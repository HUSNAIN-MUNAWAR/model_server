# semantic_search_streamlit_app.py
from __future__ import annotations

import os
import time
import json
import threading
import logging
import asyncio
from typing import List, Dict, Optional, Any, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass
import heapq
import gc
from contextlib import asynccontextmanager

# Configure basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('semantic_search_ultra_fast')

# Initialize optional libraries with default values
SentenceTransformer = None
faiss = None
ort = None
_HAS_FAISS = False
_HAS_ONNX = False

# Try to import optional/accelerated libraries
try:
    import faiss
    _HAS_FAISS = True
    logger.info("FAISS loaded successfully")
except ImportError:
    logger.warning("FAISS not available, using numpy-based index")

try:
    from sentence_transformers import SentenceTransformer
    logger.info("sentence-transformers loaded successfully")
except ImportError:
    logger.error("sentence-transformers not available - this is required!")

# Try to import ONNX Runtime for faster inference
try:
    import onnxruntime as ort
    _HAS_ONNX = True
    logger.info("ONNX Runtime loaded successfully")
except ImportError:
    logger.info("ONNX Runtime not available, using standard model")

# Import Streamlit
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

import numpy as np

# ------------------ 
# CONFIG via pydantic for clarity
# ------------------ 
class Settings:
    LOCAL_MODEL_DIR: str = 'models/all-MiniLM-L6-v2'
    MODELS_ROOT: str = 'models'
    INDEX_DIR: str = 'index'
    API_KEY: str = ''
    ALLOWED_ORIGINS: str = '*'
    HNSW_M: int = 32
    BATCH_SIZE: int = 256  # Increased for better throughput
    SINGLE_PROCESS_MODE: bool = True
    RATE_LIMIT_WINDOW: int = 60
    RATE_LIMIT_REQ: int = 1000  # Increased for higher throughput
    ENCODE_CONCURRENCY: int = 16  # Increased for better parallelization
    PERSIST_DEBOUNCE_SECONDS: int = 30  # Less frequent persistence
    COMPACT_INTERVAL: int = 14400  # Less frequent compaction (4 hours)
    SAVE_SOURCE_TEXTS: bool = False  # Disabled for speed
    
    # Optimization settings
    USE_ONNX: bool = True if _HAS_ONNX else False  # Use ONNX if available
    ONNX_MODEL_DIR: str = 'models/onnx'
    FAISS_ON_GPU: bool = False
    MAX_CACHE_ENTRIES: int = 10000  # Larger cache
    QUERY_CACHE_SIZE: int = 10000  # Larger cache
    VECTOR_CACHE_SIZE: int = 100000  # Larger cache
    MAX_QUERY_LENGTH: int = 128  # Further reduced for faster processing
    PRELOAD_MODEL: bool = True
    WARMUP_MODEL: bool = True
    PREALLOCATE_BUFFERS: bool = True  # Preallocate memory buffers
    USE_BATCHED_ENCODING: bool = True  # Process multiple queries at once
    MAX_BATCH_SIZE: int = 64  # Larger batch size
    SEARCH_CACHE_ENABLED: bool = True  # Cache search results
    FAISS_NPROBE: int = 8  # FAISS search parameter for speed/accuracy tradeoff
    CACHE_TTL_SECONDS: int = 300  # 5 minutes TTL for cache entries
    GC_COLLECTION_INTERVAL: int = 300  # 5 minutes between GC collections
    MEMORY_MONITOR_INTERVAL: int = 60  # Check memory every minute
    MEMORY_THRESHOLD_MB: int = 1024  # Clear caches if memory usage exceeds 1GB
    USE_PROCESS_POOL: bool = False  # Use process pool for CPU-intensive tasks
    PROCESS_POOL_SIZE: int = 4  # Number of processes in pool

settings = Settings()

# File paths
INDEX_DIR = settings.INDEX_DIR
MAPPING_PATH = os.path.join(INDEX_DIR, 'mapping.json')
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, 'products.index')
IDS_PATH = os.path.join(INDEX_DIR, 'ids.json')
TOMBSTONE_PATH = os.path.join(INDEX_DIR, 'tombstones.json')
MANIFEST_PATH = os.path.join(INDEX_DIR, 'manifest.json')

# Models
class ProductItem:
    def __init__(self, product_id: str, text: str):
        self.product_id = product_id
        self.text = text

class SearchResult:
    def __init__(self, product_id: str, similarity_score: float):
        self.product_id = product_id
        self.similarity_score = similarity_score

# ------------------ 
# Caching and optimization structures
# ------------------ 
@dataclass
class CachedQuery:
    results: List[SearchResult]
    timestamp: float
    access_count: int = 0

class QueryCache:
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, CachedQuery] = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.last_cleanup = time.time()
    
    def get(self, query: str, topk: int) -> Optional[List[SearchResult]]:
        if not settings.SEARCH_CACHE_ENABLED:
            return None
            
        key = f"{query}:{topk}"
        with self.lock:
            # Periodic cleanup
            if time.time() - self.last_cleanup > self.ttl_seconds:
                self._cleanup()
                
            if key in self.cache:
                cached = self.cache[key]
                cached.access_count += 1
                # Move to end (LRU)
                del self.cache[key]
                self.cache[key] = cached
                self.hits += 1
                return cached.results
            self.misses += 1
            return None
    
    def put(self, query: str, topk: int, results: List[SearchResult]):
        if not settings.SEARCH_CACHE_ENABLED:
            return
            
        key = f"{query}:{topk}"
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Remove least recently used item
                lru_key = next(iter(self.cache))
                del self.cache[lru_key]
            self.cache[key] = CachedQuery(results=results, timestamp=time.time())
    
    def _cleanup(self):
        now = time.time()
        expired_keys = [
            key for key, cached in self.cache.items() 
            if now - cached.timestamp > self.ttl_seconds
        ]
        for key in expired_keys:
            del self.cache[key]
        self.last_cleanup = now
        logger.info(f"Cache cleanup removed {len(expired_keys)} expired entries")

# ------------------ 
# Globals (guarded)
# ------------------ 
_model_lock = threading.RLock()
_model: Optional[SentenceTransformer] = None
_onnx_session: Optional[Any] = None
_model_meta: Optional[Dict[str, Any]] = None
_dim: Optional[int] = None
_index_lock = threading.RLock()
_index_impl = None
_mapping: Dict[int, str] = {}
_rev_map: Dict[str, int] = {}
_next_id_counter: int = 0
_tombstones: Set[str] = set()
_metrics = {'queries': 0, 'encodes': 0, 'avg_encode_ms': 0.0, 'avg_query_ms': 0.0}
_rate_store: Dict[str, List[float]] = {}
_rate_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_encode_semaphore = threading.BoundedSemaphore(settings.ENCODE_CONCURRENCY)
_executor = ThreadPoolExecutor(
    max_workers=max(1, settings.ENCODE_CONCURRENCY),
    thread_name_prefix="encode_worker"
)
_process_executor = ProcessPoolExecutor(
    max_workers=settings.PROCESS_POOL_SIZE
) if settings.USE_PROCESS_POOL else None
_persist_lock = threading.Lock()
_persist_scheduled = False
_persist_dirty = False

# Optimization globals
_query_cache = QueryCache(max_size=settings.QUERY_CACHE_SIZE, ttl_seconds=settings.CACHE_TTL_SECONDS)
_vector_cache: Dict[str, np.ndarray] = {}
_vector_cache_lock = threading.RLock()
_last_gc_run = time.time()
_last_memory_check = time.time()

# Preallocated buffers for faster processing
if settings.PREALLOCATE_BUFFERS:
    _encode_buffer = None
    _search_buffer = None

# ------------------ 
# Memory management and performance monitoring
# ------------------ 
def monitor_memory_usage():
    """Monitor memory usage and clear caches if needed"""
    global _last_memory_check
    current_time = time.time()
    
    # Only check every MEMORY_MONITOR_INTERVAL seconds
    if current_time - _last_memory_check < settings.MEMORY_MONITOR_INTERVAL:
        return
    
    _last_memory_check = current_time
    
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > settings.MEMORY_THRESHOLD_MB:
            logger.warning(f"Memory usage high: {memory_mb:.2f}MB, clearing caches")
            clear_caches()
            
            # Force garbage collection
            gc.collect()
    except ImportError:
        # psutil not available, skip memory monitoring
        pass

def clear_caches():
    """Clear various caches to free memory"""
    global _query_cache, _vector_cache
    
    with _query_cache.lock:
        _query_cache.cache.clear()
    
    with _vector_cache_lock:
        _vector_cache.clear()
    
    logger.info("Caches cleared to free memory")

def optimize_memory():
    """Run memory optimization routines"""
    global _last_gc_run
    current_time = time.time()
    
    # Only run every GC_COLLECTION_INTERVAL seconds
    if current_time - _last_gc_run < settings.GC_COLLECTION_INTERVAL:
        return
    
    _last_gc_run = current_time
    
    # Clear caches if memory is high
    monitor_memory_usage()
    
    # Run garbage collection
    collected = gc.collect()
    logger.debug(f"Garbage collection collected {collected} objects")

# ------------------ 
# Persistence helpers
# ------------------ 
def atomic_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def load_json(path: str, default: Any):
    try:
        if not os.path.exists(path):
            return default
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        logger.exception('Failed to load JSON from %s', path)
        return default

def save_manifest_atomic(manifest: dict) -> None:
    atomic_write_json(MANIFEST_PATH, manifest)

def save_ids_atomic(ids_list: List[int]):
    os.makedirs(os.path.dirname(IDS_PATH) or '.', exist_ok=True)
    atomic_write_json(IDS_PATH, ids_list)

# ------------------ 
# Index abstractions (Faiss & Numpy)
# ------------------ 
class BaseIndex:
    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray):
        raise NotImplementedError
    
    def remove_ids(self, ids: np.ndarray):
        raise NotImplementedError
    
    def search(self, qvec: np.ndarray, topk: int):
        raise NotImplementedError
    
    def ntotal(self) -> int:
        raise NotImplementedError
    
    def persist(self):
        raise NotImplementedError
    
    def load(self):
        raise NotImplementedError

class FaissIndex(BaseIndex):
    def __init__(self, dim: int, path: str, hnsw_m: int = 32):
        self.dim = dim
        self.path = path
        self.hnsw_m = hnsw_m
        self.index = None
        self._init_or_load()
    
    def _init_or_load(self):
        if os.path.exists(self.path) and _HAS_FAISS:
            try:
                self.index = faiss.read_index(self.path)
                # Configure for speed
                if hasattr(self.index, 'nprobe'):
                    self.index.nprobe = settings.FAISS_NPROBE
                logger.info('Loaded FAISS index from %s', self.path)
                return
            except Exception:
                logger.warning('Failed to load FAISS index; will recreate', exc_info=True)
        
        if not _HAS_FAISS:
            raise RuntimeError('FAISS not available in this environment')
        
        # Use GPU if available and requested
        if settings.FAISS_ON_GPU and faiss.get_num_gpus() > 0:
            try:
                res = faiss.StandardGpuResources()
                base = faiss.index_factory(self.dim, f'HNSW{self.hnsw_M},Flat', faiss.METRIC_INNER_PRODUCT)
                self.index = faiss.index_cpu_to_gpu(res, 0, base)
                logger.info('Created new FAISS GPU index (dim=%d)', self.dim)
            except Exception:
                logger.warning('Failed to create GPU index, falling back to CPU', exc_info=True)
                base = faiss.index_factory(self.dim, f'HNSW{self.hnsw_m},Flat', faiss.METRIC_INNER_PRODUCT)
                self.index = faiss.IndexIDMap(base)
        else:
            base = faiss.index_factory(self.dim, f'HNSW{self.hnsw_m},Flat', faiss.METRIC_INNER_PRODUCT)
            self.index = faiss.IndexIDMap(base)
            logger.info('Created new FAISS CPU index (dim=%d)', self.dim)
    
    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray):
        if vectors.dtype != np.float32:
            vectors = vectors.astype('float32')
        vectors = l2_normalize_np(vectors).astype('float32')
        self.index.add_with_ids(vectors, ids.astype(np.int64))
        
        try:
            existing = load_json(IDS_PATH, [])
            existing.extend([int(x) for x in ids.tolist()])
            seen = set(); dedup = []
            for v in existing:
                if v not in seen:
                    dedup.append(v); seen.add(v)
            save_ids_atomic(dedup)
        except Exception:
            logger.exception('Failed to update ids on disk for faiss')
    
    def remove_ids(self, ids: np.ndarray):
        try:
            sel = faiss.IDSelectorArray(ids.astype(np.int64))
            self.index.remove_ids(sel)
        except Exception:
            logger.exception('Faiss remove_ids failed')
            raise
        
        try:
            existing = load_json(IDS_PATH, [])
            remaining = [x for x in existing if int(x) not in set(int(i) for i in ids.tolist())]
            save_ids_atomic(remaining)
        except Exception:
            logger.exception('Failed to update ids file after faiss remove')
    
    def search(self, qvec: np.ndarray, topk: int):
        if qvec.dtype != np.float32:
            qvec = qvec.astype('float32')
        qvec = l2_normalize_np(qvec).astype('float32')
        D, I = self.index.search(qvec, topk)
        return D, I
    
    def ntotal(self) -> int:
        try:
            ids = load_json(IDS_PATH, [])
            return len(ids)
        except Exception:
            return int(self.index.ntotal)
    
    def persist(self):
        tmp = self.path + '.tmp'
        faiss.write_index(self.index, tmp)
        os.replace(tmp, self.path)
    
    def load(self):
        self._init_or_load()

# ------------------ 
# Helpers: mapping/tombstones
# ------------------ 
def load_mapping(path: str) -> Dict[int, str]:
    raw = load_json(path, {})
    return {int(k): v for k, v in raw.items()} if raw else {}

def save_mapping(mapping: Dict[int, str]):
    atomic_write_json(MAPPING_PATH, {str(k): v for k, v in mapping.items()})

def load_tombstones():
    global _tombstones
    _tombstones = set(load_json(TOMBSTONE_PATH, []))

def save_tombstones():
    atomic_write_json(TOMBSTONE_PATH, list(_tombstones))

def load_manifest():
    return load_json(MANIFEST_PATH, {})

def save_manifest():
    with _index_lock:
        manifest = {
            'model': _model_meta,
            'timestamp': time.time(),
            'next_id_counter': int(_next_id_counter),
            'mapping_count': len(_mapping),
        }
        try:
            save_manifest_atomic(manifest)
        except Exception:
            logger.exception('Failed to write manifest')

# ------------------ 
# Consistency routines
# ------------------ 
def _ensure_mapping_vector_consistency():
    global _mapping, _rev_map, _next_id_counter
    with _index_lock:
        try:
            vec_ids = set()
            disk_ids = set(load_json(IDS_PATH, []))
            if disk_ids:
                vec_ids = set(int(x) for x in disk_ids)
            else:
                if _HAS_FAISS and _index_impl and hasattr(_index_impl.index, 'ntotal'):
                    vec_ids = set(range(_index_impl.index.ntotal))
                else:
                    vec_ids = set(_mapping.keys()) if _mapping else set()
            
            mapping_ids = set(_mapping.keys())
            if vec_ids == mapping_ids:
                _rev_map = {v: k for k, v in _mapping.items()}
                _next_id_counter = max(mapping_ids) + 1 if mapping_ids else 0
                return
            
            if not vec_ids and mapping_ids:
                backup = MAPPING_PATH + '.backup.' + str(int(time.time()))
                atomic_write_json(backup, {str(k): v for k, v in _mapping.items()})
                _mapping = {}; _rev_map = {}; _next_id_counter = 0
                save_mapping(_mapping)
                logger.warning('Vectors missing but mapping existed: mapping backed up and cleared')
                return
            
            common = {k: v for k, v in _mapping.items() if int(k) in vec_ids}
            if len(common) != len(_mapping):
                logger.warning('Mapping and vector id sets disagreed; truncating mapping to common ids')
                _mapping = {int(k): v for k, v in common.items()}
                _rev_map = {v: k for k, v in _mapping.items()}
                _next_id_counter = max(vec_ids) + 1 if vec_ids else 0
                save_mapping(_mapping)
        except Exception:
            logger.exception('Error ensuring mapping/vector consistency')

# ------------------ 
# Index chooser
# ------------------ 
def ensure_index_impl(dim: int):
    global _index_impl
    with _index_lock:
        if _HAS_FAISS:
            _index_impl = FaissIndex(dim=dim, path=FAISS_INDEX_PATH, hnsw_m=settings.HNSW_M)
        else:
            raise RuntimeError("FAISS is required for production use")
        _index_impl.load()

# ------------------ 
# Model loading helpers (search and lazy load)
# ------------------ 
def find_local_model_dir() -> Optional[str]:
    # Prefer explicit setting if exists
    if settings.LOCAL_MODEL_DIR and os.path.exists(settings.LOCAL_MODEL_DIR):
        return settings.LOCAL_MODEL_DIR
    
    # Otherwise search models/ for directories that look like sentence-transformer models
    root = Path(settings.MODELS_ROOT)
    if not root.exists() or not root.is_dir():
        return None
    
    # Heuristics: prefer folders containing config_sentence_transformers.json or tokenizer.json
    candidates = []
    for p in root.iterdir():
        if p.is_dir():
            if (p / 'config_sentence_transformers.json').exists() or (p / 'tokenizer.json').exists() or (p / 'config.json').exists():
                candidates.append(str(p))
    
    # Sort by name or modification time (newer first)
    if not candidates:
        return None
    candidates.sort(key=lambda s: os.path.getmtime(s), reverse=True)
    return candidates[0]

def _load_model_from(path: str):
    global _model, _model_meta, _dim, _onnx_session
    if SentenceTransformer is None:
        raise RuntimeError('sentence-transformers library not available in environment')
    
    logger.info('Loading model from %s', path)
    
    # Try to load ONNX model if enabled and available
    if settings.USE_ONNX and _HAS_ONNX:
        onnx_path = os.path.join(settings.ONNX_MODEL_DIR, "model.onnx")
        if os.path.exists(onnx_path):
            try:
                # Optimize ONNX session
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                # Set providers - prioritize CUDA if available, then CPU
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
                
                _onnx_session = ort.InferenceSession(onnx_path, options, providers=providers)
                # Get model dimension from ONNX model
                model_output = _onnx_session.get_outputs()[0]
                _dim = model_output.shape[-1]
                _model_meta = {'name': os.path.basename(path.rstrip('/')), 'dim': _dim, 'onnx': True}
                logger.info('Loaded ONNX model from %s, dim=%d', onnx_path, _dim)
                return None  # ONNX model loaded, no need for SentenceTransformer
            except Exception as e:
                logger.warning('Failed to load ONNX model: %s', e)
    
    # Fall back to SentenceTransformer
    model = SentenceTransformer(path)
    dim = model.get_sentence_embedding_dimension()
    
    with _model_lock:
        _model = model
        _dim = dim
        _model_meta = {'name': os.path.basename(path.rstrip('/')), 'dim': dim, 'onnx': False}
    
    # Warmup with larger batch for better performance
    if settings.WARMUP_MODEL:
        try:
            logger.info('Warming up model...')
            warmup_texts = ['hello world'] * 16  # Warm up with multiple samples
            _ = model.encode(warmup_texts, convert_to_numpy=True, batch_size=16, show_progress_bar=False)
            logger.info('Warmup complete')
        except Exception:
            logger.exception('Warmup failed (non-fatal)')
    
    return _model

def get_model_safe() -> Optional[SentenceTransformer]:
    """Return a loaded model. If none loaded, attempt to find & load one (lazy). Thread-safe."""
    global _model, _dim, _model_meta, _onnx_session
    with _model_lock:
        if _model is not None or _onnx_session is not None:
            return _model  # Return the model (might be None if using ONNX)
        
        # Try to find model path
        model_dir = find_local_model_dir()
        if not model_dir:
            logger.warning('No local model found (searched %s and LOCAL_MODEL_DIR)', settings.MODELS_ROOT)
            return None
        
        try:
            return _load_model_from(model_dir)
        except Exception:
            logger.exception('Failed to load model from: %s', model_dir)
            _model = None
            _dim = None
            _model_meta = None
            _onnx_session = None
            return None

# ------------------ 
# Helpers
# ------------------ 
def l2_normalize_np(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        return x / (norm + 1e-12)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    return x / norms

def encode_with_onnx(texts: List[str]) -> np.ndarray:
    """Encode texts using ONNX runtime for faster inference"""
    if _onnx_session is None:
        raise RuntimeError("ONNX session not initialized")
    
    # Use the SentenceTransformer model for tokenization if available
    if _model is not None:
        features = _model.tokenize(texts)
        inputs = {
            'input_ids': features['input_ids'].numpy(),
            'attention_mask': features['attention_mask'].numpy(),
            'token_type_ids': features['token_type_ids'].numpy() if 'token_type_ids' in features else None
        }
        # Remove None values
        inputs = {k: v for k, v in inputs.items() if v is not None}
    else:
        # Fallback: create dummy inputs (not recommended for production)
        max_length = min(max(len(text) for text in texts), settings.MAX_QUERY_LENGTH)
        input_ids = np.zeros((len(texts), max_length), dtype=np.int64)
        attention_mask = np.ones((len(texts), max_length), dtype=np.int64)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    # Run inference
    outputs = _onnx_session.run(None, inputs)
    embeddings = outputs[0]
    
    # Apply pooling (mean pooling as default)
    if len(embeddings.shape) == 3:
        embeddings = np.mean(embeddings, axis=1)
    
    return l2_normalize_np(embeddings)

def _schedule_persist():
    global _persist_scheduled, _persist_dirty
    with _persist_lock:
        _persist_dirty = True
        if _persist_scheduled:
            return
        _persist_scheduled = True
    
    def worker():
        global _persist_scheduled, _persist_dirty
        time.sleep(settings.PERSIST_DEBOUNCE_SECONDS)
        try:
            with _index_lock:
                logger.info('Persisting mapping, tombstones and index...')
                save_mapping(_mapping)
                save_tombstones()
                try:
                    _index_impl.persist()
                except Exception:
                    logger.exception('Index persist failed')
                try:
                    save_manifest()
                except Exception:
                    logger.exception('Failed to persist manifest')
        finally:
            with _persist_lock:
                _persist_dirty = False
                _persist_scheduled = False
    
    t = threading.Thread(target=worker, daemon=True)
    add_script_run_ctx(t)  # Make thread compatible with Streamlit
    t.start()

# ------------------ 
# API functions adapted for Streamlit
# ------------------ 
def health():
    return {
        'status': 'ok',
        'model_loaded': get_model_safe() is not None or _onnx_session is not None,
        'model_version': _model_meta.get('name') if _model_meta else None,
        'index_size': _index_impl.ntotal() if _index_impl else 0,
        'timestamp': int(time.time()),
        'mapping_count': len(_mapping),
        'tombstones': len(_tombstones),
        'cache_stats': {'hits': _query_cache.hits, 'misses': _query_cache.misses, 'size': len(_query_cache.cache)}
    }

def search(q: str, topk: int = 10):
    """Search function for Streamlit"""
    # Run memory optimization
    optimize_memory()
    
    if not q:
        return {"error": "Missing 'q' parameter"}
    
    # Check cache first
    cached = _query_cache.get(q, topk)
    if cached is not None:
        return cached
    
    model = get_model_safe()
    if model is None and _onnx_session is None:
        return {"error": 'Model not loaded; call /admin/reload_model or ensure models/ contains a valid model'}
    
    start = time.time()
    
    try:
        # Offload encode to threadpool with timeout
        with _encode_semaphore:
            # Truncate long queries to improve performance
            truncated_q = q[:settings.MAX_QUERY_LENGTH] if len(q) > settings.MAX_QUERY_LENGTH else q
            
            if _onnx_session is not None:
                # Use ONNX for faster encoding
                qvec = encode_with_onnx([truncated_q])
            else:
                # Use SentenceTransformer for encoding
                qvec = model.encode(
                    [truncated_q], 
                    convert_to_numpy=True, 
                    batch_size=1,
                    show_progress_bar=False
                )
                qvec = np.array(qvec)
            
            qvec = l2_normalize_np(qvec).astype('float32')
            enc_ms = (time.time() - start) * 1000.0
    except Exception:
        logger.exception('Encode error')
        return {"error": 'Failed to encode query'}
    
    with _index_lock:
        # Configure FAISS for faster search if possible
        if hasattr(_index_impl.index, 'nprobe'):
            original_nprobe = _index_impl.index.nprobe
            _index_impl.index.nprobe = settings.FAISS_NPROBE
        
        D, I = _index_impl.search(qvec, topk)
        
        # Restore original nprobe setting
        if hasattr(_index_impl.index, 'nprobe'):
            _index_impl.index.nprobe = original_nprobe
            
        results: List[SearchResult] = []
        
        if I.shape[1] == 0:
            return results
        
        for score, idx in zip(D[0], I[0]):
            if int(idx) == -1:
                continue
            pid = _mapping.get(int(idx))
            if pid is None or pid in _tombstones:
                continue
            # Normalize score to 0-1 range
            norm_score = float((float(score) + 1.0) / 2.0)
            results.append(SearchResult(product_id=pid, similarity_score=norm_score))
    
    # Cache the results
    _query_cache.put(q, topk, results)
    
    query_ms = (time.time() - start) * 1000.0
    logger.info(f"Query took {query_ms:.2f}ms (encode: {enc_ms:.2f}ms)")
    
    # Update metrics
    _metrics['queries'] += 1
    _metrics['avg_query_ms'] = (_metrics['avg_query_ms'] * (_metrics['queries'] - 1) + query_ms) / _metrics['queries']
    
    return results

def embeddings_endpoint(payload):
    """Embeddings function for Streamlit"""
    # Run memory optimization
    optimize_memory()
    
    model = get_model_safe()
    if model is None and _onnx_session is None:
        return {"error": 'Model not loaded; call /admin/reload_model or ensure models/ contains a valid model'}
    
    texts: List[str] = []
    
    # Fast parsing with minimal operations
    if isinstance(payload, dict):
        if 'inputs' in payload:
            inputs = payload['inputs']
            if isinstance(inputs, str):
                texts = [inputs]
            elif isinstance(inputs, list):
                texts = [str(x) for x in inputs if isinstance(x, str)]
        elif 'input' in payload:
            input_val = payload['input']
            if isinstance(input_val, str):
                texts = [input_val]
            elif isinstance(input_val, list):
                texts = [str(x) for x in input_val if isinstance(x, str)]
        elif 'text' in payload and isinstance(payload['text'], str):
            texts = [payload['text']]
        elif 'texts' in payload and isinstance(payload['texts'], list):
            texts = [str(x) for x in payload['texts'] if isinstance(x, str)]
    elif isinstance(payload, list):
        texts = [str(x) for x in payload if isinstance(x, str)]
    elif isinstance(payload, str):
        texts = [payload]
    
    if not texts:
        return {"error": 'No valid input texts provided'}
    
    # Limit batch size for performance
    if len(texts) > 100:
        texts = texts[:100]
    
    # Encode in batches using the same executor + semaphore pattern (non-blocking)
    try:
        embeddings = []
        start_time = time.time()
        
        for i in range(0, len(texts), settings.MAX_BATCH_SIZE):
            batch = texts[i:i + settings.MAX_BATCH_SIZE]
            with _encode_semaphore:
                if _onnx_session is not None:
                    vecs = encode_with_onnx(batch)
                else:
                    vecs = model.encode(
                        batch, 
                        batch_size=len(batch), 
                        convert_to_tensor=False,  # Skip tensor conversion for speed
                        show_progress_bar=False,
                        normalize_embeddings=True  # Normalize directly
                    )
                # Convert to python floats for JSON
                embeddings.extend(vecs.tolist())
        
        encode_time = (time.time() - start_time) * 1000
        logger.info(f"Encoded {len(texts)} texts in {encode_time:.2f}ms")
        
        # Update metrics
        _metrics['encodes'] += len(texts)
        _metrics['avg_encode_ms'] = (_metrics['avg_encode_ms'] * (_metrics['encodes'] - len(texts)) + encode_time) / _metrics['encodes']
        
    except Exception:
        logger.exception('Embedding encode failed')
        return {"error": 'Failed to compute embeddings'}
    
    # Return minimal response for speed
    return {
        "embeddings": embeddings,
        "model": _model_meta.get('name', 'unknown'),
        "count": len(embeddings)
    }

def reindex(items: List[ProductItem]):
    model = get_model_safe()
    if model is None and _onnx_session is None:
        return {"error": 'Model not loaded'}
    if not items:
        return {"error": 'Request body must be non-empty'}
    
    texts = [it.text for it in items]; pids = [it.product_id for it in items]
    
    try:
        with _encode_semaphore:
            if _onnx_session is not None:
                vecs = encode_with_onnx(texts)
            else:
                vecs = model.encode(texts, convert_to_numpy=True, batch_size=settings.BATCH_SIZE, show_progress_bar=False)
            vecs = l2_normalize_np(vecs).astype('float32')
    except Exception:
        logger.exception('Failed to encode items')
        return {"error": 'Failed to encode items'}
    
    assigned = {}
    with _index_lock:
        for vec, pid in zip(vecs, pids):
            existing = _rev_map.get(pid)
            if existing is not None:
                try:
                    _index_impl.remove_ids(np.array([existing], dtype=np.int64))
                except Exception:
                    logger.debug('Backend does not support remove_ids; replacement may leave duplicates')
                _index_impl.add_with_ids(vec.reshape(1, -1), np.array([existing], dtype=np.int64))
                assigned[pid] = existing
            else:
                nid = int(_next_id_counter); _next_id_counter += 1
                _index_impl.add_with_ids(vec.reshape(1, -1), np.array([nid], dtype=np.int64))
                _mapping[nid] = pid; _rev_map[pid] = nid
                assigned[pid] = nid
    
    _schedule_persist()
    return {'indexed': len(assigned), 'assigned_index_ids': assigned}

def rebuild_index(items: List[ProductItem]):
    if not items:
        return {"error": 'Request body must be non-empty'}
    
    # Simple rebuild implementation - just call reindex for all items
    result = reindex(items)
    return {'status': 'rebuild_complete', 'indexed': result['indexed']}

def delete_items(product_ids: List[str]):
    if not product_ids:
        return {"error": 'Provide non-empty product_ids list'}
    
    to_remove_ids = []
    with _index_lock:
        for pid in product_ids:
            nid = _rev_map.get(pid)
            if nid is None:
                continue
            to_remove_ids.append(nid)
            _tombstones.add(pid)
            if nid in _mapping:
                del _mapping[nid]
            if pid in _rev_map:
                del _rev_map[pid]
    
    if not to_remove_ids:
        return {'deleted': 0}
    
    try:
        _index_impl.remove_ids(np.array(to_remove_ids, dtype=np.int64))
    except Exception:
        logger.debug('backend did not support remove_ids; tombstones added')
    
    _schedule_persist()
    return {'deleted': len(to_remove_ids)}

def reload_model():
    global _model, _model_meta, _dim, _onnx_session
    model_dir = find_local_model_dir()
    if not model_dir:
        return {"error": f'LOCAL_MODEL_DIR not found and no model in {settings.MODELS_ROOT}'}
    if SentenceTransformer is None and not _HAS_ONNX:
        return {"error": 'SentenceTransformer and ONNX not available in environment'}
    
    try:
        # Try to load ONNX model first if enabled
        if settings.USE_ONNX and _HAS_ONNX:
            onnx_path = os.path.join(settings.ONNX_MODEL_DIR, "model.onnx")
            if os.path.exists(onnx_path):
                try:
                    # Optimize ONNX session
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = 1
                    options.inter_op_num_threads = 1
                    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    
                    # Set providers - prioritize CUDA if available, then CPU
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
                    
                    new_onnx_session = ort.InferenceSession(onnx_path, options, providers=providers)
                    # Get model dimension from ONNX model
                    model_output = new_onnx_session.get_outputs()[0]
                    new_dim = model_output.shape[-1]
                    
                    if _index_impl and _index_impl.ntotal() > 0 and new_dim != _dim:
                        return {"error": f'model dimension mismatch (current index dim={_dim}, candidate dim={new_dim}). Rebuild required before switching.'}
                    
                    with _model_lock:
                        old_onnx = _onnx_session
                        _onnx_session = new_onnx_session
                        _dim = new_dim
                        _model_meta = {'name': os.path.basename(model_dir), 'dim': _dim, 'onnx': True}
                        ensure_index_impl(_dim)
                        save_manifest()
                        try:
                            del old_onnx
                        except Exception:
                            pass
                    
                    return {'status': 'model_reloaded', 'model_meta': _model_meta}
                except Exception as e:
                    logger.warning('Failed to load ONNX model: %s', e)
                    # Fall through to SentenceTransformer loading
        
        # Fall back to SentenceTransformer
        candidate = SentenceTransformer(model_dir)
        cand_dim = candidate.get_sentence_embedding_dimension()
        if _index_impl and _index_impl.ntotal() > 0 and cand_dim != _dim:
            return {"error": f'model dimension mismatch (current index dim={_dim}, candidate dim={cand_dim}). Rebuild required before switching.'}
        
        with _model_lock:
            old = _model
            _model = candidate
            _dim = cand_dim
            _model_meta = {'name': os.path.basename(model_dir), 'dim': _dim, 'onnx': False}
            ensure_index_impl(_dim)
            save_manifest()
            try:
                del old
            except Exception:
                pass
        
        return {'status': 'model_reloaded', 'model_meta': _model_meta}
    except Exception:
        logger.exception('Failed to reload model')
        return {"error": 'failed to load model'}

def warmup():
    model = get_model_safe()
    if model is None and _onnx_session is None:
        return {"error": 'Model not loaded'}
    try:
        with _encode_semaphore:
            if _onnx_session is not None:
                _ = encode_with_onnx(['warmup text'] * 16)
            else:
                _ = model.encode(['warmup text'] * 16, convert_to_numpy=True, batch_size=16, show_progress_bar=False)
    except Exception:
        logger.exception('Warmup failed')
        return {"error": 'Warmup failed'}
    return {'status': 'warmup_complete'}

def get_metrics():
    return {
        'queries': _metrics['queries'],
        'encodes': _metrics['encodes'],
        'avg_encode_ms': _metrics['avg_encode_ms'],
        'avg_query_ms': _metrics['avg_query_ms'],
        'cache_stats': {
            'hits': _query_cache.hits,
            'misses': _query_cache.misses,
            'size': len(_query_cache.cache)
        }
    }

# ------------------ 
# Streamlit UI and API endpoints
# ------------------ 
def main():
    st.set_page_config(
        page_title="Ultra Fast Semantic Search Server",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("Ultra Fast Semantic Search Server")
    
    # Initialize the system
    if 'initialized' not in st.session_state:
        startup_event()
        st.session_state.initialized = True
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["Search", "Embeddings", "Admin", "Health"])
    
    with tab1:
        st.header("Semantic Search")
        
        # Search form
        with st.form("search_form"):
            query = st.text_input("Search query", placeholder="Enter your search query...")
            topk = st.slider("Number of results", min_value=1, max_value=100, value=10)
            submitted = st.form_submit_button("Search")
            
            if submitted and query:
                with st.spinner("Searching..."):
                    results = search(query, topk)
                    if "error" in results:
                        st.error(results["error"])
                    else:
                        st.success(f"Found {len(results)} results")
                        for i, result in enumerate(results):
                            st.write(f"{i+1}. {result.product_id} (Score: {result.similarity_score:.4f})")
    
    with tab2:
        st.header("Embeddings Generation")
        
        # Embeddings form
        with st.form("embeddings_form"):
            input_type = st.radio("Input type", ["Text", "List of texts", "JSON"])
            
            if input_type == "Text":
                text_input = st.text_area("Enter text")
                payload = text_input
            elif input_type == "List of texts":
                texts_input = st.text_area("Enter texts (one per line)")
                payload = texts_input.split("\n") if texts_input else []
            else:  # JSON
                json_input = st.text_area("Enter JSON payload")
                try:
                    payload = json.loads(json_input) if json_input else {}
                except json.JSONDecodeError:
                    st.error("Invalid JSON")
                    payload = {}
            
            submitted = st.form_submit_button("Generate Embeddings")
            
            if submitted and payload:
                with st.spinner("Generating embeddings..."):
                    result = embeddings_endpoint(payload)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success(f"Generated {result['count']} embeddings")
                        st.json(result)
    
    with tab3:
        st.header("Administration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Management")
            if st.button("Reload Model"):
                result = reload_model()
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success("Model reloaded successfully")
                    st.json(result)
            
            if st.button("Warmup Model"):
                result = warmup()
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success("Model warmed up successfully")
            
            if st.button("Clear Cache"):
                clear_caches()
                st.success("Cache cleared successfully")
        
        with col2:
            st.subheader("Index Management")
            
            # Upload CSV for indexing
            uploaded_file = st.file_uploader("Upload CSV for indexing", type=["csv"])
            if uploaded_file is not None:
                import csv
                import io
                content = uploaded_file.read().decode('utf-8')
                reader = csv.DictReader(io.StringIO(content))
                items = []
                for r in reader:
                    if 'product_id' not in r or 'text' not in r:
                        continue
                    items.append(ProductItem(product_id=r['product_id'], text=r['text']))
                
                if items and st.button("Index CSV Data"):
                    with st.spinner("Indexing..."):
                        result = reindex(items)
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            st.success(f"Indexed {result['indexed']} items")
            
            # Delete items
            delete_ids = st.text_area("Product IDs to delete (one per line)")
            if delete_ids and st.button("Delete Items"):
                ids_to_delete = delete_ids.split("\n")
                result = delete_items(ids_to_delete)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"Deleted {result['deleted']} items")
    
    with tab4:
        st.header("System Health")
        
        if st.button("Refresh Health Status"):
            pass  # Just triggers a rerun
        
        health_status = health()
        st.json(health_status)
        
        st.subheader("Metrics")
        metrics = get_metrics()
        st.json(metrics)
        
        # Display some charts
        st.subheader("Performance Charts")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Queries", metrics['queries'])
            st.metric("Average Query Time", f"{metrics['avg_query_ms']:.2f} ms")
        
        with col2:
            st.metric("Total Encodes", metrics['encodes'])
            st.metric("Average Encode Time", f"{metrics['avg_encode_ms']:.2f} ms")
        
        # Cache performance
        cache_stats = metrics['cache_stats']
        hit_ratio = cache_stats['hits'] / (cache_stats['hits'] + cache_stats['misses']) if (cache_stats['hits'] + cache_stats['misses']) > 0 else 0
        st.metric("Cache Hit Ratio", f"{hit_ratio:.2%}")

# ------------------ 
# Graceful shutdown
# ------------------ 
def startup_event():
    global _mapping, _rev_map, _next_id_counter, _dim
    os.makedirs(INDEX_DIR, exist_ok=True)
    load_tombstones()
    _mapping = load_mapping(MAPPING_PATH)
    _rev_map = {v: k for k, v in _mapping.items()}
    manifest = load_manifest() or {}
    _next_id_counter = int(manifest.get('next_id_counter', max(_mapping.keys()) + 1 if _mapping else 0))
    
    # Attempt to eagerly load model if available
    get_model_safe()
    
    # Ensure index exists for dim or fallback to 384 if _dim is None
    if _dim is None:
        _dim = 384
    ensure_index_impl(_dim)
    
    try:
        _ensure_mapping_vector_consistency()
    except Exception:
        logger.exception('Consistency check failed at startup')
    
    try:
        save_manifest()
    except Exception:
        logger.exception('Failed to save manifest on startup')
    
    logger.info('Startup complete. model_loaded=%s index_size=%d', _model is not None or _onnx_session is not None, _index_impl.ntotal())

# ------------------ 
# Main entry point
# ------------------ 
if __name__ == "__main__":
    main()
