# app.py
import os
import time
import json
import threading
import logging
import requests
from typing import List, Dict, Optional, Any, Set
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dataclasses import dataclass
import gc

# Optional accelerated libs
SentenceTransformer = None
faiss = None
ort = None
_HAS_FAISS = False
_HAS_ONNX = False

try:
    import faiss
    _HAS_FAISS = True
    logging.getLogger().info("FAISS loaded successfully")
except Exception:
    logging.getLogger().warning("FAISS not available")

try:
    from sentence_transformers import SentenceTransformer
    logging.getLogger().info("sentence-transformers loaded successfully")
except Exception:
    logging.getLogger().error("sentence-transformers not available - this is required!")

try:
    import onnxruntime as ort
    _HAS_ONNX = True
    logging.getLogger().info("ONNX Runtime loaded successfully")
except Exception:
    logging.getLogger().info("ONNX Runtime not available, using standard model")

# Streamlit UI imports
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
import numpy as np

# Web API imports (FastAPI + uvicorn)
try:
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response
    import uvicorn
    _HAS_UVICORN = True
except Exception:
    _HAS_UVICORN = False

# ------------------ 
# CONFIG
# ------------------ 
class Settings:
    LOCAL_MODEL_DIR: str = 'model/all-MiniLM-L6-v2'
    MODELS_ROOT: str = 'model'
    INDEX_DIR: str = 'index'
    HNSW_M: int = 32
    BATCH_SIZE: int = 256
    ENCODE_CONCURRENCY: int = 16
    USE_ONNX: bool = False
    MAX_QUERY_LENGTH: int = 128
    PRELOAD_MODEL: bool = True
    WARMUP_MODEL: bool = True
    SEARCH_CACHE_ENABLED: bool = True
    FAISS_NPROBE: int = 8
    CACHE_TTL_SECONDS: int = 300
    PING_INTERVAL: int = 10
    PING_URL: str = os.environ.get("PING_URL", "")
    # CORS / API settings - Enhanced CORS configuration
    ALLOWED_ORIGINS: str = os.environ.get("ALLOWED_ORIGINS", "*")
    ALLOWED_METHODS: str = os.environ.get("ALLOWED_METHODS", "*")
    ALLOWED_HEADERS: str = os.environ.get("ALLOWED_HEADERS", "*")
    ALLOW_CREDENTIALS: bool = os.environ.get("ALLOW_CREDENTIALS", "False").lower() == "true"
    API_HOST: str = os.environ.get("MODEL_API_HOST", "0.0.0.0")
    API_PORT: int = int(os.environ.get("MODEL_API_PORT", "8000"))
    # New setting to control API server startup
    ENABLE_API_SERVER: bool = os.environ.get("ENABLE_API_SERVER", "False").lower() == "true"

settings = Settings()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('semantic_search_streamlit')

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, settings.INDEX_DIR)
os.makedirs(INDEX_DIR, exist_ok=True)
MAPPING_PATH = os.path.join(INDEX_DIR, 'mapping.json')
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, 'products.index')
IDS_PATH = os.path.join(INDEX_DIR, 'ids.json')
TOMBSTONE_PATH = os.path.join(INDEX_DIR, 'tombstones.json')
MANIFEST_PATH = os.path.join(INDEX_DIR, 'manifest.json')

# Models
class ProductItem:
    def __init__(self, product_id: str, text: str):
        self.product_id = product_id
        self.text = str(text)

class SearchResult:
    def __init__(self, product_id: str, similarity_score: float):
        self.product_id = product_id
        self.similarity_score = similarity_score

    def to_dict(self):
        return {"product_id": self.product_id, "similarity_score": float(self.similarity_score)}

# ------------------ 
# Caching
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
            if time.time() - self.last_cleanup > self.ttl_seconds:
                self._cleanup()
                
            if key in self.cache:
                cached = self.cache[key]
                cached.access_count += 1
                # move to end (simple LRU behavior)
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

# ------------------ 
# Globals
# ------------------ 
_model_lock = threading.RLock()
_model: Optional[SentenceTransformer] = None
_model_meta: Optional[Dict[str, Any]] = None
_dim: Optional[int] = None
_index_lock = threading.RLock()
_index_impl = None
_mapping: Dict[int, str] = {}
_rev_map: Dict[str, int] = {}
_next_id_counter: int = 0
_tombstones: Set[str] = set()
_metrics = {'queries': 0, 'encodes': 0, 'avg_encode_ms': 0.0, 'avg_query_ms': 0.0}
_encode_semaphore = threading.BoundedSemaphore(settings.ENCODE_CONCURRENCY)
_executor = ThreadPoolExecutor(
    max_workers=max(1, settings.ENCODE_CONCURRENCY),
    thread_name_prefix="encode_worker"
)
_persist_lock = threading.Lock()
_persist_scheduled = False

_query_cache = QueryCache(max_size=10000, ttl_seconds=settings.CACHE_TTL_SECONDS)
_last_gc_run = time.time()

# Ping controls
_ping_thread = None
_ping_running = False

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
        return default

def save_manifest_atomic(manifest: dict) -> None:
    atomic_write_json(MANIFEST_PATH, manifest)

def save_ids_atomic(ids_list: List[int]):
    os.makedirs(os.path.dirname(IDS_PATH) or '.', exist_ok=True)
    atomic_write_json(IDS_PATH, ids_list)

# ------------------ 
# FAISS index wrapper
# ------------------ 
class FaissIndex:
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
                if hasattr(self.index, 'nprobe'):
                    self.index.nprobe = settings.FAISS_NPROBE
                logger.info('Loaded FAISS index from %s', self.path)
                return
            except Exception:
                logger.warning('Failed to load FAISS index; will recreate')
        
        if not _HAS_FAISS:
            raise RuntimeError('FAISS not available in this environment')
        
        base = faiss.index_factory(self.dim, f'HNSW{self.hnsw_m},Flat', faiss.METRIC_INNER_PRODUCT)
        self.index = faiss.IndexIDMap(base)
        logger.info('Created new FAISS CPU index (dim=%d)', self.dim)
    
    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray):
        if vectors.dtype != np.float32:
            vectors = vectors.astype('float32')
        vectors = self.l2_normalize_np(vectors).astype('float32')
        self.index.add_with_ids(vectors, ids.astype(np.int64))
        
        try:
            existing = load_json(IDS_PATH, [])
            existing.extend([int(x) for x in ids.tolist()])
            seen = set()
            dedup = []
            for v in existing:
                if v not in seen:
                    dedup.append(v)
                    seen.add(v)
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
        qvec = self.l2_normalize_np(qvec).astype('float32')
        D, I = self.index.search(qvec, topk)
        return D, I
    
    def ntotal(self) -> int:
        try:
            ids = load_json(IDS_PATH, [])
            return len(ids)
        except Exception:
            return int(self.index.ntotal)
    
    def persist(self):
        if _HAS_FAISS:
            tmp = self.path + '.tmp'
            faiss.write_index(self.index, tmp)
            os.replace(tmp, self.path)
    
    def load(self):
        self._init_or_load()
    
    def l2_normalize_np(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            norm = np.linalg.norm(x)
            return x / (norm + 1e-12)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        return x / norms


class NumpyIndex:
    """Small dependency-free cosine index used when FAISS is unavailable.

    This is intentionally an exact-search fallback for local demos and small
    datasets. FAISS remains the preferred implementation for larger indexes.
    """

    def __init__(self, dim: int, path: str):
        self.dim = dim
        self.path = os.path.splitext(path)[0] + ".npz"
        self.index = None
        self.vectors = np.empty((0, dim), dtype="float32")
        self.ids = np.empty((0,), dtype="int64")
        self.load()

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray):
        normalized = self.l2_normalize_np(vectors).astype("float32")
        self.vectors = np.vstack((self.vectors, normalized))
        self.ids = np.concatenate((self.ids, ids.astype("int64")))
        self.persist()

    def remove_ids(self, ids: np.ndarray):
        remove = set(int(value) for value in ids.tolist())
        keep = np.array([int(value) not in remove for value in self.ids], dtype=bool)
        self.vectors = self.vectors[keep]
        self.ids = self.ids[keep]
        self.persist()

    def search(self, qvec: np.ndarray, topk: int):
        if not len(self.ids):
            return np.empty((1, 0), dtype="float32"), np.empty((1, 0), dtype="int64")
        scores = self.vectors @ self.l2_normalize_np(qvec).reshape(-1).astype("float32")
        order = np.argsort(-scores)[:topk]
        return scores[order].reshape(1, -1), self.ids[order].reshape(1, -1)

    def ntotal(self) -> int:
        return int(len(self.ids))

    def persist(self):
        tmp = self.path + ".tmp.npz"
        np.savez_compressed(tmp, vectors=self.vectors, ids=self.ids)
        os.replace(tmp, self.path)

    def load(self):
        if not os.path.exists(self.path):
            return
        try:
            with np.load(self.path) as data:
                vectors = np.asarray(data["vectors"], dtype="float32")
                ids = np.asarray(data["ids"], dtype="int64")
            if vectors.ndim != 2 or vectors.shape[1] != self.dim or len(vectors) != len(ids):
                raise ValueError("invalid NumPy index dimensions")
            self.vectors = vectors
            self.ids = ids
        except Exception:
            logger.exception("Failed to load NumPy index; starting empty")

    def l2_normalize_np(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            norm = np.linalg.norm(x)
            return x / (norm + 1e-12)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        return x / norms

# ------------------ 
# mapping/tombstones
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
# Model loading
# ------------------ 
def _load_model_from(path: str):
    global _model, _model_meta, _dim
    if SentenceTransformer is None:
        raise RuntimeError('sentence-transformers library not available in environment')
    
    logger.info('Loading model from %s', path)
    
    if not os.path.exists(path):
        logger.error('Model directory does not exist: %s', path)
        return None
    
    model = SentenceTransformer(path)
    dim = model.get_sentence_embedding_dimension()
    
    with _model_lock:
        _model = model
        _dim = dim
        _model_meta = {'name': os.path.basename(path.rstrip('/')), 'dim': dim}
    
    if settings.WARMUP_MODEL:
        try:
            logger.info('Warming up model...')
            warmup_texts = ['hello world'] * 4
            _ = model.encode(warmup_texts, convert_to_numpy=True, batch_size=4, show_progress_bar=False)
            logger.info('Warmup complete')
        except Exception:
            logger.exception('Warmup failed (non-fatal)')
    
    return _model

def get_model_safe() -> Optional[SentenceTransformer]:
    global _model, _dim, _model_meta
    with _model_lock:
        if _model is not None:
            return _model
        
        model_dir = os.path.join(BASE_DIR, settings.LOCAL_MODEL_DIR)
        if not os.path.exists(model_dir):
            logger.error('Model not found at: %s', model_dir)
            model_root = Path(os.path.join(BASE_DIR, settings.MODELS_ROOT))
            if model_root.exists() and model_root.is_dir():
                for p in model_root.iterdir():
                    if p.is_dir():
                        model_dir = str(p)
                        logger.info('Found alternative model at: %s', model_dir)
                        break
            
            if not os.path.exists(model_dir):
                logger.error('No models found in %s', settings.MODELS_ROOT)
                return None
        
        try:
            return _load_model_from(model_dir)
        except Exception:
            logger.exception('Failed to load model from: %s', model_dir)
            _model = None
            _dim = None
            _model_meta = None
            return None

# ------------------ 
# Index chooser
# ------------------ 
def ensure_index_impl(dim: int):
    global _index_impl
    with _index_lock:
        if _HAS_FAISS:
            if _index_impl is None:
                _index_impl = FaissIndex(dim=dim, path=FAISS_INDEX_PATH, hnsw_m=settings.HNSW_M)
        else:
            if _index_impl is None:
                logger.warning("FAISS unavailable; using exact NumPy index fallback")
                _index_impl = NumpyIndex(dim=dim, path=FAISS_INDEX_PATH)
        _index_impl.load()

# ------------------ 
# Core API functions
# ------------------ 
def health():
    return {
        'status': 'ok',
        'model_loaded': _model is not None,
        'model_version': _model_meta.get('name') if _model_meta else None,
        'index_size': _index_impl.ntotal() if _index_impl else 0,
        'timestamp': int(time.time()),
        'mapping_count': len(_mapping),
        'tombstones': len(_tombstones),
    }

def _normalize_topk(topk: Any) -> int:
    try:
        value = int(topk)
    except (TypeError, ValueError) as exc:
        raise ValueError("topk must be an integer") from exc
    if value < 1 or value > 100:
        raise ValueError("topk must be between 1 and 100")
    return value


def search(q: str, topk: int = 10):
    if not isinstance(q, str) or not q.strip():
        return {"error": "Missing query parameter"}

    try:
        topk = _normalize_topk(topk)
    except ValueError as exc:
        return {"error": str(exc)}
    q = q.strip()[:settings.MAX_QUERY_LENGTH]
    
    cached = _query_cache.get(q, topk)
    if cached is not None:
        return [r.to_dict() for r in cached]
    
    model = get_model_safe()
    if model is None:
        return {"error": 'Model not loaded'}
    
    if _index_impl is None:
        if _dim is None:
            return {"error": 'Model dimension not set'}
        ensure_index_impl(_dim)
    
    start = time.time()
    
    try:
        with _encode_semaphore:
            qvec = model.encode(
                [q],
                convert_to_numpy=True, 
                batch_size=1,
                show_progress_bar=False
            )
            qvec = np.array(qvec)
            qvec = _index_impl.l2_normalize_np(qvec).astype('float32')
            enc_ms = (time.time() - start) * 1000.0
    except Exception as e:
        logger.exception('Encode error: %s', e)
        return {"error": 'Failed to encode query'}
    
    with _index_lock:
        if hasattr(_index_impl.index, 'nprobe'):
            original_nprobe = _index_impl.index.nprobe
            _index_impl.index.nprobe = settings.FAISS_NPROBE
        
        D, I = _index_impl.search(qvec, topk)
        
        if hasattr(_index_impl.index, 'nprobe'):
            _index_impl.index.nprobe = original_nprobe
            
        results: List[SearchResult] = []
        
        if I.shape[1] == 0:
            return []
        
        for score, idx in zip(D[0], I[0]):
            if int(idx) == -1:
                continue
            pid = _mapping.get(int(idx))
            if pid is None or pid in _tombstones:
                continue
            norm_score = float((float(score) + 1.0) / 2.0)
            results.append(SearchResult(product_id=pid, similarity_score=norm_score))
    
    _query_cache.put(q, topk, results)
    
    query_ms = (time.time() - start) * 1000.0
    logger.info(f"Query took {query_ms:.2f}ms")
    
    _metrics['queries'] += 1
    _metrics['avg_query_ms'] = (_metrics['avg_query_ms'] * (_metrics['queries'] - 1) + query_ms) / _metrics['queries']
    
    return [r.to_dict() for r in results]

def embeddings_endpoint(payload):
    model = get_model_safe()
    if model is None:
        return {"error": 'Model not loaded'}
    
    texts: List[str] = []
    
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
    
    if len(texts) > 100:
        texts = texts[:100]
    
    try:
        embeddings = []
        start_time = time.time()
        
        for i in range(0, len(texts), settings.BATCH_SIZE):
            batch = texts[i:i + settings.BATCH_SIZE]
            with _encode_semaphore:
                vecs = model.encode(
                    batch, 
                    batch_size=len(batch), 
                    convert_to_tensor=False,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
                embeddings.extend(vecs.tolist())
        
        encode_time = (time.time() - start_time) * 1000
        logger.info(f"Encoded {len(texts)} texts in {encode_time:.2f}ms")
        
        _metrics['encodes'] += len(texts)
        _metrics['avg_encode_ms'] = (_metrics['avg_encode_ms'] * (_metrics['encodes'] - len(texts)) + encode_time) / _metrics['encodes']
        
    except Exception:
        logger.exception('Embedding encode failed')
        return {"error": 'Failed to compute embeddings'}
    
    return {
        "embeddings": embeddings,
        "model": _model_meta.get('name', 'unknown'),
        "count": len(embeddings)
    }


def reindex_items(items: Any):
    """Encode and persist product records for the local search index."""
    global _next_id_counter
    if not isinstance(items, list) or not items:
        return {"error": "Expected a non-empty list of product records"}
    if len(items) > 10_000:
        return {"error": "A reindex request may contain at most 10000 records"}

    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            return {"error": "Each record must be an object"}
        product_id = str(item.get("product_id", "")).strip()
        text = str(item.get("text", "")).strip()
        if not product_id or not text:
            return {"error": "Each record requires product_id and text"}
        if product_id in seen:
            return {"error": f"Duplicate product_id: {product_id}"}
        seen.add(product_id)
        records.append((product_id, text[:settings.MAX_QUERY_LENGTH]))

    model = get_model_safe()
    if model is None:
        return {"error": "Model not loaded"}
    if _dim is None:
        return {"error": "Model dimension not set"}
    ensure_index_impl(_dim)

    try:
        vectors = np.asarray(
            model.encode(
                [text for _, text in records],
                convert_to_numpy=True,
                batch_size=min(settings.BATCH_SIZE, len(records)),
                show_progress_bar=False,
                normalize_embeddings=True,
            ),
            dtype="float32",
        )
        ids = np.arange(_next_id_counter, _next_id_counter + len(records), dtype="int64")
        _index_impl.add_with_ids(vectors, ids)
    except Exception:
        logger.exception("Reindex failed")
        return {"error": "Failed to encode or persist records"}

    for product_id, item_id in zip((product_id for product_id, _ in records), ids):
        _mapping[int(item_id)] = product_id
        _rev_map[product_id] = int(item_id)
    _next_id_counter += len(records)
    save_mapping(_mapping)
    save_manifest()
    return {"count": len(records), "index_size": _index_impl.ntotal(), "model": _model_meta}

# ------------------ 
# Ping thread
# ------------------ 
def start_ping_thread():
    global _ping_thread, _ping_running
    
    if not settings.PING_URL:
        logger.info("No PING_URL configured, skipping ping thread")
        return
    
    _ping_running = True
    
    def ping_worker():
        while _ping_running:
            try:
                response = requests.get(settings.PING_URL, timeout=10)
                logger.info(f"Ping sent, status: {response.status_code}")
            except Exception as e:
                logger.error(f"Ping failed: {e}")
            time.sleep(settings.PING_INTERVAL)
    
    _ping_thread = threading.Thread(target=ping_worker, daemon=True)
    try:
        add_script_run_ctx(_ping_thread)
    except Exception:
        pass
    _ping_thread.start()
    logger.info("Ping thread started")

def stop_ping_thread():
    global _ping_running
    _ping_running = False
    if _ping_thread:
        _ping_thread.join(timeout=5.0)
    logger.info("Ping thread stopped")

# ------------------ 
# FastAPI server (embedded) - Enhanced CORS handling
# ------------------ 
_api_thread = None
_api_app: Optional[FastAPI] = None

def _create_api_app() -> Optional[FastAPI]:
    if not _HAS_UVICORN:
        logger.warning("uvicorn/fastapi not available - API server will not start")
        return None

    app = FastAPI(title="Model Server API (with Enhanced CORS)")

    configured_origins = [value.strip() for value in settings.ALLOWED_ORIGINS.split(",") if value.strip()]
    allow_origins = configured_origins or ["*"]
    allow_credentials = settings.ALLOW_CREDENTIALS and allow_origins != ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=[value.strip() for value in settings.ALLOWED_METHODS.split(",") if value.strip()] or ["*"] ,
        allow_headers=[value.strip() for value in settings.ALLOWED_HEADERS.split(",") if value.strip()] or ["*"] ,
        max_age=600,
    )

    # Add explicit OPTIONS handler for preflight requests
    @app.options("/{path:path}")
    async def universal_options_handler(path: str):
        return Response(status_code=200)

    @app.get("/api/health")
    async def api_health():
        return JSONResponse(content=health())

    @app.get("/api/search")
    async def api_search(q: Optional[str] = None, topk: int = 10):
        if not q:
            return JSONResponse(content={"error": "Missing query parameter 'q'"}, status_code=400)
        res = search(q, topk)
        return JSONResponse(content=res)

    @app.post("/api/search")
    async def api_search_post(req: Request):
        try:
            body = await req.json()
            q = body.get("q") or body.get("query") or body.get("text")
            topk = int(body.get("topk", 10))
            if not q:
                return JSONResponse(content={"error": "Missing 'q' in body"}, status_code=400)
            res = search(q, topk)
            return JSONResponse(content=res)
        except Exception as e:
            logger.error(f"Error in api_search_post: {e}")
            return JSONResponse(content={"error": "Invalid JSON payload"}, status_code=400)

    @app.post("/api/embeddings")
    async def api_embeddings(req: Request):
        try:
            body = await req.json()
            res = embeddings_endpoint(body)
            return JSONResponse(content=res)
        except Exception as e:
            logger.error(f"Error in api_embeddings: {e}")
            return JSONResponse(content={"error": "Invalid JSON payload"}, status_code=400)

    @app.post("/api/reindex")
    @app.post("/reindex")
    async def api_reindex(req: Request):
        try:
            body = await req.json()
            items = body.get("items") if isinstance(body, dict) and "items" in body else body
            result = reindex_items(items)
            status = 400 if "error" in result else 200
            return JSONResponse(content=result, status_code=status)
        except Exception:
            logger.exception("Error in reindex request")
            return JSONResponse(content={"error": "Invalid reindex payload"}, status_code=400)

    # Additional endpoint for direct CORS testing
    @app.get("/api/cors-test")
    async def cors_test():
        return JSONResponse(content={"message": "CORS is working!", "timestamp": time.time()})

    return app


# Export the ASGI application used by the Procfile and local uvicorn runs.
app = _create_api_app()

def start_api_server():
    global _api_thread, _api_app
    if not _HAS_UVICORN:
        logger.warning("Uvicorn not installed; skipping API startup")
        return

    if _api_thread and _api_thread.is_alive():
        logger.info("API server already running")
        return

    # Only start API server if enabled in settings
    if not settings.ENABLE_API_SERVER:
        logger.info("API server is disabled in settings")
        return

    _api_app = _create_api_app()
    if _api_app is None:
        return

    def run_uvicorn():
        try:
            config = uvicorn.Config(
                _api_app, 
                host=settings.API_HOST, 
                port=settings.API_PORT, 
                log_level="info",
                timeout_keep_alive=300
            )
            server = uvicorn.Server(config)
            server.run()
        except Exception as e:
            logger.exception("Failed to run uvicorn: %s", e)

    _api_thread = threading.Thread(target=run_uvicorn, daemon=True, name="API-Server")
    try:
        add_script_run_ctx(_api_thread)
    except Exception:
        pass
    _api_thread.start()
    logger.info("Started embedded FastAPI server on %s:%d (Enhanced CORS enabled)", settings.API_HOST, settings.API_PORT)

# ------------------ 
# Streamlit UI (preserving your UI)
# ------------------ 
def main():
    st.set_page_config(
        page_title="Ultra Fast Semantic Search Server",
        page_icon="🔍",
        layout="wide"
    )

    st.title("Ultra Fast Semantic Search Server (Streamlit + API)")

    if 'initialized' not in st.session_state:
        startup_event()
        start_ping_thread()
        # Start the API server (so external frontends get CORS-enabled endpoints)
        start_api_server()
        st.session_state.initialized = True

    tab1, tab2, tab3 = st.tabs(["Search", "Embeddings", "Health"])

    with tab1:
        st.header("Semantic Search")
        with st.form("search_form"):
            query = st.text_input("Search query", placeholder="Enter your search query...")
            topk = st.slider("Number of results", min_value=1, max_value=100, value=10)
            submitted = st.form_submit_button("Search")
            if submitted and query:
                with st.spinner("Searching..."):
                    results = search(query, topk)
                    if isinstance(results, dict) and "error" in results:
                        st.error(results["error"])
                    else:
                        st.success(f"Found {len(results)} results")
                        for i, result in enumerate(results):
                            st.write(f"{i+1}. {result['product_id']} (Score: {result['similarity_score']:.4f})")

    with tab2:
        st.header("Embeddings Generation")
        with st.form("embeddings_form"):
            input_type = st.radio("Input type", ["Text", "List of texts"])
            if input_type == "Text":
                text_input = st.text_area("Enter text")
                payload = text_input
            else:
                texts_input = st.text_area("Enter texts (one per line)")
                payload = texts_input.split("\n") if texts_input else []
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
        st.header("System Health")
        if st.button("Refresh Health Status"):
            pass
        health_status = health()
        st.json(health_status)
        st.subheader("Metrics")
        st.metric("Total Queries", _metrics['queries'])
        st.metric("Total Encodes", _metrics['encodes'])
        st.metric("Avg Query Time", f"{_metrics['avg_query_ms']:.2f} ms")
        st.metric("Avg Encode Time", f"{_metrics['avg_encode_ms']:.2f} ms")
        st.subheader("Ping Status")
        if settings.PING_URL:
            st.info(f"Pinging {settings.PING_URL} every {settings.PING_INTERVAL} seconds")
        else:
            st.warning("PING_URL not configured - app may sleep when inactive")
        
        # API server status
        st.subheader("API Server Status")
        if settings.ENABLE_API_SERVER:
            st.info("API Server: Enabled")
            if _api_thread and _api_thread.is_alive():
                st.success("API Server: Running")
            else:
                st.warning("API Server: Not Running")
        else:
            st.warning("API Server: Disabled (set ENABLE_API_SERVER=True to enable)")

# ------------------ 
# Startup
# ------------------ 
def startup_event():
    global _mapping, _rev_map, _next_id_counter, _dim
    os.makedirs(INDEX_DIR, exist_ok=True)
    load_tombstones()
    _mapping = load_mapping(MAPPING_PATH)
    _rev_map = {v: k for k, v in _mapping.items()}
    manifest = load_manifest() or {}
    _next_id_counter = int(manifest.get('next_id_counter', max(_mapping.keys()) + 1 if _mapping else 0))
    
    model = get_model_safe()
    
    if _dim is None:
        _dim = 384  # default for all-MiniLM-L6-v2
    
    if model is not None:
        try:
            ensure_index_impl(_dim)
        except Exception:
            logger.exception("Failed to initialize index")
    else:
        logger.warning("Model not loaded, index initialization skipped")
    
    try:
        save_manifest()
    except Exception:
        logger.exception('Failed to save manifest on startup')
    
    logger.info('Startup complete. model_loaded=%s index_size=%d', 
                _model is not None, 
                _index_impl.ntotal() if _index_impl else 0)

# ------------------ 
# Entrypoint
# ------------------ 
if __name__ == "__main__":
    main()
