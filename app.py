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

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('semantic_search_streamlit')

# Initialize optional libraries
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
    logger.warning("FAISS not available")

try:
    from sentence_transformers import SentenceTransformer
    logger.info("sentence-transformers loaded successfully")
except ImportError:
    logger.error("sentence-transformers not available - this is required!")

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
# CONFIG
# ------------------ 
class Settings:
    # Use your existing model directory - fixed path
    LOCAL_MODEL_DIR: str = 'model/all-MiniLM-L6-v2'
    MODELS_ROOT: str = 'model'
    INDEX_DIR: str = 'index'
    HNSW_M: int = 32
    BATCH_SIZE: int = 256
    ENCODE_CONCURRENCY: int = 16
    USE_ONNX: bool = False  # Disable ONNX for simplicity
    MAX_QUERY_LENGTH: int = 128
    PRELOAD_MODEL: bool = True
    WARMUP_MODEL: bool = True
    SEARCH_CACHE_ENABLED: bool = True
    FAISS_NPROBE: int = 8
    CACHE_TTL_SECONDS: int = 300
    # Ping settings
    PING_INTERVAL: int = 10  # Ping every 10 seconds
    PING_URL: str = os.environ.get("PING_URL", "")  # URL to ping

settings = Settings()

# File paths - using absolute paths for Streamlit Cloud compatibility
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
            if time.time() - self.last_cleanup > self.ttl_seconds:
                self._cleanup()
                
            if key in self.cache:
                cached = self.cache[key]
                cached.access_count += 1
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

# Optimization globals
_query_cache = QueryCache(max_size=10000, ttl_seconds=settings.CACHE_TTL_SECONDS)
_last_gc_run = time.time()

# Ping thread to keep app awake
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
# Index abstractions (Faiss)
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
        
        # Fixed the typo: hnsw_M -> hnsw_m
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
# Model loading
# ------------------ 
def _load_model_from(path: str):
    global _model, _model_meta, _dim
    if SentenceTransformer is None:
        raise RuntimeError('sentence-transformers library not available in environment')
    
    logger.info('Loading model from %s', path)
    
    # Check if model directory exists
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
            warmup_texts = ['hello world'] * 4  # Smaller batch for Streamlit
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
        
        # Use the model path from your repository
        model_dir = os.path.join(BASE_DIR, settings.LOCAL_MODEL_DIR)
        if not os.path.exists(model_dir):
            logger.error('Model not found at: %s', model_dir)
            # Try to find any model in the models directory
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
            _index_impl = FaissIndex(dim=dim, path=FAISS_INDEX_PATH, hnsw_m=settings.HNSW_M)
        else:
            raise RuntimeError("FAISS is required")
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

def search(q: str, topk: int = 10):
    if not q:
        return {"error": "Missing query parameter"}
    
    cached = _query_cache.get(q, topk)
    if cached is not None:
        return cached
    
    model = get_model_safe()
    if model is None:
        return {"error": 'Model not loaded'}
    
    # Ensure index is initialized
    if _index_impl is None:
        if _dim is None:
            return {"error": 'Model dimension not set'}
        ensure_index_impl(_dim)
    
    start = time.time()
    
    try:
        with _encode_semaphore:
            truncated_q = q[:settings.MAX_QUERY_LENGTH] if len(q) > settings.MAX_QUERY_LENGTH else q
            qvec = model.encode(
                [truncated_q], 
                convert_to_numpy=True, 
                batch_size=1,
                show_progress_bar=False
            )
            qvec = np.array(qvec)
            qvec = _index_impl.l2_normalize_np(qvec).astype('float32')
            enc_ms = (time.time() - start) * 1000.0
    except Exception as e:
        logger.exception('Encode error: %s', e)
        return {"error": f'Failed to encode query: {e}'}
    
    with _index_lock:
        if hasattr(_index_impl.index, 'nprobe'):
            original_nprobe = _index_impl.index.nprobe
            _index_impl.index.nprobe = settings.FAISS_NPROBE
        
        D, I = _index_impl.search(qvec, topk)
        
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
            norm_score = float((float(score) + 1.0) / 2.0)
            results.append(SearchResult(product_id=pid, similarity_score=norm_score))
    
    _query_cache.put(q, topk, results)
    
    query_ms = (time.time() - start) * 1000.0
    logger.info(f"Query took {query_ms:.2f}ms")
    
    _metrics['queries'] += 1
    _metrics['avg_query_ms'] = (_metrics['avg_query_ms'] * (_metrics['queries'] - 1) + query_ms) / _metrics['queries']
    
    return results

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

# ------------------ 
# FastAPI Endpoints (for external access)
# ------------------ 
# This section is commented out as Streamlit doesn't support running FastAPI alongside it
# For production, consider deploying a separate FastAPI service

# ------------------ 
# Ping functionality to keep app awake
# ------------------ 
def start_ping_thread():
    """Start a thread that pings the app regularly to keep it awake"""
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
    add_script_run_ctx(_ping_thread)  # Make thread compatible with Streamlit
    _ping_thread.start()
    logger.info("Ping thread started")

def stop_ping_thread():
    """Stop the ping thread"""
    global _ping_running
    _ping_running = False
    if _ping_thread:
        _ping_thread.join(timeout=5.0)
    logger.info("Ping thread stopped")

# ------------------ 
# Streamlit UI
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
        start_ping_thread()  # Start ping thread
        st.session_state.initialized = True
    
    # Create tabs for different functionalities
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
                    if "error" in results:
                        st.error(results["error"])
                    else:
                        st.success(f"Found {len(results)} results")
                        for i, result in enumerate(results):
                            st.write(f"{i+1}. {result.product_id} (Score: {result.similarity_score:.4f})")
    
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
        
        # Display ping status
        st.subheader("Ping Status")
        if settings.PING_URL:
            st.info(f"Pinging {settings.PING_URL} every {settings.PING_INTERVAL} seconds")
        else:
            st.warning("PING_URL not configured - app may sleep when inactive")

# ------------------ 
# Startup and initialization
# ------------------ 
def startup_event():
    global _mapping, _rev_map, _next_id_counter, _dim
    os.makedirs(INDEX_DIR, exist_ok=True)
    load_tombstones()
    _mapping = load_mapping(MAPPING_PATH)
    _rev_map = {v: k for k, v in _mapping.items()}
    manifest = load_manifest() or {}
    _next_id_counter = int(manifest.get('next_id_counter', max(_mapping.keys()) + 1 if _mapping else 0))
    
    # Load model - this will use your existing model directory
    model = get_model_safe()
    
    # Ensure index exists
    if _dim is None:
        _dim = 384  # Default dimension for all-MiniLM-L6-v2
    
    # Initialize index only if model is loaded
    if model is not None:
        ensure_index_impl(_dim)
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
# Main entry point
# ------------------ 
if __name__ == "__main__":
    main()
