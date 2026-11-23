                          Semantic Search Server - Overview

This is a high-performance semantic search server built with FastAPI that provides text similarity search functionality using local machine learning models.This server loads a local machine learning model (either ONNX-optimized or standard SentenceTransformer) to convert text into numerical vectors. When you add product data, it encodes the text into vectors and stores them in an efficient index (FAISS for fastest search, or NumPy as fallback). For search queries, it encodes the search text into a vector, compares it against all stored vectors using cosine similarity, and returns the closest matches with similarity scores. The system maintains consistency between stored texts and vectors, handles updates/deletions via tombstones, and persists all data to disk for reliability.The code is designed to use ONNX for faster inference if it is available and enabled (which is the default setting). If ONNX is not available or fails to load, it falls back to using the SentenceTransformer library. So, depending on your environment and configuration, it could be using either, but ONNX is preferred for performance.

Key Features

              FastAPI-based REST API with endpoints for search, embeddings, and administrative functions
              Dual backend support: FAISS (GPU/CPU) and fallback NumPy-based index
              ONNX Runtime integration for accelerated inference
              Efficient caching with query and vector caching
              Background jobs system for long-running operations
              Automatic compaction to clean up deleted items
              Rate limiting and basic authentication support
              File-based persistence with atomic writes

Core Components

                  Model Management: Supports SentenceTransformers and ONNX models with warmup and reload capabilities
                  Indexing System: FAISS (preferred) or an exact NumPy cosine-search fallback for local/small indexes

API Endpoints:

                /search - Main semantic search endpoint
                /embeddings - Text embedding generation endpoint

Admin endpoints for indexing, updates, and system management

Performance Optimizations
                          Batch processing of queries and embeddings
                          Thread pooling for concurrent encoding
                          Memmapped arrays for efficient memory usage
                          LRU caching for frequent queries
                          Preallocated buffers to reduce memory allocation overhead

Step-by-Step Setup Instructions


        1. Prerequisites
                            Python 3.8 or higher installed
                            Git installed (to clone your project if needed)

        2. Setup Steps (Windows & Linux)
                          
                          Step 1: Download or create your project folder

                                      # If using git
                                      git clone <your-repository-url>
                                      cd <your-project-folder>

   Step 2: Create and activate virtual environment

      For Windows:

                    python -m venv venv
                    venv\Scripts\activate
      
      For Linux/Mac:

                    python3 -m venv venv
                    source venv/bin/activate


   Step 3: Install dependencies

                  pip install -r requirements.txt

   Step 4: Run the server

    For Windows:
                uvicorn app:app --host 0.0.0.0 --port 9000 --reload
    For Linux:
                uvicorn app:app --host 0.0.0.0 --port 9000 --reload

3. Verify Installation

        Open your browser and go to: http://localhost:9000/docs

        Test the health endpoint: http://localhost:9000/health

4. Adding Some Data (Example)

Use curl or the API docs to add sample data:

    curl -X POST "http://localhost:9000/reindex" -H "Content-Type: application/json" -d '[
          {"product_id": "1", "text": "wireless Bluetooth headphones with noise cancellation"},
          {"product_id": "2", "text": "gaming mechanical keyboard with RGB lighting"},
          {"product_id": "3", "text": "smartphone with high-resolution camera and 5G"}
        ]'
5. Test Search

    curl -X GET "http://localhost:9000/search?q=wireless%20headphones&topk=3"




![alt text](<semantic_search_diagram (1).png>)

## Security and runtime files

Configuration is loaded from a local `.env` file. Start from
[`.env.example`](.env.example), keep `API_KEY` empty until a local credential
is configured, and never commit `.env`. The index manifests, tombstones,
Python bytecode, and `server.log` are runtime outputs and are ignored.

The repository history previously contained a non-empty `API_KEY` in `.env`.
Assume that credential is exposed: revoke and rotate it with the issuing
service before using the server, and audit any deployment that used it. This
working-tree cleanup removes the file from the current branch but does not
rewrite history.
