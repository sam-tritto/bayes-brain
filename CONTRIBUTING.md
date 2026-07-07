# Contributing to BayesianCortex

Thank you for your interest in contributing to **BayesianCortex**! We want to make contributing to this project as easy, structured, and rewarding as possible. 

Here is our strategic playbook for understanding the codebase architecture, setting up your environment, and successfully getting your changes merged.

---

## 🗺️ 1. Strategic "Extension Hooks" (Modular Architecture)

BayesianCortex is designed around modular, decoupled components. Instead of rewriting the core Thompson Sampling engine, you can build self-contained features using our predefined **Protocols** and **Base Classes**.

### Custom Embedders
Want to integrate a niche embedding model (e.g., Cohere, local Llama.cpp, etc.)? Implement the `ContextEmbedder` (synchronous) or `AsyncContextEmbedder` (asynchronous) protocols found in [src/bayesian_cortex/embeddings.py](file:///Users/sam/Locals%20Only/bayesian-cortex/src/bayesian_cortex/embeddings.py):
```python
from typing import List
from bayesian_cortex.embeddings import ContextEmbedder

class MyCustomEmbedder(ContextEmbedder):
    def embed_query(self, text: str) -> List[float]:
        # Implement your 15-line embedding call here
        pass
```

### New Storage Backends
Using DuckDB, MongoDB, or a custom internal store? Subclass `BaseStorage` (sync) or `AsyncBaseStorage` (async) in [src/bayesian_cortex/storage.py](file:///Users/sam/Locals%20Only/bayesian-cortex/src/bayesian_cortex/storage.py) and implement the abstract methods for parameters updating and telemetry logging.

### New Vector Stores
Need to support PGVector, Chroma, or Qdrant? Implement `VectorStoreProtocol` or `AsyncVectorStoreProtocol` from [src/bayesian_cortex/embeddings.py](file:///Users/sam/Locals%20Only/bayesian-cortex/src/bayesian_cortex/embeddings.py).

---

## ⚡ 2. Quickstart Developer Setup

Setting up your environment takes less than two minutes thanks to `uv`.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sam-tritto/bayesian-cortex.git
   cd bayesian-cortex
   ```
2. **Install dependencies and create a virtual environment:**
   ```bash
   # Syncs packages and installs dev dependencies / extras
   uv sync --all-extras
   ```
3. **Run the test suite:**
   ```bash
   uv run pytest
   ```

---

## 🏷️ 3. Good First Issues (with a blueprint)

When looking for issues to contribute to, check out our issue tracker. We prioritize writing **blueprints** directly inside issues labeled `good first issue` to eliminate cognitive overhead:

> ### 💡 Example Blueprint Format:
> **Good First Issue:** Implement Exponential Backoff for Remote Embedders
> - **Where to look:** [src/bayesian_cortex/embeddings.py](file:///Users/sam/Locals%20Only/bayesian-cortex/src/bayesian_cortex/embeddings.py) inside the `OpenAIEmbedder` or `GeminiEmbedder` class.
> - **The Goal:** Wrap the `httpx` HTTP call in a retry loop.
> - **Suggested Approach:** Refer to how we handle SQLite write-locks using jittered backoff in [src/bayesian_cortex/storage.py](file:///Users/sam/Locals%20Only/bayesian-cortex/src/bayesian_cortex/storage.py) for inspiration.

---

## 🚀 4. Unfinished Roadmap

Want to take on a larger, high-impact feature? Check out our long-term roadmap and claim ownership of a component by opening an issue/PR draft:

- [ ] **GraphRAG Support:** Route decisions based on structural context arrays.
- [ ] **Asynchronous Redis Cluster:** Multi-node state synchronization.
- [ ] **JAX/Triton Accelerated Math:** Optimize high-dimensional Linear Thompson Sampling matrix inversions.
- [ ] **Context-Aware Dynamic Epsilon:** Dynamically adjust exploration weights based on contextual drift velocity.

---

## 📜 5. Submission Guidelines

1. **Keep it focused:** Try to make one PR per logical change.
2. **Write tests:** Add corresponding unit tests in the `tests/` directory.
3. **Format & Type Check:** Ensure your code is clean and passes static analysis:
   ```bash
   uv run black .
   uv run mypy src/
   ```
