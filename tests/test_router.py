from typing import Optional, Sequence

import pytest

from bayes_brain.embeddings import VectorContextStore
from bayes_brain.router import BayesianToolRouter
from bayes_brain.storage import InMemoryStorage


def test_vector_context_store():
    store = VectorContextStore()
    
    # Add contexts
    store.add_context("ctx_search", [1.0, 0.0, 0.0])
    store.add_context("ctx_math", [0.0, 1.0, 0.0])

    # Query with exact match
    assert store.get_nearest_context([1.0, 0.0, 0.0], 0.9) == "ctx_search"
    assert store.get_nearest_context([0.0, 1.0, 0.0], 0.9) == "ctx_math"

    # Query with close match
    assert store.get_nearest_context([0.9, 0.1, 0.0], 0.8) == "ctx_search"
    
    # Query with no match (below threshold)
    assert store.get_nearest_context([0.5, 0.5, 0.0], 0.95) is None


class MockEmbedder:
    def embed_query(self, text: str) -> Sequence[float]:
        # Simple rule: if "search" in text -> [1, 0], if "math" -> [0, 1]
        if "search" in text.lower():
            return [1.0, 0.0]
        return [0.0, 1.0]


def test_router_without_embeddings():
    storage = InMemoryStorage()
    router = BayesianToolRouter(storage=storage, decay_factor=0.95)

    # Route should select from candidate tools
    tool = router.route("web_search_query", ["search_api", "fallback_api"])
    assert tool in ["search_api", "fallback_api"]

    # Provide feedback
    router.feedback("web_search_query", "search_api", success=True)
    a_success, b_success = storage.get_tool_params("web_search_query", "search_api")
    # Initial was (1, 1). Decayed: alpha = 1 * 0.95 + 1.0 = 1.95, beta = 1 * 0.95 + 0.0 = 0.95
    assert a_success == pytest.approx(1.95)
    assert b_success == pytest.approx(0.95)

    # Provide failure feedback
    router.feedback("web_search_query", "search_api", success=False)
    a_fail, b_fail = storage.get_tool_params("web_search_query", "search_api")
    # Decayed: alpha = 1.95 * 0.95 + 0 = 1.8525, beta = 0.95 * 0.95 + 1.0 = 1.9025
    assert a_fail == pytest.approx(1.8525)
    assert b_fail == pytest.approx(1.9025)


def test_router_with_embeddings():
    storage = InMemoryStorage()
    embedder = MockEmbedder()
    router = BayesianToolRouter(
        storage=storage,
        embedder=embedder,
        decay_factor=1.0,
        similarity_threshold=0.85
    )

    # First routing creates a new cluster context key
    tool_1, trace_1 = router.route_with_trace("find math help", ["tool_math", "tool_search"])
    # Resolve context text to a context key
    context_key_1 = router._resolve_context_key("find math help")
    assert context_key_1.startswith("ctx_")

    # Routing a similar text matches the same context cluster
    context_key_2 = router._resolve_context_key("do some math stuff")
    assert context_key_1 == context_key_2

    # Routing a search text yields a different cluster context key
    context_key_search = router._resolve_context_key("do search things")
    assert context_key_1 != context_key_search


def test_router_trace_feedback():
    storage = InMemoryStorage()
    router = BayesianToolRouter(storage=storage)

    chosen_tool, trace_id = router.route_with_trace("context_a", ["tool_x"])
    assert chosen_tool == "tool_x"

    # Feedback using trace ID
    router.feedback_by_trace(trace_id, success=True)
    
    alpha, beta = storage.get_tool_params("context_a", "tool_x")
    # (1*1 + 1) = 2.0, (1*1 + 0) = 1.0
    assert alpha == 2.0
    assert beta == 1.0


def test_router_priors_seeding():
    storage = InMemoryStorage()
    priors = {
        "highly_reliable": (90.0, 10.0),
        "unreliable": (1.0, 99.0)
    }
    router = BayesianToolRouter(storage=storage, priors=priors)

    # Highly reliable should be preferred
    chosen = router.route("some_task", ["highly_reliable", "unreliable"])
    assert chosen == "highly_reliable"

    # Verify storage contains the seeded priors
    a_rel, b_rel = storage.get_tool_params("some_task", "highly_reliable")
    assert a_rel == 90.0
    assert b_rel == 10.0


class CustomMemoryVectorStore:
    def __init__(self) -> None:
        self.vectors = {}

    def add_context(self, context_key: str, vector: Sequence[float]) -> None:
        self.vectors[context_key] = vector

    def get_nearest_context(
        self, query_vector: Sequence[float], similarity_threshold: float = 0.8
    ) -> Optional[str]:
        if self.vectors:
            return list(self.vectors.keys())[0]
        return None


def test_router_with_custom_vector_store():
    storage = InMemoryStorage()
    embedder = MockEmbedder()
    custom_store = CustomMemoryVectorStore()
    
    router = BayesianToolRouter(
        storage=storage,
        embedder=embedder,
        vector_store=custom_store
    )
    
    # Initially empty
    assert len(custom_store.vectors) == 0
    
    # Resolving context text creates a new cluster context key
    context_key = router._resolve_context_key("some search query")
    assert context_key.startswith("ctx_")
    
    # Verify custom_store has the new context key stored
    assert context_key in custom_store.vectors
    assert custom_store.get_nearest_context([1.0, 0.0]) == context_key
