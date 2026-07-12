from typing import List, Sequence

import pytest

from bayesian_cortex.exceptions import OutlierContextError
from bayesian_cortex.router import AsyncBayesianRouter, BayesianRouter
from bayesian_cortex.storage import AsyncInMemoryStorage, InMemoryStorage


class Mock3DEmbedder:
    def embed_query(self, text: str) -> Sequence[float]:
        # Maps specific prompts to 3D unit vectors
        if "search" in text.lower():
            return [1.0, 0.0, 0.0]
        elif "math" in text.lower():
            return [0.0, 1.0, 0.0]
        else:
            # Completely orthogonal/alien OOD vector
            return [0.0, 0.0, 1.0]

    async def aembed_query(self, text: str) -> Sequence[float]:
        return self.embed_query(text)

    def embed_queries(self, texts: List[str]) -> List[Sequence[float]]:
        return [self.embed_query(t) for t in texts]

    async def aembed_queries(self, texts: List[str]) -> List[Sequence[float]]:
        return [self.embed_query(t) for t in texts]


def test_bootstrap_empty_store_not_outlier():
    # When store is empty, first query shouldn't be an outlier
    embedder = Mock3DEmbedder()
    router = BayesianRouter(
        storage=InMemoryStorage(),
        embedder=embedder,
        similarity_threshold=0.8,
        outlier_threshold=0.4,
        fallback_candidate="fallback_arm",
    )

    # Empty store -> first query bootstraps a context key ctx_<uuid>
    choice, trace_id = router.route_with_trace("first search query", ["arm1", "arm2"])
    assert choice in ["arm1", "arm2"]

    # Verify a context key was created in store
    contexts = list(router._context_store._contexts.keys())
    assert len(contexts) == 1
    assert contexts[0].startswith("ctx_")


def test_sync_outlier_fallback():
    embedder = Mock3DEmbedder()
    router = BayesianRouter(
        storage=InMemoryStorage(),
        embedder=embedder,
        similarity_threshold=0.8,
        outlier_threshold=0.4,
        fallback_candidate="fallback_arm",
    )

    # 1. Bootstrap the store with an in-distribution query
    router.route("search query", ["arm1", "arm2", "fallback_arm"])

    # 2. Route an alien query (OOD)
    # The Mock3DEmbedder maps "alien" to [0, 0, 1]. Max similarity to [1, 0, 0] is 0.0.
    # 0.0 < outlier_threshold (0.4) -> should trigger outlier fallback to fallback_arm
    choice, trace_id = router.route_with_trace(
        "alien query", ["arm1", "arm2", "fallback_arm"]
    )
    assert choice == "fallback_arm"
    # The logged context key in trace ID should be __outlier_fallback__
    ctx_key, cand = router._decode_trace_id(trace_id)
    assert ctx_key == "__outlier_fallback__"
    assert cand == "fallback_arm"


def test_sync_outlier_raise():
    embedder = Mock3DEmbedder()
    router = BayesianRouter(
        storage=InMemoryStorage(),
        embedder=embedder,
        similarity_threshold=0.8,
        outlier_threshold=0.4,
        outlier_fallback_behavior="raise",
    )

    # Bootstrap
    router.route("search query", ["arm1", "arm2"])

    # OOD query should raise OutlierContextError
    with pytest.raises(OutlierContextError):
        router.route("alien query", ["arm1", "arm2"])


@pytest.mark.anyio
async def test_async_outlier_fallback():
    embedder = Mock3DEmbedder()
    router = AsyncBayesianRouter(
        storage=AsyncInMemoryStorage(),
        embedder=embedder,
        similarity_threshold=0.8,
        outlier_threshold=0.4,
        fallback_candidate="fallback_arm",
    )

    # Empty store bootstrap
    choice1, _ = await router.aroute_with_trace("first search query", ["arm1", "arm2"])
    assert choice1 in ["arm1", "arm2"]

    # OOD query fallback
    choice2, trace_id2 = await router.aroute_with_trace(
        "alien query", ["arm1", "arm2", "fallback_arm"]
    )
    assert choice2 == "fallback_arm"
    ctx_key, cand = router._decode_trace_id(trace_id2)
    assert ctx_key == "__outlier_fallback__"
    assert cand == "fallback_arm"


@pytest.mark.anyio
async def test_async_outlier_raise():
    embedder = Mock3DEmbedder()
    router = AsyncBayesianRouter(
        storage=AsyncInMemoryStorage(),
        embedder=embedder,
        similarity_threshold=0.8,
        outlier_threshold=0.4,
        outlier_fallback_behavior="raise",
    )

    # Bootstrap
    await router.aroute("search query", ["arm1", "arm2"])

    # OOD query should raise OutlierContextError
    with pytest.raises(OutlierContextError):
        await router.aroute("alien query", ["arm1", "arm2"])


def test_batch_outlier_fallback():
    embedder = Mock3DEmbedder()
    router = BayesianRouter(
        storage=InMemoryStorage(),
        embedder=embedder,
        similarity_threshold=0.8,
        outlier_threshold=0.4,
        fallback_candidate="fallback_arm",
    )

    # Bootstrap the store with "search"
    router.route("search query", ["arm1", "fallback_arm"])

    # Batch of contexts: one in-distribution (search), one outlier (alien)
    results = router.route_batch_with_trace(
        ["similar search query", "completely alien query"],
        candidates=["arm1", "fallback_arm"],
    )

    assert len(results) == 2
    # First query should route normally
    assert results[0][0] in ["arm1", "fallback_arm"]
    ctx_key1, _ = router._decode_trace_id(results[0][1])
    assert ctx_key1 != "__outlier_fallback__"

    # Second query is an outlier and should route to fallback_arm
    assert results[1][0] == "fallback_arm"
    ctx_key2, _ = router._decode_trace_id(results[1][1])
    assert ctx_key2 == "__outlier_fallback__"


@pytest.mark.anyio
async def test_async_batch_outlier_fallback():
    embedder = Mock3DEmbedder()
    router = AsyncBayesianRouter(
        storage=AsyncInMemoryStorage(),
        embedder=embedder,
        similarity_threshold=0.8,
        outlier_threshold=0.4,
        fallback_candidate="fallback_arm",
    )

    # Bootstrap
    await router.aroute("search query", ["arm1", "fallback_arm"])

    # Batch of mixed queries
    results = await router.aroute_batch_with_trace(
        ["similar search query", "completely alien query"],
        candidates=["arm1", "fallback_arm"],
    )

    assert len(results) == 2
    assert results[0][0] in ["arm1", "fallback_arm"]
    ctx_key1, _ = router._decode_trace_id(results[0][1])
    assert ctx_key1 != "__outlier_fallback__"

    assert results[1][0] == "fallback_arm"
    ctx_key2, _ = router._decode_trace_id(results[1][1])
    assert ctx_key2 == "__outlier_fallback__"
