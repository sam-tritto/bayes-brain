import pytest
from typing import Sequence
from bayesian_cortex.router import BayesianRouter, AsyncBayesianRouter
from bayesian_cortex.storage import InMemoryStorage, AsyncInMemoryStorage


class MockEmbedder:
    def embed_query(self, text: str) -> Sequence[float]:
        return [1.0, 0.0]

    async def aembed_query(self, text: str) -> Sequence[float]:
        return [1.0, 0.0]


def test_hierarchical_init_sync():
    embedder = MockEmbedder()
    storage = InMemoryStorage()

    # Define a child router config
    child_cfg = {
        "mode": "clustering",
        "candidates": ["tool_python", "tool_git"],
    }

    # Initialize root router
    root = BayesianRouter(
        storage=storage,
        embedder=embedder,
        candidates=["coder_subagent", "research_subagent"],
        children={
            "coder_subagent": child_cfg,
        }
    )

    assert root.candidates == ["coder_subagent", "research_subagent"]
    assert "coder_subagent" in root.children
    child = root.children["coder_subagent"]

    # Child should inherit the parent's storage and embedder
    assert child.storage is storage
    assert child.embedder is embedder
    assert child.candidates == ["tool_python", "tool_git"]


def test_hierarchical_from_config_sync():
    embedder = MockEmbedder()
    storage = InMemoryStorage()

    config = {
        "storage": storage,
        "embedder": embedder,
        "mode": "clustering",
        "candidates": ["coder_subagent", "research_subagent"],
        "children": {
            "coder_subagent": {
                "mode": "clustering",
                "candidates": ["tool_python", "tool_git"],
            },
            "research_subagent": {
                "mode": "clustering",
                "candidates": ["tool_search", "tool_wiki"],
            }
        }
    }

    root = BayesianRouter.from_config(config)

    assert root.candidates == ["coder_subagent", "research_subagent"]
    assert "coder_subagent" in root.children
    assert "research_subagent" in root.children

    coder_child = root.children["coder_subagent"]
    research_child = root.children["research_subagent"]

    assert coder_child.storage is storage
    assert coder_child.embedder is embedder
    assert coder_child.candidates == ["tool_python", "tool_git"]

    assert research_child.storage is storage
    assert research_child.embedder is embedder
    assert research_child.candidates == ["tool_search", "tool_wiki"]


def test_fallback_candidates_sync():
    root = BayesianRouter(
        candidates=["a", "b"],
    )

    # Route without specifying candidates, should fallback to root.candidates
    choice = root.route("some query")
    assert choice in ["a", "b"]

    # Batch route without candidates
    choices = root.route_batch(["query 1", "query 2"])
    assert len(choices) == 2
    assert all(c in ["a", "b"] for c in choices)


def test_route_hierarchical_and_feedback_sync():
    embedder = MockEmbedder()
    storage = InMemoryStorage()

    config = {
        "storage": storage,
        "embedder": embedder,
        "mode": "clustering",
        "candidates": ["coder_subagent", "research_subagent"],
        "children": {
            "coder_subagent": {
                "mode": "clustering",
                "candidates": ["tool_python", "tool_git"],
            }
        }
    }

    # Seed parent router so it consistently chooses "coder_subagent"
    root = BayesianRouter.from_config(config)
    ctx_key = root._resolve_context_key("query")
    # Give coder_subagent high prior, research_subagent low prior
    root.storage.update_candidate_params(ctx_key, "coder_subagent", 10.0, 1.0)
    root.storage.update_candidate_params(ctx_key, "research_subagent", 1.0, 10.0)

    # Route hierarchically
    path, trace_ids = root.route_hierarchical("query")

    # Path should traverse root -> coder_subagent -> tool
    assert path[0] == "coder_subagent"
    assert path[1] in ["tool_python", "tool_git"]

    assert "coder_subagent" in trace_ids
    assert path[1] in trace_ids

    parent_trace_id = trace_ids["coder_subagent"]
    child_trace_id = trace_ids[path[1]]

    # Update feedback for parent trace via root router
    root.feedback_by_trace(parent_trace_id, success=True)
    # The parent candidate is in parent router, so parent storage parameters should update
    alpha, beta = root.storage.get_candidate_params(ctx_key, "coder_subagent")
    assert alpha > 10.0

    # Update feedback for child trace via root router (delegation)
    child_candidate = path[1]
    child_router = root.children["coder_subagent"]
    child_ctx_key = child_router._resolve_context_key("query")

    root.feedback_by_trace(child_trace_id, success=True)

    # The child router should have updated parameters in child storage
    c_alpha, c_beta = child_router.storage.get_candidate_params(child_ctx_key, child_candidate)
    assert c_alpha > 1.0


@pytest.mark.anyio
async def test_hierarchical_init_async():
    embedder = MockEmbedder()
    storage = AsyncInMemoryStorage()

    child_cfg = {
        "mode": "clustering",
        "candidates": ["tool_python", "tool_git"],
    }

    root = AsyncBayesianRouter(
        storage=storage,
        embedder=embedder,
        candidates=["coder_subagent", "research_subagent"],
        children={
            "coder_subagent": child_cfg,
        }
    )

    assert root.candidates == ["coder_subagent", "research_subagent"]
    assert "coder_subagent" in root.children
    child = root.children["coder_subagent"]

    assert child.storage is storage
    assert child.embedder is embedder
    assert child.candidates == ["tool_python", "tool_git"]


@pytest.mark.anyio
async def test_hierarchical_from_config_async():
    embedder = MockEmbedder()
    storage = AsyncInMemoryStorage()

    config = {
        "storage": storage,
        "embedder": embedder,
        "mode": "clustering",
        "candidates": ["coder_subagent", "research_subagent"],
        "children": {
            "coder_subagent": {
                "mode": "clustering",
                "candidates": ["tool_python", "tool_git"],
            },
            "research_subagent": {
                "mode": "clustering",
                "candidates": ["tool_search", "tool_wiki"],
            }
        }
    }

    root = AsyncBayesianRouter.from_config(config)

    assert root.candidates == ["coder_subagent", "research_subagent"]
    assert "coder_subagent" in root.children
    assert "research_subagent" in root.children

    coder_child = root.children["coder_subagent"]
    research_child = root.children["research_subagent"]

    assert coder_child.storage is storage
    assert coder_child.embedder is embedder
    assert coder_child.candidates == ["tool_python", "tool_git"]

    assert research_child.storage is storage
    assert research_child.embedder is embedder
    assert research_child.candidates == ["tool_search", "tool_wiki"]


@pytest.mark.anyio
async def test_fallback_candidates_async():
    root = AsyncBayesianRouter(
        candidates=["a", "b"],
    )

    choice = await root.aroute("some query")
    assert choice in ["a", "b"]

    choices = await root.aroute_batch(["query 1", "query 2"])
    assert len(choices) == 2
    assert all(c in ["a", "b"] for c in choices)


@pytest.mark.anyio
async def test_route_hierarchical_and_feedback_async():
    embedder = MockEmbedder()
    storage = AsyncInMemoryStorage()

    config = {
        "storage": storage,
        "embedder": embedder,
        "mode": "clustering",
        "candidates": ["coder_subagent", "research_subagent"],
        "children": {
            "coder_subagent": {
                "mode": "clustering",
                "candidates": ["tool_python", "tool_git"],
            }
        }
    }

    root = AsyncBayesianRouter.from_config(config)
    # Ensure initialized is called or run a routing method to initialize the router and vector store
    # Seed async storage
    ctx_key = await root._resolve_context_key("query")
    await root.storage.update_candidate_params(ctx_key, "coder_subagent", 10.0, 1.0)
    await root.storage.update_candidate_params(ctx_key, "research_subagent", 1.0, 10.0)

    # Route hierarchically
    path, trace_ids = await root.aroute_hierarchical("query")

    assert path[0] == "coder_subagent"
    assert path[1] in ["tool_python", "tool_git"]

    assert "coder_subagent" in trace_ids
    assert path[1] in trace_ids

    parent_trace_id = trace_ids["coder_subagent"]
    child_trace_id = trace_ids[path[1]]

    # Update feedback for parent trace
    await root.afeedback_by_trace(parent_trace_id, success=True)
    alpha, beta = await root.storage.get_candidate_params(ctx_key, "coder_subagent")
    assert alpha > 10.0

    # Update feedback for child trace via delegation
    child_candidate = path[1]
    child_router = root.children["coder_subagent"]
    child_ctx_key = await child_router._resolve_context_key("query")

    await root.afeedback_by_trace(child_trace_id, success=True)

    c_alpha, c_beta = await child_router.storage.get_candidate_params(child_ctx_key, child_candidate)
    assert c_alpha > 1.0
