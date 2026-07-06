import json
import numpy as np
from typing import Dict, List, Optional, Protocol, Sequence

class ContextEmbedder(Protocol):
    """Protocol defining how to convert text into a vector context key."""

    def embed_query(self, text: str) -> Sequence[float]:
        """Convert a text query (prompt) into a vector of floats."""
        ...


class VectorStoreProtocol(Protocol):
    """Protocol defining the interface for context vector storage and search."""

    def add_context(self, context_key: str, vector: Sequence[float]) -> None:
        """Add or update a context vector in the index."""
        ...

    def get_nearest_context(
        self, query_vector: Sequence[float], similarity_threshold: float = 0.8
    ) -> Optional[str]:
        """
        Find the context_key whose stored vector is closest to query_vector,
        provided the cosine similarity is above the threshold.
        """
        ...


class LocalSentenceTransformerEmbedder:
    """
    Batteries-included embedder using sentence-transformers.
    Loaded lazily, requiring `pip install 'bayes-brain[local-ml]'`.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "The sentence-transformers package is required for LocalSentenceTransformerEmbedder. "
                    "Please install it with: pip install 'bayes-brain[local-ml]'"
                )
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_query(self, text: str) -> Sequence[float]:
        embedding = self.model.encode(text)
        return [float(x) for x in embedding]


class VectorContextStore:
    """
    A lightweight, in-memory vector index for storing and querying reference contexts.
    Uses cosine similarity to map query vectors to nearest contextual keys.
    """

    def __init__(self) -> None:
        # Maps context_key to its vector representation
        self._contexts: Dict[str, np.ndarray] = {}

    def add_context(self, context_key: str, vector: Sequence[float]) -> None:
        """Add or update a context vector in the index."""
        self._contexts[context_key] = np.array(vector, dtype=np.float32)

    def get_nearest_context(
        self, query_vector: Sequence[float], similarity_threshold: float = 0.8
    ) -> Optional[str]:
        """
        Find the context_key whose stored vector is closest to query_vector,
        provided the cosine similarity is above the threshold.
        """
        if not self._contexts:
            return None

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0.0:
            return None

        best_key = None
        best_similarity = -1.0

        for key, ref_vec in self._contexts.items():
            ref_norm = np.linalg.norm(ref_vec)
            if ref_norm == 0.0:
                continue
            
            # Cosine similarity calculation
            similarity = float(np.dot(q_vec, ref_vec) / (q_norm * ref_norm))
            if similarity > best_similarity:
                best_similarity = similarity
                best_key = key

        if best_similarity >= similarity_threshold:
            return best_key
            
        return None

    def to_json(self) -> str:
        """Serialize the context store to a JSON string."""
        serializable = {
            key: vec.tolist() for key, vec in self._contexts.items()
        }
        return json.dumps(serializable)

    @classmethod
    def from_json(cls, data_str: str) -> "VectorContextStore":
        """Deserialize context store from a JSON string."""
        store = cls()
        data = json.loads(data_str)
        for key, vec_list in data.items():
            store.add_context(key, vec_list)
        return store
