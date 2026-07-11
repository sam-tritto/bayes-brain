"""Custom exceptions for BayesianCortex.

These define a proper exception hierarchy to make it easier for library callers to handle
specific failure modes without catching generic Python exception types.
"""

class BayesianCortexError(Exception):
    """Base exception for all errors raised by the BayesianCortex library."""
    pass

class TamperDetectedError(BayesianCortexError, ValueError):
    """Raised when HMAC signature verification fails or trace ID is tampered."""
    pass

class EmbeddingError(BayesianCortexError, RuntimeError):
    """Raised when embedding generation or API communication fails."""
    pass
