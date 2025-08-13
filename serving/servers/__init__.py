"""Server implementations for HybridInference.

This module contains server-side implementations that provide APIs,
as opposed to client implementations in serving/providers/.
"""

from .openrouter import app as openrouter_app

__all__ = ["openrouter_app"]