"""
Vercel serverless entry point.

Vercel looks for an `app` object in api/index.py.
We simply re-export the FastAPI app from our existing server module.
"""

from app.server import app  # noqa: F401
