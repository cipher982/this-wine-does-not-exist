"""Entry point for running the FastAPI deployment."""

from wine_ai.inference.api import create_app

app = create_app()
MD
