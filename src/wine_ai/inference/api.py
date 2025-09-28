"""FastAPI integration for serving wine generations."""

from __future__ import annotations

from fastapi import FastAPI

from .generators import GenerationRequest, generate_wines


def create_app() -> FastAPI:
    app = FastAPI(title="Wine AI Dataset", version="1.0.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/generate")
    async def generate(
        wine_type: str | None = None,
        sweetness: float | None = None,
        price_low: float | None = None,
        price_high: float | None = None,
        seed: int | None = None,
        count: int = 1,
    ) -> list[dict[str, object]]:
        price_range = None
        if price_low is not None or price_high is not None:
            price_range = (
                price_low if price_low is not None else 0.0,
                price_high if price_high is not None else float("inf"),
            )
        request = GenerationRequest(
            wine_type=wine_type,
            sweetness=sweetness,
            price_range=price_range,
            seed=seed,
            count=count,
        )
        return generate_wines(request)

    return app


__all__ = ["create_app"]
