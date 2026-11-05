"""OpenEnv wire models for the order environment."""

from __future__ import annotations

from typing import Literal

from openenv.core.env_server.types import Action, Observation


class OrderAction(Action):
    tool: Literal["inspect_product", "add_to_cart", "set_destination", "checkout"]
    sku: str | None = None
    quantity: int | None = None
    city: str | None = None


class OrderObservation(Observation):
    message: str = ""
