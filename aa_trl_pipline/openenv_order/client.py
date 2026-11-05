"""Typed WebSocket client for the OpenEnv order server."""

from __future__ import annotations

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from aa_trl_pipline.openenv_order.models import OrderAction, OrderObservation


class OrderEnvClient(EnvClient[OrderAction, OrderObservation, State]):
    def _step_payload(self, action: OrderAction) -> dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict) -> StepResult[OrderObservation]:
        observation = OrderObservation(
            **payload["observation"],
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )
        return StepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    def _parse_state(self, payload: dict) -> State:
        return State(**payload)
