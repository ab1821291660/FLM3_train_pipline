"""OpenEnv server implementation backed by the deterministic local environment."""

from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from aa_trl_pipline.openenv_order.models import OrderAction, OrderObservation
from aa_trl_pipline.order_env import OrderFulfillmentEnv##===================================##===================================


class OrderServerEnvironment(Environment[OrderAction, OrderObservation,
                            State]):
    SUPPORTS_CONCURRENT_SESSIONS = True##并行

    def __init__(self) -> None:
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._env = OrderFulfillmentEnv()##===================================##===================================
        self._done = False


    def reset(self, seed=None, episode_id=None, **kwargs) -> OrderObservation:
        del seed
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._env.reset(**kwargs)##===================================##===================================
        self._done = False
        return OrderObservation(message="order environment ready", reward=0.0, done=False)


    def step(self, action: OrderAction, timeout_s=None, **kwargs) -> OrderObservation:
        del timeout_s, kwargs
        if self._done:
            return OrderObservation(message="episode is over; call reset",
                                    reward=0.0,
                                    done=True)


        self._state.step_count += 1
        if action.tool == "inspect_product":
            if action.sku is None:
                raise ValueError("inspect_product requires sku")
            message = self._env.inspect_product(action.sku)##===================================##===================================
        elif action.tool == "add_to_cart":
            if action.sku is None or action.quantity is None:
                raise ValueError("add_to_cart requires sku and quantity")
            message = self._env.add_to_cart(action.sku, action.quantity)##===================================##===================================
        elif action.tool == "set_destination":
            if action.city is None:
                raise ValueError("set_destination requires city")
            message = self._env.set_destination(action.city)##===================================##===================================
        elif action.tool == "checkout":
            message = self._env.checkout()##===================================##===================================
            self._done = True
        else:
            raise ValueError(f"unsupported tool: {action.tool}")
        reward = self._env.get_reward() if self._done   else 0.0 ##===================================##===================================
        return OrderObservation(message=message,
                                reward=reward,
                                done=self._done)

    @property
    def state(self) -> State:
        return self._state
