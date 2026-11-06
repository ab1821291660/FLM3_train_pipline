"""ASGI application for ``uvicorn ...openenv_order.server.app:app``."""

from openenv.core.env_server import create_app

from aa_trl_pipline.openenv_order.models import OrderAction, OrderObservation
from aa_trl_pipline.openenv_order.server.environment import OrderServerEnvironment


app = create_app(##===================================##===================================
    OrderServerEnvironment,##===================================
    OrderAction,
    OrderObservation,
    env_name="aa-order-env",
)
