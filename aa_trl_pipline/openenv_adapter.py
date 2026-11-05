"""TRL ``environment_factory`` adapter for the remote OpenEnv order server."""

from __future__ import annotations

from aa_trl_pipline.openenv_order import OrderAction, OrderEnvClient


class OpenEnvOrderAdapter:
    """Expose meaningful tool methods while OpenEnv owns state and reward."""

    def __init__(self, base_url: str) -> None:
        self._context = OrderEnvClient(base_url=base_url).sync()
        self._client = self._context.__enter__()
        self._reward = 0.0

    def reset(self, **kwargs: object) -> None:
        self._reward = 0.0
        self._client.reset(**kwargs)

    def get_reward(self) -> float:
        return float(self._reward)

    def inspect_product(self, sku: str) -> str:
        """Inspect one product in the remote order environment.

        Args:
            sku: Product identifier to inspect.

        Returns:
            Product details returned by OpenEnv.
        """

        result = self._client.step(OrderAction(tool="inspect_product", sku=sku))
        self._reward = float(result.reward or 0.0)
        return result.observation.message

    def add_to_cart(self, sku: str, quantity: int) -> str:
        """Add an inspected product to the remote cart.

        Args:
            sku: Product identifier already inspected.
            quantity: Positive number of units.

        Returns:
            Current cart state returned by OpenEnv.
        """

        result = self._client.step(OrderAction(tool="add_to_cart", sku=sku, quantity=quantity))
        self._reward = float(result.reward or 0.0)
        return result.observation.message

    def set_destination(self, city: str) -> str:
        """Set the destination in the remote environment.

        Args:
            city: Supported destination city.

        Returns:
            Destination confirmation returned by OpenEnv.
        """

        result = self._client.step(OrderAction(tool="set_destination", city=city))
        self._reward = float(result.reward or 0.0)
        return result.observation.message

    def checkout(self) -> str:
        """Finalize the remote cart and obtain its terminal reward.

        Returns:
            Checkout result returned by OpenEnv.
        """

        result = self._client.step(OrderAction(tool="checkout"))
        self._reward = float(result.reward or 0.0)
        return result.observation.message


class OpenEnvOrderFactory:
    """Picklable no-argument factory suitable for AsyncGRPOTrainer workers."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def __call__(self) -> OpenEnvOrderAdapter:
        return OpenEnvOrderAdapter(self.base_url)
