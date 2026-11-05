"""A deterministic stateful environment for multi-turn Agent RL.

Public methods other than ``reset`` and ``get_reward`` are automatically
exposed as tools by current TRL GRPO trainers.
"""

from __future__ import annotations

import json
from decimal import Decimal

from aa_trl_pipline.toolkit import CATALOG, SHIPPING_FEES, lookup_product, order_totals


class OrderFulfillmentEnv:
    """Stateful shopping task with an environment-owned binary reward."""

    def __init__(self) -> None:
        self.target_cart: dict[str, int] = {}
        self.target_city = ""
        self.expected_total = "0.00"
        self.cart: dict[str, int] = {}
        self.inspected: set[str] = set()
        self.destination = ""
        self.checked_out = False
        self.last_total = ""
        self.calls: list[str] = []

    def reset(
        self,
        sku_a: str,
        qty_a: int,
        sku_b: str,
        qty_b: int,
        city: str,
        expected_total: str,
        **kwargs: object,
    ) -> None:
        """Reset one episode from the non-prompt columns of a dataset row."""

        del kwargs
        self.target_cart = {sku_a: int(qty_a), sku_b: int(qty_b)}
        self.target_city = city
        self.expected_total = str(expected_total)
        self.cart = {}
        self.inspected = set()
        self.destination = ""
        self.checked_out = False
        self.last_total = ""
        self.calls = []

    def get_reward(self) -> float:
        """Return one only for a correct checked-out state and real tool sequence."""

        required = {"inspect_product", "add_to_cart", "set_destination", "checkout"}
        correct = (
            self.checked_out
            and self.cart == self.target_cart
            and self.destination == self.target_city
            and self.last_total == self.expected_total
            and self.inspected == set(self.target_cart)
            and required.issubset(self.calls)
            and self.calls.count("add_to_cart") >= 2
        )
        return float(correct)

    def inspect_product(self, sku: str) -> str:
        """Inspect a product before it can be added to the cart.

        Args:
            sku: Product identifier to inspect.

        Returns:
            Product details encoded as JSON.
        """

        result = lookup_product(sku)
        self.inspected.add(sku)
        self.calls.append("inspect_product")
        return result

    def add_to_cart(self, sku: str, quantity: int) -> str:
        """Add an inspected product and quantity to the episode cart.

        Args:
            sku: Product identifier already passed to ``inspect_product``.
            quantity: Positive number of units to add.

        Returns:
            The current cart encoded as JSON.
        """

        if sku not in self.inspected:
            raise ValueError(f"inspect {sku} before adding it")
        if quantity <= 0 or quantity > int(CATALOG[sku]["stock"]):
            raise ValueError("invalid quantity")
        self.cart[sku] = self.cart.get(sku, 0) + int(quantity)
        self.calls.append("add_to_cart")
        return json.dumps({"cart": self.cart}, ensure_ascii=False, sort_keys=True)

    def set_destination(self, city: str) -> str:
        """Set the shipping destination for the current cart.

        Args:
            city: A city supported by the shipping table.

        Returns:
            Confirmation of the selected destination.
        """

        if city not in SHIPPING_FEES:
            raise ValueError(f"unsupported city: {city}")
        self.destination = city
        self.calls.append("set_destination")
        return json.dumps({"destination": city}, ensure_ascii=False)

    def checkout(self) -> str:
        """Compute and lock the total for the current cart and destination.

        Returns:
            Checkout status and payable total encoded as JSON.
        """

        if len(self.cart) != 2:
            raise ValueError("cart must contain exactly two products")
        if not self.destination:
            raise ValueError("set destination before checkout")
        items = sorted(self.cart.items())
        totals = order_totals(
            items[0][0], items[0][1], items[1][0], items[1][1], self.destination
        )
        self.last_total = totals["grand_total"]
        self.checked_out = True
        self.calls.append("checkout")
        return json.dumps(
            {"status": "checked_out", "grand_total": self.last_total, "currency": "CNY"},
            ensure_ascii=False,
            sort_keys=True,
        )


def replay_oracle(row: dict[str, object]) -> tuple[str, float]:
    """Execute the known-good multi-turn path used by tests and data checks."""
    env = OrderFulfillmentEnv()##===================================
    env.reset(
        sku_a=str(row["sku_a"]),
        qty_a=int(row["qty_a"]),
        sku_b=str(row["sku_b"]),
        qty_b=int(row["qty_b"]),
        city=str(row["city"]),
        expected_total=str(row["expected_total"]),
    )
    env.inspect_product(str(row["sku_a"]))
    env.add_to_cart(str(row["sku_a"]), int(row["qty_a"]))
    env.inspect_product(str(row["sku_b"]))
    env.add_to_cart(str(row["sku_b"]), int(row["qty_b"]))
    env.set_destination(str(row["city"]))
    result = env.checkout()##===================================
    return result, env.get_reward()##===================================

