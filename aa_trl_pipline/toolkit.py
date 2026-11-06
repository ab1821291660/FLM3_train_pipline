"""Deterministic tools shared by data generation and TRL agent rollouts.

The tools deliberately avoid network and filesystem access.  That makes reward
checks reproducible and keeps the examples safe to execute during unit tests.
"""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from typing import Any


CATALOG: dict[str, dict[str, Any]] = {##商品价
    "SKU-A100": {"name": "便携键盘", "price": "79.90", "stock": 80},
    "SKU-A200": {"name": "无线鼠标", "price": "49.50", "stock": 120},
    "SKU-B100": {"name": "USB-C 扩展坞", "price": "129.00", "stock": 45},
    "SKU-B200": {"name": "折叠支架", "price": "35.80", "stock": 95},
    "SKU-C100": {"name": "降噪耳机", "price": "199.00", "stock": 35},
    "SKU-C200": {"name": "桌面麦克风", "price": "159.90", "stock": 30},
    "SKU-D100": {"name": "移动电源", "price": "99.00", "stock": 60},
    "SKU-D200": {"name": "编织数据线", "price": "19.90", "stock": 200},
}

SHIPPING_FEES = {##运费
    "北京": "8.00",
    "上海": "9.00",
    "广州": "10.00",
    "成都": "11.00",
    "杭州": "8.50",
}


def ____辅助函数(self):pass
def _money(value: Decimal | str | int | float) -> str:
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def ____4个工具(self):pass
def lookup_product(sku: str) -> str:
    """Look up one product in the read-only catalog.

    Args:
        sku: Product identifier such as ``SKU-A100``.

    Returns:
        A JSON string containing the SKU, name, unit price, and stock.
    """

    if sku not in CATALOG:
        raise ValueError(f"unknown sku: {sku}")
    return _json({"sku": sku,
                  **CATALOG[sku]})


def calculate_line_total(sku: str, quantity: int) -> str:
    """Calculate the merchandise total for one order line.

    Args:
        sku: Product identifier in the catalog.
        quantity: Positive number of units to purchase.

    Returns:
        A JSON string with the SKU, quantity, and two-decimal line total.
    """

    if sku not in CATALOG:
        raise ValueError(f"unknown sku: {sku}")
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    if quantity > int(CATALOG[sku]["stock"]):
        raise ValueError(f"insufficient stock for {sku}")
    total = Decimal(CATALOG[sku]["price"]) * quantity
    return _json({"sku": sku,
                  "quantity": quantity,
                  "line_total": _money(total)})


def shipping_quote(city: str, merchandise_total: float) -> str:##订单满300 CNY免运费;否则按城市价格计算。
    """Quote shipping after the merchandise subtotal is known.

    Orders of at least 300 CNY ship free; otherwise the city rate applies.

    Args:
        city: Destination city in the supported city table.
        merchandise_total: Combined value of all order lines in CNY.

    Returns:
        A JSON string with the destination and two-decimal shipping fee.
    """

    if city not in SHIPPING_FEES:
        raise ValueError(f"unsupported city: {city}")
    subtotal = Decimal(str(merchandise_total))
    fee = Decimal("0") if subtotal >= Decimal("300") else Decimal(SHIPPING_FEES[city])
    return _json({"city": city,
                  "shipping_fee": _money(fee)})


def calculate_grand_total(merchandise_total: float, shipping_fee: float) -> str:
    """Add merchandise and shipping into the final payable amount.

    Args:
        merchandise_total: Combined value of all order lines in CNY.
        shipping_fee: Shipping fee returned by ``shipping_quote``.

    Returns:
        A JSON string containing the two-decimal grand total.
    """

    total = Decimal(str(merchandise_total)) + Decimal(str(shipping_fee))
    return _json({"grand_total": _money(total),
                  "currency": "CNY"})


TOOLS = [lookup_product, calculate_line_total, shipping_quote, calculate_grand_total]##===================================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_product",
            "description": "Look up one product in the read-only catalog.",
            "parameters": {
                "type": "object",
                "properties": {"sku": {"type": "string", "description": "Product identifier."}},
                "required": ["sku"],
            },
            "return": {"type": "string", "description": "Product data encoded as JSON."},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_line_total",
            "description": "Calculate the merchandise total for one order line.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "Product identifier."},
                    "quantity": {"type": "integer", "description": "Positive number of units."},
                },
                "required": ["sku", "quantity"],
            },
            "return": {"type": "string", "description": "Line total encoded as JSON."},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shipping_quote",
            "description": "Quote shipping after the merchandise subtotal is known.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Destination city."},
                    "merchandise_total": {"type": "number", "description": "Combined line totals."},
                },
                "required": ["city", "merchandise_total"],
            },
            "return": {"type": "string", "description": "Shipping quote encoded as JSON."},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_grand_total",
            "description": "Add merchandise and shipping into the final payable amount.",
            "parameters": {
                "type": "object",
                "properties": {
                    "merchandise_total": {"type": "number", "description": "Combined line totals."},
                    "shipping_fee": {"type": "number", "description": "Quoted shipping fee."},
                },
                "required": ["merchandise_total", "shipping_fee"],
            },
            "return": {"type": "string", "description": "Grand total encoded as JSON."},
        },
    },
]


def ____总计算工具(self):pass
def order_totals(sku_a: str, qty_a: int, sku_b: str, qty_b: int, city: str) -> dict[str, str]:
    """Compute oracle totals for fixture generation and environment verification."""

    line_a = Decimal(CATALOG[sku_a]["price"]) * qty_a
    line_b = Decimal(CATALOG[sku_b]["price"]) * qty_b
    merchandise = line_a + line_b
    shipping = Decimal("0") if merchandise >= Decimal("300") else Decimal(SHIPPING_FEES[city])
    return {
        "line_a": _money(line_a),
        "line_b": _money(line_b),
        "merchandise_total": _money(merchandise),
        "shipping_fee": _money(shipping),
        "grand_total": _money(merchandise + shipping),
    }
