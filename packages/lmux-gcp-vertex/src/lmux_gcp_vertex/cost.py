"""Deprecated shim — use ``lmux_google.cost`` instead."""

from lmux_google.cost import calculate_google_cost as calculate_gcp_vertex_cost

__all__ = [
    "calculate_gcp_vertex_cost",
]
