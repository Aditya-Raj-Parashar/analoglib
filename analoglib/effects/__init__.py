"""analoglib.effects — Pluggable physical hardware effects.

Each Effect is applied to conductance matrices during a crossbar VMM,
allowing arbitrary composition of physical non-idealities.

Usage::

    from analoglib.effects import IRDrop, Thermal, Drift

    xbar = Crossbar(128, 64, device=reram, effects=[IRDrop(r_wire=1.0)])
    xbar.vmm(V)  # IR drop applied automatically in HARDWARE mode
"""

from .base import Effect, EffectContext
from .ir_drop import IRDrop
from .thermal import Thermal
from .drift import Drift

__all__ = [
    "Effect",
    "EffectContext",
    "IRDrop",
    "Thermal",
    "Drift",
]
