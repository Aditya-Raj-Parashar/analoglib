"""AnalogLib CLI — command-line interface.

Commands:
    analog info <file.analog>       — inspect metadata
    analog simulate <file.analog>   — run inference
    analog export-spice <file.analog> — generate SPICE netlist
    analog profile <file.analog>    — hardware performance report
"""

from __future__ import annotations

import sys
import os

# Ensure the parent package is importable when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _get_analoglib():
    import analoglib as al
    return al


def cmd_info(args):
    """Print metadata from a .analog file."""
    al = _get_analoglib()
    if not args or args[0].startswith("-"):
        print("Usage: analog info <file.analog>")
        return 1
    path = args[0]
    try:
        result = al.load(path)
    except Exception as e:
        print(f"Error loading {path!r}: {e}")
        return 1

    meta = result["meta"]
    xbars = result["crossbars"]
    print("=" * 48)
    print("  AnalogLib Model Info")
    print("=" * 48)
    for k, v in meta.items():
        print(f"  {k:<20}: {v}")
    print(f"  {'layers':<20}: {len(xbars)}")
    for i, xb in enumerate(xbars):
        print(f"    Layer {i}: {xb.rows}x{xb.cols}  device={xb.device.__class__.__name__}")
    print("=" * 48)
    return 0


def cmd_simulate(args):
    """Run model inference and print output."""
    import numpy as np
    al = _get_analoglib()
    if len(args) < 1:
        print("Usage: analog simulate <file.analog> [--mode ideal|device|hardware]")
        return 1
    path = args[0]
    mode = "ideal"
    for i, a in enumerate(args[1:]):
        if a == "--mode" and i + 2 < len(args):
            mode = args[i + 2]

    result = al.load(path)
    xbars = result["crossbars"]
    if not xbars:
        print("No crossbars found.")
        return 1

    engine = al.SimulationEngine(crossbars=xbars)
    V = np.random.default_rng(0).uniform(0, 1, xbars[0].rows)
    out = engine.run(V, mode=mode)
    print(f"Input shape:  {V.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output:       {out}")
    return 0


def cmd_export_spice(args):
    """Export .analog model to SPICE netlist."""
    al = _get_analoglib()
    from analoglib.exporters.spice import SpiceExporter
    if len(args) < 1:
        print("Usage: analog export-spice <file.analog> [--out circuit.cir] [--dialect ngspice]")
        return 1
    path = args[0]
    out_path = path.replace(".analog", ".cir")
    dialect = "ngspice"
    for i, a in enumerate(args[1:]):
        if a == "--out" and i + 2 < len(args):
            out_path = args[i + 2]
        if a == "--dialect" and i + 2 < len(args):
            dialect = args[i + 2]

    result = al.load(path)
    xbars = result["crossbars"]
    exporter = SpiceExporter(dialect=dialect)
    saved = exporter.export(out_path, xbars)
    print(f"SPICE netlist written to: {saved}")
    return 0


def cmd_profile(args):
    """Print hardware performance metrics."""
    import numpy as np
    al = _get_analoglib()
    from analoglib.analysis.profiler import AnalogProfiler
    if len(args) < 1:
        print("Usage: analog profile <file.analog>")
        return 1
    path = args[0]
    result = al.load(path)
    xbars = result["crossbars"]
    if not xbars:
        print("No crossbars found.")
        return 1

    V = np.ones(xbars[0].rows) * 0.5
    profiler = AnalogProfiler()
    report = profiler.profile(xbars, V)
    report.print()
    return 0


def main():
    """Entry point for `analog` CLI."""
    args = sys.argv[1:]
    if not args:
        print("AnalogLib CLI — usage: analog <command> [options]")
        print("Commands: info, simulate, export-spice, profile")
        return

    command = args[0]
    rest = args[1:]

    dispatch = {
        "info":         cmd_info,
        "simulate":     cmd_simulate,
        "export-spice": cmd_export_spice,
        "profile":      cmd_profile,
    }

    if command not in dispatch:
        print(f"Unknown command {command!r}. Commands: {list(dispatch)}")
        sys.exit(1)

    sys.exit(dispatch[command](rest) or 0)


if __name__ == "__main__":
    main()
