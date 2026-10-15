"""Thin CLI integration for the two prediction-based paper analyses.

The main CLI can register these commands with add_analysis_parsers and dispatch
with run_analysis. This module never imports the training CLI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .checkpoints import _require_new_output
from .stability_analysis import summarize_stability
from .weather_analysis import analyze_weather


def add_analysis_parsers(subparsers: argparse._SubParsersAction) -> None:
    stability = subparsers.add_parser(
        "summarize-stability", help="Summarize exactly six supplied full-stage final instances, with sample SD",
    )
    stability.add_argument("--runs", type=Path, nargs=6, required=True, metavar="DIR", help="Six distinct instance directories, each containing all three window NPZ files")
    stability.add_argument("--reference", type=Path, required=True, help="Fixed NWPSeq2Seq prediction directory with all three windows")
    stability.add_argument("--output", type=Path, required=True, help="New local JSON file; existing paths are never overwritten")

    weather = subparsers.add_parser(
        "analyze-weather", help="Compare matched Full and Head-only predictions in training-derived weather tertiles",
    )
    weather.add_argument("--data-root", type=Path, required=True, help="Authorized local raw dataset root")
    weather.add_argument("--full", type=Path, required=True, help="Full predictions (metadata variant='full') for all three windows")
    weather.add_argument("--head-only", type=Path, required=True, help="Matched Head-only predictions (metadata variant='head_only')")
    weather.add_argument("--output", type=Path, required=True, help="New local JSON file; existing paths are never overwritten")


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    """Write strict JSON and return only the compact main-CLI stdout result."""
    output = Path(args.output).expanduser()
    _require_new_output(output)
    if args.command == "summarize-stability":
        report = summarize_stability(args.runs, args.reference)
        result = {
            "command": args.command, "output": str(output.resolve()),
            "instance_count": report["instance_count"],
            "all_six_meet_selection_condition": report["all_six_meet_selection_condition"],
        }
    elif args.command == "analyze-weather":
        report = analyze_weather(args.data_root, args.full, args.head_only)
        result = {
            "command": args.command, "output": str(output.resolve()),
            "group_count": len(report["groups"]),
            "matched_station_day_count": report["matched_station_day_count"],
        }
    else:
        raise ValueError(f"Unknown analysis command: {args.command}")
    # Serialize before creating a file so invalid numeric results leave no output.
    text = json.dumps(report, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against a path appearing after preflight.
    with output.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MSWP-Fuse paper prediction analyses")
    add_analysis_parsers(parser.add_subparsers(dest="command", required=True))
    result = run_analysis(parser.parse_args(argv))
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
