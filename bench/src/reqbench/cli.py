"""Command-line entry point: `uv run reqbench <command>`."""

import argparse
import json

from reqcore.budget import Budget
from reqcore.env import load_env
from reqcore.paths import data_dir


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def cmd_data(args: argparse.Namespace) -> None:
    from reqbench.data import build, download, examples, splits

    if args.step in ("download", "all"):
        print(f"sources manifest: {download.download_all(force=args.force)}")
    if args.step in ("build", "all"):
        clean, report = build.build_promise()
        manifest = build.write_build(clean, report)
        _print({k: manifest[k] for k in ("n_raw", "n_clean", "leaf_counts", "removed")})
    if args.step in ("split", "all"):
        clean = build.load_processed()
        meta_build = json.loads(
            (data_dir() / "manifests" / f"{build.DATASET_VERSION}.json").read_text()
        )
        table, meta = splits.make_splits(clean, meta_build["cross_project_near_duplicates"])
        _print(splits.write_splits(table, meta))
    if args.step in ("examples", "all"):
        build_manifest = json.loads(
            (data_dir() / "manifests" / f"{build.DATASET_VERSION}.json").read_text()
        )
        ineligible = {r["kept"] for r in build_manifest["removed"]} | {
            rid for a, b, _ in build_manifest["cross_project_near_duplicates"] for rid in (a, b)
        }
        spec = examples.pick_examples(build.load_processed(), splits.load_splits(), ineligible)
        examples.write_examples(spec)
        _print(spec["source_ids"])


def cmd_budget(_: argparse.Namespace) -> None:
    budget = Budget()
    _print({"spent": budget.spent_by_bucket(), "remaining": budget.remaining()})


def cmd_smoke(args: argparse.Namespace) -> None:
    from reqbench.smoke import run_smoke

    _print(run_smoke(args.n_jev, args.n_haiku, args.n_opus, run=args.run))


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(prog="reqbench")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="download, build, split and pick examples")
    data.add_argument("step", choices=["download", "build", "split", "examples", "all"])
    data.add_argument("--force", action="store_true", help="re-download even if verified")
    data.set_defaults(func=cmd_data)

    sub.add_parser("budget", help="show spend and remaining budget").set_defaults(func=cmd_budget)

    smoke = sub.add_parser("smoke", help="phase-0 smoke test (dry run unless --run)")
    smoke.add_argument("--run", action="store_true", help="make real (paid) API calls")
    smoke.add_argument("--n-jev", type=int, default=20)
    smoke.add_argument("--n-haiku", type=int, default=20)
    smoke.add_argument("--n-opus", type=int, default=3)
    smoke.set_defaults(func=cmd_smoke)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
