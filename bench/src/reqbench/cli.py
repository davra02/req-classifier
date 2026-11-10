"""Command-line entry point: `uv run reqbench <command>`."""

import argparse
import json
from pathlib import Path

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


def cmd_fresh(args: argparse.Namespace) -> None:
    if args.step in ("assign", "message", "form", "reply", "kappa"):
        _fresh_agreement(args)
        return
    if args.step == "annotate":
        from reqbench.fresh.annotate import serve

        serve(port=args.port, open_browser=not args.no_browser)
        return
    from reqbench.fresh import collect

    result = collect.collect(
        refresh=args.refresh, seed=args.seed, target=args.target, per_doc=args.per_doc
    )
    eligible = result.eligible
    _print(
        {
            "retrieved_at": result.snapshot["retrieved_at"],
            "repos_found": len(result.snapshot["repos"]),
            "repos_excluded": dict(result.repo_exclusions),
            "documents_excluded": dict(result.doc_exclusions),
            "documents_eligible": len(eligible),
            "candidates_eligible": sum(len(d.candidates) for d in eligible),
            "reference_near_duplicates": sum(
                len(d.reference_hits) for d in eligible + [d for d, _ in result.excluded_docs]
            ),
            "skipped_during_draw": result.selection.skipped,
            "documents_selected": len(result.documents),
            "requirements_selected": len(result.requirements),
        }
    )
    print(result.documents[["doc_id", "license", "published_date", "max_requirements_taken",
                            "url"]].to_string(index=False))
    if args.show:
        sample = result.requirements.head(args.show)  # rows are already shuffled by seed
        for row in sample.itertuples():
            print(f"{row.req_id} {row.doc_id} {row.location}\n    {row.text}")
    if args.write:
        _print(collect.write_collection(result, args.seed, args.target, args.per_doc)["counts"])
    else:
        print("Dry run: nothing written to data/fresh/ (use --write).")


def _fresh_agreement(args: argparse.Namespace) -> None:
    import subprocess
    import sys

    import pandas as pd

    from reqbench.fresh import agreement

    path = data_dir() / "fresh" / "requirements.csv"
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    def write(frame: pd.DataFrame) -> None:
        tmp = path.with_suffix(".csv.part")
        frame.to_csv(tmp, index=False)
        tmp.replace(path)

    if args.step == "kappa":
        _print(agreement.agreement(df))
        return
    if args.step == "assign":
        out = agreement.add_subset(df, args.who) if args.who else agreement.assign_subsets(df)
        write(out)
        assigned = out[out[agreement.ASSIGN_COLUMN] != ""]
        _print(assigned.groupby(agreement.ASSIGN_COLUMN).size().to_dict())
        return
    if args.who is None:
        raise SystemExit("--who is required")
    if args.step == "form":
        out = Path(args.out or data_dir() / "raw" / "fresh" / f"form_{args.who}.html")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(agreement.render_form(df, args.who), encoding="utf-8")
        print(f"Form for {args.who.upper()} written to {out} (publish it and send the link).")
        return
    if args.step == "message":
        text = agreement.checklist_message(df, args.who)
        if args.copy:
            subprocess.run(["pbcopy"], input=text.encode(), check=True)
            print(f"Checklist for {args.who.upper()} copied to the clipboard ({len(text)} chars).")
        else:
            print(text)
        return
    if args.paste:
        reply = subprocess.run(["pbpaste"], capture_output=True, check=True).stdout.decode()
    elif args.file:
        reply = Path(args.file).read_text(encoding="utf-8")
    else:
        reply = sys.stdin.read()
    labels = agreement.parse_reply(df, args.who, reply)
    write(agreement.apply_reply(df, args.who, labels))
    print(f"Saved {len(labels)} labels to annotator_{args.who}.")


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

    fresh = sub.add_parser(
        "fresh", help="fresh set: collect (dry run unless --write), annotate, assign, form, "
        "message, reply, kappa"
    )  # fmt: skip
    fresh.add_argument(
        "step", choices=["collect", "annotate", "assign", "form", "message", "reply", "kappa"]
    )
    fresh.add_argument("--write", action="store_true", help="write data/fresh/*.csv")
    fresh.add_argument("--refresh", action="store_true", help="re-run the GitHub search")
    fresh.add_argument("--seed", type=int, default=20261010)
    fresh.add_argument("--target", type=int, default=170)
    fresh.add_argument("--per-doc", type=int, default=15)
    fresh.add_argument("--show", type=int, default=25, help="print the first N rows")
    fresh.add_argument("--port", type=int, default=8765, help="annotate: local port")
    fresh.add_argument("--no-browser", action="store_true", help="annotate: do not open it")
    fresh.add_argument(
        "--who", choices=["b", "c", "d", "e"],
        help="second annotator (assign: draw a reserve subsample for d or e)",
    )  # fmt: skip
    fresh.add_argument("--copy", action="store_true", help="message: copy to the clipboard")
    fresh.add_argument("--paste", action="store_true", help="reply: read the clipboard")
    fresh.add_argument("--file", help="reply: read the reply from this file")
    fresh.add_argument("--out", help="form: output HTML path (default data/raw/fresh/)")
    fresh.set_defaults(func=cmd_fresh)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
