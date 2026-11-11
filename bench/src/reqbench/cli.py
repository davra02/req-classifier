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


def cmd_baselines(args: argparse.Namespace) -> None:
    from reqbench.eval import cv
    from reqbench.models import baselines
    from reqbench.models.embeddings import CachedEmbedder

    factories = {
        "b0": baselines.Majority,
        "b1": baselines.Keywords,
        "b2": baselines.TfidfLR,
        "b3": lambda: baselines.EmbedLR(embedder),
    }
    embedder = CachedEmbedder() if "b3" in args.systems else None
    dev = cv.load_dev()
    rows = []
    for key in args.systems:
        oof, params = cv.run_cv(factories[key], dev)
        summary = cv.evaluate(oof, n_resamples=args.n_resamples)
        name = factories[key]().name if key != "b3" else baselines.EmbedLR.name
        cv.save(name, oof, params, summary)
        ci = summary["ci"]

        def fmt(metric: str, ci: dict = ci) -> str:
            if metric not in ci:
                return "—"
            v, (lo, hi) = ci[metric]["value"], ci[metric]["ci95"]
            return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"

        rows.append({"system": name, "macro_f1": fmt("macro_f1"), "level1_f1": fmt("level1_f1"),
                     "level2_f1": fmt("level2_macro_f1"), "ece": fmt("ece"),
                     "aurc": fmt("aurc"), "cov@95": fmt("coverage_at_95")})  # fmt: skip
        print(f"{name}: done (fold params: {params})")
    import pandas as pd

    print(f"\nDev CV, n = {int((~dev['is_example']).sum())} (examples_v1 excluded); "
          f"bootstrap {args.n_resamples} resamples. Saved to {cv.runs_dir()}")  # fmt: skip
    print(pd.DataFrame(rows).to_string(index=False))


def cmd_jev(args: argparse.Namespace) -> None:
    from reqbench.eval.jev_dev import run_jev_dev

    report = run_jev_dev(run=args.run, rotations=tuple(args.rotations),
                         n_samples=args.samples, workers=args.workers,
                         n_resamples=args.n_resamples)  # fmt: skip
    if "dry_run" in report:
        _print(report["dry_run"])
        print("Dry run: no API call made. Re-run with --run to make the calls above.")
        return
    rows = []
    for name, summary in report["summaries"].items():
        ci = summary["ci"]

        def fmt(metric: str, ci: dict = ci) -> str:
            v, (lo, hi) = ci[metric]["value"], ci[metric]["ci95"]
            return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"

        diff = report["comparisons"][name].get("B3_bge_small_lr", {}).get("macro_f1")
        rows.append({"system": name, "macro_f1": fmt("macro_f1"), "level1_f1": fmt("level1_f1"),
                     "ece": fmt("ece"), "aurc": fmt("aurc"), "cov@95": fmt("coverage_at_95"),
                     "vs_B3": (f"{diff['value']:+.3f} [{diff['ci95'][0]:+.3f}, "
                               f"{diff['ci95'][1]:+.3f}]") if diff else "—"})  # fmt: skip
    import pandas as pd

    print(pd.DataFrame(rows).to_string(index=False))
    _print({k: report[k] for k in ("usage", "stability", "order")})


def cmd_haiku(args: argparse.Namespace) -> None:
    from reqbench.eval.haiku_dev import MAIN, run_haiku_dev

    report = run_haiku_dev(run=args.run, latency_n=args.latency_n,
                           poll_seconds=args.poll_seconds,
                           n_resamples=args.n_resamples)  # fmt: skip
    if "dry_run" in report:
        _print(report["dry_run"])
        print("Dry run: no API call made. Re-run with --run to make the calls above.")
        return
    rows = []
    for name, summary in report["summaries"].items():
        ci = summary["ci"]

        def fmt(metric: str, ci: dict = ci) -> str:
            if metric not in ci:
                return "—"
            v, (lo, hi) = ci[metric]["value"], ci[metric]["ci95"]
            return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"

        rows.append({"system": name, "macro_f1": fmt("macro_f1"), "level1_f1": fmt("level1_f1"),
                     "ece": fmt("ece"), "aurc": fmt("aurc"),
                     "cov@95": fmt("coverage_at_95")})  # fmt: skip
    import pandas as pd

    print(pd.DataFrame(rows).to_string(index=False))
    for ref, diff in report["comparisons"][MAIN].items():
        m = diff["macro_f1"]
        print(f"{MAIN} - {ref}: macro-F1 {m['value']:+.3f} "
              f"[{m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}] (Holm p {m.get('p_holm', 1):.4f})")
    _print({k: report[k] for k in ("usage", "stability", "order")})


def cmd_finetune(args: argparse.Namespace) -> None:
    from reqbench.eval.finetune_dev import run_finetune_dev
    from reqbench.models.finetune import SEEDS, Recipe

    seeds = SEEDS[: args.n_seeds]
    report = run_finetune_dev(seeds, Recipe(max_epochs=args.max_epochs),
                              n_resamples=args.n_resamples)  # fmt: skip
    _print({k: report[k] for k in ("over_seeds", "per_seed", "fit_seconds_total")})
    for name, refs in report["comparisons"].items():
        for ref, diff in refs.items():
            m = diff["macro_f1"]
            print(f"{name} - {ref}: macro-F1 {m['value']:+.3f} "
                  f"[{m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}]")


def cmd_calibrate(args: argparse.Namespace) -> None:
    import pandas as pd

    from reqbench.eval.calibrate_dev import frozen_dir, run_calibration_dev

    report = run_calibration_dev(n_resamples=args.n_resamples)
    spec = json.loads((frozen_dir() / "calibrators.json").read_text())["systems"]
    rows = []
    for name, rep in report.items():
        raw, cal, thr = rep["raw"], rep["calibrated"], rep["threshold_dev"]
        lo, hi = cal["ece"]["ci95"]
        rows.append({
            "system": name,
            "calibrator": " + ".join(f"{q}:{s['name']}" if q != "system" else s["name"]
                                     for q, s in spec[name]["questions"].items()),
            "ECE raw": f"{raw['ece']['value']:.3f}",
            "ECE cal [CI]": f"{cal['ece']['value']:.3f} [{lo:.3f}, {hi:.3f}]",
            "logloss": f"{raw['log_loss']:.3f}->{cal['log_loss']:.3f}",
            "AURC": f"{raw['aurc']:.3f}->{cal['aurc']:.3f}",
            "macroF1 cal": f"{cal['macro_f1']:.3f}",
            "threshold": "—" if thr["threshold"] is None else f"{thr['threshold']:.3f}",
            "cov@95": f"{thr['coverage']:.3f}",
        })  # fmt: skip
    print(pd.DataFrame(rows).to_string(index=False))


def cmd_final(args: argparse.Namespace) -> None:
    from reqbench.eval import final_models

    if not args.skip_classical:
        _print({k: v["params"] for k, v in final_models.fit_classical().items()})
    if not args.skip_b4:
        final_models.fit_b4()


def cmd_freeze(args: argparse.Namespace) -> None:
    from reqbench.eval import heldout

    freeze = heldout.write_freeze()
    print(f"Wrote {heldout.freeze_path()} ({len(freeze['files'])} files), "
          f"freeze hash {freeze['freeze_hash']}")  # fmt: skip


def cmd_heldout(args: argparse.Namespace) -> None:
    from reqbench.eval import heldout

    mode = "run" if args.run else "cache" if args.from_cache else "dry"
    result = heldout.run(args.set, mode, n_resamples=args.n_resamples,
                         restart_reason=args.restart_reason)
    if "dry_run" in result:
        _print(result["dry_run"])
        print("Dry run: no API call made and no label read. Re-run with --run to evaluate.")
        return
    _print(result.get("decision", {}))
    print(f"Results: {heldout.out_dir(args.set) / 'results.json'}")


def cmd_reproduce(args: argparse.Namespace) -> None:
    """Regenerate every dev table, the calibration, the held-out results and the report.

    With --from-cache no API is ever called (every call must be cached); without it, missing
    dev calls are made (with the usual budget guard). The held-out sets are only re-read from
    the cache, never evaluated anew."""
    from reqbench.eval import heldout
    from reqbench.eval.calibrate_dev import run_calibration_dev
    from reqbench.eval.finetune_dev import run_finetune_dev
    from reqbench.eval.haiku_dev import run_haiku_dev
    from reqbench.eval.jev_dev import run_jev_dev

    cmd_baselines(argparse.Namespace(systems=["b0", "b1", "b2", "b3"],
                                     n_resamples=args.n_resamples))  # fmt: skip
    for name, report in (("jev", run_jev_dev(run=not args.from_cache,
                                             n_resamples=args.n_resamples)),
                         ("haiku", run_haiku_dev(run=not args.from_cache,
                                                 n_resamples=args.n_resamples))):  # fmt: skip
        if "dry_run" in report:
            raise SystemExit(f"{name}: calls missing from the cache: {report['dry_run']}")
    run_finetune_dev(n_resamples=args.n_resamples)
    run_calibration_dev(n_resamples=args.n_resamples)
    for set_name in heldout.SETS:
        if any(e["event"] == "completed" for e in heldout._log_entries(set_name)):
            heldout.run(set_name, "cache", n_resamples=args.n_resamples)
            print(f"{set_name}: re-evaluated from the cache")
    from reqbench.report import phase1

    print(f"Report: {phase1.write_report()}")


def cmd_audit(args: argparse.Namespace) -> None:
    from reqbench.eval.audit_phase1 import audit_dir, run_audit

    _print(run_audit(n_resamples=args.n_resamples))
    print(f"Written to {audit_dir() / 'audit.json'}")


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

    base = sub.add_parser("baselines", help="phase 1: B0-B3 cross-validated on dev (no cost)")
    base.add_argument("--systems", nargs="+", choices=["b0", "b1", "b2", "b3"],
                      default=["b0", "b1", "b2", "b3"])  # fmt: skip
    base.add_argument("--n-resamples", type=int, default=10_000)
    base.set_defaults(func=cmd_baselines)

    jevp = sub.add_parser("jev", help="phase 1: Jev J1-J4 on dev (dry run unless --run)")
    jevp.add_argument("--run", action="store_true", help="make real (paid) API calls")
    jevp.add_argument("--rotations", type=int, nargs="+", default=[0, 1, 2])
    jevp.add_argument("--samples", type=int, default=5, help="samples of rotation 0")
    jevp.add_argument("--workers", type=int, default=8)
    jevp.add_argument("--n-resamples", type=int, default=10_000)
    jevp.set_defaults(func=cmd_jev)

    hk = sub.add_parser("haiku", help="phase 1: B5 Claude Haiku 5.5 on dev (dry run unless --run)")
    hk.add_argument("--run", action="store_true", help="make real (paid) API calls")
    hk.add_argument("--latency-n", type=int, default=30, help="requirements for latency")
    hk.add_argument("--poll-seconds", type=float, default=30.0)
    hk.add_argument("--n-resamples", type=int, default=10_000)
    hk.set_defaults(func=cmd_haiku)

    ft = sub.add_parser("finetune", help="phase 1: B4 DeBERTa-v3-small on dev (local, no cost)")
    ft.add_argument("--n-seeds", type=int, default=5)
    ft.add_argument("--max-epochs", type=int, default=15)
    ft.add_argument("--n-resamples", type=int, default=10_000)
    ft.set_defaults(func=cmd_finetune)

    cal = sub.add_parser("calibrate", help="phase 1: calibrators and review thresholds on dev")
    cal.add_argument("--n-resamples", type=int, default=10_000)
    cal.set_defaults(func=cmd_calibrate)

    fin = sub.add_parser("final", help="phase 1: fit B0-B4 and J3 on all of dev (no cost)")
    fin.add_argument("--skip-classical", action="store_true")
    fin.add_argument("--skip-b4", action="store_true")
    fin.set_defaults(func=cmd_final)

    sub.add_parser("freeze", help="phase 1: write frozen/phase1/FREEZE.json (hashes)"
                   ).set_defaults(func=cmd_freeze)  # fmt: skip

    ho = sub.add_parser("heldout", help="phase 1: one-time evaluation on test or fresh "
                        "(dry run unless --run or --from-cache)")  # fmt: skip
    ho.add_argument("set", choices=["test", "fresh"])
    group = ho.add_mutually_exclusive_group()
    group.add_argument("--run", action="store_true", help="make the paid calls, then evaluate")
    group.add_argument("--from-cache", action="store_true", help="evaluate from the cache only")
    ho.add_argument("--n-resamples", type=int, default=10_000)
    ho.add_argument("--restart-reason", help="log a restart after an attempt that crashed "
                    "before writing results (needed only if the freeze changed)")
    ho.set_defaults(func=cmd_heldout)

    rep = sub.add_parser("reproduce", help="regenerate every table, figure and the report")
    rep.add_argument("--from-cache", action="store_true", help="never call an API")
    rep.add_argument("--n-resamples", type=int, default=10_000)
    rep.set_defaults(func=cmd_reproduce)

    aud = sub.add_parser("audit", help="phase-1 audit support analyses (no cost; some are "
                         "post hoc on test predictions)")  # fmt: skip
    aud.add_argument("--n-resamples", type=int, default=10_000)
    aud.set_defaults(func=cmd_audit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
