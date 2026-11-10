"""Second-annotator subsamples and inter-annotator agreement for the fresh set.

Annotator A labels every sentence. Two further annotators, B and C, each label a disjoint
subsample (deviation of 2026-11-10 in docs/PREREGISTRATION.md). The subsamples are drawn with
the project seed, stratified by A's label (proportional allocation, at least MIN_PER_LABEL per
label when available) and dealt alternately to B and C, so both get the same mix. A reserve
annotator (D, ...) gets a further disjoint subsample drawn the same way from the rows that
have no second annotator yet (`add_subset`).
"""

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from reqcore.hashing import content_hash
from reqcore.questions import load_taxonomy
from sklearn.metrics import cohen_kappa_score

from reqbench.fresh.collect import SEED

SUBSET_SIZE = {"b": 30, "c": 30}
EXTRA_SIZE = 30
SECOND_ANNOTATORS = ("b", "c", "d", "e")
MIN_PER_LABEL = 2
ASSIGN_COLUMN = "second_annotator"
COLUMN_ORDER = [
    "req_id", "doc_id", "location", "text", "annotator_a", "annotator_b", "annotator_c",
    ASSIGN_COLUMN, "adjudicated", "adjudication_note",
]  # fmt: skip
N_BOOTSTRAP = 2000


def _allocate(counts: pd.Series, total: int) -> dict[str, int]:
    """Proportional allocation by largest remainder, with a floor of MIN_PER_LABEL."""
    exact = counts / counts.sum() * total
    alloc = {k: max(min(MIN_PER_LABEL, int(n)), math.floor(exact[k])) for k, n in counts.items()}
    remainders = sorted(counts.index, key=lambda k: (-(exact[k] - math.floor(exact[k])), k))
    while sum(alloc.values()) < total:
        for k in remainders:
            if sum(alloc.values()) < total and alloc[k] < counts[k]:
                alloc[k] += 1
    while sum(alloc.values()) > total:  # the floors overshot: trim the largest labels
        k = max(alloc, key=lambda key: (alloc[key] - exact[key], alloc[key]))
        alloc[k] -= 1
    return alloc


def assign_subsets(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Return a copy with `annotator_c` and `second_annotator` filled for the subsamples."""
    out = df.copy()
    for column in ("annotator_b", "annotator_c", ASSIGN_COLUMN):
        if column not in out:
            out[column] = ""
    if (out["annotator_a"] == "").any():
        raise RuntimeError("Annotator A has not finished; draw the subsamples afterwards.")
    if (out[["annotator_b", "annotator_c"]] != "").any().any():
        raise RuntimeError("B or C have already started; refusing to redraw the subsamples.")
    alloc = _allocate(out["annotator_a"].value_counts(), sum(SUBSET_SIZE.values()))
    chosen: list[int] = []
    for label in sorted(alloc):
        rows = out.index[out["annotator_a"] == label]
        ranked = sorted(rows, key=lambda i: content_hash([seed, "subset", out.at[i, "req_id"]]))
        chosen += ranked[: alloc[label]]
    out[ASSIGN_COLUMN] = ""
    annotators = sorted(SUBSET_SIZE)
    for position, i in enumerate(chosen):
        out.at[i, ASSIGN_COLUMN] = annotators[position % len(annotators)]
    return _ordered(out)


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    """COLUMN_ORDER, with any extra annotator column placed before `second_annotator`."""
    extra = sorted(c for c in df if c.startswith("annotator_") and c not in COLUMN_ORDER)
    at = COLUMN_ORDER.index(ASSIGN_COLUMN)
    return df[COLUMN_ORDER[:at] + extra + COLUMN_ORDER[at:]]


def add_subset(
    df: pd.DataFrame, who: str, size: int = EXTRA_SIZE, seed: int = SEED
) -> pd.DataFrame:
    """Draw a further disjoint subsample for a reserve annotator, from unassigned rows."""
    if who not in SECOND_ANNOTATORS or who in SUBSET_SIZE:
        raise ValueError(f"{who!r} is not a reserve annotator; use one of d, e")
    if (df[ASSIGN_COLUMN] == who).any():
        raise RuntimeError(f"Annotator {who.upper()} already has a subsample.")
    if (df["annotator_a"] == "").any():
        raise RuntimeError("Annotator A has not finished; draw the subsamples afterwards.")
    out = df.copy()
    out[f"annotator_{who}"] = ""
    pool = out[out[ASSIGN_COLUMN] == ""]
    if len(pool) < size:
        raise RuntimeError(f"Only {len(pool)} rows left without a second annotator.")
    alloc = _allocate(pool["annotator_a"].value_counts(), size)
    for label in sorted(alloc):
        rows = pool.index[pool["annotator_a"] == label]
        ranked = sorted(
            rows, key=lambda i: content_hash([seed, "subset", who, out.at[i, "req_id"]])
        )
        out.loc[ranked[: alloc[label]], ASSIGN_COLUMN] = who
    return _ordered(out)


def _kappa_with_ci(a: list[str], b: list[str], seed: int) -> dict:
    n = len(a)
    if n == 0:
        return {"n": 0}
    x, y = np.array(a), np.array(b)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        if len(set(x[idx]) | set(y[idx])) > 1:
            boot.append(cohen_kappa_score(x[idx], y[idx]))
    kappa = cohen_kappa_score(x, y) if len(set(a) | set(b)) > 1 else float("nan")
    low, high = np.nanpercentile(boot, [2.5, 97.5]) if boot else (float("nan"),) * 2
    return {
        "n": n,
        "agreement": round(float((x == y).mean()), 3),
        "kappa": round(float(kappa), 3),
        "ci95": [round(float(low), 3), round(float(high), 3)],
    }


def agreement(df: pd.DataFrame, seed: int = SEED) -> dict:
    """Cohen's kappa of A against each second annotator and all pooled, leaves and level 1."""
    taxonomy = load_taxonomy()
    report: dict = {}
    assigned = sorted(w for w in df[ASSIGN_COLUMN].unique() if w)
    pairs = [(f"A-{w.upper()}", [w]) for w in assigned] + [("A-second", assigned)]
    for name, who in pairs:
        rows = df[df[ASSIGN_COLUMN].isin(who)]
        second = rows.apply(lambda r: r[f"annotator_{r[ASSIGN_COLUMN]}"], axis=1)
        done = (rows["annotator_a"] != "") & (second != "")
        a, s = rows.loc[done, "annotator_a"].tolist(), second[done].tolist()
        kept = [(p, q) for p, q in zip(a, s, strict=True) if "discard" not in (p, q)]
        report[name] = {
            "pending": int((~done).sum()),
            "leaves_with_discard": _kappa_with_ci(a, s, seed),
            "level1": _kappa_with_ci(
                [taxonomy.level1_of(p) for p, _ in kept],
                [taxonomy.level1_of(q) for _, q in kept],
                seed,
            ),
            "disagreements": [
                {"req_id": r, "a": p, "second": q}
                for r, p, q in zip(rows.loc[done, "req_id"], a, s, strict=True)
                if p != q
            ],
        }
    return report


# One letter per label for the WhatsApp checklist; definitions come from taxonomy_v1.yaml.
CODES = {
    "F": "functional", "P": "performance", "S": "security", "U": "usability",
    "C": "operational_constraint", "O": "other", "D": "discard",
}  # fmt: skip
_REPLY_ITEM = re.compile(r"(\d{1,3})\s*[-:=.)]?\s*([A-Za-z])\b")
_REPLY_WHO = re.compile(r"\banotador\s+([A-Z])\b", re.IGNORECASE)
SPANISH_NAMES = {
    "functional": "Funcional", "performance": "Rendimiento", "security": "Seguridad",
    "usability": "Usabilidad", "operational_constraint": "Restricción operativa",
    "other": "Otra calidad", "discard": "Descartar (no es un requisito)",
}  # fmt: skip
FORM_TEMPLATE = Path(__file__).with_name("form.html")


def subset(df: pd.DataFrame, who: str) -> pd.DataFrame:
    """The rows of `who`'s subsample, in the CSV order (already shuffled, never by label)."""
    rows = df[df[ASSIGN_COLUMN] == who]
    if rows.empty:
        raise RuntimeError(f"No rows assigned to {who!r}; run `reqbench fresh assign` first.")
    return rows


def checklist_message(df: pd.DataFrame, who: str) -> str:
    """Plain-text checklist to paste into WhatsApp. It never contains other annotators' labels."""
    taxonomy = load_taxonomy()
    rows = subset(df, who)
    lines = [
        f"Anotación de requisitos ({len(rows)} frases). Para cada frase elige UNA letra según "
        "lo que pide la frase, leyendo solo su texto y estas definiciones (en inglés, las mismas "
        "para todos los anotadores). No consultes a nadie ni ninguna IA.",
        "",
    ]
    for code, leaf in CODES.items():
        if leaf == "discard":
            lines.append(f"{code} = discard: not a requirement (heading, note, process rule...).")
            continue
        definition = taxonomy.classes[leaf]
        line = f"{code} = {leaf}: {definition['what']}"
        if definition.get("not_for"):
            line += f" NOT: {definition['not_for']}"
        lines.append(line)
    lines += [
        "",
        "Responde con número y letra, por ejemplo: 1F 2S 3O 4D …",
        "",
    ]
    lines += [f"{n}. {text}" for n, text in enumerate(rows["text"], start=1)]
    return "\n".join(lines)


def render_form(df: pd.DataFrame, who: str) -> str:
    """Interactive one-page checklist for `who` (published as an artifact and sent by link).

    It embeds only `who`'s sentences and the leaf definitions, never anyone's labels. The answers
    leave the page as a WhatsApp message in the format `parse_reply` reads.
    """
    taxonomy = load_taxonomy()
    labels = []
    for code, leaf in CODES.items():
        definition = taxonomy.classes.get(leaf, {})
        what = definition.get("what", "Not a requirement (heading, note, process rule...).")
        labels.append(
            {"code": code, "key": leaf, "name": SPANISH_NAMES[leaf], "what": what,
             "not_for": definition.get("not_for", "")}
        )  # fmt: skip
    items = [{"n": n, "text": t} for n, t in enumerate(subset(df, who)["text"], start=1)]
    data = json.dumps({"who": who.upper(), "items": items, "labels": labels}, ensure_ascii=False)
    data = data.replace("</", "<\\/")  # untrusted text must not close the <script> element
    return FORM_TEMPLATE.read_text().replace("__DATA__", data).replace("__WHO__", who.upper())


def parse_reply(df: pd.DataFrame, who: str, reply: str) -> dict[str, str]:
    """Map a WhatsApp reply ("1F 2S ...") to {req_id: leaf}; every item exactly once."""
    req_ids = subset(df, who)["req_id"].tolist()
    signed = {m.upper() for m in _REPLY_WHO.findall(reply)}
    if signed and signed != {who.upper()}:
        raise ValueError(f"this reply is signed by annotator {sorted(signed)}, not {who.upper()}")
    answers: dict[int, str] = {}
    errors: list[str] = []
    for number, code in _REPLY_ITEM.findall(reply):
        n, code = int(number), code.upper()
        if not 1 <= n <= len(req_ids):
            errors.append(f"item {n} does not exist")
        elif code not in CODES:
            errors.append(f"item {n}: unknown letter {code!r}")
        elif n in answers and answers[n] != CODES[code]:
            errors.append(f"item {n} answered twice ({answers[n]} and {CODES[code]})")
        else:
            answers[n] = CODES[code]
    missing = [n for n in range(1, len(req_ids) + 1) if n not in answers]
    if missing:
        errors.append(f"missing items: {missing}")
    if errors:
        raise ValueError("; ".join(errors))
    return {req_ids[n - 1]: leaf for n, leaf in answers.items()}


def apply_reply(df: pd.DataFrame, who: str, labels: dict[str, str]) -> pd.DataFrame:
    column = f"annotator_{who}"
    out = df.copy()
    if (out.loc[out["req_id"].isin(labels), column] != "").any():
        raise RuntimeError(f"{column} already has labels for this subsample; not overwriting.")
    out[column] = out.apply(lambda r: labels.get(r["req_id"], r[column]), axis=1)
    return out
