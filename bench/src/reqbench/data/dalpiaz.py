"""Read requirement texts from the Dalpiaz et al. RE'19 materials, straight from the zip.

Only the manually tagged files that ship the requirement text are read; the others (ds2, ds3,
esa-eucl-est, 8combined, INDcombined) have the text column blanked out.
"""

import zipfile
from pathlib import Path

import pandas as pd

MEMBER_DIR = "explainable-re-RE-2019-Materials-876e606/Manually tagged datasets/"
PUBLIC_TEXT_FILES = ("dronology", "leeds", "reqview", "wasp", "promise-reclass")


def read_dalpiaz_texts(zip_path: Path) -> list[str]:
    texts: list[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        for name in PUBLIC_TEXT_FILES:
            with archive.open(f"{MEMBER_DIR}{name}.csv") as fh:
                frame = pd.read_csv(fh, encoding="utf-8")
            # promise-reclass keeps the ARFF quoting: 'The system shall ...'
            texts += [str(t).strip().strip("'") for t in frame["RequirementText"].dropna()]
    return texts
