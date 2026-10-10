"""Parse the PROMISE NFR ARFF file straight from the Zenodo tarball."""

import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

ARFF_MEMBER = "nfr/nfr.arff"
# Data rows look like: 3,'The system shall ...',PE
_ROW = re.compile(r"^(\d+),'(.*)',([A-Z]{1,2})\s*$")


@dataclass(frozen=True)
class PromiseRow:
    row_index: int
    project_id: int
    text: str
    label: str


def parse_arff(content: str) -> list[PromiseRow]:
    rows: list[PromiseRow] = []
    in_data = False
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        if stripped.upper().startswith("@DATA"):
            in_data = True
            continue
        if not in_data:
            continue
        match = _ROW.match(stripped)
        if match is None:
            raise ValueError(f"Unparseable ARFF data row: {stripped[:80]!r}")
        project, text, label = match.groups()
        rows.append(PromiseRow(len(rows), int(project), text.replace("\\'", "'"), label))
    return rows


def read_promise(tar_path: Path) -> list[PromiseRow]:
    with tarfile.open(tar_path) as archive:
        member = archive.extractfile(ARFF_MEMBER)
        if member is None:
            raise FileNotFoundError(f"{ARFF_MEMBER} not found in {tar_path}")
        # The file is Windows-1252 (e.g. 0x92 for a right single quote).
        return parse_arff(member.read().decode("cp1252", errors="replace"))
