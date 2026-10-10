"""Pinned public data sources: URL, expected SHA-256, license and citation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    filename: str
    sha256: str
    license: str
    citation: str


SOURCES: dict[str, Source] = {
    "promise_nfr": Source(
        name="promise_nfr",
        url="https://zenodo.org/records/268542/files/nfr.tar?download=1",
        filename="nfr.tar",
        sha256="28f9328ce72f1c54f09b8dd45573eb811b7a78959b5bf290561cbad1d5adb0b5",
        license="CC-BY-4.0",
        citation=(
            "Cleland-Huang, J., Mazrouee, S., Liguo, H., Port, D. (2007). nfr [Data set]. "
            "Zenodo. https://doi.org/10.5281/zenodo.268542"
        ),
    ),
    "nice": Source(
        name="nice",
        url="https://zenodo.org/records/14590935/files/PROMISE-relabeled-NICE.csv?download=1",
        filename="PROMISE-relabeled-NICE.csv",
        sha256="87acb2172bc6273b8e04c55b4de8bf090cb0a9980b9d23010b20af88245f2642",
        license="CC-BY-4.0",
        citation=(
            "Rejithkumar, G., Anish, P. R. (2025). NICE: Non-Functional Requirements "
            "Identification, Classification, and Explanation Using Small Language Models "
            "[Data set]. Zenodo. https://doi.org/10.5281/zenodo.14590935"
        ),
    ),
    "dalpiaz_re19": Source(
        name="dalpiaz_re19",
        url=(
            "https://zenodo.org/records/3309669/files/"
            "explainable-re/RE-2019-Materials-v0.zip?download=1"
        ),
        filename="RE-2019-Materials-v0.zip",
        sha256="823521077757a14724b261f4869f8859ac623c87fd98f2e61fb14ab286431f1a",
        license="GPL-3.0",
        citation=(
            "Dalpiaz, F., Dell'Anna, D., Aydemir, F. B., Cevikol, S. (2019). Requirements "
            "Classification with Interpretable Machine Learning and Dependency Parsing. RE'19. "
            "https://doi.org/10.5281/zenodo.3309669"
        ),
    ),
}
