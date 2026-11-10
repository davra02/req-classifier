# Data licenses and attribution

This repository does **not** redistribute the datasets. `uv run reqbench data download` fetches
them from their original locations and verifies their SHA-256 checksums
(`bench/src/reqbench/data/sources.py`). Only derived identifiers, text hashes, split
assignments and aggregate statistics are committed. The one exception is
`questions/examples_v1.yaml`, which quotes 12 PROMISE requirements (CC BY 4.0) as few-shot
examples, with attribution in its header.

| Dataset | Use here | License | Citation |
|---|---|---|---|
| PROMISE NFR | Main benchmark (type classification) | CC BY 4.0 | Cleland-Huang, J., Mazrouee, S., Liguo, H., Port, D. (2007). nfr [Data set]. Zenodo. https://doi.org/10.5281/zenodo.268542 |
| NICE relabeled PROMISE | Reference for deduplication and multi-label analysis | CC BY 4.0 | Rejithkumar, G., Anish, P. R. (2025). NICE [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14590935 |
| RE'19 materials (Dalpiaz et al.) | External functional/quality test set; label-noise analysis | GPL-3.0 | Dalpiaz, F., Dell'Anna, D., Aydemir, F. B., Çevikol, S. (2019). Requirements Classification with Interpretable Machine Learning and Dependency Parsing. RE'19. https://doi.org/10.5281/zenodo.3309669 |

The fresh evaluation set (plan §6.9) quotes short requirement sentences from 12 public GitHub
repositories under MIT or Apache-2.0 (`data/fresh/requirements.csv`). Each sentence links to
its source through `doc_id`; `data/fresh/documents.csv` records the repository URL at the
pinned commit, its license and the retrieval date. Copyright remains with the respective
authors under those licenses.
