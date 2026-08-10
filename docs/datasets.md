# Datasets

## Status

Two candidate datasets were investigated for Phase 1. One (MVTec AD) has
verified, unambiguous licensing and is selected for use under its stated
restrictions. The other (DAVIS 2017) has contradictory licensing information
across sources and is **not selected** pending clarification.

No dataset is downloaded or committed to this repository. Only metadata,
manifest schemas, and validation code exist locally.

---

## Dataset: MVTec AD

### Official Source

https://www.mvtec.com/company/research/datasets/mvtec-ad
(also mirrored at https://www.mvtec.com/research-teaching/datasets/mvtec-ad)

### Purpose

Industrial anomaly detection with pixel-precise ground-truth annotations —
directly matches this project's anomaly-detection phase and is a real
unsupervised-anomaly-detection benchmark (not a labeled classification
dataset repurposed as "anomaly detection").

### Tasks

Unsupervised anomaly detection and localization: models train on
defect-free images only and are evaluated on both normal and anomalous
test images.

### Annotations

Pixel-precise ground-truth segmentation masks for every anomalous test
image, across 15 object/texture categories. Training sets contain only
defect-free (normal) samples, consistent with the unsupervised-AD protocol.

### License

**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0)**, as stated on the official MVTec dataset page.

### Restrictions

- **Non-commercial use only.** The official page explicitly states
  commercial use of the dataset is not permitted.
- Attribution required.
- Derivative works/redistribution must be shared under the same
  CC BY-NC-SA 4.0 license (ShareAlike).
- This project is a non-commercial educational/portfolio project, which is
  consistent with the license. If this project's scope ever changes toward
  commercial use, MVTec AD must be dropped or MVTec contacted for a
  separate license.

### Repository Policy

Raw dataset files (images, masks, archives) will **never** be committed to
this GitHub repository. `.gitignore` already excludes `data/` and
`datasets/`. Only manifests (file lists + metadata, not image content) and
code are versioned.

### Download

1. Complete MVTec's official download-request form at the URL above
   (registration is required by MVTec — this project does not bypass it).
2. Download the archive from the link MVTec provides after registration.
3. Extract locally (or in a Colab-mounted storage location) to a path
   pointed at by this project's `EVAT_DATA_ROOT` configuration — see
   `src/evat/data/paths.py`. No path is hard-coded.
4. Run `scripts/prepare_mvtec.py` to validate structure and generate a
   manifest. This script does not download the dataset — it only
   validates and indexes files already present locally.

### Version

Verified against the official MVTec AD dataset page on 2026-08-10. Exact
dataset revision string is not exposed on the page beyond "MVTec AD" (the
extended/comprehensive release described in the MVTec AD journal paper);
this should be re-confirmed against the archive's own metadata once
downloaded, and recorded in the generated manifest's `dataset_version`
field.

### Decision

**CONDITIONAL — SELECTED FOR NON-COMMERCIAL USE ONLY.**

Verified suitable for this project's stated non-commercial, educational,
portfolio purpose. Must not be used, and derivative artifacts (trained
weights, generated crops, etc.) must not be distributed, in any commercial
context without separate permission from MVTec.

---

## Dataset: DAVIS 2017

### Official Source

https://davischallenge.org/davis2017/code.html

### Purpose

Was under consideration for video object segmentation / tracking phases
(Phases 3–4).

### Tasks

Semi-supervised and unsupervised video object segmentation.

### Annotations

Per-frame object segmentation masks; first-frame annotations for the
semi-supervised track; scribbles for the interactive track.

### License

**UNRESOLVED — contradictory information found, not verified.**

- The DAVIS toolkit/code repository (`fperazzi/davis-2017` on GitHub)
  states: "DAVIS is released under the BSD License" — but this license
  statement is scoped to the *evaluation code/toolkit*, not explicitly to
  the video/annotation data itself.
- A secondary (non-official) source claims the dataset annotations are
  under "Creative Commons Attribution 4.0" restricted to "non-commercial
  research use" — but standard CC BY 4.0 has no non-commercial clause, so
  this claim is internally inconsistent and cannot be trusted without the
  official license text.
- The official davischallenge.org site (index and downloads pages) does
  not display an explicit license statement or link to a license page for
  the dataset itself.

Per this project's rule against fabricating or guessing licensing
information, **DAVIS licensing is not verified** and no conclusion is drawn.

### Restrictions

Not determinable until the license is confirmed directly with the DAVIS
maintainers (contact: davisvideochallenge@gmail.com per the official site)
or an explicit license statement is located on an official page.

### Repository Policy

Not applicable — no download, ingestion, or adapter implementation will
occur until licensing is resolved.

### Download

Not applicable at this time.

### Version

DAVIS 2017 (semi-supervised/unsupervised tracks). Investigated 2026-08-10.

### Decision

**NOT SELECTED — pending license verification.**

A minimal, disabled adapter stub is scaffolded in
`src/evat/data/datasets/davis.py` for architectural symmetry, but it
intentionally raises `NotImplementedError` and must not be enabled until
license text is confirmed from an authoritative source and this document
is updated with a verified license section.
