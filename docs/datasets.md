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

---

## Current Dataset Decisions

### MVTec AD

CC BY-NC-SA 4.0, non-commercial use only, registration required.
CONDITIONAL — selected. Purpose: industrial anomaly detection component
(Phase 8+). See full record above.

### DAVIS 2017

**BLOCKED — LICENSE UNRESOLVED.** Not selected. See full record above.
Do not use until an authoritative license statement is found.

### Video Dataset Candidates

Investigated for the segmentation/tracking/temporal component (separate
from MVTec AD, which does not provide video/tracking data). Verified
against official sources on 2026-08-10.

| Candidate | Official source | License (verified) | Segmentation | Tracking | Colab feasibility |
|---|---|---|---|---|---|
| YouTube-VOS | youtube-vos.org | CC BY 4.0 (annotations) + explicit "non-commercial research only" restriction on the dataset itself, stated on the official Terms of Use page | Yes — dense per-frame object masks | Implicit — consistent object IDs across frames (segmentation-based tracking) | Good — moderate video count (3,471 train / 507 val), direct Google Drive download, no proprietary tooling |
| MOT17 (MOTChallenge) | motchallenge.net | CC BY-NC-SA 3.0 | No — bounding boxes only, no masks | Yes — explicit multi-object tracking ground truth (IDs across frames) | Excellent — only 14 sequences total, small download |
| KITTI (tracking/MOTS) | cvlibs.net/datasets/kitti | CC BY-NC-SA 3.0 | Partial — KITTI-MOTS extension has masks for 2 classes (car, pedestrian) only | Yes | Good — moderate size, but domain-narrow (autonomous-driving only, 2 categories) |
| BDD100K | bdd100k.com / doc.bdd100k.com/license.html | Free for research/non-commercial (UC Berkeley copyright); commercial use requires separate license via BDD/BAIR Commons or UC Berkeley OTL | Yes — semantic + instance segmentation subsets | Yes — MOT subset with tracking labels | Poor for this project's scale — full dataset ~100K videos / reported in the TB range; usable only via curated subsets, adding scope/complexity not verified to fit free Colab within this project's timeline |
| SegTrack(v2) | official host not consistently reachable/maintained; no single authoritative current license page found | LICENSE UNCLEAR | Yes | Limited | Not evaluated further — license could not be verified from an official, currently-maintained source |

Comparison table (qualitative scoring, "Strong / Moderate / Weak / Unknown"):

| Criterion | YouTube-VOS | MOT17 | KITTI (MOTS) | BDD100K | SegTrack(v2) |
|---|---|---|---|---|---|
| 1. License clarity | Strong | Strong | Strong | Moderate (research free, commercial gated) | Unknown |
| 2. Research/portfolio usability | Strong | Strong | Moderate | Moderate | Unknown |
| 3. Segmentation support | Strong | Weak (none) | Moderate (2 classes only) | Strong | Moderate |
| 4. Tracking support | Moderate (implicit via mask IDs) | Strong (explicit MOT ground truth) | Strong | Strong | Weak |
| 5. Video/temporal support | Strong | Strong | Strong | Strong | Moderate |
| 6. Dataset size (Colab-appropriate) | Moderate (usable via subset) | Strong (already small) | Moderate | Weak (very large) | Unknown |
| 7. Colab feasibility | Strong | Strong | Moderate | Weak | Unknown |
| 8. Annotation quality | Strong | Strong | Strong | Strong | Unknown |
| 9. Reproducibility | Strong (direct download link, no special software) | Strong | Moderate (requires KITTI dev kit conventions) | Moderate (large infra to reproduce fully) | Unknown |
| 10. Relevance to industrial/CV engineering story | Strong (general segmentation+tracking narrative feeds Video Transformer) | Moderate (tracking-only, no masks) | Moderate (domain-narrow) | Moderate (autonomous-driving-narrow) | Unknown |

### Selected Video Dataset

**YouTube-VOS**

- **Official source:** https://youtube-vos.org/dataset/vos/ (terms: https://youtube-vos.org/dataset/term/)
- **License:** Annotations under CC BY 4.0; the dataset as a whole is explicitly restricted by the official Terms of Use to "non-commercial research purpose only." This restriction is stated directly on the official terms page (not inferred from a third party), so — unlike DAVIS — it is verified rather than contradictory.
- **Restrictions:** Non-commercial research use only. Access requires agreeing to the stated terms and conditions; sharing with colleagues is permitted only if they also agree to the same terms. No indication that raw video redistribution is permitted beyond the dataset's own distribution channel.
- **Technical capabilities:** 3,471 training videos / 507 validation videos (2019 split; 2022 split adds a larger test set), dense (6 fps) per-frame object segmentation masks, 65 training categories plus held-out categories for generalization testing. Exact total frame count and on-disk size in GB: **not verified** — not stated on the official pages reviewed.
- **Colab feasibility:** Direct Google Drive download link provided (no proprietary software, no paid API). A subset of training videos can be used initially rather than the full 3,471, keeping storage/preprocessing within free-Colab limits. Segmentation masks are lightweight (indexed PNGs), not raw video re-encoding.
- **Reason for selection:** It is the only candidate that provides both dense segmentation masks and inherent object-level continuity across frames (i.e., tracking by consistent instance ID) in one dataset, with an explicit, unambiguous license statement from the official source, no registration paywall, and a download path simple enough to script and validate the way `scripts/prepare_mvtec.py` already does for MVTec AD. It maps directly onto this project's planned Segmentation → Tracking → Temporal → Video Transformer pipeline without forcing two different datasets together.

### Backup Dataset

**MOT17 (MOTChallenge)**

- **Official source:** https://motchallenge.net/
- **License:** CC BY-NC-SA 3.0 — non-commercial, attribution required, share-alike.
- **Role:** If YouTube-VOS's mask-based implicit tracking proves insufficient for the Phase 4 tracking component (e.g., if explicit multi-object bounding-box tracking metrics are needed), MOT17 is a small (14 sequences total), clearly-licensed, Colab-trivial fallback focused purely on tracking ground truth. It does not provide segmentation masks, so it would supplement rather than replace YouTube-VOS.

### Rejected Candidates — reasons

- **DAVIS 2017:** already blocked, license unresolved (see above). Remains blocked.
- **KITTI / KITTI-MOTS:** license is clear (CC BY-NC-SA 3.0) and technically capable, but domain-narrow (autonomous-driving scenes only, MOTS masks limited to 2 object classes) — weaker general "video understanding" story than YouTube-VOS, and adds KITTI-specific devkit/calibration-file conventions not needed elsewhere in this project. Kept as a documented alternative, not selected.
- **BDD100K:** license permits non-commercial research use, but the full dataset scale (~100K videos) is not practical to work with on free Google Colab within this project's scope; would require committing to a specific curated subset whose exact size/composition was not verified from the official source in this pass. Reasonable candidate for a future phase if more compute becomes available, not selected now.
- **SegTrack(v2):** no single official, currently-maintained source with a clear license statement was found. Marked **LICENSE UNCLEAR** and not selected, per the project's rule against guessing licensing.

### Dataset Policy

Raw data (videos, frames, masks, archives) for any dataset — MVTec AD or
the selected video dataset — will **not** be committed to this GitHub
repository. `.gitignore` excludes top-level `/data/` and `/datasets/`.
Only manifests (metadata/paths) and code are versioned. Download/setup for
YouTube-VOS will follow the same pattern already established for MVTec AD
(`scripts/prepare_<dataset>.py`, `EVAT_DATA_ROOT`-relative paths) — to be
implemented as part of Phase 2, not this task.
