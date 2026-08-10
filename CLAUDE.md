# CLAUDE.md

# Edge Video Anomaly Transformer

## 1. Project Purpose

This repository is a production-oriented research prototype for computer vision
and deep learning engineering.

The system will progressively demonstrate:

- video segmentation
- object tracking
- anomaly detection
- video understanding
- temporal representation learning
- a Video Transformer implemented from scratch
- inference optimization for constrained/edge-oriented environments
- reproducible experimentation
- production-quality software engineering

The project is designed to demonstrate the ability to take a computer-vision
problem through:

Research
→ Design
→ Data
→ Experimentation
→ Model Development
→ Evaluation
→ Iteration
→ Optimization
→ Deployment-oriented Inference

The project is being built to strengthen a Deep Learning / Computer Vision
engineering portfolio and specifically address gaps identified while
evaluating the project against a Deep Learning Software Engineer role.

The project must remain technically honest.

Do not claim production deployment, edge deployment, real-time performance,
commercial use, or measured improvements unless they have actually been
implemented and measured.

---

# 2. Primary Engineering Principle

Build this as an engineer, not as a collection of disconnected notebooks.

The repository must progressively evolve into one coherent system.

Every phase must build on the previous phase.

Do not create disconnected implementations merely to satisfy a checklist.

Prefer:

- reusable modules
- explicit interfaces
- configuration-driven experiments
- automated tests
- reproducible experiments
- measurable benchmarks
- clear documentation
- small focused commits
- backward-compatible changes

Avoid:

- giant notebooks containing all logic
- duplicated code
- hard-coded paths
- hidden global state
- unexplained magic numbers
- unnecessary dependencies
- untested ML code
- fabricated metrics
- copied implementations without attribution
- premature abstraction
- premature optimization

---

# 3. Non-Negotiable Rules

These rules apply to every phase.

## 3.1 Never fabricate evidence

Never invent:

- accuracy
- F1
- IoU
- Dice
- FPS
- latency
- memory usage
- speedup
- parameter count
- training time
- dataset size
- benchmark results
- deployment results

If something has not been measured, label it as:

- TODO
- not yet measured
- expected
- hypothesis

Never present an expected result as an achieved result.

---

## 3.2 Never claim work that was not done

Do not claim:

- edge-device deployment unless an actual edge device was used
- real-time inference unless measured FPS/latency supports the claim
- production deployment unless an actual production environment exists
- commercial deployment
- industrial deployment
- state-of-the-art performance
- research novelty
- from-scratch implementation

unless the repository contains evidence supporting the claim.

---

## 3.3 Dataset licensing

Do not assume that a publicly downloadable dataset is freely licensed for
all purposes.

For every dataset:

1. Identify the official source.
2. Record the dataset name and version.
3. Record the license or usage restrictions.
4. Document whether the dataset permits research/educational use.
5. Do not redistribute restricted raw datasets in this repository.
6. Do not commit datasets to Git.
7. Provide download/setup instructions instead.

The repository should contain dataset metadata and download/preparation
scripts where permitted.

Raw data must remain outside Git.

---

# 4. Project Architecture

The project should progressively converge toward:

Video Input
    |
    v
Frame Sampling
    |
    v
Segmentation
    |
    v
Object Tracking
    |
    v
Visual Feature Extraction
    |
    v
Temporal Sequence Construction
    |
    v
Video Transformer
    |
    v
Anomaly Detection
    |
    v
Inference / Benchmarking
    |
    v
Optimized Runtime

Not every phase needs to implement the entire pipeline.

Each phase should introduce one meaningful layer and integrate it with
previous work.

---

# 5. Planned Development Phases

The project must be implemented sequentially.

Planned phases:

1. Repository and engineering foundation
2. Dataset ingestion and validation
3. Video preprocessing
4. Segmentation
5. Object tracking
6. Visual feature extraction
7. Video Transformer from scratch
8. Baselines and ablation experiments
9. Anomaly detection
10. Inference optimization
11. End-to-end pipeline
12. Production hardening
13. CI/CD and quality gates
14. Final benchmarking and documentation

These phases may be refined if engineering evidence shows that a different
order is necessary.

However, do not skip a phase without explicitly explaining why.

Do not implement multiple future phases in the same phase.

---

# 6. Phase Execution Protocol

Before starting ANY phase, Claude Code MUST:

1. Read this CLAUDE.md completely.
2. Inspect the current repository.
3. Inspect the previous phase's implementation.
4. Inspect existing tests.
5. Inspect configuration files.
6. Inspect recent Git history.
7. Identify what the previous phase established.
8. Identify what the current phase must add.
9. Check for existing reusable code.
10. Avoid duplicating functionality.
11. Determine which work is LOCAL and which work is GOOGLE COLAB.

Before implementation, prepare a concise plan containing:

- objective
- current repository state
- existing components being reused
- files likely to change
- new interfaces/components
- tests required
- benchmark/evaluation requirements
- documentation updates
- compute environment

Then implement ONLY the current phase.

Do not automatically implement future phases.

---

# 7. Compute Environment Policy

Google Colab is the required environment for computationally heavy
machine-learning work in this project.

The repository itself must remain lightweight and reproducible locally, but
heavy computation MUST be executed in Google Colab whenever practical.

The separation is:

GitHub
    |
    v
Claude Code
    |
    +----------------------+
    |                      |
    v                      v
LOCAL                  GOOGLE COLAB
lightweight work       heavy computation
    |                      |
    |                      |
unit tests              GPU training
linting                 large inference
type checking           evaluation
smoke tests             ablations
static analysis         optimization
    |                      |
    +----------+-----------+
               |
               v
         measured results
               |
               v
            GitHub

---

# 8. Tasks That MUST Run in Google Colab

The following tasks are explicitly required to run in Google Colab:

- model training
- deep-learning fine-tuning
- segmentation training
- tracking experiments involving substantial video inference
- feature extraction over large datasets
- Video Transformer training
- Transformer ablation experiments
- anomaly-detection training
- large-scale evaluation
- hyperparameter experiments
- GPU inference benchmarks
- ONNX/model optimization experiments that require substantial compute
- batch inference over datasets
- generation of large experiment result sets
- any task that materially benefits from GPU acceleration

Do not run these workloads locally merely because they can technically run
on CPU.

---

# 9. Local Execution Policy

Local execution should primarily be used for:

- unit tests
- integration tests using small fixtures
- linting
- type checking
- static analysis
- package validation
- lightweight smoke tests
- small synthetic examples
- repository development
- code review
- Git operations

Do not accidentally trigger multi-hour training jobs locally.

Do not replace a required Colab GPU experiment with a local CPU run and
present the result as equivalent.

---

# 10. Google Colab Implementation Rules

## 10.1 Repository code remains the source of truth

Do not create a separate implementation inside the notebook.

Preferred:

GitHub repository
    |
    v
Google Colab
    |
    v
install/checkout repository
    |
    v
execute src/ modules
    |
    v
save experiment outputs

Core implementation belongs under:

src/

The notebook should primarily:

- configure the experiment
- load data
- call repository modules
- execute training/evaluation
- visualize results
- save metrics

---

## 10.2 Colab notebooks

Each compute-heavy phase should have a corresponding reproducible notebook
when appropriate.

Examples:

notebooks/
    03_segmentation_training.ipynb
    04_tracking_evaluation.ipynb
    06_transformer_training.ipynb
    07_transformer_ablation.ipynb
    08_anomaly_training.ipynb
    09_inference_optimization.ipynb

Notebook names should clearly identify the phase.

---

## 10.3 Colab hardware

Do not assume a specific GPU will always be available.

The code must detect available hardware.

Preferred behavior:

1. CUDA GPU if available
2. CPU fallback for lightweight tests

Do not assume a T4, A100, L4, or any other specific GPU unless a benchmark
explicitly records that hardware.

When recording benchmark results, record:

- GPU model
- CPU
- RAM
- CUDA version
- PyTorch version
- relevant runtime versions

---

## 10.4 Long-running Colab jobs

Free Colab sessions can disconnect or terminate.

Therefore:

- checkpoint training
- save checkpoints periodically
- save metrics periodically
- make experiments resumable
- save intermediate outputs
- avoid relying on notebook memory for critical state
- make training restartable
- use deterministic configuration

Do not design experiments that require one uninterrupted long-running
session if checkpointing is reasonably possible.

---

## 10.5 Colab storage

Do not assume `/content` is persistent.

Temporary data may be stored under:

/content/

Important artifacts should be explicitly saved to persistent storage or
regenerated through deterministic scripts.

Do not commit:

- datasets
- model checkpoints
- large generated artifacts
- temporary Colab files

to Git unless there is a specific documented reason.

---

## 10.6 Heavy-compute approval rule

Claude Code MUST NOT automatically launch expensive computation.

Before a heavy experiment, Claude Code should:

1. prepare the code
2. validate the implementation locally with lightweight tests
3. identify expected compute requirements
4. identify the exact Colab notebook/command
5. identify expected inputs/outputs
6. STOP

The user will run the heavy experiment in Google Colab.

Do not silently launch expensive computation locally.

---

# 11. Phase Completion Criteria

A phase is not complete merely because the code runs once.

A phase is complete only when:

- implementation exists
- relevant tests exist
- tests pass
- lint passes
- type checking passes where applicable
- configuration is reproducible
- documentation is updated
- experiment results are recorded if applicable
- no secrets are present
- no raw datasets are committed
- Git diff has been reviewed
- implementation is compatible with the previous phase

If a requirement cannot be completed, stop and report the blocker.

Do not silently weaken the requirement.

---

# 12. Testing Philosophy

Testing is required throughout the project.

Use multiple levels of testing.

## Unit tests

Test individual components:

- dataset parsing
- validation
- frame sampling
- transformations
- segmentation components
- tracker logic
- attention
- Transformer blocks
- anomaly scoring
- configuration validation

## Integration tests

Test interactions:

- dataset → preprocessing
- preprocessing → model
- segmentation → tracking
- feature extraction → Transformer
- Transformer → anomaly detection
- full inference pipeline

## Smoke tests

Every major model must have a lightweight smoke test.

Verify:

- imports work
- model constructs
- tensors have expected shapes
- forward pass works
- inference produces valid output

## Overfitting tests

For trainable models, where practical:

- use a tiny synthetic or tiny real dataset
- verify the model can overfit a small sample

If a new model cannot overfit a tiny controlled dataset, investigate before
running large experiments.

---

# 13. Reproducibility

Every experiment must record, where applicable:

- dataset version
- dataset split
- random seed
- model configuration
- training configuration
- preprocessing configuration
- optimizer
- learning rate
- batch size
- number of epochs
- checkpoint
- evaluation metrics
- environment information
- Git commit

Use configuration files rather than hard-coded experiment parameters.

Prefer:

configs/
    segmentation.yaml
    tracking.yaml
    transformer.yaml
    anomaly.yaml
    inference.yaml

over hard-coded values inside Python files.

---

# 14. Data Leakage Prevention

Never allow test data to influence training decisions.

Prohibited:

- tuning hyperparameters against the test set
- selecting the best model using test performance
- repeatedly evaluating test data during development
- mixing train and test samples
- inconsistent preprocessing
- leaking labels into features

Use:

train
validation
test

with clear separation.

If the official dataset provides a fixed split, preserve it unless there is
a documented reason to create an additional development split.

---

# 15. Model Development Rules

Always establish a baseline before introducing a more complex model.

For example:

simple CNN
    |
    v
CNN + LSTM
    |
    v
Video Transformer

The project must answer:

Why is the more complex model necessary?

Do not add complexity merely because a technology appears in a job
description.

Every significant architecture decision must have a reason.

---

# 16. Video Transformer Requirement

The Video Transformer is a key project requirement.

The Transformer must be implemented manually rather than imported as a
complete video Transformer architecture.

The implementation should explicitly demonstrate understanding of:

- temporal tokenization
- positional information
- query/key/value projections
- scaled dot-product attention
- multi-head attention
- residual connections
- normalization
- feed-forward networks
- temporal sequence modeling
- classification/anomaly head

Allowed:

- PyTorch tensor operations
- PyTorch Linear
- PyTorch LayerNorm
- PyTorch activation functions
- standard tensor utilities

Not allowed for the from-scratch requirement:

- TimeSformer as the complete model
- VideoMAE as the complete model
- Video Swin as the complete model
- any complete pretrained Video Transformer
- hiding a pretrained Video Transformer behind a wrapper

A pretrained CNN feature extractor may be used if clearly documented.

The implementation must contain our own attention and Transformer-block logic.

---

# 17. Computer Vision Scope

The project should demonstrate practical knowledge of:

- segmentation
- object tracking
- anomaly detection
- video understanding
- temporal modeling
- object-level visual representation

Clearly distinguish:

Academic knowledge
Project implementation
Benchmark evidence
Production experience

Do not claim production expertise merely because a model was tested in
a portfolio project.

---

# 18. Segmentation Rules

Segmentation should initially prioritize:

- correctness
- clean implementation
- reproducibility
- evaluation

Use appropriate metrics such as:

- IoU
- Dice
- pixel accuracy

Do not claim "real-time segmentation" until inference latency/FPS has been
measured under a clearly specified environment.

Do not claim edge-device segmentation unless an actual edge device is used.

---

# 19. Tracking Rules

Tracking should preserve object identity across frames.

At minimum, track:

- frame ID
- track ID
- bounding box
- confidence
- relevant object/mask information

Evaluate tracking quality where appropriate.

Do not claim production-grade tracking based only on a simple IoU tracker.

If ByteTrack or another established tracker is added, document:

- why it was selected
- what it provides
- what parts are reused
- what parts are implemented by us
- license

---

# 20. Anomaly Detection Rules

Clearly distinguish:

classification

from:

anomaly detection

If the system is trained on labeled normal/abnormal examples, describe it
accurately as supervised classification unless the methodology genuinely
implements anomaly detection.

Possible evaluation metrics:

- ROC-AUC
- PR-AUC
- precision
- recall
- F1
- false-positive rate
- false-negative rate

For localization, use appropriate pixel-level metrics.

---

# 21. Optimization Rules

Never claim an optimization without a before/after benchmark.

Every optimization should have:

baseline
change
measurement
result
tradeoff

For example:

PyTorch FP32
    |
    v
ONNX Runtime
    |
    v
optimized runtime

Record:

- latency
- throughput/FPS
- memory
- model size
- accuracy/F1 impact

Optimization is successful only if the benchmark demonstrates a meaningful
benefit or clearly documents the tradeoff.

---

# 22. Edge Deployment Honesty

This project may use free Google Colab and local CPU/runtime resources.

Therefore, do not claim:

- Jetson deployment
- Raspberry Pi deployment
- embedded GPU deployment
- physical edge deployment

unless an actual device was used.

If constrained runtime optimization is performed using CPU or another
resource-limited environment, describe it as:

"constrained inference benchmark"

or:

"edge-oriented inference optimization"

Do not call it actual edge deployment.

---

# 23. Production Engineering Standards

Prefer production patterns where they provide real value.

Use:

- typed interfaces
- configuration management
- structured logging
- meaningful exceptions
- deterministic behavior
- dependency pinning where practical
- clear module boundaries
- input validation
- output validation
- reproducible builds
- automated tests
- CI
- documentation
- versioned experiment configuration

Do not add infrastructure merely to make the project look complicated.

A simple reliable implementation is preferable to a complex unreliable
implementation.

---

# 24. Error Handling

Never silently swallow errors.

Avoid:

try:
    ...
except Exception:
    pass

unless there is a documented and justified reason.

Errors should provide enough context to diagnose:

- input
- component
- configuration
- expected behavior
- actual failure

---

# 25. Logging

Use structured, useful logs.

Logs should help answer:

- what operation is running?
- what input is being processed?
- what model/checkpoint is being used?
- what configuration is active?
- how long did the operation take?
- where did the failure occur?

Avoid excessive debug output during normal execution.

---

# 26. Configuration

Do not hard-code:

- dataset paths
- model paths
- output paths
- batch sizes
- learning rates
- device names
- experiment names

Configuration should be explicit.

Environment-specific values may use environment variables.

Never store credentials in configuration files.

---

# 27. Dependency Management

Keep dependencies minimal.

Before adding a dependency:

1. Check whether existing libraries already provide the capability.
2. Check whether the dependency is maintained.
3. Check installation complexity in Google Colab.
4. Check CPU compatibility.
5. Check licensing.
6. Add only if justified.

Do not add a large framework for a small utility.

---

# 28. Google Colab Requirements

The project must remain runnable on free Google Colab where practical.

Therefore:

- avoid requiring large GPUs
- use small models initially
- support CPU for lightweight tests
- avoid multi-GPU assumptions
- checkpoint long-running experiments
- save experiment outputs
- make experiments resumable where practical
- do not depend on interactive notebook state for core logic

The notebook should call repository modules.

Do not put the primary implementation only inside the notebook.

Preferred:

notebook
    |
    v
src/

Not:

notebook
    |
    v
hundreds of lines of duplicated implementation

---

# 29. Notebook Rules

Notebooks are for:

- exploration
- visualization
- experiment execution
- result analysis
- demonstration

Notebooks are NOT the primary location for reusable production logic.

Important logic belongs under:

src/

A notebook should be reproducible from a clean kernel as much as practical.

---

# 30. Dataset Strategy

The initial project will use public datasets only after verifying their
official licensing and usage terms.

Potential datasets include:

- DAVIS 2017 for video/segmentation work
- MVTec AD for industrial anomaly detection

Before using any dataset:

1. verify the official source
2. verify license
3. verify research/educational permissions
4. record the license in docs/datasets.md
5. do not commit raw data
6. do not redistribute restricted data
7. provide official download instructions where permitted

Do not assume that "free download" means unrestricted commercial use.

The final README must clearly state dataset licensing limitations.

---

# 31. Experiment Tracking

Every meaningful experiment should produce machine-readable results.

Prefer:

results/
    experiment_name/
        config.yaml
        metrics.json
        summary.md

rather than only writing results into notebook output.

Record enough information to reproduce the experiment.

---

# 32. Benchmarking

Benchmarks must specify:

- hardware
- software/runtime
- model version
- input dimensions
- batch size
- precision
- warm-up procedure
- number of measurements
- aggregation method

Do not compare numbers measured under incompatible conditions without stating
the difference.

For latency, distinguish where relevant:

- cold-start latency
- warm inference latency
- preprocessing latency
- model latency
- postprocessing latency
- end-to-end latency

---

# 33. Model Versioning

Every trained model/checkpoint should have identifiable metadata.

At minimum:

- model name
- model version
- dataset version
- configuration
- training date
- Git commit
- metrics

Do not rely only on filenames such as:

best_model_final_v7_really_final.pt

---

# 34. Security

Never commit:

- API keys
- AWS credentials
- GitHub tokens
- passwords
- private dataset credentials
- personal access tokens

Use environment variables or secret managers.

If a secret is accidentally exposed:

1. stop
2. revoke/rotate it
3. remove it from the repository
4. inspect Git history
5. report the issue

Do not merely delete the visible file.

---

# 35. Licensing and Attribution

Every external implementation or major algorithmic dependency must be
properly attributed where required.

Do not copy code from repositories without checking the license.

For datasets, document:

- source
- license
- restrictions
- download instructions
- whether redistribution is permitted

For pretrained models, document:

- model name
- source
- license
- intended-use restrictions
- whether weights are downloaded at runtime

---

# 36. Research Integrity

When comparing approaches:

- use the same evaluation split
- use the same evaluation metrics
- document preprocessing
- document training conditions
- report failures
- report negative results when relevant

Do not selectively report only the best run.

A failed experiment can still be useful evidence.

---

# 37. Architecture Decisions

For important decisions, document:

Decision
Context
Options considered
Chosen approach
Reason
Tradeoffs

Examples:

- why U-Net
- why a particular tracker
- why a particular feature encoder
- why a particular Transformer architecture
- why ONNX Runtime
- why a particular input resolution
- why a particular sequence length

Do not create formal architecture documents for trivial decisions.

---

# 38. Backward Compatibility

Every new phase must preserve previous functionality unless there is a
documented architectural reason to change it.

Before changing an existing interface:

1. inspect current usage
2. identify affected tests
3. update dependents
4. run the relevant test suite
5. document the change

Do not break previous phases casually.

---

# 39. Refactoring Rules

Refactor when:

- duplication becomes meaningful
- interfaces are unclear
- tests are difficult to write
- responsibilities are mixed
- performance requires architectural change

Do not refactor working code merely for stylistic preference.

Avoid large unrelated refactors during feature phases.

---

# 40. Performance Rules

Do not optimize before measuring.

Use profiling or benchmarking to identify the bottleneck.

Possible bottlenecks:

- data loading
- preprocessing
- segmentation
- feature extraction
- Transformer inference
- postprocessing
- serialization

Optimize the measured bottleneck first.

Heavy profiling and benchmark workloads must run in Google Colab when they
require substantial computation.

---

# 41. Git Rules

Never work directly on `main`.

Each phase gets its own branch.

Example:

phase/00-foundation
phase/01-data
phase/02-video
phase/03-segmentation
phase/04-tracking
phase/05-features
phase/06-transformer
phase/07-experiments
phase/08-anomaly
phase/09-optimization
phase/10-pipeline
phase/11-hardening
phase/12-ci

Use focused commits.

Examples:

chore: initialize production project structure
feat(data): add reproducible dataset ingestion
feat(video): add deterministic frame sampling
feat(segmentation): add segmentation baseline
feat(tracking): add object tracking
feat(transformer): implement video transformer from scratch
feat(anomaly): add anomaly detection
perf(inference): optimize constrained inference
test: expand pipeline integration coverage
docs: publish benchmark results

Avoid vague commits such as:

update
fix stuff
changes
final

---

# 42. Git Safety

Before committing:

git status
git diff

Verify:

- no secrets
- no credentials
- no datasets
- no temporary files
- no generated binaries
- no accidental notebook artifacts
- no unrelated changes

Never force-push unless explicitly instructed.

Never rewrite unrelated user work.

Never delete existing work merely to simplify the repository.

---

# 43. Pull Request Rules

Every phase should ideally produce one focused PR.

PR description should include:

## Objective

What problem does this phase solve?

## Implementation

What changed?

## Tests

What was run?

## Results

What were the measured results?

## Compute Environment

What ran locally?

What ran in Google Colab?

What GPU was used?

## Tradeoffs

What decisions were made?

## Limitations

What remains incomplete?

## Next Phase

What will build on this work?

---

# 44. Documentation Rules

Every major component needs documentation.

At minimum:

README.md
docs/architecture.md
docs/datasets.md
docs/experiments.md
docs/benchmarking.md
docs/model-card.md

Documentation must describe what actually exists.

Do not document future functionality as implemented.

Use clear diagrams and tables where they improve understanding.

---

# 45. Final Resume Integrity Rule

The final resume may claim only what this repository actually demonstrates.

Allowed if demonstrated:

- "Implemented a Video Transformer from scratch"
- "Built segmentation and object-tracking pipelines"
- "Developed anomaly detection for industrial inspection data"
- "Optimized inference with ONNX Runtime"
- "Benchmarked latency and throughput"

Not allowed unless actually demonstrated:

- "Deployed to edge devices"
- "Productionized on Jetson"
- "Real-time edge deployment"
- "State-of-the-art performance"
- "Production industrial inspection system"

Accuracy is more important than keyword coverage.

---

# 46. What Claude Code Must NOT Do

Do not:

- implement all phases at once
- skip tests
- fabricate benchmark numbers
- fabricate deployment
- fabricate dataset permissions
- fabricate research novelty
- fabricate production experience
- copy a complete external implementation and rename it
- use a prebuilt video Transformer when the requirement is from-scratch
- commit datasets
- commit secrets
- overwrite user work without checking
- delete tests because they fail
- weaken tests merely to make CI pass
- remove validation because it slows development
- introduce unnecessary infrastructure
- add technologies only for resume keywords
- run heavy training locally
- run large dataset processing locally
- silently substitute CPU results for required Colab GPU experiments
- implement future phases before the current phase is approved

---

# 47. Required End-of-Phase Report

After each phase, Claude Code must stop and report:

## Completed

Exact functionality implemented.

## Files Changed

List all files created or modified.

## Tests

Commands executed and pass/fail status.

## Quality Checks

- lint
- type checking
- build/package checks

## Compute Environment

Report:

- what ran locally
- what ran in Google Colab
- GPU used
- CPU if relevant
- approximate runtime
- dataset size processed
- whether the experiment was resumable

## Measurements

Only measured values.

## Design Decisions

Important decisions and reasons.

## Limitations

Known issues or incomplete areas.

## Git

Report:

- branch
- commit
- whether changes are ready for review

## Next Phase

Describe only the next phase.

Then STOP.

Do not automatically continue.

---

# 48. Definition of Done

The final project is complete only when:

- the complete pipeline runs
- segmentation works
- tracking works
- anomaly detection works
- video understanding works
- Video Transformer is implemented from scratch
- baselines are included
- experiments are reproducible
- optimization is benchmarked
- end-to-end inference works
- tests pass
- CI passes
- documentation is complete
- dataset licenses are documented
- no secrets are present
- no raw datasets are committed
- limitations are clearly documented
- final benchmark results are reproducible

---

# 49. Final Engineering Principle

The goal is not to maximize the number of technologies in the repository.

The goal is to build one coherent system demonstrating:

Strong CV fundamentals
        +
Deep learning implementation
        +
Research thinking
        +
Software engineering
        +
Evaluation discipline
        +
Performance engineering
        +
Deployment awareness

Every phase should make the final system more credible.

When a simpler implementation is sufficient, choose the simpler implementation.

When a claim requires evidence, measure it.

When something is not implemented, say so.

When something fails, investigate it rather than hiding it.

Build the system so that another engineer can clone the repository, understand
the architecture, reproduce the experiments, run the tests, and verify the
claims.

---

# 50. Mandatory Instruction for Every Future Phase

Before beginning ANY future phase, Claude Code MUST first read this entire
CLAUDE.md.

Then it must:

1. Inspect the current repository.
2. Inspect the previous phase.
3. Inspect tests and configuration.
4. Inspect recent Git history.
5. Determine LOCAL vs GOOGLE COLAB work.
6. State the implementation plan.
7. Implement ONLY the requested phase.
8. Add tests.
9. Run lightweight validation locally.
10. Prepare heavy-compute work for Google Colab.
11. Do NOT execute heavy computation locally.
12. Run or document the required Colab experiment as appropriate.
13. Record measured results.
14. Update documentation.
15. Review the Git diff.
16. Create the focused phase commit.
17. Report the completed work.
18. STOP.

Never skip the CLAUDE.md review.

Never start the next phase automatically.
