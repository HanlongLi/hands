# Human-to-Revo2 mapping: collect, review, export, train

This first mapping-learning pipeline works without hardware. It pairs human hand observations
with robot-hand targets, supports operator-reviewed corrections, trains a small baseline, and
evaluates it on separate clips or operators. It does not train an object-grasping policy.

## 1. Collect separate sessions

```sh
.venv/bin/r1h-panel --backend mock --sides right
```

Start the camera, check handedness, and calibrate open/fist/pinch. Enter **Operator ID** and
**Session ID**, then record landmarks. Use varied poses, including partial closure, individual
fingers, pinch and transitions. Record several genuinely separate clips; frames from one clip
are strongly related, so randomly splitting those frames would give misleading validation.

Operator/session IDs are optional text identifiers, not account credentials. Set them before
recording: the recording freezes them and its calibration at the start. Calibration changes or
handedness swaps end the active recording. IDs on a legacy clip remain unknown.

## 2. Review robot poses

1. Record/import a clip or load the synthetic example.
2. Replay to a useful pose and pause. Open **Mapping training data** below the clip controls.
3. Click **Edit paused pose**. Wait for the frame to appear if the panel reports it is still arriving.
4. Adjust the six sliders. The solid 3D hand shows your proposed **Operator label**; motor
   targets do not change. Use **Reviewer** and **Pose / note** to identify the review.
5. Click **Save reviewed pose**. Use **Next frame** to advance one frame while paused, or resume
   replay to reach another gesture. Frame changes invalidate the editor until you select again.
6. **Download clip** saves landmarks and annotations together. Unsaved browser data disappears
   on refresh. **Remove label** deletes the selected frame's label.

A reviewed label means a person explicitly chose that kinematic target. It has not been checked
for collisions or validated against real motor motion. Approving an unchanged pose is useful,
but editing only the difficult poses can bias the dataset: label a balanced set of easy and hard
poses. Reviewers should use the same intended gesture conventions.

## 3. Export paired data

**Export mapping dataset** downloads a compressed NPZ without sending motor commands. You can
also export several saved JSON clips from the command line:

```sh
.venv/bin/python -m r1hands.data.mapping export datasets/clips/*.json --output datasets/mapping
```

Each NPZ has one row per original frame, preserving time and gaps:

| Array | Meaning |
|---|---|
| `time_s` | Original sample time in seconds |
| `human_landmarks_world` | 21×3 camera-estimated hand landmarks; missing observations use NaN |
| `human_landmarks_image` | 21×2 image coordinates |
| `human_landmarks_palm` | Wrist-centered, palm-aligned, palm-length-normalized coordinates |
| `human_features` | Six flexion/opposition angles in radians |
| `raw_robot_target` | Existing analytic mapper output, before temporal smoothing |
| `filtered_robot_target` | Mapper with acquisition gating and One Euro smoothing; holds across gaps |
| `corrected_robot_target` | Explicit operator labels; NaN on unlabeled frames |
| `observation_valid`, `tracking_valid` | Observation quality and temporal acquisition masks |
| `label_valid`, `training_mask` | Label availability and usable reviewed input/label pairs |
| `quality_reason` | Missing hand, uncertain identity, clipping, ambiguity, bad geometry, acquiring or valid |
| `label_source`, `reviewer`, `label_note` | Label provenance and annotation text |
| `collision_checked` | False on every row: collision checking is not implemented |
| `metadata_json` | Schema/version, joint order, units, source group, profile, filtering and summary |

Low-confidence observations are preserved when the right-hand candidate is identifiable, but
excluded from training. Ambiguous/missing candidates produce missing selected landmarks; the
original JSON retains every raw detector observation. Export uses the clip's recorded profile,
not a later panel calibration. Automatic targets are **pseudo-labels from our current algorithm**,
not measured robot actions or independent ground truth. Unlabeled rows do not silently become
reviewed labels. No collision-corrected target is fabricated.

The command-line exporter also writes a `.manifest.json` beside each NPZ. In the panel download,
the same metadata is stored in `metadata_json` inside the NPZ. Load safely with:

```python
import json
import numpy as np
with np.load("dataset.npz", allow_pickle=False) as data:
    metadata = json.loads(str(data["metadata_json"]))
    x = data["human_features"][data["training_mask"]]
    y = data["corrected_robot_target"][data["training_mask"]]
```

## 4. Train and evaluate a baseline

Label at least two distinct clips before training. A NumPy ridge-regression baseline maps the
six human angles to six normalized robot targets. It requires no PyTorch installation.

```sh
.venv/bin/python -m r1hands.data.mapping train datasets/mapping/*.npz --output models/mapping.npz
```

The trainer holds out whole clips (approximately 20%, at least one), fits feature normalization
on training samples only, and writes `models/mapping.report.json`. The report includes overall
and per-joint MAE, the existing analytic mapper's MAE on the same reviewed labels, sample counts,
synthetic-data counts, split groups and the exact `train_datasets`/`validation_datasets` paths.
All errors are normalized motor units, not fingertip distances or physical accuracy.

For cross-operator evaluation, collect known operator IDs from at least two operators and use:

```sh
.venv/bin/python -m r1hands.data.mapping train datasets/mapping/*.npz --split-by operator --output models/cross-operator.npz
```

The saved report lists held-out files. To score them again, pass those files only:

```sh
.venv/bin/python -m r1hands.data.mapping evaluate models/mapping.npz path/to/held-out-clip.npz
```

Evaluation rejects training clips, and operator-mode evaluation also rejects training operators
or missing operator IDs. Re-exporting the same observations with edited annotations/calibration
keeps their clip group unchanged; supplying duplicate exports together is rejected. Clip groups
are content-based rather than filename-based. This detects exact reused observations, not all
near-duplicate recordings: use genuinely separate recording sessions and operators.

For a pipeline smoke test without manual labels, `train ... --target teacher` explicitly uses
analytic pseudo-labels. Such a run tests distillation of today's mapper; it does not establish an
improved mapping. The report records this label source. The example clip is synthetic and is
flagged as such. Enough pose diversity and independent human review are needed before model
quality conclusions; a successful fit on a few labels is just a functional baseline.

## 5. Compare the learned mapping in 3D

```sh
.venv/bin/r1h-panel --backend mock --sides right --mapping-model models/mapping.npz
```

The **Mapped hand** preview now uses the trained baseline and clearly labels it as preview-only.
Live camera observations and replay both use it, with the existing tracking gates and smoothing.
The learned weights use raw human angles rather than the analytic open/fist normalization, so
the recorded/current calibration checkbox does not change learned predictions. Rewind or start
a fresh replay when comparing runs. Remove `--mapping-model` to return to the analytic preview.

This option changes the read-only mapping preview only. Motor-control skills and the explicit
arm flow continue to use the analytic mapper and existing guards. Applying a learned mapping
to hardware requires a later integration and collision/physical validation step.

## Scope and next stages

The current baseline establishes data quality, reviewed supervision, independent evaluation and
visual inspection. It does not solve finger collisions, infer object contacts, or provide a task
policy. Collision validation can later add its own target provenance and masks. Task imitation
will need synchronized scene observations, robot states/actions, episode/task labels and outcomes.

Generated `datasets/` and `models/` directories are ignored by Git. Keep downloaded raw JSON
clips as well as NPZ exports so mapping changes can be evaluated on the same observations.
