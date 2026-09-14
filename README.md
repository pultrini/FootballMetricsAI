# Football Analytics

Open-source, modular pipeline for extracting structured football analytics from match video.
The project now contains **Milestones 1 through 9**: domain schemas, canonical pitch
geometry, tested shot geometry, detector/tracker contracts, Ultralytics adapters for object
and pitch-keypoint detection, ByteTrack, robust dynamic homography, metric trajectories,
per-frame minimaps, dedicated ball tracking, smoothed possession, pass and shot event detection,
a validated historical-shot dataset, a trainable expected-goals baseline, match statistics,
tabular exports, heatmaps, shot maps, average-position maps, and the end-to-end video CLI.

## Requirements

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/) installed

## Setup and verification

```bash
git clone <repository>
cd football-analytics
uv sync
uv run pytest
uv run ruff check .
uv run football-analytics --help
```

Format the code with:

```bash
uv run ruff format .
```

The committed `uv.lock` makes application and development environments reproducible. Add
runtime dependencies with `uv add <package>` and development tools with
`uv add --dev <package>`.

## Architecture

```text
video
  -> Detector -> YoloDetector -> detection records
  -> Tracker -> ByteTrackTracker -> tracking records
  -> PitchKeypointDetector -> pitch landmark observations
  -> RANSAC homography -> confidence-gated metric positions
  -> metric trajectories -> per-frame minimap / video overlay
  -> ball association -> temporally smoothed player/team possession
  -> PassDetector / ShotDetector -> event records
  -> trained xG model -> shot probabilities and player/team xG
  -> player/team statistics -> Parquet / CSV / JSON and visualizations
```

The package uses a `src` layout. Pipeline boundaries exchange immutable Pydantic records from
`storage.schemas`; `pitch` owns the canonical 105 m x 68 m coordinate system; and `analytics`
contains pure, independently testable calculations. Player and ball positions are separate
records so future tabular exports remain normalized.

Canonical coordinates start at the top-left corner of a top-down pitch. `x` runs from 0 to
105 m along the touchline and `y` runs from 0 to 68 m across it. Shot angle is the visual angle
between vectors from the shot location to both goalposts, not an angle to the goal centre.

## Detection and tracking

The domain depends on abstract `Detector` and `Tracker` contracts. Ultralytics and Roboflow
are isolated in adapters, so either implementation can be replaced without changing schemas
or downstream analytics. `YoloDetector.detect(frame, frame_id=...)` returns immutable
`Detection` records. `ByteTrackTracker.update(...)` assigns persistent IDs and uses the bottom
centre of the bounding box as the approximate point of contact with the pitch.

Use football-specific YOLO weights whose labels distinguish `player`, `goalkeeper`, `referee`,
and `ball`. Generic COCO weights cannot distinguish football roles reliably. For experiments,
`class_name_map` can map labels such as `person` and `sports ball`, while `allowed_classes`
filters unwanted detections.

```python
import cv2

from football_analytics.detection import YoloDetector, YoloDetectorConfig
from football_analytics.tracking import ByteTrackTracker

detector = YoloDetector(
    "models/football.pt",
    YoloDetectorConfig(allowed_classes=frozenset({"player", "goalkeeper", "referee", "ball"})),
)
tracker = ByteTrackTracker()

frame = cv2.imread("frame.jpg")
detections = detector.detect(frame, frame_id=0)
person_detections = [item for item in detections if item.class_name != "ball"]
tracks = tracker.update(person_detections, frame_id=0, frame=frame)
```

Call `tracker.reset()` before processing another video. Frames must be non-empty HxWx3 BGR
arrays with `uint8` values. Model weights are deliberately not committed or downloaded by the
test suite. ByteTrack should receive person-like classes; ball observations are handled by the
dedicated metric-space ball tracker. Unconfirmed internal ByteTrack observations are omitted
until the backend assigns a non-negative persistent ID.

Runtime dependencies are managed by `uv`: Pydantic, NumPy, Polars, scikit-learn, joblib,
OpenCV, Ultralytics, Supervision, and Roboflow Trackers. PyTorch is resolved transitively by
Ultralytics. pytest and Ruff remain development dependencies.

## Pitch keypoints and homography

The canonical model contains the 32 landmarks used by Roboflow's football-field pose model:
goal-line intersections, halfway-line intersections, centre-circle extrema, penalty-area and
goal-area intersections, and penalty spots. Their index order matches `SoccerPitchConfiguration`;
the target coordinates are expressed on this project's canonical 105 m x 68 m pitch.

`YoloPitchKeypointDetector` deliberately requires per-keypoint confidence. It never invents a
score when a model emits coordinates only. Low-confidence landmarks are removed before they
reach homography estimation.

```python
from football_analytics.pitch import PitchHomography, YoloPitchKeypointDetector

keypoint_detector = YoloPitchKeypointDetector("models/football_field.onnx")
homography = PitchHomography()

keypoints = keypoint_detector.detect(frame, frame_id=frame_id)
homography.estimate_keypoints(keypoints)

if homography.validate():
    pitch_x, pitch_y = homography.transform_point(player_bottom_centre)
```

`PitchHomography.estimate()` uses OpenCV RANSAC with at least four confident, unique,
non-collinear correspondences. Its confidence combines RANSAC inlier ratio, mean keypoint
confidence, and inlier reprojection RMSE in metres. Failed recalculation clears the previous
matrix; unreliable estimates cannot transform positions. This prevents stale coordinates after
camera cuts and suppresses spatial statistics when field evidence is weak.

## Metric positions and minimap

MetricPositionEstimator transforms the bottom-centre point of each person-like track through
the current reliable homography. It rejects off-pitch results and calculates velocity in metres
per second from consecutive trusted observations. First observations and gaps larger than the
configured threshold produce unknown velocities instead of artificial zero-speed samples.

Goalkeepers and referees inherit their semantic role from detection. Player team assignments can
be supplied as a track-ID-to-team mapping when a team classifier becomes available. Unclassified
players remain valid but appear gray on the minimap.

~~~python
from football_analytics.analytics import MetricPositionEstimator
from football_analytics.visualization import MinimapRenderer

position_estimator = MetricPositionEstimator(homography)
minimap_renderer = MinimapRenderer()

timestamp = frame_id / frames_per_second
positions = position_estimator.project_tracks(
    tracks,
    timestamp=timestamp,
    team_assignments=team_assignments,
)
ball_position = position_estimator.project_ball(ball_detection, timestamp=timestamp)
minimap = minimap_renderer.render(positions, ball_position)
annotated_frame = minimap_renderer.overlay(frame, minimap)
~~~

The renderer draws the canonical field, both teams, goalkeepers, referees, unassigned players,
and the ball. Each render represents exactly one video frame; mixed-frame observations are
rejected. The overlay scales to fit the source frame and leaves its input array unchanged.

## Ball tracking and possession

BallTracker associates projected ball candidates using predicted position and metric distance.
It smooths measurements, estimates ball velocity, bridges short detection gaps with
constant-velocity predictions, decays prediction confidence, and marks every predicted position
with observed=False. Strong detections can explicitly reacquire the trajectory after a sudden
change. Predictions outside the pitch or beyond the configured gap are discarded.

BallPossessionEstimator uses player-to-ball distance with separate acquisition and release
thresholds. Acquisition, player switches, and loss of possession require configurable repeated
evidence. This hysteresis prevents frame-by-frame flicker when players contest the ball.

~~~python
from football_analytics.ball import BallPossessionEstimator, BallTracker

ball_tracker = BallTracker()
possession_estimator = BallPossessionEstimator()

ball_candidates = position_estimator.project_balls(
    ball_detections,
    timestamp=timestamp,
)
tracked_ball = ball_tracker.update(
    ball_candidates,
    frame_id=frame_id,
    timestamp=timestamp,
)
possession = possession_estimator.update(
    positions,
    tracked_ball,
    frame_id=frame_id,
    timestamp=timestamp,
)
~~~

Possession summaries are duration-weighted rather than frame-counted. They expose seconds per
player, seconds for Team A and Team B, unknown seconds, and percentages over time attributed to
known teams. Goalkeeper possession remains in the unknown-team bucket until goalkeeper team
affiliation is available.

## Passes and shots

EventFrame guarantees that possession and ball evidence belong to the same frame and timestamp.
PassDetector opens a candidate when confirmed possession leaves a player. Reception by a
teammate completes the pass; reception by a known opponent or expiration makes it incomplete.
An unknown receiver team does not create an invented interception, and short possession flicker
below the configured distance or duration is ignored.

ShotDetector is independent from xG. It requires prior possession, ball acceleration, minimum
speed, separation from the controlled origin, and a velocity vector pointing toward the
attacking goal. Predicted ball locations cannot create shots by default, and a cooldown prevents
duplicate events.

~~~python
from football_analytics.events import EventFrame, PassDetector, ShotDetector

pass_detector = PassDetector()
shot_detector = ShotDetector()

event_frame = EventFrame(possession=possession, ball=tracked_ball)
passes = pass_detector.update(event_frame)
shots = shot_detector.update(event_frame)
~~~

Passes contain start/end coordinates, distance, duration, completion and confidence. Shots
contain distance to goal, the visual angle between both posts, ball speed and confidence.
Body part, pressure, set piece and outcome remain null until trustworthy detectors exist.

## Expected goals (xG)

The xG baseline is a scikit-learn logistic regression trained on historical shots, never on
invented probabilities. Its initial feature vector is ordered as `distance, angle`, where angle
is measured in radians. Historical files may be CSV, Parquet, JSON, or NDJSON and must contain:

```text
shot_x, shot_y, distance, angle, goal
```

Optional fields are `body_part`, `assist_type`, `set_piece`, `pressure`, and
`goalkeeper_position`. Unknown extra source columns are ignored. Every row is validated before
training, including pitch bounds, non-negative distance, angle between zero and pi, and a binary
goal outcome.

Train the model outside the video pipeline:

```bash
uv run football-analytics train-xg data/shots.parquet --output models/xg.joblib
```

Then load the fitted artifact and estimate the probability of the detected `ShotEvent`:

```python
from football_analytics.analytics import ExpectedGoalsModel, aggregate_expected_goals

model = ExpectedGoalsModel.load("models/xg.joblib")
predictions = model.predict_shots(shots)
totals = aggregate_expected_goals(predictions)
```

Each prediction is constrained to `0 <= xG <= 1`. Aggregation sums the same predictions by
player and by team and validates that both overall totals are equal. Detection remains separate:
the video pipeline creates shot features, while only the previously trained model estimates xG.

`joblib` uses pickle-compatible persistence and can execute code while loading. Only load model
artifacts produced by this project from a trusted source, and use the same locked dependency
environment used for training.

## Match statistics and visualizations

`MatchStatisticsCalculator` combines trusted metric positions, possession observations, passes,
shots, and xG predictions. It returns immutable, serializable `PlayerStatistics` and
`TeamStatistics` records:

```python
from football_analytics.analytics import MatchStatisticsCalculator

statistics = MatchStatisticsCalculator().calculate(
    positions=positions,
    possessions=possession_observations,
    passes=passes,
    shots=shots,
    xg_predictions=xg_predictions,
)
```

Player records include tracked minutes, distance, average and maximum speed, possession time,
passes and accuracy, shots, known outcomes, xG, average position, and progressive distance with
the ball. Team records include possession, passes, shot and xG totals, final-third and penalty-area
entries, progressive passes, field tilt, territorial possession, average width and length,
compactness, and defensive-line position.

Team A attacks toward increasing `x` by default and Team B toward decreasing `x`. Set
`StatisticsConfig(team_a_attacks_right=False)` when the orientation is reversed. Progressive
passes advance at least 10 m by default. Field tilt is the share of confirmed possession time in
each team's attacking final third; territorial possession is the corresponding share in the
opposition half. Shape metrics require at least two known teammates in a frame. PPDA remains
`None` because the current event model does not yet provide reliable defensive actions. Unknown
shot outcomes and incomplete xG coverage also remain `None`, never fabricated as zero.

Visualizations operate directly on canonical metric coordinates and return OpenCV BGR images:

```python
import cv2

from football_analytics.visualization import (
    AveragePositionRenderer,
    HeatmapRenderer,
    ShotMapRenderer,
)

player_heatmap = HeatmapRenderer().render_positions(positions, player_id=9)
cv2.imwrite("outputs/player_9_heatmap.png", player_heatmap)
cv2.imwrite("outputs/pass_heatmap.png", HeatmapRenderer().render_passes(passes))
cv2.imwrite("outputs/shot_map.png", ShotMapRenderer().render(shots, xg_predictions))
average_positions = AveragePositionRenderer().render(positions)
cv2.imwrite("outputs/average_positions.png", average_positions)
```

The same heatmap renderer supports shot locations and confirmed-possession locations through
`render_shots()` and `render_possession()`. Shot-map marker size represents xG when a matching
prediction exists, while known goals receive a distinct ring.

Export any Pydantic record sequence to Parquet, CSV, or JSON:

```python
from football_analytics.storage import MatchDataExporter

exporter = MatchDataExporter("data")
exporter.export("positions", positions)
exporter.export("passes", passes, formats=("parquet", "csv", "json"))
exporter.export("shots", shots)
exporter.export("player_stats", statistics.players)
exporter.export("team_stats", statistics.teams)
```

## End-to-end video analysis

The `analyze` command connects object detection, ByteTrack, jersey classification, pitch
keypoints, dynamic homography, metric positions, ball tracking, possession, passes, shots,
optional xG, statistics, exports, static maps, and an annotated video:

```bash
uv run football-analytics analyze match.mp4 \
  --object-model models/player_tracker.onnx \
  --pitch-model models/football_field.onnx \
  --xg-model models/xg.joblib \
  --output outputs/match
```

The default `models/player_tracker.onnx` is a 1280 x 1280 object detector; ByteTrack still assigns
the persistent tracking IDs. Its numeric Roboflow labels are translated automatically as
`0=ball`, `1=goalkeeper`, `2=player`, and `3=referee`. Models such as `models/best.pt` that already
use semantic labels remain compatible. `models/football_field.onnx` is the 32-keypoint YOLOv8 Pose
model used for homography. `onnxruntime` is part of the locked runtime.
The similarly named `models/football_field.pt` is not a supported Ultralytics checkpoint and
must not be passed to the CLI.

The xG artifact is optional. Without `--xg-model`, the rest of the pipeline runs and xG fields
for detected shots remain unknown. Since joblib is pickle-compatible, only load an xG artifact
you trust.

For stable team identities, provide representative RGB jersey colours:

```bash
uv run football-analytics analyze match.mp4 \
  --team-a-color 220,30,30 \
  --team-b-color 30,30,220
```

If colours are omitted, the pipeline groups jersey crops into two deterministic K-means
clusters; the resulting Team A/Team B names are arbitrary. The colour-anchor mode is preferable
when attacking direction and team-specific statistics matter.

Use a short smoke test before processing a full match:

```bash
uv run football-analytics analyze match.mp4 --max-frames 100
```

Useful options include `--device cpu`, `--device mps`, `--device 0`,
`--export-format parquet csv json`, `--no-annotated-video`, and `--verbose`. The default output
directory is `outputs/<video-name>/` and contains:

```text
annotated.mp4
report.json
data/
  detections.parquet
  tracks.parquet
  pitch_keypoints.parquet
  positions.parquet
  ball_positions.parquet
  possessions.parquet
  events.parquet
  passes.parquet
  shots.parquet
  xg_predictions.parquet
  player_stats.parquet
  team_stats.parquet
visualizations/
  pass_heatmap.png
  shot_heatmap.png
  possession_map.png
  shot_map.png
  average_positions.png
  players/player_<id>_heatmap.png
```

When a frame lacks a reliable homography, metric positions and ball state are suppressed for that
frame. The video pipeline uses a calibrated minimum confidence of 0.40 for the supplied
32-keypoint model while retaining the RANSAC inlier and reprojection-error gates. This prevents
stale camera geometry from contaminating match statistics without discarding most valid frames.
