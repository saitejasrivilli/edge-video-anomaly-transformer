"""Baseline mask-aware tracker.

State machine (see ``schemas.TrackState`` for field-level docs):

    unmatched candidate  -->  NEW
    NEW/ACTIVE/MISSED, matched this frame  -->  ACTIVE
    ACTIVE/MISSED, not matched, missed_frames <= max_missed_frames  -->  MISSED
    MISSED, missed_frames > max_missed_frames  -->  TERMINATED (dropped)

Each call to ``update()`` processes one frame's candidates against the
current live tracks (NEW/ACTIVE/MISSED) and returns one ``TrackedInstance``
per candidate that was matched or newly created this frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from evat.tracking.matching import MatchingMethod, compute_score_matrix, greedy_match
from evat.tracking.schemas import ObjectCandidate, Track, TrackedInstance, TrackState


@dataclass(slots=True)
class TrackerConfig:
    """All tunable tracking parameters — never hard-coded in ``Tracker``."""

    matching_method: MatchingMethod = "mask_iou"
    iou_threshold: float = 0.3
    max_missed_frames: int = 5
    min_track_length: int = 1

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(asdict(self), sort_keys=True), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrackerConfig:
        """Load config, ignoring unrelated top-level keys (e.g. ``evaluation``)."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        known_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known_fields})


class Tracker:
    """Maintains object identity across frames via IoU-based greedy matching."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._tracks: list[Track] = []
        self._next_track_id = 1

    @property
    def tracks(self) -> list[Track]:
        """All tracks (including MISSED) currently retained by the tracker."""
        return list(self._tracks)

    def update(self, frame_id: str, candidates: list[ObjectCandidate]) -> list[TrackedInstance]:
        """Process one frame's candidates. Returns matched/new TrackedInstances."""
        live_tracks = [t for t in self._tracks if t.state != TrackState.TERMINATED]

        scores = compute_score_matrix(live_tracks, candidates, self.config.matching_method)
        matches = greedy_match(scores, self.config.iou_threshold)

        matched_track_idxs = {t_idx for t_idx, _ in matches}
        matched_candidate_idxs = {c_idx for _, c_idx in matches}

        outputs: list[TrackedInstance] = []

        for track_idx, candidate_idx in matches:
            track = live_tracks[track_idx]
            candidate = candidates[candidate_idx]
            track.mask = candidate.mask
            track.bbox = candidate.bbox
            track.last_frame_id = frame_id
            track.frame_history.append(frame_id)
            track.hits += 1
            track.missed_frames = 0
            track.age += 1
            track.state = TrackState.ACTIVE
            outputs.append(
                TrackedInstance(
                    frame_id=frame_id,
                    track_id=track.track_id,
                    state=track.state,
                    mask=track.mask,
                    bbox=track.bbox,
                )
            )

        for track_idx, track in enumerate(live_tracks):
            if track_idx in matched_track_idxs:
                continue
            track.missed_frames += 1
            track.age += 1
            track.state = (
                TrackState.TERMINATED
                if track.missed_frames > self.config.max_missed_frames
                else TrackState.MISSED
            )

        for candidate_idx, candidate in enumerate(candidates):
            if candidate_idx in matched_candidate_idxs:
                continue
            track = Track(
                track_id=self._next_track_id,
                state=TrackState.NEW,
                mask=candidate.mask,
                bbox=candidate.bbox,
                last_frame_id=frame_id,
            )
            self._next_track_id += 1
            self._tracks.append(track)
            outputs.append(
                TrackedInstance(
                    frame_id=frame_id,
                    track_id=track.track_id,
                    state=track.state,
                    mask=track.mask,
                    bbox=track.bbox,
                )
            )

        return outputs

    def finalize(self) -> list[Track]:
        """Return all tracks meeting ``min_track_length`` (call after the last frame)."""
        return [t for t in self._tracks if t.hits >= self.config.min_track_length]
