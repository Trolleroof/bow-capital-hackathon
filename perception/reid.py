"""
Re-Identification gallery with passive pre-buffering.

Flow
----
Every frame, for every tracked object:
    gallery.sample(track_id, crop)      # rate-limited to ~1/s; no-op for unknowns

After tracker produces the current active set:
    gallery.prune(active_track_ids)     # drops pre-buffers for gone unconfirmed tracks

On operator confirm:
    gallery.confirm(track_id)           # promotes that track's pre-buffer to confirmed
                                        # gallery; wipes all other pre-buffers

When confirmed target is absent from tracker:
    gallery.match(track_id, crop)       # streak-gated cosine match against confirmed
                                        # gallery; returns original ID after N hits

On unconfirm:
    gallery.clear()                     # full reset

Why this works better than snapshot-on-confirm:
    By the time the operator presses 'c', the target may have been in frame for
    several seconds -- the pre-buffer already holds diverse angles.  The confirmed
    gallery starts rich rather than cold.
"""
from __future__ import annotations

from collections import defaultdict, deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T

import config

# ---------------------------------------------------------------------------
# Extractors -- OSNet preferred, MobileNetV3 fallback
# ---------------------------------------------------------------------------

class _OSNetExtractor:
    """
    OSNet-x0.25 via torchreid -- trained with metric learning on ReID datasets.
    Same-person similarity typically 0.80-0.95; different-person 0.20-0.45.
    """
    def __init__(self, device: str) -> None:
        import torchreid
        self._fe  = torchreid.utils.FeatureExtractor(
            model_name="osnet_x0_25",
            device=device,
        )
        self._dev = device
        print("[reid] OSNet-x0.25 loaded (metric-learning backbone)")

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        h, w = crop_bgr.shape[:2]
        if min(h, w) < config.REID_MIN_CROP_PX:
            return None
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            feat = self._fe([rgb])          # (1, 512)
        vec  = feat[0].cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        return (vec / norm) if norm > 1e-6 else None


_MOBILENET_H, _MOBILENET_W = 256, 128
_mobilenet_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((_MOBILENET_H, _MOBILENET_W)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class _MobileNetExtractor(nn.Module):
    """Fallback when torchreid is not installed. Weaker discrimination."""
    def __init__(self, device: str) -> None:
        super().__init__()
        backbone = tvm.mobilenet_v3_small(
            weights=tvm.MobileNet_V3_Small_Weights.DEFAULT
        )
        self.features = backbone.features
        self.pool     = backbone.avgpool
        self._dev = torch.device("cpu" if device == "cpu" else "cuda")
        self.to(self._dev)
        self.eval()
        print("[reid] WARNING: torchreid not found, using MobileNetV3 fallback -- run: pip install torchreid")

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        h, w = crop_bgr.shape[:2]
        if min(h, w) < config.REID_MIN_CROP_PX:
            return None
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        t = _mobilenet_transform(rgb).unsqueeze(0).to(self._dev)
        with torch.no_grad():
            feat = self.pool(self.features(t))
        vec  = feat.squeeze().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        return (vec / norm) if norm > 1e-6 else None


def _build_extractor(device: str) -> _OSNetExtractor | _MobileNetExtractor:
    try:
        return _OSNetExtractor(device)
    except (ImportError, Exception) as e:
        print(f"[reid] OSNet unavailable ({e}), falling back to MobileNetV3")
        return _MobileNetExtractor(device)


# ---------------------------------------------------------------------------
# Part-based embedding
# ---------------------------------------------------------------------------

# Minimum pixel height per strip for a valid part embedding
_PART_MIN_STRIP_H = 16


def _embed(extractor: _OSNetExtractor | _MobileNetExtractor, crop_bgr: np.ndarray) -> np.ndarray | None:
    """
    Returns a single L2-normalised embedding for crop_bgr.

    If the crop is tall enough (>= REID_PART_MIN_H), splits into three
    horizontal strips and returns a weighted average of their embeddings:
        head/shoulders  weight 0.20
        torso/loadout   weight 0.50  <- most discriminative for uniformed targets
        legs            weight 0.30

    Falls back to a single global embedding for small crops.
    The output is always 512-dim and L2-normalised so all existing similarity
    logic works unchanged.
    """
    h = crop_bgr.shape[0]

    if h < config.REID_PART_MIN_H:
        return extractor.extract(crop_bgr)

    t1, t2   = h // 3, 2 * h // 3
    strips   = [crop_bgr[:t1, :], crop_bgr[t1:t2, :], crop_bgr[t2:, :]]
    weights  = config.REID_PART_WEIGHTS
    combined = None

    for strip, w in zip(strips, weights):
        if strip.shape[0] < _PART_MIN_STRIP_H:
            continue
        emb = extractor.extract(strip)
        if emb is None:
            continue
        combined = (combined + w * emb) if combined is not None else (w * emb)

    if combined is None:
        # All parts failed -- try full crop as last resort
        return extractor.extract(crop_bgr)

    norm = np.linalg.norm(combined)
    return (combined / norm) if norm > 1e-6 else None


# ---------------------------------------------------------------------------

def _is_diverse(emb: np.ndarray, existing: deque[np.ndarray], threshold: float = 0.97) -> bool:
    if not existing:
        return True
    return max(float(np.dot(emb, e)) for e in existing) < threshold


class ReIDGallery:
    def __init__(self) -> None:
        self._extractor = _build_extractor(config.DEVICE)

        # Pre-buffers: track_id → rolling deque of embeddings (unconfirmed targets)
        self._pre: dict[int, deque[np.ndarray]] = {}
        # Per-track frame counter for rate-limiting (shared for pre + confirmed)
        self._ctr: dict[int, int] = defaultdict(int)

        # Confirmed gallery
        self._gallery: deque[np.ndarray] = deque(maxlen=config.REID_GALLERY_SIZE)
        self._origin_id: int | None = None        # track_id at time of confirmation
        self._current_id: int | None = None       # current track_id (may change after ReID)

        # Streak counters for match()
        self._streaks: dict[int, int] = defaultdict(int)

        print("[reid] ReID gallery ready (passive pre-buffering, part-based embedding)")

    # ------------------------------------------------------------------
    # Per-frame sampling -- call for every tracked object

    def sample(self, track_id: int, crop: np.ndarray) -> None:
        """
        Rate-limited embedding extraction for any visible track.
        - Confirmed target    → feeds into confirmed gallery
        - Unconfirmed track   → feeds into that track's pre-buffer
        Both are no-ops if less than REID_SAMPLE_INTERVAL frames have passed.
        """
        self._ctr[track_id] += 1
        if self._ctr[track_id] % config.REID_SAMPLE_INTERVAL != 0:
            return

        emb = _embed(self._extractor, crop)
        if emb is None:
            return

        if track_id == self._current_id:
            # Confirmed target -- add to main gallery if diverse enough
            if _is_diverse(emb, self._gallery):
                self._gallery.append(emb)
        else:
            # Unconfirmed -- add to that track's pre-buffer
            if track_id not in self._pre:
                self._pre[track_id] = deque(maxlen=config.REID_PREBUFFER_SIZE)
            buf = self._pre[track_id]
            if _is_diverse(emb, buf):
                buf.append(emb)

    # ------------------------------------------------------------------
    # Lifecycle events

    def prune(self, active_track_ids: set[int]) -> None:
        """
        Drop pre-buffers for tracks that have left the frame.
        Confirmed target's buffer is never touched here.
        """
        gone = set(self._pre.keys()) - active_track_ids
        for tid in gone:
            del self._pre[tid]
            self._ctr.pop(tid, None)

    def confirm(self, track_id: int) -> None:
        """
        Promote track_id's pre-buffer into the confirmed gallery.
        All other pre-buffers are dropped immediately.
        """
        promoted = list(self._pre.get(track_id, []))
        n = len(promoted)

        # Wipe everything
        self._pre.clear()
        self._ctr.clear()
        self._streaks.clear()
        self._gallery.clear()

        self._origin_id  = track_id
        self._current_id = track_id

        for emb in promoted:
            if _is_diverse(emb, self._gallery):
                self._gallery.append(emb)

        print(f"[reid] Confirmed target {track_id} -- gallery seeded with {n} pre-buffered embeddings → {len(self._gallery)} diverse kept")

    def reassign(self, new_track_id: int) -> None:
        """Called by main loop after a successful ReID match to follow the new track ID."""
        self._current_id = new_track_id
        self._streaks.clear()
        # Pre-buffer for the new ID (if any) is now obsolete -- it was "us"
        self._pre.pop(new_track_id, None)

    def clear(self) -> None:
        """Full reset on operator unconfirm."""
        self._pre.clear()
        self._ctr.clear()
        self._gallery.clear()
        self._streaks.clear()
        self._origin_id  = None
        self._current_id = None
        print("[reid] Gallery cleared")

    # ------------------------------------------------------------------
    # Matching -- only call when confirmed target is absent from tracker

    def match(self, track_id: int, crop: np.ndarray) -> int | None:
        """
        Returns the original confirmed track_id after REID_CONSECUTIVE consecutive
        frames above REID_THRESHOLD, otherwise None.
        """
        if not self._gallery or self._origin_id is None:
            return None
        emb = _embed(self._extractor, crop)
        if emb is None:
            self._streaks[track_id] = 0
            return None

        sim = max(float(np.dot(emb, g)) for g in self._gallery)
        if sim >= config.REID_THRESHOLD:
            self._streaks[track_id] += 1
            if self._streaks[track_id] >= config.REID_CONSECUTIVE:
                print(
                    f"[reid] Re-identified {self._origin_id} via track {track_id} "
                    f"(sim={sim:.3f}, streak={self._streaks[track_id]})"
                )
                return self._origin_id
        else:
            self._streaks[track_id] = 0

        return None

    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        return bool(self._gallery) and self._origin_id is not None

    def debug_info(self) -> str:
        pre_counts = {tid: len(buf) for tid, buf in self._pre.items()}
        return (
            f"gallery={len(self._gallery)}/{config.REID_GALLERY_SIZE} "
            f"pre={pre_counts} "
            f"confirmed={self._origin_id} current={self._current_id}"
        )
