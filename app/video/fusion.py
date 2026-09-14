"""
app/video/fusion.py
─────────────────────────────────────────────────────────────────────────────
Dynamic (confidence-weighted) fusion of questionnaire and video scores, per
§19 of spec:

    R_hybrid = alpha * Rq + (1 - alpha) * Rv

This is kept SEPARATE from app.utils.combine_scores (the existing
fixed-weight q_weight=0.6/v_weight=0.4 placeholder), which is left in place
untouched as a fallback/reference implementation. app/main.py is switched to
call get_alpha()/hybrid_fusion() from this module instead.

──────────────────────────────────────────────────────────────────────────
IMPLEMENTATION DECISION — NOT A VALIDATED CLINICAL FORMULA
──────────────────────────────────────────────────────────────────────────
The paper motivating this project describes "dynamic weighting based on
video confidence" but (as provided to this implementation) does not specify
an exact alpha(confidence) formula. Rather than inventing one and presenting
it as scientifically established, this module implements ONE explicit,
documented, and swappable mapping:

    alpha(c) = ALPHA_FLOOR + (ALPHA_CEILING - ALPHA_FLOOR) * (1 - c)

where c = video_confidence in [0, 1]. Intuition: when the video model is
very confident (c -> 1), alpha shrinks toward ALPHA_FLOOR, giving video more
influence; when the video model is unconfident (c -> 0), alpha rises toward
ALPHA_CEILING, giving the questionnaire more influence. This is linear and
monotonic by construction, and both endpoints are configurable constants
below rather than hard-coded magic numbers.

This is an ENGINEERING CHOICE, presented as such. If you have (or later
obtain) empirical data on how video-confidence should trade off against
questionnaire reliability, replace get_alpha() with a better-justified
mapping — the rest of the pipeline only depends on this function's
signature (confidence: float) -> float, not its internals.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

# Bounds for alpha (weight on the questionnaire score). Configurable via env
# so this can be tuned without a code change while the exact formula is
# still under discussion/validation.
ALPHA_FLOOR = float(os.environ.get("VIDEO_FUSION_ALPHA_FLOOR", 0.4))
ALPHA_CEILING = float(os.environ.get("VIDEO_FUSION_ALPHA_CEILING", 0.9))

assert 0.0 <= ALPHA_FLOOR <= ALPHA_CEILING <= 1.0, (
    "ALPHA_FLOOR must be <= ALPHA_CEILING and both must be in [0, 1]."
)


def get_alpha(video_confidence: Optional[float]) -> float:
    """
    Map video_confidence -> alpha (weight on questionnaire score Rq).

    video_confidence=None (video unavailable/insufficient data) -> alpha=1.0
    so R_hybrid collapses exactly to the questionnaire score, per §19.
    """
    if video_confidence is None:
        return 1.0
    c = float(np.clip(video_confidence, 0.0, 1.0))
    alpha = ALPHA_FLOOR + (ALPHA_CEILING - ALPHA_FLOOR) * (1.0 - c)
    return float(np.clip(alpha, 0.0, 1.0))


def hybrid_fusion(
    questionnaire_score: float,
    video_score: Optional[float],
    video_confidence: Optional[float] = None,
) -> dict:
    """
    Compute R_hybrid = alpha * Rq + (1 - alpha) * Rv.

    Returns a dict (not just a float) so callers/logs/API responses can see
    the alpha that was actually used, which matters for auditability of a
    screening tool.
    """
    rq = float(np.clip(questionnaire_score, 0.0, 1.0))

    if video_score is None:
        alpha = 1.0
        r_hybrid = rq
    else:
        rv = float(np.clip(video_score, 0.0, 1.0))
        alpha = get_alpha(video_confidence)
        r_hybrid = alpha * rq + (1.0 - alpha) * rv
        r_hybrid = float(np.clip(r_hybrid, 0.0, 1.0))

    return {
        "questionnaire_score": rq,
        "video_score": video_score,
        "video_confidence": video_confidence,
        "alpha": alpha,
        "hybrid_risk_score": r_hybrid,
    }
