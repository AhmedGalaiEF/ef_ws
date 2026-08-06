from __future__ import annotations

import numpy as np
import pytest

from CV.rgbd_segmentation_demo import compose_views, decode_depth_frame


def test_compose_views_requires_at_least_one_view() -> None:
    with pytest.raises(ValueError, match="At least one view"):
        compose_views([])


def test_decode_depth_frame_preserves_uint16_frames() -> None:
    frame = np.array([[1, 2], [3, 4]], dtype=np.uint16)

    decoded = decode_depth_frame(frame)

    assert decoded.dtype == np.uint16
    np.testing.assert_array_equal(decoded, frame)
