from pathlib import Path

import numpy as np

from app import NumpyIndex, _normalize_topk


def test_numpy_index_persists_and_returns_nearest_item(tmp_path: Path) -> None:
    index = NumpyIndex(2, str(tmp_path / "products.index"))
    index.add_with_ids(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
        np.array([11, 22], dtype="int64"),
    )

    scores, ids = index.search(np.array([[0.9, 0.1]], dtype="float32"), 1)

    assert ids.tolist() == [[11]]
    assert scores[0, 0] > 0.9

    restored = NumpyIndex(2, str(tmp_path / "products.index"))
    assert restored.ntotal() == 2


def test_topk_is_bounded() -> None:
    assert _normalize_topk("5") == 5

    for value in (0, 101, "not-a-number"):
        try:
            _normalize_topk(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid topk value: {value}")
