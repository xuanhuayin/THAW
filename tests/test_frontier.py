import numpy as np
import pytest

from thaw.frontier import analyse, mixture_statistics, paired_matrix


def test_interpolated_baseline_and_matched_motion_vqa():
    data = np.array([
        [0, 0, 0, 0, 0],
        [1.2, 0.04, 0.0, 0.12, 0.0],
        [1.5, 0.05, 0.0, 0.04, 0.0],
    ])
    np.testing.assert_allclose(mixture_statistics(data), [0.9, 0.01])
    result = analyse(np.repeat(data[None], 4, axis=0), draws=40, seed=4)
    assert result["prompts"] == 4 and result["passes_both"]


def test_incomplete_pairs_are_rejected(tmp_path):
    file = tmp_path / "metrics.csv"
    file.write_text(
        "prompt_id,method,delta_g,delta_m,delta_q,abs_delta_s,delta_e\n"
        "a,stock,0,0,0,0,0\n"
        "a,thaw,2,0,0,0.05,1\n"
        "b,stock,0,0,0,0,0\n"
    )
    with pytest.raises(ValueError, match="missing methods"):
        paired_matrix(file, ["stock", "thaw"])
