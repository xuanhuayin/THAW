"""Four-cost CFG mixture statistics for paired, externally measured prompts."""
import argparse
import csv
import json
from pathlib import Path

COLUMNS = ("delta_g", "delta_m", "delta_q", "abs_delta_s", "delta_e")
MARGINS = (0.03, 0.03, 0.02, 1.0)
VQA_NI = -0.03


def paired_matrix(path: Path, methods: list[str]):
    """Read finite paired differences with method order [stock, ..., candidate]."""
    import numpy as np

    records = {}
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if not {"prompt_id", "method", *COLUMNS}.issubset(reader.fieldnames or ()):
            raise ValueError(f"Expected prompt_id, method and {', '.join(COLUMNS)}.")
        for line, row in enumerate(reader, 2):
            method = row["method"]
            if method not in methods:
                continue
            prompt = row["prompt_id"]
            if not prompt:
                raise ValueError(f"Missing prompt_id on line {line}.")
            if method in records.setdefault(prompt, {}):
                raise ValueError(f"Duplicate prompt and method on line {line}.")
            try:
                values = [float(row[column]) for column in COLUMNS]
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid measurements on line {line}.") from exc
            if not all(np.isfinite(values)) or min(values[3:]) < 0:
                raise ValueError(f"Expected finite measurements and nonnegative appearance costs on line {line}.")
            records[prompt][method] = values
    if not records:
        raise ValueError("The CSV contains no measurements for the requested methods.")
    for prompt, row in records.items():
        missing = set(methods) - row.keys()
        if missing:
            raise ValueError(f"Prompt {prompt!r} is missing methods: {', '.join(sorted(missing))}.")
        if any(abs(value) > 1e-12 for value in row[methods[0]]):
            raise ValueError(f"Stock differences must be zero on prompt {prompt!r}.")
    return np.array([[row[method] for method in methods] for row in records.values()])


def mixture_statistics(means):
    """Return the admissible motion gap and matched-motion VQA difference."""
    import numpy as np
    from scipy.optimize import linprog

    g, motion_vqa, subject_vqa, saturation, color = means.T
    costs = np.vstack([-motion_vqa, -subject_vqa, saturation, color])
    caps = np.maximum(costs[:, -1] + np.asarray(MARGINS), costs[:, 0])
    ones = np.ones((1, len(g) - 1))

    frontier = linprog(
        -g[:-1], A_ub=costs[:, :-1], b_ub=caps, A_eq=ones, b_eq=[1.0],
        bounds=(0, None), method="highs",
    )
    if not frontier.success:
        raise RuntimeError(f"The admissible reference LP failed: {frontier.message}")

    reference_max = g[:-1].max()
    if g[-1] > reference_max:
        vqa_ref = motion_vqa[:-1][g[:-1] == reference_max].max()
    else:
        matched = linprog(
            -motion_vqa[:-1], A_ub=-g[:-1][None, :], b_ub=[-g[-1]], A_eq=ones,
            b_eq=[1.0], bounds=(0, None), method="highs",
        )
        if not matched.success:
            raise RuntimeError(f"The matched-motion VQA LP failed: {matched.message}")
        vqa_ref = -matched.fun
    return np.array([g[-1] + frontier.fun, motion_vqa[-1] - vqa_ref])


def analyse(data, *, draws: int, seed: int):
    import numpy as np

    if draws < 1:
        raise ValueError("At least one bootstrap draw is required.")
    rng = np.random.default_rng(seed)
    estimates = mixture_statistics(data.mean(axis=0))
    boot = np.empty((draws, 2))
    for i in range(draws):
        index = rng.integers(0, data.shape[0], data.shape[0])
        boot[i] = mixture_statistics(data[index].mean(axis=0))
    null = np.array([0.0, VQA_NI])
    p = ((boot <= null).sum(axis=0) + 1) / (draws + 1)
    lower = np.percentile(boot, 10, axis=0)
    order = np.argsort(p)
    adjusted = np.empty(2)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(p[order] * [2, 1]))
    return {
        "prompts": data.shape[0],
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "motion_gap": dict(estimate=estimates[0], lower_bound_90=lower[0], p=p[0], p_holm=adjusted[0]),
        "motion_vqa_difference": dict(
            estimate=estimates[1], lower_bound_90=lower[1], p=p[1], p_holm=adjusted[1], margin=VQA_NI,
        ),
        "passes_both": bool((adjusted <= 0.10).all() and (lower > null).all()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--stock", default="stock")
    parser.add_argument("--candidate", default="thaw")
    parser.add_argument("--baseline", action="append", default=[], help="Non-stock baseline. Repeat for every setting.")
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args(argv)
    methods = [args.stock, *args.baseline, args.candidate]
    if len(set(methods)) != len(methods):
        parser.error("Stock, candidate and baseline names must be distinct.")
    try:
        result = analyse(paired_matrix(args.metrics, methods), draws=args.draws, seed=args.seed)
    except ImportError:
        parser.error("Install the analysis dependencies with pip install '.[analysis]'.")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"methods": methods, **result}, indent=2))


if __name__ == "__main__":
    main()
