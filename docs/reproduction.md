# Reproduction guide

## Generation protocol

Use the pinned preset and archive the generated JSONL records with your videos. Branches use paired prompts and initial noise. Wan text-to-video uses its standard negative prompt with UniPC flow shift 3. Wan image-to-video uses an empty negative prompt, UniPC flow shift 5 and the same conditioning image for both branches. CogVideoX-5B uses the checkpoint's DDIM scheduler with fixed guidance and eta 0. LTX-Video uses its flow scheduler, a conditioning frame rate of 25, decoder timestep 0.05 and decoder noise scale 0.025. HunyuanVideo-1.5 uses its flow scheduler and standard two-branch CFG at scale 6. Output frame rates affect playback.

The scheduler receives the original CFG update plus the residual only while the early window is active. When using the library directly, arrange all predictions as `[batch, channels, latent_frames, height, width]`. Call:

    from thaw import add_temporal_guidance

    if step < round(window * total_steps):
        guided = add_temporal_guidance(
            guided, conditional, baseline, strength=2.0, causal=True
        )

Set `causal=False` for the full temporal projection. Norms are per video in float32 and the update preserves the incoming CFG dtype. The term is zero when the residual norm is at most 0.001 times the full difference norm.

## Four-cost comparison

Install the analysis extra with `python -m pip install -e '.[analysis]'`. The command accepts **paired differences already measured against stock CFG**. Use one row per prompt and setting with these columns.

| Column | Meaning |
|---|---|
| `prompt_id` | Shared identifier for a prompt across settings |
| `method` | Stock CFG, a baseline setting, or the THAW candidate |
| `delta_g` | Gain in camera-compensated subject flow, px/frame |
| `delta_m` | Difference in motion VQA probability |
| `delta_q` | Difference in subject VQA probability |
| `abs_delta_s` | Absolute relative change in mean HSV saturation, written as a ratio |
| `delta_e` | Mean color distance using CIE76 |

Include a zero row for stock CFG on every prompt. A prompt must have a row for each requested setting. For repeated runs, first average each prompt's paired measurements across the runs in the comparison. Submit calibration and test prompts separately and pass the selected settings from calibration to the test-stage command.

    thaw-frontier metrics.csv \
      --stock stock --candidate thaw --baseline cfg-6 --baseline scheduled-cfg

The analysis maximizes mean baseline motion over convex mixtures of the listed settings under the four caps. It reports THAW's motion gap, its motion-VQA difference at matched flow, one-sided 90% lower bounds, and Holm-adjusted p-values from a paired prompt bootstrap. The default is 10,000 resamples. For test runs with fixture inputs, `--draws` sets a smaller resample count.

## Benchmark scoring and scope

[Generate paired clips](../README.md#quick-start) with the intended protocol before running external evaluators. Use [VBench](https://github.com/Vchitect/VBench) with its standard prompt lists and dimension-specific subsets. Dynamic degree and static-filtered flicker use distinct subsets. Follow the official aggregator when reporting Total. The paper's custom four-cost protocol instead uses camera-compensated motion, two VQA scorers and color metrics. Its statistics command consumes exported paired measurements.

The project provides the generation method, selected qualitative clips and the four-cost statistics code. Regenerating the paper's full tables requires their original cohorts, baseline settings, and evaluator outputs.
