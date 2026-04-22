# patches

This directory contains the canonical, source-of-truth patches that the `tts`
branch of this fork carries on top of `master`. Each patch is also reflected
as a normal commit on the branch so that consumers building from a `git`
checkout do not need to apply anything manually; the patches are retained here
for auditability and for re-application against future ggml syncs.

## Files

- `ggml-metal-chatterbox-ops.patch` — Metal backend additions used by the
  Chatterbox Turbo TTS pipeline:
  - new `GGML_OP_DIAG_MASK_INF` Metal op,
  - `ggml_pad` extended with front-padding (`lp0..lp3`) to emulate
    `ggml_pad_ext`,
  - `kernel_conv_transpose_1d` rewritten to use one threadgroup per output
    pixel with a 32-thread `simd_sum` reduction across input channels,
  - opt-in fused `MUL_MAT + ADD(bias) (+ ADD(residual))` path for the
    Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 mat-vec kernels (gated behind
    function constants so non-fused callers keep their existing behaviour).

  Equivalent to the single commit on `tts` whose subject begins with
  `feat(metal): chatterbox ops`.

## Re-applying after a ggml sync

```bash
git checkout tts
git fetch origin master
git rebase origin/master            # or merge, depending on workflow
# if conflicts inside src/ggml-metal/, drop our metal commit and reapply:
git apply patches/ggml-metal-chatterbox-ops.patch
git add -A && git commit
```
