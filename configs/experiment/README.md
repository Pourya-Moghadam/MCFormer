# Experiment configuration catalog

The committed E01 configurations drive the implemented data, model, training, and evaluation
factories. E02--E05 are deterministic analyses of frozen E01 prediction artifacts and are driven
by `mcformer-analyze`; they have no training configuration. `../../REPRODUCIBILITY_SPEC.md` fixes
the behavior for the remaining controlled experiments. Every training configuration records its
experiment ID and LaTeX table/figure labels and overrides one experimental factor at a time.

Protocol variants are created with explicit files or a recorded override, for example:

```bash
mcformer-show-config \
  --config configs/experiment/e01_toyota_cs_mcformer.yaml \
  --set data.protocol=cv1
```

Never overwrite a frozen paper configuration after results have been released. Add a new
configuration and version the result instead.

The controlled E06--E12 variants are enumerated in `../sweep/e06_e12.json`; do not create ad-hoc
unnamed overrides. `mcformer-show-sweep` resolves that matrix and injects the correct experiment ID
and variant name into every run configuration. E09's TimeSformer and MViTv2 base configurations
are committed here because their baseline and MCIM runs share all non-head settings.
