# _template_model

Blank-slate skeleton for a new model. Copy this directory, rename every occurrence of `TEMPLATE` / `_template_` / `_template_model` to your model name, then delete this README.

See [`how_to_add_a_new_model/`](../../../how_to_add_a_new_model/) at the repo root for the full guide.

## Files

- `simulation.py` — trivial 1-state Ornstein-Uhlenbeck SDE that runs out of the box.
- `sim_plots.py` — 2-panel plot (latent + observations).
- `estimation.py` — commented stub; fill in when you want posterior inference.
- `_dynamics.py` — empty; use when `estimation.py` grows beyond ~200 lines.
- `TESTING.md` — section skeleton; fill in once your model works.
- `__init__.py` — exports `TEMPLATE_MODEL`.

## Smoke test

From `version_1/`:

```bash
python simulator/run_simulator.py \
    --model models._template_model.simulation.TEMPLATE_MODEL \
    --param-set A --seed 42
```

Writes `outputs/synthetic__template__A_<timestamp>/` with NPZ + two PNG plots.
