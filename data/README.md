# Example data

One sample from `data30` for smoke tests:

```text
data/simulated_data/data30/run_1789376009_gpu1_env0/
```

Instrument meshes are **not** included here; point `--instrument` at your local mesh directory (e.g. `/data/data1/shena/data/simulated_instrument`).

```bash
python optimize_pose.py \
  --data-root data/simulated_data/data30 \
  --runs run_1789376009_gpu1_env0 \
  --instrument /path/to/simulated_instrument \
  --limit 5
```
