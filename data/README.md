# Example data

- Sample sequence: `simulated_data/data30/run_1789376009_gpu1_env0/`
- Instrument meshes: `simulated_instrument/` (`shaft.obj` / `wrist.obj` / grippers)

```bash
python optimize_pose.py \
  --data-root data/simulated_data/data30 \
  --instrument data/simulated_instrument \
  --runs run_1789376009_gpu1_env0 \
  --limit 5
```
