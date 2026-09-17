"""Render-and-compare dual-instrument pose optimization (CMA-ES or Adam)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from utils.cma_optimizer import evaluate_population, optimize_frame_adam, optimize_frame_cmaes
from utils.losses import CHANNEL_NAMES, DEFAULT_WEIGHTS, pose_error_dict
from utils.parameterization import initialize_pose
from utils.pose_geometry import ROOT, InstrumentMesh, SemanticRenderer, instruments_json
from utils.pose_visualization import validation_images
from utils.sim_data import CONVENTION, DEFAULT_ALBEDO, PoseConverter, iter_frames, list_runs

ARM_NAMES = ("left", "right")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("/data/data1/shena/data/simulated_data/data30"))
    p.add_argument("--instrument", type=Path, default=Path("/data/data1/shena/data/simulated_instrument"))
    p.add_argument("--output", type=Path, default=None,
                   help="Output directory; default runs/<sample_run_name>")
    p.add_argument("--camera", choices=("left", "right"), default="left")
    p.add_argument("--baseline", type=float, default=.005)
    p.add_argument("--height", type=int, default=288)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--renderer", choices=("nvdiffrast", "torch"), default="nvdiffrast")
    p.add_argument("--supersample", type=int, choices=(1, 2, 4), default=1)
    p.add_argument("--faces-per-part", type=int, default=0,
                   help="0 keeps the original OBJ; a positive count speeds optimization")
    p.add_argument("--runs", nargs="*", default=(), help="Optional run directory names; default is every run_*")
    p.add_argument("--num-samples", "--max-runs", type=int, default=0, dest="max_runs",
                   help="Use only the first N run_* samples under data-root (e.g. 15); 0 keeps all")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--limit", type=int, default=0,
                   help="Frames per run; when >0, take this many evenly spaced frames across the video (e.g. 50); 0 keeps all")
    p.add_argument("--init", choices=("perturb-gt", "identity", "centroid", "gt"), default="perturb-gt",
                   help="perturb-gt: first video frame starts from noisy GT (good-init protocol)")
    p.add_argument("--init-rotation-deg", type=float, default=10.,
                   help="GT rotation perturbation on the first video frame")
    p.add_argument("--init-translation-mm", type=float, default=5.,
                   help="GT translation perturbation on the first video frame, millimetres")
    p.add_argument("--init-joints-deg", type=float, default=8.,
                   help="GT joint-angle perturbation on the first video frame")
    p.add_argument("--track", action="store_true",
                   help="After the first frame, initialize from the previous prediction instead of that frame's GT")
    p.add_argument("--init-depth", type=float, default=.06)
    p.add_argument("--init-x-split", type=float, default=.04)
    p.add_argument("--min-fov-pixels", type=int, default=80)
    p.add_argument("--optimize-anyway", action="store_true",
                   help="Optimize even if the initialized render is off-screen")
    p.add_argument("--optimizer", choices=("cmaes", "adam"), default="cmaes",
                   help="Pose optimizer: CMA-ES or differentiable Adam")
    p.add_argument("--maxiter", type=int, default=None,
                   help="Iterations; default 60 for CMA-ES, 300 for Adam")
    p.add_argument("--popsize", type=int, default=12, help="CMA-ES population size")
    p.add_argument("--early-stop-tol", type=float, default=0.001,
                   help="Stop when |loss_t - loss_{t-1}| < this AND loss < --early-stop-loss; <=0 disables")
    p.add_argument("--early-stop-loss", type=float, default=0.1,
                   help="Early stop also requires loss < this value; <=0 disables the loss gate")
    p.add_argument("--sigma", type=float, default=1.)
    p.add_argument("--rotation-std", type=float, default=.25,
                   help="Initial CMA std for axis-angle coordinates (radians)")
    p.add_argument("--translation-xy-std", type=float, default=.003,
                   help="Initial CMA std for wrist x/y (metres); default 3 mm")
    p.add_argument("--translation-z-std", type=float, default=.005,
                   help="Initial CMA std for wrist z (metres); default 5 mm")
    p.add_argument("--joint-std", type=float, default=.15,
                   help="Initial CMA std for joint angles (radians)")
    p.add_argument("--translation-xy-mm", type=float, default=20.,
                   help="Search half-range for wrist x/y relative to the frame init (mm)")
    p.add_argument("--translation-z-mm", type=float, default=30.,
                   help="Search half-range for wrist z relative to the frame init (mm)")
    p.add_argument("--lr-translation", type=float, default=.1,
                   help="Adam LR for local wrist translation in millimetres (Instrument-Splatting style)")
    p.add_argument("--lr-rotation", type=float, default=.001,
                   help="Adam LR for wrist quaternion (0.01 oscillates; use 0.001)")
    p.add_argument("--lr-joints", type=float, default=.01,
                   help="Adam LR for joints (default 0.01, matching Instrument-Splatting theta_lr)")
    p.add_argument("--lr-step", type=int, default=0,
                   help="Adam StepLR period; 0 disables (dice-driven LR is preferred)")
    p.add_argument("--lr-gamma", type=float, default=.5,
                   help="Adam StepLR decay factor")
    p.add_argument("--no-dice-lr", action="store_true",
                   help="Disable Dice-driven Adam LR multipliers")
    p.add_argument("--render-chunk", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--translation-min", type=float, nargs=3, default=(-.08, -.08, .025))
    p.add_argument("--translation-max", type=float, nargs=3, default=(.08, .08, .12))
    p.add_argument("--alpha-limit-deg", type=float, default=90.)
    p.add_argument("--joint-limit-deg", type=float, default=125.)
    p.add_argument("--mask-weight", type=float, default=DEFAULT_WEIGHTS["mask"])
    p.add_argument("--tips-weight", type=float, default=DEFAULT_WEIGHTS["tips"])
    p.add_argument("--tips-gap-weight", type=float, default=DEFAULT_WEIGHTS["tips_gap"])
    p.add_argument("--save-history", action="store_true")
    p.add_argument("--no-overlays", action="store_true")
    return p


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def resolve_output(args, runs):
    if args.output is not None:
        return Path(args.output)
    if not runs:
        raise ValueError("No runs selected")
    if len(runs) == 1:
        return ROOT / "runs" / runs[0].name
    # Full / multi-run batch: use the dataset folder name (e.g. data30).
    return ROOT / "runs" / Path(args.data_root).name


def save_overlay(path, frame, result):
    batch = {"rgb": frame.rgb[None], "mask": frame.mask[None],
             "tips": frame.tips[None], "tip_confidence": frame.tip_confidence[None]}
    cpu = {key: value.detach().cpu() for key, value in result.items() if torch.is_tensor(value)}
    if "rgb" not in cpu:
        # Mask-only renderer: synthesize a flat RGB from predicted semantics for the pair panel.
        from utils.pose_visualization import mask_palette
        palette = torch.as_tensor(mask_palette(cpu["mask"].shape[1]), dtype=cpu["mask"].dtype)
        cpu["rgb"] = torch.einsum("bchw,cd->bdhw", cpu["mask"].clamp(0, 1), palette).clamp(0, 1)
    panels = validation_images(batch, cpu)
    grid = panels["overlap_GT_left_prediction_right"]
    Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(path)
    if "RGB_target_left_render_right" in panels:
        rgb = panels["RGB_target_left_render_right"]
        Image.fromarray((rgb.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(
            path.with_name(path.stem + "_rgb.png"))


def save_history_plot(path, history, optimizer="cmaes"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    gens = [row["generation"] for row in history]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(gens, [row["loss"] for row in history], color="#1f77b4")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=.3)
    for name, color in (("left_mean", "#d62728"), ("right_mean", "#2ca02c"), ("mean", "#111111")):
        axes[1].plot(gens, [row["dice"].get(name) for row in history], label=name, color=color)
    axes[1].set_ylabel("dice")
    axes[1].set_xlabel("Adam iteration" if optimizer == "adam" else "CMA-ES generation")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def mean_or_none(values):
    finite = [v for v in values if v is not None and v == v]
    return None if not finite else float(sum(finite) / len(finite))


def summarize(rows):
    summary = {
        "frames": len(rows),
        "optimized": sum(1 for row in rows if row.get("optimized")),
        "skipped_out_of_fov": sum(1 for row in rows if row.get("skipped") == "out_of_fov"),
        "mean_time_s": mean_or_none([row.get("time_s") for row in rows if row.get("optimized")]),
    }
    for name in CHANNEL_NAMES + ("mean", "left_mean", "right_mean"):
        summary[f"dice_{name}"] = mean_or_none([row.get("dice", {}).get(name) for row in rows if row.get("optimized")])
    for arm in ARM_NAMES:
        for key in (f"{arm}_translation_error_mm", f"{arm}_rotation_error_deg", f"{arm}_joints_mae_deg"):
            summary[key] = mean_or_none([row.get("pose_error", {}).get(key) for row in rows if row.get("optimized")])
    return summary


def write_csv(path, rows):
    keys = ["run", "frame", "optimized", "skipped", "init", "time_s", "loss",
            "dice_mean", "dice_left_mean", "dice_right_mean",
            *[f"dice_{name}" for name in CHANNEL_NAMES],
            "left_translation_error_mm", "left_rotation_error_deg",
            "right_translation_error_mm", "right_rotation_error_deg"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            dice, err = row.get("dice") or {}, row.get("pose_error") or {}
            values = {
                "run": row.get("run"), "frame": row.get("frame"),
                "optimized": int(bool(row.get("optimized"))),
                "skipped": row.get("skipped") or "",
                "init": (row.get("init") or {}).get("init", ""),
                "time_s": row.get("time_s"), "loss": row.get("loss"),
                "dice_mean": dice.get("mean"), "dice_left_mean": dice.get("left_mean"),
                "dice_right_mean": dice.get("right_mean"),
                **{f"dice_{name}": dice.get(name) for name in CHANNEL_NAMES},
                "left_translation_error_mm": err.get("left_translation_error_mm"),
                "left_rotation_error_deg": err.get("left_rotation_error_deg"),
                "right_translation_error_mm": err.get("right_translation_error_mm"),
                "right_rotation_error_deg": err.get("right_rotation_error_deg"),
            }
            handle.write(",".join("" if values[k] is None else str(values[k]) for k in keys) + "\n")


def run_optimizer(args, renderer, frame, vector, weights, alpha_limit, jaw_limit):
    common = dict(
        renderer=renderer, frame=frame, x0=vector, weights=weights,
        translation_min=args.translation_min, translation_max=args.translation_max,
        alpha_limit=alpha_limit, jaw_limit=jaw_limit, maxiter=args.maxiter,
        translation_xy_m=args.translation_xy_mm / 1000.,
        translation_z_m=args.translation_z_mm / 1000.,
        early_stop_tol=args.early_stop_tol,
        early_stop_loss=args.early_stop_loss,
    )
    if args.optimizer == "adam":
        return optimize_frame_adam(
            **common,
            lr_translation=args.lr_translation,
            lr_rotation=args.lr_rotation,
            lr_joints=args.lr_joints,
            lr_step=args.lr_step if args.lr_step and args.lr_step > 0 else None,
            lr_gamma=args.lr_gamma,
            dice_lr=not args.no_dice_lr,
        )
    return optimize_frame_cmaes(
        **common,
        sigma=args.sigma, popsize=args.popsize, seed=args.seed + frame.index,
        rotation_std=args.rotation_std,
        translation_xy_std=args.translation_xy_std,
        translation_z_std=args.translation_z_std,
        joint_std=args.joint_std,
        chunk=args.render_chunk,
    )


def main(args):
    if args.maxiter is None:
        args.maxiter = 300 if args.optimizer == "adam" else 60
    if min(args.height, args.width) < 32 or args.height % 2 or args.width % 2:
        raise ValueError("height/width must be even and >= 32")
    if args.frame_stride < 1 or args.render_chunk < 1:
        raise ValueError("frame-stride and render-chunk must be positive")
    device = torch.device(args.device)
    if args.renderer == "nvdiffrast" and device.type != "cuda":
        raise ValueError("nvdiffrast requires --device cuda")
    size = (args.height, args.width)
    runs = list_runs(args.data_root)
    if args.runs:
        wanted = set(args.runs)
        runs = [run for run in runs if run.name in wanted]
        missing = wanted - {run.name for run in runs}
        if missing:
            raise ValueError(f"Unknown runs: {sorted(missing)}")
    if args.max_runs:
        runs = runs[:args.max_runs]
    output = resolve_output(args, runs)
    output.mkdir(parents=True, exist_ok=True)
    # Multi-run batch (e.g. full data30): write each sample under output/<run_name>/.
    per_run_dirs = len(runs) > 1

    print("Using flat meshes without material RGB (mask + tip losses only)", flush=True)
    albedo = DEFAULT_ALBEDO
    mesh = InstrumentMesh(args.instrument, args.faces_per_part, load_appearance=False,
                          convention=CONVENTION, arms=2, part_albedo=None).to(device)
    renderer = SemanticRenderer(mesh, size, args.renderer, supersample=args.supersample, render_rgb=False)
    converter = PoseConverter(CONVENTION, args.camera, args.baseline)
    weights = {"mask": args.mask_weight, "tips": args.tips_weight, "tips_gap": args.tips_gap_weight}
    alpha_limit, jaw_limit = np.deg2rad(args.alpha_limit_deg), np.deg2rad(args.joint_limit_deg)
    config = {**vars(args), "albedo": albedo, "weights": weights, "output": str(output),
              "per_run_dirs": per_run_dirs,
              "mesh_sha256": mesh.hashes, "renderer": renderer.configuration(),
              "geometry_convention": CONVENTION}
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    (output / "config.json").write_text(json.dumps(jsonable(config), indent=2), encoding="utf-8")
    print(f"Optimizer={args.optimizer}  output={output}"
          f"{'  (per-run subdirs)' if per_run_dirs else ''}", flush=True)

    rows = []
    started = time.perf_counter()
    batch_jsonl = None if per_run_dirs else (output / "frames.jsonl").open("w", encoding="utf-8")
    try:
        for run_index, run in enumerate(runs):
            print(f"[{run_index + 1}/{len(runs)}] {run.name}", flush=True)
            run_dir = output / run.name if per_run_dirs else output
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "overlays").mkdir(exist_ok=True)
            (run_dir / "history").mkdir(exist_ok=True)
            if per_run_dirs:
                (run_dir / "config.json").write_text(
                    json.dumps(jsonable({**config, "output": str(run_dir), "runs": [run.name]}), indent=2),
                    encoding="utf-8")
            run_rows = []
            jsonl = (run_dir / "frames.jsonl").open("w", encoding="utf-8") if per_run_dirs else batch_jsonl
            previous_vector = None
            first_frame = True
            try:
                for frame in iter_frames(run, converter, args.camera, size, args.instrument,
                                         frame_stride=args.frame_stride, limit=args.limit,
                                         start=args.start_frame):
                    synchronize(device)
                    frame_start = time.perf_counter()
                    use_previous = args.track and args.init == "perturb-gt" and not first_frame and previous_vector is not None
                    _, vector, init_info = initialize_pose(
                        args.init, frame, renderer, args.init_depth, args.init_x_split,
                        args.min_fov_pixels, alpha_limit, jaw_limit,
                        translation_min=args.translation_min, translation_max=args.translation_max,
                        rotation_degrees=args.init_rotation_deg, translation_mm=args.init_translation_mm,
                        joints_degrees=args.init_joints_deg,
                        seed=args.seed + run_index * 1_000_003 + frame.index,
                        previous_vector=previous_vector if use_previous else None,
                        first_frame=first_frame or not use_previous)
                    init_eval = evaluate_population([vector], renderer, frame, weights,
                                                    alpha_limit, jaw_limit, args.render_chunk)
                    record = {
                        "run": frame.run, "frame": frame.index, "init": init_info,
                        "init_loss": float(init_eval["loss"][0]),
                        "init_dice": init_eval["dice"][0],
                        "init_terms": {name: float(value[0]) for name, value in init_eval["terms"].items()},
                        "optimized": False,
                    }
                    overlay_stem = f"{frame.index:04d}" if per_run_dirs else f"{frame.run}_{frame.index:04d}"
                    if not args.no_overlays:
                        save_overlay(run_dir / "overlays" / f"{overlay_stem}_init.png",
                                     frame, init_eval["result"])
                    if not init_info["visible"] and not args.optimize_anyway:
                        record.update(skipped="out_of_fov", time_s=time.perf_counter() - frame_start)
                        run_rows.append(record)
                        rows.append(record)
                        jsonl.write(json.dumps(jsonable(record)) + "\n")
                        jsonl.flush()
                        print(f"  frame {frame.index:04d}: skip, init {init_info['init']} "
                              f"pixels {init_info['pixels']}", flush=True)
                        previous_vector = None
                        first_frame = False
                        continue
                    best, history, elapsed, stats = run_optimizer(
                        args, renderer, frame, vector, weights, alpha_limit, jaw_limit)
                    synchronize(device)
                    gt = {key: value.to(device) for key, value in frame.pose.items()}
                    errors = pose_error_dict(best["pose"], gt, mesh.convention.pivot, mesh.convention.shaft_offset)
                    if not args.no_overlays:
                        save_overlay(run_dir / "overlays" / f"{overlay_stem}.png", frame, best["result"])
                    if args.save_history:
                        (run_dir / "history" / f"{overlay_stem}.json").write_text(
                            json.dumps(jsonable(history), indent=2), encoding="utf-8")
                        save_history_plot(run_dir / "history" / f"{overlay_stem}.png", history, args.optimizer)
                    record.update(
                        optimized=True, skipped=None, time_s=time.perf_counter() - frame_start,
                        optimize_s=elapsed, optimizer=stats, loss=best["loss"], terms=best["terms"],
                        dice=best["dice"], pose_error=errors,
                        predicted=instruments_json(best["pose"]),
                        gt=instruments_json(gt),
                    )
                    run_rows.append(record)
                    rows.append(record)
                    jsonl.write(json.dumps(jsonable(record)) + "\n")
                    jsonl.flush()
                    dice = best["dice"]
                    print(
                        f"  frame {frame.index:04d}: {elapsed:.1f}s  loss {best['loss']:.3f}  "
                        f"dice L {dice.get('left_mean')} R {dice.get('right_mean')}  "
                        f"t {errors['left_translation_error_mm']:.1f}/{errors['right_translation_error_mm']:.1f} mm  "
                        f"R {errors['left_rotation_error_deg']:.1f}/{errors['right_rotation_error_deg']:.1f} deg  "
                        f"init {init_info['init']} vis {init_info['visible']}  opt {args.optimizer}",
                        flush=True,
                    )
                    previous_vector = best["x"]
                    first_frame = False
            finally:
                if per_run_dirs:
                    jsonl.close()
            if per_run_dirs:
                run_summary = summarize(run_rows)
                run_summary["optimizer"] = args.optimizer
                run_summary["output"] = str(run_dir)
                (run_dir / "summary.json").write_text(
                    json.dumps(jsonable(run_summary), indent=2), encoding="utf-8")
                write_csv(run_dir / "summary.csv", run_rows)
    finally:
        if batch_jsonl is not None:
            batch_jsonl.close()

    summary = summarize(rows)
    summary["elapsed_s"] = time.perf_counter() - started
    summary["optimizer"] = args.optimizer
    summary["output"] = str(output)
    summary["runs"] = [run.name for run in runs]
    (output / "summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    write_csv(output / "summary.csv", rows)
    print("Summary: " + json.dumps(jsonable(summary), indent=2), flush=True)
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
