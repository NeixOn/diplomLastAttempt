import argparse
import csv
import gc
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


MESH_EXTENSIONS = {".obj", ".glb", ".ply", ".stl", ".off"}


def ensure_torch():
    try:
        return importlib.import_module("torch")
    except Exception:
        return None


def cuda_stats_start():
    torch = ensure_torch()
    if torch is None or not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def cuda_stats_end():
    torch = ensure_torch()
    if torch is None or not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**3)


def find_newest_mesh(root):
    root = Path(root)
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in MESH_EXTENSIONS
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def mesh_stats(mesh_path):
    import trimesh

    loaded = trimesh.load(mesh_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geom
            for geom in loaded.geometry.values()
            if isinstance(geom, trimesh.Trimesh)
        ]
        mesh = trimesh.util.concatenate(meshes) if meshes else None
    else:
        mesh = loaded

    if mesh is None:
        return {"vertices": 0, "faces": 0, "file_size_mb": Path(mesh_path).stat().st_size / (1024**2)}

    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "file_size_mb": Path(mesh_path).stat().st_size / (1024**2),
    }


def normalize_mesh(mesh):
    mesh = mesh.copy()
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    scale = float((bounds[1] - bounds[0]).max())
    if scale <= 0:
        raise ValueError("invalid mesh scale")
    mesh.vertices = (mesh.vertices - center) / scale
    return mesh


def load_as_mesh(mesh_path):
    import trimesh

    loaded = trimesh.load(mesh_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geom
            for geom in loaded.geometry.values()
            if isinstance(geom, trimesh.Trimesh)
        ]
        if not meshes:
            raise ValueError(f"empty mesh scene: {mesh_path}")
        return trimesh.util.concatenate(meshes)
    return loaded


def chamfer_distance(pred_mesh_path, gt_mesh_path, samples, seed):
    import numpy as np
    from scipy.spatial import cKDTree

    pred = normalize_mesh(load_as_mesh(pred_mesh_path))
    gt = normalize_mesh(load_as_mesh(gt_mesh_path))

    pred_points, _ = pred.sample(samples, return_index=True)
    gt_points, _ = gt.sample(samples, return_index=True)

    rng = np.random.default_rng(seed)
    rng.shuffle(pred_points)
    rng.shuffle(gt_points)

    pred_tree = cKDTree(pred_points)
    gt_tree = cKDTree(gt_points)

    pred_to_gt = gt_tree.query(pred_points, k=1)[0]
    gt_to_pred = pred_tree.query(gt_points, k=1)[0]
    return float(pred_to_gt.mean() + gt_to_pred.mean())


def build_eval_set(args):
    dataset_root = Path(args.dataset_root)
    metadata_path = dataset_root / "render_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} not found. Run scripts/kaggle_shapenet_chair.py render first."
        )

    with open(metadata_path, "r", encoding="utf-8") as file:
        render_metadata = json.load(file)

    items = []
    for item in render_metadata:
        image_paths = item.get("image_paths") or []
        if not image_paths:
            continue
        mesh_path = item.get("mesh_path")
        if mesh_path and not Path(mesh_path).exists():
            mesh_path = None
        items.append(
            {
                "model_id": item["model_id"],
                "image_path": image_paths[args.view_index % len(image_paths)],
                "gt_mesh_path": mesh_path,
            }
        )

    items = items[: args.limit]
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(items, file, indent=2)

    print(f"Saved {len(items)} evaluation images to {output_path}")


def run_triposr(image_path, out_dir, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(args.triposr_repo) / "run.py"),
        str(image_path),
        "--output-dir",
        str(out_dir),
    ]
    if args.triposr_bake_texture:
        command.append("--bake-texture")
    if args.triposr_extra_args:
        command.extend(args.triposr_extra_args.split())

    cuda_stats_start()
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=args.triposr_repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.perf_counter() - start
    peak_vram = cuda_stats_end()

    mesh_path = find_newest_mesh(out_dir)
    return {
        "returncode": completed.returncode,
        "elapsed_sec": elapsed,
        "peak_vram_gb": peak_vram,
        "mesh_path": str(mesh_path) if mesh_path else None,
        "log_tail": completed.stdout[-2000:],
    }


def load_hunyuan_pipeline(args):
    sys.path.insert(0, str(Path(args.hunyuan_repo).resolve()))
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    kwargs = {}
    if args.hunyuan_subfolder:
        kwargs["subfolder"] = args.hunyuan_subfolder
    if args.hunyuan_dtype == "float16":
        torch = ensure_torch()
        if torch is not None:
            kwargs["torch_dtype"] = torch.float16

    return Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.hunyuan_model,
        **kwargs,
    )


def run_hunyuan(image_path, out_dir, args, pipeline):
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "mesh.glb"

    call_kwargs = {}
    if args.hunyuan_steps is not None:
        call_kwargs["num_inference_steps"] = args.hunyuan_steps
    if args.hunyuan_seed is not None:
        call_kwargs["seed"] = args.hunyuan_seed
    if args.hunyuan_extra_kwargs:
        call_kwargs.update(json.loads(args.hunyuan_extra_kwargs))

    cuda_stats_start()
    start = time.perf_counter()
    mesh = pipeline(image=str(image_path), **call_kwargs)[0]
    elapsed = time.perf_counter() - start
    peak_vram = cuda_stats_end()

    mesh.export(output_path)
    return {
        "returncode": 0,
        "elapsed_sec": elapsed,
        "peak_vram_gb": peak_vram,
        "mesh_path": str(output_path),
        "log_tail": "",
    }


def run_models(args):
    with open(args.eval_json, "r", encoding="utf-8") as file:
        eval_items = json.load(file)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    selected_models = set(args.models)
    hunyuan_pipeline = None
    if "hunyuan3d-mini" in selected_models:
        hunyuan_pipeline = load_hunyuan_pipeline(args)

    results = []
    for index, item in enumerate(eval_items):
        model_id = item["model_id"]
        image_path = Path(item["image_path"])
        gt_mesh_path = item.get("gt_mesh_path")
        print(f"[{index + 1}/{len(eval_items)}] {model_id}: {image_path}")

        for model_name in args.models:
            out_dir = output_root / model_id / model_name
            row = {
                "model_id": model_id,
                "input_image": str(image_path),
                "gt_mesh_path": gt_mesh_path,
                "method": model_name,
                "status": "ok",
                "error": "",
            }

            try:
                if model_name == "triposr":
                    run_result = run_triposr(image_path, out_dir, args)
                elif model_name == "hunyuan3d-mini":
                    run_result = run_hunyuan(image_path, out_dir, args, hunyuan_pipeline)
                else:
                    raise ValueError(f"unknown model: {model_name}")

                row.update(run_result)
                if run_result["returncode"] != 0:
                    row["status"] = "failed"

                mesh_path = run_result.get("mesh_path")
                if mesh_path and Path(mesh_path).exists():
                    row.update(mesh_stats(mesh_path))
                    if args.compute_chamfer and gt_mesh_path:
                        row["chamfer_norm"] = chamfer_distance(
                            mesh_path,
                            gt_mesh_path,
                            samples=args.chamfer_samples,
                            seed=args.seed,
                        )
                else:
                    row["status"] = "failed"
                    row["error"] = "mesh output not found"
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = repr(exc)

            results.append(row)
            print(
                f"  {model_name}: {row['status']}, "
                f"time={row.get('elapsed_sec')}, mesh={row.get('mesh_path')}"
            )

            if ensure_torch() is not None:
                torch = ensure_torch()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            gc.collect()

    save_results(results, output_root)


def save_results(results, output_root):
    json_path = output_root / "results.json"
    csv_path = output_root / "results.csv"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    fieldnames = sorted({key for row in results for key in row.keys()})
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare TripoSR and Hunyuan3D mini on identical Kaggle inputs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser_ = subparsers.add_parser("build-eval-set")
    build_parser_.add_argument("--dataset-root", default="/kaggle/working/dataset/chair")
    build_parser_.add_argument(
        "--output-json", default="/kaggle/working/model_compare/eval_images.json"
    )
    build_parser_.add_argument("--limit", type=int, default=10)
    build_parser_.add_argument("--view-index", type=int, default=0)
    build_parser_.set_defaults(func=build_eval_set)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--eval-json", default="/kaggle/working/model_compare/eval_images.json"
    )
    run_parser.add_argument("--output-dir", default="/kaggle/working/model_compare")
    run_parser.add_argument(
        "--models",
        nargs="+",
        default=["triposr", "hunyuan3d-mini"],
        choices=["triposr", "hunyuan3d-mini"],
    )
    run_parser.add_argument("--triposr-repo", default="/kaggle/working/TripoSR")
    run_parser.add_argument("--triposr-bake-texture", action="store_true")
    run_parser.add_argument("--triposr-extra-args", default="")
    run_parser.add_argument("--hunyuan-repo", default="/kaggle/working/Hunyuan3D-2")
    run_parser.add_argument("--hunyuan-model", default="tencent/Hunyuan3D-2mini")
    run_parser.add_argument("--hunyuan-subfolder", default="hunyuan3d-dit-v2-mini-turbo")
    run_parser.add_argument("--hunyuan-dtype", choices=["float16", "float32"], default="float16")
    run_parser.add_argument("--hunyuan-steps", type=int, default=None)
    run_parser.add_argument("--hunyuan-seed", type=int, default=42)
    run_parser.add_argument("--hunyuan-extra-kwargs", default="")
    run_parser.add_argument("--compute-chamfer", action="store_true")
    run_parser.add_argument("--chamfer-samples", type=int, default=10000)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.set_defaults(func=run_models)

    return parser


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
