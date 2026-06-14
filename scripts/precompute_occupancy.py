import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


def normalize_mesh(mesh):
    mesh = mesh.copy()
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    scale = float((bounds[1] - bounds[0]).max())
    if scale <= 0:
        raise ValueError("bad mesh scale")
    mesh.vertices = (mesh.vertices - center) / scale
    return mesh


def load_mesh(mesh_path):
    import trimesh

    loaded = trimesh.load(mesh_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geom
            for geom in loaded.geometry.values()
            if isinstance(geom, trimesh.Trimesh)
        ]
        if not meshes:
            raise ValueError("empty scene")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = loaded

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("empty mesh")

    return normalize_mesh(mesh)


def sample_occupancy(mesh, points_per_model, near_surface_ratio, bbox_size, surface_sigma):
    num_near = int(points_per_model * near_surface_ratio)
    num_uniform = points_per_model - num_near

    uniform = np.random.uniform(
        -bbox_size,
        bbox_size,
        size=(num_uniform, 3),
    ).astype(np.float32)

    if num_near > 0:
        surface_points, _ = mesh.sample(num_near, return_index=True)
        noise = np.random.normal(0.0, surface_sigma, size=surface_points.shape)
        near = (surface_points + noise).astype(np.float32)
        points = np.concatenate([uniform, near], axis=0)
    else:
        points = uniform

    labels = mesh.contains(points).astype(np.uint8)
    return points, labels


def process_one(task):
    np.random.seed(task["seed"])

    model_id = task["model_id"]
    out_path = Path(task["occupancy_root"]) / f"{model_id}.npz"
    if task["skip_existing"] and out_path.exists():
        return {"model_id": model_id, "status": "skipped", "path": str(out_path)}

    mesh = load_mesh(task["mesh_path"])
    points, labels = sample_occupancy(
        mesh=mesh,
        points_per_model=task["points_per_model"],
        near_surface_ratio=task["near_surface_ratio"],
        bbox_size=task["bbox_size"],
        surface_sigma=task["surface_sigma"],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if task["float16"]:
        points = points.astype(np.float16)

    np.savez_compressed(
        out_path,
        points=points,
        labels=labels,
        model_id=model_id,
    )
    return {"model_id": model_id, "status": "created", "path": str(out_path)}


def main():
    parser = argparse.ArgumentParser(description="Precompute occupancy samples per mesh.")
    parser.add_argument("--dataset-root", default="./dataset/chair")
    parser.add_argument("--occupancy-root", default="./dataset/chair/occupancy")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--points-per-model", type=int, default=20000)
    parser.add_argument("--near-surface-ratio", type=float, default=0.5)
    parser.add_argument("--bbox-size", type=float, default=0.6)
    parser.add_argument("--surface-sigma", type=float, default=0.03)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--task-batch-size",
        type=int,
        default=None,
        help="How many models to submit to one process pool at once. Smaller values survive bad meshes better.",
    )
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--float16", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    occupancy_root = Path(args.occupancy_root)
    occupancy_root.mkdir(parents=True, exist_ok=True)

    with open(dataset_root / "metadata.json", "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if args.split != "all":
        metadata = [item for item in metadata if item.get("split") == args.split]
    if args.max_models is not None:
        metadata = metadata[: args.max_models]

    tasks = []
    for idx, item in enumerate(metadata):
        tasks.append(
            {
                "model_id": item["model_id"],
                "mesh_path": item["mesh_path"],
                "occupancy_root": str(occupancy_root),
                "points_per_model": args.points_per_model,
                "near_surface_ratio": args.near_surface_ratio,
                "bbox_size": args.bbox_size,
                "surface_sigma": args.surface_sigma,
                "float16": args.float16,
                "skip_existing": args.skip_existing,
                "seed": args.seed + idx,
            }
        )

    results = []
    failures = []

    if args.workers <= 1:
        for task in tqdm(tasks):
            try:
                results.append(process_one(task))
            except Exception as exc:
                failures.append({"model_id": task["model_id"], "reason": repr(exc)})
    else:
        task_batch_size = args.task_batch_size or max(args.workers * 4, 1)
        progress = tqdm(total=len(tasks))

        for start in range(0, len(tasks), task_batch_size):
            task_batch = tasks[start : start + task_batch_size]

            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                future_to_task = {
                    executor.submit(process_one, task): task for task in task_batch
                }

                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        failures.append(
                            {"model_id": task["model_id"], "reason": repr(exc)}
                        )
                    finally:
                        progress.update(1)

        progress.close()

    with open(dataset_root / "occupancy_metadata.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    with open(dataset_root / "failed_occupancy.json", "w", encoding="utf-8") as file:
        json.dump(failures, file, indent=2)

    print(f"created/skipped: {len(results)}")
    print(f"failed: {len(failures)}")
    print(f"occupancy files: {len(list(occupancy_root.glob('*.npz')))}")
    if failures:
        print("first failures:")
        for failure in failures[:5]:
            print(json.dumps(failure, ensure_ascii=False))


if __name__ == "__main__":
    main()
