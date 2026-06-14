import argparse
import json
import math
import os
import random
import shutil
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


CHAIR_SYNSET_ID = "03001627"


def read_hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def download_chair(args):
    from huggingface_hub import hf_hub_download

    token = args.hf_token or read_hf_token()
    if not token:
        raise RuntimeError(
            "HF token not found. Set Kaggle secret HF_TOKEN or pass --hf-token."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = hf_hub_download(
        repo_id="ShapeNet/ShapeNetCore",
        repo_type="dataset",
        filename=f"{CHAIR_SYNSET_ID}.zip",
        token=token,
        local_dir=str(out_dir),
    )

    print(zip_path)


def extract_chair(args):
    zip_path = Path(args.zip_path)
    extract_root = Path(args.extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        if args.max_models is None:
            archive.extractall(extract_root)
            print(f"Extracted all files to {extract_root}")
            return

        names = archive.namelist()
        model_ids = []
        for name in names:
            parts = Path(name).parts
            if len(parts) >= 2 and parts[0] == CHAIR_SYNSET_ID:
                model_id = parts[1]
                if model_id not in model_ids:
                    model_ids.append(model_id)

        selected = set(model_ids[: args.max_models])
        for name in names:
            parts = Path(name).parts
            if len(parts) >= 2 and parts[0] == CHAIR_SYNSET_ID and parts[1] in selected:
                archive.extract(name, extract_root)

        print(f"Extracted {len(selected)} models to {extract_root}")


def build_metadata(args):
    raw_root = Path(args.raw_root)
    dataset_root = Path(args.dataset_root)
    splits_root = dataset_root / "splits"
    splits_root.mkdir(parents=True, exist_ok=True)

    obj_files = sorted(raw_root.glob("*/models/model_normalized.obj"))
    if args.max_models is not None:
        obj_files = obj_files[: args.max_models]

    metadata = []
    failed = []

    for obj_path in obj_files:
        model_id = obj_path.parts[-3]
        try:
            if not obj_path.exists():
                failed.append({"model_id": model_id, "reason": "missing_obj"})
                continue
            if obj_path.stat().st_size == 0:
                failed.append({"model_id": model_id, "reason": "empty_obj"})
                continue

            metadata.append(
                {
                    "model_id": model_id,
                    "category_id": CHAIR_SYNSET_ID,
                    "category": "chair",
                    "mesh_path": str(obj_path),
                }
            )
        except Exception as exc:
            failed.append({"model_id": model_id, "reason": repr(exc)})

    rng = random.Random(args.seed)
    model_ids = [item["model_id"] for item in metadata]
    rng.shuffle(model_ids)

    n_total = len(model_ids)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)

    splits = {
        "train": model_ids[:n_train],
        "val": model_ids[n_train : n_train + n_val],
        "test": model_ids[n_train + n_val :],
    }

    split_by_id = {}
    for split_name, ids in splits.items():
        with open(splits_root / f"{split_name}.txt", "w", encoding="utf-8") as file:
            for model_id in ids:
                file.write(model_id + "\n")
                split_by_id[model_id] = split_name

    for item in metadata:
        item["split"] = split_by_id[item["model_id"]]

    with open(dataset_root / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    with open(dataset_root / "failed_metadata.json", "w", encoding="utf-8") as file:
        json.dump(failed, file, indent=2)

    print(f"valid models: {len(metadata)}")
    print(f"failed models: {len(failed)}")
    print(f"train: {len(splits['train'])}")
    print(f"val: {len(splits['val'])}")
    print(f"test: {len(splits['test'])}")
    print(f"saved: {dataset_root}")


def normalize_mesh_for_render(mesh):
    mesh = mesh.copy()
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    scale = float((bounds[1] - bounds[0]).max())
    if scale <= 0:
        raise ValueError("bad mesh scale")
    mesh.vertices = (mesh.vertices - center) / scale
    return mesh


def look_at(camera_position, target=None, up=None):
    import numpy as np

    if target is None:
        target = np.array([0.0, 0.0, 0.0])
    if up is None:
        up = np.array([0.0, 1.0, 0.0])

    camera_position = np.array(camera_position, dtype=np.float64)
    target = np.array(target, dtype=np.float64)

    forward = target - camera_position
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)

    right = right / right_norm
    true_up = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = camera_position
    return pose


def render_one_model(task):
    os.environ.setdefault("PYOPENGL_PLATFORM", task["opengl_platform"])

    import numpy as np
    import pyrender
    import trimesh
    from PIL import Image

    model_id = task["model_id"]
    mesh_path = task["mesh_path"]
    out_dir = Path(task["renders_root"]) / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    views = task["views"]
    image_size = task["image_size"]
    expected_paths = [out_dir / f"view_{idx:03d}.png" for idx in range(views)]

    if task["skip_existing"] and all(path.exists() for path in expected_paths):
        return {
            "model_id": model_id,
            "mesh_path": mesh_path,
            "image_paths": [str(path) for path in expected_paths],
            "cameras_path": str(out_dir / "cameras.json"),
            "status": "skipped",
        }

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

    mesh = normalize_mesh_for_render(mesh)

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.75, 0.75, 0.75, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.7,
    )
    render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)

    scene = pyrender.Scene(
        bg_color=[255, 255, 255, 0],
        ambient_light=[0.4, 0.4, 0.4],
    )
    scene.add(render_mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    renderer = pyrender.OffscreenRenderer(image_size, image_size)

    saved = []
    cameras = []
    radius = 1.8
    elevation = math.radians(task["elevation_degrees"])

    for view_id in range(views):
        azimuth = 2.0 * math.pi * view_id / views
        cam_pos = np.array(
            [
                radius * math.cos(elevation) * math.sin(azimuth),
                radius * math.sin(elevation),
                radius * math.cos(elevation) * math.cos(azimuth),
            ]
        )

        pose = look_at(cam_pos)
        cam_node = scene.add(camera, pose=pose)
        light_node = scene.add(light, pose=pose)

        color, _ = renderer.render(scene)

        scene.remove_node(cam_node)
        scene.remove_node(light_node)

        image_path = out_dir / f"view_{view_id:03d}.png"
        Image.fromarray(color).save(image_path)

        saved.append(str(image_path))
        cameras.append(
            {
                "view_id": int(view_id),
                "azimuth": float(azimuth),
                "elevation": float(elevation),
                "radius": float(radius),
                "camera_position": cam_pos.tolist(),
                "image_path": str(image_path),
            }
        )

    renderer.delete()

    with open(out_dir / "cameras.json", "w", encoding="utf-8") as file:
        json.dump(cameras, file, indent=2)

    return {
        "model_id": model_id,
        "mesh_path": mesh_path,
        "image_paths": saved,
        "cameras_path": str(out_dir / "cameras.json"),
        "status": "rendered",
    }


def render_dataset(args):
    os.environ.setdefault("PYOPENGL_PLATFORM", args.opengl_platform)

    from tqdm.auto import tqdm

    dataset_root = Path(args.dataset_root)
    renders_root = dataset_root / "renders"
    renders_root.mkdir(parents=True, exist_ok=True)

    with open(dataset_root / "metadata.json", "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if args.split != "all":
        metadata = [item for item in metadata if item.get("split") == args.split]
    if args.max_models is not None:
        metadata = metadata[: args.max_models]

    tasks = [
        {
            "model_id": item["model_id"],
            "mesh_path": item["mesh_path"],
            "renders_root": str(renders_root),
            "views": args.views,
            "image_size": args.image_size,
            "skip_existing": args.skip_existing,
            "opengl_platform": args.opengl_platform,
            "elevation_degrees": args.elevation_degrees,
        }
        for item in metadata
    ]

    render_metadata = []
    failed_render = []

    if args.workers <= 1:
        iterator = tqdm(tasks, total=len(tasks))
        for task in iterator:
            try:
                render_metadata.append(render_one_model(task))
            except Exception as exc:
                failed_render.append(
                    {"model_id": task["model_id"], "reason": repr(exc)}
                )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(render_one_model, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    render_metadata.append(future.result())
                except Exception as exc:
                    failed_render.append({"reason": repr(exc)})

    with open(dataset_root / "render_metadata.json", "w", encoding="utf-8") as file:
        json.dump(render_metadata, file, indent=2)

    with open(dataset_root / "failed_render.json", "w", encoding="utf-8") as file:
        json.dump(failed_render, file, indent=2)

    print(f"rendered/skipped models: {len(render_metadata)}")
    print(f"failed: {len(failed_render)}")
    print(f"images: {len(list(renders_root.glob('*/*.png')))}")


def clean(args):
    for raw_path in args.paths:
        path = Path(raw_path)
        if path.exists():
            print(f"Removing: {path}")
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    print("done")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Prepare ShapeNetCore chair data for Kaggle experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download-chair")
    download_parser.add_argument("--output-dir", default="/kaggle/working/ShapeNetCore")
    download_parser.add_argument("--hf-token", default=None)
    download_parser.set_defaults(func=download_chair)

    extract_parser = subparsers.add_parser("extract-chair")
    extract_parser.add_argument(
        "--zip-path", default=f"/kaggle/working/ShapeNetCore/{CHAIR_SYNSET_ID}.zip"
    )
    extract_parser.add_argument(
        "--extract-root", default="/kaggle/working/shapenet_chair"
    )
    extract_parser.add_argument("--max-models", type=int, default=None)
    extract_parser.set_defaults(func=extract_chair)

    metadata_parser = subparsers.add_parser("build-metadata")
    metadata_parser.add_argument(
        "--raw-root", default=f"/kaggle/working/shapenet_chair/{CHAIR_SYNSET_ID}"
    )
    metadata_parser.add_argument("--dataset-root", default="/kaggle/working/dataset/chair")
    metadata_parser.add_argument("--max-models", type=int, default=None)
    metadata_parser.add_argument("--seed", type=int, default=42)
    metadata_parser.add_argument("--train-ratio", type=float, default=0.8)
    metadata_parser.add_argument("--val-ratio", type=float, default=0.1)
    metadata_parser.set_defaults(func=build_metadata)

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--dataset-root", default="/kaggle/working/dataset/chair")
    render_parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    render_parser.add_argument("--max-models", type=int, default=20)
    render_parser.add_argument("--views", type=int, default=4)
    render_parser.add_argument("--image-size", type=int, default=224)
    render_parser.add_argument("--workers", type=int, default=1)
    render_parser.add_argument("--opengl-platform", default="egl")
    render_parser.add_argument("--elevation-degrees", type=float, default=20.0)
    render_parser.add_argument("--skip-existing", action="store_true")
    render_parser.set_defaults(func=render_dataset)

    clean_parser = subparsers.add_parser("clean")
    clean_parser.add_argument("paths", nargs="+")
    clean_parser.set_defaults(func=clean)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
