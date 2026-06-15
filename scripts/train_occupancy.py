import argparse
import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split_ids(dataset_root, split):
    split_path = Path(dataset_root) / "splits" / f"{split}.txt"
    with open(split_path, "r", encoding="utf-8") as file:
        return set(line.strip() for line in file if line.strip())


def load_metadata(dataset_root):
    with open(Path(dataset_root) / "metadata.json", "r", encoding="utf-8") as file:
        return json.load(file)


def load_render_metadata(dataset_root):
    with open(Path(dataset_root) / "render_metadata.json", "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_mesh(mesh):
    mesh = mesh.copy()
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    scale = float((bounds[1] - bounds[0]).max())
    if scale <= 0:
        raise ValueError("bad mesh scale")
    mesh.vertices = (mesh.vertices - center) / scale
    return mesh


@lru_cache(maxsize=512)
def load_mesh_cached(mesh_path):
    import trimesh

    loaded = trimesh.load(mesh_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geom
            for geom in loaded.geometry.values()
            if isinstance(geom, trimesh.Trimesh)
        ]
        if not meshes:
            raise ValueError(f"empty scene: {mesh_path}")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = loaded

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"empty mesh: {mesh_path}")

    return normalize_mesh(mesh)


def sample_occupancy_points(mesh, num_points, near_surface_ratio, bbox_size, surface_sigma):
    num_near = int(num_points * near_surface_ratio)
    num_uniform = num_points - num_near

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

    labels = mesh.contains(points).astype(np.float32)
    return points, labels


class ShapeNetRenderOccupancyDataset(Dataset):
    def __init__(
        self,
        dataset_root,
        split,
        image_size,
        points_per_item,
        near_surface_ratio,
        bbox_size,
        surface_sigma,
        occupancy_root=None,
        max_items=None,
    ):
        self.dataset_root = Path(dataset_root)
        self.points_per_item = points_per_item
        self.near_surface_ratio = near_surface_ratio
        self.bbox_size = bbox_size
        self.surface_sigma = surface_sigma
        self.occupancy_root = Path(occupancy_root) if occupancy_root else None

        split_ids = load_split_ids(self.dataset_root, split)
        metadata_by_id = {
            item["model_id"]: item
            for item in load_metadata(self.dataset_root)
            if item["model_id"] in split_ids
        }

        render_items = load_render_metadata(self.dataset_root)
        samples = []
        for item in render_items:
            model_id = item["model_id"]
            if model_id not in metadata_by_id:
                continue
            if self.occupancy_root is not None:
                occupancy_path = self.occupancy_root / f"{model_id}.npz"
                if not occupancy_path.exists():
                    continue
            mesh_path = metadata_by_id[model_id]["mesh_path"]
            for image_path in item["image_paths"]:
                samples.append(
                    {
                        "model_id": model_id,
                        "image_path": image_path,
                        "mesh_path": mesh_path,
                    }
                )

        if max_items is not None:
            samples = samples[:max_items]

        self.samples = samples
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.image_transform(image)

        if self.occupancy_root is not None:
            points, labels = self._sample_cached_occupancy(sample["model_id"])
        else:
            mesh = load_mesh_cached(sample["mesh_path"])
            points, labels = sample_occupancy_points(
                mesh=mesh,
                num_points=self.points_per_item,
                near_surface_ratio=self.near_surface_ratio,
                bbox_size=self.bbox_size,
                surface_sigma=self.surface_sigma,
            )

        return {
            "image": image,
            "points": torch.from_numpy(points),
            "labels": torch.from_numpy(labels),
            "model_id": sample["model_id"],
        }

    def _sample_cached_occupancy(self, model_id):
        npz_path = self.occupancy_root / f"{model_id}.npz"
        data = np.load(npz_path)
        points = data["points"]
        labels = data["labels"]

        if len(points) >= self.points_per_item:
            indices = np.random.choice(len(points), self.points_per_item, replace=False)
        else:
            indices = np.random.choice(len(points), self.points_per_item, replace=True)

        return points[indices].astype(np.float32), labels[indices].astype(np.float32)


class ImageEncoder(nn.Module):
    def __init__(self, latent_dim, pretrained=False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Linear(in_features, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, images):
        return self.proj(self.backbone(images))


class OccupancyDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers):
        super().__init__()
        layers = []
        in_dim = latent_dim + 3
        for layer_idx in range(num_layers):
            layers.append(nn.Linear(in_dim if layer_idx == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, latent, points):
        batch_size, num_points, _ = points.shape
        latent_expanded = latent[:, None, :].expand(batch_size, num_points, latent.shape[-1])
        x = torch.cat([points, latent_expanded], dim=-1)
        logits = self.net(x.reshape(batch_size * num_points, -1))
        return logits.reshape(batch_size, num_points)


class ImageToOccupancy(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers, pretrained_encoder=False):
        super().__init__()
        self.encoder = ImageEncoder(latent_dim, pretrained=pretrained_encoder)
        self.decoder = OccupancyDecoder(latent_dim, hidden_dim, num_layers)

    def forward(self, images, points):
        latent = self.encoder(images)
        return self.decoder(latent, points)


def move_batch_to_device(batch, device):
    return {
        "image": batch["image"].to(device, non_blocking=True),
        "points": batch["points"].to(device, non_blocking=True),
        "labels": batch["labels"].to(device, non_blocking=True),
    }


def compute_bce_loss(logits, labels, pos_weight_mode):
    if pos_weight_mode == "none":
        return F.binary_cross_entropy_with_logits(logits, labels)

    if pos_weight_mode == "auto":
        with torch.no_grad():
            positives = labels.sum()
            negatives = labels.numel() - positives
            pos_weight = negatives / positives.clamp_min(1.0)
            pos_weight = pos_weight.clamp(1.0, 20.0)
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

    pos_weight = torch.tensor(float(pos_weight_mode), device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


def run_epoch(model, loader, optimizer, device, train, pos_weight_mode):
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    iterator = tqdm(loader, leave=False)
    for raw_batch in iterator:
        batch = move_batch_to_device(raw_batch, device)

        with torch.set_grad_enabled(train):
            logits = model(batch["image"], batch["points"])
            loss = compute_bce_loss(logits, batch["labels"], pos_weight_mode)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (preds == batch["labels"]).sum().item()
            total_count += batch["labels"].numel()
            total_loss += loss.item() * batch["labels"].numel()

        iterator.set_postfix(loss=float(loss.item()))

    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
    }


def save_checkpoint(path, model, optimizer, epoch, args, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train image-conditioned occupancy model.")
    parser.add_argument("--dataset-root", default="./dataset/chair")
    parser.add_argument("--output-dir", default="./outputs/occupancy")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--points-per-item", type=int, default=2048)
    parser.add_argument("--near-surface-ratio", type=float, default=0.5)
    parser.add_argument("--bbox-size", type=float, default=0.6)
    parser.add_argument("--surface-sigma", type=float, default=0.03)
    parser.add_argument("--occupancy-root", default=None)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--decoder-layers", type=int, default=5)
    parser.add_argument("--pretrained-encoder", action="store_true")
    parser.add_argument(
        "--pos-weight",
        default="none",
        help="Use 'none', 'auto', or a numeric positive-class BCE weight.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--max-val-items", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = ShapeNetRenderOccupancyDataset(
        dataset_root=args.dataset_root,
        split="train",
        image_size=args.image_size,
        points_per_item=args.points_per_item,
        near_surface_ratio=args.near_surface_ratio,
        bbox_size=args.bbox_size,
        surface_sigma=args.surface_sigma,
        occupancy_root=args.occupancy_root,
        max_items=args.max_train_items,
    )
    val_dataset = ShapeNetRenderOccupancyDataset(
        dataset_root=args.dataset_root,
        split="val",
        image_size=args.image_size,
        points_per_item=args.points_per_item,
        near_surface_ratio=args.near_surface_ratio,
        bbox_size=args.bbox_size,
        surface_sigma=args.surface_sigma,
        occupancy_root=args.occupancy_root,
        max_items=args.max_val_items,
    )

    print(f"train samples: {len(train_dataset)}")
    print(f"val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = ImageToOccupancy(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.decoder_layers,
        pretrained_encoder=args.pretrained_encoder,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            pos_weight_mode=args.pos_weight,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            train=False,
            pos_weight_mode=args.pos_weight,
        )

        metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(metrics)

        with open(output_dir / "history.json", "w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            args,
            metrics,
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                args,
                metrics,
            )


if __name__ == "__main__":
    main()
