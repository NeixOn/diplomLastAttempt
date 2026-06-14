import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.measure import marching_cubes
from torchvision import transforms
from tqdm.auto import tqdm

from train_occupancy import ImageToOccupancy


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint["args"]

    model = ImageToOccupancy(
        latent_dim=saved_args["latent_dim"],
        hidden_dim=saved_args["hidden_dim"],
        num_layers=saved_args["decoder_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, saved_args


def load_image(image_path, image_size, device):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


def make_grid(resolution, bbox_size):
    lin = np.linspace(-bbox_size, bbox_size, resolution, dtype=np.float32)
    grid = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3), lin


def write_obj(path, vertices, faces):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for vertex in vertices:
            file.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face in faces:
            i, j, k = face + 1
            file.write(f"f {i} {j} {k}\n")


def reconstruct(args):
    device = torch.device(args.device)
    model, saved_args = load_model(args.checkpoint, device)
    image = load_image(args.image, saved_args["image_size"], device)

    points, lin = make_grid(args.resolution, args.bbox_size)
    probs = []

    with torch.no_grad():
        for start in tqdm(range(0, len(points), args.chunk_size)):
            chunk = points[start : start + args.chunk_size]
            chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).to(device)
            logits = model(image, chunk_tensor)
            prob = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            probs.append(prob)

    volume = np.concatenate(probs, axis=0).reshape(
        args.resolution,
        args.resolution,
        args.resolution,
    )

    spacing = (
        float(lin[1] - lin[0]),
        float(lin[1] - lin[0]),
        float(lin[1] - lin[0]),
    )
    vertices, faces, _, _ = marching_cubes(volume, level=args.level, spacing=spacing)
    vertices = vertices + np.array([-args.bbox_size, -args.bbox_size, -args.bbox_size])

    write_obj(args.output, vertices, faces)
    print(f"saved mesh: {args.output}")
    print(f"vertices: {len(vertices)}")
    print(f"faces: {len(faces)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct mesh from one rendered image.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="./outputs/reconstruction.obj")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--bbox-size", type=float, default=0.6)
    parser.add_argument("--level", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    reconstruct(parse_args())
