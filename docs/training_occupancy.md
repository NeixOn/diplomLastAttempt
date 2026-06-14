# Occupancy Training

Этот pipeline обучает базовую модель:

```text
rendered image -> ResNet18 encoder -> occupancy decoder -> Marching Cubes -> mesh
```

Это MVP, а не уровень TripoSR. Он нужен, чтобы проверить полный путь от подготовленных рендеров до `.obj` mesh.

## 1. Установка

```bash
source venv/bin/activate
pip install -r requirements-train.txt
```

`rtree` нужен для `trimesh.contains(points)`.

## 2. Быстрый smoke test обучения

```bash
python scripts/train_occupancy.py \
  --dataset-root ./dataset/chair \
  --output-dir ./outputs/occupancy_smoke \
  --max-train-items 128 \
  --max-val-items 64 \
  --epochs 1 \
  --batch-size 4 \
  --points-per-item 512 \
  --workers 2
```

Если это прошло, можно запускать дольше.

## 3. Первый нормальный запуск

```bash
python scripts/train_occupancy.py \
  --dataset-root ./dataset/chair \
  --output-dir ./outputs/occupancy_chair \
  --epochs 20 \
  --batch-size 16 \
  --points-per-item 2048 \
  --workers 8
```

На слабой GPU уменьши:

```bash
--batch-size 8 --points-per-item 1024
```

## 4. Реконструкция mesh по одной картинке

Выбери любую картинку:

```bash
find ./dataset/chair/renders -name "*.png" | head
```

Построй mesh:

```bash
python scripts/reconstruct_occupancy_mesh.py \
  --checkpoint ./outputs/occupancy_chair/best.pt \
  --image ./dataset/chair/renders/<model_id>/view_000.png \
  --output ./outputs/reconstructions/sample.obj \
  --resolution 64
```

Для более детального mesh:

```bash
--resolution 128
```

Но `128` заметно тяжелее.
