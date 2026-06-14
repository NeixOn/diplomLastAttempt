# ShapeNet Chair на Kaggle

Этот файл нужен, чтобы в Kaggle не копировать большие куски кода из чата. После `git clone` репозитория запускай команды из notebook-ячейки через `!python`.

## 1. Установка зависимостей

```python
!apt-get update -qq
!apt-get install -y -qq libosmesa6-dev freeglut3-dev
!pip install -q huggingface_hub trimesh pyrender pillow tqdm numpy
```

Если `pyrender` не работает с `egl`, перезапусти kernel и попробуй `--opengl-platform osmesa` в команде рендера.

## 2. Hugging Face token

В Kaggle добавь secret:

```text
HF_TOKEN
```

У токена должен быть доступ к public gated repositories, а условия ShapeNetCore на Hugging Face должны быть приняты.

## 3. Скачать только стулья

```python
!python scripts/kaggle_shapenet_chair.py download-chair
```

Это скачает:

```text
/kaggle/working/ShapeNetCore/03001627.zip
```

## 4. Распаковать стулья

Полная распаковка:

```python
!python scripts/kaggle_shapenet_chair.py extract-chair
```

Для быстрой проверки можно распаковать только первые 200 моделей:

```python
!python scripts/kaggle_shapenet_chair.py extract-chair --max-models 200
```

## 5. Создать metadata и split

```python
!python scripts/kaggle_shapenet_chair.py build-metadata
```

Результат:

```text
/kaggle/working/dataset/chair/metadata.json
/kaggle/working/dataset/chair/splits/train.txt
/kaggle/working/dataset/chair/splits/val.txt
/kaggle/working/dataset/chair/splits/test.txt
```

## 6. Тестовый рендер

Сначала проверь 3 модели по 4 ракурса без параллельности:

```python
!python scripts/kaggle_shapenet_chair.py render --max-models 3 --views 4 --workers 1 --skip-existing
```

Если все работает, можно быстрее:

```python
!python scripts/kaggle_shapenet_chair.py render --max-models 500 --views 8 --image-size 128 --workers 2 --skip-existing
```

Финальнее качество:

```python
!python scripts/kaggle_shapenet_chair.py render --max-models 1000 --views 12 --image-size 224 --workers 2 --skip-existing
```

Если `workers 2` стабилен, можно попробовать `--workers 4`.

## 7. Показать по одной случайной картинке на модель

```python
from pathlib import Path
import random
from IPython.display import display
from PIL import Image

RENDERS_ROOT = Path("/kaggle/working/dataset/chair/renders")
random.seed(42)

model_dirs = sorted([p for p in RENDERS_ROOT.iterdir() if p.is_dir()])
random.shuffle(model_dirs)

print("models with renders:", len(model_dirs))

for model_dir in model_dirs[:20]:
    images = sorted(model_dir.glob("*.png"))
    if not images:
        continue

    img_path = random.choice(images)
    img = Image.open(img_path)
    img.thumbnail((224, 224))

    print("model_id:", model_dir.name)
    print("image:", img_path)
    display(img)
```

## 8. Очистка места

Удалить кэш Hugging Face и рабочие папки:

```python
!python scripts/kaggle_shapenet_chair.py clean /root/.cache/huggingface /kaggle/working/ShapeNetCore
```

Удалить подготовленный датасет:

```python
!python scripts/kaggle_shapenet_chair.py clean /kaggle/working/dataset/chair
```

## Рекомендуемый порядок

Для первого запуска:

```python
!python scripts/kaggle_shapenet_chair.py download-chair
!python scripts/kaggle_shapenet_chair.py extract-chair --max-models 200
!python scripts/kaggle_shapenet_chair.py build-metadata
!python scripts/kaggle_shapenet_chair.py render --max-models 3 --views 4 --workers 1 --skip-existing
```

Когда тестовый рендер выглядит нормально:

```python
!python scripts/kaggle_shapenet_chair.py render --max-models 500 --views 8 --image-size 128 --workers 2 --skip-existing
```
