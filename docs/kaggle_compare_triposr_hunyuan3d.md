# Сравнение TripoSR и Hunyuan3D mini на Kaggle

Этот сценарий нужен для быстрого практического сравнения моделей на одинаковом наборе входных изображений. В качестве набора данных используется класс `chair` из ShapeNetCore. Из 3D-моделей рендерятся одинаковые входные изображения, после чего TripoSR и Hunyuan3D mini запускаются на одних и тех же файлах.

Такой эксперимент подходит для главы с практическим сравнением, потому что фиксируются:

1) одинаковые входные изображения;
2) одинаковое оборудование Kaggle;
3) время работы каждой модели;
4) пиковое использование видеопамяти;
5) параметры полученной mesh-модели;
6) приближенный Chamfer Distance при наличии ground truth mesh.

Важно: Chamfer Distance в этом скрипте является приближенной практической метрикой после нормализации масштаба и положения mesh-моделей. Это не полное воспроизведение протоколов из статей TripoSR или Hunyuan3D.

## 1. Установка базовых зависимостей

```python
!apt-get update -qq
!apt-get install -y -qq libosmesa6-dev freeglut3-dev
!pip install -q huggingface_hub trimesh pyrender pillow tqdm numpy scipy
```

## 2. Подготовка датасета ShapeNet chairs

Для скачивания ShapeNetCore нужен Hugging Face token с принятыми условиями доступа к датасету. В Kaggle добавь secret с именем:

```text
HF_TOKEN
```

Затем выполни:

```python
!python scripts/kaggle_shapenet_chair.py download-chair
!python scripts/kaggle_shapenet_chair.py extract-chair --max-models 50
!python scripts/kaggle_shapenet_chair.py build-metadata
!python scripts/kaggle_shapenet_chair.py render --max-models 20 --views 1 --image-size 512 --workers 1 --skip-existing
```

Для первого запуска лучше брать 5-10 объектов. После проверки можно увеличить `--max-models`.

## 3. Установка TripoSR

```python
%cd /kaggle/working
!git clone https://github.com/VAST-AI-Research/TripoSR.git
%cd /kaggle/working/TripoSR
!pip install -q -r requirements.txt
!pip install -q git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Если установка `tiny-cuda-nn` не проходит, можно сначала попробовать запустить без него. На некоторых окружениях Kaggle зависимости TripoSR меняются, поэтому при ошибке установки надо смотреть текст ошибки в notebook.

## 4. Установка Hunyuan3D mini

```python
%cd /kaggle/working
!git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git
%cd /kaggle/working/Hunyuan3D-2
!pip install -q -r requirements.txt
!pip install -q -e .
```

Для сравнения используется только shape generation без текстурирования. Это легче для Kaggle и честнее для сравнения с TripoSR как моделью реконструкции геометрии.

## 5. Создание одинакового списка входных изображений

```python
%cd /kaggle/working/diplomLastAttempt
!python scripts/kaggle_compare_3d_models.py build-eval-set --limit 10
```

Результат:

```text
/kaggle/working/model_compare/eval_images.json
```

## 6. Запуск сравнения

```python
!python scripts/kaggle_compare_3d_models.py run --compute-chamfer --chamfer-samples 10000
```

Для очень быстрого теста без Chamfer Distance:

```python
!python scripts/kaggle_compare_3d_models.py run
```

Если Hunyuan3D mini не помещается в память, можно сначала проверить только TripoSR:

```python
!python scripts/kaggle_compare_3d_models.py run --models triposr
```

Или уменьшить число объектов:

```python
!python scripts/kaggle_compare_3d_models.py build-eval-set --limit 3
!python scripts/kaggle_compare_3d_models.py run --compute-chamfer
```

## 7. Где лежат результаты

После запуска будут созданы файлы:

```text
/kaggle/working/model_compare/results.csv
/kaggle/working/model_compare/results.json
```

В `results.csv` можно брать значения для таблицы:

1) `method` — модель;
2) `elapsed_sec` — время обработки одного изображения;
3) `peak_vram_gb` — пиковое использование видеопамяти;
4) `vertices` — количество вершин в выходной mesh-модели;
5) `faces` — количество граней;
6) `file_size_mb` — размер результата;
7) `chamfer_norm` — приближенное расстояние до исходной ShapeNet mesh-модели.

## 8. Формулировка для отчета

В эксперименте использовался открытый набор ShapeNetCore, категория `chair`. Для каждой исходной 3D-модели был сформирован один RGB-рендер размером 512×512 пикселей. Полученные изображения подавались на вход моделям TripoSR и Hunyuan3D mini. Сравнение проводилось на одинаковом наборе изображений и одном вычислительном окружении Kaggle. Для каждой модели фиксировались время инференса, пиковое потребление видеопамяти, количество вершин и граней в выходной mesh-модели, а также приближенное значение Chamfer Distance после нормализации масштаба.
