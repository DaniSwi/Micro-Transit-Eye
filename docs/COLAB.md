# Colab — celdas para copiar y pegar

Regla que no se rompe: **el código vive en GitHub, los datos viven en Drive.**
El repo se clona en cada sesión (pesa kilobytes). El dataset se arma **una sola vez**
y queda en Drive para todo el grupo.

---

## Celda 1 — Montar Drive y clonar el repo

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/USUARIO/g7-micro-transit-eye.git /content/repo
%cd /content/repo
!pip install -q -r requirements.txt

import os
DRIVE = '/content/drive/MyDrive/g7'
os.makedirs(f'{DRIVE}/data', exist_ok=True)
os.makedirs(f'{DRIVE}/outputs', exist_ok=True)
```

En sesiones posteriores, reemplazar el `git clone` por:

```python
%cd /content/repo
!git pull
```

---

## Celda 2 — Armar el dataset (SOLO UNA PERSONA, UNA VEZ)

Esto es lo que evita bajar 18 GB. El script lee las anotaciones (~250 MB),
filtra, y descarga únicamente las imágenes que sobreviven, ya reescaladas.

```python
!python src/download_selective.py coco \
    --out {DRIVE}/data/raw/coco \
    --cap-per-class 1500 \
    --max-side 640
```

Tarda entre 20 y 40 minutos según la red. Es **reanudable**: si Colab se
desconecta, vuelve a correr la misma celda y salta lo ya descargado.

Al terminar imprime cuántos MB quedaron en disco. Debería ser del orden de
cientos de MB, no de gigas. Si salió en gigas, bajen `--cap-per-class`.

### CrowdHuman

No permite descarga por imagen suelta: viene en ZIPs grandes. Pero si ya
tienen el ZIP en Drive, se puede extraer selectivamente sin descomprimirlo entero:

```python
!python src/download_selective.py crowdhuman-extract \
    --zip {DRIVE}/CrowdHuman_train01.zip \
    --odgt {DRIVE}/data/raw/crowdhuman/annotation_train.odgt \
    --out {DRIVE}/data/raw/crowdhuman \
    --cap-per-class 800
```

El `.odgt` de anotaciones es liviano y se baja aparte desde la página de CrowdHuman.

### ShanghaiTech

Pesa ~300 MB. No vale la pena filtrarlo: descárguenlo de Kaggle y descompriman
Part B directo en `{DRIVE}/data/raw/shanghaitech/`.

---

## Celda 3 — Comprimir el dataset final

Una vez armado, empaquétenlo. Leer miles de archivos sueltos desde Drive montado
es lentísimo; leerlos desde el disco local de Colab es rápido. El `.tar` es el puente.

```python
!cd {DRIVE}/data && tar -cf dataset_g7.tar raw/
```

Súbanlo a una **carpeta compartida** de Drive. Los demás integrantes ya no
descargan nada: solo montan y descomprimen.

---

## Celda 4 — Preparar la sesión de entrenamiento

Esta es la que corren todos, cada vez:

```python
!mkdir -p /content/data
!tar -xf {DRIVE}/data/dataset_g7.tar -C /content/data

# datos en disco local (rápido), resultados en Drive (persistentes)
!ln -sfn /content/data/raw /content/repo/data/raw
!ln -sfn {DRIVE}/outputs /content/repo/outputs
```

Descomprimir toma unos minutos. Leer esas mismas imágenes desde Drive durante
12 épocas tomaría horas. La diferencia es real.

---

## Celda 5 — Pipeline completo

```python
!python src/labels.py --raw-dir data/raw --out data/index.csv
!python src/splits.py --index data/index.csv --seed 42
!python src/baselines.py --index data/index.csv
```

Antes de lanzar el entrenamiento completo, **prueben que corre de punta a punta**
con algo mínimo. Editen temporalmente las épocas a 1 en el YAML:

```python
!python src/train.py --config configs/resnet50.yaml
```

Si eso termina sin errores, recién ahí suban las épocas y lancen en serio.

```python
!python src/train.py --config configs/resnet50.yaml
!python src/train.py --config configs/vit_b16.yaml

!python src/evaluate.py --run outputs/resnet50 --split test
!python src/evaluate.py --run outputs/vit_b16  --split test
!python src/evaluate.py --run outputs/resnet50 --split local
!python src/evaluate.py --run outputs/vit_b16  --split local
```

Como `outputs/` apunta a Drive, los checkpoints y las figuras sobreviven aunque
se caiga la sesión.

---

## Verificar la GPU antes de entrenar

```python
!nvidia-smi
```

Si sale T4 (16 GB), ViT-B/16 con batch 32 entra. Si sale algo más chico o no hay
GPU asignada, bajen a `vit_small_patch16_224` — y entonces bajen ResNet a `resnet18`
también, y decláren1o en la lámina. Cambiar solo uno rompe la comparación.

---

## Advertencias

**Desconexión por inactividad.** Colab gratuito corta la sesión. `train.py` guarda
checkpoint cada vez que mejora el macro-F1, así que no pierden todo — pero para el
entrenamiento completo conviene que lo corra quien tenga GPU propia.

**Versiones que cambian solas.** Colab actualiza `torch` y `timm` sin avisar y un
día el código deja de correr sin que nadie haya tocado nada. Si pasa, claven la
versión exacta que les funcionaba en `requirements.txt`.

**Nunca suban checkpoints a GitHub.** Un `.pt` de ViT-B/16 pesa más de 300 MB y
GitHub rechaza archivos sobre 100 MB. El `.gitignore` ya los excluye.

**Push desde Colab.** Se puede (con un token personal en los secretos de Colab),
pero es incómodo y genera conflictos. El flujo sano: editar en el computador,
push desde ahí, `git pull` en Colab antes de entrenar.
