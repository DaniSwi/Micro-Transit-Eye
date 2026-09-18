# G7 — Micro-Transit Eye

**Aprendizaje Automático II · PUCV · Proyecto Aplicado**
Entregable inmediato: **Presentación de Avance (5%)**

---

## 1. Qué estamos construyendo

**Tarea:** clasificación de **nivel de ocupación de paraderos** a partir de una imagen de escena urbana.

| | |
|---|---|
| Entrada | Una imagen RGB de escena urbana con contexto vial (224×224) |
| Salida | `baja` / `media` / `alta` |
| Umbrales | baja = 0–3 personas · media = 4–10 · alta = >10 |
| Modelos | ResNet-50 (CNN) vs ViT-B/16 (Transformer) |
| Métrica principal | macro-F1 |

**Por qué esta tarea y no detección/conteo:** con clasificación podemos comparar ResNet y ViT
cambiando **solo el backbone**, con el mismo split, la misma pérdida y las mismas métricas.
Detección exigiría dos arquitecturas de cabeza distintas y no sería una comparación limpia.
Detección y conteo fino quedan como extensión para el informe final, no para el avance.

> Regla de oro del avance: **es mejor un resultado modesto pero real, comparable y explicado,
> que un número alto sin procedencia.** Cuatro de los ocho criterios de la rúbrica se ganan
> solo con tener los dos baselines entrenados y sus métricas en pantalla.

---

## 2. Estructura del repo

```
g7-micro-transit-eye/
├── README.md                  ← este archivo
├── requirements.txt
├── configs/
│   ├── resnet50.yaml          ← config del baseline CNN
│   └── vit_b16.yaml           ← config del baseline Transformer
├── src/
│   ├── download_selective.py  ← baja SOLO las imágenes que pasan el filtro
│   ├── labels.py              ← conteo de personas → etiqueta (baja/media/alta)
│   ├── splits.py              ← split 70/15/15 estratificado, semilla fija
│   ├── dataset.py             ← Dataset + transforms (IDÉNTICOS para ambos modelos)
│   ├── train.py               ← loop de entrenamiento (sirve para los dos)
│   ├── evaluate.py            ← métricas, matriz de confusión, desglose por fuente
│   └── baselines.py           ← baseline trivial (clase mayoritaria)
├── docs/
│   ├── COLAB.md               ← celdas para copiar y pegar en Colab
│   ├── CHECKLIST.md           ← lo que hay que tener listo, marcado uno por uno
│   ├── GUION_PRESENTACION.md  ← las 11 láminas con tiempos
│   └── RIESGOS.md             ← riesgos concretos + mitigación (criterio de la rúbrica)
├── data/                      ← NO se sube a git
│   ├── raw/                   ← datasets descargados tal cual
│   ├── index.csv              ← índice unificado que genera labels.py
│   └── local_valpo/           ← nuestras fotos de paraderos (solo test)
└── outputs/                   ← checkpoints, métricas, figuras
```

---

## 3. Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Si trabajan en Google Colab: ver `docs/COLAB.md`, tiene las celdas listas para copiar.
o cambiar el modelo a `vit_small_patch16_224` y `resnet18`.

---

## 4. Flujo de trabajo (en este orden)

```bash
# 1. Armar el dataset. NO bajen COCO completo (18 GB): este script lee las
#    anotaciones, filtra, y descarga solo lo que sirve, ya reescalado.
#    Una persona lo corre una vez y comparte el resultado por Drive.
python src/download_selective.py coco --out data/raw/coco --cap-per-class 1500

# 2. Construir el índice unificado con las etiquetas derivadas del conteo
python src/labels.py --raw-dir data/raw --out data/index.csv

# 3. Generar los splits (estratificados por clase Y por fuente, semilla fija)
python src/splits.py --index data/index.csv --seed 42

# 4. Baseline trivial (10 minutos, demuestra criterio)
python src/baselines.py --index data/index.csv

# 5. Entrenar los dos modelos con la MISMA receta
python src/train.py --config configs/resnet50.yaml
python src/train.py --config configs/vit_b16.yaml

# 6. Evaluar: test general + test local de Valparaíso
python src/evaluate.py --run outputs/resnet50 --split test
python src/evaluate.py --run outputs/vit_b16  --split test
python src/evaluate.py --run outputs/resnet50 --split local
python src/evaluate.py --run outputs/vit_b16  --split local
```

---

## 5. Datos

El truco que nos ahorra semanas: **no etiquetamos a mano**, derivamos la etiqueta de las
anotaciones que ya existen en cada dataset.

| Fuente | Qué usamos | Cómo se vuelve etiqueta |
|---|---|---|
| COCO / Open Images | Imágenes con cajas `person` **y** además `bus`, `car`, `traffic light` o `stop sign` | Contar cajas `person` → bin |
| CrowdHuman | Cajas visible-body | Contar cajas → bin (aporta clase `alta`) |
| ShanghaiTech Part B | Anotaciones de puntos, escenas de calle | Contar puntos → bin |
| Roboflow Universe (bus detection) | Imágenes con micro/bus presente | Aporta contexto de paradero |
| **`data/local_valpo/`** | **60–100 fotos nuestras del Gran Valparaíso** | **Etiquetadas a mano. SOLO TEST.** |

**Criterio de etiquetado (texto literal para la lámina de datos):**

> La etiqueta es el número de personas anotadas en la imagen, discretizado en tres bins
> (0–3 / 4–10 / >10). Se usan únicamente imágenes de exterior con contexto vial.

**Splits:** 70 / 15 / 15, estratificados por clase y por fuente, `seed=42`.
Cuando hay varios frames de la misma escena, la separación es **por escena**, no por frame,
para evitar fuga entre train y test.

**Conjunto local:** las fotos de Valparaíso **nunca** entran a entrenamiento ni a validación.
Son nuestro test de brecha de dominio y lo que justifica el criterio "relevancia local
justificada" de la rúbrica.

---

## 6. Receta de entrenamiento

Lo que se mantiene **idéntico** entre ambos modelos (esto va en una lámina propia):
mismo split, 224×224, normalización ImageNet, mismas augmentaciones
(RandomResizedCrop 0.7–1.0, flip horizontal, ColorJitter suave), mismas épocas,
misma pérdida (cross-entropy con class weights), misma métrica de early stopping.

Lo único que cambia:

| | ResNet-50 | ViT-B/16 |
|---|---|---|
| Pesos | ImageNet (torchvision) | `vit_base_patch16_224.augreg_in21k_ft_in1k` (timm) |
| LR backbone | 3e-4 | 1e-5 – 3e-5 |
| LR cabeza | 3e-4 | 1e-4 |
| Weight decay | 1e-4 | 0.05 |
| Warmup | — | 1–2 épocas |
| Label smoothing | 0.1 | 0.1 |

**Métricas a reportar:** macro-F1 (principal, las clases están desbalanceadas), accuracy,
recall por clase, matriz de confusión, curvas de entrenamiento, nº de parámetros y
tiempo de inferencia por imagen.

> **Ojo con la expectativa:** es muy probable que ViT rinda **igual o peor** que ResNet con
> pocos datos. Eso no es un fracaso, es *el hallazgo*. Sostenerlo con argumentos
> (sesgo inductivo de las convoluciones, hambre de datos del Transformer, sensibilidad
> al learning rate) da mejor nota que un número alto sin explicación.

---

## 7. División de tareas

Ajusten los nombres. Lo importante es que **nadie quede como único responsable de un baseline**:
si ese integrante falla, se pierden dos criterios completos de la rúbrica.

| Rol | Responsable | Entrega |
|---|---|---|
| Datos: descarga, `labels.py`, `splits.py` | | `data/index.csv` + tabla de conteos |
| Baseline ResNet | | `outputs/resnet50/` + curvas |
| Baseline ViT | | `outputs/vit_b16/` + curvas |
| Fotos locales Valparaíso + etiquetado | | `data/local_valpo/` + CSV |
| Evaluación, figuras, matrices de confusión | | `outputs/figures/` |
| Presentación y ensayo cronometrado | | slides + 3 láminas de respaldo |

Revisión cruzada obligatoria: quien entrena ResNet revisa el código de ViT y viceversa,
para asegurar que la receta es realmente la misma.

---

## 8. Antes de presentar

Ver `docs/CHECKLIST.md` y `docs/GUION_PRESENTACION.md`.
Ensayar cronometrado **al menos dos veces**: el criterio "Tiempo" (15 min) se pierde por
pasarse, no por quedarse corto.
