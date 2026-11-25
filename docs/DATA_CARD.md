# Data card — Micro-Transit Eye

Dataset unificado para clasificar la ocupación de paraderos de micro en tres clases.
Versión usada en los resultados finales: corrida del 23-nov-2026 (`seed=42`).

## 1. Tarea y etiquetas

- **Entrada:** imagen RGB de escena urbana, redimensionada a 224×224 para los modelos.
- **Salida:** 3 clases según el número de personas en la imagen (`BINS` en `src/labels.py`):

| Clase | Personas |
|---|---|
| `baja` | 0–3 |
| `media` | 4–10 |
| `alta` | más de 10 |

- **Criterio de etiquetado:** la etiqueta se **deriva** del número de personas ya anotadas
  en cada dataset público (cajas o puntos). No se reetiquetó a mano, salvo el set local
  de Valparaíso, que se contó a mano.

## 2. Fuentes

| Fuente | Origen | Licencia / términos de uso | Cómo se obtiene el conteo | Imágenes usadas |
|---|---|---|---|---|
| **COCO 2017** | Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014. cocodataset.org | Anotaciones: CC BY 4.0. Imágenes: términos de Flickr, con licencia Creative Commons propia de cada imagen | Cajas `person` | 4.058 |
| **CrowdHuman** | Shao et al., *CrowdHuman: A Benchmark for Detecting Human in a Crowd*, arXiv:1805.00123, 2018. crowdhuman.org | Solo investigación no comercial (términos de Megvii) | Cajas `person` sin marca `ignore` | 1.293 |
| **ShanghaiTech Part B** | Zhang et al., *Single-Image Crowd Counting via Multi-Column CNN*, CVPR 2016. Copia en Kaggle | Uso académico, citando el paper. No publica una licencia formal | Puntos de cabeza (`.mat`) | 400 |
| **Valparaíso (propio)** | Fotos tomadas por el equipo G7 en paraderos del Gran Valparaíso | Propias. Uso académico del curso; no se publican (ver §6) | Conteo a mano (`data/local_valpo/labels.csv`) | 106 |

**Filtros aplicados**
- **COCO:** solo imágenes con al menos una persona **y** algún objeto de contexto vial
  (`bus`, `car`, `truck`, `traffic light`, `stop sign`, `bicycle`). Se **descartan** las
  imágenes con alguna anotación `iscrowd=1`, porque COCO anota esas multitudes con una
  sola caja y el conteo saldría subestimado.
- **COCO y CrowdHuman:** muestreo balanceado por clase con `seed=42`
  (`download_selective.py --cap-per-class`), para que ninguna fuente aporte solo una clase.
- **Todas:** se reescalan al descargar para ahorrar espacio (lado máximo fijo). El
  recorte a 224×224 se hace después, en `dataset.py`.

## 3. Splits

Reparto 70/15/15 **por escena** (`scene_id`), no por imagen, y estratificado por clase y
fuente (`src/splits.py`, `seed=42`). El set de Valparaíso va **siempre** a `local` y nunca
entra a train ni a val: es el holdout que mide la brecha de dominio. `splits.py` tiene
asserts que fallan si una escena queda en dos splits o si una foto local entra a train o val.

**Por fuente**

| Fuente | Train | Val | Test | Local |
|---|---|---|---|---|
| COCO 2017 | 2.841 | 609 | 608 | — |
| CrowdHuman | 904 | 193 | 196 | — |
| ShanghaiTech B | 280 | 60 | 60 | — |
| Valparaíso | — | — | — | 106 |
| **Total** | **4.025** | **862** | **864** | **106** |

**Por clase**

| Clase | Train | Val | Test | Local | Total |
|---|---|---|---|---|---|
| baja | 1.201 | 257 | 258 | 77 | 1.793 |
| media | 1.443 | 309 | 310 | 26 | 2.088 |
| alta | 1.381 | 296 | 296 | 3 | 1.976 |
| **Total** | **4.025** | **862** | **864** | **106** | **5.857** |

## 4. Sesgos conocidos

1. **Atajo por fuente.** Todas las imágenes de ShanghaiTech son `alta`. Un modelo puede
   aprender a reconocer el estilo de ese dataset en vez de contar personas. Evidencia:
   ShanghaiTech saca 100% en test (ver `per_source.csv` de cada evaluación).
2. **Brecha de dominio.** Las fuentes públicas no son paraderos chilenos: COCO son fotos
   de Flickr de todo el mundo, CrowdHuman son multitudes urbanas y ShanghaiTech son
   calles de Shanghái. Ambos modelos caen ≈ 0,2 de macro-F1 en Valparaíso y sobreestiman
   la ocupación.
3. **Set local desbalanceado y pequeño.** 77 `baja`, 26 `media` y solo 3 `alta`. El recall
   de `alta` en Valparaíso se calcula con 3 fotos y no es fiable. Por eso se reportan
   intervalos de confianza por bootstrap.
4. **Ruido de etiqueta por las anotaciones de origen.** El conteo hereda los errores de
   cada dataset: personas no anotadas, cajas de reflejos o carteles, cabezas ocluidas.
   Cerca de los umbrales (3/4 y 10/11) una sola persona mal anotada cambia la clase.
5. **Umbrales arbitrarios.** 3 y 10 son una definición operativa nuestra, no un estándar.
   ≈ 90% de los errores de ambos modelos son entre clases vecinas.
6. **Condiciones de captura del set local.** No se registró la hora, el clima ni el
   paradero de cada foto, así que no se puede saber qué condiciones cubre. Sí se sabe que
   la ocupación alta casi no aparece (3 fotos).

## 5. Uso previsto y no previsto

- **Previsto:** comparar ResNet-50 y ViT-B/16 en un curso universitario y estudiar la
  brecha de dominio hacia Valparaíso.
- **No previsto:** uso operativo, como decidir frecuencias reales de micros, ni uso
  comercial. La licencia de CrowdHuman además lo prohíbe.

## 6. Privacidad y distribución

Las imágenes **no** están en el repositorio (`.gitignore`). Las fotos de Valparaíso pueden
mostrar caras de personas que no dieron consentimiento: solo se comparten dentro del equipo
por Drive y no se publican. En la presentación y la demo se usan pocas y solo con fines
académicos.

## 7. Cómo regenerar

```bash
python src/download_selective.py coco --out data/raw/coco --cap-per-class 1500
python src/labels.py --raw-dir data/raw --out data/index.csv
python src/splits.py --index data/index.csv --seed 42
```

En Colab: secciones 2 y 3 de `notebooks/Micro_Transit_Eye_final.ipynb`.
