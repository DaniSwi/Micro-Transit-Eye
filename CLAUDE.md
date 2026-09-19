# CLAUDE.md

Contexto del proyecto para Claude Code. Léelo antes de tocar cualquier archivo.

---

## Qué es esto

Proyecto universitario (Aprendizaje Automático II, PUCV). Compara **ResNet-50 vs
ViT-B/16** en una tarea de clasificación de ocupación de paraderos de micro del
Gran Valparaíso.

- **Entrada:** imagen RGB de escena urbana, 224×224
- **Salida:** 3 clases — `baja` (0–3 personas) / `media` (4–10) / `alta` (>10)
- **Métrica principal:** macro-F1 (las clases están desbalanceadas; accuracy engaña)
- **Entregable inmediato:** presentación de avance, evaluada con rúbrica

El objetivo académico **no** es maximizar una métrica. Es tener dos baselines
entrenados con un protocolo limpio y poder explicar las diferencias. Un resultado
modesto y bien explicado vale más que un número alto sin procedencia.

---

## Invariantes — no romper sin avisar al usuario

Estas reglas son la razón de ser del proyecto. Si una tarea parece requerir
romperlas, para y pregunta antes de seguir.

**1. ResNet y ViT pasan por exactamente el mismo código y los mismos datos.**
`dataset.py`, `train.py` y `evaluate.py` son compartidos. Los campos de
`configs/resnet50.yaml` y `configs/vit_b16.yaml` deben coincidir en todo salvo
los bloques `model` y `optim`. Si cambias épocas, batch size, semilla,
augmentaciones, tamaño de imagen, label smoothing o class weights, **cámbialo en
ambos YAML**. Un cambio unilateral invalida la comparación entera y le cuesta un
criterio completo de la rúbrica.

**2. El conjunto local de Valparaíso (`source == "local_valpo"`) nunca entra a
train ni a val.** Va siempre a `split == "local"`. Es el holdout que mide la
brecha de dominio. `splits.py` tiene un assert que lo verifica: no lo desactives.

**3. Los splits se reparten por `scene_id`, no por imagen.** Frames de la misma
escena en train y test es fuga de datos y las métricas salen infladas.
`splits.py` falla si detecta una escena en dos splits. Si falla, el bug está en
cómo se asignaron los `scene_id`, no en el assert.

**4. Semilla fija = 42.** En `splits.py`, en los YAML y en `set_seed()`. Sin esto
los integrantes generan repartos distintos y los números no son comparables.

**5. Los umbrales de clase (3 y 10) viven en `BINS`**, definido en `labels.py` y
replicado en `download_selective.py`. Si cambian, cambian en los dos archivos.

**6. No maquillar resultados.** Si ViT rinde peor que ResNet, o si el test local
cae fuerte respecto al test general, eso se reporta tal cual. Es el hallazgo más
interesante de la presentación, no un problema que ocultar.

---

## Estructura

```
src/download_selective.py  descarga filtrada (COCO: 18 GB -> cientos de MB)
src/labels.py              anotaciones -> data/index.csv con etiquetas
src/splits.py              agrega columna `split`, con chequeos de fuga
src/dataset.py             Dataset + transforms COMPARTIDOS por ambos modelos
src/train.py               loop único, parametrizado por YAML
src/evaluate.py            métricas, matriz de confusión, desglose por fuente
src/baselines.py           piso de comparación (clase mayoritaria)
configs/*.yaml             lo único que distingue a un modelo del otro
docs/CHECKLIST.md          tareas mapeadas a la rúbrica
docs/RIESGOS.md            los 6 riesgos concretos y su mitigación
docs/GUION_PRESENTACION.md 11 láminas con tiempos
docs/COLAB.md              celdas listas para Colab
```

Pipeline: `download_selective.py` → `labels.py` → `splits.py` → `baselines.py`
→ `train.py` (×2) → `evaluate.py` (×4: test y local, por modelo).

---

## Estado actual

El esqueleto compila pero **no se ha corrido contra datos reales todavía**.
Pendientes conocidos:

- Los loaders de `labels.py` asumen rutas y formatos estándar de cada dataset.
  Nadie los ha verificado contra los archivos descargados.
- `load_roboflow_bus()` está vacío a propósito: falta elegir el dataset concreto.
  Si el dataset no anota personas, no sirve para etiquetar ocupación — en ese
  caso se usa solo para discusión cualitativa, no para entrenar.
- `download_selective.py` no se ha probado con descargas reales. La primera
  corrida debe ser con `--cap-per-class 20` para verificar que las URLs responden.
- Faltan las fotos locales de Valparaíso (60–100, etiquetadas a mano).

---

## Convenciones

- Español en comentarios, docstrings y salidas por consola. Nombres de variables
  y funciones en inglés donde ya lo están; no renombrar por consistencia.
- Los comentarios explican **por qué**, no qué hace la línea. Los que marcan
  decisiones de diseño (por qué lr distinto en ViT, por qué class weights, por
  qué agrupar por escena) son deliberados: no los borres al refactorizar.
- Sin dependencias nuevas salvo que sean necesarias. El `requirements.txt` está
  pensado para correr en Colab sin fricción.
- Nada de `.pt`, imágenes ni `outputs/` a git. Un checkpoint de ViT-B/16 pasa los
  100 MB que GitHub rechaza.

---

## Errores probables y qué significan

**La loss de validación de ViT sube desde la primera época.** Learning rate muy
alto para el backbone. Está en `1e-5`–`3e-5` a propósito; con el `3e-4` de ResNet
el modelo preentrenado se destruye en dos épocas.

**Accuracy alto pero recall de `alta` cerca de cero.** El modelo aprendió a
predecir la clase mayoritaria. Verificar que `class_weights: true` esté activo y
reportar macro-F1, no accuracy.

**macro-F1 muy dispar entre fuentes en `per_source.csv`.** El modelo puede estar
aprendiendo el estilo del dataset en vez de la densidad de personas. Es el riesgo
1 de `docs/RIESGOS.md`. No es un bug que arreglar en silencio: es un hallazgo que
va a la presentación.

**Casi todos los errores son entre clases adyacentes.** El problema son los
umbrales, no el modelo. `evaluate.py` ya reporta ese porcentaje.

**Out of memory en Colab con ViT-B/16.** Bajar a `vit_small_patch16_224` — y
entonces bajar ResNet a `resnet18` también, y declararlo en la presentación.
Cambiar solo uno rompe el invariante 1.

---

## Qué NO hacer

- No convertir esto en detección de objetos. La clasificación existe justamente
  para que ambos backbones sean comparables cambiando una sola cosa.
- No agregar trucos de rendimiento (ensambles, TTA, backbones más grandes) antes
  de que ambos baselines estén corriendo y evaluados. No es lo que evalúa la rúbrica.
- No escribir arquitecturas desde cero. Se usa `timm` con pesos preentrenados;
  entrenar un ViT desde pesos aleatorios con este dataset no daría nada útil.
- No borrar los asserts de `splits.py`.
