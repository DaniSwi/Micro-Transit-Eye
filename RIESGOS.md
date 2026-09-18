# Riesgos identificados

La rúbrica pide "riesgos concretos (no genéricos)". Decir "puede haber overfitting"
no cuenta. Cada riesgo de abajo tiene qué lo causa, cómo lo detectamos y qué hacemos.

---

## 1. Atajo por fuente de datos

**El riesgo.** ShanghaiTech Part B es casi todo clase `alta`; COCO filtrado es casi
todo `baja`. El modelo puede aprender a reconocer el *estilo de cada dataset*
(resolución, ángulo de cámara, compresión) en vez de la densidad real de personas.
Sacaría métricas excelentes y sería inservible en un paradero de Valparaíso.

**Cómo lo detectamos.** `evaluate.py` genera `per_source.csv`. Si el macro-F1 por
fuente es muy dispar, o si una fuente sola predice casi siempre la misma clase,
hay atajo.

**Mitigación.** Muestreo balanceado por fuente dentro de cada clase; reportar métricas
por fuente además de la global; inspección visual con Grad-CAM (ResNet) y attention
rollout (ViT) para ver si el modelo mira a las personas o al fondo.

---

## 2. Umbrales arbitrarios (3 y 10)

**El riesgo.** Una imagen con 3 personas es `baja` y una con 4 es `media`. Las
imágenes fronterizas son ruido de etiqueta puro, y penalizan al modelo por algo
que no puede aprender.

**Cómo lo detectamos.** `evaluate.py` reporta qué porcentaje de los errores son
entre clases adyacentes. Si es alto, el modelo está bien y el problema es la
discretización.

**Mitigación.** Análisis de sensibilidad moviendo los umbrales (2/8, 3/10, 5/12) y
ver cómo cambia el macro-F1; reportar la métrica de error adyacente vs error de
dos saltos; en el informe final, considerar regresión ordinal.

---

## 3. Brecha de dominio Valparaíso

**El riesgo.** Ningún dataset es chileno. Los micros tienen colores y formas
distintas, las calles de Valparaíso son estrechas y en pendiente, hay oclusiones
fuertes, y buena parte del uso real sería con poca luz. El modelo puede funcionar
bien en test y fallar completamente en la calle.

**Cómo lo detectamos.** El conjunto `local_valpo` (60–100 fotos nuestras, nunca
usadas en entrenamiento) mide exactamente esa caída.

**Mitigación.** Reportar la caída sin maquillarla — es el resultado más honesto e
interesante de la presentación; augmentación de color, desenfoque y baja luz;
en el informe final, fine-tuning con un puñado de imágenes locales para medir
cuántas hacen falta para cerrar la brecha.

---

## 4. Desbalance de clases

**El riesgo.** La clase `alta` será minoritaria. Un modelo que nunca predice `alta`
puede tener accuracy alto y ser justo inútil para el caso de uso: detectar un
paradero congestionado es *exactamente* lo que queremos.

**Cómo lo detectamos.** Recall de la clase `alta` en la matriz de confusión.

**Mitigación.** macro-F1 como métrica principal en vez de accuracy; class weights
inversos a la frecuencia en la pérdida; reportar siempre recall por clase.

---

## 5. Privacidad

**El riesgo.** Son imágenes de personas en el espacio público. Un sistema de
cámaras en paraderos tiene implicancias reales, y presentarlo sin abordarlas es
una omisión que la comisión va a notar.

**Mitigación.** El sistema entrega solo conteos agregados, nunca identidades; no
hay reconocimiento facial ni re-identificación en ninguna etapa; difuminamos
rostros en las fotos locales que mostramos; declarar explícitamente que un
despliegue real requeriría revisión legal.

---

## 6. Cómputo y reproducibilidad

**El riesgo.** ViT-B/16 es pesado. Si alguien entrena en Colab con batch distinto
o menos épocas "porque no le cabía", la comparación con ResNet se rompe y no nos
damos cuenta hasta la presentación.

**Mitigación.** Toda la configuración vive en los YAML versionados; semilla fija
en `train.py`; el config usado se copia dentro de `outputs/<run>/`; revisión
cruzada entre quienes entrenan cada modelo; si hay que bajar el tamaño del modelo,
se baja **en ambos** y se declara en la lámina.
