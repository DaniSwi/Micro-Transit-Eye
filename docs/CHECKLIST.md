# Checklist — Presentación de Avance (5%)

Cada bloque corresponde a una fila de la rúbrica. Si un bloque queda sin marcar,
ese criterio baja.

---

## Problema y motivación
*Se evalúa: claridad del problema, relevancia local justificada*

- [ ] Una frase que define el problema sin jerga
- [ ] Quién usaría esto en concreto (municipio, operador de transporte, usuario)
- [ ] Al menos una foto real de un paradero del Gran Valparaíso en la lámina
- [ ] Tarea formalizada: entrada, salida, clases, umbrales
- [ ] Justificación de por qué clasificación y no detección/conteo en esta etapa

## Datos
*Se evalúa: fuente(s) definida(s), criterio de etiquetado, split train/val/test*

- [ ] Tabla de fuentes con número de imágenes por fuente
- [ ] Criterio de etiquetado escrito **textual** en la lámina (no explicado de palabra)
- [ ] Ejemplos visuales: una imagen de cada clase
- [ ] Tabla de splits con números absolutos por clase
- [ ] Mencionar: estratificación por clase y fuente, agrupación por escena, semilla fija
- [ ] Conjunto local de Valparaíso descrito como holdout puro (60–100 fotos)
- [ ] `python src/splits.py` corre sin fallar los chequeos de fuga

## Resultados preliminares ResNet
*Se evalúa: baseline entrenado, métricas iniciales presentadas*

- [ ] `outputs/resnet50/best.pt` existe
- [ ] Curvas de train/val loss
- [ ] macro-F1, accuracy, recall por clase
- [ ] Matriz de confusión
- [ ] Nº de parámetros y ms/imagen

## Resultados preliminares ViT
*Se evalúa: baseline entrenado, métricas iniciales presentadas*

- [ ] `outputs/vit_b16/best.pt` existe
- [ ] **Mismo formato de lámina que ResNet** (mismos ejes, mismas métricas, mismo orden)
- [ ] Lámina explícita de "qué mantuvimos igual entre ambos modelos"
- [ ] Baseline trivial incluido como piso de comparación
- [ ] Si ViT rinde peor: explicación preparada, no disculpa

## Riesgos identificados
*Se evalúa: riesgos concretos (no genéricos) y mitigación propuesta*

- [ ] Los 6 riesgos de `docs/RIESGOS.md`, cada uno con su mitigación
- [ ] Tabla `per_source.csv` mostrada como evidencia del riesgo de atajo por dataset
- [ ] Resultado del test local mostrado **sin maquillar**, aunque sea malo

## Claridad / Organización
- [ ] Una idea por lámina, sin párrafos largos
- [ ] Tipografía legible desde el fondo de la sala
- [ ] Todos los gráficos con ejes rotulados y unidades

## Preguntas
- [ ] Lámina de respaldo: *¿cómo saben que no aprendió el dataset?*
- [ ] Lámina de respaldo: *¿por qué esos umbrales (3 y 10)?*
- [ ] Lámina de respaldo: *¿qué da el baseline trivial?*
- [ ] Lámina de respaldo: *¿por qué ViT quedó abajo?*
- [ ] Todos los integrantes saben responder de los dos modelos, no solo del suyo

## Tiempo (15 min)
- [ ] Ensayo cronometrado #1
- [ ] Ensayo cronometrado #2
- [ ] Transiciones entre expositores practicadas

---

## Si el tiempo aprieta

Recortar **el tamaño del dataset**, nunca uno de los dos baselines.
Llegar sin ResNet o sin ViT cuesta dos criterios completos de ocho.

Plan mínimo viable: 3.000 imágenes, 6 épocas, ambos modelos, test local de 40 fotos.
Es suficiente para marcar todos los bloques de arriba.
