# Guion de presentación — 15 minutos

11 láminas. Los tiempos suman ~15:15, así que hay que ir ajustado.
Asignar un expositor por bloque y practicar las transiciones.

| # | Lámina | Tiempo | Qué tiene que quedar claro |
|---|---|---|---|
| 1 | Título y equipo | 0:15 | Nombre del proyecto, integrantes |
| 2 | Problema y motivación | 2:00 | Foto real de paradero de Valparaíso, quién usaría esto |
| 3 | Tarea formalizada | 1:30 | Entrada, salida, 3 clases, por qué clasificación y no detección |
| 4 | Datos | 2:00 | Tabla de fuentes, criterio de etiquetado textual, ejemplo por clase |
| 5 | Splits | 1:00 | Tabla con números, estratificación, holdout local |
| 6 | Setup experimental | 1:00 | Qué se mantiene idéntico entre los dos modelos |
| 7 | ResNet-50 | 2:00 | Curvas, macro-F1, matriz de confusión |
| 8 | ViT-B/16 | 2:00 | Mismo formato exacto que la lámina 7 |
| 9 | Comparación y errores | 1:30 | Tabla lado a lado + baseline trivial + test local |
| 10 | Riesgos y mitigación | 2:00 | Los concretos, con evidencia |
| 11 | Plan restante | 1:00 | Qué falta para el informe final |

**Total: 15:15** — ajustar recortando de las láminas 2 y 10 si se pasan.

---

## Notas por lámina

**Lámina 2 — Problema.** Empezar con la foto, no con la definición. "Esta es la
escena a las 8 de la mañana en [paradero]. Queremos que un sistema sepa que está
lleno antes de que llegue la micro." Recién ahí el planteamiento formal.

**Lámina 3 — Tarea.** La pregunta que van a hacer es por qué no detección. La
respuesta: para que ResNet y ViT sean comparables cambiando solo el backbone.
Tenerla lista en una frase.

**Lámina 4 — Datos.** El criterio de etiquetado va escrito en pantalla, no
solo dicho. Es literalmente lo que evalúa la rúbrica.

**Lámina 6 — Setup.** Esta lámina parece prescindible y es de las que más suma:
demuestra que la comparación fue diseñada, no improvisada.

**Láminas 7 y 8.** Mismo layout, mismos ejes, mismas escalas. Que el cambio entre
una y otra solo mueva los números. Si los ejes cambian de escala, la comparación
visual engaña y alguien lo va a notar.

**Lámina 9 — Comparación.** Si ViT quedó abajo, decirlo de frente y explicar por
qué: sesgo inductivo de las convoluciones, el Transformer necesita más datos,
sensibilidad al learning rate. Un hallazgo explicado vale más que un número alto
sin procedencia.

**Lámina 11 — Plan restante.** Terminar con preguntas abiertas propias, no con
"gracias". Muestra que el proyecto sigue vivo.

---

## Láminas de respaldo (después de la última)

1. ¿Cómo saben que no aprendió el dataset? → `per_source.csv` + mapas de atención
2. ¿Por qué umbrales 3 y 10? → análisis de sensibilidad
3. ¿Qué da el baseline trivial? → tabla de `baselines_triviales.csv`
4. ¿Por qué ViT quedó abajo? → curvas de learning rate y tamaño efectivo del dataset
5. Hiperparámetros completos de ambos modelos

---

## Reglas de ensayo

- Dos pasadas cronometradas completas, mínimo.
- Cada integrante tiene que poder responder sobre **los dos** modelos, no solo
  del que entrenó. Si preguntan por ViT y contesta solo quien lo entrenó, se nota.
- El criterio "Tiempo" se pierde por pasarse, no por quedarse corto. Si van a
  15:30, recorten contenido antes de hablar más rápido.
