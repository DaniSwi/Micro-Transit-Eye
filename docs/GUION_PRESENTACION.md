# Guion de presentación — 15 minutos

11 láminas (deck real: `Micro- Transit Eye-apresentação.pdf`). Los tiempos suman
~15:30, así que hay que ir ajustado. Asignar un expositor por bloque y practicar
las transiciones — sobre todo la 6→9, que es donde vive la comparación real.

| # | Lámina | Tiempo | Qué tiene que quedar claro |
|---|---|---|---|
| 1 | Portada | 0:15 | Nombre del proyecto, curso, equipo (G7) |
| 2 | Definición del problema | 1:30 | Impacto operacional / seguridad vial / calidad de servicio, quién lo usaría |
| 3 | Formalización del proyecto | 1:00 | Entrada 224×224, 3 clases, umbrales 0-3/4-10/+10 |
| 4 | Fuentes y datos utilizados | 1:15 | Tabla COCO+CrowdHuman+Local, qué aporta cada una |
| 5 | Criterios de etiquetado + partición | 1:30 | Qué significa cada clase, tabla de splits con números |
| 6 | Resultados preliminares — ResNet50 | 2:00 | Baseline trivial, macro-F1 test vs local |
| 7 | ResNet50 — análisis de comportamiento | 1:30 | Matriz de confusión, errores adyacentes, conclusión |
| 8 | Resultados preliminares — ViT | 2:00 | Mismo formato que la 6, comparar de memoria con ResNet |
| 9 | ViT-B/16 — análisis de comportamiento | 2:00 | Caída en local, brecha de dominio, veredicto ResNet vs ViT |
| 10 | Riesgos identificados | 2:00 | Los 4 riesgos, cada uno con su mitigación |
| 11 | Gracias | 0:30 | Cierre + dar paso a preguntas |

**Total: ~15:30** — si se pasan, recortar de las láminas 2 y 10, no de los
resultados (6-9): ahí están los cuatro criterios de la rúbrica que valen más.

---

## Notas por lámina

**Lámina 1 — Portada.** Nombrar el proyecto y el curso, nada más. No leer la
foto del paradero todavía — se usa recién en la lámina 2.

**Lámina 2 — Definición del problema.** Empezar con una de las fotos reales de
paradero, no con la lista de bullets. Algo como: *"Este es un paradero del Gran
Valparaíso. Nadie sabe si está lleno hasta que llega la micro."* Recién ahí los
tres impactos (operacional, seguridad vial, calidad de servicio) y quién lo usaría
(municipio, operadores, pasajeros).

**Lámina 3 — Formalización.** Una frase de por qué clasificación y no detección/
conteo: *"Así ResNet y ViT son comparables cambiando solo el backbone — detección
necesitaría dos cabezas distintas y la comparación dejaría de ser limpia."*
Mostrar las tres fotos de ejemplo (baja/media/alta) apoyando los umbrales.

**Lámina 4 — Fuentes y datos.** La tabla ya está armada — leerla rápido y
detenerse en la columna "Aporte": COCO da contexto urbano general, CrowdHuman
aporta las escenas densas (multitudes, oclusión), el dataset local es el que
mide si esto sirve *en Valparaíso* de verdad.

**Lámina 5 — Criterios + partición.** El criterio de etiquetado va leído tal
cual aparece en pantalla (0-3 espacio libre, 4-10 fila parcial, +10 saturado con
riesgo de desborde) — es literalmente lo que evalúa la rúbrica. En la tabla de
partición, mencionar en una frase: estratificado por clase y fuente, semilla
fija, y que el conjunto local (81 imágenes) nunca entra a train/val — es holdout
puro.

**Lámina 6 — Resultados ResNet50.** Leer primero el piso (baseline trivial:
accuracy 0.1975, macro-F1 0.110 en local) para dar contexto de cuánto aporta el
modelo real. Después las dos tablas: macro-F1 sube a 0.6306 en test y se
mantiene relativamente en 0.5683 en local — bajada moderada.

**Lámina 7 — ResNet50 análisis.** La matriz de confusión (test) muestra que el
modelo distingue bien los extremos (baja 87% correcto, alta 61% correcto) y
confunde sobre todo `media` con sus vecinas. El punto central: 87% de los
errores en test (96% en local) son entre clases adyacentes → el modelo entiende
la densidad, el problema es la rigidez de los umbrales, no el modelo en sí.

**Lámina 8 — Resultados ViT.** Mismo formato que la 6 — leer los mismos cuatro
números (macro-F1, accuracy, errores adyacentes, ms/imagen) en el mismo orden,
para que la comparación visual con ResNet sea automática. Destacar de pasada:
ViT gana en test (0.6504 vs 0.6306) pero cuesta casi 4× más por imagen
(13.3ms vs 3.56ms) y tiene 3.6× más parámetros.

**Lámina 9 — ViT análisis — la lámina más importante de toda la presentación.**
Acá se da vuelta la historia: ViT que ganaba en test (0.65) se desploma a 0.44
en el entorno local — una caída mucho más fuerte que la de ResNet (0.63→0.57).
Decirlo de frente, sin suavizarlo: *"El modelo más grande y más caro es también
el que peor generaliza a Valparaíso."* La explicación ya está en el slide: ViT
memoriza patrones de datasets de internet (COCO/CrowdHuman) pero tiene menos
sesgo inductivo que una CNN, así que necesita más datos para no sobreajustarse
al estilo de esas fuentes. Conclusión explícita: *"para este volumen de datos,
la arquitectura convolucional generalizó mejor que el Transformer."*
**Ojo con la matriz de confusión de esta lámina:** la clase `alta` en local
tiene solo 2 imágenes — el 100% de acierto ahí no es significativo
estadísticamente, decirlo si preguntan (ver láminas de respaldo).

**Lámina 10 — Riesgos.** Los cuatro riesgos con su mitigación ya están escritos
en pantalla — no leerlos palabra por palabra, parafrasear y dar énfasis a la
mitigación de cada uno. El riesgo de "Brecha de Dominio Visual" conecta directo
con lo que se acaba de mostrar en la lámina 9 — se puede referenciar: *"esto es
justo lo que vimos recién con ViT."*

**Lámina 11 — Gracias.** Cerrar y dar paso a preguntas. No hay lámina de "plan
restante" en este deck — si el jurado pregunta qué sigue, la respuesta está en
las láminas de respaldo de abajo.

---

## Láminas de respaldo (para preguntas, no se muestran salvo que pregunten)

1. **¿Por qué el baseline trivial en local (macro-F1 0.110) es más bajo que el
   de test/val (0.175)?** → el conjunto local está mucho más desbalanceado
   (63 baja / 16 media / 2 alta de 81), así que hasta la clase mayoritaria
   acierta menos proporcionalmente en macro-F1.
2. **¿Por qué el 100% de acierto en `alta` local de ViT no es un buen dato?**
   → son solo 2 imágenes de esa clase en todo el holdout local — hace falta
   juntar más fotos con paraderos saturados para que esa celda signifique algo.
3. **¿Cómo saben que no aprendió el dataset en vez de la densidad real?** →
   `per_source.csv` de cada evaluación (desglose de macro-F1 por fuente: COCO,
   CrowdHuman, local).
4. **¿Por qué esos umbrales (3 y 10)?** → discretización manual sobre el
   conteo de personas anotadas; los errores adyacentes (87-96% de todos los
   errores) sugieren que el modelo entiende la densidad continua y el problema
   está en el corte discreto, no en la predicción.
5. **¿Qué sigue para el informe final?** → cerrar la brecha de dominio con más
   fotos locales y posible fine-tuning con ellas; análisis de sensibilidad de
   umbrales; considerar Grad-CAM/attention rollout para confirmar en qué mira
   cada modelo.

---

## Reglas de ensayo

- Dos pasadas cronometradas completas, mínimo.
- Cada integrante tiene que poder responder sobre **los dos** modelos, no solo
  del que entrenó. Si preguntan por ViT y contesta solo quien lo entrenó, se nota.
- El criterio "Tiempo" se pierde por pasarse, no por quedarse corto. Si van a
  15:45+, recorten contenido antes de hablar más rápido.
