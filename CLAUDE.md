# req-classifier: contexto para futuras sesiones

Proyecto de portfolio. Clasifica requisitos de software por tipo y evalúa su calidad (ISO/IEC/IEEE 29148
+ INCOSE GtWR), usando Jev (TypeSafe AI) para los juicios semánticos. **El objetivo principal es medir con
rigor si Jev aporta valor** frente a alternativas más simples o baratas. Un resultado negativo bien
medido es válido. Plan completo: `docs/PLAN.md`. El usuario se comunica en español.

## Estado
- 2026-10-10: plan v2; las preguntas P1–P8 están resueltas (`docs/PLAN.md` §1).
- Fase 0 en curso. Hecho: workspace `uv` (Python 3.12; `packages/reqcore` + `bench`), clientes Jev y
  Claude con caché y presupuesto, datos (`uv run reqbench data all`), splits congelados (hash
  `224a6fa3…`), ejemplos `examples_v1`, `docs/DATA_CARD.md`, `docs/PREREGISTRATION.md`, 13 tests en verde.
  Pendiente: commit del prerregistro y los splits (antes de ejecutar ningún modelo), smoke test
  (`uv run reqbench smoke --run`, pide aprobación del usuario) y recogida del set fresco.
- Comandos: `uv run reqbench data all|download|build|split|examples` · `uv run reqbench budget` ·
  `uv run reqbench smoke [--run]` · `uv run --group dev pytest -q` · `uv run --group dev ruff check packages bench`. **La fase 1 (MVP) es SOLO el benchmark de tipo sobre PROMISE, con su
  informe. Nada de servicio ni interfaz hasta tener la decisión de la fase 1.**
- El usuario acepta el presupuesto de créditos de API (≤ $10).
- **Credenciales en `.env.local`** (nunca imprimir sus valores; cargar con `set -a; . ./.env.local; set +a`;
  `.env*` está en `.gitignore`):
  - `TYPESAFE_API_KEY`: clave directa de TypeSafe (es la variable que el SDK lee por defecto).
    Verificada el 2026-10-10 con `GET https://api.typesafe.ai/v1/models` (HTTP 200; lista solo los alias).
  - `ANTHROPIC_API_KEY`: API de Anthropic directa, con $10 de créditos cargados el 2026-10-10.
    Verificada con `GET /v1/models`; la cuenta tiene acceso a `claude-haiku-5-5` y `claude-opus-5-5`.
  - `AI_GATEWAY_API_KEY`: clave de Vercel AI Gateway. **No se usa** en el proyecto.
- **Decidido:** Jev se llama **directamente a TypeSafe** fijando `jev-1.13.0` (nunca por el gateway de
  Vercel: allí solo existe el alias `typesafe-ai/jev`, sin versión). Claude (Haiku 5.5 / Opus 5.5) se llama
  por la **API de Anthropic directa**, con Batch API disponible. Los otros modelos de decisión del gateway
  quedan fuera del alcance (posible extensión futura). No ejecutar `npx vercel ai-gateway setup`.
- **Skill de TypeSafe instalada** (plugin `typesafe@typesafe-ai` v0.5.7, ámbito de usuario). Usarla
  siempre que se trabaje con Jev. Según la propia skill, la documentación en vivo
  (https://docs.typesafe.ai/llms.txt, páginas en `.md`) manda: leer la página de la API o del SDK y la
  del primitivo correspondiente antes de escribir una integración.
- **Vercel AI Gateway (comprobado el 2026-10-10 con GET /v1/models):** ofrece `typesafe-ai/jev`
  (API compatible con TypeSafe en `https://ai-gateway.vercel.sh/typesafe`; la respuesta devuelve
  `"model": "typesafe-ai/jev"`, sin versión), `anthropic/claude-haiku-5.5` y `anthropic/claude-opus-5.5`.
  También ofrece otros modelos de decisión con la misma API: `liquid/d1`, `microsoft/microsoft-decision-1`,
  `openai/gpt-6-luna-decisions`, `convaiinnovations/laya` (gratis hasta el 31-10). Vercel anuncia cero
  margen sobre el precio del proveedor y $5 gratis al mes en el plan gratuito, solo para un subconjunto
  de modelos.

## Decisiones tomadas
- **Presupuesto:** < $10 en total para llamadas a modelos (previsto unos $4–5). La suscripción de Claude
  sirve para desarrollar con Claude Code, **no** como API: el benchmark usa créditos de prepago de la
  Console. Nunca usar Claude Code headless como sustituto de la API en el benchmark.
- **Dataset principal:** PROMISE NFR (Zenodo 268542, CC BY 4.0, 625 requisitos, 15 proyectos).
  Se eliminan 3 duplicados (como NICE) → 622. PROMISE_exp descartado (licencia no verificable).
- **Taxonomía `taxonomy_v1`, jerárquica:** nivel 1 functional (F) / non_functional; nivel 2 (NFR):
  performance=PE+SC · security=SE · usability=US+LF · operational_constraint=O+L+PO · other=A+FT+MN.
  La métrica principal es macro-F1 sobre las 6 hojas; F1 de nivel 1 y macro-F1 de nivel 2 se reportan
  siempre. Las 11 subclases originales y F/Q externo (Dalpiaz, GPL-3.0) son análisis secundarios.
- **Splits:** test bloqueado = proyectos completos (~30 %), elegidos por un script con semilla fija; dev =
  StratifiedGroupKFold(5) por proyecto. **Set fresco** de ~150 requisitos públicos de 2025 en adelante,
  con doble anotación (da también el κ humano para el tipo). **Test y fresco se evalúan una sola vez**,
  con todo congelado por hash.
- **Criterios de calidad:** Q1 singular, Q2a términos vagos, Q2b ambigüedad estructural, Q3 verificable,
  Q4 independiente de la solución (no se aplica a operational_constraint): reglas + Jev. Q5 conforme y
  Q6 completo/sin cláusulas abiertas: solo reglas. Hay segundo anotador disponible.
- **Sistemas del benchmark:** mayoritaria, reglas, TF-IDF+LR, bge-small+LR, DeBERTa-v3-small (MPS en
  MacBook Pro M4, fp32), Claude Haiku 5.5 (`claude-haiku-5-5`, pensamiento desactivado, k = 5) y Claude
  Opus 5.5 (`claude-opus-5-5`, solo en test y fresco, k = 1, Batch API, esfuerzo low). Jev: J1 Choice
  plano, J2 promedio de 3 órdenes, J3 probabilidades Jev → LR, J4 jerárquica (Noul `is_functional` +
  Choice `nfr_subtype`).
- **Criterios GO/NO-GO** prerregistrados en `docs/PLAN.md` §6.7 (δ = 0.03 macro-F1; en empate gana la
  opción más simple y barata). Copiarlos a `docs/PREREGISTRATION.md` y commitearlos antes de ejecutar
  modelos.
- **Stack:** todo en Python. Workspace `uv` con `packages/reqcore` (núcleo compartido: normalización,
  reglas, clientes, caché, presupuesto, calibración, pipeline), `bench/` (benchmark) y `service/`
  (FastAPI + SQLAlchemy 2 + Alembic + PostgreSQL, fase 3). Demo en Next.js (fase 4). **Spring Boot
  quedó descartado**: el servicio debe ejecutar exactamente el mismo código que se midió.

## Restricciones no negociables
- **Presupuesto:** toda ejecución que llame a APIs pasa por la guarda de `reqcore/budget.py` (*dry run*
  con estimación y subtopes: Jev $2, Haiku $2, Opus $3, reserva $3). Antes de cualquier corrida con
  coste, mostrar la estimación al usuario.
- **Jev fijado a `jev-1.13.0`**, nunca `jev-latest` ni `jev-preview`. Guardar `response.model`, el
  `usage` y la latencia en cada inferencia. Si cambia la versión: re-ejecutar dev, recalibrar, re-fijar
  umbrales y nueva evaluación en test.
- **Jev no es determinista** (σ ≈ 0.01 en Noul; Choice cambia de etiqueta en casos frontera). La clave
  de caché incluye `sample_index`. Medir siempre la estabilidad.
- **Caché** por `sha256(provider, pinned_version, question_set_hash, state, sample_index)` para Jev y LLM.
- **Calibración por pregunta** (Platt, temperatura o isotónica, elegida por log-loss en CV de dev). Nunca
  se ajusta con test. Si la identidad gana, se deja así y se reporta.
- **El texto de entrada no es confiable:** va solo en `state.requirement` (Jev) o delimitado en el
  mensaje de usuario con salida enum (LLM). Nunca se interpola en las instrucciones. Detector de
  inyección por reglas + Noul `addresses_ai`; cualquier sospecha → revisión humana.
- **Un requisito por request de Jev**, con todas sus preguntas en esa misma llamada. Choice siempre con
  opción `other`. Los ejemplos de los `criteria` solo pueden salir de dev.
- **Lo que resuelven las reglas no pasa por el modelo**, pero una regla solo es DECISIVA si su precisión
  en dev es ≥ 0.95; si no, es CANDIDATA y aporta evidencia.
- **El LLM y Jev reciben las mismas definiciones y los mismos ejemplos** (generados desde
  `questions/taxonomy_v1.yaml`). Máximo 15 iteraciones de prompt/preguntas, todas registradas.
- **Reproducibilidad:** datos descargados con checksums (no se redistribuyen), splits como IDs en git,
  semillas fijas y `uv run reqbench reproduce [--from-cache]`, que regenera todas las tablas y figuras.
- **Nunca se modifican las etiquetas de test.** Las auditorías de ruido se reportan aparte.

## Hechos de referencia (verificados el 2026-10-10)
- Jev 1.13: $0.042/Mtok de entrada (salida gratis); 80 req/s y 100K tok/s (cambian sin aviso);
  contexto de 64k (32k para state + la pregunta más larga); Noul sin `confidence` (usar |2p−1|);
  sesgo hacia la primera opción en Choice; vulnerable a contenido adversarial; SDKs en Python y JS.
- PROMISE: duplicado contradictorio («Only registered customers can purchase streaming movies.» SE/F);
  19 pares casi duplicados (17 dentro del mismo proyecto); PO = 1 ejemplo; los proyectos 11–15 no tienen
  funcionales; 13 % de requisitos con doble aspecto F+Q (reetiquetado de Dalpiaz).
- Precios Claude (por Mtok, entrada/salida): Haiku 5.5 $0.10/$0.50 · Opus 5.5 $4/$20; Batch API −50 %.
  Haiku 5.5 no acepta cambiar la temperatura (las probabilidades del LLM salen de k = 5 muestras) y
  permite desactivar el pensamiento con esfuerzo ≤ high. Opus 5.5 no permite desactivar el pensamiento.
