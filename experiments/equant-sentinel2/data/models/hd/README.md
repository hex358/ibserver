# Детальные модели для реалистичных кадров (scripts/hd_models.py)

- `ISS_stationary.glb` (44 МБ, в git не хранится) — NASA 3D Resources «ISS (stationary)», 247 тыс. треугольников,
  26 текстур. Не охраняется авторским правом США; credit NASA; эмблемы NASA не используются.
  Скачать: `curl -L -o ISS_stationary.glb https://raw.githubusercontent.com/0crispy/iss-tracker/main/project/models/iss/ISS_stationary.glb`
- `starlink_v2mini_satvis.glb` — Starlink V2 mini из генератора satvis-3d-models (github.com/Flowm/satvis-3d-models,
  MIT, © 2026 Florian Mauracher, см. `satvis_LICENSE`): размеры из заявки SpaceX в FCC (4.10.2022). Собран
  `node --experimental-strip-types satvis_gen.ts out.glb` рядом с `generators/{assembly,parts,starlink}.ts` из того
  репозитория и `npm install @gltf-transform/core@4.5.1`.
