# Детальные модели для реалистичных кадров (scripts/hd_models.py)

- `ISS_stationary.glb` (44 МБ, в git не хранится) — NASA 3D Resources «ISS (stationary)», 247 тыс. треугольников,
  26 текстур. Не охраняется авторским правом США; credit NASA; эмблемы NASA не используются.
  Скачать: `curl -L -o ISS_stationary.glb https://raw.githubusercontent.com/0crispy/iss-tracker/main/project/models/iss/ISS_stationary.glb`
- `starlink_v2mini_satvis.glb` — Starlink V2 mini из генератора satvis-3d-models (github.com/Flowm/satvis-3d-models,
  MIT, © 2026 Florian Mauracher, см. `satvis_LICENSE`): размеры из заявки SpaceX в FCC (4.10.2022). Собран
  `node --experimental-strip-types satvis_gen.ts out.glb` рядом с `generators/{assembly,parts,starlink}.ts` из того
  репозитория и `npm install @gltf-transform/core@4.5.1`.
- `starlink_malacodart/` — «Starlink Spacex Satellite» by Malacodart (https://sketchfab.com/malacodart),
  https://sketchfab.com/3d-models/starlink-spacex-satellite-0a60f6720c5141c9a1c6d71aac108b31, CC-BY-4.0
  (см. `license.txt`). Детальная модель Starlink v1 (20 тыс. треугольников, PBR-текстуры 4096²). В кадрах из её деталей
  собран V2 mini в размерах по заявке SpaceX в FCC (`hd_models.starlink_parts_mal`): две шины v1 рядом на подложке
  4.1 × 2.7 м, крылья — по две панели v1 встык (12.8 × 4.1 м), коромысла из satvis. Это приближение: детали шины —
  от v1, у ячеек ширина растянута в 1.5 раза. Взята из github.com/AlextheYounga/starlink-satellite-tracker
  (public/gltf/starlink_spacex_satellite).
