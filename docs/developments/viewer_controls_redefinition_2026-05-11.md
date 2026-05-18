# Viewer Controls Redefinition

## Objective

Redefine LoD viewer mouse controls so normal dragging performs terrain-anchored pan, mouse wheel zooms toward the terrain point under the pointer, `Ctrl+drag` rotates around the clicked terrain point, and `Shift+drag` remains a pan operation while preserving the terrain pivot projection.

## Assumptions

- The pivot point is the nearest ray intersection between the pointer and the current blended terrain mesh.
- If the pointer does not intersect the terrain mesh, pan falls back to screen-space pan and zoom falls back to regular distance scaling.
- `Ctrl+drag` preserves the clicked pivot in screen projection by rotating first and then solving `panX` and `panY`.
- `Shift+drag` is retained as a pan gesture for continuity, but now uses the same terrain-anchored projection rule as plain drag.

## Inputs

- Mouse button, wheel, `Ctrl`, and `Shift` interaction state.
- Current camera parameters: `yaw`, `pitch`, `dist`, `panX`, and `panY`.
- Current MVP matrix and inverse MVP matrix.
- Current blended terrain mesh triangles.

## Outputs

- Updated HUD control hint:
  - `drag: terrain pan | wheel: zoom to terrain | ctrl+drag: pivot rotate | shift+drag: terrain pan`
- Updated drag behavior:
  - Plain drag: terrain-anchored pan.
  - `Shift+drag`: terrain-anchored pan.
  - `Ctrl+drag`: pivot rotation.
- Updated wheel behavior:
  - Zoom distance changes while keeping the terrain point under the pointer projected under the pointer.
- Modified files:
  - `scripts/lod_viewer.html`
  - `scripts/lod_viewer.js`

## Method

1. On mouse down, raycast from the pointer to the current terrain mesh and store the nearest intersection as `dragAnchor`.
2. Set drag mode to `rotate` only when `Ctrl` is pressed; otherwise use `pan`.
3. For pan mode, solve `panX` and `panY` so `dragAnchor` projects to the current pointer position.
4. For rotate mode, update `yaw` and `pitch`, then solve `panX` and `panY` so the pivot remains fixed in screen projection.
5. On wheel, raycast the pointer before changing `dist`; after zooming, solve `panX` and `panY` against that same anchor.

## CRS and Raster Rules

- Input CRS: unchanged from the loaded viewer experiment, normally `EPSG:3857`.
- Output CRS: unchanged, `EPSG:3857` viewer grid.
- Rasterization resolution: unchanged; interaction uses the already-loaded viewer mesh resolution.
- Per-pixel aggregation rule: unchanged from the project pipeline, minimum `z` for Cloud DEM rasterization.
- Validity mask: unchanged, finite cells in the selected blended surface.

## Metrics

- No DEM comparison metrics were recomputed because this is a viewer interaction change.
- Metric formula, validity mask, and sample counts remain those declared by the loaded experiment report.

## Runtime Notes

- No DEM data were processed.
- No subfolders were processed.
- No original data were modified.
- JavaScript syntax was validated with `Get-Content -Raw scripts\lod_viewer.js | node --check`.
- Static content checks verified the new hint, drag mode selection, drag anchor, pivot-preserving rotation, and zoom anchor behavior.
