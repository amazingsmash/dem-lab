# Terrain-Anchored Ctrl Drag

## Objective

Add a `Ctrl+drag` viewer control that pans the camera while keeping the terrain mesh point initially under the pointer projected under the pointer during the drag.

## Assumptions

- The LoD viewer camera operates in normalized WebGL scene coordinates.
- The terrain surface for anchored panning is the current blended DEM surface.
- When adaptive refinement is active, the raycast uses the refined blended mesh triangles.
- When adaptive refinement is not active, the raycast uses the regular blended viewer grid triangles.
- If no terrain intersection is found, `Ctrl+drag` falls back to the existing screen-space pan behavior.

## Inputs

- Pointer screen coordinates from mouse events.
- Current camera MVP matrix.
- Current scene normalization values: `scene.cx`, `scene.cy`, `scene.cz`, and `scene.extent`.
- Current terrain mesh:
  - Regular grid: `currentTerrainGrid` and `currentBlendValues()`.
  - Refined mesh: `current.refined_mesh.layers[refinedLayerId()]`.

## Outputs

- Updated HUD control hint: `ctrl+drag: terrain pan`.
- New ray-to-terrain intersection path for anchored panning.
- Updated camera pan values `panX` and `panY` during `Ctrl+drag`.
- Modified files:
  - `scripts/lod_viewer.html`
  - `scripts/lod_viewer.js`

## Method

1. On `mousedown` with `Ctrl`, transform the pointer from screen coordinates to normalized device coordinates.
2. Invert the current MVP matrix to compute a world-space ray through the pointer.
3. Intersect the ray against the current blended terrain mesh triangles using a ray-triangle test.
4. Store the nearest intersection point in normalized scene coordinates.
5. On each mouse move, solve for `panX` and `panY` so the stored terrain point projects to the current pointer position.
6. If the projection solve fails, use the existing screen-space pan fallback.

## CRS and Raster Rules

- Input CRS: unchanged from the loaded viewer experiment, normally `EPSG:3857`.
- Output CRS: unchanged, `EPSG:3857` viewer grid.
- Rasterization resolution: unchanged; the control uses the already-loaded viewer mesh resolution.
- Per-pixel aggregation rule: unchanged from the project pipeline, minimum `z` for Cloud DEM rasterization.
- Validity mask: unchanged, finite cells in the selected blended surface.

## Metrics

- No DEM comparison metrics were recomputed because this is an interaction-only viewer change.
- The metric formula, validity mask, and sample counts remain those declared by the source experiment report.

## Runtime Notes

- No DEM data were processed.
- No subfolders were processed.
- No original data were modified.
- JavaScript syntax was validated with `Get-Content -Raw scripts\lod_viewer.js | node --check`.
