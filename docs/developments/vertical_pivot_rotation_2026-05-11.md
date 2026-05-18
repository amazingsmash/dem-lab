# Vertical Pivot Rotation

## Objective

Correct `Ctrl+drag` rotation so the camera rotates around the clicked terrain pivot using the vertical up vector as the rotation axis.

## Assumptions

- The viewer scene coordinate system uses `x` and `z` as horizontal axes and `y` as elevation.
- The vertical up rotation axis is therefore the scene `y` axis through the terrain pivot.
- `Ctrl+drag` horizontal movement controls rotation angle around this vertical axis.
- Vertical mouse movement is ignored for `Ctrl+drag` rotation because the requested axis is vertical only.

## Inputs

- Terrain pivot from ray intersection with the current blended mesh.
- Current camera target translation: `panX`, `panY`, and new `panZ`.
- Current `yaw`, `pitch`, and `dist`.
- Mouse horizontal delta `dx`.

## Outputs

- Added `panZ` camera target translation so the camera center can orbit around arbitrary terrain pivots in the horizontal plane.
- Updated `currentMvp()` and `render()` camera centers to use `[panX, panY, panZ]`.
- Updated `Ctrl+drag` to rotate `panX` and `panZ` around the pivot while incrementing `yaw`.
- Modified file:
  - `scripts/lod_viewer.js`

## Method

For pivot `P = (px, py, pz)` and camera target horizontal position `C = (panX, panZ)`, a `Ctrl+drag` angle `a` applies:

- `dx0 = panX - px`
- `dz0 = panZ - pz`
- `panX = px + dx0 * cos(a) + dz0 * sin(a)`
- `panZ = pz - dx0 * sin(a) + dz0 * cos(a)`
- `yaw = yaw + a`

This rotates the camera target and eye orbit around the vertical line through the pivot.

## CRS and Raster Rules

- Input CRS: unchanged from loaded viewer experiment, normally `EPSG:3857`.
- Output CRS: unchanged, `EPSG:3857` viewer grid.
- Rasterization resolution: unchanged; interaction uses the already-loaded viewer mesh.
- Per-pixel aggregation rule: unchanged from the project pipeline, minimum `z` for Cloud DEM rasterization.
- Validity mask: unchanged, finite cells in the selected blended surface.

## Metrics

- No DEM comparison metrics were recomputed because this is a viewer interaction change.

## Runtime Notes

- No DEM data were processed.
- No subfolders were processed.
- No original data were modified.
- JavaScript syntax was validated with `Get-Content -Raw scripts\lod_viewer.js | node --check`.
- Static content checks verified `panZ`, camera center updates, vertical pivot rotation, and absence of pitch changes in `Ctrl+drag`.
