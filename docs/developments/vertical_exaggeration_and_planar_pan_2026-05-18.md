# Vertical Exaggeration And Planar Pan

- Objective: expose vertical exaggeration as an `Effects` control and correct terrain drag so zenithal views pan across the scene instead of changing camera height.
- Assumptions: vertical exaggeration is a visualization scale applied only to rendered geometry; DEM elevations and comparison metrics remain in meters.
- Input CRS: unchanged viewer terrain coordinates, normally EPSG:3857.
- Output CRS: unchanged EPSG:3857 viewer coordinates.
- Inputs: selected DEM grids, point sample, refined blend geometry when active, `Vertical exaggeration` UI value, mouse drag events.
- Outputs: rebuilt viewer meshes using the current vertical exaggeration, updated raycast geometry, updated camera pan behavior.
- Parameters: default vertical exaggeration is `3.0`; invalid values fall back to `3.0`; non-negative values are accepted. Planar pan solves `panX` and `panZ` to keep the terrain anchor under the pointer.
- Rasterization resolution: unchanged; no DEM rasterization is performed by this change.
- Per-pixel aggregation rule: unchanged from the loaded experiment.
- Validity mask: unchanged from the loaded experiment.
- Metrics: no DEM metrics are recalculated; vertical exaggeration affects only visualization.
- Visualization: all rendered DEM meshes, point cloud positions, raycast terrain triangles, and profile line positions use `sceneElevation(z) = (z - scene.cz) * verticalExaggeration / scene.extent`.
- Camera controls: plain drag uses terrain-anchored planar pan in `panX/panZ`; fallback drag also uses `panX/panZ`; `Shift + drag` remains camera tilt.
- Runtime: no experiment data was generated.
- Errors: none during static verification.
- Warnings: changing vertical exaggeration rebuilds viewer buffers but does not re-fetch or recompute DEM data.
- Verification: `Get-Content .\scripts\lod_viewer.js -Raw | node --check -` passed.
