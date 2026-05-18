# Shift Click Camera Tilt

- Objective: add a `Shift + left click drag` interaction to tilt the LoD viewer camera.
- Assumptions: camera tilt is the viewer `pitch` angle; the interaction modifies only the camera, not terrain data.
- Input CRS: unchanged viewer terrain coordinates, normally EPSG:3857.
- Output CRS: unchanged EPSG:3857 viewer coordinates.
- Inputs: left mouse button state, `Shift` key state, mouse vertical delta, current camera pitch.
- Outputs: updated camera `pitch` during drag and updated HUD hint.
- Parameters: pitch delta is `dy * 0.008` radians per pointer move, clamped to `[-1.45, 1.45]`.
- Rasterization resolution: unchanged; no DEM data are processed.
- Per-pixel aggregation rule: unchanged from the loaded experiment.
- Validity mask: unchanged; this is a viewer camera-control change.
- Metrics: no DEM comparison metrics were recomputed.
- Visualization: `Shift + left click drag` tilts the camera; `Ctrl + drag` remains pivot rotation; plain drag remains terrain pan.
- Runtime: no experiment data was generated.
- Errors: none during static verification.
- Warnings: if both `Shift` and `Ctrl` are held, `Shift` tilt takes precedence.
- Verification: `Get-Content .\scripts\lod_viewer.js -Raw | node --check -` passed.
