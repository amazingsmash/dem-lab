# Terrain Vertical Distance Metric

- Objective: show the maximum vertical distance between the Terrarium DEM and Cloud DEM in the viewer interface.
- Assumptions: both compared elevation values are in meters. The viewer comparison grid uses EPSG:3857 Web Mercator coordinates.
- Input CRS: Terrarium tile coordinates are EPSG:3857. Cloud DEM samples are reprojected/resampled into the same EPSG:3857 grid before comparison.
- Output CRS: EPSG:3857 for grid coordinates; elevation outputs remain meters.
- Inputs: selected Terrarium LoD grid, sampled Cloud DEM on the selected Terrarium grid, current Terrarium BBOX.
- Outputs: `/api/mesh` now includes `terrain_comparison_metrics`; the viewer status block displays `Max vertical distance`.
- Parameters: selected Terrarium LoD, optional manual Terrarium BBOX, Cloud DEM source settings, Terrarium grid resolution for the selected LoD.
- Rasterization resolution: the selected Terrarium LoD pixel-center resolution reported as `resolution_m`.
- Per-pixel aggregation rule: Cloud DEM values use the existing viewer sampling path, which reprojects the Cloud DEM into the selected Terrarium grid with minimum-value resampling.
- Validity mask: cells where both `z_cloud_dem` and `z_terrarium` are finite.
- Formula: `max(abs(z_cloud_dem - z_terrarium))` over the validity mask.
- Metrics recorded: sample count, maximum absolute vertical distance in meters, signed vertical difference at the maximum absolute distance.
- Visualization: the status entry is generated from the `/api/mesh` JSON payload; no additional downsampling is introduced for this metric.
- Runtime: no experiment data was generated; this is an interface and API payload change.
- Errors: direct-path `node --check .\scripts\lod_viewer.js` failed in the sandbox before checking syntax with `EPERM: operation not permitted, lstat 'C:\Users\Jose'`.
- Warnings: existing unrelated changes are present in `scripts/lod_terrarium_viewer.py` and `scripts/lod_viewer.js`; they were left intact.
- Dependency versions: Python 3.10.6; Node.js v18.15.0.
- Verification: `python -m py_compile scripts\lod_terrarium_viewer.py` passed; `Get-Content .\scripts\lod_viewer.js -Raw | node --check -` passed; `python -m json.tool docs\developments\terrain_vertical_distance_metric_2026-05-18.json` passed.
