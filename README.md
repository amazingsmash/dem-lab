# dem-lab

Research lab for generating, comparing, and visualizing elevation models derived from LAS point clouds and Terrarium tiles.

## Purpose

This repository contains reproducible scripts to:

- Rasterize LAS files as a minimum-elevation-per-cell DEM.
- Compare the local DEM against Terrarium elevation in `EPSG:3857`.
- Log each run with Markdown and JSON reports.
- Prepare WebGL viewers for inspecting meshes, blends, and levels of detail.

## Geospatial Assumptions

- Default input CRS: `EPSG:32628`, used only when LAS headers do not include a CRS.
- Default output CRS: `EPSG:3857`.
- Default DEM resolution: `0.5 m`.
- Rasterization rule: minimum `z` from the points that fall inside each cell.
- Validity mask: cells with finite values in the cloud DEM.
- Comparison against Terrarium: bilinear Terrarium sampling at the center of each valid pixel in the local DEM.
- Reported difference: `cloud_min_z_m - terrarium_elevation_m`.

## Structure

```text
scripts/
  build_dem_terrarium_experiment.py  # main experiment and artifact logging
  viewer_wiremesh.py                 # static WebGL viewer from viewer_meshes.json
  lod_terrarium_viewer.py            # local server with LoD, blends, and refinement
launch_lod_viewer.bat                # Windows launcher for the LoD viewer
outputs/                             # generated outputs, ignored by git
```

Original data and heavy artifacts are excluded from git through `.gitignore`. Outputs must be written to result folders such as `outputs/`, `results/`, or `runs/`.

## Dependencies

Python 3.10+ is recommended.

```powershell
python -m pip install laspy numpy requests pillow pyproj rasterio
```

`lod_terrarium_viewer.py` and `build_dem_terrarium_experiment.py` download Terrarium tiles from `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/`, so they require network access during those stages.

## DEM vs Terrarium Experiment

Example using default values:

```powershell
python scripts\build_dem_terrarium_experiment.py
```

Example declaring inputs and outputs:

```powershell
python scripts\build_dem_terrarium_experiment.py `
  --las-dir "E:\266S_220ATF-CPR\03_TilesTScan" `
  --out-dir outputs\dem_terrarium_experiment `
  --source-crs EPSG:32628 `
  --target-crs EPSG:3857 `
  --pixel-size-m 0.5 `
  --chunk-size 1000000 `
  --viewer-max-cells 240
```

Main outputs:

- `cloud_minz_3857_0p5m.float32.memmap`: temporary minimum-`z` raster.
- `cloud_minz_3857_0p5m.tif`: local DEM as GeoTIFF.
- `cloud_minz_3857_0p5m_preview.png`: raster preview.
- `terrarium_covering_tile.png`: downloaded Terrarium tile.
- `terrarium_covering_tile_3857.tif`: Terrarium decoded as GeoTIFF.
- `viewer_meshes.json`: simplified meshes for visualization.
- `experiment_report.md`: traceable experiment report.
- `experiment_report.json`: structured record of parameters, runtimes, errors, paths, and metrics.

To recompute existing outputs:

```powershell
python scripts\build_dem_terrarium_experiment.py --force
```

## Viewers

Generate a static WebGL viewer from `viewer_meshes.json`:

```powershell
python scripts\viewer_wiremesh.py `
  --mesh outputs\dem_terrarium_experiment\viewer_meshes.json `
  --out outputs\dem_terrarium_experiment\wiremesh_viewer.html `
  --open
```

Run the local LoD viewer:

```powershell
python scripts\lod_terrarium_viewer.py `
  --report outputs\dem_terrarium_experiment\experiment_report.json `
  --cache-dir outputs\dem_terrarium_experiment\terrarium_lod_cache `
  --host 127.0.0.1 `
  --port 8765 `
  --max-lod 15 `
  --open
```

On Windows, this can also be used:

```powershell
.\launch_lod_viewer.bat 8765
```

## Required Traceability

Each experiment must preserve:

- Purpose, assumptions, inputs, outputs, and parameters.
- Detected or assumed input CRS and output CRS.
- Resolution, aggregation rule, and validity mask.
- Number of samples, metric formulas, and artifact paths.
- Runtime, warnings, errors, and dependency versions when relevant.

Subfolders are not processed unless the experiment explicitly declares it. Original data is not modified; scripts must write new results inside the workspace.
