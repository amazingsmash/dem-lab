# ISPRS IDW Buffer Blending Mode

## Objective

Add a selectable DEM blending mode based on Chandra et al. (2025), "Integrated Multi-Resolution DEM Generation: Merging Airborne LiDAR and CartoDEM for Seamless Terrain Modeling".

## Paper Interpretation

The paper merges a high-accuracy LiDAR DEM with a coarser support DEM by keeping the LiDAR DEM inside its footprint, keeping the support DEM outside the transition region, and adjusting the support DEM inside a buffer. The adjustment surface is built from elevation differences at the inner edge of the high-resolution DEM and zero-valued control points at the outer edge of the buffer. The paper reports a 300 m transition zone, IDW power 0.8, and 50 neighbors.

## Assumptions

- In this project, the Cloud DEM is the high-resolution DEM analogue.
- Terrarium is the coarser support DEM analogue.
- Existing viewer data are already represented in `EPSG:3857`; distances are interpreted in CRS meters.
- Vertical datum differences are not corrected by this mode. The adjustment surface only models local edge mismatch.
- The mode is implemented for the regular viewer mesh. Adaptive tessellation remains available for the pre-existing strategies and is not applied to this IDW mode.

## Inputs

- Cloud DEM grid sampled to the selected Terrarium LoD.
- Terrarium DEM grid for the selected LoD and optional BBOX.
- Cloud DEM validity mask.
- Parameters:
  - `transition_radius_m`: default `300.0`
  - `power`: default `0.8`
  - `neighbors`: default `50`

## Outputs

- API payload layer: `isprs_idw_blend`
- API payload metadata: `isprs_idw_parameters`
- UI strategy option: `ISPRS IDW Buffer`

## Method

1. Detect the Cloud DEM contour from the valid-cell mask.
2. Compute `H_diff = cloud_z - terrarium_z` at contour cells.
3. Generate zero-valued outer controls near the transition radius.
4. For invalid Cloud DEM cells reachable within the transition radius, interpolate `H_diff` with IDW.
5. Set `z = terrarium_z + H_diff` in the transition zone.
6. Preserve Cloud DEM where valid and Terrarium where outside the transition zone.

## CRS and Raster Rules

- Input CRS: inherited from the loaded experiment and viewer API, normally `EPSG:3857` after the project pipeline.
- Output CRS: `EPSG:3857` viewer grid.
- Rasterization resolution: selected Terrarium LoD for the blended layer; fixed Cloud DEM visualization resolution for the source layer.
- Per-pixel aggregation rule: unchanged from the project pipeline, minimum `z` for Cloud DEM rasterization.
- Validity mask: finite Cloud DEM cells.

## Verification

- Static checks: Python compile for `scripts/lod_terrarium_viewer.py`.
- UI checks: JavaScript syntax check with Node when available.
- Runtime DEM metrics were not recomputed because no local LAS/Terrarium experiment was executed during this development.

## Download Note

The requested `referencias/` folder was created and added to `.gitignore`. The sandbox blocked direct network download from PowerShell, so the PDF could not be written to disk from the local execution environment during this run.
