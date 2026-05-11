# Blending Information Modal

## Objective

Add an information button to the LoD viewer blending strategy selector so users can read a short explanation of the selected blending method. For the ISPRS IDW Buffer method, include a link to the original "Integrated Multi-Resolution DEM Generation" paper.

## Assumptions

- The UI surface is `scripts/lod_viewer.html`, `scripts/lod_viewer.css`, and `scripts/lod_viewer.js`.
- The information button describes the strategy currently selected in the blending settings modal.
- The modal is informational only and does not change blending parameters or generated DEM data.
- Existing DEM processing CRS behavior remains unchanged: viewer inputs and outputs are treated as `EPSG:3857` unless documented otherwise by the experiment payload.

## Inputs

- Existing blending strategy identifiers from the viewer:
  - `distance`
  - `isprs_idw`
  - `vertical_distance`
  - `blur`
  - `naive`
- Original paper URL for the ISPRS IDW method:
  - `https://isprs-annals.copernicus.org/articles/X-5-W2-2025/77/2025/isprs-annals-X-5-W2-2025-77-2025.pdf`

## Outputs

- New `i` button beside the blending strategy label.
- New modal with method title, short description, relevant parameters, and the paper link when `ISPRS IDW Buffer` is selected.
- The modal loads KaTeX from jsDelivr and renders the `ISPRS IDW Buffer` equations in a dedicated formula block instead of showing raw formula text.
- The `ISPRS IDW Buffer` formula block includes:
  - `dH_i = cloud_z_i - terrarium_z_i` at inner edge controls.
  - `dH_i = 0` at outer buffer controls.
  - `dH = sum(w_i * dH_i) / sum(w_i)`, with `w_i = 1 / (d_i^p + eps)`.
  - `z = terrarium_z + dH`.
- Modified files:
  - `scripts/lod_viewer.html`
  - `scripts/lod_viewer.css`
  - `scripts/lod_viewer.js`

## Relevant Parameters

- No raster or blending calculation parameters were changed.
- The ISPRS IDW modal text references the existing transition radius, IDW power, and neighbor controls.
- UI math rendering dependency: KaTeX `0.16.45` from jsDelivr.

## CRS and Raster Rules

- Input CRS: unchanged from viewer data, normally `EPSG:3857`.
- Output CRS: unchanged, `EPSG:3857` viewer grid.
- Rasterization resolution: unchanged; selected Terrarium LoD for blended viewer layers.
- Per-pixel aggregation rule: unchanged from the project pipeline, minimum `z` for Cloud DEM rasterization.
- Validity mask: unchanged, finite Cloud DEM cells.

## Metrics

- No DEM comparison metrics were recomputed because this is a UI-only development.
- Metric formula, validity mask, and sample counts remain whatever the source experiment report declares.

## Verification Plan

- Run JavaScript syntax validation with Node when available.
- Open the local viewer and verify the blending settings flow:
  1. Open blending settings.
  2. Press the information button.
  3. Confirm the modal text changes for the selected method.
  4. Confirm the ISPRS IDW method shows a link to the original paper.

## Runtime Notes

- No DEM data were processed.
- No subfolders were processed.
- No original data were modified.
- KaTeX is used only for client-side rendering of formulas in the informational modal.
