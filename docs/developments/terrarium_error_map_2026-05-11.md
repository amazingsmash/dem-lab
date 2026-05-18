# Terrarium Error Map Button

## Objective

Add a viewer control on the Terrarium DEM layer that calculates an error map for the currently selected Terrarium LoD. The calculation downloads Terrarium at a higher LoD for the same requested BBOX, samples the selected low-resolution Terrarium triangle mesh at the high-resolution pixel centers, and produces three image artifacts: low-LoD hillshade, high-LoD hillshade, and absolute error map.

## Assumptions

- Terrarium tile coordinates are treated as `EPSG:3857` Web Mercator.
- Elevations decoded from Terrarium RGB are meters.
- The selected LoD is the low-resolution DEM under evaluation.
- The high-resolution reference defaults to the server `--max-lod`.
- The low-resolution mesh interpolation is piecewise planar over the same regular cell split used by the WebGL viewer.
- The comparison uses the requested Terrarium BBOX rather than recursively processing subfolders or external data.

## Inputs

- Current Terrarium LoD from the viewer.
- Current Terrarium requested BBOX in `EPSG:3857`.
- Terrarium tiles from `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/`.

## Outputs

- Runtime PNG artifacts under `outputs/dem_terrarium_experiment/terrarium_error_maps/...`:
  - `terrarium_low_hillshade.png`
  - `terrarium_high_hillshade.png`
  - `terrarium_error_map.png`
- Runtime traceability reports:
  - `terrarium_error_report.json`
  - `terrarium_error_report.md`
- Browser panel showing the three generated images and metrics.

## Method

1. Download and decode the selected low LoD Terrarium grid and the high LoD Terrarium grid for the same BBOX.
2. Crop both grids to pixel centers inside the requested BBOX.
3. For every high-resolution pixel center, interpolate the low-resolution DEM on the selected regular triangle mesh.
4. Compute `abs(z_high_terrarium - z_low_triangle_interpolated)`.
5. Render low and high DEMs as grayscale hillshade images.
6. Render error as blue-to-red where blue is `0 m` and red is the maximum absolute error from the run.

## Geospatial Notes

- Input CRS: `EPSG:3857`.
- Output CRS: `EPSG:3857`.
- Rasterization resolution: native Terrarium pixel-center resolution for each LoD; no vector rasterization is performed.
- Per-pixel aggregation rule: Terrarium source elevation per tile pixel; no aggregation.
- Validity mask: finite high-resolution Terrarium samples inside the requested BBOX.
- Formula: `abs(z_high_terrarium - z_low_triangle_interpolated)`.

## Parameters

- Default high LoD: `--max-lod`.
- Safety limit: `4,000,000` high-resolution source pixels per request.
- Hillshade azimuth: `315 degrees`.
- Hillshade altitude: `45 degrees`.

## Verification

- Static checks: `python -m py_compile scripts/lod_terrarium_viewer.py`.
- Static checks: `node --check scripts/lod_viewer.js` with the bundled Node runtime.
- Runtime direct Python check generated a `z10` to `z15` error map with `1,223,540` high-resolution samples.
- UI handling was updated so backend errors open the error-map panel instead of only writing a transient status message.

## Changed Files

- `scripts/lod_terrarium_viewer.py`
- `scripts/lod_viewer.html`
- `scripts/lod_viewer.js`
- `scripts/lod_viewer.css`
