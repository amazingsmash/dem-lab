from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import laspy
import numpy as np
import requests
from PIL import Image
from pyproj import CRS, Transformer
import rasterio
from rasterio.transform import from_origin


WEB_MERCATOR_HALF_WORLD = 20037508.342789244
TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
NODATA = -9999.0


@dataclass
class StepLog:
    name: str
    start: str
    end: str | None = None
    elapsed_seconds: float | None = None
    status: str = "running"
    details: dict[str, Any] | None = None
    error: str | None = None


class ExperimentLog:
    def __init__(self) -> None:
        self.steps: list[StepLog] = []

    def start(self, name: str, details: dict[str, Any] | None = None) -> StepLog:
        step = StepLog(name=name, start=timestamp(), details=details or {})
        step.details["_t0"] = time.perf_counter()
        self.steps.append(step)
        return step

    def finish(self, step: StepLog, details: dict[str, Any] | None = None) -> None:
        t0 = step.details.pop("_t0", None) if step.details else None
        step.end = timestamp()
        step.elapsed_seconds = round(time.perf_counter() - t0, 3) if t0 else None
        step.status = "ok"
        if details:
            step.details.update(details)

    def fail(self, step: StepLog, error: Exception) -> None:
        t0 = step.details.pop("_t0", None) if step.details else None
        step.end = timestamp()
        step.elapsed_seconds = round(time.perf_counter() - t0, 3) if t0 else None
        step.status = "error"
        step.error = repr(error)

    def serializable(self) -> list[dict[str, Any]]:
        return [asdict(step) for step in self.steps]


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a minimum-z DEM from LAS files, compare it with a Terrarium elevation tile, and prepare a wiremesh viewer."
    )
    parser.add_argument("--las-dir", default=r"E:\266S_220ATF-CPR\03_TilesTScan")
    parser.add_argument("--out-dir", default="outputs/dem_terrarium_experiment")
    parser.add_argument("--source-crs", default="EPSG:32628", help="CRS for LAS coordinates when LAS headers do not contain one.")
    parser.add_argument("--target-crs", default="EPSG:3857")
    parser.add_argument("--pixel-size-m", type=float, default=0.5, help="Output DEM pixel size in target CRS meters.")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--viewer-max-cells", type=int, default=240)
    parser.add_argument("--force", action="store_true", help="Recompute outputs even if existing files are present.")
    return parser.parse_args()


def header_info(las_files: list[Path], fallback_crs: CRS) -> tuple[list[dict[str, Any]], CRS]:
    infos: list[dict[str, Any]] = []
    detected_crs: CRS | None = None
    for path in las_files:
        with laspy.open(path) as src:
            h = src.header
            crs = h.parse_crs()
            if crs is not None and detected_crs is None:
                detected_crs = crs
            infos.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "point_count": int(h.point_count),
                    "mins": [float(v) for v in h.mins],
                    "maxs": [float(v) for v in h.maxs],
                    "scales": [float(v) for v in h.scales],
                    "offsets": [float(v) for v in h.offsets],
                    "crs_in_header": crs.to_string() if crs else None,
                }
            )
    return infos, detected_crs or fallback_crs


def transformed_bounds(infos: list[dict[str, Any]], transformer: Transformer) -> tuple[float, float, float, float]:
    minx = min(info["mins"][0] for info in infos)
    miny = min(info["mins"][1] for info in infos)
    maxx = max(info["maxs"][0] for info in infos)
    maxy = max(info["maxs"][1] for info in infos)
    corners_x = [minx, minx, maxx, maxx]
    corners_y = [miny, maxy, miny, maxy]
    tx, ty = transformer.transform(corners_x, corners_y)
    return min(tx), min(ty), max(tx), max(ty)


def allocate_dem(path: Path, height: int, width: int, force: bool) -> np.memmap:
    if path.exists() and force:
        path.unlink()
    if path.exists():
        return np.memmap(path, dtype="float32", mode="r+", shape=(height, width))
    dem = np.memmap(path, dtype="float32", mode="w+", shape=(height, width))
    dem[:] = np.inf
    dem.flush()
    return dem


def rasterize_min_z(
    las_files: list[Path],
    dem: np.memmap,
    bounds: tuple[float, float, float, float],
    pixel_size: float,
    transformer: Transformer,
    chunk_size: int,
) -> dict[str, Any]:
    minx, miny, _maxx, maxy = bounds
    height, width = dem.shape
    flat_dem = dem.reshape(-1)
    total_points = 0
    used_points = 0
    per_file = []

    for path in las_files:
        file_points = 0
        file_used = 0
        t0 = time.perf_counter()
        with laspy.open(path) as src:
            for points in src.chunk_iterator(chunk_size):
                x, y = transformer.transform(points.x, points.y)
                z = np.asarray(points.z, dtype=np.float32)
                cols = np.floor((x - minx) / pixel_size).astype(np.int64)
                rows = np.floor((maxy - y) / pixel_size).astype(np.int64)
                valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height) & np.isfinite(z)
                if valid.any():
                    flat = rows[valid] * width + cols[valid]
                    np.minimum.at(flat_dem, flat, z[valid])
                    file_used += int(valid.sum())
                file_points += len(z)
        total_points += file_points
        used_points += file_used
        per_file.append(
            {
                "name": path.name,
                "points": file_points,
                "points_used": file_used,
                "elapsed_seconds": round(time.perf_counter() - t0, 3),
            }
        )
        dem.flush()

    valid_cells = int(np.isfinite(dem).sum())
    return {
        "total_points": total_points,
        "points_used": used_points,
        "valid_cells": valid_cells,
        "valid_cell_fraction": valid_cells / float(height * width),
        "per_file": per_file,
    }


def write_dem_geotiff(path: Path, dem: np.ndarray, bounds: tuple[float, float, float, float], pixel_size: float, crs: CRS) -> None:
    minx, _miny, _maxx, maxy = bounds
    transform = from_origin(minx, maxy, pixel_size, pixel_size)
    out = np.where(np.isfinite(dem), dem, NODATA).astype("float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=dem.shape[0],
        width=dem.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=NODATA,
        compress="deflate",
        tiled=True,
        BIGTIFF="IF_SAFER",
    ) as dst:
        dst.write(out, 1)


def write_preview_png(path: Path, dem: np.ndarray, max_size: int = 1800) -> None:
    step = max(1, math.ceil(max(dem.shape) / max_size))
    sample = np.asarray(dem[::step, ::step])
    valid = np.isfinite(sample)
    if not valid.any():
        Image.new("L", sample.shape[::-1], 0).save(path)
        return
    p2, p98 = np.nanpercentile(sample[valid], [2, 98])
    scaled = np.zeros(sample.shape, dtype=np.uint8)
    clipped = np.clip((sample - p2) / max(p98 - p2, 1e-6), 0, 1)
    scaled[valid] = (clipped[valid] * 255).astype(np.uint8)
    Image.fromarray(scaled, mode="L").save(path)


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(lat)
    n = 2**zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def mercator_to_tile(mx: float, my: float, zoom: int) -> tuple[int, int]:
    n = 2**zoom
    x = int(math.floor((mx + WEB_MERCATOR_HALF_WORLD) / (2 * WEB_MERCATOR_HALF_WORLD) * n))
    y = int(math.floor((WEB_MERCATOR_HALF_WORLD - my) / (2 * WEB_MERCATOR_HALF_WORLD) * n))
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    n = 2**z
    tile = 2 * WEB_MERCATOR_HALF_WORLD / n
    minx = -WEB_MERCATOR_HALF_WORLD + x * tile
    maxx = minx + tile
    maxy = WEB_MERCATOR_HALF_WORLD - y * tile
    miny = maxy - tile
    return minx, miny, maxx, maxy


def choose_covering_tile(bounds: tuple[float, float, float, float]) -> dict[str, Any]:
    minx, miny, maxx, maxy = bounds
    selected = None
    for z in range(15, -1, -1):
        tx0, ty0 = mercator_to_tile(minx, maxy, z)
        tx1, ty1 = mercator_to_tile(maxx, miny, z)
        if tx0 == tx1 and ty0 == ty1:
            selected = {"z": z, "x": tx0, "y": ty0, "bounds": tile_bounds(z, tx0, ty0)}
            break
    if selected is None:
        selected = {"z": 0, "x": 0, "y": 0, "bounds": tile_bounds(0, 0, 0)}
    tb = selected["bounds"]
    selected["resolution_m_per_pixel"] = (tb[2] - tb[0]) / 256.0
    return selected


def download_terrarium(tile: dict[str, Any], out_png: Path) -> str:
    url = TERRARIUM_URL.format(**tile)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    out_png.write_bytes(response.content)
    return url


def decode_terrarium(png_path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(png_path).convert("RGB"), dtype=np.float32)
    return rgb[:, :, 0] * 256.0 + rgb[:, :, 1] + rgb[:, :, 2] / 256.0 - 32768.0


def write_terrarium_geotiff(path: Path, elevation: np.ndarray, tile: dict[str, Any], crs: CRS) -> None:
    minx, _miny, _maxx, maxy = tile["bounds"]
    pixel = tile["resolution_m_per_pixel"]
    transform = from_origin(minx, maxy, pixel, pixel)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=elevation.shape[0],
        width=elevation.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=None,
        compress="deflate",
    ) as dst:
        dst.write(elevation.astype("float32"), 1)


def sample_terrarium(elevation: np.ndarray, tile: dict[str, Any], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    minx, _miny, _maxx, maxy = tile["bounds"]
    res = tile["resolution_m_per_pixel"]
    col = (x - minx) / res - 0.5
    row = (maxy - y) / res - 0.5
    c0 = np.floor(col).astype(np.int64)
    r0 = np.floor(row).astype(np.int64)
    c1 = c0 + 1
    r1 = r0 + 1
    wc = (col - c0).astype(np.float32)
    wr = (row - r0).astype(np.float32)
    c0 = np.clip(c0, 0, elevation.shape[1] - 1)
    c1 = np.clip(c1, 0, elevation.shape[1] - 1)
    r0 = np.clip(r0, 0, elevation.shape[0] - 1)
    r1 = np.clip(r1, 0, elevation.shape[0] - 1)
    z00 = elevation[r0, c0]
    z10 = elevation[r0, c1]
    z01 = elevation[r1, c0]
    z11 = elevation[r1, c1]
    return (z00 * (1 - wc) * (1 - wr) + z10 * wc * (1 - wr) + z01 * (1 - wc) * wr + z11 * wc * wr).astype("float32")


def compute_rmse(
    dem: np.ndarray,
    bounds: tuple[float, float, float, float],
    pixel_size: float,
    terrarium: np.ndarray,
    tile: dict[str, Any],
    block_rows: int = 512,
) -> dict[str, Any]:
    minx, _miny, _maxx, maxy = bounds
    height, width = dem.shape
    x_centers = minx + (np.arange(width, dtype=np.float64) + 0.5) * pixel_size
    sum_sq = 0.0
    sum_diff = 0.0
    count = 0
    min_diff = math.inf
    max_diff = -math.inf

    for r0 in range(0, height, block_rows):
        r1 = min(height, r0 + block_rows)
        block = np.asarray(dem[r0:r1, :])
        valid = np.isfinite(block)
        if not valid.any():
            continue
        y_centers = maxy - (np.arange(r0, r1, dtype=np.float64) + 0.5) * pixel_size
        xx, yy = np.meshgrid(x_centers, y_centers)
        terrain = sample_terrarium(terrarium, tile, xx, yy)
        diff = block[valid].astype(np.float64) - terrain[valid].astype(np.float64)
        sum_sq += float(np.sum(diff * diff))
        sum_diff += float(np.sum(diff))
        count += int(diff.size)
        min_diff = min(min_diff, float(np.min(diff)))
        max_diff = max(max_diff, float(np.max(diff)))

    return {
        "sample_count": count,
        "rmse_m": math.sqrt(sum_sq / count) if count else None,
        "mean_difference_m": sum_diff / count if count else None,
        "min_difference_m": min_diff if count else None,
        "max_difference_m": max_diff if count else None,
        "difference_definition": "cloud_min_z_m - terrarium_elevation_m",
        "sampling": "bilinear Terrarium sample at each valid cloud DEM pixel center",
    }


def create_viewer_mesh(
    path: Path,
    dem: np.ndarray,
    bounds: tuple[float, float, float, float],
    pixel_size: float,
    terrarium: np.ndarray,
    tile: dict[str, Any],
    max_cells: int,
) -> dict[str, Any]:
    minx, _miny, _maxx, maxy = bounds
    height, width = dem.shape
    stride = max(1, math.ceil(max(height, width) / max_cells))
    rows = np.arange(0, height, stride, dtype=np.int64)
    cols = np.arange(0, width, stride, dtype=np.int64)
    cloud = np.asarray(dem[np.ix_(rows, cols)], dtype=np.float32)
    xs = minx + (cols.astype(np.float64) + 0.5) * pixel_size
    ys = maxy - (rows.astype(np.float64) + 0.5) * pixel_size
    xx, yy = np.meshgrid(xs, ys)
    terrain = sample_terrarium(terrarium, tile, xx, yy)

    cloud_out = np.where(np.isfinite(cloud), cloud, np.nan).astype("float32")
    payload = {
        "metadata": {
            "description": "Downsampled regular grids for red/blue wiremesh visualization. Metrics use the full-resolution DEM, not this mesh.",
            "stride": int(stride),
            "nx": int(len(cols)),
            "ny": int(len(rows)),
            "crs": "EPSG:3857",
            "cloud_color": "red",
            "terrarium_color": "blue",
        },
        "xs": xs.round(3).tolist(),
        "ys": ys.round(3).tolist(),
        "cloud_z": none_for_nan(cloud_out),
        "terrarium_z": terrain.round(3).astype("float32").reshape(-1).tolist(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload["metadata"]


def none_for_nan(arr: np.ndarray) -> list[float | None]:
    flat = arr.reshape(-1)
    return [None if not np.isfinite(v) else round(float(v), 3) for v in flat]


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    terrarium = report.get("terrarium", {"url": "not completed", "tile": {"z": None, "x": None, "y": None, "resolution_m_per_pixel": None}})
    rmse = report.get("rmse", {"status": "not completed"})
    lines = [
        "# Experimento DEM minimo vs Terrarium",
        "",
        f"- Fecha: `{report['timestamp']}`",
        f"- LAS: `{report['inputs']['las_dir']}`",
        f"- CRS origen: `{report['crs']['source']}`",
        f"- CRS destino: `{report['crs']['target']}`",
        f"- Resolucion DEM: `{report['parameters']['pixel_size_m']} m`",
        f"- Regla por pixel: `{report['parameters']['aggregation']}`",
        "",
        "## Nota sobre resolucion",
        "",
        report["parameters"]["resolution_note"],
        "",
        "## Salidas",
        "",
    ]
    for key, value in report["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    lines += [
        "",
        "## Terrarium",
        "",
        f"- URL: `{terrarium['url']}`",
        f"- Tile z/x/y: `{terrarium['tile']['z']}/{terrarium['tile']['x']}/{terrarium['tile']['y']}`",
        f"- Resolucion tile: `{terrarium['tile']['resolution_m_per_pixel']} m/pixel`",
        "",
        "## RMSE",
        "",
    ]
    for key, value in rmse.items():
        lines.append(f"- {key}: `{value}`")
    lines += [
        "",
        "## Tiempos y pasos",
        "",
    ]
    for step in report["steps"]:
        lines.append(f"- `{step['name']}`: {step['status']} en `{step['elapsed_seconds']}` s")
        if step.get("error"):
            lines.append(f"  - Error: `{step['error']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = ExperimentLog()

    outputs = {
        "dem_memmap": str(out_dir / "cloud_minz_3857_0p5m.float32.memmap"),
        "cloud_dem_geotiff": str(out_dir / "cloud_minz_3857_0p5m.tif"),
        "cloud_dem_preview": str(out_dir / "cloud_minz_3857_0p5m_preview.png"),
        "terrarium_png": str(out_dir / "terrarium_covering_tile.png"),
        "terrarium_geotiff": str(out_dir / "terrarium_covering_tile_3857.tif"),
        "viewer_meshes": str(out_dir / "viewer_meshes.json"),
        "report_markdown": str(out_dir / "experiment_report.md"),
        "report_json": str(out_dir / "experiment_report.json"),
    }

    report: dict[str, Any] = {
        "timestamp": timestamp(),
        "inputs": {"las_dir": args.las_dir},
        "parameters": {
            "pixel_size_m": args.pixel_size_m,
            "aggregation": "minimum z of points falling inside each EPSG:3857 grid cell",
            "chunk_size": args.chunk_size,
            "resolution_note": (
                "La peticion indica 50 cm^2 por pixel. En este experimento se usa una celda de 0.5 m x 0.5 m, "
                "interpretando la intencion practica como resolucion de 50 cm. La interpretacion literal de area "
                "50 cm^2 implicaria un lado de 0.0707107 m y un raster de varios miles de millones de celdas para esta extension."
            ),
        },
        "outputs": outputs,
    }

    try:
        step = log.start("discover_las")
        las_dir = Path(args.las_dir)
        las_files = sorted(p for p in las_dir.glob("*.las") if p.is_file())
        if not las_files:
            raise FileNotFoundError(f"No LAS files found in {las_dir}")
        source_crs = CRS.from_user_input(args.source_crs)
        target_crs = CRS.from_user_input(args.target_crs)
        infos, actual_source_crs = header_info(las_files, source_crs)
        report["las_files"] = infos
        report["crs"] = {
            "source": actual_source_crs.to_string(),
            "target": target_crs.to_string(),
            "source_reason": "LAS header CRS when present, otherwise --source-crs fallback",
        }
        log.finish(step, {"file_count": len(las_files), "total_points": sum(i["point_count"] for i in infos)})

        transformer = Transformer.from_crs(actual_source_crs, target_crs, always_xy=True)
        step = log.start("compute_bounds")
        bounds = transformed_bounds(infos, transformer)
        width = int(math.ceil((bounds[2] - bounds[0]) / args.pixel_size_m))
        height = int(math.ceil((bounds[3] - bounds[1]) / args.pixel_size_m))
        report["grid"] = {
            "bounds_3857": bounds,
            "width": width,
            "height": height,
            "cell_count": int(width * height),
        }
        log.finish(step, report["grid"])

        step = log.start("rasterize_las_min_z")
        dem_path = Path(outputs["dem_memmap"])
        dem_exists = dem_path.exists() and not args.force
        dem = allocate_dem(dem_path, height, width, args.force)
        if dem_exists:
            raster_stats = {
                "resumed_from_existing_memmap": True,
                "valid_cells": int(np.isfinite(dem).sum()),
                "valid_cell_fraction": int(np.isfinite(dem).sum()) / float(height * width),
            }
        else:
            raster_stats = rasterize_min_z(las_files, dem, bounds, args.pixel_size_m, transformer, args.chunk_size)
        report["rasterization"] = raster_stats
        log.finish(step, raster_stats)

        step = log.start("write_cloud_dem_artifacts")
        if args.force or not Path(outputs["cloud_dem_geotiff"]).exists():
            write_dem_geotiff(Path(outputs["cloud_dem_geotiff"]), dem, bounds, args.pixel_size_m, target_crs)
        if args.force or not Path(outputs["cloud_dem_preview"]).exists():
            write_preview_png(Path(outputs["cloud_dem_preview"]), dem)
        log.finish(step)

        step = log.start("download_decode_terrarium")
        tile = choose_covering_tile(bounds)
        url = download_terrarium(tile, Path(outputs["terrarium_png"]))
        terrarium = decode_terrarium(Path(outputs["terrarium_png"]))
        write_terrarium_geotiff(Path(outputs["terrarium_geotiff"]), terrarium, tile, target_crs)
        report["terrarium"] = {"url": url, "tile": tile}
        log.finish(step, {"url": url, "tile": tile})

        step = log.start("compute_rmse")
        rmse = compute_rmse(dem, bounds, args.pixel_size_m, terrarium, tile)
        report["rmse"] = rmse
        log.finish(step, rmse)

        step = log.start("create_viewer_mesh")
        mesh_meta = create_viewer_mesh(
            Path(outputs["viewer_meshes"]), dem, bounds, args.pixel_size_m, terrarium, tile, args.viewer_max_cells
        )
        report["viewer"] = mesh_meta
        log.finish(step, mesh_meta)

    except Exception as exc:
        if log.steps and log.steps[-1].status == "running":
            log.fail(log.steps[-1], exc)
        report["error"] = repr(exc)
        report["steps"] = log.serializable()
        Path(outputs["report_json"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
        write_markdown_report(Path(outputs["report_markdown"]), report)
        raise

    report["steps"] = log.serializable()
    Path(outputs["report_json"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(Path(outputs["report_markdown"]), report)
    print(json.dumps({"status": "ok", "report": outputs["report_json"], "markdown": outputs["report_markdown"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
