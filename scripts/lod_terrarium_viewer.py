from __future__ import annotations

import argparse
import hashlib
import json
import heapq
import math
import time
import urllib.parse
import webbrowser
import laspy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import requests
import numpy as np
from PIL import Image
from pyproj import CRS, Transformer
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
import rasterio.windows


WEB_MERCATOR_HALF_WORLD = 20037508.342789244
TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
NODATA = -9999.0


STATIC_DIR = Path(__file__).resolve().parent
HTML_PATH = STATIC_DIR / "lod_viewer.html"
CSS_PATH = STATIC_DIR / "lod_viewer.css"
JS_PATH = STATIC_DIR / "lod_viewer.js"


class TerrainContext:
    def __init__(
        self,
        report_path: Path,
        cache_dir: Path,
        max_lod: int,
        cloud_resolution_m: float | None,
        blend_refinement_resolution_m: float | None,
        blur_radius_m: float | None,
    ) -> None:
        self.report_path = report_path
        self.root = report_path.parent.parent.parent
        self.report = json.loads(report_path.read_text(encoding="utf-8"))
        self.out_dir = report_path.parent
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        grid = self.report["grid"]
        self.terrarium_bounds = tuple(float(v) for v in grid["bounds_3857"])
        self.bounds: tuple[float, float, float, float] | None = None
        self.dem_width = 0
        self.dem_height = 0
        self.dem_pixel_size = float(self.report["parameters"]["pixel_size_m"])
        self.dem_path: Path | None = None
        self.dem_tif_path: Path | None = None
        self.cloud_loaded = False
        self.cloud_source: dict[str, Any] | None = None
        self.point_sample: list[list[float]] = []
        self.point_sample_path: Path | None = None
        self.max_lod = max_lod
        viewer_stride = int(self.report.get("viewer", {}).get("stride", 53))
        self.cloud_resolution_m = float(cloud_resolution_m) if cloud_resolution_m else viewer_stride * self.dem_pixel_size
        self.cloud_resolution_override = cloud_resolution_m is not None
        self.blend_refinement_resolution_m = (
            float(blend_refinement_resolution_m) if blend_refinement_resolution_m else self.cloud_resolution_m
        )
        self.default_blur_radius_m = max(0.0, float(blur_radius_m)) if blur_radius_m is not None else 6.0 * self.cloud_resolution_m
        self._cloud_layer: dict[str, Any] | None = None
        self._cloud_layer_arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def tile_bounds(self, z: int, x: int, y: int) -> tuple[float, float, float, float]:
        n = 2**z
        tile = 2 * WEB_MERCATOR_HALF_WORLD / n
        minx = -WEB_MERCATOR_HALF_WORLD + x * tile
        maxx = minx + tile
        maxy = WEB_MERCATOR_HALF_WORLD - y * tile
        miny = maxy - tile
        return minx, miny, maxx, maxy

    def mercator_to_tile(self, mx: float, my: float, z: int) -> tuple[int, int]:
        n = 2**z
        x = int(math.floor((mx + WEB_MERCATOR_HALF_WORLD) / (2 * WEB_MERCATOR_HALF_WORLD) * n))
        y = int(math.floor((WEB_MERCATOR_HALF_WORLD - my) / (2 * WEB_MERCATOR_HALF_WORLD) * n))
        return max(0, min(n - 1, x)), max(0, min(n - 1, y))

    def covering_tiles(
        self, z: int, bounds: tuple[float, float, float, float] | None = None
    ) -> list[tuple[int, int]]:
        minx, miny, maxx, maxy = bounds if bounds is not None else self.terrarium_bounds
        x0, y0 = self.mercator_to_tile(minx, maxy, z)
        x1, y1 = self.mercator_to_tile(maxx, miny, z)
        return [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]

    def lod_info(self) -> list[dict[str, Any]]:
        info = []
        for z in range(0, self.max_lod + 1):
            res = 2 * WEB_MERCATOR_HALF_WORLD / (2**z * 256)
            tiles = self.covering_tiles(z)
            xs = [tile[0] for tile in tiles]
            ys = [tile[1] for tile in tiles]
            info.append(
                {
                    "z": z,
                    "resolution_m": res,
                    "tiles": len(tiles),
                    "mesh_width": (max(xs) - min(xs) + 1) * 256,
                    "mesh_height": (max(ys) - min(ys) + 1) * 256,
                }
            )
        return info

    def download_tile(self, z: int, x: int, y: int) -> np.ndarray:
        path = self.cache_dir / str(z) / str(x) / f"{y}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            url = TERRARIUM_URL.format(z=z, x=x, y=y)
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        return rgb[:, :, 0] * 256.0 + rgb[:, :, 1] + rgb[:, :, 2] / 256.0 - 32768.0

    def terrarium_grid(
        self, z: int, bounds: tuple[float, float, float, float] | None = None
    ) -> dict[str, Any]:
        minx, miny, maxx, maxy = bounds if bounds is not None else self.terrarium_bounds
        res = 2 * WEB_MERCATOR_HALF_WORLD / (2**z * 256)
        x0, y0 = self.mercator_to_tile(minx, maxy, z)
        x1, y1 = self.mercator_to_tile(maxx, miny, z)
        mosaic = np.empty(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256), dtype=np.float32)
        tiles = []
        for ty, y in enumerate(range(y0, y1 + 1)):
            for tx, x in enumerate(range(x0, x1 + 1)):
                mosaic[ty * 256 : (ty + 1) * 256, tx * 256 : (tx + 1) * 256] = self.download_tile(z, x, y)
                tiles.append({"z": z, "x": x, "y": y})
        tile_minx, _tile_miny, _tile_maxx, tile_maxy = self.tile_bounds(z, x0, y0)
        _last_minx, tile_miny, tile_maxx, _last_maxy = self.tile_bounds(z, x1, y1)
        xs_full = tile_minx + (np.arange(mosaic.shape[1], dtype=np.float64) + 0.5) * res
        ys_full = tile_maxy - (np.arange(mosaic.shape[0], dtype=np.float64) + 0.5) * res
        return {
            "xs": xs_full,
            "ys": ys_full,
            "terrarium": mosaic,
            "tiles": tiles,
            "resolution": res,
            "bounds": (tile_minx, tile_miny, tile_maxx, tile_maxy),
            "requested_bounds": (minx, miny, maxx, maxy),
            "cloud_bounds": self.bounds,
        }

    def sample_cloud(self, xs: np.ndarray, ys: np.ndarray, resolution: float) -> np.ndarray:
        half = resolution / 2.0
        cloud = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
        if not self.cloud_loaded or self.dem_tif_path is None:
            return cloud
        with rasterio.open(self.dem_tif_path) as src:
            dst_transform = from_origin(float(xs[0] - half), float(ys[0] + half), resolution, resolution)
            reproject(
                source=rasterio.band(src, 1),
                destination=cloud,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=dst_transform,
                dst_crs=src.crs,
                dst_nodata=np.nan,
                resampling=Resampling.min,
            )
        if not np.isfinite(cloud).any():
            cloud = self.exact_cloud_footprint_min(xs, ys, resolution)
        return cloud

    def exact_cloud_footprint_min(self, xs: np.ndarray, ys: np.ndarray, resolution: float) -> np.ndarray:
        if not self.cloud_loaded or self.bounds is None or self.dem_tif_path is None:
            return np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
        minx, miny, maxx, maxy = self.bounds
        half = resolution / 2.0
        cloud = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
        candidate_cols = np.where((xs + half >= minx) & (xs - half <= maxx))[0]
        candidate_rows = np.where((ys + half >= miny) & (ys - half <= maxy))[0]
        if candidate_cols.size == 0 or candidate_rows.size == 0:
            return cloud
        with rasterio.open(self.dem_tif_path) as src:
            for r in candidate_rows:
                top = float(ys[r] + half)
                bottom = float(ys[r] - half)
                for c in candidate_cols:
                    left = float(xs[c] - half)
                    right = float(xs[c] + half)
                    window = rasterio.windows.from_bounds(left, bottom, right, top, src.transform)
                    window = window.round_offsets().round_lengths()
                    if window.width <= 0 or window.height <= 0:
                        continue
                    values = src.read(1, window=window, masked=True)
                    if values.count():
                        cloud[r, c] = float(values.min())
        return cloud

    def fixed_cloud_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._cloud_layer_arrays is not None:
            return self._cloud_layer_arrays
        if not self.cloud_loaded or self.bounds is None:
            xs = np.asarray([], dtype=np.float64)
            ys = np.asarray([], dtype=np.float64)
            cloud = np.empty((0, 0), dtype=np.float32)
            self._cloud_layer_arrays = (xs, ys, cloud)
            return self._cloud_layer_arrays
        minx, miny, maxx, maxy = self.bounds
        res = self.cloud_resolution_m
        nx = int(math.ceil((maxx - minx) / res))
        ny = int(math.ceil((maxy - miny) / res))
        xs = minx + (np.arange(nx, dtype=np.float64) + 0.5) * res
        ys = maxy - (np.arange(ny, dtype=np.float64) + 0.5) * res
        cloud = self.sample_cloud(xs, ys, res)
        self._cloud_layer_arrays = (xs, ys, cloud)
        return self._cloud_layer_arrays

    def fixed_cloud_layer(self) -> dict[str, Any]:
        if self._cloud_layer is not None:
            return self._cloud_layer
        xs, ys, cloud = self.fixed_cloud_arrays()
        self._cloud_layer = {
            "nx": int(len(xs)),
            "ny": int(len(ys)),
            "resolution_m": float(self.cloud_resolution_m),
            "xs": np.round(xs, 3).tolist(),
            "ys": np.round(ys, 3).tolist(),
            "z": [None if not np.isfinite(v) else round(float(v), 3) for v in cloud.reshape(-1)],
            "valid_count": int(np.isfinite(cloud).sum()),
            "loaded": bool(self.cloud_loaded),
            "note": "Fixed-resolution Cloud DEM visualization layer, independent of selected Terrarium LoD.",
        }
        return self._cloud_layer

    def bbox_dict(self, bounds: tuple[float, float, float, float]) -> dict[str, float]:
        return {
            "minx": float(bounds[0]),
            "miny": float(bounds[1]),
            "maxx": float(bounds[2]),
            "maxy": float(bounds[3]),
        }

    def header_info(self, las_files: list[Path], fallback_crs: CRS) -> tuple[list[dict[str, Any]], CRS]:
        infos: list[dict[str, Any]] = []
        detected_crs: CRS | None = None
        for path in las_files:
            with laspy.open(path) as src:
                header = src.header
                crs = header.parse_crs()
                if crs is not None and detected_crs is None:
                    detected_crs = crs
                infos.append(
                    {
                        "name": path.name,
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "point_count": int(header.point_count),
                        "mins": [float(v) for v in header.mins],
                        "maxs": [float(v) for v in header.maxs],
                        "crs_in_header": crs.to_string() if crs else None,
                    }
                )
        return infos, detected_crs or fallback_crs

    def transformed_bounds(self, infos: list[dict[str, Any]], transformer: Transformer) -> tuple[float, float, float, float]:
        minx = min(info["mins"][0] for info in infos)
        miny = min(info["mins"][1] for info in infos)
        maxx = max(info["maxs"][0] for info in infos)
        maxy = max(info["maxs"][1] for info in infos)
        corners_x = [minx, minx, maxx, maxx]
        corners_y = [miny, maxy, miny, maxy]
        tx, ty = transformer.transform(corners_x, corners_y)
        return min(tx), min(ty), max(tx), max(ty)

    def rasterize_las_min_z(
        self,
        las_files: list[Path],
        dem: np.memmap,
        bounds: tuple[float, float, float, float],
        pixel_size: float,
        transformer: Transformer,
        chunk_size: int,
    ) -> dict[str, Any]:
        minx, _miny, _maxx, maxy = bounds
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

    def write_cloud_geotiff(
        self, path: Path, dem: np.ndarray, bounds: tuple[float, float, float, float], pixel_size: float, crs: CRS
    ) -> None:
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

    def cloud_cache_key(
        self, las_dir: Path, las_files: list[Path], source_crs_text: str, actual_source_crs: CRS, pixel_size_m: float
    ) -> str:
        payload = {
            "las_dir": str(las_dir.resolve()),
            "source_crs_requested": source_crs_text,
            "source_crs_actual": actual_source_crs.to_string(),
            "target_crs": "EPSG:3857",
            "pixel_size_m": pixel_size_m,
            "files": [
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
                for path in las_files
            ],
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def sample_las_points(
        self,
        las_files: list[Path],
        infos: list[dict[str, Any]],
        transformer: Transformer,
        sample_count: int,
        chunk_size: int,
        seed_text: str,
    ) -> list[list[float]]:
        total_points = sum(int(info["point_count"]) for info in infos)
        if total_points <= 0:
            return []
        count = min(sample_count, total_points)
        rng = np.random.default_rng(int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16))
        selected = np.sort(rng.choice(total_points, size=count, replace=False))
        selected_pos = 0
        global_offset = 0
        sampled: list[list[float]] = []
        for path in las_files:
            with laspy.open(path) as src:
                for points in src.chunk_iterator(chunk_size):
                    chunk_len = len(points.x)
                    chunk_start = global_offset
                    chunk_end = global_offset + chunk_len
                    while selected_pos < count and selected[selected_pos] < chunk_start:
                        selected_pos += 1
                    start_pos = selected_pos
                    while selected_pos < count and selected[selected_pos] < chunk_end:
                        selected_pos += 1
                    if selected_pos > start_pos:
                        local_idx = selected[start_pos:selected_pos] - chunk_start
                        chunk_x = np.asarray(points.x)
                        chunk_y = np.asarray(points.y)
                        chunk_z = np.asarray(points.z)
                        x, y = transformer.transform(chunk_x[local_idx], chunk_y[local_idx])
                        z = np.asarray(chunk_z[local_idx], dtype=np.float64)
                        for px, py, pz in zip(x, y, z):
                            sampled.append([round(float(px), 3), round(float(py), 3), round(float(pz), 3)])
                    global_offset = chunk_end
        return sampled

    def activate_cloud_assets(
        self,
        dem_path: Path,
        geotiff_path: Path,
        point_sample_path: Path,
        source: dict[str, Any],
        bounds: tuple[float, float, float, float],
        width: int,
        height: int,
        pixel_size_m: float,
    ) -> None:
        if not self.cloud_resolution_override:
            self.cloud_resolution_m = max(pixel_size_m, max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 240.0)
            self.blend_refinement_resolution_m = self.cloud_resolution_m
            self.default_blur_radius_m = 6.0 * self.cloud_resolution_m
        self.bounds = bounds
        self.dem_width = width
        self.dem_height = height
        self.dem_pixel_size = pixel_size_m
        self.dem_path = dem_path
        self.dem_tif_path = geotiff_path
        self.cloud_loaded = True
        self.point_sample_path = point_sample_path
        self.point_sample = json.loads(point_sample_path.read_text(encoding="utf-8"))["points"] if point_sample_path.exists() else []
        self.cloud_source = source
        self._cloud_layer = None
        self._cloud_layer_arrays = None

    def load_cloud_dem_from_las(
        self, las_dir: Path, source_crs_text: str, pixel_size_m: float, chunk_size: int
    ) -> dict[str, Any]:
        start = time.perf_counter()
        if not las_dir.exists() or not las_dir.is_dir():
            raise ValueError(f"LAS folder does not exist: {las_dir}")
        las_files = sorted(path for path in las_dir.iterdir() if path.is_file() and path.suffix.lower() == ".las")
        if not las_files:
            raise ValueError(f"No direct .las files found in: {las_dir}")
        if pixel_size_m <= 0:
            raise ValueError("pixel_size_m must be greater than zero")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        fallback_crs = CRS.from_user_input(source_crs_text)
        target_crs = CRS.from_epsg(3857)
        infos, actual_source_crs = self.header_info(las_files, fallback_crs)
        transformer = Transformer.from_crs(actual_source_crs, target_crs, always_xy=True)
        bounds = self.transformed_bounds(infos, transformer)
        width = int(math.ceil((bounds[2] - bounds[0]) / pixel_size_m))
        height = int(math.ceil((bounds[3] - bounds[1]) / pixel_size_m))
        if width <= 0 or height <= 0:
            raise ValueError("Computed Cloud DEM grid is empty")
        cache_key = self.cloud_cache_key(las_dir, las_files, source_crs_text, actual_source_crs, pixel_size_m)
        run_dir = self.out_dir / "dynamic_cloud_dem" / "cache" / cache_key
        run_dir.mkdir(parents=True, exist_ok=True)
        dem_path = run_dir / "cloud_minz_3857.float32.memmap"
        geotiff_path = run_dir / "cloud_minz_3857.tif"
        point_sample_path = run_dir / "point_sample_3857.json"
        report_path = run_dir / "cloud_dem_load_report.json"
        cache_hit = dem_path.exists() and geotiff_path.exists() and point_sample_path.exists() and report_path.exists()
        if cache_hit:
            source = json.loads(report_path.read_text(encoding="utf-8"))
            source["cache_hit"] = True
            self.activate_cloud_assets(dem_path, geotiff_path, point_sample_path, source, bounds, width, height, pixel_size_m)
            payload = dict(source)
            payload["bbox"] = self.bbox_dict(bounds)
            payload["cloud_layer"] = self.fixed_cloud_layer()
            payload["point_sample"] = self.point_sample
            payload["report_json"] = str(report_path)
            payload["report_markdown"] = str(run_dir / "cloud_dem_load_report.md")
            return payload
        dem = np.memmap(dem_path, dtype="float32", mode="w+", shape=(height, width))
        dem[:] = np.inf
        dem.flush()
        raster_stats = self.rasterize_las_min_z(las_files, dem, bounds, pixel_size_m, transformer, chunk_size)
        self.write_cloud_geotiff(geotiff_path, dem, bounds, pixel_size_m, target_crs)
        dem.flush()
        sampled_points = self.sample_las_points(las_files, infos, transformer, 10_000, chunk_size, cache_key)
        point_sample_path.write_text(
            json.dumps(
                {
                    "crs": "EPSG:3857",
                    "sample_count": len(sampled_points),
                    "sampling": "deterministic random sample without replacement over all direct LAS points",
                    "points": sampled_points,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        source = {
            "las_dir": str(las_dir),
            "las_file_count": len(las_files),
            "cache_key": cache_key,
            "cache_hit": False,
            "source_crs": actual_source_crs.to_string(),
            "target_crs": target_crs.to_string(),
            "pixel_size_m": pixel_size_m,
            "chunk_size": chunk_size,
            "bounds_3857": bounds,
            "width": width,
            "height": height,
            "outputs": {"height_texture": str(geotiff_path), "memmap": str(dem_path), "point_sample": str(point_sample_path)},
            "rasterization": raster_stats,
            "point_sample": {
                "count": len(sampled_points),
                "crs": "EPSG:3857",
                "path": str(point_sample_path),
                "sampling": "deterministic random sample without replacement over all direct LAS points",
            },
            "elapsed_seconds": round(time.perf_counter() - start, 3),
        }
        self.activate_cloud_assets(dem_path, geotiff_path, point_sample_path, source, bounds, width, height, pixel_size_m)
        report_path.write_text(json.dumps(source, indent=2), encoding="utf-8")
        md_path = run_dir / "cloud_dem_load_report.md"
        md_path.write_text(
            "\n".join(
                [
                    "# Cloud DEM Load Report",
                    "",
                    f"- LAS folder: `{las_dir}`",
                    f"- LAS files: `{len(las_files)}`",
                    f"- Source CRS: `{actual_source_crs.to_string()}`",
                    "- Target CRS: `EPSG:3857`",
                    f"- Pixel size: `{pixel_size_m} m`",
                    "- Rasterization rule: minimum `z` per EPSG:3857 cell.",
                    f"- Cache key: `{cache_key}`",
                    "- Cache hit: `false`",
                    f"- Valid cells: `{raster_stats['valid_cells']}`",
                    f"- Point sample: `{len(sampled_points)}` points",
                    f"- Runtime: `{source['elapsed_seconds']} s`",
                    f"- GeoTIFF: `{geotiff_path}`",
                    f"- Memmap: `{dem_path}`",
                    f"- Point sample file: `{point_sample_path}`",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        payload = dict(source)
        payload["bbox"] = self.bbox_dict(bounds)
        payload["cloud_layer"] = self.fixed_cloud_layer()
        payload["point_sample"] = self.point_sample
        payload["report_json"] = str(report_path)
        payload["report_markdown"] = str(md_path)
        return payload

    def distance_blend(self, cloud: np.ndarray, terrarium: np.ndarray, xs: np.ndarray, ys: np.ndarray, radius: float) -> np.ndarray:
        dist, nearest_z = self.cloud_distance_field(cloud, xs, ys, radius)
        valid = np.isfinite(cloud)
        if not np.isfinite(nearest_z).any():
            return terrarium.copy()
        return self.apply_distance_blend(cloud, terrarium, dist, nearest_z, radius)

    def cloud_distance_field(
        self, cloud: np.ndarray, xs: np.ndarray, ys: np.ndarray, radius: float
    ) -> tuple[np.ndarray, np.ndarray]:
        valid = np.isfinite(cloud)
        ny, nx = cloud.shape
        dx = float(abs(np.median(np.diff(xs)))) if nx > 1 else radius / 6.0
        dy = float(abs(np.median(np.diff(ys)))) if ny > 1 else dx
        dist = np.full((ny, nx), np.inf, dtype=np.float32)
        nearest_z = np.full((ny, nx), np.nan, dtype=np.float32)
        if not valid.any():
            return dist, nearest_z
        queue: list[tuple[float, int, int, float]] = []
        for r, c in np.argwhere(valid):
            z = float(cloud[r, c])
            dist[r, c] = 0.0
            nearest_z[r, c] = z
            heapq.heappush(queue, (0.0, int(r), int(c), z))
        neighbors = [
            (-1, 0, dy),
            (1, 0, dy),
            (0, -1, dx),
            (0, 1, dx),
            (-1, -1, math.hypot(dx, dy)),
            (-1, 1, math.hypot(dx, dy)),
            (1, -1, math.hypot(dx, dy)),
            (1, 1, math.hypot(dx, dy)),
        ]
        max_distance = radius
        while queue:
            d, r, c, z = heapq.heappop(queue)
            if d != float(dist[r, c]) or d > max_distance:
                continue
            for dr, dc, step in neighbors:
                rr = r + dr
                cc = c + dc
                if rr < 0 or rr >= ny or cc < 0 or cc >= nx:
                    continue
                nd = d + step
                if nd < dist[rr, cc] and nd <= max_distance:
                    dist[rr, cc] = nd
                    nearest_z[rr, cc] = z
                    heapq.heappush(queue, (nd, rr, cc, z))
        return dist, nearest_z

    def apply_distance_blend(
        self, cloud: np.ndarray, terrarium: np.ndarray, dist: np.ndarray, nearest_z: np.ndarray, radius: float
    ) -> np.ndarray:
        valid = np.isfinite(cloud)
        blended = terrarium.copy()
        invalid = ~valid
        reachable = invalid & np.isfinite(dist)
        t = np.clip(dist[reachable] / radius, 0.0, 1.0)
        w = 0.5 * (1.0 + np.cos(np.pi * t))
        blended[reachable] = w * nearest_z[reachable] + (1.0 - w) * blended[reachable]
        blended[valid] = cloud[valid]
        return blended

    def vertical_distance_blend(
        self, cloud: np.ndarray, terrarium: np.ndarray, xs: np.ndarray, ys: np.ndarray, radius: float
    ) -> np.ndarray:
        dist, nearest_z = self.cloud_distance_field(cloud, xs, ys, radius)
        valid = np.isfinite(cloud)
        blended = terrarium.copy()
        invalid = ~valid
        reachable = invalid & np.isfinite(nearest_z) & np.isfinite(dist)
        if np.any(reachable):
            dz = np.abs(nearest_z[reachable] - terrarium[reachable])
            t = np.clip(dz / radius, 0.0, 1.0)
            w = 0.5 * (1.0 + np.cos(np.pi * t))
            blended[reachable] = w * nearest_z[reachable] + (1.0 - w) * blended[reachable]
        blended[valid] = cloud[valid]
        return blended

    def sample_regular_grid(self, values: np.ndarray, xs: np.ndarray, ys: np.ndarray, x: float, y: float) -> float:
        if values.size == 0 or len(xs) == 0 or len(ys) == 0:
            return float("nan")
        ny, nx = values.shape
        res_x = self.cloud_resolution_m if nx == 1 else abs(float(xs[1] - xs[0]))
        res_y = self.cloud_resolution_m if ny == 1 else abs(float(ys[1] - ys[0]))
        col = int(round((x - float(xs[0])) / res_x))
        row = int(round((float(ys[0]) - y) / res_y))
        if row < 0 or row >= ny or col < 0 or col >= nx:
            return float("nan")
        value = float(values[row, col])
        return value if np.isfinite(value) else float("nan")

    def cloud_contour_mask(self, cloud: np.ndarray) -> np.ndarray:
        valid = np.isfinite(cloud)
        contour = np.zeros(valid.shape, dtype=bool)
        if not valid.any():
            return contour
        ny, nx = valid.shape
        for r, c in np.argwhere(valid):
            if r == 0 or r == ny - 1 or c == 0 or c == nx - 1:
                contour[r, c] = True
                continue
            if not (valid[r - 1, c] and valid[r + 1, c] and valid[r, c - 1] and valid[r, c + 1]):
                contour[r, c] = True
        return contour

    def blur_blend_value(
        self,
        x: float,
        y: float,
        terrarium_z: float,
        cloud: np.ndarray,
        cloud_xs: np.ndarray,
        cloud_ys: np.ndarray,
        contour: np.ndarray,
        radius_m: float,
    ) -> float:
        cloud_z = self.sample_regular_grid(cloud, cloud_xs, cloud_ys, x, y)
        if np.isfinite(cloud_z):
            return cloud_z
        if radius_m <= 0.0:
            return float(terrarium_z)
        res_x = self.cloud_resolution_m if len(cloud_xs) == 1 else abs(float(cloud_xs[1] - cloud_xs[0]))
        res_y = self.cloud_resolution_m if len(cloud_ys) == 1 else abs(float(cloud_ys[1] - cloud_ys[0]))
        col_f = (x - float(cloud_xs[0])) / res_x
        row_f = (float(cloud_ys[0]) - y) / res_y
        row_radius = radius_m / res_y
        col_radius = radius_m / res_x
        r0 = max(0, int(math.floor(row_f - row_radius)))
        r1 = min(cloud.shape[0] - 1, int(math.ceil(row_f + row_radius)))
        c0 = max(0, int(math.floor(col_f - col_radius)))
        c1 = min(cloud.shape[1] - 1, int(math.ceil(col_f + col_radius)))
        if r0 > r1 or c0 > c1:
            return float(terrarium_z)
        weights: list[float] = []
        weighted_z = 0.0
        weight_sum = 0.0
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if not contour[rr, cc]:
                    continue
                distance = math.hypot(x - float(cloud_xs[cc]), y - float(cloud_ys[rr]))
                if distance > radius_m:
                    continue
                t = max(0.0, min(distance / radius_m, 1.0))
                weight = 0.5 * (1.0 + math.cos(math.pi * t))
                if weight <= 0.0:
                    continue
                z = float(cloud[rr, cc])
                weights.append(weight)
                weighted_z += weight * z
                weight_sum += weight
        if not weights or weight_sum <= 0.0:
            return float(terrarium_z)
        w_avg = float(sum(weights) / len(weights))
        z_contour_avg = weighted_z / weight_sum
        return w_avg * z_contour_avg + (1.0 - w_avg) * float(terrarium_z)

    def blur_blend(
        self, terrarium: np.ndarray, xs: np.ndarray, ys: np.ndarray, radius_m: float
    ) -> np.ndarray:
        cloud_xs, cloud_ys, cloud = self.fixed_cloud_arrays()
        if cloud.size == 0:
            return terrarium.copy()
        contour = self.cloud_contour_mask(cloud)
        blended = terrarium.copy()
        if not contour.any():
            return blended
        for row, y in enumerate(ys):
            for col, x in enumerate(xs):
                blended[row, col] = self.blur_blend_value(
                    float(x), float(y), float(terrarium[row, col]), cloud, cloud_xs, cloud_ys, contour, radius_m
                )
        return blended

    def bilinear_sample(self, values: np.ndarray, xs: np.ndarray, ys: np.ndarray, x: float, y: float) -> float:
        ny, nx = values.shape
        if nx == 1 or ny == 1:
            row = 0 if ny == 1 else int(np.clip(round((float(ys[0]) - y) / abs(float(ys[1] - ys[0]))), 0, ny - 1))
            col = 0 if nx == 1 else int(np.clip(round((x - float(xs[0])) / abs(float(xs[1] - xs[0]))), 0, nx - 1))
            return float(values[row, col])
        dx = abs(float(xs[1] - xs[0]))
        dy = abs(float(ys[1] - ys[0]))
        col = np.clip((x - float(xs[0])) / dx, 0.0, nx - 1.0)
        row = np.clip((float(ys[0]) - y) / dy, 0.0, ny - 1.0)
        c0 = int(math.floor(col))
        r0 = int(math.floor(row))
        c1 = min(c0 + 1, nx - 1)
        r1 = min(r0 + 1, ny - 1)
        tx = col - c0
        ty = row - r0
        top = (1.0 - tx) * float(values[r0, c0]) + tx * float(values[r0, c1])
        bottom = (1.0 - tx) * float(values[r1, c0]) + tx * float(values[r1, c1])
        return (1.0 - ty) * top + ty * bottom

    def nearest_cloud_value(
        self, cloud: np.ndarray, cloud_xs: np.ndarray, cloud_ys: np.ndarray, x: float, y: float
    ) -> float:
        if len(cloud_xs) == 0 or len(cloud_ys) == 0:
            return float("nan")
        res_x = self.cloud_resolution_m if len(cloud_xs) == 1 else abs(float(cloud_xs[1] - cloud_xs[0]))
        res_y = self.cloud_resolution_m if len(cloud_ys) == 1 else abs(float(cloud_ys[1] - cloud_ys[0]))
        col = int(round((x - float(cloud_xs[0])) / res_x))
        row = int(round((float(cloud_ys[0]) - y) / res_y))
        best_distance = float("inf")
        best_value = float("nan")
        for rr in range(max(0, row - 1), min(cloud.shape[0], row + 2)):
            if abs(y - float(cloud_ys[rr])) > res_y / 2.0:
                continue
            for cc in range(max(0, col - 1), min(cloud.shape[1], col + 2)):
                if abs(x - float(cloud_xs[cc])) > res_x / 2.0:
                    continue
                value = float(cloud[rr, cc])
                if not np.isfinite(value):
                    continue
                distance = math.hypot(x - float(cloud_xs[cc]), y - float(cloud_ys[rr]))
                if distance < best_distance:
                    best_distance = distance
                    best_value = value
        return best_value

    def nearest_grid_value(self, values: np.ndarray, grid_xs: np.ndarray, grid_ys: np.ndarray, x: float, y: float) -> float:
        if len(grid_xs) == 0 or len(grid_ys) == 0:
            return float("nan")
        res_x = self.cloud_resolution_m if len(grid_xs) == 1 else abs(float(grid_xs[1] - grid_xs[0]))
        res_y = self.cloud_resolution_m if len(grid_ys) == 1 else abs(float(grid_ys[1] - grid_ys[0]))
        col = int(round((x - float(grid_xs[0])) / res_x))
        row = int(round((float(grid_ys[0]) - y) / res_y))
        if row < 0 or row >= values.shape[0] or col < 0 or col >= values.shape[1]:
            return float("nan")
        value = float(values[row, col])
        return value if np.isfinite(value) else float("nan")

    def adaptive_blend_mesh(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        terrarium: np.ndarray,
        resolution: float,
        radius: float,
        blur_radius_m: float,
        tessellation_strategy: str = "quadtree",
    ) -> dict[str, Any]:
        if not self.cloud_loaded:
            return {"enabled": True, "applied": False, "reason": "Cloud DEM not loaded"}
        if tessellation_strategy == "nvb":
            return self.adaptive_nvb_blend_mesh(xs, ys, terrarium, resolution, radius, blur_radius_m)
        if tessellation_strategy == "diamond48":
            return self.adaptive_diamond48_blend_mesh(xs, ys, terrarium, resolution, radius, blur_radius_m)
        return self.adaptive_quadtree_blend_mesh(xs, ys, terrarium, resolution, radius, blur_radius_m)

    def adaptive_quadtree_blend_mesh(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        terrarium: np.ndarray,
        resolution: float,
        radius: float,
        blur_radius_m: float,
    ) -> dict[str, Any]:
        target = max(float(self.blend_refinement_resolution_m), float(self.dem_pixel_size))
        if resolution <= target:
            return {
                "enabled": True,
                "applied": False,
                "tessellation_strategy": "Quadtree quads",
                "target_resolution_m": target,
                "reason": "selected Terrarium mesh is already at or finer than the refinement target",
            }
        cloud_xs, cloud_ys, cloud = self.fixed_cloud_arrays()
        valid = np.isfinite(cloud)
        if not valid.any():
            return {
                "enabled": True,
                "applied": False,
                "tessellation_strategy": "Quadtree quads",
                "target_resolution_m": target,
                "reason": "fixed Cloud DEM layer has no valid support",
            }

        valid_ascending_y = valid[::-1, :].astype(np.int32)
        prefix = valid_ascending_y.cumsum(axis=0).cumsum(axis=1)
        cloud_ys_ascending = cloud_ys[::-1]
        footprint_pad = target / 2.0
        fixed_dist, fixed_nearest_z = self.cloud_distance_field(cloud, cloud_xs, cloud_ys, radius)
        contour = self.cloud_contour_mask(cloud)

        def prefix_sum(r0: int, c0: int, r1: int, c1: int) -> int:
            total = int(prefix[r1, c1])
            if r0 > 0:
                total -= int(prefix[r0 - 1, c1])
            if c0 > 0:
                total -= int(prefix[r1, c0 - 1])
            if r0 > 0 and c0 > 0:
                total += int(prefix[r0 - 1, c0 - 1])
            return total

        def contains_cloud_support(x0: float, y0: float, x1: float, y1: float, pad: float) -> bool:
            minx = min(x0, x1) - pad
            maxx = max(x0, x1) + pad
            miny = min(y0, y1) - pad
            maxy = max(y0, y1) + pad
            c0 = int(np.searchsorted(cloud_xs, minx, side="left"))
            c1 = int(np.searchsorted(cloud_xs, maxx, side="right")) - 1
            r0 = int(np.searchsorted(cloud_ys_ascending, miny, side="left"))
            r1 = int(np.searchsorted(cloud_ys_ascending, maxy, side="right")) - 1
            c0 = max(0, c0)
            r0 = max(0, r0)
            c1 = min(len(cloud_xs) - 1, c1)
            r1 = min(len(cloud_ys_ascending) - 1, r1)
            if c0 > c1 or r0 > r1:
                return False
            return prefix_sum(r0, c0, r1, c1) > 0

        leaves: list[tuple[float, float, float, float, int]] = []
        max_depth = 0

        def subdivide(x0: float, y0: float, x1: float, y1: float, depth: int) -> None:
            nonlocal max_depth
            width = abs(x1 - x0)
            height = abs(y1 - y0)
            if max(width, height) > target and contains_cloud_support(x0, y0, x1, y1, footprint_pad):
                midx = (x0 + x1) / 2.0
                midy = (y0 + y1) / 2.0
                subdivide(x0, y0, midx, midy, depth + 1)
                subdivide(midx, y0, x1, midy, depth + 1)
                subdivide(x0, midy, midx, y1, depth + 1)
                subdivide(midx, midy, x1, y1, depth + 1)
                return
            max_depth = max(max_depth, depth)
            leaves.append((x0, y0, x1, y1, depth))

        for row in range(len(ys) - 1):
            for col in range(len(xs) - 1):
                subdivide(float(xs[col]), float(ys[row]), float(xs[col + 1]), float(ys[row + 1]), 0)

        z_cache: dict[tuple[float, float], tuple[float, float, float, float]] = {}

        def z_values(x: float, y: float) -> tuple[float, float, float, float]:
            key = (round(x, 6), round(y, 6))
            cached = z_cache.get(key)
            if cached is not None:
                return cached
            terrarium_z = self.bilinear_sample(terrarium, xs, ys, x, y)
            cloud_z = self.nearest_cloud_value(cloud, cloud_xs, cloud_ys, x, y)
            replacement_z = cloud_z if np.isfinite(cloud_z) else terrarium_z
            blur_z = self.blur_blend_value(x, y, terrarium_z, cloud, cloud_xs, cloud_ys, contour, blur_radius_m)
            if np.isfinite(cloud_z):
                blend_z = cloud_z
                vertical_blend_z = cloud_z
            else:
                nearest_z = self.nearest_grid_value(fixed_nearest_z, cloud_xs, cloud_ys, x, y)
                distance = self.nearest_grid_value(fixed_dist, cloud_xs, cloud_ys, x, y)
                if np.isfinite(nearest_z) and np.isfinite(distance) and distance <= radius:
                    t = max(0.0, min(distance / radius, 1.0))
                    weight = 0.5 * (1.0 + math.cos(math.pi * t))
                    blend_z = weight * nearest_z + (1.0 - weight) * terrarium_z
                    dz = abs(nearest_z - terrarium_z)
                    vt = max(0.0, min(dz / radius, 1.0))
                    vweight = 0.5 * (1.0 + math.cos(math.pi * vt))
                    vertical_blend_z = vweight * nearest_z + (1.0 - vweight) * terrarium_z
                else:
                    blend_z = terrarium_z
                    vertical_blend_z = terrarium_z
            cached = (replacement_z, blend_z, blur_z, vertical_blend_z)
            z_cache[key] = cached
            return cached

        replacement_quads = []
        blend_quads = []
        blur_quads = []
        vertical_blend_quads = []
        for x0, y0, x1, y1, _depth in leaves:
            repl_00, blend_00, blur_00, vertical_00 = z_values(x0, y0)
            repl_10, blend_10, blur_10, vertical_10 = z_values(x1, y0)
            repl_01, blend_01, blur_01, vertical_01 = z_values(x0, y1)
            repl_11, blend_11, blur_11, vertical_11 = z_values(x1, y1)
            xy = [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)]
            replacement_quads.append(xy + [round(repl_00, 3), round(repl_10, 3), round(repl_01, 3), round(repl_11, 3)])
            blend_quads.append(xy + [round(blend_00, 3), round(blend_10, 3), round(blend_01, 3), round(blend_11, 3)])
            blur_quads.append(xy + [round(blur_00, 3), round(blur_10, 3), round(blur_01, 3), round(blur_11, 3)])
            vertical_blend_quads.append(xy + [round(vertical_00, 3), round(vertical_10, 3), round(vertical_01, 3), round(vertical_11, 3)])

        return {
            "enabled": True,
            "applied": True,
            "tessellation_strategy": "Quadtree quads",
            "geometry": "quads",
            "target_resolution_m": target,
            "blend_radius_m": float(radius),
            "source": "fixed Cloud DEM visualization grid",
            "rule": "Recursively split any blend quad that overlaps valid Cloud DEM support while its edge length is coarser than the target resolution.",
            "base_quad_count": int((len(xs) - 1) * (len(ys) - 1)),
            "quad_count": len(leaves),
            "max_depth": max_depth,
            "layers": {
                "cloud_replacement": {"quads": replacement_quads},
                "distance_blend": {"quads": blend_quads},
                "vertical_distance_blend": {"quads": vertical_blend_quads},
                "blur_blend": {"quads": blur_quads},
            },
        }

    def adaptive_nvb_blend_mesh(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        terrarium: np.ndarray,
        resolution: float,
        radius: float,
        blur_radius_m: float,
    ) -> dict[str, Any]:
        target = max(float(self.blend_refinement_resolution_m), float(self.dem_pixel_size))
        if resolution <= target:
            return {
                "enabled": True,
                "applied": False,
                "tessellation_strategy": "Newest Vertex Bisection (NVB)",
                "target_resolution_m": target,
                "reason": "selected Terrarium mesh is already at or finer than the refinement target",
            }
        cloud_xs, cloud_ys, cloud = self.fixed_cloud_arrays()
        valid = np.isfinite(cloud)
        if not valid.any():
            return {
                "enabled": True,
                "applied": False,
                "tessellation_strategy": "Newest Vertex Bisection (NVB)",
                "target_resolution_m": target,
                "reason": "fixed Cloud DEM layer has no valid support",
            }

        valid_ascending_y = valid[::-1, :].astype(np.int32)
        prefix = valid_ascending_y.cumsum(axis=0).cumsum(axis=1)
        cloud_ys_ascending = cloud_ys[::-1]
        footprint_pad = target / 2.0
        fixed_dist, fixed_nearest_z = self.cloud_distance_field(cloud, cloud_xs, cloud_ys, radius)
        contour = self.cloud_contour_mask(cloud)

        def prefix_sum(r0: int, c0: int, r1: int, c1: int) -> int:
            total = int(prefix[r1, c1])
            if r0 > 0:
                total -= int(prefix[r0 - 1, c1])
            if c0 > 0:
                total -= int(prefix[r1, c0 - 1])
            if r0 > 0 and c0 > 0:
                total += int(prefix[r0 - 1, c0 - 1])
            return total

        def contains_cloud_support_bbox(minx: float, miny: float, maxx: float, maxy: float, pad: float) -> bool:
            c0 = int(np.searchsorted(cloud_xs, minx - pad, side="left"))
            c1 = int(np.searchsorted(cloud_xs, maxx + pad, side="right")) - 1
            r0 = int(np.searchsorted(cloud_ys_ascending, miny - pad, side="left"))
            r1 = int(np.searchsorted(cloud_ys_ascending, maxy + pad, side="right")) - 1
            c0 = max(0, c0)
            r0 = max(0, r0)
            c1 = min(len(cloud_xs) - 1, c1)
            r1 = min(len(cloud_ys_ascending) - 1, r1)
            if c0 > c1 or r0 > r1:
                return False
            return prefix_sum(r0, c0, r1, c1) > 0

        Point2 = tuple[float, float]
        Triangle = tuple[tuple[Point2, Point2, Point2], int, int]

        def key_point(p: Point2) -> tuple[int, int]:
            return (int(round(p[0] * 1000.0)), int(round(p[1] * 1000.0)))

        def edge_key(a: Point2, b: Point2) -> tuple[tuple[int, int], tuple[int, int]]:
            ka, kb = key_point(a), key_point(b)
            return (ka, kb) if ka <= kb else (kb, ka)

        def refinement_edge(triangle: Triangle) -> tuple[tuple[int, int], tuple[int, int]]:
            vertices, newest, _depth = triangle
            edge_vertices = [vertices[i] for i in range(3) if i != newest]
            return edge_key(edge_vertices[0], edge_vertices[1])

        def max_edge_length(vertices: tuple[Point2, Point2, Point2]) -> float:
            return max(
                math.hypot(vertices[0][0] - vertices[1][0], vertices[0][1] - vertices[1][1]),
                math.hypot(vertices[1][0] - vertices[2][0], vertices[1][1] - vertices[2][1]),
                math.hypot(vertices[2][0] - vertices[0][0], vertices[2][1] - vertices[0][1]),
            )

        def triangle_has_cloud_support(vertices: tuple[Point2, Point2, Point2]) -> bool:
            minx = min(p[0] for p in vertices)
            maxx = max(p[0] for p in vertices)
            miny = min(p[1] for p in vertices)
            maxy = max(p[1] for p in vertices)
            return contains_cloud_support_bbox(minx, miny, maxx, maxy, footprint_pad)

        def should_refine(triangle: Triangle) -> bool:
            vertices, _newest, _depth = triangle
            return max_edge_length(vertices) > target and triangle_has_cloud_support(vertices)

        triangles: list[Triangle] = []
        for row in range(len(ys) - 1):
            for col in range(len(xs) - 1):
                a = (float(xs[col]), float(ys[row]))
                b = (float(xs[col + 1]), float(ys[row]))
                c = (float(xs[col]), float(ys[row + 1]))
                d = (float(xs[col + 1]), float(ys[row + 1]))
                triangles.append(((a, c, b), 2, 0))
                triangles.append(((b, c, d), 2, 0))

        max_depth = 0
        max_triangles = 500000

        def bisect(triangle: Triangle) -> tuple[Triangle, Triangle]:
            vertices, newest, depth = triangle
            edge_vertices = [vertices[i] for i in range(3) if i != newest]
            newest_vertex = vertices[newest]
            a, b = edge_vertices
            midpoint = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
            next_depth = depth + 1
            return ((newest_vertex, a, midpoint), 2, next_depth), ((b, newest_vertex, midpoint), 2, next_depth)

        def refine_triangle(triangle: Triangle) -> None:
            nonlocal max_depth
            if len(triangles) > max_triangles:
                return
            vertices, _newest, depth = triangle
            if not should_refine(triangle):
                max_depth = max(max_depth, depth)
                leaves.append(triangle)
                return
            left, right = bisect(triangle)
            refine_triangle(left)
            refine_triangle(right)

        base_triangles = triangles
        triangles = []
        leaves: list[Triangle] = []
        for triangle in base_triangles:
            refine_triangle(triangle)
            if len(leaves) > max_triangles:
                return {
                    "enabled": True,
                    "applied": False,
                    "tessellation_strategy": "Newest Vertex Bisection (NVB)",
                    "target_resolution_m": target,
                    "reason": f"NVB refinement exceeded safety limit of {max_triangles} triangles",
                }
        triangles = leaves

        z_cache: dict[tuple[float, float], tuple[float, float, float, float]] = {}

        def z_values(x: float, y: float) -> tuple[float, float, float, float]:
            key = (round(x, 6), round(y, 6))
            cached = z_cache.get(key)
            if cached is not None:
                return cached
            terrarium_z = self.bilinear_sample(terrarium, xs, ys, x, y)
            cloud_z = self.nearest_cloud_value(cloud, cloud_xs, cloud_ys, x, y)
            replacement_z = cloud_z if np.isfinite(cloud_z) else terrarium_z
            blur_z = self.blur_blend_value(x, y, terrarium_z, cloud, cloud_xs, cloud_ys, contour, blur_radius_m)
            if np.isfinite(cloud_z):
                blend_z = cloud_z
                vertical_blend_z = cloud_z
            else:
                nearest_z = self.nearest_grid_value(fixed_nearest_z, cloud_xs, cloud_ys, x, y)
                distance = self.nearest_grid_value(fixed_dist, cloud_xs, cloud_ys, x, y)
                if np.isfinite(nearest_z) and np.isfinite(distance) and distance <= radius:
                    t = max(0.0, min(distance / radius, 1.0))
                    weight = 0.5 * (1.0 + math.cos(math.pi * t))
                    blend_z = weight * nearest_z + (1.0 - weight) * terrarium_z
                    dz = abs(nearest_z - terrarium_z)
                    vt = max(0.0, min(dz / radius, 1.0))
                    vweight = 0.5 * (1.0 + math.cos(math.pi * vt))
                    vertical_blend_z = vweight * nearest_z + (1.0 - vweight) * terrarium_z
                else:
                    blend_z = terrarium_z
                    vertical_blend_z = terrarium_z
            cached = (replacement_z, blend_z, blur_z, vertical_blend_z)
            z_cache[key] = cached
            return cached

        replacement_triangles = []
        blend_triangles = []
        blur_triangles = []
        vertical_blend_triangles = []
        for vertices, _newest, _depth in triangles:
            zs = [z_values(p[0], p[1]) for p in vertices]
            xy = [round(vertices[0][0], 3), round(vertices[0][1], 3), round(vertices[1][0], 3), round(vertices[1][1], 3), round(vertices[2][0], 3), round(vertices[2][1], 3)]
            replacement_triangles.append(xy + [round(zs[0][0], 3), round(zs[1][0], 3), round(zs[2][0], 3)])
            blend_triangles.append(xy + [round(zs[0][1], 3), round(zs[1][1], 3), round(zs[2][1], 3)])
            blur_triangles.append(xy + [round(zs[0][2], 3), round(zs[1][2], 3), round(zs[2][2], 3)])
            vertical_blend_triangles.append(xy + [round(zs[0][3], 3), round(zs[1][3], 3), round(zs[2][3], 3)])

        return {
            "enabled": True,
            "applied": True,
            "tessellation_strategy": "Newest Vertex Bisection (NVB)",
            "geometry": "triangles",
            "target_resolution_m": target,
            "blend_radius_m": float(radius),
            "source": "fixed Cloud DEM visualization grid",
            "rule": "Bisect marked triangles across the edge opposite the newest vertex. This viewer implementation keeps the refinement local and bounded for interactive use.",
            "base_triangle_count": int((len(xs) - 1) * (len(ys) - 1) * 2),
            "triangle_count": len(triangles),
            "closure_refinements": 0,
            "max_depth": max_depth,
            "layers": {
                "cloud_replacement": {"triangles": replacement_triangles},
                "distance_blend": {"triangles": blend_triangles},
                "vertical_distance_blend": {"triangles": vertical_blend_triangles},
                "blur_blend": {"triangles": blur_triangles},
            },
        }

    def adaptive_diamond48_blend_mesh(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        terrarium: np.ndarray,
        resolution: float,
        radius: float,
        blur_radius_m: float,
    ) -> dict[str, Any]:
        target = max(float(self.blend_refinement_resolution_m), float(self.dem_pixel_size))
        label = "4-8 Diamond edge splits"
        if resolution <= target:
            return {
                "enabled": True,
                "applied": False,
                "tessellation_strategy": label,
                "target_resolution_m": target,
                "reason": "selected Terrarium mesh is already at or finer than the refinement target",
            }
        cloud_xs, cloud_ys, cloud = self.fixed_cloud_arrays()
        valid = np.isfinite(cloud)
        if not valid.any():
            return {
                "enabled": True,
                "applied": False,
                "tessellation_strategy": label,
                "target_resolution_m": target,
                "reason": "fixed Cloud DEM layer has no valid support",
            }

        valid_ascending_y = valid[::-1, :].astype(np.int32)
        prefix = valid_ascending_y.cumsum(axis=0).cumsum(axis=1)
        cloud_ys_ascending = cloud_ys[::-1]
        footprint_pad = target / 2.0
        fixed_dist, fixed_nearest_z = self.cloud_distance_field(cloud, cloud_xs, cloud_ys, radius)
        contour = self.cloud_contour_mask(cloud)

        Point2 = tuple[float, float]
        Triangle = tuple[tuple[Point2, Point2, Point2], int]

        def key_point(p: Point2) -> tuple[int, int]:
            return (int(round(p[0] * 1000.0)), int(round(p[1] * 1000.0)))

        def edge_key(a: Point2, b: Point2) -> tuple[tuple[int, int], tuple[int, int]]:
            ka, kb = key_point(a), key_point(b)
            return (ka, kb) if ka <= kb else (kb, ka)

        def prefix_sum(r0: int, c0: int, r1: int, c1: int) -> int:
            total = int(prefix[r1, c1])
            if r0 > 0:
                total -= int(prefix[r0 - 1, c1])
            if c0 > 0:
                total -= int(prefix[r1, c0 - 1])
            if r0 > 0 and c0 > 0:
                total += int(prefix[r0 - 1, c0 - 1])
            return total

        def contains_cloud_support_bbox(minx: float, miny: float, maxx: float, maxy: float, pad: float) -> bool:
            c0 = int(np.searchsorted(cloud_xs, minx - pad, side="left"))
            c1 = int(np.searchsorted(cloud_xs, maxx + pad, side="right")) - 1
            r0 = int(np.searchsorted(cloud_ys_ascending, miny - pad, side="left"))
            r1 = int(np.searchsorted(cloud_ys_ascending, maxy + pad, side="right")) - 1
            c0 = max(0, c0)
            r0 = max(0, r0)
            c1 = min(len(cloud_xs) - 1, c1)
            r1 = min(len(cloud_ys_ascending) - 1, r1)
            if c0 > c1 or r0 > r1:
                return False
            return prefix_sum(r0, c0, r1, c1) > 0

        def edge_length(a: Point2, b: Point2) -> float:
            return math.hypot(a[0] - b[0], a[1] - b[1])

        def triangle_has_cloud_support(vertices: tuple[Point2, Point2, Point2]) -> bool:
            minx = min(p[0] for p in vertices)
            maxx = max(p[0] for p in vertices)
            miny = min(p[1] for p in vertices)
            maxy = max(p[1] for p in vertices)
            return contains_cloud_support_bbox(minx, miny, maxx, maxy, footprint_pad)

        def longest_edge(vertices: tuple[Point2, Point2, Point2]) -> tuple[Point2, Point2]:
            edges = [(vertices[0], vertices[1]), (vertices[1], vertices[2]), (vertices[2], vertices[0])]
            return max(edges, key=lambda e: (edge_length(e[0], e[1]), edge_key(e[0], e[1])))

        def max_edge_length(vertices: tuple[Point2, Point2, Point2]) -> float:
            return max(edge_length(vertices[0], vertices[1]), edge_length(vertices[1], vertices[2]), edge_length(vertices[2], vertices[0]))

        triangles: list[Triangle] = []
        for row in range(len(ys) - 1):
            for col in range(len(xs) - 1):
                a = (float(xs[col]), float(ys[row]))
                b = (float(xs[col + 1]), float(ys[row]))
                c = (float(xs[col]), float(ys[row + 1]))
                d = (float(xs[col + 1]), float(ys[row + 1]))
                triangles.append(((a, c, b), 0))
                triangles.append(((b, c, d), 0))

        max_depth = 0
        split_rounds = 0
        max_triangles = 500000
        while True:
            split_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
            for vertices, _depth in triangles:
                if max_edge_length(vertices) > target and triangle_has_cloud_support(vertices):
                    a, b = longest_edge(vertices)
                    split_edges.add(edge_key(a, b))
            if not split_edges:
                break
            split_rounds += 1
            midpoint_cache: dict[tuple[tuple[int, int], tuple[int, int]], Point2] = {}

            def midpoint(a: Point2, b: Point2) -> Point2:
                key = edge_key(a, b)
                cached = midpoint_cache.get(key)
                if cached is not None:
                    return cached
                value = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
                midpoint_cache[key] = value
                return value

            next_triangles: list[Triangle] = []
            for vertices, depth in triangles:
                a, b, c = vertices
                split_ab = edge_key(a, b) in split_edges
                split_bc = edge_key(b, c) in split_edges
                split_ca = edge_key(c, a) in split_edges
                count = int(split_ab) + int(split_bc) + int(split_ca)
                if count == 0:
                    next_triangles.append((vertices, depth))
                    max_depth = max(max_depth, depth)
                    continue
                next_depth = depth + 1
                max_depth = max(max_depth, next_depth)
                if count == 1:
                    if split_ab:
                        m = midpoint(a, b)
                        next_triangles.extend([((a, m, c), next_depth), ((m, b, c), next_depth)])
                    elif split_bc:
                        m = midpoint(b, c)
                        next_triangles.extend([((b, m, a), next_depth), ((m, c, a), next_depth)])
                    else:
                        m = midpoint(c, a)
                        next_triangles.extend([((c, m, b), next_depth), ((m, a, b), next_depth)])
                elif count == 2:
                    if not split_ab:
                        m_bc = midpoint(b, c)
                        m_ca = midpoint(c, a)
                        next_triangles.extend([((c, m_ca, m_bc), next_depth), ((a, b, m_bc), next_depth), ((a, m_bc, m_ca), next_depth)])
                    elif not split_bc:
                        m_ca = midpoint(c, a)
                        m_ab = midpoint(a, b)
                        next_triangles.extend([((a, m_ab, m_ca), next_depth), ((b, c, m_ca), next_depth), ((b, m_ca, m_ab), next_depth)])
                    else:
                        m_ab = midpoint(a, b)
                        m_bc = midpoint(b, c)
                        next_triangles.extend([((b, m_bc, m_ab), next_depth), ((c, a, m_ab), next_depth), ((c, m_ab, m_bc), next_depth)])
                else:
                    m_ab = midpoint(a, b)
                    m_bc = midpoint(b, c)
                    m_ca = midpoint(c, a)
                    next_triangles.extend([
                        ((a, m_ab, m_ca), next_depth),
                        ((m_ab, b, m_bc), next_depth),
                        ((m_ca, m_bc, c), next_depth),
                        ((m_ab, m_bc, m_ca), next_depth),
                    ])
            triangles = next_triangles
            if len(triangles) > max_triangles:
                return {
                    "enabled": True,
                    "applied": False,
                    "tessellation_strategy": label,
                    "target_resolution_m": target,
                    "reason": f"4-8 diamond refinement exceeded safety limit of {max_triangles} triangles",
                }

        edge_counts: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
        for vertices, _depth in triangles:
            for a, b in ((vertices[0], vertices[1]), (vertices[1], vertices[2]), (vertices[2], vertices[0])):
                key = edge_key(a, b)
                edge_counts[key] = edge_counts.get(key, 0) + 1
        nonconforming_edges = sum(1 for count in edge_counts.values() if count > 2)

        z_cache: dict[tuple[float, float], tuple[float, float, float, float]] = {}

        def z_values(x: float, y: float) -> tuple[float, float, float, float]:
            key = (round(x, 6), round(y, 6))
            cached = z_cache.get(key)
            if cached is not None:
                return cached
            terrarium_z = self.bilinear_sample(terrarium, xs, ys, x, y)
            cloud_z = self.nearest_cloud_value(cloud, cloud_xs, cloud_ys, x, y)
            replacement_z = cloud_z if np.isfinite(cloud_z) else terrarium_z
            blur_z = self.blur_blend_value(x, y, terrarium_z, cloud, cloud_xs, cloud_ys, contour, blur_radius_m)
            if np.isfinite(cloud_z):
                blend_z = cloud_z
                vertical_blend_z = cloud_z
            else:
                nearest_z = self.nearest_grid_value(fixed_nearest_z, cloud_xs, cloud_ys, x, y)
                distance = self.nearest_grid_value(fixed_dist, cloud_xs, cloud_ys, x, y)
                if np.isfinite(nearest_z) and np.isfinite(distance) and distance <= radius:
                    t = max(0.0, min(distance / radius, 1.0))
                    weight = 0.5 * (1.0 + math.cos(math.pi * t))
                    blend_z = weight * nearest_z + (1.0 - weight) * terrarium_z
                    dz = abs(nearest_z - terrarium_z)
                    vt = max(0.0, min(dz / radius, 1.0))
                    vweight = 0.5 * (1.0 + math.cos(math.pi * vt))
                    vertical_blend_z = vweight * nearest_z + (1.0 - vweight) * terrarium_z
                else:
                    blend_z = terrarium_z
                    vertical_blend_z = terrarium_z
            cached = (replacement_z, blend_z, blur_z, vertical_blend_z)
            z_cache[key] = cached
            return cached

        replacement_triangles = []
        blend_triangles = []
        blur_triangles = []
        vertical_blend_triangles = []
        for vertices, _depth in triangles:
            zs = [z_values(p[0], p[1]) for p in vertices]
            xy = [round(vertices[0][0], 3), round(vertices[0][1], 3), round(vertices[1][0], 3), round(vertices[1][1], 3), round(vertices[2][0], 3), round(vertices[2][1], 3)]
            replacement_triangles.append(xy + [round(zs[0][0], 3), round(zs[1][0], 3), round(zs[2][0], 3)])
            blend_triangles.append(xy + [round(zs[0][1], 3), round(zs[1][1], 3), round(zs[2][1], 3)])
            blur_triangles.append(xy + [round(zs[0][2], 3), round(zs[1][2], 3), round(zs[2][2], 3)])
            vertical_blend_triangles.append(xy + [round(zs[0][3], 3), round(zs[1][3], 3), round(zs[2][3], 3)])

        return {
            "enabled": True,
            "applied": True,
            "tessellation_strategy": label,
            "geometry": "triangles",
            "target_resolution_m": target,
            "blend_radius_m": float(radius),
            "source": "fixed Cloud DEM visualization grid",
            "rule": "Adaptive edge-centered 4-8 style diamond refinement. Marked edges are split globally, and every incident triangle is re-triangulated with the exact same edge midpoint endpoints.",
            "base_triangle_count": int((len(xs) - 1) * (len(ys) - 1) * 2),
            "triangle_count": len(triangles),
            "split_rounds": split_rounds,
            "max_depth": max_depth,
            "nonconforming_edges": nonconforming_edges,
            "layers": {
                "cloud_replacement": {"triangles": replacement_triangles},
                "distance_blend": {"triangles": blend_triangles},
                "vertical_distance_blend": {"triangles": vertical_blend_triangles},
                "blur_blend": {"triangles": blur_triangles},
            },
        }

    def mesh(
        self,
        z: int,
        refine_blends: bool = False,
        tessellation_strategy: str = "quadtree",
        blur_radius_m: float | None = None,
        terrarium_bounds: tuple[float, float, float, float] | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        grid = self.terrarium_grid(z, terrarium_bounds)
        xs = grid["xs"]
        ys = grid["ys"]
        terrarium = grid["terrarium"]
        cloud = self.sample_cloud(xs, ys, float(grid["resolution"]))
        replacement = np.where(np.isfinite(cloud), cloud, terrarium)
        radius = 6.0 * float(grid["resolution"])
        blur_radius = self.default_blur_radius_m if blur_radius_m is None else max(0.0, float(blur_radius_m))
        blend = self.distance_blend(cloud, terrarium, xs, ys, radius)
        vertical_blend = self.vertical_distance_blend(cloud, terrarium, xs, ys, radius)
        blur = self.blur_blend(terrarium, xs, ys, blur_radius)
        payload = {
            "lod": z,
            "resolution_m": float(grid["resolution"]),
            "nx": int(len(xs)),
            "ny": int(len(ys)),
            "xs": np.round(xs, 3).tolist(),
            "ys": np.round(ys, 3).tolist(),
            "terrarium": np.round(terrarium.reshape(-1), 3).tolist(),
            "cloud_dem": [None if not np.isfinite(v) else round(float(v), 3) for v in cloud.reshape(-1)],
            "cloud_replacement": np.round(replacement.reshape(-1), 3).tolist(),
            "distance_blend": np.round(blend.reshape(-1), 3).tolist(),
            "vertical_distance_blend": np.round(vertical_blend.reshape(-1), 3).tolist(),
            "blur_blend": np.round(blur.reshape(-1), 3).tolist(),
            "cloud_layer": self.fixed_cloud_layer(),
            "cloud_valid_count": int(np.isfinite(cloud).sum()),
            "blend_radius_m": radius,
            "blur_radius_m": blur_radius,
            "blend_formula": "Horizontal Distance: If Cloud DEM is defined, z=cloud_z. Otherwise t=clamp(d_horizontal/R,0,1), w=0.5*(1+cos(pi*t)), z=w*z_nearest_cloud+(1-w)*z_terrarium.",
            "vertical_blend_formula": "Vertical Distance: If Cloud DEM is defined, z=cloud_z. Otherwise use the horizontally nearest valid Cloud DEM value, t=clamp(abs(z_nearest_cloud-z_terrarium)/R,0,1), w=0.5*(1+cos(pi*t)), z=w*z_nearest_cloud+(1-w)*z_terrarium.",
            "blur_formula": "If Cloud DEM is defined, z=cloud_z. Otherwise collect valid Cloud DEM contour points within R meters. For each contour point, W=0.5*(1+cos(pi*d_m/R)). Then w_avg=mean(W), z_contour_avg=sum(W*z_contour)/sum(W), and z=w_avg*z_contour_avg+(1-w_avg)*z_terrarium.",
            "tiles": grid["tiles"],
            "bbox": {
                "terrarium_dem": self.bbox_dict(grid["bounds"]),
                "terrarium_requested": self.bbox_dict(grid["requested_bounds"]),
                "point_cloud": self.bbox_dict(self.bounds) if self.bounds is not None else None,
            },
            "cloud_loaded": bool(self.cloud_loaded),
            "cloud_source": self.cloud_source,
            "point_sample": self.point_sample,
            "elapsed_seconds": round(time.perf_counter() - start, 3),
        }
        if refine_blends:
            payload["refined_mesh"] = self.adaptive_blend_mesh(
                xs, ys, terrarium, float(grid["resolution"]), radius, blur_radius, tessellation_strategy
            )
            payload["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        return payload


class Handler(BaseHTTPRequestHandler):
    context: TerrainContext

    def file_response(self, path: Path, content_type: str) -> None:
        encoded = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def json_response(self, payload: Any) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self.file_response(HTML_PATH, "text/html; charset=utf-8")
            return
        if parsed.path == "/lod_viewer.css":
            self.file_response(CSS_PATH, "text/css; charset=utf-8")
            return
        if parsed.path == "/lod_viewer.js":
            self.file_response(JS_PATH, "application/javascript; charset=utf-8")
            return
        if parsed.path == "/api/info":
            lods = self.context.lod_info()
            self.json_response(
                {
                    "lods": lods,
                    "default_lod": 10,
                    "cloud_dem": {
                        "resolution_m": self.context.cloud_resolution_m,
                        "description": "Cloud DEM visualization resolution is fixed and independent of Terrarium LoD.",
                    },
                    "default_blur_radius_m": self.context.default_blur_radius_m,
                    "cloud_loaded": self.context.cloud_loaded,
                    "cloud_bbox": self.context.bbox_dict(self.context.bounds) if self.context.bounds is not None else None,
                    "cloud_source": self.context.cloud_source,
                }
            )
            return
        if parsed.path == "/api/cloud_dem":
            query = urllib.parse.parse_qs(parsed.query)
            las_dir_text = query.get("las_dir", [""])[0].strip()
            if not las_dir_text:
                self.send_error(400, "las_dir is required")
                return
            source_crs = query.get("source_crs", [self.context.report.get("crs", {}).get("source", "EPSG:32630")])[0]
            try:
                pixel_size_m = float(query.get("pixel_size_m", [str(self.context.dem_pixel_size)])[0])
                chunk_size = int(query.get("chunk_size", ["1000000"])[0])
                payload = self.context.load_cloud_dem_from_las(Path(las_dir_text), source_crs, pixel_size_m, chunk_size)
            except Exception as exc:
                self.send_error(400, str(exc))
                return
            self.json_response(payload)
            return
        if parsed.path == "/api/mesh":
            query = urllib.parse.parse_qs(parsed.query)
            try:
                z = int(query.get("z", ["10"])[0])
            except ValueError:
                self.send_error(400, "LoD must be an integer")
                return
            if z < 0 or z > self.context.max_lod:
                self.send_error(400, f"LoD must be between 0 and {self.context.max_lod}")
                return
            refine_blends = query.get("refine", ["0"])[0].lower() in {"1", "true", "yes", "on"}
            tessellation = query.get("tessellation", ["quadtree"])[0].lower()
            if tessellation == "none":
                tessellation = "quadtree"
                refine_blends = False
            if tessellation not in {"quadtree", "nvb", "diamond48"}:
                self.send_error(400, "tessellation must be 'none', 'quadtree', 'nvb', or 'diamond48'")
                return
            try:
                blur_radius_m = float(query.get("blur_radius_m", [str(self.context.default_blur_radius_m)])[0])
            except ValueError:
                self.send_error(400, "blur_radius_m must be a number")
                return
            terrarium_bounds = None
            bbox_keys = ("bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy")
            if any(key in query for key in bbox_keys):
                if not all(key in query for key in bbox_keys):
                    self.send_error(400, "Terrarium BBOX requires bbox_minx, bbox_miny, bbox_maxx, and bbox_maxy")
                    return
                try:
                    minx = float(query["bbox_minx"][0])
                    miny = float(query["bbox_miny"][0])
                    maxx = float(query["bbox_maxx"][0])
                    maxy = float(query["bbox_maxy"][0])
                except ValueError:
                    self.send_error(400, "Terrarium BBOX values must be numbers")
                    return
                if not all(math.isfinite(v) for v in (minx, miny, maxx, maxy)):
                    self.send_error(400, "Terrarium BBOX values must be finite")
                    return
                if minx >= maxx or miny >= maxy:
                    self.send_error(400, "Terrarium BBOX min values must be lower than max values")
                    return
                if (
                    minx < -WEB_MERCATOR_HALF_WORLD
                    or maxx > WEB_MERCATOR_HALF_WORLD
                    or miny < -WEB_MERCATOR_HALF_WORLD
                    or maxy > WEB_MERCATOR_HALF_WORLD
                ):
                    self.send_error(400, "Terrarium BBOX must be inside EPSG:3857 Web Mercator bounds")
                    return
                terrarium_bounds = (minx, miny, maxx, maxy)
            self.json_response(
                self.context.mesh(
                    z,
                    refine_blends=refine_blends,
                    tessellation_strategy=tessellation,
                    blur_radius_m=blur_radius_m,
                    terrarium_bounds=terrarium_bounds,
                )
            )
            return
        self.send_error(404)

    def log_message(self, format: str, *args: Any) -> None:
        print("%s - %s" % (self.address_string(), format % args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve an interactive LoD Terrarium terrain viewer.")
    parser.add_argument("--report", default="outputs/dem_terrarium_experiment/experiment_report.json")
    parser.add_argument("--cache-dir", default="outputs/dem_terrarium_experiment/terrarium_lod_cache")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-lod", type=int, default=15)
    parser.add_argument(
        "--cloud-resolution-m",
        type=float,
        default=None,
        help="Fixed Cloud DEM visualization grid resolution in meters. Default uses report viewer stride * DEM pixel size.",
    )
    parser.add_argument(
        "--blend-refinement-resolution-m",
        type=float,
        default=None,
        help="Target edge length for optional blend mesh refinement. Default matches --cloud-resolution-m.",
    )
    parser.add_argument(
        "--blur-radius-m",
        type=float,
        default=None,
        help="Default blur-blend neighborhood radius in CRS meters. Default is six Cloud DEM visualization cells.",
    )
    parser.add_argument("--open", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    context = TerrainContext(
        Path(args.report),
        Path(args.cache_dir),
        args.max_lod,
        args.cloud_resolution_m,
        args.blend_refinement_resolution_m,
        args.blur_radius_m,
    )
    Handler.context = context
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(url)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

