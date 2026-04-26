# dem-lab

Laboratorio de investigacion para generar, comparar y visualizar modelos de elevacion derivados de nubes de puntos LAS y teselas Terrarium.

## Objetivo

El repositorio contiene scripts reproducibles para:

- Rasterizar archivos LAS como un DEM de minima elevacion por celda.
- Comparar el DEM local contra elevacion Terrarium en `EPSG:3857`.
- Registrar cada ejecucion con reportes Markdown y JSON.
- Preparar visores WebGL para inspeccionar mallas, mezclas y niveles de detalle.

## Supuestos geoespaciales

- CRS de entrada por defecto: `EPSG:32628`, usado solo cuando los encabezados LAS no incluyen CRS.
- CRS de salida por defecto: `EPSG:3857`.
- Resolucion DEM por defecto: `0.5 m`.
- Regla de rasterizacion: minimo `z` de los puntos que caen dentro de cada celda.
- Mascara de validez: celdas con valores finitos en el DEM de nube.
- Comparacion contra Terrarium: muestreo bilineal de Terrarium en el centro de cada pixel valido del DEM local.
- Diferencia reportada: `cloud_min_z_m - terrarium_elevation_m`.

## Estructura

```text
scripts/
  build_dem_terrarium_experiment.py  # experimento principal y registro de artefactos
  viewer_wiremesh.py                 # visor WebGL estatico desde viewer_meshes.json
  lod_terrarium_viewer.py            # servidor local con LoD, mezclas y refinamiento
launch_lod_viewer.bat                # lanzador Windows del visor LoD
outputs/                             # salidas generadas, ignoradas por git
```

Los datos originales y artefactos pesados estan excluidos de git mediante `.gitignore`. Las salidas deben escribirse en carpetas de resultados como `outputs/`, `results/` o `runs/`.

## Dependencias

Python 3.10+ recomendado.

```powershell
python -m pip install laspy numpy requests pillow pyproj rasterio
```

`lod_terrarium_viewer.py` y `build_dem_terrarium_experiment.py` descargan teselas Terrarium desde `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/`, por lo que requieren acceso de red durante esas etapas.

## Experimento DEM vs Terrarium

Ejemplo con valores por defecto:

```powershell
python scripts\build_dem_terrarium_experiment.py
```

Ejemplo declarando entradas y salidas:

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

Salidas principales:

- `cloud_minz_3857_0p5m.float32.memmap`: raster temporal de minimo `z`.
- `cloud_minz_3857_0p5m.tif`: DEM local en GeoTIFF.
- `cloud_minz_3857_0p5m_preview.png`: previsualizacion raster.
- `terrarium_covering_tile.png`: tesela Terrarium descargada.
- `terrarium_covering_tile_3857.tif`: Terrarium decodificado como GeoTIFF.
- `viewer_meshes.json`: mallas simplificadas para visualizacion.
- `experiment_report.md`: reporte trazable del experimento.
- `experiment_report.json`: registro estructurado de parametros, tiempos, errores, rutas y metricas.

Para recomputar salidas existentes:

```powershell
python scripts\build_dem_terrarium_experiment.py --force
```

## Visores

Generar un visor WebGL estatico desde `viewer_meshes.json`:

```powershell
python scripts\viewer_wiremesh.py `
  --mesh outputs\dem_terrarium_experiment\viewer_meshes.json `
  --out outputs\dem_terrarium_experiment\wiremesh_viewer.html `
  --open
```

Ejecutar el visor LoD local:

```powershell
python scripts\lod_terrarium_viewer.py `
  --report outputs\dem_terrarium_experiment\experiment_report.json `
  --cache-dir outputs\dem_terrarium_experiment\terrarium_lod_cache `
  --host 127.0.0.1 `
  --port 8765 `
  --max-lod 15 `
  --open
```

En Windows tambien se puede usar:

```powershell
.\launch_lod_viewer.bat 8765
```

## Trazabilidad requerida

Cada experimento debe conservar:

- Objetivo, supuestos, entradas, salidas y parametros.
- CRS de entrada detectado o asumido y CRS de salida.
- Resolucion, regla de agregacion y mascara de validez.
- Numero de muestras, formula de metricas y rutas de artefactos.
- Tiempos de ejecucion, advertencias, errores y versiones de dependencias cuando sean relevantes.

No se procesan subcarpetas salvo que el experimento lo declare de forma explicita. No se modifican datos originales; los scripts deben escribir resultados nuevos dentro del workspace.
