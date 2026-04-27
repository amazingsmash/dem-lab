from __future__ import annotations

import argparse
import html
import json
import webbrowser
from pathlib import Path

import numpy as np


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DEM vs Terrarium Viewer</title>
<style>
html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; background: #111; color: #eee; font-family: Segoe UI, Arial, sans-serif; }
#gl { width: 100vw; height: 100vh; display: block; }
#hud { position: fixed; left: 12px; top: 12px; width: 260px; background: rgba(0,0,0,.76); padding: 10px 12px; border: 1px solid #444; border-radius: 6px; font-size: 13px; line-height: 1.45; }
#hud label { display: flex; align-items: center; gap: 8px; margin: 5px 0; }
#hud select { width: 100%; margin: 6px 0 8px; color: #eee; background: #1e1e1e; border: 1px solid #555; border-radius: 4px; padding: 5px 7px; }
#hud input[type=range] { width: 100%; }
#hud details { margin: 8px 0; }
#hud summary { cursor: pointer; user-select: none; }
.row { display: grid; grid-template-columns: 72px 1fr 38px; gap: 8px; align-items: center; margin: 6px 0; }
.swatch { width: 12px; height: 12px; border-radius: 2px; display: inline-block; }
.red { background: #ff4a4a; } .blue { background: #4aa3ff; } .green { background: #52d273; } .yellow { background: #f2c94c; } .cyan { background: #4fd7d7; }
.hint { color: #bbb; margin-top: 8px; }
</style>
</head>
<body>
<canvas id="gl"></canvas>
<div id="hud">
  <label for="mode">Render</label>
  <select id="mode">
    <option value="wire">Wireframe</option>
    <option value="flat">Flat shading</option>
    <option value="phong">Phong shading</option>
  </select>
  <details open>
    <summary>Layers</summary>
    <label><input id="showTerrarium" type="checkbox" checked><span class="swatch blue"></span>Terrarium</label>
    <label><input id="showCloud" type="checkbox" checked><span class="swatch red"></span>Cloud DEM</label>
    <label><input id="showNaive" type="checkbox"><span class="swatch green"></span>Naive Combination</label>
    <label><input id="showDistanceBlend" type="checkbox"><span class="swatch yellow"></span>Horizontal Distance</label>
    <label><input id="showVerticalDistanceBlend" type="checkbox"><span class="swatch yellow"></span>Vertical Distance</label>
    <label><input id="showBlurBlend" type="checkbox"><span class="swatch cyan"></span>Blur Blend</label>
  </details>
  <details open>
    <summary>Sun</summary>
    <div class="row"><span>Azimuth</span><input id="sunAzimuth" type="range" min="0" max="360" value="315"><span id="sunAzimuthValue"></span></div>
    <div class="row"><span>Elevation</span><input id="sunElevation" type="range" min="1" max="89" value="45"><span id="sunElevationValue"></span></div>
    <div class="row"><span>Ambient</span><input id="ambient" type="range" min="0" max="80" value="28"><span id="ambientValue"></span></div>
    <div class="row"><span>Specular</span><input id="specular" type="range" min="0" max="80" value="18"><span id="specularValue"></span></div>
  </details>
  <div id="meta"></div>
  <div class="hint">drag: rotate | wheel: zoom | shift+drag: pan</div>
</div>
<script id="mesh-data" type="application/json">__MESH_JSON__</script>
<script>
const data = JSON.parse(document.getElementById("mesh-data").textContent);
const canvas = document.getElementById("gl");
const gl = canvas.getContext("webgl", {antialias: true});
if (!gl) throw new Error("WebGL unavailable");

const xs = data.xs, ys = data.ys, nx = data.metadata.nx, ny = data.metadata.ny;
const naiveZ = data.terrarium_z.map((z, i) => data.cloud_z[i] === null ? z : data.cloud_z[i]);
const distanceBlendZ = data.distance_blend_z;
const verticalDistanceBlendZ = data.vertical_distance_blend_z;
const blurBlendZ = data.blur_blend_z;
document.getElementById("meta").textContent = `${nx} x ${ny}, stride ${data.metadata.stride}`;

const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
const cloudVals = data.cloud_z.filter(v => v !== null);
const allZ = cloudVals.concat(data.terrarium_z);
const cz = (Math.min(...allZ) + Math.max(...allZ)) / 2;
const extent = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 1);
const zScale = 3.0;

function shader(type, source) {
  const s = gl.createShader(type);
  gl.shaderSource(s, source);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

function program(vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, shader(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, shader(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

const wireProgram = program(`
attribute vec3 p;
uniform mat4 mvp;
void main() { gl_Position = mvp * vec4(p, 1.0); }
`, `
precision mediump float;
uniform vec3 color;
void main() { gl_FragColor = vec4(color, 1.0); }
`);

const phongProgram = program(`
attribute vec3 p;
attribute vec3 n;
uniform mat4 mvp;
varying vec3 vn;
varying vec3 vp;
void main() {
  vn = normalize(n);
  vp = p;
  gl_Position = mvp * vec4(p, 1.0);
}
`, `
precision mediump float;
uniform vec3 color;
uniform vec3 lightDir;
uniform float ambient;
uniform float specularStrength;
varying vec3 vn;
varying vec3 vp;
void main() {
  vec3 N = normalize(vn);
  vec3 L = normalize(lightDir);
  vec3 V = normalize(vec3(0.0, 0.0, 1.0) - vp);
  vec3 R = reflect(-L, N);
  // One-sided Lambert lighting: back-facing normals receive no direct sun.
  float diffuse = max(dot(N, L), 0.0);
  float specular = pow(max(dot(R, V), 0.0), 24.0) * specularStrength;
  vec3 shaded = color * (ambient + (1.0 - ambient) * diffuse) + vec3(specular);
  gl_FragColor = vec4(shaded, 1.0);
}
`);

function valid(zs, i) { return zs[i] !== null && Number.isFinite(zs[i]); }
function point(zs, i) {
  const x = i % nx;
  const y = Math.floor(i / nx);
  return [(xs[x] - cx) / extent, (zs[i] - cz) * zScale / extent, (ys[y] - cy) / extent];
}

function buildLines(zs) {
  const lines = [];
  const edges = new Set();
  function push(a, b) {
    const lo = Math.min(a, b), hi = Math.max(a, b);
    const key = `${lo}:${hi}`;
    if (edges.has(key)) return;
    edges.add(key);
    lines.push(...point(zs, a), ...point(zs, b));
  }
  function tri(a, b, c) {
    if (!valid(zs, a) || !valid(zs, b) || !valid(zs, c)) return;
    push(a, b); push(b, c); push(c, a);
  }
  for (let y = 0; y < ny - 1; y++) {
    for (let x = 0; x < nx - 1; x++) {
      const a = y * nx + x, b = a + 1, c = a + nx, d = c + 1;
      tri(a, c, b);
      tri(b, c, d);
    }
  }
  return new Float32Array(lines);
}

function cross(a, b) {
  return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
}

function add(a, b) { a[0] += b[0]; a[1] += b[1]; a[2] += b[2]; }
function sub(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function norm(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0]/l, v[1]/l, v[2]/l];
}
function pointKey(p) { return `${p[0].toFixed(7)}:${p[1].toFixed(7)}:${p[2].toFixed(7)}`; }

function buildTriangles(zs) {
  const points = new Array(nx * ny);
  const smoothNormals = new Array(nx * ny);
  for (let i = 0; i < points.length; i++) {
    points[i] = valid(zs, i) ? point(zs, i) : null;
    smoothNormals[i] = [0, 0, 0];
  }
  const tris = [];
  const faceNormals = [];
  function tri(a, b, c) {
    if (!points[a] || !points[b] || !points[c]) return;
    let n = cross(sub(points[b], points[a]), sub(points[c], points[a]));
    // Terrain normals must point upward in object space (+Y). Some grid
    // windings produce downward normals, which makes one-sided lighting dark.
    if (n[1] < 0) n = [-n[0], -n[1], -n[2]];
    add(smoothNormals[a], n); add(smoothNormals[b], n); add(smoothNormals[c], n);
    tris.push(a, b, c);
    faceNormals.push(norm(n));
  }
  for (let y = 0; y < ny - 1; y++) {
    for (let x = 0; x < nx - 1; x++) {
      const a = y * nx + x, b = a + 1, c = a + nx, d = c + 1;
      tri(a, c, b);
      tri(b, c, d);
    }
  }
  const vertices = [];
  const normalData = [];
  const flatNormalData = [];
  for (let t = 0; t < tris.length; t += 3) {
    const face = faceNormals[Math.floor(t / 3)];
    for (const i of [tris[t], tris[t + 1], tris[t + 2]]) {
      vertices.push(...points[i]);
      normalData.push(...norm(smoothNormals[i]));
      flatNormalData.push(...face);
    }
  }
  return {positions: new Float32Array(vertices), normals: new Float32Array(normalData), flatNormals: new Float32Array(flatNormalData)};
}

function buffer(data) {
  const b = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, b);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  return b;
}

function makeMesh(zs) {
  const lines = buildLines(zs);
  const tris = buildTriangles(zs);
  return {
    lineBuffer: buffer(lines),
    lineCount: lines.length / 3,
    triBuffer: buffer(tris.positions),
    normalBuffer: buffer(tris.normals),
    flatNormalBuffer: buffer(tris.flatNormals),
    triCount: tris.positions.length / 3
  };
}

const meshes = {
  terrarium: {mesh: makeMesh(data.terrarium_z), color: [0.2, 0.55, 1.0], control: "showTerrarium"},
  cloud: {mesh: makeMesh(data.cloud_z), color: [1.0, 0.18, 0.15], control: "showCloud"},
  naive: {mesh: makeMesh(naiveZ), color: [0.28, 0.78, 0.43], control: "showNaive"},
  distanceBlend: {mesh: makeMesh(distanceBlendZ), color: [0.95, 0.70, 0.18], control: "showDistanceBlend"},
  verticalDistanceBlend: {mesh: makeMesh(verticalDistanceBlendZ), color: [0.95, 0.88, 0.25], control: "showVerticalDistanceBlend"},
  blurBlend: {mesh: makeMesh(blurBlendZ), color: [0.25, 0.84, 0.84], control: "showBlurBlend"}
};

function sliderNumber(id) {
  return Number(document.getElementById(id).value);
}

function updateSliderLabels() {
  document.getElementById("sunAzimuthValue").textContent = `${sliderNumber("sunAzimuth")}deg`;
  document.getElementById("sunElevationValue").textContent = `${sliderNumber("sunElevation")}deg`;
  document.getElementById("ambientValue").textContent = (sliderNumber("ambient") / 100).toFixed(2);
  document.getElementById("specularValue").textContent = (sliderNumber("specular") / 100).toFixed(2);
}

function sunDirection() {
  const az = sliderNumber("sunAzimuth") * Math.PI / 180;
  const el = sliderNumber("sunElevation") * Math.PI / 180;
  return [
    Math.sin(az) * Math.cos(el),
    Math.sin(el),
    Math.cos(az) * Math.cos(el)
  ];
}

for (const id of ["sunAzimuth", "sunElevation", "ambient", "specular"]) {
  document.getElementById(id).addEventListener("input", updateSliderLabels);
}
updateSliderLabels();

let yaw = -0.7, pitch = 0.8, dist = 2.1, panX = 0, panY = 0;
let dragging = false, lastX = 0, lastY = 0, panning = false;
canvas.addEventListener("mousedown", e => { dragging = true; panning = e.shiftKey; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener("mouseup", () => dragging = false);
window.addEventListener("mousemove", e => {
  if (!dragging) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  if (panning) { panX += dx / canvas.clientWidth * dist; panY -= dy / canvas.clientHeight * dist; }
  else { yaw += dx * 0.008; pitch = Math.max(-1.45, Math.min(1.45, pitch + dy * 0.008)); }
});
canvas.addEventListener("wheel", e => { e.preventDefault(); dist *= Math.exp(e.deltaY * 0.001); dist = Math.max(0.4, Math.min(12, dist)); }, {passive:false});

function resize() {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.floor(canvas.clientWidth * dpr), h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; gl.viewport(0, 0, w, h); }
}

function matMul(a,b){const r=new Float32Array(16);for(let c=0;c<4;c++)for(let rr=0;rr<4;rr++)r[c*4+rr]=a[0*4+rr]*b[c*4+0]+a[1*4+rr]*b[c*4+1]+a[2*4+rr]*b[c*4+2]+a[3*4+rr]*b[c*4+3];return r;}
function perspective(fovy, aspect, near, far){const f=1/Math.tan(fovy/2), nf=1/(near-far);return new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(far+near)*nf,-1,0,0,2*far*near*nf,0]);}
function lookAt(eye, center, up){
  let zx=eye[0]-center[0], zy=eye[1]-center[1], zz=eye[2]-center[2]; let zl=Math.hypot(zx,zy,zz); zx/=zl; zy/=zl; zz/=zl;
  let xx=up[1]*zz-up[2]*zy, xy=up[2]*zx-up[0]*zz, xz=up[0]*zy-up[1]*zx; let xl=Math.hypot(xx,xy,xz); xx/=xl; xy/=xl; xz/=xl;
  let yx=zy*xz-zz*xy, yy=zz*xx-zx*xz, yz=zx*xy-zy*xx;
  return new Float32Array([xx,yx,zx,0, xy,yy,zy,0, xz,yz,zz,0, -(xx*eye[0]+xy*eye[1]+xz*eye[2]), -(yx*eye[0]+yy*eye[1]+yz*eye[2]), -(zx*eye[0]+zy*eye[1]+zz*eye[2]), 1]);
}

function selectedLayers() {
  return Object.values(meshes).filter(layer => document.getElementById(layer.control).checked);
}

function drawWire(layer, mvp) {
  gl.useProgram(wireProgram);
  gl.bindBuffer(gl.ARRAY_BUFFER, layer.mesh.lineBuffer);
  const pLoc = gl.getAttribLocation(wireProgram, "p");
  gl.vertexAttribPointer(pLoc, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(pLoc);
  gl.uniformMatrix4fv(gl.getUniformLocation(wireProgram, "mvp"), false, mvp);
  gl.uniform3fv(gl.getUniformLocation(wireProgram, "color"), new Float32Array(layer.color));
  gl.drawArrays(gl.LINES, 0, layer.mesh.lineCount);
}

function drawShaded(layer, mvp, normalBuffer) {
  gl.useProgram(phongProgram);
  const pLoc = gl.getAttribLocation(phongProgram, "p");
  const nLoc = gl.getAttribLocation(phongProgram, "n");
  gl.bindBuffer(gl.ARRAY_BUFFER, layer.mesh.triBuffer);
  gl.vertexAttribPointer(pLoc, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(pLoc);
  gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
  gl.vertexAttribPointer(nLoc, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(nLoc);
  gl.uniformMatrix4fv(gl.getUniformLocation(phongProgram, "mvp"), false, mvp);
  gl.uniform3fv(gl.getUniformLocation(phongProgram, "color"), new Float32Array(layer.color));
  gl.uniform3fv(gl.getUniformLocation(phongProgram, "lightDir"), new Float32Array(sunDirection()));
  gl.uniform1f(gl.getUniformLocation(phongProgram, "ambient"), sliderNumber("ambient") / 100);
  gl.uniform1f(gl.getUniformLocation(phongProgram, "specularStrength"), sliderNumber("specular") / 100);
  gl.drawArrays(gl.TRIANGLES, 0, layer.mesh.triCount);
}
function drawFlat(layer, mvp) { drawShaded(layer, mvp, layer.mesh.flatNormalBuffer); }
function drawPhong(layer, mvp) { drawShaded(layer, mvp, layer.mesh.normalBuffer); }

function render() {
  resize();
  gl.clearColor(0.065, 0.065, 0.07, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  const aspect = canvas.width / Math.max(canvas.height, 1);
  const eye = [Math.sin(yaw)*Math.cos(pitch)*dist + panX, Math.sin(pitch)*dist + panY, Math.cos(yaw)*Math.cos(pitch)*dist];
  const center = [panX, panY, 0];
  const mvp = matMul(perspective(Math.PI/4, aspect, 0.01, 100), lookAt(eye, center, [0,1,0]));
  const mode = document.getElementById("mode").value;
  for (const layer of selectedLayers()) {
    if (mode === "phong") drawPhong(layer, mvp);
    else if (mode === "flat") drawFlat(layer, mvp);
    else drawWire(layer, mvp);
  }
  requestAnimationFrame(render);
}
render();
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an interactive WebGL viewer from viewer_meshes.json.")
    parser.add_argument("--mesh", default="outputs/dem_terrarium_experiment/viewer_meshes.json")
    parser.add_argument("--out", default=None)
    parser.add_argument("--open", action="store_true")
    parser.add_argument(
        "--blend-radius-m",
        type=float,
        default=None,
        help="Horizontal/vertical distance-blend influence radius in mesh CRS meters. Default is six mesh cells.",
    )
    return parser.parse_args()


def compute_distance_blend(mesh_json: dict, radius_m: float | None) -> dict:
    xs = np.asarray(mesh_json["xs"], dtype=np.float64)
    ys = np.asarray(mesh_json["ys"], dtype=np.float64)
    nx = int(mesh_json["metadata"]["nx"])
    ny = int(mesh_json["metadata"]["ny"])
    cloud = np.asarray([np.nan if value is None else value for value in mesh_json["cloud_z"]], dtype=np.float64)
    terrarium = np.asarray(mesh_json["terrarium_z"], dtype=np.float64)

    xx, yy = np.meshgrid(xs, ys)
    all_xy = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    valid = np.isfinite(cloud)
    if not valid.any():
        mesh_json["distance_blend_z"] = np.round(terrarium, 3).tolist()
        mesh_json["vertical_distance_blend_z"] = np.round(terrarium, 3).tolist()
        mesh_json["metadata"]["distance_blend"] = {
            "status": "no_cloud_values",
            "formula": "Fallback to Terrarium because cloud_z has no valid vertices.",
        }
        mesh_json["metadata"]["vertical_distance_blend"] = {
            "status": "no_cloud_values",
            "formula": "Fallback to Terrarium because cloud_z has no valid vertices.",
        }
        return mesh_json

    dx = float(np.median(np.diff(xs))) if nx > 1 else 1.0
    dy = float(abs(np.median(np.diff(ys)))) if ny > 1 else dx
    radius = float(radius_m) if radius_m is not None else 6.0 * max(dx, dy)
    radius = max(radius, max(dx, dy))

    valid_xy = all_xy[valid]
    valid_z = cloud[valid]
    nearest_distance = np.empty(all_xy.shape[0], dtype=np.float64)
    nearest_z = np.empty(all_xy.shape[0], dtype=np.float64)

    # Exact nearest valid cloud vertex, chunked to keep peak memory bounded.
    chunk = 2048
    for start in range(0, all_xy.shape[0], chunk):
        stop = min(start + chunk, all_xy.shape[0])
        delta = all_xy[start:stop, None, :] - valid_xy[None, :, :]
        dist2 = np.sum(delta * delta, axis=2)
        nearest = np.argmin(dist2, axis=1)
        nearest_distance[start:stop] = np.sqrt(dist2[np.arange(stop - start), nearest])
        nearest_z[start:stop] = valid_z[nearest]

    blended = terrarium.copy()
    invalid = ~valid
    t = np.clip(nearest_distance[invalid] / radius, 0.0, 1.0)
    weights = 0.5 * (1.0 + np.cos(np.pi * t))
    blended[invalid] = weights * nearest_z[invalid] + (1.0 - weights) * terrarium[invalid]
    blended[valid] = cloud[valid]

    mesh_json["distance_blend_z"] = np.round(blended, 3).tolist()
    vertical_blended = terrarium.copy()
    dz = np.abs(nearest_z[invalid] - terrarium[invalid])
    vertical_weights = 0.5 * (1.0 + np.cos(np.pi * np.clip(dz / radius, 0.0, 1.0)))
    vertical_blended[invalid] = vertical_weights * nearest_z[invalid] + (1.0 - vertical_weights) * terrarium[invalid]
    vertical_blended[valid] = cloud[valid]
    mesh_json["vertical_distance_blend_z"] = np.round(vertical_blended, 3).tolist()
    mesh_json["metadata"]["distance_blend"] = {
        "status": "ok",
        "radius_m": radius,
        "grid_spacing_x_m": dx,
        "grid_spacing_y_m": dy,
        "nearest_cloud_vertices": int(valid.sum()),
        "formula": (
            "If cloud_z is defined, z=cloud_z. Otherwise d is the horizontal distance to the nearest valid cloud mesh "
            "vertex, t=clamp(d/R,0,1), w=0.5*(1+cos(pi*t)), and z=w*z_nearest_cloud+(1-w)*z_terrarium."
        ),
    }
    mesh_json["metadata"]["vertical_distance_blend"] = {
        "status": "ok",
        "radius_m": radius,
        "formula": (
            "If cloud_z is defined, z=cloud_z. Otherwise use the horizontally nearest valid cloud mesh vertex, "
            "t=clamp(abs(z_nearest_cloud-z_terrarium)/R,0,1), w=0.5*(1+cos(pi*t)), "
            "and z=w*z_nearest_cloud+(1-w)*z_terrarium."
        ),
    }
    return mesh_json


def cloud_contour_mask(cloud: np.ndarray, nx: int, ny: int) -> np.ndarray:
    valid = np.isfinite(cloud).reshape(ny, nx)
    contour = np.zeros((ny, nx), dtype=bool)
    if not valid.any():
        return contour
    for r, c in np.argwhere(valid):
        if r == 0 or r == ny - 1 or c == 0 or c == nx - 1:
            contour[r, c] = True
            continue
        if not (valid[r - 1, c] and valid[r + 1, c] and valid[r, c - 1] and valid[r, c + 1]):
            contour[r, c] = True
    return contour


def compute_blur_blend(mesh_json: dict, radius_m: float | None) -> dict:
    xs = np.asarray(mesh_json["xs"], dtype=np.float64)
    ys = np.asarray(mesh_json["ys"], dtype=np.float64)
    nx = int(mesh_json["metadata"]["nx"])
    ny = int(mesh_json["metadata"]["ny"])
    cloud = np.asarray([np.nan if value is None else value for value in mesh_json["cloud_z"]], dtype=np.float64)
    terrarium = np.asarray(mesh_json["terrarium_z"], dtype=np.float64)
    dx = float(np.median(np.diff(xs))) if nx > 1 else 1.0
    dy = float(abs(np.median(np.diff(ys)))) if ny > 1 else dx
    radius = float(radius_m) if radius_m is not None else 6.0 * max(dx, dy)
    row_radius = max(1, int(np.ceil(radius / dy))) if radius > 0.0 else 0
    col_radius = max(1, int(np.ceil(radius / dx))) if radius > 0.0 else 0
    contour = cloud_contour_mask(cloud, nx, ny)
    cloud_grid = cloud.reshape(ny, nx)
    blended = terrarium.reshape(ny, nx).copy()
    if contour.any() and radius > 0.0:
        for row in range(ny):
            for col in range(nx):
                r0 = max(0, row - row_radius)
                r1 = min(ny - 1, row + row_radius)
                c0 = max(0, col - col_radius)
                c1 = min(nx - 1, col + col_radius)
                weights = []
                weighted_z = 0.0
                weight_sum = 0.0
                for rr in range(r0, r1 + 1):
                    for cc in range(c0, c1 + 1):
                        if not contour[rr, cc]:
                            continue
                        distance = float(np.hypot(float(xs[col] - xs[cc]), float(ys[row] - ys[rr])))
                        if distance > radius:
                            continue
                        t = np.clip(distance / radius, 0.0, 1.0)
                        weight = float(0.5 * (1.0 + np.cos(np.pi * t)))
                        if weight <= 0.0:
                            continue
                        weights.append(weight)
                        weighted_z += weight * float(cloud_grid[rr, cc])
                        weight_sum += weight
                if weights and weight_sum > 0.0:
                    w_avg = float(sum(weights) / len(weights))
                    z_contour_avg = weighted_z / weight_sum
                    blended[row, col] = w_avg * z_contour_avg + (1.0 - w_avg) * blended[row, col]

    mesh_json["blur_blend_z"] = np.round(blended.reshape(-1), 3).tolist()
    mesh_json["metadata"]["blur_blend"] = {
        "status": "ok" if contour.any() else "no_cloud_contour",
        "radius_m": radius,
        "contour_vertices": int(contour.sum()),
        "formula": (
            "For each final DEM vertex, collect valid Cloud DEM contour points within R meters. For each contour point, "
            "W=0.5*(1+cos(pi*d_m/R)). Then w_avg=mean(W), z_contour_avg=sum(W*z_contour)/sum(W), "
            "and z=w_avg*z_contour_avg+(1-w_avg)*z_terrarium."
        ),
    }
    return mesh_json


def main() -> int:
    args = parse_args()
    mesh_path = Path(args.mesh)
    mesh_json = json.loads(mesh_path.read_text(encoding="utf-8"))
    mesh_json = compute_distance_blend(mesh_json, args.blend_radius_m)
    mesh_json = compute_blur_blend(mesh_json, args.blend_radius_m)
    out_path = Path(args.out) if args.out else mesh_path.with_name("wiremesh_viewer.html")
    escaped_json = html.escape(json.dumps(mesh_json), quote=False)
    out_path.write_text(HTML_TEMPLATE.replace("__MESH_JSON__", escaped_json), encoding="utf-8")
    print(out_path.resolve())
    if args.open:
        webbrowser.open(out_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
