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


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LoD Terrarium Blend Viewer</title>
<style>
html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; background: #111; color: #eee; font-family: Segoe UI, Arial, sans-serif; }
#gl { width: 100vw; height: 100vh; display: block; }
#hud { position: fixed; left: 12px; top: 12px; width: 300px; max-height: calc(100vh - 24px); overflow: auto; background: rgba(0,0,0,.78); padding: 10px 12px; border: 1px solid #444; border-radius: 6px; font-size: 13px; line-height: 1.45; }
#hud label { display: flex; align-items: center; gap: 8px; margin: 5px 0; }
#hud select, #hud button, #hud input[type=number] { width: 100%; margin: 6px 0 8px; color: #eee; background: #1e1e1e; border: 1px solid #555; border-radius: 4px; padding: 6px 7px; box-sizing: border-box; }
#hud button { cursor: pointer; background: #263247; display: flex; align-items: center; justify-content: center; gap: 7px; }
#hud button:disabled { opacity: .55; cursor: wait; }
#hud input[type=range] { width: 100%; }
#hud details { margin: 8px 0; }
#hud summary { cursor: pointer; user-select: none; }
.layer-line { display: flex; align-items: center; gap: 8px; margin: 5px 0; }
.layer-line label { margin: 0; flex: 1; }
.icon-button { width: 30px !important; height: 28px; margin: 0 !important; padding: 0 !important; font-size: 17px; line-height: 1; }
.row { display: grid; grid-template-columns: 82px 1fr 44px; gap: 8px; align-items: center; margin: 6px 0; }
.bbox-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 8px; }
.bbox-grid label { display: block; margin: 0; }
.bbox-grid span { display: block; color: #bbb; font-size: 12px; margin-bottom: -2px; }
.swatch { width: 12px; height: 12px; border-radius: 2px; display: inline-block; }
.blue { background: #4aa3ff; } .red { background: #ff4a4a; } .green { background: #52d273; } .yellow { background: #f2c94c; }
.hint { color: #bbb; margin-top: 8px; }
.status { color: #ddd; white-space: pre-wrap; }
#busy { display: none; align-items: center; gap: 8px; margin: 8px 0; color: #f2c94c; }
#busy.active { display: flex; }
.spinner { width: 14px; height: 14px; border: 2px solid #555; border-top-color: #f2c94c; border-radius: 50%; animation: spin .8s linear infinite; }
#profilePanel { position: fixed; left: 320px; right: 12px; bottom: 12px; height: 230px; display: none; background: rgba(0,0,0,.82); border: 1px solid #444; border-radius: 6px; padding: 10px 12px; z-index: 4; }
#profilePanel.active { display: block; }
#profileHeader { display: flex; align-items: center; justify-content: space-between; color: #eee; font-size: 13px; margin-bottom: 6px; }
#profileHeader button { width: 28px; height: 24px; color: #eee; background: #263247; border: 1px solid #555; border-radius: 4px; cursor: pointer; }
#profileChart { width: 100%; height: 190px; }
.modal-backdrop { position: fixed; inset: 0; display: none; align-items: center; justify-content: center; background: rgba(0,0,0,.58); z-index: 10; }
.modal-backdrop.active { display: flex; }
.modal { width: min(420px, calc(100vw - 28px)); max-height: calc(100vh - 28px); overflow: auto; color: #eee; background: #151515; border: 1px solid #555; border-radius: 6px; box-shadow: 0 18px 60px rgba(0,0,0,.48); padding: 14px; }
.modal h2 { margin: 0 0 12px; font-size: 16px; font-weight: 600; }
.modal label { display: flex; align-items: center; gap: 8px; margin: 8px 0; }
.modal .bbox-grid label { display: block; margin: 0; }
.modal select, .modal button, .modal input[type=number], .modal input[type=text] { width: 100%; margin: 6px 0 10px; color: #eee; background: #1e1e1e; border: 1px solid #555; border-radius: 4px; padding: 7px; box-sizing: border-box; }
.modal button { cursor: pointer; background: #263247; }
.modal-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; }
.modal-actions button { margin: 0; }
@keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<canvas id="gl"></canvas>
<div id="hud">
  <div id="busy"><span class="spinner"></span><span id="busyText">Waiting for backend...</span></div>
  <label for="mode">Render</label>
  <select id="mode">
    <option value="wire">Wireframe</option>
    <option value="flat">Flat shading</option>
    <option value="phong">Phong shading</option>
  </select>
  <button id="profileTool">Profile Tool</button>
  <details open>
    <summary>Layers</summary>
    <div class="layer-line">
      <label><input id="showTerrarium" type="checkbox" checked><span class="swatch blue"></span>Terrarium</label>
      <button id="frameTerrarium" class="icon-button" title="Frame Terrarium DEM" aria-label="Frame Terrarium DEM">&#128247;</button>
      <button id="openTerrariumSettings" class="icon-button" title="Terrarium settings" aria-label="Terrarium settings">&#9881;</button>
    </div>
    <div class="layer-line">
      <label><input id="showCloud" type="checkbox"><span class="swatch red"></span>Cloud DEM</label>
      <button id="frameCloud" class="icon-button" title="Frame Point Cloud" aria-label="Frame Point Cloud">&#128247;</button>
      <button id="openCloudSettings" class="icon-button" title="Cloud DEM settings" aria-label="Cloud DEM settings">&#9881;</button>
    </div>
    <label><input id="showPointCloud" type="checkbox"><span class="swatch green"></span>Point Cloud</label>
    <div class="layer-line">
      <label><input id="showBlended" type="checkbox"><span class="swatch yellow"></span>Blended DEM</label>
      <button id="openBlendSettings" class="icon-button" title="Blending settings" aria-label="Blending settings">&#9881;</button>
    </div>
  </details>
  <details open>
    <summary>Sun</summary>
    <div class="row"><span>Azimuth</span><input id="sunAzimuth" type="range" min="0" max="360" value="315"><span id="sunAzimuthValue"></span></div>
    <div class="row"><span>Elevation</span><input id="sunElevation" type="range" min="1" max="89" value="45"><span id="sunElevationValue"></span></div>
    <div class="row"><span>Ambient</span><input id="ambient" type="range" min="0" max="80" value="28"><span id="ambientValue"></span></div>
    <div class="row"><span>Specular</span><input id="specular" type="range" min="0" max="80" value="18"><span id="specularValue"></span></div>
  </details>
  <div id="meta" class="status">Loading metadata...</div>
  <div class="hint">drag: rotate | wheel: zoom | shift+drag: pan</div>
</div>
<div id="profilePanel">
  <div id="profileHeader"><span id="profileTitle">Terrain profile</span><button id="closeProfile" aria-label="Close profile">x</button></div>
  <canvas id="profileChart"></canvas>
</div>
<div id="cloudModal" class="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="cloudModalTitle">
  <div class="modal">
    <h2 id="cloudModalTitle">Cloud DEM Settings</h2>
    <label for="cloudLasDir">LAS folder</label>
    <input id="cloudLasDir" type="text" placeholder="E:\path\to\las_folder">
    <label for="cloudSourceCrs">Source CRS when LAS has no CRS</label>
    <input id="cloudSourceCrs" type="text" value="EPSG:32630">
    <label for="cloudPixelSize">DEM pixel size (m)</label>
    <input id="cloudPixelSize" type="number" min="0.01" step="0.01" value="0.5">
    <label for="cloudChunkSize">Chunk size</label>
    <input id="cloudChunkSize" type="number" min="1" step="100000" value="1000000">
    <div class="modal-actions">
      <button id="acceptCloudSettings">Load</button>
      <button id="cancelCloudSettings">Cancel</button>
    </div>
  </div>
</div>
<div id="terrariumModal" class="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="terrariumModalTitle">
  <div class="modal">
    <h2 id="terrariumModalTitle">Terrarium Settings</h2>
    <label for="lod">Terrarium LoD</label>
    <select id="lod"></select>
    <label><input id="useManualBbox" type="checkbox">Manual BBOX</label>
    <div class="bbox-grid">
      <label><span>Min X</span><input id="bboxMinX" type="number" step="0.01"></label>
      <label><span>Min Y</span><input id="bboxMinY" type="number" step="0.01"></label>
      <label><span>Max X</span><input id="bboxMaxX" type="number" step="0.01"></label>
      <label><span>Max Y</span><input id="bboxMaxY" type="number" step="0.01"></label>
    </div>
    <button id="resetBbox">Use Cloud BBOX</button>
    <div class="modal-actions">
      <button id="acceptTerrariumSettings">Accept</button>
      <button id="cancelTerrariumSettings">Cancel</button>
    </div>
  </div>
</div>
<div id="blendModal" class="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="blendModalTitle">
  <div class="modal">
    <h2 id="blendModalTitle">Blending Settings</h2>
    <label for="blendStrategy">Strategy</label>
    <select id="blendStrategy">
      <option value="distance">Horizontal Distance</option>
      <option value="vertical_distance">Vertical Distance</option>
      <option value="blur">Blur</option>
      <option value="naive">Naive</option>
    </select>
    <label for="tessellationStrategy">Tessellation</label>
    <select id="tessellationStrategy">
      <option value="quadtree">Quadtree quads</option>
      <option value="nvb">Newest Vertex Bisection (NVB)</option>
      <option value="diamond48">4-8 Diamond edge splits</option>
    </select>
    <div class="row"><span>Blur R</span><input id="blurRadiusM" type="range" min="0" max="500" step="5" value="160"><span id="blurRadiusMValue"></span></div>
    <label><input id="refineBlends" type="checkbox">Refine blends over cloud</label>
    <div class="modal-actions">
      <button id="acceptBlendSettings">Accept</button>
      <button id="cancelBlendSettings">Cancel</button>
    </div>
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script>
const canvas = document.getElementById("gl");
const gl = canvas.getContext("webgl", {antialias: true});
if (!gl) throw new Error("WebGL unavailable");
let current = null;
let currentTerrainGrid = null;
let currentCloudGrid = null;
let meshes = {};
let scene = {cx: 0, cy: 0, cz: 0, extent: 1};
const INITIAL_LOD = 10;
const metaEl = document.getElementById("meta");
const busyEl = document.getElementById("busy");
const busyTextEl = document.getElementById("busyText");
let appliedBlendSettings = {
  strategy: "distance",
  tessellation: "quadtree",
  blurRadiusM: 160,
  refineBlends: false
};
let appliedTerrariumSettings = {
  lod: INITIAL_LOD,
  useManualBbox: false,
  bbox: null
};
let appliedCloudSettings = {
  lasDir: "",
  sourceCrs: "EPSG:32630",
  pixelSizeM: 0.5,
  chunkSize: 1000000
};
let profileMode = false;
let profilePoints = [];
let profileChart = null;
let profileLineBuffer = null;

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
void main() { vn = normalize(n); vp = p; gl_Position = mvp * vec4(p, 1.0); }
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
  float diffuse = max(dot(N, L), 0.0);
  float specular = pow(max(dot(R, V), 0.0), 24.0) * specularStrength;
  vec3 shaded = color * (ambient + (1.0 - ambient) * diffuse) + vec3(specular);
  gl_FragColor = vec4(shaded, 1.0);
}
`);

const pointProgram = program(`
attribute vec3 p;
uniform mat4 mvp;
uniform float pointSize;
void main() { gl_Position = mvp * vec4(p, 1.0); gl_PointSize = pointSize; }
`, `
precision mediump float;
uniform vec3 color;
void main() { gl_FragColor = vec4(color, 1.0); }
`);

function finite(v) { return v !== null && Number.isFinite(v); }
function point(data, zs, i) {
  const x = i % data.nx;
  const y = Math.floor(i / data.nx);
  return [(data.xs[x] - scene.cx) / scene.extent, (zs[i] - scene.cz) * 3.0 / scene.extent, (data.ys[y] - scene.cy) / scene.extent];
}
function buildLines(data, zs) {
  const lines = [];
  const edges = new Set();
  function push(a, b) {
    const lo = Math.min(a, b), hi = Math.max(a, b);
    const key = `${lo}:${hi}`;
    if (edges.has(key)) return;
    edges.add(key);
    lines.push(...point(data, zs, a), ...point(data, zs, b));
  }
  function tri(a, b, c) {
    if (!finite(zs[a]) || !finite(zs[b]) || !finite(zs[c])) return;
    push(a, b); push(b, c); push(c, a);
  }
  for (let y = 0; y < data.ny - 1; y++) {
    for (let x = 0; x < data.nx - 1; x++) {
      const a = y * data.nx + x, b = a + 1, c = a + data.nx, d = c + 1;
      tri(a, c, b); tri(b, c, d);
    }
  }
  return new Float32Array(lines);
}
function cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function add(a, b) { a[0]+=b[0]; a[1]+=b[1]; a[2]+=b[2]; }
function sub(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function norm(v) { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0]/l, v[1]/l, v[2]/l]; }
function pointKey(p) { return `${p[0].toFixed(7)}:${p[1].toFixed(7)}:${p[2].toFixed(7)}`; }
function buildTriangles(data, zs) {
  const pts = new Array(data.nx * data.ny);
  const smoothNormals = new Array(data.nx * data.ny);
  for (let i = 0; i < pts.length; i++) { pts[i] = finite(zs[i]) ? point(data, zs, i) : null; smoothNormals[i] = [0,0,0]; }
  const tris = [];
  const faceNormals = [];
  function tri(a, b, c) {
    if (!pts[a] || !pts[b] || !pts[c]) return;
    let n = cross(sub(pts[b], pts[a]), sub(pts[c], pts[a]));
    if (n[1] < 0) n = [-n[0], -n[1], -n[2]];
    add(smoothNormals[a], n); add(smoothNormals[b], n); add(smoothNormals[c], n);
    tris.push(a, b, c);
    faceNormals.push(norm(n));
  }
  for (let y = 0; y < data.ny - 1; y++) for (let x = 0; x < data.nx - 1; x++) {
    const a = y * data.nx + x, b = a + 1, c = a + data.nx, d = c + 1;
    tri(a, c, b); tri(b, c, d);
  }
  const p = [], smooth = [], flat = [];
  for (let t = 0; t < tris.length; t += 3) {
    const face = faceNormals[Math.floor(t / 3)];
    for (const i of [tris[t], tris[t + 1], tris[t + 2]]) {
      p.push(...pts[i]);
      smooth.push(...norm(smoothNormals[i]));
      flat.push(...face);
    }
  }
  return {positions: new Float32Array(p), normals: new Float32Array(smooth), flatNormals: new Float32Array(flat)};
}
function buffer(values) { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, values, gl.STATIC_DRAW); return b; }
function makeMesh(data, zs) {
  const lines = buildLines(data, zs);
  const tris = buildTriangles(data, zs);
  return {lineBuffer: buffer(lines), lineCount: lines.length / 3, triBuffer: buffer(tris.positions), normalBuffer: buffer(tris.normals), flatNormalBuffer: buffer(tris.flatNormals), triCount: tris.positions.length / 3};
}

function makePointCloud(points) {
  const positions = [];
  for (const p of points || []) {
    positions.push((p[0] - scene.cx) / scene.extent, (p[2] - scene.cz) * 3.0 / scene.extent, (p[1] - scene.cy) / scene.extent);
  }
  return {pointBuffer: buffer(new Float32Array(positions)), pointCount: positions.length / 3};
}

function explicitPoint(p) {
  return [(p[0] - scene.cx) / scene.extent, (p[2] - scene.cz) * 3.0 / scene.extent, (p[1] - scene.cy) / scene.extent];
}
function makeQuadMesh(quads) {
  const lines = [];
  const positions = [];
  const faceNormals = [];
  const smoothByPoint = new Map();
  function pushLine(a, b) { lines.push(...a, ...b); }
  function pushTri(a, b, c) {
    let n = cross(sub(b, a), sub(c, a));
    if (n[1] < 0) n = [-n[0], -n[1], -n[2]];
    n = norm(n);
    positions.push(...a, ...b, ...c);
    faceNormals.push(...n, ...n, ...n);
    for (const p of [a, b, c]) {
      const key = pointKey(p);
      if (!smoothByPoint.has(key)) smoothByPoint.set(key, [0,0,0]);
      add(smoothByPoint.get(key), n);
    }
  }
  for (const q of quads) {
    const a = explicitPoint([q[0], q[1], q[4]]);
    const b = explicitPoint([q[2], q[1], q[5]]);
    const c = explicitPoint([q[0], q[3], q[6]]);
    const d = explicitPoint([q[2], q[3], q[7]]);
    pushLine(a, b); pushLine(b, d); pushLine(d, c); pushLine(c, a); pushLine(b, c);
    pushTri(a, c, b); pushTri(b, c, d);
  }
  const smoothNormals = [];
  for (let i = 0; i < positions.length; i += 3) {
    smoothNormals.push(...norm(smoothByPoint.get(pointKey([positions[i], positions[i + 1], positions[i + 2]]))));
  }
  return {
    lineBuffer: buffer(new Float32Array(lines)),
    lineCount: lines.length / 3,
    triBuffer: buffer(new Float32Array(positions)),
    normalBuffer: buffer(new Float32Array(smoothNormals)),
    flatNormalBuffer: buffer(new Float32Array(faceNormals)),
    triCount: positions.length / 3
  };
}

function makeTriangleMesh(triangles) {
  const lines = [];
  const positions = [];
  const faceNormals = [];
  const smoothByPoint = new Map();
  function pushLine(a, b) { lines.push(...a, ...b); }
  function pushTri(a, b, c) {
    let n = cross(sub(b, a), sub(c, a));
    if (n[1] < 0) n = [-n[0], -n[1], -n[2]];
    n = norm(n);
    positions.push(...a, ...b, ...c);
    faceNormals.push(...n, ...n, ...n);
    for (const p of [a, b, c]) {
      const key = pointKey(p);
      if (!smoothByPoint.has(key)) smoothByPoint.set(key, [0,0,0]);
      add(smoothByPoint.get(key), n);
    }
    pushLine(a, b); pushLine(b, c); pushLine(c, a);
  }
  for (const t of triangles) {
    const a = explicitPoint([t[0], t[1], t[6]]);
    const b = explicitPoint([t[2], t[3], t[7]]);
    const c = explicitPoint([t[4], t[5], t[8]]);
    pushTri(a, b, c);
  }
  const smoothNormals = [];
  for (let i = 0; i < positions.length; i += 3) {
    smoothNormals.push(...norm(smoothByPoint.get(pointKey([positions[i], positions[i + 1], positions[i + 2]]))));
  }
  return {
    lineBuffer: buffer(new Float32Array(lines)),
    lineCount: lines.length / 3,
    triBuffer: buffer(new Float32Array(positions)),
    normalBuffer: buffer(new Float32Array(smoothNormals)),
    flatNormalBuffer: buffer(new Float32Array(faceNormals)),
    triCount: positions.length / 3
  };
}

function finiteValues(values) {
  return values.filter(finite);
}

function valueRange(values) {
  let minValue = Infinity;
  let maxValue = -Infinity;
  let count = 0;
  for (const value of values) {
    if (!finite(value)) continue;
    if (value < minValue) minValue = value;
    if (value > maxValue) maxValue = value;
    count++;
  }
  return {min: minValue, max: maxValue, count};
}

function bboxCenter(bbox) {
  return [(bbox.minx + bbox.maxx) / 2, (bbox.miny + bbox.maxy) / 2];
}

function bboxExtent(bbox) {
  return Math.max(bbox.maxx - bbox.minx, bbox.maxy - bbox.miny, 1);
}

function valuesInBbox(grid, zs, bbox) {
  const values = [];
  for (let y = 0; y < grid.ny; y++) {
    const gy = grid.ys[y];
    if (gy < bbox.miny || gy > bbox.maxy) continue;
    for (let x = 0; x < grid.nx; x++) {
      const gx = grid.xs[x];
      if (gx < bbox.minx || gx > bbox.maxx) continue;
      const i = y * grid.nx + x;
      if (finite(zs[i])) values.push(zs[i]);
    }
  }
  return values;
}

function blendStrategy() {
  return appliedBlendSettings.strategy;
}

function blendStrategyLabel() {
  const strategy = blendStrategy();
  if (strategy === "naive") return "Naive";
  if (strategy === "blur") return "Blur";
  if (strategy === "vertical_distance") return "Vertical Distance";
  return "Horizontal Distance";
}

function currentBlendValues() {
  if (!current) return [];
  const strategy = blendStrategy();
  if (strategy === "naive") return current.cloud_replacement;
  if (strategy === "blur") return current.blur_blend;
  if (strategy === "vertical_distance") return current.vertical_distance_blend;
  return current.distance_blend;
}

function setSceneFrame(bbox) {
  if (!current) return;
  const center = bboxCenter(bbox);
  let zValues = valuesInBbox(currentCloudGrid, currentCloudGrid.z, bbox)
    .concat(valuesInBbox(currentTerrainGrid, currentBlendValues(), bbox));
  if (!zValues.length) {
    zValues = finiteValues(current.cloud_layer.z)
      .concat(finiteValues(currentBlendValues()));
  }
  scene.cx = center[0];
  scene.cy = center[1];
  const zRange = valueRange(zValues);
  scene.cz = zRange.count ? (zRange.min + zRange.max) / 2 : 0;
  scene.extent = bboxExtent(bbox) * 1.15;
  panX = 0;
  panY = 0;
  dist = 2.1;
}

function rebuildMeshes() {
  if (!current || !currentTerrainGrid || !currentCloudGrid) return;
  const refined = current.refined_mesh && current.refined_mesh.applied;
  const strategy = blendStrategy();
  const refinedLayer = strategy === "naive" ? "cloud_replacement" : (strategy === "blur" ? "blur_blend" : (strategy === "vertical_distance" ? "vertical_distance_blend" : "distance_blend"));
  let blendedMesh = makeMesh(currentTerrainGrid, currentBlendValues());
  if (refined) {
    const layer = current.refined_mesh.layers[refinedLayer];
    blendedMesh = current.refined_mesh.geometry === "triangles" ? makeTriangleMesh(layer.triangles) : makeQuadMesh(layer.quads);
  }
  meshes = {
    terrarium: {mesh: makeMesh(currentTerrainGrid, current.terrarium), color: [0.2, 0.55, 1.0], control: "showTerrarium"},
    cloud: {mesh: makeMesh(currentCloudGrid, currentCloudGrid.z), color: [1.0, 0.18, 0.15], control: "showCloud"},
    blended: {mesh: blendedMesh, color: [0.95, 0.70, 0.18], control: "showBlended"},
    points: {mesh: makePointCloud(current.point_sample), color: [0.32, 0.82, 0.45], control: "showPointCloud", kind: "points"}
  };
}

function framePointCloud() {
  if (!current || !current.bbox || !current.bbox.point_cloud) {
    metaEl.textContent = "Cloud DEM is not loaded.";
    return;
  }
  setSceneFrame(current.bbox.point_cloud);
  rebuildMeshes();
}

function frameTerrarium() {
  if (!current) return;
  setSceneFrame(current.bbox.terrarium_dem);
  rebuildMeshes();
}

function setBboxInputs(bbox) {
  if (!bbox) return;
  document.getElementById("bboxMinX").value = bbox.minx.toFixed(2);
  document.getElementById("bboxMinY").value = bbox.miny.toFixed(2);
  document.getElementById("bboxMaxX").value = bbox.maxx.toFixed(2);
  document.getElementById("bboxMaxY").value = bbox.maxy.toFixed(2);
}

function readBboxFromForm() {
  return {
    minx: Number(document.getElementById("bboxMinX").value),
    miny: Number(document.getElementById("bboxMinY").value),
    maxx: Number(document.getElementById("bboxMaxX").value),
    maxy: Number(document.getElementById("bboxMaxY").value)
  };
}

function validateBbox(bbox) {
  if (!Object.values(bbox).every(Number.isFinite)) throw new Error("Manual BBOX values must be numeric");
  if (bbox.minx >= bbox.maxx || bbox.miny >= bbox.maxy) throw new Error("Manual BBOX min values must be lower than max values");
}

function readTerrariumSettingsFromForm() {
  const useManualBbox = document.getElementById("useManualBbox").checked;
  const bbox = readBboxFromForm();
  if (useManualBbox) validateBbox(bbox);
  return {
    lod: Number(document.getElementById("lod").value),
    useManualBbox,
    bbox
  };
}

function writeTerrariumSettingsToForm(settings) {
  document.getElementById("lod").value = String(settings.lod);
  document.getElementById("useManualBbox").checked = settings.useManualBbox;
  if (settings.bbox) setBboxInputs(settings.bbox);
}

function readCloudSettingsFromForm() {
  const settings = {
    lasDir: document.getElementById("cloudLasDir").value.trim(),
    sourceCrs: document.getElementById("cloudSourceCrs").value.trim(),
    pixelSizeM: Number(document.getElementById("cloudPixelSize").value),
    chunkSize: Number(document.getElementById("cloudChunkSize").value)
  };
  if (!settings.lasDir) throw new Error("LAS folder is required");
  if (!settings.sourceCrs) throw new Error("Source CRS is required");
  if (!Number.isFinite(settings.pixelSizeM) || settings.pixelSizeM <= 0) throw new Error("DEM pixel size must be greater than zero");
  if (!Number.isInteger(settings.chunkSize) || settings.chunkSize <= 0) throw new Error("Chunk size must be a positive integer");
  return settings;
}

function writeCloudSettingsToForm(settings) {
  document.getElementById("cloudLasDir").value = settings.lasDir;
  document.getElementById("cloudSourceCrs").value = settings.sourceCrs;
  document.getElementById("cloudPixelSize").value = String(settings.pixelSizeM);
  document.getElementById("cloudChunkSize").value = String(settings.chunkSize);
}

function openCloudModal() {
  writeCloudSettingsToForm(appliedCloudSettings);
  document.getElementById("cloudModal").classList.add("active");
}

function closeCloudModal() {
  document.getElementById("cloudModal").classList.remove("active");
}

function readManualBbox(settings = appliedTerrariumSettings) {
  if (!settings.useManualBbox) return null;
  validateBbox(settings.bbox);
  return settings.bbox;
}

function openTerrariumModal() {
  writeTerrariumSettingsToForm(appliedTerrariumSettings);
  document.getElementById("terrariumModal").classList.add("active");
}

function closeTerrariumModal() {
  document.getElementById("terrariumModal").classList.remove("active");
}

function resetBboxToCloud() {
  const cloudBbox = current && current.bbox && current.bbox.point_cloud ? current.bbox.point_cloud : appliedTerrariumSettings.bbox;
  document.getElementById("useManualBbox").checked = false;
  if (cloudBbox) setBboxInputs(cloudBbox);
}

function bboxQueryString(bbox) {
  if (!bbox) return "";
  return `&bbox_minx=${encodeURIComponent(bbox.minx)}&bbox_miny=${encodeURIComponent(bbox.miny)}&bbox_maxx=${encodeURIComponent(bbox.maxx)}&bbox_maxy=${encodeURIComponent(bbox.maxy)}`;
}

function readBlendSettingsFromForm() {
  return {
    strategy: document.getElementById("blendStrategy").value,
    tessellation: document.getElementById("tessellationStrategy").value,
    blurRadiusM: Number(document.getElementById("blurRadiusM").value),
    refineBlends: document.getElementById("refineBlends").checked
  };
}

function writeBlendSettingsToForm(settings) {
  document.getElementById("blendStrategy").value = settings.strategy;
  document.getElementById("tessellationStrategy").value = settings.tessellation;
  document.getElementById("blurRadiusM").value = String(settings.blurRadiusM);
  document.getElementById("refineBlends").checked = settings.refineBlends;
  updateSliderLabels();
}

function openBlendModal() {
  writeBlendSettingsToForm(appliedBlendSettings);
  document.getElementById("blendModal").classList.add("active");
}

function closeBlendModal() {
  document.getElementById("blendModal").classList.remove("active");
}

async function loadInfo() {
  setBusy(true, "Loading metadata...");
  const info = await fetch("/api/info").then(r => r.json());
  const lod = document.getElementById("lod");
  lod.innerHTML = "";
  for (const item of info.lods) {
    const opt = document.createElement("option");
    opt.value = item.z;
    opt.textContent = `z${item.z} | ${item.resolution_m.toFixed(2)} m/px | ${item.tiles} tiles | ${item.mesh_width}x${item.mesh_height}`;
    if (item.z === info.default_lod) opt.selected = true;
    lod.appendChild(opt);
  }
  lod.value = String(info.default_lod ?? INITIAL_LOD);
  appliedTerrariumSettings = {
    lod: Number(lod.value),
    useManualBbox: false,
    bbox: info.cloud_bbox
  };
  writeTerrariumSettingsToForm(appliedTerrariumSettings);
  if (info.cloud_source && info.cloud_source.source_crs) {
    appliedCloudSettings.sourceCrs = info.cloud_source.source_crs;
  }
  const blurRadius = document.getElementById("blurRadiusM");
  const defaultBlurRadius = Number(info.default_blur_radius_m ?? blurRadius.value);
  blurRadius.value = String(Math.round(defaultBlurRadius));
  blurRadius.max = String(Math.max(500, Math.ceil(defaultBlurRadius * 3 / 50) * 50));
  appliedBlendSettings.blurRadiusM = Number(blurRadius.value);
  writeBlendSettingsToForm(appliedBlendSettings);
  setBboxInputs(info.cloud_bbox);
  updateSliderLabels();
  metaEl.textContent = `Choose a Terrarium LoD and recalculate.\nCloud DEM: ${info.cloud_loaded ? "loaded" : "not loaded"}\nCloud BBOX: ${formatBbox(info.cloud_bbox)}`;
  setBusy(false);
}

function setBusy(active, message = "Waiting for backend...") {
  busyTextEl.textContent = message;
  busyEl.classList.toggle("active", active);
}

function formatBbox(bbox) {
  if (!bbox) return "n/a";
  return `[${bbox.minx.toFixed(2)}, ${bbox.miny.toFixed(2)}, ${bbox.maxx.toFixed(2)}, ${bbox.maxy.toFixed(2)}]`;
}

function updateMetaText() {
  if (!current) return;
  const cloudGrid = current.cloud_layer;
  let refinementText = "\nBlend refinement: off";
  if (current.refined_mesh) {
    if (current.refined_mesh.applied) {
      const elementCount = current.refined_mesh.geometry === "triangles" ? `${current.refined_mesh.triangle_count} triangles` : `${current.refined_mesh.quad_count} quads`;
      refinementText = `\nBlend refinement: on | ${current.refined_mesh.tessellation_strategy} | target ${current.refined_mesh.target_resolution_m.toFixed(2)} m | ${elementCount} | max depth ${current.refined_mesh.max_depth}`;
    } else {
      refinementText = `\nBlend refinement: on | not applied (${current.refined_mesh.reason})`;
    }
  }
  const pointCount = current.point_sample ? current.point_sample.length : 0;
  const cacheText = current.cloud_source && current.cloud_source.cache_hit ? " | cache hit" : "";
  metaEl.textContent = `Terrarium z${current.lod} | ${current.resolution_m.toFixed(2)} m/px | ${current.nx}x${current.ny}\nCloud DEM ${current.cloud_loaded ? "loaded" : "not loaded"}${cacheText} | fixed view ${cloudGrid.resolution_m.toFixed(2)} m/px | ${cloudGrid.nx}x${cloudGrid.ny}\nPoint Cloud sample: ${pointCount} points\nBlended DEM strategy: ${blendStrategyLabel()}\n${current.tiles.length} tiles | blend cloud cells: ${current.cloud_valid_count}\nBlend radius: ${current.blend_radius_m.toFixed(2)} m | blur radius: ${current.blur_radius_m.toFixed(2)} m${refinementText}\nTerrarium DEM BBOX: ${formatBbox(current.bbox.terrarium_dem)}\nPoint Cloud BBOX: ${formatBbox(current.bbox.point_cloud)}`;
}

async function loadMesh(forcedLod = null, frameTarget = null) {
  const lod = document.getElementById("lod");
  const requestedBlendSettings = readBlendSettingsFromForm();
  let requestedTerrariumSettings;
  try {
    requestedTerrariumSettings = readTerrariumSettingsFromForm();
  } catch (e) {
    metaEl.textContent = `Failed: ${e.message}`;
    return;
  }
  const useManualBbox = document.getElementById("useManualBbox");
  const bboxInputs = ["bboxMinX", "bboxMinY", "bboxMaxX", "bboxMaxY"].map(id => document.getElementById(id));
  const bboxButtons = ["resetBbox", "openBlendSettings", "openTerrariumSettings", "openCloudSettings"].map(id => document.getElementById(id));
  const blendControls = ["blendStrategy", "tessellationStrategy", "blurRadiusM", "refineBlends", "acceptBlendSettings", "cancelBlendSettings"].map(id => document.getElementById(id));
  const terrariumControls = ["lod", "useManualBbox", "acceptTerrariumSettings", "cancelTerrariumSettings"].map(id => document.getElementById(id));
  const cloudControls = ["cloudLasDir", "cloudSourceCrs", "cloudPixelSize", "cloudChunkSize", "acceptCloudSettings", "cancelCloudSettings"].map(id => document.getElementById(id));
  if (Number.isInteger(forcedLod)) {
    requestedTerrariumSettings.lod = forcedLod;
    lod.value = String(forcedLod);
  }
  const z = String(requestedTerrariumSettings.lod);
  const refineQuery = requestedBlendSettings.refineBlends ? "&refine=1" : "";
  const tessellationQuery = `&tessellation=${encodeURIComponent(requestedBlendSettings.tessellation)}`;
  const blurRadiusQuery = `&blur_radius_m=${encodeURIComponent(requestedBlendSettings.blurRadiusM)}`;
  let bboxQuery = "";
  try {
    bboxQuery = bboxQueryString(readManualBbox(requestedTerrariumSettings));
  } catch (e) {
    metaEl.textContent = `Failed: ${e.message}`;
    return;
  }
  lod.disabled = true;
  useManualBbox.disabled = true;
  for (const element of bboxInputs.concat(bboxButtons).concat(blendControls).concat(terrariumControls).concat(cloudControls)) element.disabled = true;
  setBusy(true, `Waiting for backend: downloading / recalculating z${z}...`);
  metaEl.textContent = `Downloading / recalculating z${z}...`;
  try {
    const data = await fetch(`/api/mesh?z=${encodeURIComponent(z)}${refineQuery}${tessellationQuery}${blurRadiusQuery}${bboxQuery}`).then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    });
    current = data;
    currentTerrainGrid = {xs: data.xs, ys: data.ys, nx: data.nx, ny: data.ny};
    currentCloudGrid = data.cloud_layer;
    appliedBlendSettings = requestedBlendSettings;
    appliedTerrariumSettings = requestedTerrariumSettings;
    if (frameTarget === "cloud" && data.bbox && data.bbox.point_cloud) framePointCloud();
    else if (frameTarget === "terrarium" || !data.bbox || !data.bbox.point_cloud) frameTerrarium();
    else rebuildMeshes();
    updateMetaText();
  } catch (e) {
    metaEl.textContent = `Failed: ${e.message}`;
  } finally {
    lod.disabled = false;
    useManualBbox.disabled = false;
    for (const element of bboxInputs.concat(bboxButtons).concat(blendControls).concat(terrariumControls).concat(cloudControls)) element.disabled = false;
    setBusy(false);
  }
}

async function loadCloudDem() {
  let settings;
  try {
    settings = readCloudSettingsFromForm();
  } catch (e) {
    metaEl.textContent = `Failed: ${e.message}`;
    return;
  }
  closeCloudModal();
  setBusy(true, "Waiting for backend: building Cloud DEM from LAS...");
  metaEl.textContent = "Building Cloud DEM from LAS files...";
  const controls = ["openCloudSettings", "cloudLasDir", "cloudSourceCrs", "cloudPixelSize", "cloudChunkSize", "acceptCloudSettings", "cancelCloudSettings"].map(id => document.getElementById(id));
  for (const element of controls) element.disabled = true;
  try {
    const query = `las_dir=${encodeURIComponent(settings.lasDir)}&source_crs=${encodeURIComponent(settings.sourceCrs)}&pixel_size_m=${encodeURIComponent(settings.pixelSizeM)}&chunk_size=${encodeURIComponent(settings.chunkSize)}`;
    const result = await fetch(`/api/cloud_dem?${query}`).then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    });
    appliedCloudSettings = settings;
    appliedTerrariumSettings = {
      lod: appliedTerrariumSettings.lod,
      useManualBbox: true,
      bbox: result.bbox
    };
    writeTerrariumSettingsToForm(appliedTerrariumSettings);
    document.getElementById("showCloud").checked = true;
    document.getElementById("showPointCloud").checked = true;
    await loadMesh(null, "cloud");
  } catch (e) {
    metaEl.textContent = `Failed: ${e.message}`;
  } finally {
    for (const element of controls) element.disabled = false;
    setBusy(false);
  }
}

let yaw = -0.7, pitch = 0.8, dist = 2.1, panX = 0, panY = 0;
let dragging = false, lastX = 0, lastY = 0, panning = false, dragMoved = false;
canvas.addEventListener("mousedown", e => { dragging = true; dragMoved = false; panning = e.shiftKey; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener("mouseup", () => dragging = false);
window.addEventListener("mousemove", e => {
  if (!dragging) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  if (Math.hypot(dx, dy) > 2) dragMoved = true;
  if (panning) { panX += dx / canvas.clientWidth * dist; panY -= dy / canvas.clientHeight * dist; }
  else { yaw += dx * 0.008; pitch = Math.max(-1.45, Math.min(1.45, pitch + dy * 0.008)); }
});
canvas.addEventListener("wheel", e => { e.preventDefault(); dist *= Math.exp(e.deltaY * 0.001); dist = Math.max(0.4, Math.min(12, dist)); }, {passive:false});

function sliderNumber(id) { return Number(document.getElementById(id).value); }
function updateSliderLabels() {
  document.getElementById("blurRadiusMValue").textContent = `${sliderNumber("blurRadiusM").toFixed(0)}m`;
  document.getElementById("sunAzimuthValue").textContent = `${sliderNumber("sunAzimuth")}deg`;
  document.getElementById("sunElevationValue").textContent = `${sliderNumber("sunElevation")}deg`;
  document.getElementById("ambientValue").textContent = (sliderNumber("ambient") / 100).toFixed(2);
  document.getElementById("specularValue").textContent = (sliderNumber("specular") / 100).toFixed(2);
}
function sunDirection() {
  const az = sliderNumber("sunAzimuth") * Math.PI / 180;
  const el = sliderNumber("sunElevation") * Math.PI / 180;
  return [Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el)];
}
for (const id of ["blurRadiusM", "sunAzimuth", "sunElevation", "ambient", "specular"]) document.getElementById(id).addEventListener("input", updateSliderLabels);
updateSliderLabels();

function matInv(m) {
  const inv = new Float32Array(16);
  inv[0] = m[5]*m[10]*m[15]-m[5]*m[11]*m[14]-m[9]*m[6]*m[15]+m[9]*m[7]*m[14]+m[13]*m[6]*m[11]-m[13]*m[7]*m[10];
  inv[4] = -m[4]*m[10]*m[15]+m[4]*m[11]*m[14]+m[8]*m[6]*m[15]-m[8]*m[7]*m[14]-m[12]*m[6]*m[11]+m[12]*m[7]*m[10];
  inv[8] = m[4]*m[9]*m[15]-m[4]*m[11]*m[13]-m[8]*m[5]*m[15]+m[8]*m[7]*m[13]+m[12]*m[5]*m[11]-m[12]*m[7]*m[9];
  inv[12] = -m[4]*m[9]*m[14]+m[4]*m[10]*m[13]+m[8]*m[5]*m[14]-m[8]*m[6]*m[13]-m[12]*m[5]*m[10]+m[12]*m[6]*m[9];
  inv[1] = -m[1]*m[10]*m[15]+m[1]*m[11]*m[14]+m[9]*m[2]*m[15]-m[9]*m[3]*m[14]-m[13]*m[2]*m[11]+m[13]*m[3]*m[10];
  inv[5] = m[0]*m[10]*m[15]-m[0]*m[11]*m[14]-m[8]*m[2]*m[15]+m[8]*m[3]*m[14]+m[12]*m[2]*m[11]-m[12]*m[3]*m[10];
  inv[9] = -m[0]*m[9]*m[15]+m[0]*m[11]*m[13]+m[8]*m[1]*m[15]-m[8]*m[3]*m[13]-m[12]*m[1]*m[11]+m[12]*m[3]*m[9];
  inv[13] = m[0]*m[9]*m[14]-m[0]*m[10]*m[13]-m[8]*m[1]*m[14]+m[8]*m[2]*m[13]+m[12]*m[1]*m[10]-m[12]*m[2]*m[9];
  inv[2] = m[1]*m[6]*m[15]-m[1]*m[7]*m[14]-m[5]*m[2]*m[15]+m[5]*m[3]*m[14]+m[13]*m[2]*m[7]-m[13]*m[3]*m[6];
  inv[6] = -m[0]*m[6]*m[15]+m[0]*m[7]*m[14]+m[4]*m[2]*m[15]-m[4]*m[3]*m[14]-m[12]*m[2]*m[7]+m[12]*m[3]*m[6];
  inv[10] = m[0]*m[5]*m[15]-m[0]*m[7]*m[13]-m[4]*m[1]*m[15]+m[4]*m[3]*m[13]+m[12]*m[1]*m[7]-m[12]*m[3]*m[5];
  inv[14] = -m[0]*m[5]*m[14]+m[0]*m[6]*m[13]+m[4]*m[1]*m[14]-m[4]*m[2]*m[13]-m[12]*m[1]*m[6]+m[12]*m[2]*m[5];
  inv[3] = -m[1]*m[6]*m[11]+m[1]*m[7]*m[10]+m[5]*m[2]*m[11]-m[5]*m[3]*m[10]-m[9]*m[2]*m[7]+m[9]*m[3]*m[6];
  inv[7] = m[0]*m[6]*m[11]-m[0]*m[7]*m[10]-m[4]*m[2]*m[11]+m[4]*m[3]*m[10]+m[8]*m[2]*m[7]-m[8]*m[3]*m[6];
  inv[11] = -m[0]*m[5]*m[11]+m[0]*m[7]*m[9]+m[4]*m[1]*m[11]-m[4]*m[3]*m[9]-m[8]*m[1]*m[7]+m[8]*m[3]*m[5];
  inv[15] = m[0]*m[5]*m[10]-m[0]*m[6]*m[9]-m[4]*m[1]*m[10]+m[4]*m[2]*m[9]+m[8]*m[1]*m[6]-m[8]*m[2]*m[5];
  let det = m[0]*inv[0]+m[1]*inv[4]+m[2]*inv[8]+m[3]*inv[12];
  if (!det) return null;
  det = 1.0 / det;
  for (let i = 0; i < 16; i++) inv[i] *= det;
  return inv;
}

function transform4(m, v) {
  return [
    m[0]*v[0]+m[4]*v[1]+m[8]*v[2]+m[12]*v[3],
    m[1]*v[0]+m[5]*v[1]+m[9]*v[2]+m[13]*v[3],
    m[2]*v[0]+m[6]*v[1]+m[10]*v[2]+m[14]*v[3],
    m[3]*v[0]+m[7]*v[1]+m[11]*v[2]+m[15]*v[3]
  ];
}

function currentMvp() {
  const aspect = canvas.width / Math.max(canvas.height, 1);
  const eye = [Math.sin(yaw)*Math.cos(pitch)*dist + panX, Math.sin(pitch)*dist + panY, Math.cos(yaw)*Math.cos(pitch)*dist];
  const center = [panX, panY, 0];
  return matMul(perspective(Math.PI/4, aspect, 0.01, 100), lookAt(eye, center, [0,1,0]));
}

function screenToMapPoint(e) {
  resize();
  const rect = canvas.getBoundingClientRect();
  const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  const y = 1 - ((e.clientY - rect.top) / rect.height) * 2;
  const inv = matInv(currentMvp());
  if (!inv) return null;
  const near = transform4(inv, [x, y, -1, 1]);
  const far = transform4(inv, [x, y, 1, 1]);
  for (const p of [near, far]) { p[0] /= p[3]; p[1] /= p[3]; p[2] /= p[3]; }
  const dy = far[1] - near[1];
  if (Math.abs(dy) < 1e-8) return null;
  const t = -near[1] / dy;
  const nx = near[0] + (far[0] - near[0]) * t;
  const nz = near[2] + (far[2] - near[2]) * t;
  return {x: nx * scene.extent + scene.cx, y: nz * scene.extent + scene.cy};
}

function sampleGrid(grid, values, x, y) {
  if (!grid || !values || !grid.nx || !grid.ny || grid.nx < 1 || grid.ny < 1) return null;
  const nx = grid.nx, ny = grid.ny;
  const dx = nx > 1 ? Math.abs(grid.xs[1] - grid.xs[0]) : 1;
  const dy = ny > 1 ? Math.abs(grid.ys[1] - grid.ys[0]) : dx;
  const col = (x - grid.xs[0]) / dx;
  const row = (grid.ys[0] - y) / dy;
  if (col < 0 || row < 0 || col > nx - 1 || row > ny - 1) return null;
  const c0 = Math.floor(col), r0 = Math.floor(row);
  const c1 = Math.min(nx - 1, c0 + 1), r1 = Math.min(ny - 1, r0 + 1);
  const tx = col - c0, ty = row - r0;
  const idx = (r, c) => r * nx + c;
  const z00 = values[idx(r0, c0)], z10 = values[idx(r0, c1)], z01 = values[idx(r1, c0)], z11 = values[idx(r1, c1)];
  const valid = [z00, z10, z01, z11].filter(finite);
  if (!valid.length) return null;
  if (![z00, z10, z01, z11].every(finite)) return valid.reduce((a, b) => a + b, 0) / valid.length;
  return z00*(1-tx)*(1-ty) + z10*tx*(1-ty) + z01*(1-tx)*ty + z11*tx*ty;
}

function buildProfileSamples(a, b) {
  const distance = Math.hypot(b.x - a.x, b.y - a.y);
  const sampleStepM = 0.5;
  const steps = Math.max(1, Math.ceil(distance / sampleStepM));
  const terrarium = [];
  const cloud = [];
  const blended = [];
  const blendValues = currentBlendValues();
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const x = a.x + (b.x - a.x) * t;
    const y = a.y + (b.y - a.y) * t;
    const d = distance * t;
    const cloudZ = sampleGrid(currentCloudGrid, currentCloudGrid.z, x, y);
    let blendedZ = sampleGrid(currentTerrainGrid, blendValues, x, y);
    if (finite(cloudZ)) blendedZ = cloudZ;
    terrarium.push({x: d, y: sampleGrid(currentTerrainGrid, current.terrarium, x, y)});
    cloud.push({x: d, y: cloudZ});
    blended.push({x: d, y: blendedZ});
  }
  return {distance, terrarium, cloud, blended};
}

function updateProfileChart() {
  if (!current || profilePoints.length !== 2) return;
  if (!window.Chart) {
    metaEl.textContent = "Chart.js is not available; profile chart cannot be rendered.";
    return;
  }
  const profile = buildProfileSamples(profilePoints[0], profilePoints[1]);
  document.getElementById("profilePanel").classList.add("active");
  document.getElementById("profileTitle").textContent = `Terrain profile | ${(profile.distance).toFixed(1)} m`;
  const ctx = document.getElementById("profileChart");
  const datasets = [
    {label: "Terrarium DEM", data: profile.terrarium, borderColor: "#4aa3ff", backgroundColor: "#4aa3ff"},
    {label: "Cloud DEM", data: profile.cloud, borderColor: "#ff4a4a", backgroundColor: "#ff4a4a"},
    {label: "Blended DEM", data: profile.blended, borderColor: "#f2c94c", backgroundColor: "#f2c94c"}
  ].map(ds => ({...ds, showLine: true, spanGaps: true, pointRadius: 0, pointHoverRadius: 0, borderWidth: 2, tension: 0}));
  if (profileChart) profileChart.destroy();
  profileChart = new Chart(ctx, {
    type: "scatter",
    data: {datasets},
    options: {
      maintainAspectRatio: false,
      animation: false,
      parsing: false,
      scales: {
        x: {type: "linear", title: {display: true, text: "Distance (m)", color: "#ddd"}, ticks: {color: "#ccc"}, grid: {color: "rgba(255,255,255,.12)"}},
        y: {title: {display: true, text: "Elevation (m)", color: "#ddd"}, ticks: {color: "#ccc"}, grid: {color: "rgba(255,255,255,.12)"}}
      },
      plugins: {legend: {labels: {color: "#ddd"}}}
    }
  });
}

function profileLineMesh() {
  if (profilePoints.length !== 2 || !current) return null;
  const points = profilePoints.map(p => {
    const z = sampleGrid(currentTerrainGrid, currentBlendValues(), p.x, p.y);
    return [(p.x - scene.cx) / scene.extent, ((finite(z) ? z : scene.cz) - scene.cz + 2.0) * 3.0 / scene.extent, (p.y - scene.cy) / scene.extent];
  });
  if (!profileLineBuffer) profileLineBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, profileLineBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([...points[0], ...points[1]]), gl.DYNAMIC_DRAW);
  return {lineBuffer: profileLineBuffer, lineCount: 2};
}

function handleProfileClick(e) {
  if (!profileMode || !current) return;
  if (dragMoved) return;
  e.preventDefault();
  const point = screenToMapPoint(e);
  if (!point) return;
  if (profilePoints.length >= 2) profilePoints = [];
  profilePoints.push(point);
  if (profilePoints.length === 2) updateProfileChart();
  else metaEl.textContent = "Profile tool: click the second point.";
}

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
function selectedLayers() { return Object.values(meshes).filter(layer => document.getElementById(layer.control).checked); }
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
function drawPoints(layer, mvp) {
  gl.useProgram(pointProgram);
  gl.bindBuffer(gl.ARRAY_BUFFER, layer.mesh.pointBuffer);
  const pLoc = gl.getAttribLocation(pointProgram, "p");
  gl.vertexAttribPointer(pLoc, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(pLoc);
  gl.uniformMatrix4fv(gl.getUniformLocation(pointProgram, "mvp"), false, mvp);
  gl.uniform3fv(gl.getUniformLocation(pointProgram, "color"), new Float32Array(layer.color));
  gl.uniform1f(gl.getUniformLocation(pointProgram, "pointSize"), 4.0);
  gl.drawArrays(gl.POINTS, 0, layer.mesh.pointCount);
}
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
    if (layer.kind === "points") drawPoints(layer, mvp);
    else if (mode === "phong") drawPhong(layer, mvp);
    else if (mode === "flat") drawFlat(layer, mvp);
    else drawWire(layer, mvp);
  }
  const profileLine = profileLineMesh();
  if (profileLine) drawWire({mesh: profileLine, color: [0.32, 0.95, 0.45]}, mvp);
  requestAnimationFrame(render);
}
canvas.addEventListener("click", handleProfileClick);
document.getElementById("frameCloud").addEventListener("click", framePointCloud);
document.getElementById("frameTerrarium").addEventListener("click", frameTerrarium);
document.getElementById("openCloudSettings").addEventListener("click", openCloudModal);
document.getElementById("acceptCloudSettings").addEventListener("click", loadCloudDem);
document.getElementById("cancelCloudSettings").addEventListener("click", () => {
  writeCloudSettingsToForm(appliedCloudSettings);
  closeCloudModal();
});
document.getElementById("cloudModal").addEventListener("click", e => {
  if (e.target.id === "cloudModal") {
    writeCloudSettingsToForm(appliedCloudSettings);
    closeCloudModal();
  }
});
document.getElementById("openTerrariumSettings").addEventListener("click", openTerrariumModal);
document.getElementById("acceptTerrariumSettings").addEventListener("click", () => {
  try {
    readTerrariumSettingsFromForm();
  } catch (e) {
    metaEl.textContent = `Failed: ${e.message}`;
    return;
  }
  closeTerrariumModal();
  loadMesh();
});
document.getElementById("cancelTerrariumSettings").addEventListener("click", () => {
  writeTerrariumSettingsToForm(appliedTerrariumSettings);
  closeTerrariumModal();
});
document.getElementById("terrariumModal").addEventListener("click", e => {
  if (e.target.id === "terrariumModal") {
    writeTerrariumSettingsToForm(appliedTerrariumSettings);
    closeTerrariumModal();
  }
});
document.getElementById("openBlendSettings").addEventListener("click", openBlendModal);
document.getElementById("acceptBlendSettings").addEventListener("click", () => {
  closeBlendModal();
  loadMesh();
});
document.getElementById("cancelBlendSettings").addEventListener("click", () => {
  writeBlendSettingsToForm(appliedBlendSettings);
  closeBlendModal();
});
document.getElementById("blendModal").addEventListener("click", e => {
  if (e.target.id === "blendModal") {
    writeBlendSettingsToForm(appliedBlendSettings);
    closeBlendModal();
  }
});
document.getElementById("resetBbox").addEventListener("click", () => {
  resetBboxToCloud();
});
document.getElementById("profileTool").addEventListener("click", () => {
  profileMode = !profileMode;
  profilePoints = [];
  document.getElementById("profileTool").textContent = profileMode ? "Profile Tool: On" : "Profile Tool";
  metaEl.textContent = profileMode ? "Profile tool: click the first point." : "Profile tool disabled.";
});
document.getElementById("closeProfile").addEventListener("click", () => {
  document.getElementById("profilePanel").classList.remove("active");
  profilePoints = [];
  if (profileChart) {
    profileChart.destroy();
    profileChart = null;
  }
});
loadInfo().then(() => loadMesh(INITIAL_LOD, "terrarium"));
render();
</script>
</body>
</html>
"""


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
            encoded = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
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
            if tessellation not in {"quadtree", "nvb", "diamond48"}:
                self.send_error(400, "tessellation must be 'quadtree', 'nvb', or 'diamond48'")
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
