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
  tessellation: "none",
  blurRadiusM: 160,
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
let profileSurfaceCache = null;

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
  };
}

function writeBlendSettingsToForm(settings) {
  document.getElementById("blendStrategy").value = settings.strategy;
  document.getElementById("tessellationStrategy").value = settings.tessellation;
  document.getElementById("blurRadiusM").value = String(settings.blurRadiusM);
  updateSliderLabels();
  updateBlendModalVisibility();
}

function openBlendModal() {
  writeBlendSettingsToForm(appliedBlendSettings);
  document.getElementById("blendModal").classList.add("active");
}

function closeBlendModal() {
  document.getElementById("blendModal").classList.remove("active");
}

function updateBlendModalVisibility() {
  document.getElementById("blurRadiusRow").style.display = document.getElementById("blendStrategy").value === "blur" ? "grid" : "none";
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
  const blendControls = ["blendStrategy", "tessellationStrategy", "blurRadiusM", "acceptBlendSettings", "cancelBlendSettings"].map(id => document.getElementById(id));
  const terrariumControls = ["lod", "useManualBbox", "acceptTerrariumSettings", "cancelTerrariumSettings"].map(id => document.getElementById(id));
  const cloudControls = ["cloudLasDir", "cloudSourceCrs", "cloudPixelSize", "cloudChunkSize", "acceptCloudSettings", "cancelCloudSettings"].map(id => document.getElementById(id));
  if (Number.isInteger(forcedLod)) {
    requestedTerrariumSettings.lod = forcedLod;
    lod.value = String(forcedLod);
  }
  const z = String(requestedTerrariumSettings.lod);
  const refineQuery = requestedBlendSettings.tessellation !== "none" ? "&refine=1" : "";
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
    profileSurfaceCache = null;
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
document.getElementById("blendStrategy").addEventListener("change", updateBlendModalVisibility);
updateSliderLabels();
updateBlendModalVisibility();

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

function pointInTriangle(x, y, t) {
  const ax = t[0], ay = t[1], bx = t[3], by = t[4], cx = t[6], cy = t[7];
  const den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
  if (Math.abs(den) < 1e-12) return null;
  const w0 = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / den;
  const w1 = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / den;
  const w2 = 1 - w0 - w1;
  const eps = -1e-8;
  if (w0 < eps || w1 < eps || w2 < eps) return null;
  return w0 * t[2] + w1 * t[5] + w2 * t[8];
}

function sampleRegularMesh(grid, values, x, y) {
  if (!grid || !values || !grid.nx || !grid.ny || grid.nx < 2 || grid.ny < 2) return null;
  const nx = grid.nx, ny = grid.ny;
  const dx = Math.abs(grid.xs[1] - grid.xs[0]);
  const dy = Math.abs(grid.ys[1] - grid.ys[0]);
  const col = Math.floor((x - grid.xs[0]) / dx);
  const row = Math.floor((grid.ys[0] - y) / dy);
  if (col < 0 || row < 0 || col >= nx - 1 || row >= ny - 1) return null;
  const idx = (r, c) => r * nx + c;
  const a = [grid.xs[col], grid.ys[row], values[idx(row, col)]];
  const b = [grid.xs[col + 1], grid.ys[row], values[idx(row, col + 1)]];
  const c = [grid.xs[col], grid.ys[row + 1], values[idx(row + 1, col)]];
  const d = [grid.xs[col + 1], grid.ys[row + 1], values[idx(row + 1, col + 1)]];
  const tri1 = [a, c, b];
  const tri2 = [b, c, d];
  for (const tri of [tri1, tri2]) {
    if (!tri.every(p => finite(p[2]))) continue;
    const z = pointInTriangle(x, y, [tri[0][0], tri[0][1], tri[0][2], tri[1][0], tri[1][1], tri[1][2], tri[2][0], tri[2][1], tri[2][2]]);
    if (finite(z)) return z;
  }
  return null;
}

function refinedLayerId() {
  const strategy = blendStrategy();
  if (strategy === "naive") return "cloud_replacement";
  if (strategy === "blur") return "blur_blend";
  if (strategy === "vertical_distance") return "vertical_distance_blend";
  return "distance_blend";
}

function trianglesFromRefinedLayer(layer, geometry) {
  const triangles = [];
  if (!layer) return triangles;
  if (geometry === "triangles") {
    for (const t of layer.triangles || []) {
      triangles.push([t[0], t[1], t[6], t[2], t[3], t[7], t[4], t[5], t[8]]);
    }
  } else {
    for (const q of layer.quads || []) {
      const a = [q[0], q[1], q[4]], b = [q[2], q[1], q[5]], c = [q[0], q[3], q[6]], d = [q[2], q[3], q[7]];
      triangles.push([a[0], a[1], a[2], c[0], c[1], c[2], b[0], b[1], b[2]]);
      triangles.push([b[0], b[1], b[2], c[0], c[1], c[2], d[0], d[1], d[2]]);
    }
  }
  return triangles;
}

function buildTriangleIndex(triangles) {
  if (!triangles.length) return null;
  let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
  for (const t of triangles) {
    minx = Math.min(minx, t[0], t[3], t[6]); maxx = Math.max(maxx, t[0], t[3], t[6]);
    miny = Math.min(miny, t[1], t[4], t[7]); maxy = Math.max(maxy, t[1], t[4], t[7]);
  }
  const bins = 64, cells = Array.from({length: bins * bins}, () => []);
  const sx = bins / Math.max(maxx - minx, 1e-9), sy = bins / Math.max(maxy - miny, 1e-9);
  function clamp(v) { return Math.max(0, Math.min(bins - 1, v)); }
  triangles.forEach((t, i) => {
    const tminx = Math.min(t[0], t[3], t[6]), tmaxx = Math.max(t[0], t[3], t[6]);
    const tminy = Math.min(t[1], t[4], t[7]), tmaxy = Math.max(t[1], t[4], t[7]);
    const c0 = clamp(Math.floor((tminx - minx) * sx)), c1 = clamp(Math.floor((tmaxx - minx) * sx));
    const r0 = clamp(Math.floor((tminy - miny) * sy)), r1 = clamp(Math.floor((tmaxy - miny) * sy));
    for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) cells[r * bins + c].push(i);
  });
  return {triangles, minx, miny, maxx, maxy, bins, cells, sx, sy};
}

function sampleTriangleIndex(index, x, y) {
  if (!index || x < index.minx || x > index.maxx || y < index.miny || y > index.maxy) return null;
  const c = Math.max(0, Math.min(index.bins - 1, Math.floor((x - index.minx) * index.sx)));
  const r = Math.max(0, Math.min(index.bins - 1, Math.floor((y - index.miny) * index.sy)));
  for (const i of index.cells[r * index.bins + c]) {
    const z = pointInTriangle(x, y, index.triangles[i]);
    if (finite(z)) return z;
  }
  return null;
}

function profileSurfaces() {
  if (profileSurfaceCache) return profileSurfaceCache;
  const refined = current.refined_mesh && current.refined_mesh.applied;
  let blended = {type: "regular", grid: currentTerrainGrid, values: currentBlendValues()};
  if (refined) {
    const layer = current.refined_mesh.layers[refinedLayerId()];
    blended = {type: "indexed", index: buildTriangleIndex(trianglesFromRefinedLayer(layer, current.refined_mesh.geometry))};
  }
  profileSurfaceCache = {
    terrarium: {type: "regular", grid: currentTerrainGrid, values: current.terrarium},
    cloud: {type: "regular", grid: currentCloudGrid, values: currentCloudGrid.z},
    blended
  };
  return profileSurfaceCache;
}

function sampleSurface(surface, x, y) {
  if (!surface) return null;
  if (surface.type === "indexed") return sampleTriangleIndex(surface.index, x, y);
  return sampleRegularMesh(surface.grid, surface.values, x, y);
}

function buildProfileSamples(a, b) {
  const distance = Math.hypot(b.x - a.x, b.y - a.y);
  const sampleStepM = 0.5;
  const steps = Math.max(1, Math.ceil(distance / sampleStepM));
  const terrarium = [];
  const cloud = [];
  const blended = [];
  const surfaces = profileSurfaces();
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const x = a.x + (b.x - a.x) * t;
    const y = a.y + (b.y - a.y) * t;
    const d = distance * t;
    terrarium.push({x: d, y: sampleSurface(surfaces.terrarium, x, y)});
    cloud.push({x: d, y: sampleSurface(surfaces.cloud, x, y)});
    blended.push({x: d, y: sampleSurface(surfaces.blended, x, y)});
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

