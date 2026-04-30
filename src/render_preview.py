import trimesh
import base64
from pathlib import Path

def render_shadow_preview_threejs(stl_path, output_path):
    mesh = trimesh.load(stl_path)

    # Normalize mesh
    mesh.apply_translation(-mesh.centroid)
    scale = 1.5 / max(mesh.extents)
    mesh.apply_scale(scale)

    # Export to GLB
    glb_path = str(Path(output_path).with_suffix(".glb"))
    mesh.export(glb_path)
    with open(glb_path, "rb") as f:
        glb_base64 = base64.b64encode(f.read()).decode()

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Shadow Box Preview</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: #0a0a0f; display: flex; align-items: center; justify-content: center; height: 100vh; font-family: monospace; }}
    #info {{ position: absolute; top: 16px; left: 50%; transform: translateX(-50%);
             color: #8899bb; font-size: 11px; letter-spacing: 0.15em; text-transform: uppercase; opacity: 0.7; }}
    canvas {{ display: block; }}
  </style>
</head>
<body>
<div id="info">drag to rotate · scroll to zoom</div>
<script type="module">
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160/build/three.module.js';
import {{ OrbitControls }} from 'https://cdn.jsdelivr.net/npm/three@0.160/examples/jsm/controls/OrbitControls.js';
import {{ GLTFLoader }} from 'https://cdn.jsdelivr.net/npm/three@0.160/examples/jsm/loaders/GLTFLoader.js';

// ─── Scene ────────────────────────────────────────────────────────────────────
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0f);
scene.fog = new THREE.FogExp2(0x0a0a0f, 0.18);

// ─── Camera — orthographic-ish perspective matching photo angle ───────────────
const aspect = window.innerWidth / window.innerHeight;
const camera = new THREE.PerspectiveCamera(38, aspect, 0.1, 100);
// Elevated 3/4 view: slightly left-forward, looking down into the box
camera.position.set(-3.2, 3.0, 3.8);
camera.lookAt(0, 0, 0);

// ─── Renderer ─────────────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
document.body.appendChild(renderer.domElement);

// ─── Orbit controls — constrained to preserve the diorama feel ────────────────
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.minDistance = 4;
controls.maxDistance = 12;
controls.maxPolarAngle = Math.PI * 0.52; // don't go below floor
controls.update();

// ─── Box dimensions ───────────────────────────────────────────────────────────
const BOX = 3.0;   // half-width/height of interior
const HALF = BOX / 2;

// ─── Wall material — warm MDF/kraft-paper tone like photo ─────────────────────
const wallMat = new THREE.MeshStandardMaterial({{
  color: 0xc8a96e,       // warm tan / laser-cut MDF
  roughness: 0.92,
  metalness: 0.0,
  side: THREE.FrontSide,
}});

function addWall(w, h, px, py, pz, rx, ry) {{
  const geo = new THREE.PlaneGeometry(w, h);
  const mesh = new THREE.Mesh(geo, wallMat);
  mesh.position.set(px, py, pz);
  mesh.rotation.set(rx, ry, 0);
  mesh.receiveShadow = true;
  scene.add(mesh);
}}

// Floor
addWall(BOX, BOX,   0,     -HALF,  0,    -Math.PI / 2,  0);
// Back wall (facing +Z camera)
addWall(BOX, BOX,   0,      0,    -HALF,  0,             0);
// Left wall (facing +X camera)
addWall(BOX, BOX,  -HALF,   0,     0,     0,             Math.PI / 2);

// ─── Lighting ─────────────────────────────────────────────────────────────────

// 1. Blue/purple point light — matches the colored LED in the photo (left side)
const blueLight = new THREE.PointLight(0x4466ff, 6.0, 8, 2);
blueLight.position.set(-HALF + 0.2, 0.2, 0.2); // near left wall
blueLight.castShadow = true;
blueLight.shadow.mapSize.set(2048, 2048);
blueLight.shadow.camera.near = 0.05;
blueLight.shadow.camera.far = 10;
blueLight.shadow.radius = 2; // softer penumbra
scene.add(blueLight);

// 2. Warm top spotlight — projects circle on floor like photo's gobo light
const topSpot = new THREE.SpotLight(0xffe8b0, 5.0);
topSpot.position.set(0.1, HALF - 0.05, 0.1); // just below ceiling centre
topSpot.target.position.set(0, -HALF, 0);
topSpot.angle = 0.38;
topSpot.penumbra = 0.45;
topSpot.decay = 1.8;
topSpot.castShadow = true;
topSpot.shadow.mapSize.set(2048, 2048);
topSpot.shadow.camera.near = 0.1;
topSpot.shadow.camera.far = 10;
scene.add(topSpot);
scene.add(topSpot.target);

// 3. Very dim ambient so shadowed surfaces aren't pitch black
const ambient = new THREE.AmbientLight(0x1a1a2e, 1.8);
scene.add(ambient);

// ─── Suspended object ─────────────────────────────────────────────────────────
const loader = new GLTFLoader();
const data = "data:model/gltf-binary;base64,{glb_base64}";

loader.load(data, (gltf) => {{
  const obj = gltf.scene;

  // Center vertically in the box (slightly above mid)
  const box3 = new THREE.Box3().setFromObject(obj);
  const center = new THREE.Vector3();
  box3.getCenter(center);
  obj.position.sub(center);
  obj.position.y += 0.15; // hang slightly above mid-height

  obj.traverse((child) => {{
    if (child.isMesh) {{
      child.castShadow = true;
      child.receiveShadow = false;
      // Blue-tinted grey plastic — matches the 3D-printed piece in the photo
      child.material = new THREE.MeshStandardMaterial({{
        color: 0x4477cc,
        roughness: 0.45,
        metalness: 0.15,
      }});
    }}
  }});

  scene.add(obj);

  // ── Suspension wires ────────────────────────────────────────────────────────
  // 4 thin lines from object top to ceiling, like the photo's strings
  const wireMat = new THREE.LineBasicMaterial({{ color: 0xffffff, opacity: 0.35, transparent: true }});
  const topY = HALF - 0.01;
  const objTopY = 0.15 + 0.75; // approximate

  [[-0.3, -0.3], [0.3, -0.3], [-0.3, 0.3], [0.3, 0.3]].forEach(([x, z]) => {{
    const pts = [
      new THREE.Vector3(x * 0.6, objTopY, z * 0.6),
      new THREE.Vector3(x * 0.8, topY, z * 0.8),
    ];
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    scene.add(new THREE.Line(geo, wireMat));
  }});
}});

// ─── Box frame edges — laser-cut finger-joint look ────────────────────────────
// Thin dark lines along box corners
const edgeMat = new THREE.LineBasicMaterial({{ color: 0x4a3520, opacity: 0.8, transparent: true }});
const corners = [
  // vertical edges
  [[-HALF,-HALF,-HALF],[-HALF, HALF,-HALF]],
  [[-HALF,-HALF,-HALF],[-HALF,-HALF, HALF]],
  [[-HALF,-HALF,-HALF],[ HALF,-HALF,-HALF]], // along floor back
];
corners.forEach(([a, b]) => {{
  const geo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(...a), new THREE.Vector3(...b)
  ]);
  scene.add(new THREE.Line(geo, edgeMat));
}});

// ─── Animate ──────────────────────────────────────────────────────────────────
function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();

// ─── Resize ───────────────────────────────────────────────────────────────────
window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body>
</html>"""

    html_path = output_path.replace(".png", ".html")
    with open(html_path, "w") as f:
        f.write(html)
    return html_path