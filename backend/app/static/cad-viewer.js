import * as THREE from "./vendor/three.module.js";
import { OrbitControls } from "./vendor/OrbitControls.js";
import { RoomEnvironment } from "./vendor/RoomEnvironment.js";
import { toCreasedNormals } from "./vendor/BufferGeometryUtils.js";

const BACKGROUND = 0xf3f5f7;
const MIN_VIEW_SIZE = 320;
const CREASE_ANGLE = Math.PI / 5;

function colorComponents(value) {
  const color = new THREE.Color(value || "#9198a2");
  const hsl = {};
  color.getHSL(hsl);
  color.setHSL(
    hsl.h,
    Math.min(hsl.s, 0.08),
    THREE.MathUtils.clamp(0.31 + hsl.l * 0.18, 0.35, 0.52),
  );
  return [color.r, color.g, color.b];
}

function triangulateFaces(faces) {
  const positions = [];
  const colors = [];

  for (const face of faces || []) {
    const points = Array.isArray(face.points) ? face.points : [];
    if (points.length < 3) {
      continue;
    }

    const [red, green, blue] = colorComponents(face.color);
    for (let index = 1; index < points.length - 1; index += 1) {
      for (const point of [points[0], points[index], points[index + 1]]) {
        positions.push(Number(point.x) || 0, Number(point.y) || 0, Number(point.z) || 0);
        colors.push(red, green, blue);
      }
    }
  }

  if (!positions.length) {
    throw new Error("The CAD preview does not contain renderable faces.");
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  return toCreasedNormals(geometry, CREASE_ANGLE);
}

function materialSettings(materialName) {
  const name = String(materialName || "").toLowerCase();
  if (/(steel|aluminium|aluminum|iron|metal)/.test(name)) {
    return { metalness: 0.38, roughness: 0.38, envMapIntensity: 0.8 };
  }
  if (/(rubber|elastomer|epdm|silicone)/.test(name)) {
    return { metalness: 0.0, roughness: 0.56, envMapIntensity: 0.38 };
  }
  return { metalness: 0.04, roughness: 0.52, envMapIntensity: 0.62 };
}

function cameraDirection(cameraState) {
  const pitch = THREE.MathUtils.clamp(Number(cameraState.rotationX) || -0.55, -1.35, 1.35);
  const yaw = Number(cameraState.rotationY) || 0.78;
  const horizontal = Math.cos(pitch);
  return new THREE.Vector3(
    Math.sin(yaw) * horizontal,
    -Math.sin(pitch),
    Math.cos(yaw) * horizontal,
  ).normalize();
}

function addOrientationBadge(container) {
  const badge = document.createElement("div");
  badge.className = "cad-viewer-axis";
  badge.setAttribute("aria-hidden", "true");
  badge.innerHTML = `
    <span class="cad-viewer-axis-line cad-viewer-axis-x"><span class="cad-viewer-axis-label">X</span></span>
    <span class="cad-viewer-axis-line cad-viewer-axis-y"><span class="cad-viewer-axis-label">Y</span></span>
    <span class="cad-viewer-axis-line cad-viewer-axis-z"><span class="cad-viewer-axis-label">Z</span></span>
  `;
  container.appendChild(badge);

  const hint = document.createElement("div");
  hint.className = "cad-viewer-hint";
  hint.textContent = "Drag to orbit · Scroll to zoom";
  container.appendChild(hint);
}

function updateCameraState(camera, target, cameraState) {
  const direction = camera.position.clone().sub(target).normalize();
  cameraState.rotationX = THREE.MathUtils.clamp(
    -Math.asin(THREE.MathUtils.clamp(direction.y, -1, 1)),
    -1.35,
    1.35,
  );
  cameraState.rotationY = Math.atan2(direction.x, direction.z);
  cameraState.zoom = camera.zoom;
}

function disposeMaterial(material) {
  if (Array.isArray(material)) {
    material.forEach(disposeMaterial);
    return;
  }
  material?.dispose();
}

function createContactShadow(radius) {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 128;
  const context = canvas.getContext("2d");
  const gradient = context.createRadialGradient(64, 64, 4, 64, 64, 62);
  gradient.addColorStop(0, "rgba(15, 23, 42, 0.5)");
  gradient.addColorStop(0.42, "rgba(15, 23, 42, 0.25)");
  gradient.addColorStop(1, "rgba(15, 23, 42, 0)");
  context.fillStyle = gradient;
  context.fillRect(0, 0, 128, 128);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const geometry = new THREE.PlaneGeometry(radius * 3.4, radius * 2.5);
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    opacity: 0.52,
    depthWrite: false,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.rotation.x = -Math.PI / 2;
  mesh.renderOrder = 1;
  return { mesh, geometry, material, texture };
}

export function createCadViewer({
  container,
  faces,
  cameraState,
  materialName = "",
}) {
  if (!(container instanceof HTMLElement)) {
    throw new Error("A CAD viewer container is required.");
  }

  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: false,
    preserveDrawingBuffer: true,
    powerPreference: "high-performance",
  });
  renderer.setClearColor(BACKGROUND, 1);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.98;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.domElement.setAttribute("aria-label", "Interactive shaded CAD preview");
  renderer.domElement.setAttribute("role", "img");
  renderer.domElement.tabIndex = 0;

  container.classList.add("viewer3d-webgl");
  container.dataset.renderer = "three";
  container.appendChild(renderer.domElement);
  addOrientationBadge(container);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(BACKGROUND);

  const geometry = triangulateFaces(faces);
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  const bounds = geometry.boundingBox;
  const sphere = geometry.boundingSphere;
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  const radius = Math.max(sphere.radius, 1);

  const settings = materialSettings(materialName);
  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
    metalness: settings.metalness,
    roughness: settings.roughness,
    envMapIntensity: settings.envMapIntensity,
  });
  const solid = new THREE.Mesh(geometry, material);
  solid.castShadow = true;
  solid.receiveShadow = true;
  scene.add(solid);

  const edgeGeometry = new THREE.EdgesGeometry(geometry, 34);
  const edgeMaterial = new THREE.LineBasicMaterial({
    color: 0x1b2634,
    transparent: true,
    opacity: 0.2,
    depthWrite: false,
  });
  const edges = new THREE.LineSegments(edgeGeometry, edgeMaterial);
  edges.renderOrder = 2;
  scene.add(edges);

  const environment = new RoomEnvironment();
  const pmremGenerator = new THREE.PMREMGenerator(renderer);
  scene.environment = pmremGenerator.fromScene(environment, 0.04).texture;
  environment.dispose();
  pmremGenerator.dispose();

  const hemisphere = new THREE.HemisphereLight(0xffffff, 0xaab6c4, 0.78);
  scene.add(hemisphere);

  const keyLight = new THREE.DirectionalLight(0xffffff, 2.6);
  keyLight.position.set(
    center.x + radius * 2.6,
    center.y + radius * 3.4,
    center.z + radius * 2.1,
  );
  keyLight.target.position.copy(center);
  keyLight.intensity = 2.35;
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(2048, 2048);
  keyLight.shadow.bias = -0.00015;
  keyLight.shadow.normalBias = 0.025;
  const shadowExtent = radius * 2.2;
  keyLight.shadow.camera.left = -shadowExtent;
  keyLight.shadow.camera.right = shadowExtent;
  keyLight.shadow.camera.top = shadowExtent;
  keyLight.shadow.camera.bottom = -shadowExtent;
  keyLight.shadow.camera.near = 0.1;
  keyLight.shadow.camera.far = radius * 9;
  scene.add(keyLight, keyLight.target);

  const fillLight = new THREE.DirectionalLight(0xc8dbf4, 0.42);
  fillLight.position.set(center.x - radius * 2, center.y + radius, center.z - radius * 2);
  scene.add(fillLight);

  const groundY = bounds.min.y - Math.max(radius * 0.035, size.y * 0.018, 0.15);
  const groundGeometry = new THREE.PlaneGeometry(radius * 7, radius * 7);
  const groundMaterial = new THREE.ShadowMaterial({
    color: 0x15202e,
    transparent: true,
    opacity: 0.24,
  });
  const ground = new THREE.Mesh(groundGeometry, groundMaterial);
  ground.rotation.x = -Math.PI / 2;
  ground.position.set(center.x, groundY, center.z);
  ground.receiveShadow = true;
  scene.add(ground);

  const contactShadow = createContactShadow(radius);
  contactShadow.mesh.position.set(center.x, groundY + radius * 0.012, center.z);
  scene.add(contactShadow.mesh);

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, radius * 30);
  const target = center.clone();
  const viewDirection = cameraDirection(cameraState);
  camera.position.copy(target).addScaledVector(viewDirection, radius * 4.2);
  camera.up.set(0, 1, 0);
  camera.zoom = THREE.MathUtils.clamp(Number(cameraState.zoom) || 1, 0.55, 2.8);
  camera.lookAt(target);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(target);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.screenSpacePanning = true;
  controls.minZoom = 0.55;
  controls.maxZoom = 3.5;
  controls.zoomToCursor = true;
  controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
  controls.mouseButtons.MIDDLE = THREE.MOUSE.DOLLY;
  controls.mouseButtons.RIGHT = THREE.MOUSE.PAN;
  controls.addEventListener("change", () => updateCameraState(camera, target, cameraState));

  let width = 0;
  let height = 0;
  let frameId = 0;
  let disposed = false;

  function resize() {
    const nextWidth = Math.max(MIN_VIEW_SIZE, Math.floor(container.clientWidth || 720));
    const nextHeight = Math.max(MIN_VIEW_SIZE, Math.floor(container.clientHeight || 470));
    if (nextWidth === width && nextHeight === height) {
      return;
    }
    width = nextWidth;
    height = nextHeight;
    renderer.setSize(width, height, false);
    const aspect = width / height;
    const viewHeight = radius * 2.35;
    camera.left = -viewHeight * aspect / 2;
    camera.right = viewHeight * aspect / 2;
    camera.top = viewHeight / 2;
    camera.bottom = -viewHeight / 2;
    camera.updateProjectionMatrix();
  }

  function animate() {
    if (disposed) {
      return;
    }
    frameId = window.requestAnimationFrame(animate);
    resize();
    controls.update();
    renderer.render(scene, camera);
  }

  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(container);
  resize();
  animate();

  return {
    canvas: renderer.domElement,
    renderer: "three",
    dispose() {
      disposed = true;
      window.cancelAnimationFrame(frameId);
      resizeObserver.disconnect();
      controls.dispose();
      scene.environment?.dispose();
      geometry.dispose();
      material.dispose();
      edgeGeometry.dispose();
      edgeMaterial.dispose();
      groundGeometry.dispose();
      groundMaterial.dispose();
      contactShadow.geometry.dispose();
      contactShadow.material.dispose();
      contactShadow.texture.dispose();
      scene.traverse((object) => {
        if (object !== solid && object !== edges && object !== ground && object !== contactShadow.mesh) {
          object.geometry?.dispose();
          disposeMaterial(object.material);
        }
      });
      renderer.dispose();
      renderer.forceContextLoss();
    },
  };
}
