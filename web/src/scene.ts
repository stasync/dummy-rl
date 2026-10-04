// Three.js view: the MuJoCo robot (one mesh per geom, moved from geom_xpos/geom_xmat each frame),
// an industrial firing range around it, and game-feel effects (sparks, camera shake, tints).
// The whole scene uses MuJoCo's z-up coordinates (Object3D.DEFAULT_UP = +z), so positions from
// physics and raycast hits need no axis conversion. The robot faces +x, toward the shooter.

import * as THREE from 'three';
import type { MainModule, MjData, MjModel } from '@mujoco/mujoco';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const CAMERA_POS = new THREE.Vector3(4.0, -0.5, 1.45); // shooter's eye, ~4 m in front of the robot
const CAMERA_TARGET = new THREE.Vector3(0, 0, 0.8);

export class Scene {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  /** Robot meshes only (raycast targets); mesh.userData.bodyId = MuJoCo body id. */
  readonly robotMeshes: THREE.Mesh[] = [];
  /** Static range geometry: shots that miss the robot hit these (sparks on walls/floor). */
  readonly rangeMeshes: THREE.Object3D[] = [];
  private dynamic: { mesh: THREE.Mesh; geom: number }[] = [];
  private tmp = new THREE.Matrix4();
  private sparks: Sparks;
  private trauma = 0; // camera shake amount, decays over time

  constructor(canvas: HTMLCanvasElement, mj: MainModule, private model: MjModel) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;

    this.scene.background = new THREE.Color(0x111214);
    this.scene.fog = new THREE.Fog(0x111214, 7, 22);

    this.camera = new THREE.PerspectiveCamera(50, 1, 0.05, 100);
    this.camera.position.copy(CAMERA_POS);
    this.camera.lookAt(CAMERA_TARGET);

    this.buildLights();
    this.buildRange();
    this.buildGeoms(mj);
    this.sparks = new Sparks(this.scene);
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  // --- robot --------------------------------------------------------------------------------

  private buildGeoms(mj: MainModule): void {
    const m = this.model;
    const T = mj.mjtGeom;
    const types = m.geom_type, size = m.geom_size, rgba = m.geom_rgba, body = m.geom_bodyid;
    for (let g = 0; g < m.ngeom; g++) {
      if (body[g] === 0) continue; // world geoms (the floor plane) are drawn by buildRange
      const sx = size[g * 3], sy = size[g * 3 + 1], sz = size[g * 3 + 2];
      let geo: THREE.BufferGeometry;
      switch (types[g]) {
        case T.mjGEOM_SPHERE.value:
          geo = new THREE.SphereGeometry(sx, 32, 16);
          break;
        case T.mjGEOM_CAPSULE.value:
          geo = new THREE.CapsuleGeometry(sx, 2 * sy, 8, 24).rotateX(Math.PI / 2); // Three: along y; MuJoCo: along z
          break;
        case T.mjGEOM_CYLINDER.value:
          geo = new THREE.CylinderGeometry(sx, sx, 2 * sy, 32).rotateX(Math.PI / 2);
          break;
        case T.mjGEOM_BOX.value:
          geo = new THREE.BoxGeometry(2 * sx, 2 * sy, 2 * sz);
          break;
        default:
          continue; // no meshes/heightfields in this model
      }
      const color = new THREE.Color(rgba[g * 4], rgba[g * 4 + 1], rgba[g * 4 + 2]);
      const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.05 });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.matrixAutoUpdate = false;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.userData.bodyId = body[g];
      this.scene.add(mesh);
      this.dynamic.push({ mesh, geom: g });
      this.robotMeshes.push(mesh);
    }
  }

  /** Copy every robot geom's world pose into its mesh. geom_xmat is row-major, like Matrix4.set. */
  update(data: MjData): void {
    const p = data.geom_xpos, R = data.geom_xmat;
    for (const { mesh, geom: g } of this.dynamic) {
      const r = g * 9, o = g * 3;
      this.tmp.set(R[r], R[r + 1], R[r + 2], p[o], R[r + 3], R[r + 4], R[r + 5], p[o + 1], R[r + 6], R[r + 7], R[r + 8], p[o + 2], 0, 0, 0, 1);
      mesh.matrix.copy(this.tmp);
      mesh.matrixWorldNeedsUpdate = true;
    }
  }

  /** Emissive tint per MuJoCo body (debug overlay: torque load, foot contact). Empty map clears. */
  setBodyTints(tints: Map<number, THREE.Color>): void {
    for (const mesh of this.robotMeshes) {
      const mat = mesh.material as THREE.MeshStandardMaterial;
      const c = tints.get(mesh.userData.bodyId);
      if (c) mat.emissive.copy(c);
      else mat.emissive.setRGB(0, 0, 0);
    }
  }

  // --- firing range (look pass; original design: concrete, lane paint, hanging work lights) ---

  private buildLights(): void {
    this.scene.add(new THREE.HemisphereLight(0xc9d3dd, 0x2b2622, 0.6));
    const key = new THREE.DirectionalLight(0xfff1e0, 1.6);
    key.position.set(3, -2.5, 6);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    const s = key.shadow.camera;
    s.left = -3; s.right = 3; s.top = 3; s.bottom = -3; s.near = 0.5; s.far = 20;
    key.shadow.bias = -0.0005;
    this.scene.add(key);
  }

  private buildRange(): void {
    const concrete = (shade: number, repeat: number) => {
      const tex = new THREE.CanvasTexture(concreteCanvas(shade));
      tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
      tex.repeat.set(repeat, repeat);
      tex.colorSpace = THREE.SRGBColorSpace;
      return new THREE.MeshStandardMaterial({ map: tex, roughness: 0.95 });
    };
    const add = (mesh: THREE.Mesh, shadows = true) => {
      mesh.receiveShadow = shadows;
      this.scene.add(mesh);
      this.rangeMeshes.push(mesh);
      return mesh;
    };

    // Floor (the physics floor is an infinite plane at z = 0; this is its visible part).
    add(new THREE.Mesh(new THREE.PlaneGeometry(30, 14), concrete(120, 6)));

    // Paint: firing line at the shooter's feet, lane edges, and a stand marker under the robot.
    const paint = (w: number, h: number, x: number, y: number, color: number) => {
      const m = new THREE.Mesh(
        new THREE.PlaneGeometry(w, h),
        new THREE.MeshStandardMaterial({ color, roughness: 0.8, polygonOffset: true, polygonOffsetFactor: -1 }),
      );
      m.position.set(x, y, 0.001);
      add(m);
    };
    paint(0.12, 6, 3.3, 0, 0xd8a31a);
    paint(14, 0.06, -3.5, 1.6, 0xd8a31a);
    paint(14, 0.06, -3.5, -1.6, 0xd8a31a);
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.42, 0.47, 48),
      new THREE.MeshStandardMaterial({ color: 0xd8a31a, roughness: 0.8, polygonOffset: true, polygonOffsetFactor: -1 }),
    );
    ring.position.z = 0.001;
    add(ring);

    // Walls: backstop behind the robot, side walls, ceiling beams.
    const wall = concrete(95, 3);
    const box = (sx: number, sy: number, sz: number, x: number, y: number, z: number, mat: THREE.Material) => {
      const m = add(new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), mat));
      m.position.set(x, y, z);
      m.castShadow = true;
      return m;
    };
    box(0.4, 12, 5, -8, 0, 2.5, wall);
    box(16, 0.4, 5, -1, 5.5, 2.5, wall);
    box(16, 0.4, 5, -1, -5.5, 2.5, wall);
    const steel = new THREE.MeshStandardMaterial({ color: 0x3a3d42, roughness: 0.6, metalness: 0.4 });
    for (let x = -6; x <= 4; x += 2.5) box(0.25, 11, 0.35, x, 0, 4.2, steel);
    // Angled baffle panels on the backstop (classic range look, catches stray rounds).
    for (let z = 0.8; z < 4.5; z += 1.1) {
      const b = box(0.08, 11, 0.7, -7.6, 0, z, steel);
      b.rotation.y = -0.6;
    }

    // Hanging work lights: emissive housings + warm spot lights (no shadows: one shadow caster is enough).
    const housing = new THREE.MeshStandardMaterial({ color: 0x222222, emissive: 0xffd9a0, emissiveIntensity: 1.2 });
    for (const [x, y] of [[0, 0], [-3.5, 0], [2.2, -2.6], [-1.5, 2.8]] as const) {
      const lamp = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.18, 0.06), housing);
      lamp.position.set(x, y, 3.3);
      this.scene.add(lamp);
      const cable = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.9).rotateX(Math.PI / 2), steel);
      cable.position.set(x, y, 3.75);
      this.scene.add(cable);
      const spot = new THREE.SpotLight(0xffd9a0, 18, 9, 0.8, 0.6, 1.6);
      spot.position.set(x, y, 3.25);
      spot.target.position.set(x, y, 0);
      this.scene.add(spot, spot.target);
    }
  }

  // --- effects ------------------------------------------------------------------------------

  spark(point: THREE.Vector3, normal: THREE.Vector3, strength: number): void {
    this.sparks.emit(point, normal, strength);
  }

  /** Camera shake; amount ~0.1 (rifle) .. 1 (cannon). */
  shake(amount: number): void {
    this.trauma = Math.min(1, this.trauma + amount);
  }

  render(dt: number): void {
    this.sparks.update(dt);
    // Shake = random offset scaled by trauma^2 (small hits barely move it, big ones jolt it).
    this.trauma = Math.max(0, this.trauma - dt * 2.5);
    const k = this.trauma * this.trauma * 0.06;
    this.camera.position.set(CAMERA_POS.x, CAMERA_POS.y + (Math.random() * 2 - 1) * k, CAMERA_POS.z + (Math.random() * 2 - 1) * k);
    this.camera.lookAt(CAMERA_TARGET);
    this.renderer.render(this.scene, this.camera);
  }

  private resize(): void {
    const w = window.innerWidth, h = window.innerHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }
}

/** Procedural concrete: grey noise with a few darker stains. */
function concreteCanvas(shade: number): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const g = c.getContext('2d')!;
  const img = g.createImageData(256, 256);
  for (let i = 0; i < 256 * 256; i++) {
    const v = shade + (Math.random() - 0.5) * 22;
    img.data[i * 4] = v;
    img.data[i * 4 + 1] = v * 0.98;
    img.data[i * 4 + 2] = v * 0.95;
    img.data[i * 4 + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  for (let i = 0; i < 12; i++) {
    g.fillStyle = `rgba(0,0,0,${0.03 + Math.random() * 0.05})`;
    g.beginPath();
    g.arc(Math.random() * 256, Math.random() * 256, 10 + Math.random() * 40, 0, Math.PI * 2);
    g.fill();
  }
  return c;
}

/** Hit sparks: one pooled Points object, particles fly out along the surface normal and fade. */
class Sparks {
  private static N = 256;
  private pos = new Float32Array(Sparks.N * 3);
  private col = new Float32Array(Sparks.N * 3);
  private vel = new Float32Array(Sparks.N * 3);
  private life = new Float32Array(Sparks.N);
  private next = 0;
  private geo = new THREE.BufferGeometry();

  constructor(scene: THREE.Scene) {
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    this.geo.setAttribute('color', new THREE.BufferAttribute(this.col, 3));
    const mat = new THREE.PointsMaterial({ size: 0.035, vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
    const pts = new THREE.Points(this.geo, mat);
    pts.frustumCulled = false;
    scene.add(pts);
  }

  emit(p: THREE.Vector3, n: THREE.Vector3, strength: number): void {
    const count = Math.round(8 + 22 * Math.min(1.5, strength));
    for (let k = 0; k < count; k++) {
      const i = this.next;
      this.next = (this.next + 1) % Sparks.N;
      const speed = 1.5 + Math.random() * 3.5 * Math.min(1.5, strength + 0.3);
      const dir = new THREE.Vector3(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).multiplyScalar(1.4).add(n).normalize();
      this.pos.set([p.x, p.y, p.z], i * 3);
      this.vel.set([dir.x * speed, dir.y * speed, dir.z * speed], i * 3);
      this.life[i] = 0.12 + Math.random() * 0.25;
    }
  }

  update(dt: number): void {
    for (let i = 0; i < Sparks.N; i++) {
      if (this.life[i] <= 0) {
        this.col[i * 3] = this.col[i * 3 + 1] = this.col[i * 3 + 2] = 0;
        continue;
      }
      this.life[i] -= dt;
      this.vel[i * 3 + 2] -= 9.81 * dt;
      for (let a = 0; a < 3; a++) this.pos[i * 3 + a] += this.vel[i * 3 + a] * dt;
      const f = Math.max(0, Math.min(1, this.life[i] * 4));
      this.col[i * 3] = 1.0 * f;
      this.col[i * 3 + 1] = 0.7 * f;
      this.col[i * 3 + 2] = 0.25 * f;
    }
    this.geo.attributes.position.needsUpdate = true;
    this.geo.attributes.color.needsUpdate = true;
  }
}
