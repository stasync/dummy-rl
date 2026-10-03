// Three.js view of the MuJoCo model: one mesh per geom, moved from geom_xpos/geom_xmat each frame.
// The whole scene uses MuJoCo's z-up coordinates directly (Object3D.DEFAULT_UP = +z), so
// positions from physics and raycast hits need no axis conversion.

import * as THREE from 'three';
import type { MainModule, MjData, MjModel } from '@mujoco/mujoco';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

export class Scene {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  /** Robot meshes only (raycast targets); mesh.userData.bodyId = MuJoCo body id. */
  readonly robotMeshes: THREE.Mesh[] = [];
  private dynamic: { mesh: THREE.Mesh; geom: number }[] = [];
  private tmp = new THREE.Matrix4();

  constructor(canvas: HTMLCanvasElement, mj: MainModule, private model: MjModel) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.scene.background = new THREE.Color(0x15171a);
    this.scene.fog = new THREE.Fog(0x15171a, 8, 30);

    // Shooter's view: standing ~4 m in front of the robot (+x), eye height, aiming at its chest.
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.05, 100);
    this.camera.position.set(4.0, -0.6, 1.45);
    this.camera.lookAt(0, 0, 0.8);

    this.scene.add(new THREE.HemisphereLight(0xdfe6ee, 0x2a2622, 1.2));
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(3, -2, 6);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    const s = key.shadow.camera;
    s.left = -3; s.right = 3; s.top = 3; s.bottom = -3; s.near = 0.5; s.far = 20;
    this.scene.add(key);

    this.buildGeoms(mj);
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  private buildGeoms(mj: MainModule): void {
    const m = this.model;
    const T = mj.mjtGeom;
    const types = m.geom_type, size = m.geom_size, rgba = m.geom_rgba, matid = m.geom_matid, matRgba = m.mat_rgba, body = m.geom_bodyid;
    for (let g = 0; g < m.ngeom; g++) {
      const sx = size[g * 3], sy = size[g * 3 + 1], sz = size[g * 3 + 2];
      let geo: THREE.BufferGeometry;
      switch (types[g]) {
        case T.mjGEOM_PLANE.value:
          geo = new THREE.PlaneGeometry(2 * sx, 2 * sy);
          break;
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
      const src = matid[g] >= 0 ? matRgba : rgba;
      const o = (matid[g] >= 0 ? matid[g] : g) * 4;
      const isFloor = body[g] === 0;
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(src[o], src[o + 1], src[o + 2]),
        roughness: isFloor ? 0.95 : 0.6,
        metalness: 0.0,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.matrixAutoUpdate = false;
      mesh.receiveShadow = true;
      mesh.castShadow = !isFloor;
      mesh.userData.bodyId = body[g];
      this.scene.add(mesh);
      if (isFloor) {
        this.place(mesh, g, null);
      } else {
        this.dynamic.push({ mesh, geom: g });
        this.robotMeshes.push(mesh);
      }
    }
  }

  /** Copy a geom's world pose into its mesh. geom_xmat is row-major, like Matrix4.set's arguments. */
  private place(mesh: THREE.Mesh, g: number, data: MjData | null): void {
    const p = data ? data.geom_xpos : this.model.geom_pos;
    const R = data ? data.geom_xmat : null;
    if (R) {
      const r = g * 9;
      this.tmp.set(R[r], R[r + 1], R[r + 2], p[g * 3], R[r + 3], R[r + 4], R[r + 5], p[g * 3 + 1], R[r + 6], R[r + 7], R[r + 8], p[g * 3 + 2], 0, 0, 0, 1);
    } else {
      this.tmp.makeTranslation(p[g * 3], p[g * 3 + 1], p[g * 3 + 2]); // static world geoms: floor is axis-aligned
    }
    mesh.matrix.copy(this.tmp);
    mesh.matrixWorldNeedsUpdate = true;
  }

  update(data: MjData): void {
    for (const { mesh, geom } of this.dynamic) this.place(mesh, geom, data);
  }

  render(): void {
    this.renderer.render(this.scene, this.camera);
  }

  private resize(): void {
    const w = window.innerWidth, h = window.innerHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }
}
