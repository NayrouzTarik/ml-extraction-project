import { CommonModule } from '@angular/common';
import { Component, ElementRef, Input, OnChanges, OnDestroy, SimpleChanges, ViewChild } from '@angular/core';

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';

@Component({
  selector: 'app-obj-viewer',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="viewer-card">
      <div class="viewer-header">
        <div class="title">3D Viewer</div>
        <div class="muted">{{ label }}</div>
      </div>
      <div #canvasHost class="viewer"></div>
    </div>
  `,
  styles: [`
    .viewer-card { background:#fff; border-radius:12px; box-shadow:0 2px 12px rgba(0,0,0,.08); overflow:hidden; }
    .viewer-header { padding:12px 16px; border-bottom:1px solid #edf2f7; display:flex; justify-content:space-between; gap:12px; }
    .title { font-weight:800; }
    .muted { color:#64748b; font-size:12px; }
    .viewer { width: 100%; height: 420px; }
  `]
})
export class ObjViewerComponent implements OnChanges, OnDestroy {
  @Input() objUrl: string = '';
  @Input() label: string = '';

  @ViewChild('canvasHost', { static: true }) canvasHost!: ElementRef<HTMLDivElement>;

  private renderer!: THREE.WebGLRenderer;
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private controls!: OrbitControls;
  private loader = new OBJLoader();
  private currentObj: THREE.Object3D | null = null;
  private animId: number | null = null;

  ngOnChanges(changes: SimpleChanges): void {
    if (!this.renderer) this.initThree();
    if (changes['objUrl'] && this.objUrl) this.loadObj(this.objUrl);
  }

  private initThree() {
    const host = this.canvasHost.nativeElement;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xf7f7f7);

    const w = host.clientWidth || 600;
    const h = host.clientHeight || 420;

    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1000);
    this.camera.position.set(0, 1, 3);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(w, h);
    host.innerHTML = '';
    host.appendChild(this.renderer.domElement);

    // Lights
    const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 1.0);
    this.scene.add(hemi);

    const dir = new THREE.DirectionalLight(0xffffff, 1.2);
    dir.position.set(3, 5, 2);
    this.scene.add(dir);

    // Ground subtle
    const grid = new THREE.GridHelper(10, 10, 0xdddddd, 0xeeeeee);
    (grid.material as THREE.Material).opacity = 0.25;
    (grid.material as THREE.Material).transparent = true;
    this.scene.add(grid);

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;

    // Resize handling
    const ro = new ResizeObserver(() => this.onResize());
    ro.observe(host);

    this.animate();
  }

  private onResize() {
    if (!this.renderer || !this.camera) return;
    const host = this.canvasHost.nativeElement;
    const w = host.clientWidth || 600;
    const h = host.clientHeight || 420;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  private loadObj(url: string) {
    // remove previous
    if (this.currentObj) {
      this.scene.remove(this.currentObj);
      this.currentObj = null;
    }

    this.loader.load(
      url,
      (obj) => {
        // normalize scale + center
        const box = new THREE.Box3().setFromObject(obj);
        const size = new THREE.Vector3();
        box.getSize(size);
        const center = new THREE.Vector3();
        box.getCenter(center);

        obj.position.sub(center);

        const maxDim = Math.max(size.x, size.y, size.z) || 1;
        const scale = 1.5 / maxDim;
        obj.scale.setScalar(scale);

        // nicer material
        obj.traverse((child: any) => {
          if (child.isMesh) {
            child.material = new THREE.MeshStandardMaterial({ color: 0x666666, roughness: 0.8, metalness: 0.1 });
          }
        });

        this.currentObj = obj;
        this.scene.add(obj);

        // reset camera
        this.camera.position.set(0, 1, 3);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
      },
      undefined,
      (err) => console.error('OBJ load error', err)
    );
  }

  private animate() {
    this.animId = requestAnimationFrame(() => this.animate());
    this.controls?.update();
    this.renderer?.render(this.scene, this.camera);
  }

  ngOnDestroy(): void {
    if (this.animId) cancelAnimationFrame(this.animId);
    this.renderer?.dispose();
  }
}
