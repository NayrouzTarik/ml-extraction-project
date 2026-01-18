import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Cbir3dResult, Cbir3dService } from '../../services/cbir3d.service';
import { ObjViewerComponent } from '../obj-viewer/obj-viewer.component';

@Component({
  selector: 'app-cbir3d-search',
  standalone: true,
  imports: [CommonModule, FormsModule, ObjViewerComponent],
  templateUrl: './cbir3d-search.component.html',
  styles: [`
    .card { max-width: 900px; margin: 0 auto; padding: 24px; background: white; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,.08); }
    .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
    .btn { padding: 10px 16px; border: none; border-radius: 10px; cursor: pointer; font-weight: 600; }
    .btn-primary { background: #3182ce; color: #fff; }
    .btn-secondary { background: #edf2f7; color: #1a202c; }
    input[type="number"] { width: 90px; padding: 8px; border-radius: 8px; border: 1px solid #e2e8f0; }
    .muted { color: #64748b; }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; }
    th, td { text-align: left; padding: 10px; border-bottom: 1px solid #edf2f7; }
    .pill { display:inline-block; padding: 4px 10px; border-radius: 999px; background:#f1f5f9; }
    pre { background:#0b1220; color:#e5e7eb; padding:12px; border-radius:10px; overflow:auto; }
  `]
})
export class Cbir3dSearchComponent {
  // query upload
  selectedFile: File | null = null;

  // params
  limit = 50;
  topK = 5;
  nViews = 12;
  imgSize = 256;

  // state
  indexing = false;
  searching = false;

  statusInfo: any = null;
  results: Cbir3dResult[] = [];
  lastQueryName = '';
  errorMsg = '';
  lastRaw: any = null;

  // backend + display
  backendBase = 'http://127.0.0.1:5000';
  queryPreviewUrl = '';
  viewerUrl = '';
  viewerLabel = '';

  constructor(private cbir3d: Cbir3dService) {
    this.refreshStatus();
  }

  onFileChange(ev: Event) {
    const input = ev.target as HTMLInputElement;
    this.selectedFile = input.files && input.files.length ? input.files[0] : null;
  }

  refreshStatus() {
    this.cbir3d.status().subscribe({
      next: (s) => { this.statusInfo = s; this.errorMsg = ''; },
      error: () => { this.errorMsg = 'Backend inaccessible. Lance Flask sur 127.0.0.1:5000'; }
    });
  }

  runIndex() {
    this.indexing = true;
    this.errorMsg = '';
    this.cbir3d.index(this.limit, this.nViews, this.imgSize).subscribe({
      next: (res) => {
        this.lastRaw = res;
        this.indexing = false;
        this.refreshStatus();
      },
      error: (e) => {
        this.indexing = false;
        this.errorMsg = e?.error?.error || 'Erreur lors de l’indexation';
      }
    });
  }

  openModel(modelName: string) {
    this.viewerUrl = this.backendBase + '/api3d/model/' + modelName;
    this.viewerLabel = modelName;
  }

  runSearch() {
    if (!this.selectedFile) {
      this.errorMsg = 'Choisis un fichier .obj (query) avant de lancer la recherche.';
      return;
    }

    this.searching = true;
    this.errorMsg = '';
    this.results = [];
    this.queryPreviewUrl = '';

    this.cbir3d.search(this.selectedFile, this.topK, this.nViews, this.imgSize).subscribe({
      next: (res) => {
        this.lastRaw = res;
        this.lastQueryName = res.query;
        this.results = res.top_k || [];

        // ✅ query preview image (silhouette)
        this.queryPreviewUrl = res.query_preview_url
          ? this.backendBase + res.query_preview_url
          : '';

        // ✅ show query model in viewer (requires /api3d/query/<file> route)
        this.viewerUrl = this.backendBase + '/api3d/query/' + res.query;
        this.viewerLabel = 'Query: ' + res.query;

        this.searching = false;
      },
      error: (e) => {
        this.searching = false;
        this.errorMsg = e?.error?.error || 'Erreur lors de la recherche';
      }
    });
  }
}
