import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { ApiService } from '../../services/api.service';
import { DescriptorsViewerComponent } from '../descriptors-viewer/descriptors-viewer.component';

@Component({
  selector: 'app-upload',
  standalone: true,
  imports: [CommonModule, FormsModule, DescriptorsViewerComponent],
  template: `
    <div class="upload-page">
      <div class="page-header">
        <h1>Upload et Analyse d'Image</h1>
        <p class="subtitle">Uploadez une image pour détecter les objets et analyser leurs descripteurs visuels</p>
      </div>

      <div class="content-container">
        <!-- Upload Zone -->
        <div class="upload-card">
          <div class="upload-zone" 
               [class.dragover]="isDragging"
               (dragover)="onDragOver($event)"
               (dragleave)="onDragLeave($event)"
               (drop)="onDrop($event)">
            <input 
              type="file" 
              #fileInput 
              accept="image/*" 
              (change)="onFileSelected($event)"
              class="file-input"
            >
            
            <div *ngIf="!uploading && !currentImage" class="upload-placeholder">
              <svg class="upload-icon" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
              <h3>Glissez-déposez une image ici</h3>
              <p>ou</p>
              <button class="btn btn-primary" (click)="fileInput.click()">
                Parcourir les fichiers
              </button>
              <p class="file-info">Formats acceptés: JPG, PNG, GIF (Max 16MB)</p>
            </div>

            <div *ngIf="uploading" class="upload-progress">
              <div class="spinner"></div>
              <p>Traitement de l'image en cours...</p>
            </div>

            <div *ngIf="uploadError" class="error-alert">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="8" x2="12" y2="12"></line>
                <line x1="12" y1="16" x2="12.01" y2="16"></line>
              </svg>
              <span>{{ uploadError }}</span>
            </div>
          </div>
        </div>

        <!-- Current Image Analysis -->
        <div *ngIf="currentImage && currentImageDescriptor" class="analysis-card">
          <div class="card-header">
            <h2>Analyse de l'image</h2>
            <button class="btn-icon" (click)="clearImage()" title="Réinitialiser">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
          </div>

          <div class="image-preview">
            <img [src]="getImageUrl(currentImage.filepath)" [alt]="currentImage.filename" class="preview-image">
          </div>

          <div class="detection-results">
            <div class="detection-header">
              <h3>Objets détectés ({{ currentImageDescriptor.objects.length }})</h3>
            </div>

            <div *ngFor="let obj of currentImageDescriptor.objects; let i = index" class="detection-item">
              <div class="detection-badge">
                <span class="object-number">#{{ i + 1 }}</span>
                <span class="object-class">{{ obj.class_name }}</span>
                <span class="confidence-score">{{ (obj.confidence_score * 100).toFixed(1) }}%</span>
              </div>

              <div class="descriptors-container">
                <app-descriptors-viewer 
                  [descriptors]="getObjectDescriptors(obj.object_id)"
                  [objectInfo]="obj">
                </app-descriptors-viewer>
              </div>
            </div>
          </div>

          <div class="action-bar">
            <button 
              class="btn btn-primary btn-large" 
              (click)="findSimilar()"
              [disabled]="searching || !currentImageDescriptor.objects.length">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="11" cy="11" r="8"></circle>
                <path d="m21 21-4.35-4.35"></path>
              </svg>
              {{ searching ? 'Recherche en cours...' : 'Rechercher des images similaires' }}
            </button>
          </div>
        </div>

        <!-- Search Results -->
        <div *ngIf="searchResults && searchResults.results.length > 0" class="results-card">
          <div class="card-header">
            <h2>Images similaires (Top 5)</h2>
            <span class="results-count">{{ searchResults.num_results }} résultat(s)</span>
          </div>

          <div class="results-grid">
            <div *ngFor="let result of searchResults.results; let i = index" class="result-item">
              <div class="result-card-header">
                <span class="rank">#{{ i + 1 }}</span>
                <span class="similarity">{{ (result.similarity * 100).toFixed(1) }}%</span>
              </div>

              <div class="result-image-wrapper">
                <img [src]="getImageUrl(result.filepath, result.source)" [alt]="result.filename" class="result-image">
              </div>

              <div class="result-meta">
                <p class="result-filename">{{ result.filename }}</p>
                <span class="result-category">{{ result.class_name }}</span>
              </div>

              <button 
                class="btn btn-secondary btn-fullwidth" 
                (click)="toggleDetails(result.object_id)">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                  <circle cx="12" cy="12" r="3"></circle>
                </svg>
                {{ selectedDetailsId === result.object_id ? 'Masquer' : 'Détails' }}
              </button>

              <div *ngIf="selectedDetailsId === result.object_id" class="result-details-expanded">
                <div class="details-metrics">
                  <div class="metric">
                    <span class="metric-label">Distance</span>
                    <span class="metric-value">{{ result.distance.toFixed(4) }}</span>
                  </div>
                  <div class="metric">
                    <span class="metric-label">Similarité</span>
                    <span class="metric-value">{{ (result.similarity * 100).toFixed(2) }}%</span>
                  </div>
                  <div class="metric">
                    <span class="metric-label">Confiance</span>
                    <span class="metric-value">{{ (result.confidence_score * 100).toFixed(1) }}%</span>
                  </div>
                </div>

                <div class="result-descriptors">
                  <h4>Descripteurs visuels</h4>
                  <app-descriptors-viewer 
                    [descriptors]="getObjectDescriptors(result.object_id)"
                    [objectInfo]="result">
                  </app-descriptors-viewer>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .upload-page {
      min-height: 100vh;
      background: #f8fafc;
      padding: 40px 20px;
    }

    .page-header {
      max-width: 1200px;
      margin: 0 auto 40px;
      text-align: center;
    }

    .page-header h1 {
      font-size: 36px;
      font-weight: 700;
      color: #1a202c;
      margin-bottom: 12px;
    }

    .subtitle {
      font-size: 18px;
      color: #64748b;
    }

    .content-container {
      max-width: 1200px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    .upload-card, .analysis-card, .results-card {
      background: white;
      border-radius: 16px;
      padding: 32px;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
      border: 1px solid #e2e8f0;
    }

    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 2px solid #f1f5f9;
    }

    .card-header h2 {
      font-size: 24px;
      font-weight: 600;
      color: #1a202c;
      margin: 0;
    }

    .btn-icon {
      background: none;
      border: none;
      color: #64748b;
      cursor: pointer;
      padding: 8px;
      border-radius: 8px;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .btn-icon:hover {
      background: #f1f5f9;
      color: #1a202c;
    }

    .upload-zone {
      border: 2px dashed #cbd5e1;
      border-radius: 12px;
      padding: 60px 20px;
      text-align: center;
      transition: all 0.3s ease;
      background: #f8fafc;
    }

    .upload-zone.dragover {
      border-color: #4299e1;
      background: #ebf8ff;
    }

    .file-input {
      display: none;
    }

    .upload-placeholder {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 16px;
    }

    .upload-icon {
      color: #94a3b8;
      margin-bottom: 8px;
    }

    .upload-placeholder h3 {
      font-size: 20px;
      font-weight: 600;
      color: #334155;
      margin: 0;
    }

    .upload-placeholder p {
      color: #64748b;
      margin: 0;
    }

    .file-info {
      font-size: 14px;
      color: #94a3b8;
    }

    .upload-progress {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 16px;
    }

    .spinner {
      border: 4px solid #e2e8f0;
      border-top: 4px solid #4299e1;
      border-radius: 50%;
      width: 48px;
      height: 48px;
      animation: spin 1s linear infinite;
    }

    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }

    .error-alert {
      display: flex;
      align-items: center;
      gap: 12px;
      background: #fee2e2;
      color: #991b1b;
      padding: 16px;
      border-radius: 8px;
      border: 1px solid #fecaca;
    }

    .image-preview {
      margin-bottom: 32px;
      text-align: center;
    }

    .preview-image {
      max-width: 100%;
      max-height: 500px;
      border-radius: 12px;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .detection-results {
      margin-bottom: 32px;
    }

    .detection-header {
      margin-bottom: 24px;
    }

    .detection-header h3 {
      font-size: 20px;
      font-weight: 600;
      color: #1a202c;
    }

    .detection-item {
      margin-bottom: 32px;
      padding-bottom: 32px;
      border-bottom: 1px solid #e2e8f0;
    }

    .detection-item:last-child {
      border-bottom: none;
      margin-bottom: 0;
      padding-bottom: 0;
    }

    .detection-badge {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 20px;
    }

    .object-number {
      background: #1a202c;
      color: white;
      padding: 6px 12px;
      border-radius: 6px;
      font-weight: 600;
      font-size: 14px;
    }

    .object-class {
      background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
      color: white;
      padding: 6px 16px;
      border-radius: 20px;
      font-weight: 500;
      font-size: 14px;
    }

    .confidence-score {
      background: #dcfce7;
      color: #166534;
      padding: 6px 12px;
      border-radius: 6px;
      font-weight: 600;
      font-size: 14px;
    }

    .descriptors-container {
      margin-top: 20px;
    }

    .action-bar {
      margin-top: 32px;
      padding-top: 32px;
      border-top: 2px solid #f1f5f9;
      text-align: center;
    }

    .btn {
      padding: 12px 24px;
      border: none;
      border-radius: 8px;
      font-size: 16px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      text-decoration: none;
    }

    .btn-primary {
      background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
      color: white;
    }

    .btn-primary:hover:not(:disabled) {
      background: linear-gradient(135deg, #3182ce 0%, #2c5282 100%);
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(66, 153, 225, 0.4);
    }

    .btn-secondary {
      background: #f1f5f9;
      color: #334155;
    }

    .btn-secondary:hover {
      background: #e2e8f0;
    }

    .btn-large {
      padding: 16px 32px;
      font-size: 18px;
    }

    .btn-fullwidth {
      width: 100%;
      justify-content: center;
    }

    .btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }

    .results-count {
      color: #64748b;
      font-size: 14px;
      font-weight: 500;
    }

    .results-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 24px;
    }

    .result-item {
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 16px;
      transition: all 0.2s;
    }

    .result-item:hover {
      border-color: #4299e1;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }

    .result-card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }

    .rank {
      background: #1a202c;
      color: white;
      padding: 4px 10px;
      border-radius: 6px;
      font-weight: 600;
      font-size: 12px;
    }

    .similarity {
      background: #dcfce7;
      color: #166534;
      padding: 4px 10px;
      border-radius: 6px;
      font-weight: 600;
      font-size: 12px;
    }

    .result-image-wrapper {
      width: 100%;
      padding-top: 75%;
      position: relative;
      border-radius: 8px;
      overflow: hidden;
      background: #e2e8f0;
      margin-bottom: 12px;
    }

    .result-image {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
    }

    .result-meta {
      margin-bottom: 12px;
    }

    .result-filename {
      font-weight: 600;
      color: #1a202c;
      font-size: 14px;
      margin-bottom: 8px;
      word-break: break-word;
    }

    .result-category {
      display: inline-block;
      background: #e0e7ff;
      color: #4338ca;
      padding: 4px 10px;
      border-radius: 12px;
      font-size: 12px;
      font-weight: 500;
    }

    .result-details-expanded {
      margin-top: 16px;
      padding-top: 16px;
      border-top: 1px solid #e2e8f0;
    }

    .details-metrics {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-bottom: 20px;
    }

    .metric {
      background: white;
      padding: 12px;
      border-radius: 8px;
      text-align: center;
      border: 1px solid #e2e8f0;
    }

    .metric-label {
      display: block;
      font-size: 12px;
      color: #64748b;
      margin-bottom: 4px;
      text-transform: uppercase;
      font-weight: 500;
    }

    .metric-value {
      display: block;
      font-size: 18px;
      font-weight: 600;
      color: #1a202c;
    }

    .result-descriptors h4 {
      font-size: 16px;
      font-weight: 600;
      color: #1a202c;
      margin-bottom: 16px;
    }

    @media (max-width: 768px) {
      .results-grid {
        grid-template-columns: 1fr;
      }

      .details-metrics {
        grid-template-columns: 1fr;
      }
    }
  `]
})
export class UploadComponent implements OnInit {
  currentImage: any = null;
  currentImageDescriptor: any = null;
  uploading = false;
  uploadError: string | null = null;
  searching = false;
  searchResults: any = null;
  selectedDetailsId: number | null = null;
  objectDescriptorsCache: Map<number, any> = new Map();
  isDragging = false;

  constructor(private apiService: ApiService, private router: Router) {}

  ngOnInit() {
    // Vérifier si un image_id est passé en paramètre
    const params = new URLSearchParams(window.location.search);
    const imageId = params.get('image_id');
    if (imageId) {
      this.loadImageFromId(Number(imageId));
    }
  }

  loadImageFromId(imageId: number) {
    this.apiService.getImages().subscribe({
      next: (response) => {
        const image = response.images.find((img: any) => img.image_id === imageId);
        if (image) {
          this.currentImage = {
            image_id: image.image_id,
            filename: image.filename,
            filepath: image.filepath
          };
          this.loadImageDescriptors(image.image_id);
        }
      },
      error: (error) => {
        console.error('Error loading image:', error);
      }
    });
  }

  onDragOver(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging = true;
  }

  onDragLeave(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging = false;
  }

  onDrop(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging = false;

    const files = event.dataTransfer?.files;
    if (files && files.length > 0) {
      this.uploadImage(files[0]);
    }
  }

  onFileSelected(event: Event) {
    const target = event.target as HTMLInputElement;
    if (target.files && target.files.length > 0) {
      this.uploadImage(target.files[0]);
    }
  }

  uploadImage(file: File) {
    this.uploading = true;
    this.uploadError = null;
    this.currentImage = null;
    this.currentImageDescriptor = null;
    this.searchResults = null;
    this.selectedDetailsId = null;
    this.objectDescriptorsCache.clear();

    this.apiService.uploadSingleImage(file).subscribe({
      next: (response: any) => {
        this.uploading = false;
        if (response.images && response.images.length > 0) {
          const uploadedImage = response.images[0];
          this.currentImage = {
            image_id: uploadedImage.image_id,
            filename: uploadedImage.filename,
            filepath: uploadedImage.filepath || `uploads/${uploadedImage.filename}`
          };
          this.loadImageDescriptors(uploadedImage.image_id);
        }
      },
      error: (error) => {
        this.uploading = false;
        this.uploadError = error.error?.error || 'Erreur lors de l\'upload';
      }
    });
  }

  loadImageDescriptors(imageId: number) {
    this.apiService.getImageDescriptors(imageId, false).subscribe({
      next: (descriptors) => {
        this.currentImageDescriptor = descriptors;
        descriptors.objects.forEach((obj: any) => {
          if (obj.object_id) {
            this.loadObjectDescriptors(obj.object_id);
          }
        });
      },
      error: (error) => {
        console.error('Error loading descriptors:', error);
      }
    });
  }

  loadObjectDescriptors(objectId: number) {
    if (this.objectDescriptorsCache.has(objectId)) {
      return;
    }

    this.apiService.getObject(objectId, true).subscribe({
      next: (obj) => {
        if (obj.descriptors) {
          this.objectDescriptorsCache.set(objectId, obj.descriptors);
        }
      },
      error: (error) => {
        console.error('Error loading object descriptors:', error);
      }
    });
  }

  getObjectDescriptors(objectId: number): any {
    return this.objectDescriptorsCache.get(objectId) || null;
  }

  findSimilar() {
    if (!this.currentImageDescriptor || !this.currentImageDescriptor.objects.length) {
      return;
    }

    const objects = this.currentImageDescriptor.objects.filter((obj: any) => obj.object_id);
    if (objects.length === 0) {
      return;
    }

    this.searching = true;
    this.searchResults = null;
    this.selectedDetailsId = null;

    // Rechercher des similaires pour CHAQUE objet détecté
    const searchPromises = objects.map((obj: any) => 
      this.apiService.searchSimilar(obj.object_id, undefined, true).toPromise()
    );

    Promise.all(searchPromises).then((resultsArray: any[]) => {
      // Combiner tous les résultats
      const allResults: any[] = [];
      const seenObjectIds = new Set<number>(); // Pour éviter les doublons

      resultsArray.forEach((results, index) => {
        if (results && results.results && results.results.length > 0) {
          results.results.forEach((result: any) => {
            // Éviter les doublons basés sur object_id
            if (!seenObjectIds.has(result.object_id)) {
              seenObjectIds.add(result.object_id);
              // Ajouter une référence à l'objet source
              result.sourceObject = objects[index];
              allResults.push(result);
            }
          });
        }
      });

      // Trier par similarité (plus haute = mieux)
      allResults.sort((a, b) => b.similarity - a.similarity);

      // Créer une structure de résultats combinée
      this.searchResults = {
        query_object_id: objects[0].object_id,
        query_image_id: this.currentImage.id,
        query_class_name: objects.map((o: any) => o.class_name).join(', '),
        num_results: allResults.length,
        results: allResults
      };

      // Charger les descripteurs pour chaque résultat
      allResults.forEach((result: any) => {
        if (result.descriptors) {
          this.objectDescriptorsCache.set(result.object_id, result.descriptors);
        } else {
          this.loadObjectDescriptors(result.object_id);
        }
      });

      this.searching = false;
      console.log(`Recherche terminée: ${allResults.length} résultats uniques pour ${objects.length} objet(s)`);
    }).catch((error) => {
      console.error('Error searching:', error);
      this.searching = false;
      alert('Erreur lors de la recherche: ' + (error.error?.error || error.message || 'Erreur inconnue'));
    });
  }

  toggleDetails(objectId: number) {
    if (this.selectedDetailsId === objectId) {
      this.selectedDetailsId = null;
    } else {
      this.selectedDetailsId = objectId;
      this.loadObjectDescriptors(objectId);
    }
  }

  clearImage() {
    this.currentImage = null;
    this.currentImageDescriptor = null;
    this.searchResults = null;
    this.selectedDetailsId = null;
    this.objectDescriptorsCache.clear();
  }

  getImageUrl(filepath: string, source?: string): string {
    return this.apiService.getImageUrl(filepath, source);
  }
}

