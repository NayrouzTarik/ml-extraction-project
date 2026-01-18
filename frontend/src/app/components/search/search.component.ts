import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router, RouterModule } from '@angular/router';
import { ApiService, ImageDescriptor, SearchResponse, SearchResult, Image } from '../../services/api.service';

@Component({
  selector: 'app-search',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterModule],
  template: `
    <div class="container">
      <div class="card">
        <h2 class="card-title">Recherche par Similarité</h2>

        <!-- Selection d'image -->
        <div class="search-section">
          <div class="form-group">
            <label class="form-label">Choisir une image pour la recherche</label>
            <select class="form-control" [(ngModel)]="selectedImageId" (change)="onImageSelected()">
              <option value="">-- Sélectionner une image --</option>
              <option *ngFor="let img of allImages" [value]="img.image_id">
                {{ img.filename }} ({{ img.num_objects }} objets)
              </option>
            </select>
          </div>

          <div *ngIf="selectedImage" class="selected-image-container">
            <div class="image-display-wrapper">
              <img [src]="getImageUrl(selectedImage.filepath)" [alt]="selectedImage.filename" 
                   class="display-image" #imageElement (load)="onImageLoad()">
              <canvas #bboxCanvas class="bbox-canvas"></canvas>
            </div>

            <div class="objects-panel">
              <h4>Objets détectés</h4>
              <div class="objects-list">
                <div *ngFor="let obj of imageObjects" 
                     class="object-item"
                     [class.selected]="selectedObject?.object_id === obj.object_id"
                     (click)="selectObject(obj)">
                  <span class="badge badge-primary">{{ obj.class_name }}</span>
                  <span class="confidence">{{ (obj.confidence_score * 100).toFixed(1) }}%</span>
                  <button class="btn btn-primary btn-sm" (click)="searchForObject(obj.object_id)">
                    Rechercher
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Résultats de recherche -->
        <div *ngIf="searching" class="searching">
          <div class="spinner"></div>
          <p>Recherche en cours...</p>
        </div>

        <div *ngIf="searchResults && searchResults.results.length > 0" class="results-section">
          <h3 class="results-title">
            {{ searchResults.num_results }} image(s) similaire(s) trouvée(s) 
            (classe: {{ searchResults.query_class_name }})
          </h3>
          
          <div class="results-grid">
            <div *ngFor="let result of searchResults.results; let i = index" class="result-card">
              <div class="result-rank">#{{ i + 1 }}</div>
              <div class="image-wrapper">
                <img [src]="getImageUrl(result.filepath)" [alt]="result.filename" 
                     class="result-thumbnail">
                <div class="result-overlay">
                  <div class="similarity-badge">
                    {{ (result.similarity * 100).toFixed(1) }}% similaire
                  </div>
                </div>
              </div>
              <div class="result-info">
                <p class="result-filename">{{ result.filename }}</p>
                <span class="badge badge-primary">{{ result.class_name }}</span>
                <p class="result-meta">
                  Distance: {{ result.distance.toFixed(4) }}<br>
                  Confiance: {{ (result.confidence_score * 100).toFixed(1) }}%
                </p>
                <button class="btn btn-primary btn-sm" 
                        (click)="viewImageDetails(result.image_id)">
                  Voir détails
                </button>
              </div>
            </div>
          </div>
        </div>

        <div *ngIf="searchResults && searchResults.results.length === 0" class="no-results">
          <p>Aucun résultat trouvé.</p>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .search-section {
      margin-bottom: 32px;
    }

    .selected-image-container {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 24px;
      margin-top: 24px;
    }

    .image-display-wrapper {
      position: relative;
      border-radius: 8px;
      overflow: hidden;
      background: #f7fafc;
    }

    .display-image {
      width: 100%;
      height: auto;
      display: block;
    }

    .bbox-canvas {
      position: absolute;
      top: 0;
      left: 0;
      pointer-events: none;
    }

    .objects-panel {
      background: #f7fafc;
      border-radius: 8px;
      padding: 20px;
      max-height: 600px;
      overflow-y: auto;
    }

    .objects-panel h4 {
      margin-bottom: 16px;
      color: #2d3748;
      font-size: 18px;
    }

    .objects-list {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .object-item {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px;
      background: white;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.3s ease;
      border: 2px solid transparent;
    }

    .object-item:hover {
      background: #edf2f7;
      border-color: #667eea;
    }

    .object-item.selected {
      background: #e6fffa;
      border-color: #38a169;
    }

    .confidence {
      color: #4a5568;
      font-size: 14px;
      flex: 1;
    }

    .searching {
      text-align: center;
      padding: 40px;
    }

    .searching p {
      margin-top: 16px;
      color: #4a5568;
    }

    .results-section {
      margin-top: 32px;
    }

    .results-title {
      font-size: 22px;
      font-weight: 600;
      color: #2d3748;
      margin-bottom: 24px;
    }

    .results-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 24px;
    }

    .result-card {
      background: white;
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      transition: transform 0.3s ease;
      position: relative;
    }

    .result-card:hover {
      transform: translateY(-5px);
      box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    .result-rank {
      position: absolute;
      top: 8px;
      left: 8px;
      background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
      color: white;
      padding: 4px 12px;
      border-radius: 12px;
      font-weight: 600;
      font-size: 14px;
      z-index: 10;
    }

    .image-wrapper {
      position: relative;
      width: 100%;
      padding-top: 75%;
      overflow: hidden;
      background: #f7fafc;
    }

    .result-thumbnail {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
    }

    .result-overlay {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0, 0, 0, 0.5);
      display: flex;
      align-items: center;
      justify-content: center;
      opacity: 0;
      transition: opacity 0.3s ease;
    }

    .result-card:hover .result-overlay {
      opacity: 1;
    }

    .similarity-badge {
      background: white;
      color: #2d3748;
      padding: 8px 16px;
      border-radius: 20px;
      font-weight: 600;
      font-size: 16px;
    }

    .result-info {
      padding: 16px;
    }

    .result-filename {
      font-weight: 600;
      color: #2d3748;
      margin-bottom: 8px;
      word-break: break-word;
    }

    .result-meta {
      color: #718096;
      font-size: 12px;
      margin: 8px 0;
      line-height: 1.6;
    }

    .no-results {
      text-align: center;
      padding: 60px 20px;
      color: #718096;
      font-size: 18px;
    }

    .btn-sm {
      padding: 8px 16px;
      font-size: 14px;
    }

    @media (max-width: 768px) {
      .selected-image-container {
        grid-template-columns: 1fr;
      }
    }
  `]
})
export class SearchComponent implements OnInit {
  allImages: Image[] = [];
  selectedImageId: number | null = null;
  selectedImage: Image | null = null;
  imageDescriptors: ImageDescriptor | null = null;
  imageObjects: any[] = [];
  selectedObject: any = null;
  searchResults: SearchResponse | null = null;
  searching = false;

  constructor(
    private apiService: ApiService,
    private route: ActivatedRoute,
    private router: Router
  ) {}

  ngOnInit() {
    // Charger toutes les images
    this.loadAllImages();

    // Vérifier si un object_id est passé en paramètre
    this.route.queryParams.subscribe(params => {
      const objectId = params['object_id'];
      if (objectId) {
        this.loadObjectAndSearch(Number(objectId));
      }
    });
  }

  loadAllImages() {
    this.apiService.getImages().subscribe({
      next: (response) => {
        this.allImages = response.images;
      },
      error: (error) => {
        console.error('Error loading images:', error);
      }
    });
  }

  onImageSelected() {
    if (!this.selectedImageId) {
      this.selectedImage = null;
      this.imageDescriptors = null;
      this.imageObjects = [];
      this.selectedObject = null;
      return;
    }

    const img = this.allImages.find(i => i.image_id === this.selectedImageId);
    if (!img) return;

    this.selectedImage = img;
    this.selectedObject = null;
    this.searchResults = null;

    // Charger les descripteurs de l'image
    this.apiService.getImageDescriptors(img.image_id).subscribe({
      next: (descriptors) => {
        this.imageDescriptors = descriptors;
        this.imageObjects = descriptors.objects || [];
        // Dessiner les bounding boxes après que l'image soit chargée
        setTimeout(() => this.drawBoundingBoxes(), 100);
      },
      error: (error) => {
        console.error('Error loading descriptors:', error);
      }
    });
  }

  onImageLoad() {
    this.drawBoundingBoxes();
  }

  drawBoundingBoxes() {
    const canvas = document.querySelector('.image-display-wrapper canvas.bbox-canvas') as HTMLCanvasElement;
    const img = document.querySelector('.image-display-wrapper img.display-image') as HTMLImageElement;
    
    if (!canvas || !img || !this.selectedImage || !this.imageDescriptors) return;
    
    if (!canvas || !img || !this.imageDescriptors) return;

    const rect = img.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    this.imageObjects.forEach((obj) => {
      const x = (obj.bbox[0] / this.selectedImage!.width) * rect.width;
      const y = (obj.bbox[1] / this.selectedImage!.height) * rect.height;
      const width = ((obj.bbox[2] - obj.bbox[0]) / this.selectedImage!.width) * rect.width;
      const height = ((obj.bbox[3] - obj.bbox[1]) / this.selectedImage!.height) * rect.height;

      const isSelected = this.selectedObject?.object_id === obj.object_id;
      ctx.strokeStyle = isSelected ? '#38a169' : '#4299e1';
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeRect(x, y, width, height);

      // Dessiner le label
      ctx.fillStyle = isSelected ? '#38a169' : '#4299e1';
      ctx.fillRect(x, y - 20, Math.min(width, ctx.measureText(obj.class_name).width + 10), 20);
      ctx.fillStyle = 'white';
      ctx.font = '12px Arial';
      ctx.fillText(obj.class_name, x + 5, y - 5);
    });
  }

  selectObject(obj: any) {
    this.selectedObject = obj;
  }

  searchForObject(objectId: number) {
    this.searching = true;
    this.searchResults = null;

    this.apiService.searchSimilar(objectId, undefined, true).subscribe({
      next: (results) => {
        this.searchResults = results;
        this.searching = false;
      },
      error: (error) => {
        console.error('Error searching:', error);
        this.searching = false;
      }
    });
  }

  loadObjectAndSearch(objectId: number) {
    this.apiService.getObject(objectId).subscribe({
      next: (obj) => {
        // Charger l'image correspondante
        this.apiService.getImages().subscribe({
          next: (response) => {
            this.allImages = response.images;
            const img = response.images.find(i => i.image_id === obj.image_id);
            if (img) {
              this.selectedImageId = img.image_id;
              this.onImageSelected();
              // Sélectionner l'objet et rechercher
              setTimeout(() => {
                const foundObj = this.imageObjects.find(o => o.object_id === objectId);
                if (foundObj) {
                  this.selectObject(foundObj);
                  this.searchForObject(objectId);
                }
              }, 500);
            }
          }
        });
      },
      error: (error) => {
        console.error('Error loading object:', error);
      }
    });
  }

  viewImageDetails(imageId: number) {
    this.router.navigate(['/gallery'], { 
      queryParams: { image_id: imageId } 
    });
  }

  getImageUrl(filepath: string): string {
    return this.apiService.getImageUrl(filepath);
  }
}

