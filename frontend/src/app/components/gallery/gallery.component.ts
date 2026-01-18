import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService, Image, ImageDescriptor } from '../../services/api.service';
import { RouterModule, ActivatedRoute } from '@angular/router';
import { DescriptorsViewerComponent } from '../descriptors-viewer/descriptors-viewer.component';

@Component({
  selector: 'app-gallery',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterModule, DescriptorsViewerComponent],
  template: `
    <div class="container">
      <div class="card">
        <h2 class="card-title">Galerie d'Images</h2>
        
        <!-- Upload Section -->
        <div class="upload-section">
          <input 
            type="file" 
            #fileInput 
            multiple 
            accept="image/*" 
            (change)="onFileSelected($event)"
            style="display: none"
          >
          <button class="btn btn-primary" (click)="fileInput.click()">
            Uploader des Images
          </button>
          <span class="upload-info" *ngIf="selectedFiles && selectedFiles.length > 0">
            {{ selectedFiles.length }} fichier(s) sélectionné(s)
          </span>
        </div>

        <div *ngIf="uploading" class="upload-status">
          <div class="spinner"></div>
          <p>Upload et traitement en cours...</p>
        </div>

        <div *ngIf="uploadError" class="alert alert-error">
          {{ uploadError }}
        </div>

        <div *ngIf="uploadSuccess" class="alert alert-success">
          Images uploadées avec succès !
        </div>

        <!-- Filter Section -->
        <div class="filter-section">
          <select class="form-control" [(ngModel)]="selectedCategory" (change)="loadImages()">
            <option value="">Toutes les catégories</option>
            <option *ngFor="let cat of categories" [value]="cat">{{ cat }}</option>
          </select>
        </div>

        <!-- Images Grid -->
        <div *ngIf="loading" class="loading">
          <div class="spinner"></div>
        </div>

        <div *ngIf="!loading && images.length === 0" class="empty-state">
          <p>Aucune image dans la galerie. Uploadez des images pour commencer.</p>
        </div>

        <div class="images-grid" *ngIf="!loading && images.length > 0">
          <div class="image-card" *ngFor="let image of images">
            <div class="image-wrapper" (click)="viewImage(image)">
              <img [src]="getImageUrl(image.filepath)" [alt]="image.filename" class="image-thumbnail">
              <div class="image-overlay">
                <span class="object-count">{{ image.num_objects }} objet(s)</span>
              </div>
            </div>
            <div class="image-info">
              <p class="image-filename">{{ image.filename }}</p>
              <p class="image-meta">{{ image.width }}x{{ image.height }}</p>
              <div class="image-actions">
                <button class="btn btn-primary btn-sm" (click)="viewImage(image)">
                  Voir détails
                </button>
                <button class="btn btn-danger btn-sm" (click)="deleteImage(image.image_id)">
                  Supprimer
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Image Detail Modal -->
        <div class="modal" *ngIf="selectedImage" (click)="closeModal()">
          <div class="modal-content" (click)="$event.stopPropagation()">
            <span class="close" (click)="closeModal()">&times;</span>
            <h3>{{ selectedImage.filename }}</h3>
            <div class="modal-image-container">
              <img [src]="getImageUrl(selectedImage.filepath)" 
                   [alt]="selectedImage.filename" 
                   class="modal-image"
                   #modalImage
                   (load)="onImageLoad()">
              <canvas #bboxCanvas class="bbox-canvas"></canvas>
            </div>

            <div class="detections" *ngIf="imageDescriptors">
              <h4>Objets détectés ({{ imageDescriptors.objects.length }})</h4>
              <div class="objects-list">
                <div class="object-item" 
                     *ngFor="let obj of imageDescriptors.objects"
                     [class.highlighted]="highlightedObjectId === obj.object_id"
                     (mouseenter)="highlightObject(obj.object_id)"
                     (mouseleave)="highlightObject(null)">
                  <span class="badge badge-primary">{{ obj.class_name }}</span>
                  <span class="confidence">Confiance: {{ (obj.confidence_score * 100).toFixed(1) }}%</span>
                  <div class="object-actions">
                    <button class="btn btn-success btn-sm" (click)="searchSimilar(obj.object_id)">
                      Rechercher similaire
                    </button>
                    <button class="btn btn-secondary btn-sm" (click)="viewDescriptors(obj.object_id)">
                      Voir descripteurs
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <!-- Transformations Section -->
            <div class="transformations-section">
              <h4>Transformations</h4>
              <div class="transform-buttons">
                <button class="btn btn-secondary btn-sm" (click)="openTransformModal('resize')">
                  Redimensionner
                </button>
                <button class="btn btn-secondary btn-sm" (click)="openTransformModal('crop')">
                  Recadrer
                </button>
                <button class="btn btn-secondary btn-sm" (click)="openTransformModal('rotate')">
                  Rotation
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .upload-section {
      display: flex;
      align-items: center;
      gap: 16px;
      margin-bottom: 24px;
      flex-wrap: wrap;
    }

    .upload-info {
      color: #4a5568;
      font-weight: 500;
    }

    .upload-status {
      text-align: center;
      padding: 40px;
      p {
        margin-top: 16px;
        color: #4a5568;
      }
    }

    .filter-section {
      margin-bottom: 24px;
      max-width: 300px;
    }

    .images-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 24px;
    }

    .image-card {
      background: white;
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      transition: transform 0.3s ease;

      &:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
      }
    }

    .image-wrapper {
      position: relative;
      cursor: pointer;
    }

    .image-overlay {
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

      .image-wrapper:hover & {
        opacity: 1;
      }
    }

    .object-count {
      color: white;
      font-weight: 600;
      font-size: 18px;
    }

    .image-info {
      padding: 16px;

      .image-filename {
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 8px;
        word-break: break-word;
      }

      .image-meta {
        color: #718096;
        font-size: 14px;
        margin-bottom: 12px;
      }

      .image-actions {
        display: flex;
        gap: 8px;
      }
    }

    .btn-sm {
      padding: 8px 16px;
      font-size: 14px;
    }

    .modal {
      display: flex;
      position: fixed;
      z-index: 1000;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
      background-color: rgba(0, 0, 0, 0.8);
      align-items: center;
      justify-content: center;
    }

    .modal-content {
      background-color: white;
      padding: 32px;
      border-radius: 12px;
      max-width: 90%;
      max-height: 90%;
      overflow-y: auto;
      position: relative;

      h3 {
        margin-bottom: 20px;
        color: #2d3748;
      }
    }

    .close {
      position: absolute;
      right: 20px;
      top: 20px;
      font-size: 32px;
      font-weight: bold;
      color: #999;
      cursor: pointer;

      &:hover {
        color: #333;
      }
    }

    .modal-image {
      max-width: 100%;
      max-height: 500px;
      border-radius: 8px;
      margin-bottom: 24px;
    }

    .detections {
      h4 {
        margin-bottom: 16px;
        color: #2d3748;
      }
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
      background: #f7fafc;
      border-radius: 8px;
    }

    .confidence {
      color: #4a5568;
      font-size: 14px;
    }

    .empty-state {
      text-align: center;
      padding: 60px 20px;
      color: #718096;
      font-size: 18px;
    }

    .modal-image-container {
      position: relative;
      display: inline-block;
      max-width: 100%;
      margin-bottom: 24px;
    }

    .bbox-canvas {
      position: absolute;
      top: 0;
      left: 0;
      pointer-events: none;
    }

    .object-item.highlighted {
      background: #e6fffa;
      border: 2px solid #38a169;
    }

    .object-actions {
      display: flex;
      gap: 8px;
    }

    .transformations-section {
      margin-top: 24px;
      padding-top: 24px;
      border-top: 1px solid #e2e8f0;
    }

    .transformations-section h4 {
      margin-bottom: 16px;
      color: #2d3748;
    }

    .transform-buttons {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }

    .transform-modal {
      max-width: 500px;
    }

    .transform-form {
      margin: 24px 0;
    }

    .transform-actions {
      display: flex;
      gap: 12px;
      justify-content: flex-end;
      margin-top: 24px;
    }

    .descriptors-modal {
      max-width: 1200px;
      max-height: 90vh;
      overflow-y: auto;
    }
  `]
})
export class GalleryComponent implements OnInit {
  images: Image[] = [];
  loading = false;
  uploading = false;
  uploadError: string | null = null;
  uploadSuccess = false;
  selectedFiles: FileList | null = null;
  selectedCategory = '';
  categories: string[] = [];
  selectedImage: Image | null = null;
  imageDescriptors: ImageDescriptor | null = null;
  highlightedObjectId: number | null = null;
  showTransformModal = false;
  showDescriptorsModal = false;
  selectedObjectId: number | null = null;
  selectedObjectDescriptors: any = null;
  selectedObjectInfo: any = null;
  currentTransform: 'resize' | 'crop' | 'rotate' | null = null;
  transformParams: any = {};
  transforming = false;

  constructor(private apiService: ApiService, private route: ActivatedRoute) {}

  ngOnInit() {
    this.loadImages();
    // Vérifier si un image_id est passé en paramètre
    this.route.queryParams.subscribe(params => {
      const imageId = params['image_id'];
      if (imageId) {
        this.loadImages().then(() => {
          const img = this.images.find(i => i.image_id === Number(imageId));
          if (img) {
            this.viewImage(img);
          }
        });
      }
    });
  }

  onFileSelected(event: Event) {
    const target = event.target as HTMLInputElement;
    if (target.files && target.files.length > 0) {
      this.selectedFiles = target.files;
      this.uploadImages();
    }
  }

  uploadImages() {
    if (!this.selectedFiles || this.selectedFiles.length === 0) return;

    this.uploading = true;
    this.uploadError = null;
    this.uploadSuccess = false;

    this.apiService.uploadImages(this.selectedFiles).subscribe({
      next: (response: any) => {
        this.uploading = false;
        this.uploadSuccess = true;
        this.selectedFiles = null;
        this.loadImages();
        setTimeout(() => {
          this.uploadSuccess = false;
        }, 3000);
      },
      error: (error) => {
        this.uploading = false;
        this.uploadError = error.error?.error || 'Erreur lors de l\'upload';
      }
    });
  }

  loadImages(): Promise<void> {
    return new Promise((resolve) => {
      this.loading = true;
      const category = this.selectedCategory || undefined;
      this.apiService.getImages(category).subscribe({
        next: (response) => {
          this.images = response.images;
          this.loading = false;
          // Extraire les catégories uniques
          const cats = new Set<string>();
          response.images.forEach(img => {
            if (img.category) cats.add(img.category);
          });
          this.categories = Array.from(cats);
          resolve();
        },
        error: (error) => {
          console.error('Error loading images:', error);
          this.loading = false;
          resolve();
        }
      });
    });
  }

  viewImage(image: Image) {
    this.selectedImage = image;
    this.apiService.getImageDescriptors(image.image_id).subscribe({
      next: (descriptors) => {
        this.imageDescriptors = descriptors;
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
    const canvas = document.querySelector('canvas.bbox-canvas') as HTMLCanvasElement;
    const img = document.querySelector('img.modal-image') as HTMLImageElement;
    
    if (!canvas || !img || !this.imageDescriptors) return;

    const rect = img.getBoundingClientRect();
    const scaleX = img.naturalWidth / rect.width;
    const scaleY = img.naturalHeight / rect.height;

    canvas.width = rect.width;
    canvas.height = rect.height;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    this.imageDescriptors.objects.forEach((obj, index) => {
      const x = (obj.bbox[0] / this.selectedImage!.width) * rect.width;
      const y = (obj.bbox[1] / this.selectedImage!.height) * rect.height;
      const width = ((obj.bbox[2] - obj.bbox[0]) / this.selectedImage!.width) * rect.width;
      const height = ((obj.bbox[3] - obj.bbox[1]) / this.selectedImage!.height) * rect.height;

      // Choisir la couleur selon si l'objet est highlighté
      const isHighlighted = this.highlightedObjectId === obj.object_id;
      ctx.strokeStyle = isHighlighted ? '#38a169' : '#667eea';
      ctx.lineWidth = isHighlighted ? 3 : 2;
      ctx.strokeRect(x, y, width, height);

      // Dessiner le label
      ctx.fillStyle = isHighlighted ? '#38a169' : '#667eea';
      ctx.fillRect(x, y - 20, Math.min(width, ctx.measureText(obj.class_name).width + 10), 20);
      ctx.fillStyle = 'white';
      ctx.font = '12px Arial';
      ctx.fillText(obj.class_name, x + 5, y - 5);
    });
  }

  highlightObject(objectId: number | null) {
    this.highlightedObjectId = objectId;
    this.drawBoundingBoxes();
  }

  closeModal() {
    this.selectedImage = null;
    this.imageDescriptors = null;
    this.highlightedObjectId = null;
  }

  viewDescriptors(objectId: number) {
    this.showDescriptorsModal = true;
    this.selectedObjectId = objectId;
    
    // Récupérer les descripteurs de l'objet
    this.apiService.getObject(objectId, true).subscribe({
      next: (obj) => {
        this.selectedObjectDescriptors = obj.descriptors || {};
        this.selectedObjectInfo = {
          object_id: obj.object_id,
          class_name: obj.class_name,
          confidence_score: obj.confidence_score
        };
      },
      error: (error) => {
        console.error('Error loading object descriptors:', error);
        alert('Erreur lors du chargement des descripteurs');
      }
    });
  }

  closeDescriptorsModal() {
    this.showDescriptorsModal = false;
    this.selectedObjectId = null;
    this.selectedObjectDescriptors = null;
    this.selectedObjectInfo = null;
  }

  openTransformModal(transform: 'resize' | 'crop' | 'rotate') {
    if (!this.selectedImage) return;
    
    this.currentTransform = transform;
    this.showTransformModal = true;
    
    // Initialiser les paramètres selon le type de transformation
    if (transform === 'resize') {
      this.transformParams = {
        width: this.selectedImage.width,
        height: this.selectedImage.height
      };
    } else if (transform === 'crop') {
      this.transformParams = {
        x: 0,
        y: 0,
        width: this.selectedImage.width,
        height: this.selectedImage.height
      };
    } else if (transform === 'rotate') {
      this.transformParams = {
        angle: 0,
        expand: true
      };
    }
  }

  closeTransformModal() {
    this.showTransformModal = false;
    this.currentTransform = null;
    this.transformParams = {};
  }

  applyTransform() {
    if (!this.selectedImage || !this.currentTransform) return;

    this.transforming = true;
    const payload = {
      image_id: this.selectedImage.image_id,
      transform: this.currentTransform,
      ...this.transformParams
    };

    this.apiService.transformImage(payload).subscribe({
      next: (response: any) => {
        this.transforming = false;
        this.closeTransformModal();
        this.loadImages().then(() => {
          alert('Image transformée avec succès !');
        });
      },
      error: (error) => {
        console.error('Error transforming image:', error);
        this.transforming = false;
        alert('Erreur lors de la transformation');
      }
    });
  }

  deleteImage(imageId: number) {
    if (confirm('Êtes-vous sûr de vouloir supprimer cette image ?')) {
      this.apiService.deleteImage(imageId).subscribe({
        next: () => {
          this.loadImages();
          if (this.selectedImage?.image_id === imageId) {
            this.closeModal();
          }
        },
        error: (error) => {
          console.error('Error deleting image:', error);
          alert('Erreur lors de la suppression');
        }
      });
    }
  }

  searchSimilar(objectId: number) {
    // Rediriger vers la page de recherche avec l'object_id
    window.location.href = `/search?object_id=${objectId}`;
  }

  getImageUrl(filepath: string): string {
    return this.apiService.getImageUrl(filepath);
  }
}

