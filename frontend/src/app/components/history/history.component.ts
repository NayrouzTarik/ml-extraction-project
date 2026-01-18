import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-history',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="history-page">
      <div class="page-header">
        <h1>Historique des Images</h1>
        <p class="subtitle">Consultez et gérez toutes vos images uploadées</p>
      </div>

      <div class="content-container">
        <!-- Filters and Stats -->
        <div class="controls-bar">
          <div class="stats">
            <div class="stat-item">
              <span class="stat-value">{{ historyImages.length }}</span>
              <span class="stat-label">Images</span>
            </div>
            <div class="stat-item">
              <span class="stat-value">{{ totalObjects }}</span>
              <span class="stat-label">Objets détectés</span>
            </div>
          </div>

          <div class="filters">
            <select class="filter-select" [(ngModel)]="selectedCategory" (change)="applyFilters()">
              <option value="">Toutes les catégories</option>
              <option *ngFor="let cat of categories" [value]="cat">{{ cat }}</option>
            </select>
            <button class="btn btn-secondary" (click)="loadHistory()">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="23 4 23 10 17 10"></polyline>
                <polyline points="1 20 1 14 7 14"></polyline>
                <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
              </svg>
              Actualiser
            </button>
          </div>
        </div>

        <!-- Loading State -->
        <div *ngIf="loading" class="loading-state">
          <div class="spinner"></div>
          <p>Chargement de l'historique...</p>
        </div>

        <!-- Empty State -->
        <div *ngIf="!loading && filteredImages.length === 0" class="empty-state">
          <svg class="empty-icon" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
            <circle cx="8.5" cy="8.5" r="1.5"></circle>
            <polyline points="21 15 16 10 5 21"></polyline>
          </svg>
          <h3>Aucune image trouvée</h3>
          <p>{{ selectedCategory ? 'Aucune image dans cette catégorie' : 'Commencez par uploader une image' }}</p>
          <button class="btn btn-primary" (click)="goToUpload()">
            Uploader une image
          </button>
        </div>

        <!-- Images Grid -->
        <div *ngIf="!loading && filteredImages.length > 0" class="images-grid">
          <div *ngFor="let img of filteredImages" class="image-card">
            <div class="image-card-header">
              <span class="image-id">#{{ img.image_id }}</span>
              <button 
                class="btn-icon-danger" 
                (click)="deleteImage(img.image_id)"
                title="Supprimer">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <polyline points="3 6 5 6 21 6"></polyline>
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                </svg>
              </button>
            </div>

            <div class="image-thumbnail-wrapper" (click)="viewImage(img)">
              <img [src]="getImageUrl(img.filepath)" [alt]="img.filename" class="image-thumbnail">
              <div class="image-overlay">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                  <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                  <circle cx="12" cy="12" r="3"></circle>
                </svg>
                <span>Voir détails</span>
              </div>
            </div>

            <div class="image-card-body">
              <p class="image-filename" [title]="img.filename">{{ img.filename }}</p>
              
              <div class="image-meta">
                <div class="meta-item">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                    <line x1="9" y1="3" x2="9" y2="21"></line>
                  </svg>
                  <span>{{ img.width }}×{{ img.height }}</span>
                </div>
                <div class="meta-item">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                  </svg>
                  <span>{{ img.num_objects }} objet(s)</span>
                </div>
              </div>

              <div *ngIf="img.category" class="image-categories">
                <span class="category-badge">{{ img.category }}</span>
              </div>

              <div class="image-date">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <circle cx="12" cy="12" r="10"></circle>
                  <polyline points="12 6 12 12 16 14"></polyline>
                </svg>
                <span>{{ formatDate(img.upload_date) }}</span>
              </div>
            </div>

            <button class="btn btn-primary btn-fullwidth" (click)="viewImage(img)">
              Analyser cette image
            </button>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .history-page {
      min-height: 100vh;
      background: #f8fafc;
      padding: 40px 20px;
    }

    .page-header {
      max-width: 1400px;
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
      max-width: 1400px;
      margin: 0 auto;
    }

    .controls-bar {
      background: white;
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 24px;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 20px;
    }

    .stats {
      display: flex;
      gap: 32px;
    }

    .stat-item {
      display: flex;
      flex-direction: column;
    }

    .stat-value {
      font-size: 28px;
      font-weight: 700;
      color: #1a202c;
      line-height: 1;
    }

    .stat-label {
      font-size: 14px;
      color: #64748b;
      margin-top: 4px;
    }

    .filters {
      display: flex;
      gap: 12px;
      align-items: center;
    }

    .filter-select {
      padding: 10px 16px;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      background: white;
      color: #334155;
      font-size: 14px;
      cursor: pointer;
      min-width: 200px;
    }

    .filter-select:focus {
      outline: none;
      border-color: #4299e1;
      box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.1);
    }

    .loading-state, .empty-state {
      background: white;
      border-radius: 12px;
      padding: 60px 20px;
      text-align: center;
    }

    .empty-icon {
      color: #cbd5e1;
      margin-bottom: 20px;
    }

    .empty-state h3 {
      font-size: 24px;
      font-weight: 600;
      color: #1a202c;
      margin-bottom: 8px;
    }

    .empty-state p {
      color: #64748b;
      margin-bottom: 24px;
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
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
      border: 1px solid #e2e8f0;
      transition: all 0.2s;
    }

    .image-card:hover {
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
      transform: translateY(-2px);
    }

    .image-card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 16px;
      background: #f8fafc;
      border-bottom: 1px solid #e2e8f0;
    }

    .image-id {
      font-size: 12px;
      font-weight: 600;
      color: #64748b;
    }

    .btn-icon-danger {
      background: none;
      border: none;
      color: #ef4444;
      cursor: pointer;
      padding: 6px;
      border-radius: 6px;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .btn-icon-danger:hover {
      background: #fee2e2;
      color: #dc2626;
    }

    .image-thumbnail-wrapper {
      position: relative;
      width: 100%;
      padding-top: 75%;
      overflow: hidden;
      background: #e2e8f0;
      cursor: pointer;
    }

    .image-thumbnail {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
      transition: transform 0.3s;
    }

    .image-thumbnail-wrapper:hover .image-thumbnail {
      transform: scale(1.05);
    }

    .image-overlay {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0, 0, 0, 0.6);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 8px;
      opacity: 0;
      transition: opacity 0.3s;
      color: white;
    }

    .image-thumbnail-wrapper:hover .image-overlay {
      opacity: 1;
    }

    .image-overlay span {
      font-weight: 500;
      font-size: 14px;
    }

    .image-card-body {
      padding: 16px;
    }

    .image-filename {
      font-weight: 600;
      color: #1a202c;
      font-size: 14px;
      margin-bottom: 12px;
      word-break: break-word;
      overflow: hidden;
      text-overflow: ellipsis;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
    }

    .image-meta {
      display: flex;
      gap: 16px;
      margin-bottom: 12px;
      padding-bottom: 12px;
      border-bottom: 1px solid #f1f5f9;
    }

    .meta-item {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 13px;
      color: #64748b;
    }

    .meta-item svg {
      color: #94a3b8;
    }

    .image-categories {
      margin-bottom: 12px;
    }

    .category-badge {
      display: inline-block;
      background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
      color: white;
      padding: 4px 12px;
      border-radius: 12px;
      font-size: 12px;
      font-weight: 500;
    }

    .image-date {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      color: #94a3b8;
      margin-bottom: 16px;
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

    .btn-primary:hover {
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

    .btn-fullwidth {
      width: 100%;
      justify-content: center;
    }

    .spinner {
      border: 4px solid #e2e8f0;
      border-top: 4px solid #4299e1;
      border-radius: 50%;
      width: 48px;
      height: 48px;
      animation: spin 1s linear infinite;
      margin: 0 auto 16px;
    }

    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }

    @media (max-width: 768px) {
      .controls-bar {
        flex-direction: column;
        align-items: stretch;
      }

      .stats {
        justify-content: space-around;
      }

      .filters {
        flex-direction: column;
      }

      .filter-select {
        width: 100%;
      }

      .images-grid {
        grid-template-columns: 1fr;
      }
    }
  `]
})
export class HistoryComponent implements OnInit {
  historyImages: any[] = [];
  filteredImages: any[] = [];
  loading = false;
  selectedCategory = '';
  categories: string[] = [];

  constructor(private apiService: ApiService, private router: Router) {}

  ngOnInit() {
    this.loadHistory();
  }

  loadHistory() {
    this.loading = true;
    this.apiService.getImages().subscribe({
      next: (response) => {
        this.historyImages = response.images;
        this.extractCategories();
        this.applyFilters();
        this.loading = false;
      },
      error: (error) => {
        console.error('Error loading history:', error);
        this.loading = false;
      }
    });
  }

  extractCategories() {
    const cats = new Set<string>();
    this.historyImages.forEach(img => {
      if (img.category) {
        cats.add(img.category);
      }
    });
    this.categories = Array.from(cats).sort();
  }

  applyFilters() {
    if (!this.selectedCategory) {
      this.filteredImages = this.historyImages;
    } else {
      this.filteredImages = this.historyImages.filter(img => img.category === this.selectedCategory);
    }
  }

  deleteImage(imageId: number) {
    if (!confirm('Êtes-vous sûr de vouloir supprimer cette image ?')) {
      return;
    }

    this.apiService.deleteImage(imageId).subscribe({
      next: () => {
        this.loadHistory();
      },
      error: (error) => {
        console.error('Error deleting image:', error);
        alert('Erreur lors de la suppression');
      }
    });
  }

  viewImage(image: any) {
    // Naviguer vers la page upload avec l'image pré-chargée
    this.router.navigate(['/upload'], { 
      queryParams: { image_id: image.image_id } 
    });
  }

  goToUpload() {
    this.router.navigate(['/upload']);
  }

  formatDate(dateString: string): string {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleDateString('fr-FR', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  }

  get totalObjects(): number {
    return this.historyImages.reduce((sum, img) => sum + (img.num_objects || 0), 0);
  }

  getImageUrl(filepath: string): string {
    return this.apiService.getImageUrl(filepath);
  }
}

