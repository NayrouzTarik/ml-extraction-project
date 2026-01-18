import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [CommonModule, RouterModule],
  template: `
    <div class="container">
      <div class="hero-section">
        <h2>Bienvenue dans l'Application de Recherche d'Images</h2>
        <p class="subtitle">
          Détectez des objets avec YOLOv8n et recherchez des images similaires 
          via des descripteurs visuels avancés
        </p>
      </div>

      <div class="features-grid">
        <div class="feature-card">
          <div class="feature-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
          </div>
          <h3>Upload d'Images</h3>
          <p>Uploadez une ou plusieurs images pour détecter les objets avec YOLOv8n</p>
          <a routerLink="/gallery" class="btn btn-primary">Commencer</a>
        </div>

        <div class="feature-card">
          <div class="feature-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
              <line x1="9" y1="3" x2="9" y2="21"></line>
              <line x1="15" y1="3" x2="15" y2="21"></line>
            </svg>
          </div>
          <h3>Détection d'Objets</h3>
          <p>Détection automatique de 15 catégories d'objets avec bounding boxes</p>
        </div>

        <div class="feature-card">
          <div class="feature-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="11" cy="11" r="8"></circle>
              <path d="m21 21-4.35-4.35"></path>
            </svg>
          </div>
          <h3>Recherche par Similarité</h3>
          <p>Trouvez les 10 images les plus similaires basées sur les descripteurs visuels</p>
          <a routerLink="/search" class="btn btn-primary">Rechercher</a>
        </div>

        <div class="feature-card">
          <div class="feature-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="18" y1="20" x2="18" y2="10"></line>
              <line x1="12" y1="20" x2="12" y2="4"></line>
              <line x1="6" y1="20" x2="6" y2="14"></line>
            </svg>
          </div>
          <h3>Descripteurs Avancés</h3>
          <p>Histogrammes, couleurs dominantes, Tamura, Gabor, Moments de Hu, HOG, LBP</p>
        </div>
      </div>

      <div class="info-section">
        <h3>Classes Détectées (15 catégories)</h3>
        <div class="classes-list">
          <span class="class-badge" *ngFor="let cls of classes">{{ cls }}</span>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .hero-section {
      text-align: center;
      padding: 40px 20px;
      margin-bottom: 40px;

      h2 {
        font-size: 36px;
        font-weight: 700;
        color: white;
        margin-bottom: 16px;
      }

      .subtitle {
        font-size: 18px;
        color: rgba(255, 255, 255, 0.9);
        max-width: 800px;
        margin: 0 auto;
      }
    }

    .features-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 24px;
      margin-bottom: 40px;
    }

    .feature-card {
      background: white;
      border-radius: 12px;
      padding: 32px;
      text-align: center;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      transition: transform 0.3s ease;

      &:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
      }

      .feature-icon {
        width: 48px;
        height: 48px;
        margin: 0 auto 16px;
        color: #667eea;
        display: flex;
        align-items: center;
        justify-content: center;
      }

      h3 {
        font-size: 22px;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 12px;
      }

      p {
        color: #718096;
        margin-bottom: 20px;
        line-height: 1.6;
      }
    }

    .info-section {
      background: white;
      border-radius: 12px;
      padding: 32px;
      text-align: center;

      h3 {
        font-size: 24px;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 24px;
      }

      .classes-list {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        justify-content: center;
      }

      .class-badge {
        display: inline-block;
        padding: 8px 16px;
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
        color: white;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 500;
      }
    }
  `]
})
export class HomeComponent {
  classes = [
    'basketball', 'orange', 'lemon', 'banana', 'Granny Smith',
    'corn', 'acorn', 'umbrella', 'strawberry', 'cucumber, cuke',
    'pineapple, ananas', 'tiger shark', 'black swan', 'airliner'
  ];
}

