import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, CommonModule, RouterLink, RouterLinkActive],
  template: `
    <div class="app-container">
      <header class="app-header">
        <div class="header-container">
          <div class="logo">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
              <circle cx="8.5" cy="8.5" r="1.5"></circle>
              <polyline points="21 15 16 10 5 21"></polyline>
            </svg>
            <span class="logo-text">Image Search</span>
          </div>
          <nav class="main-nav">
            <a routerLink="/search-3d" routerLinkActive="active">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2l4 4-4 4-4-4 4-4z"></path>
                <path d="M2 12l4 4-4 4-4-4 4-4z" transform="translate(4 0)"></path>
                <path d="M12 14l4 4-4 4-4-4 4-4z"></path>
              </svg>
              Recherche 3D
            </a>

            <a routerLink="/upload" routerLinkActive="active">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
              Upload
            </a>
            <a routerLink="/history" routerLinkActive="active">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                <line x1="9" y1="3" x2="9" y2="21"></line>
              </svg>
              Historique
            </a>
          </nav>
        </div>
      </header>
      <main class="app-main">
        <router-outlet></router-outlet>
      </main>
    </div>
  `,
  styles: [`
    .app-container {
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    .app-header {
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(10px);
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
      padding: 20px 0;
      position: sticky;
      top: 0;
      z-index: 100;

    }

    .header-container {
      max-width: 1400px;
      margin: 0 auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0 20px;
    }

    .logo {
      display: flex;
      align-items: center;
      gap: 12px;
      color: #1a202c;
    }

    .logo svg {
      color: #4299e1;
    }

    .logo-text {
      font-size: 20px;
      font-weight: 700;
    }

    .main-nav {
      display: flex;
      gap: 8px;
    }

    .main-nav a {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 10px 20px;
      text-decoration: none;
      color: #64748b;
      font-weight: 500;
      border-radius: 8px;
      transition: all 0.2s;
    }

    .main-nav a:hover {
      background: #f1f5f9;
      color: #1a202c;
    }

    .main-nav a.active {
      background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
      color: white;
    }

    .app-main {
      flex: 1;
      padding: 40px 0;
    }
  `]
})
export class AppComponent {
  title = 'Image Search Application';
}

