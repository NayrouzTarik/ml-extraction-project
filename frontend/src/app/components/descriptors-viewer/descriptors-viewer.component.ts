import { Component, Input, OnInit, OnChanges, SimpleChanges, ViewChild, ElementRef, AfterViewInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Chart, ChartConfiguration, ChartData, registerables } from 'chart.js';

// Enregistrer tous les composants Chart.js
Chart.register(...registerables);

@Component({
  selector: 'app-descriptors-viewer',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="descriptors-viewer">
      <div class="descriptors-header">
        <h3>Descripteurs Visuels</h3>
        <p *ngIf="objectInfo" class="object-info">
          Objet: <strong>{{ objectInfo.class_name }}</strong> 
          (Confiance: {{ (objectInfo.confidence_score * 100).toFixed(1) }}%)
        </p>
      </div>

      <div *ngIf="!hasDescriptors()" class="no-descriptors">
        <p>Aucun descripteur disponible pour cet objet.</p>
      </div>

      <div *ngIf="hasDescriptors()" class="descriptors-grid">
        
        <!-- Histogramme des Couleurs -->
        <div *ngIf="descriptors.color_hist" class="descriptor-card">
          <h4>Histogramme des Couleurs</h4>
          <div class="chart-container">
            <canvas #colorHistCanvas></canvas>
          </div>
        </div>

        <!-- Couleurs Dominantes -->
        <div *ngIf="descriptors.dominant_colors" class="descriptor-card">
          <h4>Couleurs Dominantes</h4>
          <div class="dominant-colors">
            <div *ngFor="let color of dominantColorsList" class="color-item">
              <div class="color-box" [style.background-color]="color.rgb"></div>
              <div class="color-info">
                <div class="color-percentage">{{ color.percentage.toFixed(1) }}%</div>
                <div class="color-rgb">RGB: {{ color.rgb }}</div>
              </div>
            </div>
          </div>
        </div>

        <!-- Descripteurs de Tamura -->
        <div *ngIf="descriptors.tamura" class="descriptor-card">
          <h4>Descripteurs de Tamura (Texture)</h4>
          <div class="chart-container">
            <canvas #tamuraCanvas></canvas>
          </div>
          <div class="descriptor-values">
            <div class="value-item">
              <span class="value-label">Rugosité (Coarseness):</span>
              <span class="value-number">{{ tamuraValues[0]?.toFixed(4) }}</span>
            </div>
            <div class="value-item">
              <span class="value-label">Contraste:</span>
              <span class="value-number">{{ tamuraValues[1]?.toFixed(4) }}</span>
            </div>
            <div class="value-item">
              <span class="value-label">Directionnalité:</span>
              <span class="value-number">{{ tamuraValues[2]?.toFixed(4) }}</span>
            </div>
          </div>
        </div>

        <!-- Filtres de Gabor -->
        <div *ngIf="descriptors.gabor" class="descriptor-card">
          <h4>Filtres de Gabor (Texture)</h4>
          <div class="chart-container">
            <canvas #gaborCanvas></canvas>
          </div>
        </div>

        <!-- Moments de Hu -->
        <div *ngIf="descriptors.hu_moments" class="descriptor-card">
          <h4>Moments de Hu (Forme)</h4>
          <div class="chart-container">
            <canvas #huMomentsCanvas></canvas>
          </div>
          <div class="descriptor-values">
            <div *ngFor="let moment of huMomentsList; let i = index" class="value-item">
              <span class="value-label">Moment {{ i + 1 }}:</span>
              <span class="value-number">{{ moment.toFixed(6) }}</span>
            </div>
          </div>
        </div>

        <!-- HOG (Histogram of Oriented Gradients) -->
        <div *ngIf="descriptors.hog" class="descriptor-card">
          <h4>HOG (Histogram of Oriented Gradients)</h4>
          <div class="chart-container">
            <canvas #hogCanvas></canvas>
          </div>
        </div>

        <!-- LBP (Local Binary Patterns) -->
        <div *ngIf="descriptors.lbp" class="descriptor-card">
          <h4>LBP (Local Binary Patterns)</h4>
          <div class="chart-container">
            <canvas #lbpCanvas></canvas>
          </div>
        </div>

      </div>
    </div>
  `,
  styles: [`
    .descriptors-viewer {
      background: white;
      border-radius: 12px;
      padding: 24px;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .descriptors-header {
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 2px solid #e2e8f0;
    }

    .descriptors-header h3 {
      font-size: 24px;
      font-weight: 600;
      color: #2d3748;
      margin-bottom: 8px;
    }

    .object-info {
      color: #4a5568;
      font-size: 14px;
    }

    .no-descriptors {
      text-align: center;
      padding: 40px;
      color: #718096;
    }

    .descriptors-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
      gap: 24px;
    }

    .descriptor-card {
      background: #f7fafc;
      border-radius: 8px;
      padding: 20px;
      border: 1px solid #e2e8f0;
    }

    .descriptor-card h4 {
      font-size: 18px;
      font-weight: 600;
      color: #2d3748;
      margin-bottom: 16px;
    }

    .chart-container {
      position: relative;
      height: 300px;
      margin-bottom: 16px;
    }

    .dominant-colors {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .color-item {
      display: flex;
      align-items: center;
      gap: 16px;
      padding: 12px;
      background: white;
      border-radius: 8px;
      border: 1px solid #e2e8f0;
    }

    .color-box {
      width: 60px;
      height: 60px;
      border-radius: 8px;
      border: 2px solid #cbd5e0;
      flex-shrink: 0;
    }

    .color-info {
      flex: 1;
    }

    .color-percentage {
      font-size: 18px;
      font-weight: 600;
      color: #2d3748;
      margin-bottom: 4px;
    }

    .color-rgb {
      font-size: 14px;
      color: #718096;
      font-family: 'Courier New', monospace;
    }

    .descriptor-values {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }

    .value-item {
      display: flex;
      flex-direction: column;
      padding: 12px;
      background: white;
      border-radius: 6px;
      border: 1px solid #e2e8f0;
    }

    .value-label {
      font-size: 12px;
      color: #718096;
      margin-bottom: 4px;
      text-transform: uppercase;
      font-weight: 500;
    }

    .value-number {
      font-size: 16px;
      font-weight: 600;
      color: #2d3748;
      font-family: 'Courier New', monospace;
    }

    @media (max-width: 768px) {
      .descriptors-grid {
        grid-template-columns: 1fr;
      }
      
      .chart-container {
        height: 250px;
      }
    }
  `]
})
export class DescriptorsViewerComponent implements OnInit, OnChanges, AfterViewInit, OnDestroy {
  @Input() descriptors: any;
  @Input() objectInfo: any;

  @ViewChild('colorHistCanvas') colorHistCanvas!: ElementRef<HTMLCanvasElement>;
  @ViewChild('tamuraCanvas') tamuraCanvas!: ElementRef<HTMLCanvasElement>;
  @ViewChild('gaborCanvas') gaborCanvas!: ElementRef<HTMLCanvasElement>;
  @ViewChild('huMomentsCanvas') huMomentsCanvas!: ElementRef<HTMLCanvasElement>;
  @ViewChild('hogCanvas') hogCanvas!: ElementRef<HTMLCanvasElement>;
  @ViewChild('lbpCanvas') lbpCanvas!: ElementRef<HTMLCanvasElement>;

  private charts: Chart[] = [];

  dominantColorsList: Array<{ rgb: string; percentage: number }> = [];
  tamuraValues: number[] = [];
  huMomentsList: number[] = [];

  ngOnInit() {
    this.updateData();
  }

  ngAfterViewInit() {
    setTimeout(() => this.createCharts(), 100);
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['descriptors']) {
      this.updateData();
      setTimeout(() => {
        this.destroyCharts();
        this.createCharts();
      }, 100);
    }
  }

  hasDescriptors(): boolean {
    return this.descriptors && Object.keys(this.descriptors).length > 0;
  }

  private updateData() {
    if (!this.descriptors) return;

    // Couleurs dominantes
    if (this.descriptors.dominant_colors) {
      const dc = this.descriptors.dominant_colors;
      if (dc.colors && Array.isArray(dc.colors) && dc.proportions && Array.isArray(dc.proportions)) {
        this.dominantColorsList = dc.colors.map((color: any, index: number) => {
          const rgbArray = Array.isArray(color) ? color : (color.color || [128, 128, 128]);
          const rgb = `rgb(${rgbArray[0]}, ${rgbArray[1]}, ${rgbArray[2]})`;
          const proportion = dc.proportions[index] || 0;
          return {
            rgb,
            percentage: proportion * 100
          };
        }).slice(0, 5);
      }
    }

    // Tamura
    if (this.descriptors.tamura) {
      this.tamuraValues = Array.isArray(this.descriptors.tamura) ? this.descriptors.tamura : [];
    }

    // Moments de Hu
    if (this.descriptors.hu_moments) {
      this.huMomentsList = Array.isArray(this.descriptors.hu_moments) ? this.descriptors.hu_moments : [];
    }
  }

  private destroyCharts() {
    this.charts.forEach(chart => chart.destroy());
    this.charts = [];
  }

  private createCharts() {
    if (!this.descriptors) return;

    // Histogramme des couleurs
    if (this.descriptors.color_hist && this.colorHistCanvas) {
      const hist = Array.isArray(this.descriptors.color_hist) ? this.descriptors.color_hist : [];
      const bins = Math.min(hist.length, 32);
      const labels = Array.from({ length: bins }, (_, i) => `Bin ${i + 1}`);
      const data = hist.slice(0, bins);

      const config: ChartConfiguration<'bar'> = {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            data,
            backgroundColor: 'rgba(66, 153, 225, 0.6)',
            borderColor: 'rgba(66, 153, 225, 1)',
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: { y: { beginAtZero: true } }
        }
      };
      this.charts.push(new Chart(this.colorHistCanvas.nativeElement, config));
    }

    // Tamura (radar)
    if (this.descriptors.tamura && this.tamuraCanvas && this.tamuraValues.length >= 3) {
      const config: ChartConfiguration<'radar'> = {
        type: 'radar',
        data: {
          labels: ['Rugosité', 'Contraste', 'Directionnalité'],
          datasets: [{
            data: [this.tamuraValues[0] || 0, this.tamuraValues[1] || 0, this.tamuraValues[2] || 0],
            backgroundColor: 'rgba(66, 153, 225, 0.2)',
            borderColor: 'rgba(66, 153, 225, 1)',
            borderWidth: 2,
            pointBackgroundColor: 'rgba(66, 153, 225, 1)'
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: { r: { beginAtZero: true } }
        }
      };
      this.charts.push(new Chart(this.tamuraCanvas.nativeElement, config));
    }

    // Gabor
    if (this.descriptors.gabor && this.gaborCanvas) {
      const gabor = Array.isArray(this.descriptors.gabor) ? this.descriptors.gabor : [];
      const sampleSize = Math.min(gabor.length, 50);
      const labels = Array.from({ length: sampleSize }, (_, i) => `F${i + 1}`);

      const config: ChartConfiguration<'line'> = {
        type: 'line',
        data: {
          labels,
          datasets: [{
            label: 'Magnitude',
            data: gabor.slice(0, sampleSize),
            borderColor: 'rgba(139, 92, 246, 1)',
            backgroundColor: 'rgba(139, 92, 246, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.4
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: { y: { beginAtZero: true } }
        }
      };
      this.charts.push(new Chart(this.gaborCanvas.nativeElement, config));
    }

    // Moments de Hu
    if (this.descriptors.hu_moments && this.huMomentsCanvas && this.huMomentsList.length > 0) {
      const labels = Array.from({ length: Math.min(this.huMomentsList.length, 7) }, (_, i) => `M${i + 1}`);

      const config: ChartConfiguration<'bar'> = {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            data: this.huMomentsList.slice(0, 7),
            backgroundColor: 'rgba(34, 197, 94, 0.6)',
            borderColor: 'rgba(34, 197, 94, 1)',
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: { y: { beginAtZero: true } }
        }
      };
      this.charts.push(new Chart(this.huMomentsCanvas.nativeElement, config));
    }

    // HOG
    if (this.descriptors.hog && this.hogCanvas) {
      const hog = Array.isArray(this.descriptors.hog) ? this.descriptors.hog : [];
      const sampleSize = Math.min(hog.length, 36);
      const labels = Array.from({ length: sampleSize }, (_, i) => `H${i + 1}`);

      const config: ChartConfiguration<'bar'> = {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            data: hog.slice(0, sampleSize),
            backgroundColor: 'rgba(245, 158, 11, 0.6)',
            borderColor: 'rgba(245, 158, 11, 1)',
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: { y: { beginAtZero: true } }
        }
      };
      this.charts.push(new Chart(this.hogCanvas.nativeElement, config));
    }

    // LBP
    if (this.descriptors.lbp && this.lbpCanvas) {
      const lbp = Array.isArray(this.descriptors.lbp) ? this.descriptors.lbp : [];
      const labels = Array.from({ length: lbp.length }, (_, i) => `L${i + 1}`);

      const config: ChartConfiguration<'bar'> = {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            data: lbp,
            backgroundColor: 'rgba(236, 72, 153, 0.6)',
            borderColor: 'rgba(236, 72, 153, 1)',
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: { y: { beginAtZero: true } }
        }
      };
      this.charts.push(new Chart(this.lbpCanvas.nativeElement, config));
    }
  }

  ngOnDestroy() {
    this.destroyCharts();
  }
}
