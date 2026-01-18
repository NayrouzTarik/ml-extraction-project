import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

export interface Cbir3dResult {
  model: string;
  distance: number;
  preview_url?: string;
}

export interface ModelItem {
  model: string;
  preview_url: string;
  model_url: string;
}

@Injectable({ providedIn: 'root' })
export class Cbir3dService {

  private readonly baseUrl = 'http://127.0.0.1:5000';

  constructor(private http: HttpClient) {}

  status(): Observable<any> {
    return this.http.get(`${this.baseUrl}/api3d/status`);
  }

  index(limit = 50, nViews = 12, imgSize = 256): Observable<any> {
    return this.http.post(`${this.baseUrl}/api3d/index`, {
      limit,
      n_views: nViews,
      img_size: imgSize
    });
  }

  search(
    file: File,
    topK = 5,
    nViews = 12,
    imgSize = 256
  ): Observable<{ status: string; query: string; query_preview_url?: string; top_k: Cbir3dResult[] }> {

    const form = new FormData();
    form.append('file', file);
    form.append('top_k', String(topK));
    form.append('n_views', String(nViews));
    form.append('img_size', String(imgSize));

    return this.http.post<{ status: string; query: string; query_preview_url?: string; top_k: Cbir3dResult[] }>(
      `${this.baseUrl}/api3d/search`,
      form
    );
  }

  // ✅ ADD THIS METHOD (INSIDE THE CLASS)
  listModels(limit = 80): Observable<{ status: string; models: ModelItem[] }> {
    return this.http.get<{ status: string; models: ModelItem[] }>(
      `${this.baseUrl}/api3d/models?limit=${limit}`
    );
  }

  // ✅ ADD THIS METHOD (INSIDE THE CLASS)
  searchByName(
    model: string,
    topK = 5,
    nViews = 12,
    imgSize = 256
  ): Observable<{ status: string; query: string; query_preview_url?: string; top_k: Cbir3dResult[] }> {

    return this.http.post<{ status: string; query: string; query_preview_url?: string; top_k: Cbir3dResult[] }>(
      `${this.baseUrl}/api3d/search_by_name`,
      {
        model,
        top_k: topK,
        n_views: nViews,
        img_size: imgSize
      }
    );
  }
}
