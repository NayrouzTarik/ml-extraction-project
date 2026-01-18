import { Injectable } from '@angular/core';
import { HttpClient, HttpEvent, HttpEventType } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

const API_BASE_URL = 'http://localhost:5000/api';

export interface Image {
  image_id: number;
  filename: string;
  filepath: string;
  category?: string;
  width: number;
  height: number;
  upload_date: string;
  num_objects: number;
}

export interface ObjectDetection {
  object_id: number;
  class_id: number;
  class_name: string;
  bbox: number[];
  confidence_score: number;
}

export interface ImageDescriptor {
  image_id: number;
  filename: string;
  width: number;
  height: number;
  category?: string;
  upload_date: string;
  num_objects: number;
  objects: ObjectDetection[];
}

export interface SearchResult {
  object_id: number;
  image_id: number;
  filename: string;
  filepath: string;
  class_id: number;
  class_name: string;
  bbox: number[];
  confidence_score: number;
  distance: number;
  similarity: number;
}

export interface SearchResponse {
  query_object_id?: number;
  query_image_id?: number;
  query_class_name: string;
  num_results: number;
  results: SearchResult[];
}

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  constructor(private http: HttpClient) {}

  uploadImages(files: FileList): Observable<any> {
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('images', files[i]);
    }
    return this.http.post(`${API_BASE_URL}/upload`, formData);
  }

  uploadSingleImage(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('images', file);
    return this.http.post(`${API_BASE_URL}/upload`, formData);
  }

  getImages(category?: string): Observable<{ total: number; images: Image[] }> {
    const url = category 
      ? `${API_BASE_URL}/images?category=${category}`
      : `${API_BASE_URL}/images`;
    return this.http.get<{ total: number; images: Image[] }>(url);
  }

  getImageDescriptors(imageId: number, includeDescriptors = false): Observable<ImageDescriptor> {
    const url = includeDescriptors
      ? `${API_BASE_URL}/descriptors/${imageId}?include_descriptors=true`
      : `${API_BASE_URL}/descriptors/${imageId}`;
    return this.http.get<ImageDescriptor>(url);
  }

  searchSimilar(objectId?: number, imageId?: number, excludeSelf = true): Observable<SearchResponse> {
    const body: any = { exclude_self: excludeSelf };
    if (objectId) {
      body.object_id = objectId;
    } else if (imageId) {
      body.image_id = imageId;
    }
    return this.http.post<SearchResponse>(`${API_BASE_URL}/search`, body);
  }

  deleteImage(imageId: number): Observable<any> {
    return this.http.delete(`${API_BASE_URL}/images/${imageId}`);
  }

  getObject(objectId: number, includeDescriptors = false): Observable<any> {
    const url = includeDescriptors
      ? `${API_BASE_URL}/objects/${objectId}?include_descriptors=true`
      : `${API_BASE_URL}/objects/${objectId}`;
    return this.http.get(url);
  }

  getImageUrl(filepath: string, source?: string): string {
    const baseUrl = 'http://localhost:5000';
    
    // Si source est val_dataset, utiliser le endpoint val_images
    if (source === 'val_dataset') {
      const filename = filepath.split(/[/\\]/).pop() || filepath;
      return `${baseUrl}/val_images/${filename}`;
    }
    
    // Convertir le chemin du serveur en URL pour les uploads
    // Si le chemin contient déjà 'uploads', utiliser directement
    if (filepath.includes('uploads')) {
      return `${baseUrl}/${filepath}`;
    }
    return `${baseUrl}/uploads/${filepath.split(/[/\\]/).pop()}`;
  }

  transformImage(params: any): Observable<any> {
    return this.http.post(`${API_BASE_URL}/transform`, params);
  }
}

