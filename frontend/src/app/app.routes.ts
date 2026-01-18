import { Routes } from '@angular/router';
import { Cbir3dSearchComponent } from './components/cbir3d-search/cbir3d-search.component';
import { HistoryComponent } from './components/history/history.component';
import { UploadComponent } from './components/upload/upload.component';


export const routes: Routes = [
  { path: '', redirectTo: '/upload', pathMatch: 'full' },
  { path: 'upload', component: UploadComponent },
  { path: 'history', component: HistoryComponent },
  { path: 'search-3d', component: Cbir3dSearchComponent },

  { path: '**', redirectTo: '/upload' }
];

