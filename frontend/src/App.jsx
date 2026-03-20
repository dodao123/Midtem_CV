/**
 * App root — React Router configuration.
 * Routes map to Dashboard, Filtering, Stereo, and Stitching pages.
 */
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import DashboardPage from './pages/DashboardPage';
import FilteringPage from './pages/FilteringPage';
import StereoPage from './pages/StereoPage';
import StitchingPage from './pages/StitchingPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<DashboardPage />} />
          <Route path="filtering" element={<FilteringPage />} />
          <Route path="stereo" element={<StereoPage />} />
          <Route path="stitching" element={<StitchingPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
