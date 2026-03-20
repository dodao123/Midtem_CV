/**
 * Part A: Image Filtering page.
 * Upload → apply single filter → run comparative analysis.
 */
import { useState, useRef } from 'react';
import { Upload, Play, BarChart3, Sparkles } from 'lucide-react';
import { applyFilter, compareFilters } from '../services/api';
import CompareSection from '../components/CompareSection';

export default function FilteringPage() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [singleResult, setSingleResult] = useState(null);
  const [compareData, setCompareData] = useState(null);
  const [filterType, setFilterType] = useState('gaussian');
  const [kernelSize, setKernelSize] = useState(5);
  const [noiseType, setNoiseType] = useState('gaussian');
  const [sigma, setSigma] = useState(25);
  const fileRef = useRef();

  const handleFile = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setSingleResult(null);
    setCompareData(null);
  };

  const handleApply = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const blob = await applyFilter(file, filterType, { kernelSize, sigma });
      setSingleResult(URL.createObjectURL(blob));
    } catch (err) { console.error(err); }
    setLoading(false);
  };

  const handleCompare = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const data = await compareFilters(file, noiseType, sigma);
      setCompareData(data);
    } catch (err) { console.error(err); }
    setLoading(false);
  };

  return (
    <>
      <div className="page-header">
        <h1 className="page-title"><Sparkles size={22} style={{ display: 'inline', marginRight: 8 }} />Part A — Image Filtering</h1>
        <p className="page-desc">Apply traditional filters for noise reduction. Compare PSNR & SSIM metrics.</p>
      </div>

      {/* Upload */}
      <div className="card">
        <div className="card-title"><Upload /> Upload Image</div>
        <div className="upload-zone" onClick={() => fileRef.current?.click()}>
          <input type="file" ref={fileRef} accept="image/*" onChange={handleFile} />
          <Upload size={28} color="var(--color-text-muted)" />
          <div className="upload-text">{file ? file.name : 'Click to select an image'}</div>
          <div className="upload-hint">Supports JPG, PNG, BMP</div>
        </div>
        {preview && (
          <div className="image-grid" style={{ marginTop: 16 }}>
            <div className="image-item">
              <img src={preview} alt="Original uploaded image" />
              <div className="image-caption">Original</div>
            </div>
          </div>
        )}
      </div>

      {/* Single Filter */}
      <div className="card">
        <div className="card-title"><Play /> Apply Single Filter</div>
        <div className="controls-row">
          <div className="control-group">
            <label className="control-label">Filter</label>
            <select value={filterType} onChange={(e) => setFilterType(e.target.value)}>
              <option value="mean">Mean</option>
              <option value="gaussian">Gaussian</option>
              <option value="median">Median</option>
              <option value="laplacian">Laplacian Sharpening</option>
            </select>
          </div>
          <div className="control-group">
            <label className="control-label">Kernel</label>
            <select value={kernelSize} onChange={(e) => setKernelSize(Number(e.target.value))}>
              <option value={3}>3×3</option>
              <option value={5}>5×5</option>
              <option value={7}>7×7</option>
            </select>
          </div>
          <button className="btn btn-primary" onClick={handleApply} disabled={!file || loading}>
            {loading ? <span className="spinner" /> : <Play size={14} />} Apply
          </button>
        </div>
        {singleResult && (
          <div className="image-grid">
            <div className="image-item">
              <img src={preview} alt="Original" />
              <div className="image-caption">Original</div>
            </div>
            <div className="image-item">
              <img src={singleResult} alt="Filtered result" />
              <div className="image-caption">{filterType} (k={kernelSize})</div>
            </div>
          </div>
        )}
      </div>

      {/* Compare */}
      <CompareSection
        file={file}
        noiseType={noiseType}
        setNoiseType={setNoiseType}
        sigma={sigma}
        setSigma={setSigma}
        loading={loading}
        data={compareData}
        onCompare={handleCompare}
      />
    </>
  );
}
