/**
 * Part C: Image Stitching page.
 * Upload multiple images → detect features → match → stitch panorama.
 */
import { useState, useRef } from 'react';
import { Upload, Play, Search, Layers, Image } from 'lucide-react';
import { matchFeatures, stitchImages } from '../services/api';

export default function StitchingPage() {
    const [files, setFiles] = useState([]);
    const [previews, setPreviews] = useState([]);
    const [loading, setLoading] = useState(false);
    const [detector, setDetector] = useState('sift');
    const [matchResult, setMatchResult] = useState(null);
    const [panorama, setPanorama] = useState(null);
    const fileRef = useRef();

    const handleFiles = (e) => {
        const selected = Array.from(e.target.files);
        setFiles(selected);
        setPreviews(selected.map((f) => URL.createObjectURL(f)));
        setMatchResult(null);
        setPanorama(null);
    };

    const handleMatch = async () => {
        if (files.length < 2) return;
        setLoading(true);
        try {
            const data = await matchFeatures(files[0], files[1], detector);
            setMatchResult(data);
        } catch (err) { console.error(err); }
        setLoading(false);
    };

    const handleStitch = async () => {
        if (files.length < 2) return;
        setLoading(true);
        try {
            const blob = await stitchImages(files, detector);
            setPanorama(URL.createObjectURL(blob));
        } catch (err) { console.error(err); }
        setLoading(false);
    };

    return (
        <>
            <div className="page-header">
                <h1 className="page-title">
                    <Layers size={22} style={{ display: 'inline', marginRight: 8 }} />
                    Part C — Image Stitching
                </h1>
                <p className="page-desc">
                    Feature detection, matching, and panorama stitching using SIFT & ORB.
                </p>
            </div>

            {/* Upload multiple images */}
            <div className="card">
                <div className="card-title"><Upload /> Upload Images (2–6)</div>
                <div className="upload-zone" onClick={() => fileRef.current?.click()}>
                    <input
                        type="file" ref={fileRef} accept="image/*"
                        multiple onChange={handleFiles}
                    />
                    <Upload size={28} color="var(--color-text-muted)" />
                    <div className="upload-text">
                        {files.length > 0
                            ? `${files.length} images selected`
                            : 'Click to select overlapping images'}
                    </div>
                    <div className="upload-hint">Select 2–6 images with overlapping regions</div>
                </div>
                {previews.length > 0 && (
                    <div className="image-grid" style={{ marginTop: 16 }}>
                        {previews.map((src, i) => (
                            <div className="image-item" key={i}>
                                <img src={src} alt={`Input image ${i + 1}`} />
                                <div className="image-caption">Image {i + 1}</div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Detector + Actions */}
            <div className="card">
                <div className="card-title"><Search /> Feature Detection & Matching</div>
                <div className="controls-row">
                    <div className="control-group">
                        <label className="control-label">Detector</label>
                        <select value={detector} onChange={(e) => setDetector(e.target.value)}>
                            <option value="sift">SIFT</option>
                            <option value="orb">ORB</option>
                        </select>
                    </div>
                    <button className="btn btn-secondary" onClick={handleMatch}
                        disabled={files.length < 2 || loading}>
                        {loading ? <span className="spinner" /> : <Search size={14} />}
                        Match Features
                    </button>
                    <button className="btn btn-primary" onClick={handleStitch}
                        disabled={files.length < 2 || loading}>
                        {loading ? <span className="spinner" /> : <Image size={14} />}
                        Stitch Panorama
                    </button>
                </div>

                {matchResult && (
                    <>
                        <div className="result-summary">
                            <span>Detector: <strong>{matchResult.detector}</strong></span>
                            <span>Matches: <strong>{matchResult.num_matches}</strong></span>
                        </div>
                        <div className="image-grid" style={{ marginTop: 16, gridTemplateColumns: '1fr' }}>
                            <div className="image-item">
                                <img
                                    src={`data:image/png;base64,${matchResult.visualization_b64}`}
                                    alt="Feature matches visualization"
                                />
                                <div className="image-caption">
                                    Feature Matches ({matchResult.detector.toUpperCase()})
                                </div>
                            </div>
                        </div>
                    </>
                )}
            </div>

            {/* Panorama Result */}
            {panorama && (
                <div className="card">
                    <div className="card-title"><Layers /> Stitched Panorama</div>
                    <div className="image-grid" style={{ gridTemplateColumns: '1fr' }}>
                        <div className="image-item">
                            <img src={panorama} alt="Stitched panorama result" />
                            <div className="image-caption">
                                Panorama ({detector.toUpperCase()}, {files.length} images)
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
}
