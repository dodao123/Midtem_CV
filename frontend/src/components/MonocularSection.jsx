/**
 * Monocular 3D Reconstruction section.
 * Upload a single image → depth estimation → stereo synthesis → 3D model.
 */
import { useState, useRef } from 'react';
import { Camera, Play, Eye } from 'lucide-react';
import PlyViewer from './PlyViewer';

const API_BASE = 'http://localhost:8000';

/**
 * MonocularSection component for single-image 3D reconstruction.
 * @param {object} props - Component props.
 * @param {Function} props.onModelCreated - Callback when a new model is made.
 */
export default function MonocularSection({ onModelCreated }) {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [viewType, setViewType] = useState('mesh');
    const fileRef = useRef();

    const handleFile = (event) => {
        const selected = event.target.files[0];
        if (!selected) return;
        setFile(selected);
        setPreview(URL.createObjectURL(selected));
        setResult(null);
    };

    const handleReconstruct = async () => {
        if (!file) return;
        setLoading(true);
        setResult(null);
        try {
            const response = await import('../services/api')
                .then((m) => m.monocularReconstruct3D(file));
            setResult(response);
            if (onModelCreated) onModelCreated();
        } catch (error) {
            console.error('Monocular reconstruction failed:', error);
        }
        setLoading(false);
    };

    return (
        <div className="card">
            <div className="card-title">
                <Camera /> Monocular 3D — Single Image to 3D
            </div>
            <p style={descStyle}>
                Upload any photo → AI estimates depth → generates stereo pair → reconstructs 3D model.
            </p>

            <MonocularUpload
                fileRef={fileRef}
                file={file}
                onChange={handleFile}
            />

            {preview && (
                <div className="image-grid" style={{ marginTop: 16 }}>
                    <div className="image-item">
                        <img src={preview} alt="Input" />
                        <div className="image-caption">Input Image</div>
                    </div>
                </div>
            )}

            <button
                className="btn btn-primary"
                onClick={handleReconstruct}
                disabled={!file || loading}
                style={{ marginTop: 16 }}
            >
                {loading
                    ? <><span className="spinner" /> Processing (may take a minute)...</>
                    : <><Play size={14} /> Reconstruct 3D from Single Image</>
                }
            </button>

            {result && <MonocularResults result={result} viewType={viewType} setViewType={setViewType} />}
        </div>
    );
}

/** Upload zone for a single image. */
function MonocularUpload({ fileRef, file, onChange }) {
    return (
        <div className="upload-zone" onClick={() => fileRef.current?.click()}>
            <input type="file" ref={fileRef} accept="image/*" onChange={onChange} />
            <Camera size={24} color="var(--color-text-muted)" />
            <div className="upload-text">
                {file ? file.name : 'Click to upload any photo'}
            </div>
        </div>
    );
}

/** Display pipeline results: depth, stereo pair, 3D viewer. */
function MonocularResults({ result, viewType, setViewType }) {
    const plyUrl = viewType === 'mesh'
        ? `${API_BASE}${result.mesh_url}`
        : `${API_BASE}${result.pointcloud_url}`;

    return (
        <div style={{ marginTop: 20 }}>
            <h3 style={sectionTitle}>
                <Eye size={16} /> Pipeline Results
            </h3>

            {/* Step 1: Depth Map */}
            <div className="image-grid" style={{ marginTop: 12 }}>
                <div className="image-item">
                    <img src={`${API_BASE}${result.depth_url}`} alt="Depth Map" />
                    <div className="image-caption">
                        Step 1 — AI Depth Estimation (MiDaS)
                    </div>
                </div>
            </div>

            {/* Step 2: Stereo Pair */}
            <div className="image-grid" style={{ marginTop: 12, gridTemplateColumns: '1fr 1fr' }}>
                <div className="image-item">
                    <img src={`${API_BASE}${result.left_url}`} alt="Left" />
                    <div className="image-caption">Step 2a — Left (Original)</div>
                </div>
                <div className="image-item">
                    <img src={`${API_BASE}${result.right_url}`} alt="Right" />
                    <div className="image-caption">Step 2b — Right (Synthesized)</div>
                </div>
            </div>

            {/* Step 3: 3D Model */}
            <h3 style={{ ...sectionTitle, marginTop: 20 }}>
                Step 3 — 3D Reconstruction (from Depth)
            </h3>
            <div className="controls-row" style={{ marginBottom: 12 }}>
                <div className="control-group">
                    <label className="control-label">View</label>
                    <select value={viewType} onChange={(e) => setViewType(e.target.value)}>
                        <option value="mesh">Mesh</option>
                        <option value="pointcloud">Point Cloud</option>
                    </select>
                </div>
            </div>
            <PlyViewer url={plyUrl} type={viewType} key={plyUrl} />
        </div>
    );
}

const descStyle = {
    fontSize: '0.85rem',
    color: 'var(--color-text-muted)',
    marginBottom: 16,
    lineHeight: 1.5,
};

const sectionTitle = {
    fontSize: '0.95rem',
    fontFamily: 'var(--font-heading)',
    color: 'var(--color-text)',
    marginBottom: 8,
    display: 'flex',
    alignItems: 'center',
    gap: 6,
};
