/**
 * Part B: Stereo 3D Reconstruction page.
 * Upload stereo pair → compute disparity → reconstruct 3D → epipolar.
 * Includes 3D model viewer for viewing generated PLY meshes.
 */
import { useState, useRef, useEffect } from 'react';
import { Upload, Play, GitCompare, Scan, Box } from 'lucide-react';
import { computeDisparity, computeEpipolar, list3dModels, reconstruct3D } from '../services/api';
import ModelViewerSection from '../components/ModelViewerSection';


export default function StereoPage() {
    const [leftFile, setLeftFile] = useState(null);
    const [rightFile, setRightFile] = useState(null);
    const [leftPreview, setLeftPreview] = useState(null);
    const [rightPreview, setRightPreview] = useState(null);
    const [loading, setLoading] = useState(false);
    const [dispResult, setDispResult] = useState(null);
    const [epiResult, setEpiResult] = useState(null);
    const [method, setMethod] = useState('sgbm');
    const [numDisp, setNumDisp] = useState(64);
    const [blockSize, setBlockSize] = useState(5);
    const [models, setModels] = useState([]);
    const [reconstructing, setReconstructing] = useState(false);
    const leftRef = useRef();
    const rightRef = useRef();

    const refreshModels = () => list3dModels().then(setModels).catch(console.error);
    useEffect(() => { refreshModels(); }, []);

    const handleLeft = (e) => {
        const f = e.target.files[0]; if (!f) return;
        setLeftFile(f); setLeftPreview(URL.createObjectURL(f));
    };

    const handleRight = (e) => {
        const f = e.target.files[0]; if (!f) return;
        setRightFile(f); setRightPreview(URL.createObjectURL(f));
    };

    const handleDisparity = async () => {
        if (!leftFile || !rightFile) return;
        setLoading(true);
        try {
            const blob = await computeDisparity(leftFile, rightFile, method, {
                numDisparities: numDisp, blockSize,
            });
            setDispResult(URL.createObjectURL(blob));
        } catch (err) { console.error(err); }
        setLoading(false);
    };

    const handleReconstruct = async () => {
        if (!leftFile || !rightFile) return;
        setReconstructing(true);
        try {
            await reconstruct3D(leftFile, rightFile, method, {
                numDisparities: numDisp, blockSize,
            });
            await refreshModels();
        } catch (err) { console.error(err); }
        setReconstructing(false);
    };

    const handleEpipolar = async () => {
        if (!leftFile || !rightFile) return;
        setLoading(true);
        try {
            const blob = await computeEpipolar(leftFile, rightFile);
            setEpiResult(URL.createObjectURL(blob));
        } catch (err) { console.error(err); }
        setLoading(false);
    };

    const hasBoth = leftFile && rightFile;

    return (
        <>
            <div className="page-header">
                <h1 className="page-title">
                    <GitCompare size={22} style={{ display: 'inline', marginRight: 8 }} />
                    Part B — 3D Stereo Reconstruction
                </h1>
                <p className="page-desc">
                    Compute disparity maps, reconstruct 3D models, and view interactive results.
                </p>
            </div>

            <ModelViewerSection models={models} />



            {/* Upload */}
            <div className="card">
                <div className="card-title"><Upload /> Upload Stereo Pair</div>
                <div className="image-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
                    <StereoUpload refEl={leftRef} file={leftFile} label="Left Image" onChange={handleLeft} />
                    <StereoUpload refEl={rightRef} file={rightFile} label="Right Image" onChange={handleRight} />
                </div>
                {leftPreview && rightPreview && (
                    <div className="image-grid" style={{ marginTop: 16, gridTemplateColumns: '1fr 1fr' }}>
                        <div className="image-item"><img src={leftPreview} alt="Left" /><div className="image-caption">Left</div></div>
                        <div className="image-item"><img src={rightPreview} alt="Right" /><div className="image-caption">Right</div></div>
                    </div>
                )}
            </div>

            {/* Disparity + Reconstruct 3D */}
            <DisparitySection
                method={method} setMethod={setMethod}
                numDisp={numDisp} setNumDisp={setNumDisp}
                blockSize={blockSize} setBlockSize={setBlockSize}
                hasBoth={hasBoth} loading={loading}
                onCompute={handleDisparity} result={dispResult}
                reconstructing={reconstructing} onReconstruct={handleReconstruct}
            />

            {/* Epipolar */}
            <div className="card">
                <div className="card-title"><GitCompare /> Epipolar Lines</div>
                <button className="btn btn-primary" onClick={handleEpipolar} disabled={!hasBoth || loading}>
                    {loading ? <span className="spinner" /> : <Play size={14} />} Compute Epipolar
                </button>
                {epiResult && (
                    <div className="image-grid" style={{ marginTop: 16, gridTemplateColumns: '1fr' }}>
                        <div className="image-item"><img src={epiResult} alt="Epipolar lines" /><div className="image-caption">Epipolar Lines (SIFT + RANSAC)</div></div>
                    </div>
                )}
            </div>
        </>
    );
}

/** Reusable stereo upload zone. */
function StereoUpload({ refEl, file, label, onChange }) {
    return (
        <div className="upload-zone" onClick={() => refEl.current?.click()}>
            <input type="file" ref={refEl} accept="image/*" onChange={onChange} />
            <Upload size={24} color="var(--color-text-muted)" />
            <div className="upload-text">{file ? file.name : label}</div>
        </div>
    );
}

/** Disparity controls + Reconstruct 3D button. */
function DisparitySection({ method, setMethod, numDisp, setNumDisp, blockSize, setBlockSize, hasBoth, loading, onCompute, result, reconstructing, onReconstruct }) {
    return (
        <div className="card">
            <div className="card-title"><Scan /> Disparity Map & 3D Reconstruction</div>
            <div className="controls-row">
                <div className="control-group"><label className="control-label">Method</label>
                    <select value={method} onChange={(e) => setMethod(e.target.value)}>
                        <option value="bm">Block Matching</option><option value="sgbm">SGBM</option>
                    </select>
                </div>
                <div className="control-group"><label className="control-label">Disparities</label>
                    <select value={numDisp} onChange={(e) => setNumDisp(Number(e.target.value))}>
                        <option value={16}>16</option><option value={32}>32</option>
                        <option value={64}>64</option><option value={128}>128</option>
                    </select>
                </div>
                <div className="control-group"><label className="control-label">Block Size</label>
                    <select value={blockSize} onChange={(e) => setBlockSize(Number(e.target.value))}>
                        <option value={3}>3</option><option value={5}>5</option>
                        <option value={7}>7</option><option value={15}>15</option>
                    </select>
                </div>
                <button className="btn btn-primary" onClick={onCompute} disabled={!hasBoth || loading}>
                    {loading ? <span className="spinner" /> : <Play size={14} />} Disparity
                </button>
                <button className="btn btn-secondary" onClick={onReconstruct} disabled={!hasBoth || reconstructing}
                    style={{ borderColor: 'var(--color-cta)', color: 'var(--color-cta)' }}>
                    {reconstructing ? <><span className="spinner" /> Building 3D...</> : <><Box size={14} /> Reconstruct 3D</>}
                </button>
            </div>
            {result && (
                <div className="image-grid" style={{ gridTemplateColumns: '1fr' }}>
                    <div className="image-item"><img src={result} alt="Disparity map" />
                        <div className="image-caption">{method.toUpperCase()} Disparity (d={numDisp}, b={blockSize})</div>
                    </div>
                </div>
            )}
        </div>
    );
}
