/**
 * 3D Model viewer section for StereoPage.
 * Lists available PLY models, lets user pick one, and renders it.
 */
import { useState } from 'react';
import { Box, RotateCcw } from 'lucide-react';
import PlyViewer from './PlyViewer';

const API_BASE = 'http://localhost:8000';

/**
 * Section card showing dataset selector and 3D viewer.
 * @param {object} props - Component props.
 * @param {Array} props.models - Available model metadata from API.
 */
export default function ModelViewerSection({ models }) {
    const [selected, setSelected] = useState(null);
    const [viewType, setViewType] = useState('mesh');

    const handleSelect = (model) => {
        setSelected(model);
        setViewType(model.has_mesh ? 'mesh' : 'pointcloud');
    };

    const plyUrl = selected
        ? `${API_BASE}${viewType === 'mesh' ? selected.mesh_url : selected.pointcloud_url}`
        : null;

    return (
        <div className="card">
            <div className="card-title"><Box /> 3D Model Viewer</div>

            {models.length === 0 && (
                <div className="loading-overlay">
                    <span className="spinner" /> Loading available models...
                </div>
            )}

            {models.length > 0 && (
                <>
                    <div className="controls-row">
                        <div className="control-group">
                            <label className="control-label">Dataset</label>
                            <select
                                value={selected?.name || ''}
                                onChange={(e) => {
                                    const model = models.find((m) => m.name === e.target.value);
                                    if (model) handleSelect(model);
                                }}
                            >
                                <option value="" disabled>Select a model...</option>
                                {models.map((m) => (
                                    <option key={m.name} value={m.name}>{m.label}</option>
                                ))}
                            </select>
                        </div>

                        {selected && (
                            <div className="control-group">
                                <label className="control-label">View</label>
                                <select
                                    value={viewType}
                                    onChange={(e) => setViewType(e.target.value)}
                                >
                                    {selected.has_mesh && <option value="mesh">Mesh</option>}
                                    {selected.has_pointcloud && (
                                        <option value="pointcloud">Point Cloud</option>
                                    )}
                                </select>
                            </div>
                        )}

                        {selected && (
                            <button
                                className="btn btn-secondary"
                                onClick={() => handleSelect(selected)}
                            >
                                <RotateCcw size={14} /> Reset View
                            </button>
                        )}
                    </div>

                    {selected?.mesh_size_mb && (
                        <div style={{
                            fontSize: '0.78rem',
                            color: 'var(--color-text-muted)',
                            marginBottom: 12,
                            fontFamily: 'var(--font-heading)',
                        }}>
                            File size: {selected.mesh_size_mb} MB — Drag to rotate, scroll to zoom
                        </div>
                    )}
                </>
            )}

            {plyUrl && <PlyViewer url={plyUrl} type={viewType} key={plyUrl} />}
        </div>
    );
}
