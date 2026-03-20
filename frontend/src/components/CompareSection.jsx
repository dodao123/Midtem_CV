/**
 * Comparative analysis section for FilteringPage.
 * Shows noise controls, runs all filters, displays metrics table + images.
 */
import { BarChart3 } from 'lucide-react';

export default function CompareSection({
    file, noiseType, setNoiseType, sigma, setSigma, loading, data, onCompare,
}) {
    return (
        <div className="card">
            <div className="card-title"><BarChart3 /> Comparative Analysis</div>
            <div className="controls-row">
                <div className="control-group">
                    <label className="control-label">Noise</label>
                    <select value={noiseType} onChange={(e) => setNoiseType(e.target.value)}>
                        <option value="gaussian">Gaussian</option>
                        <option value="salt_pepper">Salt & Pepper</option>
                    </select>
                </div>
                <div className="control-group">
                    <label className="control-label">Sigma</label>
                    <input
                        type="number" value={sigma}
                        onChange={(e) => setSigma(Number(e.target.value))}
                        min={1} max={100} style={{ width: 80 }}
                    />
                </div>
                <button className="btn btn-primary" onClick={onCompare} disabled={!file || loading}>
                    {loading ? <span className="spinner" /> : <BarChart3 size={14} />}
                    Compare All
                </button>
            </div>

            {data && (
                <>
                    <table className="metrics-table">
                        <thead>
                            <tr><th>Filter</th><th>PSNR (dB)</th><th>SSIM</th></tr>
                        </thead>
                        <tbody>
                            {data.filters.map((f, i) => (
                                <tr key={i}>
                                    <td>{f.name}</td>
                                    <td className={f.name === data.best_psnr ? 'best-value' : ''}>
                                        {f.psnr}
                                    </td>
                                    <td className={f.name === data.best_ssim ? 'best-value' : ''}>
                                        {f.ssim}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>

                    <div className="result-summary">
                        <span>Best PSNR: <strong>{data.best_psnr}</strong></span>
                        <span>Best SSIM: <strong>{data.best_ssim}</strong></span>
                    </div>

                    <div className="image-grid">
                        <div className="image-item">
                            <img src={`data:image/png;base64,${data.original_b64}`} alt="Original" />
                            <div className="image-caption">Original</div>
                        </div>
                        <div className="image-item">
                            <img src={`data:image/png;base64,${data.noisy_b64}`} alt="Noisy" />
                            <div className="image-caption">Noisy ({noiseType})</div>
                        </div>
                        {data.filters.slice(0, 4).map((f, i) => (
                            <div className="image-item" key={i}>
                                <img src={`data:image/png;base64,${f.image_b64}`} alt={f.name} />
                                <div className="image-caption">{f.name}</div>
                            </div>
                        ))}
                    </div>
                </>
            )}
        </div>
    );
}
