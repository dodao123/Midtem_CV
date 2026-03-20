/**
 * Dashboard page — project overview with 3 part cards.
 */
import { Link } from 'react-router-dom';
import { Palette, Box, Layers, CheckCircle2 } from 'lucide-react';

const PARTS = [
    {
        icon: Palette,
        iconClass: 'blue',
        title: 'Part A — Image Filtering',
        desc: 'Mean, Gaussian, Median filters + Laplacian sharpening. PSNR & SSIM comparative analysis.',
        link: '/filtering',
        badge: '20 pts',
    },
    {
        icon: Box,
        iconClass: 'amber',
        title: 'Part B — 3D Reconstruction',
        desc: 'Stereo disparity maps (BM vs SGBM), epipolar geometry, 3D point cloud & mesh.',
        link: '/stereo',
        badge: '25 pts',
    },
    {
        icon: Layers,
        iconClass: 'green',
        title: 'Part C — Image Stitching',
        desc: 'SIFT/ORB feature detection, homography estimation, and panorama blending.',
        link: '/stitching',
        badge: '25 pts',
    },
];

const REQUIREMENTS = [
    'Only traditional image processing methods (no deep learning)',
    'Libraries: OpenCV, NumPy, Matplotlib, Open3D',
    'Compare ≥2 methods per part with quantitative metrics',
    'FastAPI backend + React frontend architecture',
];

export default function DashboardPage() {
    return (
        <>
            <div className="page-header">
                <h1 className="page-title">&gt;_ Computer Vision Midterm</h1>
                <p className="page-desc">
                    Traditional Image Processing — Filtering, 3D Reconstruction, and Image Stitching
                </p>
            </div>

            <div className="dashboard-grid">
                {PARTS.map((part) => (
                    <Link to={part.link} className="dash-card" key={part.link}>
                        <div className="card">
                            <div className={`card-icon ${part.iconClass}`}>
                                <part.icon />
                            </div>
                            <div className="card-title" style={{ justifyContent: 'space-between' }}>
                                {part.title}
                                <span className="badge badge-cta">{part.badge}</span>
                            </div>
                            <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.85rem' }}>
                                {part.desc}
                            </p>
                        </div>
                    </Link>
                ))}
            </div>

            <div className="card" style={{ marginTop: 8 }}>
                <div className="card-title">
                    <CheckCircle2 /> Project Requirements
                </div>
                <ul style={{ paddingLeft: 20, color: 'var(--color-text-secondary)', lineHeight: 2, fontSize: '0.88rem' }}>
                    {REQUIREMENTS.map((req, i) => (
                        <li key={i}>{req}</li>
                    ))}
                </ul>
            </div>
        </>
    );
}
