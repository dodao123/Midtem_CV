/**
 * Sidebar layout with navigation for all 3 parts.
 * Uses Lucide icons (no emojis per design system rules).
 */
import { NavLink, Outlet } from 'react-router-dom';
import { LayoutDashboard, Palette, Box, Layers } from 'lucide-react';

const NAV_ITEMS = [
    { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
    { to: '/filtering', label: 'Part A — Filtering', icon: Palette },
    { to: '/stereo', label: 'Part B — 3D Stereo', icon: Box },
    { to: '/stitching', label: 'Part C — Stitching', icon: Layers },
];

export default function Layout() {
    return (
        <div className="app-layout">
            <aside className="sidebar">
                <div className="sidebar-brand">
                    <div className="sidebar-logo">&gt;_ CV Midterm</div>
                </div>
                <div className="sidebar-subtitle">INS3155 // Traditional CV</div>
                <nav className="sidebar-nav">
                    {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
                        <NavLink
                            key={to}
                            to={to}
                            end={end}
                            className={({ isActive }) =>
                                `nav-link${isActive ? ' active' : ''}`
                            }
                        >
                            <Icon />
                            {label}
                        </NavLink>
                    ))}
                </nav>
            </aside>
            <main className="main-content">
                <Outlet />
            </main>
        </div>
    );
}
