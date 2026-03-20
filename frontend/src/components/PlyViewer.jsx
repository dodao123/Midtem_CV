/**
 * Three.js PLY model viewer component.
 * Renders mesh or point cloud from PLY files with orbit controls.
 * Uses Phong shading to match Open3D rendering quality.
 */
import { Suspense, useState, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Center } from '@react-three/drei';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader';
import * as THREE from 'three';

/**
 * Internal component that loads and renders a PLY model.
 * @param {object} props - Component props.
 * @param {string} props.url - URL to the PLY file.
 * @param {string} props.type - 'mesh' or 'pointcloud'.
 */
function PlyModel({ url, type }) {
    const [geometry, setGeometry] = useState(null);

    useMemo(() => {
        const loader = new PLYLoader();
        loader.load(url, (geo) => {
            geo.computeVertexNormals();
            geo.center();

            /* Scale to fit viewer */
            geo.computeBoundingBox();
            const box = geo.boundingBox;
            const maxDim = Math.max(
                box.max.x - box.min.x,
                box.max.y - box.min.y,
                box.max.z - box.min.z,
            );
            const scale = 3 / maxDim;
            geo.scale(scale, scale, scale);

            setGeometry(geo);
        });
    }, [url]);

    if (!geometry) return null;

    if (type === 'pointcloud') {
        return (
            <points geometry={geometry}>
                <pointsMaterial
                    size={0.02}
                    vertexColors={!!geometry.attributes.color}
                    sizeAttenuation
                />
            </points>
        );
    }

    const hasColors = !!geometry.attributes.color;
    return (
        <mesh geometry={geometry}>
            <meshBasicMaterial
                vertexColors={hasColors}
                color={hasColors ? undefined : '#3B82F6'}
                side={THREE.DoubleSide}
            />
        </mesh>
    );
}

/**
 * 3D viewer with canvas, lighting, and orbit controls.
 * Lighting setup mimics Open3D default viewer for consistent quality.
 * @param {object} props - Component props.
 * @param {string} props.url - PLY file URL.
 * @param {string} props.type - 'mesh' or 'pointcloud'.
 * @param {number} [props.height=450] - Canvas height in pixels.
 */
export default function PlyViewer({ url, type = 'mesh', height = 450 }) {
    return (
        <div
            style={{
                width: '100%',
                height,
                borderRadius: 'var(--radius-sm)',
                overflow: 'hidden',
                background: '#080c14',
                border: '1px solid var(--color-border)',
            }}
        >
            <Canvas
                camera={{ position: [0, 0, 5], fov: 50 }}
                gl={{ antialias: true, toneMapping: THREE.NoToneMapping }}
                linear
            >
                <ambientLight intensity={0.7} />
                <directionalLight position={[5, 5, 5]} intensity={1.0} />
                <directionalLight position={[-5, 3, 3]} intensity={0.5} />
                <directionalLight position={[0, -3, 5]} intensity={0.3} />
                <Suspense fallback={null}>
                    <Center>
                        <PlyModel url={url} type={type} />
                    </Center>
                </Suspense>
                <OrbitControls
                    enableDamping
                    dampingFactor={0.1}
                    rotateSpeed={0.8}
                    zoomSpeed={0.8}
                />
            </Canvas>
        </div>
    );
}
