/**
 * API service for communicating with the FastAPI backend.
 * All endpoints proxy through Vite dev server to port 8000.
 */

const API_BASE = '/api';

/**
 * Apply a single filter to an image.
 * @param {File} imageFile - The image file to filter.
 * @param {string} filterType - Filter type name.
 * @param {object} params - Filter parameters.
 * @returns {Promise<Blob>} Filtered image as blob.
 */
export async function applyFilter(imageFile, filterType, params = {}) {
    const form = new FormData();
    form.append('image', imageFile);
    form.append('filter_type', filterType);
    form.append('kernel_size', params.kernelSize || 5);
    form.append('sigma', params.sigma || 1.0);
    form.append('alpha', params.alpha || 0.5);

    const res = await fetch(`${API_BASE}/filtering/apply`, {
        method: 'POST',
        body: form,
    });
    return res.blob();
}

/**
 * Add noise to an image.
 * @param {File} imageFile - The image file.
 * @param {string} noiseType - 'gaussian' or 'salt_pepper'.
 * @param {object} params - Noise parameters.
 * @returns {Promise<Blob>} Noisy image blob.
 */
export async function addNoise(imageFile, noiseType, params = {}) {
    const form = new FormData();
    form.append('image', imageFile);
    form.append('noise_type', noiseType);
    form.append('sigma', params.sigma || 25.0);
    form.append('salt_prob', params.saltProb || 0.05);
    form.append('pepper_prob', params.pepperProb || 0.05);

    const res = await fetch(`${API_BASE}/filtering/add-noise`, {
        method: 'POST',
        body: form,
    });
    return res.blob();
}

/**
 * Run comparative filter analysis.
 * @param {File} imageFile - Clean image file.
 * @param {string} noiseType - Noise type to apply.
 * @param {number} sigma - Noise sigma.
 * @returns {Promise<object>} JSON with metrics and base64 images.
 */
export async function compareFilters(imageFile, noiseType = 'gaussian', sigma = 25.0) {
    const form = new FormData();
    form.append('image', imageFile);
    form.append('noise_type', noiseType);
    form.append('sigma', sigma);

    const res = await fetch(`${API_BASE}/filtering/compare`, {
        method: 'POST',
        body: form,
    });
    return res.json();
}

/**
 * Compute disparity map from stereo pair.
 * @param {File} leftFile - Left image.
 * @param {File} rightFile - Right image.
 * @param {string} method - 'bm' or 'sgbm'.
 * @param {object} params - Stereo parameters.
 * @returns {Promise<Blob>} Colorized disparity map blob.
 */
export async function computeDisparity(leftFile, rightFile, method, params = {}) {
    const form = new FormData();
    form.append('left_image', leftFile);
    form.append('right_image', rightFile);
    form.append('method', method);
    form.append('num_disparities', params.numDisparities || 64);
    form.append('block_size', params.blockSize || 5);

    const res = await fetch(`${API_BASE}/stereo/disparity`, {
        method: 'POST',
        body: form,
    });
    return res.blob();
}

/**
 * Compute epipolar lines visualization.
 * @param {File} leftFile - Left image.
 * @param {File} rightFile - Right image.
 * @returns {Promise<Blob>} Image with epipolar lines.
 */
export async function computeEpipolar(leftFile, rightFile) {
    const form = new FormData();
    form.append('left_image', leftFile);
    form.append('right_image', rightFile);

    const res = await fetch(`${API_BASE}/stereo/epipolar`, {
        method: 'POST',
        body: form,
    });
    return res.blob();
}

/**
 * Match features between two images.
 * @param {File} img1 - First image.
 * @param {File} img2 - Second image.
 * @param {string} detector - 'sift' or 'orb'.
 * @returns {Promise<object>} JSON with match count and visualization.
 */
export async function matchFeatures(img1, img2, detector = 'sift') {
    const form = new FormData();
    form.append('image1', img1);
    form.append('image2', img2);
    form.append('detector', detector);

    const res = await fetch(`${API_BASE}/stitching/match-features`, {
        method: 'POST',
        body: form,
    });
    return res.json();
}

/**
 * Stitch multiple images into panorama.
 * @param {File[]} imageFiles - Array of image files.
 * @param {string} detector - 'sift' or 'orb'.
 * @returns {Promise<Blob>} Panorama image blob.
 */
export async function stitchImages(imageFiles, detector = 'sift') {
    const form = new FormData();
    imageFiles.forEach((f) => form.append('images', f));
    form.append('detector', detector);

    const res = await fetch(`${API_BASE}/stitching/stitch`, {
        method: 'POST',
        body: form,
    });
    return res.blob();
}

/**
 * List available 3D models (PLY files).
 * @returns {Promise<Array>} Array of model metadata objects.
 */
export async function list3dModels() {
    const res = await fetch(`${API_BASE}/stereo/models`);
    const data = await res.json();
    return data.models || [];
}

/**
 * Reconstruct 3D model from stereo pair.
 * @param {File} leftFile - Left stereo image.
 * @param {File} rightFile - Right stereo image.
 * @param {string} method - 'bm' or 'sgbm'.
 * @param {object} params - Stereo parameters.
 * @returns {Promise<object>} JSON with model name and PLY URLs.
 */
export async function reconstruct3D(leftFile, rightFile, method = 'sgbm', params = {}) {
    const form = new FormData();
    form.append('left_image', leftFile);
    form.append('right_image', rightFile);
    form.append('method', method);
    form.append('num_disparities', params.numDisparities || 64);
    form.append('block_size', params.blockSize || 5);

    const res = await fetch(`${API_BASE}/stereo/reconstruct`, {
        method: 'POST',
        body: form,
    });
    return res.json();
}

/**
 * Reconstruct 3D model from a single image (monocular).
 * @param {File} imageFile - Single input image.
 * @returns {Promise<object>} JSON with output URLs.
 */
export async function monocularReconstruct3D(imageFile) {
    const form = new FormData();
    form.append('image', imageFile);

    const res = await fetch(`${API_BASE}/monocular/reconstruct`, {
        method: 'POST',
        body: form,
    });
    return res.json();
}
