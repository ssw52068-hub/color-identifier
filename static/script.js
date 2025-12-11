// ===== Color Identifier App - Frontend JavaScript =====
// Handles image upload, camera capture, API calls, and results display

console.log('🎨 Color Identifier App initialized');

// ===== DOM Elements =====
const uploadBox = document.getElementById('uploadBox');
const imageInput = document.getElementById('imageInput');
const cameraBtn = document.getElementById('cameraBtn');

const uploadSection = document.getElementById('uploadSection');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');

const originalImage = document.getElementById('originalImage');
const segmentedImage = document.getElementById('segmentedImage');
const colorsList = document.getElementById('colorsList');
const modelInfo = document.getElementById('modelInfo');

const retryBtn = document.getElementById('retryBtn');
const downloadBtn = document.getElementById('downloadBtn');

// ===== Global State =====
let currentImageData = null;
let analysisResults = null;

// ===== Event Listeners Setup =====

// Upload box click
uploadBox.addEventListener('click', () => {
    imageInput.click();
});

// File input change
imageInput.addEventListener('change', handleFileSelect);

// Drag and drop events
uploadBox.addEventListener('dragover', handleDragOver);
uploadBox.addEventListener('dragleave', handleDragLeave);
uploadBox.addEventListener('drop', handleDrop);

// Camera button
cameraBtn.addEventListener('click', handleCameraClick);

// Retry button
retryBtn.addEventListener('click', resetApp);

// Download button
downloadBtn.addEventListener('click', handleDownload);

// ===== File Upload Handlers =====

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    uploadBox.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadBox.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    uploadBox.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function processFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please upload an image file (JPG, PNG, etc.)');
        return;
    }

    // Check file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('Image too large! Please use an image smaller than 10MB.');
        return;
    }

    // Read file as data URL
    const reader = new FileReader();
    reader.onload = (e) => {
        currentImageData = e.target.result;
        analyzeImage(currentImageData);
    };
    reader.onerror = () => {
        showError('Error reading file. Please try again.');
    };
    reader.readAsDataURL(file);
}

// ===== Camera Capture Handler =====

function handleCameraClick() {
    // Check if getUserMedia is supported
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showError('Camera not supported in this browser. Please upload an image instead.');
        return;
    }

    // Request camera access
    navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment' } // Use back camera on mobile
    })
    .then(stream => {
        showCameraModal(stream);
    })
    .catch(error => {
        console.error('Camera error:', error);
        if (error.name === 'NotAllowedError') {
            showError('Camera access denied. Please allow camera access and try again.');
        } else {
            showError('Could not access camera. Please upload an image instead.');
        }
    });
}

function showCameraModal(stream) {
    // Create modal overlay
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.95);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 10000;
        padding: 1rem;
    `;

    // Create video element
    const video = document.createElement('video');
    video.srcObject = stream;
    video.autoplay = true;
    video.playsInline = true;
    video.style.cssText = `
        max-width: 90%;
        max-height: 60vh;
        border-radius: 1rem;
        margin-bottom: 1.5rem;
    `;

    // Create canvas (hidden, for capturing)
    const canvas = document.createElement('canvas');

    // Create buttons container
    const buttonsDiv = document.createElement('div');
    buttonsDiv.style.cssText = `
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        justify-content: center;
    `;

    // Capture button
    const captureBtn = document.createElement('button');
    captureBtn.innerHTML = '📸 Capture Photo';
    captureBtn.style.cssText = `
        padding: 1rem 2rem;
        background: #10b981;
        color: white;
        border: none;
        border-radius: 0.5rem;
        font-size: 1.125rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    `;
    captureBtn.onmouseover = () => captureBtn.style.background = '#059669';
    captureBtn.onmouseout = () => captureBtn.style.background = '#10b981';

    // Close button
    const closeBtn = document.createElement('button');
    closeBtn.innerHTML = '✕ Close';
    closeBtn.style.cssText = `
        padding: 1rem 2rem;
        background: #ef4444;
        color: white;
        border: none;
        border-radius: 0.5rem;
        font-size: 1.125rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    `;
    closeBtn.onmouseover = () => closeBtn.style.background = '#dc2626';
    closeBtn.onmouseout = () => closeBtn.style.background = '#ef4444';

    // Capture button click
    captureBtn.addEventListener('click', () => {
        // Set canvas size to video size
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Draw video frame to canvas
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);

        // Get image as data URL
        const imageData = canvas.toDataURL('image/png');
        currentImageData = imageData;

        // Stop camera
        stream.getTracks().forEach(track => track.stop());

        // Remove modal
        document.body.removeChild(modal);

        // Analyze image
        analyzeImage(imageData);
    });

    // Close button click
    closeBtn.addEventListener('click', () => {
        stream.getTracks().forEach(track => track.stop());
        document.body.removeChild(modal);
    });

    // Assemble modal
    buttonsDiv.appendChild(captureBtn);
    buttonsDiv.appendChild(closeBtn);
    modal.appendChild(video);
    modal.appendChild(buttonsDiv);
    document.body.appendChild(modal);
}

// ===== API Communication =====

async function analyzeImage(imageData) {
    console.log('Starting image analysis...');

    // Show loading screen
    uploadSection.style.display = 'none';
    loadingSection.style.display = 'block';
    resultsSection.style.display = 'none';

    try {
        // Call Flask API
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageData
            })
        });

        // Check response status
        if (!response.ok) {
            throw new Error(`API error: ${response.status} ${response.statusText}`);
        }

        // Parse JSON response
        const data = await response.json();

        // Check if analysis was successful
        if (!data.success) {
            throw new Error(data.error || 'Analysis failed');
        }

        console.log('Analysis successful!', data);

        // Store results
        analysisResults = data;

        // Display results
        displayResults(data, imageData);

    } catch (error) {
        console.error('Analysis error:', error);
        
        // Hide loading, show upload section
        loadingSection.style.display = 'none';
        uploadSection.style.display = 'block';
        
        // Show error message
        showError(`Analysis failed: ${error.message}`);
    }
}

// ===== Results Display =====

function displayResults(data, originalImageData) {
    console.log('Displaying results...');

    // Hide loading, show results
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'block';

    // Display images
    originalImage.src = originalImageData;
    segmentedImage.src = data.segmented_image;

    // Display detected colors
    displayColors(data.results);

    // Display model info
    displayModelInfo(data.model_info);

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displayColors(colors) {
    colorsList.innerHTML = '';

    colors.forEach((color, index) => {
        const colorItem = document.createElement('div');
        colorItem.className = 'color-item';
        colorItem.style.animationDelay = `${index * 0.1}s`;

        const rgb = color.rgb;
        const rgbString = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
        const colorName = color.color_name.replace(/_/g, ' ');

        colorItem.innerHTML = `
            <div class="color-swatch" style="background-color: ${rgbString}"></div>
            <div class="color-details">
                <div class="color-name">${colorName}</div>
                <div class="color-stats">
                    <div class="stat">
                        <strong>Coverage:</strong> ${color.coverage.toFixed(1)}%
                    </div>
                    <div class="stat">
                        <strong>Confidence:</strong> ${color.confidence.toFixed(1)}%
                    </div>
                    <div class="stat">
                        <strong>RGB:</strong> (${rgb[0]}, ${rgb[1]}, ${rgb[2]})
                    </div>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${color.confidence}%"></div>
                </div>
            </div>
        `;

        colorsList.appendChild(colorItem);
    });

    console.log(`Displayed ${colors.length} colors`);
}

function displayModelInfo(info) {
    modelInfo.innerHTML = `
        <div class="info-item">
            <div class="info-label">Algorithm</div>
            <div class="info-value">${info.algorithm}</div>
        </div>
        <div class="info-item">
            <div class="info-label">Training Samples</div>
            <div class="info-value">${info.training_samples.toLocaleString()}</div>
        </div>
        <div class="info-item">
            <div class="info-label">Model Accuracy</div>
            <div class="info-value">${info.accuracy.toFixed(1)}%</div>
        </div>
        <div class="info-item">
            <div class="info-label">Colors Detected</div>
            <div class="info-value">${info.colors_detected}</div>
        </div>
    `;
}

// ===== Utility Functions =====

function resetApp() {
    console.log('Resetting app...');

    // Clear state
    currentImageData = null;
    analysisResults = null;
    imageInput.value = '';

    // Show upload section
    uploadSection.style.display = 'block';
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'none';

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function handleDownload() {
    if (!analysisResults) {
        showError('No results to download');
        return;
    }

    // Create text report
    let report = '🎨 Color Analysis Report\n';
    report += '=' .repeat(50) + '\n\n';
    report += `Algorithm: ${analysisResults.model_info.algorithm}\n`;
    report += `Accuracy: ${analysisResults.model_info.accuracy.toFixed(1)}%\n\n`;
    report += 'Detected Colors:\n';
    report += '-'.repeat(50) + '\n';

    analysisResults.results.forEach((color, index) => {
        const name = color.color_name.replace(/_/g, ' ');
        report += `${index + 1}. ${name}\n`;
        report += `   RGB: (${color.rgb.join(', ')})\n`;
        report += `   Coverage: ${color.coverage.toFixed(1)}%\n`;
        report += `   Confidence: ${color.confidence.toFixed(1)}%\n\n`;
    });

    report += '=' .repeat(50) + '\n';
    report += 'Generated by Color Identifier App\n';
    report += 'Team: ACDT 31조\n';

    // Create blob and download
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'color-analysis-report.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    console.log('Report downloaded');
}

function showError(message) {
    alert('❌ ' + message);
}

// ===== API Health Check =====

async function checkAPIHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        console.log('✅ API Health Check:', data);
        return true;
    } catch (error) {
        console.error('❌ API Health Check Failed:', error);
        showError('Backend API is not responding. Please make sure Flask server is running.');
        return false;
    }
}

// ===== Initialize App =====

// Check API on page load
window.addEventListener('load', () => {
    console.log('Page loaded, checking API...');
    checkAPIHealth();
});

console.log('✅ JavaScript initialized successfully');
