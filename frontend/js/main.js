const API_URL = "http://localhost:5000";

document.addEventListener("DOMContentLoaded", () => {
    const uploadArea = document.getElementById("uploadArea");
    const imageInput = document.getElementById("imageInput");
    const canvasContainer = document.getElementById("canvasContainer");
    const sourceImage = document.getElementById("sourceImage");
    const detectionCanvas = document.getElementById("detectionCanvas");
    const detailsPanel = document.getElementById("detailsPanel");
    const objectsList = document.getElementById("objectsList");
    const descriptorView = document.getElementById("descriptorView");

    // Event Listeners for Upload
    uploadArea.addEventListener("click", () => imageInput.click());

    uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = "#6366f1";
    });

    uploadArea.addEventListener("dragleave", () => {
        uploadArea.style.borderColor = "#334155";
    });

    uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = "#334155";
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith("image/")) {
            handleImageUpload(file);
        }
    });

    imageInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) handleImageUpload(file);
    });

    // Handle Image Upload & Detection
    async function handleImageUpload(file) {
        // 1. Show Image Locally
        const reader = new FileReader();
        reader.onload = (e) => {
            sourceImage.src = e.target.result;
            sourceImage.onload = () => {
                canvasContainer.style.display = "block";
                uploadArea.style.display = "none";
                resizeCanvas();
            };
        };
        reader.readAsDataURL(file);

        // 2. Send to Backend for Detection
        const formData = new FormData();
        formData.append("image", file);

        try {
            // We use /descriptors endpoint to get everything: boxes + descriptors
            const response = await fetch(`${API_URL}/descriptors`, {
                method: "POST",
                body: formData
            });

            if (!response.ok) throw new Error("API Error");

            const data = await response.json();
            displayObjects(data);
            drawBoundingBoxes(data);

            // Store current file globally for search if needed
            window.currentImageFile = file;
            window.currentObjects = data;

        } catch (error) {
            console.error(Error, error);
            alert("Error processing image. Is the backend running?");
        }
    }

    function resizeCanvas() {
        detectionCanvas.width = sourceImage.width;
        detectionCanvas.height = sourceImage.height;
    }

    function drawBoundingBoxes(objects) {
        const ctx = detectionCanvas.getContext("2d");
        ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);

        // Scale factors if image displayed size differs from natural size
        // For simplicity, we assume the image is rendered at natural size or handled via CSS max-width
        // But for canvas overlay, we need to match the *displayed* size if it shrinks.
        // HOWEVER, the logic below assumes sourceImage.width IS the rendered width.
        // A robust way uses naturalWidth vs clientWidth.

        const scaleX = sourceImage.width / sourceImage.naturalWidth;
        const scaleY = sourceImage.height / sourceImage.naturalHeight;

        // Note: The backend boxes might be normalized or absolute. 
        // Based on yolo_detector.py, they are [x1, y1, x2, y2] absolute coordinates.
        // We need to verify if backend returns coordinates based on the ORIGINAL image size.
        // YES, YOLO returns original image coordinates.

        // Wait, sourceImage.width in JS gives the *rendered* width if styled with CSS?
        // Actually, if we just set src, .width is the intrinsic width unless limited by CSS.
        // But our CSS says max-width: 100%.
        // So we strictly need to use clientWidth/Height for canvas size.

        detectionCanvas.width = sourceImage.clientWidth;
        detectionCanvas.height = sourceImage.clientHeight;

        const realScaleX = sourceImage.clientWidth / sourceImage.naturalWidth;
        const realScaleY = sourceImage.clientHeight / sourceImage.naturalHeight;

        objects.forEach((obj, index) => {
            // Backend sends descriptors object which has 'label' (int).
            // But we actually need the BBOX.
            // Wait, descriptors.py/pipeline.py does NOT include bbox in the /descriptors response?
            // Let's check pipeline.py.
            // pipeline.py: return results (list of dicts). 
            // dict keys: "label", "hu", "orientation_hist", etc.
            // IT DOES NOT RETURN BBOX! 
            // FIX REQUIRED IN BACKEND.

            // Assuming we will fix backend to return bbox.
            if (!obj.bbox) return;

            const [x1, y1, x2, y2] = obj.bbox;

            ctx.strokeStyle = "#22c55e";
            ctx.lineWidth = 3;
            ctx.strokeRect(x1 * realScaleX, y1 * realScaleY, (x2 - x1) * realScaleX, (y2 - y1) * realScaleY);

            ctx.fillStyle = "#22c55e";
            ctx.font = "14px Inter";
            ctx.fillText(`#${index} ${obj.category}`, x1 * realScaleX, (y1 * realScaleY) - 5);
        });
    }

    function displayObjects(objects) {
        const emptyState = detailsPanel.querySelector(".empty-state");
        const resultsContent = detailsPanel.querySelector(".results-content");
        const countSpan = document.getElementById("objectCount");

        emptyState.style.display = "none";
        resultsContent.style.display = "block";
        countSpan.textContent = `${objects.length} Found`;

        objectsList.innerHTML = "";

        objects.forEach((obj, index) => {
            const div = document.createElement("div");
            div.className = "object-item";
            div.innerHTML = `
                <i class="fa-solid fa-cube"></i><br>
                <strong>${obj.category}</strong><br>
                <small>#${index}</small>
            `;
            div.onclick = () => showDescriptors(obj, index);
            objectsList.appendChild(div);
        });
    }

    let colorChart = null;

    function showDescriptors(obj, index) {
        descriptorView.style.display = "block";

        // Scroll to view
        descriptorView.scrollIntoView({ behavior: 'smooth' });

        // populate metrics
        document.getElementById("huMomentsData").textContent = JSON.stringify(obj.descriptors.hu_moments.values.map(n => n.toFixed(4)), null, 2);

        // Texture data summary
        document.getElementById("textureData").textContent =
            `LBP (first 10): ${obj.descriptors.lbp.values.slice(0, 10).map(n => n.toFixed(3))}...\nHOG Length: ${obj.descriptors.hog.length}`;

        // Render Color Chart
        renderColorChart(obj.descriptors.color_histogram.values || []); // Backend needs to ensure this structure or I fix it

        // Trigger Search
        findSimilarImages(index);
    }

    function renderColorChart(data) {
        const ctx = document.getElementById("colorHistChart").getContext("2d");

        // Mock data if empty for visualization (or formatting backend data)
        // Backend 'color_hist' is flat 256*3 array? Or just 3 arrays?
        // pipeline.py says: np.concatenate(hist) -> 768 length array (B, G, R sequential).

        // We will visualize 3 series.
        const binCount = 256;
        const b = data.slice(0, 256);
        const g = data.slice(256, 512);
        const r = data.slice(512, 768);

        const labels = Array.from({ length: 256 }, (_, i) => i);

        if (colorChart) colorChart.destroy();

        colorChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    { label: 'Red', data: r, borderColor: 'red', borderWidth: 1, pointRadius: 0 },
                    { label: 'Green', data: g, borderColor: 'green', borderWidth: 1, pointRadius: 0 },
                    { label: 'Blue', data: b, borderColor: 'blue', borderWidth: 1, pointRadius: 0 }
                ]
            },
            options: {
                responsive: true,
                scales: { x: { display: false }, y: { display: false } },
                plugins: { legend: { display: true } }
            }
        });
    }

    async function findSimilarImages(objectIndex) {
        const grid = document.getElementById("similarImagesGrid");
        grid.innerHTML = '<p>Searching...</p>';

        // Since /search expects a new file upload which re-runs detection...
        // We should arguably optimize this API designs. 
        // But per `app.py`, /search takes `file` and finds matches for discovered objects.
        // It returns results for ALL objects. We just filter for the one selected.

        const formData = new FormData();
        formData.append("image", window.currentImageFile);

        try {
            const res = await fetch(`${API_URL}/search`, { method: "POST", body: formData });
            const allResults = await res.json();

            // allResults is a list of objs, we match by index (assuming deterministic order which YOLO usually is)
            const myResult = allResults[objectIndex];

            grid.innerHTML = "";

            if (myResult && myResult.best_match) {
                // Currently only shows 1 best match per logic.
                // We show it as an image. 
                // Note: The path is local server path "backend/data/uploads/...". 
                // Browser cannot access this directly due to security.
                // We need an endpoint to serve images OR assuming we are testing locally and opened file via local server?
                // NO, we are on a web page. We cannot file:// link to local hard drive.
                // WE NEED A SERVE ENDPOINT.

                // For this demo, let's just create a dummy div with the path text
                // OR we add a quick static route in backend.

                const div = document.createElement("div");
                div.innerHTML = `
                    <div style="background: #333; color: #fff; padding: 10px; border-radius: 4px;">
                        <i class="fa-solid fa-image"></i>
                        <p>${myResult.best_match.split('/').pop()}</p>
                        <p>Score: ${(myResult.score * 100).toFixed(1)}%</p>
                    </div>
                `;
                grid.appendChild(div);
            } else {
                grid.innerHTML = "<p>No matches found in DB.</p>";
            }

        } catch (e) {
            console.error(e);
            grid.innerHTML = "<p>Error searching.</p>";
        }
    }

    window.closeDescriptors = () => {
        descriptorView.style.display = "none";
    };
});
