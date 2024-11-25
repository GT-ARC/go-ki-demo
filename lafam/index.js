window.addEventListener("error", (event) => {
    addLogMsg(event.message);
});
window.addEventListener("unhandledrejection", (event) => {
    addLogMsg(event.reason);
});

function addLogMsg(msg) {
    debug_log.textContent += msg + "\n";
    debug_log.scrollTop = debug_log.scrollHeight;
}

const debug_devices = document.getElementById("devices");
const debug_log = document.getElementById("log");

const INPUT_WIDTH = 224;
const INPUT_HEIGHT = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const TOP_N = 14;

let include_groups;
let exclude_groups;
let grouper = null;

let selectionEnabled = true;

class ModelWorker {
    constructor(url) {
        let $this = this;
        this.initCamera();

        this.worker = new window.Worker(url);
        this.worker.onmessage = (e) => {
            $this._onmessage(e);
        };

        this.palettes;
        this.currentPalette;

        this.initElements();
        this.initEvents();

        fetch("./imagenet_class_index.json")
            .then((response) => response.json())
            .then((data) => (this.imagenet_classes = data));

        fetch("palettes.json")
            .then((response) => response.json())
            .then((data) => {
                $this.palettes = data;
                $this.currentPalette = Object.keys(data)[0];
                for (const palette in data) {
                    const option = document.createElement("option");
                    option.value = palette;
                    option.textContent = palette;
                    $this.paletteSelect.appendChild(option);
                }
            });
    }

    initElements() {
        this.paletteSelect = document.getElementById("palette-select");
        this.showLogits = document.getElementById("btn-show-logits");
        this.mainSection = document.getElementById("main-section");
        this.video = document.createElement("video");
        this.predictionList = document.getElementById("prediction-list");
        this.hidden_canvas = document.createElement("canvas");
        this.ctx_hidden = this.hidden_canvas.getContext("2d");
        this.img_canvas = document.getElementById("img-canvas");
        this.ctx_img = this.img_canvas.getContext("2d");
        this.heatmap_canvas = document.getElementById("heatmap-canvas");
        this.ctx_heatmap = this.heatmap_canvas.getContext("2d");
        this.startButton = document.getElementById("start-button");
        this.switchCameraButton = document.getElementById("switch-camera-button");
        this.heatmapOpacity = document.getElementById("heatmap-opacity");
        this.predefinedFiles = document.getElementById("predefined-files");
        this.uploadButton = document.getElementById("upload-button");
        this.uploadInput = document.getElementById("upload-input");
        this.heatmapGrid = document.getElementById("grid");
        this.clearSelection = document.getElementById("clear-selection");
    }

    initEvents() {
        const $this = this;

        this.predefinedFiles.addEventListener("change", () => {
            $this._clearSelections();
            $this._clearSquareResults();
            $this.load_selected_image();
        });

        this.uploadButton.addEventListener("click", () => {
            this.uploadInput.click();
        });

        this.uploadInput.addEventListener("change", (e) => {
            const file = e.target.files[0];
            console.log('upload file', file)
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    this.setSize(img.width, img.height);
                    this._postMessage("predict", this.getImage(img));
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        });

        this.predictionList.addEventListener("click", (event) => {
            if (!selectionEnabled) return;
            if (!this.video.paused) return;
            const div = event.target.closest(".prediction");
            if (!div) return; // Clicked outside of a prediction div

            let selectedIdxs = [];
            if (!div.classList.contains("selected")) {
                selectedIdxs.push(parseInt(div.getAttribute("data-idx")));
                div.classList.add("selected");
            } else {
                div.classList.remove("selected");
            }

            for (const el of document.querySelectorAll(".prediction.selected")) {
                if (el !== div) {
                    selectedIdxs.push(parseInt(el.getAttribute("data-idx")));
                }
            }

            if (selectedIdxs.length > 0) {
                this.clearSelection.disabled = false;
                this.worker.postMessage({
                    status: "heatmap_by_class",
                    classIdxs: selectedIdxs,
                });
            } else {
                this.clearSelection.disabled = true;
                this.updateResults(
                    this.results.heatmap,
                    this.results.predictions,
                    this.results.logits
                );
            }
        });

        this.clearSelection.addEventListener("click", () => {
            this._clearSelections();
            this.updateResults(
                this.results.heatmap,
                this.results.predictions,
                this.results.logits
            );
        });

        this.heatmapOpacity.oninput = () => {
            $this.heatmap_canvas.style.opacity = $this.heatmapOpacity.value;
        };

        this.paletteSelect.onchange = function () {
            if (!selectionEnabled) return;
            $this.currentPalette = this.value;
            $this.updateHeatmap($this.currentHeatmap);
        };

        this.startButton.addEventListener("click", (e) => {
            if ($this.video.paused) {
                this._clearSelections();
                $this.video.play();
                $this.mainSection.classList.remove("paused");
            } else {
                $this.video.pause();
                $this.mainSection.classList.add("paused");
            }
        });

        this.showLogits.onclick = (e) => {
            this._clearSelections();
            this._clearSquareResults();

            // todo: disable selections

            this.load_selected_image("predict_per_square");
        }

        // root event listener for cells (divs)
        this.heatmapGrid.addEventListener("click", (e) => {
            const div = e.target.closest("div");
            if (!div) return;

            this._clearPredictionSelections();
            let classIdxs = [];
            if (!div.classList.contains("selected")) {
                classIdxs.push(parseInt(div.getAttribute("data-idx")));
                div.classList.add("selected");
            } else {
                div.classList.remove("selected");
            }

            for (const el of document.querySelectorAll("#grid div.selected")) {
                if (el !== div) {
                    classIdxs.push(parseInt(el.getAttribute("data-idx")));
                }
            }

            if (classIdxs.length > 0) {
                this.clearSelection.disabled = false;
                $this.worker.postMessage({
                    status: "class_by_heatmap",
                    classIdxs: classIdxs,
                });
            } else {
                this.clearSelection.disabled = true;
                this.updateResults(
                    this.results.heatmap,
                    this.results.predictions,
                    this.results.logits
                );
            }
        });

        selectionEnabled = false;
    }

    setSize(width, height) {
        this.width = width;
        this.height = height;
        this.min_side = Math.min(width, height);

        this.hidden_canvas.width = width;
        this.hidden_canvas.height = height;

        this.img_canvas.width = this.min_side;
        this.img_canvas.height = this.min_side;

        this.heatmap_canvas.width = this.min_side;
        this.heatmap_canvas.height = this.min_side;
    }

    getImage(img) {
        this.ctx_hidden.drawImage(img, 0, 0);
        return this.ctx_hidden.getImageData(0, 0, this.width, this.height);
    }

    load_selected_image(post_status = "predict") {
        const file = document.getElementById("predefined-files").value;
        if (file) {
            fetch(file)
                .then((response) => response.blob())
                .then((blob) => {
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        const img = new Image();
                        img.onload = () => {
                            this.width = img.width;
                            this.height = img.height;
                            this.min_side = Math.min(img.width, img.height);

                            this.hidden_canvas.width = this.width;
                            this.hidden_canvas.height = this.height;

                            this.img_canvas.width = this.min_side;
                            this.img_canvas.height = this.min_side;

                            this.heatmap_canvas.width = this.min_side;
                            this.heatmap_canvas.height = this.min_side;

                            this.ctx_hidden.drawImage(img, 0, 0);
                            let imgData = this.ctx_hidden.getImageData(
                                0,
                                0,
                                img.width,
                                img.height
                            );
                            this._postMessage(post_status, imgData);
                        };
                        img.src = e.target.result;
                    };
                    reader.readAsDataURL(blob);
                });
        }
    }

    async initCamera() {
        let $this = this;

        const state = await navigator.permissions.query({ name: "camera" });
        if (state.state === "denied")
            return;

        let cameras = await navigator.mediaDevices.enumerateDevices();
        debug_devices.textContent = JSON.stringify(cameras, null, 2);
        cameras = cameras.filter((device) => device.kind === "videoinput");

        if (cameras.length === 0) {
            // startButton.textContent = "No camera";
            return;
        }

        this.startButton.style.display = "block";

        this.currentCameraId = cameras[cameras.length - 1].deviceId;
        this.localMediaStream = await navigator.mediaDevices.getUserMedia({
            video: {
                deviceId: $this.currentCameraId,
                height: { ideal: 1024 },
            },
        });

        this.video.srcObject = this.localMediaStream;

        if (cameras.length > 1) {
            // $this.mainSection.classList.add("multiple-cameras");
            document.body.classList.add("multiple-cameras");
            this.switchCameraButton.onclick = async () => {
                $this.video.pause();
                $this.video.removeEventListener("play", $this._onPlay);
                $this.localMediaStream.getTracks().forEach((track) => track.stop());

                $this.currentCameraId = cameras.find(
                    (camera) => camera.deviceId !== $this.currentCameraId
                ).deviceId;

                $this.localMediaStream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        deviceId: $this.currentCameraId,
                        height: { ideal: 1024 },
                    },
                });

                $this.video.srcObject = $this.localMediaStream;
                $this.video.addEventListener(
                    "play",
                    () => {
                        $this._onPlay();
                    },
                    0
                );
                $this.video.play();
                $this.mainSection.classList.remove("paused");
            };
        }

        this.video.addEventListener(
            "play",
            () => {
                this._onPlay();
            },
            0
        );
    }

    _onPlay() {
        const settings = this.localMediaStream.getVideoTracks()[0].getSettings();
        this.width = settings.width;
        this.height = settings.height;
        this.min_side = Math.min(this.width, this.height);

        this.hidden_canvas.width = this.width;
        this.hidden_canvas.height = this.height;

        this.img_canvas.width = this.min_side;
        this.img_canvas.height = this.min_side;

        this.heatmap_canvas.width = this.min_side;
        this.heatmap_canvas.height = this.min_side;

        this._postMessage("predict", this.getImage(this.video));
    }

    _onmessage(e) {
        const { data } = e;
        if (data.status === "ready") {
            this.startButton.disabled = false;
            this.switchCameraButton.disabled = false;
            document.getElementById("loading-indicator").style.display = "none";
            this.mainSection.classList.add("ready");
        }
        if (data.status === "results") {
            this.results = data;
            this.updateResults(data.heatmap, data.predictions, data.logits);
        }
        if (data.status === "weighted_heatmap") {
            this.updateHeatmap(data.heatmap);
        }
        if (data.status === "class_by_heatmap") {
            const top_n_idx = argmax_top_n(data.logits, TOP_N, 1.7);
            this._updatePredictionList(top_n_idx, data.predictions, data.logits);
        }

        if (data.status === "square_results") {
            this.updateSquareResults(data.data);
        }
    }

    _clearPredictionSelections() {
        for (const el of document.querySelectorAll(".prediction.selected")) {
            el.classList.remove("selected");
        }
    }

    _clearHeatmapSelections() {
        // remove all selected classes
        for (const el of document.querySelectorAll("#grid div.selected")) {
            el.classList.remove("selected");
        }
    }

    _clearSelections() {
        this._clearPredictionSelections();
        this._clearHeatmapSelections();
        this.clearSelection.disabled = true;
    }

    _postMessage(status, imgData) {
        const croppedFrame = ImageProcessor.fromImageData(imgData).squareCrop();

        this.ctx_img.putImageData(
            new ImageData(
                ImageProcessor.toImageData(croppedFrame),
                this.min_side,
                this.min_side
            ),
            0,
            0
        );

        const transformed_img = croppedFrame
            .resize(INPUT_WIDTH, INPUT_HEIGHT, "bilinear")
            .normalize(MEAN, STD);

        this.worker.postMessage({
            status: status,
            tensor: ImageProcessor.toTensor(transformed_img),
        });
    }

    updateResults(heatmap, predictions, logits) {
        this.updateHeatmap(heatmap);

        let top_n_idx = argmax_top_n(logits, TOP_N, 1.7);

        this._updatePredictionList(top_n_idx, predictions, logits);

        if (!this.video.paused) {
            this._postMessage("predict", this.getImage(this.video));
        }
    }

    updateSquareResults(data) {
        if(!grouper) {
            grouper = new ClassGrouper();
        }

        const squareData = data.map(square => {
            const { logits, predictions } = square;
            const top_index = argmax_top_n(logits, 1, 0)[0];

            return {
                classId: top_index,
                groupId: grouper.classToGroup(top_index),
                logit: logits[top_index],
                probability: predictions[top_index],
            }
        });

        // show logits+classId in square
        squareData.forEach((square, i) => {
            const div = document.querySelector(`div[data-idx='${i}']`);
            if (div !== null) {
                const logit = square.logit.toFixed(2);
                const cls = square.classId;
                const group = square.groupId;
                div.innerHTML = `${logit}<br/>${cls}, ${group}`;
                div.classList.add("class-display");
            } else {
                addLogMsg('Error: could not find div with data-idx=' + i);
            }
        });

        // calc and show top classes for the image
        const topClasses = Object.values(squareData
            .reduce((acc, obj) => {
                if (!acc[obj.classId]) {
                    acc[obj.classId] = JSON.parse(JSON.stringify(obj));
                    acc[obj.classId].count = 1;
                } else {
                    acc[obj.classId].count++;
                }
                return acc;
            }, {}))
            .sort((a, b) => b.count - a.count);

        // todo: show top classes on the right with color palette values
        // todo: proper palette

        const heatmap = this.makeClassHeatmap(squareData);
        this.updateHeatmap(heatmap, "rainbow");

        const groupGrid2d = to2DArray(squareData.map(square => square.groupId), 7, 7);
        const areas = findAreas(groupGrid2d, [-1]).filter(area => area.length > 1);
        const boundingBoxes = findBoundingBoxes(areas);
        drawBoundingBoxes(boundingBoxes, 'heatmap-canvas');
    }

    makeClassHeatmap(squareData) {
        const classIds = squareData.map(square => square.classId);
        const groupIds = classIds.map(classId => grouper.classToGroup(classId));
        const min = Math.min(...groupIds);
        const max = Math.max(...groupIds);
        return groupIds.map(groupId => (groupId - min) / (max - min + 1e-6));
    }

    _clearSquareResults() {
        for (let div of document.querySelectorAll("#grid div")) {
            div.innerHTML = "";
            div.classList.remove("class-display");
        }
        selectionEnabled = true;
    }

    updateHeatmap(data, palette = null) {
        this.currentHeatmap = data;

        palette = palette === null ? this.currentPalette : palette;
        let heatmap = mapToPalette(data, this.palettes[palette]);
        heatmap = new ImageProcessor(heatmap, 7, 7).resize(
            this.min_side,
            this.min_side,
            "nearest"
        );

        this.ctx_heatmap.putImageData(
            new ImageData(
                ImageProcessor.toImageData(heatmap),
                this.min_side,
                this.min_side
            ),
            0,
            0
        );
    }

    _updatePredictionList(indices, predictions, logits) {
        // create list with progress bars
        this.predictionList.innerHTML = "";
        const fragment = document.createDocumentFragment();
        for (let i = 0; i < indices.length; i++) {
            const idx = indices[i];
            let div = document.createElement("div");
            div.classList.add("prediction");
            div.setAttribute("data-idx", idx);
            // div.setAttribute("data-logits", logits[idx]);

            let label = document.createElement("label");
            const cls = this.imagenet_classes[idx];
            const prob = predictions[idx];
            const l = logits[idx].toFixed(2);
            label.innerHTML = `<b>${cls}</b> (Logits: ${l}; Softmax: ${Math.round(
                prob * 100
            )}%)`;
            div.appendChild(label);

            let progress = document.createElement("progress");
            progress.value = prob;
            progress.max = 1;
            div.appendChild(progress);

            fragment.appendChild(div);
        }

        this.predictionList.appendChild(fragment);
    }
}

// document.getElementById("grid").addEventListener("mousemove", (e) => {
//   for (const date of document.querySelectorAll("#grid div")) {
//     const rect = date.getBoundingClientRect(),
//       x = e.clientX - rect.left,
//       y = e.clientY - rect.top;

//     date.style.setProperty("--mouse-x", `${x}px`);
//     date.style.setProperty("--mouse-y", `${y}px`);
//   }
// });

async function init() {
    if ("serviceWorker" in navigator) {
        try {
            await navigator.serviceWorker.register("service-worker.js");
            console.log("Service Worker Registered");
        } catch (error) {
            console.log("Service Worker Registration Failed");
        }
    }

    include_groups = await fetch("include_groups.json").then(res => res.json());
    exclude_groups = await fetch("exclude_groups.json").then(res => res.json());

    new ModelWorker("worker.js");
}

function mapToPalette(x, palette) {
    let heatmap = [
        new Float32Array(x.length),
        new Float32Array(x.length),
        new Float32Array(x.length),
    ];
    for (let i = 0; i < x.length; i++) {
        const color = palette[Math.round(x[i] * (palette.length - 1))];
        heatmap[0][i] = color[0];
        heatmap[1][i] = color[1];
        heatmap[2][i] = color[2];
    }
    return heatmap;
}

class ImageProcessor {
    constructor(channels, width, height) {
        this.channels = channels;
        this.width = width;
        this.height = height;
    }

    static fromImageData(imageData) {
        let img = new ImageProcessor();
        img.width = imageData.width;
        img.height = imageData.height;
        img.channels = ImageProcessor.toNdarray(imageData, 3);

        return img;
    }

    static toTensor(img) {
        let tensor = new Float32Array(img.channels.length * img.width * img.height);
        for (let i = 0; i < img.channels.length; i++) {
            tensor.set(img.channels[i], i * img.width * img.height);
        }
        return tensor;
    }

    squareCrop() {
        const size = Math.min(this.width, this.height);
        const startX = Math.round((this.width - size) / 2);
        const startY = Math.round((this.height - size) / 2);
        const endY = startY + size;

        let channels = [];
        for (let i = 0; i < this.channels.length; i++) {
            channels.push(new Float32Array(size * size));
        }

        let croppedIndex = 0;

        for (let y = startY; y < endY; y++) {
            const rowStart = y * this.width + startX;
            const rowEnd = rowStart + size;

            for (let channel = 0; channel < this.channels.length; channel++) {
                channels[channel].set(
                    this.channels[channel].slice(rowStart, rowEnd),
                    croppedIndex
                );
            }

            croppedIndex += size;
        }

        return new ImageProcessor(channels, size, size);
    }

    resize(newWidth, newHeight, interpolation = "bilinear") {
        const resized = [];
        for (let i = 0; i < this.channels.length; i++) {
            resized.push(new Float32Array(newWidth * newHeight));
        }

        const xRatio = this.width / newWidth;
        const yRatio = this.height / newHeight;

        for (let y = 0; y < newHeight; y++) {
            for (let x = 0; x < newWidth; x++) {
                if (interpolation === "nearest") {
                    this._nearestNeighbor(x, y, xRatio, yRatio, resized, newWidth);
                } else if (interpolation === "bilinear") {
                    this._bilinearInterpolation(x, y, xRatio, yRatio, resized, newWidth);
                } else {
                    throw new Error(
                        `interpolation method ${interpolation} is not supported.`
                    );
                }
            }
        }

        return new ImageProcessor(resized, newWidth, newHeight);
    }

    normalize(mean, std) {
        // Validate mean and std have 3 elements each
        if (mean.length !== 3 || std.length !== 3) {
            throw new Error("Mean and standard deviation must each have 3 elements.");
        }

        const length = this.width * this.height;
        const n_channels = this.channels.length;
        let channels = [
            new Float32Array(length),
            new Float32Array(length),
            new Float32Array(length),
        ];

        for (let i = 0; i < length; i++) {
            // Normalize each channel
            for (let channel = 0; channel < n_channels; channel++) {
                channels[channel][i] =
                    (this.channels[channel][i] / 255 - mean[channel]) / std[channel];
            }
        }

        return new ImageProcessor(channels, this.width, this.height);
    }

    demoralize(mean, std) {
        // Validate mean and std have 3 elements each
        if (mean.length !== 3 || std.length !== 3) {
            throw new Error("Mean and standard deviation must each have 3 elements.");
        }

        const length = this.width * this.height;
        let channels = [
            new Float32Array(length),
            new Float32Array(length),
            new Float32Array(length),
        ];

        for (let i = 0; i < length; i++) {
            // Denormalize each channel
            for (let channel = 0; channel < 3; channel++) {
                channels[channel][i] =
                    (this.channels[channel][i] * std[channel] + mean[channel]) * 255;
            }
        }

        return new ImageProcessor(channels, this.width, this.height);
    }

    static toImageData(img) {
        const length = img.width * img.height;
        let rgba = new Uint8ClampedArray(length * 4);
        for (let i = 0; i < length * 4; i += 4) {
            for (let channel = 0; channel < img.channels.length; channel++) {
                rgba[i + channel] = img.channels[channel][i / 4];
            }
            rgba[i + 3] = 255;
        }

        return rgba;
    }

    static toNdarray(imageData, n_channels) {
        const length = imageData.width * imageData.height;
        let channels = [];
        for (let i = 0; i < n_channels; i++) {
            channels.push(new Float32Array(length));
        }

        for (let i = 0; i < imageData.data.length; i += 4) {
            for (let channel = 0; channel < n_channels; channel++) {
                channels[channel][i / 4] = imageData.data[i + channel];
            }
            // ignore [i + 4] as it is alpha channel
        }
        // 3. Concatenate RGB to transpose [224, 224, 3] -> [3, 224, 224] to a number array
        // const transposedData = redArray.concat(greenArray).concat(blueArray);

        return channels;
    }

    _nearestNeighbor(x, y, xRatio, yRatio, output, newWidth) {
        const nearestX = Math.floor((x + 0.5) * xRatio);
        const nearestY = Math.floor((y + 0.5) * yRatio);
        const idxSrc = nearestY * this.width + nearestX;
        const idxDest = y * newWidth + x;

        for (let channel = 0; channel < this.channels.length; channel++) {
            output[channel][idxDest] = this.channels[channel][idxSrc];
        }
    }

    _bilinearInterpolation(x, y, xRatio, yRatio, output, newWidth) {
        const xL = Math.floor(x * xRatio);
        const yL = Math.floor(y * yRatio);
        const xH = Math.ceil(x * xRatio);
        const yH = Math.ceil(y * yRatio);
        const xWeight = x * xRatio - xL;
        const yWeight = y * yRatio - yL;
        const idxDest = y * newWidth + x;

        for (let channel = 0; channel < 3; channel++) {
            const valTL = this.channels[channel][yL * this.width + xL];
            const valTR = this.channels[channel][yL * this.width + xH];
            const valBL = this.channels[channel][yH * this.width + xL];
            const valBR = this.channels[channel][yH * this.width + xH];

            const top = valTL + (valTR - valTL) * xWeight;
            const bottom = valBL + (valBR - valBL) * xWeight;
            output[channel][idxDest] = top + (bottom - top) * yWeight;
        }
    }
}

class ClassGrouper {
    constructor() {

        // todo: needed here?
        this.includeGroups = {"dog (canid)": [151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275], "fox": [277, 278, 279, 280], "cat": [281, 282, 283, 284, 285], "bird": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102], "fish": [389, 390, 391, 392, 393, 394, 395, 396], "snake": [52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68], "monkey": [365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381], "bear": [294, 295, 296, 297], "big_cats": [286, 287, 288, 289, 290, 291, 292, 293], "horse": [304, 305, 306], "deer": [301, 302, 303], "rabbit": [330, 331, 332], "squirrel": [335, 336], "cattle": [345, 346, 347], "sheep": [348, 349], "elephant": [386, 387], "bike": [444, 870, 880], "vehicle": [407, 436, 468, 479, 511, 555, 569, 573, 581, 586, 609, 627, 654, 656, 661, 675, 717, 734, 751, 757, 779, 817, 864, 866, 867]};
        this.excludeGroups = {"furniture": [508, 509, 510, 511, 512, 513, 514, 515, 516, 894], "electronics": [754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769], "sports_equipment": [701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736], "containers": [463, 464, 465, 466, 467, 468, 469, 470, 471], "tools": [845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865], "clothing": [600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621], "interior_items": [517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599], "misc": [403, 677, 784, 799, 908, 977]};
        this.includedClasses = new Set(
            Object.values(this.includeGroups).flat()
        );
        this.excludedClasses = new Set(
            Object.values(this.excludeGroups).flat()
        );

        // mapping for group names
        this._classToGroup = {};
        for (let [groupName, indices] of Object.entries(this.includeGroups)) {
            for (let idx of indices) {
                this._classToGroup[idx] = groupName;
            }
        }

        this._groupToId = {};
        Object.entries(this.includeGroups).forEach(([group, _], groupId) => {
            this._groupToId[group] = groupId;
        });

    }

    classToGroup(classId) {
        if (this.excludedClasses.has(classId)) return -1;
        const group = this._classToGroup[classId];
        const groupId = this.getGroupId(group);
        if (groupId < 0 || groupId === undefined) return -1;
        return groupId;
    }

    getGroupId(group) {
        return this._groupToId[group];
    }

    getGroupName(groupId) {
        if (groupId < 0) return '---';
        return Array.from(Object.keys(this.includeGroups))[groupId];
    }
}

function argmax_top_n(logits, n, threshold = 0) {
    let indices = logits.map((e, i) => i);
    indices.sort((a, b) => logits[b] - logits[a]);
    let top_n = [];
    for (let i = 0; i < n; i++) {
        if (logits[indices[i]] < threshold) break;

        top_n.push(indices[i]);
    }
    return top_n; //.slice(0, n);
}

function updateServiceWorker() {
    if ("caches" in window) {
        caches
            .keys()
            .then((keyList) => {
                return Promise.all(
                    keyList.map((key) => {
                        return caches.delete(key);
                    })
                );
            })
            .then(() => {
                console.log("All caches cleared.");
            })
            .catch((err) => {
                console.error("Error clearing caches:", err);
            });
    }

    if ("serviceWorker" in navigator) {
        navigator.serviceWorker
            .getRegistrations()
            .then((registrations) => {
                // Unregister all service workers
                for (let registration of registrations) {
                    registration.unregister();
                }
            })
            .catch((error) => {
                console.error("Error unregistering service worker:", error);
            });
    }
    window.location.reload();
}

let loadingText = document.getElementById("loading-text");

if ("serviceWorker" in navigator) {
    navigator.serviceWorker.addEventListener("message", (event) => {
        if (event.data && event.data.type === "progress") {
            loadingText.textContent = `Downloading ${event.data.name} (${event.data.progress}%)`;
        }
    });
}

// release camera on page unload
window.addEventListener("beforeunload", () => {
    if (window.localMediaStream) {
        window.localMediaStream.getTracks().forEach((track) => track.stop());
    }
});

init();
