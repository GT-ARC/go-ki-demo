// imort onnxruntime-web by url into worker
importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js");
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

const INPUT_WIDTH = 224;
const INPUT_HEIGHT = 224;

let layer4;
let fc;
let results;
let activations;
let output_weights;

onmessage = async (e) => {
  const { data } = e;
  if (data.status === "predict") {
    const imgDataTensor = new ort.Tensor("float32", data.tensor, [
      1,
      3,
      INPUT_HEIGHT,
      INPUT_WIDTH,
    ]);

    activations = await layer4.run({ l_x_: imgDataTensor });
    activations = activations.resnet_layer4_1.cpuData;
    const activationsTensor = new ort.Tensor(
      "float32",
      activations,
      [1, 2048, 7, 7]
    );
    results = await fc.run({ l_activations_: activationsTensor });
    results = results.fc_1.cpuData;
    const predictions = Array.from(softmax(results));
    const heatmap = averageHeatmap(activations, [2048, 7, 7]);

    postMessage({
      status: "results",
      output: results,
      predictions: predictions,
      heatmap: heatmap,
    });
  }

  if (data.status === "heatmap_by_class") {
    let output = new Float32Array(1000).fill(0);
    output[data.classIdx] = results[data.classIdx];
    //w = torch.mm(output, resnet_output_weights)
    const w = matrixMultiply(output, output_weights);
    const weighted_heatmap = averageHeatmap(activations, [2048, 7, 7], w);

    postMessage({
      status: "weighted_heatmap",
      heatmap: weighted_heatmap,
    });
  }

  if (data.status === "class_by_heatmap") {
    const idx = data.cellIdx;

    let a = new Float32Array(2048 * 7 * 7).fill(0);
    for (let i = 0; i < 2048; i++) {
      a[i * 7 * 7 + idx] = 20*activations[i * 7 * 7 + idx];
    }


    let logits = await fc.run({
      l_activations_: new ort.Tensor("float32", a, [1, 2048, 7, 7]),
    });

    logits = logits.fc_1.cpuData;

    const predictions = Array.from(softmax(logits));
    postMessage({
      status: "class_by_heatmap",
      predictions: predictions,
    });
  }
};

(async () => {
  layer4 = await ort.InferenceSession.create("resnet50_imagenet_layer4.onnx", {
    executionProviders: ["wasm"],
  });

  fc = await ort.InferenceSession.create("resnet50_imagenet_fc.onnx", {
    executionProviders: ["wasm"],
  });

  output_weights = await fetch("resnet_output_weights.bin").then((r) =>
    r.arrayBuffer()
  );
  output_weights = new Float32Array(output_weights);

  postMessage({ status: "ready" });
})();

function softmax(arr) {
  return arr.map(function (value, index) {
    return (
      Math.exp(value) /
      arr
        .map(function (y) {
          return Math.exp(y);
        })
        .reduce(function (a, b) {
          return a + b;
        })
    );
  });
}

function averageHeatmap(arr, shape, weights = null, normalize = true) {
  let heatmap = new Float32Array(shape[1] * shape[2]);
  for (let i = 0; i < shape[1]; i++) {
    for (let j = 0; j < shape[2]; j++) {
      let sum = 0;

      for (let k = 0; k < shape[0]; k++) {
        let w = weights ? weights[k] : 1;
        w = w < 0 ? 0 : w;
        sum += w * arr[k * shape[1] * shape[2] + i * shape[2] + j];
      }

      heatmap[i * shape[2] + j] = sum / shape[0];
    }
  }
  if (normalize) {
    let max = Math.max(...heatmap);
    let min = Math.min(...heatmap);
    heatmap = heatmap.map((x) => (x - min) / (max - min + 1e-6));
  }
  return heatmap;
}

function matrixMultiply(r_out, fc_weightsFlat) {
  const n_activations = 2048; // Known/implicit number of columns in fc_weights
  let result = new Float32Array(n_activations).fill(0);
  const n_classes = r_out.length;

  for (let col = 0; col < n_activations; col++) {
    let sum = 0;
    for (let row = 0; row < n_classes; row++) {
      sum += r_out[row] * fc_weightsFlat[row * n_activations + col];
    }
    result[col] = sum;
  }
  return result;
}

function weightHeatmap(w, heatmap) {
  let weighted_heatmap = new Float32Array(49).fill(0);
  for (let i = 0; i < 49; i++) {
    weighted_heatmap[i] = heatmap[i] * w;
  }
  return weighted_heatmap;
}
