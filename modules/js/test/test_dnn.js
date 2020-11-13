if (typeof module !== 'undefined' && module.exports) {
    // The environment is Node.js
    var cv = require('./opencv.js'); // eslint-disable-line no-var
}

QUnit.module('DNN', {});

createFileFromUrl = function(path, url, callback) {
    let request = new XMLHttpRequest();
    request.open('GET', url, true);
    request.responseType = 'arraybuffer';
    request.onload = function(ev) {
        if (request.readyState === 4) {
            if (request.status === 200) {
                let data = new Uint8Array(request.response);
                cv.FS_createDataFile('/', path, data, true, false, false);
                callback();
            } else {
                console.error('Failed to load ' + url + ' status: ' + request.status);
            }
        }
    };
    request.send();
  };

// get name of model and config file from url
function getNameFromUrl(url) {
    const modelParts = url.modelUrl.split('/');
    const modelPath = modelParts[modelParts.length-1];
    const configParts = url.configUrl.split('/');
    const configPath = configParts[configParts.length-1];
    return {
        modelPath: modelPath,
        configPath: configPath
    }
}

let modelLoaded = [];
loadModel = async function(url) {
    path = getNameFromUrl(url);
    return new Promise((resolve) => {
        // check if the model has been loaded before
        if(modelLoaded.indexOf(path.modelPath) == -1){
            createFileFromUrl(path.modelPath, url.modelUrl, () => {
                modelLoaded.push(path.modelPath);
                // check if need to load config file
                if(url.configUrl !== "") {
                    createFileFromUrl(path.configPath, url.configUrl, () => {
                        resolve(path);
                    });
                } else {
                    resolve(path);
                }
            });
        } else {
            resolve(path);
        }
    });
}

function asyncForwardWrapper(net) {
    let outputs = new cv.MatVector();
    net.forward1(outputs);
    return new Promise(function(resolve) {
        Module.Asyncify.asyncFinalizers.push(function() {
          resolve(outputs.get(0));
          outputs.delete();
        });
    });
}

async function loadAndComputeCaffeLayer(url, inputSize) {
    const path = await loadModel(url);
    let net;
    const input = cv.Mat.MatND(inputSize, cv.CV_32F);

    net = cv.readNetFromCaffe(path.modelPath, '');
    net.setInput(input);
    net.setPreferableBackend(cv.DNN_BACKEND_WEBGPU);
    net.setPreferableTarget(cv.DNN_TARGET_WEBGPU);
    const start = performance.now();
    const out = await asyncForwardWrapper(net);
    const time = (performance.now() - start).toFixed(3);
    console.log(`WebGPU backend time cost ${time} ms.`);

    input.delete();
    net.delete();
    return out;
}

QUnit.test('test_layer_softmax', async function(assert) {
    const layer_softmax = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/layers/layer_softmax.prototxt";
    const inputSize = [2, 6, 75, 113];
    const url = {
        modelUrl: layer_softmax,
        configUrl: ""
    };

    const path = await loadModel(url);
    let net;
    const input = cv.Mat.MatND(inputSize, cv.CV_32F);
    net = cv.readNetFromCaffe(path.modelPath, '');
    net.setInput(input);
    const start = performance.now();
    const out = net.forward();
    const time = (performance.now() - start).toFixed(3);
    console.log(`CPUbackend time cost ${time} ms.`);
    assert.ok(out.empty() === false);

    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    Module.preinitializedWebGPUDevice = device;
    const out1 = await loadAndComputeCaffeLayer(url, inputSize);

    const EPSILON = 0.00001;
    const data1 = out.data;
    const data2 = out1.data;
    assert.equal(data1.length, data2.length);
    let absSum = 0.0;
    const len = data1.length;
    for(let i = 0; i < len; i++)
    {
        absSum += Math.abs(data1[i] - data2[i]) / len;
    }
    console.log(`L1 norm: ${absSum}`);
    assert.ok(absSum < EPSILON);

    out.delete();
    out1.delete();
});