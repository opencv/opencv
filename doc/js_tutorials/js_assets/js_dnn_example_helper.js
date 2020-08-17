getBlodFromImage = function(inputSize, mean, std, swapRB, canvasId) {
    let mat = cv.imread(canvasId);
    let matC3 = new cv.Mat(mat.matSize[0], mat.matSize[1], cv.CV_8UC3);
    cv.cvtColor(mat, matC3, cv.COLOR_RGBA2RGB);
    let input = cv.blobFromImage(matC3, std, new cv.Size(inputSize[0], inputSize[1]),
                                 new cv.Scalar(mean[0], mean[1], mean[2]), swapRB);
    mat.delete();
    matC3.delete();
    return input;
}

getBlodFromFrame = function(inputSize, mean, std, swapRB, frame) {
    let matC3 = new cv.Mat(frame.matSize[0],frame.matSize[1],cv.CV_8UC3);
    cv.cvtColor(frame, matC3, cv.COLOR_RGBA2RGB);
    let input = cv.blobFromImage(matC3, std, new cv.Size(inputSize[0], inputSize[1]),
                                 new cv.Scalar(mean[0], mean[1], mean[2]), swapRB);
    matC3.delete();
    return input;
}

loadLables = async function(labelsUrl) {
    let response = await fetch(labelsUrl);
    let label = await response.text();
    label = label.split('\n');
    return label;
}

getNameFromUrl = function(url) {
    const modelParts = url.modelUrl.split('/');
    const modelPath = modelParts[modelParts.length-1];
    const configParts = url.configUrl.split('/');
    const configPath = configParts[configParts.length-1];
    return {
        modelPath: modelPath,
        configPath: configPath
    };
}

loadModel = async function(url) {
    path = getNameFromUrl(url);
    return new Promise((resolve) => {
        // check if the model has been loaded before
        if(modelLoaded.indexOf(path.modelPath) == -1){
            document.getElementById('status').innerHTML = 'Loading model...';
            utils.createFileFromUrl(path.modelPath, url.modelUrl, () => {
                modelLoaded.push(path.modelPath);
                // check if need to load config file
                if(url.configUrl !== "") {
                    utils.createFileFromUrl(path.configPath, url.configUrl, () => {
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

getModelType = function(modelPath) {
    let parts = modelPath.split('.');
    let format = parts[parts.length-1];
    parts.pop();
    let modelType;
    switch(format) {
        case "caffemodel": {
            modelType = "Caffe";
        } break;
        case "onnx": {
            modelType = "ONNX";
        } break;
        case "pb": {
            modelType = "Tensorflow";
        } break;
        case "weights": {
            modelType = "Darknet";
        } break;
        case "t7":
        case "net": {
            modelType = "Torch"
        } break;
        default: break;
    }

    return modelType;
}

loadImageToCanvas = function(e) {
    let files = e.target.files;
    let imgUrl = URL.createObjectURL(files[0]);
    let canvas = document.getElementById("canvasInput");
    let ctx = canvas.getContext('2d');
    let img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = imgUrl;
    img.onload = function() {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
}

drawInfoTable = async function(jsonUrl, divId) {
    let response = await fetch(jsonUrl);
    let json = await response.json();

    let appendix = document.getElementById(divId);
    for(key of Object.keys(json)) {
        let h3 = document.createElement('h3');
        h3.textContent = key + " model";
        appendix.appendChild(h3);

        let table = document.createElement('table');
        let head_tr = document.createElement('tr');
        for(head of Object.keys(json[key][0])) {
            if(head !== "modelUrl" && head !== "configUrl") {
                let th = document.createElement('th');
                th.textContent = head;
                th.style.border = "1px solid black";
                head_tr.appendChild(th);
            }
        }
        table.appendChild(head_tr)

        for(model of json[key]) {
            let tr = document.createElement('tr');
            for(params of Object.keys(model)) {
                if(params !== "modelUrl" && params !== "configUrl") {
                    let td = document.createElement('td');
                    td.textContent = model[params];
                    td.style.border = "1px solid black";
                    tr.appendChild(td);
                }
            }
            table.appendChild(tr);
        }
        table.style.width = "800px";
        table.style.borderCollapse = "collapse";
        appendix.appendChild(table);
    }
}
