getBlobFromImage = function(inputSize, mean, std, swapRB, image) {
    let mat;
    if (typeof(image) === 'string') {
        mat = cv.imread(image);
    } else {
        mat = image;
    }

    let matC3 = new cv.Mat(mat.matSize[0], mat.matSize[1], cv.CV_8UC3);
    cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR);
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

loadModel = async function(e) {
    return new Promise((resolve) => {
        let file = e.target.files[0];
        let path = file.name;
        let reader = new FileReader();
        reader.readAsArrayBuffer(file);
        reader.onload = function(ev) {
            if (reader.readyState === 2) {
                let buffer = reader.result;
                let data = new Uint8Array(buffer);
                cv.FS_createDataFile('/', path, data, true, false, false);
                resolve(path);
            }
        }
    });
}

getTopClasses = function(probs, labels, topK = 3) {
    probs = Array.from(probs);
    let indexes = probs.map((prob, index) => [prob, index]);
    let sorted = indexes.sort((a, b) => {
        if (a[0] === b[0]) {return 0;}
        return a[0] < b[0] ? -1 : 1;
    });
    sorted.reverse();
    let classes = [];
    for (let i = 0; i < topK; ++i) {
        let prob = sorted[i][0];
        let index = sorted[i][1];
        let c = {
            label: labels[index],
            prob: (prob * 100).toFixed(2)
        }
        classes.push(c);
    }
    return classes;
}

loadImageToCanvas = function(e, canvasId) {
    let files = e.target.files;
    let imgUrl = URL.createObjectURL(files[0]);
    let canvas = document.getElementById(canvasId);
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
    for (key of Object.keys(json)) {
        let h3 = document.createElement('h3');
        h3.textContent = key + " model";
        appendix.appendChild(h3);

        let table = document.createElement('table');
        let head_tr = document.createElement('tr');
        for (head of Object.keys(json[key][0])) {
            let th = document.createElement('th');
            th.textContent = head;
            th.style.border = "1px solid black";
            head_tr.appendChild(th);
        }
        table.appendChild(head_tr)

        for (model of json[key]) {
            let tr = document.createElement('tr');
            for (params of Object.keys(model)) {
                let td = document.createElement('td');
                td.style.border = "1px solid black";
                if (params !== "modelUrl" && params !== "configUrl" && params !== "labelsUrl") {
                    td.textContent = model[params];
                    tr.appendChild(td);
                } else {
                    let a = document.createElement('a');
                    let link = document.createTextNode('link');
                    a.append(link);
                    a.href = model[params];
                    td.appendChild(a);
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
