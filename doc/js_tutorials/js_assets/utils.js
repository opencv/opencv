function loadImageToCanvas(url, cavansId) { // eslint-disable-line no-unused-vars
    let canvas = document.getElementById(cavansId);
    let ctx = canvas.getContext('2d');
    let img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = function() {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0, img.width, img.height);
    };
    img.src = url;
}

function executeCode(codeEditorId, errorOutputId) { // eslint-disable-line no-unused-vars
    let code = document.getElementById(codeEditorId).value;
    try {
        eval(code);
        document.getElementById(errorOutputId).innerHTML = ' ';
    } catch (err) {
        handleError(err, errorOutputId);
    }
}

function handleError(err, errorOutputId) {
    if (typeof err === 'number') {
        if (!isNaN(err)) {
            err = 'Exception: ' + cv.exceptionFromPtr(err).msg;
        }
    } else if (typeof err === 'string') {
        let ptr = Number(err.split(' ')[0]);
        if (!isNaN(ptr)) {
            err = 'Exception: ' + cv.exceptionFromPtr(ptr).msg;
        }
    } else if (err instanceof Error) {
        err = err.stack.replace(/\n/g, '<br>');
    }
    document.getElementById(errorOutputId).innerHTML = err;
}

function loadCode(scriptId, codeEditorId) { // eslint-disable-line no-unused-vars
    let scriptNode = document.getElementById(scriptId);
    let codeEditor = document.getElementById(codeEditorId);
    if (scriptNode.type !== 'text/code-snippet') {
        throw Error('Unknown code snippet type');
    }
    codeEditor.value = scriptNode.text.replace(/^\n/, '');
}

function addFileInputHandler(fileInputId, canvasId) { // eslint-disable-line no-unused-vars
    let inputElement = document.getElementById(fileInputId);
    inputElement.addEventListener('change', (e) => {
        let imgUrl = URL.createObjectURL(e.target.files[0]);
        loadImageToCanvas(imgUrl, canvasId);
    }, false);
}

function onOpenCvLoadError(errorOutputId) { // eslint-disable-line no-unused-vars
    document.getElementById(errorOutputId).innerHTML = 'Failed to load opencv.js';
}

function startCamera( // eslint-disable-line no-unused-vars
    resolution, videoId, errorOutputId, callback) {
    const constraints = {
        'qvga': {width: {exact: 320}, height: {exact: 240}},
        'vga': {width: {exact: 640}, height: {exact: 480}}};
    let video = document.getElementById(videoId);
    if (!video) {
        video = document.createElement('video');
    }

    let videoConstraint = constraints[resolution];
    if (!videoConstraint) {
        videoConstraint = true;
    }

    let errorOutput = document.getElementById(errorOutputId);
    navigator.mediaDevices.getUserMedia({video: videoConstraint, audio: false})
        .then(function(stream) {
            video.srcObject = stream;
            video.play();
            video.addEventListener('canplay', () => {
                if (callback) {
                    callback(stream, video);
                }
            }, false);
            if (errorOutput) {
                errorOutput.innerHTML = ' ';
            }
        })
        .catch(function(err) {
            if (errorOutput) {
                errorOutput.innerHTML =
                    'Camera Error: ' + err.name + ' ' + err.message;
            }
        });
}

function stopCamera(stream, video) { // eslint-disable-line no-unused-vars
    if (video) {
        video.pause();
        video.srcObject = null;
    }
    if (stream) {
        stream.getVideoTracks()[0].stop();
    }
}
