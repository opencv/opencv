Module["imread"] = function(canvasID) {
    var canvas = document.getElementById(canvasID);
    if (canvas === null || !(canvas instanceof HTMLCanvasElement))  { console.warn("Please input the valid canvas id."); return; }
    var ctx = canvas.getContext("2d");
    var imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    var src = cv.matFromArray(imgData, cv.CV_8UC4);
    var dst = src.clone();
    src.delete();
    return dst;
}

Module["imshow"] = function(canvasID, mat) {
    if (mat.type() !== cv.CV_8UC4) { console.warn("Please convert mat type to cv.CV_8UC4 first."); return; }
    var imgData = new ImageData(new Uint8ClampedArray(mat.data()), mat.cols, mat.rows);
    var canvas = document.getElementById(canvasID);
    if (canvas === null || !(canvas instanceof HTMLCanvasElement))  { console.warn("Please input the valid canvas id."); return; }
    var ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.width = imgData.width;
    canvas.height = imgData.height;
    ctx.putImageData(imgData, 0, 0);
}