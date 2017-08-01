Module["imread"] = function(canvasID) {
    var canvas = document.getElementById(canvasID);
    if (canvas === null || !(canvas instanceof HTMLCanvasElement))
        { throw("Please input the valid canvas id."); return; }
    var ctx = canvas.getContext("2d");
    var imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return cv.matFromArray(imgData, cv.CV_8UC4);
}

Module["imshow"] = function(canvasID, mat) {
    var canvas = document.getElementById(canvasID);
    if (canvas === null || !(canvas instanceof HTMLCanvasElement))
        { throw("Please input the valid canvas id."); return; }
    if (!(mat instanceof cv.Mat))
        { throw("Please input the valid cv.Mat instance."); return; }

    // convert the mat type to cv.CV_8U
    var img = new cv.Mat();
    var depth = mat.type()%8;
    var scale = depth <= cv.CV_8S? 1.0 : (depth <= cv.CV_32S? 1.0/256.0 : 255.0)
    var shift = (depth === cv.CV_8S || depth === cv.CV_16S)? 128.0 : 0.0;
    mat.convertTo(img, cv.CV_8U, scale, shift);

    // convert the img type to cv.CV_8UC4
    switch (img.type()) {
        case cv.CV_8UC1:
            cv.cvtColor(img, img, cv.COLOR_GRAY2RGBA);
            break;
        case cv.CV_8UC3:
            cv.cvtColor(img, img, cv.COLOR_RGB2RGBA);
            break;
        case cv.CV_8UC4:
            break;
        default:
            throw("Bad number of channels (Source image must have 1, 3 or 4 channels)");
            return;
    }
    var imgData = new ImageData(new Uint8ClampedArray(img.data()), img.cols, img.rows);
    var ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.width = imgData.width;
    canvas.height = imgData.height;
    ctx.putImageData(imgData, 0, 0);
    img.delete();
}

Module["VideoCapture"] = function(videoID) {
    var video = document.getElementById(videoID);
    if (video === null || !(video instanceof HTMLVideoElement))
        { throw("Please input the valid video id."); return; }
    var canvas = document.createElement("canvas");
    canvas.width = video.width;
    canvas.height = video.height;
    var ctx = canvas.getContext("2d");

    this.read = function(frame) {
        if (!(frame instanceof cv.Mat))
            { throw("Please input the valid cv.Mat instance."); return; }
        if (frame.type() !== cv.CV_8UC4)
            { throw("Bad type of input mat: the type should be cv.CV_8UC4."); return; }
        if (frame.cols !== video.width || frame.rows !== video.height)
            { throw("Bad size of input mat: the size should be same as the video."); return; }
        ctx.drawImage(video, 0, 0, video.width, video.height);
        frame.data().set(ctx.getImageData(0, 0, video.width, video.height).data);
    };
}