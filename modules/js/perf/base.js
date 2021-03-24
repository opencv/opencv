if (typeof window === 'undefined') {
  var cv = require("../opencv");
  if (cv instanceof Promise) {
    loadOpenCV();
  } else {
    cv.onRuntimeInitialized = perf;
  }
}

let gCvSize;

function getCvSize() {
  if (gCvSize === undefined) {
    gCvSize = {
      szODD: new cv.Size(127, 61),
      szQVGA: new cv.Size(320, 240),
      szVGA: new cv.Size(640, 480),
      szSVGA: new cv.Size(800, 600),
      szqHD: new cv.Size(960, 540),
      szXGA: new cv.Size(1024, 768),
      sz720p: new cv.Size(1280, 720),
      szSXGA: new cv.Size(1280, 1024),
      sz1080p: new cv.Size(1920, 1080),
      sz130x60: new cv.Size(130, 60),
      sz213x120: new cv.Size(120 * 1280 / 720, 120),
    };
  }

  return gCvSize;
}

async function loadOpenCV() {
  cv = await cv;
}

if (typeof window === 'undefined') {
  exports.getCvSize = getCvSize;
}