if (typeof window === 'undefined') {
  var cv = require("../opencv");
}

const cvSize = {
  szODD: new cv.Size(127, 61),
  szQVGA: new cv.Size(320, 240),
  szVGA: new cv.Size(640, 480),
  szqHD: new cv.Size(960, 540),
  sz720p: new cv.Size(1280, 720),
  sz1080p: new cv.Size(1920, 1080),
  sz130x60: new cv.Size(130, 60),
  sz213x120: new cv.Size(120 * 1280 / 720, 120),
}

if (typeof window === 'undefined') {
  exports.cvSize = cvSize;
}