// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

if (typeof module !== 'undefined' && module.exports) {
  // The envrionment is Node.js
  var cv = require('./opencv.js'); // eslint-disable-line no-var
}

QUnit.module('Camera Calibration and 3D Reconstruction', {});

QUnit.test('constants', function(assert) {
  assert.strictEqual(typeof cv.LMEDS, 'number');
  assert.strictEqual(typeof cv.RANSAC, 'number');
  assert.strictEqual(typeof cv.RHO, 'number');
});

QUnit.test('findHomography', function(assert) {
  let srcPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
    56,
    65,
    368,
    52,
    28,
    387,
    389,
    390,
  ]);
  let dstPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
    0,
    0,
    300,
    0,
    0,
    300,
    300,
    300,
  ]);

  const mat = cv.findHomography(srcPoints, dstPoints);

  assert.ok(mat instanceof cv.Mat);
});
