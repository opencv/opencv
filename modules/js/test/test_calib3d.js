// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

if (typeof module !== 'undefined' && module.exports) {
  // The environment is Node.js
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

QUnit.test('Rodrigues', function(assert) {
  // Converts a rotation matrix to a rotation vector and vice versa
  // data64F is the output array
  const rvec0 = cv.matFromArray(1, 3, cv.CV_64F, [1,1,1]);
  let rMat0 = new cv.Mat();
  let rvec1 = new cv.Mat();

  // Args: input Mat, output Mat. The function mutates the output Mat, so the function does not return anything.
  // cv.Rodrigues (InputArray=src, OutputArray=dst, jacobian=0)
  // https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#void%20Rodrigues(InputArray%20src,%20OutputArray%20dst,%20OutputArray%20jacobian)
  // vec to Mat, starting number is 3 long and each element is 1.
  cv.Rodrigues(rvec0, rMat0);

  assert.ok(rMat0.data64F.length == 9);
  assert.ok(0.23 > rMat0.data64F[0] > 0.22);

  // convert Mat to Vec, should be same as what we started with, 3 long and each item should be a 1.
  cv.Rodrigues(rMat0, rvec1);

  assert.ok(rvec1.data64F.length == 3);
  assert.ok(1.01 > rvec1.data64F[0] > 0.9);
  // Answer should be around 1: 0.9999999999999999
});

QUnit.test('estimateAffine2D', function(assert) {
   const inputs = cv.matFromArray(4, 1, cv.CV_32FC2, [
    1, 1,
    80, 0,
    0, 80,
    80, 80
  ]);
  const outputs = cv.matFromArray(4, 1, cv.CV_32FC2, [
    21, 51,
    70, 77,
    40, 40,
    10, 70
  ]);
  const M = cv.estimateAffine2D(inputs, outputs);
  assert.ok(M instanceof cv.Mat);
  assert.deepEqual(Array.from(M.data), [
     23,  55,  97, 126,  87, 139, 227,  63,   0,   0,
      0,   0,   0,   0, 232, 191,  71, 246,  12,  68,
    165,  35,  53,  64,  99,  56,  27,  66,  14, 254,
    212,  63, 103, 102, 102, 102, 102, 102, 182, 191,
    195, 252, 174,  22,  55,  97,  73,  64
  ]);
});
