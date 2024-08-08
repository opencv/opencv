// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

function generateTestFrame(width, height) {
  let w = width || 200;
  let h = height || 200;
  let img = new cv.Mat(h, w, cv.CV_8UC1, new cv.Scalar(0, 0, 0, 0));
  let s = new cv.Scalar(255, 255, 255, 255);
  let s128 = new cv.Scalar(128, 128, 128, 128);
  let rect = new cv.Rect(w / 4, h / 4, w / 2, h / 2);
  img.roi(rect).setTo(s);
  img.roi(new cv.Rect(w / 2 - w / 8, h / 2 - h / 8, w / 4, h / 4)).setTo(s128);
  cv.rectangle(img, new cv.Point(w / 8, h / 8), new cv.Point(w - w / 8, h - h / 8), s, 5);
  cv.rectangle(img, new cv.Point(w / 5, h / 5), new cv.Point(w - w / 5, h - h / 5), s128, 3);
  cv.line(img, new cv.Point(-w, 0), new cv.Point(w / 2, h / 2), s128, 5);
  cv.line(img, new cv.Point(2*w, 0), new cv.Point(w / 2, h / 2), s, 5);
  return img;
}

QUnit.module('Features2D', {});
QUnit.test('Detectors', function(assert) {
  let image = generateTestFrame();

  let kp = new cv.KeyPointVector();

  let orb = new cv.ORB();
  orb.detect(image, kp);
  assert.equal(kp.size(), 67, 'ORB');

  /* TODO: Fix test failure Expected: 7 Result: 0
  bug: https://github.com/opencv/opencv/issues/25862
  let mser = new cv.MSER();
  mser.detect(image, kp);
  assert.equal(kp.size(), 7, 'MSER');
  */

  let brisk = new cv.BRISK();
  brisk.detect(image, kp);
  assert.equal(kp.size(), 191, 'BRISK');

  let ffd = new cv.FastFeatureDetector();
  ffd.detect(image, kp);
  assert.equal(kp.size(), 12, 'FastFeatureDetector');

  let afd = new cv.AgastFeatureDetector();
  afd.detect(image, kp);
  assert.equal(kp.size(), 67, 'AgastFeatureDetector');

  let gftt = new cv.GFTTDetector();
  gftt.detect(image, kp);
  assert.equal(kp.size(), 168, 'GFTTDetector');

  let kaze = new cv.KAZE();
  kaze.detect(image, kp);
  assert.equal(kp.size(), 159, 'KAZE');

  let akaze = new cv.AKAZE();
  akaze.detect(image, kp);
  assert.equal(kp.size(), 53, 'AKAZE');
});

QUnit.test('SimpleBlobDetector', function(assert) {
  let image = generateTestFrame();

  let kp = new cv.KeyPointVector();
  let sbd = new cv.SimpleBlobDetector();
  sbd.detect(image, kp);
  assert.equal(kp.size(), 0);
});

QUnit.test('BFMatcher', function(assert) {
  // Generate key points.
  let image = generateTestFrame();

  let kp = new cv.KeyPointVector();
  let descriptors = new cv.Mat();
  let orb = new cv.ORB();
  orb.detectAndCompute(image, new cv.Mat(), kp, descriptors);

  assert.equal(kp.size(), 67);

  // Run a matcher.
  let dm = new cv.DMatchVector();
  let matcher = new cv.BFMatcher();
  matcher.match(descriptors, descriptors, dm);

  assert.equal(dm.size(), 67);
});

QUnit.test('Drawing', function(assert) {
  // Generate key points.
  let image = generateTestFrame();

  let kp = new cv.KeyPointVector();
  let descriptors = new cv.Mat();
  let orb = new cv.ORB();
  orb.detectAndCompute(image, new cv.Mat(), kp, descriptors);
  assert.equal(kp.size(), 67);

  let dst = new cv.Mat();
  cv.drawKeypoints(image, kp, dst);
  assert.equal(dst.rows, image.rows);
  assert.equal(dst.cols, image.cols);

  // Run a matcher.
  let dm = new cv.DMatchVector();
  let matcher = new cv.BFMatcher();
  matcher.match(descriptors, descriptors, dm);
  assert.equal(dm.size(), 67);

  cv.drawMatches(image, kp, image, kp, dm, dst);
  assert.equal(dst.rows, image.rows);
  assert.equal(dst.cols, 2 * image.cols);

  dm = new cv.DMatchVectorVector();
  matcher.knnMatch(descriptors, descriptors, dm, 2);
  assert.equal(dm.size(), 67);
  cv.drawMatchesKnn(image, kp, image, kp, dm, dst);
  assert.equal(dst.rows, image.rows);
  assert.equal(dst.cols, 2 * image.cols);
});
