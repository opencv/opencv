/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/*M///////////////////////////////////////////////////////////////////////////////////////
// Author: Sajjad Taheri, University of California, Irvine. sajjadt[at]uci[dot]edu
//
//                             LICENSE AGREEMENT
// Copyright (c) 2015 The Regents of the University of California (Regents)
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. Neither the name of the University nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//M*/

if (typeof module !== 'undefined' && module.exports) {
    // The envrionment is Node.js
    var cv = require('./opencv.js');
    cv.FS_createLazyFile('/', 'haarcascade_eye.xml', 'data/haarcascade_eye.xml', true, false);
    cv.FS_createLazyFile('/', 'haarcascade_frontalface_default.xml', 'data/haarcascade_frontalface_default.xml', true, false);
    cv.FS_createLazyFile('/', 'hogcascade_pedestrians.xml', 'data/hogcascade_pedestrians.xml', true, false);
}

QUnit.module ("Object Detection", {});
QUnit.test("Cascade classification", function(assert) {
	// Group rectangle
	// CV_EXPORTS_W void groupRectangles(CV_IN_OUT std::vector<Rect>& rectList, CV_OUT std::vector<int>& weights,
	//                                int groupThreshold, double eps = 0.2);
	{
		let rectList = new cv.RectVector(),
			weights = new cv.IntVector(),
			groupThreshold = 1,
			eps = 0.2;

		let rect1 = new cv.Rect(1, 2, 3, 4),
			rect2 = new cv.Rect(1, 4, 2, 3);

		rectList.push_back(rect1);
		rectList.push_back(rect2);

		cv.groupRectangles(rectList, weights, groupThreshold, eps);


		rectList.delete();
		weights.delete();
	}

	// CascadeClassifier
	{
		let classifier = new cv.CascadeClassifier(),
			modelPath = '/haarcascade_frontalface_default.xml';

		assert.equal(classifier.empty(), true);


		classifier.load(modelPath);
		assert.equal(classifier.empty(), false);

		// cv.HAAR = 0
		//assert.equal(classifier.getFeatureType(), 0);

		let image = cv.Mat.eye({height: 10, width: 10}, cv.CV_8UC3),
			objects = new cv.RectVector(),
			numDetections = new cv.IntVector(),
			scaleFactor = 1.1,
			minNeighbors = 3,
			flags = 0,
			minSize = {height: 0, width: 0},
			maxSize = {height: 10, width: 10};

		classifier.detectMultiScale2(image, objects, numDetections, scaleFactor,
		minNeighbors, flags, minSize, maxSize);

		// test default parameters
		classifier.detectMultiScale2(image, objects, numDetections, scaleFactor,
		minNeighbors, flags, minSize);
		classifier.detectMultiScale2(image, objects, numDetections, scaleFactor,
		minNeighbors, flags);
		classifier.detectMultiScale2(image, objects, numDetections, scaleFactor,
		minNeighbors);
		classifier.detectMultiScale2(image, objects, numDetections, scaleFactor);

		classifier.delete();
		objects.delete();
		numDetections.delete();
	}

	// HOGDescriptor
	{
		let hog = new cv.HOGDescriptor(),
			mat = new cv.Mat({height: 10, width: 10}, cv.CV_8UC1),
			descriptors = new cv.FloatVector(),
			locations = new cv.PointVector();


		assert.equal(hog.winSize.height, 128);
		assert.equal(hog.winSize.width, 64);
		assert.equal(hog.nbins, 9);
		assert.equal(hog.derivAperture, 1);
		assert.equal(hog.winSigma, -1);
		assert.equal(hog.histogramNormType, 0);
		assert.equal(hog.nlevels, 64);

		hog.nlevels = 32;
		assert.equal(hog.nlevels, 32);

		//assert.equal(hog.empty(), false);

		//hog.compute(mat, descriptors, [4, 4], [4, 4], locations);

		// hog.detectMultiScale();

		// hog.computeGradient();

		hog.delete();
		mat.delete();
		descriptors.delete();
		locations.delete();
	}
});
