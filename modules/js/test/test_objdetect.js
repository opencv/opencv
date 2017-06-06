/*/////////////////////////////////////////////////////////////////////////////
AUTHOR: Sajjad Taheri sajjadt[at]uci[at]edu

                             LICENSE AGREEMENT
Copyright (c) 2015, University of california, Irvine

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. All advertising materials mentioning features or use of this software
   must display the following acknowledgement:
   This product includes software developed by the UC Irvine.
4. Neither the name of the UC Irvine nor the
   names of its contributors may be used to endorse or promote products
   derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY UC IRVINE ''AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL UC IRVINE OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/////////////////////////////////////////////////////////////////////////////*/

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
		rect1.delete();
		rect2.delete();
	}

	// CascadeClassifier
	{
		let classifier = new cv.CascadeClassifier(),
			modelPath = '../../test/data/haarcascade_frontalface_default.xml';

		assert.equal(classifier.empty(), true);


		classifier.load(modelPath);
		assert.equal(classifier.empty(), false);

		// cv.HAAR = 0
		assert.equal(classifier.getFeatureType(), 0);

		let image = cv.Mat.eye([10, 10], cv.CV_8UC3),
			objects = new cv.RectVector(),
			numDetections = new cv.IntVector(),
			scaleFactor = 1.1,
			minNeighbors = 3,
			flags = 0,
			minSize = [0, 0],
			maxSize = [10, 10];

		classifier.detectMultiScale2(image, objects, numDetections, scaleFactor,
		minNeighbors, flags, minSize, maxSize);

		classifier.delete();
		objects.delete();
		numDetections.delete();
	}

	// HOGDescriptor
	{
		let hog = new cv.HOGDescriptor(),
			mat = new cv.Mat([10, 10], cv.CV_8UC1),
			descriptors = new cv.FloatVector(),
			locations = new cv.PointVector();


		assert.equal(hog.winSize[0], 128);
		assert.equal(hog.winSize[1], 64);
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
