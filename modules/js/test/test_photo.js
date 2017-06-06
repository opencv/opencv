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

QUnit.module ("Computational Photography", {});
QUnit.test("Test Inpinting", function(assert) {
	// Inpaint
	{
		let mat = new cv.Mat([4, 4], cv.CV_8UC3),
			mask = cv.Mat.eye([4, 4], cv.CV_8UC1),
			dest = new cv.Mat();

		cv.inpaint(mat, mask, dest, 2, cv.INPAINT_TELEA);

		let size = dest.size();
		assert.equal(size.get(0), 4);
		assert.equal(size.get(1), 4);
		assert.equal(dest.channels(), 3);

		mat.delete();
		mask.delete();
		dest.delete();
	}
});

QUnit.test("Test Denoising", function(assert) {
	//  void fastNlMeansDenoising( InputArray src, OutputArray dst, float h = 3,
	//      int templateWindowSize = 7, int searchWindowSize = 21);
	{
		let mat = new cv.Mat([4, 4], cv.CV_8UC3),
			dest = new cv.Mat();

		cv.fastNlMeansDenoising(mat, dest, 3, 7, 21);

		let size = dest.size();
		assert.equal(size.get(0), 4);
		assert.equal(size.get(1), 4);
		assert.equal(dest.channels(), 3);

		size.delete();
		mat.delete();
		dest.delete();
	}
	// void fastNlMeansDenoisingColoredMulti( InputArrayOfArrays srcImgs, OutputArray dst,
	//      int imgToDenoiseIndex, int temporalWindowSize,
	//      float h = 3, float hColor = 3,
	//      int templateWindowSize = 7, int searchWindowSize = 21);
	{
		let mat1 = new cv.Mat([4, 4], cv.CV_8UC3),
			mat2 = new cv.Mat([4, 4], cv.CV_8UC3),
			mat3 = new cv.Mat([4, 4], cv.CV_8UC3);

		let inputArray = new cv.MatVector();
		inputArray.push_back(mat1);
		inputArray.push_back(mat2);
		inputArray.push_back(mat3);

		let dest = new cv.Mat();

		cv.fastNlMeansDenoisingColoredMulti(inputArray, dest, 1, 1, 3, 3, 7, 21);

		let size = dest.size();
		assert.equal(size.get(0), 4);
		assert.equal(size.get(1), 4);
		assert.equal(dest.channels(), 3);

		size.delete();
		mat1.delete();
		mat2.delete();
		mat3.delete();
		dest.delete();
		inputArray.delete();
	}
	// void denoise_TVL1(const std::vector<Mat>& observations,Mat& result, double lambda=1.0, int niters=30);
	{
		let mat1 = new cv.Mat([4, 4], cv.CV_8UC1),
			mat2 = new cv.Mat([4, 4], cv.CV_8UC1),
			mat3 = new cv.Mat([4, 4], cv.CV_8UC1),
			dest = new cv.Mat(),
			inputArray = new cv.MatVector();

		inputArray.push_back(mat1);
		inputArray.push_back(mat2);
		inputArray.push_back(mat3);

		cv.denoise_TVL1(inputArray, dest, 1.0, 30);

		let size = dest.size();
		assert.equal(size.get(0), 4);
		assert.equal(size.get(1), 4);
		assert.equal(dest.channels(), 1);

		size.delete();
		mat1.delete();
		mat2.delete();
		mat3.delete();
		dest.delete();
		inputArray.delete();
	}
});

QUnit.test("Tone Mapping", function(assert) {
	// Linear Mapper
	{
		let gamma = 1.0,
			mapper = new cv.Tonemap(gamma),
			mat = new cv.Mat([4, 4], cv.CV_32SC3),
			dst = new cv.Mat();

		assert.equal(mapper.getGamma(), 1);
		mapper.process(mat, dst);

		let size = dst.size();
		assert.equal(size.get(0), 4);
		assert.equal(size.get(1), 4);
		assert.equal(dst.channels(), 3);

		size.delete();
		mapper.delete();
		mat.delete();
		dst.delete();
	}
	// Durand Mapper
	{
		let gamma = 1.0,
			contrast = 4.0,
			saturation = 1.0,
			sigma_space = 2.0,
			sigma_color = 2.0,
			mapper = new cv.TonemapDurand(gamma, contrast, saturation, sigma_space, sigma_color);


		assert.equal(mapper.getGamma(), gamma);
		assert.equal(mapper.getContrast(), contrast);
		assert.equal(mapper.getSaturation(), saturation);
		assert.equal(mapper.getSigmaSpace(), sigma_space);
		assert.equal(mapper.getSigmaColor(), sigma_color);


		let mat = new cv.Mat([4, 4], cv.CV_32SC3),
			dst = new cv.Mat();

		mapper.process(mat, dst);

		let size = dst.size();
		assert.equal(size.get(0), 4);
		assert.equal(size.get(1), 4);
		assert.equal(dst.channels(), 3);

		size.delete();
		mapper.delete();
		mat.delete();
		dst.delete();
	}
});

QUnit.test("Image Allignment", function(assert) {
	// AlignMTB
	{
		let maxBits = 6,
			excludeRange = 4,
			cut = true,
			mtb = new cv.AlignMTB(maxBits, excludeRange, cut);

		assert.equal(mtb.getMaxBits(), maxBits);
		assert.equal(mtb.getExcludeRange(), excludeRange);
		assert.equal(mtb.getCut(), cut);

		let mat = new cv.Mat([4, 4], cv.CV_8UC1),
			mat2 = cv.Mat.eye([4, 4], cv.CV_8UC1),
			point = mtb.calculateShift(mat, mat2),
			dst = new cv.Mat(),
			dst2 = new cv.Mat();

		mtb.computeBitmaps(mat, dst, dst2);

		let size = dst.size();
		assert.equal(size.get(0), 4);
		assert.equal(size.get(1), 4);
		assert.equal(dst.channels(), 1);


		size = dst2.size();
		assert.equal(size.get(0), 4);
		assert.equal(size.get(1), 4);
		assert.equal(dst2.channels(), 1);

		mtb.delete();
		mat.delete();
		mat2.delete();
		dst.delete();
		dst2.delete();
	}
	// Robertson Calibrate
	{
		let maxIter = 30,
			threshold = 0.01;

		let mat = cv.Mat.eye([4,4], cv.CV_8UC3),
			mat2 = cv.Mat.ones([4,4], cv.CV_8UC3),
			times = cv.Mat.ones([1,2], cv.CV_8UC3),
			dst = new cv.Mat(),
			inputVector = new cv.MatVector();

		inputVector.push_back(mat);
		inputVector.push_back(mat2);

		let calib = new cv.CalibrateRobertson(maxIter, threshold);

		assert.equal(calib.getMaxIter(), maxIter);
		assert.equal(Math.abs(calib.getThreshold()-threshold) < 0.0001, true);

		calib.process(inputVector, dst, times);

		let size = dst.size();
		assert.equal(size.get(0), 256);
		assert.equal(size.get(1), 1);
		assert.equal(dst.channels(), 3);

		mat.delete();
		mat2.delete();
		dst.delete();
		times.delete();
		inputVector.delete();
	}
	//
	{

	}
});
