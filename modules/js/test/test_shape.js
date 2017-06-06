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

QUnit.module ("Shapes", {});
QUnit.test("Test transformers", function(assert) {
	// ShapeTransformer
	{
		let regParamSize = 0,
		    transformer = new cv.ThinPlateSplineShapeTransformer(regParamSize);

		assert.equal(transformer.getRegularizationParameter(), 0);

		transformer.setRegularizationParameter(1);
		assert.equal(transformer.getRegularizationParameter(), 1);

		transformer.delete();
	}
	// AffineTransformer
	{
	let transformer = new cv.AffineTransformer(true);
	assert.equal(transformer.getFullAffine(), true);
	transformer.delete();
	}

});


QUnit.test("Test Histogram Extractor", function(assert) {
	{
		let flag = cv.DistanceTypes.DIST_L2.value,
		    numDummies = 20,
		    cost = 80;

		let extractor = new cv.NormHistogramCostExtractor(flag, numDummies, cost);

		//assert.equal(extractor.getNormFlag(), flag);
		assert.equal(extractor.getNDummies(), numDummies);
		assert.equal(extractor.getDefaultCost(), cost);

		let matDim = 10;

		let mat1 = cv.Mat.eye([matDim, matDim], cv.CV_8UC4),
		mat2 = cv.Mat.ones([matDim, matDim], cv.CV_8UC4),
		mat3 = new cv.Mat();

		extractor.buildCostMatrix(mat1, mat2, mat3);

		assert.equal(mat3.rows, matDim + numDummies);
		assert.equal(mat3.channels(), 1);
		assert.equal(mat3.elemSize1(), 4);

		mat1.delete();
		mat2.delete();
		mat3.delete();
		extractor.delete();
	}
});
