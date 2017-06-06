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

QUnit.module ("Video", {});
//QUnit.test("Tracking", function(assert) {
// meanShift
//{
//}
// buildOpticalFlowPyramid
//{
//}
// DualTVL1OpticalFlow
//{
//}
//});

QUnit.test("Background Segmentation", function(assert) {

	// BackgroundSubtractorMOG2
	{
		let history= 600,
			varThreshold=15,
			detectShadows=true;

		let mog2 = new cv.BackgroundSubtractorMOG2(history, varThreshold, detectShadows);

		assert.equal(mog2.getVarThreshold(), 15);
		assert.equal(mog2.getDetectShadows(), true);

		mog2.delete();
	}

	// BackgroundSubtractorKNN
	{
		let history = 500,
			dist2Threshold = 350,
			numSamples = 10,
			detectShadows = false;

		let bsknn = new cv.BackgroundSubtractorKNN(history, dist2Threshold, detectShadows);
		bsknn.setNSamples(numSamples);
		assert.equal(bsknn.getDetectShadows(), detectShadows);
		assert.equal(bsknn.getHistory(), history);
		assert.equal(bsknn.getDist2Threshold(), dist2Threshold);
		assert.equal(bsknn.getNSamples(), numSamples);

		bsknn.delete();
	}

	// BackgroundSubtractorMOG2
	{
		let history = 500,
			numMixtures = 8,
			varThreshold = 16,
			detectShadows = true;

		var bsmog2 = new cv.BackgroundSubtractorMOG2(history, varThreshold, detectShadows);
		bsmog2.setNMixtures(numMixtures);
		assert.equal(bsmog2.getDetectShadows(), detectShadows, "BSMOG2.getDetectShadows");
		assert.equal(bsmog2.getHistory(), history, "BSMOG2.getHistory");
		assert.equal(bsmog2.getVarThreshold(), varThreshold, "BSMOG2.getVarThreshold");
		assert.equal(bsmog2.getNMixtures(), numMixtures, "BSMOG2.getNMixtures");

		bsmog2.delete();
	}

});
