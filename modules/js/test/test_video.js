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
if (typeof module !== 'undefined' && module.exports) {
    // The envrionment is Node.js
    var cv = require('./opencv.js');
}

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

		assert.equal(mog2 instanceof cv.BackgroundSubtractorMOG2, true);

		mog2.delete();

		mog2 = new cv.BackgroundSubtractorMOG2();

		assert.equal(mog2 instanceof cv.BackgroundSubtractorMOG2, true);

		mog2.delete();

		mog2 = new cv.BackgroundSubtractorMOG2(history);

		assert.equal(mog2 instanceof cv.BackgroundSubtractorMOG2, true);

		mog2.delete();

		mog2 = new cv.BackgroundSubtractorMOG2(history, varThreshold);

		assert.equal(mog2 instanceof cv.BackgroundSubtractorMOG2, true);

		mog2.delete();
	}

});
