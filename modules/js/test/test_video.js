//  //////////////////////////////////////////////////////////////////////////////////////
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

// //////////////////////////////////////////////////////////////////////////////////////
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
//

QUnit.module('Video', {});
QUnit.test('Background Segmentation', function(assert) {
    // BackgroundSubtractorMOG2
    {
        const history = 600;
        const varThreshold = 15;
        const detectShadows = true;

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

QUnit.test('TrackerMIL', function(assert) {
    {
        let src1 = cv.Mat.zeros(100, 100, cv.CV_8UC1);
        let src2 = cv.Mat.zeros(100, 100, cv.CV_8UC1);

        let tracker = new cv.TrackerMIL();

        assert.equal(tracker instanceof cv.TrackerMIL, true);
        assert.equal(tracker instanceof cv.Tracker, true);

        let rect = new cv.Rect(10, 10, 50, 60);
        tracker.init(src1, rect);

        let [updated, rect2] = tracker.update(src2);

        assert.equal(updated, true);
        assert.equal(rect2.width, 50);
        assert.equal(rect2.height, 60);

        tracker.delete();
        src1.delete();
        src2.delete();
    }
});
