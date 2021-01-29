// //////////////////////////////////////////////////////////////////////////////////////
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

if (typeof module !== 'undefined' && module.exports) {
    // The environment is Node.js
    var cv = require('./opencv.js'); // eslint-disable-line no-var
    cv.FS_createLazyFile('/', 'haarcascade_frontalface_default.xml', // eslint-disable-line new-cap
                         'haarcascade_frontalface_default.xml', true, false);
}

QUnit.module('Object Detection', {});
QUnit.test('Cascade classification', function(assert) {
    // Group rectangle
    {
        let rectList = new cv.RectVector();
        let weights = new cv.IntVector();
        let groupThreshold = 1;
        const eps = 0.2;

        let rect1 = new cv.Rect(1, 2, 3, 4);
        let rect2 = new cv.Rect(1, 4, 2, 3);

        rectList.push_back(rect1);
        rectList.push_back(rect2);

        cv.groupRectangles(rectList, weights, groupThreshold, eps);


        rectList.delete();
        weights.delete();
    }

    // CascadeClassifier
    {
        let classifier = new cv.CascadeClassifier();
        const modelPath = '/haarcascade_frontalface_default.xml';

        assert.equal(classifier.empty(), true);


        classifier.load(modelPath);
        assert.equal(classifier.empty(), false);

        let image = cv.Mat.eye({height: 10, width: 10}, cv.CV_8UC3);
        let objects = new cv.RectVector();
        let numDetections = new cv.IntVector();
        const scaleFactor = 1.1;
        const minNeighbors = 3;
        const flags = 0;
        const minSize = {height: 0, width: 0};
        const maxSize = {height: 10, width: 10};

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
        let hog = new cv.HOGDescriptor();
        let mat = new cv.Mat({height: 10, width: 10}, cv.CV_8UC1);
        let descriptors = new cv.FloatVector();
        let locations = new cv.PointVector();


        assert.equal(hog.winSize.height, 128);
        assert.equal(hog.winSize.width, 64);
        assert.equal(hog.nbins, 9);
        assert.equal(hog.derivAperture, 1);
        assert.equal(hog.winSigma, -1);
        assert.equal(hog.histogramNormType, 0);
        assert.equal(hog.nlevels, 64);

        hog.nlevels = 32;
        assert.equal(hog.nlevels, 32);

        hog.delete();
        mat.delete();
        descriptors.delete();
        locations.delete();
    }
});
QUnit.test('QR code detect and decode', function (assert) {
    {
        assert.expect(12);
        let done = assert.async(2);

        const detector = new cv.QRCodeDetector();
        const imgSingleQrCode = new Image();
        imgSingleQrCode.crossOrigin = 'anonymous';
        imgSingleQrCode.onload = function () {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = imgSingleQrCode.width;
            canvas.height = imgSingleQrCode.height;
            ctx.drawImage(imgSingleQrCode, 0, 0, imgSingleQrCode.width,
                imgSingleQrCode.height);
            const imgMat = cv.imread(canvas);
            assert.ok(imgMat);

            // test detect
            let points = new cv.Mat();
            let retVal = detector.detect(imgMat, points);
            assert.ok(retVal);
            assert.equal(points.size().width, 4)
            assert.equal(points.size().height, 1);

            // test decode
            retVal = detector.decode(imgMat, points);
            assert.equal(retVal, "https://opencv.org/");

            // test detectAndDecode
            retVal = '';
            points = new cv.Mat();
            retVal = detector.detectAndDecode(imgMat, points);
            assert.equal(retVal, "https://opencv.org/");
            assert.equal(points.size().width, 4);
            assert.equal(points.size().height, 1);
            points.delete();
            imgMat.delete();
            done();

        };
        imgSingleQrCode.src = '/link_ocv.jpg';

        const imgMultiQrCode = new Image();
        imgMultiQrCode.crossOrigin = 'anonymous';
        imgMultiQrCode.onload = function () {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = imgMultiQrCode.width;
            canvas.height = imgMultiQrCode.height;
            ctx.drawImage(imgMultiQrCode, 0, 0, imgMultiQrCode.width,
                imgMultiQrCode.height);
            const imgMat = cv.imread(canvas);
            assert.ok(imgMat);

            // test detectMulti
            const points = new cv.Mat();
            let retVal = detector.detectMulti(imgMat, points);
            assert.ok(retVal);
            assert.equal(points.size().width, 4);
            assert.equal(points.size().height, 6);
            points.delete();
            imgMat.delete();
            done();

        };
        imgMultiQrCode.src = "/6_qrcodes.png";
    }
});