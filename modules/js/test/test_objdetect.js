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
}

QUnit.module('Object Detection', {});
QUnit.test('QR code detect and decode', function (assert) {
    {
        const detector = new cv.QRCodeDetector();
        let mat = cv.Mat.ones(800, 600, cv.CV_8U);
        assert.ok(mat);

        // test detect
        let points = new cv.Mat();
        let qrCodeFound = detector.detect(mat, points);
        assert.equal(points.rows, 0)
        assert.equal(points.cols, 0)
        assert.equal(qrCodeFound, false);

        // test detectMult
        qrCodeFound = detector.detectMulti(mat, points);
        assert.equal(points.rows, 0)
        assert.equal(points.cols, 0)
        assert.equal(qrCodeFound, false);

        // test decode (with random numbers)
        let decodeTestPoints = cv.matFromArray(1, 4, cv.CV_32FC2, [10, 20, 30, 40, 60, 80, 90, 100]);
        let qrCodeContent = detector.decode(mat, decodeTestPoints);
        assert.equal(typeof qrCodeContent, 'string');
        assert.equal(qrCodeContent, '');

        //test detectAndDecode
        qrCodeContent = detector.detectAndDecode(mat);
        assert.equal(typeof qrCodeContent, 'string');
        assert.equal(qrCodeContent, '');

        // test decodeCurved
        qrCodeContent = detector.decodeCurved(mat, decodeTestPoints);
        assert.equal(typeof qrCodeContent, 'string');
        assert.equal(qrCodeContent, '');

        decodeTestPoints.delete();
        points.delete();
        mat.delete();

    }
});
QUnit.test('Aruco-based QR code detect', function (assert) {
    {
        let qrcode_params = new cv.QRCodeDetectorAruco_Params();
        let detector = new cv.QRCodeDetectorAruco();
        let mat = cv.Mat.ones(800, 600, cv.CV_8U);
        assert.ok(mat);

        detector.setDetectorParameters(qrcode_params);

        let points = new cv.Mat();
        let qrCodeFound = detector.detect(mat, points);
        assert.equal(points.rows, 0)
        assert.equal(points.cols, 0)
        assert.equal(qrCodeFound, false);

        qrcode_params.delete();
        detector.delete();
        points.delete();
        mat.delete();
    }
});
QUnit.test('Bar code detect', function (assert) {
    {
        let detector = new cv.barcode_BarcodeDetector();
        let mat = cv.Mat.ones(800, 600, cv.CV_8U);
        assert.ok(mat);

        let points = new cv.Mat();
        let codeFound = detector.detect(mat, points);
        assert.equal(points.rows, 0)
        assert.equal(points.cols, 0)
        assert.equal(codeFound, false);

        codeContent = detector.detectAndDecode(mat);
        assert.equal(typeof codeContent, 'string');
        assert.equal(codeContent, '');

        detector.delete();
        points.delete();
        mat.delete();
    }
});
QUnit.test('Aruco detector', function (assert) {
    {
        let dictionary = cv.getPredefinedDictionary(cv.DICT_4X4_50);
        let aruco_image = new cv.Mat();
        let detectorParameters = new cv.aruco_DetectorParameters();
        let refineParameters = new cv.aruco_RefineParameters(10, 3, true);
        let detector = new cv.aruco_ArucoDetector(dictionary, detectorParameters,refineParameters);
        let corners = new cv.MatVector();
        let ids = new cv.Mat();

        dictionary.generateImageMarker(10, 128, aruco_image);
        assert.ok(!aruco_image.empty());

        detector.detectMarkers(aruco_image, corners, ids);

        dictionary.delete();
        aruco_image.delete();
        detectorParameters.delete();
        refineParameters.delete();
        detector.delete();
        corners.delete();
        ids.delete();
    }
});
QUnit.test('Charuco detector', function (assert) {
    {
        let dictionary = new cv.getPredefinedDictionary(cv.DICT_4X4_50);
        let boardIds = new cv.Mat();
        let board = new cv.aruco_CharucoBoard(new cv.Size(3, 5), 64, 32, dictionary, boardIds);
        let charucoParameters = new cv.aruco_CharucoParameters();
        let detectorParameters = new cv.aruco_DetectorParameters();
        let refineParameters = new cv.aruco_RefineParameters(10, 3, true);
        let detector = new cv.aruco_CharucoDetector(board, charucoParameters, detectorParameters, refineParameters);
        let board_image = new cv.Mat();
        let corners = new cv.Mat();
        let ids = new cv.Mat();

        board.generateImage(new cv.Size(300, 500), board_image);
        assert.ok(!board_image.empty());

        detector.detectBoard(board_image, corners, ids);
        assert.ok(!corners.empty());
        assert.ok(!ids.empty());

        dictionary.delete();
        boardIds.delete();
        board.delete();
        board_image.delete();
        charucoParameters.delete();
        detectorParameters.delete();
        refineParameters.delete();
        detector.delete();
        corners.delete();
        ids.delete();
    }
});
