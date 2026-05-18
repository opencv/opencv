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

        let chess_corners = board.getChessboardCorners();

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
        chess_corners.delete();
    }
});

QUnit.test('ArUco2 detection', function (assert) {
    let params = new cv.aruco2_DetectionParameters();
    params.boxFilterSize = 5;
    let DICT = cv.aruco2_DictionaryType.DICT_4X4_50;

    let img = new cv.Mat();
    cv.aruco2_getFiducialMarker(img, DICT, 0, 100, true);
    assert.ok(!img.empty());

    let markers = cv.aruco2_detectFiducialMarkers(img, DICT, params);
    assert.equal(markers.size(), 1);
    assert.equal(markers.get(0).id, 0);
    assert.equal(markers.get(0).corners.size(), 4);

    params.delete();
    img.delete();
    markers.delete();
});

QUnit.test('ArUco2 MIP dictionary detection', function (assert) {
    let DICT = cv.aruco2_DictionaryType.DICT_ARUCO_MIP_36h12;
    let img = new cv.Mat();
    cv.aruco2_getFiducialMarker(img, DICT, 42, 200, true);
    assert.ok(!img.empty());

    let markers = cv.aruco2_detectFiducialMarkers(img, DICT);
    assert.equal(markers.size(), 1);
    assert.equal(markers.get(0).id, 42);
    assert.equal(markers.get(0).corners.size(), 4);

    img.delete();
    markers.delete();
});

QUnit.test('ArUco2 drawFiducialMarkers', function (assert) {
    let DICT = cv.aruco2_DictionaryType.DICT_4X4_50;
    let img = new cv.Mat();
    cv.aruco2_getFiducialMarker(img, DICT, 5, 100, true);
    assert.ok(!img.empty());

    let markers = cv.aruco2_detectFiducialMarkers(img, DICT);
    assert.equal(markers.size(), 1);

    let colorImg = new cv.Mat();
    cv.cvtColor(img, colorImg, cv.COLOR_GRAY2BGR);
    cv.aruco2_drawFiducialMarkers(colorImg, markers);
    assert.ok(!colorImg.empty());

    img.delete();
    colorImg.delete();
    markers.delete();
});

QUnit.test('ArUco2 GridBoard generate and detect', function (assert) {
    let DICT = cv.aruco2_DictionaryType.DICT_ARUCO_MIP_36h12;
    let img = new cv.Mat();
    let gridSize = new cv.Size(3, 2);
    cv.aruco2_getGridBoard(img, gridSize, DICT, 20);
    assert.ok(!img.empty());

    let canvas = new cv.Mat(img.rows + 100, img.cols + 100, cv.CV_8UC1);
    canvas.setTo([255]);
    let roi = new cv.Rect(50, 50, img.cols, img.rows);
    let sub = canvas.roi(roi);
    img.copyTo(sub);
    sub.delete();

    let board = new cv.aruco2_GridBoard();
    let found = cv.aruco2_detectGridBoard(canvas, gridSize, DICT, board);
    assert.ok(found);
    assert.equal(board.markers.size(), 6);

    let colorCanvas = new cv.Mat();
    cv.cvtColor(canvas, colorCanvas, cv.COLOR_GRAY2BGR);
    cv.aruco2_drawGridBoard(colorCanvas, board);
    assert.ok(!colorCanvas.empty());

    let objPts = new cv.Mat();
    let imgPts = new cv.Mat();
    cv.aruco2_getSolvePnpPoints1(board, objPts, imgPts);
    // 3x2 board → (3+1)×(2+1) = 12 corners
    assert.equal(objPts.rows, 12);
    assert.equal(imgPts.rows, 12);

    img.delete(); canvas.delete(); board.delete(); colorCanvas.delete();
    objPts.delete(); imgPts.delete();
});

QUnit.test('ArUco2 getSolvePnpPoints FiducialMarker', function (assert) {
    let DICT = cv.aruco2_DictionaryType.DICT_ARUCO_MIP_36h12;
    let img = new cv.Mat();
    cv.aruco2_getFiducialMarker(img, DICT, 100, 20, false);

    let canvas = new cv.Mat(img.rows * 2, img.cols * 2, cv.CV_8UC1);
    canvas.setTo([255]);
    let roi = new cv.Rect(img.cols / 2 | 0, img.rows / 2 | 0, img.cols, img.rows);
    let sub = canvas.roi(roi);
    img.copyTo(sub);
    sub.delete();

    let markers = cv.aruco2_detectFiducialMarkers(canvas, DICT);
    assert.equal(markers.size(), 1);

    let objPts = new cv.Mat();
    let imgPts = new cv.Mat();
    cv.aruco2_getSolvePnpPoints(markers.get(0), objPts, imgPts);
    assert.equal(objPts.rows, 4);
    assert.equal(imgPts.rows, 4);

    img.delete(); canvas.delete(); markers.delete(); objPts.delete(); imgPts.delete();
});

QUnit.test('ArUco2 multi-dict detection', function (assert) {
    let DICT1 = cv.aruco2_DictionaryType.DICT_4X4_50;
    let DICT2 = cv.aruco2_DictionaryType.DICT_ARUCO_MIP_36h12;

    let img1 = new cv.Mat();
    let img2 = new cv.Mat();
    cv.aruco2_getFiducialMarker(img1, DICT1, 5, 20, false);
    cv.aruco2_getFiducialMarker(img2, DICT2, 10, 20, false);

    let canvas = new cv.Mat(600, 600, cv.CV_8UC1);
    canvas.setTo([255]);
    let sub1 = canvas.roi(new cv.Rect(100, 100, img1.cols, img1.rows));
    img1.copyTo(sub1);
    sub1.delete();
    let sub2 = canvas.roi(new cv.Rect(300, 300, img2.cols, img2.rows));
    img2.copyTo(sub2);
    sub2.delete();

    let dicts = new cv.DictionaryTypeVector();
    dicts.push_back(DICT1);
    dicts.push_back(DICT2);
    let markers = cv.aruco2_detectFiducialMarkers1(canvas, dicts);
    assert.equal(markers.size(), 2);

    img1.delete(); img2.delete(); canvas.delete(); dicts.delete(); markers.delete();
});

QUnit.test('ArUco2 drawAxis', function (assert) {
    let DICT = cv.aruco2_DictionaryType.DICT_ARUCO_MIP_36h12;
    let img = new cv.Mat();
    cv.aruco2_getFiducialMarker(img, DICT, 42, 20, true);

    let colorImg = new cv.Mat();
    cv.cvtColor(img, colorImg, cv.COLOR_GRAY2BGR);

    let cameraMatrix = cv.matFromArray(3, 3, cv.CV_64F, [500, 0, 200, 0, 500, 200, 0, 0, 1]);
    let distCoeffs = cv.Mat.zeros(4, 1, cv.CV_64F);
    let rvec = cv.Mat.zeros(3, 1, cv.CV_64F);
    let tvec = cv.matFromArray(3, 1, cv.CV_64F, [0, 0, 1]);

    cv.aruco2_drawAxis(colorImg, cameraMatrix, distCoeffs, rvec, tvec, 0.1);
    assert.ok(!colorImg.empty());

    img.delete(); colorImg.delete();
    cameraMatrix.delete(); distCoeffs.delete(); rvec.delete(); tvec.delete();
});

QUnit.test('ArUco2 Diamond workflow', function (assert) {
    let DICT = cv.aruco2_DictionaryType.DICT_ARUCO_MIP_36h12;
    let diamondImg = new cv.Mat();
    cv.aruco2_getDiamondImage(diamondImg, DICT, [5, 10, 15, 20], 20);
    assert.ok(!diamondImg.empty());

    let canvas = new cv.Mat(diamondImg.rows + 100, diamondImg.cols + 100, cv.CV_8UC1);
    canvas.setTo([255]);
    let sub = canvas.roi(new cv.Rect(50, 50, diamondImg.cols, diamondImg.rows));
    diamondImg.copyTo(sub);
    sub.delete();

    let diamonds = cv.aruco2_detectDiamonds(canvas, DICT);
    assert.equal(diamonds.size(), 1);

    let colorCanvas = new cv.Mat();
    cv.cvtColor(canvas, colorCanvas, cv.COLOR_GRAY2BGR);
    cv.aruco2_drawDiamonds(colorCanvas, diamonds);
    assert.ok(!colorCanvas.empty());

    let objPts = new cv.Mat();
    let imgPts = new cv.Mat();
    cv.aruco2_getSolvePnpPoints2(diamonds.get(0), objPts, imgPts);
    // Diamond → 3x3 grid = 9 points
    assert.equal(objPts.rows, 9);
    assert.equal(imgPts.rows, 9);

    diamondImg.delete(); canvas.delete(); diamonds.delete();
    colorCanvas.delete(); objPts.delete(); imgPts.delete();
});

QUnit.test('ArUco2 Fractal workflow', function (assert) {
    let fractalImg = new cv.Mat();
    cv.aruco2_getFractalImage(fractalImg, cv.aruco2_FractalType.FRACTAL_2L_6, 40);
    assert.ok(!fractalImg.empty());
    assert.equal(fractalImg.rows, fractalImg.cols);

    let canvas = new cv.Mat(fractalImg.rows + 100, fractalImg.cols + 100, cv.CV_8UC1);
    canvas.setTo([255]);
    let sub = canvas.roi(new cv.Rect(50, 50, fractalImg.cols, fractalImg.rows));
    fractalImg.copyTo(sub);
    sub.delete();

    let fractals = cv.aruco2_detectFractals(canvas, cv.aruco2_FractalType.FRACTAL_2L_6);
    assert.equal(fractals.size(), 1);

    let colorCanvas = new cv.Mat();
    cv.cvtColor(canvas, colorCanvas, cv.COLOR_GRAY2BGR);
    cv.aruco2_drawFractals(colorCanvas, fractals);
    assert.ok(!colorCanvas.empty());

    let objPts = new cv.Mat();
    let imgPts = new cv.Mat();
    cv.aruco2_getSolvePnpPoints3(fractals.get(0), objPts, imgPts);
    assert.ok(objPts.rows >= 4);
    assert.equal(objPts.rows, imgPts.rows);

    fractalImg.delete(); canvas.delete(); fractals.delete();
    colorCanvas.delete(); objPts.delete(); imgPts.delete();
});
