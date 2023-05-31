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

// Author : Rijubrata Bhaumik, Intel Corporation. rijubrata.bhaumik[at]intel[dot]com

if (typeof module !== 'undefined' && module.exports) {
    // The environment is Node.js
    var cv = require('./opencv.js'); // eslint-disable-line no-var
}


QUnit.module('Photo', {});

QUnit.test('test_photo', function(assert) {
    // CalibrateDebevec
    {
        let calibration = new cv.CalibrateDebevec();
        assert.ok(true, calibration);
        //let response = calibration.process(images, exposures);
    }
    // CalibrateRobertson
    {
        let calibration = new cv.CalibrateRobertson();
        assert.ok(true, calibration);
        //let response = calibration.process(images, exposures);
    }

    // MergeDebevec
    {
        let merge = new cv.MergeDebevec();
        assert.ok(true, merge);
        //let hdr = merge.process(images, exposures, response);
    }
    // MergeMertens
    {
        let merge = new cv.MergeMertens();
        assert.ok(true, merge);
        //let hdr = merge.process(images, exposures, response);
    }
    // MergeRobertson
    {
        let merge = new cv.MergeRobertson();
        assert.ok(true, merge);
        //let hdr = merge.process(images, exposures, response);
    }

    // TonemapDrago
    {
        let tonemap = new cv.TonemapDrago();
        assert.ok(true, tonemap);
        // let ldr = new cv.Mat();
        // let retval = tonemap.process(hdr, ldr);
    }
    // TonemapMantiuk
    {
        let tonemap = new cv.TonemapMantiuk();
        assert.ok(true, tonemap);
        // let ldr = new cv.Mat();
        // let retval = tonemap.process(hdr, ldr);
    }
    // TonemapReinhard
    {
        let tonemap = new cv.TonemapReinhard();
        assert.ok(true, tonemap);
        // let ldr = new cv.Mat();
        // let retval = tonemap.process(hdr, ldr);
    }
    // Inpaint
    {
        let src = new cv.Mat(100, 100, cv.CV_8UC3, new cv.Scalar(127, 127, 127, 255));
        let mask = new cv.Mat(100, 100, cv.CV_8UC1, new cv.Scalar(0, 0, 0, 0));
        let dst = new cv.Mat();
        cv.line(mask, new cv.Point(10, 50), new cv.Point(90, 50), new cv.Scalar(255, 255, 255, 255),5);
        cv.inpaint(src, mask, dst, 3, cv.INPAINT_TELEA);
        assert.equal(dst.rows, 100);
        assert.equal(dst.cols, 100);
        assert.equal(dst.channels(), 3);
    }
});
