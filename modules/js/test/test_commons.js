/*M///////////////////////////////////////////////////////////////////////////////////////
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
//M*/

/*M///////////////////////////////////////////////////////////////////////////////////////
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
//M*/

var Commons = {
	// Utility functions
	clamp: function(num, min, max) {
		return num < min ? min : num > max ? max : num;
	},

	createAlphaMat: function(mat) {
		let UCHAR_MAX =  255;
		for (var i = 0; i < mat.rows; ++i) {
			for (var j = 0; j < mat.cols; ++j) {
				var rgba = mat.ptr(i, j);
				rgba[0] = this.clamp((mat.rows - i)/(mat.rows) * UCHAR_MAX, 0, UCHAR_MAX); // Red
				rgba[1] = this.clamp((mat.cols - j)/(mat.cols) * UCHAR_MAX, 0, UCHAR_MAX); // Green
				rgba[2] = UCHAR_MAX; // Blue
				rgba[3] = this.clamp(0.5 * (rgba[1] + rgba[2]), 0, UCHAR_MAX); // Alpha
			}
		}
	},

	showImage: function(mat) {
		let canvas = document.createElement('canvas');
		canvas.style.zIndex   = 8;
		canvas.style.border   = "1px solid";
		document.body.appendChild(canvas);

		ctx = canvas.getContext("2d");
		ctx.clearRect(0, 0, canvas.width, canvas.height);

		canvas.width = mat.cols;
		canvas.height = mat.rows;

		let imdata = ctx.createImageData(mat.cols, mat.rows);
		let data = mat.data(),
		channels = mat.channels(),
		channelSize = mat.elemSize1();

		for (var i = 0, j=0; i < data.length; i += channels, j+=4) {
			imdata.data[j] = data[i];
			imdata.data[j + 1] = data[i+1%channels];
			imdata.data[j + 2] = data[i+2%channels];
			imdata.data[j + 3] = 255;
		}

		ctx.putImageData(imdata, 0, 0);
	}

	// deepEqualWithTolerance: function(collection1, collection2, assert) {
	// 	assert(collection1.size() = collection2.size());
	// 	var size = collection1.size();
	// 	for (i = 0; i < size; i+=1) {
	// 	  assert.closeEn
	// 	}
	// }
	//
};

if (typeof module !== 'undefined' && module.exports) exports.Commons = Commons;
