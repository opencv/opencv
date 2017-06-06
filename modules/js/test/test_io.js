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

QUnit.module ("IO", {});
QUnit.test("Test IO", function(assert) {
    const CV_LOAD_IMAGE_COLOR = 1;
    // Imwrite/Imread
    {
        let mat = new cv.Mat(48, 64, cv.CV_8UC4);
        Commons.createAlphaMat(mat);

        compressionParams = new cv.IntVector();
        compressionParams.push_back(16);
        compressionParams.push_back(9);

        cv.imwrite("alpha.png", mat, compressionParams);

        let mat2 = cv.imread("alpha.png", 1); // RGB
        assert.equal(mat.total(), mat2.total());
        assert.equal(mat2.channels(), 3);

        let mat3 = cv.imread("alpha.png", 0); //Grayscale
        assert.equal(mat.total(), mat3.total());
        assert.equal(mat3.channels(), 1);

        mat.delete();
        mat2.delete();
        mat3.delete();
        compressionParams.delete();
    }
    // Imencode/Imdecode
    {
        let mat = new cv.Mat(480, 640, cv.CV_8UC4),
        buff = new cv.UCharVector(),
        param = new cv.IntVector();

        Commons.createAlphaMat(mat);
        param.push_back(1); // CV_IMWRITE_JPEG_QUALITY
        param.push_back(95);
        cv.imencode(".png", mat, buff, param);

        let mat2 = cv.imdecode(new cv.Mat(buff), CV_LOAD_IMAGE_COLOR);

        assert.equal(mat.total(), mat2.total())

        mat.delete();
        buff.delete();
        mat2.delete();
        param.delete();
    }
    // Show image
    {
        let mat = new cv.Mat([50, 50], cv.CV_8UC4);

        Commons.createAlphaMat(mat);
        Commons.showImage(mat);

        mat.delete();
    }
});
