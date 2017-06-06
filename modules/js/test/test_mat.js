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

QUnit.module( "Core", {});

QUnit.test("test_mat_creation", function(assert) {
    // Mat constructors.
    // Mat::Mat(int rows, int cols, int type)
    {
        let mat = new cv.Mat(10, 20, cv.CV_8UC3);

        assert.equal(mat.type(), cv.CV_8UC3);
        assert.equal(mat.depth(), cv.CV_8U);
        assert.equal(mat.channels(), 3);
        assert.ok(mat.empty() === false);

        let size = mat.size();
        assert.ok(size.size() === 2);
        assert.equal(size.get(0), 10);
        assert.equal(size.get(1), 20);

        size.delete();
        mat.delete();
    }

    // Mat::Mat(const Mat &)
    //{
        //  : Copy from another Mat
        //let mat1 = new cv.Mat(10, 20, cv.CV_8UC3);
        //let mat2 = new cv.Mat(mat1);

        //assert.equal(mat2.type(), mat1.type());
        //assert.equal(mat2.depth(), mat1.depth());
        //assert.equal(mat2.channels(), mat1.channels());
        //assert.equal(mat2.empty(), mat1.empty());

        //let size1 = mat1.size();
        //let size2 = mat2.size();
        //assert.ok(size1.size() === size2.size());
        //assert.ok(size1.get(0) === size2.get(0));
        //assert.ok(size1.get(1) === size2.get(1));

        //mat1.delete();
        //mat2.delete();
    //}

    // Mat::Mat(Size size, int type, void *data, size_t step=AUTO_STEP)
    {
        // 10 * 10 and one channel
        let data = cv._malloc(10 * 10 * 1);
        let mat = new cv.Mat([10, 10], cv.CV_8UC1, data, 0);

        assert.equal(mat.type(), cv.CV_8UC1);
        assert.equal(mat.depth(), cv.CV_8U);
        assert.equal(mat.channels(), 1);
        assert.ok(mat.empty() === false);

        let size = mat.size();
        assert.ok(size.size() === 2);
        assert.ok(size.get(0) === 10);
        assert.ok(size.get(1) === 10);

        size.delete();
        mat.delete();
    }

    //  Mat::create(int, int, int)
    {
        let mat = new cv.Mat();
        mat.create(10, 5, cv.CV_8UC3);
        let size = mat.size();

        assert.ok(mat.type() === cv.CV_8UC3);
        assert.ok(size.get(0) === 10);
        assert.ok(size.get(1) === 5);
        assert.ok(mat.channels() === 3);

        size.delete();
        mat.delete();
    }
    //  Mat::create(Size, int)
    {
        let mat = new cv.Mat();
        mat.create([10, 5], cv.CV_8UC4);
        let size = mat.size();

        assert.ok(mat.type() === cv.CV_8UC4);
        assert.ok(size.get(0) === 10);
        assert.ok(size.get(1) === 5);
        assert.ok(mat.channels() === 4);

        size.delete();
        mat.delete();
    }
    //   clone
    {
    let mat = cv.Mat.ones(5, 5, cv.CV_8UC1);
        let mat2 = mat.clone();

        assert.equal(mat.channels, mat2.channels);
        assert.equal(mat.size()[0], mat2.size()[0]);
        assert.equal(mat.size()[1], mat2.size()[1]);

        assert.deepEqual(mat.data(), mat2.data());


        mat.delete();
        mat2.delete();
    }
});

QUnit.test("test_mat_ptr", function(assert) {
    const RValue = 3;
    const GValue = 7;
    const BValue = 197;

    // cv.CV_8UC1 + Mat::ptr(int).
    {
        let mat = new cv.Mat(10, 10, cv.CV_8UC1);
        let view = mat.data();

        // Alter matrix[2, 1].
        let step = 10;
        view[2 * step + 1] = RValue;

        // Access matrix[2, 1].
        view = mat.ptr(2);

        assert.equal(view[1], RValue);
    }

    // cv.CV_8UC3 + Mat::ptr(int).
    {
        let mat = new cv.Mat(10, 10, cv.CV_8UC3);
        let view = mat.data();

        // Alter matrix[2, 1].
        let step = 3 * 10;
        view[2 * step + 3] = RValue;
        view[2 * step + 3 + 1] = GValue;
        view[2 * step + 3 + 2] = BValue;

        // Access matrix[2, 1].
        view = mat.ptr(2);

        assert.equal(view[3], RValue);
        assert.equal(view[3 + 1], GValue);
        assert.equal(view[3 + 2], BValue);
    }

    // cv.CV_8UC3 + Mat::ptr(int, int).
    {
        let mat = new cv.Mat(10, 10, cv.CV_8UC3);
        let view = mat.data();

        // Alter matrix[2, 1].
        let step = 3 * 10;
        view[2 * step + 3] = RValue;
        view[2 * step + 3 + 1] = GValue;
        view[2 * step + 3 + 2] = BValue;

        // Access matrix[2, 1].
        view = mat.ptr(2, 1);

        assert.equal(view[0], RValue);
        assert.equal(view[1], GValue);
        assert.equal(view[2], BValue);
    }
});

QUnit.test("test_mat_zeros", function(assert) {
    zeros = new Uint8Array(10*10).fill(0);
    // Mat::zeros(int, int, int)
    {
        let mat = cv.Mat.zeros(10, 10, cv.CV_8UC1);
        let view = mat.data();

        assert.deepEqual(view, zeros);

        mat.delete();
    }

    // Mat::zeros(Size, int)
    {
        let mat = cv.Mat.zeros([10, 10], cv.CV_8UC1);
        let view = mat.data();

        assert.deepEqual(view, zeros);

        mat.delete();
    }
});

QUnit.test("test_mat_ones", function(assert) {
    let ones = new Uint8Array(10*10).fill(1);
    // Mat::ones(int, int, int)
    {
        var mat = cv.Mat.ones(10, 10, cv.CV_8UC1);
        var view = mat.data();

        assert.deepEqual(view, ones);
    }
    // Mat::ones(Size, int)
    {
        var mat = cv.Mat.ones([10, 10], cv.CV_8UC1);
        var view = mat.data();

        assert.deepEqual(view, ones);
    }
});

QUnit.test("test_mat_eye", function(assert) {
    let eye4by4 = new Uint8Array([1, 0, 0, 0,
                                  0, 1, 0, 0,
                                  0, 0, 1, 0,
                                  0, 0, 0, 1])
    // Mat::eye(int, int, int)
    {
        var mat = cv.Mat.eye(4, 4, cv.CV_8UC1);
        var view = mat.data();

        assert.deepEqual(view, eye4by4);
    }

    // Mat::eye(Size, int)
    {
        var mat = cv.Mat.eye([4, 4], cv.CV_8UC1);
        var view = mat.data();

        assert.deepEqual(view, eye4by4);
    }
});

QUnit.test("test_mat_miscs", function(assert) {
    // Mat::col(int)
    {
        let mat = cv.Mat.ones(5, 5, cv.CV_8UC2);
        let col = mat.col(1);
        let view = col.data();
        assert.equal(view[0], 1);
        assert.equal(view[4], 1);

        col.delete();
        mat.delete();
    }

    // Mat::row(int)
    {
        let mat = cv.Mat.zeros(5, 5, cv.CV_8UC2);
        let row = mat.row(1);
        let view = row.data();
        assert.equal(view[0], 0);
        assert.equal(view[4], 0);

        row.delete();
        mat.delete();
    }

    // Mat::convertTo(Mat, int, double, double)
    {
        let mat = cv.Mat.ones(5, 5, cv.CV_8UC3);
        let grayMat = cv.Mat.zeros(5, 5, cv.CV_8UC1);

        mat.convertTo(grayMat, cv.CV_8U, 2, 1);
        // dest = 2 * source(x, y) + 1.
        let view = grayMat.data();
        assert.equal(view[0], (1 * 2) + 1);

        grayMat.delete();
        mat.delete();
    }

    // C++
    //   void split(InputArray, OutputArrayOfArrays)
    // Embind
    //   void split(VecotrMat, VectorMat)
    {
        const R =7;
        const G =13;
        const B =29;

        let mat = cv.Mat.ones(5, 5, cv.CV_8UC3);
        let view = mat.data();
        view[0] = R;
        view[1] = G;
        view[2] = B;

        let bgr_planes = new cv.MatVector();
        cv.split(mat, bgr_planes);
        assert.equal(bgr_planes.size(), 3);

        let rMat = bgr_planes.get(0);
        view = rMat.data();
        assert.equal(view[0], R);

        let gMat = bgr_planes.get(1);
        view = gMat.data();
        assert.equal(view[0], G);

        let bMat = bgr_planes.get(2);
        view = bMat.data();
        assert.equal(view[0], B);

        mat.delete();
        rMat.delete();
        gMat.delete();
        bgr_planes.delete();
        bMat.delete();
    }

    // C++
    //   size_t Mat::elemSize() const
    {
        let mat = cv.Mat.ones(5, 5, cv.CV_8UC3);
        assert.equal(mat.elemSize(), 3);
        assert.equal(mat.elemSize1(), 1);

        let mat2 = cv.Mat.zeros(5, 5, cv.CV_8UC1);
        assert.equal(mat2.elemSize(), 1);
        assert.equal(mat2.elemSize1(), 1);

        let mat3 = cv.Mat.eye(5, 5, cv.CV_16UC3);
        assert.equal(mat3.elemSize(), 2 * 3);
        assert.equal(mat3.elemSize1(), 2);

        mat.delete();
        mat2.delete();
        mat3.delete();
    }

    //   double Mat::dot(const Mat&) const
    {
        let mat = cv.Mat.ones(5, 5, cv.CV_8UC1),
            mat2 = cv.Mat.eye(5, 5, cv.CV_8UC1);

        assert.equal(mat.dot(mat), 25);
        assert.equal(mat.dot(mat2), 5);
        assert.equal(mat2.dot(mat2), 5);

        mat.delete();
        mat2.delete();
    }

    //   Element-wise multiplication
    //   double Mat::mul(const Mat&) const
    {
        const FACTOR = 5;
        let mat = cv.Mat.ones(4, 4, cv.CV_8UC1),
            mat2 = cv.Mat.eye(4, 4, cv.CV_8UC1);

        let expected = new Uint8Array([FACTOR, 0, 0, 0,
                                      0, FACTOR, 0, 0,
                                      0, 0, FACTOR, 0,
                                      0, 0, 0, FACTOR])
        let mat3 = mat.mul(mat2, FACTOR);

        assert.deepEqual(mat3.data(), expected);

        mat.delete();
        mat2.delete();
        mat3.delete();
    }

});


QUnit.test("test mat access", function(assert) {
    // test memory view
    {
        let data = new Uint8Array([0, 0, 0, 255, 0, 1, 2, 3]),
            dataPtr = cv._malloc(8);

        let dataHeap = new Uint8Array(cv.HEAPU8.buffer, dataPtr, 8);
        dataHeap.set(new Uint8Array(data.buffer));

        let mat = new cv.Mat([8, 1], cv.CV_8UC1, dataPtr, 0);


        let unsignedCharView = new Uint8Array(data.buffer),
        charView = new Int8Array(data.buffer),
        shortView = new Int16Array(data.buffer),
        unsignedShortView = new Uint16Array(data.buffer),
        intView = new Int32Array(data.buffer),
        float32View = new Float32Array(data.buffer),
        float64View = new Float64Array(data.buffer);


        assert.deepEqual(unsignedCharView, mat.data());
        assert.deepEqual(charView, mat.data8S());
        assert.deepEqual(shortView, mat.data16s());
        assert.deepEqual(unsignedShortView, mat.data16u());
        assert.deepEqual(intView, mat.data32s());
        assert.deepEqual(float32View, mat.data32f());
        assert.deepEqual(float64View, mat.data64f());
    }

    // test get_uchar(i)
    {
        let data = new Uint8Array([0, 0, 0, 255, 0, 1, 2, 3]),
        dataPtr = cv._malloc(8);

        let dataHeap = new Uint8Array(cv.HEAPU8.buffer, dataPtr, 8);
        dataHeap.set(new Uint8Array(data.buffer));

        let mat = new cv.Mat([8, 1], cv.CV_8UC1, dataPtr, 0);

        assert.equal(mat.get_uchar_at(0), 0);
        assert.equal(mat.get_uchar_at(1), 0);
        assert.equal(mat.get_uchar_at(2), 0);
        assert.equal(mat.get_uchar_at(3), 255);
        assert.equal(mat.get_uchar_at(4), 0);
        assert.equal(mat.get_uchar_at(5), 1);
        assert.equal(mat.get_uchar_at(6), 2);
        assert.equal(mat.get_uchar_at(7), 3);
    }

    // test get_ushort(i)
    {
        let data = new Uint16Array([0, 1000, 65000, 255, 0, 1, 2, 3]),
            dataPtr = cv._malloc(16);

        let dataHeap = new Uint16Array(cv.HEAPU8.buffer, dataPtr, 8);
        dataHeap.set(new Uint16Array(data.buffer));

        let mat = new cv.Mat([8, 1], cv.CV_16SC1, dataPtr, 0);

        assert.equal(mat.get_ushort_at(0), 0);
        assert.equal(mat.get_ushort_at(1), 1000);
        assert.equal(mat.get_ushort_at(2), 65000);
        assert.equal(mat.get_ushort_at(3), 255);
        assert.equal(mat.get_ushort_at(4), 0);
        assert.equal(mat.get_ushort_at(5), 1);
        assert.equal(mat.get_ushort_at(6), 2);
        assert.equal(mat.get_ushort_at(7), 3);
    }

    // test get_int(i)
    {
        let data = new Int32Array([0, -1000, 65000, 255, -2000000, -1, 2, 3]),
            dataPtr = cv._malloc(32);

        let dataHeap = new Int32Array(cv.HEAPU32.buffer, dataPtr, 8);
        dataHeap.set(new Int32Array(data.buffer));

        let mat = new cv.Mat([8, 1], cv.CV_32SC1, dataPtr, 0);

        assert.equal(mat.get_int_at(0), 0);
        assert.equal(mat.get_int_at(1), -1000);
        assert.equal(mat.get_int_at(2), 65000);
        assert.equal(mat.get_int_at(3), 255);
        assert.equal(mat.get_int_at(4), -2000000);
        assert.equal(mat.get_int_at(5), -1);
        assert.equal(mat.get_int_at(6), 2);
        assert.equal(mat.get_int_at(7), 3);
    }

    // test get_float(i)
    {
        const EPSILON = 0.001;
        let data = new Float32Array([0, -10.5, 650.001, 255, -20.1, -1.2, 2, 3.5]),
            dataPtr = cv._malloc(32);

        let dataHeap = new Float32Array(cv.HEAPU32.buffer, dataPtr, 8);
        dataHeap.set(new Float32Array(data.buffer));

        let mat = new cv.Mat([8, 1], cv.CV_32FC1, dataPtr, 0);

        assert.equal(Math.abs(mat.get_float_at(0)-0)       < EPSILON, true);
        assert.equal(Math.abs(mat.get_float_at(1)+10.5)    < EPSILON, true);
        assert.equal(Math.abs(mat.get_float_at(2)-650.001) < EPSILON, true);
        assert.equal(Math.abs(mat.get_float_at(3)-255)     < EPSILON, true);
        assert.equal(Math.abs(mat.get_float_at(4)+20.1)    < EPSILON, true);
        assert.equal(Math.abs(mat.get_float_at(5)+1.2)     < EPSILON, true);
        assert.equal(Math.abs(mat.get_float_at(6)-2)       < EPSILON, true);
        assert.equal(Math.abs(mat.get_float_at(7)-3.5)     < EPSILON, true);
    }

    // test get_int(i,j)
    {
        let mat = cv.Mat.eye([3, 3], cv.CV_32SC1);

        assert.equal(mat.get_int_at(0, 0), 1);
        assert.equal(mat.get_int_at(0, 1), 0);
        assert.equal(mat.get_int_at(0, 2), 0);
        assert.equal(mat.get_int_at(1, 0), 0);
        assert.equal(mat.get_int_at(1, 1), 1);
        assert.equal(mat.get_int_at(1, 2), 0);
        assert.equal(mat.get_int_at(2, 0), 0);
        assert.equal(mat.get_int_at(2, 1), 0);
        assert.equal(mat.get_int_at(2, 2), 1);

        mat.delete();
    }

});
