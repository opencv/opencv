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




QUnit.module("Image Processing", {});

QUnit.test("test_imgProc", function(assert) {
 // calcHist
  {
    var aa = 1;
    var source = new cv.MatVector();
    var channels = new cv.IntVector();
    var histSize = new cv.IntVector();
    var ranges = new cv.FloatVector();

    //var vec0 = new cv.Mat.zeros([20, 20], cv.CV_8UC1);
    var vec1 = new cv.Mat.ones([20, 20], cv.CV_8UC1);
    //source.push_back(vec0);
    source.push_back(vec1);
    channels.push_back(0);
    histSize.push_back(256);
    ranges.push_back(0); ranges.push_back(256);

    let hist = new cv.Mat();
    let mask = new cv.Mat();
    let binSize = cv._malloc(4);
    let binView = new Int32Array(cv.HEAP8.buffer, binSize);
    // Or, let binView = cv.HEAP32.subarray(binSize >> 2);
    binView[0] = 10;
    // TBD
    // float **: change this parameter to vector?
    cv.calcHist(source, channels, mask, hist, histSize, ranges, false);

    // hist should contains a N X 1 arrary.
    let size = hist.size();
    assert.equal(size.size(), 2);
    assert.equal(size.get(0), 256);
    assert.equal(size.get(1), 1);

    // Do we need to verify data in histogram?
    let dataView = hist.data();

    // Free resource
    cv._free(binSize);
    mask.delete();
    hist.delete();
    source.delete();
  }

  // C++
  //   void cvtColor(InputArray, OutputArray, int, int)
  // Embind
  //   void cvtColor(const Mat &, Mat &, int, int);
  {
    let source = new cv.Mat(10, 10, cv.CV_8UC3);
    let dest = new cv.Mat();

    cv.cvtColor(source, dest, cv.ColorConversionCodes.COLOR_BGR2GRAY.value, 0);
    assert.equal(dest.channels(), 1);

    cv.cvtColor(source, dest, cv.ColorConversionCodes.COLOR_BGR2BGRA.value, 0);
    assert.equal(dest.channels(), 4);

    dest.delete();
    source.delete();
  }
  // C++
  //   void equalizeHist(InputArray, OutputArray);
  // Embind
  //   void equalizeHist(const Mat &, Mat &);
  {
    let source = new cv.Mat(10, 10, cv.CV_8UC1);
    let dest = new cv.Mat();

    cv.equalizeHist(source, dest);

    // eualizeHist changes the content of a image, but does not alter meta data
    // of it.
    assert.equal(source.channels(), dest.channels());
    assert.equal(source.type(), dest.type());
    assert.equal(source.size().size(), dest.size().size());
    // Varifiy content>

    dest.delete();
    source.delete();
  }
});

QUnit.test("test_segmentation", function(assert) {
  const THRESHOLD = 127.0;
  const THRESHOLD_MAX = 210.0;

  // C++
  //   double threshold(InputArray, OutputArray, double, double, int)
  // Embind
  //   double threshold(const Mat&, Mat&, double, double, int)
  {
    let source = new cv.Mat(1, 5, cv.CV_8UC1);
    let sourceView = source.data();
    sourceView[0] = 0;   // < threshold
    sourceView[1] = 100; // < threshold
    sourceView[2] = 200; // > threshold

    let dest = new cv.Mat();

    cv.threshold(source, dest, THRESHOLD, THRESHOLD_MAX, cv.ThresholdTypes.THRESH_BINARY.value);

    let destView = dest.data();
    assert.equal(destView[0], 0);
    assert.equal(destView[1], 0);
    assert.equal(destView[2], THRESHOLD_MAX);
  }

  // C++
  //   void adaptiveThreshold(InputArray, OutputArray, double, int, int, int, double);
  // Embind
  //   void adaptiveThreshold(const Mat &, Mat &, double, int, int, int, double);
  {
    let source = cv.Mat.zeros(1, 5, cv.CV_8UC1);
    let sourceView = source.data();
    sourceView[0] = 50;
    sourceView[1] = 150;
    sourceView[2] = 200;

    let dest = new cv.Mat();
    let C = 0;
    const block_size = 3;
    cv.adaptiveThreshold(source, dest, THRESHOLD_MAX,
        cv.AdaptiveThresholdTypes.ADAPTIVE_THRESH_MEAN_C.value,
        cv.ThresholdTypes.THRESH_BINARY.value, block_size, C);

    let destView = dest.data();
    assert.equal(destView[0], 0);
    assert.equal(destView[1], THRESHOLD_MAX);
    assert.equal(destView[2], THRESHOLD_MAX);
  }
});

QUnit.test("test_filter", function(assert) {
  // C++
  //   void blur(InputArray, OutputArray, Size ksize, Point, int);
  // Embind
  //   void blur(const Mat &, Mat &, Size ksize, Point, int);
  {
      let mat1 = cv.Mat.ones(5, 5, cv.CV_8UC3);
      let mat2 = new cv.Mat();

      cv.blur(mat1, mat2, [3, 3], [-1, -1], cv.BORDER_DEFAULT);

      // Verify result.
      let view = mat2.data();
      let size = mat2.size();
      assert.equal(mat2.channels(), 3);
      assert.equal(size.get(0), 5);
      assert.equal(size.get(1), 5);
  }
  // C++
  //  void GaussianBlur(InputArray, OutputArray, Size, double, double, int);
  // Embind
  //  void GaussianBlur(Mat &, Mat&, Size, double, double, int);
  {
      let mat1 = cv.Mat.ones(7, 7, cv.CV_8UC1);
      let mat2 = new cv.Mat();

      cv.GaussianBlur(mat1, mat2, [3, 3], 0, 0, cv.BORDER_DEFAULT);

      // Verify result.
      let view = mat2.data();
      let size = mat2.size();
      assert.equal(mat2.channels(), 1);
      assert.equal(size.get(0), 7);
      assert.equal(size.get(1), 7);
  }

  // C++
  //   void medianBlur(InputArray, OutputArray, int);
  // Embind
  //   void medianBlur(Mat &, Mat &, int);
  {
      let mat1 = cv.Mat.ones(9, 9, cv.CV_8UC3);
      let mat2 = new cv.Mat();

      cv.medianBlur(mat1, mat2, 3);

      // Verify result.
      let view = mat2.data();
      let size = mat2.size();
      assert.equal(mat2.channels(), 3);
      assert.equal(size.get(0), 9);
      assert.equal(size.get(1), 9);
  }

  // Transpose
  {
      let mat1 = cv.Mat.eye(9, 9, cv.CV_8UC3);
      let mat2 = new cv.Mat();

      cv.transpose(mat1, mat2);

      // Verify result.
      let view = mat2.data();
      let size = mat2.size();
      assert.equal(mat2.channels(), 3);
      assert.equal(size.get(0), 9);
      assert.equal(size.get(1), 9);
  }

  // C++
  //   void bilateralFilter(InputArray, OutputArray, int, double, double, int borderType);
  // Embind
  //   void bilateralFilter(Mat &, Mat &, int, double, double, int borderType);
  {
      let mat1 = cv.Mat.ones(11, 11, cv.CV_8UC3);
      let mat2 = new cv.Mat();

      cv.bilateralFilter(mat1, mat2, 3, 6, 1.5, cv.BORDER_DEFAULT);

      // Verify result.
      let view = mat2.data();
      let size = mat2.size();
      assert.equal(mat2.channels(), 3);
      assert.equal(size.get(0), 11);
      assert.equal(size.get(1), 11);

  }

  // Watershed
  {
      let mat = cv.Mat.ones(11, 11, cv.CV_8UC3);
      let out = new cv.Mat(11, 11, cv.CV_32SC1);

      cv.watershed(mat, out);

      // Verify result.
      let size = out.size();
      assert.equal(out.channels(), 1);
      assert.equal(size.get(0), 11);
      assert.equal(size.get(1), 11);
      assert.equal(out.elemSize1(), 4);

      mat.delete();
      out.delete();
  }

  // Concat
  {
      let mat = cv.Mat.ones([10, 5], cv.CV_8UC3);
      let mat2 = cv.Mat.eye([10, 5], cv.CV_8UC3);
      let mat3 = cv.Mat.eye([10, 5], cv.CV_8UC3);


      let out = new cv.Mat();
      let input = new cv.MatVector();
      input.push_back(mat);
      input.push_back(mat2);
      input.push_back(mat3);

      cv.vconcat(input, out);

      // Verify result.
      let size = out.size();
      assert.equal(out.channels(), 3);
      assert.equal(size.get(0), 30);
      assert.equal(size.get(1), 5);
      assert.equal(out.elemSize1(), 1);

      cv.hconcat(input, out);

      // Verify result.
      size = out.size();
      assert.equal(out.channels(), 3);
      assert.equal(size.get(0), 10);
      assert.equal(size.get(1), 15);
      assert.equal(out.elemSize1(), 1);

      mat.delete();
      mat2.delete();
      mat3.delete();
      out.delete();
      input.delete();
  }


  // distanceTransform variants
  {
      let mat = cv.Mat.ones(11, 11, cv.CV_8UC1);
      let out = new cv.Mat(11, 11, cv.CV_32FC1);
      let labels = new cv.Mat(11, 11, cv.CV_32FC1);
      let maskSize = 3;
      cv.distanceTransform(mat, out, cv.DistanceTypes.DIST_L2.value, maskSize, cv.CV_32F);

      // Verify result.
      let size = out.size();
      assert.equal(out.channels(), 1);
      assert.equal(size.get(0), 11);
      assert.equal(size.get(1), 11);
      assert.equal(out.elemSize1(), 4);


      cv.distanceTransformWithLabels(mat, out, labels, cv.DistanceTypes.DIST_L2.value, maskSize, cv.DistanceTransformLabelTypes.DIST_LABEL_CCOMP.value);

      // Verify result.
      size = out.size();
      assert.equal(out.channels(), 1);
      assert.equal(size.get(0), 11);
      assert.equal(size.get(1), 11);
      assert.equal(out.elemSize1(), 4);

      size = labels.size();
      assert.equal(labels.channels(), 1);
      assert.equal(size.get(0), 11);
      assert.equal(size.get(1), 11);
      assert.equal(labels.elemSize1(), 4);




      mat.delete();
      out.delete();
      labels.delete();
  }

  // Min, Max
  {
      var data1 = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
      var data2 = new Uint8Array([0, 4, 0, 8, 0, 12, 0, 16, 0]);

      var expectedMin = new Uint8Array([0, 2, 0, 4, 0, 6, 0, 8, 0]);
      var expectedMax = new Uint8Array([1, 4, 3, 8, 5, 12, 7, 16, 9]);

      var dataPtr = cv._malloc(3*3*1);
      var dataPtr2 = cv._malloc(3*3*1);

      var dataHeap = new Uint8Array(cv.HEAPU8.buffer, dataPtr, 3*3*1);
      dataHeap.set(new Uint8Array(data1.buffer));

      var dataHeap2 = new Uint8Array(cv.HEAPU8.buffer, dataPtr2, 3*3*1);
      dataHeap2.set(new Uint8Array(data2.buffer));


      let mat1 = new cv.Mat([3, 3], cv.CV_8UC1, dataPtr, 0);
      let mat2 = new cv.Mat([3, 3], cv.CV_8UC1, dataPtr2, 0);

      let mat3 = new cv.Mat();

      cv.min(mat1, mat2, mat3);
      // Verify result.
      let view = mat2.data();
      let size = mat2.size();
      assert.equal(mat2.channels(), 1);
      assert.equal(size.get(0), 3);
      assert.equal(size.get(1), 3);

      assert.deepEqual(mat3.data(), expectedMin);


      cv.max(mat1, mat2, mat3);
      // Verify result.
      view = mat2.data();
      size = mat2.size();
      assert.equal(mat2.channels(), 1);
      assert.equal(size.get(0), 3);
      assert.equal(size.get(1), 3);

      assert.deepEqual(mat3.data(), expectedMax);

      cv._free(dataPtr);
      cv._free(dataPtr2);
  }

  // Bitwise operations
  {
      var data1 = new Uint8Array([0, 1, 2, 4, 8, 16, 32, 64, 128]);
      var data2 = new Uint8Array([255, 255, 255, 255, 255, 255, 255, 255, 255]);

      var expectedAnd = new Uint8Array([0, 1, 2, 4, 8, 16, 32, 64, 128]);
      var expectedOr = new Uint8Array([255, 255, 255, 255, 255, 255, 255, 255, 255]);
      var expectedXor = new Uint8Array([255, 254, 253, 251, 247, 239, 223, 191, 127]);

      var expectedNot = new Uint8Array([255, 254, 253, 251, 247, 239, 223, 191, 127]);

      var dataPtr = cv._malloc(3*3*1);
      var dataPtr2 = cv._malloc(3*3*1);

      var dataHeap = new Uint8Array(cv.HEAPU8.buffer, dataPtr, 3*3*1);
      dataHeap.set(new Uint8Array(data1.buffer));

      var dataHeap2 = new Uint8Array(cv.HEAPU8.buffer, dataPtr2, 3*3*1);
      dataHeap2.set(new Uint8Array(data2.buffer));


      let mat1 = new cv.Mat([3, 3], cv.CV_8UC1, dataPtr, 0);
      let mat2 = new cv.Mat([3, 3], cv.CV_8UC1, dataPtr2, 0);

      let mat3 = new cv.Mat();
      let none = new cv.Mat();

      cv.bitwise_not(mat1, mat3, none);
      // Verify result.
      let view = mat3.data();
      let size = mat3.size();
      assert.equal(mat3.channels(), 1);
      assert.equal(size.get(0), 3);
      assert.equal(size.get(1), 3);

      assert.deepEqual(mat3.data(), expectedNot);

      cv.bitwise_and(mat1, mat2, mat3, none);
      // Verify result.
      view = mat3.data();
      size = mat3.size();
      assert.equal(mat3.channels(), 1);
      assert.equal(size.get(0), 3);
      assert.equal(size.get(1), 3);

      assert.deepEqual(mat3.data(), expectedAnd);


      cv.bitwise_or(mat1, mat2, mat3, none);
      // Verify result.
      view = mat3.data();
      size = mat3.size();
      assert.equal(mat3.channels(), 1);
      assert.equal(size.get(0), 3);
      assert.equal(size.get(1), 3);

      assert.deepEqual(mat3.data(), expectedOr);

      cv.bitwise_xor(mat1, mat2, mat3, none);
      // Verify result.
      size = mat3.size();
      assert.equal(mat3.channels(), 1);
      assert.equal(size.get(0), 3);
      assert.equal(size.get(1), 3);

      assert.deepEqual(mat3.data(), expectedXor);

      cv._free(dataPtr);
      cv._free(dataPtr2);
  }


  // Acc
  {
      var data1 = new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7, 8]);
      var data2 = new Uint8Array([0, 2, 4, 6, 8, 10, 12, 14, 16]);
      var data3 = new Float32Array([0, 1, 0, 1, 0, 1, 0, 1, 0]);

      // data3 += data1
      var expectedAcc = new Float32Array([0, 2, 2, 4, 4, 6, 6, 8, 8]);
      // data3 += data1*data2
      var expectedAccMul = new Float32Array([0, 3, 8, 19, 32, 51, 72, 99, 128]);
      // data3 += data1*data1
      var expectedAccSquare = new Float32Array([0, 3, 8, 15, 24, 37, 48, 63, 80]);

      var dataPtr1 = cv._malloc(3*3*1);
      var dataPtr2 = cv._malloc(3*3*1);
      var dataPtr3 = cv._malloc(3*3*4);

      var dataHeap = new Uint8Array(cv.HEAPU8.buffer, dataPtr1, 3*3*1);
      dataHeap.set(new Uint8Array(data1.buffer));
      dataHeap = new Uint8Array(cv.HEAPU8.buffer, dataPtr2, 3*3*1);
      dataHeap.set(new Uint8Array(data2.buffer));
      dataHeap = new Uint8Array(cv.HEAPU8.buffer, dataPtr3, 3*3*4);
      dataHeap.set(new Uint8Array(data3.buffer));


      let mat1 = new cv.Mat([3, 3], cv.CV_8UC1, dataPtr1, 0);
          mat2 = new cv.Mat([3, 3], cv.CV_8UC1, dataPtr2, 0),
          mat3 = new cv.Mat([3, 3], cv.CV_32FC1, dataPtr3, 0),
          none = new cv.Mat();


      let mat4 = mat3.clone();
      cv.accumulate(mat1, mat4, none);
      // Verify result.
      size = mat4.size();
      assert.equal(mat4.channels(), 1);
      assert.equal(size.get(0), 3);
      assert.equal(size.get(1), 3);

      assert.deepEqual(mat4.data32f(), expectedAcc);

      cv.accumulateProduct(mat1, mat2, mat3, none);
      // Verify result.
      size = mat3.size();
      assert.equal(mat3.channels(), 1);
      assert.equal(size.get(0), 3);
      assert.equal(size.get(1), 3);

      assert.deepEqual(mat3.data32f(), expectedAccMul);

  }
  // Arithmatic operations
  {
      var data1 = new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7, 8]);
      var data2 = new Uint8Array([0, 2, 4, 6, 8, 10, 12, 14, 16]);
      var data3 = new Uint8Array([0, 1, 0, 1, 0, 1, 0, 1, 0]);

      // |data1 - data2|
      var expectedAbsDiff = new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7, 8]);
      var expectedAdd = new Uint8Array([0, 3, 6, 9, 12, 15, 18, 21, 24]);

      // data2 += data1
      var expectedAcc = new Uint8Array([0, 3, 6, 9, 12, 15, 18, 21, 24]);
      // data3 += data1*data2
      var expectedAccMul = new Uint8Array([0, 3, 8, 19, 12, 16, 72, 99, 128]);
      // data2 += data1*data1
      var expectedAccSquare = new Uint8Array([0, 3, 8, 15, 24, 37, 48, 63, 80]);

      let alpha = 4,
          beta = -1,
          gamma = 3;
      // 4*data1 - data2 + 3
      var expectedWeightedAdd = new Uint8Array([3, 5, 7, 9, 11, 13, 15, 17, 19]);

      var dataPtr = cv._malloc(3*3*1);
      var dataPtr2 = cv._malloc(3*3*1);
      var dataPtr3 = cv._malloc(3*3*1);

      var dataHeap = new Uint8Array(cv.HEAPU8.buffer, dataPtr, 3*3*1);
      dataHeap.set(new Uint8Array(data1.buffer));
      var dataHeap2 = new Uint8Array(cv.HEAPU8.buffer, dataPtr2, 3*3*1);
      dataHeap2.set(new Uint8Array(data2.buffer));
      var dataHeap3 = new Uint8Array(cv.HEAPU8.buffer, dataPtr3, 3*3*1);
      dataHeap3.set(new Uint8Array(data3.buffer));

      let mat1 = new cv.Mat([3, 3], cv.CV_8UC1, dataPtr, 0);
      let mat2 = new cv.Mat([3, 3], cv.CV_8UC1, dataPtr2, 0);
      let mat3 = new cv.Mat([3, 3], cv.CV_8UC1, dataPtr3, 0);

      let dst = new cv.Mat();
      let none = new cv.Mat();

      cv.absdiff(mat1, mat2, dst);
      // Verify result.
      let view = dst.data();
      let size = dst.size();
      assert.equal(dst.channels(), 1);
      assert.equal(size.get(0), 3);
      assert.equal(size.get(1), 3);

      assert.deepEqual(dst.data(), expectedAbsDiff);

      cv.add(mat1, mat2, dst, none, -1);
      // Verify result.
      view = dst.data();
      size = dst.size();
      assert.equal(dst.channels(), 1);
      assert.equal(size.get(0), 3);
      assert.equal(size.get(1), 3);

      assert.deepEqual(dst.data(), expectedAdd);

      cv.addWeighted(mat1, alpha, mat2, beta, gamma, dst, -1);
      // Verify result.
      view = dst.data();
      size = dst.size();
      assert.equal(dst.channels(), 1);
      assert.equal(size.get(0), 3);
      assert.equal(size.get(1), 3);

      assert.deepEqual(dst.data(), expectedWeightedAdd);

      mat1.delete();
      mat2.delete();
      mat3.delete();
      none.delete();
  }


  // getStructuringElement, Erode, dilate
  {
      let mat1 = cv.Mat.ones([100, 100], cv.CV_8UC3);
      let mat2 = new cv.Mat();
      let size = 3;

      let element = cv.getStructuringElement(cv.MorphShapes.MORPH_RECT.value,
                                        [2*size + 1, 2*size+1 ],
                                        [size, size] );

      cv.erode(mat1, mat2, element, [-1, -1], 1, cv.BORDER_CONSTANT, cv.Scalar.all(Number.MAX_VALUE));

      // Verify result.
      let view = mat2.data();
      let matSize = mat2.size();
      assert.equal(mat2.channels(), 3);
      assert.equal(matSize.get(0), 100);
      assert.equal(matSize.get(1), 100);

      element = cv.getStructuringElement(cv.MorphShapes.MORPH_ELLIPSE.value,
                                       [2*size + 1, 2*size+1],
                                       [size, size] );
      cv.dilate(mat1, mat2, element, [-1, -1], 1, cv.BORDER_CONSTANT, cv.Scalar.all(Number.MAX_VALUE));
      // Verify result.
      view = mat2.data();
      matSize = mat2.size();
      assert.equal(mat2.channels(), 3);
      assert.equal(matSize.get(0), 100);
      assert.equal(matSize.get(1), 100);

      mat1.delete();
      mat2.delete();
      element.delete();
  }

  // Integral variants
  {
      let mat = cv.Mat.eye([100, 100], cv.CV_8UC3);
      let sum = new cv.Mat();
      let sqSum = new cv.Mat();
      let title = new cv.Mat();

      cv.integral(mat, sum, -1);

      // Verify result.
      let size = sum.size();
      assert.equal(sum.channels(), 3);
      assert.equal(size.get(0), 100+1);
      assert.equal(size.get(1), 100+1);

      cv.integral2(mat, sum, sqSum, -1, -1);
      // Verify result.
      size = sum.size();
      assert.equal(sum.channels(), 3);
      assert.equal(size.get(0), 100+1);
      assert.equal(size.get(1), 100+1);

      size = sqSum.size();
      assert.equal(sqSum.channels(), 3);
      assert.equal(size.get(0), 100+1);
      assert.equal(size.get(1), 100+1);

      cv.integral3(mat, sum, sqSum, title, -1, -1);
      // Verify result.
      size = sum.size();
      assert.equal(sum.channels(), 3);
      assert.equal(size.get(0), 100+1);
      assert.equal(size.get(1), 100+1);

      size = sqSum.size();
      assert.equal(sqSum.channels(), 3);
      assert.equal(size.get(0), 100+1);
      assert.equal(size.get(1), 100+1);

      size = title.size();
      assert.equal(title.channels(), 3);
      assert.equal(size.get(0), 100+1);
      assert.equal(size.get(1), 100+1);

      mat.delete();
      sum.delete();
      sqSum.delete();
      title.delete();
  }

  // Mean, meanSTDev
  {
      let mat = cv.Mat.eye([100, 100], cv.CV_8UC3);
      let sum = new cv.Mat();
      let sqSum = new cv.Mat();
      let title = new cv.Mat();

      cv.integral(mat, sum, -1);

      // Verify result.
      let size = sum.size();
      assert.equal(sum.channels(), 3);
      assert.equal(size.get(0), 100+1);
      assert.equal(size.get(1), 100+1);

      cv.integral2(mat, sum, sqSum, -1, -1);
      // Verify result.
      size = sum.size();
      assert.equal(sum.channels(), 3);
      assert.equal(size.get(0), 100+1);
      assert.equal(size.get(1), 100+1);

      size = sqSum.size();
      assert.equal(sqSum.channels(), 3);
      assert.equal(size.get(0), 100+1);
      assert.equal(size.get(1), 100+1);

      cv.integral3(mat, sum, sqSum, title, -1, -1);
      // Verify result.
      size = sum.size();
      assert.equal(sum.channels(), 3);
      assert.equal(size.get(0), 100+1);
      assert.equal(size.get(1), 100+1);

      size = sqSum.size();
      assert.equal(sqSum.channels(), 3);
      assert.equal(size.get(0), 100+1);
      assert.equal(size.get(1), 100+1);

      size = title.size();
      assert.equal(title.channels(), 3);
      assert.equal(size.get(0), 100+1);
      assert.equal(size.get(1), 100+1);

      mat.delete();
      sum.delete();
      sqSum.delete();
      title.delete();
  }

  // Invert
  {
      let inv1 = new cv.Mat(),
          inv2 = new cv.Mat(),
          inv3 = new cv.Mat(),
          inv4 = new cv.Mat();


      var data1 = new Float32Array([1, 0, 0,
                                    0, 1, 0,
                                    0, 0, 1]);
      var data2 = new Float32Array([0, 0, 0,
                                    0, 5, 0,
                                    0, 0, 0]);
      var data3 = new Float32Array([1, 1, 1, 0,
                                    0, 3, 1, 2,
                                    2, 3, 1, 0,
                                    1, 0, 2, 1]);
      var data4 = new Float32Array([1, 4, 5,
                                    4, 2, 2,
                                    5, 2, 2]);

      var expected1 = new Float32Array([1, 0, 0,
                                        0, 1, 0,
                                        0, 0, 1]);
      // Inverse does not exist!
      var expected2 = new Float32Array([1, 0, 0,
                                        0, 0, 0,
                                        0, 0, 1]);
      var expected3 = new Float32Array([-3, -1/2, 3/2, 1,
                                        1, 1/4, -1/4, -1/2,
                                        3, 1/4, -5/4, -1/2,
                                        -3, 0, 1, 1]);
      var expected4 = new Float32Array([0, -1, 1,
                                        -1, 23/2, -9,
                                        1, -9, 7]);

      var dataPtr1 = cv._malloc(3*3*4);
      var dataPtr2 = cv._malloc(3*3*4);
      var dataPtr3 = cv._malloc(4*4*4);
      var dataPtr4 = cv._malloc(3*3*4);

      var dataHeap = new Float32Array(cv.HEAP32.buffer, dataPtr1, 3*3);
      dataHeap.set(new Float32Array(data1.buffer));
      var dataHeap2 = new Float32Array(cv.HEAP32.buffer, dataPtr2, 3*3);
      dataHeap2.set(new Float32Array(data2.buffer));
      var dataHeap3 = new Float32Array(cv.HEAP32.buffer, dataPtr3, 4*4);
      dataHeap3.set(new Float32Array(data3.buffer));
      var dataHeap4 = new Float32Array(cv.HEAP32.buffer, dataPtr4, 3*3);
      dataHeap4.set(new Float32Array(data4.buffer));

      let mat1 = new cv.Mat([3, 3], cv.CV_32FC1, dataPtr1, 0);
      let mat2 = new cv.Mat([3, 3], cv.CV_32FC1, dataPtr2, 0);
      let mat3 = new cv.Mat([4, 4], cv.CV_32FC1, dataPtr3, 0);
      let mat4 = new cv.Mat([3, 3], cv.CV_32FC1, dataPtr4, 0);

      let dst = new cv.Mat();
      let none = new cv.Mat();

      QUnit.assert.deepEqualWithTolerance = function( value, expected, tolerance ) {
        for (i = 0 ; i < value.length; i= i+1) {
          this.pushResult( {
              result: Math.abs(value[i]-expected[i]) < tolerance,
              actual: value[i],
              expected: expected[i],
//              message: message
          } );
        }
      };


      // DECOMP_LU       = 0
      // DECOMP_SVD      = 1

      // Matrix must be symmetric
      // DECOMP_EIG      = 2
      // DECOMP_CHOLESKY = 3


      cv.invert(mat1, inv1, 0);
      // Verify result.
      let size = inv1.size();
      assert.equal(inv1.channels(), 1);
      assert.equal(size.get(0), 3);
      assert.equal(size.get(1), 3);
      assert.deepEqualWithTolerance(inv1.data32f(), expected1, 0.0001);


      cv.invert(mat2, inv2, 0);
      // Verify result.
      assert.deepEqualWithTolerance(inv3.data32f(), expected3, 0.0001);



      cv.invert(mat3, inv3, 0);
      // Verify result.
      size = inv3.size();
      assert.equal(inv3.channels(), 1);
      assert.equal(size.get(0), 4);
      assert.equal(size.get(1), 4);
      assert.deepEqualWithTolerance(inv3.data32f(), expected3, 0.0001);
      console.log(inv3.data32f());


      cv.invert(mat3, inv3, 1);
      // Verify result.
      assert.deepEqualWithTolerance(inv3.data32f(), expected3, 0.0001);

      cv.invert(mat4, inv4, 2);
      // Verify result.
      assert.deepEqualWithTolerance(inv4.data32f(), expected4, 0.0001);

      cv.invert(mat4, inv4, 3);
      // Verify result.
      assert.deepEqualWithTolerance(inv4.data32f(), expected4, 0.0001);

      mat1.delete();
      mat2.delete();
      mat3.delete();
      mat4.delete();
      inv1.delete();
      inv2.delete();
      inv3.delete();
      inv4.delete();
  }


/*
  function("Canny", select_overload<void(const cv::Mat&, cv::Mat&, double, double, int, bool)>(&Wrappers::Canny_wrapper));
  function("HoughCircles", select_overload<void(const cv::Mat&, cv::Mat&, int, double, double, double, double, int, int)>(&Wrappers::HoughCircles_wrapper));
  function("HoughLines", select_overload<void(const cv::Mat&, cv::Mat&, double, double, int, double, double, double, double)>(&Wrappers::HoughLines_wrapper));
  function("HoughLinesP", select_overload<void(const cv::Mat&, cv::Mat&, double, double, int, double, double)>(&Wrappers::HoughLinesP_wrapper));
  function("HuMoments", select_overload<void(const Moments&, cv::Mat&)>(&Wrappers::HuMoments_wrapper));
  function("LUT", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&)>(&Wrappers::LUT_wrapper));
  function("Laplacian", select_overload<void(const cv::Mat&, cv::Mat&, int, int, double, double, int)>(&Wrappers::Laplacian_wrapper));
  function("Mahalanobis", select_overload<double(const cv::Mat&, const cv::Mat&, const cv::Mat&)>(&Wrappers::Mahalanobis_wrapper));
  function("PCABackProject", select_overload<void(const cv::Mat&, const cv::Mat&, const cv::Mat&, cv::Mat&)>(&Wrappers::PCABackProject_wrapper));
  function("PCACompute", select_overload<void(const cv::Mat&, cv::Mat&, cv::Mat&, int)>(&Wrappers::PCACompute_wrapper));
  function("PCACompute1", select_overload<void(const cv::Mat&, cv::Mat&, cv::Mat&, double)>(&Wrappers::PCACompute_wrapper1));
  function("PCAProject", select_overload<void(const cv::Mat&, const cv::Mat&, const cv::Mat&, cv::Mat&)>(&Wrappers::PCAProject_wrapper));
  function("PSNR", select_overload<double(const cv::Mat&, const cv::Mat&)>(&Wrappers::PSNR_wrapper));
  function("SVBackSubst", select_overload<void(const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&, cv::Mat&)>(&Wrappers::SVBackSubst_wrapper));
  function("SVDecomp", select_overload<void(const cv::Mat&, cv::Mat&, cv::Mat&, cv::Mat&, int)>(&Wrappers::SVDecomp_wrapper));
  function("Scharr", select_overload<void(const cv::Mat&, cv::Mat&, int, int, int, double, double, int)>(&Wrappers::Scharr_wrapper));
  function("Sobel", select_overload<void(const cv::Mat&, cv::Mat&, int, int, int, int, double, double, int)>(&Wrappers::Sobel_wrapper));
  function("applyColorMap", select_overload<void(const cv::Mat&, cv::Mat&, int)>(&Wrappers::applyColorMap_wrapper));
  function("approxPolyDP", select_overload<void(const cv::Mat&, cv::Mat&, double, bool)>(&Wrappers::approxPolyDP_wrapper));
  function("arcLength", select_overload<double(const cv::Mat&, bool)>(&Wrappers::arcLength_wrapper));
  function("arrowedLine", select_overload<void(cv::Mat&, Point, Point, const Scalar&, int, int, int, double)>(&Wrappers::arrowedLine_wrapper));
  function("batchDistance", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, int, cv::Mat&, int, int, const cv::Mat&, int, bool)>(&Wrappers::batchDistance_wrapper));
  function("borderInterpolate", select_overload<int(int, int, int)>(&Wrappers::borderInterpolate_wrapper));
  function("boundingRect", select_overload<Rect(const cv::Mat&)>(&Wrappers::boundingRect_wrapper));
  function("boxFilter", select_overload<void(const cv::Mat&, cv::Mat&, int, Size, Point, bool, int)>(&Wrappers::boxFilter_wrapper));
  function("boxPoints", select_overload<void(RotatedRect, cv::Mat&)>(&Wrappers::boxPoints_wrapper));
  function("calcBackProject", select_overload<void(const std::vector<cv::Mat>&, const std::vector<int>&, const cv::Mat&, cv::Mat&, const std::vector<float>&, double)>(&Wrappers::calcBackProject_wrapper));
  function("calcCovarMatrix", select_overload<void(const cv::Mat&, cv::Mat&, cv::Mat&, int, int)>(&Wrappers::calcCovarMatrix_wrapper));
  function("cartToPolar", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, cv::Mat&, bool)>(&Wrappers::cartToPolar_wrapper));
  function("circle", select_overload<void(cv::Mat&, Point, int, const Scalar&, int, int, int)>(&Wrappers::circle_wrapper));
  function("clipLine", select_overload<bool(Rect, Point&, Point&)>(&Wrappers::clipLine_wrapper));
  function("compare", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, int)>(&Wrappers::compare_wrapper));
  function("compareHist", select_overload<double(const cv::Mat&, const cv::Mat&, int)>(&Wrappers::compareHist_wrapper));
  function("completeSymm", select_overload<void(cv::Mat&, bool)>(&Wrappers::completeSymm_wrapper));
  function("connectedComponents", select_overload<int(const cv::Mat&, cv::Mat&, int, int)>(&Wrappers::connectedComponents_wrapper));
  function("connectedComponentsWithStats", select_overload<int(const cv::Mat&, cv::Mat&, cv::Mat&, cv::Mat&, int, int)>(&Wrappers::connectedComponentsWithStats_wrapper));
  function("contourArea", select_overload<double(const cv::Mat&, bool)>(&Wrappers::contourArea_wrapper));
  function("convertMaps", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, cv::Mat&, int, bool)>(&Wrappers::convertMaps_wrapper));
  function("convertScaleAbs", select_overload<void(const cv::Mat&, cv::Mat&, double, double)>(&Wrappers::convertScaleAbs_wrapper));
  function("convexHull", select_overload<void(const cv::Mat&, cv::Mat&, bool, bool)>(&Wrappers::convexHull_wrapper));
  function("convexityDefects", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&)>(&Wrappers::convexityDefects_wrapper));
  function("copyMakeBorder", select_overload<void(const cv::Mat&, cv::Mat&, int, int, int, int, int, const Scalar&)>(&Wrappers::copyMakeBorder_wrapper));
  function("cornerEigenValsAndVecs", select_overload<void(const cv::Mat&, cv::Mat&, int, int, int)>(&Wrappers::cornerEigenValsAndVecs_wrapper));
  function("cornerHarris", select_overload<void(const cv::Mat&, cv::Mat&, int, int, double, int)>(&Wrappers::cornerHarris_wrapper));
  function("cornerMinEigenVal", select_overload<void(const cv::Mat&, cv::Mat&, int, int, int)>(&Wrappers::cornerMinEigenVal_wrapper));
  function("cornerSubPix", select_overload<void(const cv::Mat&, cv::Mat&, Size, Size, TermCriteria)>(&Wrappers::cornerSubPix_wrapper));
  function("countNonZero", select_overload<int(const cv::Mat&)>(&Wrappers::countNonZero_wrapper));
  function("createCLAHE", select_overload<Ptr<CLAHE>(double, Size)>(&Wrappers::createCLAHE_wrapper));
  function("createHanningWindow", select_overload<void(cv::Mat&, Size, int)>(&Wrappers::createHanningWindow_wrapper));
  function("createLineSegmentDetector", select_overload<Ptr<LineSegmentDetector>(int, double, double, double, double, double, double, int)>(&Wrappers::createLineSegmentDetector_wrapper));
  function("dct", select_overload<void(const cv::Mat&, cv::Mat&, int)>(&Wrappers::dct_wrapper));
  function("demosaicing", select_overload<void(const cv::Mat&, cv::Mat&, int, int)>(&Wrappers::demosaicing_wrapper));
  function("determinant", select_overload<double(const cv::Mat&)>(&Wrappers::determinant_wrapper));
  function("dft", select_overload<void(const cv::Mat&, cv::Mat&, int, int)>(&Wrappers::dft_wrapper));
  function("divide", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, double, int)>(&Wrappers::divide_wrapper));
  function("divide1", select_overload<void(double, const cv::Mat&, cv::Mat&, int)>(&Wrappers::divide_wrapper1));
  function("drawContours", select_overload<void(cv::Mat&, const std::vector<cv::Mat>&, int, const Scalar&, int, int, const cv::Mat&, int, Point)>(&Wrappers::drawContours_wrapper));
  function("eigen", select_overload<bool(const cv::Mat&, cv::Mat&, cv::Mat&)>(&Wrappers::eigen_wrapper));
  function("ellipse", select_overload<void(cv::Mat&, Point, Size, double, double, double, const Scalar&, int, int, int)>(&Wrappers::ellipse_wrapper));
  function("ellipse1", select_overload<void(cv::Mat&, const RotatedRect&, const Scalar&, int, int)>(&Wrappers::ellipse_wrapper1));
  function("ellipse2Poly", select_overload<void(Point, Size, int, int, int, int, std::vector<Point>&)>(&Wrappers::ellipse2Poly_wrapper));
  function("equalizeHist", select_overload<void(const cv::Mat&, cv::Mat&)>(&Wrappers::equalizeHist_wrapper));
  function("exp", select_overload<void(const cv::Mat&, cv::Mat&)>(&Wrappers::exp_wrapper));
  function("extractChannel", select_overload<void(const cv::Mat&, cv::Mat&, int)>(&Wrappers::extractChannel_wrapper));
  function("fillConvexPoly", select_overload<void(cv::Mat&, const cv::Mat&, const Scalar&, int, int)>(&Wrappers::fillConvexPoly_wrapper));
  function("fillPoly", select_overload<void(cv::Mat&, const std::vector<cv::Mat>&, const Scalar&, int, int, Point)>(&Wrappers::fillPoly_wrapper));
  function("filter2D", select_overload<void(const cv::Mat&, cv::Mat&, int, const cv::Mat&, Point, double, int)>(&Wrappers::filter2D_wrapper));
  function("findContours", select_overload<void(cv::Mat&, std::vector<cv::Mat>&, cv::Mat&, int, int, Point)>(&Wrappers::findContours_wrapper));
  function("findNonZero", select_overload<void(const cv::Mat&, cv::Mat&)>(&Wrappers::findNonZero_wrapper));
  function("fitEllipse", select_overload<RotatedRect(const cv::Mat&)>(&Wrappers::fitEllipse_wrapper));
  function("fitLine", select_overload<void(const cv::Mat&, cv::Mat&, int, double, double, double)>(&Wrappers::fitLine_wrapper));
  function("flip", select_overload<void(const cv::Mat&, cv::Mat&, int)>(&Wrappers::flip_wrapper));
  function("gemm", select_overload<void(const cv::Mat&, const cv::Mat&, double, const cv::Mat&, double, cv::Mat&, int)>(&Wrappers::gemm_wrapper));
  function("getAffineTransform", select_overload<Mat(const cv::Mat&, const cv::Mat&)>(&Wrappers::getAffineTransform_wrapper));
  function("getDefaultNewCameraMatrix", select_overload<Mat(const cv::Mat&, Size, bool)>(&Wrappers::getDefaultNewCameraMatrix_wrapper));
  function("getDerivKernels", select_overload<void(cv::Mat&, cv::Mat&, int, int, int, bool, int)>(&Wrappers::getDerivKernels_wrapper));
  function("getGaborKernel", select_overload<Mat(Size, double, double, double, double, double, int)>(&Wrappers::getGaborKernel_wrapper));
  function("getGaussianKernel", select_overload<Mat(int, double, int)>(&Wrappers::getGaussianKernel_wrapper));
  function("getOptimalDFTSize", select_overload<int(int)>(&Wrappers::getOptimalDFTSize_wrapper));
  function("getPerspectiveTransform", select_overload<Mat(const cv::Mat&, const cv::Mat&)>(&Wrappers::getPerspectiveTransform_wrapper));
  function("getRectSubPix", select_overload<void(const cv::Mat&, Size, Point2f, cv::Mat&, int)>(&Wrappers::getRectSubPix_wrapper));
  function("getRotationMatrix2D", select_overload<Mat(Point2f, double, double)>(&Wrappers::getRotationMatrix2D_wrapper));
  function("getTextSize", select_overload<Size(const String&, int, double, int, int*)>(&Wrappers::getTextSize_wrapper), allow_raw_pointers());
  function("goodFeaturesToTrack", select_overload<void(const cv::Mat&, cv::Mat&, int, double, double, const cv::Mat&, int, bool, double)>(&Wrappers::goodFeaturesToTrack_wrapper));
  function("grabCut", select_overload<void(const cv::Mat&, cv::Mat&, Rect, cv::Mat&, cv::Mat&, int, int)>(&Wrappers::grabCut_wrapper));
  function("idct", select_overload<void(const cv::Mat&, cv::Mat&, int)>(&Wrappers::idct_wrapper));
  function("idft", select_overload<void(const cv::Mat&, cv::Mat&, int, int)>(&Wrappers::idft_wrapper));
  function("inRange", select_overload<void(const cv::Mat&, const cv::Mat&, const cv::Mat&, cv::Mat&)>(&Wrappers::inRange_wrapper));
  function("initUndistortRectifyMap", select_overload<void(const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&, Size, int, cv::Mat&, cv::Mat&)>(&Wrappers::initUndistortRectifyMap_wrapper));
  function("initWideAngleProjMap", select_overload<float(const cv::Mat&, const cv::Mat&, Size, int, int, cv::Mat&, cv::Mat&, int, double)>(&Wrappers::initWideAngleProjMap_wrapper));
  function("insertChannel", select_overload<void(const cv::Mat&, cv::Mat&, int)>(&Wrappers::insertChannel_wrapper));
  function("intersectConvexConvex", select_overload<float(const cv::Mat&, const cv::Mat&, cv::Mat&, bool)>(&Wrappers::intersectConvexConvex_wrapper));
  function("invertAffineTransform", select_overload<void(const cv::Mat&, cv::Mat&)>(&Wrappers::invertAffineTransform_wrapper));
  function("isContourConvex", select_overload<bool(const cv::Mat&)>(&Wrappers::isContourConvex_wrapper));
  function("kmeans", select_overload<double(const cv::Mat&, int, cv::Mat&, TermCriteria, int, int, cv::Mat&)>(&Wrappers::kmeans_wrapper));
  function("line", select_overload<void(cv::Mat&, Point, Point, const Scalar&, int, int, int)>(&Wrappers::line_wrapper));
  function("linearPolar", select_overload<void(const cv::Mat&, cv::Mat&, Point2f, double, int)>(&Wrappers::linearPolar_wrapper));
  function("log", select_overload<void(const cv::Mat&, cv::Mat&)>(&Wrappers::log_wrapper));
  function("logPolar", select_overload<void(const cv::Mat&, cv::Mat&, Point2f, double, int)>(&Wrappers::logPolar_wrapper));
  function("magnitude", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&)>(&Wrappers::magnitude_wrapper));
  function("matchShapes", select_overload<double(const cv::Mat&, const cv::Mat&, int, double)>(&Wrappers::matchShapes_wrapper));
  function("matchTemplate", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, int, const cv::Mat&)>(&Wrappers::matchTemplate_wrapper));
  function("mean", select_overload<Scalar(const cv::Mat&, const cv::Mat&)>(&Wrappers::mean_wrapper));
  function("meanStdDev", select_overload<void(const cv::Mat&, cv::Mat&, cv::Mat&, const cv::Mat&)>(&Wrappers::meanStdDev_wrapper));
  function("merge", select_overload<void(const std::vector<cv::Mat>&, cv::Mat&)>(&Wrappers::merge_wrapper));
  function("minAreaRect", select_overload<RotatedRect(const cv::Mat&)>(&Wrappers::minAreaRect_wrapper));
  function("minEnclosingTriangle", select_overload<double(const cv::Mat&, cv::Mat&)>(&Wrappers::minEnclosingTriangle_wrapper));
  function("mixChannels", select_overload<void(const std::vector<cv::Mat>&, InputOutputArrayOfArrays, const std::vector<int>&)>(&Wrappers::mixChannels_wrapper));
  function("moments", select_overload<Moments(const cv::Mat&, bool)>(&Wrappers::moments_wrapper));
  function("morphologyEx", select_overload<void(const cv::Mat&, cv::Mat&, int, const cv::Mat&, Point, int, int, const Scalar&)>(&Wrappers::morphologyEx_wrapper));
  function("mulSpectrums", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, int, bool)>(&Wrappers::mulSpectrums_wrapper));
  function("mulTransposed", select_overload<void(const cv::Mat&, cv::Mat&, bool, const cv::Mat&, double, int)>(&Wrappers::mulTransposed_wrapper));
  function("multiply", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, double, int)>(&Wrappers::multiply_wrapper));
  function("norm", select_overload<double(const cv::Mat&, int, const cv::Mat&)>(&Wrappers::norm_wrapper));
  function("norm1", select_overload<double(const cv::Mat&, const cv::Mat&, int, const cv::Mat&)>(&Wrappers::norm_wrapper1));
  function("normalize", select_overload<void(const cv::Mat&, cv::Mat&, double, double, int, int, const cv::Mat&)>(&Wrappers::normalize_wrapper));
  function("patchNaNs", select_overload<void(cv::Mat&, double)>(&Wrappers::patchNaNs_wrapper));
  function("perspectiveTransform", select_overload<void(const cv::Mat&, cv::Mat&, const cv::Mat&)>(&Wrappers::perspectiveTransform_wrapper));
  function("phase", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, bool)>(&Wrappers::phase_wrapper));
  function("pointPolygonTest", select_overload<double(const cv::Mat&, Point2f, bool)>(&Wrappers::pointPolygonTest_wrapper));
  function("polarToCart", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, cv::Mat&, bool)>(&Wrappers::polarToCart_wrapper));
  function("polylines", select_overload<void(cv::Mat&, const std::vector<cv::Mat>&, bool, const Scalar&, int, int, int)>(&Wrappers::polylines_wrapper));
  function("pow", select_overload<void(const cv::Mat&, double, cv::Mat&)>(&Wrappers::pow_wrapper));
  function("preCornerDetect", select_overload<void(const cv::Mat&, cv::Mat&, int, int)>(&Wrappers::preCornerDetect_wrapper));
  function("putText", select_overload<void(cv::Mat&, const String&, Point, int, double, Scalar, int, int, bool)>(&Wrappers::putText_wrapper));
  function("pyrDown", select_overload<void(const cv::Mat&, cv::Mat&, const Size&, int)>(&Wrappers::pyrDown_wrapper));
  function("pyrMeanShiftFiltering", select_overload<void(const cv::Mat&, cv::Mat&, double, double, int, TermCriteria)>(&Wrappers::pyrMeanShiftFiltering_wrapper));
  function("pyrUp", select_overload<void(const cv::Mat&, cv::Mat&, const Size&, int)>(&Wrappers::pyrUp_wrapper));
  function("randn", select_overload<void(cv::Mat&, const cv::Mat&, const cv::Mat&)>(&Wrappers::randn_wrapper));
  function("randu", select_overload<void(cv::Mat&, const cv::Mat&, const cv::Mat&)>(&Wrappers::randu_wrapper));
  function("rectangle", select_overload<void(cv::Mat&, Point, Point, const Scalar&, int, int, int)>(&Wrappers::rectangle_wrapper));
  function("reduce", select_overload<void(const cv::Mat&, cv::Mat&, int, int, int)>(&Wrappers::reduce_wrapper));
  function("remap", select_overload<void(const cv::Mat&, cv::Mat&, const cv::Mat&, const cv::Mat&, int, int, const Scalar&)>(&Wrappers::remap_wrapper));
  function("repeat", select_overload<void(const cv::Mat&, int, int, cv::Mat&)>(&Wrappers::repeat_wrapper));
  function("resize", select_overload<void(const cv::Mat&, cv::Mat&, Size, double, double, int)>(&Wrappers::resize_wrapper));
  function("rotatedRectangleIntersection", select_overload<int(const RotatedRect&, const RotatedRect&, cv::Mat&)>(&Wrappers::rotatedRectangleIntersection_wrapper));
  function("scaleAdd", select_overload<void(const cv::Mat&, double, const cv::Mat&, cv::Mat&)>(&Wrappers::scaleAdd_wrapper));
  function("sepFilter2D", select_overload<void(const cv::Mat&, cv::Mat&, int, const cv::Mat&, const cv::Mat&, Point, double, int)>(&Wrappers::sepFilter2D_wrapper));
  function("setIdentity", select_overload<void(cv::Mat&, const Scalar&)>(&Wrappers::setIdentity_wrapper));
  function("solve", select_overload<bool(const cv::Mat&, const cv::Mat&, cv::Mat&, int)>(&Wrappers::solve_wrapper));
  function("solveCubic", select_overload<int(const cv::Mat&, cv::Mat&)>(&Wrappers::solveCubic_wrapper));
  function("solvePoly", select_overload<double(const cv::Mat&, cv::Mat&, int)>(&Wrappers::solvePoly_wrapper));
  function("sort", select_overload<void(const cv::Mat&, cv::Mat&, int)>(&Wrappers::sort_wrapper));
  function("sortIdx", select_overload<void(const cv::Mat&, cv::Mat&, int)>(&Wrappers::sortIdx_wrapper));
  function("split", select_overload<void(const cv::Mat&, std::vector<cv::Mat>&)>(&Wrappers::split_wrapper));
  function("sqrBoxFilter", select_overload<void(const cv::Mat&, cv::Mat&, int, Size, Point, bool, int)>(&Wrappers::sqrBoxFilter_wrapper));
  function("sqrt", select_overload<void(const cv::Mat&, cv::Mat&)>(&Wrappers::sqrt_wrapper));
  function("subtract", select_overload<void(const cv::Mat&, const cv::Mat&, cv::Mat&, const cv::Mat&, int)>(&Wrappers::subtract_wrapper));
  function("sumElems", select_overload<Scalar(const cv::Mat&)>(&Wrappers::sum_wrapper));
  function("trace", select_overload<Scalar(const cv::Mat&)>(&Wrappers::trace_wrapper));
  function("transform", select_overload<void(const cv::Mat&, cv::Mat&, const cv::Mat&)>(&Wrappers::transform_wrapper));
  function("transpose", select_overload<void(const cv::Mat&, cv::Mat&)>(&Wrappers::transpose_wrapper));
  function("undistort", select_overload<void(const cv::Mat&, cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&)>(&Wrappers::undistort_wrapper));
  function("undistortPoints", select_overload<void(const cv::Mat&, cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&)>(&Wrappers::undistortPoints_wrapper));
  function("warpAffine", select_overload<void(const cv::Mat&, cv::Mat&, const cv::Mat&, Size, int, int, const Scalar&)>(&Wrappers::warpAffine_wrapper));
  function("warpPerspective", select_overload<void(const cv::Mat&, cv::Mat&, const cv::Mat&, Size, int, int, const Scalar&)>(&Wrappers::warpPerspective_wrapper));
  function("finish", select_overload<void()>(&Wrappers::finish_wrapper));
*/


});
