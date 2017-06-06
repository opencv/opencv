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

QUnit.module ("ML", {});
QUnit.test("Stat Models", function(assert) {

	// TrainData, Logistic Regression
	{
		let lr = new cv.ml_LogisticRegression();

		lr.setLearningRate(0.001);
		lr.setIterations(10);
		lr.setRegularization(cv.ml_LogisticRegression_RegKinds.REG_L2.value);
		lr.setTrainMethod(cv.ml_LogisticRegression_Methods.BATCH.value);
		lr.setMiniBatchSize(1);

		assert.equal(Math.abs(lr.getLearningRate()-0.001) < 0.0001, true);
		assert.equal(lr.getIterations(), 10);
		assert.equal(lr.getRegularization(),
		             cv.ml_LogisticRegression_RegKinds.REG_L2.value);
		assert.equal(lr.getTrainMethod(),
		             cv.ml_LogisticRegression_Methods.BATCH.value);
		assert.equal(lr.getMiniBatchSize(), 1);


		// Training
		let labelsMat = new cv.Mat(4, 1, cv.CV_32FC1),
			labelsData = labelsMat.data32f();

		labelsData.set([1, -1, -1, -1]);

		let trainingDataMat = new cv.Mat(4, 2, cv.CV_32FC1),
			trainingData = trainingDataMat.data32f();

		trainingData.set([501, 255, 501, 10, 10, 10, 255, 501]); // { {501, 10}, {255, 10}, {501, 255}, {10, 501} }

		let none = new cv.Mat();
		let trainData = new cv.ml_TrainData(trainingDataMat,
			                                cv.ml_SampleTypes.ROW_SAMPLE.value,
			                                labelsMat, none, none, none, none);

		let samples = trainData.getSamples(),
		responses = trainData.getResponses();

		assert.equal(samples.columns, responses.columns);
		assert.deepEqual(samples.data32f(), trainingData);
		assert.deepEqual(responses.data32f(), labelsData);


		let trainStatus = lr.train(trainData, 0);
		assert.equal(trainStatus, true);

		// test

		trainingDataMat.delete();
		labelsMat.delete();
		lr.delete();
	}

	// Support Vector Machine
	{
		let svm = new cv.ml_SVM(),
		type = cv.ml_SVM_Types.C_SVC.value,
		kernel = cv.ml_SVM_KernelTypes.LINEAR.value,
		gamma = 3;

		svm.setType(type);
		svm.setKernel(kernel);
		svm.setGamma(gamma);

		assert.equal(svm.getType(), type);
		assert.equal(svm.getKernelType(), kernel);
		assert.equal(svm.getGamma(), gamma);


		// Training
		let labelsMat = new cv.Mat(4, 1, cv.CV_32SC1);
		let labelsData = labelsMat.data32s();
		labelsData.set([1, -1, -1, -1]);

		let trainingDataMat = new cv.Mat(4, 2, cv.CV_32FC1);
		let trainingData = trainingDataMat.data32f();

		// { {501, 10}, {255, 10}, {501, 255}, {10, 501} }
		trainingData.set([501, 255, 501, 10, 10, 10, 255, 501]);

		let trainStatus = svm.train1(trainingDataMat,
			    		             cv.ml_SampleTypes.ROW_SAMPLE.value,
									 labelsMat);

		assert.equal(trainStatus, true);
		// Test single prediction
		{
			let sampleMat = new cv.Mat(1, 2, cv.CV_32FC1);
			let samepleData = sampleMat.data32f();  // { {1000, 1}
			samepleData[0] = 1000; samepleData[1] = 1;
			let outputs = new cv.Mat(),
			flags = 0;
			let response = svm.predict(sampleMat, outputs, flags);

			assert.deepEqual(outputs.data32f(), new Float32Array([1.0]));

			sampleMat.delete();
			outputs.delete();
		}
		// Test multiple prediction
		{
			let sampleMat = new cv.Mat(2, 2, cv.CV_32FC1);
			let samepleData = sampleMat.data32f();  // { {10000, 1}, {1, 1000},
			samepleData[0] = 10000; samepleData[1] = 1;
			samepleData[2] = 1; samepleData[3] = 1000;
			let outputs = new cv.Mat(),
			flags = 0;
			let response = svm.predict(sampleMat, outputs, flags);

			assert.deepEqual(outputs.data32f(), new Float32Array([1.0, -1.0]));
			sampleMat.delete();
			outputs.delete();
		}

		trainingDataMat.delete();
		labelsMat.delete();
		svm.delete();
	}

});
