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

QUnit.module( "Feature2D", {});
QUnit.test( "Test Feature detection/extraction", function(assert) {

	// Prepare input
	var raw_image = cv.imread("../../test/data/cv.png", 1),
	image = new cv.Mat();
	cv.cvtColor(raw_image, image, cv.ColorConversionCodes.COLOR_RGB2GRAY.value, 0);

	//BRISK
	{
		let v1 = new cv.FloatVector(),
			v2 = new cv.IntVector(),
			v3 = new cv.IntVector(),
			keyPoints = new cv.KeyPointVector(),
			descriptors = new cv.Mat(),
			mask = new cv.Mat();

		let brisk = new cv.BRISK(30, 3, 1.0);
		assert.equal(brisk.empty(), true);

		brisk.delete();

		v1.push_back(1);v1.push_back(2);v1.push_back(3);v1.push_back(4);v1.push_back(5);
		v2.push_back(1);v2.push_back(4);v2.push_back(6);v2.push_back(8);v2.push_back(10);

		brisk = new cv.BRISK(v1, v2, 5.8, 8.2, v3);
		brisk.detect(image, keyPoints, mask);

		for(var i=0; i < keyPoints.size(); i+=1) {
			var keyPoint = keyPoints.get(i);
			assert.notEqual(keyPoint.angle, -1);
			keyPoint.delete();
		}

		brisk.compute(image, keyPoints, descriptors);


		v1.delete();
		v2.delete();
		v3.delete();
		keyPoints.delete();
		descriptors.delete();
		brisk.delete();
		mask.delete();
	}

	//ORB
	{
		let numFeatures = 900,
			scaleFactor = 1.2,
			numLevels = 8,
			edgeThreshold = 31,
			firstLevel =0,
			WTA_K= 2,
			scoreType = 0, //ORB::HARRIS_SCORE
			patchSize = 31,
			fastThreshold=20,
			keyPoints = new cv.KeyPointVector(),
			descriptors = new cv.Mat(),
			mask = new cv.Mat();

		let orb = new cv.ORB(numFeatures, scaleFactor, numLevels, edgeThreshold, firstLevel,
									WTA_K, scoreType, patchSize, fastThreshold);

		orb.detect(image, keyPoints, mask);
		orb.compute(image, keyPoints, descriptors);

		keyPoints.delete();
		descriptors.delete();
		orb.delete();
		mask.delete();
	}

	//MSER
	{
		let delta=5,
			minArea=60,
			maxArea=14400,
			maxVariation=0.25,
			minDiversity=.2,
			maxEvolution=200,
			areaThreshold=1.01,
			minMargin=0.003,
			edgeBlurSize=5,
			mask = new cv.Mat(),
			keyPoints = new cv.KeyPointVector();

		let mser = new cv.MSER(delta, minArea, maxArea, maxVariation, minDiversity,
										maxEvolution, areaThreshold, minMargin, edgeBlurSize)

		mser.detect(image, keyPoints, mask);



		keyPoints.delete();
		mser.delete();
		mask.delete();
	}

	//Fast
	{
		let threshold = 10,
			nonMaxSuppression = true,
			type = 2, // FastFeatureDetector::TYPE_9_16
			keyPoints = new cv.KeyPointVector(),
			mask = new cv.Mat();

		let fast = new cv.FastFeatureDetector(threshold, nonMaxSuppression, type);

		assert.equal(fast.getThreshold(), threshold);
		assert.equal(fast.getNonmaxSuppression(), nonMaxSuppression);
		assert.equal(fast.getType(), type);

		fast.detect(image, keyPoints, mask);
		// fast.compute()
		// fast.detectAndCompute()
		keyPoints.delete();
		fast.delete();
		mask.delete();
	}
	//GFTTDetector
	{
		let maxCorners=1000,
			qualityLevel=0.01,
			minDistance=1,
			blockSize=3,
			useHarrisDetector=false,
			k=0.04,
			keyPoints = new cv.KeyPointVector(),
			mask = new cv.Mat();

		let detector = new cv.GFTTDetector(maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector, k);


		detector.detect(image, keyPoints, mask);


		keyPoints.delete();
		detector.delete();
		mask.delete();
	}
	// SimpleBlobDetector
	{
		let params = new cv.SimpleBlobDetector_Params(),
			keyPoints = new cv.KeyPointVector(),
			mask = new cv.Mat();

		/*
		thresholdStep = 10;
		minThreshold = 50;
		maxThreshold = 220;
		minRepeatability = 2;
		minDistBetweenBlobs = 10;

		filterByColor = true;
		blobColor = 0;

		filterByArea = true;
		minArea = 25;
		maxArea = 5000;

		filterByCircularity = false;
		minCircularity = 0.8f;
		maxCircularity = std::numeric_limits<float>::max();

		filterByInertia = true;
		//minInertiaRatio = 0.6;
		minInertiaRatio = 0.1f;
		maxInertiaRatio = std::numeric_limits<float>::max();

		filterByConvexity = true;
		minConvexity = 0.95f;
		maxConvexity = std::numeric_limits<float>::max();
		*/
		let detector = new cv.SimpleBlobDetector(params);

		detector.detect(image, keyPoints, mask);
		// detector.compute()
		// detector.detectAndCompute()
		keyPoints.delete();
		params.delete();
		detector.delete();
		mask.delete();
	}
	// Kaze
	{
		let extended=false,
			upright=false,
			threshold = 0.001,
			numOctaves = 4,
			numOctaveLayers = 4,
			keyPoints = new cv.KeyPointVector(),
			mask = new cv.Mat(),
			descr = new cv.Mat(),
			diffusivity = 1; // KAZE::DIFF_PM_G2

		let kaze = new cv.KAZE(extended, upright, threshold, numOctaves, numOctaveLayers, diffusivity);

		assert.equal(kaze.getExtended(), extended);
		assert.equal(kaze.getUpright(), upright);
		assert.equal(kaze.getNOctaves(), numOctaves);
		assert.equal(kaze.getNOctaveLayers(), numOctaveLayers);
		assert.equal(kaze.getDiffusivity(), diffusivity);

		kaze.detectAndCompute(image, mask, keyPoints, descr, false);

		keyPoints.delete();
		kaze.delete();
		mask.delete();
	}
	//AKAZE
	{
		let akaze = new cv.AKAZE(5, 0, 3, 0.001, 4, 4, 1),
			keyPoints = new cv.KeyPointVector(),
			mask = new cv.Mat();

		assert.equal(akaze.getDescriptorType(),5);
		assert.equal(akaze.getDescriptorSize(), 0);
		assert.equal(akaze.getDescriptorChannels(), 3);
		assert.equal(Math.abs(akaze.getThreshold()-0.001) < 0.0001, true);
		assert.equal(akaze.getNOctaves(), 4);
		assert.equal(akaze.getNOctaveLayers(), 4);
		assert.equal(akaze.getDiffusivity(), 1);


		//TODO akaze.detect(image, keyPoints, mask);


		keyPoints.delete();
		akaze.delete();
		mask.delete();
		}

		raw_image.delete();
		image.delete();
	});

	QUnit.test("Descriptor matcher", function(assert) {
	// DescriptorMatcher
	{
		// Types could be:
		// `BruteForce` (it uses L2 )
		// `BruteForce-L1`
		// `BruteForce-Hamming`
		// `BruteForce-Hamming(2)`
		// `FlannBased`
		let type = "BruteForce";
		let matcher = new cv.DescriptorMatcher(type);

		assert.equal(matcher.empty(), true);

		//matcher.add(descriptors);


		//matcher.train();
		//matcher.match();
		//matcher.knnMatch();


		matcher.delete();
	}
	// Brute Force Matcher
	{
		let bfmatcher = new cv.BFMatcher(4, false);

		assert.equal(bfmatcher.isMaskSupported(), true)

		bfmatcher.delete();

	}
	// TODO: Flann Based Matcher
});

QUnit.test("Bag of visual words", function(assert) {
	// BOWKMeansTrainer
	{
		let criteria = new cv.TermCriteria(),
			bowkmeans = new cv.BOWKMeansTrainer(3, criteria, 3, 2);


		assert.equal(bowkmeans.descriptorsCount(), 0);

		let descriptors1 = new cv.Mat(),
			descriptors2 = new cv.Mat();

		//bowkmeans.add(descriptors1);
		//bowkmeans.add(descriptors2);

		assert.equal(bowkmeans.descriptorsCount(), 0);

		let descriptorsVector = bowkmeans.getDescriptors();
		assert.equal(descriptorsVector.size(), 0);

		//let voc = new cv.Mat();

		//bowkmeans.cluster();


		descriptors1.delete();
		descriptors2.delete();
		descriptorsVector.delete();
		criteria.delete();
		bowkmeans.delete();
	}
	// BOWImgDescriptorExtractor
	{

	}
});
