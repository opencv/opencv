# Classes and methods whitelist
core = {'': ['absdiff', 'add', 'addWeighted', 'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor', 'cartToPolar',\
             'compare', 'convertScaleAbs', 'copyMakeBorder', 'countNonZero', 'determinant', 'dft', 'divide', 'eigen', \
             'exp', 'flip', 'getOptimalDFTSize','gemm', 'hconcat', 'inRange', 'invert', 'kmeans', 'log', 'magnitude', \
             'max', 'mean', 'meanStdDev', 'merge', 'min', 'minMaxLoc', 'mixChannels', 'multiply', 'norm', 'normalize', \
             'perspectiveTransform', 'polarToCart', 'pow', 'randn', 'randu', 'reduce', 'repeat', 'rotate', 'setIdentity', 'setRNGSeed', \
             'solve', 'solvePoly', 'split', 'sqrt', 'subtract', 'trace', 'transform', 'transpose', 'vconcat'],
        'Algorithm': []}

imgproc = {'': ['Canny', 'GaussianBlur', 'Laplacian', 'HoughLines', 'HoughLinesP', 'HoughCircles', 'Scharr','Sobel', \
                'adaptiveThreshold','approxPolyDP','arcLength','bilateralFilter','blur','boundingRect','boxFilter',\
                'calcBackProject','calcHist','circle','compareHist','connectedComponents','connectedComponentsWithStats', \
                'contourArea', 'convexHull', 'convexityDefects', 'cornerHarris','cornerMinEigenVal','createCLAHE', \
                'createLineSegmentDetector','cvtColor','demosaicing','dilate', 'distanceTransform','distanceTransformWithLabels', \
                'drawContours','ellipse','ellipse2Poly','equalizeHist','erode', 'filter2D', 'findContours','fitEllipse', \
                'fitLine', 'floodFill','getAffineTransform', 'getPerspectiveTransform', 'getRotationMatrix2D', 'getStructuringElement', \
                'goodFeaturesToTrack','grabCut','initUndistortRectifyMap', 'integral','integral2', 'isContourConvex', 'line', \
                'matchShapes', 'matchTemplate','medianBlur', 'minAreaRect', 'minEnclosingCircle', 'moments', 'morphologyEx', \
                'pointPolygonTest', 'putText','pyrDown','pyrUp','rectangle','remap', 'resize','sepFilter2D','threshold', \
                'undistort','warpAffine','warpPerspective','warpPolar','watershed', \
                'fillPoly', 'fillConvexPoly'],
           'CLAHE': ['apply', 'collectGarbage', 'getClipLimit', 'getTilesGridSize', 'setClipLimit', 'setTilesGridSize']}

objdetect = {'': ['groupRectangles'],
             'HOGDescriptor': ['load', 'HOGDescriptor', 'getDefaultPeopleDetector', 'getDaimlerPeopleDetector', 'setSVMDetector', 'detectMultiScale'],
             'CascadeClassifier': ['load', 'detectMultiScale2', 'CascadeClassifier', 'detectMultiScale3', 'empty', 'detectMultiScale']}

video = {'': ['CamShift', 'calcOpticalFlowFarneback', 'calcOpticalFlowPyrLK', 'createBackgroundSubtractorMOG2', \
             'findTransformECC', 'meanShift'],
         'BackgroundSubtractorMOG2': ['BackgroundSubtractorMOG2', 'apply'],
         'BackgroundSubtractor': ['apply', 'getBackgroundImage']}

dnn = {'dnn_Net': ['setInput', 'forward'],
       '': ['readNetFromCaffe', 'readNetFromTensorflow', 'readNetFromTorch', 'readNetFromDarknet',
            'readNetFromONNX', 'readNet', 'blobFromImage']}

features2d = {'Feature2D': ['detect', 'compute', 'detectAndCompute', 'descriptorSize', 'descriptorType', 'defaultNorm', 'empty', 'getDefaultName'],
              'BRISK': ['create', 'getDefaultName'],
              'ORB': ['create', 'setMaxFeatures', 'setScaleFactor', 'setNLevels', 'setEdgeThreshold', 'setFirstLevel', 'setWTA_K', 'setScoreType', 'setPatchSize', 'getFastThreshold', 'getDefaultName'],
              'MSER': ['create', 'detectRegions', 'setDelta', 'getDelta', 'setMinArea', 'getMinArea', 'setMaxArea', 'getMaxArea', 'setPass2Only', 'getPass2Only', 'getDefaultName'],
              'FastFeatureDetector': ['create', 'setThreshold', 'getThreshold', 'setNonmaxSuppression', 'getNonmaxSuppression', 'setType', 'getType', 'getDefaultName'],
              'AgastFeatureDetector': ['create', 'setThreshold', 'getThreshold', 'setNonmaxSuppression', 'getNonmaxSuppression', 'setType', 'getType', 'getDefaultName'],
              'GFTTDetector': ['create', 'setMaxFeatures', 'getMaxFeatures', 'setQualityLevel', 'getQualityLevel', 'setMinDistance', 'getMinDistance', 'setBlockSize', 'getBlockSize', 'setHarrisDetector', 'getHarrisDetector', 'setK', 'getK', 'getDefaultName'],
              # 'SimpleBlobDetector': ['create'],
              'KAZE': ['create', 'setExtended', 'getExtended', 'setUpright', 'getUpright', 'setThreshold', 'getThreshold', 'setNOctaves', 'getNOctaves', 'setNOctaveLayers', 'getNOctaveLayers', 'setDiffusivity', 'getDiffusivity', 'getDefaultName'],
              'AKAZE': ['create', 'setDescriptorType', 'getDescriptorType', 'setDescriptorSize', 'getDescriptorSize', 'setDescriptorChannels', 'getDescriptorChannels', 'setThreshold', 'getThreshold', 'setNOctaves', 'getNOctaves', 'setNOctaveLayers', 'getNOctaveLayers', 'setDiffusivity', 'getDiffusivity', 'getDefaultName'],
              'DescriptorMatcher': ['add', 'clear', 'empty', 'isMaskSupported', 'train', 'match', 'knnMatch', 'radiusMatch', 'clone', 'create'],
              'BFMatcher': ['isMaskSupported', 'create'],
              '': ['drawKeypoints', 'drawMatches', 'drawMatchesKnn']}

photo = {'': ['createAlignMTB', 'createCalibrateDebevec', 'createCalibrateRobertson', \
              'createMergeDebevec', 'createMergeMertens', 'createMergeRobertson', \
              'createTonemapDrago', 'createTonemapMantiuk', 'createTonemapReinhard', 'inpaint'],
        'CalibrateCRF': ['process'],
        'AlignMTB' : ['calculateShift', 'shiftMat', 'computeBitmaps', 'getMaxBits', 'setMaxBits', \
                      'getExcludeRange', 'setExcludeRange', 'getCut', 'setCut'],
        'CalibrateDebevec' : ['getLambda', 'setLambda', 'getSamples', 'setSamples', 'getRandom', 'setRandom'],
        'CalibrateRobertson' : ['getMaxIter', 'setMaxIter', 'getThreshold', 'setThreshold', 'getRadiance'],
        'MergeExposures' : ['process'],
        'MergeDebevec' : ['process'],
        'MergeMertens' : ['process', 'getContrastWeight', 'setContrastWeight', 'getSaturationWeight', \
                          'setSaturationWeight', 'getExposureWeight', 'setExposureWeight'],
        'MergeRobertson' : ['process'],
        'Tonemap' : ['process' , 'getGamma', 'setGamma'],
        'TonemapDrago' : ['getSaturation', 'setSaturation', 'getBias', 'setBias', \
                          'getSigmaColor', 'setSigmaColor', 'getSigmaSpace','setSigmaSpace'],
        'TonemapMantiuk' : ['getScale', 'setScale', 'getSaturation', 'setSaturation'],
        'TonemapReinhard' : ['getIntensity', 'setIntensity', 'getLightAdaptation', 'setLightAdaptation', \
                             'getColorAdaptation', 'setColorAdaptation']
        }

aruco = {'': ['detectMarkers', 'drawDetectedMarkers', 'drawAxis', 'estimatePoseSingleMarkers', 'estimatePoseBoard', 'estimatePoseCharucoBoard', 'interpolateCornersCharuco', 'drawDetectedCornersCharuco'],
        'aruco_Dictionary': ['get', 'drawMarker'],
        'aruco_Board': ['create'],
        'aruco_GridBoard': ['create', 'draw'],
        'aruco_CharucoBoard': ['create', 'draw'],
        }

calib3d = {'': ['findHomography', 'calibrateCameraExtended', 'drawFrameAxes', 'estimateAffine2D', 'getDefaultNewCameraMatrix', 'initUndistortRectifyMap', 'Rodrigues']}


white_list = makeWhiteList([core, imgproc, objdetect, video, dnn, features2d, photo, aruco, calib3d])
