# Classes and methods whitelist

core = {
    '': [
        'absdiff', 'add', 'addWeighted', 'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor', 'cartToPolar',
        'compare', 'convertScaleAbs', 'copyMakeBorder', 'countNonZero', 'determinant', 'dft', 'divide', 'eigen',
        'exp', 'flip', 'getOptimalDFTSize','gemm', 'hconcat', 'inRange', 'invert', 'kmeans', 'log', 'magnitude',
        'max', 'mean', 'meanStdDev', 'merge', 'min', 'minMaxLoc', 'mixChannels', 'multiply', 'norm', 'normalize',
        'perspectiveTransform', 'polarToCart', 'pow', 'randn', 'randu', 'reduce', 'repeat', 'rotate', 'setIdentity', 'setRNGSeed',
        'solve', 'solvePoly', 'split', 'sqrt', 'subtract', 'trace', 'transform', 'transpose', 'vconcat',
        'setLogLevel', 'getLogLevel',
        'LUT',
    ],
    'Algorithm': [],
}

imgproc = {
    '': [
        'Canny',
        'GaussianBlur',
        'Laplacian',
        'HoughLines',
        'HoughLinesP',
        'HoughCircles',
        'Scharr',
        'Sobel',
        'adaptiveThreshold',
        'approxPolyDP',
        'arcLength',
        'bilateralFilter',
        'blur',
        'boundingRect',
        'boxFilter',
        'calcBackProject',
        'calcHist',
        'circle',
        'compareHist',
        'connectedComponents',
        'connectedComponentsWithStats',
        'contourArea',
        'convexHull',
        'convexityDefects',
        'cornerHarris',
        'cornerMinEigenVal',
        'createCLAHE',
        'createLineSegmentDetector',
        'cvtColor',
        'demosaicing',
        'dilate',
        'distanceTransform',
        'distanceTransformWithLabels',
        'drawContours',
        'ellipse',
        'ellipse2Poly',
        'equalizeHist',
        'erode',
        'filter2D',
        'findContours',
        'fitEllipse',
        'fitLine',
        'floodFill',
        'getAffineTransform',
        'getPerspectiveTransform',
        'getRotationMatrix2D',
        'getStructuringElement',
        'goodFeaturesToTrack',
        'grabCut',
        #'initUndistortRectifyMap',  # 4.x: moved to calib3d
        'integral',
        'integral2',
        'isContourConvex',
        'line',
        'matchShapes',
        'matchTemplate',
        'medianBlur',
        'minAreaRect',
        'minEnclosingCircle',
        'moments',
        'morphologyEx',
        'pointPolygonTest',
        'putText',
        'pyrDown',
        'pyrUp',
        'rectangle',
        'remap',
        'resize',
        'sepFilter2D',
        'threshold',
        #'undistort',  # 4.x: moved to calib3d
        'warpAffine',
        'warpPerspective',
        'warpPolar',
        'watershed',
        'fillPoly',
        'fillConvexPoly',
        'polylines',
    ],
    'CLAHE': ['apply', 'collectGarbage', 'getClipLimit', 'getTilesGridSize', 'setClipLimit', 'setTilesGridSize'],
    'segmentation_IntelligentScissorsMB': [
        'IntelligentScissorsMB',
        'setWeights',
        'setGradientMagnitudeMaxLimit',
        'setEdgeFeatureZeroCrossingParameters',
        'setEdgeFeatureCannyParameters',
        'applyImage',
        'applyImageFeatures',
        'buildMap',
        'getContour'
    ],
}

objdetect = {'': ['getPredefinedDictionary', 'extendDictionary',
                  'drawDetectedMarkers', 'generateImageMarker', 'drawDetectedCornersCharuco',
                  'drawDetectedDiamonds'],
             'GraphicalCodeDetector': ['decode', 'detect', 'detectAndDecode', 'detectMulti', 'decodeMulti', 'detectAndDecodeMulti'],
             'QRCodeDetector': ['QRCodeDetector', 'decode', 'detect', 'detectAndDecode', 'detectMulti', 'decodeMulti', 'detectAndDecodeMulti', 'decodeCurved', 'detectAndDecodeCurved', 'setEpsX', 'setEpsY'],
             'aruco_PredefinedDictionaryType': [],
             'aruco_Dictionary': ['Dictionary', 'getDistanceToId', 'generateImageMarker', 'getByteListFromBits', 'getBitsFromByteList'],
             'aruco_Board': ['Board', 'matchImagePoints', 'generateImage'],
             'aruco_GridBoard': ['GridBoard', 'generateImage', 'getGridSize', 'getMarkerLength', 'getMarkerSeparation', 'matchImagePoints'],
             'aruco_CharucoParameters': ['CharucoParameters'],
             'aruco_CharucoBoard': ['CharucoBoard', 'generateImage', 'getChessboardCorners', 'getNearestMarkerCorners', 'checkCharucoCornersCollinear', 'matchImagePoints', 'getLegacyPattern', 'setLegacyPattern'],
             'aruco_DetectorParameters': ['DetectorParameters'],
             'aruco_RefineParameters': ['RefineParameters'],
             'aruco_ArucoDetector': ['ArucoDetector', 'detectMarkers', 'refineDetectedMarkers', 'setDictionary', 'setDetectorParameters', 'setRefineParameters'],
             'aruco_CharucoDetector': ['CharucoDetector', 'setBoard', 'setCharucoParameters', 'setDetectorParameters', 'setRefineParameters', 'detectBoard', 'detectDiamonds'],
             'QRCodeDetectorAruco_Params': ['Params'],
             'QRCodeDetectorAruco': ['QRCodeDetectorAruco', 'decode', 'detect', 'detectAndDecode', 'detectMulti', 'decodeMulti', 'detectAndDecodeMulti', 'setDetectorParameters', 'setArucoParameters'],
             'barcode_BarcodeDetector': ['BarcodeDetector', 'decode', 'detect', 'detectAndDecode', 'detectMulti', 'decodeMulti', 'detectAndDecodeMulti', 'decodeWithType', 'detectAndDecodeWithType'],
             'FaceDetectorYN': ['setInputSize', 'getInputSize', 'setScoreThreshold', 'getScoreThreshold', 'setNMSThreshold', 'getNMSThreshold',
                                'setTopK', 'getTopK', 'detect', 'create'],
}

video = {
    '': [
        'CamShift',
        'calcOpticalFlowFarneback',
        'calcOpticalFlowPyrLK',
        'createBackgroundSubtractorMOG2',
        'findTransformECC',
        'meanShift',
    ],
    'BackgroundSubtractorMOG2': ['BackgroundSubtractorMOG2', 'apply'],
    'BackgroundSubtractor': ['apply', 'getBackgroundImage'],
    # issue #21070: 'Tracker': ['init', 'update'],
    'TrackerMIL': ['create'],
    'TrackerMIL_Params': [],
}

dnn = {'dnn_Net': ['setInput', 'forward', 'setPreferableBackend','getUnconnectedOutLayersNames'],
       '': ['readNetFromCaffe', 'readNetFromTensorflow', 'readNetFromDarknet',
            'readNetFromONNX', 'readNetFromTFLite', 'readNet', 'blobFromImage']}

features = {'Feature2D': ['detect', 'compute', 'detectAndCompute', 'descriptorSize', 'descriptorType', 'defaultNorm', 'empty', 'getDefaultName'],
              'ORB': ['create', 'setMaxFeatures', 'setScaleFactor', 'setNLevels', 'setEdgeThreshold', 'setFastThreshold', 'setFirstLevel', 'setWTA_K', 'setScoreType', 'setPatchSize', 'getFastThreshold', 'getDefaultName'],
              'MSER': ['create', 'detectRegions', 'setDelta', 'getDelta', 'setMinArea', 'getMinArea', 'setMaxArea', 'getMaxArea', 'setPass2Only', 'getPass2Only', 'getDefaultName'],
              'FastFeatureDetector': ['create', 'setThreshold', 'getThreshold', 'setNonmaxSuppression', 'getNonmaxSuppression', 'setType', 'getType', 'getDefaultName'],
              'GFTTDetector': ['create', 'setMaxFeatures', 'getMaxFeatures', 'setQualityLevel', 'getQualityLevel', 'setMinDistance', 'getMinDistance', 'setBlockSize', 'getBlockSize', 'setHarrisDetector', 'getHarrisDetector', 'setK', 'getK', 'getDefaultName'],
              'SimpleBlobDetector': ['create', 'setParams', 'getParams', 'getDefaultName'],
              'SimpleBlobDetector_Params': [],
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

_3d = {
    '': [
        'findHomography',
        'calibrateCameraExtended',
        'drawFrameAxes',
        'estimateAffine2D',
        'getDefaultNewCameraMatrix',
        'initUndistortRectifyMap',
        'Rodrigues',
        'solvePnP',
        'solvePnPRansac',
        'solvePnPRefineLM',
        'projectPoints',
        'undistort',
    ],
}

calib = {
    '': [

        # cv::fisheye namespace
        'fisheye_initUndistortRectifyMap',
        'fisheye_projectPoints',
    ],
}


white_list = makeWhiteList([core, imgproc, objdetect, video, dnn, features, photo, _3d, calib])

# namespace_prefix_override['dnn'] = ''  # compatibility stuff (enabled by default)
# namespace_prefix_override['aruco'] = ''  # compatibility stuff (enabled by default)
