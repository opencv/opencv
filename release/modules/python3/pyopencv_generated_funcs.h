static PyObject* pyopencv_cv_AKAZE_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int descriptor_type=AKAZE::DESCRIPTOR_MLDB;
    int descriptor_size=0;
    int descriptor_channels=3;
    float threshold=0.001f;
    int nOctaves=4;
    int nOctaveLayers=4;
    int diffusivity=KAZE::DIFF_PM_G2;
    Ptr<AKAZE> retval;

    const char* keywords[] = { "descriptor_type", "descriptor_size", "descriptor_channels", "threshold", "nOctaves", "nOctaveLayers", "diffusivity", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|iiifiii:AKAZE_create", (char**)keywords, &descriptor_type, &descriptor_size, &descriptor_channels, &threshold, &nOctaves, &nOctaveLayers, &diffusivity) )
    {
        ERRWRAP2(retval = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves, nOctaveLayers, diffusivity));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_AgastFeatureDetector_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int threshold=10;
    bool nonmaxSuppression=true;
    int type=AgastFeatureDetector::OAST_9_16;
    Ptr<AgastFeatureDetector> retval;

    const char* keywords[] = { "threshold", "nonmaxSuppression", "type", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|ibi:AgastFeatureDetector_create", (char**)keywords, &threshold, &nonmaxSuppression, &type) )
    {
        ERRWRAP2(retval = cv::AgastFeatureDetector::create(threshold, nonmaxSuppression, type));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_BFMatcher_BFMatcher(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int normType=NORM_L2;
    bool crossCheck=false;
    pyopencv_BFMatcher_t* self = 0;

    const char* keywords[] = { "normType", "crossCheck", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|ib:BFMatcher", (char**)keywords, &normType, &crossCheck) )
    {
        self = PyObject_NEW(pyopencv_BFMatcher_t, &pyopencv_BFMatcher_Type);
        new (&(self->v)) Ptr<cv::BFMatcher>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::BFMatcher(normType, crossCheck)));
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_cv_BFMatcher_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int normType=NORM_L2;
    bool crossCheck=false;
    Ptr<BFMatcher> retval;

    const char* keywords[] = { "normType", "crossCheck", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|ib:BFMatcher_create", (char**)keywords, &normType, &crossCheck) )
    {
        ERRWRAP2(retval = cv::BFMatcher::create(normType, crossCheck));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_BOWImgDescriptorExtractor_BOWImgDescriptorExtractor(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_dextractor = NULL;
    Ptr<DescriptorExtractor> dextractor;
    PyObject* pyobj_dmatcher = NULL;
    Ptr<DescriptorMatcher> dmatcher;
    pyopencv_BOWImgDescriptorExtractor_t* self = 0;

    const char* keywords[] = { "dextractor", "dmatcher", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:BOWImgDescriptorExtractor", (char**)keywords, &pyobj_dextractor, &pyobj_dmatcher) &&
        pyopencv_to(pyobj_dextractor, dextractor, ArgInfo("dextractor", 0)) &&
        pyopencv_to(pyobj_dmatcher, dmatcher, ArgInfo("dmatcher", 0)) )
    {
        self = PyObject_NEW(pyopencv_BOWImgDescriptorExtractor_t, &pyopencv_BOWImgDescriptorExtractor_Type);
        new (&(self->v)) Ptr<cv::BOWImgDescriptorExtractor>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::BOWImgDescriptorExtractor(dextractor, dmatcher)));
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_cv_BOWKMeansTrainer_BOWKMeansTrainer(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int clusterCount=0;
    PyObject* pyobj_termcrit = NULL;
    TermCriteria termcrit;
    int attempts=3;
    int flags=KMEANS_PP_CENTERS;
    pyopencv_BOWKMeansTrainer_t* self = 0;

    const char* keywords[] = { "clusterCount", "termcrit", "attempts", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i|Oii:BOWKMeansTrainer", (char**)keywords, &clusterCount, &pyobj_termcrit, &attempts, &flags) &&
        pyopencv_to(pyobj_termcrit, termcrit, ArgInfo("termcrit", 0)) )
    {
        self = PyObject_NEW(pyopencv_BOWKMeansTrainer_t, &pyopencv_BOWKMeansTrainer_Type);
        new (&(self->v)) Ptr<cv::BOWKMeansTrainer>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::BOWKMeansTrainer(clusterCount, termcrit, attempts, flags)));
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_cv_BRISK_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    int thresh=30;
    int octaves=3;
    float patternScale=1.0f;
    Ptr<BRISK> retval;

    const char* keywords[] = { "thresh", "octaves", "patternScale", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|iif:BRISK_create", (char**)keywords, &thresh, &octaves, &patternScale) )
    {
        ERRWRAP2(retval = cv::BRISK::create(thresh, octaves, patternScale));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_radiusList = NULL;
    vector_float radiusList;
    PyObject* pyobj_numberList = NULL;
    vector_int numberList;
    float dMax=5.85f;
    float dMin=8.2f;
    PyObject* pyobj_indexChange = NULL;
    vector_int indexChange=std::vector<int>();
    Ptr<BRISK> retval;

    const char* keywords[] = { "radiusList", "numberList", "dMax", "dMin", "indexChange", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|ffO:BRISK_create", (char**)keywords, &pyobj_radiusList, &pyobj_numberList, &dMax, &dMin, &pyobj_indexChange) &&
        pyopencv_to(pyobj_radiusList, radiusList, ArgInfo("radiusList", 0)) &&
        pyopencv_to(pyobj_numberList, numberList, ArgInfo("numberList", 0)) &&
        pyopencv_to(pyobj_indexChange, indexChange, ArgInfo("indexChange", 0)) )
    {
        ERRWRAP2(retval = cv::BRISK::create(radiusList, numberList, dMax, dMin, indexChange));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_CamShift(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_probImage = NULL;
    Mat probImage;
    PyObject* pyobj_window = NULL;
    Rect window;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria;
    RotatedRect retval;

    const char* keywords[] = { "probImage", "window", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:CamShift", (char**)keywords, &pyobj_probImage, &pyobj_window, &pyobj_criteria) &&
        pyopencv_to(pyobj_probImage, probImage, ArgInfo("probImage", 0)) &&
        pyopencv_to(pyobj_window, window, ArgInfo("window", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(retval = cv::CamShift(probImage, window, criteria));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(window));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_probImage = NULL;
    UMat probImage;
    PyObject* pyobj_window = NULL;
    Rect window;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria;
    RotatedRect retval;

    const char* keywords[] = { "probImage", "window", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:CamShift", (char**)keywords, &pyobj_probImage, &pyobj_window, &pyobj_criteria) &&
        pyopencv_to(pyobj_probImage, probImage, ArgInfo("probImage", 0)) &&
        pyopencv_to(pyobj_window, window, ArgInfo("window", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(retval = cv::CamShift(probImage, window, criteria));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(window));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_Canny(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_edges = NULL;
    Mat edges;
    double threshold1=0;
    double threshold2=0;
    int apertureSize=3;
    bool L2gradient=false;

    const char* keywords[] = { "image", "threshold1", "threshold2", "edges", "apertureSize", "L2gradient", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odd|Oib:Canny", (char**)keywords, &pyobj_image, &threshold1, &threshold2, &pyobj_edges, &apertureSize, &L2gradient) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_edges, edges, ArgInfo("edges", 1)) )
    {
        ERRWRAP2(cv::Canny(image, edges, threshold1, threshold2, apertureSize, L2gradient));
        return pyopencv_from(edges);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_edges = NULL;
    UMat edges;
    double threshold1=0;
    double threshold2=0;
    int apertureSize=3;
    bool L2gradient=false;

    const char* keywords[] = { "image", "threshold1", "threshold2", "edges", "apertureSize", "L2gradient", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odd|Oib:Canny", (char**)keywords, &pyobj_image, &threshold1, &threshold2, &pyobj_edges, &apertureSize, &L2gradient) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_edges, edges, ArgInfo("edges", 1)) )
    {
        ERRWRAP2(cv::Canny(image, edges, threshold1, threshold2, apertureSize, L2gradient));
        return pyopencv_from(edges);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_dx = NULL;
    Mat dx;
    PyObject* pyobj_dy = NULL;
    Mat dy;
    PyObject* pyobj_edges = NULL;
    Mat edges;
    double threshold1=0;
    double threshold2=0;
    bool L2gradient=false;

    const char* keywords[] = { "dx", "dy", "threshold1", "threshold2", "edges", "L2gradient", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOdd|Ob:Canny", (char**)keywords, &pyobj_dx, &pyobj_dy, &threshold1, &threshold2, &pyobj_edges, &L2gradient) &&
        pyopencv_to(pyobj_dx, dx, ArgInfo("dx", 0)) &&
        pyopencv_to(pyobj_dy, dy, ArgInfo("dy", 0)) &&
        pyopencv_to(pyobj_edges, edges, ArgInfo("edges", 1)) )
    {
        ERRWRAP2(cv::Canny(dx, dy, edges, threshold1, threshold2, L2gradient));
        return pyopencv_from(edges);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_dx = NULL;
    UMat dx;
    PyObject* pyobj_dy = NULL;
    UMat dy;
    PyObject* pyobj_edges = NULL;
    UMat edges;
    double threshold1=0;
    double threshold2=0;
    bool L2gradient=false;

    const char* keywords[] = { "dx", "dy", "threshold1", "threshold2", "edges", "L2gradient", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOdd|Ob:Canny", (char**)keywords, &pyobj_dx, &pyobj_dy, &threshold1, &threshold2, &pyobj_edges, &L2gradient) &&
        pyopencv_to(pyobj_dx, dx, ArgInfo("dx", 0)) &&
        pyopencv_to(pyobj_dy, dy, ArgInfo("dy", 0)) &&
        pyopencv_to(pyobj_edges, edges, ArgInfo("edges", 1)) )
    {
        ERRWRAP2(cv::Canny(dx, dy, edges, threshold1, threshold2, L2gradient));
        return pyopencv_from(edges);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_CascadeClassifier_CascadeClassifier(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    pyopencv_CascadeClassifier_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_CascadeClassifier_t, &pyopencv_CascadeClassifier_Type);
        new (&(self->v)) Ptr<cv::CascadeClassifier>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::CascadeClassifier()));
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_filename = NULL;
    String filename;
    pyopencv_CascadeClassifier_t* self = 0;

    const char* keywords[] = { "filename", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:CascadeClassifier", (char**)keywords, &pyobj_filename) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) )
    {
        self = PyObject_NEW(pyopencv_CascadeClassifier_t, &pyopencv_CascadeClassifier_Type);
        new (&(self->v)) Ptr<cv::CascadeClassifier>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::CascadeClassifier(filename)));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_CascadeClassifier_convert(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_oldcascade = NULL;
    String oldcascade;
    PyObject* pyobj_newcascade = NULL;
    String newcascade;
    bool retval;

    const char* keywords[] = { "oldcascade", "newcascade", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:CascadeClassifier_convert", (char**)keywords, &pyobj_oldcascade, &pyobj_newcascade) &&
        pyopencv_to(pyobj_oldcascade, oldcascade, ArgInfo("oldcascade", 0)) &&
        pyopencv_to(pyobj_newcascade, newcascade, ArgInfo("newcascade", 0)) )
    {
        ERRWRAP2(retval = cv::CascadeClassifier::convert(oldcascade, newcascade));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_DMatch_DMatch(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    pyopencv_DMatch_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_DMatch_t, &pyopencv_DMatch_Type);
        if(self) ERRWRAP2(new (&(self->v)) cv::DMatch());
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    int _queryIdx=0;
    int _trainIdx=0;
    float _distance=0.f;
    pyopencv_DMatch_t* self = 0;

    const char* keywords[] = { "_queryIdx", "_trainIdx", "_distance", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "iif:DMatch", (char**)keywords, &_queryIdx, &_trainIdx, &_distance) )
    {
        self = PyObject_NEW(pyopencv_DMatch_t, &pyopencv_DMatch_Type);
        if(self) ERRWRAP2(new (&(self->v)) cv::DMatch(_queryIdx, _trainIdx, _distance));
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    int _queryIdx=0;
    int _trainIdx=0;
    int _imgIdx=0;
    float _distance=0.f;
    pyopencv_DMatch_t* self = 0;

    const char* keywords[] = { "_queryIdx", "_trainIdx", "_imgIdx", "_distance", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "iiif:DMatch", (char**)keywords, &_queryIdx, &_trainIdx, &_imgIdx, &_distance) )
    {
        self = PyObject_NEW(pyopencv_DMatch_t, &pyopencv_DMatch_Type);
        if(self) ERRWRAP2(new (&(self->v)) cv::DMatch(_queryIdx, _trainIdx, _imgIdx, _distance));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_DescriptorMatcher_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_descriptorMatcherType = NULL;
    String descriptorMatcherType;
    Ptr<DescriptorMatcher> retval;

    const char* keywords[] = { "descriptorMatcherType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:DescriptorMatcher_create", (char**)keywords, &pyobj_descriptorMatcherType) &&
        pyopencv_to(pyobj_descriptorMatcherType, descriptorMatcherType, ArgInfo("descriptorMatcherType", 0)) )
    {
        ERRWRAP2(retval = cv::DescriptorMatcher::create(descriptorMatcherType));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    int matcherType=0;
    Ptr<DescriptorMatcher> retval;

    const char* keywords[] = { "matcherType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:DescriptorMatcher_create", (char**)keywords, &matcherType) )
    {
        ERRWRAP2(retval = cv::DescriptorMatcher::create(matcherType));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_DualTVL1OpticalFlow_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    double tau=0.25;
    double lambda=0.15;
    double theta=0.3;
    int nscales=5;
    int warps=5;
    double epsilon=0.01;
    int innnerIterations=30;
    int outerIterations=10;
    double scaleStep=0.8;
    double gamma=0.0;
    int medianFiltering=5;
    bool useInitialFlow=false;
    Ptr<DualTVL1OpticalFlow> retval;

    const char* keywords[] = { "tau", "lambda", "theta", "nscales", "warps", "epsilon", "innnerIterations", "outerIterations", "scaleStep", "gamma", "medianFiltering", "useInitialFlow", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|dddiidiiddib:DualTVL1OpticalFlow_create", (char**)keywords, &tau, &lambda, &theta, &nscales, &warps, &epsilon, &innnerIterations, &outerIterations, &scaleStep, &gamma, &medianFiltering, &useInitialFlow) )
    {
        ERRWRAP2(retval = cv::DualTVL1OpticalFlow::create(tau, lambda, theta, nscales, warps, epsilon, innnerIterations, outerIterations, scaleStep, gamma, medianFiltering, useInitialFlow));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FarnebackOpticalFlow_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int numLevels=5;
    double pyrScale=0.5;
    bool fastPyramids=false;
    int winSize=13;
    int numIters=10;
    int polyN=5;
    double polySigma=1.1;
    int flags=0;
    Ptr<FarnebackOpticalFlow> retval;

    const char* keywords[] = { "numLevels", "pyrScale", "fastPyramids", "winSize", "numIters", "polyN", "polySigma", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|idbiiidi:FarnebackOpticalFlow_create", (char**)keywords, &numLevels, &pyrScale, &fastPyramids, &winSize, &numIters, &polyN, &polySigma, &flags) )
    {
        ERRWRAP2(retval = cv::FarnebackOpticalFlow::create(numLevels, pyrScale, fastPyramids, winSize, numIters, polyN, polySigma, flags));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FastFeatureDetector_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int threshold=10;
    bool nonmaxSuppression=true;
    int type=FastFeatureDetector::TYPE_9_16;
    Ptr<FastFeatureDetector> retval;

    const char* keywords[] = { "threshold", "nonmaxSuppression", "type", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|ibi:FastFeatureDetector_create", (char**)keywords, &threshold, &nonmaxSuppression, &type) )
    {
        ERRWRAP2(retval = cv::FastFeatureDetector::create(threshold, nonmaxSuppression, type));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FileNode_FileNode(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    pyopencv_FileNode_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_FileNode_t, &pyopencv_FileNode_Type);
        if(self) ERRWRAP2(new (&(self->v)) cv::FileNode());
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_cv_FileStorage_FileStorage(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    pyopencv_FileStorage_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_FileStorage_t, &pyopencv_FileStorage_Type);
        new (&(self->v)) Ptr<cv::FileStorage>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::FileStorage()));
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_source = NULL;
    String source;
    int flags=0;
    PyObject* pyobj_encoding = NULL;
    String encoding;
    pyopencv_FileStorage_t* self = 0;

    const char* keywords[] = { "source", "flags", "encoding", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|O:FileStorage", (char**)keywords, &pyobj_source, &flags, &pyobj_encoding) &&
        pyopencv_to(pyobj_source, source, ArgInfo("source", 0)) &&
        pyopencv_to(pyobj_encoding, encoding, ArgInfo("encoding", 0)) )
    {
        self = PyObject_NEW(pyopencv_FileStorage_t, &pyopencv_FileStorage_Type);
        new (&(self->v)) Ptr<cv::FileStorage>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::FileStorage(source, flags, encoding)));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_FlannBasedMatcher_FlannBasedMatcher(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_indexParams = NULL;
    Ptr<flann::IndexParams> indexParams=makePtr<flann::KDTreeIndexParams>();
    PyObject* pyobj_searchParams = NULL;
    Ptr<flann::SearchParams> searchParams=makePtr<flann::SearchParams>();
    pyopencv_FlannBasedMatcher_t* self = 0;

    const char* keywords[] = { "indexParams", "searchParams", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|OO:FlannBasedMatcher", (char**)keywords, &pyobj_indexParams, &pyobj_searchParams) &&
        pyopencv_to(pyobj_indexParams, indexParams, ArgInfo("indexParams", 0)) &&
        pyopencv_to(pyobj_searchParams, searchParams, ArgInfo("searchParams", 0)) )
    {
        self = PyObject_NEW(pyopencv_FlannBasedMatcher_t, &pyopencv_FlannBasedMatcher_Type);
        new (&(self->v)) Ptr<cv::FlannBasedMatcher>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::FlannBasedMatcher(indexParams, searchParams)));
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_cv_FlannBasedMatcher_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    Ptr<FlannBasedMatcher> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::FlannBasedMatcher::create());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_GFTTDetector_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int maxCorners=1000;
    double qualityLevel=0.01;
    double minDistance=1;
    int blockSize=3;
    bool useHarrisDetector=false;
    double k=0.04;
    Ptr<GFTTDetector> retval;

    const char* keywords[] = { "maxCorners", "qualityLevel", "minDistance", "blockSize", "useHarrisDetector", "k", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|iddibd:GFTTDetector_create", (char**)keywords, &maxCorners, &qualityLevel, &minDistance, &blockSize, &useHarrisDetector, &k) )
    {
        ERRWRAP2(retval = cv::GFTTDetector::create(maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector, k));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_GaussianBlur(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_ksize = NULL;
    Size ksize;
    double sigmaX=0;
    double sigmaY=0;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "ksize", "sigmaX", "dst", "sigmaY", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOd|Odi:GaussianBlur", (char**)keywords, &pyobj_src, &pyobj_ksize, &sigmaX, &pyobj_dst, &sigmaY, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_ksize, ksize, ArgInfo("ksize", 0)) )
    {
        ERRWRAP2(cv::GaussianBlur(src, dst, ksize, sigmaX, sigmaY, borderType));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_ksize = NULL;
    Size ksize;
    double sigmaX=0;
    double sigmaY=0;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "ksize", "sigmaX", "dst", "sigmaY", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOd|Odi:GaussianBlur", (char**)keywords, &pyobj_src, &pyobj_ksize, &sigmaX, &pyobj_dst, &sigmaY, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_ksize, ksize, ArgInfo("ksize", 0)) )
    {
        ERRWRAP2(cv::GaussianBlur(src, dst, ksize, sigmaX, sigmaY, borderType));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_HOGDescriptor_HOGDescriptor(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    pyopencv_HOGDescriptor_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_HOGDescriptor_t, &pyopencv_HOGDescriptor_Type);
        new (&(self->v)) Ptr<cv::HOGDescriptor>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::HOGDescriptor()));
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj__winSize = NULL;
    Size _winSize;
    PyObject* pyobj__blockSize = NULL;
    Size _blockSize;
    PyObject* pyobj__blockStride = NULL;
    Size _blockStride;
    PyObject* pyobj__cellSize = NULL;
    Size _cellSize;
    int _nbins=0;
    int _derivAperture=1;
    double _winSigma=-1;
    int _histogramNormType=HOGDescriptor::L2Hys;
    double _L2HysThreshold=0.2;
    bool _gammaCorrection=false;
    int _nlevels=HOGDescriptor::DEFAULT_NLEVELS;
    bool _signedGradient=false;
    pyopencv_HOGDescriptor_t* self = 0;

    const char* keywords[] = { "_winSize", "_blockSize", "_blockStride", "_cellSize", "_nbins", "_derivAperture", "_winSigma", "_histogramNormType", "_L2HysThreshold", "_gammaCorrection", "_nlevels", "_signedGradient", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOi|ididbib:HOGDescriptor", (char**)keywords, &pyobj__winSize, &pyobj__blockSize, &pyobj__blockStride, &pyobj__cellSize, &_nbins, &_derivAperture, &_winSigma, &_histogramNormType, &_L2HysThreshold, &_gammaCorrection, &_nlevels, &_signedGradient) &&
        pyopencv_to(pyobj__winSize, _winSize, ArgInfo("_winSize", 0)) &&
        pyopencv_to(pyobj__blockSize, _blockSize, ArgInfo("_blockSize", 0)) &&
        pyopencv_to(pyobj__blockStride, _blockStride, ArgInfo("_blockStride", 0)) &&
        pyopencv_to(pyobj__cellSize, _cellSize, ArgInfo("_cellSize", 0)) )
    {
        self = PyObject_NEW(pyopencv_HOGDescriptor_t, &pyopencv_HOGDescriptor_Type);
        new (&(self->v)) Ptr<cv::HOGDescriptor>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::HOGDescriptor(_winSize, _blockSize, _blockStride, _cellSize, _nbins, _derivAperture, _winSigma, _histogramNormType, _L2HysThreshold, _gammaCorrection, _nlevels, _signedGradient)));
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_filename = NULL;
    String filename;
    pyopencv_HOGDescriptor_t* self = 0;

    const char* keywords[] = { "filename", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:HOGDescriptor", (char**)keywords, &pyobj_filename) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) )
    {
        self = PyObject_NEW(pyopencv_HOGDescriptor_t, &pyopencv_HOGDescriptor_Type);
        new (&(self->v)) Ptr<cv::HOGDescriptor>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::HOGDescriptor(filename)));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_HOGDescriptor_getDaimlerPeopleDetector(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    std::vector<float> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::HOGDescriptor::getDaimlerPeopleDetector());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_HOGDescriptor_getDefaultPeopleDetector(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    std::vector<float> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::HOGDescriptor::getDefaultPeopleDetector());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_HoughCircles(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_circles = NULL;
    Mat circles;
    int method=0;
    double dp=0;
    double minDist=0;
    double param1=100;
    double param2=100;
    int minRadius=0;
    int maxRadius=0;

    const char* keywords[] = { "image", "method", "dp", "minDist", "circles", "param1", "param2", "minRadius", "maxRadius", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oidd|Oddii:HoughCircles", (char**)keywords, &pyobj_image, &method, &dp, &minDist, &pyobj_circles, &param1, &param2, &minRadius, &maxRadius) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_circles, circles, ArgInfo("circles", 1)) )
    {
        ERRWRAP2(cv::HoughCircles(image, circles, method, dp, minDist, param1, param2, minRadius, maxRadius));
        return pyopencv_from(circles);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_circles = NULL;
    UMat circles;
    int method=0;
    double dp=0;
    double minDist=0;
    double param1=100;
    double param2=100;
    int minRadius=0;
    int maxRadius=0;

    const char* keywords[] = { "image", "method", "dp", "minDist", "circles", "param1", "param2", "minRadius", "maxRadius", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oidd|Oddii:HoughCircles", (char**)keywords, &pyobj_image, &method, &dp, &minDist, &pyobj_circles, &param1, &param2, &minRadius, &maxRadius) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_circles, circles, ArgInfo("circles", 1)) )
    {
        ERRWRAP2(cv::HoughCircles(image, circles, method, dp, minDist, param1, param2, minRadius, maxRadius));
        return pyopencv_from(circles);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_HoughLines(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_lines = NULL;
    Mat lines;
    double rho=0;
    double theta=0;
    int threshold=0;
    double srn=0;
    double stn=0;
    double min_theta=0;
    double max_theta=CV_PI;

    const char* keywords[] = { "image", "rho", "theta", "threshold", "lines", "srn", "stn", "min_theta", "max_theta", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oddi|Odddd:HoughLines", (char**)keywords, &pyobj_image, &rho, &theta, &threshold, &pyobj_lines, &srn, &stn, &min_theta, &max_theta) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_lines, lines, ArgInfo("lines", 1)) )
    {
        ERRWRAP2(cv::HoughLines(image, lines, rho, theta, threshold, srn, stn, min_theta, max_theta));
        return pyopencv_from(lines);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_lines = NULL;
    UMat lines;
    double rho=0;
    double theta=0;
    int threshold=0;
    double srn=0;
    double stn=0;
    double min_theta=0;
    double max_theta=CV_PI;

    const char* keywords[] = { "image", "rho", "theta", "threshold", "lines", "srn", "stn", "min_theta", "max_theta", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oddi|Odddd:HoughLines", (char**)keywords, &pyobj_image, &rho, &theta, &threshold, &pyobj_lines, &srn, &stn, &min_theta, &max_theta) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_lines, lines, ArgInfo("lines", 1)) )
    {
        ERRWRAP2(cv::HoughLines(image, lines, rho, theta, threshold, srn, stn, min_theta, max_theta));
        return pyopencv_from(lines);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_HoughLinesP(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_lines = NULL;
    Mat lines;
    double rho=0;
    double theta=0;
    int threshold=0;
    double minLineLength=0;
    double maxLineGap=0;

    const char* keywords[] = { "image", "rho", "theta", "threshold", "lines", "minLineLength", "maxLineGap", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oddi|Odd:HoughLinesP", (char**)keywords, &pyobj_image, &rho, &theta, &threshold, &pyobj_lines, &minLineLength, &maxLineGap) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_lines, lines, ArgInfo("lines", 1)) )
    {
        ERRWRAP2(cv::HoughLinesP(image, lines, rho, theta, threshold, minLineLength, maxLineGap));
        return pyopencv_from(lines);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_lines = NULL;
    UMat lines;
    double rho=0;
    double theta=0;
    int threshold=0;
    double minLineLength=0;
    double maxLineGap=0;

    const char* keywords[] = { "image", "rho", "theta", "threshold", "lines", "minLineLength", "maxLineGap", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oddi|Odd:HoughLinesP", (char**)keywords, &pyobj_image, &rho, &theta, &threshold, &pyobj_lines, &minLineLength, &maxLineGap) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_lines, lines, ArgInfo("lines", 1)) )
    {
        ERRWRAP2(cv::HoughLinesP(image, lines, rho, theta, threshold, minLineLength, maxLineGap));
        return pyopencv_from(lines);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_HuMoments(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_m = NULL;
    Moments m;
    PyObject* pyobj_hu = NULL;
    Mat hu;

    const char* keywords[] = { "m", "hu", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:HuMoments", (char**)keywords, &pyobj_m, &pyobj_hu) &&
        pyopencv_to(pyobj_m, m, ArgInfo("m", 0)) &&
        pyopencv_to(pyobj_hu, hu, ArgInfo("hu", 1)) )
    {
        ERRWRAP2(cv::HuMoments(m, hu));
        return pyopencv_from(hu);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_m = NULL;
    Moments m;
    PyObject* pyobj_hu = NULL;
    UMat hu;

    const char* keywords[] = { "m", "hu", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:HuMoments", (char**)keywords, &pyobj_m, &pyobj_hu) &&
        pyopencv_to(pyobj_m, m, ArgInfo("m", 0)) &&
        pyopencv_to(pyobj_hu, hu, ArgInfo("hu", 1)) )
    {
        ERRWRAP2(cv::HuMoments(m, hu));
        return pyopencv_from(hu);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_KAZE_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    bool extended=false;
    bool upright=false;
    float threshold=0.001f;
    int nOctaves=4;
    int nOctaveLayers=4;
    int diffusivity=KAZE::DIFF_PM_G2;
    Ptr<KAZE> retval;

    const char* keywords[] = { "extended", "upright", "threshold", "nOctaves", "nOctaveLayers", "diffusivity", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|bbfiii:KAZE_create", (char**)keywords, &extended, &upright, &threshold, &nOctaves, &nOctaveLayers, &diffusivity) )
    {
        ERRWRAP2(retval = cv::KAZE::create(extended, upright, threshold, nOctaves, nOctaveLayers, diffusivity));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_KalmanFilter_KalmanFilter(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    pyopencv_KalmanFilter_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_KalmanFilter_t, &pyopencv_KalmanFilter_Type);
        new (&(self->v)) Ptr<cv::KalmanFilter>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::KalmanFilter()));
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    int dynamParams=0;
    int measureParams=0;
    int controlParams=0;
    int type=CV_32F;
    pyopencv_KalmanFilter_t* self = 0;

    const char* keywords[] = { "dynamParams", "measureParams", "controlParams", "type", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "ii|ii:KalmanFilter", (char**)keywords, &dynamParams, &measureParams, &controlParams, &type) )
    {
        self = PyObject_NEW(pyopencv_KalmanFilter_t, &pyopencv_KalmanFilter_Type);
        new (&(self->v)) Ptr<cv::KalmanFilter>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::KalmanFilter(dynamParams, measureParams, controlParams, type)));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_KeyPoint_KeyPoint(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    pyopencv_KeyPoint_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_KeyPoint_t, &pyopencv_KeyPoint_Type);
        if(self) ERRWRAP2(new (&(self->v)) cv::KeyPoint());
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    float x=0.f;
    float y=0.f;
    float _size=0.f;
    float _angle=-1;
    float _response=0;
    int _octave=0;
    int _class_id=-1;
    pyopencv_KeyPoint_t* self = 0;

    const char* keywords[] = { "x", "y", "_size", "_angle", "_response", "_octave", "_class_id", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "fff|ffii:KeyPoint", (char**)keywords, &x, &y, &_size, &_angle, &_response, &_octave, &_class_id) )
    {
        self = PyObject_NEW(pyopencv_KeyPoint_t, &pyopencv_KeyPoint_Type);
        if(self) ERRWRAP2(new (&(self->v)) cv::KeyPoint(x, y, _size, _angle, _response, _octave, _class_id));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_KeyPoint_convert(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_keypoints = NULL;
    vector_KeyPoint keypoints;
    vector_Point2f points2f;
    PyObject* pyobj_keypointIndexes = NULL;
    vector_int keypointIndexes=std::vector<int>();

    const char* keywords[] = { "keypoints", "keypointIndexes", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:KeyPoint_convert", (char**)keywords, &pyobj_keypoints, &pyobj_keypointIndexes) &&
        pyopencv_to(pyobj_keypoints, keypoints, ArgInfo("keypoints", 0)) &&
        pyopencv_to(pyobj_keypointIndexes, keypointIndexes, ArgInfo("keypointIndexes", 0)) )
    {
        ERRWRAP2(cv::KeyPoint::convert(keypoints, points2f, keypointIndexes));
        return pyopencv_from(points2f);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_points2f = NULL;
    vector_Point2f points2f;
    vector_KeyPoint keypoints;
    float size=1;
    float response=1;
    int octave=0;
    int class_id=-1;

    const char* keywords[] = { "points2f", "size", "response", "octave", "class_id", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|ffii:KeyPoint_convert", (char**)keywords, &pyobj_points2f, &size, &response, &octave, &class_id) &&
        pyopencv_to(pyobj_points2f, points2f, ArgInfo("points2f", 0)) )
    {
        ERRWRAP2(cv::KeyPoint::convert(points2f, keypoints, size, response, octave, class_id));
        return pyopencv_from(keypoints);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_KeyPoint_overlap(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_kp1 = NULL;
    KeyPoint kp1;
    PyObject* pyobj_kp2 = NULL;
    KeyPoint kp2;
    float retval;

    const char* keywords[] = { "kp1", "kp2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:KeyPoint_overlap", (char**)keywords, &pyobj_kp1, &pyobj_kp2) &&
        pyopencv_to(pyobj_kp1, kp1, ArgInfo("kp1", 0)) &&
        pyopencv_to(pyobj_kp2, kp2, ArgInfo("kp2", 0)) )
    {
        ERRWRAP2(retval = cv::KeyPoint::overlap(kp1, kp2));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_LUT(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_lut = NULL;
    Mat lut;
    PyObject* pyobj_dst = NULL;
    Mat dst;

    const char* keywords[] = { "src", "lut", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:LUT", (char**)keywords, &pyobj_src, &pyobj_lut, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_lut, lut, ArgInfo("lut", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::LUT(src, lut, dst));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_lut = NULL;
    UMat lut;
    PyObject* pyobj_dst = NULL;
    UMat dst;

    const char* keywords[] = { "src", "lut", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:LUT", (char**)keywords, &pyobj_src, &pyobj_lut, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_lut, lut, ArgInfo("lut", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::LUT(src, lut, dst));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_Laplacian(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int ddepth=0;
    int ksize=1;
    double scale=1;
    double delta=0;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "ddepth", "dst", "ksize", "scale", "delta", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|Oiddi:Laplacian", (char**)keywords, &pyobj_src, &ddepth, &pyobj_dst, &ksize, &scale, &delta, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::Laplacian(src, dst, ddepth, ksize, scale, delta, borderType));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int ddepth=0;
    int ksize=1;
    double scale=1;
    double delta=0;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "ddepth", "dst", "ksize", "scale", "delta", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|Oiddi:Laplacian", (char**)keywords, &pyobj_src, &ddepth, &pyobj_dst, &ksize, &scale, &delta, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::Laplacian(src, dst, ddepth, ksize, scale, delta, borderType));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_MSER_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int _delta=5;
    int _min_area=60;
    int _max_area=14400;
    double _max_variation=0.25;
    double _min_diversity=.2;
    int _max_evolution=200;
    double _area_threshold=1.01;
    double _min_margin=0.003;
    int _edge_blur_size=5;
    Ptr<MSER> retval;

    const char* keywords[] = { "_delta", "_min_area", "_max_area", "_max_variation", "_min_diversity", "_max_evolution", "_area_threshold", "_min_margin", "_edge_blur_size", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|iiiddiddi:MSER_create", (char**)keywords, &_delta, &_min_area, &_max_area, &_max_variation, &_min_diversity, &_max_evolution, &_area_threshold, &_min_margin, &_edge_blur_size) )
    {
        ERRWRAP2(retval = cv::MSER::create(_delta, _min_area, _max_area, _max_variation, _min_diversity, _max_evolution, _area_threshold, _min_margin, _edge_blur_size));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_Mahalanobis(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_v1 = NULL;
    Mat v1;
    PyObject* pyobj_v2 = NULL;
    Mat v2;
    PyObject* pyobj_icovar = NULL;
    Mat icovar;
    double retval;

    const char* keywords[] = { "v1", "v2", "icovar", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:Mahalanobis", (char**)keywords, &pyobj_v1, &pyobj_v2, &pyobj_icovar) &&
        pyopencv_to(pyobj_v1, v1, ArgInfo("v1", 0)) &&
        pyopencv_to(pyobj_v2, v2, ArgInfo("v2", 0)) &&
        pyopencv_to(pyobj_icovar, icovar, ArgInfo("icovar", 0)) )
    {
        ERRWRAP2(retval = cv::Mahalanobis(v1, v2, icovar));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_v1 = NULL;
    UMat v1;
    PyObject* pyobj_v2 = NULL;
    UMat v2;
    PyObject* pyobj_icovar = NULL;
    UMat icovar;
    double retval;

    const char* keywords[] = { "v1", "v2", "icovar", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:Mahalanobis", (char**)keywords, &pyobj_v1, &pyobj_v2, &pyobj_icovar) &&
        pyopencv_to(pyobj_v1, v1, ArgInfo("v1", 0)) &&
        pyopencv_to(pyobj_v2, v2, ArgInfo("v2", 0)) &&
        pyopencv_to(pyobj_icovar, icovar, ArgInfo("icovar", 0)) )
    {
        ERRWRAP2(retval = cv::Mahalanobis(v1, v2, icovar));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_MultiTracker_MultiTracker(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_trackerType = NULL;
    String trackerType="";
    pyopencv_MultiTracker_t* self = 0;

    const char* keywords[] = { "trackerType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|O:MultiTracker", (char**)keywords, &pyobj_trackerType) &&
        pyopencv_to(pyobj_trackerType, trackerType, ArgInfo("trackerType", 0)) )
    {
        self = PyObject_NEW(pyopencv_MultiTracker_t, &pyopencv_MultiTracker_Type);
        new (&(self->v)) Ptr<cv::MultiTracker>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::MultiTracker(trackerType)));
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ORB_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int nfeatures=500;
    float scaleFactor=1.2f;
    int nlevels=8;
    int edgeThreshold=31;
    int firstLevel=0;
    int WTA_K=2;
    int scoreType=ORB::HARRIS_SCORE;
    int patchSize=31;
    int fastThreshold=20;
    Ptr<ORB> retval;

    const char* keywords[] = { "nfeatures", "scaleFactor", "nlevels", "edgeThreshold", "firstLevel", "WTA_K", "scoreType", "patchSize", "fastThreshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|ifiiiiiii:ORB_create", (char**)keywords, &nfeatures, &scaleFactor, &nlevels, &edgeThreshold, &firstLevel, &WTA_K, &scoreType, &patchSize, &fastThreshold) )
    {
        ERRWRAP2(retval = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_PCABackProject(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_data = NULL;
    Mat data;
    PyObject* pyobj_mean = NULL;
    Mat mean;
    PyObject* pyobj_eigenvectors = NULL;
    Mat eigenvectors;
    PyObject* pyobj_result = NULL;
    Mat result;

    const char* keywords[] = { "data", "mean", "eigenvectors", "result", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|O:PCABackProject", (char**)keywords, &pyobj_data, &pyobj_mean, &pyobj_eigenvectors, &pyobj_result) &&
        pyopencv_to(pyobj_data, data, ArgInfo("data", 0)) &&
        pyopencv_to(pyobj_mean, mean, ArgInfo("mean", 0)) &&
        pyopencv_to(pyobj_eigenvectors, eigenvectors, ArgInfo("eigenvectors", 0)) &&
        pyopencv_to(pyobj_result, result, ArgInfo("result", 1)) )
    {
        ERRWRAP2(cv::PCABackProject(data, mean, eigenvectors, result));
        return pyopencv_from(result);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_data = NULL;
    UMat data;
    PyObject* pyobj_mean = NULL;
    UMat mean;
    PyObject* pyobj_eigenvectors = NULL;
    UMat eigenvectors;
    PyObject* pyobj_result = NULL;
    UMat result;

    const char* keywords[] = { "data", "mean", "eigenvectors", "result", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|O:PCABackProject", (char**)keywords, &pyobj_data, &pyobj_mean, &pyobj_eigenvectors, &pyobj_result) &&
        pyopencv_to(pyobj_data, data, ArgInfo("data", 0)) &&
        pyopencv_to(pyobj_mean, mean, ArgInfo("mean", 0)) &&
        pyopencv_to(pyobj_eigenvectors, eigenvectors, ArgInfo("eigenvectors", 0)) &&
        pyopencv_to(pyobj_result, result, ArgInfo("result", 1)) )
    {
        ERRWRAP2(cv::PCABackProject(data, mean, eigenvectors, result));
        return pyopencv_from(result);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_PCACompute(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_data = NULL;
    Mat data;
    PyObject* pyobj_mean = NULL;
    Mat mean;
    PyObject* pyobj_eigenvectors = NULL;
    Mat eigenvectors;
    int maxComponents=0;

    const char* keywords[] = { "data", "mean", "eigenvectors", "maxComponents", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Oi:PCACompute", (char**)keywords, &pyobj_data, &pyobj_mean, &pyobj_eigenvectors, &maxComponents) &&
        pyopencv_to(pyobj_data, data, ArgInfo("data", 0)) &&
        pyopencv_to(pyobj_mean, mean, ArgInfo("mean", 1)) &&
        pyopencv_to(pyobj_eigenvectors, eigenvectors, ArgInfo("eigenvectors", 1)) )
    {
        ERRWRAP2(cv::PCACompute(data, mean, eigenvectors, maxComponents));
        return Py_BuildValue("(NN)", pyopencv_from(mean), pyopencv_from(eigenvectors));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_data = NULL;
    UMat data;
    PyObject* pyobj_mean = NULL;
    UMat mean;
    PyObject* pyobj_eigenvectors = NULL;
    UMat eigenvectors;
    int maxComponents=0;

    const char* keywords[] = { "data", "mean", "eigenvectors", "maxComponents", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Oi:PCACompute", (char**)keywords, &pyobj_data, &pyobj_mean, &pyobj_eigenvectors, &maxComponents) &&
        pyopencv_to(pyobj_data, data, ArgInfo("data", 0)) &&
        pyopencv_to(pyobj_mean, mean, ArgInfo("mean", 1)) &&
        pyopencv_to(pyobj_eigenvectors, eigenvectors, ArgInfo("eigenvectors", 1)) )
    {
        ERRWRAP2(cv::PCACompute(data, mean, eigenvectors, maxComponents));
        return Py_BuildValue("(NN)", pyopencv_from(mean), pyopencv_from(eigenvectors));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_data = NULL;
    Mat data;
    PyObject* pyobj_mean = NULL;
    Mat mean;
    PyObject* pyobj_eigenvectors = NULL;
    Mat eigenvectors;
    double retainedVariance=0;

    const char* keywords[] = { "data", "mean", "retainedVariance", "eigenvectors", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOd|O:PCACompute", (char**)keywords, &pyobj_data, &pyobj_mean, &retainedVariance, &pyobj_eigenvectors) &&
        pyopencv_to(pyobj_data, data, ArgInfo("data", 0)) &&
        pyopencv_to(pyobj_mean, mean, ArgInfo("mean", 1)) &&
        pyopencv_to(pyobj_eigenvectors, eigenvectors, ArgInfo("eigenvectors", 1)) )
    {
        ERRWRAP2(cv::PCACompute(data, mean, eigenvectors, retainedVariance));
        return Py_BuildValue("(NN)", pyopencv_from(mean), pyopencv_from(eigenvectors));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_data = NULL;
    UMat data;
    PyObject* pyobj_mean = NULL;
    UMat mean;
    PyObject* pyobj_eigenvectors = NULL;
    UMat eigenvectors;
    double retainedVariance=0;

    const char* keywords[] = { "data", "mean", "retainedVariance", "eigenvectors", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOd|O:PCACompute", (char**)keywords, &pyobj_data, &pyobj_mean, &retainedVariance, &pyobj_eigenvectors) &&
        pyopencv_to(pyobj_data, data, ArgInfo("data", 0)) &&
        pyopencv_to(pyobj_mean, mean, ArgInfo("mean", 1)) &&
        pyopencv_to(pyobj_eigenvectors, eigenvectors, ArgInfo("eigenvectors", 1)) )
    {
        ERRWRAP2(cv::PCACompute(data, mean, eigenvectors, retainedVariance));
        return Py_BuildValue("(NN)", pyopencv_from(mean), pyopencv_from(eigenvectors));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_PCAProject(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_data = NULL;
    Mat data;
    PyObject* pyobj_mean = NULL;
    Mat mean;
    PyObject* pyobj_eigenvectors = NULL;
    Mat eigenvectors;
    PyObject* pyobj_result = NULL;
    Mat result;

    const char* keywords[] = { "data", "mean", "eigenvectors", "result", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|O:PCAProject", (char**)keywords, &pyobj_data, &pyobj_mean, &pyobj_eigenvectors, &pyobj_result) &&
        pyopencv_to(pyobj_data, data, ArgInfo("data", 0)) &&
        pyopencv_to(pyobj_mean, mean, ArgInfo("mean", 0)) &&
        pyopencv_to(pyobj_eigenvectors, eigenvectors, ArgInfo("eigenvectors", 0)) &&
        pyopencv_to(pyobj_result, result, ArgInfo("result", 1)) )
    {
        ERRWRAP2(cv::PCAProject(data, mean, eigenvectors, result));
        return pyopencv_from(result);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_data = NULL;
    UMat data;
    PyObject* pyobj_mean = NULL;
    UMat mean;
    PyObject* pyobj_eigenvectors = NULL;
    UMat eigenvectors;
    PyObject* pyobj_result = NULL;
    UMat result;

    const char* keywords[] = { "data", "mean", "eigenvectors", "result", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|O:PCAProject", (char**)keywords, &pyobj_data, &pyobj_mean, &pyobj_eigenvectors, &pyobj_result) &&
        pyopencv_to(pyobj_data, data, ArgInfo("data", 0)) &&
        pyopencv_to(pyobj_mean, mean, ArgInfo("mean", 0)) &&
        pyopencv_to(pyobj_eigenvectors, eigenvectors, ArgInfo("eigenvectors", 0)) &&
        pyopencv_to(pyobj_result, result, ArgInfo("result", 1)) )
    {
        ERRWRAP2(cv::PCAProject(data, mean, eigenvectors, result));
        return pyopencv_from(result);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_PSNR(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    PyObject* pyobj_src2 = NULL;
    Mat src2;
    double retval;

    const char* keywords[] = { "src1", "src2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:PSNR", (char**)keywords, &pyobj_src1, &pyobj_src2) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) )
    {
        ERRWRAP2(retval = cv::PSNR(src1, src2));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src1 = NULL;
    UMat src1;
    PyObject* pyobj_src2 = NULL;
    UMat src2;
    double retval;

    const char* keywords[] = { "src1", "src2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:PSNR", (char**)keywords, &pyobj_src1, &pyobj_src2) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) )
    {
        ERRWRAP2(retval = cv::PSNR(src1, src2));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_RQDecomp3x3(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_mtxR = NULL;
    Mat mtxR;
    PyObject* pyobj_mtxQ = NULL;
    Mat mtxQ;
    PyObject* pyobj_Qx = NULL;
    Mat Qx;
    PyObject* pyobj_Qy = NULL;
    Mat Qy;
    PyObject* pyobj_Qz = NULL;
    Mat Qz;
    Vec3d retval;

    const char* keywords[] = { "src", "mtxR", "mtxQ", "Qx", "Qy", "Qz", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOOOO:RQDecomp3x3", (char**)keywords, &pyobj_src, &pyobj_mtxR, &pyobj_mtxQ, &pyobj_Qx, &pyobj_Qy, &pyobj_Qz) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_mtxR, mtxR, ArgInfo("mtxR", 1)) &&
        pyopencv_to(pyobj_mtxQ, mtxQ, ArgInfo("mtxQ", 1)) &&
        pyopencv_to(pyobj_Qx, Qx, ArgInfo("Qx", 1)) &&
        pyopencv_to(pyobj_Qy, Qy, ArgInfo("Qy", 1)) &&
        pyopencv_to(pyobj_Qz, Qz, ArgInfo("Qz", 1)) )
    {
        ERRWRAP2(retval = cv::RQDecomp3x3(src, mtxR, mtxQ, Qx, Qy, Qz));
        return Py_BuildValue("(NNNNNN)", pyopencv_from(retval), pyopencv_from(mtxR), pyopencv_from(mtxQ), pyopencv_from(Qx), pyopencv_from(Qy), pyopencv_from(Qz));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_mtxR = NULL;
    UMat mtxR;
    PyObject* pyobj_mtxQ = NULL;
    UMat mtxQ;
    PyObject* pyobj_Qx = NULL;
    UMat Qx;
    PyObject* pyobj_Qy = NULL;
    UMat Qy;
    PyObject* pyobj_Qz = NULL;
    UMat Qz;
    Vec3d retval;

    const char* keywords[] = { "src", "mtxR", "mtxQ", "Qx", "Qy", "Qz", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOOOO:RQDecomp3x3", (char**)keywords, &pyobj_src, &pyobj_mtxR, &pyobj_mtxQ, &pyobj_Qx, &pyobj_Qy, &pyobj_Qz) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_mtxR, mtxR, ArgInfo("mtxR", 1)) &&
        pyopencv_to(pyobj_mtxQ, mtxQ, ArgInfo("mtxQ", 1)) &&
        pyopencv_to(pyobj_Qx, Qx, ArgInfo("Qx", 1)) &&
        pyopencv_to(pyobj_Qy, Qy, ArgInfo("Qy", 1)) &&
        pyopencv_to(pyobj_Qz, Qz, ArgInfo("Qz", 1)) )
    {
        ERRWRAP2(retval = cv::RQDecomp3x3(src, mtxR, mtxQ, Qx, Qy, Qz));
        return Py_BuildValue("(NNNNNN)", pyopencv_from(retval), pyopencv_from(mtxR), pyopencv_from(mtxQ), pyopencv_from(Qx), pyopencv_from(Qy), pyopencv_from(Qz));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_Rodrigues(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_jacobian = NULL;
    Mat jacobian;

    const char* keywords[] = { "src", "dst", "jacobian", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OO:Rodrigues", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_jacobian) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_jacobian, jacobian, ArgInfo("jacobian", 1)) )
    {
        ERRWRAP2(cv::Rodrigues(src, dst, jacobian));
        return Py_BuildValue("(NN)", pyopencv_from(dst), pyopencv_from(jacobian));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_jacobian = NULL;
    UMat jacobian;

    const char* keywords[] = { "src", "dst", "jacobian", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OO:Rodrigues", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_jacobian) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_jacobian, jacobian, ArgInfo("jacobian", 1)) )
    {
        ERRWRAP2(cv::Rodrigues(src, dst, jacobian));
        return Py_BuildValue("(NN)", pyopencv_from(dst), pyopencv_from(jacobian));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_SVBackSubst(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_w = NULL;
    Mat w;
    PyObject* pyobj_u = NULL;
    Mat u;
    PyObject* pyobj_vt = NULL;
    Mat vt;
    PyObject* pyobj_rhs = NULL;
    Mat rhs;
    PyObject* pyobj_dst = NULL;
    Mat dst;

    const char* keywords[] = { "w", "u", "vt", "rhs", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|O:SVBackSubst", (char**)keywords, &pyobj_w, &pyobj_u, &pyobj_vt, &pyobj_rhs, &pyobj_dst) &&
        pyopencv_to(pyobj_w, w, ArgInfo("w", 0)) &&
        pyopencv_to(pyobj_u, u, ArgInfo("u", 0)) &&
        pyopencv_to(pyobj_vt, vt, ArgInfo("vt", 0)) &&
        pyopencv_to(pyobj_rhs, rhs, ArgInfo("rhs", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::SVBackSubst(w, u, vt, rhs, dst));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_w = NULL;
    UMat w;
    PyObject* pyobj_u = NULL;
    UMat u;
    PyObject* pyobj_vt = NULL;
    UMat vt;
    PyObject* pyobj_rhs = NULL;
    UMat rhs;
    PyObject* pyobj_dst = NULL;
    UMat dst;

    const char* keywords[] = { "w", "u", "vt", "rhs", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|O:SVBackSubst", (char**)keywords, &pyobj_w, &pyobj_u, &pyobj_vt, &pyobj_rhs, &pyobj_dst) &&
        pyopencv_to(pyobj_w, w, ArgInfo("w", 0)) &&
        pyopencv_to(pyobj_u, u, ArgInfo("u", 0)) &&
        pyopencv_to(pyobj_vt, vt, ArgInfo("vt", 0)) &&
        pyopencv_to(pyobj_rhs, rhs, ArgInfo("rhs", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::SVBackSubst(w, u, vt, rhs, dst));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_SVDecomp(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_w = NULL;
    Mat w;
    PyObject* pyobj_u = NULL;
    Mat u;
    PyObject* pyobj_vt = NULL;
    Mat vt;
    int flags=0;

    const char* keywords[] = { "src", "w", "u", "vt", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOOi:SVDecomp", (char**)keywords, &pyobj_src, &pyobj_w, &pyobj_u, &pyobj_vt, &flags) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_w, w, ArgInfo("w", 1)) &&
        pyopencv_to(pyobj_u, u, ArgInfo("u", 1)) &&
        pyopencv_to(pyobj_vt, vt, ArgInfo("vt", 1)) )
    {
        ERRWRAP2(cv::SVDecomp(src, w, u, vt, flags));
        return Py_BuildValue("(NNN)", pyopencv_from(w), pyopencv_from(u), pyopencv_from(vt));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_w = NULL;
    UMat w;
    PyObject* pyobj_u = NULL;
    UMat u;
    PyObject* pyobj_vt = NULL;
    UMat vt;
    int flags=0;

    const char* keywords[] = { "src", "w", "u", "vt", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOOi:SVDecomp", (char**)keywords, &pyobj_src, &pyobj_w, &pyobj_u, &pyobj_vt, &flags) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_w, w, ArgInfo("w", 1)) &&
        pyopencv_to(pyobj_u, u, ArgInfo("u", 1)) &&
        pyopencv_to(pyobj_vt, vt, ArgInfo("vt", 1)) )
    {
        ERRWRAP2(cv::SVDecomp(src, w, u, vt, flags));
        return Py_BuildValue("(NNN)", pyopencv_from(w), pyopencv_from(u), pyopencv_from(vt));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_Scharr(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int ddepth=0;
    int dx=0;
    int dy=0;
    double scale=1;
    double delta=0;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "ddepth", "dx", "dy", "dst", "scale", "delta", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oiii|Oddi:Scharr", (char**)keywords, &pyobj_src, &ddepth, &dx, &dy, &pyobj_dst, &scale, &delta, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::Scharr(src, dst, ddepth, dx, dy, scale, delta, borderType));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int ddepth=0;
    int dx=0;
    int dy=0;
    double scale=1;
    double delta=0;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "ddepth", "dx", "dy", "dst", "scale", "delta", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oiii|Oddi:Scharr", (char**)keywords, &pyobj_src, &ddepth, &dx, &dy, &pyobj_dst, &scale, &delta, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::Scharr(src, dst, ddepth, dx, dy, scale, delta, borderType));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_SimpleBlobDetector_Params_SimpleBlobDetector_Params(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    pyopencv_SimpleBlobDetector_Params_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_SimpleBlobDetector_Params_t, &pyopencv_SimpleBlobDetector_Params_Type);
        if(self) ERRWRAP2(new (&(self->v)) cv::SimpleBlobDetector::Params());
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_cv_SimpleBlobDetector_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_parameters = NULL;
    SimpleBlobDetector_Params parameters=SimpleBlobDetector::Params();
    Ptr<SimpleBlobDetector> retval;

    const char* keywords[] = { "parameters", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|O:SimpleBlobDetector_create", (char**)keywords, &pyobj_parameters) &&
        pyopencv_to(pyobj_parameters, parameters, ArgInfo("parameters", 0)) )
    {
        ERRWRAP2(retval = cv::SimpleBlobDetector::create(parameters));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_Sobel(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int ddepth=0;
    int dx=0;
    int dy=0;
    int ksize=3;
    double scale=1;
    double delta=0;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "ddepth", "dx", "dy", "dst", "ksize", "scale", "delta", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oiii|Oiddi:Sobel", (char**)keywords, &pyobj_src, &ddepth, &dx, &dy, &pyobj_dst, &ksize, &scale, &delta, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::Sobel(src, dst, ddepth, dx, dy, ksize, scale, delta, borderType));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int ddepth=0;
    int dx=0;
    int dy=0;
    int ksize=3;
    double scale=1;
    double delta=0;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "ddepth", "dx", "dy", "dst", "ksize", "scale", "delta", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oiii|Oiddi:Sobel", (char**)keywords, &pyobj_src, &ddepth, &dx, &dy, &pyobj_dst, &ksize, &scale, &delta, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::Sobel(src, dst, ddepth, dx, dy, ksize, scale, delta, borderType));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_SparsePyrLKOpticalFlow_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_winSize = NULL;
    Size winSize=Size(21, 21);
    int maxLevel=3;
    PyObject* pyobj_crit = NULL;
    TermCriteria crit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
    int flags=0;
    double minEigThreshold=1e-4;
    Ptr<SparsePyrLKOpticalFlow> retval;

    const char* keywords[] = { "winSize", "maxLevel", "crit", "flags", "minEigThreshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|OiOid:SparsePyrLKOpticalFlow_create", (char**)keywords, &pyobj_winSize, &maxLevel, &pyobj_crit, &flags, &minEigThreshold) &&
        pyopencv_to(pyobj_winSize, winSize, ArgInfo("winSize", 0)) &&
        pyopencv_to(pyobj_crit, crit, ArgInfo("crit", 0)) )
    {
        ERRWRAP2(retval = cv::SparsePyrLKOpticalFlow::create(winSize, maxLevel, crit, flags, minEigThreshold));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoBM_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int numDisparities=0;
    int blockSize=21;
    Ptr<StereoBM> retval;

    const char* keywords[] = { "numDisparities", "blockSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|ii:StereoBM_create", (char**)keywords, &numDisparities, &blockSize) )
    {
        ERRWRAP2(retval = cv::StereoBM::create(numDisparities, blockSize));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoSGBM_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int minDisparity=0;
    int numDisparities=0;
    int blockSize=0;
    int P1=0;
    int P2=0;
    int disp12MaxDiff=0;
    int preFilterCap=0;
    int uniquenessRatio=0;
    int speckleWindowSize=0;
    int speckleRange=0;
    int mode=StereoSGBM::MODE_SGBM;
    Ptr<StereoSGBM> retval;

    const char* keywords[] = { "minDisparity", "numDisparities", "blockSize", "P1", "P2", "disp12MaxDiff", "preFilterCap", "uniquenessRatio", "speckleWindowSize", "speckleRange", "mode", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "iii|iiiiiiii:StereoSGBM_create", (char**)keywords, &minDisparity, &numDisparities, &blockSize, &P1, &P2, &disp12MaxDiff, &preFilterCap, &uniquenessRatio, &speckleWindowSize, &speckleRange, &mode) )
    {
        ERRWRAP2(retval = cv::StereoSGBM::create(minDisparity, numDisparities, blockSize, P1, P2, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, mode));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_Subdiv2D_Subdiv2D(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    pyopencv_Subdiv2D_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_Subdiv2D_t, &pyopencv_Subdiv2D_Type);
        new (&(self->v)) Ptr<cv::Subdiv2D>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::Subdiv2D()));
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_rect = NULL;
    Rect rect;
    pyopencv_Subdiv2D_t* self = 0;

    const char* keywords[] = { "rect", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:Subdiv2D", (char**)keywords, &pyobj_rect) &&
        pyopencv_to(pyobj_rect, rect, ArgInfo("rect", 0)) )
    {
        self = PyObject_NEW(pyopencv_Subdiv2D_t, &pyopencv_Subdiv2D_Type);
        new (&(self->v)) Ptr<cv::Subdiv2D>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::Subdiv2D(rect)));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_TickMeter_TickMeter(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    pyopencv_TickMeter_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_TickMeter_t, &pyopencv_TickMeter_Type);
        new (&(self->v)) Ptr<cv::TickMeter>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::TickMeter()));
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_cv_Tracker_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_trackerType = NULL;
    String trackerType;
    Ptr<Tracker> retval;

    const char* keywords[] = { "trackerType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:Tracker_create", (char**)keywords, &pyobj_trackerType) &&
        pyopencv_to(pyobj_trackerType, trackerType, ArgInfo("trackerType", 0)) )
    {
        ERRWRAP2(retval = cv::Tracker::create(trackerType));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_VideoCapture_VideoCapture(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    pyopencv_VideoCapture_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_VideoCapture_t, &pyopencv_VideoCapture_Type);
        new (&(self->v)) Ptr<cv::VideoCapture>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::VideoCapture()));
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_filename = NULL;
    String filename;
    pyopencv_VideoCapture_t* self = 0;

    const char* keywords[] = { "filename", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:VideoCapture", (char**)keywords, &pyobj_filename) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) )
    {
        self = PyObject_NEW(pyopencv_VideoCapture_t, &pyopencv_VideoCapture_Type);
        new (&(self->v)) Ptr<cv::VideoCapture>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::VideoCapture(filename)));
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_filename = NULL;
    String filename;
    int apiPreference=0;
    pyopencv_VideoCapture_t* self = 0;

    const char* keywords[] = { "filename", "apiPreference", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi:VideoCapture", (char**)keywords, &pyobj_filename, &apiPreference) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) )
    {
        self = PyObject_NEW(pyopencv_VideoCapture_t, &pyopencv_VideoCapture_Type);
        new (&(self->v)) Ptr<cv::VideoCapture>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::VideoCapture(filename, apiPreference)));
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    int index=0;
    pyopencv_VideoCapture_t* self = 0;

    const char* keywords[] = { "index", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:VideoCapture", (char**)keywords, &index) )
    {
        self = PyObject_NEW(pyopencv_VideoCapture_t, &pyopencv_VideoCapture_Type);
        new (&(self->v)) Ptr<cv::VideoCapture>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::VideoCapture(index)));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_VideoWriter_VideoWriter(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    pyopencv_VideoWriter_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_VideoWriter_t, &pyopencv_VideoWriter_Type);
        new (&(self->v)) Ptr<cv::VideoWriter>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::VideoWriter()));
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_filename = NULL;
    String filename;
    int fourcc=0;
    double fps=0;
    PyObject* pyobj_frameSize = NULL;
    Size frameSize;
    bool isColor=true;
    pyopencv_VideoWriter_t* self = 0;

    const char* keywords[] = { "filename", "fourcc", "fps", "frameSize", "isColor", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OidO|b:VideoWriter", (char**)keywords, &pyobj_filename, &fourcc, &fps, &pyobj_frameSize, &isColor) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) &&
        pyopencv_to(pyobj_frameSize, frameSize, ArgInfo("frameSize", 0)) )
    {
        self = PyObject_NEW(pyopencv_VideoWriter_t, &pyopencv_VideoWriter_Type);
        new (&(self->v)) Ptr<cv::VideoWriter>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::VideoWriter(filename, fourcc, fps, frameSize, isColor)));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_VideoWriter_fourcc(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_c1 = NULL;
    char c1;
    PyObject* pyobj_c2 = NULL;
    char c2;
    PyObject* pyobj_c3 = NULL;
    char c3;
    PyObject* pyobj_c4 = NULL;
    char c4;
    int retval;

    const char* keywords[] = { "c1", "c2", "c3", "c4", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO:VideoWriter_fourcc", (char**)keywords, &pyobj_c1, &pyobj_c2, &pyobj_c3, &pyobj_c4) &&
        convert_to_char(pyobj_c1, &c1, ArgInfo("c1", 0)) &&
        convert_to_char(pyobj_c2, &c2, ArgInfo("c2", 0)) &&
        convert_to_char(pyobj_c3, &c3, ArgInfo("c3", 0)) &&
        convert_to_char(pyobj_c4, &c4, ArgInfo("c4", 0)) )
    {
        ERRWRAP2(retval = cv::VideoWriter::fourcc(c1, c2, c3, c4));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_absdiff(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    PyObject* pyobj_src2 = NULL;
    Mat src2;
    PyObject* pyobj_dst = NULL;
    Mat dst;

    const char* keywords[] = { "src1", "src2", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:absdiff", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::absdiff(src1, src2, dst));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src1 = NULL;
    UMat src1;
    PyObject* pyobj_src2 = NULL;
    UMat src2;
    PyObject* pyobj_dst = NULL;
    UMat dst;

    const char* keywords[] = { "src1", "src2", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:absdiff", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::absdiff(src1, src2, dst));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_accumulate(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_mask = NULL;
    Mat mask;

    const char* keywords[] = { "src", "dst", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:accumulate", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_mask) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::accumulate(src, dst, mask));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_mask = NULL;
    UMat mask;

    const char* keywords[] = { "src", "dst", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:accumulate", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_mask) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::accumulate(src, dst, mask));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_accumulateProduct(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    PyObject* pyobj_src2 = NULL;
    Mat src2;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_mask = NULL;
    Mat mask;

    const char* keywords[] = { "src1", "src2", "dst", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|O:accumulateProduct", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &pyobj_mask) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::accumulateProduct(src1, src2, dst, mask));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src1 = NULL;
    UMat src1;
    PyObject* pyobj_src2 = NULL;
    UMat src2;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_mask = NULL;
    UMat mask;

    const char* keywords[] = { "src1", "src2", "dst", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|O:accumulateProduct", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &pyobj_mask) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::accumulateProduct(src1, src2, dst, mask));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_accumulateSquare(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_mask = NULL;
    Mat mask;

    const char* keywords[] = { "src", "dst", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:accumulateSquare", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_mask) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::accumulateSquare(src, dst, mask));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_mask = NULL;
    UMat mask;

    const char* keywords[] = { "src", "dst", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:accumulateSquare", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_mask) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::accumulateSquare(src, dst, mask));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_accumulateWeighted(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    double alpha=0;
    PyObject* pyobj_mask = NULL;
    Mat mask;

    const char* keywords[] = { "src", "dst", "alpha", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOd|O:accumulateWeighted", (char**)keywords, &pyobj_src, &pyobj_dst, &alpha, &pyobj_mask) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::accumulateWeighted(src, dst, alpha, mask));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    double alpha=0;
    PyObject* pyobj_mask = NULL;
    UMat mask;

    const char* keywords[] = { "src", "dst", "alpha", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOd|O:accumulateWeighted", (char**)keywords, &pyobj_src, &pyobj_dst, &alpha, &pyobj_mask) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::accumulateWeighted(src, dst, alpha, mask));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_adaptiveThreshold(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    double maxValue=0;
    int adaptiveMethod=0;
    int thresholdType=0;
    int blockSize=0;
    double C=0;

    const char* keywords[] = { "src", "maxValue", "adaptiveMethod", "thresholdType", "blockSize", "C", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odiiid|O:adaptiveThreshold", (char**)keywords, &pyobj_src, &maxValue, &adaptiveMethod, &thresholdType, &blockSize, &C, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::adaptiveThreshold(src, dst, maxValue, adaptiveMethod, thresholdType, blockSize, C));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    double maxValue=0;
    int adaptiveMethod=0;
    int thresholdType=0;
    int blockSize=0;
    double C=0;

    const char* keywords[] = { "src", "maxValue", "adaptiveMethod", "thresholdType", "blockSize", "C", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odiiid|O:adaptiveThreshold", (char**)keywords, &pyobj_src, &maxValue, &adaptiveMethod, &thresholdType, &blockSize, &C, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::adaptiveThreshold(src, dst, maxValue, adaptiveMethod, thresholdType, blockSize, C));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_add(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    PyObject* pyobj_src2 = NULL;
    Mat src2;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    int dtype=-1;

    const char* keywords[] = { "src1", "src2", "dst", "mask", "dtype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOi:add", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &pyobj_mask, &dtype) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::add(src1, src2, dst, mask, dtype));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src1 = NULL;
    UMat src1;
    PyObject* pyobj_src2 = NULL;
    UMat src2;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    int dtype=-1;

    const char* keywords[] = { "src1", "src2", "dst", "mask", "dtype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOi:add", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &pyobj_mask, &dtype) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::add(src1, src2, dst, mask, dtype));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_addText(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_text = NULL;
    String text;
    PyObject* pyobj_org = NULL;
    Point org;
    PyObject* pyobj_nameFont = NULL;
    String nameFont;
    int pointSize=-1;
    PyObject* pyobj_color = NULL;
    Scalar color=Scalar::all(0);
    int weight=QT_FONT_NORMAL;
    int style=QT_STYLE_NORMAL;
    int spacing=0;

    const char* keywords[] = { "img", "text", "org", "nameFont", "pointSize", "color", "weight", "style", "spacing", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|iOiii:addText", (char**)keywords, &pyobj_img, &pyobj_text, &pyobj_org, &pyobj_nameFont, &pointSize, &pyobj_color, &weight, &style, &spacing) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_text, text, ArgInfo("text", 0)) &&
        pyopencv_to(pyobj_org, org, ArgInfo("org", 0)) &&
        pyopencv_to(pyobj_nameFont, nameFont, ArgInfo("nameFont", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2(cv::addText(img, text, org, nameFont, pointSize, color, weight, style, spacing));
        Py_RETURN_NONE;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_text = NULL;
    String text;
    PyObject* pyobj_org = NULL;
    Point org;
    PyObject* pyobj_nameFont = NULL;
    String nameFont;
    int pointSize=-1;
    PyObject* pyobj_color = NULL;
    Scalar color=Scalar::all(0);
    int weight=QT_FONT_NORMAL;
    int style=QT_STYLE_NORMAL;
    int spacing=0;

    const char* keywords[] = { "img", "text", "org", "nameFont", "pointSize", "color", "weight", "style", "spacing", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|iOiii:addText", (char**)keywords, &pyobj_img, &pyobj_text, &pyobj_org, &pyobj_nameFont, &pointSize, &pyobj_color, &weight, &style, &spacing) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_text, text, ArgInfo("text", 0)) &&
        pyopencv_to(pyobj_org, org, ArgInfo("org", 0)) &&
        pyopencv_to(pyobj_nameFont, nameFont, ArgInfo("nameFont", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2(cv::addText(img, text, org, nameFont, pointSize, color, weight, style, spacing));
        Py_RETURN_NONE;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_addWeighted(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    double alpha=0;
    PyObject* pyobj_src2 = NULL;
    Mat src2;
    double beta=0;
    double gamma=0;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int dtype=-1;

    const char* keywords[] = { "src1", "alpha", "src2", "beta", "gamma", "dst", "dtype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OdOdd|Oi:addWeighted", (char**)keywords, &pyobj_src1, &alpha, &pyobj_src2, &beta, &gamma, &pyobj_dst, &dtype) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::addWeighted(src1, alpha, src2, beta, gamma, dst, dtype));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src1 = NULL;
    UMat src1;
    double alpha=0;
    PyObject* pyobj_src2 = NULL;
    UMat src2;
    double beta=0;
    double gamma=0;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int dtype=-1;

    const char* keywords[] = { "src1", "alpha", "src2", "beta", "gamma", "dst", "dtype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OdOdd|Oi:addWeighted", (char**)keywords, &pyobj_src1, &alpha, &pyobj_src2, &beta, &gamma, &pyobj_dst, &dtype) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::addWeighted(src1, alpha, src2, beta, gamma, dst, dtype));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_applyColorMap(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int colormap=0;

    const char* keywords[] = { "src", "colormap", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|O:applyColorMap", (char**)keywords, &pyobj_src, &colormap, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::applyColorMap(src, dst, colormap));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int colormap=0;

    const char* keywords[] = { "src", "colormap", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|O:applyColorMap", (char**)keywords, &pyobj_src, &colormap, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::applyColorMap(src, dst, colormap));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_approxPolyDP(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_curve = NULL;
    Mat curve;
    PyObject* pyobj_approxCurve = NULL;
    Mat approxCurve;
    double epsilon=0;
    bool closed=0;

    const char* keywords[] = { "curve", "epsilon", "closed", "approxCurve", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odb|O:approxPolyDP", (char**)keywords, &pyobj_curve, &epsilon, &closed, &pyobj_approxCurve) &&
        pyopencv_to(pyobj_curve, curve, ArgInfo("curve", 0)) &&
        pyopencv_to(pyobj_approxCurve, approxCurve, ArgInfo("approxCurve", 1)) )
    {
        ERRWRAP2(cv::approxPolyDP(curve, approxCurve, epsilon, closed));
        return pyopencv_from(approxCurve);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_curve = NULL;
    UMat curve;
    PyObject* pyobj_approxCurve = NULL;
    UMat approxCurve;
    double epsilon=0;
    bool closed=0;

    const char* keywords[] = { "curve", "epsilon", "closed", "approxCurve", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odb|O:approxPolyDP", (char**)keywords, &pyobj_curve, &epsilon, &closed, &pyobj_approxCurve) &&
        pyopencv_to(pyobj_curve, curve, ArgInfo("curve", 0)) &&
        pyopencv_to(pyobj_approxCurve, approxCurve, ArgInfo("approxCurve", 1)) )
    {
        ERRWRAP2(cv::approxPolyDP(curve, approxCurve, epsilon, closed));
        return pyopencv_from(approxCurve);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_arcLength(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_curve = NULL;
    Mat curve;
    bool closed=0;
    double retval;

    const char* keywords[] = { "curve", "closed", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Ob:arcLength", (char**)keywords, &pyobj_curve, &closed) &&
        pyopencv_to(pyobj_curve, curve, ArgInfo("curve", 0)) )
    {
        ERRWRAP2(retval = cv::arcLength(curve, closed));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_curve = NULL;
    UMat curve;
    bool closed=0;
    double retval;

    const char* keywords[] = { "curve", "closed", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Ob:arcLength", (char**)keywords, &pyobj_curve, &closed) &&
        pyopencv_to(pyobj_curve, curve, ArgInfo("curve", 0)) )
    {
        ERRWRAP2(retval = cv::arcLength(curve, closed));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_arrowedLine(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_pt1 = NULL;
    Point pt1;
    PyObject* pyobj_pt2 = NULL;
    Point pt2;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int thickness=1;
    int line_type=8;
    int shift=0;
    double tipLength=0.1;

    const char* keywords[] = { "img", "pt1", "pt2", "color", "thickness", "line_type", "shift", "tipLength", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|iiid:arrowedLine", (char**)keywords, &pyobj_img, &pyobj_pt1, &pyobj_pt2, &pyobj_color, &thickness, &line_type, &shift, &tipLength) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_pt1, pt1, ArgInfo("pt1", 0)) &&
        pyopencv_to(pyobj_pt2, pt2, ArgInfo("pt2", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2(cv::arrowedLine(img, pt1, pt2, color, thickness, line_type, shift, tipLength));
        return pyopencv_from(img);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_img = NULL;
    UMat img;
    PyObject* pyobj_pt1 = NULL;
    Point pt1;
    PyObject* pyobj_pt2 = NULL;
    Point pt2;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int thickness=1;
    int line_type=8;
    int shift=0;
    double tipLength=0.1;

    const char* keywords[] = { "img", "pt1", "pt2", "color", "thickness", "line_type", "shift", "tipLength", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|iiid:arrowedLine", (char**)keywords, &pyobj_img, &pyobj_pt1, &pyobj_pt2, &pyobj_color, &thickness, &line_type, &shift, &tipLength) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_pt1, pt1, ArgInfo("pt1", 0)) &&
        pyopencv_to(pyobj_pt2, pt2, ArgInfo("pt2", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2(cv::arrowedLine(img, pt1, pt2, color, thickness, line_type, shift, tipLength));
        return pyopencv_from(img);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_batchDistance(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    PyObject* pyobj_src2 = NULL;
    Mat src2;
    PyObject* pyobj_dist = NULL;
    Mat dist;
    int dtype=0;
    PyObject* pyobj_nidx = NULL;
    Mat nidx;
    int normType=NORM_L2;
    int K=0;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    int update=0;
    bool crosscheck=false;

    const char* keywords[] = { "src1", "src2", "dtype", "dist", "nidx", "normType", "K", "mask", "update", "crosscheck", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi|OOiiOib:batchDistance", (char**)keywords, &pyobj_src1, &pyobj_src2, &dtype, &pyobj_dist, &pyobj_nidx, &normType, &K, &pyobj_mask, &update, &crosscheck) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dist, dist, ArgInfo("dist", 1)) &&
        pyopencv_to(pyobj_nidx, nidx, ArgInfo("nidx", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::batchDistance(src1, src2, dist, dtype, nidx, normType, K, mask, update, crosscheck));
        return Py_BuildValue("(NN)", pyopencv_from(dist), pyopencv_from(nidx));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src1 = NULL;
    UMat src1;
    PyObject* pyobj_src2 = NULL;
    UMat src2;
    PyObject* pyobj_dist = NULL;
    UMat dist;
    int dtype=0;
    PyObject* pyobj_nidx = NULL;
    UMat nidx;
    int normType=NORM_L2;
    int K=0;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    int update=0;
    bool crosscheck=false;

    const char* keywords[] = { "src1", "src2", "dtype", "dist", "nidx", "normType", "K", "mask", "update", "crosscheck", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi|OOiiOib:batchDistance", (char**)keywords, &pyobj_src1, &pyobj_src2, &dtype, &pyobj_dist, &pyobj_nidx, &normType, &K, &pyobj_mask, &update, &crosscheck) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dist, dist, ArgInfo("dist", 1)) &&
        pyopencv_to(pyobj_nidx, nidx, ArgInfo("nidx", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::batchDistance(src1, src2, dist, dtype, nidx, normType, K, mask, update, crosscheck));
        return Py_BuildValue("(NN)", pyopencv_from(dist), pyopencv_from(nidx));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_bilateralFilter(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int d=0;
    double sigmaColor=0;
    double sigmaSpace=0;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "d", "sigmaColor", "sigmaSpace", "dst", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oidd|Oi:bilateralFilter", (char**)keywords, &pyobj_src, &d, &sigmaColor, &sigmaSpace, &pyobj_dst, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::bilateralFilter(src, dst, d, sigmaColor, sigmaSpace, borderType));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int d=0;
    double sigmaColor=0;
    double sigmaSpace=0;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "d", "sigmaColor", "sigmaSpace", "dst", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oidd|Oi:bilateralFilter", (char**)keywords, &pyobj_src, &d, &sigmaColor, &sigmaSpace, &pyobj_dst, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::bilateralFilter(src, dst, d, sigmaColor, sigmaSpace, borderType));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_bitwise_and(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    PyObject* pyobj_src2 = NULL;
    Mat src2;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_mask = NULL;
    Mat mask;

    const char* keywords[] = { "src1", "src2", "dst", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OO:bitwise_and", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &pyobj_mask) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::bitwise_and(src1, src2, dst, mask));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src1 = NULL;
    UMat src1;
    PyObject* pyobj_src2 = NULL;
    UMat src2;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_mask = NULL;
    UMat mask;

    const char* keywords[] = { "src1", "src2", "dst", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OO:bitwise_and", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &pyobj_mask) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::bitwise_and(src1, src2, dst, mask));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_bitwise_not(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_mask = NULL;
    Mat mask;

    const char* keywords[] = { "src", "dst", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OO:bitwise_not", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_mask) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::bitwise_not(src, dst, mask));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_mask = NULL;
    UMat mask;

    const char* keywords[] = { "src", "dst", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OO:bitwise_not", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_mask) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::bitwise_not(src, dst, mask));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_bitwise_or(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    PyObject* pyobj_src2 = NULL;
    Mat src2;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_mask = NULL;
    Mat mask;

    const char* keywords[] = { "src1", "src2", "dst", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OO:bitwise_or", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &pyobj_mask) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::bitwise_or(src1, src2, dst, mask));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src1 = NULL;
    UMat src1;
    PyObject* pyobj_src2 = NULL;
    UMat src2;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_mask = NULL;
    UMat mask;

    const char* keywords[] = { "src1", "src2", "dst", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OO:bitwise_or", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &pyobj_mask) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::bitwise_or(src1, src2, dst, mask));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_bitwise_xor(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    PyObject* pyobj_src2 = NULL;
    Mat src2;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_mask = NULL;
    Mat mask;

    const char* keywords[] = { "src1", "src2", "dst", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OO:bitwise_xor", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &pyobj_mask) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::bitwise_xor(src1, src2, dst, mask));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src1 = NULL;
    UMat src1;
    PyObject* pyobj_src2 = NULL;
    UMat src2;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_mask = NULL;
    UMat mask;

    const char* keywords[] = { "src1", "src2", "dst", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OO:bitwise_xor", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &pyobj_mask) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::bitwise_xor(src1, src2, dst, mask));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_blur(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_ksize = NULL;
    Size ksize;
    PyObject* pyobj_anchor = NULL;
    Point anchor=Point(-1,-1);
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "ksize", "dst", "anchor", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOi:blur", (char**)keywords, &pyobj_src, &pyobj_ksize, &pyobj_dst, &pyobj_anchor, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_ksize, ksize, ArgInfo("ksize", 0)) &&
        pyopencv_to(pyobj_anchor, anchor, ArgInfo("anchor", 0)) )
    {
        ERRWRAP2(cv::blur(src, dst, ksize, anchor, borderType));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_ksize = NULL;
    Size ksize;
    PyObject* pyobj_anchor = NULL;
    Point anchor=Point(-1,-1);
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "ksize", "dst", "anchor", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOi:blur", (char**)keywords, &pyobj_src, &pyobj_ksize, &pyobj_dst, &pyobj_anchor, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_ksize, ksize, ArgInfo("ksize", 0)) &&
        pyopencv_to(pyobj_anchor, anchor, ArgInfo("anchor", 0)) )
    {
        ERRWRAP2(cv::blur(src, dst, ksize, anchor, borderType));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_borderInterpolate(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int p=0;
    int len=0;
    int borderType=0;
    int retval;

    const char* keywords[] = { "p", "len", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "iii:borderInterpolate", (char**)keywords, &p, &len, &borderType) )
    {
        ERRWRAP2(retval = cv::borderInterpolate(p, len, borderType));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_boundingRect(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_points = NULL;
    Mat points;
    Rect retval;

    const char* keywords[] = { "points", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:boundingRect", (char**)keywords, &pyobj_points) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 0)) )
    {
        ERRWRAP2(retval = cv::boundingRect(points));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_points = NULL;
    UMat points;
    Rect retval;

    const char* keywords[] = { "points", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:boundingRect", (char**)keywords, &pyobj_points) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 0)) )
    {
        ERRWRAP2(retval = cv::boundingRect(points));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_boxFilter(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int ddepth=0;
    PyObject* pyobj_ksize = NULL;
    Size ksize;
    PyObject* pyobj_anchor = NULL;
    Point anchor=Point(-1,-1);
    bool normalize=true;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "ddepth", "ksize", "dst", "anchor", "normalize", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OiO|OObi:boxFilter", (char**)keywords, &pyobj_src, &ddepth, &pyobj_ksize, &pyobj_dst, &pyobj_anchor, &normalize, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_ksize, ksize, ArgInfo("ksize", 0)) &&
        pyopencv_to(pyobj_anchor, anchor, ArgInfo("anchor", 0)) )
    {
        ERRWRAP2(cv::boxFilter(src, dst, ddepth, ksize, anchor, normalize, borderType));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int ddepth=0;
    PyObject* pyobj_ksize = NULL;
    Size ksize;
    PyObject* pyobj_anchor = NULL;
    Point anchor=Point(-1,-1);
    bool normalize=true;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "ddepth", "ksize", "dst", "anchor", "normalize", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OiO|OObi:boxFilter", (char**)keywords, &pyobj_src, &ddepth, &pyobj_ksize, &pyobj_dst, &pyobj_anchor, &normalize, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_ksize, ksize, ArgInfo("ksize", 0)) &&
        pyopencv_to(pyobj_anchor, anchor, ArgInfo("anchor", 0)) )
    {
        ERRWRAP2(cv::boxFilter(src, dst, ddepth, ksize, anchor, normalize, borderType));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_boxPoints(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_box = NULL;
    RotatedRect box;
    PyObject* pyobj_points = NULL;
    Mat points;

    const char* keywords[] = { "box", "points", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:boxPoints", (char**)keywords, &pyobj_box, &pyobj_points) &&
        pyopencv_to(pyobj_box, box, ArgInfo("box", 0)) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 1)) )
    {
        ERRWRAP2(cv::boxPoints(box, points));
        return pyopencv_from(points);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_box = NULL;
    RotatedRect box;
    PyObject* pyobj_points = NULL;
    UMat points;

    const char* keywords[] = { "box", "points", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:boxPoints", (char**)keywords, &pyobj_box, &pyobj_points) &&
        pyopencv_to(pyobj_box, box, ArgInfo("box", 0)) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 1)) )
    {
        ERRWRAP2(cv::boxPoints(box, points));
        return pyopencv_from(points);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_buildOpticalFlowPyramid(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_pyramid = NULL;
    vector_Mat pyramid;
    PyObject* pyobj_winSize = NULL;
    Size winSize;
    int maxLevel=0;
    bool withDerivatives=true;
    int pyrBorder=BORDER_REFLECT_101;
    int derivBorder=BORDER_CONSTANT;
    bool tryReuseInputImage=true;
    int retval;

    const char* keywords[] = { "img", "winSize", "maxLevel", "pyramid", "withDerivatives", "pyrBorder", "derivBorder", "tryReuseInputImage", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi|Obiib:buildOpticalFlowPyramid", (char**)keywords, &pyobj_img, &pyobj_winSize, &maxLevel, &pyobj_pyramid, &withDerivatives, &pyrBorder, &derivBorder, &tryReuseInputImage) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_pyramid, pyramid, ArgInfo("pyramid", 1)) &&
        pyopencv_to(pyobj_winSize, winSize, ArgInfo("winSize", 0)) )
    {
        ERRWRAP2(retval = cv::buildOpticalFlowPyramid(img, pyramid, winSize, maxLevel, withDerivatives, pyrBorder, derivBorder, tryReuseInputImage));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(pyramid));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_img = NULL;
    UMat img;
    PyObject* pyobj_pyramid = NULL;
    vector_Mat pyramid;
    PyObject* pyobj_winSize = NULL;
    Size winSize;
    int maxLevel=0;
    bool withDerivatives=true;
    int pyrBorder=BORDER_REFLECT_101;
    int derivBorder=BORDER_CONSTANT;
    bool tryReuseInputImage=true;
    int retval;

    const char* keywords[] = { "img", "winSize", "maxLevel", "pyramid", "withDerivatives", "pyrBorder", "derivBorder", "tryReuseInputImage", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi|Obiib:buildOpticalFlowPyramid", (char**)keywords, &pyobj_img, &pyobj_winSize, &maxLevel, &pyobj_pyramid, &withDerivatives, &pyrBorder, &derivBorder, &tryReuseInputImage) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_pyramid, pyramid, ArgInfo("pyramid", 1)) &&
        pyopencv_to(pyobj_winSize, winSize, ArgInfo("winSize", 0)) )
    {
        ERRWRAP2(retval = cv::buildOpticalFlowPyramid(img, pyramid, winSize, maxLevel, withDerivatives, pyrBorder, derivBorder, tryReuseInputImage));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(pyramid));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_calcBackProject(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_images = NULL;
    vector_Mat images;
    PyObject* pyobj_channels = NULL;
    vector_int channels;
    PyObject* pyobj_hist = NULL;
    Mat hist;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_ranges = NULL;
    vector_float ranges;
    double scale=0;

    const char* keywords[] = { "images", "channels", "hist", "ranges", "scale", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOd|O:calcBackProject", (char**)keywords, &pyobj_images, &pyobj_channels, &pyobj_hist, &pyobj_ranges, &scale, &pyobj_dst) &&
        pyopencv_to(pyobj_images, images, ArgInfo("images", 0)) &&
        pyopencv_to(pyobj_channels, channels, ArgInfo("channels", 0)) &&
        pyopencv_to(pyobj_hist, hist, ArgInfo("hist", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_ranges, ranges, ArgInfo("ranges", 0)) )
    {
        ERRWRAP2(cv::calcBackProject(images, channels, hist, dst, ranges, scale));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_images = NULL;
    vector_Mat images;
    PyObject* pyobj_channels = NULL;
    vector_int channels;
    PyObject* pyobj_hist = NULL;
    UMat hist;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_ranges = NULL;
    vector_float ranges;
    double scale=0;

    const char* keywords[] = { "images", "channels", "hist", "ranges", "scale", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOd|O:calcBackProject", (char**)keywords, &pyobj_images, &pyobj_channels, &pyobj_hist, &pyobj_ranges, &scale, &pyobj_dst) &&
        pyopencv_to(pyobj_images, images, ArgInfo("images", 0)) &&
        pyopencv_to(pyobj_channels, channels, ArgInfo("channels", 0)) &&
        pyopencv_to(pyobj_hist, hist, ArgInfo("hist", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_ranges, ranges, ArgInfo("ranges", 0)) )
    {
        ERRWRAP2(cv::calcBackProject(images, channels, hist, dst, ranges, scale));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_calcCovarMatrix(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_samples = NULL;
    Mat samples;
    PyObject* pyobj_covar = NULL;
    Mat covar;
    PyObject* pyobj_mean = NULL;
    Mat mean;
    int flags=0;
    int ctype=CV_64F;

    const char* keywords[] = { "samples", "mean", "flags", "covar", "ctype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi|Oi:calcCovarMatrix", (char**)keywords, &pyobj_samples, &pyobj_mean, &flags, &pyobj_covar, &ctype) &&
        pyopencv_to(pyobj_samples, samples, ArgInfo("samples", 0)) &&
        pyopencv_to(pyobj_covar, covar, ArgInfo("covar", 1)) &&
        pyopencv_to(pyobj_mean, mean, ArgInfo("mean", 1)) )
    {
        ERRWRAP2(cv::calcCovarMatrix(samples, covar, mean, flags, ctype));
        return Py_BuildValue("(NN)", pyopencv_from(covar), pyopencv_from(mean));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_samples = NULL;
    UMat samples;
    PyObject* pyobj_covar = NULL;
    UMat covar;
    PyObject* pyobj_mean = NULL;
    UMat mean;
    int flags=0;
    int ctype=CV_64F;

    const char* keywords[] = { "samples", "mean", "flags", "covar", "ctype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi|Oi:calcCovarMatrix", (char**)keywords, &pyobj_samples, &pyobj_mean, &flags, &pyobj_covar, &ctype) &&
        pyopencv_to(pyobj_samples, samples, ArgInfo("samples", 0)) &&
        pyopencv_to(pyobj_covar, covar, ArgInfo("covar", 1)) &&
        pyopencv_to(pyobj_mean, mean, ArgInfo("mean", 1)) )
    {
        ERRWRAP2(cv::calcCovarMatrix(samples, covar, mean, flags, ctype));
        return Py_BuildValue("(NN)", pyopencv_from(covar), pyopencv_from(mean));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_calcHist(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_images = NULL;
    vector_Mat images;
    PyObject* pyobj_channels = NULL;
    vector_int channels;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    PyObject* pyobj_hist = NULL;
    Mat hist;
    PyObject* pyobj_histSize = NULL;
    vector_int histSize;
    PyObject* pyobj_ranges = NULL;
    vector_float ranges;
    bool accumulate=false;

    const char* keywords[] = { "images", "channels", "mask", "histSize", "ranges", "hist", "accumulate", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOO|Ob:calcHist", (char**)keywords, &pyobj_images, &pyobj_channels, &pyobj_mask, &pyobj_histSize, &pyobj_ranges, &pyobj_hist, &accumulate) &&
        pyopencv_to(pyobj_images, images, ArgInfo("images", 0)) &&
        pyopencv_to(pyobj_channels, channels, ArgInfo("channels", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) &&
        pyopencv_to(pyobj_hist, hist, ArgInfo("hist", 1)) &&
        pyopencv_to(pyobj_histSize, histSize, ArgInfo("histSize", 0)) &&
        pyopencv_to(pyobj_ranges, ranges, ArgInfo("ranges", 0)) )
    {
        ERRWRAP2(cv::calcHist(images, channels, mask, hist, histSize, ranges, accumulate));
        return pyopencv_from(hist);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_images = NULL;
    vector_Mat images;
    PyObject* pyobj_channels = NULL;
    vector_int channels;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    PyObject* pyobj_hist = NULL;
    UMat hist;
    PyObject* pyobj_histSize = NULL;
    vector_int histSize;
    PyObject* pyobj_ranges = NULL;
    vector_float ranges;
    bool accumulate=false;

    const char* keywords[] = { "images", "channels", "mask", "histSize", "ranges", "hist", "accumulate", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOO|Ob:calcHist", (char**)keywords, &pyobj_images, &pyobj_channels, &pyobj_mask, &pyobj_histSize, &pyobj_ranges, &pyobj_hist, &accumulate) &&
        pyopencv_to(pyobj_images, images, ArgInfo("images", 0)) &&
        pyopencv_to(pyobj_channels, channels, ArgInfo("channels", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) &&
        pyopencv_to(pyobj_hist, hist, ArgInfo("hist", 1)) &&
        pyopencv_to(pyobj_histSize, histSize, ArgInfo("histSize", 0)) &&
        pyopencv_to(pyobj_ranges, ranges, ArgInfo("ranges", 0)) )
    {
        ERRWRAP2(cv::calcHist(images, channels, mask, hist, histSize, ranges, accumulate));
        return pyopencv_from(hist);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_calcOpticalFlowFarneback(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_prev = NULL;
    Mat prev;
    PyObject* pyobj_next = NULL;
    Mat next;
    PyObject* pyobj_flow = NULL;
    Mat flow;
    double pyr_scale=0;
    int levels=0;
    int winsize=0;
    int iterations=0;
    int poly_n=0;
    double poly_sigma=0;
    int flags=0;

    const char* keywords[] = { "prev", "next", "flow", "pyr_scale", "levels", "winsize", "iterations", "poly_n", "poly_sigma", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOdiiiidi:calcOpticalFlowFarneback", (char**)keywords, &pyobj_prev, &pyobj_next, &pyobj_flow, &pyr_scale, &levels, &winsize, &iterations, &poly_n, &poly_sigma, &flags) &&
        pyopencv_to(pyobj_prev, prev, ArgInfo("prev", 0)) &&
        pyopencv_to(pyobj_next, next, ArgInfo("next", 0)) &&
        pyopencv_to(pyobj_flow, flow, ArgInfo("flow", 1)) )
    {
        ERRWRAP2(cv::calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags));
        return pyopencv_from(flow);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_prev = NULL;
    UMat prev;
    PyObject* pyobj_next = NULL;
    UMat next;
    PyObject* pyobj_flow = NULL;
    UMat flow;
    double pyr_scale=0;
    int levels=0;
    int winsize=0;
    int iterations=0;
    int poly_n=0;
    double poly_sigma=0;
    int flags=0;

    const char* keywords[] = { "prev", "next", "flow", "pyr_scale", "levels", "winsize", "iterations", "poly_n", "poly_sigma", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOdiiiidi:calcOpticalFlowFarneback", (char**)keywords, &pyobj_prev, &pyobj_next, &pyobj_flow, &pyr_scale, &levels, &winsize, &iterations, &poly_n, &poly_sigma, &flags) &&
        pyopencv_to(pyobj_prev, prev, ArgInfo("prev", 0)) &&
        pyopencv_to(pyobj_next, next, ArgInfo("next", 0)) &&
        pyopencv_to(pyobj_flow, flow, ArgInfo("flow", 1)) )
    {
        ERRWRAP2(cv::calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags));
        return pyopencv_from(flow);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_calcOpticalFlowPyrLK(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_prevImg = NULL;
    Mat prevImg;
    PyObject* pyobj_nextImg = NULL;
    Mat nextImg;
    PyObject* pyobj_prevPts = NULL;
    Mat prevPts;
    PyObject* pyobj_nextPts = NULL;
    Mat nextPts;
    PyObject* pyobj_status = NULL;
    Mat status;
    PyObject* pyobj_err = NULL;
    Mat err;
    PyObject* pyobj_winSize = NULL;
    Size winSize=Size(21,21);
    int maxLevel=3;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
    int flags=0;
    double minEigThreshold=1e-4;

    const char* keywords[] = { "prevImg", "nextImg", "prevPts", "nextPts", "status", "err", "winSize", "maxLevel", "criteria", "flags", "minEigThreshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OOOiOid:calcOpticalFlowPyrLK", (char**)keywords, &pyobj_prevImg, &pyobj_nextImg, &pyobj_prevPts, &pyobj_nextPts, &pyobj_status, &pyobj_err, &pyobj_winSize, &maxLevel, &pyobj_criteria, &flags, &minEigThreshold) &&
        pyopencv_to(pyobj_prevImg, prevImg, ArgInfo("prevImg", 0)) &&
        pyopencv_to(pyobj_nextImg, nextImg, ArgInfo("nextImg", 0)) &&
        pyopencv_to(pyobj_prevPts, prevPts, ArgInfo("prevPts", 0)) &&
        pyopencv_to(pyobj_nextPts, nextPts, ArgInfo("nextPts", 1)) &&
        pyopencv_to(pyobj_status, status, ArgInfo("status", 1)) &&
        pyopencv_to(pyobj_err, err, ArgInfo("err", 1)) &&
        pyopencv_to(pyobj_winSize, winSize, ArgInfo("winSize", 0)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(cv::calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, status, err, winSize, maxLevel, criteria, flags, minEigThreshold));
        return Py_BuildValue("(NNN)", pyopencv_from(nextPts), pyopencv_from(status), pyopencv_from(err));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_prevImg = NULL;
    UMat prevImg;
    PyObject* pyobj_nextImg = NULL;
    UMat nextImg;
    PyObject* pyobj_prevPts = NULL;
    UMat prevPts;
    PyObject* pyobj_nextPts = NULL;
    UMat nextPts;
    PyObject* pyobj_status = NULL;
    UMat status;
    PyObject* pyobj_err = NULL;
    UMat err;
    PyObject* pyobj_winSize = NULL;
    Size winSize=Size(21,21);
    int maxLevel=3;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
    int flags=0;
    double minEigThreshold=1e-4;

    const char* keywords[] = { "prevImg", "nextImg", "prevPts", "nextPts", "status", "err", "winSize", "maxLevel", "criteria", "flags", "minEigThreshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OOOiOid:calcOpticalFlowPyrLK", (char**)keywords, &pyobj_prevImg, &pyobj_nextImg, &pyobj_prevPts, &pyobj_nextPts, &pyobj_status, &pyobj_err, &pyobj_winSize, &maxLevel, &pyobj_criteria, &flags, &minEigThreshold) &&
        pyopencv_to(pyobj_prevImg, prevImg, ArgInfo("prevImg", 0)) &&
        pyopencv_to(pyobj_nextImg, nextImg, ArgInfo("nextImg", 0)) &&
        pyopencv_to(pyobj_prevPts, prevPts, ArgInfo("prevPts", 0)) &&
        pyopencv_to(pyobj_nextPts, nextPts, ArgInfo("nextPts", 1)) &&
        pyopencv_to(pyobj_status, status, ArgInfo("status", 1)) &&
        pyopencv_to(pyobj_err, err, ArgInfo("err", 1)) &&
        pyopencv_to(pyobj_winSize, winSize, ArgInfo("winSize", 0)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(cv::calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, status, err, winSize, maxLevel, criteria, flags, minEigThreshold));
        return Py_BuildValue("(NNN)", pyopencv_from(nextPts), pyopencv_from(status), pyopencv_from(err));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_calibrateCamera(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_objectPoints = NULL;
    vector_Mat objectPoints;
    PyObject* pyobj_imagePoints = NULL;
    vector_Mat imagePoints;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    Mat distCoeffs;
    PyObject* pyobj_rvecs = NULL;
    vector_Mat rvecs;
    PyObject* pyobj_tvecs = NULL;
    vector_Mat tvecs;
    int flags=0;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria=TermCriteria( TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON);
    double retval;

    const char* keywords[] = { "objectPoints", "imagePoints", "imageSize", "cameraMatrix", "distCoeffs", "rvecs", "tvecs", "flags", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOO|OOiO:calibrateCamera", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints, &pyobj_imageSize, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvecs, &pyobj_tvecs, &flags, &pyobj_criteria) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints, imagePoints, ArgInfo("imagePoints", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 1)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 1)) &&
        pyopencv_to(pyobj_rvecs, rvecs, ArgInfo("rvecs", 1)) &&
        pyopencv_to(pyobj_tvecs, tvecs, ArgInfo("tvecs", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(retval = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags, criteria));
        return Py_BuildValue("(NNNNN)", pyopencv_from(retval), pyopencv_from(cameraMatrix), pyopencv_from(distCoeffs), pyopencv_from(rvecs), pyopencv_from(tvecs));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_objectPoints = NULL;
    vector_Mat objectPoints;
    PyObject* pyobj_imagePoints = NULL;
    vector_Mat imagePoints;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    UMat distCoeffs;
    PyObject* pyobj_rvecs = NULL;
    vector_Mat rvecs;
    PyObject* pyobj_tvecs = NULL;
    vector_Mat tvecs;
    int flags=0;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria=TermCriteria( TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON);
    double retval;

    const char* keywords[] = { "objectPoints", "imagePoints", "imageSize", "cameraMatrix", "distCoeffs", "rvecs", "tvecs", "flags", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOO|OOiO:calibrateCamera", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints, &pyobj_imageSize, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvecs, &pyobj_tvecs, &flags, &pyobj_criteria) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints, imagePoints, ArgInfo("imagePoints", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 1)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 1)) &&
        pyopencv_to(pyobj_rvecs, rvecs, ArgInfo("rvecs", 1)) &&
        pyopencv_to(pyobj_tvecs, tvecs, ArgInfo("tvecs", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(retval = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags, criteria));
        return Py_BuildValue("(NNNNN)", pyopencv_from(retval), pyopencv_from(cameraMatrix), pyopencv_from(distCoeffs), pyopencv_from(rvecs), pyopencv_from(tvecs));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_calibrateCameraExtended(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_objectPoints = NULL;
    vector_Mat objectPoints;
    PyObject* pyobj_imagePoints = NULL;
    vector_Mat imagePoints;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    Mat distCoeffs;
    PyObject* pyobj_rvecs = NULL;
    vector_Mat rvecs;
    PyObject* pyobj_tvecs = NULL;
    vector_Mat tvecs;
    PyObject* pyobj_stdDeviationsIntrinsics = NULL;
    Mat stdDeviationsIntrinsics;
    PyObject* pyobj_stdDeviationsExtrinsics = NULL;
    Mat stdDeviationsExtrinsics;
    PyObject* pyobj_perViewErrors = NULL;
    Mat perViewErrors;
    int flags=0;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria=TermCriteria( TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON);
    double retval;

    const char* keywords[] = { "objectPoints", "imagePoints", "imageSize", "cameraMatrix", "distCoeffs", "rvecs", "tvecs", "stdDeviationsIntrinsics", "stdDeviationsExtrinsics", "perViewErrors", "flags", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOO|OOOOOiO:calibrateCameraExtended", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints, &pyobj_imageSize, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvecs, &pyobj_tvecs, &pyobj_stdDeviationsIntrinsics, &pyobj_stdDeviationsExtrinsics, &pyobj_perViewErrors, &flags, &pyobj_criteria) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints, imagePoints, ArgInfo("imagePoints", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 1)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 1)) &&
        pyopencv_to(pyobj_rvecs, rvecs, ArgInfo("rvecs", 1)) &&
        pyopencv_to(pyobj_tvecs, tvecs, ArgInfo("tvecs", 1)) &&
        pyopencv_to(pyobj_stdDeviationsIntrinsics, stdDeviationsIntrinsics, ArgInfo("stdDeviationsIntrinsics", 1)) &&
        pyopencv_to(pyobj_stdDeviationsExtrinsics, stdDeviationsExtrinsics, ArgInfo("stdDeviationsExtrinsics", 1)) &&
        pyopencv_to(pyobj_perViewErrors, perViewErrors, ArgInfo("perViewErrors", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(retval = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors, flags, criteria));
        return Py_BuildValue("(NNNNNNNN)", pyopencv_from(retval), pyopencv_from(cameraMatrix), pyopencv_from(distCoeffs), pyopencv_from(rvecs), pyopencv_from(tvecs), pyopencv_from(stdDeviationsIntrinsics), pyopencv_from(stdDeviationsExtrinsics), pyopencv_from(perViewErrors));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_objectPoints = NULL;
    vector_Mat objectPoints;
    PyObject* pyobj_imagePoints = NULL;
    vector_Mat imagePoints;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    UMat distCoeffs;
    PyObject* pyobj_rvecs = NULL;
    vector_Mat rvecs;
    PyObject* pyobj_tvecs = NULL;
    vector_Mat tvecs;
    PyObject* pyobj_stdDeviationsIntrinsics = NULL;
    UMat stdDeviationsIntrinsics;
    PyObject* pyobj_stdDeviationsExtrinsics = NULL;
    UMat stdDeviationsExtrinsics;
    PyObject* pyobj_perViewErrors = NULL;
    UMat perViewErrors;
    int flags=0;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria=TermCriteria( TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON);
    double retval;

    const char* keywords[] = { "objectPoints", "imagePoints", "imageSize", "cameraMatrix", "distCoeffs", "rvecs", "tvecs", "stdDeviationsIntrinsics", "stdDeviationsExtrinsics", "perViewErrors", "flags", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOO|OOOOOiO:calibrateCameraExtended", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints, &pyobj_imageSize, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvecs, &pyobj_tvecs, &pyobj_stdDeviationsIntrinsics, &pyobj_stdDeviationsExtrinsics, &pyobj_perViewErrors, &flags, &pyobj_criteria) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints, imagePoints, ArgInfo("imagePoints", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 1)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 1)) &&
        pyopencv_to(pyobj_rvecs, rvecs, ArgInfo("rvecs", 1)) &&
        pyopencv_to(pyobj_tvecs, tvecs, ArgInfo("tvecs", 1)) &&
        pyopencv_to(pyobj_stdDeviationsIntrinsics, stdDeviationsIntrinsics, ArgInfo("stdDeviationsIntrinsics", 1)) &&
        pyopencv_to(pyobj_stdDeviationsExtrinsics, stdDeviationsExtrinsics, ArgInfo("stdDeviationsExtrinsics", 1)) &&
        pyopencv_to(pyobj_perViewErrors, perViewErrors, ArgInfo("perViewErrors", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(retval = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors, flags, criteria));
        return Py_BuildValue("(NNNNNNNN)", pyopencv_from(retval), pyopencv_from(cameraMatrix), pyopencv_from(distCoeffs), pyopencv_from(rvecs), pyopencv_from(tvecs), pyopencv_from(stdDeviationsIntrinsics), pyopencv_from(stdDeviationsExtrinsics), pyopencv_from(perViewErrors));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_calibrationMatrixValues(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    double apertureWidth=0;
    double apertureHeight=0;
    double fovx;
    double fovy;
    double focalLength;
    Point2d principalPoint;
    double aspectRatio;

    const char* keywords[] = { "cameraMatrix", "imageSize", "apertureWidth", "apertureHeight", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOdd:calibrationMatrixValues", (char**)keywords, &pyobj_cameraMatrix, &pyobj_imageSize, &apertureWidth, &apertureHeight) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) )
    {
        ERRWRAP2(cv::calibrationMatrixValues(cameraMatrix, imageSize, apertureWidth, apertureHeight, fovx, fovy, focalLength, principalPoint, aspectRatio));
        return Py_BuildValue("(NNNNN)", pyopencv_from(fovx), pyopencv_from(fovy), pyopencv_from(focalLength), pyopencv_from(principalPoint), pyopencv_from(aspectRatio));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    double apertureWidth=0;
    double apertureHeight=0;
    double fovx;
    double fovy;
    double focalLength;
    Point2d principalPoint;
    double aspectRatio;

    const char* keywords[] = { "cameraMatrix", "imageSize", "apertureWidth", "apertureHeight", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOdd:calibrationMatrixValues", (char**)keywords, &pyobj_cameraMatrix, &pyobj_imageSize, &apertureWidth, &apertureHeight) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) )
    {
        ERRWRAP2(cv::calibrationMatrixValues(cameraMatrix, imageSize, apertureWidth, apertureHeight, fovx, fovy, focalLength, principalPoint, aspectRatio));
        return Py_BuildValue("(NNNNN)", pyopencv_from(fovx), pyopencv_from(fovy), pyopencv_from(focalLength), pyopencv_from(principalPoint), pyopencv_from(aspectRatio));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_cartToPolar(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_x = NULL;
    Mat x;
    PyObject* pyobj_y = NULL;
    Mat y;
    PyObject* pyobj_magnitude = NULL;
    Mat magnitude;
    PyObject* pyobj_angle = NULL;
    Mat angle;
    bool angleInDegrees=false;

    const char* keywords[] = { "x", "y", "magnitude", "angle", "angleInDegrees", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOb:cartToPolar", (char**)keywords, &pyobj_x, &pyobj_y, &pyobj_magnitude, &pyobj_angle, &angleInDegrees) &&
        pyopencv_to(pyobj_x, x, ArgInfo("x", 0)) &&
        pyopencv_to(pyobj_y, y, ArgInfo("y", 0)) &&
        pyopencv_to(pyobj_magnitude, magnitude, ArgInfo("magnitude", 1)) &&
        pyopencv_to(pyobj_angle, angle, ArgInfo("angle", 1)) )
    {
        ERRWRAP2(cv::cartToPolar(x, y, magnitude, angle, angleInDegrees));
        return Py_BuildValue("(NN)", pyopencv_from(magnitude), pyopencv_from(angle));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_x = NULL;
    UMat x;
    PyObject* pyobj_y = NULL;
    UMat y;
    PyObject* pyobj_magnitude = NULL;
    UMat magnitude;
    PyObject* pyobj_angle = NULL;
    UMat angle;
    bool angleInDegrees=false;

    const char* keywords[] = { "x", "y", "magnitude", "angle", "angleInDegrees", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOb:cartToPolar", (char**)keywords, &pyobj_x, &pyobj_y, &pyobj_magnitude, &pyobj_angle, &angleInDegrees) &&
        pyopencv_to(pyobj_x, x, ArgInfo("x", 0)) &&
        pyopencv_to(pyobj_y, y, ArgInfo("y", 0)) &&
        pyopencv_to(pyobj_magnitude, magnitude, ArgInfo("magnitude", 1)) &&
        pyopencv_to(pyobj_angle, angle, ArgInfo("angle", 1)) )
    {
        ERRWRAP2(cv::cartToPolar(x, y, magnitude, angle, angleInDegrees));
        return Py_BuildValue("(NN)", pyopencv_from(magnitude), pyopencv_from(angle));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_checkHardwareSupport(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int feature=0;
    bool retval;

    const char* keywords[] = { "feature", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:checkHardwareSupport", (char**)keywords, &feature) )
    {
        ERRWRAP2(retval = cv::checkHardwareSupport(feature));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_checkRange(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_a = NULL;
    Mat a;
    bool quiet=true;
    Point pos;
    double minVal=-DBL_MAX;
    double maxVal=DBL_MAX;
    bool retval;

    const char* keywords[] = { "a", "quiet", "minVal", "maxVal", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|bdd:checkRange", (char**)keywords, &pyobj_a, &quiet, &minVal, &maxVal) &&
        pyopencv_to(pyobj_a, a, ArgInfo("a", 0)) )
    {
        ERRWRAP2(retval = cv::checkRange(a, quiet, &pos, minVal, maxVal));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(pos));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_a = NULL;
    UMat a;
    bool quiet=true;
    Point pos;
    double minVal=-DBL_MAX;
    double maxVal=DBL_MAX;
    bool retval;

    const char* keywords[] = { "a", "quiet", "minVal", "maxVal", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|bdd:checkRange", (char**)keywords, &pyobj_a, &quiet, &minVal, &maxVal) &&
        pyopencv_to(pyobj_a, a, ArgInfo("a", 0)) )
    {
        ERRWRAP2(retval = cv::checkRange(a, quiet, &pos, minVal, maxVal));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(pos));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_circle(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_center = NULL;
    Point center;
    int radius=0;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int thickness=1;
    int lineType=LINE_8;
    int shift=0;

    const char* keywords[] = { "img", "center", "radius", "color", "thickness", "lineType", "shift", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOiO|iii:circle", (char**)keywords, &pyobj_img, &pyobj_center, &radius, &pyobj_color, &thickness, &lineType, &shift) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_center, center, ArgInfo("center", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2(cv::circle(img, center, radius, color, thickness, lineType, shift));
        return pyopencv_from(img);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_img = NULL;
    UMat img;
    PyObject* pyobj_center = NULL;
    Point center;
    int radius=0;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int thickness=1;
    int lineType=LINE_8;
    int shift=0;

    const char* keywords[] = { "img", "center", "radius", "color", "thickness", "lineType", "shift", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOiO|iii:circle", (char**)keywords, &pyobj_img, &pyobj_center, &radius, &pyobj_color, &thickness, &lineType, &shift) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_center, center, ArgInfo("center", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2(cv::circle(img, center, radius, color, thickness, lineType, shift));
        return pyopencv_from(img);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_clipLine(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_imgRect = NULL;
    Rect imgRect;
    PyObject* pyobj_pt1 = NULL;
    Point pt1;
    PyObject* pyobj_pt2 = NULL;
    Point pt2;
    bool retval;

    const char* keywords[] = { "imgRect", "pt1", "pt2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:clipLine", (char**)keywords, &pyobj_imgRect, &pyobj_pt1, &pyobj_pt2) &&
        pyopencv_to(pyobj_imgRect, imgRect, ArgInfo("imgRect", 0)) &&
        pyopencv_to(pyobj_pt1, pt1, ArgInfo("pt1", 1)) &&
        pyopencv_to(pyobj_pt2, pt2, ArgInfo("pt2", 1)) )
    {
        ERRWRAP2(retval = cv::clipLine(imgRect, pt1, pt2));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(pt1), pyopencv_from(pt2));
    }

    return NULL;
}

static PyObject* pyopencv_cv_colorChange(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    float red_mul=1.0f;
    float green_mul=1.0f;
    float blue_mul=1.0f;

    const char* keywords[] = { "src", "mask", "dst", "red_mul", "green_mul", "blue_mul", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Offf:colorChange", (char**)keywords, &pyobj_src, &pyobj_mask, &pyobj_dst, &red_mul, &green_mul, &blue_mul) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::colorChange(src, mask, dst, red_mul, green_mul, blue_mul));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    float red_mul=1.0f;
    float green_mul=1.0f;
    float blue_mul=1.0f;

    const char* keywords[] = { "src", "mask", "dst", "red_mul", "green_mul", "blue_mul", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Offf:colorChange", (char**)keywords, &pyobj_src, &pyobj_mask, &pyobj_dst, &red_mul, &green_mul, &blue_mul) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::colorChange(src, mask, dst, red_mul, green_mul, blue_mul));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_compare(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    PyObject* pyobj_src2 = NULL;
    Mat src2;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int cmpop=0;

    const char* keywords[] = { "src1", "src2", "cmpop", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi|O:compare", (char**)keywords, &pyobj_src1, &pyobj_src2, &cmpop, &pyobj_dst) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::compare(src1, src2, dst, cmpop));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src1 = NULL;
    UMat src1;
    PyObject* pyobj_src2 = NULL;
    UMat src2;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int cmpop=0;

    const char* keywords[] = { "src1", "src2", "cmpop", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi|O:compare", (char**)keywords, &pyobj_src1, &pyobj_src2, &cmpop, &pyobj_dst) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::compare(src1, src2, dst, cmpop));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_compareHist(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_H1 = NULL;
    Mat H1;
    PyObject* pyobj_H2 = NULL;
    Mat H2;
    int method=0;
    double retval;

    const char* keywords[] = { "H1", "H2", "method", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi:compareHist", (char**)keywords, &pyobj_H1, &pyobj_H2, &method) &&
        pyopencv_to(pyobj_H1, H1, ArgInfo("H1", 0)) &&
        pyopencv_to(pyobj_H2, H2, ArgInfo("H2", 0)) )
    {
        ERRWRAP2(retval = cv::compareHist(H1, H2, method));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_H1 = NULL;
    UMat H1;
    PyObject* pyobj_H2 = NULL;
    UMat H2;
    int method=0;
    double retval;

    const char* keywords[] = { "H1", "H2", "method", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi:compareHist", (char**)keywords, &pyobj_H1, &pyobj_H2, &method) &&
        pyopencv_to(pyobj_H1, H1, ArgInfo("H1", 0)) &&
        pyopencv_to(pyobj_H2, H2, ArgInfo("H2", 0)) )
    {
        ERRWRAP2(retval = cv::compareHist(H1, H2, method));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_completeSymm(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_mtx = NULL;
    Mat mtx;
    bool lowerToUpper=false;

    const char* keywords[] = { "mtx", "lowerToUpper", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|b:completeSymm", (char**)keywords, &pyobj_mtx, &lowerToUpper) &&
        pyopencv_to(pyobj_mtx, mtx, ArgInfo("mtx", 1)) )
    {
        ERRWRAP2(cv::completeSymm(mtx, lowerToUpper));
        return pyopencv_from(mtx);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_mtx = NULL;
    UMat mtx;
    bool lowerToUpper=false;

    const char* keywords[] = { "mtx", "lowerToUpper", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|b:completeSymm", (char**)keywords, &pyobj_mtx, &lowerToUpper) &&
        pyopencv_to(pyobj_mtx, mtx, ArgInfo("mtx", 1)) )
    {
        ERRWRAP2(cv::completeSymm(mtx, lowerToUpper));
        return pyopencv_from(mtx);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_composeRT(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_rvec1 = NULL;
    Mat rvec1;
    PyObject* pyobj_tvec1 = NULL;
    Mat tvec1;
    PyObject* pyobj_rvec2 = NULL;
    Mat rvec2;
    PyObject* pyobj_tvec2 = NULL;
    Mat tvec2;
    PyObject* pyobj_rvec3 = NULL;
    Mat rvec3;
    PyObject* pyobj_tvec3 = NULL;
    Mat tvec3;
    PyObject* pyobj_dr3dr1 = NULL;
    Mat dr3dr1;
    PyObject* pyobj_dr3dt1 = NULL;
    Mat dr3dt1;
    PyObject* pyobj_dr3dr2 = NULL;
    Mat dr3dr2;
    PyObject* pyobj_dr3dt2 = NULL;
    Mat dr3dt2;
    PyObject* pyobj_dt3dr1 = NULL;
    Mat dt3dr1;
    PyObject* pyobj_dt3dt1 = NULL;
    Mat dt3dt1;
    PyObject* pyobj_dt3dr2 = NULL;
    Mat dt3dr2;
    PyObject* pyobj_dt3dt2 = NULL;
    Mat dt3dt2;

    const char* keywords[] = { "rvec1", "tvec1", "rvec2", "tvec2", "rvec3", "tvec3", "dr3dr1", "dr3dt1", "dr3dr2", "dr3dt2", "dt3dr1", "dt3dt1", "dt3dr2", "dt3dt2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OOOOOOOOOO:composeRT", (char**)keywords, &pyobj_rvec1, &pyobj_tvec1, &pyobj_rvec2, &pyobj_tvec2, &pyobj_rvec3, &pyobj_tvec3, &pyobj_dr3dr1, &pyobj_dr3dt1, &pyobj_dr3dr2, &pyobj_dr3dt2, &pyobj_dt3dr1, &pyobj_dt3dt1, &pyobj_dt3dr2, &pyobj_dt3dt2) &&
        pyopencv_to(pyobj_rvec1, rvec1, ArgInfo("rvec1", 0)) &&
        pyopencv_to(pyobj_tvec1, tvec1, ArgInfo("tvec1", 0)) &&
        pyopencv_to(pyobj_rvec2, rvec2, ArgInfo("rvec2", 0)) &&
        pyopencv_to(pyobj_tvec2, tvec2, ArgInfo("tvec2", 0)) &&
        pyopencv_to(pyobj_rvec3, rvec3, ArgInfo("rvec3", 1)) &&
        pyopencv_to(pyobj_tvec3, tvec3, ArgInfo("tvec3", 1)) &&
        pyopencv_to(pyobj_dr3dr1, dr3dr1, ArgInfo("dr3dr1", 1)) &&
        pyopencv_to(pyobj_dr3dt1, dr3dt1, ArgInfo("dr3dt1", 1)) &&
        pyopencv_to(pyobj_dr3dr2, dr3dr2, ArgInfo("dr3dr2", 1)) &&
        pyopencv_to(pyobj_dr3dt2, dr3dt2, ArgInfo("dr3dt2", 1)) &&
        pyopencv_to(pyobj_dt3dr1, dt3dr1, ArgInfo("dt3dr1", 1)) &&
        pyopencv_to(pyobj_dt3dt1, dt3dt1, ArgInfo("dt3dt1", 1)) &&
        pyopencv_to(pyobj_dt3dr2, dt3dr2, ArgInfo("dt3dr2", 1)) &&
        pyopencv_to(pyobj_dt3dt2, dt3dt2, ArgInfo("dt3dt2", 1)) )
    {
        ERRWRAP2(cv::composeRT(rvec1, tvec1, rvec2, tvec2, rvec3, tvec3, dr3dr1, dr3dt1, dr3dr2, dr3dt2, dt3dr1, dt3dt1, dt3dr2, dt3dt2));
        return Py_BuildValue("(NNNNNNNNNN)", pyopencv_from(rvec3), pyopencv_from(tvec3), pyopencv_from(dr3dr1), pyopencv_from(dr3dt1), pyopencv_from(dr3dr2), pyopencv_from(dr3dt2), pyopencv_from(dt3dr1), pyopencv_from(dt3dt1), pyopencv_from(dt3dr2), pyopencv_from(dt3dt2));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_rvec1 = NULL;
    UMat rvec1;
    PyObject* pyobj_tvec1 = NULL;
    UMat tvec1;
    PyObject* pyobj_rvec2 = NULL;
    UMat rvec2;
    PyObject* pyobj_tvec2 = NULL;
    UMat tvec2;
    PyObject* pyobj_rvec3 = NULL;
    UMat rvec3;
    PyObject* pyobj_tvec3 = NULL;
    UMat tvec3;
    PyObject* pyobj_dr3dr1 = NULL;
    UMat dr3dr1;
    PyObject* pyobj_dr3dt1 = NULL;
    UMat dr3dt1;
    PyObject* pyobj_dr3dr2 = NULL;
    UMat dr3dr2;
    PyObject* pyobj_dr3dt2 = NULL;
    UMat dr3dt2;
    PyObject* pyobj_dt3dr1 = NULL;
    UMat dt3dr1;
    PyObject* pyobj_dt3dt1 = NULL;
    UMat dt3dt1;
    PyObject* pyobj_dt3dr2 = NULL;
    UMat dt3dr2;
    PyObject* pyobj_dt3dt2 = NULL;
    UMat dt3dt2;

    const char* keywords[] = { "rvec1", "tvec1", "rvec2", "tvec2", "rvec3", "tvec3", "dr3dr1", "dr3dt1", "dr3dr2", "dr3dt2", "dt3dr1", "dt3dt1", "dt3dr2", "dt3dt2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OOOOOOOOOO:composeRT", (char**)keywords, &pyobj_rvec1, &pyobj_tvec1, &pyobj_rvec2, &pyobj_tvec2, &pyobj_rvec3, &pyobj_tvec3, &pyobj_dr3dr1, &pyobj_dr3dt1, &pyobj_dr3dr2, &pyobj_dr3dt2, &pyobj_dt3dr1, &pyobj_dt3dt1, &pyobj_dt3dr2, &pyobj_dt3dt2) &&
        pyopencv_to(pyobj_rvec1, rvec1, ArgInfo("rvec1", 0)) &&
        pyopencv_to(pyobj_tvec1, tvec1, ArgInfo("tvec1", 0)) &&
        pyopencv_to(pyobj_rvec2, rvec2, ArgInfo("rvec2", 0)) &&
        pyopencv_to(pyobj_tvec2, tvec2, ArgInfo("tvec2", 0)) &&
        pyopencv_to(pyobj_rvec3, rvec3, ArgInfo("rvec3", 1)) &&
        pyopencv_to(pyobj_tvec3, tvec3, ArgInfo("tvec3", 1)) &&
        pyopencv_to(pyobj_dr3dr1, dr3dr1, ArgInfo("dr3dr1", 1)) &&
        pyopencv_to(pyobj_dr3dt1, dr3dt1, ArgInfo("dr3dt1", 1)) &&
        pyopencv_to(pyobj_dr3dr2, dr3dr2, ArgInfo("dr3dr2", 1)) &&
        pyopencv_to(pyobj_dr3dt2, dr3dt2, ArgInfo("dr3dt2", 1)) &&
        pyopencv_to(pyobj_dt3dr1, dt3dr1, ArgInfo("dt3dr1", 1)) &&
        pyopencv_to(pyobj_dt3dt1, dt3dt1, ArgInfo("dt3dt1", 1)) &&
        pyopencv_to(pyobj_dt3dr2, dt3dr2, ArgInfo("dt3dr2", 1)) &&
        pyopencv_to(pyobj_dt3dt2, dt3dt2, ArgInfo("dt3dt2", 1)) )
    {
        ERRWRAP2(cv::composeRT(rvec1, tvec1, rvec2, tvec2, rvec3, tvec3, dr3dr1, dr3dt1, dr3dr2, dr3dt2, dt3dr1, dt3dt1, dt3dr2, dt3dt2));
        return Py_BuildValue("(NNNNNNNNNN)", pyopencv_from(rvec3), pyopencv_from(tvec3), pyopencv_from(dr3dr1), pyopencv_from(dr3dt1), pyopencv_from(dr3dr2), pyopencv_from(dr3dt2), pyopencv_from(dt3dr1), pyopencv_from(dt3dt1), pyopencv_from(dt3dr2), pyopencv_from(dt3dt2));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_computeCorrespondEpilines(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_points = NULL;
    Mat points;
    int whichImage=0;
    PyObject* pyobj_F = NULL;
    Mat F;
    PyObject* pyobj_lines = NULL;
    Mat lines;

    const char* keywords[] = { "points", "whichImage", "F", "lines", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OiO|O:computeCorrespondEpilines", (char**)keywords, &pyobj_points, &whichImage, &pyobj_F, &pyobj_lines) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 0)) &&
        pyopencv_to(pyobj_F, F, ArgInfo("F", 0)) &&
        pyopencv_to(pyobj_lines, lines, ArgInfo("lines", 1)) )
    {
        ERRWRAP2(cv::computeCorrespondEpilines(points, whichImage, F, lines));
        return pyopencv_from(lines);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_points = NULL;
    UMat points;
    int whichImage=0;
    PyObject* pyobj_F = NULL;
    UMat F;
    PyObject* pyobj_lines = NULL;
    UMat lines;

    const char* keywords[] = { "points", "whichImage", "F", "lines", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OiO|O:computeCorrespondEpilines", (char**)keywords, &pyobj_points, &whichImage, &pyobj_F, &pyobj_lines) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 0)) &&
        pyopencv_to(pyobj_F, F, ArgInfo("F", 0)) &&
        pyopencv_to(pyobj_lines, lines, ArgInfo("lines", 1)) )
    {
        ERRWRAP2(cv::computeCorrespondEpilines(points, whichImage, F, lines));
        return pyopencv_from(lines);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_connectedComponents(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_labels = NULL;
    Mat labels;
    int connectivity=8;
    int ltype=CV_32S;
    int retval;

    const char* keywords[] = { "image", "labels", "connectivity", "ltype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oii:connectedComponents", (char**)keywords, &pyobj_image, &pyobj_labels, &connectivity, &ltype) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_labels, labels, ArgInfo("labels", 1)) )
    {
        ERRWRAP2(retval = cv::connectedComponents(image, labels, connectivity, ltype));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(labels));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_labels = NULL;
    UMat labels;
    int connectivity=8;
    int ltype=CV_32S;
    int retval;

    const char* keywords[] = { "image", "labels", "connectivity", "ltype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oii:connectedComponents", (char**)keywords, &pyobj_image, &pyobj_labels, &connectivity, &ltype) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_labels, labels, ArgInfo("labels", 1)) )
    {
        ERRWRAP2(retval = cv::connectedComponents(image, labels, connectivity, ltype));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(labels));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_connectedComponentsWithAlgorithm(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_labels = NULL;
    Mat labels;
    int connectivity=0;
    int ltype=0;
    int ccltype=0;
    int retval;

    const char* keywords[] = { "image", "connectivity", "ltype", "ccltype", "labels", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oiii|O:connectedComponentsWithAlgorithm", (char**)keywords, &pyobj_image, &connectivity, &ltype, &ccltype, &pyobj_labels) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_labels, labels, ArgInfo("labels", 1)) )
    {
        ERRWRAP2(retval = cv::connectedComponents(image, labels, connectivity, ltype, ccltype));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(labels));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_labels = NULL;
    UMat labels;
    int connectivity=0;
    int ltype=0;
    int ccltype=0;
    int retval;

    const char* keywords[] = { "image", "connectivity", "ltype", "ccltype", "labels", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oiii|O:connectedComponentsWithAlgorithm", (char**)keywords, &pyobj_image, &connectivity, &ltype, &ccltype, &pyobj_labels) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_labels, labels, ArgInfo("labels", 1)) )
    {
        ERRWRAP2(retval = cv::connectedComponents(image, labels, connectivity, ltype, ccltype));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(labels));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_connectedComponentsWithStats(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_labels = NULL;
    Mat labels;
    PyObject* pyobj_stats = NULL;
    Mat stats;
    PyObject* pyobj_centroids = NULL;
    Mat centroids;
    int connectivity=8;
    int ltype=CV_32S;
    int retval;

    const char* keywords[] = { "image", "labels", "stats", "centroids", "connectivity", "ltype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOOii:connectedComponentsWithStats", (char**)keywords, &pyobj_image, &pyobj_labels, &pyobj_stats, &pyobj_centroids, &connectivity, &ltype) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_labels, labels, ArgInfo("labels", 1)) &&
        pyopencv_to(pyobj_stats, stats, ArgInfo("stats", 1)) &&
        pyopencv_to(pyobj_centroids, centroids, ArgInfo("centroids", 1)) )
    {
        ERRWRAP2(retval = cv::connectedComponentsWithStats(image, labels, stats, centroids, connectivity, ltype));
        return Py_BuildValue("(NNNN)", pyopencv_from(retval), pyopencv_from(labels), pyopencv_from(stats), pyopencv_from(centroids));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_labels = NULL;
    UMat labels;
    PyObject* pyobj_stats = NULL;
    UMat stats;
    PyObject* pyobj_centroids = NULL;
    UMat centroids;
    int connectivity=8;
    int ltype=CV_32S;
    int retval;

    const char* keywords[] = { "image", "labels", "stats", "centroids", "connectivity", "ltype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOOii:connectedComponentsWithStats", (char**)keywords, &pyobj_image, &pyobj_labels, &pyobj_stats, &pyobj_centroids, &connectivity, &ltype) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_labels, labels, ArgInfo("labels", 1)) &&
        pyopencv_to(pyobj_stats, stats, ArgInfo("stats", 1)) &&
        pyopencv_to(pyobj_centroids, centroids, ArgInfo("centroids", 1)) )
    {
        ERRWRAP2(retval = cv::connectedComponentsWithStats(image, labels, stats, centroids, connectivity, ltype));
        return Py_BuildValue("(NNNN)", pyopencv_from(retval), pyopencv_from(labels), pyopencv_from(stats), pyopencv_from(centroids));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_connectedComponentsWithStatsWithAlgorithm(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_labels = NULL;
    Mat labels;
    PyObject* pyobj_stats = NULL;
    Mat stats;
    PyObject* pyobj_centroids = NULL;
    Mat centroids;
    int connectivity=0;
    int ltype=0;
    int ccltype=0;
    int retval;

    const char* keywords[] = { "image", "connectivity", "ltype", "ccltype", "labels", "stats", "centroids", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oiii|OOO:connectedComponentsWithStatsWithAlgorithm", (char**)keywords, &pyobj_image, &connectivity, &ltype, &ccltype, &pyobj_labels, &pyobj_stats, &pyobj_centroids) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_labels, labels, ArgInfo("labels", 1)) &&
        pyopencv_to(pyobj_stats, stats, ArgInfo("stats", 1)) &&
        pyopencv_to(pyobj_centroids, centroids, ArgInfo("centroids", 1)) )
    {
        ERRWRAP2(retval = cv::connectedComponentsWithStats(image, labels, stats, centroids, connectivity, ltype, ccltype));
        return Py_BuildValue("(NNNN)", pyopencv_from(retval), pyopencv_from(labels), pyopencv_from(stats), pyopencv_from(centroids));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_labels = NULL;
    UMat labels;
    PyObject* pyobj_stats = NULL;
    UMat stats;
    PyObject* pyobj_centroids = NULL;
    UMat centroids;
    int connectivity=0;
    int ltype=0;
    int ccltype=0;
    int retval;

    const char* keywords[] = { "image", "connectivity", "ltype", "ccltype", "labels", "stats", "centroids", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oiii|OOO:connectedComponentsWithStatsWithAlgorithm", (char**)keywords, &pyobj_image, &connectivity, &ltype, &ccltype, &pyobj_labels, &pyobj_stats, &pyobj_centroids) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_labels, labels, ArgInfo("labels", 1)) &&
        pyopencv_to(pyobj_stats, stats, ArgInfo("stats", 1)) &&
        pyopencv_to(pyobj_centroids, centroids, ArgInfo("centroids", 1)) )
    {
        ERRWRAP2(retval = cv::connectedComponentsWithStats(image, labels, stats, centroids, connectivity, ltype, ccltype));
        return Py_BuildValue("(NNNN)", pyopencv_from(retval), pyopencv_from(labels), pyopencv_from(stats), pyopencv_from(centroids));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_contourArea(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_contour = NULL;
    Mat contour;
    bool oriented=false;
    double retval;

    const char* keywords[] = { "contour", "oriented", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|b:contourArea", (char**)keywords, &pyobj_contour, &oriented) &&
        pyopencv_to(pyobj_contour, contour, ArgInfo("contour", 0)) )
    {
        ERRWRAP2(retval = cv::contourArea(contour, oriented));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_contour = NULL;
    UMat contour;
    bool oriented=false;
    double retval;

    const char* keywords[] = { "contour", "oriented", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|b:contourArea", (char**)keywords, &pyobj_contour, &oriented) &&
        pyopencv_to(pyobj_contour, contour, ArgInfo("contour", 0)) )
    {
        ERRWRAP2(retval = cv::contourArea(contour, oriented));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_convertFp16(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:convertFp16", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::convertFp16(src, dst));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:convertFp16", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::convertFp16(src, dst));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_convertMaps(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_map1 = NULL;
    Mat map1;
    PyObject* pyobj_map2 = NULL;
    Mat map2;
    PyObject* pyobj_dstmap1 = NULL;
    Mat dstmap1;
    PyObject* pyobj_dstmap2 = NULL;
    Mat dstmap2;
    int dstmap1type=0;
    bool nninterpolation=false;

    const char* keywords[] = { "map1", "map2", "dstmap1type", "dstmap1", "dstmap2", "nninterpolation", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi|OOb:convertMaps", (char**)keywords, &pyobj_map1, &pyobj_map2, &dstmap1type, &pyobj_dstmap1, &pyobj_dstmap2, &nninterpolation) &&
        pyopencv_to(pyobj_map1, map1, ArgInfo("map1", 0)) &&
        pyopencv_to(pyobj_map2, map2, ArgInfo("map2", 0)) &&
        pyopencv_to(pyobj_dstmap1, dstmap1, ArgInfo("dstmap1", 1)) &&
        pyopencv_to(pyobj_dstmap2, dstmap2, ArgInfo("dstmap2", 1)) )
    {
        ERRWRAP2(cv::convertMaps(map1, map2, dstmap1, dstmap2, dstmap1type, nninterpolation));
        return Py_BuildValue("(NN)", pyopencv_from(dstmap1), pyopencv_from(dstmap2));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_map1 = NULL;
    UMat map1;
    PyObject* pyobj_map2 = NULL;
    UMat map2;
    PyObject* pyobj_dstmap1 = NULL;
    UMat dstmap1;
    PyObject* pyobj_dstmap2 = NULL;
    UMat dstmap2;
    int dstmap1type=0;
    bool nninterpolation=false;

    const char* keywords[] = { "map1", "map2", "dstmap1type", "dstmap1", "dstmap2", "nninterpolation", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi|OOb:convertMaps", (char**)keywords, &pyobj_map1, &pyobj_map2, &dstmap1type, &pyobj_dstmap1, &pyobj_dstmap2, &nninterpolation) &&
        pyopencv_to(pyobj_map1, map1, ArgInfo("map1", 0)) &&
        pyopencv_to(pyobj_map2, map2, ArgInfo("map2", 0)) &&
        pyopencv_to(pyobj_dstmap1, dstmap1, ArgInfo("dstmap1", 1)) &&
        pyopencv_to(pyobj_dstmap2, dstmap2, ArgInfo("dstmap2", 1)) )
    {
        ERRWRAP2(cv::convertMaps(map1, map2, dstmap1, dstmap2, dstmap1type, nninterpolation));
        return Py_BuildValue("(NN)", pyopencv_from(dstmap1), pyopencv_from(dstmap2));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_convertPointsFromHomogeneous(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:convertPointsFromHomogeneous", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::convertPointsFromHomogeneous(src, dst));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:convertPointsFromHomogeneous", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::convertPointsFromHomogeneous(src, dst));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_convertPointsToHomogeneous(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:convertPointsToHomogeneous", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::convertPointsToHomogeneous(src, dst));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:convertPointsToHomogeneous", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::convertPointsToHomogeneous(src, dst));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_convertScaleAbs(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    double alpha=1;
    double beta=0;

    const char* keywords[] = { "src", "dst", "alpha", "beta", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Odd:convertScaleAbs", (char**)keywords, &pyobj_src, &pyobj_dst, &alpha, &beta) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::convertScaleAbs(src, dst, alpha, beta));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    double alpha=1;
    double beta=0;

    const char* keywords[] = { "src", "dst", "alpha", "beta", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Odd:convertScaleAbs", (char**)keywords, &pyobj_src, &pyobj_dst, &alpha, &beta) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::convertScaleAbs(src, dst, alpha, beta));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_convexHull(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_points = NULL;
    Mat points;
    PyObject* pyobj_hull = NULL;
    Mat hull;
    bool clockwise=false;
    bool returnPoints=true;

    const char* keywords[] = { "points", "hull", "clockwise", "returnPoints", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Obb:convexHull", (char**)keywords, &pyobj_points, &pyobj_hull, &clockwise, &returnPoints) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 0)) &&
        pyopencv_to(pyobj_hull, hull, ArgInfo("hull", 1)) )
    {
        ERRWRAP2(cv::convexHull(points, hull, clockwise, returnPoints));
        return pyopencv_from(hull);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_points = NULL;
    UMat points;
    PyObject* pyobj_hull = NULL;
    UMat hull;
    bool clockwise=false;
    bool returnPoints=true;

    const char* keywords[] = { "points", "hull", "clockwise", "returnPoints", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Obb:convexHull", (char**)keywords, &pyobj_points, &pyobj_hull, &clockwise, &returnPoints) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 0)) &&
        pyopencv_to(pyobj_hull, hull, ArgInfo("hull", 1)) )
    {
        ERRWRAP2(cv::convexHull(points, hull, clockwise, returnPoints));
        return pyopencv_from(hull);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_convexityDefects(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_contour = NULL;
    Mat contour;
    PyObject* pyobj_convexhull = NULL;
    Mat convexhull;
    PyObject* pyobj_convexityDefects = NULL;
    Mat convexityDefects;

    const char* keywords[] = { "contour", "convexhull", "convexityDefects", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:convexityDefects", (char**)keywords, &pyobj_contour, &pyobj_convexhull, &pyobj_convexityDefects) &&
        pyopencv_to(pyobj_contour, contour, ArgInfo("contour", 0)) &&
        pyopencv_to(pyobj_convexhull, convexhull, ArgInfo("convexhull", 0)) &&
        pyopencv_to(pyobj_convexityDefects, convexityDefects, ArgInfo("convexityDefects", 1)) )
    {
        ERRWRAP2(cv::convexityDefects(contour, convexhull, convexityDefects));
        return pyopencv_from(convexityDefects);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_contour = NULL;
    UMat contour;
    PyObject* pyobj_convexhull = NULL;
    UMat convexhull;
    PyObject* pyobj_convexityDefects = NULL;
    UMat convexityDefects;

    const char* keywords[] = { "contour", "convexhull", "convexityDefects", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:convexityDefects", (char**)keywords, &pyobj_contour, &pyobj_convexhull, &pyobj_convexityDefects) &&
        pyopencv_to(pyobj_contour, contour, ArgInfo("contour", 0)) &&
        pyopencv_to(pyobj_convexhull, convexhull, ArgInfo("convexhull", 0)) &&
        pyopencv_to(pyobj_convexityDefects, convexityDefects, ArgInfo("convexityDefects", 1)) )
    {
        ERRWRAP2(cv::convexityDefects(contour, convexhull, convexityDefects));
        return pyopencv_from(convexityDefects);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_copyMakeBorder(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int top=0;
    int bottom=0;
    int left=0;
    int right=0;
    int borderType=0;
    PyObject* pyobj_value = NULL;
    Scalar value;

    const char* keywords[] = { "src", "top", "bottom", "left", "right", "borderType", "dst", "value", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oiiiii|OO:copyMakeBorder", (char**)keywords, &pyobj_src, &top, &bottom, &left, &right, &borderType, &pyobj_dst, &pyobj_value) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_value, value, ArgInfo("value", 0)) )
    {
        ERRWRAP2(cv::copyMakeBorder(src, dst, top, bottom, left, right, borderType, value));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int top=0;
    int bottom=0;
    int left=0;
    int right=0;
    int borderType=0;
    PyObject* pyobj_value = NULL;
    Scalar value;

    const char* keywords[] = { "src", "top", "bottom", "left", "right", "borderType", "dst", "value", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oiiiii|OO:copyMakeBorder", (char**)keywords, &pyobj_src, &top, &bottom, &left, &right, &borderType, &pyobj_dst, &pyobj_value) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_value, value, ArgInfo("value", 0)) )
    {
        ERRWRAP2(cv::copyMakeBorder(src, dst, top, bottom, left, right, borderType, value));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_cornerEigenValsAndVecs(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int blockSize=0;
    int ksize=0;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "blockSize", "ksize", "dst", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii|Oi:cornerEigenValsAndVecs", (char**)keywords, &pyobj_src, &blockSize, &ksize, &pyobj_dst, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::cornerEigenValsAndVecs(src, dst, blockSize, ksize, borderType));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int blockSize=0;
    int ksize=0;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "blockSize", "ksize", "dst", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii|Oi:cornerEigenValsAndVecs", (char**)keywords, &pyobj_src, &blockSize, &ksize, &pyobj_dst, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::cornerEigenValsAndVecs(src, dst, blockSize, ksize, borderType));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_cornerHarris(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int blockSize=0;
    int ksize=0;
    double k=0;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "blockSize", "ksize", "k", "dst", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oiid|Oi:cornerHarris", (char**)keywords, &pyobj_src, &blockSize, &ksize, &k, &pyobj_dst, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::cornerHarris(src, dst, blockSize, ksize, k, borderType));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int blockSize=0;
    int ksize=0;
    double k=0;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "blockSize", "ksize", "k", "dst", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oiid|Oi:cornerHarris", (char**)keywords, &pyobj_src, &blockSize, &ksize, &k, &pyobj_dst, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::cornerHarris(src, dst, blockSize, ksize, k, borderType));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_cornerMinEigenVal(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int blockSize=0;
    int ksize=3;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "blockSize", "dst", "ksize", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|Oii:cornerMinEigenVal", (char**)keywords, &pyobj_src, &blockSize, &pyobj_dst, &ksize, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::cornerMinEigenVal(src, dst, blockSize, ksize, borderType));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int blockSize=0;
    int ksize=3;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "blockSize", "dst", "ksize", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|Oii:cornerMinEigenVal", (char**)keywords, &pyobj_src, &blockSize, &pyobj_dst, &ksize, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::cornerMinEigenVal(src, dst, blockSize, ksize, borderType));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_cornerSubPix(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_corners = NULL;
    Mat corners;
    PyObject* pyobj_winSize = NULL;
    Size winSize;
    PyObject* pyobj_zeroZone = NULL;
    Size zeroZone;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria;

    const char* keywords[] = { "image", "corners", "winSize", "zeroZone", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOO:cornerSubPix", (char**)keywords, &pyobj_image, &pyobj_corners, &pyobj_winSize, &pyobj_zeroZone, &pyobj_criteria) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_corners, corners, ArgInfo("corners", 1)) &&
        pyopencv_to(pyobj_winSize, winSize, ArgInfo("winSize", 0)) &&
        pyopencv_to(pyobj_zeroZone, zeroZone, ArgInfo("zeroZone", 0)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(cv::cornerSubPix(image, corners, winSize, zeroZone, criteria));
        return pyopencv_from(corners);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_corners = NULL;
    UMat corners;
    PyObject* pyobj_winSize = NULL;
    Size winSize;
    PyObject* pyobj_zeroZone = NULL;
    Size zeroZone;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria;

    const char* keywords[] = { "image", "corners", "winSize", "zeroZone", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOO:cornerSubPix", (char**)keywords, &pyobj_image, &pyobj_corners, &pyobj_winSize, &pyobj_zeroZone, &pyobj_criteria) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_corners, corners, ArgInfo("corners", 1)) &&
        pyopencv_to(pyobj_winSize, winSize, ArgInfo("winSize", 0)) &&
        pyopencv_to(pyobj_zeroZone, zeroZone, ArgInfo("zeroZone", 0)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(cv::cornerSubPix(image, corners, winSize, zeroZone, criteria));
        return pyopencv_from(corners);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_correctMatches(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_F = NULL;
    Mat F;
    PyObject* pyobj_points1 = NULL;
    Mat points1;
    PyObject* pyobj_points2 = NULL;
    Mat points2;
    PyObject* pyobj_newPoints1 = NULL;
    Mat newPoints1;
    PyObject* pyobj_newPoints2 = NULL;
    Mat newPoints2;

    const char* keywords[] = { "F", "points1", "points2", "newPoints1", "newPoints2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|OO:correctMatches", (char**)keywords, &pyobj_F, &pyobj_points1, &pyobj_points2, &pyobj_newPoints1, &pyobj_newPoints2) &&
        pyopencv_to(pyobj_F, F, ArgInfo("F", 0)) &&
        pyopencv_to(pyobj_points1, points1, ArgInfo("points1", 0)) &&
        pyopencv_to(pyobj_points2, points2, ArgInfo("points2", 0)) &&
        pyopencv_to(pyobj_newPoints1, newPoints1, ArgInfo("newPoints1", 1)) &&
        pyopencv_to(pyobj_newPoints2, newPoints2, ArgInfo("newPoints2", 1)) )
    {
        ERRWRAP2(cv::correctMatches(F, points1, points2, newPoints1, newPoints2));
        return Py_BuildValue("(NN)", pyopencv_from(newPoints1), pyopencv_from(newPoints2));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_F = NULL;
    UMat F;
    PyObject* pyobj_points1 = NULL;
    UMat points1;
    PyObject* pyobj_points2 = NULL;
    UMat points2;
    PyObject* pyobj_newPoints1 = NULL;
    UMat newPoints1;
    PyObject* pyobj_newPoints2 = NULL;
    UMat newPoints2;

    const char* keywords[] = { "F", "points1", "points2", "newPoints1", "newPoints2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|OO:correctMatches", (char**)keywords, &pyobj_F, &pyobj_points1, &pyobj_points2, &pyobj_newPoints1, &pyobj_newPoints2) &&
        pyopencv_to(pyobj_F, F, ArgInfo("F", 0)) &&
        pyopencv_to(pyobj_points1, points1, ArgInfo("points1", 0)) &&
        pyopencv_to(pyobj_points2, points2, ArgInfo("points2", 0)) &&
        pyopencv_to(pyobj_newPoints1, newPoints1, ArgInfo("newPoints1", 1)) &&
        pyopencv_to(pyobj_newPoints2, newPoints2, ArgInfo("newPoints2", 1)) )
    {
        ERRWRAP2(cv::correctMatches(F, points1, points2, newPoints1, newPoints2));
        return Py_BuildValue("(NN)", pyopencv_from(newPoints1), pyopencv_from(newPoints2));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_countNonZero(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    int retval;

    const char* keywords[] = { "src", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:countNonZero", (char**)keywords, &pyobj_src) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) )
    {
        ERRWRAP2(retval = cv::countNonZero(src));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    int retval;

    const char* keywords[] = { "src", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:countNonZero", (char**)keywords, &pyobj_src) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) )
    {
        ERRWRAP2(retval = cv::countNonZero(src));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_createAffineTransformer(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    bool fullAffine=0;
    Ptr<AffineTransformer> retval;

    const char* keywords[] = { "fullAffine", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:createAffineTransformer", (char**)keywords, &fullAffine) )
    {
        ERRWRAP2(retval = cv::createAffineTransformer(fullAffine));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_createAlignMTB(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int max_bits=6;
    int exclude_range=4;
    bool cut=true;
    Ptr<AlignMTB> retval;

    const char* keywords[] = { "max_bits", "exclude_range", "cut", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|iib:createAlignMTB", (char**)keywords, &max_bits, &exclude_range, &cut) )
    {
        ERRWRAP2(retval = cv::createAlignMTB(max_bits, exclude_range, cut));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_createBackgroundSubtractorKNN(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int history=500;
    double dist2Threshold=400.0;
    bool detectShadows=true;
    Ptr<BackgroundSubtractorKNN> retval;

    const char* keywords[] = { "history", "dist2Threshold", "detectShadows", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|idb:createBackgroundSubtractorKNN", (char**)keywords, &history, &dist2Threshold, &detectShadows) )
    {
        ERRWRAP2(retval = cv::createBackgroundSubtractorKNN(history, dist2Threshold, detectShadows));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_createBackgroundSubtractorMOG2(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int history=500;
    double varThreshold=16;
    bool detectShadows=true;
    Ptr<BackgroundSubtractorMOG2> retval;

    const char* keywords[] = { "history", "varThreshold", "detectShadows", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|idb:createBackgroundSubtractorMOG2", (char**)keywords, &history, &varThreshold, &detectShadows) )
    {
        ERRWRAP2(retval = cv::createBackgroundSubtractorMOG2(history, varThreshold, detectShadows));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_createCLAHE(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    double clipLimit=40.0;
    PyObject* pyobj_tileGridSize = NULL;
    Size tileGridSize=Size(8, 8);
    Ptr<CLAHE> retval;

    const char* keywords[] = { "clipLimit", "tileGridSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|dO:createCLAHE", (char**)keywords, &clipLimit, &pyobj_tileGridSize) &&
        pyopencv_to(pyobj_tileGridSize, tileGridSize, ArgInfo("tileGridSize", 0)) )
    {
        ERRWRAP2(retval = cv::createCLAHE(clipLimit, tileGridSize));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_createCalibrateDebevec(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int samples=70;
    float lambda=10.0f;
    bool random=false;
    Ptr<CalibrateDebevec> retval;

    const char* keywords[] = { "samples", "lambda", "random", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|ifb:createCalibrateDebevec", (char**)keywords, &samples, &lambda, &random) )
    {
        ERRWRAP2(retval = cv::createCalibrateDebevec(samples, lambda, random));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_createCalibrateRobertson(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int max_iter=30;
    float threshold=0.01f;
    Ptr<CalibrateRobertson> retval;

    const char* keywords[] = { "max_iter", "threshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|if:createCalibrateRobertson", (char**)keywords, &max_iter, &threshold) )
    {
        ERRWRAP2(retval = cv::createCalibrateRobertson(max_iter, threshold));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_createChiHistogramCostExtractor(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int nDummies=25;
    float defaultCost=0.2f;
    Ptr<HistogramCostExtractor> retval;

    const char* keywords[] = { "nDummies", "defaultCost", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|if:createChiHistogramCostExtractor", (char**)keywords, &nDummies, &defaultCost) )
    {
        ERRWRAP2(retval = cv::createChiHistogramCostExtractor(nDummies, defaultCost));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_createEMDHistogramCostExtractor(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int flag=DIST_L2;
    int nDummies=25;
    float defaultCost=0.2f;
    Ptr<HistogramCostExtractor> retval;

    const char* keywords[] = { "flag", "nDummies", "defaultCost", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|iif:createEMDHistogramCostExtractor", (char**)keywords, &flag, &nDummies, &defaultCost) )
    {
        ERRWRAP2(retval = cv::createEMDHistogramCostExtractor(flag, nDummies, defaultCost));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_createEMDL1HistogramCostExtractor(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int nDummies=25;
    float defaultCost=0.2f;
    Ptr<HistogramCostExtractor> retval;

    const char* keywords[] = { "nDummies", "defaultCost", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|if:createEMDL1HistogramCostExtractor", (char**)keywords, &nDummies, &defaultCost) )
    {
        ERRWRAP2(retval = cv::createEMDL1HistogramCostExtractor(nDummies, defaultCost));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_createHanningWindow(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_winSize = NULL;
    Size winSize;
    int type=0;

    const char* keywords[] = { "winSize", "type", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|O:createHanningWindow", (char**)keywords, &pyobj_winSize, &type, &pyobj_dst) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_winSize, winSize, ArgInfo("winSize", 0)) )
    {
        ERRWRAP2(cv::createHanningWindow(dst, winSize, type));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_winSize = NULL;
    Size winSize;
    int type=0;

    const char* keywords[] = { "winSize", "type", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|O:createHanningWindow", (char**)keywords, &pyobj_winSize, &type, &pyobj_dst) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_winSize, winSize, ArgInfo("winSize", 0)) )
    {
        ERRWRAP2(cv::createHanningWindow(dst, winSize, type));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_createHausdorffDistanceExtractor(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int distanceFlag=cv::NORM_L2;
    float rankProp=0.6f;
    Ptr<HausdorffDistanceExtractor> retval;

    const char* keywords[] = { "distanceFlag", "rankProp", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|if:createHausdorffDistanceExtractor", (char**)keywords, &distanceFlag, &rankProp) )
    {
        ERRWRAP2(retval = cv::createHausdorffDistanceExtractor(distanceFlag, rankProp));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_createLineSegmentDetector(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int _refine=LSD_REFINE_STD;
    double _scale=0.8;
    double _sigma_scale=0.6;
    double _quant=2.0;
    double _ang_th=22.5;
    double _log_eps=0;
    double _density_th=0.7;
    int _n_bins=1024;
    Ptr<LineSegmentDetector> retval;

    const char* keywords[] = { "_refine", "_scale", "_sigma_scale", "_quant", "_ang_th", "_log_eps", "_density_th", "_n_bins", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|iddddddi:createLineSegmentDetector", (char**)keywords, &_refine, &_scale, &_sigma_scale, &_quant, &_ang_th, &_log_eps, &_density_th, &_n_bins) )
    {
        ERRWRAP2(retval = cv::createLineSegmentDetector(_refine, _scale, _sigma_scale, _quant, _ang_th, _log_eps, _density_th, _n_bins));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_createMergeDebevec(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    Ptr<MergeDebevec> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::createMergeDebevec());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_createMergeMertens(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    float contrast_weight=1.0f;
    float saturation_weight=1.0f;
    float exposure_weight=0.0f;
    Ptr<MergeMertens> retval;

    const char* keywords[] = { "contrast_weight", "saturation_weight", "exposure_weight", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|fff:createMergeMertens", (char**)keywords, &contrast_weight, &saturation_weight, &exposure_weight) )
    {
        ERRWRAP2(retval = cv::createMergeMertens(contrast_weight, saturation_weight, exposure_weight));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_createMergeRobertson(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    Ptr<MergeRobertson> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::createMergeRobertson());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_createNormHistogramCostExtractor(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int flag=DIST_L2;
    int nDummies=25;
    float defaultCost=0.2f;
    Ptr<HistogramCostExtractor> retval;

    const char* keywords[] = { "flag", "nDummies", "defaultCost", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|iif:createNormHistogramCostExtractor", (char**)keywords, &flag, &nDummies, &defaultCost) )
    {
        ERRWRAP2(retval = cv::createNormHistogramCostExtractor(flag, nDummies, defaultCost));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_createOptFlow_DualTVL1(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    Ptr<DualTVL1OpticalFlow> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::createOptFlow_DualTVL1());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_createShapeContextDistanceExtractor(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int nAngularBins=12;
    int nRadialBins=4;
    float innerRadius=0.2f;
    float outerRadius=2;
    int iterations=3;
    PyObject* pyobj_comparer = NULL;
    Ptr<HistogramCostExtractor> comparer=createChiHistogramCostExtractor();
    PyObject* pyobj_transformer = NULL;
    Ptr<ShapeTransformer> transformer=createThinPlateSplineShapeTransformer();
    Ptr<ShapeContextDistanceExtractor> retval;

    const char* keywords[] = { "nAngularBins", "nRadialBins", "innerRadius", "outerRadius", "iterations", "comparer", "transformer", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|iiffiOO:createShapeContextDistanceExtractor", (char**)keywords, &nAngularBins, &nRadialBins, &innerRadius, &outerRadius, &iterations, &pyobj_comparer, &pyobj_transformer) &&
        pyopencv_to(pyobj_comparer, comparer, ArgInfo("comparer", 0)) &&
        pyopencv_to(pyobj_transformer, transformer, ArgInfo("transformer", 0)) )
    {
        ERRWRAP2(retval = cv::createShapeContextDistanceExtractor(nAngularBins, nRadialBins, innerRadius, outerRadius, iterations, comparer, transformer));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_createStitcher(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    bool try_use_gpu=false;
    Ptr<Stitcher> retval;

    const char* keywords[] = { "try_use_gpu", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|b:createStitcher", (char**)keywords, &try_use_gpu) )
    {
        ERRWRAP2(retval = cv::createStitcher(try_use_gpu));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_createThinPlateSplineShapeTransformer(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    double regularizationParameter=0;
    Ptr<ThinPlateSplineShapeTransformer> retval;

    const char* keywords[] = { "regularizationParameter", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|d:createThinPlateSplineShapeTransformer", (char**)keywords, &regularizationParameter) )
    {
        ERRWRAP2(retval = cv::createThinPlateSplineShapeTransformer(regularizationParameter));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_createTonemap(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    float gamma=1.0f;
    Ptr<Tonemap> retval;

    const char* keywords[] = { "gamma", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|f:createTonemap", (char**)keywords, &gamma) )
    {
        ERRWRAP2(retval = cv::createTonemap(gamma));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_createTonemapDrago(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    float gamma=1.0f;
    float saturation=1.0f;
    float bias=0.85f;
    Ptr<TonemapDrago> retval;

    const char* keywords[] = { "gamma", "saturation", "bias", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|fff:createTonemapDrago", (char**)keywords, &gamma, &saturation, &bias) )
    {
        ERRWRAP2(retval = cv::createTonemapDrago(gamma, saturation, bias));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_createTonemapDurand(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    float gamma=1.0f;
    float contrast=4.0f;
    float saturation=1.0f;
    float sigma_space=2.0f;
    float sigma_color=2.0f;
    Ptr<TonemapDurand> retval;

    const char* keywords[] = { "gamma", "contrast", "saturation", "sigma_space", "sigma_color", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|fffff:createTonemapDurand", (char**)keywords, &gamma, &contrast, &saturation, &sigma_space, &sigma_color) )
    {
        ERRWRAP2(retval = cv::createTonemapDurand(gamma, contrast, saturation, sigma_space, sigma_color));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_createTonemapMantiuk(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    float gamma=1.0f;
    float scale=0.7f;
    float saturation=1.0f;
    Ptr<TonemapMantiuk> retval;

    const char* keywords[] = { "gamma", "scale", "saturation", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|fff:createTonemapMantiuk", (char**)keywords, &gamma, &scale, &saturation) )
    {
        ERRWRAP2(retval = cv::createTonemapMantiuk(gamma, scale, saturation));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_createTonemapReinhard(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    float gamma=1.0f;
    float intensity=0.0f;
    float light_adapt=1.0f;
    float color_adapt=0.0f;
    Ptr<TonemapReinhard> retval;

    const char* keywords[] = { "gamma", "intensity", "light_adapt", "color_adapt", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|ffff:createTonemapReinhard", (char**)keywords, &gamma, &intensity, &light_adapt, &color_adapt) )
    {
        ERRWRAP2(retval = cv::createTonemapReinhard(gamma, intensity, light_adapt, color_adapt));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_cubeRoot(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    float val=0.f;
    float retval;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:cubeRoot", (char**)keywords, &val) )
    {
        ERRWRAP2(retval = cv::cubeRoot(val));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_cvtColor(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int code=0;
    int dstCn=0;

    const char* keywords[] = { "src", "code", "dst", "dstCn", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|Oi:cvtColor", (char**)keywords, &pyobj_src, &code, &pyobj_dst, &dstCn) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::cvtColor(src, dst, code, dstCn));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int code=0;
    int dstCn=0;

    const char* keywords[] = { "src", "code", "dst", "dstCn", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|Oi:cvtColor", (char**)keywords, &pyobj_src, &code, &pyobj_dst, &dstCn) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::cvtColor(src, dst, code, dstCn));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_dct(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int flags=0;

    const char* keywords[] = { "src", "dst", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oi:dct", (char**)keywords, &pyobj_src, &pyobj_dst, &flags) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::dct(src, dst, flags));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int flags=0;

    const char* keywords[] = { "src", "dst", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oi:dct", (char**)keywords, &pyobj_src, &pyobj_dst, &flags) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::dct(src, dst, flags));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_decolor(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_grayscale = NULL;
    Mat grayscale;
    PyObject* pyobj_color_boost = NULL;
    Mat color_boost;

    const char* keywords[] = { "src", "grayscale", "color_boost", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OO:decolor", (char**)keywords, &pyobj_src, &pyobj_grayscale, &pyobj_color_boost) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_grayscale, grayscale, ArgInfo("grayscale", 1)) &&
        pyopencv_to(pyobj_color_boost, color_boost, ArgInfo("color_boost", 1)) )
    {
        ERRWRAP2(cv::decolor(src, grayscale, color_boost));
        return Py_BuildValue("(NN)", pyopencv_from(grayscale), pyopencv_from(color_boost));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_grayscale = NULL;
    UMat grayscale;
    PyObject* pyobj_color_boost = NULL;
    UMat color_boost;

    const char* keywords[] = { "src", "grayscale", "color_boost", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OO:decolor", (char**)keywords, &pyobj_src, &pyobj_grayscale, &pyobj_color_boost) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_grayscale, grayscale, ArgInfo("grayscale", 1)) &&
        pyopencv_to(pyobj_color_boost, color_boost, ArgInfo("color_boost", 1)) )
    {
        ERRWRAP2(cv::decolor(src, grayscale, color_boost));
        return Py_BuildValue("(NN)", pyopencv_from(grayscale), pyopencv_from(color_boost));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_decomposeEssentialMat(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_E = NULL;
    Mat E;
    PyObject* pyobj_R1 = NULL;
    Mat R1;
    PyObject* pyobj_R2 = NULL;
    Mat R2;
    PyObject* pyobj_t = NULL;
    Mat t;

    const char* keywords[] = { "E", "R1", "R2", "t", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOO:decomposeEssentialMat", (char**)keywords, &pyobj_E, &pyobj_R1, &pyobj_R2, &pyobj_t) &&
        pyopencv_to(pyobj_E, E, ArgInfo("E", 0)) &&
        pyopencv_to(pyobj_R1, R1, ArgInfo("R1", 1)) &&
        pyopencv_to(pyobj_R2, R2, ArgInfo("R2", 1)) &&
        pyopencv_to(pyobj_t, t, ArgInfo("t", 1)) )
    {
        ERRWRAP2(cv::decomposeEssentialMat(E, R1, R2, t));
        return Py_BuildValue("(NNN)", pyopencv_from(R1), pyopencv_from(R2), pyopencv_from(t));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_E = NULL;
    UMat E;
    PyObject* pyobj_R1 = NULL;
    UMat R1;
    PyObject* pyobj_R2 = NULL;
    UMat R2;
    PyObject* pyobj_t = NULL;
    UMat t;

    const char* keywords[] = { "E", "R1", "R2", "t", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOO:decomposeEssentialMat", (char**)keywords, &pyobj_E, &pyobj_R1, &pyobj_R2, &pyobj_t) &&
        pyopencv_to(pyobj_E, E, ArgInfo("E", 0)) &&
        pyopencv_to(pyobj_R1, R1, ArgInfo("R1", 1)) &&
        pyopencv_to(pyobj_R2, R2, ArgInfo("R2", 1)) &&
        pyopencv_to(pyobj_t, t, ArgInfo("t", 1)) )
    {
        ERRWRAP2(cv::decomposeEssentialMat(E, R1, R2, t));
        return Py_BuildValue("(NNN)", pyopencv_from(R1), pyopencv_from(R2), pyopencv_from(t));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_decomposeHomographyMat(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_H = NULL;
    Mat H;
    PyObject* pyobj_K = NULL;
    Mat K;
    PyObject* pyobj_rotations = NULL;
    vector_Mat rotations;
    PyObject* pyobj_translations = NULL;
    vector_Mat translations;
    PyObject* pyobj_normals = NULL;
    vector_Mat normals;
    int retval;

    const char* keywords[] = { "H", "K", "rotations", "translations", "normals", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOO:decomposeHomographyMat", (char**)keywords, &pyobj_H, &pyobj_K, &pyobj_rotations, &pyobj_translations, &pyobj_normals) &&
        pyopencv_to(pyobj_H, H, ArgInfo("H", 0)) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 0)) &&
        pyopencv_to(pyobj_rotations, rotations, ArgInfo("rotations", 1)) &&
        pyopencv_to(pyobj_translations, translations, ArgInfo("translations", 1)) &&
        pyopencv_to(pyobj_normals, normals, ArgInfo("normals", 1)) )
    {
        ERRWRAP2(retval = cv::decomposeHomographyMat(H, K, rotations, translations, normals));
        return Py_BuildValue("(NNNN)", pyopencv_from(retval), pyopencv_from(rotations), pyopencv_from(translations), pyopencv_from(normals));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_H = NULL;
    UMat H;
    PyObject* pyobj_K = NULL;
    UMat K;
    PyObject* pyobj_rotations = NULL;
    vector_Mat rotations;
    PyObject* pyobj_translations = NULL;
    vector_Mat translations;
    PyObject* pyobj_normals = NULL;
    vector_Mat normals;
    int retval;

    const char* keywords[] = { "H", "K", "rotations", "translations", "normals", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOO:decomposeHomographyMat", (char**)keywords, &pyobj_H, &pyobj_K, &pyobj_rotations, &pyobj_translations, &pyobj_normals) &&
        pyopencv_to(pyobj_H, H, ArgInfo("H", 0)) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 0)) &&
        pyopencv_to(pyobj_rotations, rotations, ArgInfo("rotations", 1)) &&
        pyopencv_to(pyobj_translations, translations, ArgInfo("translations", 1)) &&
        pyopencv_to(pyobj_normals, normals, ArgInfo("normals", 1)) )
    {
        ERRWRAP2(retval = cv::decomposeHomographyMat(H, K, rotations, translations, normals));
        return Py_BuildValue("(NNNN)", pyopencv_from(retval), pyopencv_from(rotations), pyopencv_from(translations), pyopencv_from(normals));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_decomposeProjectionMatrix(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_projMatrix = NULL;
    Mat projMatrix;
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_rotMatrix = NULL;
    Mat rotMatrix;
    PyObject* pyobj_transVect = NULL;
    Mat transVect;
    PyObject* pyobj_rotMatrixX = NULL;
    Mat rotMatrixX;
    PyObject* pyobj_rotMatrixY = NULL;
    Mat rotMatrixY;
    PyObject* pyobj_rotMatrixZ = NULL;
    Mat rotMatrixZ;
    PyObject* pyobj_eulerAngles = NULL;
    Mat eulerAngles;

    const char* keywords[] = { "projMatrix", "cameraMatrix", "rotMatrix", "transVect", "rotMatrixX", "rotMatrixY", "rotMatrixZ", "eulerAngles", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOOOOOO:decomposeProjectionMatrix", (char**)keywords, &pyobj_projMatrix, &pyobj_cameraMatrix, &pyobj_rotMatrix, &pyobj_transVect, &pyobj_rotMatrixX, &pyobj_rotMatrixY, &pyobj_rotMatrixZ, &pyobj_eulerAngles) &&
        pyopencv_to(pyobj_projMatrix, projMatrix, ArgInfo("projMatrix", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 1)) &&
        pyopencv_to(pyobj_rotMatrix, rotMatrix, ArgInfo("rotMatrix", 1)) &&
        pyopencv_to(pyobj_transVect, transVect, ArgInfo("transVect", 1)) &&
        pyopencv_to(pyobj_rotMatrixX, rotMatrixX, ArgInfo("rotMatrixX", 1)) &&
        pyopencv_to(pyobj_rotMatrixY, rotMatrixY, ArgInfo("rotMatrixY", 1)) &&
        pyopencv_to(pyobj_rotMatrixZ, rotMatrixZ, ArgInfo("rotMatrixZ", 1)) &&
        pyopencv_to(pyobj_eulerAngles, eulerAngles, ArgInfo("eulerAngles", 1)) )
    {
        ERRWRAP2(cv::decomposeProjectionMatrix(projMatrix, cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles));
        return Py_BuildValue("(NNNNNNN)", pyopencv_from(cameraMatrix), pyopencv_from(rotMatrix), pyopencv_from(transVect), pyopencv_from(rotMatrixX), pyopencv_from(rotMatrixY), pyopencv_from(rotMatrixZ), pyopencv_from(eulerAngles));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_projMatrix = NULL;
    UMat projMatrix;
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    PyObject* pyobj_rotMatrix = NULL;
    UMat rotMatrix;
    PyObject* pyobj_transVect = NULL;
    UMat transVect;
    PyObject* pyobj_rotMatrixX = NULL;
    UMat rotMatrixX;
    PyObject* pyobj_rotMatrixY = NULL;
    UMat rotMatrixY;
    PyObject* pyobj_rotMatrixZ = NULL;
    UMat rotMatrixZ;
    PyObject* pyobj_eulerAngles = NULL;
    UMat eulerAngles;

    const char* keywords[] = { "projMatrix", "cameraMatrix", "rotMatrix", "transVect", "rotMatrixX", "rotMatrixY", "rotMatrixZ", "eulerAngles", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOOOOOO:decomposeProjectionMatrix", (char**)keywords, &pyobj_projMatrix, &pyobj_cameraMatrix, &pyobj_rotMatrix, &pyobj_transVect, &pyobj_rotMatrixX, &pyobj_rotMatrixY, &pyobj_rotMatrixZ, &pyobj_eulerAngles) &&
        pyopencv_to(pyobj_projMatrix, projMatrix, ArgInfo("projMatrix", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 1)) &&
        pyopencv_to(pyobj_rotMatrix, rotMatrix, ArgInfo("rotMatrix", 1)) &&
        pyopencv_to(pyobj_transVect, transVect, ArgInfo("transVect", 1)) &&
        pyopencv_to(pyobj_rotMatrixX, rotMatrixX, ArgInfo("rotMatrixX", 1)) &&
        pyopencv_to(pyobj_rotMatrixY, rotMatrixY, ArgInfo("rotMatrixY", 1)) &&
        pyopencv_to(pyobj_rotMatrixZ, rotMatrixZ, ArgInfo("rotMatrixZ", 1)) &&
        pyopencv_to(pyobj_eulerAngles, eulerAngles, ArgInfo("eulerAngles", 1)) )
    {
        ERRWRAP2(cv::decomposeProjectionMatrix(projMatrix, cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles));
        return Py_BuildValue("(NNNNNNN)", pyopencv_from(cameraMatrix), pyopencv_from(rotMatrix), pyopencv_from(transVect), pyopencv_from(rotMatrixX), pyopencv_from(rotMatrixY), pyopencv_from(rotMatrixZ), pyopencv_from(eulerAngles));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_demosaicing(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj__src = NULL;
    Mat _src;
    PyObject* pyobj__dst = NULL;
    Mat _dst;
    int code=0;
    int dcn=0;

    const char* keywords[] = { "_src", "code", "_dst", "dcn", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|Oi:demosaicing", (char**)keywords, &pyobj__src, &code, &pyobj__dst, &dcn) &&
        pyopencv_to(pyobj__src, _src, ArgInfo("_src", 0)) &&
        pyopencv_to(pyobj__dst, _dst, ArgInfo("_dst", 1)) )
    {
        ERRWRAP2(cv::demosaicing(_src, _dst, code, dcn));
        return pyopencv_from(_dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj__src = NULL;
    UMat _src;
    PyObject* pyobj__dst = NULL;
    UMat _dst;
    int code=0;
    int dcn=0;

    const char* keywords[] = { "_src", "code", "_dst", "dcn", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|Oi:demosaicing", (char**)keywords, &pyobj__src, &code, &pyobj__dst, &dcn) &&
        pyopencv_to(pyobj__src, _src, ArgInfo("_src", 0)) &&
        pyopencv_to(pyobj__dst, _dst, ArgInfo("_dst", 1)) )
    {
        ERRWRAP2(cv::demosaicing(_src, _dst, code, dcn));
        return pyopencv_from(_dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_denoise_TVL1(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_observations = NULL;
    vector_Mat observations;
    PyObject* pyobj_result = NULL;
    Mat result;
    double lambda=1.0;
    int niters=30;

    const char* keywords[] = { "observations", "result", "lambda", "niters", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|di:denoise_TVL1", (char**)keywords, &pyobj_observations, &pyobj_result, &lambda, &niters) &&
        pyopencv_to(pyobj_observations, observations, ArgInfo("observations", 0)) &&
        pyopencv_to(pyobj_result, result, ArgInfo("result", 0)) )
    {
        ERRWRAP2(cv::denoise_TVL1(observations, result, lambda, niters));
        Py_RETURN_NONE;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_observations = NULL;
    vector_Mat observations;
    PyObject* pyobj_result = NULL;
    Mat result;
    double lambda=1.0;
    int niters=30;

    const char* keywords[] = { "observations", "result", "lambda", "niters", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|di:denoise_TVL1", (char**)keywords, &pyobj_observations, &pyobj_result, &lambda, &niters) &&
        pyopencv_to(pyobj_observations, observations, ArgInfo("observations", 0)) &&
        pyopencv_to(pyobj_result, result, ArgInfo("result", 0)) )
    {
        ERRWRAP2(cv::denoise_TVL1(observations, result, lambda, niters));
        Py_RETURN_NONE;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_destroyAllWindows(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;


    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(cv::destroyAllWindows());
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_destroyWindow(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_winname = NULL;
    String winname;

    const char* keywords[] = { "winname", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:destroyWindow", (char**)keywords, &pyobj_winname) &&
        pyopencv_to(pyobj_winname, winname, ArgInfo("winname", 0)) )
    {
        ERRWRAP2(cv::destroyWindow(winname));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_detailEnhance(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    float sigma_s=10;
    float sigma_r=0.15f;

    const char* keywords[] = { "src", "dst", "sigma_s", "sigma_r", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Off:detailEnhance", (char**)keywords, &pyobj_src, &pyobj_dst, &sigma_s, &sigma_r) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::detailEnhance(src, dst, sigma_s, sigma_r));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    float sigma_s=10;
    float sigma_r=0.15f;

    const char* keywords[] = { "src", "dst", "sigma_s", "sigma_r", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Off:detailEnhance", (char**)keywords, &pyobj_src, &pyobj_dst, &sigma_s, &sigma_r) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::detailEnhance(src, dst, sigma_s, sigma_r));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_determinant(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_mtx = NULL;
    Mat mtx;
    double retval;

    const char* keywords[] = { "mtx", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:determinant", (char**)keywords, &pyobj_mtx) &&
        pyopencv_to(pyobj_mtx, mtx, ArgInfo("mtx", 0)) )
    {
        ERRWRAP2(retval = cv::determinant(mtx));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_mtx = NULL;
    UMat mtx;
    double retval;

    const char* keywords[] = { "mtx", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:determinant", (char**)keywords, &pyobj_mtx) &&
        pyopencv_to(pyobj_mtx, mtx, ArgInfo("mtx", 0)) )
    {
        ERRWRAP2(retval = cv::determinant(mtx));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_dft(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int flags=0;
    int nonzeroRows=0;

    const char* keywords[] = { "src", "dst", "flags", "nonzeroRows", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oii:dft", (char**)keywords, &pyobj_src, &pyobj_dst, &flags, &nonzeroRows) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::dft(src, dst, flags, nonzeroRows));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int flags=0;
    int nonzeroRows=0;

    const char* keywords[] = { "src", "dst", "flags", "nonzeroRows", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oii:dft", (char**)keywords, &pyobj_src, &pyobj_dst, &flags, &nonzeroRows) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::dft(src, dst, flags, nonzeroRows));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_dilate(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_kernel = NULL;
    Mat kernel;
    PyObject* pyobj_anchor = NULL;
    Point anchor=Point(-1,-1);
    int iterations=1;
    int borderType=BORDER_CONSTANT;
    PyObject* pyobj_borderValue = NULL;
    Scalar borderValue=morphologyDefaultBorderValue();

    const char* keywords[] = { "src", "kernel", "dst", "anchor", "iterations", "borderType", "borderValue", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOiiO:dilate", (char**)keywords, &pyobj_src, &pyobj_kernel, &pyobj_dst, &pyobj_anchor, &iterations, &borderType, &pyobj_borderValue) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 0)) &&
        pyopencv_to(pyobj_anchor, anchor, ArgInfo("anchor", 0)) &&
        pyopencv_to(pyobj_borderValue, borderValue, ArgInfo("borderValue", 0)) )
    {
        ERRWRAP2(cv::dilate(src, dst, kernel, anchor, iterations, borderType, borderValue));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_kernel = NULL;
    UMat kernel;
    PyObject* pyobj_anchor = NULL;
    Point anchor=Point(-1,-1);
    int iterations=1;
    int borderType=BORDER_CONSTANT;
    PyObject* pyobj_borderValue = NULL;
    Scalar borderValue=morphologyDefaultBorderValue();

    const char* keywords[] = { "src", "kernel", "dst", "anchor", "iterations", "borderType", "borderValue", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOiiO:dilate", (char**)keywords, &pyobj_src, &pyobj_kernel, &pyobj_dst, &pyobj_anchor, &iterations, &borderType, &pyobj_borderValue) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 0)) &&
        pyopencv_to(pyobj_anchor, anchor, ArgInfo("anchor", 0)) &&
        pyopencv_to(pyobj_borderValue, borderValue, ArgInfo("borderValue", 0)) )
    {
        ERRWRAP2(cv::dilate(src, dst, kernel, anchor, iterations, borderType, borderValue));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_displayOverlay(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_winname = NULL;
    String winname;
    PyObject* pyobj_text = NULL;
    String text;
    int delayms=0;

    const char* keywords[] = { "winname", "text", "delayms", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|i:displayOverlay", (char**)keywords, &pyobj_winname, &pyobj_text, &delayms) &&
        pyopencv_to(pyobj_winname, winname, ArgInfo("winname", 0)) &&
        pyopencv_to(pyobj_text, text, ArgInfo("text", 0)) )
    {
        ERRWRAP2(cv::displayOverlay(winname, text, delayms));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_displayStatusBar(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_winname = NULL;
    String winname;
    PyObject* pyobj_text = NULL;
    String text;
    int delayms=0;

    const char* keywords[] = { "winname", "text", "delayms", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|i:displayStatusBar", (char**)keywords, &pyobj_winname, &pyobj_text, &delayms) &&
        pyopencv_to(pyobj_winname, winname, ArgInfo("winname", 0)) &&
        pyopencv_to(pyobj_text, text, ArgInfo("text", 0)) )
    {
        ERRWRAP2(cv::displayStatusBar(winname, text, delayms));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_distanceTransform(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int distanceType=0;
    int maskSize=0;
    int dstType=CV_32F;

    const char* keywords[] = { "src", "distanceType", "maskSize", "dst", "dstType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii|Oi:distanceTransform", (char**)keywords, &pyobj_src, &distanceType, &maskSize, &pyobj_dst, &dstType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::distanceTransform(src, dst, distanceType, maskSize, dstType));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int distanceType=0;
    int maskSize=0;
    int dstType=CV_32F;

    const char* keywords[] = { "src", "distanceType", "maskSize", "dst", "dstType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii|Oi:distanceTransform", (char**)keywords, &pyobj_src, &distanceType, &maskSize, &pyobj_dst, &dstType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::distanceTransform(src, dst, distanceType, maskSize, dstType));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_distanceTransformWithLabels(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_labels = NULL;
    Mat labels;
    int distanceType=0;
    int maskSize=0;
    int labelType=DIST_LABEL_CCOMP;

    const char* keywords[] = { "src", "distanceType", "maskSize", "dst", "labels", "labelType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii|OOi:distanceTransformWithLabels", (char**)keywords, &pyobj_src, &distanceType, &maskSize, &pyobj_dst, &pyobj_labels, &labelType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_labels, labels, ArgInfo("labels", 1)) )
    {
        ERRWRAP2(cv::distanceTransform(src, dst, labels, distanceType, maskSize, labelType));
        return Py_BuildValue("(NN)", pyopencv_from(dst), pyopencv_from(labels));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_labels = NULL;
    UMat labels;
    int distanceType=0;
    int maskSize=0;
    int labelType=DIST_LABEL_CCOMP;

    const char* keywords[] = { "src", "distanceType", "maskSize", "dst", "labels", "labelType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii|OOi:distanceTransformWithLabels", (char**)keywords, &pyobj_src, &distanceType, &maskSize, &pyobj_dst, &pyobj_labels, &labelType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_labels, labels, ArgInfo("labels", 1)) )
    {
        ERRWRAP2(cv::distanceTransform(src, dst, labels, distanceType, maskSize, labelType));
        return Py_BuildValue("(NN)", pyopencv_from(dst), pyopencv_from(labels));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_divide(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    PyObject* pyobj_src2 = NULL;
    Mat src2;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    double scale=1;
    int dtype=-1;

    const char* keywords[] = { "src1", "src2", "dst", "scale", "dtype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Odi:divide", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &scale, &dtype) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::divide(src1, src2, dst, scale, dtype));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src1 = NULL;
    UMat src1;
    PyObject* pyobj_src2 = NULL;
    UMat src2;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    double scale=1;
    int dtype=-1;

    const char* keywords[] = { "src1", "src2", "dst", "scale", "dtype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Odi:divide", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &scale, &dtype) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::divide(src1, src2, dst, scale, dtype));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    double scale=0;
    PyObject* pyobj_src2 = NULL;
    Mat src2;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int dtype=-1;

    const char* keywords[] = { "scale", "src2", "dst", "dtype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "dO|Oi:divide", (char**)keywords, &scale, &pyobj_src2, &pyobj_dst, &dtype) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::divide(scale, src2, dst, dtype));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    double scale=0;
    PyObject* pyobj_src2 = NULL;
    UMat src2;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int dtype=-1;

    const char* keywords[] = { "scale", "src2", "dst", "dtype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "dO|Oi:divide", (char**)keywords, &scale, &pyobj_src2, &pyobj_dst, &dtype) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::divide(scale, src2, dst, dtype));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_drawChessboardCorners(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_patternSize = NULL;
    Size patternSize;
    PyObject* pyobj_corners = NULL;
    Mat corners;
    bool patternWasFound=0;

    const char* keywords[] = { "image", "patternSize", "corners", "patternWasFound", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOb:drawChessboardCorners", (char**)keywords, &pyobj_image, &pyobj_patternSize, &pyobj_corners, &patternWasFound) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 1)) &&
        pyopencv_to(pyobj_patternSize, patternSize, ArgInfo("patternSize", 0)) &&
        pyopencv_to(pyobj_corners, corners, ArgInfo("corners", 0)) )
    {
        ERRWRAP2(cv::drawChessboardCorners(image, patternSize, corners, patternWasFound));
        return pyopencv_from(image);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_patternSize = NULL;
    Size patternSize;
    PyObject* pyobj_corners = NULL;
    UMat corners;
    bool patternWasFound=0;

    const char* keywords[] = { "image", "patternSize", "corners", "patternWasFound", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOb:drawChessboardCorners", (char**)keywords, &pyobj_image, &pyobj_patternSize, &pyobj_corners, &patternWasFound) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 1)) &&
        pyopencv_to(pyobj_patternSize, patternSize, ArgInfo("patternSize", 0)) &&
        pyopencv_to(pyobj_corners, corners, ArgInfo("corners", 0)) )
    {
        ERRWRAP2(cv::drawChessboardCorners(image, patternSize, corners, patternWasFound));
        return pyopencv_from(image);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_drawContours(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_contours = NULL;
    vector_Mat contours;
    int contourIdx=0;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int thickness=1;
    int lineType=LINE_8;
    PyObject* pyobj_hierarchy = NULL;
    Mat hierarchy;
    int maxLevel=INT_MAX;
    PyObject* pyobj_offset = NULL;
    Point offset;

    const char* keywords[] = { "image", "contours", "contourIdx", "color", "thickness", "lineType", "hierarchy", "maxLevel", "offset", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOiO|iiOiO:drawContours", (char**)keywords, &pyobj_image, &pyobj_contours, &contourIdx, &pyobj_color, &thickness, &lineType, &pyobj_hierarchy, &maxLevel, &pyobj_offset) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 1)) &&
        pyopencv_to(pyobj_contours, contours, ArgInfo("contours", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) &&
        pyopencv_to(pyobj_hierarchy, hierarchy, ArgInfo("hierarchy", 0)) &&
        pyopencv_to(pyobj_offset, offset, ArgInfo("offset", 0)) )
    {
        ERRWRAP2(cv::drawContours(image, contours, contourIdx, color, thickness, lineType, hierarchy, maxLevel, offset));
        return pyopencv_from(image);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_contours = NULL;
    vector_Mat contours;
    int contourIdx=0;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int thickness=1;
    int lineType=LINE_8;
    PyObject* pyobj_hierarchy = NULL;
    UMat hierarchy;
    int maxLevel=INT_MAX;
    PyObject* pyobj_offset = NULL;
    Point offset;

    const char* keywords[] = { "image", "contours", "contourIdx", "color", "thickness", "lineType", "hierarchy", "maxLevel", "offset", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOiO|iiOiO:drawContours", (char**)keywords, &pyobj_image, &pyobj_contours, &contourIdx, &pyobj_color, &thickness, &lineType, &pyobj_hierarchy, &maxLevel, &pyobj_offset) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 1)) &&
        pyopencv_to(pyobj_contours, contours, ArgInfo("contours", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) &&
        pyopencv_to(pyobj_hierarchy, hierarchy, ArgInfo("hierarchy", 0)) &&
        pyopencv_to(pyobj_offset, offset, ArgInfo("offset", 0)) )
    {
        ERRWRAP2(cv::drawContours(image, contours, contourIdx, color, thickness, lineType, hierarchy, maxLevel, offset));
        return pyopencv_from(image);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_drawKeypoints(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_keypoints = NULL;
    vector_KeyPoint keypoints;
    PyObject* pyobj_outImage = NULL;
    Mat outImage;
    PyObject* pyobj_color = NULL;
    Scalar color=Scalar::all(-1);
    int flags=DrawMatchesFlags::DEFAULT;

    const char* keywords[] = { "image", "keypoints", "outImage", "color", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|Oi:drawKeypoints", (char**)keywords, &pyobj_image, &pyobj_keypoints, &pyobj_outImage, &pyobj_color, &flags) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_keypoints, keypoints, ArgInfo("keypoints", 0)) &&
        pyopencv_to(pyobj_outImage, outImage, ArgInfo("outImage", 1)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2(cv::drawKeypoints(image, keypoints, outImage, color, flags));
        return pyopencv_from(outImage);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_keypoints = NULL;
    vector_KeyPoint keypoints;
    PyObject* pyobj_outImage = NULL;
    UMat outImage;
    PyObject* pyobj_color = NULL;
    Scalar color=Scalar::all(-1);
    int flags=DrawMatchesFlags::DEFAULT;

    const char* keywords[] = { "image", "keypoints", "outImage", "color", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|Oi:drawKeypoints", (char**)keywords, &pyobj_image, &pyobj_keypoints, &pyobj_outImage, &pyobj_color, &flags) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_keypoints, keypoints, ArgInfo("keypoints", 0)) &&
        pyopencv_to(pyobj_outImage, outImage, ArgInfo("outImage", 1)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2(cv::drawKeypoints(image, keypoints, outImage, color, flags));
        return pyopencv_from(outImage);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_drawMarker(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_position = NULL;
    Point position;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int markerType=MARKER_CROSS;
    int markerSize=20;
    int thickness=1;
    int line_type=8;

    const char* keywords[] = { "img", "position", "color", "markerType", "markerSize", "thickness", "line_type", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|iiii:drawMarker", (char**)keywords, &pyobj_img, &pyobj_position, &pyobj_color, &markerType, &markerSize, &thickness, &line_type) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_position, position, ArgInfo("position", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2(cv::drawMarker(img, position, color, markerType, markerSize, thickness, line_type));
        return pyopencv_from(img);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_position = NULL;
    Point position;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int markerType=MARKER_CROSS;
    int markerSize=20;
    int thickness=1;
    int line_type=8;

    const char* keywords[] = { "img", "position", "color", "markerType", "markerSize", "thickness", "line_type", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|iiii:drawMarker", (char**)keywords, &pyobj_img, &pyobj_position, &pyobj_color, &markerType, &markerSize, &thickness, &line_type) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_position, position, ArgInfo("position", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2(cv::drawMarker(img, position, color, markerType, markerSize, thickness, line_type));
        return pyopencv_from(img);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_drawMatches(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_img1 = NULL;
    Mat img1;
    PyObject* pyobj_keypoints1 = NULL;
    vector_KeyPoint keypoints1;
    PyObject* pyobj_img2 = NULL;
    Mat img2;
    PyObject* pyobj_keypoints2 = NULL;
    vector_KeyPoint keypoints2;
    PyObject* pyobj_matches1to2 = NULL;
    vector_DMatch matches1to2;
    PyObject* pyobj_outImg = NULL;
    Mat outImg;
    PyObject* pyobj_matchColor = NULL;
    Scalar matchColor=Scalar::all(-1);
    PyObject* pyobj_singlePointColor = NULL;
    Scalar singlePointColor=Scalar::all(-1);
    PyObject* pyobj_matchesMask = NULL;
    vector_char matchesMask=std::vector<char>();
    int flags=DrawMatchesFlags::DEFAULT;

    const char* keywords[] = { "img1", "keypoints1", "img2", "keypoints2", "matches1to2", "outImg", "matchColor", "singlePointColor", "matchesMask", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOO|OOOi:drawMatches", (char**)keywords, &pyobj_img1, &pyobj_keypoints1, &pyobj_img2, &pyobj_keypoints2, &pyobj_matches1to2, &pyobj_outImg, &pyobj_matchColor, &pyobj_singlePointColor, &pyobj_matchesMask, &flags) &&
        pyopencv_to(pyobj_img1, img1, ArgInfo("img1", 0)) &&
        pyopencv_to(pyobj_keypoints1, keypoints1, ArgInfo("keypoints1", 0)) &&
        pyopencv_to(pyobj_img2, img2, ArgInfo("img2", 0)) &&
        pyopencv_to(pyobj_keypoints2, keypoints2, ArgInfo("keypoints2", 0)) &&
        pyopencv_to(pyobj_matches1to2, matches1to2, ArgInfo("matches1to2", 0)) &&
        pyopencv_to(pyobj_outImg, outImg, ArgInfo("outImg", 1)) &&
        pyopencv_to(pyobj_matchColor, matchColor, ArgInfo("matchColor", 0)) &&
        pyopencv_to(pyobj_singlePointColor, singlePointColor, ArgInfo("singlePointColor", 0)) &&
        pyopencv_to(pyobj_matchesMask, matchesMask, ArgInfo("matchesMask", 0)) )
    {
        ERRWRAP2(cv::drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg, matchColor, singlePointColor, matchesMask, flags));
        return pyopencv_from(outImg);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_img1 = NULL;
    UMat img1;
    PyObject* pyobj_keypoints1 = NULL;
    vector_KeyPoint keypoints1;
    PyObject* pyobj_img2 = NULL;
    UMat img2;
    PyObject* pyobj_keypoints2 = NULL;
    vector_KeyPoint keypoints2;
    PyObject* pyobj_matches1to2 = NULL;
    vector_DMatch matches1to2;
    PyObject* pyobj_outImg = NULL;
    UMat outImg;
    PyObject* pyobj_matchColor = NULL;
    Scalar matchColor=Scalar::all(-1);
    PyObject* pyobj_singlePointColor = NULL;
    Scalar singlePointColor=Scalar::all(-1);
    PyObject* pyobj_matchesMask = NULL;
    vector_char matchesMask=std::vector<char>();
    int flags=DrawMatchesFlags::DEFAULT;

    const char* keywords[] = { "img1", "keypoints1", "img2", "keypoints2", "matches1to2", "outImg", "matchColor", "singlePointColor", "matchesMask", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOO|OOOi:drawMatches", (char**)keywords, &pyobj_img1, &pyobj_keypoints1, &pyobj_img2, &pyobj_keypoints2, &pyobj_matches1to2, &pyobj_outImg, &pyobj_matchColor, &pyobj_singlePointColor, &pyobj_matchesMask, &flags) &&
        pyopencv_to(pyobj_img1, img1, ArgInfo("img1", 0)) &&
        pyopencv_to(pyobj_keypoints1, keypoints1, ArgInfo("keypoints1", 0)) &&
        pyopencv_to(pyobj_img2, img2, ArgInfo("img2", 0)) &&
        pyopencv_to(pyobj_keypoints2, keypoints2, ArgInfo("keypoints2", 0)) &&
        pyopencv_to(pyobj_matches1to2, matches1to2, ArgInfo("matches1to2", 0)) &&
        pyopencv_to(pyobj_outImg, outImg, ArgInfo("outImg", 1)) &&
        pyopencv_to(pyobj_matchColor, matchColor, ArgInfo("matchColor", 0)) &&
        pyopencv_to(pyobj_singlePointColor, singlePointColor, ArgInfo("singlePointColor", 0)) &&
        pyopencv_to(pyobj_matchesMask, matchesMask, ArgInfo("matchesMask", 0)) )
    {
        ERRWRAP2(cv::drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg, matchColor, singlePointColor, matchesMask, flags));
        return pyopencv_from(outImg);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_drawMatchesKnn(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_img1 = NULL;
    Mat img1;
    PyObject* pyobj_keypoints1 = NULL;
    vector_KeyPoint keypoints1;
    PyObject* pyobj_img2 = NULL;
    Mat img2;
    PyObject* pyobj_keypoints2 = NULL;
    vector_KeyPoint keypoints2;
    PyObject* pyobj_matches1to2 = NULL;
    vector_vector_DMatch matches1to2;
    PyObject* pyobj_outImg = NULL;
    Mat outImg;
    PyObject* pyobj_matchColor = NULL;
    Scalar matchColor=Scalar::all(-1);
    PyObject* pyobj_singlePointColor = NULL;
    Scalar singlePointColor=Scalar::all(-1);
    PyObject* pyobj_matchesMask = NULL;
    vector_vector_char matchesMask=std::vector<std::vector<char> >();
    int flags=DrawMatchesFlags::DEFAULT;

    const char* keywords[] = { "img1", "keypoints1", "img2", "keypoints2", "matches1to2", "outImg", "matchColor", "singlePointColor", "matchesMask", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOO|OOOi:drawMatchesKnn", (char**)keywords, &pyobj_img1, &pyobj_keypoints1, &pyobj_img2, &pyobj_keypoints2, &pyobj_matches1to2, &pyobj_outImg, &pyobj_matchColor, &pyobj_singlePointColor, &pyobj_matchesMask, &flags) &&
        pyopencv_to(pyobj_img1, img1, ArgInfo("img1", 0)) &&
        pyopencv_to(pyobj_keypoints1, keypoints1, ArgInfo("keypoints1", 0)) &&
        pyopencv_to(pyobj_img2, img2, ArgInfo("img2", 0)) &&
        pyopencv_to(pyobj_keypoints2, keypoints2, ArgInfo("keypoints2", 0)) &&
        pyopencv_to(pyobj_matches1to2, matches1to2, ArgInfo("matches1to2", 0)) &&
        pyopencv_to(pyobj_outImg, outImg, ArgInfo("outImg", 1)) &&
        pyopencv_to(pyobj_matchColor, matchColor, ArgInfo("matchColor", 0)) &&
        pyopencv_to(pyobj_singlePointColor, singlePointColor, ArgInfo("singlePointColor", 0)) &&
        pyopencv_to(pyobj_matchesMask, matchesMask, ArgInfo("matchesMask", 0)) )
    {
        ERRWRAP2(cv::drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg, matchColor, singlePointColor, matchesMask, flags));
        return pyopencv_from(outImg);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_img1 = NULL;
    UMat img1;
    PyObject* pyobj_keypoints1 = NULL;
    vector_KeyPoint keypoints1;
    PyObject* pyobj_img2 = NULL;
    UMat img2;
    PyObject* pyobj_keypoints2 = NULL;
    vector_KeyPoint keypoints2;
    PyObject* pyobj_matches1to2 = NULL;
    vector_vector_DMatch matches1to2;
    PyObject* pyobj_outImg = NULL;
    UMat outImg;
    PyObject* pyobj_matchColor = NULL;
    Scalar matchColor=Scalar::all(-1);
    PyObject* pyobj_singlePointColor = NULL;
    Scalar singlePointColor=Scalar::all(-1);
    PyObject* pyobj_matchesMask = NULL;
    vector_vector_char matchesMask=std::vector<std::vector<char> >();
    int flags=DrawMatchesFlags::DEFAULT;

    const char* keywords[] = { "img1", "keypoints1", "img2", "keypoints2", "matches1to2", "outImg", "matchColor", "singlePointColor", "matchesMask", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOO|OOOi:drawMatchesKnn", (char**)keywords, &pyobj_img1, &pyobj_keypoints1, &pyobj_img2, &pyobj_keypoints2, &pyobj_matches1to2, &pyobj_outImg, &pyobj_matchColor, &pyobj_singlePointColor, &pyobj_matchesMask, &flags) &&
        pyopencv_to(pyobj_img1, img1, ArgInfo("img1", 0)) &&
        pyopencv_to(pyobj_keypoints1, keypoints1, ArgInfo("keypoints1", 0)) &&
        pyopencv_to(pyobj_img2, img2, ArgInfo("img2", 0)) &&
        pyopencv_to(pyobj_keypoints2, keypoints2, ArgInfo("keypoints2", 0)) &&
        pyopencv_to(pyobj_matches1to2, matches1to2, ArgInfo("matches1to2", 0)) &&
        pyopencv_to(pyobj_outImg, outImg, ArgInfo("outImg", 1)) &&
        pyopencv_to(pyobj_matchColor, matchColor, ArgInfo("matchColor", 0)) &&
        pyopencv_to(pyobj_singlePointColor, singlePointColor, ArgInfo("singlePointColor", 0)) &&
        pyopencv_to(pyobj_matchesMask, matchesMask, ArgInfo("matchesMask", 0)) )
    {
        ERRWRAP2(cv::drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg, matchColor, singlePointColor, matchesMask, flags));
        return pyopencv_from(outImg);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_edgePreservingFilter(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int flags=1;
    float sigma_s=60;
    float sigma_r=0.4f;

    const char* keywords[] = { "src", "dst", "flags", "sigma_s", "sigma_r", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oiff:edgePreservingFilter", (char**)keywords, &pyobj_src, &pyobj_dst, &flags, &sigma_s, &sigma_r) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::edgePreservingFilter(src, dst, flags, sigma_s, sigma_r));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int flags=1;
    float sigma_s=60;
    float sigma_r=0.4f;

    const char* keywords[] = { "src", "dst", "flags", "sigma_s", "sigma_r", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oiff:edgePreservingFilter", (char**)keywords, &pyobj_src, &pyobj_dst, &flags, &sigma_s, &sigma_r) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::edgePreservingFilter(src, dst, flags, sigma_s, sigma_r));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_eigen(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_eigenvalues = NULL;
    Mat eigenvalues;
    PyObject* pyobj_eigenvectors = NULL;
    Mat eigenvectors;
    bool retval;

    const char* keywords[] = { "src", "eigenvalues", "eigenvectors", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OO:eigen", (char**)keywords, &pyobj_src, &pyobj_eigenvalues, &pyobj_eigenvectors) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_eigenvalues, eigenvalues, ArgInfo("eigenvalues", 1)) &&
        pyopencv_to(pyobj_eigenvectors, eigenvectors, ArgInfo("eigenvectors", 1)) )
    {
        ERRWRAP2(retval = cv::eigen(src, eigenvalues, eigenvectors));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(eigenvalues), pyopencv_from(eigenvectors));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_eigenvalues = NULL;
    UMat eigenvalues;
    PyObject* pyobj_eigenvectors = NULL;
    UMat eigenvectors;
    bool retval;

    const char* keywords[] = { "src", "eigenvalues", "eigenvectors", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OO:eigen", (char**)keywords, &pyobj_src, &pyobj_eigenvalues, &pyobj_eigenvectors) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_eigenvalues, eigenvalues, ArgInfo("eigenvalues", 1)) &&
        pyopencv_to(pyobj_eigenvectors, eigenvectors, ArgInfo("eigenvectors", 1)) )
    {
        ERRWRAP2(retval = cv::eigen(src, eigenvalues, eigenvectors));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(eigenvalues), pyopencv_from(eigenvectors));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ellipse(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_center = NULL;
    Point center;
    PyObject* pyobj_axes = NULL;
    Size axes;
    double angle=0;
    double startAngle=0;
    double endAngle=0;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int thickness=1;
    int lineType=LINE_8;
    int shift=0;

    const char* keywords[] = { "img", "center", "axes", "angle", "startAngle", "endAngle", "color", "thickness", "lineType", "shift", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOdddO|iii:ellipse", (char**)keywords, &pyobj_img, &pyobj_center, &pyobj_axes, &angle, &startAngle, &endAngle, &pyobj_color, &thickness, &lineType, &shift) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_center, center, ArgInfo("center", 0)) &&
        pyopencv_to(pyobj_axes, axes, ArgInfo("axes", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2(cv::ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness, lineType, shift));
        return pyopencv_from(img);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_img = NULL;
    UMat img;
    PyObject* pyobj_center = NULL;
    Point center;
    PyObject* pyobj_axes = NULL;
    Size axes;
    double angle=0;
    double startAngle=0;
    double endAngle=0;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int thickness=1;
    int lineType=LINE_8;
    int shift=0;

    const char* keywords[] = { "img", "center", "axes", "angle", "startAngle", "endAngle", "color", "thickness", "lineType", "shift", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOdddO|iii:ellipse", (char**)keywords, &pyobj_img, &pyobj_center, &pyobj_axes, &angle, &startAngle, &endAngle, &pyobj_color, &thickness, &lineType, &shift) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_center, center, ArgInfo("center", 0)) &&
        pyopencv_to(pyobj_axes, axes, ArgInfo("axes", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2(cv::ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness, lineType, shift));
        return pyopencv_from(img);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_box = NULL;
    RotatedRect box;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int thickness=1;
    int lineType=LINE_8;

    const char* keywords[] = { "img", "box", "color", "thickness", "lineType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|ii:ellipse", (char**)keywords, &pyobj_img, &pyobj_box, &pyobj_color, &thickness, &lineType) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_box, box, ArgInfo("box", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2(cv::ellipse(img, box, color, thickness, lineType));
        return pyopencv_from(img);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_img = NULL;
    UMat img;
    PyObject* pyobj_box = NULL;
    RotatedRect box;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int thickness=1;
    int lineType=LINE_8;

    const char* keywords[] = { "img", "box", "color", "thickness", "lineType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|ii:ellipse", (char**)keywords, &pyobj_img, &pyobj_box, &pyobj_color, &thickness, &lineType) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_box, box, ArgInfo("box", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2(cv::ellipse(img, box, color, thickness, lineType));
        return pyopencv_from(img);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ellipse2Poly(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_center = NULL;
    Point center;
    PyObject* pyobj_axes = NULL;
    Size axes;
    int angle=0;
    int arcStart=0;
    int arcEnd=0;
    int delta=0;
    vector_Point pts;

    const char* keywords[] = { "center", "axes", "angle", "arcStart", "arcEnd", "delta", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOiiii:ellipse2Poly", (char**)keywords, &pyobj_center, &pyobj_axes, &angle, &arcStart, &arcEnd, &delta) &&
        pyopencv_to(pyobj_center, center, ArgInfo("center", 0)) &&
        pyopencv_to(pyobj_axes, axes, ArgInfo("axes", 0)) )
    {
        ERRWRAP2(cv::ellipse2Poly(center, axes, angle, arcStart, arcEnd, delta, pts));
        return pyopencv_from(pts);
    }

    return NULL;
}

static PyObject* pyopencv_cv_equalizeHist(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:equalizeHist", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::equalizeHist(src, dst));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:equalizeHist", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::equalizeHist(src, dst));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_erode(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_kernel = NULL;
    Mat kernel;
    PyObject* pyobj_anchor = NULL;
    Point anchor=Point(-1,-1);
    int iterations=1;
    int borderType=BORDER_CONSTANT;
    PyObject* pyobj_borderValue = NULL;
    Scalar borderValue=morphologyDefaultBorderValue();

    const char* keywords[] = { "src", "kernel", "dst", "anchor", "iterations", "borderType", "borderValue", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOiiO:erode", (char**)keywords, &pyobj_src, &pyobj_kernel, &pyobj_dst, &pyobj_anchor, &iterations, &borderType, &pyobj_borderValue) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 0)) &&
        pyopencv_to(pyobj_anchor, anchor, ArgInfo("anchor", 0)) &&
        pyopencv_to(pyobj_borderValue, borderValue, ArgInfo("borderValue", 0)) )
    {
        ERRWRAP2(cv::erode(src, dst, kernel, anchor, iterations, borderType, borderValue));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_kernel = NULL;
    UMat kernel;
    PyObject* pyobj_anchor = NULL;
    Point anchor=Point(-1,-1);
    int iterations=1;
    int borderType=BORDER_CONSTANT;
    PyObject* pyobj_borderValue = NULL;
    Scalar borderValue=morphologyDefaultBorderValue();

    const char* keywords[] = { "src", "kernel", "dst", "anchor", "iterations", "borderType", "borderValue", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOiiO:erode", (char**)keywords, &pyobj_src, &pyobj_kernel, &pyobj_dst, &pyobj_anchor, &iterations, &borderType, &pyobj_borderValue) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 0)) &&
        pyopencv_to(pyobj_anchor, anchor, ArgInfo("anchor", 0)) &&
        pyopencv_to(pyobj_borderValue, borderValue, ArgInfo("borderValue", 0)) )
    {
        ERRWRAP2(cv::erode(src, dst, kernel, anchor, iterations, borderType, borderValue));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_estimateAffine2D(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_from = NULL;
    Mat from;
    PyObject* pyobj_to = NULL;
    Mat to;
    PyObject* pyobj_inliers = NULL;
    Mat inliers;
    int method=RANSAC;
    double ransacReprojThreshold=3;
    size_t maxIters=2000;
    double confidence=0.99;
    size_t refineIters=10;
    cv::Mat retval;

    const char* keywords[] = { "from", "to", "inliers", "method", "ransacReprojThreshold", "maxIters", "confidence", "refineIters", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OidIdI:estimateAffine2D", (char**)keywords, &pyobj_from, &pyobj_to, &pyobj_inliers, &method, &ransacReprojThreshold, &maxIters, &confidence, &refineIters) &&
        pyopencv_to(pyobj_from, from, ArgInfo("from", 0)) &&
        pyopencv_to(pyobj_to, to, ArgInfo("to", 0)) &&
        pyopencv_to(pyobj_inliers, inliers, ArgInfo("inliers", 1)) )
    {
        ERRWRAP2(retval = cv::estimateAffine2D(from, to, inliers, method, ransacReprojThreshold, maxIters, confidence, refineIters));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(inliers));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_from = NULL;
    UMat from;
    PyObject* pyobj_to = NULL;
    UMat to;
    PyObject* pyobj_inliers = NULL;
    UMat inliers;
    int method=RANSAC;
    double ransacReprojThreshold=3;
    size_t maxIters=2000;
    double confidence=0.99;
    size_t refineIters=10;
    cv::Mat retval;

    const char* keywords[] = { "from", "to", "inliers", "method", "ransacReprojThreshold", "maxIters", "confidence", "refineIters", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OidIdI:estimateAffine2D", (char**)keywords, &pyobj_from, &pyobj_to, &pyobj_inliers, &method, &ransacReprojThreshold, &maxIters, &confidence, &refineIters) &&
        pyopencv_to(pyobj_from, from, ArgInfo("from", 0)) &&
        pyopencv_to(pyobj_to, to, ArgInfo("to", 0)) &&
        pyopencv_to(pyobj_inliers, inliers, ArgInfo("inliers", 1)) )
    {
        ERRWRAP2(retval = cv::estimateAffine2D(from, to, inliers, method, ransacReprojThreshold, maxIters, confidence, refineIters));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(inliers));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_estimateAffine3D(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_out = NULL;
    Mat out;
    PyObject* pyobj_inliers = NULL;
    Mat inliers;
    double ransacThreshold=3;
    double confidence=0.99;
    int retval;

    const char* keywords[] = { "src", "dst", "out", "inliers", "ransacThreshold", "confidence", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOdd:estimateAffine3D", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_out, &pyobj_inliers, &ransacThreshold, &confidence) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 0)) &&
        pyopencv_to(pyobj_out, out, ArgInfo("out", 1)) &&
        pyopencv_to(pyobj_inliers, inliers, ArgInfo("inliers", 1)) )
    {
        ERRWRAP2(retval = cv::estimateAffine3D(src, dst, out, inliers, ransacThreshold, confidence));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(out), pyopencv_from(inliers));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_out = NULL;
    UMat out;
    PyObject* pyobj_inliers = NULL;
    UMat inliers;
    double ransacThreshold=3;
    double confidence=0.99;
    int retval;

    const char* keywords[] = { "src", "dst", "out", "inliers", "ransacThreshold", "confidence", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOdd:estimateAffine3D", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_out, &pyobj_inliers, &ransacThreshold, &confidence) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 0)) &&
        pyopencv_to(pyobj_out, out, ArgInfo("out", 1)) &&
        pyopencv_to(pyobj_inliers, inliers, ArgInfo("inliers", 1)) )
    {
        ERRWRAP2(retval = cv::estimateAffine3D(src, dst, out, inliers, ransacThreshold, confidence));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(out), pyopencv_from(inliers));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_estimateAffinePartial2D(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_from = NULL;
    Mat from;
    PyObject* pyobj_to = NULL;
    Mat to;
    PyObject* pyobj_inliers = NULL;
    Mat inliers;
    int method=RANSAC;
    double ransacReprojThreshold=3;
    size_t maxIters=2000;
    double confidence=0.99;
    size_t refineIters=10;
    cv::Mat retval;

    const char* keywords[] = { "from", "to", "inliers", "method", "ransacReprojThreshold", "maxIters", "confidence", "refineIters", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OidIdI:estimateAffinePartial2D", (char**)keywords, &pyobj_from, &pyobj_to, &pyobj_inliers, &method, &ransacReprojThreshold, &maxIters, &confidence, &refineIters) &&
        pyopencv_to(pyobj_from, from, ArgInfo("from", 0)) &&
        pyopencv_to(pyobj_to, to, ArgInfo("to", 0)) &&
        pyopencv_to(pyobj_inliers, inliers, ArgInfo("inliers", 1)) )
    {
        ERRWRAP2(retval = cv::estimateAffinePartial2D(from, to, inliers, method, ransacReprojThreshold, maxIters, confidence, refineIters));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(inliers));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_from = NULL;
    UMat from;
    PyObject* pyobj_to = NULL;
    UMat to;
    PyObject* pyobj_inliers = NULL;
    UMat inliers;
    int method=RANSAC;
    double ransacReprojThreshold=3;
    size_t maxIters=2000;
    double confidence=0.99;
    size_t refineIters=10;
    cv::Mat retval;

    const char* keywords[] = { "from", "to", "inliers", "method", "ransacReprojThreshold", "maxIters", "confidence", "refineIters", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OidIdI:estimateAffinePartial2D", (char**)keywords, &pyobj_from, &pyobj_to, &pyobj_inliers, &method, &ransacReprojThreshold, &maxIters, &confidence, &refineIters) &&
        pyopencv_to(pyobj_from, from, ArgInfo("from", 0)) &&
        pyopencv_to(pyobj_to, to, ArgInfo("to", 0)) &&
        pyopencv_to(pyobj_inliers, inliers, ArgInfo("inliers", 1)) )
    {
        ERRWRAP2(retval = cv::estimateAffinePartial2D(from, to, inliers, method, ransacReprojThreshold, maxIters, confidence, refineIters));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(inliers));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_estimateRigidTransform(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    bool fullAffine=0;
    Mat retval;

    const char* keywords[] = { "src", "dst", "fullAffine", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOb:estimateRigidTransform", (char**)keywords, &pyobj_src, &pyobj_dst, &fullAffine) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 0)) )
    {
        ERRWRAP2(retval = cv::estimateRigidTransform(src, dst, fullAffine));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    bool fullAffine=0;
    Mat retval;

    const char* keywords[] = { "src", "dst", "fullAffine", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOb:estimateRigidTransform", (char**)keywords, &pyobj_src, &pyobj_dst, &fullAffine) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 0)) )
    {
        ERRWRAP2(retval = cv::estimateRigidTransform(src, dst, fullAffine));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_exp(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:exp", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::exp(src, dst));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:exp", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::exp(src, dst));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_extractChannel(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int coi=0;

    const char* keywords[] = { "src", "coi", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|O:extractChannel", (char**)keywords, &pyobj_src, &coi, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::extractChannel(src, dst, coi));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int coi=0;

    const char* keywords[] = { "src", "coi", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|O:extractChannel", (char**)keywords, &pyobj_src, &coi, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::extractChannel(src, dst, coi));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_fastAtan2(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    float y=0.f;
    float x=0.f;
    float retval;

    const char* keywords[] = { "y", "x", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "ff:fastAtan2", (char**)keywords, &y, &x) )
    {
        ERRWRAP2(retval = cv::fastAtan2(y, x));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_fastNlMeansDenoising(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    float h=3;
    int templateWindowSize=7;
    int searchWindowSize=21;

    const char* keywords[] = { "src", "dst", "h", "templateWindowSize", "searchWindowSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Ofii:fastNlMeansDenoising", (char**)keywords, &pyobj_src, &pyobj_dst, &h, &templateWindowSize, &searchWindowSize) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::fastNlMeansDenoising(src, dst, h, templateWindowSize, searchWindowSize));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    float h=3;
    int templateWindowSize=7;
    int searchWindowSize=21;

    const char* keywords[] = { "src", "dst", "h", "templateWindowSize", "searchWindowSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Ofii:fastNlMeansDenoising", (char**)keywords, &pyobj_src, &pyobj_dst, &h, &templateWindowSize, &searchWindowSize) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::fastNlMeansDenoising(src, dst, h, templateWindowSize, searchWindowSize));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_h = NULL;
    vector_float h;
    int templateWindowSize=7;
    int searchWindowSize=21;
    int normType=NORM_L2;

    const char* keywords[] = { "src", "h", "dst", "templateWindowSize", "searchWindowSize", "normType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Oiii:fastNlMeansDenoising", (char**)keywords, &pyobj_src, &pyobj_h, &pyobj_dst, &templateWindowSize, &searchWindowSize, &normType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_h, h, ArgInfo("h", 0)) )
    {
        ERRWRAP2(cv::fastNlMeansDenoising(src, dst, h, templateWindowSize, searchWindowSize, normType));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_h = NULL;
    vector_float h;
    int templateWindowSize=7;
    int searchWindowSize=21;
    int normType=NORM_L2;

    const char* keywords[] = { "src", "h", "dst", "templateWindowSize", "searchWindowSize", "normType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Oiii:fastNlMeansDenoising", (char**)keywords, &pyobj_src, &pyobj_h, &pyobj_dst, &templateWindowSize, &searchWindowSize, &normType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_h, h, ArgInfo("h", 0)) )
    {
        ERRWRAP2(cv::fastNlMeansDenoising(src, dst, h, templateWindowSize, searchWindowSize, normType));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_fastNlMeansDenoisingColored(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    float h=3;
    float hColor=3;
    int templateWindowSize=7;
    int searchWindowSize=21;

    const char* keywords[] = { "src", "dst", "h", "hColor", "templateWindowSize", "searchWindowSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Offii:fastNlMeansDenoisingColored", (char**)keywords, &pyobj_src, &pyobj_dst, &h, &hColor, &templateWindowSize, &searchWindowSize) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::fastNlMeansDenoisingColored(src, dst, h, hColor, templateWindowSize, searchWindowSize));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    float h=3;
    float hColor=3;
    int templateWindowSize=7;
    int searchWindowSize=21;

    const char* keywords[] = { "src", "dst", "h", "hColor", "templateWindowSize", "searchWindowSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Offii:fastNlMeansDenoisingColored", (char**)keywords, &pyobj_src, &pyobj_dst, &h, &hColor, &templateWindowSize, &searchWindowSize) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::fastNlMeansDenoisingColored(src, dst, h, hColor, templateWindowSize, searchWindowSize));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_fastNlMeansDenoisingColoredMulti(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_srcImgs = NULL;
    vector_Mat srcImgs;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int imgToDenoiseIndex=0;
    int temporalWindowSize=0;
    float h=3;
    float hColor=3;
    int templateWindowSize=7;
    int searchWindowSize=21;

    const char* keywords[] = { "srcImgs", "imgToDenoiseIndex", "temporalWindowSize", "dst", "h", "hColor", "templateWindowSize", "searchWindowSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii|Offii:fastNlMeansDenoisingColoredMulti", (char**)keywords, &pyobj_srcImgs, &imgToDenoiseIndex, &temporalWindowSize, &pyobj_dst, &h, &hColor, &templateWindowSize, &searchWindowSize) &&
        pyopencv_to(pyobj_srcImgs, srcImgs, ArgInfo("srcImgs", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::fastNlMeansDenoisingColoredMulti(srcImgs, dst, imgToDenoiseIndex, temporalWindowSize, h, hColor, templateWindowSize, searchWindowSize));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_srcImgs = NULL;
    vector_Mat srcImgs;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int imgToDenoiseIndex=0;
    int temporalWindowSize=0;
    float h=3;
    float hColor=3;
    int templateWindowSize=7;
    int searchWindowSize=21;

    const char* keywords[] = { "srcImgs", "imgToDenoiseIndex", "temporalWindowSize", "dst", "h", "hColor", "templateWindowSize", "searchWindowSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii|Offii:fastNlMeansDenoisingColoredMulti", (char**)keywords, &pyobj_srcImgs, &imgToDenoiseIndex, &temporalWindowSize, &pyobj_dst, &h, &hColor, &templateWindowSize, &searchWindowSize) &&
        pyopencv_to(pyobj_srcImgs, srcImgs, ArgInfo("srcImgs", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::fastNlMeansDenoisingColoredMulti(srcImgs, dst, imgToDenoiseIndex, temporalWindowSize, h, hColor, templateWindowSize, searchWindowSize));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_fastNlMeansDenoisingMulti(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_srcImgs = NULL;
    vector_Mat srcImgs;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int imgToDenoiseIndex=0;
    int temporalWindowSize=0;
    float h=3;
    int templateWindowSize=7;
    int searchWindowSize=21;

    const char* keywords[] = { "srcImgs", "imgToDenoiseIndex", "temporalWindowSize", "dst", "h", "templateWindowSize", "searchWindowSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii|Ofii:fastNlMeansDenoisingMulti", (char**)keywords, &pyobj_srcImgs, &imgToDenoiseIndex, &temporalWindowSize, &pyobj_dst, &h, &templateWindowSize, &searchWindowSize) &&
        pyopencv_to(pyobj_srcImgs, srcImgs, ArgInfo("srcImgs", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::fastNlMeansDenoisingMulti(srcImgs, dst, imgToDenoiseIndex, temporalWindowSize, h, templateWindowSize, searchWindowSize));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_srcImgs = NULL;
    vector_Mat srcImgs;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int imgToDenoiseIndex=0;
    int temporalWindowSize=0;
    float h=3;
    int templateWindowSize=7;
    int searchWindowSize=21;

    const char* keywords[] = { "srcImgs", "imgToDenoiseIndex", "temporalWindowSize", "dst", "h", "templateWindowSize", "searchWindowSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii|Ofii:fastNlMeansDenoisingMulti", (char**)keywords, &pyobj_srcImgs, &imgToDenoiseIndex, &temporalWindowSize, &pyobj_dst, &h, &templateWindowSize, &searchWindowSize) &&
        pyopencv_to(pyobj_srcImgs, srcImgs, ArgInfo("srcImgs", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::fastNlMeansDenoisingMulti(srcImgs, dst, imgToDenoiseIndex, temporalWindowSize, h, templateWindowSize, searchWindowSize));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_srcImgs = NULL;
    vector_Mat srcImgs;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int imgToDenoiseIndex=0;
    int temporalWindowSize=0;
    PyObject* pyobj_h = NULL;
    vector_float h;
    int templateWindowSize=7;
    int searchWindowSize=21;
    int normType=NORM_L2;

    const char* keywords[] = { "srcImgs", "imgToDenoiseIndex", "temporalWindowSize", "h", "dst", "templateWindowSize", "searchWindowSize", "normType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OiiO|Oiii:fastNlMeansDenoisingMulti", (char**)keywords, &pyobj_srcImgs, &imgToDenoiseIndex, &temporalWindowSize, &pyobj_h, &pyobj_dst, &templateWindowSize, &searchWindowSize, &normType) &&
        pyopencv_to(pyobj_srcImgs, srcImgs, ArgInfo("srcImgs", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_h, h, ArgInfo("h", 0)) )
    {
        ERRWRAP2(cv::fastNlMeansDenoisingMulti(srcImgs, dst, imgToDenoiseIndex, temporalWindowSize, h, templateWindowSize, searchWindowSize, normType));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_srcImgs = NULL;
    vector_Mat srcImgs;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int imgToDenoiseIndex=0;
    int temporalWindowSize=0;
    PyObject* pyobj_h = NULL;
    vector_float h;
    int templateWindowSize=7;
    int searchWindowSize=21;
    int normType=NORM_L2;

    const char* keywords[] = { "srcImgs", "imgToDenoiseIndex", "temporalWindowSize", "h", "dst", "templateWindowSize", "searchWindowSize", "normType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OiiO|Oiii:fastNlMeansDenoisingMulti", (char**)keywords, &pyobj_srcImgs, &imgToDenoiseIndex, &temporalWindowSize, &pyobj_h, &pyobj_dst, &templateWindowSize, &searchWindowSize, &normType) &&
        pyopencv_to(pyobj_srcImgs, srcImgs, ArgInfo("srcImgs", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_h, h, ArgInfo("h", 0)) )
    {
        ERRWRAP2(cv::fastNlMeansDenoisingMulti(srcImgs, dst, imgToDenoiseIndex, temporalWindowSize, h, templateWindowSize, searchWindowSize, normType));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_fillConvexPoly(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_points = NULL;
    Mat points;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int lineType=LINE_8;
    int shift=0;

    const char* keywords[] = { "img", "points", "color", "lineType", "shift", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|ii:fillConvexPoly", (char**)keywords, &pyobj_img, &pyobj_points, &pyobj_color, &lineType, &shift) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2(cv::fillConvexPoly(img, points, color, lineType, shift));
        return pyopencv_from(img);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_img = NULL;
    UMat img;
    PyObject* pyobj_points = NULL;
    UMat points;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int lineType=LINE_8;
    int shift=0;

    const char* keywords[] = { "img", "points", "color", "lineType", "shift", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|ii:fillConvexPoly", (char**)keywords, &pyobj_img, &pyobj_points, &pyobj_color, &lineType, &shift) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2(cv::fillConvexPoly(img, points, color, lineType, shift));
        return pyopencv_from(img);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_fillPoly(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_pts = NULL;
    vector_Mat pts;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int lineType=LINE_8;
    int shift=0;
    PyObject* pyobj_offset = NULL;
    Point offset;

    const char* keywords[] = { "img", "pts", "color", "lineType", "shift", "offset", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|iiO:fillPoly", (char**)keywords, &pyobj_img, &pyobj_pts, &pyobj_color, &lineType, &shift, &pyobj_offset) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_pts, pts, ArgInfo("pts", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) &&
        pyopencv_to(pyobj_offset, offset, ArgInfo("offset", 0)) )
    {
        ERRWRAP2(cv::fillPoly(img, pts, color, lineType, shift, offset));
        return pyopencv_from(img);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_img = NULL;
    UMat img;
    PyObject* pyobj_pts = NULL;
    vector_Mat pts;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int lineType=LINE_8;
    int shift=0;
    PyObject* pyobj_offset = NULL;
    Point offset;

    const char* keywords[] = { "img", "pts", "color", "lineType", "shift", "offset", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|iiO:fillPoly", (char**)keywords, &pyobj_img, &pyobj_pts, &pyobj_color, &lineType, &shift, &pyobj_offset) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_pts, pts, ArgInfo("pts", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) &&
        pyopencv_to(pyobj_offset, offset, ArgInfo("offset", 0)) )
    {
        ERRWRAP2(cv::fillPoly(img, pts, color, lineType, shift, offset));
        return pyopencv_from(img);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_filter2D(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int ddepth=0;
    PyObject* pyobj_kernel = NULL;
    Mat kernel;
    PyObject* pyobj_anchor = NULL;
    Point anchor=Point(-1,-1);
    double delta=0;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "ddepth", "kernel", "dst", "anchor", "delta", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OiO|OOdi:filter2D", (char**)keywords, &pyobj_src, &ddepth, &pyobj_kernel, &pyobj_dst, &pyobj_anchor, &delta, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 0)) &&
        pyopencv_to(pyobj_anchor, anchor, ArgInfo("anchor", 0)) )
    {
        ERRWRAP2(cv::filter2D(src, dst, ddepth, kernel, anchor, delta, borderType));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int ddepth=0;
    PyObject* pyobj_kernel = NULL;
    UMat kernel;
    PyObject* pyobj_anchor = NULL;
    Point anchor=Point(-1,-1);
    double delta=0;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "ddepth", "kernel", "dst", "anchor", "delta", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OiO|OOdi:filter2D", (char**)keywords, &pyobj_src, &ddepth, &pyobj_kernel, &pyobj_dst, &pyobj_anchor, &delta, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 0)) &&
        pyopencv_to(pyobj_anchor, anchor, ArgInfo("anchor", 0)) )
    {
        ERRWRAP2(cv::filter2D(src, dst, ddepth, kernel, anchor, delta, borderType));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_filterSpeckles(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_img = NULL;
    Mat img;
    double newVal=0;
    int maxSpeckleSize=0;
    double maxDiff=0;
    PyObject* pyobj_buf = NULL;
    Mat buf;

    const char* keywords[] = { "img", "newVal", "maxSpeckleSize", "maxDiff", "buf", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odid|O:filterSpeckles", (char**)keywords, &pyobj_img, &newVal, &maxSpeckleSize, &maxDiff, &pyobj_buf) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_buf, buf, ArgInfo("buf", 1)) )
    {
        ERRWRAP2(cv::filterSpeckles(img, newVal, maxSpeckleSize, maxDiff, buf));
        return Py_BuildValue("(NN)", pyopencv_from(img), pyopencv_from(buf));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_img = NULL;
    UMat img;
    double newVal=0;
    int maxSpeckleSize=0;
    double maxDiff=0;
    PyObject* pyobj_buf = NULL;
    UMat buf;

    const char* keywords[] = { "img", "newVal", "maxSpeckleSize", "maxDiff", "buf", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odid|O:filterSpeckles", (char**)keywords, &pyobj_img, &newVal, &maxSpeckleSize, &maxDiff, &pyobj_buf) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_buf, buf, ArgInfo("buf", 1)) )
    {
        ERRWRAP2(cv::filterSpeckles(img, newVal, maxSpeckleSize, maxDiff, buf));
        return Py_BuildValue("(NN)", pyopencv_from(img), pyopencv_from(buf));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_findChessboardCorners(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_patternSize = NULL;
    Size patternSize;
    PyObject* pyobj_corners = NULL;
    Mat corners;
    int flags=CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE;
    bool retval;

    const char* keywords[] = { "image", "patternSize", "corners", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Oi:findChessboardCorners", (char**)keywords, &pyobj_image, &pyobj_patternSize, &pyobj_corners, &flags) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_patternSize, patternSize, ArgInfo("patternSize", 0)) &&
        pyopencv_to(pyobj_corners, corners, ArgInfo("corners", 1)) )
    {
        ERRWRAP2(retval = cv::findChessboardCorners(image, patternSize, corners, flags));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(corners));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_patternSize = NULL;
    Size patternSize;
    PyObject* pyobj_corners = NULL;
    UMat corners;
    int flags=CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE;
    bool retval;

    const char* keywords[] = { "image", "patternSize", "corners", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Oi:findChessboardCorners", (char**)keywords, &pyobj_image, &pyobj_patternSize, &pyobj_corners, &flags) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_patternSize, patternSize, ArgInfo("patternSize", 0)) &&
        pyopencv_to(pyobj_corners, corners, ArgInfo("corners", 1)) )
    {
        ERRWRAP2(retval = cv::findChessboardCorners(image, patternSize, corners, flags));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(corners));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_findCirclesGrid(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_patternSize = NULL;
    Size patternSize;
    PyObject* pyobj_centers = NULL;
    Mat centers;
    int flags=CALIB_CB_SYMMETRIC_GRID;
    PyObject* pyobj_blobDetector = NULL;
    Ptr<FeatureDetector> blobDetector=SimpleBlobDetector::create();
    bool retval;

    const char* keywords[] = { "image", "patternSize", "centers", "flags", "blobDetector", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OiO:findCirclesGrid", (char**)keywords, &pyobj_image, &pyobj_patternSize, &pyobj_centers, &flags, &pyobj_blobDetector) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_patternSize, patternSize, ArgInfo("patternSize", 0)) &&
        pyopencv_to(pyobj_centers, centers, ArgInfo("centers", 1)) &&
        pyopencv_to(pyobj_blobDetector, blobDetector, ArgInfo("blobDetector", 0)) )
    {
        ERRWRAP2(retval = cv::findCirclesGrid(image, patternSize, centers, flags, blobDetector));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(centers));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_patternSize = NULL;
    Size patternSize;
    PyObject* pyobj_centers = NULL;
    UMat centers;
    int flags=CALIB_CB_SYMMETRIC_GRID;
    PyObject* pyobj_blobDetector = NULL;
    Ptr<FeatureDetector> blobDetector=SimpleBlobDetector::create();
    bool retval;

    const char* keywords[] = { "image", "patternSize", "centers", "flags", "blobDetector", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OiO:findCirclesGrid", (char**)keywords, &pyobj_image, &pyobj_patternSize, &pyobj_centers, &flags, &pyobj_blobDetector) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_patternSize, patternSize, ArgInfo("patternSize", 0)) &&
        pyopencv_to(pyobj_centers, centers, ArgInfo("centers", 1)) &&
        pyopencv_to(pyobj_blobDetector, blobDetector, ArgInfo("blobDetector", 0)) )
    {
        ERRWRAP2(retval = cv::findCirclesGrid(image, patternSize, centers, flags, blobDetector));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(centers));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_findContours(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_contours = NULL;
    vector_Mat contours;
    PyObject* pyobj_hierarchy = NULL;
    Mat hierarchy;
    int mode=0;
    int method=0;
    PyObject* pyobj_offset = NULL;
    Point offset;

    const char* keywords[] = { "image", "mode", "method", "contours", "hierarchy", "offset", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii|OOO:findContours", (char**)keywords, &pyobj_image, &mode, &method, &pyobj_contours, &pyobj_hierarchy, &pyobj_offset) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 1)) &&
        pyopencv_to(pyobj_contours, contours, ArgInfo("contours", 1)) &&
        pyopencv_to(pyobj_hierarchy, hierarchy, ArgInfo("hierarchy", 1)) &&
        pyopencv_to(pyobj_offset, offset, ArgInfo("offset", 0)) )
    {
        ERRWRAP2(cv::findContours(image, contours, hierarchy, mode, method, offset));
        return Py_BuildValue("(NNN)", pyopencv_from(image), pyopencv_from(contours), pyopencv_from(hierarchy));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_contours = NULL;
    vector_Mat contours;
    PyObject* pyobj_hierarchy = NULL;
    UMat hierarchy;
    int mode=0;
    int method=0;
    PyObject* pyobj_offset = NULL;
    Point offset;

    const char* keywords[] = { "image", "mode", "method", "contours", "hierarchy", "offset", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii|OOO:findContours", (char**)keywords, &pyobj_image, &mode, &method, &pyobj_contours, &pyobj_hierarchy, &pyobj_offset) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 1)) &&
        pyopencv_to(pyobj_contours, contours, ArgInfo("contours", 1)) &&
        pyopencv_to(pyobj_hierarchy, hierarchy, ArgInfo("hierarchy", 1)) &&
        pyopencv_to(pyobj_offset, offset, ArgInfo("offset", 0)) )
    {
        ERRWRAP2(cv::findContours(image, contours, hierarchy, mode, method, offset));
        return Py_BuildValue("(NNN)", pyopencv_from(image), pyopencv_from(contours), pyopencv_from(hierarchy));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_findEssentialMat(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_points1 = NULL;
    Mat points1;
    PyObject* pyobj_points2 = NULL;
    Mat points2;
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    int method=RANSAC;
    double prob=0.999;
    double threshold=1.0;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    Mat retval;

    const char* keywords[] = { "points1", "points2", "cameraMatrix", "method", "prob", "threshold", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|iddO:findEssentialMat", (char**)keywords, &pyobj_points1, &pyobj_points2, &pyobj_cameraMatrix, &method, &prob, &threshold, &pyobj_mask) &&
        pyopencv_to(pyobj_points1, points1, ArgInfo("points1", 0)) &&
        pyopencv_to(pyobj_points2, points2, ArgInfo("points2", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 1)) )
    {
        ERRWRAP2(retval = cv::findEssentialMat(points1, points2, cameraMatrix, method, prob, threshold, mask));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(mask));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_points1 = NULL;
    UMat points1;
    PyObject* pyobj_points2 = NULL;
    UMat points2;
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    int method=RANSAC;
    double prob=0.999;
    double threshold=1.0;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    Mat retval;

    const char* keywords[] = { "points1", "points2", "cameraMatrix", "method", "prob", "threshold", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|iddO:findEssentialMat", (char**)keywords, &pyobj_points1, &pyobj_points2, &pyobj_cameraMatrix, &method, &prob, &threshold, &pyobj_mask) &&
        pyopencv_to(pyobj_points1, points1, ArgInfo("points1", 0)) &&
        pyopencv_to(pyobj_points2, points2, ArgInfo("points2", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 1)) )
    {
        ERRWRAP2(retval = cv::findEssentialMat(points1, points2, cameraMatrix, method, prob, threshold, mask));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(mask));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_points1 = NULL;
    Mat points1;
    PyObject* pyobj_points2 = NULL;
    Mat points2;
    double focal=1.0;
    PyObject* pyobj_pp = NULL;
    Point2d pp=Point2d(0, 0);
    int method=RANSAC;
    double prob=0.999;
    double threshold=1.0;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    Mat retval;

    const char* keywords[] = { "points1", "points2", "focal", "pp", "method", "prob", "threshold", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|dOiddO:findEssentialMat", (char**)keywords, &pyobj_points1, &pyobj_points2, &focal, &pyobj_pp, &method, &prob, &threshold, &pyobj_mask) &&
        pyopencv_to(pyobj_points1, points1, ArgInfo("points1", 0)) &&
        pyopencv_to(pyobj_points2, points2, ArgInfo("points2", 0)) &&
        pyopencv_to(pyobj_pp, pp, ArgInfo("pp", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 1)) )
    {
        ERRWRAP2(retval = cv::findEssentialMat(points1, points2, focal, pp, method, prob, threshold, mask));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(mask));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_points1 = NULL;
    UMat points1;
    PyObject* pyobj_points2 = NULL;
    UMat points2;
    double focal=1.0;
    PyObject* pyobj_pp = NULL;
    Point2d pp=Point2d(0, 0);
    int method=RANSAC;
    double prob=0.999;
    double threshold=1.0;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    Mat retval;

    const char* keywords[] = { "points1", "points2", "focal", "pp", "method", "prob", "threshold", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|dOiddO:findEssentialMat", (char**)keywords, &pyobj_points1, &pyobj_points2, &focal, &pyobj_pp, &method, &prob, &threshold, &pyobj_mask) &&
        pyopencv_to(pyobj_points1, points1, ArgInfo("points1", 0)) &&
        pyopencv_to(pyobj_points2, points2, ArgInfo("points2", 0)) &&
        pyopencv_to(pyobj_pp, pp, ArgInfo("pp", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 1)) )
    {
        ERRWRAP2(retval = cv::findEssentialMat(points1, points2, focal, pp, method, prob, threshold, mask));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(mask));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_findFundamentalMat(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_points1 = NULL;
    Mat points1;
    PyObject* pyobj_points2 = NULL;
    Mat points2;
    int method=FM_RANSAC;
    double param1=3.;
    double param2=0.99;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    Mat retval;

    const char* keywords[] = { "points1", "points2", "method", "param1", "param2", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|iddO:findFundamentalMat", (char**)keywords, &pyobj_points1, &pyobj_points2, &method, &param1, &param2, &pyobj_mask) &&
        pyopencv_to(pyobj_points1, points1, ArgInfo("points1", 0)) &&
        pyopencv_to(pyobj_points2, points2, ArgInfo("points2", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 1)) )
    {
        ERRWRAP2(retval = cv::findFundamentalMat(points1, points2, method, param1, param2, mask));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(mask));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_points1 = NULL;
    UMat points1;
    PyObject* pyobj_points2 = NULL;
    UMat points2;
    int method=FM_RANSAC;
    double param1=3.;
    double param2=0.99;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    Mat retval;

    const char* keywords[] = { "points1", "points2", "method", "param1", "param2", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|iddO:findFundamentalMat", (char**)keywords, &pyobj_points1, &pyobj_points2, &method, &param1, &param2, &pyobj_mask) &&
        pyopencv_to(pyobj_points1, points1, ArgInfo("points1", 0)) &&
        pyopencv_to(pyobj_points2, points2, ArgInfo("points2", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 1)) )
    {
        ERRWRAP2(retval = cv::findFundamentalMat(points1, points2, method, param1, param2, mask));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(mask));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_findHomography(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_srcPoints = NULL;
    Mat srcPoints;
    PyObject* pyobj_dstPoints = NULL;
    Mat dstPoints;
    int method=0;
    double ransacReprojThreshold=3;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    int maxIters=2000;
    double confidence=0.995;
    Mat retval;

    const char* keywords[] = { "srcPoints", "dstPoints", "method", "ransacReprojThreshold", "mask", "maxIters", "confidence", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|idOid:findHomography", (char**)keywords, &pyobj_srcPoints, &pyobj_dstPoints, &method, &ransacReprojThreshold, &pyobj_mask, &maxIters, &confidence) &&
        pyopencv_to(pyobj_srcPoints, srcPoints, ArgInfo("srcPoints", 0)) &&
        pyopencv_to(pyobj_dstPoints, dstPoints, ArgInfo("dstPoints", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 1)) )
    {
        ERRWRAP2(retval = cv::findHomography(srcPoints, dstPoints, method, ransacReprojThreshold, mask, maxIters, confidence));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(mask));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_srcPoints = NULL;
    UMat srcPoints;
    PyObject* pyobj_dstPoints = NULL;
    UMat dstPoints;
    int method=0;
    double ransacReprojThreshold=3;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    int maxIters=2000;
    double confidence=0.995;
    Mat retval;

    const char* keywords[] = { "srcPoints", "dstPoints", "method", "ransacReprojThreshold", "mask", "maxIters", "confidence", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|idOid:findHomography", (char**)keywords, &pyobj_srcPoints, &pyobj_dstPoints, &method, &ransacReprojThreshold, &pyobj_mask, &maxIters, &confidence) &&
        pyopencv_to(pyobj_srcPoints, srcPoints, ArgInfo("srcPoints", 0)) &&
        pyopencv_to(pyobj_dstPoints, dstPoints, ArgInfo("dstPoints", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 1)) )
    {
        ERRWRAP2(retval = cv::findHomography(srcPoints, dstPoints, method, ransacReprojThreshold, mask, maxIters, confidence));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(mask));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_findNonZero(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_idx = NULL;
    Mat idx;

    const char* keywords[] = { "src", "idx", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:findNonZero", (char**)keywords, &pyobj_src, &pyobj_idx) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_idx, idx, ArgInfo("idx", 1)) )
    {
        ERRWRAP2(cv::findNonZero(src, idx));
        return pyopencv_from(idx);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_idx = NULL;
    UMat idx;

    const char* keywords[] = { "src", "idx", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:findNonZero", (char**)keywords, &pyobj_src, &pyobj_idx) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_idx, idx, ArgInfo("idx", 1)) )
    {
        ERRWRAP2(cv::findNonZero(src, idx));
        return pyopencv_from(idx);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_findTransformECC(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_templateImage = NULL;
    Mat templateImage;
    PyObject* pyobj_inputImage = NULL;
    Mat inputImage;
    PyObject* pyobj_warpMatrix = NULL;
    Mat warpMatrix;
    int motionType=MOTION_AFFINE;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 50, 0.001);
    PyObject* pyobj_inputMask = NULL;
    Mat inputMask;
    double retval;

    const char* keywords[] = { "templateImage", "inputImage", "warpMatrix", "motionType", "criteria", "inputMask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|iOO:findTransformECC", (char**)keywords, &pyobj_templateImage, &pyobj_inputImage, &pyobj_warpMatrix, &motionType, &pyobj_criteria, &pyobj_inputMask) &&
        pyopencv_to(pyobj_templateImage, templateImage, ArgInfo("templateImage", 0)) &&
        pyopencv_to(pyobj_inputImage, inputImage, ArgInfo("inputImage", 0)) &&
        pyopencv_to(pyobj_warpMatrix, warpMatrix, ArgInfo("warpMatrix", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) &&
        pyopencv_to(pyobj_inputMask, inputMask, ArgInfo("inputMask", 0)) )
    {
        ERRWRAP2(retval = cv::findTransformECC(templateImage, inputImage, warpMatrix, motionType, criteria, inputMask));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(warpMatrix));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_templateImage = NULL;
    UMat templateImage;
    PyObject* pyobj_inputImage = NULL;
    UMat inputImage;
    PyObject* pyobj_warpMatrix = NULL;
    UMat warpMatrix;
    int motionType=MOTION_AFFINE;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 50, 0.001);
    PyObject* pyobj_inputMask = NULL;
    UMat inputMask;
    double retval;

    const char* keywords[] = { "templateImage", "inputImage", "warpMatrix", "motionType", "criteria", "inputMask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|iOO:findTransformECC", (char**)keywords, &pyobj_templateImage, &pyobj_inputImage, &pyobj_warpMatrix, &motionType, &pyobj_criteria, &pyobj_inputMask) &&
        pyopencv_to(pyobj_templateImage, templateImage, ArgInfo("templateImage", 0)) &&
        pyopencv_to(pyobj_inputImage, inputImage, ArgInfo("inputImage", 0)) &&
        pyopencv_to(pyobj_warpMatrix, warpMatrix, ArgInfo("warpMatrix", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) &&
        pyopencv_to(pyobj_inputMask, inputMask, ArgInfo("inputMask", 0)) )
    {
        ERRWRAP2(retval = cv::findTransformECC(templateImage, inputImage, warpMatrix, motionType, criteria, inputMask));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(warpMatrix));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_fitEllipse(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_points = NULL;
    Mat points;
    RotatedRect retval;

    const char* keywords[] = { "points", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:fitEllipse", (char**)keywords, &pyobj_points) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 0)) )
    {
        ERRWRAP2(retval = cv::fitEllipse(points));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_points = NULL;
    UMat points;
    RotatedRect retval;

    const char* keywords[] = { "points", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:fitEllipse", (char**)keywords, &pyobj_points) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 0)) )
    {
        ERRWRAP2(retval = cv::fitEllipse(points));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_fitLine(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_points = NULL;
    Mat points;
    PyObject* pyobj_line = NULL;
    Mat line;
    int distType=0;
    double param=0;
    double reps=0;
    double aeps=0;

    const char* keywords[] = { "points", "distType", "param", "reps", "aeps", "line", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oiddd|O:fitLine", (char**)keywords, &pyobj_points, &distType, &param, &reps, &aeps, &pyobj_line) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 0)) &&
        pyopencv_to(pyobj_line, line, ArgInfo("line", 1)) )
    {
        ERRWRAP2(cv::fitLine(points, line, distType, param, reps, aeps));
        return pyopencv_from(line);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_points = NULL;
    UMat points;
    PyObject* pyobj_line = NULL;
    UMat line;
    int distType=0;
    double param=0;
    double reps=0;
    double aeps=0;

    const char* keywords[] = { "points", "distType", "param", "reps", "aeps", "line", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oiddd|O:fitLine", (char**)keywords, &pyobj_points, &distType, &param, &reps, &aeps, &pyobj_line) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 0)) &&
        pyopencv_to(pyobj_line, line, ArgInfo("line", 1)) )
    {
        ERRWRAP2(cv::fitLine(points, line, distType, param, reps, aeps));
        return pyopencv_from(line);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_flip(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int flipCode=0;

    const char* keywords[] = { "src", "flipCode", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|O:flip", (char**)keywords, &pyobj_src, &flipCode, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::flip(src, dst, flipCode));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int flipCode=0;

    const char* keywords[] = { "src", "flipCode", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|O:flip", (char**)keywords, &pyobj_src, &flipCode, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::flip(src, dst, flipCode));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_floodFill(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    PyObject* pyobj_seedPoint = NULL;
    Point seedPoint;
    PyObject* pyobj_newVal = NULL;
    Scalar newVal;
    Rect rect;
    PyObject* pyobj_loDiff = NULL;
    Scalar loDiff;
    PyObject* pyobj_upDiff = NULL;
    Scalar upDiff;
    int flags=4;
    int retval;

    const char* keywords[] = { "image", "mask", "seedPoint", "newVal", "loDiff", "upDiff", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OOi:floodFill", (char**)keywords, &pyobj_image, &pyobj_mask, &pyobj_seedPoint, &pyobj_newVal, &pyobj_loDiff, &pyobj_upDiff, &flags) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 1)) &&
        pyopencv_to(pyobj_seedPoint, seedPoint, ArgInfo("seedPoint", 0)) &&
        pyopencv_to(pyobj_newVal, newVal, ArgInfo("newVal", 0)) &&
        pyopencv_to(pyobj_loDiff, loDiff, ArgInfo("loDiff", 0)) &&
        pyopencv_to(pyobj_upDiff, upDiff, ArgInfo("upDiff", 0)) )
    {
        ERRWRAP2(retval = cv::floodFill(image, mask, seedPoint, newVal, &rect, loDiff, upDiff, flags));
        return Py_BuildValue("(NNNN)", pyopencv_from(retval), pyopencv_from(image), pyopencv_from(mask), pyopencv_from(rect));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    PyObject* pyobj_seedPoint = NULL;
    Point seedPoint;
    PyObject* pyobj_newVal = NULL;
    Scalar newVal;
    Rect rect;
    PyObject* pyobj_loDiff = NULL;
    Scalar loDiff;
    PyObject* pyobj_upDiff = NULL;
    Scalar upDiff;
    int flags=4;
    int retval;

    const char* keywords[] = { "image", "mask", "seedPoint", "newVal", "loDiff", "upDiff", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OOi:floodFill", (char**)keywords, &pyobj_image, &pyobj_mask, &pyobj_seedPoint, &pyobj_newVal, &pyobj_loDiff, &pyobj_upDiff, &flags) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 1)) &&
        pyopencv_to(pyobj_seedPoint, seedPoint, ArgInfo("seedPoint", 0)) &&
        pyopencv_to(pyobj_newVal, newVal, ArgInfo("newVal", 0)) &&
        pyopencv_to(pyobj_loDiff, loDiff, ArgInfo("loDiff", 0)) &&
        pyopencv_to(pyobj_upDiff, upDiff, ArgInfo("upDiff", 0)) )
    {
        ERRWRAP2(retval = cv::floodFill(image, mask, seedPoint, newVal, &rect, loDiff, upDiff, flags));
        return Py_BuildValue("(NNNN)", pyopencv_from(retval), pyopencv_from(image), pyopencv_from(mask), pyopencv_from(rect));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_gemm(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    PyObject* pyobj_src2 = NULL;
    Mat src2;
    double alpha=0;
    PyObject* pyobj_src3 = NULL;
    Mat src3;
    double beta=0;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int flags=0;

    const char* keywords[] = { "src1", "src2", "alpha", "src3", "beta", "dst", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOdOd|Oi:gemm", (char**)keywords, &pyobj_src1, &pyobj_src2, &alpha, &pyobj_src3, &beta, &pyobj_dst, &flags) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_src3, src3, ArgInfo("src3", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::gemm(src1, src2, alpha, src3, beta, dst, flags));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src1 = NULL;
    UMat src1;
    PyObject* pyobj_src2 = NULL;
    UMat src2;
    double alpha=0;
    PyObject* pyobj_src3 = NULL;
    UMat src3;
    double beta=0;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int flags=0;

    const char* keywords[] = { "src1", "src2", "alpha", "src3", "beta", "dst", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOdOd|Oi:gemm", (char**)keywords, &pyobj_src1, &pyobj_src2, &alpha, &pyobj_src3, &beta, &pyobj_dst, &flags) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_src3, src3, ArgInfo("src3", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::gemm(src1, src2, alpha, src3, beta, dst, flags));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_getAffineTransform(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    Mat retval;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:getAffineTransform", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 0)) )
    {
        ERRWRAP2(retval = cv::getAffineTransform(src, dst));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    Mat retval;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:getAffineTransform", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 0)) )
    {
        ERRWRAP2(retval = cv::getAffineTransform(src, dst));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_getBuildInformation(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    String retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::getBuildInformation());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_getCPUTickCount(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int64 retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::getCPUTickCount());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_getDefaultNewCameraMatrix(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_imgsize = NULL;
    Size imgsize;
    bool centerPrincipalPoint=false;
    Mat retval;

    const char* keywords[] = { "cameraMatrix", "imgsize", "centerPrincipalPoint", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Ob:getDefaultNewCameraMatrix", (char**)keywords, &pyobj_cameraMatrix, &pyobj_imgsize, &centerPrincipalPoint) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_imgsize, imgsize, ArgInfo("imgsize", 0)) )
    {
        ERRWRAP2(retval = cv::getDefaultNewCameraMatrix(cameraMatrix, imgsize, centerPrincipalPoint));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    PyObject* pyobj_imgsize = NULL;
    Size imgsize;
    bool centerPrincipalPoint=false;
    Mat retval;

    const char* keywords[] = { "cameraMatrix", "imgsize", "centerPrincipalPoint", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Ob:getDefaultNewCameraMatrix", (char**)keywords, &pyobj_cameraMatrix, &pyobj_imgsize, &centerPrincipalPoint) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_imgsize, imgsize, ArgInfo("imgsize", 0)) )
    {
        ERRWRAP2(retval = cv::getDefaultNewCameraMatrix(cameraMatrix, imgsize, centerPrincipalPoint));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_getDerivKernels(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_kx = NULL;
    Mat kx;
    PyObject* pyobj_ky = NULL;
    Mat ky;
    int dx=0;
    int dy=0;
    int ksize=0;
    bool normalize=false;
    int ktype=CV_32F;

    const char* keywords[] = { "dx", "dy", "ksize", "kx", "ky", "normalize", "ktype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "iii|OObi:getDerivKernels", (char**)keywords, &dx, &dy, &ksize, &pyobj_kx, &pyobj_ky, &normalize, &ktype) &&
        pyopencv_to(pyobj_kx, kx, ArgInfo("kx", 1)) &&
        pyopencv_to(pyobj_ky, ky, ArgInfo("ky", 1)) )
    {
        ERRWRAP2(cv::getDerivKernels(kx, ky, dx, dy, ksize, normalize, ktype));
        return Py_BuildValue("(NN)", pyopencv_from(kx), pyopencv_from(ky));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_kx = NULL;
    UMat kx;
    PyObject* pyobj_ky = NULL;
    UMat ky;
    int dx=0;
    int dy=0;
    int ksize=0;
    bool normalize=false;
    int ktype=CV_32F;

    const char* keywords[] = { "dx", "dy", "ksize", "kx", "ky", "normalize", "ktype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "iii|OObi:getDerivKernels", (char**)keywords, &dx, &dy, &ksize, &pyobj_kx, &pyobj_ky, &normalize, &ktype) &&
        pyopencv_to(pyobj_kx, kx, ArgInfo("kx", 1)) &&
        pyopencv_to(pyobj_ky, ky, ArgInfo("ky", 1)) )
    {
        ERRWRAP2(cv::getDerivKernels(kx, ky, dx, dy, ksize, normalize, ktype));
        return Py_BuildValue("(NN)", pyopencv_from(kx), pyopencv_from(ky));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_getGaborKernel(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_ksize = NULL;
    Size ksize;
    double sigma=0;
    double theta=0;
    double lambd=0;
    double gamma=0;
    double psi=CV_PI*0.5;
    int ktype=CV_64F;
    Mat retval;

    const char* keywords[] = { "ksize", "sigma", "theta", "lambd", "gamma", "psi", "ktype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odddd|di:getGaborKernel", (char**)keywords, &pyobj_ksize, &sigma, &theta, &lambd, &gamma, &psi, &ktype) &&
        pyopencv_to(pyobj_ksize, ksize, ArgInfo("ksize", 0)) )
    {
        ERRWRAP2(retval = cv::getGaborKernel(ksize, sigma, theta, lambd, gamma, psi, ktype));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_getGaussianKernel(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int ksize=0;
    double sigma=0;
    int ktype=CV_64F;
    Mat retval;

    const char* keywords[] = { "ksize", "sigma", "ktype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "id|i:getGaussianKernel", (char**)keywords, &ksize, &sigma, &ktype) )
    {
        ERRWRAP2(retval = cv::getGaussianKernel(ksize, sigma, ktype));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_getNumThreads(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::getNumThreads());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_getNumberOfCPUs(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::getNumberOfCPUs());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_getOptimalDFTSize(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int vecsize=0;
    int retval;

    const char* keywords[] = { "vecsize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:getOptimalDFTSize", (char**)keywords, &vecsize) )
    {
        ERRWRAP2(retval = cv::getOptimalDFTSize(vecsize));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_getOptimalNewCameraMatrix(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    Mat distCoeffs;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    double alpha=0;
    PyObject* pyobj_newImgSize = NULL;
    Size newImgSize;
    Rect validPixROI;
    bool centerPrincipalPoint=false;
    Mat retval;

    const char* keywords[] = { "cameraMatrix", "distCoeffs", "imageSize", "alpha", "newImgSize", "centerPrincipalPoint", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOd|Ob:getOptimalNewCameraMatrix", (char**)keywords, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_imageSize, &alpha, &pyobj_newImgSize, &centerPrincipalPoint) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_newImgSize, newImgSize, ArgInfo("newImgSize", 0)) )
    {
        ERRWRAP2(retval = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, alpha, newImgSize, &validPixROI, centerPrincipalPoint));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(validPixROI));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    UMat distCoeffs;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    double alpha=0;
    PyObject* pyobj_newImgSize = NULL;
    Size newImgSize;
    Rect validPixROI;
    bool centerPrincipalPoint=false;
    Mat retval;

    const char* keywords[] = { "cameraMatrix", "distCoeffs", "imageSize", "alpha", "newImgSize", "centerPrincipalPoint", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOd|Ob:getOptimalNewCameraMatrix", (char**)keywords, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_imageSize, &alpha, &pyobj_newImgSize, &centerPrincipalPoint) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_newImgSize, newImgSize, ArgInfo("newImgSize", 0)) )
    {
        ERRWRAP2(retval = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, alpha, newImgSize, &validPixROI, centerPrincipalPoint));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(validPixROI));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_getPerspectiveTransform(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    Mat retval;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:getPerspectiveTransform", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 0)) )
    {
        ERRWRAP2(retval = cv::getPerspectiveTransform(src, dst));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    Mat retval;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:getPerspectiveTransform", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 0)) )
    {
        ERRWRAP2(retval = cv::getPerspectiveTransform(src, dst));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_getRectSubPix(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_patchSize = NULL;
    Size patchSize;
    PyObject* pyobj_center = NULL;
    Point2f center;
    PyObject* pyobj_patch = NULL;
    Mat patch;
    int patchType=-1;

    const char* keywords[] = { "image", "patchSize", "center", "patch", "patchType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|Oi:getRectSubPix", (char**)keywords, &pyobj_image, &pyobj_patchSize, &pyobj_center, &pyobj_patch, &patchType) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_patchSize, patchSize, ArgInfo("patchSize", 0)) &&
        pyopencv_to(pyobj_center, center, ArgInfo("center", 0)) &&
        pyopencv_to(pyobj_patch, patch, ArgInfo("patch", 1)) )
    {
        ERRWRAP2(cv::getRectSubPix(image, patchSize, center, patch, patchType));
        return pyopencv_from(patch);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_patchSize = NULL;
    Size patchSize;
    PyObject* pyobj_center = NULL;
    Point2f center;
    PyObject* pyobj_patch = NULL;
    UMat patch;
    int patchType=-1;

    const char* keywords[] = { "image", "patchSize", "center", "patch", "patchType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|Oi:getRectSubPix", (char**)keywords, &pyobj_image, &pyobj_patchSize, &pyobj_center, &pyobj_patch, &patchType) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_patchSize, patchSize, ArgInfo("patchSize", 0)) &&
        pyopencv_to(pyobj_center, center, ArgInfo("center", 0)) &&
        pyopencv_to(pyobj_patch, patch, ArgInfo("patch", 1)) )
    {
        ERRWRAP2(cv::getRectSubPix(image, patchSize, center, patch, patchType));
        return pyopencv_from(patch);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_getRotationMatrix2D(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_center = NULL;
    Point2f center;
    double angle=0;
    double scale=0;
    Mat retval;

    const char* keywords[] = { "center", "angle", "scale", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odd:getRotationMatrix2D", (char**)keywords, &pyobj_center, &angle, &scale) &&
        pyopencv_to(pyobj_center, center, ArgInfo("center", 0)) )
    {
        ERRWRAP2(retval = cv::getRotationMatrix2D(center, angle, scale));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_getStructuringElement(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int shape=0;
    PyObject* pyobj_ksize = NULL;
    Size ksize;
    PyObject* pyobj_anchor = NULL;
    Point anchor=Point(-1,-1);
    Mat retval;

    const char* keywords[] = { "shape", "ksize", "anchor", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "iO|O:getStructuringElement", (char**)keywords, &shape, &pyobj_ksize, &pyobj_anchor) &&
        pyopencv_to(pyobj_ksize, ksize, ArgInfo("ksize", 0)) &&
        pyopencv_to(pyobj_anchor, anchor, ArgInfo("anchor", 0)) )
    {
        ERRWRAP2(retval = cv::getStructuringElement(shape, ksize, anchor));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_getTextSize(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_text = NULL;
    String text;
    int fontFace=0;
    double fontScale=0;
    int thickness=0;
    int baseLine;
    Size retval;

    const char* keywords[] = { "text", "fontFace", "fontScale", "thickness", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oidi:getTextSize", (char**)keywords, &pyobj_text, &fontFace, &fontScale, &thickness) &&
        pyopencv_to(pyobj_text, text, ArgInfo("text", 0)) )
    {
        ERRWRAP2(retval = cv::getTextSize(text, fontFace, fontScale, thickness, &baseLine));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(baseLine));
    }

    return NULL;
}

static PyObject* pyopencv_cv_getThreadNum(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::getThreadNum());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_getTickCount(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int64 retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::getTickCount());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_getTickFrequency(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::getTickFrequency());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_getTrackbarPos(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_trackbarname = NULL;
    String trackbarname;
    PyObject* pyobj_winname = NULL;
    String winname;
    int retval;

    const char* keywords[] = { "trackbarname", "winname", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:getTrackbarPos", (char**)keywords, &pyobj_trackbarname, &pyobj_winname) &&
        pyopencv_to(pyobj_trackbarname, trackbarname, ArgInfo("trackbarname", 0)) &&
        pyopencv_to(pyobj_winname, winname, ArgInfo("winname", 0)) )
    {
        ERRWRAP2(retval = cv::getTrackbarPos(trackbarname, winname));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_getValidDisparityROI(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_roi1 = NULL;
    Rect roi1;
    PyObject* pyobj_roi2 = NULL;
    Rect roi2;
    int minDisparity=0;
    int numberOfDisparities=0;
    int SADWindowSize=0;
    Rect retval;

    const char* keywords[] = { "roi1", "roi2", "minDisparity", "numberOfDisparities", "SADWindowSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOiii:getValidDisparityROI", (char**)keywords, &pyobj_roi1, &pyobj_roi2, &minDisparity, &numberOfDisparities, &SADWindowSize) &&
        pyopencv_to(pyobj_roi1, roi1, ArgInfo("roi1", 0)) &&
        pyopencv_to(pyobj_roi2, roi2, ArgInfo("roi2", 0)) )
    {
        ERRWRAP2(retval = cv::getValidDisparityROI(roi1, roi2, minDisparity, numberOfDisparities, SADWindowSize));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_getWindowProperty(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_winname = NULL;
    String winname;
    int prop_id=0;
    double retval;

    const char* keywords[] = { "winname", "prop_id", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi:getWindowProperty", (char**)keywords, &pyobj_winname, &prop_id) &&
        pyopencv_to(pyobj_winname, winname, ArgInfo("winname", 0)) )
    {
        ERRWRAP2(retval = cv::getWindowProperty(winname, prop_id));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_goodFeaturesToTrack(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_corners = NULL;
    Mat corners;
    int maxCorners=0;
    double qualityLevel=0;
    double minDistance=0;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    int blockSize=3;
    bool useHarrisDetector=false;
    double k=0.04;

    const char* keywords[] = { "image", "maxCorners", "qualityLevel", "minDistance", "corners", "mask", "blockSize", "useHarrisDetector", "k", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oidd|OOibd:goodFeaturesToTrack", (char**)keywords, &pyobj_image, &maxCorners, &qualityLevel, &minDistance, &pyobj_corners, &pyobj_mask, &blockSize, &useHarrisDetector, &k) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_corners, corners, ArgInfo("corners", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::goodFeaturesToTrack(image, corners, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k));
        return pyopencv_from(corners);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_corners = NULL;
    UMat corners;
    int maxCorners=0;
    double qualityLevel=0;
    double minDistance=0;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    int blockSize=3;
    bool useHarrisDetector=false;
    double k=0.04;

    const char* keywords[] = { "image", "maxCorners", "qualityLevel", "minDistance", "corners", "mask", "blockSize", "useHarrisDetector", "k", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oidd|OOibd:goodFeaturesToTrack", (char**)keywords, &pyobj_image, &maxCorners, &qualityLevel, &minDistance, &pyobj_corners, &pyobj_mask, &blockSize, &useHarrisDetector, &k) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_corners, corners, ArgInfo("corners", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::goodFeaturesToTrack(image, corners, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k));
        return pyopencv_from(corners);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_grabCut(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    PyObject* pyobj_rect = NULL;
    Rect rect;
    PyObject* pyobj_bgdModel = NULL;
    Mat bgdModel;
    PyObject* pyobj_fgdModel = NULL;
    Mat fgdModel;
    int iterCount=0;
    int mode=GC_EVAL;

    const char* keywords[] = { "img", "mask", "rect", "bgdModel", "fgdModel", "iterCount", "mode", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOi|i:grabCut", (char**)keywords, &pyobj_img, &pyobj_mask, &pyobj_rect, &pyobj_bgdModel, &pyobj_fgdModel, &iterCount, &mode) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 1)) &&
        pyopencv_to(pyobj_rect, rect, ArgInfo("rect", 0)) &&
        pyopencv_to(pyobj_bgdModel, bgdModel, ArgInfo("bgdModel", 1)) &&
        pyopencv_to(pyobj_fgdModel, fgdModel, ArgInfo("fgdModel", 1)) )
    {
        ERRWRAP2(cv::grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode));
        return Py_BuildValue("(NNN)", pyopencv_from(mask), pyopencv_from(bgdModel), pyopencv_from(fgdModel));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_img = NULL;
    UMat img;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    PyObject* pyobj_rect = NULL;
    Rect rect;
    PyObject* pyobj_bgdModel = NULL;
    UMat bgdModel;
    PyObject* pyobj_fgdModel = NULL;
    UMat fgdModel;
    int iterCount=0;
    int mode=GC_EVAL;

    const char* keywords[] = { "img", "mask", "rect", "bgdModel", "fgdModel", "iterCount", "mode", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOi|i:grabCut", (char**)keywords, &pyobj_img, &pyobj_mask, &pyobj_rect, &pyobj_bgdModel, &pyobj_fgdModel, &iterCount, &mode) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 1)) &&
        pyopencv_to(pyobj_rect, rect, ArgInfo("rect", 0)) &&
        pyopencv_to(pyobj_bgdModel, bgdModel, ArgInfo("bgdModel", 1)) &&
        pyopencv_to(pyobj_fgdModel, fgdModel, ArgInfo("fgdModel", 1)) )
    {
        ERRWRAP2(cv::grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode));
        return Py_BuildValue("(NNN)", pyopencv_from(mask), pyopencv_from(bgdModel), pyopencv_from(fgdModel));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_groupRectangles(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_rectList = NULL;
    vector_Rect rectList;
    vector_int weights;
    int groupThreshold=0;
    double eps=0.2;

    const char* keywords[] = { "rectList", "groupThreshold", "eps", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|d:groupRectangles", (char**)keywords, &pyobj_rectList, &groupThreshold, &eps) &&
        pyopencv_to(pyobj_rectList, rectList, ArgInfo("rectList", 1)) )
    {
        ERRWRAP2(cv::groupRectangles(rectList, weights, groupThreshold, eps));
        return Py_BuildValue("(NN)", pyopencv_from(rectList), pyopencv_from(weights));
    }

    return NULL;
}

static PyObject* pyopencv_cv_haveOpenVX(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::haveOpenVX());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_hconcat(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:hconcat", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::hconcat(src, dst));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:hconcat", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::hconcat(src, dst));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_idct(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int flags=0;

    const char* keywords[] = { "src", "dst", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oi:idct", (char**)keywords, &pyobj_src, &pyobj_dst, &flags) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::idct(src, dst, flags));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int flags=0;

    const char* keywords[] = { "src", "dst", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oi:idct", (char**)keywords, &pyobj_src, &pyobj_dst, &flags) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::idct(src, dst, flags));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_idft(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int flags=0;
    int nonzeroRows=0;

    const char* keywords[] = { "src", "dst", "flags", "nonzeroRows", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oii:idft", (char**)keywords, &pyobj_src, &pyobj_dst, &flags, &nonzeroRows) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::idft(src, dst, flags, nonzeroRows));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int flags=0;
    int nonzeroRows=0;

    const char* keywords[] = { "src", "dst", "flags", "nonzeroRows", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oii:idft", (char**)keywords, &pyobj_src, &pyobj_dst, &flags, &nonzeroRows) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::idft(src, dst, flags, nonzeroRows));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_illuminationChange(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    float alpha=0.2f;
    float beta=0.4f;

    const char* keywords[] = { "src", "mask", "dst", "alpha", "beta", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Off:illuminationChange", (char**)keywords, &pyobj_src, &pyobj_mask, &pyobj_dst, &alpha, &beta) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::illuminationChange(src, mask, dst, alpha, beta));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    float alpha=0.2f;
    float beta=0.4f;

    const char* keywords[] = { "src", "mask", "dst", "alpha", "beta", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Off:illuminationChange", (char**)keywords, &pyobj_src, &pyobj_mask, &pyobj_dst, &alpha, &beta) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::illuminationChange(src, mask, dst, alpha, beta));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_imdecode(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_buf = NULL;
    Mat buf;
    int flags=0;
    Mat retval;

    const char* keywords[] = { "buf", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi:imdecode", (char**)keywords, &pyobj_buf, &flags) &&
        pyopencv_to(pyobj_buf, buf, ArgInfo("buf", 0)) )
    {
        ERRWRAP2(retval = cv::imdecode(buf, flags));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_buf = NULL;
    UMat buf;
    int flags=0;
    Mat retval;

    const char* keywords[] = { "buf", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi:imdecode", (char**)keywords, &pyobj_buf, &flags) &&
        pyopencv_to(pyobj_buf, buf, ArgInfo("buf", 0)) )
    {
        ERRWRAP2(retval = cv::imdecode(buf, flags));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_imencode(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_ext = NULL;
    String ext;
    PyObject* pyobj_img = NULL;
    Mat img;
    vector_uchar buf;
    PyObject* pyobj_params = NULL;
    vector_int params=std::vector<int>();
    bool retval;

    const char* keywords[] = { "ext", "img", "params", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:imencode", (char**)keywords, &pyobj_ext, &pyobj_img, &pyobj_params) &&
        pyopencv_to(pyobj_ext, ext, ArgInfo("ext", 0)) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_params, params, ArgInfo("params", 0)) )
    {
        ERRWRAP2(retval = cv::imencode(ext, img, buf, params));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(buf));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_ext = NULL;
    String ext;
    PyObject* pyobj_img = NULL;
    UMat img;
    vector_uchar buf;
    PyObject* pyobj_params = NULL;
    vector_int params=std::vector<int>();
    bool retval;

    const char* keywords[] = { "ext", "img", "params", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:imencode", (char**)keywords, &pyobj_ext, &pyobj_img, &pyobj_params) &&
        pyopencv_to(pyobj_ext, ext, ArgInfo("ext", 0)) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_params, params, ArgInfo("params", 0)) )
    {
        ERRWRAP2(retval = cv::imencode(ext, img, buf, params));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(buf));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_imread(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_filename = NULL;
    String filename;
    int flags=IMREAD_COLOR;
    Mat retval;

    const char* keywords[] = { "filename", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|i:imread", (char**)keywords, &pyobj_filename, &flags) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) )
    {
        ERRWRAP2(retval = cv::imread(filename, flags));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_imreadmulti(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_filename = NULL;
    String filename;
    PyObject* pyobj_mats = NULL;
    vector_Mat mats;
    int flags=IMREAD_ANYCOLOR;
    bool retval;

    const char* keywords[] = { "filename", "mats", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|i:imreadmulti", (char**)keywords, &pyobj_filename, &pyobj_mats, &flags) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) &&
        pyopencv_to(pyobj_mats, mats, ArgInfo("mats", 0)) )
    {
        ERRWRAP2(retval = cv::imreadmulti(filename, mats, flags));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_filename = NULL;
    String filename;
    PyObject* pyobj_mats = NULL;
    vector_Mat mats;
    int flags=IMREAD_ANYCOLOR;
    bool retval;

    const char* keywords[] = { "filename", "mats", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|i:imreadmulti", (char**)keywords, &pyobj_filename, &pyobj_mats, &flags) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) &&
        pyopencv_to(pyobj_mats, mats, ArgInfo("mats", 0)) )
    {
        ERRWRAP2(retval = cv::imreadmulti(filename, mats, flags));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_imshow(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_winname = NULL;
    String winname;
    PyObject* pyobj_mat = NULL;
    Mat mat;

    const char* keywords[] = { "winname", "mat", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:imshow", (char**)keywords, &pyobj_winname, &pyobj_mat) &&
        pyopencv_to(pyobj_winname, winname, ArgInfo("winname", 0)) &&
        pyopencv_to(pyobj_mat, mat, ArgInfo("mat", 0)) )
    {
        ERRWRAP2(cv::imshow(winname, mat));
        Py_RETURN_NONE;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_winname = NULL;
    String winname;
    PyObject* pyobj_mat = NULL;
    UMat mat;

    const char* keywords[] = { "winname", "mat", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:imshow", (char**)keywords, &pyobj_winname, &pyobj_mat) &&
        pyopencv_to(pyobj_winname, winname, ArgInfo("winname", 0)) &&
        pyopencv_to(pyobj_mat, mat, ArgInfo("mat", 0)) )
    {
        ERRWRAP2(cv::imshow(winname, mat));
        Py_RETURN_NONE;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_imwrite(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_filename = NULL;
    String filename;
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_params = NULL;
    vector_int params=std::vector<int>();
    bool retval;

    const char* keywords[] = { "filename", "img", "params", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:imwrite", (char**)keywords, &pyobj_filename, &pyobj_img, &pyobj_params) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_params, params, ArgInfo("params", 0)) )
    {
        ERRWRAP2(retval = cv::imwrite(filename, img, params));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_filename = NULL;
    String filename;
    PyObject* pyobj_img = NULL;
    UMat img;
    PyObject* pyobj_params = NULL;
    vector_int params=std::vector<int>();
    bool retval;

    const char* keywords[] = { "filename", "img", "params", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:imwrite", (char**)keywords, &pyobj_filename, &pyobj_img, &pyobj_params) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_params, params, ArgInfo("params", 0)) )
    {
        ERRWRAP2(retval = cv::imwrite(filename, img, params));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_inRange(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_lowerb = NULL;
    Mat lowerb;
    PyObject* pyobj_upperb = NULL;
    Mat upperb;
    PyObject* pyobj_dst = NULL;
    Mat dst;

    const char* keywords[] = { "src", "lowerb", "upperb", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|O:inRange", (char**)keywords, &pyobj_src, &pyobj_lowerb, &pyobj_upperb, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_lowerb, lowerb, ArgInfo("lowerb", 0)) &&
        pyopencv_to(pyobj_upperb, upperb, ArgInfo("upperb", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::inRange(src, lowerb, upperb, dst));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_lowerb = NULL;
    UMat lowerb;
    PyObject* pyobj_upperb = NULL;
    UMat upperb;
    PyObject* pyobj_dst = NULL;
    UMat dst;

    const char* keywords[] = { "src", "lowerb", "upperb", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|O:inRange", (char**)keywords, &pyobj_src, &pyobj_lowerb, &pyobj_upperb, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_lowerb, lowerb, ArgInfo("lowerb", 0)) &&
        pyopencv_to(pyobj_upperb, upperb, ArgInfo("upperb", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::inRange(src, lowerb, upperb, dst));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_initCameraMatrix2D(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_objectPoints = NULL;
    vector_Mat objectPoints;
    PyObject* pyobj_imagePoints = NULL;
    vector_Mat imagePoints;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    double aspectRatio=1.0;
    Mat retval;

    const char* keywords[] = { "objectPoints", "imagePoints", "imageSize", "aspectRatio", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|d:initCameraMatrix2D", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints, &pyobj_imageSize, &aspectRatio) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints, imagePoints, ArgInfo("imagePoints", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) )
    {
        ERRWRAP2(retval = cv::initCameraMatrix2D(objectPoints, imagePoints, imageSize, aspectRatio));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_objectPoints = NULL;
    vector_Mat objectPoints;
    PyObject* pyobj_imagePoints = NULL;
    vector_Mat imagePoints;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    double aspectRatio=1.0;
    Mat retval;

    const char* keywords[] = { "objectPoints", "imagePoints", "imageSize", "aspectRatio", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|d:initCameraMatrix2D", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints, &pyobj_imageSize, &aspectRatio) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints, imagePoints, ArgInfo("imagePoints", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) )
    {
        ERRWRAP2(retval = cv::initCameraMatrix2D(objectPoints, imagePoints, imageSize, aspectRatio));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_initUndistortRectifyMap(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    Mat distCoeffs;
    PyObject* pyobj_R = NULL;
    Mat R;
    PyObject* pyobj_newCameraMatrix = NULL;
    Mat newCameraMatrix;
    PyObject* pyobj_size = NULL;
    Size size;
    int m1type=0;
    PyObject* pyobj_map1 = NULL;
    Mat map1;
    PyObject* pyobj_map2 = NULL;
    Mat map2;

    const char* keywords[] = { "cameraMatrix", "distCoeffs", "R", "newCameraMatrix", "size", "m1type", "map1", "map2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOi|OO:initUndistortRectifyMap", (char**)keywords, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_R, &pyobj_newCameraMatrix, &pyobj_size, &m1type, &pyobj_map1, &pyobj_map2) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 0)) &&
        pyopencv_to(pyobj_newCameraMatrix, newCameraMatrix, ArgInfo("newCameraMatrix", 0)) &&
        pyopencv_to(pyobj_size, size, ArgInfo("size", 0)) &&
        pyopencv_to(pyobj_map1, map1, ArgInfo("map1", 1)) &&
        pyopencv_to(pyobj_map2, map2, ArgInfo("map2", 1)) )
    {
        ERRWRAP2(cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type, map1, map2));
        return Py_BuildValue("(NN)", pyopencv_from(map1), pyopencv_from(map2));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    UMat distCoeffs;
    PyObject* pyobj_R = NULL;
    UMat R;
    PyObject* pyobj_newCameraMatrix = NULL;
    UMat newCameraMatrix;
    PyObject* pyobj_size = NULL;
    Size size;
    int m1type=0;
    PyObject* pyobj_map1 = NULL;
    UMat map1;
    PyObject* pyobj_map2 = NULL;
    UMat map2;

    const char* keywords[] = { "cameraMatrix", "distCoeffs", "R", "newCameraMatrix", "size", "m1type", "map1", "map2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOi|OO:initUndistortRectifyMap", (char**)keywords, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_R, &pyobj_newCameraMatrix, &pyobj_size, &m1type, &pyobj_map1, &pyobj_map2) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 0)) &&
        pyopencv_to(pyobj_newCameraMatrix, newCameraMatrix, ArgInfo("newCameraMatrix", 0)) &&
        pyopencv_to(pyobj_size, size, ArgInfo("size", 0)) &&
        pyopencv_to(pyobj_map1, map1, ArgInfo("map1", 1)) &&
        pyopencv_to(pyobj_map2, map2, ArgInfo("map2", 1)) )
    {
        ERRWRAP2(cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type, map1, map2));
        return Py_BuildValue("(NN)", pyopencv_from(map1), pyopencv_from(map2));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_initWideAngleProjMap(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    Mat distCoeffs;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    int destImageWidth=0;
    int m1type=0;
    PyObject* pyobj_map1 = NULL;
    Mat map1;
    PyObject* pyobj_map2 = NULL;
    Mat map2;
    int projType=PROJ_SPHERICAL_EQRECT;
    double alpha=0;
    float retval;

    const char* keywords[] = { "cameraMatrix", "distCoeffs", "imageSize", "destImageWidth", "m1type", "map1", "map2", "projType", "alpha", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOii|OOid:initWideAngleProjMap", (char**)keywords, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_imageSize, &destImageWidth, &m1type, &pyobj_map1, &pyobj_map2, &projType, &alpha) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_map1, map1, ArgInfo("map1", 1)) &&
        pyopencv_to(pyobj_map2, map2, ArgInfo("map2", 1)) )
    {
        ERRWRAP2(retval = cv::initWideAngleProjMap(cameraMatrix, distCoeffs, imageSize, destImageWidth, m1type, map1, map2, projType, alpha));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(map1), pyopencv_from(map2));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    UMat distCoeffs;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    int destImageWidth=0;
    int m1type=0;
    PyObject* pyobj_map1 = NULL;
    UMat map1;
    PyObject* pyobj_map2 = NULL;
    UMat map2;
    int projType=PROJ_SPHERICAL_EQRECT;
    double alpha=0;
    float retval;

    const char* keywords[] = { "cameraMatrix", "distCoeffs", "imageSize", "destImageWidth", "m1type", "map1", "map2", "projType", "alpha", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOii|OOid:initWideAngleProjMap", (char**)keywords, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_imageSize, &destImageWidth, &m1type, &pyobj_map1, &pyobj_map2, &projType, &alpha) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_map1, map1, ArgInfo("map1", 1)) &&
        pyopencv_to(pyobj_map2, map2, ArgInfo("map2", 1)) )
    {
        ERRWRAP2(retval = cv::initWideAngleProjMap(cameraMatrix, distCoeffs, imageSize, destImageWidth, m1type, map1, map2, projType, alpha));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(map1), pyopencv_from(map2));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_inpaint(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_inpaintMask = NULL;
    Mat inpaintMask;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    double inpaintRadius=0;
    int flags=0;

    const char* keywords[] = { "src", "inpaintMask", "inpaintRadius", "flags", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOdi|O:inpaint", (char**)keywords, &pyobj_src, &pyobj_inpaintMask, &inpaintRadius, &flags, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_inpaintMask, inpaintMask, ArgInfo("inpaintMask", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::inpaint(src, inpaintMask, dst, inpaintRadius, flags));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_inpaintMask = NULL;
    UMat inpaintMask;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    double inpaintRadius=0;
    int flags=0;

    const char* keywords[] = { "src", "inpaintMask", "inpaintRadius", "flags", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOdi|O:inpaint", (char**)keywords, &pyobj_src, &pyobj_inpaintMask, &inpaintRadius, &flags, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_inpaintMask, inpaintMask, ArgInfo("inpaintMask", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::inpaint(src, inpaintMask, dst, inpaintRadius, flags));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_insertChannel(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int coi=0;

    const char* keywords[] = { "src", "dst", "coi", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi:insertChannel", (char**)keywords, &pyobj_src, &pyobj_dst, &coi) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::insertChannel(src, dst, coi));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int coi=0;

    const char* keywords[] = { "src", "dst", "coi", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi:insertChannel", (char**)keywords, &pyobj_src, &pyobj_dst, &coi) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::insertChannel(src, dst, coi));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_integral(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_sum = NULL;
    Mat sum;
    int sdepth=-1;

    const char* keywords[] = { "src", "sum", "sdepth", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oi:integral", (char**)keywords, &pyobj_src, &pyobj_sum, &sdepth) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_sum, sum, ArgInfo("sum", 1)) )
    {
        ERRWRAP2(cv::integral(src, sum, sdepth));
        return pyopencv_from(sum);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_sum = NULL;
    UMat sum;
    int sdepth=-1;

    const char* keywords[] = { "src", "sum", "sdepth", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oi:integral", (char**)keywords, &pyobj_src, &pyobj_sum, &sdepth) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_sum, sum, ArgInfo("sum", 1)) )
    {
        ERRWRAP2(cv::integral(src, sum, sdepth));
        return pyopencv_from(sum);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_integral2(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_sum = NULL;
    Mat sum;
    PyObject* pyobj_sqsum = NULL;
    Mat sqsum;
    int sdepth=-1;
    int sqdepth=-1;

    const char* keywords[] = { "src", "sum", "sqsum", "sdepth", "sqdepth", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOii:integral2", (char**)keywords, &pyobj_src, &pyobj_sum, &pyobj_sqsum, &sdepth, &sqdepth) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_sum, sum, ArgInfo("sum", 1)) &&
        pyopencv_to(pyobj_sqsum, sqsum, ArgInfo("sqsum", 1)) )
    {
        ERRWRAP2(cv::integral(src, sum, sqsum, sdepth, sqdepth));
        return Py_BuildValue("(NN)", pyopencv_from(sum), pyopencv_from(sqsum));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_sum = NULL;
    UMat sum;
    PyObject* pyobj_sqsum = NULL;
    UMat sqsum;
    int sdepth=-1;
    int sqdepth=-1;

    const char* keywords[] = { "src", "sum", "sqsum", "sdepth", "sqdepth", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOii:integral2", (char**)keywords, &pyobj_src, &pyobj_sum, &pyobj_sqsum, &sdepth, &sqdepth) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_sum, sum, ArgInfo("sum", 1)) &&
        pyopencv_to(pyobj_sqsum, sqsum, ArgInfo("sqsum", 1)) )
    {
        ERRWRAP2(cv::integral(src, sum, sqsum, sdepth, sqdepth));
        return Py_BuildValue("(NN)", pyopencv_from(sum), pyopencv_from(sqsum));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_integral3(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_sum = NULL;
    Mat sum;
    PyObject* pyobj_sqsum = NULL;
    Mat sqsum;
    PyObject* pyobj_tilted = NULL;
    Mat tilted;
    int sdepth=-1;
    int sqdepth=-1;

    const char* keywords[] = { "src", "sum", "sqsum", "tilted", "sdepth", "sqdepth", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOOii:integral3", (char**)keywords, &pyobj_src, &pyobj_sum, &pyobj_sqsum, &pyobj_tilted, &sdepth, &sqdepth) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_sum, sum, ArgInfo("sum", 1)) &&
        pyopencv_to(pyobj_sqsum, sqsum, ArgInfo("sqsum", 1)) &&
        pyopencv_to(pyobj_tilted, tilted, ArgInfo("tilted", 1)) )
    {
        ERRWRAP2(cv::integral(src, sum, sqsum, tilted, sdepth, sqdepth));
        return Py_BuildValue("(NNN)", pyopencv_from(sum), pyopencv_from(sqsum), pyopencv_from(tilted));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_sum = NULL;
    UMat sum;
    PyObject* pyobj_sqsum = NULL;
    UMat sqsum;
    PyObject* pyobj_tilted = NULL;
    UMat tilted;
    int sdepth=-1;
    int sqdepth=-1;

    const char* keywords[] = { "src", "sum", "sqsum", "tilted", "sdepth", "sqdepth", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOOii:integral3", (char**)keywords, &pyobj_src, &pyobj_sum, &pyobj_sqsum, &pyobj_tilted, &sdepth, &sqdepth) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_sum, sum, ArgInfo("sum", 1)) &&
        pyopencv_to(pyobj_sqsum, sqsum, ArgInfo("sqsum", 1)) &&
        pyopencv_to(pyobj_tilted, tilted, ArgInfo("tilted", 1)) )
    {
        ERRWRAP2(cv::integral(src, sum, sqsum, tilted, sdepth, sqdepth));
        return Py_BuildValue("(NNN)", pyopencv_from(sum), pyopencv_from(sqsum), pyopencv_from(tilted));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_intersectConvexConvex(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj__p1 = NULL;
    Mat _p1;
    PyObject* pyobj__p2 = NULL;
    Mat _p2;
    PyObject* pyobj__p12 = NULL;
    Mat _p12;
    bool handleNested=true;
    float retval;

    const char* keywords[] = { "_p1", "_p2", "_p12", "handleNested", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Ob:intersectConvexConvex", (char**)keywords, &pyobj__p1, &pyobj__p2, &pyobj__p12, &handleNested) &&
        pyopencv_to(pyobj__p1, _p1, ArgInfo("_p1", 0)) &&
        pyopencv_to(pyobj__p2, _p2, ArgInfo("_p2", 0)) &&
        pyopencv_to(pyobj__p12, _p12, ArgInfo("_p12", 1)) )
    {
        ERRWRAP2(retval = cv::intersectConvexConvex(_p1, _p2, _p12, handleNested));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(_p12));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj__p1 = NULL;
    UMat _p1;
    PyObject* pyobj__p2 = NULL;
    UMat _p2;
    PyObject* pyobj__p12 = NULL;
    UMat _p12;
    bool handleNested=true;
    float retval;

    const char* keywords[] = { "_p1", "_p2", "_p12", "handleNested", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Ob:intersectConvexConvex", (char**)keywords, &pyobj__p1, &pyobj__p2, &pyobj__p12, &handleNested) &&
        pyopencv_to(pyobj__p1, _p1, ArgInfo("_p1", 0)) &&
        pyopencv_to(pyobj__p2, _p2, ArgInfo("_p2", 0)) &&
        pyopencv_to(pyobj__p12, _p12, ArgInfo("_p12", 1)) )
    {
        ERRWRAP2(retval = cv::intersectConvexConvex(_p1, _p2, _p12, handleNested));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(_p12));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_invert(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int flags=DECOMP_LU;
    double retval;

    const char* keywords[] = { "src", "dst", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oi:invert", (char**)keywords, &pyobj_src, &pyobj_dst, &flags) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(retval = cv::invert(src, dst, flags));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(dst));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int flags=DECOMP_LU;
    double retval;

    const char* keywords[] = { "src", "dst", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oi:invert", (char**)keywords, &pyobj_src, &pyobj_dst, &flags) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(retval = cv::invert(src, dst, flags));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(dst));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_invertAffineTransform(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_M = NULL;
    Mat M;
    PyObject* pyobj_iM = NULL;
    Mat iM;

    const char* keywords[] = { "M", "iM", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:invertAffineTransform", (char**)keywords, &pyobj_M, &pyobj_iM) &&
        pyopencv_to(pyobj_M, M, ArgInfo("M", 0)) &&
        pyopencv_to(pyobj_iM, iM, ArgInfo("iM", 1)) )
    {
        ERRWRAP2(cv::invertAffineTransform(M, iM));
        return pyopencv_from(iM);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_M = NULL;
    UMat M;
    PyObject* pyobj_iM = NULL;
    UMat iM;

    const char* keywords[] = { "M", "iM", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:invertAffineTransform", (char**)keywords, &pyobj_M, &pyobj_iM) &&
        pyopencv_to(pyobj_M, M, ArgInfo("M", 0)) &&
        pyopencv_to(pyobj_iM, iM, ArgInfo("iM", 1)) )
    {
        ERRWRAP2(cv::invertAffineTransform(M, iM));
        return pyopencv_from(iM);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_isContourConvex(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_contour = NULL;
    Mat contour;
    bool retval;

    const char* keywords[] = { "contour", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:isContourConvex", (char**)keywords, &pyobj_contour) &&
        pyopencv_to(pyobj_contour, contour, ArgInfo("contour", 0)) )
    {
        ERRWRAP2(retval = cv::isContourConvex(contour));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_contour = NULL;
    UMat contour;
    bool retval;

    const char* keywords[] = { "contour", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:isContourConvex", (char**)keywords, &pyobj_contour) &&
        pyopencv_to(pyobj_contour, contour, ArgInfo("contour", 0)) )
    {
        ERRWRAP2(retval = cv::isContourConvex(contour));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_kmeans(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_data = NULL;
    Mat data;
    int K=0;
    PyObject* pyobj_bestLabels = NULL;
    Mat bestLabels;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria;
    int attempts=0;
    int flags=0;
    PyObject* pyobj_centers = NULL;
    Mat centers;
    double retval;

    const char* keywords[] = { "data", "K", "bestLabels", "criteria", "attempts", "flags", "centers", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OiOOii|O:kmeans", (char**)keywords, &pyobj_data, &K, &pyobj_bestLabels, &pyobj_criteria, &attempts, &flags, &pyobj_centers) &&
        pyopencv_to(pyobj_data, data, ArgInfo("data", 0)) &&
        pyopencv_to(pyobj_bestLabels, bestLabels, ArgInfo("bestLabels", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) &&
        pyopencv_to(pyobj_centers, centers, ArgInfo("centers", 1)) )
    {
        ERRWRAP2(retval = cv::kmeans(data, K, bestLabels, criteria, attempts, flags, centers));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(bestLabels), pyopencv_from(centers));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_data = NULL;
    UMat data;
    int K=0;
    PyObject* pyobj_bestLabels = NULL;
    UMat bestLabels;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria;
    int attempts=0;
    int flags=0;
    PyObject* pyobj_centers = NULL;
    UMat centers;
    double retval;

    const char* keywords[] = { "data", "K", "bestLabels", "criteria", "attempts", "flags", "centers", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OiOOii|O:kmeans", (char**)keywords, &pyobj_data, &K, &pyobj_bestLabels, &pyobj_criteria, &attempts, &flags, &pyobj_centers) &&
        pyopencv_to(pyobj_data, data, ArgInfo("data", 0)) &&
        pyopencv_to(pyobj_bestLabels, bestLabels, ArgInfo("bestLabels", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) &&
        pyopencv_to(pyobj_centers, centers, ArgInfo("centers", 1)) )
    {
        ERRWRAP2(retval = cv::kmeans(data, K, bestLabels, criteria, attempts, flags, centers));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(bestLabels), pyopencv_from(centers));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_line(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_pt1 = NULL;
    Point pt1;
    PyObject* pyobj_pt2 = NULL;
    Point pt2;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int thickness=1;
    int lineType=LINE_8;
    int shift=0;

    const char* keywords[] = { "img", "pt1", "pt2", "color", "thickness", "lineType", "shift", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|iii:line", (char**)keywords, &pyobj_img, &pyobj_pt1, &pyobj_pt2, &pyobj_color, &thickness, &lineType, &shift) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_pt1, pt1, ArgInfo("pt1", 0)) &&
        pyopencv_to(pyobj_pt2, pt2, ArgInfo("pt2", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2(cv::line(img, pt1, pt2, color, thickness, lineType, shift));
        return pyopencv_from(img);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_img = NULL;
    UMat img;
    PyObject* pyobj_pt1 = NULL;
    Point pt1;
    PyObject* pyobj_pt2 = NULL;
    Point pt2;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int thickness=1;
    int lineType=LINE_8;
    int shift=0;

    const char* keywords[] = { "img", "pt1", "pt2", "color", "thickness", "lineType", "shift", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|iii:line", (char**)keywords, &pyobj_img, &pyobj_pt1, &pyobj_pt2, &pyobj_color, &thickness, &lineType, &shift) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_pt1, pt1, ArgInfo("pt1", 0)) &&
        pyopencv_to(pyobj_pt2, pt2, ArgInfo("pt2", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2(cv::line(img, pt1, pt2, color, thickness, lineType, shift));
        return pyopencv_from(img);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_linearPolar(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_center = NULL;
    Point2f center;
    double maxRadius=0;
    int flags=0;

    const char* keywords[] = { "src", "center", "maxRadius", "flags", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOdi|O:linearPolar", (char**)keywords, &pyobj_src, &pyobj_center, &maxRadius, &flags, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_center, center, ArgInfo("center", 0)) )
    {
        ERRWRAP2(cv::linearPolar(src, dst, center, maxRadius, flags));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_center = NULL;
    Point2f center;
    double maxRadius=0;
    int flags=0;

    const char* keywords[] = { "src", "center", "maxRadius", "flags", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOdi|O:linearPolar", (char**)keywords, &pyobj_src, &pyobj_center, &maxRadius, &flags, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_center, center, ArgInfo("center", 0)) )
    {
        ERRWRAP2(cv::linearPolar(src, dst, center, maxRadius, flags));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_log(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:log", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::log(src, dst));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:log", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::log(src, dst));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_logPolar(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_center = NULL;
    Point2f center;
    double M=0;
    int flags=0;

    const char* keywords[] = { "src", "center", "M", "flags", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOdi|O:logPolar", (char**)keywords, &pyobj_src, &pyobj_center, &M, &flags, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_center, center, ArgInfo("center", 0)) )
    {
        ERRWRAP2(cv::logPolar(src, dst, center, M, flags));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_center = NULL;
    Point2f center;
    double M=0;
    int flags=0;

    const char* keywords[] = { "src", "center", "M", "flags", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOdi|O:logPolar", (char**)keywords, &pyobj_src, &pyobj_center, &M, &flags, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_center, center, ArgInfo("center", 0)) )
    {
        ERRWRAP2(cv::logPolar(src, dst, center, M, flags));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_magnitude(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_x = NULL;
    Mat x;
    PyObject* pyobj_y = NULL;
    Mat y;
    PyObject* pyobj_magnitude = NULL;
    Mat magnitude;

    const char* keywords[] = { "x", "y", "magnitude", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:magnitude", (char**)keywords, &pyobj_x, &pyobj_y, &pyobj_magnitude) &&
        pyopencv_to(pyobj_x, x, ArgInfo("x", 0)) &&
        pyopencv_to(pyobj_y, y, ArgInfo("y", 0)) &&
        pyopencv_to(pyobj_magnitude, magnitude, ArgInfo("magnitude", 1)) )
    {
        ERRWRAP2(cv::magnitude(x, y, magnitude));
        return pyopencv_from(magnitude);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_x = NULL;
    UMat x;
    PyObject* pyobj_y = NULL;
    UMat y;
    PyObject* pyobj_magnitude = NULL;
    UMat magnitude;

    const char* keywords[] = { "x", "y", "magnitude", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:magnitude", (char**)keywords, &pyobj_x, &pyobj_y, &pyobj_magnitude) &&
        pyopencv_to(pyobj_x, x, ArgInfo("x", 0)) &&
        pyopencv_to(pyobj_y, y, ArgInfo("y", 0)) &&
        pyopencv_to(pyobj_magnitude, magnitude, ArgInfo("magnitude", 1)) )
    {
        ERRWRAP2(cv::magnitude(x, y, magnitude));
        return pyopencv_from(magnitude);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_matMulDeriv(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_A = NULL;
    Mat A;
    PyObject* pyobj_B = NULL;
    Mat B;
    PyObject* pyobj_dABdA = NULL;
    Mat dABdA;
    PyObject* pyobj_dABdB = NULL;
    Mat dABdB;

    const char* keywords[] = { "A", "B", "dABdA", "dABdB", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OO:matMulDeriv", (char**)keywords, &pyobj_A, &pyobj_B, &pyobj_dABdA, &pyobj_dABdB) &&
        pyopencv_to(pyobj_A, A, ArgInfo("A", 0)) &&
        pyopencv_to(pyobj_B, B, ArgInfo("B", 0)) &&
        pyopencv_to(pyobj_dABdA, dABdA, ArgInfo("dABdA", 1)) &&
        pyopencv_to(pyobj_dABdB, dABdB, ArgInfo("dABdB", 1)) )
    {
        ERRWRAP2(cv::matMulDeriv(A, B, dABdA, dABdB));
        return Py_BuildValue("(NN)", pyopencv_from(dABdA), pyopencv_from(dABdB));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_A = NULL;
    UMat A;
    PyObject* pyobj_B = NULL;
    UMat B;
    PyObject* pyobj_dABdA = NULL;
    UMat dABdA;
    PyObject* pyobj_dABdB = NULL;
    UMat dABdB;

    const char* keywords[] = { "A", "B", "dABdA", "dABdB", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OO:matMulDeriv", (char**)keywords, &pyobj_A, &pyobj_B, &pyobj_dABdA, &pyobj_dABdB) &&
        pyopencv_to(pyobj_A, A, ArgInfo("A", 0)) &&
        pyopencv_to(pyobj_B, B, ArgInfo("B", 0)) &&
        pyopencv_to(pyobj_dABdA, dABdA, ArgInfo("dABdA", 1)) &&
        pyopencv_to(pyobj_dABdB, dABdB, ArgInfo("dABdB", 1)) )
    {
        ERRWRAP2(cv::matMulDeriv(A, B, dABdA, dABdB));
        return Py_BuildValue("(NN)", pyopencv_from(dABdA), pyopencv_from(dABdB));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_matchShapes(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_contour1 = NULL;
    Mat contour1;
    PyObject* pyobj_contour2 = NULL;
    Mat contour2;
    int method=0;
    double parameter=0;
    double retval;

    const char* keywords[] = { "contour1", "contour2", "method", "parameter", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOid:matchShapes", (char**)keywords, &pyobj_contour1, &pyobj_contour2, &method, &parameter) &&
        pyopencv_to(pyobj_contour1, contour1, ArgInfo("contour1", 0)) &&
        pyopencv_to(pyobj_contour2, contour2, ArgInfo("contour2", 0)) )
    {
        ERRWRAP2(retval = cv::matchShapes(contour1, contour2, method, parameter));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_contour1 = NULL;
    UMat contour1;
    PyObject* pyobj_contour2 = NULL;
    UMat contour2;
    int method=0;
    double parameter=0;
    double retval;

    const char* keywords[] = { "contour1", "contour2", "method", "parameter", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOid:matchShapes", (char**)keywords, &pyobj_contour1, &pyobj_contour2, &method, &parameter) &&
        pyopencv_to(pyobj_contour1, contour1, ArgInfo("contour1", 0)) &&
        pyopencv_to(pyobj_contour2, contour2, ArgInfo("contour2", 0)) )
    {
        ERRWRAP2(retval = cv::matchShapes(contour1, contour2, method, parameter));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_matchTemplate(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_templ = NULL;
    Mat templ;
    PyObject* pyobj_result = NULL;
    Mat result;
    int method=0;
    PyObject* pyobj_mask = NULL;
    Mat mask;

    const char* keywords[] = { "image", "templ", "method", "result", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi|OO:matchTemplate", (char**)keywords, &pyobj_image, &pyobj_templ, &method, &pyobj_result, &pyobj_mask) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_templ, templ, ArgInfo("templ", 0)) &&
        pyopencv_to(pyobj_result, result, ArgInfo("result", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::matchTemplate(image, templ, result, method, mask));
        return pyopencv_from(result);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_templ = NULL;
    UMat templ;
    PyObject* pyobj_result = NULL;
    UMat result;
    int method=0;
    PyObject* pyobj_mask = NULL;
    UMat mask;

    const char* keywords[] = { "image", "templ", "method", "result", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi|OO:matchTemplate", (char**)keywords, &pyobj_image, &pyobj_templ, &method, &pyobj_result, &pyobj_mask) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_templ, templ, ArgInfo("templ", 0)) &&
        pyopencv_to(pyobj_result, result, ArgInfo("result", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::matchTemplate(image, templ, result, method, mask));
        return pyopencv_from(result);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_max(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    PyObject* pyobj_src2 = NULL;
    Mat src2;
    PyObject* pyobj_dst = NULL;
    Mat dst;

    const char* keywords[] = { "src1", "src2", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:max", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::max(src1, src2, dst));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src1 = NULL;
    UMat src1;
    PyObject* pyobj_src2 = NULL;
    UMat src2;
    PyObject* pyobj_dst = NULL;
    UMat dst;

    const char* keywords[] = { "src1", "src2", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:max", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::max(src1, src2, dst));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_mean(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    Scalar retval;

    const char* keywords[] = { "src", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:mean", (char**)keywords, &pyobj_src, &pyobj_mask) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(retval = cv::mean(src, mask));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    Scalar retval;

    const char* keywords[] = { "src", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:mean", (char**)keywords, &pyobj_src, &pyobj_mask) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(retval = cv::mean(src, mask));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_meanShift(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_probImage = NULL;
    Mat probImage;
    PyObject* pyobj_window = NULL;
    Rect window;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria;
    int retval;

    const char* keywords[] = { "probImage", "window", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:meanShift", (char**)keywords, &pyobj_probImage, &pyobj_window, &pyobj_criteria) &&
        pyopencv_to(pyobj_probImage, probImage, ArgInfo("probImage", 0)) &&
        pyopencv_to(pyobj_window, window, ArgInfo("window", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(retval = cv::meanShift(probImage, window, criteria));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(window));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_probImage = NULL;
    UMat probImage;
    PyObject* pyobj_window = NULL;
    Rect window;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria;
    int retval;

    const char* keywords[] = { "probImage", "window", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:meanShift", (char**)keywords, &pyobj_probImage, &pyobj_window, &pyobj_criteria) &&
        pyopencv_to(pyobj_probImage, probImage, ArgInfo("probImage", 0)) &&
        pyopencv_to(pyobj_window, window, ArgInfo("window", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(retval = cv::meanShift(probImage, window, criteria));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(window));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_meanStdDev(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_mean = NULL;
    Mat mean;
    PyObject* pyobj_stddev = NULL;
    Mat stddev;
    PyObject* pyobj_mask = NULL;
    Mat mask;

    const char* keywords[] = { "src", "mean", "stddev", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOO:meanStdDev", (char**)keywords, &pyobj_src, &pyobj_mean, &pyobj_stddev, &pyobj_mask) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_mean, mean, ArgInfo("mean", 1)) &&
        pyopencv_to(pyobj_stddev, stddev, ArgInfo("stddev", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::meanStdDev(src, mean, stddev, mask));
        return Py_BuildValue("(NN)", pyopencv_from(mean), pyopencv_from(stddev));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_mean = NULL;
    UMat mean;
    PyObject* pyobj_stddev = NULL;
    UMat stddev;
    PyObject* pyobj_mask = NULL;
    UMat mask;

    const char* keywords[] = { "src", "mean", "stddev", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOO:meanStdDev", (char**)keywords, &pyobj_src, &pyobj_mean, &pyobj_stddev, &pyobj_mask) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_mean, mean, ArgInfo("mean", 1)) &&
        pyopencv_to(pyobj_stddev, stddev, ArgInfo("stddev", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::meanStdDev(src, mean, stddev, mask));
        return Py_BuildValue("(NN)", pyopencv_from(mean), pyopencv_from(stddev));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_medianBlur(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int ksize=0;

    const char* keywords[] = { "src", "ksize", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|O:medianBlur", (char**)keywords, &pyobj_src, &ksize, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::medianBlur(src, dst, ksize));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int ksize=0;

    const char* keywords[] = { "src", "ksize", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|O:medianBlur", (char**)keywords, &pyobj_src, &ksize, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::medianBlur(src, dst, ksize));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_merge(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_mv = NULL;
    vector_Mat mv;
    PyObject* pyobj_dst = NULL;
    Mat dst;

    const char* keywords[] = { "mv", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:merge", (char**)keywords, &pyobj_mv, &pyobj_dst) &&
        pyopencv_to(pyobj_mv, mv, ArgInfo("mv", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::merge(mv, dst));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_mv = NULL;
    vector_Mat mv;
    PyObject* pyobj_dst = NULL;
    UMat dst;

    const char* keywords[] = { "mv", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:merge", (char**)keywords, &pyobj_mv, &pyobj_dst) &&
        pyopencv_to(pyobj_mv, mv, ArgInfo("mv", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::merge(mv, dst));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_min(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    PyObject* pyobj_src2 = NULL;
    Mat src2;
    PyObject* pyobj_dst = NULL;
    Mat dst;

    const char* keywords[] = { "src1", "src2", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:min", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::min(src1, src2, dst));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src1 = NULL;
    UMat src1;
    PyObject* pyobj_src2 = NULL;
    UMat src2;
    PyObject* pyobj_dst = NULL;
    UMat dst;

    const char* keywords[] = { "src1", "src2", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:min", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::min(src1, src2, dst));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_minAreaRect(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_points = NULL;
    Mat points;
    RotatedRect retval;

    const char* keywords[] = { "points", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:minAreaRect", (char**)keywords, &pyobj_points) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 0)) )
    {
        ERRWRAP2(retval = cv::minAreaRect(points));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_points = NULL;
    UMat points;
    RotatedRect retval;

    const char* keywords[] = { "points", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:minAreaRect", (char**)keywords, &pyobj_points) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 0)) )
    {
        ERRWRAP2(retval = cv::minAreaRect(points));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_minEnclosingCircle(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_points = NULL;
    Mat points;
    Point2f center;
    float radius;

    const char* keywords[] = { "points", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:minEnclosingCircle", (char**)keywords, &pyobj_points) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 0)) )
    {
        ERRWRAP2(cv::minEnclosingCircle(points, center, radius));
        return Py_BuildValue("(NN)", pyopencv_from(center), pyopencv_from(radius));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_points = NULL;
    UMat points;
    Point2f center;
    float radius;

    const char* keywords[] = { "points", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:minEnclosingCircle", (char**)keywords, &pyobj_points) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 0)) )
    {
        ERRWRAP2(cv::minEnclosingCircle(points, center, radius));
        return Py_BuildValue("(NN)", pyopencv_from(center), pyopencv_from(radius));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_minEnclosingTriangle(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_points = NULL;
    Mat points;
    PyObject* pyobj_triangle = NULL;
    Mat triangle;
    double retval;

    const char* keywords[] = { "points", "triangle", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:minEnclosingTriangle", (char**)keywords, &pyobj_points, &pyobj_triangle) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 0)) &&
        pyopencv_to(pyobj_triangle, triangle, ArgInfo("triangle", 1)) )
    {
        ERRWRAP2(retval = cv::minEnclosingTriangle(points, triangle));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(triangle));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_points = NULL;
    UMat points;
    PyObject* pyobj_triangle = NULL;
    UMat triangle;
    double retval;

    const char* keywords[] = { "points", "triangle", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:minEnclosingTriangle", (char**)keywords, &pyobj_points, &pyobj_triangle) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 0)) &&
        pyopencv_to(pyobj_triangle, triangle, ArgInfo("triangle", 1)) )
    {
        ERRWRAP2(retval = cv::minEnclosingTriangle(points, triangle));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(triangle));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_minMaxLoc(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    PyObject* pyobj_mask = NULL;
    Mat mask;

    const char* keywords[] = { "src", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:minMaxLoc", (char**)keywords, &pyobj_src, &pyobj_mask) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc, mask));
        return Py_BuildValue("(NNNN)", pyopencv_from(minVal), pyopencv_from(maxVal), pyopencv_from(minLoc), pyopencv_from(maxLoc));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    PyObject* pyobj_mask = NULL;
    UMat mask;

    const char* keywords[] = { "src", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:minMaxLoc", (char**)keywords, &pyobj_src, &pyobj_mask) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc, mask));
        return Py_BuildValue("(NNNN)", pyopencv_from(minVal), pyopencv_from(maxVal), pyopencv_from(minLoc), pyopencv_from(maxLoc));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_mixChannels(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    vector_Mat dst;
    PyObject* pyobj_fromTo = NULL;
    vector_int fromTo;

    const char* keywords[] = { "src", "dst", "fromTo", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:mixChannels", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_fromTo) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_fromTo, fromTo, ArgInfo("fromTo", 0)) )
    {
        ERRWRAP2(cv::mixChannels(src, dst, fromTo));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    vector_Mat dst;
    PyObject* pyobj_fromTo = NULL;
    vector_int fromTo;

    const char* keywords[] = { "src", "dst", "fromTo", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:mixChannels", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_fromTo) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_fromTo, fromTo, ArgInfo("fromTo", 0)) )
    {
        ERRWRAP2(cv::mixChannels(src, dst, fromTo));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_moments(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_array = NULL;
    Mat array;
    bool binaryImage=false;
    Moments retval;

    const char* keywords[] = { "array", "binaryImage", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|b:moments", (char**)keywords, &pyobj_array, &binaryImage) &&
        pyopencv_to(pyobj_array, array, ArgInfo("array", 0)) )
    {
        ERRWRAP2(retval = cv::moments(array, binaryImage));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_array = NULL;
    UMat array;
    bool binaryImage=false;
    Moments retval;

    const char* keywords[] = { "array", "binaryImage", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|b:moments", (char**)keywords, &pyobj_array, &binaryImage) &&
        pyopencv_to(pyobj_array, array, ArgInfo("array", 0)) )
    {
        ERRWRAP2(retval = cv::moments(array, binaryImage));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_morphologyEx(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int op=0;
    PyObject* pyobj_kernel = NULL;
    Mat kernel;
    PyObject* pyobj_anchor = NULL;
    Point anchor=Point(-1,-1);
    int iterations=1;
    int borderType=BORDER_CONSTANT;
    PyObject* pyobj_borderValue = NULL;
    Scalar borderValue=morphologyDefaultBorderValue();

    const char* keywords[] = { "src", "op", "kernel", "dst", "anchor", "iterations", "borderType", "borderValue", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OiO|OOiiO:morphologyEx", (char**)keywords, &pyobj_src, &op, &pyobj_kernel, &pyobj_dst, &pyobj_anchor, &iterations, &borderType, &pyobj_borderValue) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 0)) &&
        pyopencv_to(pyobj_anchor, anchor, ArgInfo("anchor", 0)) &&
        pyopencv_to(pyobj_borderValue, borderValue, ArgInfo("borderValue", 0)) )
    {
        ERRWRAP2(cv::morphologyEx(src, dst, op, kernel, anchor, iterations, borderType, borderValue));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int op=0;
    PyObject* pyobj_kernel = NULL;
    UMat kernel;
    PyObject* pyobj_anchor = NULL;
    Point anchor=Point(-1,-1);
    int iterations=1;
    int borderType=BORDER_CONSTANT;
    PyObject* pyobj_borderValue = NULL;
    Scalar borderValue=morphologyDefaultBorderValue();

    const char* keywords[] = { "src", "op", "kernel", "dst", "anchor", "iterations", "borderType", "borderValue", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OiO|OOiiO:morphologyEx", (char**)keywords, &pyobj_src, &op, &pyobj_kernel, &pyobj_dst, &pyobj_anchor, &iterations, &borderType, &pyobj_borderValue) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 0)) &&
        pyopencv_to(pyobj_anchor, anchor, ArgInfo("anchor", 0)) &&
        pyopencv_to(pyobj_borderValue, borderValue, ArgInfo("borderValue", 0)) )
    {
        ERRWRAP2(cv::morphologyEx(src, dst, op, kernel, anchor, iterations, borderType, borderValue));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_moveWindow(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_winname = NULL;
    String winname;
    int x=0;
    int y=0;

    const char* keywords[] = { "winname", "x", "y", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii:moveWindow", (char**)keywords, &pyobj_winname, &x, &y) &&
        pyopencv_to(pyobj_winname, winname, ArgInfo("winname", 0)) )
    {
        ERRWRAP2(cv::moveWindow(winname, x, y));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_mulSpectrums(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_a = NULL;
    Mat a;
    PyObject* pyobj_b = NULL;
    Mat b;
    PyObject* pyobj_c = NULL;
    Mat c;
    int flags=0;
    bool conjB=false;

    const char* keywords[] = { "a", "b", "flags", "c", "conjB", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi|Ob:mulSpectrums", (char**)keywords, &pyobj_a, &pyobj_b, &flags, &pyobj_c, &conjB) &&
        pyopencv_to(pyobj_a, a, ArgInfo("a", 0)) &&
        pyopencv_to(pyobj_b, b, ArgInfo("b", 0)) &&
        pyopencv_to(pyobj_c, c, ArgInfo("c", 1)) )
    {
        ERRWRAP2(cv::mulSpectrums(a, b, c, flags, conjB));
        return pyopencv_from(c);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_a = NULL;
    UMat a;
    PyObject* pyobj_b = NULL;
    UMat b;
    PyObject* pyobj_c = NULL;
    UMat c;
    int flags=0;
    bool conjB=false;

    const char* keywords[] = { "a", "b", "flags", "c", "conjB", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi|Ob:mulSpectrums", (char**)keywords, &pyobj_a, &pyobj_b, &flags, &pyobj_c, &conjB) &&
        pyopencv_to(pyobj_a, a, ArgInfo("a", 0)) &&
        pyopencv_to(pyobj_b, b, ArgInfo("b", 0)) &&
        pyopencv_to(pyobj_c, c, ArgInfo("c", 1)) )
    {
        ERRWRAP2(cv::mulSpectrums(a, b, c, flags, conjB));
        return pyopencv_from(c);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_mulTransposed(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    bool aTa=0;
    PyObject* pyobj_delta = NULL;
    Mat delta;
    double scale=1;
    int dtype=-1;

    const char* keywords[] = { "src", "aTa", "dst", "delta", "scale", "dtype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Ob|OOdi:mulTransposed", (char**)keywords, &pyobj_src, &aTa, &pyobj_dst, &pyobj_delta, &scale, &dtype) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_delta, delta, ArgInfo("delta", 0)) )
    {
        ERRWRAP2(cv::mulTransposed(src, dst, aTa, delta, scale, dtype));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    bool aTa=0;
    PyObject* pyobj_delta = NULL;
    UMat delta;
    double scale=1;
    int dtype=-1;

    const char* keywords[] = { "src", "aTa", "dst", "delta", "scale", "dtype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Ob|OOdi:mulTransposed", (char**)keywords, &pyobj_src, &aTa, &pyobj_dst, &pyobj_delta, &scale, &dtype) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_delta, delta, ArgInfo("delta", 0)) )
    {
        ERRWRAP2(cv::mulTransposed(src, dst, aTa, delta, scale, dtype));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_multiply(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    PyObject* pyobj_src2 = NULL;
    Mat src2;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    double scale=1;
    int dtype=-1;

    const char* keywords[] = { "src1", "src2", "dst", "scale", "dtype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Odi:multiply", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &scale, &dtype) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::multiply(src1, src2, dst, scale, dtype));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src1 = NULL;
    UMat src1;
    PyObject* pyobj_src2 = NULL;
    UMat src2;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    double scale=1;
    int dtype=-1;

    const char* keywords[] = { "src1", "src2", "dst", "scale", "dtype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Odi:multiply", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &scale, &dtype) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::multiply(src1, src2, dst, scale, dtype));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_namedWindow(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_winname = NULL;
    String winname;
    int flags=WINDOW_AUTOSIZE;

    const char* keywords[] = { "winname", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|i:namedWindow", (char**)keywords, &pyobj_winname, &flags) &&
        pyopencv_to(pyobj_winname, winname, ArgInfo("winname", 0)) )
    {
        ERRWRAP2(cv::namedWindow(winname, flags));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_norm(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    int normType=NORM_L2;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    double retval;

    const char* keywords[] = { "src1", "normType", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|iO:norm", (char**)keywords, &pyobj_src1, &normType, &pyobj_mask) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(retval = cv::norm(src1, normType, mask));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src1 = NULL;
    UMat src1;
    int normType=NORM_L2;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    double retval;

    const char* keywords[] = { "src1", "normType", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|iO:norm", (char**)keywords, &pyobj_src1, &normType, &pyobj_mask) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(retval = cv::norm(src1, normType, mask));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    PyObject* pyobj_src2 = NULL;
    Mat src2;
    int normType=NORM_L2;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    double retval;

    const char* keywords[] = { "src1", "src2", "normType", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|iO:norm", (char**)keywords, &pyobj_src1, &pyobj_src2, &normType, &pyobj_mask) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(retval = cv::norm(src1, src2, normType, mask));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src1 = NULL;
    UMat src1;
    PyObject* pyobj_src2 = NULL;
    UMat src2;
    int normType=NORM_L2;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    double retval;

    const char* keywords[] = { "src1", "src2", "normType", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|iO:norm", (char**)keywords, &pyobj_src1, &pyobj_src2, &normType, &pyobj_mask) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(retval = cv::norm(src1, src2, normType, mask));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_normalize(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    double alpha=1;
    double beta=0;
    int norm_type=NORM_L2;
    int dtype=-1;
    PyObject* pyobj_mask = NULL;
    Mat mask;

    const char* keywords[] = { "src", "dst", "alpha", "beta", "norm_type", "dtype", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|ddiiO:normalize", (char**)keywords, &pyobj_src, &pyobj_dst, &alpha, &beta, &norm_type, &dtype, &pyobj_mask) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::normalize(src, dst, alpha, beta, norm_type, dtype, mask));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    double alpha=1;
    double beta=0;
    int norm_type=NORM_L2;
    int dtype=-1;
    PyObject* pyobj_mask = NULL;
    UMat mask;

    const char* keywords[] = { "src", "dst", "alpha", "beta", "norm_type", "dtype", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|ddiiO:normalize", (char**)keywords, &pyobj_src, &pyobj_dst, &alpha, &beta, &norm_type, &dtype, &pyobj_mask) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::normalize(src, dst, alpha, beta, norm_type, dtype, mask));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_patchNaNs(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_a = NULL;
    Mat a;
    double val=0;

    const char* keywords[] = { "a", "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|d:patchNaNs", (char**)keywords, &pyobj_a, &val) &&
        pyopencv_to(pyobj_a, a, ArgInfo("a", 1)) )
    {
        ERRWRAP2(cv::patchNaNs(a, val));
        return pyopencv_from(a);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_a = NULL;
    UMat a;
    double val=0;

    const char* keywords[] = { "a", "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|d:patchNaNs", (char**)keywords, &pyobj_a, &val) &&
        pyopencv_to(pyobj_a, a, ArgInfo("a", 1)) )
    {
        ERRWRAP2(cv::patchNaNs(a, val));
        return pyopencv_from(a);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_pencilSketch(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst1 = NULL;
    Mat dst1;
    PyObject* pyobj_dst2 = NULL;
    Mat dst2;
    float sigma_s=60;
    float sigma_r=0.07f;
    float shade_factor=0.02f;

    const char* keywords[] = { "src", "dst1", "dst2", "sigma_s", "sigma_r", "shade_factor", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOfff:pencilSketch", (char**)keywords, &pyobj_src, &pyobj_dst1, &pyobj_dst2, &sigma_s, &sigma_r, &shade_factor) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst1, dst1, ArgInfo("dst1", 1)) &&
        pyopencv_to(pyobj_dst2, dst2, ArgInfo("dst2", 1)) )
    {
        ERRWRAP2(cv::pencilSketch(src, dst1, dst2, sigma_s, sigma_r, shade_factor));
        return Py_BuildValue("(NN)", pyopencv_from(dst1), pyopencv_from(dst2));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst1 = NULL;
    UMat dst1;
    PyObject* pyobj_dst2 = NULL;
    UMat dst2;
    float sigma_s=60;
    float sigma_r=0.07f;
    float shade_factor=0.02f;

    const char* keywords[] = { "src", "dst1", "dst2", "sigma_s", "sigma_r", "shade_factor", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOfff:pencilSketch", (char**)keywords, &pyobj_src, &pyobj_dst1, &pyobj_dst2, &sigma_s, &sigma_r, &shade_factor) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst1, dst1, ArgInfo("dst1", 1)) &&
        pyopencv_to(pyobj_dst2, dst2, ArgInfo("dst2", 1)) )
    {
        ERRWRAP2(cv::pencilSketch(src, dst1, dst2, sigma_s, sigma_r, shade_factor));
        return Py_BuildValue("(NN)", pyopencv_from(dst1), pyopencv_from(dst2));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_perspectiveTransform(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_m = NULL;
    Mat m;

    const char* keywords[] = { "src", "m", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:perspectiveTransform", (char**)keywords, &pyobj_src, &pyobj_m, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_m, m, ArgInfo("m", 0)) )
    {
        ERRWRAP2(cv::perspectiveTransform(src, dst, m));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_m = NULL;
    UMat m;

    const char* keywords[] = { "src", "m", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:perspectiveTransform", (char**)keywords, &pyobj_src, &pyobj_m, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_m, m, ArgInfo("m", 0)) )
    {
        ERRWRAP2(cv::perspectiveTransform(src, dst, m));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_phase(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_x = NULL;
    Mat x;
    PyObject* pyobj_y = NULL;
    Mat y;
    PyObject* pyobj_angle = NULL;
    Mat angle;
    bool angleInDegrees=false;

    const char* keywords[] = { "x", "y", "angle", "angleInDegrees", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Ob:phase", (char**)keywords, &pyobj_x, &pyobj_y, &pyobj_angle, &angleInDegrees) &&
        pyopencv_to(pyobj_x, x, ArgInfo("x", 0)) &&
        pyopencv_to(pyobj_y, y, ArgInfo("y", 0)) &&
        pyopencv_to(pyobj_angle, angle, ArgInfo("angle", 1)) )
    {
        ERRWRAP2(cv::phase(x, y, angle, angleInDegrees));
        return pyopencv_from(angle);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_x = NULL;
    UMat x;
    PyObject* pyobj_y = NULL;
    UMat y;
    PyObject* pyobj_angle = NULL;
    UMat angle;
    bool angleInDegrees=false;

    const char* keywords[] = { "x", "y", "angle", "angleInDegrees", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Ob:phase", (char**)keywords, &pyobj_x, &pyobj_y, &pyobj_angle, &angleInDegrees) &&
        pyopencv_to(pyobj_x, x, ArgInfo("x", 0)) &&
        pyopencv_to(pyobj_y, y, ArgInfo("y", 0)) &&
        pyopencv_to(pyobj_angle, angle, ArgInfo("angle", 1)) )
    {
        ERRWRAP2(cv::phase(x, y, angle, angleInDegrees));
        return pyopencv_from(angle);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_phaseCorrelate(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    PyObject* pyobj_src2 = NULL;
    Mat src2;
    PyObject* pyobj_window = NULL;
    Mat window;
    double response;
    Point2d retval;

    const char* keywords[] = { "src1", "src2", "window", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:phaseCorrelate", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_window) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_window, window, ArgInfo("window", 0)) )
    {
        ERRWRAP2(retval = cv::phaseCorrelate(src1, src2, window, &response));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(response));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src1 = NULL;
    UMat src1;
    PyObject* pyobj_src2 = NULL;
    UMat src2;
    PyObject* pyobj_window = NULL;
    UMat window;
    double response;
    Point2d retval;

    const char* keywords[] = { "src1", "src2", "window", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:phaseCorrelate", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_window) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_window, window, ArgInfo("window", 0)) )
    {
        ERRWRAP2(retval = cv::phaseCorrelate(src1, src2, window, &response));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(response));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_pointPolygonTest(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_contour = NULL;
    Mat contour;
    PyObject* pyobj_pt = NULL;
    Point2f pt;
    bool measureDist=0;
    double retval;

    const char* keywords[] = { "contour", "pt", "measureDist", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOb:pointPolygonTest", (char**)keywords, &pyobj_contour, &pyobj_pt, &measureDist) &&
        pyopencv_to(pyobj_contour, contour, ArgInfo("contour", 0)) &&
        pyopencv_to(pyobj_pt, pt, ArgInfo("pt", 0)) )
    {
        ERRWRAP2(retval = cv::pointPolygonTest(contour, pt, measureDist));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_contour = NULL;
    UMat contour;
    PyObject* pyobj_pt = NULL;
    Point2f pt;
    bool measureDist=0;
    double retval;

    const char* keywords[] = { "contour", "pt", "measureDist", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOb:pointPolygonTest", (char**)keywords, &pyobj_contour, &pyobj_pt, &measureDist) &&
        pyopencv_to(pyobj_contour, contour, ArgInfo("contour", 0)) &&
        pyopencv_to(pyobj_pt, pt, ArgInfo("pt", 0)) )
    {
        ERRWRAP2(retval = cv::pointPolygonTest(contour, pt, measureDist));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_polarToCart(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_magnitude = NULL;
    Mat magnitude;
    PyObject* pyobj_angle = NULL;
    Mat angle;
    PyObject* pyobj_x = NULL;
    Mat x;
    PyObject* pyobj_y = NULL;
    Mat y;
    bool angleInDegrees=false;

    const char* keywords[] = { "magnitude", "angle", "x", "y", "angleInDegrees", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOb:polarToCart", (char**)keywords, &pyobj_magnitude, &pyobj_angle, &pyobj_x, &pyobj_y, &angleInDegrees) &&
        pyopencv_to(pyobj_magnitude, magnitude, ArgInfo("magnitude", 0)) &&
        pyopencv_to(pyobj_angle, angle, ArgInfo("angle", 0)) &&
        pyopencv_to(pyobj_x, x, ArgInfo("x", 1)) &&
        pyopencv_to(pyobj_y, y, ArgInfo("y", 1)) )
    {
        ERRWRAP2(cv::polarToCart(magnitude, angle, x, y, angleInDegrees));
        return Py_BuildValue("(NN)", pyopencv_from(x), pyopencv_from(y));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_magnitude = NULL;
    UMat magnitude;
    PyObject* pyobj_angle = NULL;
    UMat angle;
    PyObject* pyobj_x = NULL;
    UMat x;
    PyObject* pyobj_y = NULL;
    UMat y;
    bool angleInDegrees=false;

    const char* keywords[] = { "magnitude", "angle", "x", "y", "angleInDegrees", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOb:polarToCart", (char**)keywords, &pyobj_magnitude, &pyobj_angle, &pyobj_x, &pyobj_y, &angleInDegrees) &&
        pyopencv_to(pyobj_magnitude, magnitude, ArgInfo("magnitude", 0)) &&
        pyopencv_to(pyobj_angle, angle, ArgInfo("angle", 0)) &&
        pyopencv_to(pyobj_x, x, ArgInfo("x", 1)) &&
        pyopencv_to(pyobj_y, y, ArgInfo("y", 1)) )
    {
        ERRWRAP2(cv::polarToCart(magnitude, angle, x, y, angleInDegrees));
        return Py_BuildValue("(NN)", pyopencv_from(x), pyopencv_from(y));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_polylines(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_pts = NULL;
    vector_Mat pts;
    bool isClosed=0;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int thickness=1;
    int lineType=LINE_8;
    int shift=0;

    const char* keywords[] = { "img", "pts", "isClosed", "color", "thickness", "lineType", "shift", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OObO|iii:polylines", (char**)keywords, &pyobj_img, &pyobj_pts, &isClosed, &pyobj_color, &thickness, &lineType, &shift) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_pts, pts, ArgInfo("pts", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2(cv::polylines(img, pts, isClosed, color, thickness, lineType, shift));
        return pyopencv_from(img);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_img = NULL;
    UMat img;
    PyObject* pyobj_pts = NULL;
    vector_Mat pts;
    bool isClosed=0;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int thickness=1;
    int lineType=LINE_8;
    int shift=0;

    const char* keywords[] = { "img", "pts", "isClosed", "color", "thickness", "lineType", "shift", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OObO|iii:polylines", (char**)keywords, &pyobj_img, &pyobj_pts, &isClosed, &pyobj_color, &thickness, &lineType, &shift) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_pts, pts, ArgInfo("pts", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2(cv::polylines(img, pts, isClosed, color, thickness, lineType, shift));
        return pyopencv_from(img);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_pow(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    double power=0;
    PyObject* pyobj_dst = NULL;
    Mat dst;

    const char* keywords[] = { "src", "power", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Od|O:pow", (char**)keywords, &pyobj_src, &power, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::pow(src, power, dst));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    double power=0;
    PyObject* pyobj_dst = NULL;
    UMat dst;

    const char* keywords[] = { "src", "power", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Od|O:pow", (char**)keywords, &pyobj_src, &power, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::pow(src, power, dst));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_preCornerDetect(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int ksize=0;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "ksize", "dst", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|Oi:preCornerDetect", (char**)keywords, &pyobj_src, &ksize, &pyobj_dst, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::preCornerDetect(src, dst, ksize, borderType));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int ksize=0;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "ksize", "dst", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|Oi:preCornerDetect", (char**)keywords, &pyobj_src, &ksize, &pyobj_dst, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::preCornerDetect(src, dst, ksize, borderType));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_projectPoints(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_objectPoints = NULL;
    Mat objectPoints;
    PyObject* pyobj_rvec = NULL;
    Mat rvec;
    PyObject* pyobj_tvec = NULL;
    Mat tvec;
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    Mat distCoeffs;
    PyObject* pyobj_imagePoints = NULL;
    Mat imagePoints;
    PyObject* pyobj_jacobian = NULL;
    Mat jacobian;
    double aspectRatio=0;

    const char* keywords[] = { "objectPoints", "rvec", "tvec", "cameraMatrix", "distCoeffs", "imagePoints", "jacobian", "aspectRatio", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOO|OOd:projectPoints", (char**)keywords, &pyobj_objectPoints, &pyobj_rvec, &pyobj_tvec, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_imagePoints, &pyobj_jacobian, &aspectRatio) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_rvec, rvec, ArgInfo("rvec", 0)) &&
        pyopencv_to(pyobj_tvec, tvec, ArgInfo("tvec", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_imagePoints, imagePoints, ArgInfo("imagePoints", 1)) &&
        pyopencv_to(pyobj_jacobian, jacobian, ArgInfo("jacobian", 1)) )
    {
        ERRWRAP2(cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints, jacobian, aspectRatio));
        return Py_BuildValue("(NN)", pyopencv_from(imagePoints), pyopencv_from(jacobian));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_objectPoints = NULL;
    UMat objectPoints;
    PyObject* pyobj_rvec = NULL;
    UMat rvec;
    PyObject* pyobj_tvec = NULL;
    UMat tvec;
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    UMat distCoeffs;
    PyObject* pyobj_imagePoints = NULL;
    UMat imagePoints;
    PyObject* pyobj_jacobian = NULL;
    UMat jacobian;
    double aspectRatio=0;

    const char* keywords[] = { "objectPoints", "rvec", "tvec", "cameraMatrix", "distCoeffs", "imagePoints", "jacobian", "aspectRatio", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOO|OOd:projectPoints", (char**)keywords, &pyobj_objectPoints, &pyobj_rvec, &pyobj_tvec, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_imagePoints, &pyobj_jacobian, &aspectRatio) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_rvec, rvec, ArgInfo("rvec", 0)) &&
        pyopencv_to(pyobj_tvec, tvec, ArgInfo("tvec", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_imagePoints, imagePoints, ArgInfo("imagePoints", 1)) &&
        pyopencv_to(pyobj_jacobian, jacobian, ArgInfo("jacobian", 1)) )
    {
        ERRWRAP2(cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints, jacobian, aspectRatio));
        return Py_BuildValue("(NN)", pyopencv_from(imagePoints), pyopencv_from(jacobian));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_putText(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_text = NULL;
    String text;
    PyObject* pyobj_org = NULL;
    Point org;
    int fontFace=0;
    double fontScale=0;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int thickness=1;
    int lineType=LINE_8;
    bool bottomLeftOrigin=false;

    const char* keywords[] = { "img", "text", "org", "fontFace", "fontScale", "color", "thickness", "lineType", "bottomLeftOrigin", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOidO|iib:putText", (char**)keywords, &pyobj_img, &pyobj_text, &pyobj_org, &fontFace, &fontScale, &pyobj_color, &thickness, &lineType, &bottomLeftOrigin) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_text, text, ArgInfo("text", 0)) &&
        pyopencv_to(pyobj_org, org, ArgInfo("org", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2(cv::putText(img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin));
        return pyopencv_from(img);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_img = NULL;
    UMat img;
    PyObject* pyobj_text = NULL;
    String text;
    PyObject* pyobj_org = NULL;
    Point org;
    int fontFace=0;
    double fontScale=0;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int thickness=1;
    int lineType=LINE_8;
    bool bottomLeftOrigin=false;

    const char* keywords[] = { "img", "text", "org", "fontFace", "fontScale", "color", "thickness", "lineType", "bottomLeftOrigin", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOidO|iib:putText", (char**)keywords, &pyobj_img, &pyobj_text, &pyobj_org, &fontFace, &fontScale, &pyobj_color, &thickness, &lineType, &bottomLeftOrigin) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_text, text, ArgInfo("text", 0)) &&
        pyopencv_to(pyobj_org, org, ArgInfo("org", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2(cv::putText(img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin));
        return pyopencv_from(img);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_pyrDown(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_dstsize = NULL;
    Size dstsize;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "dst", "dstsize", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOi:pyrDown", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_dstsize, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_dstsize, dstsize, ArgInfo("dstsize", 0)) )
    {
        ERRWRAP2(cv::pyrDown(src, dst, dstsize, borderType));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_dstsize = NULL;
    Size dstsize;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "dst", "dstsize", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOi:pyrDown", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_dstsize, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_dstsize, dstsize, ArgInfo("dstsize", 0)) )
    {
        ERRWRAP2(cv::pyrDown(src, dst, dstsize, borderType));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_pyrMeanShiftFiltering(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    double sp=0;
    double sr=0;
    int maxLevel=1;
    PyObject* pyobj_termcrit = NULL;
    TermCriteria termcrit=TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS,5,1);

    const char* keywords[] = { "src", "sp", "sr", "dst", "maxLevel", "termcrit", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odd|OiO:pyrMeanShiftFiltering", (char**)keywords, &pyobj_src, &sp, &sr, &pyobj_dst, &maxLevel, &pyobj_termcrit) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_termcrit, termcrit, ArgInfo("termcrit", 0)) )
    {
        ERRWRAP2(cv::pyrMeanShiftFiltering(src, dst, sp, sr, maxLevel, termcrit));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    double sp=0;
    double sr=0;
    int maxLevel=1;
    PyObject* pyobj_termcrit = NULL;
    TermCriteria termcrit=TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS,5,1);

    const char* keywords[] = { "src", "sp", "sr", "dst", "maxLevel", "termcrit", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odd|OiO:pyrMeanShiftFiltering", (char**)keywords, &pyobj_src, &sp, &sr, &pyobj_dst, &maxLevel, &pyobj_termcrit) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_termcrit, termcrit, ArgInfo("termcrit", 0)) )
    {
        ERRWRAP2(cv::pyrMeanShiftFiltering(src, dst, sp, sr, maxLevel, termcrit));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_pyrUp(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_dstsize = NULL;
    Size dstsize;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "dst", "dstsize", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOi:pyrUp", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_dstsize, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_dstsize, dstsize, ArgInfo("dstsize", 0)) )
    {
        ERRWRAP2(cv::pyrUp(src, dst, dstsize, borderType));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_dstsize = NULL;
    Size dstsize;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "dst", "dstsize", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOi:pyrUp", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_dstsize, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_dstsize, dstsize, ArgInfo("dstsize", 0)) )
    {
        ERRWRAP2(cv::pyrUp(src, dst, dstsize, borderType));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_randShuffle(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_dst = NULL;
    Mat dst;
    double iterFactor=1.;

    const char* keywords[] = { "dst", "iterFactor", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|d:randShuffle", (char**)keywords, &pyobj_dst, &iterFactor) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::randShuffle(dst, iterFactor, 0));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_dst = NULL;
    UMat dst;
    double iterFactor=1.;

    const char* keywords[] = { "dst", "iterFactor", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|d:randShuffle", (char**)keywords, &pyobj_dst, &iterFactor) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::randShuffle(dst, iterFactor, 0));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_randn(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_mean = NULL;
    Mat mean;
    PyObject* pyobj_stddev = NULL;
    Mat stddev;

    const char* keywords[] = { "dst", "mean", "stddev", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:randn", (char**)keywords, &pyobj_dst, &pyobj_mean, &pyobj_stddev) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_mean, mean, ArgInfo("mean", 0)) &&
        pyopencv_to(pyobj_stddev, stddev, ArgInfo("stddev", 0)) )
    {
        ERRWRAP2(cv::randn(dst, mean, stddev));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_mean = NULL;
    UMat mean;
    PyObject* pyobj_stddev = NULL;
    UMat stddev;

    const char* keywords[] = { "dst", "mean", "stddev", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:randn", (char**)keywords, &pyobj_dst, &pyobj_mean, &pyobj_stddev) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_mean, mean, ArgInfo("mean", 0)) &&
        pyopencv_to(pyobj_stddev, stddev, ArgInfo("stddev", 0)) )
    {
        ERRWRAP2(cv::randn(dst, mean, stddev));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_randu(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_low = NULL;
    Mat low;
    PyObject* pyobj_high = NULL;
    Mat high;

    const char* keywords[] = { "dst", "low", "high", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:randu", (char**)keywords, &pyobj_dst, &pyobj_low, &pyobj_high) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_low, low, ArgInfo("low", 0)) &&
        pyopencv_to(pyobj_high, high, ArgInfo("high", 0)) )
    {
        ERRWRAP2(cv::randu(dst, low, high));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_low = NULL;
    UMat low;
    PyObject* pyobj_high = NULL;
    UMat high;

    const char* keywords[] = { "dst", "low", "high", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:randu", (char**)keywords, &pyobj_dst, &pyobj_low, &pyobj_high) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_low, low, ArgInfo("low", 0)) &&
        pyopencv_to(pyobj_high, high, ArgInfo("high", 0)) )
    {
        ERRWRAP2(cv::randu(dst, low, high));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_recoverPose(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_E = NULL;
    Mat E;
    PyObject* pyobj_points1 = NULL;
    Mat points1;
    PyObject* pyobj_points2 = NULL;
    Mat points2;
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_R = NULL;
    Mat R;
    PyObject* pyobj_t = NULL;
    Mat t;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    int retval;

    const char* keywords[] = { "E", "points1", "points2", "cameraMatrix", "R", "t", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OOO:recoverPose", (char**)keywords, &pyobj_E, &pyobj_points1, &pyobj_points2, &pyobj_cameraMatrix, &pyobj_R, &pyobj_t, &pyobj_mask) &&
        pyopencv_to(pyobj_E, E, ArgInfo("E", 0)) &&
        pyopencv_to(pyobj_points1, points1, ArgInfo("points1", 0)) &&
        pyopencv_to(pyobj_points2, points2, ArgInfo("points2", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 1)) &&
        pyopencv_to(pyobj_t, t, ArgInfo("t", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 1)) )
    {
        ERRWRAP2(retval = cv::recoverPose(E, points1, points2, cameraMatrix, R, t, mask));
        return Py_BuildValue("(NNNN)", pyopencv_from(retval), pyopencv_from(R), pyopencv_from(t), pyopencv_from(mask));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_E = NULL;
    UMat E;
    PyObject* pyobj_points1 = NULL;
    UMat points1;
    PyObject* pyobj_points2 = NULL;
    UMat points2;
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    PyObject* pyobj_R = NULL;
    UMat R;
    PyObject* pyobj_t = NULL;
    UMat t;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    int retval;

    const char* keywords[] = { "E", "points1", "points2", "cameraMatrix", "R", "t", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OOO:recoverPose", (char**)keywords, &pyobj_E, &pyobj_points1, &pyobj_points2, &pyobj_cameraMatrix, &pyobj_R, &pyobj_t, &pyobj_mask) &&
        pyopencv_to(pyobj_E, E, ArgInfo("E", 0)) &&
        pyopencv_to(pyobj_points1, points1, ArgInfo("points1", 0)) &&
        pyopencv_to(pyobj_points2, points2, ArgInfo("points2", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 1)) &&
        pyopencv_to(pyobj_t, t, ArgInfo("t", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 1)) )
    {
        ERRWRAP2(retval = cv::recoverPose(E, points1, points2, cameraMatrix, R, t, mask));
        return Py_BuildValue("(NNNN)", pyopencv_from(retval), pyopencv_from(R), pyopencv_from(t), pyopencv_from(mask));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_E = NULL;
    Mat E;
    PyObject* pyobj_points1 = NULL;
    Mat points1;
    PyObject* pyobj_points2 = NULL;
    Mat points2;
    PyObject* pyobj_R = NULL;
    Mat R;
    PyObject* pyobj_t = NULL;
    Mat t;
    double focal=1.0;
    PyObject* pyobj_pp = NULL;
    Point2d pp=Point2d(0, 0);
    PyObject* pyobj_mask = NULL;
    Mat mask;
    int retval;

    const char* keywords[] = { "E", "points1", "points2", "R", "t", "focal", "pp", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|OOdOO:recoverPose", (char**)keywords, &pyobj_E, &pyobj_points1, &pyobj_points2, &pyobj_R, &pyobj_t, &focal, &pyobj_pp, &pyobj_mask) &&
        pyopencv_to(pyobj_E, E, ArgInfo("E", 0)) &&
        pyopencv_to(pyobj_points1, points1, ArgInfo("points1", 0)) &&
        pyopencv_to(pyobj_points2, points2, ArgInfo("points2", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 1)) &&
        pyopencv_to(pyobj_t, t, ArgInfo("t", 1)) &&
        pyopencv_to(pyobj_pp, pp, ArgInfo("pp", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 1)) )
    {
        ERRWRAP2(retval = cv::recoverPose(E, points1, points2, R, t, focal, pp, mask));
        return Py_BuildValue("(NNNN)", pyopencv_from(retval), pyopencv_from(R), pyopencv_from(t), pyopencv_from(mask));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_E = NULL;
    UMat E;
    PyObject* pyobj_points1 = NULL;
    UMat points1;
    PyObject* pyobj_points2 = NULL;
    UMat points2;
    PyObject* pyobj_R = NULL;
    UMat R;
    PyObject* pyobj_t = NULL;
    UMat t;
    double focal=1.0;
    PyObject* pyobj_pp = NULL;
    Point2d pp=Point2d(0, 0);
    PyObject* pyobj_mask = NULL;
    UMat mask;
    int retval;

    const char* keywords[] = { "E", "points1", "points2", "R", "t", "focal", "pp", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|OOdOO:recoverPose", (char**)keywords, &pyobj_E, &pyobj_points1, &pyobj_points2, &pyobj_R, &pyobj_t, &focal, &pyobj_pp, &pyobj_mask) &&
        pyopencv_to(pyobj_E, E, ArgInfo("E", 0)) &&
        pyopencv_to(pyobj_points1, points1, ArgInfo("points1", 0)) &&
        pyopencv_to(pyobj_points2, points2, ArgInfo("points2", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 1)) &&
        pyopencv_to(pyobj_t, t, ArgInfo("t", 1)) &&
        pyopencv_to(pyobj_pp, pp, ArgInfo("pp", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 1)) )
    {
        ERRWRAP2(retval = cv::recoverPose(E, points1, points2, R, t, focal, pp, mask));
        return Py_BuildValue("(NNNN)", pyopencv_from(retval), pyopencv_from(R), pyopencv_from(t), pyopencv_from(mask));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_rectangle(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_pt1 = NULL;
    Point pt1;
    PyObject* pyobj_pt2 = NULL;
    Point pt2;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int thickness=1;
    int lineType=LINE_8;
    int shift=0;

    const char* keywords[] = { "img", "pt1", "pt2", "color", "thickness", "lineType", "shift", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|iii:rectangle", (char**)keywords, &pyobj_img, &pyobj_pt1, &pyobj_pt2, &pyobj_color, &thickness, &lineType, &shift) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_pt1, pt1, ArgInfo("pt1", 0)) &&
        pyopencv_to(pyobj_pt2, pt2, ArgInfo("pt2", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2(cv::rectangle(img, pt1, pt2, color, thickness, lineType, shift));
        return pyopencv_from(img);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_img = NULL;
    UMat img;
    PyObject* pyobj_pt1 = NULL;
    Point pt1;
    PyObject* pyobj_pt2 = NULL;
    Point pt2;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int thickness=1;
    int lineType=LINE_8;
    int shift=0;

    const char* keywords[] = { "img", "pt1", "pt2", "color", "thickness", "lineType", "shift", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|iii:rectangle", (char**)keywords, &pyobj_img, &pyobj_pt1, &pyobj_pt2, &pyobj_color, &thickness, &lineType, &shift) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_pt1, pt1, ArgInfo("pt1", 0)) &&
        pyopencv_to(pyobj_pt2, pt2, ArgInfo("pt2", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2(cv::rectangle(img, pt1, pt2, color, thickness, lineType, shift));
        return pyopencv_from(img);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_rectify3Collinear(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_cameraMatrix1 = NULL;
    Mat cameraMatrix1;
    PyObject* pyobj_distCoeffs1 = NULL;
    Mat distCoeffs1;
    PyObject* pyobj_cameraMatrix2 = NULL;
    Mat cameraMatrix2;
    PyObject* pyobj_distCoeffs2 = NULL;
    Mat distCoeffs2;
    PyObject* pyobj_cameraMatrix3 = NULL;
    Mat cameraMatrix3;
    PyObject* pyobj_distCoeffs3 = NULL;
    Mat distCoeffs3;
    PyObject* pyobj_imgpt1 = NULL;
    vector_Mat imgpt1;
    PyObject* pyobj_imgpt3 = NULL;
    vector_Mat imgpt3;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    PyObject* pyobj_R12 = NULL;
    Mat R12;
    PyObject* pyobj_T12 = NULL;
    Mat T12;
    PyObject* pyobj_R13 = NULL;
    Mat R13;
    PyObject* pyobj_T13 = NULL;
    Mat T13;
    PyObject* pyobj_R1 = NULL;
    Mat R1;
    PyObject* pyobj_R2 = NULL;
    Mat R2;
    PyObject* pyobj_R3 = NULL;
    Mat R3;
    PyObject* pyobj_P1 = NULL;
    Mat P1;
    PyObject* pyobj_P2 = NULL;
    Mat P2;
    PyObject* pyobj_P3 = NULL;
    Mat P3;
    PyObject* pyobj_Q = NULL;
    Mat Q;
    double alpha=0;
    PyObject* pyobj_newImgSize = NULL;
    Size newImgSize;
    Rect roi1;
    Rect roi2;
    int flags=0;
    float retval;

    const char* keywords[] = { "cameraMatrix1", "distCoeffs1", "cameraMatrix2", "distCoeffs2", "cameraMatrix3", "distCoeffs3", "imgpt1", "imgpt3", "imageSize", "R12", "T12", "R13", "T13", "alpha", "newImgSize", "flags", "R1", "R2", "R3", "P1", "P2", "P3", "Q", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOOOOOOOOdOi|OOOOOOO:rectify3Collinear", (char**)keywords, &pyobj_cameraMatrix1, &pyobj_distCoeffs1, &pyobj_cameraMatrix2, &pyobj_distCoeffs2, &pyobj_cameraMatrix3, &pyobj_distCoeffs3, &pyobj_imgpt1, &pyobj_imgpt3, &pyobj_imageSize, &pyobj_R12, &pyobj_T12, &pyobj_R13, &pyobj_T13, &alpha, &pyobj_newImgSize, &flags, &pyobj_R1, &pyobj_R2, &pyobj_R3, &pyobj_P1, &pyobj_P2, &pyobj_P3, &pyobj_Q) &&
        pyopencv_to(pyobj_cameraMatrix1, cameraMatrix1, ArgInfo("cameraMatrix1", 0)) &&
        pyopencv_to(pyobj_distCoeffs1, distCoeffs1, ArgInfo("distCoeffs1", 0)) &&
        pyopencv_to(pyobj_cameraMatrix2, cameraMatrix2, ArgInfo("cameraMatrix2", 0)) &&
        pyopencv_to(pyobj_distCoeffs2, distCoeffs2, ArgInfo("distCoeffs2", 0)) &&
        pyopencv_to(pyobj_cameraMatrix3, cameraMatrix3, ArgInfo("cameraMatrix3", 0)) &&
        pyopencv_to(pyobj_distCoeffs3, distCoeffs3, ArgInfo("distCoeffs3", 0)) &&
        pyopencv_to(pyobj_imgpt1, imgpt1, ArgInfo("imgpt1", 0)) &&
        pyopencv_to(pyobj_imgpt3, imgpt3, ArgInfo("imgpt3", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_R12, R12, ArgInfo("R12", 0)) &&
        pyopencv_to(pyobj_T12, T12, ArgInfo("T12", 0)) &&
        pyopencv_to(pyobj_R13, R13, ArgInfo("R13", 0)) &&
        pyopencv_to(pyobj_T13, T13, ArgInfo("T13", 0)) &&
        pyopencv_to(pyobj_R1, R1, ArgInfo("R1", 1)) &&
        pyopencv_to(pyobj_R2, R2, ArgInfo("R2", 1)) &&
        pyopencv_to(pyobj_R3, R3, ArgInfo("R3", 1)) &&
        pyopencv_to(pyobj_P1, P1, ArgInfo("P1", 1)) &&
        pyopencv_to(pyobj_P2, P2, ArgInfo("P2", 1)) &&
        pyopencv_to(pyobj_P3, P3, ArgInfo("P3", 1)) &&
        pyopencv_to(pyobj_Q, Q, ArgInfo("Q", 1)) &&
        pyopencv_to(pyobj_newImgSize, newImgSize, ArgInfo("newImgSize", 0)) )
    {
        ERRWRAP2(retval = cv::rectify3Collinear(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, cameraMatrix3, distCoeffs3, imgpt1, imgpt3, imageSize, R12, T12, R13, T13, R1, R2, R3, P1, P2, P3, Q, alpha, newImgSize, &roi1, &roi2, flags));
        return Py_BuildValue("(NNNNNNNNNN)", pyopencv_from(retval), pyopencv_from(R1), pyopencv_from(R2), pyopencv_from(R3), pyopencv_from(P1), pyopencv_from(P2), pyopencv_from(P3), pyopencv_from(Q), pyopencv_from(roi1), pyopencv_from(roi2));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_cameraMatrix1 = NULL;
    UMat cameraMatrix1;
    PyObject* pyobj_distCoeffs1 = NULL;
    UMat distCoeffs1;
    PyObject* pyobj_cameraMatrix2 = NULL;
    UMat cameraMatrix2;
    PyObject* pyobj_distCoeffs2 = NULL;
    UMat distCoeffs2;
    PyObject* pyobj_cameraMatrix3 = NULL;
    UMat cameraMatrix3;
    PyObject* pyobj_distCoeffs3 = NULL;
    UMat distCoeffs3;
    PyObject* pyobj_imgpt1 = NULL;
    vector_Mat imgpt1;
    PyObject* pyobj_imgpt3 = NULL;
    vector_Mat imgpt3;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    PyObject* pyobj_R12 = NULL;
    UMat R12;
    PyObject* pyobj_T12 = NULL;
    UMat T12;
    PyObject* pyobj_R13 = NULL;
    UMat R13;
    PyObject* pyobj_T13 = NULL;
    UMat T13;
    PyObject* pyobj_R1 = NULL;
    UMat R1;
    PyObject* pyobj_R2 = NULL;
    UMat R2;
    PyObject* pyobj_R3 = NULL;
    UMat R3;
    PyObject* pyobj_P1 = NULL;
    UMat P1;
    PyObject* pyobj_P2 = NULL;
    UMat P2;
    PyObject* pyobj_P3 = NULL;
    UMat P3;
    PyObject* pyobj_Q = NULL;
    UMat Q;
    double alpha=0;
    PyObject* pyobj_newImgSize = NULL;
    Size newImgSize;
    Rect roi1;
    Rect roi2;
    int flags=0;
    float retval;

    const char* keywords[] = { "cameraMatrix1", "distCoeffs1", "cameraMatrix2", "distCoeffs2", "cameraMatrix3", "distCoeffs3", "imgpt1", "imgpt3", "imageSize", "R12", "T12", "R13", "T13", "alpha", "newImgSize", "flags", "R1", "R2", "R3", "P1", "P2", "P3", "Q", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOOOOOOOOdOi|OOOOOOO:rectify3Collinear", (char**)keywords, &pyobj_cameraMatrix1, &pyobj_distCoeffs1, &pyobj_cameraMatrix2, &pyobj_distCoeffs2, &pyobj_cameraMatrix3, &pyobj_distCoeffs3, &pyobj_imgpt1, &pyobj_imgpt3, &pyobj_imageSize, &pyobj_R12, &pyobj_T12, &pyobj_R13, &pyobj_T13, &alpha, &pyobj_newImgSize, &flags, &pyobj_R1, &pyobj_R2, &pyobj_R3, &pyobj_P1, &pyobj_P2, &pyobj_P3, &pyobj_Q) &&
        pyopencv_to(pyobj_cameraMatrix1, cameraMatrix1, ArgInfo("cameraMatrix1", 0)) &&
        pyopencv_to(pyobj_distCoeffs1, distCoeffs1, ArgInfo("distCoeffs1", 0)) &&
        pyopencv_to(pyobj_cameraMatrix2, cameraMatrix2, ArgInfo("cameraMatrix2", 0)) &&
        pyopencv_to(pyobj_distCoeffs2, distCoeffs2, ArgInfo("distCoeffs2", 0)) &&
        pyopencv_to(pyobj_cameraMatrix3, cameraMatrix3, ArgInfo("cameraMatrix3", 0)) &&
        pyopencv_to(pyobj_distCoeffs3, distCoeffs3, ArgInfo("distCoeffs3", 0)) &&
        pyopencv_to(pyobj_imgpt1, imgpt1, ArgInfo("imgpt1", 0)) &&
        pyopencv_to(pyobj_imgpt3, imgpt3, ArgInfo("imgpt3", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_R12, R12, ArgInfo("R12", 0)) &&
        pyopencv_to(pyobj_T12, T12, ArgInfo("T12", 0)) &&
        pyopencv_to(pyobj_R13, R13, ArgInfo("R13", 0)) &&
        pyopencv_to(pyobj_T13, T13, ArgInfo("T13", 0)) &&
        pyopencv_to(pyobj_R1, R1, ArgInfo("R1", 1)) &&
        pyopencv_to(pyobj_R2, R2, ArgInfo("R2", 1)) &&
        pyopencv_to(pyobj_R3, R3, ArgInfo("R3", 1)) &&
        pyopencv_to(pyobj_P1, P1, ArgInfo("P1", 1)) &&
        pyopencv_to(pyobj_P2, P2, ArgInfo("P2", 1)) &&
        pyopencv_to(pyobj_P3, P3, ArgInfo("P3", 1)) &&
        pyopencv_to(pyobj_Q, Q, ArgInfo("Q", 1)) &&
        pyopencv_to(pyobj_newImgSize, newImgSize, ArgInfo("newImgSize", 0)) )
    {
        ERRWRAP2(retval = cv::rectify3Collinear(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, cameraMatrix3, distCoeffs3, imgpt1, imgpt3, imageSize, R12, T12, R13, T13, R1, R2, R3, P1, P2, P3, Q, alpha, newImgSize, &roi1, &roi2, flags));
        return Py_BuildValue("(NNNNNNNNNN)", pyopencv_from(retval), pyopencv_from(R1), pyopencv_from(R2), pyopencv_from(R3), pyopencv_from(P1), pyopencv_from(P2), pyopencv_from(P3), pyopencv_from(Q), pyopencv_from(roi1), pyopencv_from(roi2));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_reduce(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int dim=0;
    int rtype=0;
    int dtype=-1;

    const char* keywords[] = { "src", "dim", "rtype", "dst", "dtype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii|Oi:reduce", (char**)keywords, &pyobj_src, &dim, &rtype, &pyobj_dst, &dtype) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::reduce(src, dst, dim, rtype, dtype));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int dim=0;
    int rtype=0;
    int dtype=-1;

    const char* keywords[] = { "src", "dim", "rtype", "dst", "dtype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii|Oi:reduce", (char**)keywords, &pyobj_src, &dim, &rtype, &pyobj_dst, &dtype) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::reduce(src, dst, dim, rtype, dtype));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_remap(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_map1 = NULL;
    Mat map1;
    PyObject* pyobj_map2 = NULL;
    Mat map2;
    int interpolation=0;
    int borderMode=BORDER_CONSTANT;
    PyObject* pyobj_borderValue = NULL;
    Scalar borderValue;

    const char* keywords[] = { "src", "map1", "map2", "interpolation", "dst", "borderMode", "borderValue", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOi|OiO:remap", (char**)keywords, &pyobj_src, &pyobj_map1, &pyobj_map2, &interpolation, &pyobj_dst, &borderMode, &pyobj_borderValue) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_map1, map1, ArgInfo("map1", 0)) &&
        pyopencv_to(pyobj_map2, map2, ArgInfo("map2", 0)) &&
        pyopencv_to(pyobj_borderValue, borderValue, ArgInfo("borderValue", 0)) )
    {
        ERRWRAP2(cv::remap(src, dst, map1, map2, interpolation, borderMode, borderValue));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_map1 = NULL;
    UMat map1;
    PyObject* pyobj_map2 = NULL;
    UMat map2;
    int interpolation=0;
    int borderMode=BORDER_CONSTANT;
    PyObject* pyobj_borderValue = NULL;
    Scalar borderValue;

    const char* keywords[] = { "src", "map1", "map2", "interpolation", "dst", "borderMode", "borderValue", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOi|OiO:remap", (char**)keywords, &pyobj_src, &pyobj_map1, &pyobj_map2, &interpolation, &pyobj_dst, &borderMode, &pyobj_borderValue) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_map1, map1, ArgInfo("map1", 0)) &&
        pyopencv_to(pyobj_map2, map2, ArgInfo("map2", 0)) &&
        pyopencv_to(pyobj_borderValue, borderValue, ArgInfo("borderValue", 0)) )
    {
        ERRWRAP2(cv::remap(src, dst, map1, map2, interpolation, borderMode, borderValue));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_repeat(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    int ny=0;
    int nx=0;
    PyObject* pyobj_dst = NULL;
    Mat dst;

    const char* keywords[] = { "src", "ny", "nx", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii|O:repeat", (char**)keywords, &pyobj_src, &ny, &nx, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::repeat(src, ny, nx, dst));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    int ny=0;
    int nx=0;
    PyObject* pyobj_dst = NULL;
    UMat dst;

    const char* keywords[] = { "src", "ny", "nx", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii|O:repeat", (char**)keywords, &pyobj_src, &ny, &nx, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::repeat(src, ny, nx, dst));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_reprojectImageTo3D(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_disparity = NULL;
    Mat disparity;
    PyObject* pyobj__3dImage = NULL;
    Mat _3dImage;
    PyObject* pyobj_Q = NULL;
    Mat Q;
    bool handleMissingValues=false;
    int ddepth=-1;

    const char* keywords[] = { "disparity", "Q", "_3dImage", "handleMissingValues", "ddepth", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Obi:reprojectImageTo3D", (char**)keywords, &pyobj_disparity, &pyobj_Q, &pyobj__3dImage, &handleMissingValues, &ddepth) &&
        pyopencv_to(pyobj_disparity, disparity, ArgInfo("disparity", 0)) &&
        pyopencv_to(pyobj__3dImage, _3dImage, ArgInfo("_3dImage", 1)) &&
        pyopencv_to(pyobj_Q, Q, ArgInfo("Q", 0)) )
    {
        ERRWRAP2(cv::reprojectImageTo3D(disparity, _3dImage, Q, handleMissingValues, ddepth));
        return pyopencv_from(_3dImage);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_disparity = NULL;
    UMat disparity;
    PyObject* pyobj__3dImage = NULL;
    UMat _3dImage;
    PyObject* pyobj_Q = NULL;
    UMat Q;
    bool handleMissingValues=false;
    int ddepth=-1;

    const char* keywords[] = { "disparity", "Q", "_3dImage", "handleMissingValues", "ddepth", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Obi:reprojectImageTo3D", (char**)keywords, &pyobj_disparity, &pyobj_Q, &pyobj__3dImage, &handleMissingValues, &ddepth) &&
        pyopencv_to(pyobj_disparity, disparity, ArgInfo("disparity", 0)) &&
        pyopencv_to(pyobj__3dImage, _3dImage, ArgInfo("_3dImage", 1)) &&
        pyopencv_to(pyobj_Q, Q, ArgInfo("Q", 0)) )
    {
        ERRWRAP2(cv::reprojectImageTo3D(disparity, _3dImage, Q, handleMissingValues, ddepth));
        return pyopencv_from(_3dImage);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_resize(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_dsize = NULL;
    Size dsize;
    double fx=0;
    double fy=0;
    int interpolation=INTER_LINEAR;

    const char* keywords[] = { "src", "dsize", "dst", "fx", "fy", "interpolation", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Oddi:resize", (char**)keywords, &pyobj_src, &pyobj_dsize, &pyobj_dst, &fx, &fy, &interpolation) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_dsize, dsize, ArgInfo("dsize", 0)) )
    {
        ERRWRAP2(cv::resize(src, dst, dsize, fx, fy, interpolation));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_dsize = NULL;
    Size dsize;
    double fx=0;
    double fy=0;
    int interpolation=INTER_LINEAR;

    const char* keywords[] = { "src", "dsize", "dst", "fx", "fy", "interpolation", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Oddi:resize", (char**)keywords, &pyobj_src, &pyobj_dsize, &pyobj_dst, &fx, &fy, &interpolation) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_dsize, dsize, ArgInfo("dsize", 0)) )
    {
        ERRWRAP2(cv::resize(src, dst, dsize, fx, fy, interpolation));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_resizeWindow(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_winname = NULL;
    String winname;
    int width=0;
    int height=0;

    const char* keywords[] = { "winname", "width", "height", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii:resizeWindow", (char**)keywords, &pyobj_winname, &width, &height) &&
        pyopencv_to(pyobj_winname, winname, ArgInfo("winname", 0)) )
    {
        ERRWRAP2(cv::resizeWindow(winname, width, height));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_rotate(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int rotateCode=0;

    const char* keywords[] = { "src", "rotateCode", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|O:rotate", (char**)keywords, &pyobj_src, &rotateCode, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::rotate(src, dst, rotateCode));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int rotateCode=0;

    const char* keywords[] = { "src", "rotateCode", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|O:rotate", (char**)keywords, &pyobj_src, &rotateCode, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::rotate(src, dst, rotateCode));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_rotatedRectangleIntersection(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_rect1 = NULL;
    RotatedRect rect1;
    PyObject* pyobj_rect2 = NULL;
    RotatedRect rect2;
    PyObject* pyobj_intersectingRegion = NULL;
    Mat intersectingRegion;
    int retval;

    const char* keywords[] = { "rect1", "rect2", "intersectingRegion", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:rotatedRectangleIntersection", (char**)keywords, &pyobj_rect1, &pyobj_rect2, &pyobj_intersectingRegion) &&
        pyopencv_to(pyobj_rect1, rect1, ArgInfo("rect1", 0)) &&
        pyopencv_to(pyobj_rect2, rect2, ArgInfo("rect2", 0)) &&
        pyopencv_to(pyobj_intersectingRegion, intersectingRegion, ArgInfo("intersectingRegion", 1)) )
    {
        ERRWRAP2(retval = cv::rotatedRectangleIntersection(rect1, rect2, intersectingRegion));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(intersectingRegion));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_rect1 = NULL;
    RotatedRect rect1;
    PyObject* pyobj_rect2 = NULL;
    RotatedRect rect2;
    PyObject* pyobj_intersectingRegion = NULL;
    UMat intersectingRegion;
    int retval;

    const char* keywords[] = { "rect1", "rect2", "intersectingRegion", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:rotatedRectangleIntersection", (char**)keywords, &pyobj_rect1, &pyobj_rect2, &pyobj_intersectingRegion) &&
        pyopencv_to(pyobj_rect1, rect1, ArgInfo("rect1", 0)) &&
        pyopencv_to(pyobj_rect2, rect2, ArgInfo("rect2", 0)) &&
        pyopencv_to(pyobj_intersectingRegion, intersectingRegion, ArgInfo("intersectingRegion", 1)) )
    {
        ERRWRAP2(retval = cv::rotatedRectangleIntersection(rect1, rect2, intersectingRegion));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(intersectingRegion));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_sampsonDistance(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_pt1 = NULL;
    Mat pt1;
    PyObject* pyobj_pt2 = NULL;
    Mat pt2;
    PyObject* pyobj_F = NULL;
    Mat F;
    double retval;

    const char* keywords[] = { "pt1", "pt2", "F", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:sampsonDistance", (char**)keywords, &pyobj_pt1, &pyobj_pt2, &pyobj_F) &&
        pyopencv_to(pyobj_pt1, pt1, ArgInfo("pt1", 0)) &&
        pyopencv_to(pyobj_pt2, pt2, ArgInfo("pt2", 0)) &&
        pyopencv_to(pyobj_F, F, ArgInfo("F", 0)) )
    {
        ERRWRAP2(retval = cv::sampsonDistance(pt1, pt2, F));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_pt1 = NULL;
    UMat pt1;
    PyObject* pyobj_pt2 = NULL;
    UMat pt2;
    PyObject* pyobj_F = NULL;
    UMat F;
    double retval;

    const char* keywords[] = { "pt1", "pt2", "F", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:sampsonDistance", (char**)keywords, &pyobj_pt1, &pyobj_pt2, &pyobj_F) &&
        pyopencv_to(pyobj_pt1, pt1, ArgInfo("pt1", 0)) &&
        pyopencv_to(pyobj_pt2, pt2, ArgInfo("pt2", 0)) &&
        pyopencv_to(pyobj_F, F, ArgInfo("F", 0)) )
    {
        ERRWRAP2(retval = cv::sampsonDistance(pt1, pt2, F));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_scaleAdd(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    double alpha=0;
    PyObject* pyobj_src2 = NULL;
    Mat src2;
    PyObject* pyobj_dst = NULL;
    Mat dst;

    const char* keywords[] = { "src1", "alpha", "src2", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OdO|O:scaleAdd", (char**)keywords, &pyobj_src1, &alpha, &pyobj_src2, &pyobj_dst) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::scaleAdd(src1, alpha, src2, dst));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src1 = NULL;
    UMat src1;
    double alpha=0;
    PyObject* pyobj_src2 = NULL;
    UMat src2;
    PyObject* pyobj_dst = NULL;
    UMat dst;

    const char* keywords[] = { "src1", "alpha", "src2", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OdO|O:scaleAdd", (char**)keywords, &pyobj_src1, &alpha, &pyobj_src2, &pyobj_dst) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::scaleAdd(src1, alpha, src2, dst));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_seamlessClone(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    PyObject* pyobj_p = NULL;
    Point p;
    PyObject* pyobj_blend = NULL;
    Mat blend;
    int flags=0;

    const char* keywords[] = { "src", "dst", "mask", "p", "flags", "blend", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOi|O:seamlessClone", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_mask, &pyobj_p, &flags, &pyobj_blend) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) &&
        pyopencv_to(pyobj_p, p, ArgInfo("p", 0)) &&
        pyopencv_to(pyobj_blend, blend, ArgInfo("blend", 1)) )
    {
        ERRWRAP2(cv::seamlessClone(src, dst, mask, p, blend, flags));
        return pyopencv_from(blend);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    PyObject* pyobj_p = NULL;
    Point p;
    PyObject* pyobj_blend = NULL;
    UMat blend;
    int flags=0;

    const char* keywords[] = { "src", "dst", "mask", "p", "flags", "blend", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOi|O:seamlessClone", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_mask, &pyobj_p, &flags, &pyobj_blend) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) &&
        pyopencv_to(pyobj_p, p, ArgInfo("p", 0)) &&
        pyopencv_to(pyobj_blend, blend, ArgInfo("blend", 1)) )
    {
        ERRWRAP2(cv::seamlessClone(src, dst, mask, p, blend, flags));
        return pyopencv_from(blend);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_selectROI(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_img = NULL;
    Mat img;
    bool fromCenter=true;
    Rect2d retval;

    const char* keywords[] = { "img", "fromCenter", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|b:selectROI", (char**)keywords, &pyobj_img, &fromCenter) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) )
    {
        ERRWRAP2(retval = cv::selectROI(img, fromCenter));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_img = NULL;
    Mat img;
    bool fromCenter=true;
    Rect2d retval;

    const char* keywords[] = { "img", "fromCenter", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|b:selectROI", (char**)keywords, &pyobj_img, &fromCenter) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) )
    {
        ERRWRAP2(retval = cv::selectROI(img, fromCenter));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_windowName = NULL;
    String windowName;
    PyObject* pyobj_img = NULL;
    Mat img;
    bool showCrossair=true;
    bool fromCenter=true;
    Rect2d retval;

    const char* keywords[] = { "windowName", "img", "showCrossair", "fromCenter", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|bb:selectROI", (char**)keywords, &pyobj_windowName, &pyobj_img, &showCrossair, &fromCenter) &&
        pyopencv_to(pyobj_windowName, windowName, ArgInfo("windowName", 0)) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) )
    {
        ERRWRAP2(retval = cv::selectROI(windowName, img, showCrossair, fromCenter));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_windowName = NULL;
    String windowName;
    PyObject* pyobj_img = NULL;
    Mat img;
    bool showCrossair=true;
    bool fromCenter=true;
    Rect2d retval;

    const char* keywords[] = { "windowName", "img", "showCrossair", "fromCenter", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|bb:selectROI", (char**)keywords, &pyobj_windowName, &pyobj_img, &showCrossair, &fromCenter) &&
        pyopencv_to(pyobj_windowName, windowName, ArgInfo("windowName", 0)) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) )
    {
        ERRWRAP2(retval = cv::selectROI(windowName, img, showCrossair, fromCenter));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_windowName = NULL;
    String windowName;
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_boundingBox = NULL;
    vector_Rect2d boundingBox;
    bool fromCenter=true;

    const char* keywords[] = { "windowName", "img", "boundingBox", "fromCenter", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|b:selectROI", (char**)keywords, &pyobj_windowName, &pyobj_img, &pyobj_boundingBox, &fromCenter) &&
        pyopencv_to(pyobj_windowName, windowName, ArgInfo("windowName", 0)) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_boundingBox, boundingBox, ArgInfo("boundingBox", 0)) )
    {
        ERRWRAP2(cv::selectROI(windowName, img, boundingBox, fromCenter));
        Py_RETURN_NONE;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_windowName = NULL;
    String windowName;
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_boundingBox = NULL;
    vector_Rect2d boundingBox;
    bool fromCenter=true;

    const char* keywords[] = { "windowName", "img", "boundingBox", "fromCenter", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|b:selectROI", (char**)keywords, &pyobj_windowName, &pyobj_img, &pyobj_boundingBox, &fromCenter) &&
        pyopencv_to(pyobj_windowName, windowName, ArgInfo("windowName", 0)) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_boundingBox, boundingBox, ArgInfo("boundingBox", 0)) )
    {
        ERRWRAP2(cv::selectROI(windowName, img, boundingBox, fromCenter));
        Py_RETURN_NONE;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_sepFilter2D(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int ddepth=0;
    PyObject* pyobj_kernelX = NULL;
    Mat kernelX;
    PyObject* pyobj_kernelY = NULL;
    Mat kernelY;
    PyObject* pyobj_anchor = NULL;
    Point anchor=Point(-1,-1);
    double delta=0;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "ddepth", "kernelX", "kernelY", "dst", "anchor", "delta", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OiOO|OOdi:sepFilter2D", (char**)keywords, &pyobj_src, &ddepth, &pyobj_kernelX, &pyobj_kernelY, &pyobj_dst, &pyobj_anchor, &delta, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_kernelX, kernelX, ArgInfo("kernelX", 0)) &&
        pyopencv_to(pyobj_kernelY, kernelY, ArgInfo("kernelY", 0)) &&
        pyopencv_to(pyobj_anchor, anchor, ArgInfo("anchor", 0)) )
    {
        ERRWRAP2(cv::sepFilter2D(src, dst, ddepth, kernelX, kernelY, anchor, delta, borderType));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int ddepth=0;
    PyObject* pyobj_kernelX = NULL;
    UMat kernelX;
    PyObject* pyobj_kernelY = NULL;
    UMat kernelY;
    PyObject* pyobj_anchor = NULL;
    Point anchor=Point(-1,-1);
    double delta=0;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "ddepth", "kernelX", "kernelY", "dst", "anchor", "delta", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OiOO|OOdi:sepFilter2D", (char**)keywords, &pyobj_src, &ddepth, &pyobj_kernelX, &pyobj_kernelY, &pyobj_dst, &pyobj_anchor, &delta, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_kernelX, kernelX, ArgInfo("kernelX", 0)) &&
        pyopencv_to(pyobj_kernelY, kernelY, ArgInfo("kernelY", 0)) &&
        pyopencv_to(pyobj_anchor, anchor, ArgInfo("anchor", 0)) )
    {
        ERRWRAP2(cv::sepFilter2D(src, dst, ddepth, kernelX, kernelY, anchor, delta, borderType));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_setIdentity(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_mtx = NULL;
    Mat mtx;
    PyObject* pyobj_s = NULL;
    Scalar s=Scalar(1);

    const char* keywords[] = { "mtx", "s", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:setIdentity", (char**)keywords, &pyobj_mtx, &pyobj_s) &&
        pyopencv_to(pyobj_mtx, mtx, ArgInfo("mtx", 1)) &&
        pyopencv_to(pyobj_s, s, ArgInfo("s", 0)) )
    {
        ERRWRAP2(cv::setIdentity(mtx, s));
        return pyopencv_from(mtx);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_mtx = NULL;
    UMat mtx;
    PyObject* pyobj_s = NULL;
    Scalar s=Scalar(1);

    const char* keywords[] = { "mtx", "s", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:setIdentity", (char**)keywords, &pyobj_mtx, &pyobj_s) &&
        pyopencv_to(pyobj_mtx, mtx, ArgInfo("mtx", 1)) &&
        pyopencv_to(pyobj_s, s, ArgInfo("s", 0)) )
    {
        ERRWRAP2(cv::setIdentity(mtx, s));
        return pyopencv_from(mtx);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_setNumThreads(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int nthreads=0;

    const char* keywords[] = { "nthreads", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:setNumThreads", (char**)keywords, &nthreads) )
    {
        ERRWRAP2(cv::setNumThreads(nthreads));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_setRNGSeed(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int seed=0;

    const char* keywords[] = { "seed", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:setRNGSeed", (char**)keywords, &seed) )
    {
        ERRWRAP2(cv::setRNGSeed(seed));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_setTrackbarMax(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_trackbarname = NULL;
    String trackbarname;
    PyObject* pyobj_winname = NULL;
    String winname;
    int maxval=0;

    const char* keywords[] = { "trackbarname", "winname", "maxval", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi:setTrackbarMax", (char**)keywords, &pyobj_trackbarname, &pyobj_winname, &maxval) &&
        pyopencv_to(pyobj_trackbarname, trackbarname, ArgInfo("trackbarname", 0)) &&
        pyopencv_to(pyobj_winname, winname, ArgInfo("winname", 0)) )
    {
        ERRWRAP2(cv::setTrackbarMax(trackbarname, winname, maxval));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_setTrackbarMin(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_trackbarname = NULL;
    String trackbarname;
    PyObject* pyobj_winname = NULL;
    String winname;
    int minval=0;

    const char* keywords[] = { "trackbarname", "winname", "minval", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi:setTrackbarMin", (char**)keywords, &pyobj_trackbarname, &pyobj_winname, &minval) &&
        pyopencv_to(pyobj_trackbarname, trackbarname, ArgInfo("trackbarname", 0)) &&
        pyopencv_to(pyobj_winname, winname, ArgInfo("winname", 0)) )
    {
        ERRWRAP2(cv::setTrackbarMin(trackbarname, winname, minval));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_setTrackbarPos(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_trackbarname = NULL;
    String trackbarname;
    PyObject* pyobj_winname = NULL;
    String winname;
    int pos=0;

    const char* keywords[] = { "trackbarname", "winname", "pos", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi:setTrackbarPos", (char**)keywords, &pyobj_trackbarname, &pyobj_winname, &pos) &&
        pyopencv_to(pyobj_trackbarname, trackbarname, ArgInfo("trackbarname", 0)) &&
        pyopencv_to(pyobj_winname, winname, ArgInfo("winname", 0)) )
    {
        ERRWRAP2(cv::setTrackbarPos(trackbarname, winname, pos));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_setUseOpenVX(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    bool flag=0;

    const char* keywords[] = { "flag", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:setUseOpenVX", (char**)keywords, &flag) )
    {
        ERRWRAP2(cv::setUseOpenVX(flag));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_setUseOptimized(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    bool onoff=0;

    const char* keywords[] = { "onoff", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:setUseOptimized", (char**)keywords, &onoff) )
    {
        ERRWRAP2(cv::setUseOptimized(onoff));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_setWindowProperty(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_winname = NULL;
    String winname;
    int prop_id=0;
    double prop_value=0;

    const char* keywords[] = { "winname", "prop_id", "prop_value", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oid:setWindowProperty", (char**)keywords, &pyobj_winname, &prop_id, &prop_value) &&
        pyopencv_to(pyobj_winname, winname, ArgInfo("winname", 0)) )
    {
        ERRWRAP2(cv::setWindowProperty(winname, prop_id, prop_value));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_setWindowTitle(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    PyObject* pyobj_winname = NULL;
    String winname;
    PyObject* pyobj_title = NULL;
    String title;

    const char* keywords[] = { "winname", "title", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:setWindowTitle", (char**)keywords, &pyobj_winname, &pyobj_title) &&
        pyopencv_to(pyobj_winname, winname, ArgInfo("winname", 0)) &&
        pyopencv_to(pyobj_title, title, ArgInfo("title", 0)) )
    {
        ERRWRAP2(cv::setWindowTitle(winname, title));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_solve(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    PyObject* pyobj_src2 = NULL;
    Mat src2;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int flags=DECOMP_LU;
    bool retval;

    const char* keywords[] = { "src1", "src2", "dst", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Oi:solve", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &flags) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(retval = cv::solve(src1, src2, dst, flags));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(dst));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src1 = NULL;
    UMat src1;
    PyObject* pyobj_src2 = NULL;
    UMat src2;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int flags=DECOMP_LU;
    bool retval;

    const char* keywords[] = { "src1", "src2", "dst", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Oi:solve", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &flags) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(retval = cv::solve(src1, src2, dst, flags));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(dst));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_solveCubic(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_coeffs = NULL;
    Mat coeffs;
    PyObject* pyobj_roots = NULL;
    Mat roots;
    int retval;

    const char* keywords[] = { "coeffs", "roots", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:solveCubic", (char**)keywords, &pyobj_coeffs, &pyobj_roots) &&
        pyopencv_to(pyobj_coeffs, coeffs, ArgInfo("coeffs", 0)) &&
        pyopencv_to(pyobj_roots, roots, ArgInfo("roots", 1)) )
    {
        ERRWRAP2(retval = cv::solveCubic(coeffs, roots));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(roots));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_coeffs = NULL;
    UMat coeffs;
    PyObject* pyobj_roots = NULL;
    UMat roots;
    int retval;

    const char* keywords[] = { "coeffs", "roots", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:solveCubic", (char**)keywords, &pyobj_coeffs, &pyobj_roots) &&
        pyopencv_to(pyobj_coeffs, coeffs, ArgInfo("coeffs", 0)) &&
        pyopencv_to(pyobj_roots, roots, ArgInfo("roots", 1)) )
    {
        ERRWRAP2(retval = cv::solveCubic(coeffs, roots));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(roots));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_solveLP(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_Func = NULL;
    Mat Func;
    PyObject* pyobj_Constr = NULL;
    Mat Constr;
    PyObject* pyobj_z = NULL;
    Mat z;
    int retval;

    const char* keywords[] = { "Func", "Constr", "z", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:solveLP", (char**)keywords, &pyobj_Func, &pyobj_Constr, &pyobj_z) &&
        pyopencv_to(pyobj_Func, Func, ArgInfo("Func", 0)) &&
        pyopencv_to(pyobj_Constr, Constr, ArgInfo("Constr", 0)) &&
        pyopencv_to(pyobj_z, z, ArgInfo("z", 0)) )
    {
        ERRWRAP2(retval = cv::solveLP(Func, Constr, z));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_Func = NULL;
    Mat Func;
    PyObject* pyobj_Constr = NULL;
    Mat Constr;
    PyObject* pyobj_z = NULL;
    Mat z;
    int retval;

    const char* keywords[] = { "Func", "Constr", "z", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:solveLP", (char**)keywords, &pyobj_Func, &pyobj_Constr, &pyobj_z) &&
        pyopencv_to(pyobj_Func, Func, ArgInfo("Func", 0)) &&
        pyopencv_to(pyobj_Constr, Constr, ArgInfo("Constr", 0)) &&
        pyopencv_to(pyobj_z, z, ArgInfo("z", 0)) )
    {
        ERRWRAP2(retval = cv::solveLP(Func, Constr, z));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_solvePnP(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_objectPoints = NULL;
    Mat objectPoints;
    PyObject* pyobj_imagePoints = NULL;
    Mat imagePoints;
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    Mat distCoeffs;
    PyObject* pyobj_rvec = NULL;
    Mat rvec;
    PyObject* pyobj_tvec = NULL;
    Mat tvec;
    bool useExtrinsicGuess=false;
    int flags=SOLVEPNP_ITERATIVE;
    bool retval;

    const char* keywords[] = { "objectPoints", "imagePoints", "cameraMatrix", "distCoeffs", "rvec", "tvec", "useExtrinsicGuess", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OObi:solvePnP", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvec, &pyobj_tvec, &useExtrinsicGuess, &flags) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints, imagePoints, ArgInfo("imagePoints", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_rvec, rvec, ArgInfo("rvec", 1)) &&
        pyopencv_to(pyobj_tvec, tvec, ArgInfo("tvec", 1)) )
    {
        ERRWRAP2(retval = cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess, flags));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(rvec), pyopencv_from(tvec));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_objectPoints = NULL;
    UMat objectPoints;
    PyObject* pyobj_imagePoints = NULL;
    UMat imagePoints;
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    UMat distCoeffs;
    PyObject* pyobj_rvec = NULL;
    UMat rvec;
    PyObject* pyobj_tvec = NULL;
    UMat tvec;
    bool useExtrinsicGuess=false;
    int flags=SOLVEPNP_ITERATIVE;
    bool retval;

    const char* keywords[] = { "objectPoints", "imagePoints", "cameraMatrix", "distCoeffs", "rvec", "tvec", "useExtrinsicGuess", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OObi:solvePnP", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvec, &pyobj_tvec, &useExtrinsicGuess, &flags) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints, imagePoints, ArgInfo("imagePoints", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_rvec, rvec, ArgInfo("rvec", 1)) &&
        pyopencv_to(pyobj_tvec, tvec, ArgInfo("tvec", 1)) )
    {
        ERRWRAP2(retval = cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess, flags));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(rvec), pyopencv_from(tvec));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_solvePnPRansac(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_objectPoints = NULL;
    Mat objectPoints;
    PyObject* pyobj_imagePoints = NULL;
    Mat imagePoints;
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    Mat distCoeffs;
    PyObject* pyobj_rvec = NULL;
    Mat rvec;
    PyObject* pyobj_tvec = NULL;
    Mat tvec;
    bool useExtrinsicGuess=false;
    int iterationsCount=100;
    float reprojectionError=8.0;
    double confidence=0.99;
    PyObject* pyobj_inliers = NULL;
    Mat inliers;
    int flags=SOLVEPNP_ITERATIVE;
    bool retval;

    const char* keywords[] = { "objectPoints", "imagePoints", "cameraMatrix", "distCoeffs", "rvec", "tvec", "useExtrinsicGuess", "iterationsCount", "reprojectionError", "confidence", "inliers", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OObifdOi:solvePnPRansac", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvec, &pyobj_tvec, &useExtrinsicGuess, &iterationsCount, &reprojectionError, &confidence, &pyobj_inliers, &flags) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints, imagePoints, ArgInfo("imagePoints", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_rvec, rvec, ArgInfo("rvec", 1)) &&
        pyopencv_to(pyobj_tvec, tvec, ArgInfo("tvec", 1)) &&
        pyopencv_to(pyobj_inliers, inliers, ArgInfo("inliers", 1)) )
    {
        ERRWRAP2(retval = cv::solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess, iterationsCount, reprojectionError, confidence, inliers, flags));
        return Py_BuildValue("(NNNN)", pyopencv_from(retval), pyopencv_from(rvec), pyopencv_from(tvec), pyopencv_from(inliers));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_objectPoints = NULL;
    UMat objectPoints;
    PyObject* pyobj_imagePoints = NULL;
    UMat imagePoints;
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    UMat distCoeffs;
    PyObject* pyobj_rvec = NULL;
    UMat rvec;
    PyObject* pyobj_tvec = NULL;
    UMat tvec;
    bool useExtrinsicGuess=false;
    int iterationsCount=100;
    float reprojectionError=8.0;
    double confidence=0.99;
    PyObject* pyobj_inliers = NULL;
    UMat inliers;
    int flags=SOLVEPNP_ITERATIVE;
    bool retval;

    const char* keywords[] = { "objectPoints", "imagePoints", "cameraMatrix", "distCoeffs", "rvec", "tvec", "useExtrinsicGuess", "iterationsCount", "reprojectionError", "confidence", "inliers", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OObifdOi:solvePnPRansac", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvec, &pyobj_tvec, &useExtrinsicGuess, &iterationsCount, &reprojectionError, &confidence, &pyobj_inliers, &flags) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints, imagePoints, ArgInfo("imagePoints", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_rvec, rvec, ArgInfo("rvec", 1)) &&
        pyopencv_to(pyobj_tvec, tvec, ArgInfo("tvec", 1)) &&
        pyopencv_to(pyobj_inliers, inliers, ArgInfo("inliers", 1)) )
    {
        ERRWRAP2(retval = cv::solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess, iterationsCount, reprojectionError, confidence, inliers, flags));
        return Py_BuildValue("(NNNN)", pyopencv_from(retval), pyopencv_from(rvec), pyopencv_from(tvec), pyopencv_from(inliers));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_solvePoly(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_coeffs = NULL;
    Mat coeffs;
    PyObject* pyobj_roots = NULL;
    Mat roots;
    int maxIters=300;
    double retval;

    const char* keywords[] = { "coeffs", "roots", "maxIters", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oi:solvePoly", (char**)keywords, &pyobj_coeffs, &pyobj_roots, &maxIters) &&
        pyopencv_to(pyobj_coeffs, coeffs, ArgInfo("coeffs", 0)) &&
        pyopencv_to(pyobj_roots, roots, ArgInfo("roots", 1)) )
    {
        ERRWRAP2(retval = cv::solvePoly(coeffs, roots, maxIters));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(roots));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_coeffs = NULL;
    UMat coeffs;
    PyObject* pyobj_roots = NULL;
    UMat roots;
    int maxIters=300;
    double retval;

    const char* keywords[] = { "coeffs", "roots", "maxIters", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oi:solvePoly", (char**)keywords, &pyobj_coeffs, &pyobj_roots, &maxIters) &&
        pyopencv_to(pyobj_coeffs, coeffs, ArgInfo("coeffs", 0)) &&
        pyopencv_to(pyobj_roots, roots, ArgInfo("roots", 1)) )
    {
        ERRWRAP2(retval = cv::solvePoly(coeffs, roots, maxIters));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(roots));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_sort(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int flags=0;

    const char* keywords[] = { "src", "flags", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|O:sort", (char**)keywords, &pyobj_src, &flags, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::sort(src, dst, flags));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int flags=0;

    const char* keywords[] = { "src", "flags", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|O:sort", (char**)keywords, &pyobj_src, &flags, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::sort(src, dst, flags));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_sortIdx(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int flags=0;

    const char* keywords[] = { "src", "flags", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|O:sortIdx", (char**)keywords, &pyobj_src, &flags, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::sortIdx(src, dst, flags));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int flags=0;

    const char* keywords[] = { "src", "flags", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|O:sortIdx", (char**)keywords, &pyobj_src, &flags, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::sortIdx(src, dst, flags));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_spatialGradient(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dx = NULL;
    Mat dx;
    PyObject* pyobj_dy = NULL;
    Mat dy;
    int ksize=3;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "dx", "dy", "ksize", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOii:spatialGradient", (char**)keywords, &pyobj_src, &pyobj_dx, &pyobj_dy, &ksize, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dx, dx, ArgInfo("dx", 1)) &&
        pyopencv_to(pyobj_dy, dy, ArgInfo("dy", 1)) )
    {
        ERRWRAP2(cv::spatialGradient(src, dx, dy, ksize, borderType));
        return Py_BuildValue("(NN)", pyopencv_from(dx), pyopencv_from(dy));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dx = NULL;
    UMat dx;
    PyObject* pyobj_dy = NULL;
    UMat dy;
    int ksize=3;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "dx", "dy", "ksize", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOii:spatialGradient", (char**)keywords, &pyobj_src, &pyobj_dx, &pyobj_dy, &ksize, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dx, dx, ArgInfo("dx", 1)) &&
        pyopencv_to(pyobj_dy, dy, ArgInfo("dy", 1)) )
    {
        ERRWRAP2(cv::spatialGradient(src, dx, dy, ksize, borderType));
        return Py_BuildValue("(NN)", pyopencv_from(dx), pyopencv_from(dy));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_split(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_m = NULL;
    Mat m;
    PyObject* pyobj_mv = NULL;
    vector_Mat mv;

    const char* keywords[] = { "m", "mv", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:split", (char**)keywords, &pyobj_m, &pyobj_mv) &&
        pyopencv_to(pyobj_m, m, ArgInfo("m", 0)) &&
        pyopencv_to(pyobj_mv, mv, ArgInfo("mv", 1)) )
    {
        ERRWRAP2(cv::split(m, mv));
        return pyopencv_from(mv);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_m = NULL;
    UMat m;
    PyObject* pyobj_mv = NULL;
    vector_Mat mv;

    const char* keywords[] = { "m", "mv", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:split", (char**)keywords, &pyobj_m, &pyobj_mv) &&
        pyopencv_to(pyobj_m, m, ArgInfo("m", 0)) &&
        pyopencv_to(pyobj_mv, mv, ArgInfo("mv", 1)) )
    {
        ERRWRAP2(cv::split(m, mv));
        return pyopencv_from(mv);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_sqrBoxFilter(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj__src = NULL;
    Mat _src;
    PyObject* pyobj__dst = NULL;
    Mat _dst;
    int ddepth=0;
    PyObject* pyobj_ksize = NULL;
    Size ksize;
    PyObject* pyobj_anchor = NULL;
    Point anchor=Point(-1, -1);
    bool normalize=true;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "_src", "ddepth", "ksize", "_dst", "anchor", "normalize", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OiO|OObi:sqrBoxFilter", (char**)keywords, &pyobj__src, &ddepth, &pyobj_ksize, &pyobj__dst, &pyobj_anchor, &normalize, &borderType) &&
        pyopencv_to(pyobj__src, _src, ArgInfo("_src", 0)) &&
        pyopencv_to(pyobj__dst, _dst, ArgInfo("_dst", 1)) &&
        pyopencv_to(pyobj_ksize, ksize, ArgInfo("ksize", 0)) &&
        pyopencv_to(pyobj_anchor, anchor, ArgInfo("anchor", 0)) )
    {
        ERRWRAP2(cv::sqrBoxFilter(_src, _dst, ddepth, ksize, anchor, normalize, borderType));
        return pyopencv_from(_dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj__src = NULL;
    UMat _src;
    PyObject* pyobj__dst = NULL;
    UMat _dst;
    int ddepth=0;
    PyObject* pyobj_ksize = NULL;
    Size ksize;
    PyObject* pyobj_anchor = NULL;
    Point anchor=Point(-1, -1);
    bool normalize=true;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "_src", "ddepth", "ksize", "_dst", "anchor", "normalize", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OiO|OObi:sqrBoxFilter", (char**)keywords, &pyobj__src, &ddepth, &pyobj_ksize, &pyobj__dst, &pyobj_anchor, &normalize, &borderType) &&
        pyopencv_to(pyobj__src, _src, ArgInfo("_src", 0)) &&
        pyopencv_to(pyobj__dst, _dst, ArgInfo("_dst", 1)) &&
        pyopencv_to(pyobj_ksize, ksize, ArgInfo("ksize", 0)) &&
        pyopencv_to(pyobj_anchor, anchor, ArgInfo("anchor", 0)) )
    {
        ERRWRAP2(cv::sqrBoxFilter(_src, _dst, ddepth, ksize, anchor, normalize, borderType));
        return pyopencv_from(_dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_sqrt(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:sqrt", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::sqrt(src, dst));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:sqrt", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::sqrt(src, dst));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_startWindowThread(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::startWindowThread());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_stereoCalibrate(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_objectPoints = NULL;
    vector_Mat objectPoints;
    PyObject* pyobj_imagePoints1 = NULL;
    vector_Mat imagePoints1;
    PyObject* pyobj_imagePoints2 = NULL;
    vector_Mat imagePoints2;
    PyObject* pyobj_cameraMatrix1 = NULL;
    Mat cameraMatrix1;
    PyObject* pyobj_distCoeffs1 = NULL;
    Mat distCoeffs1;
    PyObject* pyobj_cameraMatrix2 = NULL;
    Mat cameraMatrix2;
    PyObject* pyobj_distCoeffs2 = NULL;
    Mat distCoeffs2;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    PyObject* pyobj_R = NULL;
    Mat R;
    PyObject* pyobj_T = NULL;
    Mat T;
    PyObject* pyobj_E = NULL;
    Mat E;
    PyObject* pyobj_F = NULL;
    Mat F;
    int flags=CALIB_FIX_INTRINSIC;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6);
    double retval;

    const char* keywords[] = { "objectPoints", "imagePoints1", "imagePoints2", "cameraMatrix1", "distCoeffs1", "cameraMatrix2", "distCoeffs2", "imageSize", "R", "T", "E", "F", "flags", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOOO|OOOOiO:stereoCalibrate", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints1, &pyobj_imagePoints2, &pyobj_cameraMatrix1, &pyobj_distCoeffs1, &pyobj_cameraMatrix2, &pyobj_distCoeffs2, &pyobj_imageSize, &pyobj_R, &pyobj_T, &pyobj_E, &pyobj_F, &flags, &pyobj_criteria) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints1, imagePoints1, ArgInfo("imagePoints1", 0)) &&
        pyopencv_to(pyobj_imagePoints2, imagePoints2, ArgInfo("imagePoints2", 0)) &&
        pyopencv_to(pyobj_cameraMatrix1, cameraMatrix1, ArgInfo("cameraMatrix1", 1)) &&
        pyopencv_to(pyobj_distCoeffs1, distCoeffs1, ArgInfo("distCoeffs1", 1)) &&
        pyopencv_to(pyobj_cameraMatrix2, cameraMatrix2, ArgInfo("cameraMatrix2", 1)) &&
        pyopencv_to(pyobj_distCoeffs2, distCoeffs2, ArgInfo("distCoeffs2", 1)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 1)) &&
        pyopencv_to(pyobj_T, T, ArgInfo("T", 1)) &&
        pyopencv_to(pyobj_E, E, ArgInfo("E", 1)) &&
        pyopencv_to(pyobj_F, F, ArgInfo("F", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(retval = cv::stereoCalibrate(objectPoints, imagePoints1, imagePoints2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, E, F, flags, criteria));
        return Py_BuildValue("(NNNNNNNNN)", pyopencv_from(retval), pyopencv_from(cameraMatrix1), pyopencv_from(distCoeffs1), pyopencv_from(cameraMatrix2), pyopencv_from(distCoeffs2), pyopencv_from(R), pyopencv_from(T), pyopencv_from(E), pyopencv_from(F));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_objectPoints = NULL;
    vector_Mat objectPoints;
    PyObject* pyobj_imagePoints1 = NULL;
    vector_Mat imagePoints1;
    PyObject* pyobj_imagePoints2 = NULL;
    vector_Mat imagePoints2;
    PyObject* pyobj_cameraMatrix1 = NULL;
    UMat cameraMatrix1;
    PyObject* pyobj_distCoeffs1 = NULL;
    UMat distCoeffs1;
    PyObject* pyobj_cameraMatrix2 = NULL;
    UMat cameraMatrix2;
    PyObject* pyobj_distCoeffs2 = NULL;
    UMat distCoeffs2;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    PyObject* pyobj_R = NULL;
    UMat R;
    PyObject* pyobj_T = NULL;
    UMat T;
    PyObject* pyobj_E = NULL;
    UMat E;
    PyObject* pyobj_F = NULL;
    UMat F;
    int flags=CALIB_FIX_INTRINSIC;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6);
    double retval;

    const char* keywords[] = { "objectPoints", "imagePoints1", "imagePoints2", "cameraMatrix1", "distCoeffs1", "cameraMatrix2", "distCoeffs2", "imageSize", "R", "T", "E", "F", "flags", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOOO|OOOOiO:stereoCalibrate", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints1, &pyobj_imagePoints2, &pyobj_cameraMatrix1, &pyobj_distCoeffs1, &pyobj_cameraMatrix2, &pyobj_distCoeffs2, &pyobj_imageSize, &pyobj_R, &pyobj_T, &pyobj_E, &pyobj_F, &flags, &pyobj_criteria) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints1, imagePoints1, ArgInfo("imagePoints1", 0)) &&
        pyopencv_to(pyobj_imagePoints2, imagePoints2, ArgInfo("imagePoints2", 0)) &&
        pyopencv_to(pyobj_cameraMatrix1, cameraMatrix1, ArgInfo("cameraMatrix1", 1)) &&
        pyopencv_to(pyobj_distCoeffs1, distCoeffs1, ArgInfo("distCoeffs1", 1)) &&
        pyopencv_to(pyobj_cameraMatrix2, cameraMatrix2, ArgInfo("cameraMatrix2", 1)) &&
        pyopencv_to(pyobj_distCoeffs2, distCoeffs2, ArgInfo("distCoeffs2", 1)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 1)) &&
        pyopencv_to(pyobj_T, T, ArgInfo("T", 1)) &&
        pyopencv_to(pyobj_E, E, ArgInfo("E", 1)) &&
        pyopencv_to(pyobj_F, F, ArgInfo("F", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(retval = cv::stereoCalibrate(objectPoints, imagePoints1, imagePoints2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, E, F, flags, criteria));
        return Py_BuildValue("(NNNNNNNNN)", pyopencv_from(retval), pyopencv_from(cameraMatrix1), pyopencv_from(distCoeffs1), pyopencv_from(cameraMatrix2), pyopencv_from(distCoeffs2), pyopencv_from(R), pyopencv_from(T), pyopencv_from(E), pyopencv_from(F));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_stereoRectify(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_cameraMatrix1 = NULL;
    Mat cameraMatrix1;
    PyObject* pyobj_distCoeffs1 = NULL;
    Mat distCoeffs1;
    PyObject* pyobj_cameraMatrix2 = NULL;
    Mat cameraMatrix2;
    PyObject* pyobj_distCoeffs2 = NULL;
    Mat distCoeffs2;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    PyObject* pyobj_R = NULL;
    Mat R;
    PyObject* pyobj_T = NULL;
    Mat T;
    PyObject* pyobj_R1 = NULL;
    Mat R1;
    PyObject* pyobj_R2 = NULL;
    Mat R2;
    PyObject* pyobj_P1 = NULL;
    Mat P1;
    PyObject* pyobj_P2 = NULL;
    Mat P2;
    PyObject* pyobj_Q = NULL;
    Mat Q;
    int flags=CALIB_ZERO_DISPARITY;
    double alpha=-1;
    PyObject* pyobj_newImageSize = NULL;
    Size newImageSize;
    Rect validPixROI1;
    Rect validPixROI2;

    const char* keywords[] = { "cameraMatrix1", "distCoeffs1", "cameraMatrix2", "distCoeffs2", "imageSize", "R", "T", "R1", "R2", "P1", "P2", "Q", "flags", "alpha", "newImageSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOO|OOOOOidO:stereoRectify", (char**)keywords, &pyobj_cameraMatrix1, &pyobj_distCoeffs1, &pyobj_cameraMatrix2, &pyobj_distCoeffs2, &pyobj_imageSize, &pyobj_R, &pyobj_T, &pyobj_R1, &pyobj_R2, &pyobj_P1, &pyobj_P2, &pyobj_Q, &flags, &alpha, &pyobj_newImageSize) &&
        pyopencv_to(pyobj_cameraMatrix1, cameraMatrix1, ArgInfo("cameraMatrix1", 0)) &&
        pyopencv_to(pyobj_distCoeffs1, distCoeffs1, ArgInfo("distCoeffs1", 0)) &&
        pyopencv_to(pyobj_cameraMatrix2, cameraMatrix2, ArgInfo("cameraMatrix2", 0)) &&
        pyopencv_to(pyobj_distCoeffs2, distCoeffs2, ArgInfo("distCoeffs2", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 0)) &&
        pyopencv_to(pyobj_T, T, ArgInfo("T", 0)) &&
        pyopencv_to(pyobj_R1, R1, ArgInfo("R1", 1)) &&
        pyopencv_to(pyobj_R2, R2, ArgInfo("R2", 1)) &&
        pyopencv_to(pyobj_P1, P1, ArgInfo("P1", 1)) &&
        pyopencv_to(pyobj_P2, P2, ArgInfo("P2", 1)) &&
        pyopencv_to(pyobj_Q, Q, ArgInfo("Q", 1)) &&
        pyopencv_to(pyobj_newImageSize, newImageSize, ArgInfo("newImageSize", 0)) )
    {
        ERRWRAP2(cv::stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, R1, R2, P1, P2, Q, flags, alpha, newImageSize, &validPixROI1, &validPixROI2));
        return Py_BuildValue("(NNNNNNN)", pyopencv_from(R1), pyopencv_from(R2), pyopencv_from(P1), pyopencv_from(P2), pyopencv_from(Q), pyopencv_from(validPixROI1), pyopencv_from(validPixROI2));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_cameraMatrix1 = NULL;
    UMat cameraMatrix1;
    PyObject* pyobj_distCoeffs1 = NULL;
    UMat distCoeffs1;
    PyObject* pyobj_cameraMatrix2 = NULL;
    UMat cameraMatrix2;
    PyObject* pyobj_distCoeffs2 = NULL;
    UMat distCoeffs2;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    PyObject* pyobj_R = NULL;
    UMat R;
    PyObject* pyobj_T = NULL;
    UMat T;
    PyObject* pyobj_R1 = NULL;
    UMat R1;
    PyObject* pyobj_R2 = NULL;
    UMat R2;
    PyObject* pyobj_P1 = NULL;
    UMat P1;
    PyObject* pyobj_P2 = NULL;
    UMat P2;
    PyObject* pyobj_Q = NULL;
    UMat Q;
    int flags=CALIB_ZERO_DISPARITY;
    double alpha=-1;
    PyObject* pyobj_newImageSize = NULL;
    Size newImageSize;
    Rect validPixROI1;
    Rect validPixROI2;

    const char* keywords[] = { "cameraMatrix1", "distCoeffs1", "cameraMatrix2", "distCoeffs2", "imageSize", "R", "T", "R1", "R2", "P1", "P2", "Q", "flags", "alpha", "newImageSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOO|OOOOOidO:stereoRectify", (char**)keywords, &pyobj_cameraMatrix1, &pyobj_distCoeffs1, &pyobj_cameraMatrix2, &pyobj_distCoeffs2, &pyobj_imageSize, &pyobj_R, &pyobj_T, &pyobj_R1, &pyobj_R2, &pyobj_P1, &pyobj_P2, &pyobj_Q, &flags, &alpha, &pyobj_newImageSize) &&
        pyopencv_to(pyobj_cameraMatrix1, cameraMatrix1, ArgInfo("cameraMatrix1", 0)) &&
        pyopencv_to(pyobj_distCoeffs1, distCoeffs1, ArgInfo("distCoeffs1", 0)) &&
        pyopencv_to(pyobj_cameraMatrix2, cameraMatrix2, ArgInfo("cameraMatrix2", 0)) &&
        pyopencv_to(pyobj_distCoeffs2, distCoeffs2, ArgInfo("distCoeffs2", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 0)) &&
        pyopencv_to(pyobj_T, T, ArgInfo("T", 0)) &&
        pyopencv_to(pyobj_R1, R1, ArgInfo("R1", 1)) &&
        pyopencv_to(pyobj_R2, R2, ArgInfo("R2", 1)) &&
        pyopencv_to(pyobj_P1, P1, ArgInfo("P1", 1)) &&
        pyopencv_to(pyobj_P2, P2, ArgInfo("P2", 1)) &&
        pyopencv_to(pyobj_Q, Q, ArgInfo("Q", 1)) &&
        pyopencv_to(pyobj_newImageSize, newImageSize, ArgInfo("newImageSize", 0)) )
    {
        ERRWRAP2(cv::stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, R1, R2, P1, P2, Q, flags, alpha, newImageSize, &validPixROI1, &validPixROI2));
        return Py_BuildValue("(NNNNNNN)", pyopencv_from(R1), pyopencv_from(R2), pyopencv_from(P1), pyopencv_from(P2), pyopencv_from(Q), pyopencv_from(validPixROI1), pyopencv_from(validPixROI2));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_stereoRectifyUncalibrated(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_points1 = NULL;
    Mat points1;
    PyObject* pyobj_points2 = NULL;
    Mat points2;
    PyObject* pyobj_F = NULL;
    Mat F;
    PyObject* pyobj_imgSize = NULL;
    Size imgSize;
    PyObject* pyobj_H1 = NULL;
    Mat H1;
    PyObject* pyobj_H2 = NULL;
    Mat H2;
    double threshold=5;
    bool retval;

    const char* keywords[] = { "points1", "points2", "F", "imgSize", "H1", "H2", "threshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OOd:stereoRectifyUncalibrated", (char**)keywords, &pyobj_points1, &pyobj_points2, &pyobj_F, &pyobj_imgSize, &pyobj_H1, &pyobj_H2, &threshold) &&
        pyopencv_to(pyobj_points1, points1, ArgInfo("points1", 0)) &&
        pyopencv_to(pyobj_points2, points2, ArgInfo("points2", 0)) &&
        pyopencv_to(pyobj_F, F, ArgInfo("F", 0)) &&
        pyopencv_to(pyobj_imgSize, imgSize, ArgInfo("imgSize", 0)) &&
        pyopencv_to(pyobj_H1, H1, ArgInfo("H1", 1)) &&
        pyopencv_to(pyobj_H2, H2, ArgInfo("H2", 1)) )
    {
        ERRWRAP2(retval = cv::stereoRectifyUncalibrated(points1, points2, F, imgSize, H1, H2, threshold));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(H1), pyopencv_from(H2));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_points1 = NULL;
    UMat points1;
    PyObject* pyobj_points2 = NULL;
    UMat points2;
    PyObject* pyobj_F = NULL;
    UMat F;
    PyObject* pyobj_imgSize = NULL;
    Size imgSize;
    PyObject* pyobj_H1 = NULL;
    UMat H1;
    PyObject* pyobj_H2 = NULL;
    UMat H2;
    double threshold=5;
    bool retval;

    const char* keywords[] = { "points1", "points2", "F", "imgSize", "H1", "H2", "threshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OOd:stereoRectifyUncalibrated", (char**)keywords, &pyobj_points1, &pyobj_points2, &pyobj_F, &pyobj_imgSize, &pyobj_H1, &pyobj_H2, &threshold) &&
        pyopencv_to(pyobj_points1, points1, ArgInfo("points1", 0)) &&
        pyopencv_to(pyobj_points2, points2, ArgInfo("points2", 0)) &&
        pyopencv_to(pyobj_F, F, ArgInfo("F", 0)) &&
        pyopencv_to(pyobj_imgSize, imgSize, ArgInfo("imgSize", 0)) &&
        pyopencv_to(pyobj_H1, H1, ArgInfo("H1", 1)) &&
        pyopencv_to(pyobj_H2, H2, ArgInfo("H2", 1)) )
    {
        ERRWRAP2(retval = cv::stereoRectifyUncalibrated(points1, points2, F, imgSize, H1, H2, threshold));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(H1), pyopencv_from(H2));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_stylization(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    float sigma_s=60;
    float sigma_r=0.45f;

    const char* keywords[] = { "src", "dst", "sigma_s", "sigma_r", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Off:stylization", (char**)keywords, &pyobj_src, &pyobj_dst, &sigma_s, &sigma_r) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::stylization(src, dst, sigma_s, sigma_r));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    float sigma_s=60;
    float sigma_r=0.45f;

    const char* keywords[] = { "src", "dst", "sigma_s", "sigma_r", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Off:stylization", (char**)keywords, &pyobj_src, &pyobj_dst, &sigma_s, &sigma_r) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::stylization(src, dst, sigma_s, sigma_r));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_subtract(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    PyObject* pyobj_src2 = NULL;
    Mat src2;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    int dtype=-1;

    const char* keywords[] = { "src1", "src2", "dst", "mask", "dtype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOi:subtract", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &pyobj_mask, &dtype) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::subtract(src1, src2, dst, mask, dtype));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src1 = NULL;
    UMat src1;
    PyObject* pyobj_src2 = NULL;
    UMat src2;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    int dtype=-1;

    const char* keywords[] = { "src1", "src2", "dst", "mask", "dtype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOi:subtract", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &pyobj_mask, &dtype) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::subtract(src1, src2, dst, mask, dtype));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_sumElems(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    Scalar retval;

    const char* keywords[] = { "src", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:sumElems", (char**)keywords, &pyobj_src) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) )
    {
        ERRWRAP2(retval = cv::sum(src));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    Scalar retval;

    const char* keywords[] = { "src", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:sumElems", (char**)keywords, &pyobj_src) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) )
    {
        ERRWRAP2(retval = cv::sum(src));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_textureFlattening(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    float low_threshold=30;
    float high_threshold=45;
    int kernel_size=3;

    const char* keywords[] = { "src", "mask", "dst", "low_threshold", "high_threshold", "kernel_size", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Offi:textureFlattening", (char**)keywords, &pyobj_src, &pyobj_mask, &pyobj_dst, &low_threshold, &high_threshold, &kernel_size) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::textureFlattening(src, mask, dst, low_threshold, high_threshold, kernel_size));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    float low_threshold=30;
    float high_threshold=45;
    int kernel_size=3;

    const char* keywords[] = { "src", "mask", "dst", "low_threshold", "high_threshold", "kernel_size", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Offi:textureFlattening", (char**)keywords, &pyobj_src, &pyobj_mask, &pyobj_dst, &low_threshold, &high_threshold, &kernel_size) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::textureFlattening(src, mask, dst, low_threshold, high_threshold, kernel_size));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_threshold(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    double thresh=0;
    double maxval=0;
    int type=0;
    double retval;

    const char* keywords[] = { "src", "thresh", "maxval", "type", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oddi|O:threshold", (char**)keywords, &pyobj_src, &thresh, &maxval, &type, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(retval = cv::threshold(src, dst, thresh, maxval, type));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(dst));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    double thresh=0;
    double maxval=0;
    int type=0;
    double retval;

    const char* keywords[] = { "src", "thresh", "maxval", "type", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oddi|O:threshold", (char**)keywords, &pyobj_src, &thresh, &maxval, &type, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(retval = cv::threshold(src, dst, thresh, maxval, type));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(dst));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_trace(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_mtx = NULL;
    Mat mtx;
    Scalar retval;

    const char* keywords[] = { "mtx", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:trace", (char**)keywords, &pyobj_mtx) &&
        pyopencv_to(pyobj_mtx, mtx, ArgInfo("mtx", 0)) )
    {
        ERRWRAP2(retval = cv::trace(mtx));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_mtx = NULL;
    UMat mtx;
    Scalar retval;

    const char* keywords[] = { "mtx", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:trace", (char**)keywords, &pyobj_mtx) &&
        pyopencv_to(pyobj_mtx, mtx, ArgInfo("mtx", 0)) )
    {
        ERRWRAP2(retval = cv::trace(mtx));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_transform(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_m = NULL;
    Mat m;

    const char* keywords[] = { "src", "m", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:transform", (char**)keywords, &pyobj_src, &pyobj_m, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_m, m, ArgInfo("m", 0)) )
    {
        ERRWRAP2(cv::transform(src, dst, m));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_m = NULL;
    UMat m;

    const char* keywords[] = { "src", "m", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:transform", (char**)keywords, &pyobj_src, &pyobj_m, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_m, m, ArgInfo("m", 0)) )
    {
        ERRWRAP2(cv::transform(src, dst, m));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_transpose(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:transpose", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::transpose(src, dst));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:transpose", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::transpose(src, dst));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_triangulatePoints(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_projMatr1 = NULL;
    Mat projMatr1;
    PyObject* pyobj_projMatr2 = NULL;
    Mat projMatr2;
    PyObject* pyobj_projPoints1 = NULL;
    Mat projPoints1;
    PyObject* pyobj_projPoints2 = NULL;
    Mat projPoints2;
    PyObject* pyobj_points4D = NULL;
    Mat points4D;

    const char* keywords[] = { "projMatr1", "projMatr2", "projPoints1", "projPoints2", "points4D", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|O:triangulatePoints", (char**)keywords, &pyobj_projMatr1, &pyobj_projMatr2, &pyobj_projPoints1, &pyobj_projPoints2, &pyobj_points4D) &&
        pyopencv_to(pyobj_projMatr1, projMatr1, ArgInfo("projMatr1", 0)) &&
        pyopencv_to(pyobj_projMatr2, projMatr2, ArgInfo("projMatr2", 0)) &&
        pyopencv_to(pyobj_projPoints1, projPoints1, ArgInfo("projPoints1", 0)) &&
        pyopencv_to(pyobj_projPoints2, projPoints2, ArgInfo("projPoints2", 0)) &&
        pyopencv_to(pyobj_points4D, points4D, ArgInfo("points4D", 1)) )
    {
        ERRWRAP2(cv::triangulatePoints(projMatr1, projMatr2, projPoints1, projPoints2, points4D));
        return pyopencv_from(points4D);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_projMatr1 = NULL;
    UMat projMatr1;
    PyObject* pyobj_projMatr2 = NULL;
    UMat projMatr2;
    PyObject* pyobj_projPoints1 = NULL;
    UMat projPoints1;
    PyObject* pyobj_projPoints2 = NULL;
    UMat projPoints2;
    PyObject* pyobj_points4D = NULL;
    UMat points4D;

    const char* keywords[] = { "projMatr1", "projMatr2", "projPoints1", "projPoints2", "points4D", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|O:triangulatePoints", (char**)keywords, &pyobj_projMatr1, &pyobj_projMatr2, &pyobj_projPoints1, &pyobj_projPoints2, &pyobj_points4D) &&
        pyopencv_to(pyobj_projMatr1, projMatr1, ArgInfo("projMatr1", 0)) &&
        pyopencv_to(pyobj_projMatr2, projMatr2, ArgInfo("projMatr2", 0)) &&
        pyopencv_to(pyobj_projPoints1, projPoints1, ArgInfo("projPoints1", 0)) &&
        pyopencv_to(pyobj_projPoints2, projPoints2, ArgInfo("projPoints2", 0)) &&
        pyopencv_to(pyobj_points4D, points4D, ArgInfo("points4D", 1)) )
    {
        ERRWRAP2(cv::triangulatePoints(projMatr1, projMatr2, projPoints1, projPoints2, points4D));
        return pyopencv_from(points4D);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_undistort(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    Mat distCoeffs;
    PyObject* pyobj_newCameraMatrix = NULL;
    Mat newCameraMatrix;

    const char* keywords[] = { "src", "cameraMatrix", "distCoeffs", "dst", "newCameraMatrix", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|OO:undistort", (char**)keywords, &pyobj_src, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_dst, &pyobj_newCameraMatrix) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_newCameraMatrix, newCameraMatrix, ArgInfo("newCameraMatrix", 0)) )
    {
        ERRWRAP2(cv::undistort(src, dst, cameraMatrix, distCoeffs, newCameraMatrix));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    UMat distCoeffs;
    PyObject* pyobj_newCameraMatrix = NULL;
    UMat newCameraMatrix;

    const char* keywords[] = { "src", "cameraMatrix", "distCoeffs", "dst", "newCameraMatrix", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|OO:undistort", (char**)keywords, &pyobj_src, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_dst, &pyobj_newCameraMatrix) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_newCameraMatrix, newCameraMatrix, ArgInfo("newCameraMatrix", 0)) )
    {
        ERRWRAP2(cv::undistort(src, dst, cameraMatrix, distCoeffs, newCameraMatrix));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_undistortPoints(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    Mat distCoeffs;
    PyObject* pyobj_R = NULL;
    Mat R;
    PyObject* pyobj_P = NULL;
    Mat P;

    const char* keywords[] = { "src", "cameraMatrix", "distCoeffs", "dst", "R", "P", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|OOO:undistortPoints", (char**)keywords, &pyobj_src, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_dst, &pyobj_R, &pyobj_P) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 0)) &&
        pyopencv_to(pyobj_P, P, ArgInfo("P", 0)) )
    {
        ERRWRAP2(cv::undistortPoints(src, dst, cameraMatrix, distCoeffs, R, P));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    UMat distCoeffs;
    PyObject* pyobj_R = NULL;
    UMat R;
    PyObject* pyobj_P = NULL;
    UMat P;

    const char* keywords[] = { "src", "cameraMatrix", "distCoeffs", "dst", "R", "P", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|OOO:undistortPoints", (char**)keywords, &pyobj_src, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_dst, &pyobj_R, &pyobj_P) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 0)) &&
        pyopencv_to(pyobj_P, P, ArgInfo("P", 0)) )
    {
        ERRWRAP2(cv::undistortPoints(src, dst, cameraMatrix, distCoeffs, R, P));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_useOpenVX(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::useOpenVX());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_useOptimized(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::useOptimized());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_validateDisparity(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_disparity = NULL;
    Mat disparity;
    PyObject* pyobj_cost = NULL;
    Mat cost;
    int minDisparity=0;
    int numberOfDisparities=0;
    int disp12MaxDisp=1;

    const char* keywords[] = { "disparity", "cost", "minDisparity", "numberOfDisparities", "disp12MaxDisp", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOii|i:validateDisparity", (char**)keywords, &pyobj_disparity, &pyobj_cost, &minDisparity, &numberOfDisparities, &disp12MaxDisp) &&
        pyopencv_to(pyobj_disparity, disparity, ArgInfo("disparity", 1)) &&
        pyopencv_to(pyobj_cost, cost, ArgInfo("cost", 0)) )
    {
        ERRWRAP2(cv::validateDisparity(disparity, cost, minDisparity, numberOfDisparities, disp12MaxDisp));
        return pyopencv_from(disparity);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_disparity = NULL;
    UMat disparity;
    PyObject* pyobj_cost = NULL;
    UMat cost;
    int minDisparity=0;
    int numberOfDisparities=0;
    int disp12MaxDisp=1;

    const char* keywords[] = { "disparity", "cost", "minDisparity", "numberOfDisparities", "disp12MaxDisp", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOii|i:validateDisparity", (char**)keywords, &pyobj_disparity, &pyobj_cost, &minDisparity, &numberOfDisparities, &disp12MaxDisp) &&
        pyopencv_to(pyobj_disparity, disparity, ArgInfo("disparity", 1)) &&
        pyopencv_to(pyobj_cost, cost, ArgInfo("cost", 0)) )
    {
        ERRWRAP2(cv::validateDisparity(disparity, cost, minDisparity, numberOfDisparities, disp12MaxDisp));
        return pyopencv_from(disparity);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_vconcat(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:vconcat", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::vconcat(src, dst));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:vconcat", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::vconcat(src, dst));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_waitKey(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int delay=0;
    int retval;

    const char* keywords[] = { "delay", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|i:waitKey", (char**)keywords, &delay) )
    {
        ERRWRAP2(retval = cv::waitKey(delay));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_waitKeyEx(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    int delay=0;
    int retval;

    const char* keywords[] = { "delay", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|i:waitKeyEx", (char**)keywords, &delay) )
    {
        ERRWRAP2(retval = cv::waitKeyEx(delay));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_warpAffine(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_M = NULL;
    Mat M;
    PyObject* pyobj_dsize = NULL;
    Size dsize;
    int flags=INTER_LINEAR;
    int borderMode=BORDER_CONSTANT;
    PyObject* pyobj_borderValue = NULL;
    Scalar borderValue;

    const char* keywords[] = { "src", "M", "dsize", "dst", "flags", "borderMode", "borderValue", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|OiiO:warpAffine", (char**)keywords, &pyobj_src, &pyobj_M, &pyobj_dsize, &pyobj_dst, &flags, &borderMode, &pyobj_borderValue) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_M, M, ArgInfo("M", 0)) &&
        pyopencv_to(pyobj_dsize, dsize, ArgInfo("dsize", 0)) &&
        pyopencv_to(pyobj_borderValue, borderValue, ArgInfo("borderValue", 0)) )
    {
        ERRWRAP2(cv::warpAffine(src, dst, M, dsize, flags, borderMode, borderValue));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_M = NULL;
    UMat M;
    PyObject* pyobj_dsize = NULL;
    Size dsize;
    int flags=INTER_LINEAR;
    int borderMode=BORDER_CONSTANT;
    PyObject* pyobj_borderValue = NULL;
    Scalar borderValue;

    const char* keywords[] = { "src", "M", "dsize", "dst", "flags", "borderMode", "borderValue", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|OiiO:warpAffine", (char**)keywords, &pyobj_src, &pyobj_M, &pyobj_dsize, &pyobj_dst, &flags, &borderMode, &pyobj_borderValue) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_M, M, ArgInfo("M", 0)) &&
        pyopencv_to(pyobj_dsize, dsize, ArgInfo("dsize", 0)) &&
        pyopencv_to(pyobj_borderValue, borderValue, ArgInfo("borderValue", 0)) )
    {
        ERRWRAP2(cv::warpAffine(src, dst, M, dsize, flags, borderMode, borderValue));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_warpPerspective(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_M = NULL;
    Mat M;
    PyObject* pyobj_dsize = NULL;
    Size dsize;
    int flags=INTER_LINEAR;
    int borderMode=BORDER_CONSTANT;
    PyObject* pyobj_borderValue = NULL;
    Scalar borderValue;

    const char* keywords[] = { "src", "M", "dsize", "dst", "flags", "borderMode", "borderValue", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|OiiO:warpPerspective", (char**)keywords, &pyobj_src, &pyobj_M, &pyobj_dsize, &pyobj_dst, &flags, &borderMode, &pyobj_borderValue) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_M, M, ArgInfo("M", 0)) &&
        pyopencv_to(pyobj_dsize, dsize, ArgInfo("dsize", 0)) &&
        pyopencv_to(pyobj_borderValue, borderValue, ArgInfo("borderValue", 0)) )
    {
        ERRWRAP2(cv::warpPerspective(src, dst, M, dsize, flags, borderMode, borderValue));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_M = NULL;
    UMat M;
    PyObject* pyobj_dsize = NULL;
    Size dsize;
    int flags=INTER_LINEAR;
    int borderMode=BORDER_CONSTANT;
    PyObject* pyobj_borderValue = NULL;
    Scalar borderValue;

    const char* keywords[] = { "src", "M", "dsize", "dst", "flags", "borderMode", "borderValue", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|OiiO:warpPerspective", (char**)keywords, &pyobj_src, &pyobj_M, &pyobj_dsize, &pyobj_dst, &flags, &borderMode, &pyobj_borderValue) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_M, M, ArgInfo("M", 0)) &&
        pyopencv_to(pyobj_dsize, dsize, ArgInfo("dsize", 0)) &&
        pyopencv_to(pyobj_borderValue, borderValue, ArgInfo("borderValue", 0)) )
    {
        ERRWRAP2(cv::warpPerspective(src, dst, M, dsize, flags, borderMode, borderValue));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_watershed(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_markers = NULL;
    Mat markers;

    const char* keywords[] = { "image", "markers", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:watershed", (char**)keywords, &pyobj_image, &pyobj_markers) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_markers, markers, ArgInfo("markers", 1)) )
    {
        ERRWRAP2(cv::watershed(image, markers));
        return pyopencv_from(markers);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_markers = NULL;
    UMat markers;

    const char* keywords[] = { "image", "markers", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:watershed", (char**)keywords, &pyobj_image, &pyobj_markers) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_markers, markers, ArgInfo("markers", 1)) )
    {
        ERRWRAP2(cv::watershed(image, markers));
        return pyopencv_from(markers);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_Board_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    {
    PyObject* pyobj_objPoints = NULL;
    vector_Mat objPoints;
    PyObject* pyobj_dictionary = NULL;
    Ptr<Dictionary> dictionary;
    PyObject* pyobj_ids = NULL;
    Mat ids;
    Ptr<Board> retval;

    const char* keywords[] = { "objPoints", "dictionary", "ids", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:Board_create", (char**)keywords, &pyobj_objPoints, &pyobj_dictionary, &pyobj_ids) &&
        pyopencv_to(pyobj_objPoints, objPoints, ArgInfo("objPoints", 0)) &&
        pyopencv_to(pyobj_dictionary, dictionary, ArgInfo("dictionary", 0)) &&
        pyopencv_to(pyobj_ids, ids, ArgInfo("ids", 0)) )
    {
        ERRWRAP2(retval = cv::aruco::Board::create(objPoints, dictionary, ids));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_objPoints = NULL;
    vector_Mat objPoints;
    PyObject* pyobj_dictionary = NULL;
    Ptr<Dictionary> dictionary;
    PyObject* pyobj_ids = NULL;
    UMat ids;
    Ptr<Board> retval;

    const char* keywords[] = { "objPoints", "dictionary", "ids", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:Board_create", (char**)keywords, &pyobj_objPoints, &pyobj_dictionary, &pyobj_ids) &&
        pyopencv_to(pyobj_objPoints, objPoints, ArgInfo("objPoints", 0)) &&
        pyopencv_to(pyobj_dictionary, dictionary, ArgInfo("dictionary", 0)) &&
        pyopencv_to(pyobj_ids, ids, ArgInfo("ids", 0)) )
    {
        ERRWRAP2(retval = cv::aruco::Board::create(objPoints, dictionary, ids));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_CharucoBoard_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    int squaresX=0;
    int squaresY=0;
    float squareLength=0.f;
    float markerLength=0.f;
    PyObject* pyobj_dictionary = NULL;
    Ptr<Dictionary> dictionary;
    Ptr<CharucoBoard> retval;

    const char* keywords[] = { "squaresX", "squaresY", "squareLength", "markerLength", "dictionary", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "iiffO:CharucoBoard_create", (char**)keywords, &squaresX, &squaresY, &squareLength, &markerLength, &pyobj_dictionary) &&
        pyopencv_to(pyobj_dictionary, dictionary, ArgInfo("dictionary", 0)) )
    {
        ERRWRAP2(retval = cv::aruco::CharucoBoard::create(squaresX, squaresY, squareLength, markerLength, dictionary));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_DetectorParameters_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    Ptr<DetectorParameters> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::aruco::DetectorParameters::create());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_Dictionary_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    int nMarkers=0;
    int markerSize=0;
    Ptr<Dictionary> retval;

    const char* keywords[] = { "nMarkers", "markerSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "ii:Dictionary_create", (char**)keywords, &nMarkers, &markerSize) )
    {
        ERRWRAP2(retval = cv::aruco::Dictionary::create(nMarkers, markerSize));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_Dictionary_create_from(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    int nMarkers=0;
    int markerSize=0;
    PyObject* pyobj_baseDictionary = NULL;
    Ptr<Dictionary> baseDictionary;
    Ptr<Dictionary> retval;

    const char* keywords[] = { "nMarkers", "markerSize", "baseDictionary", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "iiO:Dictionary_create_from", (char**)keywords, &nMarkers, &markerSize, &pyobj_baseDictionary) &&
        pyopencv_to(pyobj_baseDictionary, baseDictionary, ArgInfo("baseDictionary", 0)) )
    {
        ERRWRAP2(retval = cv::aruco::Dictionary::create(nMarkers, markerSize, baseDictionary));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_Dictionary_get(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    int dict=0;
    Ptr<Dictionary> retval;

    const char* keywords[] = { "dict", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:Dictionary_get", (char**)keywords, &dict) )
    {
        ERRWRAP2(retval = cv::aruco::Dictionary::get(dict));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_GridBoard_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    int markersX=0;
    int markersY=0;
    float markerLength=0.f;
    float markerSeparation=0.f;
    PyObject* pyobj_dictionary = NULL;
    Ptr<Dictionary> dictionary;
    int firstMarker=0;
    Ptr<GridBoard> retval;

    const char* keywords[] = { "markersX", "markersY", "markerLength", "markerSeparation", "dictionary", "firstMarker", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "iiffO|i:GridBoard_create", (char**)keywords, &markersX, &markersY, &markerLength, &markerSeparation, &pyobj_dictionary, &firstMarker) &&
        pyopencv_to(pyobj_dictionary, dictionary, ArgInfo("dictionary", 0)) )
    {
        ERRWRAP2(retval = cv::aruco::GridBoard::create(markersX, markersY, markerLength, markerSeparation, dictionary, firstMarker));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_calibrateCameraAruco(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    {
    PyObject* pyobj_corners = NULL;
    vector_Mat corners;
    PyObject* pyobj_ids = NULL;
    Mat ids;
    PyObject* pyobj_counter = NULL;
    Mat counter;
    PyObject* pyobj_board = NULL;
    Ptr<Board> board;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    Mat distCoeffs;
    PyObject* pyobj_rvecs = NULL;
    vector_Mat rvecs;
    PyObject* pyobj_tvecs = NULL;
    vector_Mat tvecs;
    int flags=0;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria=TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON);
    double retval;

    const char* keywords[] = { "corners", "ids", "counter", "board", "imageSize", "cameraMatrix", "distCoeffs", "rvecs", "tvecs", "flags", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOO|OOiO:calibrateCameraAruco", (char**)keywords, &pyobj_corners, &pyobj_ids, &pyobj_counter, &pyobj_board, &pyobj_imageSize, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvecs, &pyobj_tvecs, &flags, &pyobj_criteria) &&
        pyopencv_to(pyobj_corners, corners, ArgInfo("corners", 0)) &&
        pyopencv_to(pyobj_ids, ids, ArgInfo("ids", 0)) &&
        pyopencv_to(pyobj_counter, counter, ArgInfo("counter", 0)) &&
        pyopencv_to(pyobj_board, board, ArgInfo("board", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 1)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 1)) &&
        pyopencv_to(pyobj_rvecs, rvecs, ArgInfo("rvecs", 1)) &&
        pyopencv_to(pyobj_tvecs, tvecs, ArgInfo("tvecs", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(retval = cv::aruco::calibrateCameraAruco(corners, ids, counter, board, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags, criteria));
        return Py_BuildValue("(NNNNN)", pyopencv_from(retval), pyopencv_from(cameraMatrix), pyopencv_from(distCoeffs), pyopencv_from(rvecs), pyopencv_from(tvecs));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_corners = NULL;
    vector_Mat corners;
    PyObject* pyobj_ids = NULL;
    UMat ids;
    PyObject* pyobj_counter = NULL;
    UMat counter;
    PyObject* pyobj_board = NULL;
    Ptr<Board> board;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    UMat distCoeffs;
    PyObject* pyobj_rvecs = NULL;
    vector_Mat rvecs;
    PyObject* pyobj_tvecs = NULL;
    vector_Mat tvecs;
    int flags=0;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria=TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON);
    double retval;

    const char* keywords[] = { "corners", "ids", "counter", "board", "imageSize", "cameraMatrix", "distCoeffs", "rvecs", "tvecs", "flags", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOO|OOiO:calibrateCameraAruco", (char**)keywords, &pyobj_corners, &pyobj_ids, &pyobj_counter, &pyobj_board, &pyobj_imageSize, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvecs, &pyobj_tvecs, &flags, &pyobj_criteria) &&
        pyopencv_to(pyobj_corners, corners, ArgInfo("corners", 0)) &&
        pyopencv_to(pyobj_ids, ids, ArgInfo("ids", 0)) &&
        pyopencv_to(pyobj_counter, counter, ArgInfo("counter", 0)) &&
        pyopencv_to(pyobj_board, board, ArgInfo("board", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 1)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 1)) &&
        pyopencv_to(pyobj_rvecs, rvecs, ArgInfo("rvecs", 1)) &&
        pyopencv_to(pyobj_tvecs, tvecs, ArgInfo("tvecs", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(retval = cv::aruco::calibrateCameraAruco(corners, ids, counter, board, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags, criteria));
        return Py_BuildValue("(NNNNN)", pyopencv_from(retval), pyopencv_from(cameraMatrix), pyopencv_from(distCoeffs), pyopencv_from(rvecs), pyopencv_from(tvecs));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_calibrateCameraArucoExtended(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    {
    PyObject* pyobj_corners = NULL;
    vector_Mat corners;
    PyObject* pyobj_ids = NULL;
    Mat ids;
    PyObject* pyobj_counter = NULL;
    Mat counter;
    PyObject* pyobj_board = NULL;
    Ptr<Board> board;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    Mat distCoeffs;
    PyObject* pyobj_rvecs = NULL;
    vector_Mat rvecs;
    PyObject* pyobj_tvecs = NULL;
    vector_Mat tvecs;
    PyObject* pyobj_stdDeviationsIntrinsics = NULL;
    Mat stdDeviationsIntrinsics;
    PyObject* pyobj_stdDeviationsExtrinsics = NULL;
    Mat stdDeviationsExtrinsics;
    PyObject* pyobj_perViewErrors = NULL;
    Mat perViewErrors;
    int flags=0;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria=TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON);
    double retval;

    const char* keywords[] = { "corners", "ids", "counter", "board", "imageSize", "cameraMatrix", "distCoeffs", "rvecs", "tvecs", "stdDeviationsIntrinsics", "stdDeviationsExtrinsics", "perViewErrors", "flags", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOO|OOOOOiO:calibrateCameraArucoExtended", (char**)keywords, &pyobj_corners, &pyobj_ids, &pyobj_counter, &pyobj_board, &pyobj_imageSize, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvecs, &pyobj_tvecs, &pyobj_stdDeviationsIntrinsics, &pyobj_stdDeviationsExtrinsics, &pyobj_perViewErrors, &flags, &pyobj_criteria) &&
        pyopencv_to(pyobj_corners, corners, ArgInfo("corners", 0)) &&
        pyopencv_to(pyobj_ids, ids, ArgInfo("ids", 0)) &&
        pyopencv_to(pyobj_counter, counter, ArgInfo("counter", 0)) &&
        pyopencv_to(pyobj_board, board, ArgInfo("board", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 1)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 1)) &&
        pyopencv_to(pyobj_rvecs, rvecs, ArgInfo("rvecs", 1)) &&
        pyopencv_to(pyobj_tvecs, tvecs, ArgInfo("tvecs", 1)) &&
        pyopencv_to(pyobj_stdDeviationsIntrinsics, stdDeviationsIntrinsics, ArgInfo("stdDeviationsIntrinsics", 1)) &&
        pyopencv_to(pyobj_stdDeviationsExtrinsics, stdDeviationsExtrinsics, ArgInfo("stdDeviationsExtrinsics", 1)) &&
        pyopencv_to(pyobj_perViewErrors, perViewErrors, ArgInfo("perViewErrors", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(retval = cv::aruco::calibrateCameraAruco(corners, ids, counter, board, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors, flags, criteria));
        return Py_BuildValue("(NNNNNNNN)", pyopencv_from(retval), pyopencv_from(cameraMatrix), pyopencv_from(distCoeffs), pyopencv_from(rvecs), pyopencv_from(tvecs), pyopencv_from(stdDeviationsIntrinsics), pyopencv_from(stdDeviationsExtrinsics), pyopencv_from(perViewErrors));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_corners = NULL;
    vector_Mat corners;
    PyObject* pyobj_ids = NULL;
    UMat ids;
    PyObject* pyobj_counter = NULL;
    UMat counter;
    PyObject* pyobj_board = NULL;
    Ptr<Board> board;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    UMat distCoeffs;
    PyObject* pyobj_rvecs = NULL;
    vector_Mat rvecs;
    PyObject* pyobj_tvecs = NULL;
    vector_Mat tvecs;
    PyObject* pyobj_stdDeviationsIntrinsics = NULL;
    UMat stdDeviationsIntrinsics;
    PyObject* pyobj_stdDeviationsExtrinsics = NULL;
    UMat stdDeviationsExtrinsics;
    PyObject* pyobj_perViewErrors = NULL;
    UMat perViewErrors;
    int flags=0;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria=TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON);
    double retval;

    const char* keywords[] = { "corners", "ids", "counter", "board", "imageSize", "cameraMatrix", "distCoeffs", "rvecs", "tvecs", "stdDeviationsIntrinsics", "stdDeviationsExtrinsics", "perViewErrors", "flags", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOO|OOOOOiO:calibrateCameraArucoExtended", (char**)keywords, &pyobj_corners, &pyobj_ids, &pyobj_counter, &pyobj_board, &pyobj_imageSize, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvecs, &pyobj_tvecs, &pyobj_stdDeviationsIntrinsics, &pyobj_stdDeviationsExtrinsics, &pyobj_perViewErrors, &flags, &pyobj_criteria) &&
        pyopencv_to(pyobj_corners, corners, ArgInfo("corners", 0)) &&
        pyopencv_to(pyobj_ids, ids, ArgInfo("ids", 0)) &&
        pyopencv_to(pyobj_counter, counter, ArgInfo("counter", 0)) &&
        pyopencv_to(pyobj_board, board, ArgInfo("board", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 1)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 1)) &&
        pyopencv_to(pyobj_rvecs, rvecs, ArgInfo("rvecs", 1)) &&
        pyopencv_to(pyobj_tvecs, tvecs, ArgInfo("tvecs", 1)) &&
        pyopencv_to(pyobj_stdDeviationsIntrinsics, stdDeviationsIntrinsics, ArgInfo("stdDeviationsIntrinsics", 1)) &&
        pyopencv_to(pyobj_stdDeviationsExtrinsics, stdDeviationsExtrinsics, ArgInfo("stdDeviationsExtrinsics", 1)) &&
        pyopencv_to(pyobj_perViewErrors, perViewErrors, ArgInfo("perViewErrors", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(retval = cv::aruco::calibrateCameraAruco(corners, ids, counter, board, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors, flags, criteria));
        return Py_BuildValue("(NNNNNNNN)", pyopencv_from(retval), pyopencv_from(cameraMatrix), pyopencv_from(distCoeffs), pyopencv_from(rvecs), pyopencv_from(tvecs), pyopencv_from(stdDeviationsIntrinsics), pyopencv_from(stdDeviationsExtrinsics), pyopencv_from(perViewErrors));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_calibrateCameraCharuco(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    {
    PyObject* pyobj_charucoCorners = NULL;
    vector_Mat charucoCorners;
    PyObject* pyobj_charucoIds = NULL;
    vector_Mat charucoIds;
    PyObject* pyobj_board = NULL;
    Ptr<CharucoBoard> board;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    Mat distCoeffs;
    PyObject* pyobj_rvecs = NULL;
    vector_Mat rvecs;
    PyObject* pyobj_tvecs = NULL;
    vector_Mat tvecs;
    int flags=0;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria=TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON);
    double retval;

    const char* keywords[] = { "charucoCorners", "charucoIds", "board", "imageSize", "cameraMatrix", "distCoeffs", "rvecs", "tvecs", "flags", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOO|OOiO:calibrateCameraCharuco", (char**)keywords, &pyobj_charucoCorners, &pyobj_charucoIds, &pyobj_board, &pyobj_imageSize, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvecs, &pyobj_tvecs, &flags, &pyobj_criteria) &&
        pyopencv_to(pyobj_charucoCorners, charucoCorners, ArgInfo("charucoCorners", 0)) &&
        pyopencv_to(pyobj_charucoIds, charucoIds, ArgInfo("charucoIds", 0)) &&
        pyopencv_to(pyobj_board, board, ArgInfo("board", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 1)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 1)) &&
        pyopencv_to(pyobj_rvecs, rvecs, ArgInfo("rvecs", 1)) &&
        pyopencv_to(pyobj_tvecs, tvecs, ArgInfo("tvecs", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(retval = cv::aruco::calibrateCameraCharuco(charucoCorners, charucoIds, board, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags, criteria));
        return Py_BuildValue("(NNNNN)", pyopencv_from(retval), pyopencv_from(cameraMatrix), pyopencv_from(distCoeffs), pyopencv_from(rvecs), pyopencv_from(tvecs));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_charucoCorners = NULL;
    vector_Mat charucoCorners;
    PyObject* pyobj_charucoIds = NULL;
    vector_Mat charucoIds;
    PyObject* pyobj_board = NULL;
    Ptr<CharucoBoard> board;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    UMat distCoeffs;
    PyObject* pyobj_rvecs = NULL;
    vector_Mat rvecs;
    PyObject* pyobj_tvecs = NULL;
    vector_Mat tvecs;
    int flags=0;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria=TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON);
    double retval;

    const char* keywords[] = { "charucoCorners", "charucoIds", "board", "imageSize", "cameraMatrix", "distCoeffs", "rvecs", "tvecs", "flags", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOO|OOiO:calibrateCameraCharuco", (char**)keywords, &pyobj_charucoCorners, &pyobj_charucoIds, &pyobj_board, &pyobj_imageSize, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvecs, &pyobj_tvecs, &flags, &pyobj_criteria) &&
        pyopencv_to(pyobj_charucoCorners, charucoCorners, ArgInfo("charucoCorners", 0)) &&
        pyopencv_to(pyobj_charucoIds, charucoIds, ArgInfo("charucoIds", 0)) &&
        pyopencv_to(pyobj_board, board, ArgInfo("board", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 1)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 1)) &&
        pyopencv_to(pyobj_rvecs, rvecs, ArgInfo("rvecs", 1)) &&
        pyopencv_to(pyobj_tvecs, tvecs, ArgInfo("tvecs", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(retval = cv::aruco::calibrateCameraCharuco(charucoCorners, charucoIds, board, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags, criteria));
        return Py_BuildValue("(NNNNN)", pyopencv_from(retval), pyopencv_from(cameraMatrix), pyopencv_from(distCoeffs), pyopencv_from(rvecs), pyopencv_from(tvecs));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_calibrateCameraCharucoExtended(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    {
    PyObject* pyobj_charucoCorners = NULL;
    vector_Mat charucoCorners;
    PyObject* pyobj_charucoIds = NULL;
    vector_Mat charucoIds;
    PyObject* pyobj_board = NULL;
    Ptr<CharucoBoard> board;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    Mat distCoeffs;
    PyObject* pyobj_rvecs = NULL;
    vector_Mat rvecs;
    PyObject* pyobj_tvecs = NULL;
    vector_Mat tvecs;
    PyObject* pyobj_stdDeviationsIntrinsics = NULL;
    Mat stdDeviationsIntrinsics;
    PyObject* pyobj_stdDeviationsExtrinsics = NULL;
    Mat stdDeviationsExtrinsics;
    PyObject* pyobj_perViewErrors = NULL;
    Mat perViewErrors;
    int flags=0;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria=TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON);
    double retval;

    const char* keywords[] = { "charucoCorners", "charucoIds", "board", "imageSize", "cameraMatrix", "distCoeffs", "rvecs", "tvecs", "stdDeviationsIntrinsics", "stdDeviationsExtrinsics", "perViewErrors", "flags", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOO|OOOOOiO:calibrateCameraCharucoExtended", (char**)keywords, &pyobj_charucoCorners, &pyobj_charucoIds, &pyobj_board, &pyobj_imageSize, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvecs, &pyobj_tvecs, &pyobj_stdDeviationsIntrinsics, &pyobj_stdDeviationsExtrinsics, &pyobj_perViewErrors, &flags, &pyobj_criteria) &&
        pyopencv_to(pyobj_charucoCorners, charucoCorners, ArgInfo("charucoCorners", 0)) &&
        pyopencv_to(pyobj_charucoIds, charucoIds, ArgInfo("charucoIds", 0)) &&
        pyopencv_to(pyobj_board, board, ArgInfo("board", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 1)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 1)) &&
        pyopencv_to(pyobj_rvecs, rvecs, ArgInfo("rvecs", 1)) &&
        pyopencv_to(pyobj_tvecs, tvecs, ArgInfo("tvecs", 1)) &&
        pyopencv_to(pyobj_stdDeviationsIntrinsics, stdDeviationsIntrinsics, ArgInfo("stdDeviationsIntrinsics", 1)) &&
        pyopencv_to(pyobj_stdDeviationsExtrinsics, stdDeviationsExtrinsics, ArgInfo("stdDeviationsExtrinsics", 1)) &&
        pyopencv_to(pyobj_perViewErrors, perViewErrors, ArgInfo("perViewErrors", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(retval = cv::aruco::calibrateCameraCharuco(charucoCorners, charucoIds, board, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors, flags, criteria));
        return Py_BuildValue("(NNNNNNNN)", pyopencv_from(retval), pyopencv_from(cameraMatrix), pyopencv_from(distCoeffs), pyopencv_from(rvecs), pyopencv_from(tvecs), pyopencv_from(stdDeviationsIntrinsics), pyopencv_from(stdDeviationsExtrinsics), pyopencv_from(perViewErrors));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_charucoCorners = NULL;
    vector_Mat charucoCorners;
    PyObject* pyobj_charucoIds = NULL;
    vector_Mat charucoIds;
    PyObject* pyobj_board = NULL;
    Ptr<CharucoBoard> board;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    UMat distCoeffs;
    PyObject* pyobj_rvecs = NULL;
    vector_Mat rvecs;
    PyObject* pyobj_tvecs = NULL;
    vector_Mat tvecs;
    PyObject* pyobj_stdDeviationsIntrinsics = NULL;
    UMat stdDeviationsIntrinsics;
    PyObject* pyobj_stdDeviationsExtrinsics = NULL;
    UMat stdDeviationsExtrinsics;
    PyObject* pyobj_perViewErrors = NULL;
    UMat perViewErrors;
    int flags=0;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria=TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON);
    double retval;

    const char* keywords[] = { "charucoCorners", "charucoIds", "board", "imageSize", "cameraMatrix", "distCoeffs", "rvecs", "tvecs", "stdDeviationsIntrinsics", "stdDeviationsExtrinsics", "perViewErrors", "flags", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOO|OOOOOiO:calibrateCameraCharucoExtended", (char**)keywords, &pyobj_charucoCorners, &pyobj_charucoIds, &pyobj_board, &pyobj_imageSize, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvecs, &pyobj_tvecs, &pyobj_stdDeviationsIntrinsics, &pyobj_stdDeviationsExtrinsics, &pyobj_perViewErrors, &flags, &pyobj_criteria) &&
        pyopencv_to(pyobj_charucoCorners, charucoCorners, ArgInfo("charucoCorners", 0)) &&
        pyopencv_to(pyobj_charucoIds, charucoIds, ArgInfo("charucoIds", 0)) &&
        pyopencv_to(pyobj_board, board, ArgInfo("board", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 1)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 1)) &&
        pyopencv_to(pyobj_rvecs, rvecs, ArgInfo("rvecs", 1)) &&
        pyopencv_to(pyobj_tvecs, tvecs, ArgInfo("tvecs", 1)) &&
        pyopencv_to(pyobj_stdDeviationsIntrinsics, stdDeviationsIntrinsics, ArgInfo("stdDeviationsIntrinsics", 1)) &&
        pyopencv_to(pyobj_stdDeviationsExtrinsics, stdDeviationsExtrinsics, ArgInfo("stdDeviationsExtrinsics", 1)) &&
        pyopencv_to(pyobj_perViewErrors, perViewErrors, ArgInfo("perViewErrors", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(retval = cv::aruco::calibrateCameraCharuco(charucoCorners, charucoIds, board, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors, flags, criteria));
        return Py_BuildValue("(NNNNNNNN)", pyopencv_from(retval), pyopencv_from(cameraMatrix), pyopencv_from(distCoeffs), pyopencv_from(rvecs), pyopencv_from(tvecs), pyopencv_from(stdDeviationsIntrinsics), pyopencv_from(stdDeviationsExtrinsics), pyopencv_from(perViewErrors));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_custom_dictionary(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    int nMarkers=0;
    int markerSize=0;
    Ptr<Dictionary> retval;

    const char* keywords[] = { "nMarkers", "markerSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "ii:custom_dictionary", (char**)keywords, &nMarkers, &markerSize) )
    {
        ERRWRAP2(retval = cv::aruco::generateCustomDictionary(nMarkers, markerSize));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_custom_dictionary_from(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    int nMarkers=0;
    int markerSize=0;
    PyObject* pyobj_baseDictionary = NULL;
    Ptr<Dictionary> baseDictionary;
    Ptr<Dictionary> retval;

    const char* keywords[] = { "nMarkers", "markerSize", "baseDictionary", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "iiO:custom_dictionary_from", (char**)keywords, &nMarkers, &markerSize, &pyobj_baseDictionary) &&
        pyopencv_to(pyobj_baseDictionary, baseDictionary, ArgInfo("baseDictionary", 0)) )
    {
        ERRWRAP2(retval = cv::aruco::generateCustomDictionary(nMarkers, markerSize, baseDictionary));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_detectCharucoDiamond(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_markerCorners = NULL;
    vector_Mat markerCorners;
    PyObject* pyobj_markerIds = NULL;
    Mat markerIds;
    float squareMarkerLengthRate=0.f;
    PyObject* pyobj_diamondCorners = NULL;
    vector_Mat diamondCorners;
    PyObject* pyobj_diamondIds = NULL;
    Mat diamondIds;
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    Mat distCoeffs;

    const char* keywords[] = { "image", "markerCorners", "markerIds", "squareMarkerLengthRate", "diamondCorners", "diamondIds", "cameraMatrix", "distCoeffs", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOf|OOOO:detectCharucoDiamond", (char**)keywords, &pyobj_image, &pyobj_markerCorners, &pyobj_markerIds, &squareMarkerLengthRate, &pyobj_diamondCorners, &pyobj_diamondIds, &pyobj_cameraMatrix, &pyobj_distCoeffs) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_markerCorners, markerCorners, ArgInfo("markerCorners", 0)) &&
        pyopencv_to(pyobj_markerIds, markerIds, ArgInfo("markerIds", 0)) &&
        pyopencv_to(pyobj_diamondCorners, diamondCorners, ArgInfo("diamondCorners", 1)) &&
        pyopencv_to(pyobj_diamondIds, diamondIds, ArgInfo("diamondIds", 1)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) )
    {
        ERRWRAP2(cv::aruco::detectCharucoDiamond(image, markerCorners, markerIds, squareMarkerLengthRate, diamondCorners, diamondIds, cameraMatrix, distCoeffs));
        return Py_BuildValue("(NN)", pyopencv_from(diamondCorners), pyopencv_from(diamondIds));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_markerCorners = NULL;
    vector_Mat markerCorners;
    PyObject* pyobj_markerIds = NULL;
    UMat markerIds;
    float squareMarkerLengthRate=0.f;
    PyObject* pyobj_diamondCorners = NULL;
    vector_Mat diamondCorners;
    PyObject* pyobj_diamondIds = NULL;
    UMat diamondIds;
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    UMat distCoeffs;

    const char* keywords[] = { "image", "markerCorners", "markerIds", "squareMarkerLengthRate", "diamondCorners", "diamondIds", "cameraMatrix", "distCoeffs", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOf|OOOO:detectCharucoDiamond", (char**)keywords, &pyobj_image, &pyobj_markerCorners, &pyobj_markerIds, &squareMarkerLengthRate, &pyobj_diamondCorners, &pyobj_diamondIds, &pyobj_cameraMatrix, &pyobj_distCoeffs) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_markerCorners, markerCorners, ArgInfo("markerCorners", 0)) &&
        pyopencv_to(pyobj_markerIds, markerIds, ArgInfo("markerIds", 0)) &&
        pyopencv_to(pyobj_diamondCorners, diamondCorners, ArgInfo("diamondCorners", 1)) &&
        pyopencv_to(pyobj_diamondIds, diamondIds, ArgInfo("diamondIds", 1)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) )
    {
        ERRWRAP2(cv::aruco::detectCharucoDiamond(image, markerCorners, markerIds, squareMarkerLengthRate, diamondCorners, diamondIds, cameraMatrix, distCoeffs));
        return Py_BuildValue("(NN)", pyopencv_from(diamondCorners), pyopencv_from(diamondIds));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_detectMarkers(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_dictionary = NULL;
    Ptr<Dictionary> dictionary;
    PyObject* pyobj_corners = NULL;
    vector_Mat corners;
    PyObject* pyobj_ids = NULL;
    Mat ids;
    PyObject* pyobj_parameters = NULL;
    Ptr<DetectorParameters> parameters=DetectorParameters::create();
    PyObject* pyobj_rejectedImgPoints = NULL;
    vector_Mat rejectedImgPoints;

    const char* keywords[] = { "image", "dictionary", "corners", "ids", "parameters", "rejectedImgPoints", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOOO:detectMarkers", (char**)keywords, &pyobj_image, &pyobj_dictionary, &pyobj_corners, &pyobj_ids, &pyobj_parameters, &pyobj_rejectedImgPoints) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_dictionary, dictionary, ArgInfo("dictionary", 0)) &&
        pyopencv_to(pyobj_corners, corners, ArgInfo("corners", 1)) &&
        pyopencv_to(pyobj_ids, ids, ArgInfo("ids", 1)) &&
        pyopencv_to(pyobj_parameters, parameters, ArgInfo("parameters", 0)) &&
        pyopencv_to(pyobj_rejectedImgPoints, rejectedImgPoints, ArgInfo("rejectedImgPoints", 1)) )
    {
        ERRWRAP2(cv::aruco::detectMarkers(image, dictionary, corners, ids, parameters, rejectedImgPoints));
        return Py_BuildValue("(NNN)", pyopencv_from(corners), pyopencv_from(ids), pyopencv_from(rejectedImgPoints));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_dictionary = NULL;
    Ptr<Dictionary> dictionary;
    PyObject* pyobj_corners = NULL;
    vector_Mat corners;
    PyObject* pyobj_ids = NULL;
    UMat ids;
    PyObject* pyobj_parameters = NULL;
    Ptr<DetectorParameters> parameters=DetectorParameters::create();
    PyObject* pyobj_rejectedImgPoints = NULL;
    vector_Mat rejectedImgPoints;

    const char* keywords[] = { "image", "dictionary", "corners", "ids", "parameters", "rejectedImgPoints", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOOO:detectMarkers", (char**)keywords, &pyobj_image, &pyobj_dictionary, &pyobj_corners, &pyobj_ids, &pyobj_parameters, &pyobj_rejectedImgPoints) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_dictionary, dictionary, ArgInfo("dictionary", 0)) &&
        pyopencv_to(pyobj_corners, corners, ArgInfo("corners", 1)) &&
        pyopencv_to(pyobj_ids, ids, ArgInfo("ids", 1)) &&
        pyopencv_to(pyobj_parameters, parameters, ArgInfo("parameters", 0)) &&
        pyopencv_to(pyobj_rejectedImgPoints, rejectedImgPoints, ArgInfo("rejectedImgPoints", 1)) )
    {
        ERRWRAP2(cv::aruco::detectMarkers(image, dictionary, corners, ids, parameters, rejectedImgPoints));
        return Py_BuildValue("(NNN)", pyopencv_from(corners), pyopencv_from(ids), pyopencv_from(rejectedImgPoints));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_drawAxis(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    Mat distCoeffs;
    PyObject* pyobj_rvec = NULL;
    Mat rvec;
    PyObject* pyobj_tvec = NULL;
    Mat tvec;
    float length=0.f;

    const char* keywords[] = { "image", "cameraMatrix", "distCoeffs", "rvec", "tvec", "length", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOf:drawAxis", (char**)keywords, &pyobj_image, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvec, &pyobj_tvec, &length) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 1)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_rvec, rvec, ArgInfo("rvec", 0)) &&
        pyopencv_to(pyobj_tvec, tvec, ArgInfo("tvec", 0)) )
    {
        ERRWRAP2(cv::aruco::drawAxis(image, cameraMatrix, distCoeffs, rvec, tvec, length));
        return pyopencv_from(image);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    UMat distCoeffs;
    PyObject* pyobj_rvec = NULL;
    UMat rvec;
    PyObject* pyobj_tvec = NULL;
    UMat tvec;
    float length=0.f;

    const char* keywords[] = { "image", "cameraMatrix", "distCoeffs", "rvec", "tvec", "length", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOf:drawAxis", (char**)keywords, &pyobj_image, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvec, &pyobj_tvec, &length) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 1)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_rvec, rvec, ArgInfo("rvec", 0)) &&
        pyopencv_to(pyobj_tvec, tvec, ArgInfo("tvec", 0)) )
    {
        ERRWRAP2(cv::aruco::drawAxis(image, cameraMatrix, distCoeffs, rvec, tvec, length));
        return pyopencv_from(image);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_drawDetectedCornersCharuco(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_charucoCorners = NULL;
    Mat charucoCorners;
    PyObject* pyobj_charucoIds = NULL;
    Mat charucoIds;
    PyObject* pyobj_cornerColor = NULL;
    Scalar cornerColor=Scalar(255, 0, 0);

    const char* keywords[] = { "image", "charucoCorners", "charucoIds", "cornerColor", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OO:drawDetectedCornersCharuco", (char**)keywords, &pyobj_image, &pyobj_charucoCorners, &pyobj_charucoIds, &pyobj_cornerColor) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 1)) &&
        pyopencv_to(pyobj_charucoCorners, charucoCorners, ArgInfo("charucoCorners", 0)) &&
        pyopencv_to(pyobj_charucoIds, charucoIds, ArgInfo("charucoIds", 0)) &&
        pyopencv_to(pyobj_cornerColor, cornerColor, ArgInfo("cornerColor", 0)) )
    {
        ERRWRAP2(cv::aruco::drawDetectedCornersCharuco(image, charucoCorners, charucoIds, cornerColor));
        return pyopencv_from(image);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_charucoCorners = NULL;
    UMat charucoCorners;
    PyObject* pyobj_charucoIds = NULL;
    UMat charucoIds;
    PyObject* pyobj_cornerColor = NULL;
    Scalar cornerColor=Scalar(255, 0, 0);

    const char* keywords[] = { "image", "charucoCorners", "charucoIds", "cornerColor", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OO:drawDetectedCornersCharuco", (char**)keywords, &pyobj_image, &pyobj_charucoCorners, &pyobj_charucoIds, &pyobj_cornerColor) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 1)) &&
        pyopencv_to(pyobj_charucoCorners, charucoCorners, ArgInfo("charucoCorners", 0)) &&
        pyopencv_to(pyobj_charucoIds, charucoIds, ArgInfo("charucoIds", 0)) &&
        pyopencv_to(pyobj_cornerColor, cornerColor, ArgInfo("cornerColor", 0)) )
    {
        ERRWRAP2(cv::aruco::drawDetectedCornersCharuco(image, charucoCorners, charucoIds, cornerColor));
        return pyopencv_from(image);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_drawDetectedDiamonds(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_diamondCorners = NULL;
    vector_Mat diamondCorners;
    PyObject* pyobj_diamondIds = NULL;
    Mat diamondIds;
    PyObject* pyobj_borderColor = NULL;
    Scalar borderColor=Scalar(0, 0, 255);

    const char* keywords[] = { "image", "diamondCorners", "diamondIds", "borderColor", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OO:drawDetectedDiamonds", (char**)keywords, &pyobj_image, &pyobj_diamondCorners, &pyobj_diamondIds, &pyobj_borderColor) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 1)) &&
        pyopencv_to(pyobj_diamondCorners, diamondCorners, ArgInfo("diamondCorners", 0)) &&
        pyopencv_to(pyobj_diamondIds, diamondIds, ArgInfo("diamondIds", 0)) &&
        pyopencv_to(pyobj_borderColor, borderColor, ArgInfo("borderColor", 0)) )
    {
        ERRWRAP2(cv::aruco::drawDetectedDiamonds(image, diamondCorners, diamondIds, borderColor));
        return pyopencv_from(image);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_diamondCorners = NULL;
    vector_Mat diamondCorners;
    PyObject* pyobj_diamondIds = NULL;
    UMat diamondIds;
    PyObject* pyobj_borderColor = NULL;
    Scalar borderColor=Scalar(0, 0, 255);

    const char* keywords[] = { "image", "diamondCorners", "diamondIds", "borderColor", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OO:drawDetectedDiamonds", (char**)keywords, &pyobj_image, &pyobj_diamondCorners, &pyobj_diamondIds, &pyobj_borderColor) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 1)) &&
        pyopencv_to(pyobj_diamondCorners, diamondCorners, ArgInfo("diamondCorners", 0)) &&
        pyopencv_to(pyobj_diamondIds, diamondIds, ArgInfo("diamondIds", 0)) &&
        pyopencv_to(pyobj_borderColor, borderColor, ArgInfo("borderColor", 0)) )
    {
        ERRWRAP2(cv::aruco::drawDetectedDiamonds(image, diamondCorners, diamondIds, borderColor));
        return pyopencv_from(image);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_drawDetectedMarkers(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_corners = NULL;
    vector_Mat corners;
    PyObject* pyobj_ids = NULL;
    Mat ids;
    PyObject* pyobj_borderColor = NULL;
    Scalar borderColor=Scalar(0, 255, 0);

    const char* keywords[] = { "image", "corners", "ids", "borderColor", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OO:drawDetectedMarkers", (char**)keywords, &pyobj_image, &pyobj_corners, &pyobj_ids, &pyobj_borderColor) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 1)) &&
        pyopencv_to(pyobj_corners, corners, ArgInfo("corners", 0)) &&
        pyopencv_to(pyobj_ids, ids, ArgInfo("ids", 0)) &&
        pyopencv_to(pyobj_borderColor, borderColor, ArgInfo("borderColor", 0)) )
    {
        ERRWRAP2(cv::aruco::drawDetectedMarkers(image, corners, ids, borderColor));
        return pyopencv_from(image);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_corners = NULL;
    vector_Mat corners;
    PyObject* pyobj_ids = NULL;
    UMat ids;
    PyObject* pyobj_borderColor = NULL;
    Scalar borderColor=Scalar(0, 255, 0);

    const char* keywords[] = { "image", "corners", "ids", "borderColor", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OO:drawDetectedMarkers", (char**)keywords, &pyobj_image, &pyobj_corners, &pyobj_ids, &pyobj_borderColor) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 1)) &&
        pyopencv_to(pyobj_corners, corners, ArgInfo("corners", 0)) &&
        pyopencv_to(pyobj_ids, ids, ArgInfo("ids", 0)) &&
        pyopencv_to(pyobj_borderColor, borderColor, ArgInfo("borderColor", 0)) )
    {
        ERRWRAP2(cv::aruco::drawDetectedMarkers(image, corners, ids, borderColor));
        return pyopencv_from(image);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_drawMarker(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    {
    PyObject* pyobj_dictionary = NULL;
    Ptr<Dictionary> dictionary;
    int id=0;
    int sidePixels=0;
    PyObject* pyobj_img = NULL;
    Mat img;
    int borderBits=1;

    const char* keywords[] = { "dictionary", "id", "sidePixels", "img", "borderBits", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii|Oi:drawMarker", (char**)keywords, &pyobj_dictionary, &id, &sidePixels, &pyobj_img, &borderBits) &&
        pyopencv_to(pyobj_dictionary, dictionary, ArgInfo("dictionary", 0)) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) )
    {
        ERRWRAP2(cv::aruco::drawMarker(dictionary, id, sidePixels, img, borderBits));
        return pyopencv_from(img);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_dictionary = NULL;
    Ptr<Dictionary> dictionary;
    int id=0;
    int sidePixels=0;
    PyObject* pyobj_img = NULL;
    UMat img;
    int borderBits=1;

    const char* keywords[] = { "dictionary", "id", "sidePixels", "img", "borderBits", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii|Oi:drawMarker", (char**)keywords, &pyobj_dictionary, &id, &sidePixels, &pyobj_img, &borderBits) &&
        pyopencv_to(pyobj_dictionary, dictionary, ArgInfo("dictionary", 0)) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) )
    {
        ERRWRAP2(cv::aruco::drawMarker(dictionary, id, sidePixels, img, borderBits));
        return pyopencv_from(img);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_drawPlanarBoard(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    {
    PyObject* pyobj_board = NULL;
    Ptr<Board> board;
    PyObject* pyobj_outSize = NULL;
    Size outSize;
    PyObject* pyobj_img = NULL;
    Mat img;
    int marginSize=0;
    int borderBits=1;

    const char* keywords[] = { "board", "outSize", "img", "marginSize", "borderBits", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Oii:drawPlanarBoard", (char**)keywords, &pyobj_board, &pyobj_outSize, &pyobj_img, &marginSize, &borderBits) &&
        pyopencv_to(pyobj_board, board, ArgInfo("board", 0)) &&
        pyopencv_to(pyobj_outSize, outSize, ArgInfo("outSize", 0)) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) )
    {
        ERRWRAP2(cv::aruco::drawPlanarBoard(board, outSize, img, marginSize, borderBits));
        return pyopencv_from(img);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_board = NULL;
    Ptr<Board> board;
    PyObject* pyobj_outSize = NULL;
    Size outSize;
    PyObject* pyobj_img = NULL;
    UMat img;
    int marginSize=0;
    int borderBits=1;

    const char* keywords[] = { "board", "outSize", "img", "marginSize", "borderBits", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Oii:drawPlanarBoard", (char**)keywords, &pyobj_board, &pyobj_outSize, &pyobj_img, &marginSize, &borderBits) &&
        pyopencv_to(pyobj_board, board, ArgInfo("board", 0)) &&
        pyopencv_to(pyobj_outSize, outSize, ArgInfo("outSize", 0)) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) )
    {
        ERRWRAP2(cv::aruco::drawPlanarBoard(board, outSize, img, marginSize, borderBits));
        return pyopencv_from(img);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_estimatePoseBoard(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    {
    PyObject* pyobj_corners = NULL;
    vector_Mat corners;
    PyObject* pyobj_ids = NULL;
    Mat ids;
    PyObject* pyobj_board = NULL;
    Ptr<Board> board;
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    Mat distCoeffs;
    PyObject* pyobj_rvec = NULL;
    Mat rvec;
    PyObject* pyobj_tvec = NULL;
    Mat tvec;
    bool useExtrinsicGuess=false;
    int retval;

    const char* keywords[] = { "corners", "ids", "board", "cameraMatrix", "distCoeffs", "rvec", "tvec", "useExtrinsicGuess", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOO|OOb:estimatePoseBoard", (char**)keywords, &pyobj_corners, &pyobj_ids, &pyobj_board, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvec, &pyobj_tvec, &useExtrinsicGuess) &&
        pyopencv_to(pyobj_corners, corners, ArgInfo("corners", 0)) &&
        pyopencv_to(pyobj_ids, ids, ArgInfo("ids", 0)) &&
        pyopencv_to(pyobj_board, board, ArgInfo("board", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_rvec, rvec, ArgInfo("rvec", 1)) &&
        pyopencv_to(pyobj_tvec, tvec, ArgInfo("tvec", 1)) )
    {
        ERRWRAP2(retval = cv::aruco::estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(rvec), pyopencv_from(tvec));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_corners = NULL;
    vector_Mat corners;
    PyObject* pyobj_ids = NULL;
    UMat ids;
    PyObject* pyobj_board = NULL;
    Ptr<Board> board;
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    UMat distCoeffs;
    PyObject* pyobj_rvec = NULL;
    UMat rvec;
    PyObject* pyobj_tvec = NULL;
    UMat tvec;
    bool useExtrinsicGuess=false;
    int retval;

    const char* keywords[] = { "corners", "ids", "board", "cameraMatrix", "distCoeffs", "rvec", "tvec", "useExtrinsicGuess", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOO|OOb:estimatePoseBoard", (char**)keywords, &pyobj_corners, &pyobj_ids, &pyobj_board, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvec, &pyobj_tvec, &useExtrinsicGuess) &&
        pyopencv_to(pyobj_corners, corners, ArgInfo("corners", 0)) &&
        pyopencv_to(pyobj_ids, ids, ArgInfo("ids", 0)) &&
        pyopencv_to(pyobj_board, board, ArgInfo("board", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_rvec, rvec, ArgInfo("rvec", 1)) &&
        pyopencv_to(pyobj_tvec, tvec, ArgInfo("tvec", 1)) )
    {
        ERRWRAP2(retval = cv::aruco::estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(rvec), pyopencv_from(tvec));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_estimatePoseCharucoBoard(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    {
    PyObject* pyobj_charucoCorners = NULL;
    Mat charucoCorners;
    PyObject* pyobj_charucoIds = NULL;
    Mat charucoIds;
    PyObject* pyobj_board = NULL;
    Ptr<CharucoBoard> board;
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    Mat distCoeffs;
    PyObject* pyobj_rvec = NULL;
    Mat rvec;
    PyObject* pyobj_tvec = NULL;
    Mat tvec;
    bool useExtrinsicGuess=false;
    bool retval;

    const char* keywords[] = { "charucoCorners", "charucoIds", "board", "cameraMatrix", "distCoeffs", "rvec", "tvec", "useExtrinsicGuess", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOO|OOb:estimatePoseCharucoBoard", (char**)keywords, &pyobj_charucoCorners, &pyobj_charucoIds, &pyobj_board, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvec, &pyobj_tvec, &useExtrinsicGuess) &&
        pyopencv_to(pyobj_charucoCorners, charucoCorners, ArgInfo("charucoCorners", 0)) &&
        pyopencv_to(pyobj_charucoIds, charucoIds, ArgInfo("charucoIds", 0)) &&
        pyopencv_to(pyobj_board, board, ArgInfo("board", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_rvec, rvec, ArgInfo("rvec", 1)) &&
        pyopencv_to(pyobj_tvec, tvec, ArgInfo("tvec", 1)) )
    {
        ERRWRAP2(retval = cv::aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, board, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(rvec), pyopencv_from(tvec));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_charucoCorners = NULL;
    UMat charucoCorners;
    PyObject* pyobj_charucoIds = NULL;
    UMat charucoIds;
    PyObject* pyobj_board = NULL;
    Ptr<CharucoBoard> board;
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    UMat distCoeffs;
    PyObject* pyobj_rvec = NULL;
    UMat rvec;
    PyObject* pyobj_tvec = NULL;
    UMat tvec;
    bool useExtrinsicGuess=false;
    bool retval;

    const char* keywords[] = { "charucoCorners", "charucoIds", "board", "cameraMatrix", "distCoeffs", "rvec", "tvec", "useExtrinsicGuess", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOO|OOb:estimatePoseCharucoBoard", (char**)keywords, &pyobj_charucoCorners, &pyobj_charucoIds, &pyobj_board, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvec, &pyobj_tvec, &useExtrinsicGuess) &&
        pyopencv_to(pyobj_charucoCorners, charucoCorners, ArgInfo("charucoCorners", 0)) &&
        pyopencv_to(pyobj_charucoIds, charucoIds, ArgInfo("charucoIds", 0)) &&
        pyopencv_to(pyobj_board, board, ArgInfo("board", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_rvec, rvec, ArgInfo("rvec", 1)) &&
        pyopencv_to(pyobj_tvec, tvec, ArgInfo("tvec", 1)) )
    {
        ERRWRAP2(retval = cv::aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, board, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(rvec), pyopencv_from(tvec));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_estimatePoseSingleMarkers(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    {
    PyObject* pyobj_corners = NULL;
    vector_Mat corners;
    float markerLength=0.f;
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    Mat distCoeffs;
    PyObject* pyobj_rvecs = NULL;
    Mat rvecs;
    PyObject* pyobj_tvecs = NULL;
    Mat tvecs;

    const char* keywords[] = { "corners", "markerLength", "cameraMatrix", "distCoeffs", "rvecs", "tvecs", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OfOO|OO:estimatePoseSingleMarkers", (char**)keywords, &pyobj_corners, &markerLength, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvecs, &pyobj_tvecs) &&
        pyopencv_to(pyobj_corners, corners, ArgInfo("corners", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_rvecs, rvecs, ArgInfo("rvecs", 1)) &&
        pyopencv_to(pyobj_tvecs, tvecs, ArgInfo("tvecs", 1)) )
    {
        ERRWRAP2(cv::aruco::estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs));
        return Py_BuildValue("(NN)", pyopencv_from(rvecs), pyopencv_from(tvecs));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_corners = NULL;
    vector_Mat corners;
    float markerLength=0.f;
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    UMat distCoeffs;
    PyObject* pyobj_rvecs = NULL;
    UMat rvecs;
    PyObject* pyobj_tvecs = NULL;
    UMat tvecs;

    const char* keywords[] = { "corners", "markerLength", "cameraMatrix", "distCoeffs", "rvecs", "tvecs", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OfOO|OO:estimatePoseSingleMarkers", (char**)keywords, &pyobj_corners, &markerLength, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvecs, &pyobj_tvecs) &&
        pyopencv_to(pyobj_corners, corners, ArgInfo("corners", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_rvecs, rvecs, ArgInfo("rvecs", 1)) &&
        pyopencv_to(pyobj_tvecs, tvecs, ArgInfo("tvecs", 1)) )
    {
        ERRWRAP2(cv::aruco::estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs));
        return Py_BuildValue("(NN)", pyopencv_from(rvecs), pyopencv_from(tvecs));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_getPredefinedDictionary(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    int dict=0;
    Ptr<Dictionary> retval;

    const char* keywords[] = { "dict", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:getPredefinedDictionary", (char**)keywords, &dict) )
    {
        ERRWRAP2(retval = cv::aruco::getPredefinedDictionary(dict));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_interpolateCornersCharuco(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    {
    PyObject* pyobj_markerCorners = NULL;
    vector_Mat markerCorners;
    PyObject* pyobj_markerIds = NULL;
    Mat markerIds;
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_board = NULL;
    Ptr<CharucoBoard> board;
    PyObject* pyobj_charucoCorners = NULL;
    Mat charucoCorners;
    PyObject* pyobj_charucoIds = NULL;
    Mat charucoIds;
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    Mat distCoeffs;
    int minMarkers=2;
    int retval;

    const char* keywords[] = { "markerCorners", "markerIds", "image", "board", "charucoCorners", "charucoIds", "cameraMatrix", "distCoeffs", "minMarkers", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OOOOi:interpolateCornersCharuco", (char**)keywords, &pyobj_markerCorners, &pyobj_markerIds, &pyobj_image, &pyobj_board, &pyobj_charucoCorners, &pyobj_charucoIds, &pyobj_cameraMatrix, &pyobj_distCoeffs, &minMarkers) &&
        pyopencv_to(pyobj_markerCorners, markerCorners, ArgInfo("markerCorners", 0)) &&
        pyopencv_to(pyobj_markerIds, markerIds, ArgInfo("markerIds", 0)) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_board, board, ArgInfo("board", 0)) &&
        pyopencv_to(pyobj_charucoCorners, charucoCorners, ArgInfo("charucoCorners", 1)) &&
        pyopencv_to(pyobj_charucoIds, charucoIds, ArgInfo("charucoIds", 1)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) )
    {
        ERRWRAP2(retval = cv::aruco::interpolateCornersCharuco(markerCorners, markerIds, image, board, charucoCorners, charucoIds, cameraMatrix, distCoeffs, minMarkers));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(charucoCorners), pyopencv_from(charucoIds));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_markerCorners = NULL;
    vector_Mat markerCorners;
    PyObject* pyobj_markerIds = NULL;
    UMat markerIds;
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_board = NULL;
    Ptr<CharucoBoard> board;
    PyObject* pyobj_charucoCorners = NULL;
    UMat charucoCorners;
    PyObject* pyobj_charucoIds = NULL;
    UMat charucoIds;
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    UMat distCoeffs;
    int minMarkers=2;
    int retval;

    const char* keywords[] = { "markerCorners", "markerIds", "image", "board", "charucoCorners", "charucoIds", "cameraMatrix", "distCoeffs", "minMarkers", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OOOOi:interpolateCornersCharuco", (char**)keywords, &pyobj_markerCorners, &pyobj_markerIds, &pyobj_image, &pyobj_board, &pyobj_charucoCorners, &pyobj_charucoIds, &pyobj_cameraMatrix, &pyobj_distCoeffs, &minMarkers) &&
        pyopencv_to(pyobj_markerCorners, markerCorners, ArgInfo("markerCorners", 0)) &&
        pyopencv_to(pyobj_markerIds, markerIds, ArgInfo("markerIds", 0)) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_board, board, ArgInfo("board", 0)) &&
        pyopencv_to(pyobj_charucoCorners, charucoCorners, ArgInfo("charucoCorners", 1)) &&
        pyopencv_to(pyobj_charucoIds, charucoIds, ArgInfo("charucoIds", 1)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) )
    {
        ERRWRAP2(retval = cv::aruco::interpolateCornersCharuco(markerCorners, markerIds, image, board, charucoCorners, charucoIds, cameraMatrix, distCoeffs, minMarkers));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(charucoCorners), pyopencv_from(charucoIds));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_aruco_refineDetectedMarkers(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::aruco;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_board = NULL;
    Ptr<Board> board;
    PyObject* pyobj_detectedCorners = NULL;
    vector_Mat detectedCorners;
    PyObject* pyobj_detectedIds = NULL;
    Mat detectedIds;
    PyObject* pyobj_rejectedCorners = NULL;
    vector_Mat rejectedCorners;
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    Mat distCoeffs;
    float minRepDistance=10.f;
    float errorCorrectionRate=3.f;
    bool checkAllOrders=true;
    PyObject* pyobj_recoveredIdxs = NULL;
    Mat recoveredIdxs;
    PyObject* pyobj_parameters = NULL;
    Ptr<DetectorParameters> parameters=DetectorParameters::create();

    const char* keywords[] = { "image", "board", "detectedCorners", "detectedIds", "rejectedCorners", "cameraMatrix", "distCoeffs", "minRepDistance", "errorCorrectionRate", "checkAllOrders", "recoveredIdxs", "parameters", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOO|OOffbOO:refineDetectedMarkers", (char**)keywords, &pyobj_image, &pyobj_board, &pyobj_detectedCorners, &pyobj_detectedIds, &pyobj_rejectedCorners, &pyobj_cameraMatrix, &pyobj_distCoeffs, &minRepDistance, &errorCorrectionRate, &checkAllOrders, &pyobj_recoveredIdxs, &pyobj_parameters) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_board, board, ArgInfo("board", 0)) &&
        pyopencv_to(pyobj_detectedCorners, detectedCorners, ArgInfo("detectedCorners", 1)) &&
        pyopencv_to(pyobj_detectedIds, detectedIds, ArgInfo("detectedIds", 1)) &&
        pyopencv_to(pyobj_rejectedCorners, rejectedCorners, ArgInfo("rejectedCorners", 1)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_recoveredIdxs, recoveredIdxs, ArgInfo("recoveredIdxs", 1)) &&
        pyopencv_to(pyobj_parameters, parameters, ArgInfo("parameters", 0)) )
    {
        ERRWRAP2(cv::aruco::refineDetectedMarkers(image, board, detectedCorners, detectedIds, rejectedCorners, cameraMatrix, distCoeffs, minRepDistance, errorCorrectionRate, checkAllOrders, recoveredIdxs, parameters));
        return Py_BuildValue("(NNNN)", pyopencv_from(detectedCorners), pyopencv_from(detectedIds), pyopencv_from(rejectedCorners), pyopencv_from(recoveredIdxs));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_board = NULL;
    Ptr<Board> board;
    PyObject* pyobj_detectedCorners = NULL;
    vector_Mat detectedCorners;
    PyObject* pyobj_detectedIds = NULL;
    UMat detectedIds;
    PyObject* pyobj_rejectedCorners = NULL;
    vector_Mat rejectedCorners;
    PyObject* pyobj_cameraMatrix = NULL;
    UMat cameraMatrix;
    PyObject* pyobj_distCoeffs = NULL;
    UMat distCoeffs;
    float minRepDistance=10.f;
    float errorCorrectionRate=3.f;
    bool checkAllOrders=true;
    PyObject* pyobj_recoveredIdxs = NULL;
    UMat recoveredIdxs;
    PyObject* pyobj_parameters = NULL;
    Ptr<DetectorParameters> parameters=DetectorParameters::create();

    const char* keywords[] = { "image", "board", "detectedCorners", "detectedIds", "rejectedCorners", "cameraMatrix", "distCoeffs", "minRepDistance", "errorCorrectionRate", "checkAllOrders", "recoveredIdxs", "parameters", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOO|OOffbOO:refineDetectedMarkers", (char**)keywords, &pyobj_image, &pyobj_board, &pyobj_detectedCorners, &pyobj_detectedIds, &pyobj_rejectedCorners, &pyobj_cameraMatrix, &pyobj_distCoeffs, &minRepDistance, &errorCorrectionRate, &checkAllOrders, &pyobj_recoveredIdxs, &pyobj_parameters) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_board, board, ArgInfo("board", 0)) &&
        pyopencv_to(pyobj_detectedCorners, detectedCorners, ArgInfo("detectedCorners", 1)) &&
        pyopencv_to(pyobj_detectedIds, detectedIds, ArgInfo("detectedIds", 1)) &&
        pyopencv_to(pyobj_rejectedCorners, rejectedCorners, ArgInfo("rejectedCorners", 1)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_recoveredIdxs, recoveredIdxs, ArgInfo("recoveredIdxs", 1)) &&
        pyopencv_to(pyobj_parameters, parameters, ArgInfo("parameters", 0)) )
    {
        ERRWRAP2(cv::aruco::refineDetectedMarkers(image, board, detectedCorners, detectedIds, rejectedCorners, cameraMatrix, distCoeffs, minRepDistance, errorCorrectionRate, checkAllOrders, recoveredIdxs, parameters));
        return Py_BuildValue("(NNNN)", pyopencv_from(detectedCorners), pyopencv_from(detectedIds), pyopencv_from(rejectedCorners), pyopencv_from(recoveredIdxs));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_bgsegm_createBackgroundSubtractorGMG(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::bgsegm;

    int initializationFrames=120;
    double decisionThreshold=0.8;
    Ptr<BackgroundSubtractorGMG> retval;

    const char* keywords[] = { "initializationFrames", "decisionThreshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|id:createBackgroundSubtractorGMG", (char**)keywords, &initializationFrames, &decisionThreshold) )
    {
        ERRWRAP2(retval = cv::bgsegm::createBackgroundSubtractorGMG(initializationFrames, decisionThreshold));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_bgsegm_createBackgroundSubtractorMOG(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::bgsegm;

    int history=200;
    int nmixtures=5;
    double backgroundRatio=0.7;
    double noiseSigma=0;
    Ptr<BackgroundSubtractorMOG> retval;

    const char* keywords[] = { "history", "nmixtures", "backgroundRatio", "noiseSigma", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|iidd:createBackgroundSubtractorMOG", (char**)keywords, &history, &nmixtures, &backgroundRatio, &noiseSigma) )
    {
        ERRWRAP2(retval = cv::bgsegm::createBackgroundSubtractorMOG(history, nmixtures, backgroundRatio, noiseSigma));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_bioinspired_createRetina(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::bioinspired;

    {
    PyObject* pyobj_inputSize = NULL;
    Size inputSize;
    Ptr<Retina> retval;

    const char* keywords[] = { "inputSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:createRetina", (char**)keywords, &pyobj_inputSize) &&
        pyopencv_to(pyobj_inputSize, inputSize, ArgInfo("inputSize", 0)) )
    {
        ERRWRAP2(retval = cv::bioinspired::createRetina(inputSize));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_inputSize = NULL;
    Size inputSize;
    bool colorMode=0;
    int colorSamplingMethod=RETINA_COLOR_BAYER;
    bool useRetinaLogSampling=false;
    float reductionFactor=1.0f;
    float samplingStrenght=10.0f;
    Ptr<Retina> retval;

    const char* keywords[] = { "inputSize", "colorMode", "colorSamplingMethod", "useRetinaLogSampling", "reductionFactor", "samplingStrenght", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Ob|ibff:createRetina", (char**)keywords, &pyobj_inputSize, &colorMode, &colorSamplingMethod, &useRetinaLogSampling, &reductionFactor, &samplingStrenght) &&
        pyopencv_to(pyobj_inputSize, inputSize, ArgInfo("inputSize", 0)) )
    {
        ERRWRAP2(retval = cv::bioinspired::createRetina(inputSize, colorMode, colorSamplingMethod, useRetinaLogSampling, reductionFactor, samplingStrenght));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_bioinspired_createRetinaFastToneMapping(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::bioinspired;

    PyObject* pyobj_inputSize = NULL;
    Size inputSize;
    Ptr<RetinaFastToneMapping> retval;

    const char* keywords[] = { "inputSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:createRetinaFastToneMapping", (char**)keywords, &pyobj_inputSize) &&
        pyopencv_to(pyobj_inputSize, inputSize, ArgInfo("inputSize", 0)) )
    {
        ERRWRAP2(retval = cv::bioinspired::createRetinaFastToneMapping(inputSize));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_bioinspired_createTransientAreasSegmentationModule(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::bioinspired;

    PyObject* pyobj_inputSize = NULL;
    Size inputSize;
    Ptr<TransientAreasSegmentationModule> retval;

    const char* keywords[] = { "inputSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:createTransientAreasSegmentationModule", (char**)keywords, &pyobj_inputSize) &&
        pyopencv_to(pyobj_inputSize, inputSize, ArgInfo("inputSize", 0)) )
    {
        ERRWRAP2(retval = cv::bioinspired::createTransientAreasSegmentationModule(inputSize));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_AbsLayer_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    Ptr<AbsLayer> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::dnn::AbsLayer::create());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_BNLLLayer_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    Ptr<BNLLLayer> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::dnn::BNLLLayer::create());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_BatchNormLayer_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    bool hasWeights=0;
    bool hasBias=0;
    float epsilon=0.f;
    Ptr<BatchNormLayer> retval;

    const char* keywords[] = { "hasWeights", "hasBias", "epsilon", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "bbf:BatchNormLayer_create", (char**)keywords, &hasWeights, &hasBias, &epsilon) )
    {
        ERRWRAP2(retval = cv::dnn::BatchNormLayer::create(hasWeights, hasBias, epsilon));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_ChannelsPReLULayer_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    Ptr<ChannelsPReLULayer> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::dnn::ChannelsPReLULayer::create());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_ConcatLayer_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    int axis=1;
    Ptr<ConcatLayer> retval;

    const char* keywords[] = { "axis", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|i:ConcatLayer_create", (char**)keywords, &axis) )
    {
        ERRWRAP2(retval = cv::dnn::ConcatLayer::create(axis));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_ConvolutionLayer_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    PyObject* pyobj_kernel = NULL;
    Size kernel=Size(3, 3);
    PyObject* pyobj_stride = NULL;
    Size stride=Size(1, 1);
    PyObject* pyobj_pad = NULL;
    Size pad=Size(0, 0);
    PyObject* pyobj_dilation = NULL;
    Size dilation=Size(1, 1);
    Ptr<BaseConvolutionLayer> retval;

    const char* keywords[] = { "kernel", "stride", "pad", "dilation", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|OOOO:ConvolutionLayer_create", (char**)keywords, &pyobj_kernel, &pyobj_stride, &pyobj_pad, &pyobj_dilation) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 0)) &&
        pyopencv_to(pyobj_stride, stride, ArgInfo("stride", 0)) &&
        pyopencv_to(pyobj_pad, pad, ArgInfo("pad", 0)) &&
        pyopencv_to(pyobj_dilation, dilation, ArgInfo("dilation", 0)) )
    {
        ERRWRAP2(retval = cv::dnn::ConvolutionLayer::create(kernel, stride, pad, dilation));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_DeconvolutionLayer_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    PyObject* pyobj_kernel = NULL;
    Size kernel=Size(3, 3);
    PyObject* pyobj_stride = NULL;
    Size stride=Size(1, 1);
    PyObject* pyobj_pad = NULL;
    Size pad=Size(0, 0);
    PyObject* pyobj_dilation = NULL;
    Size dilation=Size(1, 1);
    PyObject* pyobj_adjustPad = NULL;
    Size adjustPad;
    Ptr<BaseConvolutionLayer> retval;

    const char* keywords[] = { "kernel", "stride", "pad", "dilation", "adjustPad", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|OOOOO:DeconvolutionLayer_create", (char**)keywords, &pyobj_kernel, &pyobj_stride, &pyobj_pad, &pyobj_dilation, &pyobj_adjustPad) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 0)) &&
        pyopencv_to(pyobj_stride, stride, ArgInfo("stride", 0)) &&
        pyopencv_to(pyobj_pad, pad, ArgInfo("pad", 0)) &&
        pyopencv_to(pyobj_dilation, dilation, ArgInfo("dilation", 0)) &&
        pyopencv_to(pyobj_adjustPad, adjustPad, ArgInfo("adjustPad", 0)) )
    {
        ERRWRAP2(retval = cv::dnn::DeconvolutionLayer::create(kernel, stride, pad, dilation, adjustPad));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_InnerProductLayer_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    int axis=1;
    Ptr<InnerProductLayer> retval;

    const char* keywords[] = { "axis", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|i:InnerProductLayer_create", (char**)keywords, &axis) )
    {
        ERRWRAP2(retval = cv::dnn::InnerProductLayer::create(axis));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_LRNLayer_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    int type=LRNLayer::CHANNEL_NRM;
    int size=5;
    double alpha=1;
    double beta=0.75;
    double bias=1;
    bool normBySize=true;
    Ptr<LRNLayer> retval;

    const char* keywords[] = { "type", "size", "alpha", "beta", "bias", "normBySize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|iidddb:LRNLayer_create", (char**)keywords, &type, &size, &alpha, &beta, &bias, &normBySize) )
    {
        ERRWRAP2(retval = cv::dnn::LRNLayer::create(type, size, alpha, beta, bias, normBySize));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_LSTMLayer_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    Ptr<LSTMLayer> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::dnn::LSTMLayer::create());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_MVNLayer_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    bool normVariance=true;
    bool acrossChannels=false;
    double eps=1e-9;
    Ptr<MVNLayer> retval;

    const char* keywords[] = { "normVariance", "acrossChannels", "eps", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|bbd:MVNLayer_create", (char**)keywords, &normVariance, &acrossChannels, &eps) )
    {
        ERRWRAP2(retval = cv::dnn::MVNLayer::create(normVariance, acrossChannels, eps));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_MaxUnpoolLayer_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    PyObject* pyobj_unpoolSize = NULL;
    Size unpoolSize;
    Ptr<MaxUnpoolLayer> retval;

    const char* keywords[] = { "unpoolSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:MaxUnpoolLayer_create", (char**)keywords, &pyobj_unpoolSize) &&
        pyopencv_to(pyobj_unpoolSize, unpoolSize, ArgInfo("unpoolSize", 0)) )
    {
        ERRWRAP2(retval = cv::dnn::MaxUnpoolLayer::create(unpoolSize));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_dnn_Net_Net(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    pyopencv_dnn_Net_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_dnn_Net_t, &pyopencv_dnn_Net_Type);
        if(self) ERRWRAP2(new (&(self->v)) cv::dnn::Net());
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_PoolingLayer_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    int type=PoolingLayer::MAX;
    PyObject* pyobj_kernel = NULL;
    Size kernel=Size(2, 2);
    PyObject* pyobj_stride = NULL;
    Size stride=Size(1, 1);
    PyObject* pyobj_pad = NULL;
    Size pad=Size(0, 0);
    PyObject* pyobj_padMode = NULL;
    String padMode="";
    Ptr<PoolingLayer> retval;

    const char* keywords[] = { "type", "kernel", "stride", "pad", "padMode", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|iOOOO:PoolingLayer_create", (char**)keywords, &type, &pyobj_kernel, &pyobj_stride, &pyobj_pad, &pyobj_padMode) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 0)) &&
        pyopencv_to(pyobj_stride, stride, ArgInfo("stride", 0)) &&
        pyopencv_to(pyobj_pad, pad, ArgInfo("pad", 0)) &&
        pyopencv_to(pyobj_padMode, padMode, ArgInfo("padMode", 0)) )
    {
        ERRWRAP2(retval = cv::dnn::PoolingLayer::create(type, kernel, stride, pad, padMode));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_PoolingLayer_createGlobal(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    int type=PoolingLayer::MAX;
    Ptr<PoolingLayer> retval;

    const char* keywords[] = { "type", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|i:PoolingLayer_createGlobal", (char**)keywords, &type) )
    {
        ERRWRAP2(retval = cv::dnn::PoolingLayer::createGlobal(type));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_PowerLayer_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    double power=1;
    double scale=1;
    double shift=0;
    Ptr<PowerLayer> retval;

    const char* keywords[] = { "power", "scale", "shift", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|ddd:PowerLayer_create", (char**)keywords, &power, &scale, &shift) )
    {
        ERRWRAP2(retval = cv::dnn::PowerLayer::create(power, scale, shift));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_RNNLayer_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    Ptr<RNNLayer> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::dnn::RNNLayer::create());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_ReLULayer_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    double negativeSlope=0;
    Ptr<ReLULayer> retval;

    const char* keywords[] = { "negativeSlope", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|d:ReLULayer_create", (char**)keywords, &negativeSlope) )
    {
        ERRWRAP2(retval = cv::dnn::ReLULayer::create(negativeSlope));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_ReshapeLayer_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    PyObject* pyobj_newShape = NULL;
    BlobShape newShape;
    PyObject* pyobj_applyingRange = NULL;
    Range applyingRange=Range::all();
    bool enableReordering=false;
    Ptr<ReshapeLayer> retval;

    const char* keywords[] = { "newShape", "applyingRange", "enableReordering", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Ob:ReshapeLayer_create", (char**)keywords, &pyobj_newShape, &pyobj_applyingRange, &enableReordering) &&
        pyopencv_to(pyobj_newShape, newShape, ArgInfo("newShape", 0)) &&
        pyopencv_to(pyobj_applyingRange, applyingRange, ArgInfo("applyingRange", 0)) )
    {
        ERRWRAP2(retval = cv::dnn::ReshapeLayer::create(newShape, applyingRange, enableReordering));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_ScaleLayer_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    bool hasBias=0;
    Ptr<ScaleLayer> retval;

    const char* keywords[] = { "hasBias", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:ScaleLayer_create", (char**)keywords, &hasBias) )
    {
        ERRWRAP2(retval = cv::dnn::ScaleLayer::create(hasBias));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_SigmoidLayer_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    Ptr<SigmoidLayer> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::dnn::SigmoidLayer::create());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_SliceLayer_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    {
    int axis=0;
    Ptr<SliceLayer> retval;

    const char* keywords[] = { "axis", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:SliceLayer_create", (char**)keywords, &axis) )
    {
        ERRWRAP2(retval = cv::dnn::SliceLayer::create(axis));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    int axis=0;
    PyObject* pyobj_sliceIndices = NULL;
    vector_int sliceIndices;
    Ptr<SliceLayer> retval;

    const char* keywords[] = { "axis", "sliceIndices", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "iO:SliceLayer_create", (char**)keywords, &axis, &pyobj_sliceIndices) &&
        pyopencv_to(pyobj_sliceIndices, sliceIndices, ArgInfo("sliceIndices", 0)) )
    {
        ERRWRAP2(retval = cv::dnn::SliceLayer::create(axis, sliceIndices));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_SoftmaxLayer_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    int axis=1;
    Ptr<SoftmaxLayer> retval;

    const char* keywords[] = { "axis", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|i:SoftmaxLayer_create", (char**)keywords, &axis) )
    {
        ERRWRAP2(retval = cv::dnn::SoftmaxLayer::create(axis));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_SplitLayer_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    int outputsCount=-1;
    Ptr<SplitLayer> retval;

    const char* keywords[] = { "outputsCount", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|i:SplitLayer_create", (char**)keywords, &outputsCount) )
    {
        ERRWRAP2(retval = cv::dnn::SplitLayer::create(outputsCount));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_TanHLayer_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    Ptr<TanHLayer> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::dnn::TanHLayer::create());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_createCaffeImporter(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    PyObject* pyobj_prototxt = NULL;
    String prototxt;
    PyObject* pyobj_caffeModel = NULL;
    String caffeModel;
    Ptr<Importer> retval;

    const char* keywords[] = { "prototxt", "caffeModel", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:createCaffeImporter", (char**)keywords, &pyobj_prototxt, &pyobj_caffeModel) &&
        pyopencv_to(pyobj_prototxt, prototxt, ArgInfo("prototxt", 0)) &&
        pyopencv_to(pyobj_caffeModel, caffeModel, ArgInfo("caffeModel", 0)) )
    {
        ERRWRAP2(retval = cv::dnn::createCaffeImporter(prototxt, caffeModel));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_createTorchImporter(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    PyObject* pyobj_filename = NULL;
    String filename;
    bool isBinary=true;
    Ptr<Importer> retval;

    const char* keywords[] = { "filename", "isBinary", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|b:createTorchImporter", (char**)keywords, &pyobj_filename, &isBinary) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) )
    {
        ERRWRAP2(retval = cv::dnn::createTorchImporter(filename, isBinary));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_initModule(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;


    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(cv::dnn::initModule());
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_readNetFromCaffe(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    PyObject* pyobj_prototxt = NULL;
    String prototxt;
    PyObject* pyobj_caffeModel = NULL;
    String caffeModel;
    Net retval;

    const char* keywords[] = { "prototxt", "caffeModel", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:readNetFromCaffe", (char**)keywords, &pyobj_prototxt, &pyobj_caffeModel) &&
        pyopencv_to(pyobj_prototxt, prototxt, ArgInfo("prototxt", 0)) &&
        pyopencv_to(pyobj_caffeModel, caffeModel, ArgInfo("caffeModel", 0)) )
    {
        ERRWRAP2(retval = cv::dnn::readNetFromCaffe(prototxt, caffeModel));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_dnn_readTorchBlob(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::dnn;

    PyObject* pyobj_filename = NULL;
    String filename;
    bool isBinary=true;
    Blob retval;

    const char* keywords[] = { "filename", "isBinary", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|b:readTorchBlob", (char**)keywords, &pyobj_filename, &isBinary) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) )
    {
        ERRWRAP2(retval = cv::dnn::readTorchBlob(filename, isBinary));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_face_StandardCollector_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::face;

    double threshold=DBL_MAX;
    Ptr<StandardCollector> retval;

    const char* keywords[] = { "threshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|d:StandardCollector_create", (char**)keywords, &threshold) )
    {
        ERRWRAP2(retval = cv::face::StandardCollector::create(threshold));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_face_createBIF(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::face;

    int num_bands=8;
    int num_rotations=12;
    cv::Ptr<BIF> retval;

    const char* keywords[] = { "num_bands", "num_rotations", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|ii:createBIF", (char**)keywords, &num_bands, &num_rotations) )
    {
        ERRWRAP2(retval = cv::face::createBIF(num_bands, num_rotations));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_face_createEigenFaceRecognizer(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::face;

    int num_components=0;
    double threshold=DBL_MAX;
    Ptr<BasicFaceRecognizer> retval;

    const char* keywords[] = { "num_components", "threshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|id:createEigenFaceRecognizer", (char**)keywords, &num_components, &threshold) )
    {
        ERRWRAP2(retval = cv::face::createEigenFaceRecognizer(num_components, threshold));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_face_createFisherFaceRecognizer(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::face;

    int num_components=0;
    double threshold=DBL_MAX;
    Ptr<BasicFaceRecognizer> retval;

    const char* keywords[] = { "num_components", "threshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|id:createFisherFaceRecognizer", (char**)keywords, &num_components, &threshold) )
    {
        ERRWRAP2(retval = cv::face::createFisherFaceRecognizer(num_components, threshold));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_face_createLBPHFaceRecognizer(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::face;

    int radius=1;
    int neighbors=8;
    int grid_x=8;
    int grid_y=8;
    double threshold=DBL_MAX;
    Ptr<LBPHFaceRecognizer> retval;

    const char* keywords[] = { "radius", "neighbors", "grid_x", "grid_y", "threshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|iiiid:createLBPHFaceRecognizer", (char**)keywords, &radius, &neighbors, &grid_x, &grid_y, &threshold) )
    {
        ERRWRAP2(retval = cv::face::createLBPHFaceRecognizer(radius, neighbors, grid_x, grid_y, threshold));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_fisheye_calibrate(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::fisheye;

    {
    PyObject* pyobj_objectPoints = NULL;
    vector_Mat objectPoints;
    PyObject* pyobj_imagePoints = NULL;
    vector_Mat imagePoints;
    PyObject* pyobj_image_size = NULL;
    Size image_size;
    PyObject* pyobj_K = NULL;
    Mat K;
    PyObject* pyobj_D = NULL;
    Mat D;
    PyObject* pyobj_rvecs = NULL;
    vector_Mat rvecs;
    PyObject* pyobj_tvecs = NULL;
    vector_Mat tvecs;
    int flags=0;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria=TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON);
    double retval;

    const char* keywords[] = { "objectPoints", "imagePoints", "image_size", "K", "D", "rvecs", "tvecs", "flags", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOO|OOiO:calibrate", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints, &pyobj_image_size, &pyobj_K, &pyobj_D, &pyobj_rvecs, &pyobj_tvecs, &flags, &pyobj_criteria) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints, imagePoints, ArgInfo("imagePoints", 0)) &&
        pyopencv_to(pyobj_image_size, image_size, ArgInfo("image_size", 0)) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 1)) &&
        pyopencv_to(pyobj_D, D, ArgInfo("D", 1)) &&
        pyopencv_to(pyobj_rvecs, rvecs, ArgInfo("rvecs", 1)) &&
        pyopencv_to(pyobj_tvecs, tvecs, ArgInfo("tvecs", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(retval = cv::fisheye::calibrate(objectPoints, imagePoints, image_size, K, D, rvecs, tvecs, flags, criteria));
        return Py_BuildValue("(NNNNN)", pyopencv_from(retval), pyopencv_from(K), pyopencv_from(D), pyopencv_from(rvecs), pyopencv_from(tvecs));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_objectPoints = NULL;
    vector_Mat objectPoints;
    PyObject* pyobj_imagePoints = NULL;
    vector_Mat imagePoints;
    PyObject* pyobj_image_size = NULL;
    Size image_size;
    PyObject* pyobj_K = NULL;
    UMat K;
    PyObject* pyobj_D = NULL;
    UMat D;
    PyObject* pyobj_rvecs = NULL;
    vector_Mat rvecs;
    PyObject* pyobj_tvecs = NULL;
    vector_Mat tvecs;
    int flags=0;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria=TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON);
    double retval;

    const char* keywords[] = { "objectPoints", "imagePoints", "image_size", "K", "D", "rvecs", "tvecs", "flags", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOO|OOiO:calibrate", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints, &pyobj_image_size, &pyobj_K, &pyobj_D, &pyobj_rvecs, &pyobj_tvecs, &flags, &pyobj_criteria) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints, imagePoints, ArgInfo("imagePoints", 0)) &&
        pyopencv_to(pyobj_image_size, image_size, ArgInfo("image_size", 0)) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 1)) &&
        pyopencv_to(pyobj_D, D, ArgInfo("D", 1)) &&
        pyopencv_to(pyobj_rvecs, rvecs, ArgInfo("rvecs", 1)) &&
        pyopencv_to(pyobj_tvecs, tvecs, ArgInfo("tvecs", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(retval = cv::fisheye::calibrate(objectPoints, imagePoints, image_size, K, D, rvecs, tvecs, flags, criteria));
        return Py_BuildValue("(NNNNN)", pyopencv_from(retval), pyopencv_from(K), pyopencv_from(D), pyopencv_from(rvecs), pyopencv_from(tvecs));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_fisheye_distortPoints(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::fisheye;

    {
    PyObject* pyobj_undistorted = NULL;
    Mat undistorted;
    PyObject* pyobj_distorted = NULL;
    Mat distorted;
    PyObject* pyobj_K = NULL;
    Mat K;
    PyObject* pyobj_D = NULL;
    Mat D;
    double alpha=0;

    const char* keywords[] = { "undistorted", "K", "D", "distorted", "alpha", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|Od:distortPoints", (char**)keywords, &pyobj_undistorted, &pyobj_K, &pyobj_D, &pyobj_distorted, &alpha) &&
        pyopencv_to(pyobj_undistorted, undistorted, ArgInfo("undistorted", 0)) &&
        pyopencv_to(pyobj_distorted, distorted, ArgInfo("distorted", 1)) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 0)) &&
        pyopencv_to(pyobj_D, D, ArgInfo("D", 0)) )
    {
        ERRWRAP2(cv::fisheye::distortPoints(undistorted, distorted, K, D, alpha));
        return pyopencv_from(distorted);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_undistorted = NULL;
    UMat undistorted;
    PyObject* pyobj_distorted = NULL;
    UMat distorted;
    PyObject* pyobj_K = NULL;
    UMat K;
    PyObject* pyobj_D = NULL;
    UMat D;
    double alpha=0;

    const char* keywords[] = { "undistorted", "K", "D", "distorted", "alpha", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|Od:distortPoints", (char**)keywords, &pyobj_undistorted, &pyobj_K, &pyobj_D, &pyobj_distorted, &alpha) &&
        pyopencv_to(pyobj_undistorted, undistorted, ArgInfo("undistorted", 0)) &&
        pyopencv_to(pyobj_distorted, distorted, ArgInfo("distorted", 1)) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 0)) &&
        pyopencv_to(pyobj_D, D, ArgInfo("D", 0)) )
    {
        ERRWRAP2(cv::fisheye::distortPoints(undistorted, distorted, K, D, alpha));
        return pyopencv_from(distorted);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_fisheye_estimateNewCameraMatrixForUndistortRectify(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::fisheye;

    {
    PyObject* pyobj_K = NULL;
    Mat K;
    PyObject* pyobj_D = NULL;
    Mat D;
    PyObject* pyobj_image_size = NULL;
    Size image_size;
    PyObject* pyobj_R = NULL;
    Mat R;
    PyObject* pyobj_P = NULL;
    Mat P;
    double balance=0.0;
    PyObject* pyobj_new_size = NULL;
    Size new_size;
    double fov_scale=1.0;

    const char* keywords[] = { "K", "D", "image_size", "R", "P", "balance", "new_size", "fov_scale", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OdOd:estimateNewCameraMatrixForUndistortRectify", (char**)keywords, &pyobj_K, &pyobj_D, &pyobj_image_size, &pyobj_R, &pyobj_P, &balance, &pyobj_new_size, &fov_scale) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 0)) &&
        pyopencv_to(pyobj_D, D, ArgInfo("D", 0)) &&
        pyopencv_to(pyobj_image_size, image_size, ArgInfo("image_size", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 0)) &&
        pyopencv_to(pyobj_P, P, ArgInfo("P", 1)) &&
        pyopencv_to(pyobj_new_size, new_size, ArgInfo("new_size", 0)) )
    {
        ERRWRAP2(cv::fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, image_size, R, P, balance, new_size, fov_scale));
        return pyopencv_from(P);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_K = NULL;
    UMat K;
    PyObject* pyobj_D = NULL;
    UMat D;
    PyObject* pyobj_image_size = NULL;
    Size image_size;
    PyObject* pyobj_R = NULL;
    UMat R;
    PyObject* pyobj_P = NULL;
    UMat P;
    double balance=0.0;
    PyObject* pyobj_new_size = NULL;
    Size new_size;
    double fov_scale=1.0;

    const char* keywords[] = { "K", "D", "image_size", "R", "P", "balance", "new_size", "fov_scale", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OdOd:estimateNewCameraMatrixForUndistortRectify", (char**)keywords, &pyobj_K, &pyobj_D, &pyobj_image_size, &pyobj_R, &pyobj_P, &balance, &pyobj_new_size, &fov_scale) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 0)) &&
        pyopencv_to(pyobj_D, D, ArgInfo("D", 0)) &&
        pyopencv_to(pyobj_image_size, image_size, ArgInfo("image_size", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 0)) &&
        pyopencv_to(pyobj_P, P, ArgInfo("P", 1)) &&
        pyopencv_to(pyobj_new_size, new_size, ArgInfo("new_size", 0)) )
    {
        ERRWRAP2(cv::fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, image_size, R, P, balance, new_size, fov_scale));
        return pyopencv_from(P);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_fisheye_initUndistortRectifyMap(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::fisheye;

    {
    PyObject* pyobj_K = NULL;
    Mat K;
    PyObject* pyobj_D = NULL;
    Mat D;
    PyObject* pyobj_R = NULL;
    Mat R;
    PyObject* pyobj_P = NULL;
    Mat P;
    PyObject* pyobj_size = NULL;
    Size size;
    int m1type=0;
    PyObject* pyobj_map1 = NULL;
    Mat map1;
    PyObject* pyobj_map2 = NULL;
    Mat map2;

    const char* keywords[] = { "K", "D", "R", "P", "size", "m1type", "map1", "map2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOi|OO:initUndistortRectifyMap", (char**)keywords, &pyobj_K, &pyobj_D, &pyobj_R, &pyobj_P, &pyobj_size, &m1type, &pyobj_map1, &pyobj_map2) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 0)) &&
        pyopencv_to(pyobj_D, D, ArgInfo("D", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 0)) &&
        pyopencv_to(pyobj_P, P, ArgInfo("P", 0)) &&
        pyopencv_to(pyobj_size, size, ArgInfo("size", 0)) &&
        pyopencv_to(pyobj_map1, map1, ArgInfo("map1", 1)) &&
        pyopencv_to(pyobj_map2, map2, ArgInfo("map2", 1)) )
    {
        ERRWRAP2(cv::fisheye::initUndistortRectifyMap(K, D, R, P, size, m1type, map1, map2));
        return Py_BuildValue("(NN)", pyopencv_from(map1), pyopencv_from(map2));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_K = NULL;
    UMat K;
    PyObject* pyobj_D = NULL;
    UMat D;
    PyObject* pyobj_R = NULL;
    UMat R;
    PyObject* pyobj_P = NULL;
    UMat P;
    PyObject* pyobj_size = NULL;
    Size size;
    int m1type=0;
    PyObject* pyobj_map1 = NULL;
    UMat map1;
    PyObject* pyobj_map2 = NULL;
    UMat map2;

    const char* keywords[] = { "K", "D", "R", "P", "size", "m1type", "map1", "map2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOi|OO:initUndistortRectifyMap", (char**)keywords, &pyobj_K, &pyobj_D, &pyobj_R, &pyobj_P, &pyobj_size, &m1type, &pyobj_map1, &pyobj_map2) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 0)) &&
        pyopencv_to(pyobj_D, D, ArgInfo("D", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 0)) &&
        pyopencv_to(pyobj_P, P, ArgInfo("P", 0)) &&
        pyopencv_to(pyobj_size, size, ArgInfo("size", 0)) &&
        pyopencv_to(pyobj_map1, map1, ArgInfo("map1", 1)) &&
        pyopencv_to(pyobj_map2, map2, ArgInfo("map2", 1)) )
    {
        ERRWRAP2(cv::fisheye::initUndistortRectifyMap(K, D, R, P, size, m1type, map1, map2));
        return Py_BuildValue("(NN)", pyopencv_from(map1), pyopencv_from(map2));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_fisheye_projectPoints(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::fisheye;

    {
    PyObject* pyobj_objectPoints = NULL;
    Mat objectPoints;
    PyObject* pyobj_imagePoints = NULL;
    Mat imagePoints;
    PyObject* pyobj_rvec = NULL;
    Mat rvec;
    PyObject* pyobj_tvec = NULL;
    Mat tvec;
    PyObject* pyobj_K = NULL;
    Mat K;
    PyObject* pyobj_D = NULL;
    Mat D;
    double alpha=0;
    PyObject* pyobj_jacobian = NULL;
    Mat jacobian;

    const char* keywords[] = { "objectPoints", "rvec", "tvec", "K", "D", "imagePoints", "alpha", "jacobian", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOO|OdO:projectPoints", (char**)keywords, &pyobj_objectPoints, &pyobj_rvec, &pyobj_tvec, &pyobj_K, &pyobj_D, &pyobj_imagePoints, &alpha, &pyobj_jacobian) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints, imagePoints, ArgInfo("imagePoints", 1)) &&
        pyopencv_to(pyobj_rvec, rvec, ArgInfo("rvec", 0)) &&
        pyopencv_to(pyobj_tvec, tvec, ArgInfo("tvec", 0)) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 0)) &&
        pyopencv_to(pyobj_D, D, ArgInfo("D", 0)) &&
        pyopencv_to(pyobj_jacobian, jacobian, ArgInfo("jacobian", 1)) )
    {
        ERRWRAP2(cv::fisheye::projectPoints(objectPoints, imagePoints, rvec, tvec, K, D, alpha, jacobian));
        return Py_BuildValue("(NN)", pyopencv_from(imagePoints), pyopencv_from(jacobian));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_objectPoints = NULL;
    UMat objectPoints;
    PyObject* pyobj_imagePoints = NULL;
    UMat imagePoints;
    PyObject* pyobj_rvec = NULL;
    UMat rvec;
    PyObject* pyobj_tvec = NULL;
    UMat tvec;
    PyObject* pyobj_K = NULL;
    UMat K;
    PyObject* pyobj_D = NULL;
    UMat D;
    double alpha=0;
    PyObject* pyobj_jacobian = NULL;
    UMat jacobian;

    const char* keywords[] = { "objectPoints", "rvec", "tvec", "K", "D", "imagePoints", "alpha", "jacobian", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOO|OdO:projectPoints", (char**)keywords, &pyobj_objectPoints, &pyobj_rvec, &pyobj_tvec, &pyobj_K, &pyobj_D, &pyobj_imagePoints, &alpha, &pyobj_jacobian) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints, imagePoints, ArgInfo("imagePoints", 1)) &&
        pyopencv_to(pyobj_rvec, rvec, ArgInfo("rvec", 0)) &&
        pyopencv_to(pyobj_tvec, tvec, ArgInfo("tvec", 0)) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 0)) &&
        pyopencv_to(pyobj_D, D, ArgInfo("D", 0)) &&
        pyopencv_to(pyobj_jacobian, jacobian, ArgInfo("jacobian", 1)) )
    {
        ERRWRAP2(cv::fisheye::projectPoints(objectPoints, imagePoints, rvec, tvec, K, D, alpha, jacobian));
        return Py_BuildValue("(NN)", pyopencv_from(imagePoints), pyopencv_from(jacobian));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_fisheye_stereoCalibrate(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::fisheye;

    {
    PyObject* pyobj_objectPoints = NULL;
    vector_Mat objectPoints;
    PyObject* pyobj_imagePoints1 = NULL;
    vector_Mat imagePoints1;
    PyObject* pyobj_imagePoints2 = NULL;
    vector_Mat imagePoints2;
    PyObject* pyobj_K1 = NULL;
    Mat K1;
    PyObject* pyobj_D1 = NULL;
    Mat D1;
    PyObject* pyobj_K2 = NULL;
    Mat K2;
    PyObject* pyobj_D2 = NULL;
    Mat D2;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    PyObject* pyobj_R = NULL;
    Mat R;
    PyObject* pyobj_T = NULL;
    Mat T;
    int flags=fisheye::CALIB_FIX_INTRINSIC;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria=TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON);
    double retval;

    const char* keywords[] = { "objectPoints", "imagePoints1", "imagePoints2", "K1", "D1", "K2", "D2", "imageSize", "R", "T", "flags", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOOO|OOiO:stereoCalibrate", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints1, &pyobj_imagePoints2, &pyobj_K1, &pyobj_D1, &pyobj_K2, &pyobj_D2, &pyobj_imageSize, &pyobj_R, &pyobj_T, &flags, &pyobj_criteria) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints1, imagePoints1, ArgInfo("imagePoints1", 0)) &&
        pyopencv_to(pyobj_imagePoints2, imagePoints2, ArgInfo("imagePoints2", 0)) &&
        pyopencv_to(pyobj_K1, K1, ArgInfo("K1", 1)) &&
        pyopencv_to(pyobj_D1, D1, ArgInfo("D1", 1)) &&
        pyopencv_to(pyobj_K2, K2, ArgInfo("K2", 1)) &&
        pyopencv_to(pyobj_D2, D2, ArgInfo("D2", 1)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 1)) &&
        pyopencv_to(pyobj_T, T, ArgInfo("T", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(retval = cv::fisheye::stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, R, T, flags, criteria));
        return Py_BuildValue("(NNNNNNN)", pyopencv_from(retval), pyopencv_from(K1), pyopencv_from(D1), pyopencv_from(K2), pyopencv_from(D2), pyopencv_from(R), pyopencv_from(T));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_objectPoints = NULL;
    vector_Mat objectPoints;
    PyObject* pyobj_imagePoints1 = NULL;
    vector_Mat imagePoints1;
    PyObject* pyobj_imagePoints2 = NULL;
    vector_Mat imagePoints2;
    PyObject* pyobj_K1 = NULL;
    UMat K1;
    PyObject* pyobj_D1 = NULL;
    UMat D1;
    PyObject* pyobj_K2 = NULL;
    UMat K2;
    PyObject* pyobj_D2 = NULL;
    UMat D2;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    PyObject* pyobj_R = NULL;
    UMat R;
    PyObject* pyobj_T = NULL;
    UMat T;
    int flags=fisheye::CALIB_FIX_INTRINSIC;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria=TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON);
    double retval;

    const char* keywords[] = { "objectPoints", "imagePoints1", "imagePoints2", "K1", "D1", "K2", "D2", "imageSize", "R", "T", "flags", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOOO|OOiO:stereoCalibrate", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints1, &pyobj_imagePoints2, &pyobj_K1, &pyobj_D1, &pyobj_K2, &pyobj_D2, &pyobj_imageSize, &pyobj_R, &pyobj_T, &flags, &pyobj_criteria) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints1, imagePoints1, ArgInfo("imagePoints1", 0)) &&
        pyopencv_to(pyobj_imagePoints2, imagePoints2, ArgInfo("imagePoints2", 0)) &&
        pyopencv_to(pyobj_K1, K1, ArgInfo("K1", 1)) &&
        pyopencv_to(pyobj_D1, D1, ArgInfo("D1", 1)) &&
        pyopencv_to(pyobj_K2, K2, ArgInfo("K2", 1)) &&
        pyopencv_to(pyobj_D2, D2, ArgInfo("D2", 1)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 1)) &&
        pyopencv_to(pyobj_T, T, ArgInfo("T", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2(retval = cv::fisheye::stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, R, T, flags, criteria));
        return Py_BuildValue("(NNNNNNN)", pyopencv_from(retval), pyopencv_from(K1), pyopencv_from(D1), pyopencv_from(K2), pyopencv_from(D2), pyopencv_from(R), pyopencv_from(T));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_fisheye_stereoRectify(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::fisheye;

    {
    PyObject* pyobj_K1 = NULL;
    Mat K1;
    PyObject* pyobj_D1 = NULL;
    Mat D1;
    PyObject* pyobj_K2 = NULL;
    Mat K2;
    PyObject* pyobj_D2 = NULL;
    Mat D2;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    PyObject* pyobj_R = NULL;
    Mat R;
    PyObject* pyobj_tvec = NULL;
    Mat tvec;
    PyObject* pyobj_R1 = NULL;
    Mat R1;
    PyObject* pyobj_R2 = NULL;
    Mat R2;
    PyObject* pyobj_P1 = NULL;
    Mat P1;
    PyObject* pyobj_P2 = NULL;
    Mat P2;
    PyObject* pyobj_Q = NULL;
    Mat Q;
    int flags=0;
    PyObject* pyobj_newImageSize = NULL;
    Size newImageSize;
    double balance=0.0;
    double fov_scale=1.0;

    const char* keywords[] = { "K1", "D1", "K2", "D2", "imageSize", "R", "tvec", "flags", "R1", "R2", "P1", "P2", "Q", "newImageSize", "balance", "fov_scale", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOOi|OOOOOOdd:stereoRectify", (char**)keywords, &pyobj_K1, &pyobj_D1, &pyobj_K2, &pyobj_D2, &pyobj_imageSize, &pyobj_R, &pyobj_tvec, &flags, &pyobj_R1, &pyobj_R2, &pyobj_P1, &pyobj_P2, &pyobj_Q, &pyobj_newImageSize, &balance, &fov_scale) &&
        pyopencv_to(pyobj_K1, K1, ArgInfo("K1", 0)) &&
        pyopencv_to(pyobj_D1, D1, ArgInfo("D1", 0)) &&
        pyopencv_to(pyobj_K2, K2, ArgInfo("K2", 0)) &&
        pyopencv_to(pyobj_D2, D2, ArgInfo("D2", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 0)) &&
        pyopencv_to(pyobj_tvec, tvec, ArgInfo("tvec", 0)) &&
        pyopencv_to(pyobj_R1, R1, ArgInfo("R1", 1)) &&
        pyopencv_to(pyobj_R2, R2, ArgInfo("R2", 1)) &&
        pyopencv_to(pyobj_P1, P1, ArgInfo("P1", 1)) &&
        pyopencv_to(pyobj_P2, P2, ArgInfo("P2", 1)) &&
        pyopencv_to(pyobj_Q, Q, ArgInfo("Q", 1)) &&
        pyopencv_to(pyobj_newImageSize, newImageSize, ArgInfo("newImageSize", 0)) )
    {
        ERRWRAP2(cv::fisheye::stereoRectify(K1, D1, K2, D2, imageSize, R, tvec, R1, R2, P1, P2, Q, flags, newImageSize, balance, fov_scale));
        return Py_BuildValue("(NNNNN)", pyopencv_from(R1), pyopencv_from(R2), pyopencv_from(P1), pyopencv_from(P2), pyopencv_from(Q));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_K1 = NULL;
    UMat K1;
    PyObject* pyobj_D1 = NULL;
    UMat D1;
    PyObject* pyobj_K2 = NULL;
    UMat K2;
    PyObject* pyobj_D2 = NULL;
    UMat D2;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    PyObject* pyobj_R = NULL;
    UMat R;
    PyObject* pyobj_tvec = NULL;
    UMat tvec;
    PyObject* pyobj_R1 = NULL;
    UMat R1;
    PyObject* pyobj_R2 = NULL;
    UMat R2;
    PyObject* pyobj_P1 = NULL;
    UMat P1;
    PyObject* pyobj_P2 = NULL;
    UMat P2;
    PyObject* pyobj_Q = NULL;
    UMat Q;
    int flags=0;
    PyObject* pyobj_newImageSize = NULL;
    Size newImageSize;
    double balance=0.0;
    double fov_scale=1.0;

    const char* keywords[] = { "K1", "D1", "K2", "D2", "imageSize", "R", "tvec", "flags", "R1", "R2", "P1", "P2", "Q", "newImageSize", "balance", "fov_scale", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOOi|OOOOOOdd:stereoRectify", (char**)keywords, &pyobj_K1, &pyobj_D1, &pyobj_K2, &pyobj_D2, &pyobj_imageSize, &pyobj_R, &pyobj_tvec, &flags, &pyobj_R1, &pyobj_R2, &pyobj_P1, &pyobj_P2, &pyobj_Q, &pyobj_newImageSize, &balance, &fov_scale) &&
        pyopencv_to(pyobj_K1, K1, ArgInfo("K1", 0)) &&
        pyopencv_to(pyobj_D1, D1, ArgInfo("D1", 0)) &&
        pyopencv_to(pyobj_K2, K2, ArgInfo("K2", 0)) &&
        pyopencv_to(pyobj_D2, D2, ArgInfo("D2", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 0)) &&
        pyopencv_to(pyobj_tvec, tvec, ArgInfo("tvec", 0)) &&
        pyopencv_to(pyobj_R1, R1, ArgInfo("R1", 1)) &&
        pyopencv_to(pyobj_R2, R2, ArgInfo("R2", 1)) &&
        pyopencv_to(pyobj_P1, P1, ArgInfo("P1", 1)) &&
        pyopencv_to(pyobj_P2, P2, ArgInfo("P2", 1)) &&
        pyopencv_to(pyobj_Q, Q, ArgInfo("Q", 1)) &&
        pyopencv_to(pyobj_newImageSize, newImageSize, ArgInfo("newImageSize", 0)) )
    {
        ERRWRAP2(cv::fisheye::stereoRectify(K1, D1, K2, D2, imageSize, R, tvec, R1, R2, P1, P2, Q, flags, newImageSize, balance, fov_scale));
        return Py_BuildValue("(NNNNN)", pyopencv_from(R1), pyopencv_from(R2), pyopencv_from(P1), pyopencv_from(P2), pyopencv_from(Q));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_fisheye_undistortImage(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::fisheye;

    {
    PyObject* pyobj_distorted = NULL;
    Mat distorted;
    PyObject* pyobj_undistorted = NULL;
    Mat undistorted;
    PyObject* pyobj_K = NULL;
    Mat K;
    PyObject* pyobj_D = NULL;
    Mat D;
    PyObject* pyobj_Knew = NULL;
    Mat Knew=cv::Mat();
    PyObject* pyobj_new_size = NULL;
    Size new_size;

    const char* keywords[] = { "distorted", "K", "D", "undistorted", "Knew", "new_size", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|OOO:undistortImage", (char**)keywords, &pyobj_distorted, &pyobj_K, &pyobj_D, &pyobj_undistorted, &pyobj_Knew, &pyobj_new_size) &&
        pyopencv_to(pyobj_distorted, distorted, ArgInfo("distorted", 0)) &&
        pyopencv_to(pyobj_undistorted, undistorted, ArgInfo("undistorted", 1)) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 0)) &&
        pyopencv_to(pyobj_D, D, ArgInfo("D", 0)) &&
        pyopencv_to(pyobj_Knew, Knew, ArgInfo("Knew", 0)) &&
        pyopencv_to(pyobj_new_size, new_size, ArgInfo("new_size", 0)) )
    {
        ERRWRAP2(cv::fisheye::undistortImage(distorted, undistorted, K, D, Knew, new_size));
        return pyopencv_from(undistorted);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_distorted = NULL;
    UMat distorted;
    PyObject* pyobj_undistorted = NULL;
    UMat undistorted;
    PyObject* pyobj_K = NULL;
    UMat K;
    PyObject* pyobj_D = NULL;
    UMat D;
    PyObject* pyobj_Knew = NULL;
    UMat Knew=cv::UMat();
    PyObject* pyobj_new_size = NULL;
    Size new_size;

    const char* keywords[] = { "distorted", "K", "D", "undistorted", "Knew", "new_size", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|OOO:undistortImage", (char**)keywords, &pyobj_distorted, &pyobj_K, &pyobj_D, &pyobj_undistorted, &pyobj_Knew, &pyobj_new_size) &&
        pyopencv_to(pyobj_distorted, distorted, ArgInfo("distorted", 0)) &&
        pyopencv_to(pyobj_undistorted, undistorted, ArgInfo("undistorted", 1)) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 0)) &&
        pyopencv_to(pyobj_D, D, ArgInfo("D", 0)) &&
        pyopencv_to(pyobj_Knew, Knew, ArgInfo("Knew", 0)) &&
        pyopencv_to(pyobj_new_size, new_size, ArgInfo("new_size", 0)) )
    {
        ERRWRAP2(cv::fisheye::undistortImage(distorted, undistorted, K, D, Knew, new_size));
        return pyopencv_from(undistorted);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_fisheye_undistortPoints(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::fisheye;

    {
    PyObject* pyobj_distorted = NULL;
    Mat distorted;
    PyObject* pyobj_undistorted = NULL;
    Mat undistorted;
    PyObject* pyobj_K = NULL;
    Mat K;
    PyObject* pyobj_D = NULL;
    Mat D;
    PyObject* pyobj_R = NULL;
    Mat R;
    PyObject* pyobj_P = NULL;
    Mat P;

    const char* keywords[] = { "distorted", "K", "D", "undistorted", "R", "P", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|OOO:undistortPoints", (char**)keywords, &pyobj_distorted, &pyobj_K, &pyobj_D, &pyobj_undistorted, &pyobj_R, &pyobj_P) &&
        pyopencv_to(pyobj_distorted, distorted, ArgInfo("distorted", 0)) &&
        pyopencv_to(pyobj_undistorted, undistorted, ArgInfo("undistorted", 1)) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 0)) &&
        pyopencv_to(pyobj_D, D, ArgInfo("D", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 0)) &&
        pyopencv_to(pyobj_P, P, ArgInfo("P", 0)) )
    {
        ERRWRAP2(cv::fisheye::undistortPoints(distorted, undistorted, K, D, R, P));
        return pyopencv_from(undistorted);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_distorted = NULL;
    UMat distorted;
    PyObject* pyobj_undistorted = NULL;
    UMat undistorted;
    PyObject* pyobj_K = NULL;
    UMat K;
    PyObject* pyobj_D = NULL;
    UMat D;
    PyObject* pyobj_R = NULL;
    UMat R;
    PyObject* pyobj_P = NULL;
    UMat P;

    const char* keywords[] = { "distorted", "K", "D", "undistorted", "R", "P", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|OOO:undistortPoints", (char**)keywords, &pyobj_distorted, &pyobj_K, &pyobj_D, &pyobj_undistorted, &pyobj_R, &pyobj_P) &&
        pyopencv_to(pyobj_distorted, distorted, ArgInfo("distorted", 0)) &&
        pyopencv_to(pyobj_undistorted, undistorted, ArgInfo("undistorted", 1)) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 0)) &&
        pyopencv_to(pyobj_D, D, ArgInfo("D", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 0)) &&
        pyopencv_to(pyobj_P, P, ArgInfo("P", 0)) )
    {
        ERRWRAP2(cv::fisheye::undistortPoints(distorted, undistorted, K, D, R, P));
        return pyopencv_from(undistorted);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_flann_flann_Index_Index(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::flann;

    {
    pyopencv_flann_Index_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_flann_Index_t, &pyopencv_flann_Index_Type);
        new (&(self->v)) Ptr<cv::flann::Index>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::flann::Index()));
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_features = NULL;
    Mat features;
    PyObject* pyobj_params = NULL;
    IndexParams params;
    PyObject* pyobj_distType = NULL;
    cvflann_flann_distance_t distType=cvflann::FLANN_DIST_L2;
    pyopencv_flann_Index_t* self = 0;

    const char* keywords[] = { "features", "params", "distType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:Index", (char**)keywords, &pyobj_features, &pyobj_params, &pyobj_distType) &&
        pyopencv_to(pyobj_features, features, ArgInfo("features", 0)) &&
        pyopencv_to(pyobj_params, params, ArgInfo("params", 0)) &&
        pyopencv_to(pyobj_distType, distType, ArgInfo("distType", 0)) )
    {
        self = PyObject_NEW(pyopencv_flann_Index_t, &pyopencv_flann_Index_Type);
        new (&(self->v)) Ptr<cv::flann::Index>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::flann::Index(features, params, distType)));
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_features = NULL;
    UMat features;
    PyObject* pyobj_params = NULL;
    IndexParams params;
    PyObject* pyobj_distType = NULL;
    cvflann_flann_distance_t distType=cvflann::FLANN_DIST_L2;
    pyopencv_flann_Index_t* self = 0;

    const char* keywords[] = { "features", "params", "distType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:Index", (char**)keywords, &pyobj_features, &pyobj_params, &pyobj_distType) &&
        pyopencv_to(pyobj_features, features, ArgInfo("features", 0)) &&
        pyopencv_to(pyobj_params, params, ArgInfo("params", 0)) &&
        pyopencv_to(pyobj_distType, distType, ArgInfo("distType", 0)) )
    {
        self = PyObject_NEW(pyopencv_flann_Index_t, &pyopencv_flann_Index_Type);
        new (&(self->v)) Ptr<cv::flann::Index>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::flann::Index(features, params, distType)));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ft_FT02D_components(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ft;

    {
    PyObject* pyobj_matrix = NULL;
    Mat matrix;
    PyObject* pyobj_kernel = NULL;
    Mat kernel;
    PyObject* pyobj_components = NULL;
    Mat components;

    const char* keywords[] = { "matrix", "kernel", "components", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:FT02D_components", (char**)keywords, &pyobj_matrix, &pyobj_kernel, &pyobj_components) &&
        pyopencv_to(pyobj_matrix, matrix, ArgInfo("matrix", 0)) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 0)) &&
        pyopencv_to(pyobj_components, components, ArgInfo("components", 1)) )
    {
        ERRWRAP2(cv::ft::FT02D_components(matrix, kernel, components));
        return pyopencv_from(components);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_matrix = NULL;
    UMat matrix;
    PyObject* pyobj_kernel = NULL;
    UMat kernel;
    PyObject* pyobj_components = NULL;
    UMat components;

    const char* keywords[] = { "matrix", "kernel", "components", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:FT02D_components", (char**)keywords, &pyobj_matrix, &pyobj_kernel, &pyobj_components) &&
        pyopencv_to(pyobj_matrix, matrix, ArgInfo("matrix", 0)) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 0)) &&
        pyopencv_to(pyobj_components, components, ArgInfo("components", 1)) )
    {
        ERRWRAP2(cv::ft::FT02D_components(matrix, kernel, components));
        return pyopencv_from(components);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ft_FT02D_components1(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ft;

    {
    PyObject* pyobj_matrix = NULL;
    Mat matrix;
    PyObject* pyobj_kernel = NULL;
    Mat kernel;
    PyObject* pyobj_components = NULL;
    Mat components;
    PyObject* pyobj_mask = NULL;
    Mat mask;

    const char* keywords[] = { "matrix", "kernel", "mask", "components", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|O:FT02D_components1", (char**)keywords, &pyobj_matrix, &pyobj_kernel, &pyobj_mask, &pyobj_components) &&
        pyopencv_to(pyobj_matrix, matrix, ArgInfo("matrix", 0)) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 0)) &&
        pyopencv_to(pyobj_components, components, ArgInfo("components", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::ft::FT02D_components(matrix, kernel, components, mask));
        return pyopencv_from(components);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_matrix = NULL;
    UMat matrix;
    PyObject* pyobj_kernel = NULL;
    UMat kernel;
    PyObject* pyobj_components = NULL;
    UMat components;
    PyObject* pyobj_mask = NULL;
    UMat mask;

    const char* keywords[] = { "matrix", "kernel", "mask", "components", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|O:FT02D_components1", (char**)keywords, &pyobj_matrix, &pyobj_kernel, &pyobj_mask, &pyobj_components) &&
        pyopencv_to(pyobj_matrix, matrix, ArgInfo("matrix", 0)) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 0)) &&
        pyopencv_to(pyobj_components, components, ArgInfo("components", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::ft::FT02D_components(matrix, kernel, components, mask));
        return pyopencv_from(components);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ft_FT02D_inverseFT(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ft;

    {
    PyObject* pyobj_components = NULL;
    Mat components;
    PyObject* pyobj_kernel = NULL;
    Mat kernel;
    PyObject* pyobj_output = NULL;
    Mat output;
    int width=0;
    int height=0;

    const char* keywords[] = { "components", "kernel", "width", "height", "output", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOii|O:FT02D_inverseFT", (char**)keywords, &pyobj_components, &pyobj_kernel, &width, &height, &pyobj_output) &&
        pyopencv_to(pyobj_components, components, ArgInfo("components", 0)) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 0)) &&
        pyopencv_to(pyobj_output, output, ArgInfo("output", 1)) )
    {
        ERRWRAP2(cv::ft::FT02D_inverseFT(components, kernel, output, width, height));
        return pyopencv_from(output);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_components = NULL;
    UMat components;
    PyObject* pyobj_kernel = NULL;
    UMat kernel;
    PyObject* pyobj_output = NULL;
    UMat output;
    int width=0;
    int height=0;

    const char* keywords[] = { "components", "kernel", "width", "height", "output", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOii|O:FT02D_inverseFT", (char**)keywords, &pyobj_components, &pyobj_kernel, &width, &height, &pyobj_output) &&
        pyopencv_to(pyobj_components, components, ArgInfo("components", 0)) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 0)) &&
        pyopencv_to(pyobj_output, output, ArgInfo("output", 1)) )
    {
        ERRWRAP2(cv::ft::FT02D_inverseFT(components, kernel, output, width, height));
        return pyopencv_from(output);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ft_FT02D_iteration(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ft;

    {
    PyObject* pyobj_matrix = NULL;
    Mat matrix;
    PyObject* pyobj_kernel = NULL;
    Mat kernel;
    PyObject* pyobj_output = NULL;
    Mat output;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    PyObject* pyobj_maskOutput = NULL;
    Mat maskOutput;
    bool firstStop=0;
    int retval;

    const char* keywords[] = { "matrix", "kernel", "mask", "firstStop", "output", "maskOutput", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOb|OO:FT02D_iteration", (char**)keywords, &pyobj_matrix, &pyobj_kernel, &pyobj_mask, &firstStop, &pyobj_output, &pyobj_maskOutput) &&
        pyopencv_to(pyobj_matrix, matrix, ArgInfo("matrix", 0)) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 0)) &&
        pyopencv_to(pyobj_output, output, ArgInfo("output", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) &&
        pyopencv_to(pyobj_maskOutput, maskOutput, ArgInfo("maskOutput", 1)) )
    {
        ERRWRAP2(retval = cv::ft::FT02D_iteration(matrix, kernel, output, mask, maskOutput, firstStop));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(output), pyopencv_from(maskOutput));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_matrix = NULL;
    UMat matrix;
    PyObject* pyobj_kernel = NULL;
    UMat kernel;
    PyObject* pyobj_output = NULL;
    UMat output;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    PyObject* pyobj_maskOutput = NULL;
    UMat maskOutput;
    bool firstStop=0;
    int retval;

    const char* keywords[] = { "matrix", "kernel", "mask", "firstStop", "output", "maskOutput", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOb|OO:FT02D_iteration", (char**)keywords, &pyobj_matrix, &pyobj_kernel, &pyobj_mask, &firstStop, &pyobj_output, &pyobj_maskOutput) &&
        pyopencv_to(pyobj_matrix, matrix, ArgInfo("matrix", 0)) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 0)) &&
        pyopencv_to(pyobj_output, output, ArgInfo("output", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) &&
        pyopencv_to(pyobj_maskOutput, maskOutput, ArgInfo("maskOutput", 1)) )
    {
        ERRWRAP2(retval = cv::ft::FT02D_iteration(matrix, kernel, output, mask, maskOutput, firstStop));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(output), pyopencv_from(maskOutput));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ft_FT02D_process(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ft;

    {
    PyObject* pyobj_matrix = NULL;
    Mat matrix;
    PyObject* pyobj_kernel = NULL;
    Mat kernel;
    PyObject* pyobj_output = NULL;
    Mat output;

    const char* keywords[] = { "matrix", "kernel", "output", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:FT02D_process", (char**)keywords, &pyobj_matrix, &pyobj_kernel, &pyobj_output) &&
        pyopencv_to(pyobj_matrix, matrix, ArgInfo("matrix", 0)) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 0)) &&
        pyopencv_to(pyobj_output, output, ArgInfo("output", 1)) )
    {
        ERRWRAP2(cv::ft::FT02D_process(matrix, kernel, output));
        return pyopencv_from(output);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_matrix = NULL;
    UMat matrix;
    PyObject* pyobj_kernel = NULL;
    UMat kernel;
    PyObject* pyobj_output = NULL;
    UMat output;

    const char* keywords[] = { "matrix", "kernel", "output", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:FT02D_process", (char**)keywords, &pyobj_matrix, &pyobj_kernel, &pyobj_output) &&
        pyopencv_to(pyobj_matrix, matrix, ArgInfo("matrix", 0)) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 0)) &&
        pyopencv_to(pyobj_output, output, ArgInfo("output", 1)) )
    {
        ERRWRAP2(cv::ft::FT02D_process(matrix, kernel, output));
        return pyopencv_from(output);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ft_FT02D_process1(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ft;

    {
    PyObject* pyobj_matrix = NULL;
    Mat matrix;
    PyObject* pyobj_kernel = NULL;
    Mat kernel;
    PyObject* pyobj_output = NULL;
    Mat output;
    PyObject* pyobj_mask = NULL;
    Mat mask;

    const char* keywords[] = { "matrix", "kernel", "mask", "output", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|O:FT02D_process1", (char**)keywords, &pyobj_matrix, &pyobj_kernel, &pyobj_mask, &pyobj_output) &&
        pyopencv_to(pyobj_matrix, matrix, ArgInfo("matrix", 0)) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 0)) &&
        pyopencv_to(pyobj_output, output, ArgInfo("output", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::ft::FT02D_process(matrix, kernel, output, mask));
        return pyopencv_from(output);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_matrix = NULL;
    UMat matrix;
    PyObject* pyobj_kernel = NULL;
    UMat kernel;
    PyObject* pyobj_output = NULL;
    UMat output;
    PyObject* pyobj_mask = NULL;
    UMat mask;

    const char* keywords[] = { "matrix", "kernel", "mask", "output", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|O:FT02D_process1", (char**)keywords, &pyobj_matrix, &pyobj_kernel, &pyobj_mask, &pyobj_output) &&
        pyopencv_to(pyobj_matrix, matrix, ArgInfo("matrix", 0)) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 0)) &&
        pyopencv_to(pyobj_output, output, ArgInfo("output", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::ft::FT02D_process(matrix, kernel, output, mask));
        return pyopencv_from(output);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ft_createKernel(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ft;

    {
    int function=0;
    int radius=0;
    PyObject* pyobj_kernel = NULL;
    Mat kernel;
    int chn=0;

    const char* keywords[] = { "function", "radius", "chn", "kernel", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "iii|O:createKernel", (char**)keywords, &function, &radius, &chn, &pyobj_kernel) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 1)) )
    {
        ERRWRAP2(cv::ft::createKernel(function, radius, kernel, chn));
        return pyopencv_from(kernel);
    }
    }
    PyErr_Clear();

    {
    int function=0;
    int radius=0;
    PyObject* pyobj_kernel = NULL;
    UMat kernel;
    int chn=0;

    const char* keywords[] = { "function", "radius", "chn", "kernel", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "iii|O:createKernel", (char**)keywords, &function, &radius, &chn, &pyobj_kernel) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 1)) )
    {
        ERRWRAP2(cv::ft::createKernel(function, radius, kernel, chn));
        return pyopencv_from(kernel);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ft_createKernel1(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ft;

    {
    PyObject* pyobj_A = NULL;
    Mat A;
    PyObject* pyobj_B = NULL;
    Mat B;
    PyObject* pyobj_kernel = NULL;
    Mat kernel;
    int chn=0;

    const char* keywords[] = { "A", "B", "chn", "kernel", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi|O:createKernel1", (char**)keywords, &pyobj_A, &pyobj_B, &chn, &pyobj_kernel) &&
        pyopencv_to(pyobj_A, A, ArgInfo("A", 0)) &&
        pyopencv_to(pyobj_B, B, ArgInfo("B", 0)) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 1)) )
    {
        ERRWRAP2(cv::ft::createKernel(A, B, kernel, chn));
        return pyopencv_from(kernel);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_A = NULL;
    UMat A;
    PyObject* pyobj_B = NULL;
    UMat B;
    PyObject* pyobj_kernel = NULL;
    UMat kernel;
    int chn=0;

    const char* keywords[] = { "A", "B", "chn", "kernel", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi|O:createKernel1", (char**)keywords, &pyobj_A, &pyobj_B, &chn, &pyobj_kernel) &&
        pyopencv_to(pyobj_A, A, ArgInfo("A", 0)) &&
        pyopencv_to(pyobj_B, B, ArgInfo("B", 0)) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 1)) )
    {
        ERRWRAP2(cv::ft::createKernel(A, B, kernel, chn));
        return pyopencv_from(kernel);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ft_filter(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ft;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_kernel = NULL;
    Mat kernel;
    PyObject* pyobj_output = NULL;
    Mat output;

    const char* keywords[] = { "image", "kernel", "output", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:filter", (char**)keywords, &pyobj_image, &pyobj_kernel, &pyobj_output) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 0)) &&
        pyopencv_to(pyobj_output, output, ArgInfo("output", 1)) )
    {
        ERRWRAP2(cv::ft::filter(image, kernel, output));
        return pyopencv_from(output);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_kernel = NULL;
    UMat kernel;
    PyObject* pyobj_output = NULL;
    UMat output;

    const char* keywords[] = { "image", "kernel", "output", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:filter", (char**)keywords, &pyobj_image, &pyobj_kernel, &pyobj_output) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_kernel, kernel, ArgInfo("kernel", 0)) &&
        pyopencv_to(pyobj_output, output, ArgInfo("output", 1)) )
    {
        ERRWRAP2(cv::ft::filter(image, kernel, output));
        return pyopencv_from(output);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ft_inpaint(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ft;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    PyObject* pyobj_output = NULL;
    Mat output;
    int radius=0;
    int function=0;
    int algorithm=0;

    const char* keywords[] = { "image", "mask", "radius", "function", "algorithm", "output", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOiii|O:inpaint", (char**)keywords, &pyobj_image, &pyobj_mask, &radius, &function, &algorithm, &pyobj_output) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) &&
        pyopencv_to(pyobj_output, output, ArgInfo("output", 1)) )
    {
        ERRWRAP2(cv::ft::inpaint(image, mask, output, radius, function, algorithm));
        return pyopencv_from(output);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    PyObject* pyobj_output = NULL;
    UMat output;
    int radius=0;
    int function=0;
    int algorithm=0;

    const char* keywords[] = { "image", "mask", "radius", "function", "algorithm", "output", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOiii|O:inpaint", (char**)keywords, &pyobj_image, &pyobj_mask, &radius, &function, &algorithm, &pyobj_output) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) &&
        pyopencv_to(pyobj_output, output, ArgInfo("output", 1)) )
    {
        ERRWRAP2(cv::ft::inpaint(image, mask, output, radius, function, algorithm));
        return pyopencv_from(output);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ipp_setUseIPP(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ipp;

    bool flag=0;

    const char* keywords[] = { "flag", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:setUseIPP", (char**)keywords, &flag) )
    {
        ERRWRAP2(cv::ipp::setUseIPP(flag));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ipp_useIPP(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ipp;

    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::ipp::useIPP());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ANN_MLP_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    Ptr<ANN_MLP> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::ml::ANN_MLP::create());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ANN_MLP_load(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    PyObject* pyobj_filepath = NULL;
    String filepath;
    Ptr<ANN_MLP> retval;

    const char* keywords[] = { "filepath", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:ANN_MLP_load", (char**)keywords, &pyobj_filepath) &&
        pyopencv_to(pyobj_filepath, filepath, ArgInfo("filepath", 0)) )
    {
        ERRWRAP2(retval = cv::ml::ANN_MLP::load(filepath));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_Boost_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    Ptr<Boost> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::ml::Boost::create());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_Boost_load(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    PyObject* pyobj_filepath = NULL;
    String filepath;
    PyObject* pyobj_nodeName = NULL;
    String nodeName;
    Ptr<Boost> retval;

    const char* keywords[] = { "filepath", "nodeName", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:Boost_load", (char**)keywords, &pyobj_filepath, &pyobj_nodeName) &&
        pyopencv_to(pyobj_filepath, filepath, ArgInfo("filepath", 0)) &&
        pyopencv_to(pyobj_nodeName, nodeName, ArgInfo("nodeName", 0)) )
    {
        ERRWRAP2(retval = cv::ml::Boost::load(filepath, nodeName));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_DTrees_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    Ptr<DTrees> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::ml::DTrees::create());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_DTrees_load(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    PyObject* pyobj_filepath = NULL;
    String filepath;
    PyObject* pyobj_nodeName = NULL;
    String nodeName;
    Ptr<DTrees> retval;

    const char* keywords[] = { "filepath", "nodeName", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:DTrees_load", (char**)keywords, &pyobj_filepath, &pyobj_nodeName) &&
        pyopencv_to(pyobj_filepath, filepath, ArgInfo("filepath", 0)) &&
        pyopencv_to(pyobj_nodeName, nodeName, ArgInfo("nodeName", 0)) )
    {
        ERRWRAP2(retval = cv::ml::DTrees::load(filepath, nodeName));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_EM_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    Ptr<EM> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::ml::EM::create());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_EM_load(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    PyObject* pyobj_filepath = NULL;
    String filepath;
    PyObject* pyobj_nodeName = NULL;
    String nodeName;
    Ptr<EM> retval;

    const char* keywords[] = { "filepath", "nodeName", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:EM_load", (char**)keywords, &pyobj_filepath, &pyobj_nodeName) &&
        pyopencv_to(pyobj_filepath, filepath, ArgInfo("filepath", 0)) &&
        pyopencv_to(pyobj_nodeName, nodeName, ArgInfo("nodeName", 0)) )
    {
        ERRWRAP2(retval = cv::ml::EM::load(filepath, nodeName));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_KNearest_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    Ptr<KNearest> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::ml::KNearest::create());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_LogisticRegression_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    Ptr<LogisticRegression> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::ml::LogisticRegression::create());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_LogisticRegression_load(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    PyObject* pyobj_filepath = NULL;
    String filepath;
    PyObject* pyobj_nodeName = NULL;
    String nodeName;
    Ptr<LogisticRegression> retval;

    const char* keywords[] = { "filepath", "nodeName", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:LogisticRegression_load", (char**)keywords, &pyobj_filepath, &pyobj_nodeName) &&
        pyopencv_to(pyobj_filepath, filepath, ArgInfo("filepath", 0)) &&
        pyopencv_to(pyobj_nodeName, nodeName, ArgInfo("nodeName", 0)) )
    {
        ERRWRAP2(retval = cv::ml::LogisticRegression::load(filepath, nodeName));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_NormalBayesClassifier_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    Ptr<NormalBayesClassifier> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::ml::NormalBayesClassifier::create());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_NormalBayesClassifier_load(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    PyObject* pyobj_filepath = NULL;
    String filepath;
    PyObject* pyobj_nodeName = NULL;
    String nodeName;
    Ptr<NormalBayesClassifier> retval;

    const char* keywords[] = { "filepath", "nodeName", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:NormalBayesClassifier_load", (char**)keywords, &pyobj_filepath, &pyobj_nodeName) &&
        pyopencv_to(pyobj_filepath, filepath, ArgInfo("filepath", 0)) &&
        pyopencv_to(pyobj_nodeName, nodeName, ArgInfo("nodeName", 0)) )
    {
        ERRWRAP2(retval = cv::ml::NormalBayesClassifier::load(filepath, nodeName));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_RTrees_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    Ptr<RTrees> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::ml::RTrees::create());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_RTrees_load(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    PyObject* pyobj_filepath = NULL;
    String filepath;
    PyObject* pyobj_nodeName = NULL;
    String nodeName;
    Ptr<RTrees> retval;

    const char* keywords[] = { "filepath", "nodeName", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:RTrees_load", (char**)keywords, &pyobj_filepath, &pyobj_nodeName) &&
        pyopencv_to(pyobj_filepath, filepath, ArgInfo("filepath", 0)) &&
        pyopencv_to(pyobj_nodeName, nodeName, ArgInfo("nodeName", 0)) )
    {
        ERRWRAP2(retval = cv::ml::RTrees::load(filepath, nodeName));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_SVMSGD_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    Ptr<SVMSGD> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::ml::SVMSGD::create());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_SVMSGD_load(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    PyObject* pyobj_filepath = NULL;
    String filepath;
    PyObject* pyobj_nodeName = NULL;
    String nodeName;
    Ptr<SVMSGD> retval;

    const char* keywords[] = { "filepath", "nodeName", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:SVMSGD_load", (char**)keywords, &pyobj_filepath, &pyobj_nodeName) &&
        pyopencv_to(pyobj_filepath, filepath, ArgInfo("filepath", 0)) &&
        pyopencv_to(pyobj_nodeName, nodeName, ArgInfo("nodeName", 0)) )
    {
        ERRWRAP2(retval = cv::ml::SVMSGD::load(filepath, nodeName));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_SVM_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    Ptr<SVM> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::ml::SVM::create());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_SVM_load(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    PyObject* pyobj_filepath = NULL;
    String filepath;
    Ptr<SVM> retval;

    const char* keywords[] = { "filepath", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:SVM_load", (char**)keywords, &pyobj_filepath) &&
        pyopencv_to(pyobj_filepath, filepath, ArgInfo("filepath", 0)) )
    {
        ERRWRAP2(retval = cv::ml::SVM::load(filepath));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_TrainData_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    {
    PyObject* pyobj_samples = NULL;
    Mat samples;
    int layout=0;
    PyObject* pyobj_responses = NULL;
    Mat responses;
    PyObject* pyobj_varIdx = NULL;
    Mat varIdx;
    PyObject* pyobj_sampleIdx = NULL;
    Mat sampleIdx;
    PyObject* pyobj_sampleWeights = NULL;
    Mat sampleWeights;
    PyObject* pyobj_varType = NULL;
    Mat varType;
    Ptr<TrainData> retval;

    const char* keywords[] = { "samples", "layout", "responses", "varIdx", "sampleIdx", "sampleWeights", "varType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OiO|OOOO:TrainData_create", (char**)keywords, &pyobj_samples, &layout, &pyobj_responses, &pyobj_varIdx, &pyobj_sampleIdx, &pyobj_sampleWeights, &pyobj_varType) &&
        pyopencv_to(pyobj_samples, samples, ArgInfo("samples", 0)) &&
        pyopencv_to(pyobj_responses, responses, ArgInfo("responses", 0)) &&
        pyopencv_to(pyobj_varIdx, varIdx, ArgInfo("varIdx", 0)) &&
        pyopencv_to(pyobj_sampleIdx, sampleIdx, ArgInfo("sampleIdx", 0)) &&
        pyopencv_to(pyobj_sampleWeights, sampleWeights, ArgInfo("sampleWeights", 0)) &&
        pyopencv_to(pyobj_varType, varType, ArgInfo("varType", 0)) )
    {
        ERRWRAP2(retval = cv::ml::TrainData::create(samples, layout, responses, varIdx, sampleIdx, sampleWeights, varType));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_samples = NULL;
    UMat samples;
    int layout=0;
    PyObject* pyobj_responses = NULL;
    UMat responses;
    PyObject* pyobj_varIdx = NULL;
    UMat varIdx;
    PyObject* pyobj_sampleIdx = NULL;
    UMat sampleIdx;
    PyObject* pyobj_sampleWeights = NULL;
    UMat sampleWeights;
    PyObject* pyobj_varType = NULL;
    UMat varType;
    Ptr<TrainData> retval;

    const char* keywords[] = { "samples", "layout", "responses", "varIdx", "sampleIdx", "sampleWeights", "varType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OiO|OOOO:TrainData_create", (char**)keywords, &pyobj_samples, &layout, &pyobj_responses, &pyobj_varIdx, &pyobj_sampleIdx, &pyobj_sampleWeights, &pyobj_varType) &&
        pyopencv_to(pyobj_samples, samples, ArgInfo("samples", 0)) &&
        pyopencv_to(pyobj_responses, responses, ArgInfo("responses", 0)) &&
        pyopencv_to(pyobj_varIdx, varIdx, ArgInfo("varIdx", 0)) &&
        pyopencv_to(pyobj_sampleIdx, sampleIdx, ArgInfo("sampleIdx", 0)) &&
        pyopencv_to(pyobj_sampleWeights, sampleWeights, ArgInfo("sampleWeights", 0)) &&
        pyopencv_to(pyobj_varType, varType, ArgInfo("varType", 0)) )
    {
        ERRWRAP2(retval = cv::ml::TrainData::create(samples, layout, responses, varIdx, sampleIdx, sampleWeights, varType));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_TrainData_getSubVector(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    {
    PyObject* pyobj_vec = NULL;
    Mat vec;
    PyObject* pyobj_idx = NULL;
    Mat idx;
    Mat retval;

    const char* keywords[] = { "vec", "idx", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:TrainData_getSubVector", (char**)keywords, &pyobj_vec, &pyobj_idx) &&
        pyopencv_to(pyobj_vec, vec, ArgInfo("vec", 0)) &&
        pyopencv_to(pyobj_idx, idx, ArgInfo("idx", 0)) )
    {
        ERRWRAP2(retval = cv::ml::TrainData::getSubVector(vec, idx));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_vec = NULL;
    Mat vec;
    PyObject* pyobj_idx = NULL;
    Mat idx;
    Mat retval;

    const char* keywords[] = { "vec", "idx", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:TrainData_getSubVector", (char**)keywords, &pyobj_vec, &pyobj_idx) &&
        pyopencv_to(pyobj_vec, vec, ArgInfo("vec", 0)) &&
        pyopencv_to(pyobj_idx, idx, ArgInfo("idx", 0)) )
    {
        ERRWRAP2(retval = cv::ml::TrainData::getSubVector(vec, idx));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_motempl_calcGlobalOrientation(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::motempl;

    {
    PyObject* pyobj_orientation = NULL;
    Mat orientation;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    PyObject* pyobj_mhi = NULL;
    Mat mhi;
    double timestamp=0;
    double duration=0;
    double retval;

    const char* keywords[] = { "orientation", "mask", "mhi", "timestamp", "duration", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOdd:calcGlobalOrientation", (char**)keywords, &pyobj_orientation, &pyobj_mask, &pyobj_mhi, &timestamp, &duration) &&
        pyopencv_to(pyobj_orientation, orientation, ArgInfo("orientation", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) &&
        pyopencv_to(pyobj_mhi, mhi, ArgInfo("mhi", 0)) )
    {
        ERRWRAP2(retval = cv::motempl::calcGlobalOrientation(orientation, mask, mhi, timestamp, duration));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_orientation = NULL;
    UMat orientation;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    PyObject* pyobj_mhi = NULL;
    UMat mhi;
    double timestamp=0;
    double duration=0;
    double retval;

    const char* keywords[] = { "orientation", "mask", "mhi", "timestamp", "duration", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOdd:calcGlobalOrientation", (char**)keywords, &pyobj_orientation, &pyobj_mask, &pyobj_mhi, &timestamp, &duration) &&
        pyopencv_to(pyobj_orientation, orientation, ArgInfo("orientation", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) &&
        pyopencv_to(pyobj_mhi, mhi, ArgInfo("mhi", 0)) )
    {
        ERRWRAP2(retval = cv::motempl::calcGlobalOrientation(orientation, mask, mhi, timestamp, duration));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_motempl_calcMotionGradient(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::motempl;

    {
    PyObject* pyobj_mhi = NULL;
    Mat mhi;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    PyObject* pyobj_orientation = NULL;
    Mat orientation;
    double delta1=0;
    double delta2=0;
    int apertureSize=3;

    const char* keywords[] = { "mhi", "delta1", "delta2", "mask", "orientation", "apertureSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odd|OOi:calcMotionGradient", (char**)keywords, &pyobj_mhi, &delta1, &delta2, &pyobj_mask, &pyobj_orientation, &apertureSize) &&
        pyopencv_to(pyobj_mhi, mhi, ArgInfo("mhi", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 1)) &&
        pyopencv_to(pyobj_orientation, orientation, ArgInfo("orientation", 1)) )
    {
        ERRWRAP2(cv::motempl::calcMotionGradient(mhi, mask, orientation, delta1, delta2, apertureSize));
        return Py_BuildValue("(NN)", pyopencv_from(mask), pyopencv_from(orientation));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_mhi = NULL;
    UMat mhi;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    PyObject* pyobj_orientation = NULL;
    UMat orientation;
    double delta1=0;
    double delta2=0;
    int apertureSize=3;

    const char* keywords[] = { "mhi", "delta1", "delta2", "mask", "orientation", "apertureSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odd|OOi:calcMotionGradient", (char**)keywords, &pyobj_mhi, &delta1, &delta2, &pyobj_mask, &pyobj_orientation, &apertureSize) &&
        pyopencv_to(pyobj_mhi, mhi, ArgInfo("mhi", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 1)) &&
        pyopencv_to(pyobj_orientation, orientation, ArgInfo("orientation", 1)) )
    {
        ERRWRAP2(cv::motempl::calcMotionGradient(mhi, mask, orientation, delta1, delta2, apertureSize));
        return Py_BuildValue("(NN)", pyopencv_from(mask), pyopencv_from(orientation));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_motempl_segmentMotion(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::motempl;

    {
    PyObject* pyobj_mhi = NULL;
    Mat mhi;
    PyObject* pyobj_segmask = NULL;
    Mat segmask;
    vector_Rect boundingRects;
    double timestamp=0;
    double segThresh=0;

    const char* keywords[] = { "mhi", "timestamp", "segThresh", "segmask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odd|O:segmentMotion", (char**)keywords, &pyobj_mhi, &timestamp, &segThresh, &pyobj_segmask) &&
        pyopencv_to(pyobj_mhi, mhi, ArgInfo("mhi", 0)) &&
        pyopencv_to(pyobj_segmask, segmask, ArgInfo("segmask", 1)) )
    {
        ERRWRAP2(cv::motempl::segmentMotion(mhi, segmask, boundingRects, timestamp, segThresh));
        return Py_BuildValue("(NN)", pyopencv_from(segmask), pyopencv_from(boundingRects));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_mhi = NULL;
    UMat mhi;
    PyObject* pyobj_segmask = NULL;
    UMat segmask;
    vector_Rect boundingRects;
    double timestamp=0;
    double segThresh=0;

    const char* keywords[] = { "mhi", "timestamp", "segThresh", "segmask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odd|O:segmentMotion", (char**)keywords, &pyobj_mhi, &timestamp, &segThresh, &pyobj_segmask) &&
        pyopencv_to(pyobj_mhi, mhi, ArgInfo("mhi", 0)) &&
        pyopencv_to(pyobj_segmask, segmask, ArgInfo("segmask", 1)) )
    {
        ERRWRAP2(cv::motempl::segmentMotion(mhi, segmask, boundingRects, timestamp, segThresh));
        return Py_BuildValue("(NN)", pyopencv_from(segmask), pyopencv_from(boundingRects));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_motempl_updateMotionHistory(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::motempl;

    {
    PyObject* pyobj_silhouette = NULL;
    Mat silhouette;
    PyObject* pyobj_mhi = NULL;
    Mat mhi;
    double timestamp=0;
    double duration=0;

    const char* keywords[] = { "silhouette", "mhi", "timestamp", "duration", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOdd:updateMotionHistory", (char**)keywords, &pyobj_silhouette, &pyobj_mhi, &timestamp, &duration) &&
        pyopencv_to(pyobj_silhouette, silhouette, ArgInfo("silhouette", 0)) &&
        pyopencv_to(pyobj_mhi, mhi, ArgInfo("mhi", 1)) )
    {
        ERRWRAP2(cv::motempl::updateMotionHistory(silhouette, mhi, timestamp, duration));
        return pyopencv_from(mhi);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_silhouette = NULL;
    UMat silhouette;
    PyObject* pyobj_mhi = NULL;
    UMat mhi;
    double timestamp=0;
    double duration=0;

    const char* keywords[] = { "silhouette", "mhi", "timestamp", "duration", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOdd:updateMotionHistory", (char**)keywords, &pyobj_silhouette, &pyobj_mhi, &timestamp, &duration) &&
        pyopencv_to(pyobj_silhouette, silhouette, ArgInfo("silhouette", 0)) &&
        pyopencv_to(pyobj_mhi, mhi, ArgInfo("mhi", 1)) )
    {
        ERRWRAP2(cv::motempl::updateMotionHistory(silhouette, mhi, timestamp, duration));
        return pyopencv_from(mhi);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ocl_finish(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ocl;


    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(cv::ocl::finish());
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ocl_haveAmdBlas(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ocl;

    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::ocl::haveAmdBlas());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ocl_haveAmdFft(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ocl;

    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::ocl::haveAmdFft());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ocl_haveOpenCL(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ocl;

    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::ocl::haveOpenCL());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ocl_setUseOpenCL(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ocl;

    bool flag=0;

    const char* keywords[] = { "flag", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:setUseOpenCL", (char**)keywords, &flag) )
    {
        ERRWRAP2(cv::ocl::setUseOpenCL(flag));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ocl_useOpenCL(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ocl;

    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::ocl::useOpenCL());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_omnidir_calibrate(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::omnidir;

    {
    PyObject* pyobj_objectPoints = NULL;
    vector_Mat objectPoints;
    PyObject* pyobj_imagePoints = NULL;
    vector_Mat imagePoints;
    PyObject* pyobj_size = NULL;
    Size size;
    PyObject* pyobj_K = NULL;
    Mat K;
    PyObject* pyobj_xi = NULL;
    Mat xi;
    PyObject* pyobj_D = NULL;
    Mat D;
    PyObject* pyobj_rvecs = NULL;
    vector_Mat rvecs;
    PyObject* pyobj_tvecs = NULL;
    vector_Mat tvecs;
    int flags=0;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria;
    PyObject* pyobj_idx = NULL;
    Mat idx;
    double retval;

    const char* keywords[] = { "objectPoints", "imagePoints", "size", "K", "xi", "D", "flags", "criteria", "rvecs", "tvecs", "idx", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOiO|OOO:calibrate", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints, &pyobj_size, &pyobj_K, &pyobj_xi, &pyobj_D, &flags, &pyobj_criteria, &pyobj_rvecs, &pyobj_tvecs, &pyobj_idx) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints, imagePoints, ArgInfo("imagePoints", 0)) &&
        pyopencv_to(pyobj_size, size, ArgInfo("size", 0)) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 1)) &&
        pyopencv_to(pyobj_xi, xi, ArgInfo("xi", 1)) &&
        pyopencv_to(pyobj_D, D, ArgInfo("D", 1)) &&
        pyopencv_to(pyobj_rvecs, rvecs, ArgInfo("rvecs", 1)) &&
        pyopencv_to(pyobj_tvecs, tvecs, ArgInfo("tvecs", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) &&
        pyopencv_to(pyobj_idx, idx, ArgInfo("idx", 1)) )
    {
        ERRWRAP2(retval = cv::omnidir::calibrate(objectPoints, imagePoints, size, K, xi, D, rvecs, tvecs, flags, criteria, idx));
        return Py_BuildValue("(NNNNNNN)", pyopencv_from(retval), pyopencv_from(K), pyopencv_from(xi), pyopencv_from(D), pyopencv_from(rvecs), pyopencv_from(tvecs), pyopencv_from(idx));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_objectPoints = NULL;
    vector_Mat objectPoints;
    PyObject* pyobj_imagePoints = NULL;
    vector_Mat imagePoints;
    PyObject* pyobj_size = NULL;
    Size size;
    PyObject* pyobj_K = NULL;
    UMat K;
    PyObject* pyobj_xi = NULL;
    UMat xi;
    PyObject* pyobj_D = NULL;
    UMat D;
    PyObject* pyobj_rvecs = NULL;
    vector_Mat rvecs;
    PyObject* pyobj_tvecs = NULL;
    vector_Mat tvecs;
    int flags=0;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria;
    PyObject* pyobj_idx = NULL;
    UMat idx;
    double retval;

    const char* keywords[] = { "objectPoints", "imagePoints", "size", "K", "xi", "D", "flags", "criteria", "rvecs", "tvecs", "idx", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOiO|OOO:calibrate", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints, &pyobj_size, &pyobj_K, &pyobj_xi, &pyobj_D, &flags, &pyobj_criteria, &pyobj_rvecs, &pyobj_tvecs, &pyobj_idx) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints, imagePoints, ArgInfo("imagePoints", 0)) &&
        pyopencv_to(pyobj_size, size, ArgInfo("size", 0)) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 1)) &&
        pyopencv_to(pyobj_xi, xi, ArgInfo("xi", 1)) &&
        pyopencv_to(pyobj_D, D, ArgInfo("D", 1)) &&
        pyopencv_to(pyobj_rvecs, rvecs, ArgInfo("rvecs", 1)) &&
        pyopencv_to(pyobj_tvecs, tvecs, ArgInfo("tvecs", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) &&
        pyopencv_to(pyobj_idx, idx, ArgInfo("idx", 1)) )
    {
        ERRWRAP2(retval = cv::omnidir::calibrate(objectPoints, imagePoints, size, K, xi, D, rvecs, tvecs, flags, criteria, idx));
        return Py_BuildValue("(NNNNNNN)", pyopencv_from(retval), pyopencv_from(K), pyopencv_from(xi), pyopencv_from(D), pyopencv_from(rvecs), pyopencv_from(tvecs), pyopencv_from(idx));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_omnidir_initUndistortRectifyMap(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::omnidir;

    {
    PyObject* pyobj_K = NULL;
    Mat K;
    PyObject* pyobj_D = NULL;
    Mat D;
    PyObject* pyobj_xi = NULL;
    Mat xi;
    PyObject* pyobj_R = NULL;
    Mat R;
    PyObject* pyobj_P = NULL;
    Mat P;
    PyObject* pyobj_size = NULL;
    Size size;
    int mltype=0;
    PyObject* pyobj_map1 = NULL;
    Mat map1;
    PyObject* pyobj_map2 = NULL;
    Mat map2;
    int flags=0;

    const char* keywords[] = { "K", "D", "xi", "R", "P", "size", "mltype", "flags", "map1", "map2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOii|OO:initUndistortRectifyMap", (char**)keywords, &pyobj_K, &pyobj_D, &pyobj_xi, &pyobj_R, &pyobj_P, &pyobj_size, &mltype, &flags, &pyobj_map1, &pyobj_map2) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 0)) &&
        pyopencv_to(pyobj_D, D, ArgInfo("D", 0)) &&
        pyopencv_to(pyobj_xi, xi, ArgInfo("xi", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 0)) &&
        pyopencv_to(pyobj_P, P, ArgInfo("P", 0)) &&
        pyopencv_to(pyobj_size, size, ArgInfo("size", 0)) &&
        pyopencv_to(pyobj_map1, map1, ArgInfo("map1", 1)) &&
        pyopencv_to(pyobj_map2, map2, ArgInfo("map2", 1)) )
    {
        ERRWRAP2(cv::omnidir::initUndistortRectifyMap(K, D, xi, R, P, size, mltype, map1, map2, flags));
        return Py_BuildValue("(NN)", pyopencv_from(map1), pyopencv_from(map2));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_K = NULL;
    UMat K;
    PyObject* pyobj_D = NULL;
    UMat D;
    PyObject* pyobj_xi = NULL;
    UMat xi;
    PyObject* pyobj_R = NULL;
    UMat R;
    PyObject* pyobj_P = NULL;
    UMat P;
    PyObject* pyobj_size = NULL;
    Size size;
    int mltype=0;
    PyObject* pyobj_map1 = NULL;
    UMat map1;
    PyObject* pyobj_map2 = NULL;
    UMat map2;
    int flags=0;

    const char* keywords[] = { "K", "D", "xi", "R", "P", "size", "mltype", "flags", "map1", "map2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOii|OO:initUndistortRectifyMap", (char**)keywords, &pyobj_K, &pyobj_D, &pyobj_xi, &pyobj_R, &pyobj_P, &pyobj_size, &mltype, &flags, &pyobj_map1, &pyobj_map2) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 0)) &&
        pyopencv_to(pyobj_D, D, ArgInfo("D", 0)) &&
        pyopencv_to(pyobj_xi, xi, ArgInfo("xi", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 0)) &&
        pyopencv_to(pyobj_P, P, ArgInfo("P", 0)) &&
        pyopencv_to(pyobj_size, size, ArgInfo("size", 0)) &&
        pyopencv_to(pyobj_map1, map1, ArgInfo("map1", 1)) &&
        pyopencv_to(pyobj_map2, map2, ArgInfo("map2", 1)) )
    {
        ERRWRAP2(cv::omnidir::initUndistortRectifyMap(K, D, xi, R, P, size, mltype, map1, map2, flags));
        return Py_BuildValue("(NN)", pyopencv_from(map1), pyopencv_from(map2));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_omnidir_projectPoints(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::omnidir;

    {
    PyObject* pyobj_objectPoints = NULL;
    Mat objectPoints;
    PyObject* pyobj_imagePoints = NULL;
    Mat imagePoints;
    PyObject* pyobj_rvec = NULL;
    Mat rvec;
    PyObject* pyobj_tvec = NULL;
    Mat tvec;
    PyObject* pyobj_K = NULL;
    Mat K;
    double xi=0;
    PyObject* pyobj_D = NULL;
    Mat D;
    PyObject* pyobj_jacobian = NULL;
    Mat jacobian;

    const char* keywords[] = { "objectPoints", "rvec", "tvec", "K", "xi", "D", "imagePoints", "jacobian", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOdO|OO:projectPoints", (char**)keywords, &pyobj_objectPoints, &pyobj_rvec, &pyobj_tvec, &pyobj_K, &xi, &pyobj_D, &pyobj_imagePoints, &pyobj_jacobian) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints, imagePoints, ArgInfo("imagePoints", 1)) &&
        pyopencv_to(pyobj_rvec, rvec, ArgInfo("rvec", 0)) &&
        pyopencv_to(pyobj_tvec, tvec, ArgInfo("tvec", 0)) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 0)) &&
        pyopencv_to(pyobj_D, D, ArgInfo("D", 0)) &&
        pyopencv_to(pyobj_jacobian, jacobian, ArgInfo("jacobian", 1)) )
    {
        ERRWRAP2(cv::omnidir::projectPoints(objectPoints, imagePoints, rvec, tvec, K, xi, D, jacobian));
        return Py_BuildValue("(NN)", pyopencv_from(imagePoints), pyopencv_from(jacobian));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_objectPoints = NULL;
    UMat objectPoints;
    PyObject* pyobj_imagePoints = NULL;
    UMat imagePoints;
    PyObject* pyobj_rvec = NULL;
    UMat rvec;
    PyObject* pyobj_tvec = NULL;
    UMat tvec;
    PyObject* pyobj_K = NULL;
    UMat K;
    double xi=0;
    PyObject* pyobj_D = NULL;
    UMat D;
    PyObject* pyobj_jacobian = NULL;
    UMat jacobian;

    const char* keywords[] = { "objectPoints", "rvec", "tvec", "K", "xi", "D", "imagePoints", "jacobian", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOdO|OO:projectPoints", (char**)keywords, &pyobj_objectPoints, &pyobj_rvec, &pyobj_tvec, &pyobj_K, &xi, &pyobj_D, &pyobj_imagePoints, &pyobj_jacobian) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints, imagePoints, ArgInfo("imagePoints", 1)) &&
        pyopencv_to(pyobj_rvec, rvec, ArgInfo("rvec", 0)) &&
        pyopencv_to(pyobj_tvec, tvec, ArgInfo("tvec", 0)) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 0)) &&
        pyopencv_to(pyobj_D, D, ArgInfo("D", 0)) &&
        pyopencv_to(pyobj_jacobian, jacobian, ArgInfo("jacobian", 1)) )
    {
        ERRWRAP2(cv::omnidir::projectPoints(objectPoints, imagePoints, rvec, tvec, K, xi, D, jacobian));
        return Py_BuildValue("(NN)", pyopencv_from(imagePoints), pyopencv_from(jacobian));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_omnidir_stereoCalibrate(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::omnidir;

    {
    PyObject* pyobj_objectPoints = NULL;
    vector_Mat objectPoints;
    PyObject* pyobj_imagePoints1 = NULL;
    vector_Mat imagePoints1;
    PyObject* pyobj_imagePoints2 = NULL;
    vector_Mat imagePoints2;
    PyObject* pyobj_imageSize1 = NULL;
    Size imageSize1;
    PyObject* pyobj_imageSize2 = NULL;
    Size imageSize2;
    PyObject* pyobj_K1 = NULL;
    Mat K1;
    PyObject* pyobj_xi1 = NULL;
    Mat xi1;
    PyObject* pyobj_D1 = NULL;
    Mat D1;
    PyObject* pyobj_K2 = NULL;
    Mat K2;
    PyObject* pyobj_xi2 = NULL;
    Mat xi2;
    PyObject* pyobj_D2 = NULL;
    Mat D2;
    PyObject* pyobj_rvec = NULL;
    Mat rvec;
    PyObject* pyobj_tvec = NULL;
    Mat tvec;
    PyObject* pyobj_rvecsL = NULL;
    vector_Mat rvecsL;
    PyObject* pyobj_tvecsL = NULL;
    vector_Mat tvecsL;
    int flags=0;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria;
    PyObject* pyobj_idx = NULL;
    Mat idx;
    double retval;

    const char* keywords[] = { "objectPoints", "imagePoints1", "imagePoints2", "imageSize1", "imageSize2", "K1", "xi1", "D1", "K2", "xi2", "D2", "flags", "criteria", "rvec", "tvec", "rvecsL", "tvecsL", "idx", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOOOOOOiO|OOOOO:stereoCalibrate", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints1, &pyobj_imagePoints2, &pyobj_imageSize1, &pyobj_imageSize2, &pyobj_K1, &pyobj_xi1, &pyobj_D1, &pyobj_K2, &pyobj_xi2, &pyobj_D2, &flags, &pyobj_criteria, &pyobj_rvec, &pyobj_tvec, &pyobj_rvecsL, &pyobj_tvecsL, &pyobj_idx) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 1)) &&
        pyopencv_to(pyobj_imagePoints1, imagePoints1, ArgInfo("imagePoints1", 1)) &&
        pyopencv_to(pyobj_imagePoints2, imagePoints2, ArgInfo("imagePoints2", 1)) &&
        pyopencv_to(pyobj_imageSize1, imageSize1, ArgInfo("imageSize1", 0)) &&
        pyopencv_to(pyobj_imageSize2, imageSize2, ArgInfo("imageSize2", 0)) &&
        pyopencv_to(pyobj_K1, K1, ArgInfo("K1", 1)) &&
        pyopencv_to(pyobj_xi1, xi1, ArgInfo("xi1", 1)) &&
        pyopencv_to(pyobj_D1, D1, ArgInfo("D1", 1)) &&
        pyopencv_to(pyobj_K2, K2, ArgInfo("K2", 1)) &&
        pyopencv_to(pyobj_xi2, xi2, ArgInfo("xi2", 1)) &&
        pyopencv_to(pyobj_D2, D2, ArgInfo("D2", 1)) &&
        pyopencv_to(pyobj_rvec, rvec, ArgInfo("rvec", 1)) &&
        pyopencv_to(pyobj_tvec, tvec, ArgInfo("tvec", 1)) &&
        pyopencv_to(pyobj_rvecsL, rvecsL, ArgInfo("rvecsL", 1)) &&
        pyopencv_to(pyobj_tvecsL, tvecsL, ArgInfo("tvecsL", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) &&
        pyopencv_to(pyobj_idx, idx, ArgInfo("idx", 1)) )
    {
        ERRWRAP2(retval = cv::omnidir::stereoCalibrate(objectPoints, imagePoints1, imagePoints2, imageSize1, imageSize2, K1, xi1, D1, K2, xi2, D2, rvec, tvec, rvecsL, tvecsL, flags, criteria, idx));
        return Py_BuildValue("(NNNNNNNNNNNNNNN)", pyopencv_from(retval), pyopencv_from(objectPoints), pyopencv_from(imagePoints1), pyopencv_from(imagePoints2), pyopencv_from(K1), pyopencv_from(xi1), pyopencv_from(D1), pyopencv_from(K2), pyopencv_from(xi2), pyopencv_from(D2), pyopencv_from(rvec), pyopencv_from(tvec), pyopencv_from(rvecsL), pyopencv_from(tvecsL), pyopencv_from(idx));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_objectPoints = NULL;
    vector_Mat objectPoints;
    PyObject* pyobj_imagePoints1 = NULL;
    vector_Mat imagePoints1;
    PyObject* pyobj_imagePoints2 = NULL;
    vector_Mat imagePoints2;
    PyObject* pyobj_imageSize1 = NULL;
    Size imageSize1;
    PyObject* pyobj_imageSize2 = NULL;
    Size imageSize2;
    PyObject* pyobj_K1 = NULL;
    UMat K1;
    PyObject* pyobj_xi1 = NULL;
    UMat xi1;
    PyObject* pyobj_D1 = NULL;
    UMat D1;
    PyObject* pyobj_K2 = NULL;
    UMat K2;
    PyObject* pyobj_xi2 = NULL;
    UMat xi2;
    PyObject* pyobj_D2 = NULL;
    UMat D2;
    PyObject* pyobj_rvec = NULL;
    UMat rvec;
    PyObject* pyobj_tvec = NULL;
    UMat tvec;
    PyObject* pyobj_rvecsL = NULL;
    vector_Mat rvecsL;
    PyObject* pyobj_tvecsL = NULL;
    vector_Mat tvecsL;
    int flags=0;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria;
    PyObject* pyobj_idx = NULL;
    UMat idx;
    double retval;

    const char* keywords[] = { "objectPoints", "imagePoints1", "imagePoints2", "imageSize1", "imageSize2", "K1", "xi1", "D1", "K2", "xi2", "D2", "flags", "criteria", "rvec", "tvec", "rvecsL", "tvecsL", "idx", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOOOOOOiO|OOOOO:stereoCalibrate", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints1, &pyobj_imagePoints2, &pyobj_imageSize1, &pyobj_imageSize2, &pyobj_K1, &pyobj_xi1, &pyobj_D1, &pyobj_K2, &pyobj_xi2, &pyobj_D2, &flags, &pyobj_criteria, &pyobj_rvec, &pyobj_tvec, &pyobj_rvecsL, &pyobj_tvecsL, &pyobj_idx) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 1)) &&
        pyopencv_to(pyobj_imagePoints1, imagePoints1, ArgInfo("imagePoints1", 1)) &&
        pyopencv_to(pyobj_imagePoints2, imagePoints2, ArgInfo("imagePoints2", 1)) &&
        pyopencv_to(pyobj_imageSize1, imageSize1, ArgInfo("imageSize1", 0)) &&
        pyopencv_to(pyobj_imageSize2, imageSize2, ArgInfo("imageSize2", 0)) &&
        pyopencv_to(pyobj_K1, K1, ArgInfo("K1", 1)) &&
        pyopencv_to(pyobj_xi1, xi1, ArgInfo("xi1", 1)) &&
        pyopencv_to(pyobj_D1, D1, ArgInfo("D1", 1)) &&
        pyopencv_to(pyobj_K2, K2, ArgInfo("K2", 1)) &&
        pyopencv_to(pyobj_xi2, xi2, ArgInfo("xi2", 1)) &&
        pyopencv_to(pyobj_D2, D2, ArgInfo("D2", 1)) &&
        pyopencv_to(pyobj_rvec, rvec, ArgInfo("rvec", 1)) &&
        pyopencv_to(pyobj_tvec, tvec, ArgInfo("tvec", 1)) &&
        pyopencv_to(pyobj_rvecsL, rvecsL, ArgInfo("rvecsL", 1)) &&
        pyopencv_to(pyobj_tvecsL, tvecsL, ArgInfo("tvecsL", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) &&
        pyopencv_to(pyobj_idx, idx, ArgInfo("idx", 1)) )
    {
        ERRWRAP2(retval = cv::omnidir::stereoCalibrate(objectPoints, imagePoints1, imagePoints2, imageSize1, imageSize2, K1, xi1, D1, K2, xi2, D2, rvec, tvec, rvecsL, tvecsL, flags, criteria, idx));
        return Py_BuildValue("(NNNNNNNNNNNNNNN)", pyopencv_from(retval), pyopencv_from(objectPoints), pyopencv_from(imagePoints1), pyopencv_from(imagePoints2), pyopencv_from(K1), pyopencv_from(xi1), pyopencv_from(D1), pyopencv_from(K2), pyopencv_from(xi2), pyopencv_from(D2), pyopencv_from(rvec), pyopencv_from(tvec), pyopencv_from(rvecsL), pyopencv_from(tvecsL), pyopencv_from(idx));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_omnidir_stereoReconstruct(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::omnidir;

    {
    PyObject* pyobj_image1 = NULL;
    Mat image1;
    PyObject* pyobj_image2 = NULL;
    Mat image2;
    PyObject* pyobj_K1 = NULL;
    Mat K1;
    PyObject* pyobj_D1 = NULL;
    Mat D1;
    PyObject* pyobj_xi1 = NULL;
    Mat xi1;
    PyObject* pyobj_K2 = NULL;
    Mat K2;
    PyObject* pyobj_D2 = NULL;
    Mat D2;
    PyObject* pyobj_xi2 = NULL;
    Mat xi2;
    PyObject* pyobj_R = NULL;
    Mat R;
    PyObject* pyobj_T = NULL;
    Mat T;
    int flag=0;
    int numDisparities=0;
    int SADWindowSize=0;
    PyObject* pyobj_disparity = NULL;
    Mat disparity;
    PyObject* pyobj_image1Rec = NULL;
    Mat image1Rec;
    PyObject* pyobj_image2Rec = NULL;
    Mat image2Rec;
    PyObject* pyobj_newSize = NULL;
    Size newSize;
    PyObject* pyobj_Knew = NULL;
    Mat Knew=cv::Mat();
    PyObject* pyobj_pointCloud = NULL;
    Mat pointCloud;
    int pointType=XYZRGB;

    const char* keywords[] = { "image1", "image2", "K1", "D1", "xi1", "K2", "D2", "xi2", "R", "T", "flag", "numDisparities", "SADWindowSize", "disparity", "image1Rec", "image2Rec", "newSize", "Knew", "pointCloud", "pointType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOOOOOiii|OOOOOOi:stereoReconstruct", (char**)keywords, &pyobj_image1, &pyobj_image2, &pyobj_K1, &pyobj_D1, &pyobj_xi1, &pyobj_K2, &pyobj_D2, &pyobj_xi2, &pyobj_R, &pyobj_T, &flag, &numDisparities, &SADWindowSize, &pyobj_disparity, &pyobj_image1Rec, &pyobj_image2Rec, &pyobj_newSize, &pyobj_Knew, &pyobj_pointCloud, &pointType) &&
        pyopencv_to(pyobj_image1, image1, ArgInfo("image1", 0)) &&
        pyopencv_to(pyobj_image2, image2, ArgInfo("image2", 0)) &&
        pyopencv_to(pyobj_K1, K1, ArgInfo("K1", 0)) &&
        pyopencv_to(pyobj_D1, D1, ArgInfo("D1", 0)) &&
        pyopencv_to(pyobj_xi1, xi1, ArgInfo("xi1", 0)) &&
        pyopencv_to(pyobj_K2, K2, ArgInfo("K2", 0)) &&
        pyopencv_to(pyobj_D2, D2, ArgInfo("D2", 0)) &&
        pyopencv_to(pyobj_xi2, xi2, ArgInfo("xi2", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 0)) &&
        pyopencv_to(pyobj_T, T, ArgInfo("T", 0)) &&
        pyopencv_to(pyobj_disparity, disparity, ArgInfo("disparity", 1)) &&
        pyopencv_to(pyobj_image1Rec, image1Rec, ArgInfo("image1Rec", 1)) &&
        pyopencv_to(pyobj_image2Rec, image2Rec, ArgInfo("image2Rec", 1)) &&
        pyopencv_to(pyobj_newSize, newSize, ArgInfo("newSize", 0)) &&
        pyopencv_to(pyobj_Knew, Knew, ArgInfo("Knew", 0)) &&
        pyopencv_to(pyobj_pointCloud, pointCloud, ArgInfo("pointCloud", 1)) )
    {
        ERRWRAP2(cv::omnidir::stereoReconstruct(image1, image2, K1, D1, xi1, K2, D2, xi2, R, T, flag, numDisparities, SADWindowSize, disparity, image1Rec, image2Rec, newSize, Knew, pointCloud, pointType));
        return Py_BuildValue("(NNNN)", pyopencv_from(disparity), pyopencv_from(image1Rec), pyopencv_from(image2Rec), pyopencv_from(pointCloud));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image1 = NULL;
    UMat image1;
    PyObject* pyobj_image2 = NULL;
    UMat image2;
    PyObject* pyobj_K1 = NULL;
    UMat K1;
    PyObject* pyobj_D1 = NULL;
    UMat D1;
    PyObject* pyobj_xi1 = NULL;
    UMat xi1;
    PyObject* pyobj_K2 = NULL;
    UMat K2;
    PyObject* pyobj_D2 = NULL;
    UMat D2;
    PyObject* pyobj_xi2 = NULL;
    UMat xi2;
    PyObject* pyobj_R = NULL;
    UMat R;
    PyObject* pyobj_T = NULL;
    UMat T;
    int flag=0;
    int numDisparities=0;
    int SADWindowSize=0;
    PyObject* pyobj_disparity = NULL;
    UMat disparity;
    PyObject* pyobj_image1Rec = NULL;
    UMat image1Rec;
    PyObject* pyobj_image2Rec = NULL;
    UMat image2Rec;
    PyObject* pyobj_newSize = NULL;
    Size newSize;
    PyObject* pyobj_Knew = NULL;
    UMat Knew=cv::UMat();
    PyObject* pyobj_pointCloud = NULL;
    UMat pointCloud;
    int pointType=XYZRGB;

    const char* keywords[] = { "image1", "image2", "K1", "D1", "xi1", "K2", "D2", "xi2", "R", "T", "flag", "numDisparities", "SADWindowSize", "disparity", "image1Rec", "image2Rec", "newSize", "Knew", "pointCloud", "pointType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOOOOOiii|OOOOOOi:stereoReconstruct", (char**)keywords, &pyobj_image1, &pyobj_image2, &pyobj_K1, &pyobj_D1, &pyobj_xi1, &pyobj_K2, &pyobj_D2, &pyobj_xi2, &pyobj_R, &pyobj_T, &flag, &numDisparities, &SADWindowSize, &pyobj_disparity, &pyobj_image1Rec, &pyobj_image2Rec, &pyobj_newSize, &pyobj_Knew, &pyobj_pointCloud, &pointType) &&
        pyopencv_to(pyobj_image1, image1, ArgInfo("image1", 0)) &&
        pyopencv_to(pyobj_image2, image2, ArgInfo("image2", 0)) &&
        pyopencv_to(pyobj_K1, K1, ArgInfo("K1", 0)) &&
        pyopencv_to(pyobj_D1, D1, ArgInfo("D1", 0)) &&
        pyopencv_to(pyobj_xi1, xi1, ArgInfo("xi1", 0)) &&
        pyopencv_to(pyobj_K2, K2, ArgInfo("K2", 0)) &&
        pyopencv_to(pyobj_D2, D2, ArgInfo("D2", 0)) &&
        pyopencv_to(pyobj_xi2, xi2, ArgInfo("xi2", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 0)) &&
        pyopencv_to(pyobj_T, T, ArgInfo("T", 0)) &&
        pyopencv_to(pyobj_disparity, disparity, ArgInfo("disparity", 1)) &&
        pyopencv_to(pyobj_image1Rec, image1Rec, ArgInfo("image1Rec", 1)) &&
        pyopencv_to(pyobj_image2Rec, image2Rec, ArgInfo("image2Rec", 1)) &&
        pyopencv_to(pyobj_newSize, newSize, ArgInfo("newSize", 0)) &&
        pyopencv_to(pyobj_Knew, Knew, ArgInfo("Knew", 0)) &&
        pyopencv_to(pyobj_pointCloud, pointCloud, ArgInfo("pointCloud", 1)) )
    {
        ERRWRAP2(cv::omnidir::stereoReconstruct(image1, image2, K1, D1, xi1, K2, D2, xi2, R, T, flag, numDisparities, SADWindowSize, disparity, image1Rec, image2Rec, newSize, Knew, pointCloud, pointType));
        return Py_BuildValue("(NNNN)", pyopencv_from(disparity), pyopencv_from(image1Rec), pyopencv_from(image2Rec), pyopencv_from(pointCloud));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_omnidir_stereoRectify(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::omnidir;

    {
    PyObject* pyobj_R = NULL;
    Mat R;
    PyObject* pyobj_T = NULL;
    Mat T;
    PyObject* pyobj_R1 = NULL;
    Mat R1;
    PyObject* pyobj_R2 = NULL;
    Mat R2;

    const char* keywords[] = { "R", "T", "R1", "R2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OO:stereoRectify", (char**)keywords, &pyobj_R, &pyobj_T, &pyobj_R1, &pyobj_R2) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 0)) &&
        pyopencv_to(pyobj_T, T, ArgInfo("T", 0)) &&
        pyopencv_to(pyobj_R1, R1, ArgInfo("R1", 1)) &&
        pyopencv_to(pyobj_R2, R2, ArgInfo("R2", 1)) )
    {
        ERRWRAP2(cv::omnidir::stereoRectify(R, T, R1, R2));
        return Py_BuildValue("(NN)", pyopencv_from(R1), pyopencv_from(R2));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_R = NULL;
    UMat R;
    PyObject* pyobj_T = NULL;
    UMat T;
    PyObject* pyobj_R1 = NULL;
    UMat R1;
    PyObject* pyobj_R2 = NULL;
    UMat R2;

    const char* keywords[] = { "R", "T", "R1", "R2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OO:stereoRectify", (char**)keywords, &pyobj_R, &pyobj_T, &pyobj_R1, &pyobj_R2) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 0)) &&
        pyopencv_to(pyobj_T, T, ArgInfo("T", 0)) &&
        pyopencv_to(pyobj_R1, R1, ArgInfo("R1", 1)) &&
        pyopencv_to(pyobj_R2, R2, ArgInfo("R2", 1)) )
    {
        ERRWRAP2(cv::omnidir::stereoRectify(R, T, R1, R2));
        return Py_BuildValue("(NN)", pyopencv_from(R1), pyopencv_from(R2));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_omnidir_undistortImage(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::omnidir;

    {
    PyObject* pyobj_distorted = NULL;
    Mat distorted;
    PyObject* pyobj_undistorted = NULL;
    Mat undistorted;
    PyObject* pyobj_K = NULL;
    Mat K;
    PyObject* pyobj_D = NULL;
    Mat D;
    PyObject* pyobj_xi = NULL;
    Mat xi;
    int flags=0;
    PyObject* pyobj_Knew = NULL;
    Mat Knew=cv::Mat();
    PyObject* pyobj_new_size = NULL;
    Size new_size;
    PyObject* pyobj_R = NULL;
    Mat R=Mat::eye(3, 3, CV_64F);

    const char* keywords[] = { "distorted", "K", "D", "xi", "flags", "undistorted", "Knew", "new_size", "R", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOi|OOOO:undistortImage", (char**)keywords, &pyobj_distorted, &pyobj_K, &pyobj_D, &pyobj_xi, &flags, &pyobj_undistorted, &pyobj_Knew, &pyobj_new_size, &pyobj_R) &&
        pyopencv_to(pyobj_distorted, distorted, ArgInfo("distorted", 0)) &&
        pyopencv_to(pyobj_undistorted, undistorted, ArgInfo("undistorted", 1)) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 0)) &&
        pyopencv_to(pyobj_D, D, ArgInfo("D", 0)) &&
        pyopencv_to(pyobj_xi, xi, ArgInfo("xi", 0)) &&
        pyopencv_to(pyobj_Knew, Knew, ArgInfo("Knew", 0)) &&
        pyopencv_to(pyobj_new_size, new_size, ArgInfo("new_size", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 0)) )
    {
        ERRWRAP2(cv::omnidir::undistortImage(distorted, undistorted, K, D, xi, flags, Knew, new_size, R));
        return pyopencv_from(undistorted);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_distorted = NULL;
    UMat distorted;
    PyObject* pyobj_undistorted = NULL;
    UMat undistorted;
    PyObject* pyobj_K = NULL;
    UMat K;
    PyObject* pyobj_D = NULL;
    UMat D;
    PyObject* pyobj_xi = NULL;
    UMat xi;
    int flags=0;
    PyObject* pyobj_Knew = NULL;
    UMat Knew=cv::UMat();
    PyObject* pyobj_new_size = NULL;
    Size new_size;
    PyObject* pyobj_R = NULL;
    UMat R=UMat::eye(3, 3, CV_64F);

    const char* keywords[] = { "distorted", "K", "D", "xi", "flags", "undistorted", "Knew", "new_size", "R", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOi|OOOO:undistortImage", (char**)keywords, &pyobj_distorted, &pyobj_K, &pyobj_D, &pyobj_xi, &flags, &pyobj_undistorted, &pyobj_Knew, &pyobj_new_size, &pyobj_R) &&
        pyopencv_to(pyobj_distorted, distorted, ArgInfo("distorted", 0)) &&
        pyopencv_to(pyobj_undistorted, undistorted, ArgInfo("undistorted", 1)) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 0)) &&
        pyopencv_to(pyobj_D, D, ArgInfo("D", 0)) &&
        pyopencv_to(pyobj_xi, xi, ArgInfo("xi", 0)) &&
        pyopencv_to(pyobj_Knew, Knew, ArgInfo("Knew", 0)) &&
        pyopencv_to(pyobj_new_size, new_size, ArgInfo("new_size", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 0)) )
    {
        ERRWRAP2(cv::omnidir::undistortImage(distorted, undistorted, K, D, xi, flags, Knew, new_size, R));
        return pyopencv_from(undistorted);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_omnidir_undistortPoints(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::omnidir;

    {
    PyObject* pyobj_distorted = NULL;
    Mat distorted;
    PyObject* pyobj_undistorted = NULL;
    Mat undistorted;
    PyObject* pyobj_K = NULL;
    Mat K;
    PyObject* pyobj_D = NULL;
    Mat D;
    PyObject* pyobj_xi = NULL;
    Mat xi;
    PyObject* pyobj_R = NULL;
    Mat R;

    const char* keywords[] = { "distorted", "K", "D", "xi", "R", "undistorted", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOO|O:undistortPoints", (char**)keywords, &pyobj_distorted, &pyobj_K, &pyobj_D, &pyobj_xi, &pyobj_R, &pyobj_undistorted) &&
        pyopencv_to(pyobj_distorted, distorted, ArgInfo("distorted", 0)) &&
        pyopencv_to(pyobj_undistorted, undistorted, ArgInfo("undistorted", 1)) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 0)) &&
        pyopencv_to(pyobj_D, D, ArgInfo("D", 0)) &&
        pyopencv_to(pyobj_xi, xi, ArgInfo("xi", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 0)) )
    {
        ERRWRAP2(cv::omnidir::undistortPoints(distorted, undistorted, K, D, xi, R));
        return pyopencv_from(undistorted);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_distorted = NULL;
    UMat distorted;
    PyObject* pyobj_undistorted = NULL;
    UMat undistorted;
    PyObject* pyobj_K = NULL;
    UMat K;
    PyObject* pyobj_D = NULL;
    UMat D;
    PyObject* pyobj_xi = NULL;
    UMat xi;
    PyObject* pyobj_R = NULL;
    UMat R;

    const char* keywords[] = { "distorted", "K", "D", "xi", "R", "undistorted", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOO|O:undistortPoints", (char**)keywords, &pyobj_distorted, &pyobj_K, &pyobj_D, &pyobj_xi, &pyobj_R, &pyobj_undistorted) &&
        pyopencv_to(pyobj_distorted, distorted, ArgInfo("distorted", 0)) &&
        pyopencv_to(pyobj_undistorted, undistorted, ArgInfo("undistorted", 1)) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 0)) &&
        pyopencv_to(pyobj_D, D, ArgInfo("D", 0)) &&
        pyopencv_to(pyobj_xi, xi, ArgInfo("xi", 0)) &&
        pyopencv_to(pyobj_R, R, ArgInfo("R", 0)) )
    {
        ERRWRAP2(cv::omnidir::undistortPoints(distorted, undistorted, K, D, xi, R));
        return pyopencv_from(undistorted);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_optflow_calcOpticalFlowSF(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::optflow;

    {
    PyObject* pyobj_from = NULL;
    Mat from;
    PyObject* pyobj_to = NULL;
    Mat to;
    PyObject* pyobj_flow = NULL;
    Mat flow;
    int layers=0;
    int averaging_block_size=0;
    int max_flow=0;

    const char* keywords[] = { "from", "to", "layers", "averaging_block_size", "max_flow", "flow", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOiii|O:calcOpticalFlowSF", (char**)keywords, &pyobj_from, &pyobj_to, &layers, &averaging_block_size, &max_flow, &pyobj_flow) &&
        pyopencv_to(pyobj_from, from, ArgInfo("from", 0)) &&
        pyopencv_to(pyobj_to, to, ArgInfo("to", 0)) &&
        pyopencv_to(pyobj_flow, flow, ArgInfo("flow", 1)) )
    {
        ERRWRAP2(cv::optflow::calcOpticalFlowSF(from, to, flow, layers, averaging_block_size, max_flow));
        return pyopencv_from(flow);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_from = NULL;
    UMat from;
    PyObject* pyobj_to = NULL;
    UMat to;
    PyObject* pyobj_flow = NULL;
    UMat flow;
    int layers=0;
    int averaging_block_size=0;
    int max_flow=0;

    const char* keywords[] = { "from", "to", "layers", "averaging_block_size", "max_flow", "flow", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOiii|O:calcOpticalFlowSF", (char**)keywords, &pyobj_from, &pyobj_to, &layers, &averaging_block_size, &max_flow, &pyobj_flow) &&
        pyopencv_to(pyobj_from, from, ArgInfo("from", 0)) &&
        pyopencv_to(pyobj_to, to, ArgInfo("to", 0)) &&
        pyopencv_to(pyobj_flow, flow, ArgInfo("flow", 1)) )
    {
        ERRWRAP2(cv::optflow::calcOpticalFlowSF(from, to, flow, layers, averaging_block_size, max_flow));
        return pyopencv_from(flow);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_from = NULL;
    Mat from;
    PyObject* pyobj_to = NULL;
    Mat to;
    PyObject* pyobj_flow = NULL;
    Mat flow;
    int layers=0;
    int averaging_block_size=0;
    int max_flow=0;
    double sigma_dist=0;
    double sigma_color=0;
    int postprocess_window=0;
    double sigma_dist_fix=0;
    double sigma_color_fix=0;
    double occ_thr=0;
    int upscale_averaging_radius=0;
    double upscale_sigma_dist=0;
    double upscale_sigma_color=0;
    double speed_up_thr=0;

    const char* keywords[] = { "from", "to", "layers", "averaging_block_size", "max_flow", "sigma_dist", "sigma_color", "postprocess_window", "sigma_dist_fix", "sigma_color_fix", "occ_thr", "upscale_averaging_radius", "upscale_sigma_dist", "upscale_sigma_color", "speed_up_thr", "flow", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOiiiddidddiddd|O:calcOpticalFlowSF", (char**)keywords, &pyobj_from, &pyobj_to, &layers, &averaging_block_size, &max_flow, &sigma_dist, &sigma_color, &postprocess_window, &sigma_dist_fix, &sigma_color_fix, &occ_thr, &upscale_averaging_radius, &upscale_sigma_dist, &upscale_sigma_color, &speed_up_thr, &pyobj_flow) &&
        pyopencv_to(pyobj_from, from, ArgInfo("from", 0)) &&
        pyopencv_to(pyobj_to, to, ArgInfo("to", 0)) &&
        pyopencv_to(pyobj_flow, flow, ArgInfo("flow", 1)) )
    {
        ERRWRAP2(cv::optflow::calcOpticalFlowSF(from, to, flow, layers, averaging_block_size, max_flow, sigma_dist, sigma_color, postprocess_window, sigma_dist_fix, sigma_color_fix, occ_thr, upscale_averaging_radius, upscale_sigma_dist, upscale_sigma_color, speed_up_thr));
        return pyopencv_from(flow);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_from = NULL;
    UMat from;
    PyObject* pyobj_to = NULL;
    UMat to;
    PyObject* pyobj_flow = NULL;
    UMat flow;
    int layers=0;
    int averaging_block_size=0;
    int max_flow=0;
    double sigma_dist=0;
    double sigma_color=0;
    int postprocess_window=0;
    double sigma_dist_fix=0;
    double sigma_color_fix=0;
    double occ_thr=0;
    int upscale_averaging_radius=0;
    double upscale_sigma_dist=0;
    double upscale_sigma_color=0;
    double speed_up_thr=0;

    const char* keywords[] = { "from", "to", "layers", "averaging_block_size", "max_flow", "sigma_dist", "sigma_color", "postprocess_window", "sigma_dist_fix", "sigma_color_fix", "occ_thr", "upscale_averaging_radius", "upscale_sigma_dist", "upscale_sigma_color", "speed_up_thr", "flow", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOiiiddidddiddd|O:calcOpticalFlowSF", (char**)keywords, &pyobj_from, &pyobj_to, &layers, &averaging_block_size, &max_flow, &sigma_dist, &sigma_color, &postprocess_window, &sigma_dist_fix, &sigma_color_fix, &occ_thr, &upscale_averaging_radius, &upscale_sigma_dist, &upscale_sigma_color, &speed_up_thr, &pyobj_flow) &&
        pyopencv_to(pyobj_from, from, ArgInfo("from", 0)) &&
        pyopencv_to(pyobj_to, to, ArgInfo("to", 0)) &&
        pyopencv_to(pyobj_flow, flow, ArgInfo("flow", 1)) )
    {
        ERRWRAP2(cv::optflow::calcOpticalFlowSF(from, to, flow, layers, averaging_block_size, max_flow, sigma_dist, sigma_color, postprocess_window, sigma_dist_fix, sigma_color_fix, occ_thr, upscale_averaging_radius, upscale_sigma_dist, upscale_sigma_color, speed_up_thr));
        return pyopencv_from(flow);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_optflow_calcOpticalFlowSparseToDense(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::optflow;

    {
    PyObject* pyobj_from = NULL;
    Mat from;
    PyObject* pyobj_to = NULL;
    Mat to;
    PyObject* pyobj_flow = NULL;
    Mat flow;
    int grid_step=8;
    int k=128;
    float sigma=0.05f;
    bool use_post_proc=true;
    float fgs_lambda=500.0f;
    float fgs_sigma=1.5f;

    const char* keywords[] = { "from", "to", "flow", "grid_step", "k", "sigma", "use_post_proc", "fgs_lambda", "fgs_sigma", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Oiifbff:calcOpticalFlowSparseToDense", (char**)keywords, &pyobj_from, &pyobj_to, &pyobj_flow, &grid_step, &k, &sigma, &use_post_proc, &fgs_lambda, &fgs_sigma) &&
        pyopencv_to(pyobj_from, from, ArgInfo("from", 0)) &&
        pyopencv_to(pyobj_to, to, ArgInfo("to", 0)) &&
        pyopencv_to(pyobj_flow, flow, ArgInfo("flow", 1)) )
    {
        ERRWRAP2(cv::optflow::calcOpticalFlowSparseToDense(from, to, flow, grid_step, k, sigma, use_post_proc, fgs_lambda, fgs_sigma));
        return pyopencv_from(flow);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_from = NULL;
    UMat from;
    PyObject* pyobj_to = NULL;
    UMat to;
    PyObject* pyobj_flow = NULL;
    UMat flow;
    int grid_step=8;
    int k=128;
    float sigma=0.05f;
    bool use_post_proc=true;
    float fgs_lambda=500.0f;
    float fgs_sigma=1.5f;

    const char* keywords[] = { "from", "to", "flow", "grid_step", "k", "sigma", "use_post_proc", "fgs_lambda", "fgs_sigma", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Oiifbff:calcOpticalFlowSparseToDense", (char**)keywords, &pyobj_from, &pyobj_to, &pyobj_flow, &grid_step, &k, &sigma, &use_post_proc, &fgs_lambda, &fgs_sigma) &&
        pyopencv_to(pyobj_from, from, ArgInfo("from", 0)) &&
        pyopencv_to(pyobj_to, to, ArgInfo("to", 0)) &&
        pyopencv_to(pyobj_flow, flow, ArgInfo("flow", 1)) )
    {
        ERRWRAP2(cv::optflow::calcOpticalFlowSparseToDense(from, to, flow, grid_step, k, sigma, use_post_proc, fgs_lambda, fgs_sigma));
        return pyopencv_from(flow);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_optflow_createOptFlow_DIS(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::optflow;

    int preset=DISOpticalFlow::PRESET_FAST;
    Ptr<DISOpticalFlow> retval;

    const char* keywords[] = { "preset", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|i:createOptFlow_DIS", (char**)keywords, &preset) )
    {
        ERRWRAP2(retval = cv::optflow::createOptFlow_DIS(preset));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_optflow_createOptFlow_DeepFlow(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::optflow;

    Ptr<DenseOpticalFlow> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::optflow::createOptFlow_DeepFlow());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_optflow_createOptFlow_Farneback(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::optflow;

    Ptr<DenseOpticalFlow> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::optflow::createOptFlow_Farneback());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_optflow_createOptFlow_PCAFlow(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::optflow;

    Ptr<DenseOpticalFlow> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::optflow::createOptFlow_PCAFlow());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_optflow_createOptFlow_SimpleFlow(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::optflow;

    Ptr<DenseOpticalFlow> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::optflow::createOptFlow_SimpleFlow());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_optflow_createOptFlow_SparseToDense(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::optflow;

    Ptr<DenseOpticalFlow> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::optflow::createOptFlow_SparseToDense());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_optflow_createVariationalFlowRefinement(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::optflow;

    Ptr<VariationalRefinement> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::optflow::createVariationalFlowRefinement());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_optflow_readOpticalFlow(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::optflow;

    PyObject* pyobj_path = NULL;
    String path;
    Mat retval;

    const char* keywords[] = { "path", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:readOpticalFlow", (char**)keywords, &pyobj_path) &&
        pyopencv_to(pyobj_path, path, ArgInfo("path", 0)) )
    {
        ERRWRAP2(retval = cv::optflow::readOpticalFlow(path));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_optflow_writeOpticalFlow(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::optflow;

    {
    PyObject* pyobj_path = NULL;
    String path;
    PyObject* pyobj_flow = NULL;
    Mat flow;
    bool retval;

    const char* keywords[] = { "path", "flow", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:writeOpticalFlow", (char**)keywords, &pyobj_path, &pyobj_flow) &&
        pyopencv_to(pyobj_path, path, ArgInfo("path", 0)) &&
        pyopencv_to(pyobj_flow, flow, ArgInfo("flow", 0)) )
    {
        ERRWRAP2(retval = cv::optflow::writeOpticalFlow(path, flow));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_path = NULL;
    String path;
    PyObject* pyobj_flow = NULL;
    UMat flow;
    bool retval;

    const char* keywords[] = { "path", "flow", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:writeOpticalFlow", (char**)keywords, &pyobj_path, &pyobj_flow) &&
        pyopencv_to(pyobj_path, path, ArgInfo("path", 0)) &&
        pyopencv_to(pyobj_flow, flow, ArgInfo("flow", 0)) )
    {
        ERRWRAP2(retval = cv::optflow::writeOpticalFlow(path, flow));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_plot_createPlot2d(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::plot;

    {
    PyObject* pyobj_data = NULL;
    Mat data;
    Ptr<Plot2d> retval;

    const char* keywords[] = { "data", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:createPlot2d", (char**)keywords, &pyobj_data) &&
        pyopencv_to(pyobj_data, data, ArgInfo("data", 0)) )
    {
        ERRWRAP2(retval = cv::plot::createPlot2d(data));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_data = NULL;
    UMat data;
    Ptr<Plot2d> retval;

    const char* keywords[] = { "data", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:createPlot2d", (char**)keywords, &pyobj_data) &&
        pyopencv_to(pyobj_data, data, ArgInfo("data", 0)) )
    {
        ERRWRAP2(retval = cv::plot::createPlot2d(data));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_dataX = NULL;
    Mat dataX;
    PyObject* pyobj_dataY = NULL;
    Mat dataY;
    Ptr<Plot2d> retval;

    const char* keywords[] = { "dataX", "dataY", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:createPlot2d", (char**)keywords, &pyobj_dataX, &pyobj_dataY) &&
        pyopencv_to(pyobj_dataX, dataX, ArgInfo("dataX", 0)) &&
        pyopencv_to(pyobj_dataY, dataY, ArgInfo("dataY", 0)) )
    {
        ERRWRAP2(retval = cv::plot::createPlot2d(dataX, dataY));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_dataX = NULL;
    UMat dataX;
    PyObject* pyobj_dataY = NULL;
    UMat dataY;
    Ptr<Plot2d> retval;

    const char* keywords[] = { "dataX", "dataY", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:createPlot2d", (char**)keywords, &pyobj_dataX, &pyobj_dataY) &&
        pyopencv_to(pyobj_dataX, dataX, ArgInfo("dataX", 0)) &&
        pyopencv_to(pyobj_dataY, dataY, ArgInfo("dataY", 0)) )
    {
        ERRWRAP2(retval = cv::plot::createPlot2d(dataX, dataY));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ppf_match_3d_ppf_match_3d_ICP_ICP(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ppf_match_3d;

    {
    pyopencv_ppf_match_3d_ICP_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_ppf_match_3d_ICP_t, &pyopencv_ppf_match_3d_ICP_Type);
        new (&(self->v)) Ptr<cv::ppf_match_3d::ICP>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::ppf_match_3d::ICP()));
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    int iterations=0;
    float tolerence=0.05f;
    float rejectionScale=2.5f;
    int numLevels=6;
    int sampleType=ICP::ICP_SAMPLING_TYPE_UNIFORM;
    int numMaxCorr=1;
    pyopencv_ppf_match_3d_ICP_t* self = 0;

    const char* keywords[] = { "iterations", "tolerence", "rejectionScale", "numLevels", "sampleType", "numMaxCorr", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i|ffiii:ICP", (char**)keywords, &iterations, &tolerence, &rejectionScale, &numLevels, &sampleType, &numMaxCorr) )
    {
        self = PyObject_NEW(pyopencv_ppf_match_3d_ICP_t, &pyopencv_ppf_match_3d_ICP_Type);
        new (&(self->v)) Ptr<cv::ppf_match_3d::ICP>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::ppf_match_3d::ICP(iterations, tolerence, rejectionScale, numLevels, sampleType, numMaxCorr)));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ppf_match_3d_computeNormalsPC3d(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ppf_match_3d;

    {
    PyObject* pyobj_PC = NULL;
    Mat PC;
    PyObject* pyobj_PCNormals = NULL;
    Mat PCNormals;
    int NumNeighbors=0;
    bool FlipViewpoint=0;
    PyObject* pyobj_viewpoint = NULL;
    Vec3d viewpoint;
    int retval;

    const char* keywords[] = { "PC", "NumNeighbors", "FlipViewpoint", "viewpoint", "PCNormals", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OibO|O:computeNormalsPC3d", (char**)keywords, &pyobj_PC, &NumNeighbors, &FlipViewpoint, &pyobj_viewpoint, &pyobj_PCNormals) &&
        pyopencv_to(pyobj_PC, PC, ArgInfo("PC", 0)) &&
        pyopencv_to(pyobj_PCNormals, PCNormals, ArgInfo("PCNormals", 1)) &&
        pyopencv_to(pyobj_viewpoint, viewpoint, ArgInfo("viewpoint", 0)) )
    {
        ERRWRAP2(retval = cv::ppf_match_3d::computeNormalsPC3d(PC, PCNormals, NumNeighbors, FlipViewpoint, viewpoint));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(PCNormals));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_PC = NULL;
    Mat PC;
    PyObject* pyobj_PCNormals = NULL;
    Mat PCNormals;
    int NumNeighbors=0;
    bool FlipViewpoint=0;
    PyObject* pyobj_viewpoint = NULL;
    Vec3d viewpoint;
    int retval;

    const char* keywords[] = { "PC", "NumNeighbors", "FlipViewpoint", "viewpoint", "PCNormals", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OibO|O:computeNormalsPC3d", (char**)keywords, &pyobj_PC, &NumNeighbors, &FlipViewpoint, &pyobj_viewpoint, &pyobj_PCNormals) &&
        pyopencv_to(pyobj_PC, PC, ArgInfo("PC", 0)) &&
        pyopencv_to(pyobj_PCNormals, PCNormals, ArgInfo("PCNormals", 1)) &&
        pyopencv_to(pyobj_viewpoint, viewpoint, ArgInfo("viewpoint", 0)) )
    {
        ERRWRAP2(retval = cv::ppf_match_3d::computeNormalsPC3d(PC, PCNormals, NumNeighbors, FlipViewpoint, viewpoint));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(PCNormals));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_reg_reg_MapAffine_MapAffine(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::reg;

    pyopencv_reg_MapAffine_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_reg_MapAffine_t, &pyopencv_reg_MapAffine_Type);
        new (&(self->v)) Ptr<cv::reg::MapAffine>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::reg::MapAffine()));
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_cv_reg_reg_MapProjec_MapProjec(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::reg;

    pyopencv_reg_MapProjec_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_reg_MapProjec_t, &pyopencv_reg_MapProjec_Type);
        new (&(self->v)) Ptr<cv::reg::MapProjec>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::reg::MapProjec()));
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_cv_reg_reg_MapShift_MapShift(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::reg;

    {
    pyopencv_reg_MapShift_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_reg_MapShift_t, &pyopencv_reg_MapShift_Type);
        new (&(self->v)) Ptr<cv::reg::MapShift>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::reg::MapShift()));
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_shift = NULL;
    Mat shift;
    pyopencv_reg_MapShift_t* self = 0;

    const char* keywords[] = { "shift", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:MapShift", (char**)keywords, &pyobj_shift) &&
        pyopencv_to(pyobj_shift, shift, ArgInfo("shift", 0)) )
    {
        self = PyObject_NEW(pyopencv_reg_MapShift_t, &pyopencv_reg_MapShift_Type);
        new (&(self->v)) Ptr<cv::reg::MapShift>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::reg::MapShift(shift)));
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_shift = NULL;
    UMat shift;
    pyopencv_reg_MapShift_t* self = 0;

    const char* keywords[] = { "shift", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:MapShift", (char**)keywords, &pyobj_shift) &&
        pyopencv_to(pyobj_shift, shift, ArgInfo("shift", 0)) )
    {
        self = PyObject_NEW(pyopencv_reg_MapShift_t, &pyopencv_reg_MapShift_Type);
        new (&(self->v)) Ptr<cv::reg::MapShift>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::reg::MapShift(shift)));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_reg_MapTypeCaster_toAffine(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::reg;

    PyObject* pyobj_sourceMap = NULL;
    Ptr<Map> sourceMap;
    Ptr<MapAffine> retval;

    const char* keywords[] = { "sourceMap", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:MapTypeCaster_toAffine", (char**)keywords, &pyobj_sourceMap) &&
        pyopencv_to(pyobj_sourceMap, sourceMap, ArgInfo("sourceMap", 0)) )
    {
        ERRWRAP2(retval = cv::reg::MapTypeCaster::toAffine(sourceMap));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_reg_MapTypeCaster_toProjec(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::reg;

    PyObject* pyobj_sourceMap = NULL;
    Ptr<Map> sourceMap;
    Ptr<MapProjec> retval;

    const char* keywords[] = { "sourceMap", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:MapTypeCaster_toProjec", (char**)keywords, &pyobj_sourceMap) &&
        pyopencv_to(pyobj_sourceMap, sourceMap, ArgInfo("sourceMap", 0)) )
    {
        ERRWRAP2(retval = cv::reg::MapTypeCaster::toProjec(sourceMap));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_reg_MapTypeCaster_toShift(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::reg;

    PyObject* pyobj_sourceMap = NULL;
    Ptr<Map> sourceMap;
    Ptr<MapShift> retval;

    const char* keywords[] = { "sourceMap", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:MapTypeCaster_toShift", (char**)keywords, &pyobj_sourceMap) &&
        pyopencv_to(pyobj_sourceMap, sourceMap, ArgInfo("sourceMap", 0)) )
    {
        ERRWRAP2(retval = cv::reg::MapTypeCaster::toShift(sourceMap));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_reg_reg_MapperGradAffine_MapperGradAffine(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::reg;

    pyopencv_reg_MapperGradAffine_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_reg_MapperGradAffine_t, &pyopencv_reg_MapperGradAffine_Type);
        new (&(self->v)) Ptr<cv::reg::MapperGradAffine>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::reg::MapperGradAffine()));
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_cv_reg_reg_MapperGradEuclid_MapperGradEuclid(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::reg;

    pyopencv_reg_MapperGradEuclid_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_reg_MapperGradEuclid_t, &pyopencv_reg_MapperGradEuclid_Type);
        new (&(self->v)) Ptr<cv::reg::MapperGradEuclid>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::reg::MapperGradEuclid()));
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_cv_reg_reg_MapperGradProj_MapperGradProj(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::reg;

    pyopencv_reg_MapperGradProj_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_reg_MapperGradProj_t, &pyopencv_reg_MapperGradProj_Type);
        new (&(self->v)) Ptr<cv::reg::MapperGradProj>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::reg::MapperGradProj()));
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_cv_reg_reg_MapperGradShift_MapperGradShift(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::reg;

    pyopencv_reg_MapperGradShift_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_reg_MapperGradShift_t, &pyopencv_reg_MapperGradShift_Type);
        new (&(self->v)) Ptr<cv::reg::MapperGradShift>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::reg::MapperGradShift()));
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_cv_reg_reg_MapperGradSimilar_MapperGradSimilar(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::reg;

    pyopencv_reg_MapperGradSimilar_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_reg_MapperGradSimilar_t, &pyopencv_reg_MapperGradSimilar_Type);
        new (&(self->v)) Ptr<cv::reg::MapperGradSimilar>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::reg::MapperGradSimilar()));
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_cv_reg_reg_MapperPyramid_MapperPyramid(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::reg;

    PyObject* pyobj_baseMapper = NULL;
    Ptr<Mapper> baseMapper;
    pyopencv_reg_MapperPyramid_t* self = 0;

    const char* keywords[] = { "baseMapper", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:MapperPyramid", (char**)keywords, &pyobj_baseMapper) &&
        pyopencv_to(pyobj_baseMapper, baseMapper, ArgInfo("baseMapper", 0)) )
    {
        self = PyObject_NEW(pyopencv_reg_MapperPyramid_t, &pyopencv_reg_MapperPyramid_Type);
        new (&(self->v)) Ptr<cv::reg::MapperPyramid>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v.reset(new cv::reg::MapperPyramid(baseMapper)));
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_cv_rgbd_depthTo3d(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::rgbd;

    {
    PyObject* pyobj_depth = NULL;
    Mat depth;
    PyObject* pyobj_K = NULL;
    Mat K;
    PyObject* pyobj_points3d = NULL;
    Mat points3d;
    PyObject* pyobj_mask = NULL;
    Mat mask;

    const char* keywords[] = { "depth", "K", "points3d", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OO:depthTo3d", (char**)keywords, &pyobj_depth, &pyobj_K, &pyobj_points3d, &pyobj_mask) &&
        pyopencv_to(pyobj_depth, depth, ArgInfo("depth", 0)) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 0)) &&
        pyopencv_to(pyobj_points3d, points3d, ArgInfo("points3d", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::rgbd::depthTo3d(depth, K, points3d, mask));
        return pyopencv_from(points3d);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_depth = NULL;
    UMat depth;
    PyObject* pyobj_K = NULL;
    UMat K;
    PyObject* pyobj_points3d = NULL;
    UMat points3d;
    PyObject* pyobj_mask = NULL;
    UMat mask;

    const char* keywords[] = { "depth", "K", "points3d", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OO:depthTo3d", (char**)keywords, &pyobj_depth, &pyobj_K, &pyobj_points3d, &pyobj_mask) &&
        pyopencv_to(pyobj_depth, depth, ArgInfo("depth", 0)) &&
        pyopencv_to(pyobj_K, K, ArgInfo("K", 0)) &&
        pyopencv_to(pyobj_points3d, points3d, ArgInfo("points3d", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(cv::rgbd::depthTo3d(depth, K, points3d, mask));
        return pyopencv_from(points3d);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_rgbd_warpFrame(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::rgbd;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_depth = NULL;
    Mat depth;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    PyObject* pyobj_Rt = NULL;
    Mat Rt;
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_distCoeff = NULL;
    Mat distCoeff;
    PyObject* pyobj_warpedImage = NULL;
    Mat warpedImage;
    PyObject* pyobj_warpedDepth = NULL;
    Mat warpedDepth;
    PyObject* pyobj_warpedMask = NULL;
    Mat warpedMask;

    const char* keywords[] = { "image", "depth", "mask", "Rt", "cameraMatrix", "distCoeff", "warpedImage", "warpedDepth", "warpedMask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOO|OOO:warpFrame", (char**)keywords, &pyobj_image, &pyobj_depth, &pyobj_mask, &pyobj_Rt, &pyobj_cameraMatrix, &pyobj_distCoeff, &pyobj_warpedImage, &pyobj_warpedDepth, &pyobj_warpedMask) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_depth, depth, ArgInfo("depth", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) &&
        pyopencv_to(pyobj_Rt, Rt, ArgInfo("Rt", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeff, distCoeff, ArgInfo("distCoeff", 0)) &&
        pyopencv_to(pyobj_warpedImage, warpedImage, ArgInfo("warpedImage", 1)) &&
        pyopencv_to(pyobj_warpedDepth, warpedDepth, ArgInfo("warpedDepth", 1)) &&
        pyopencv_to(pyobj_warpedMask, warpedMask, ArgInfo("warpedMask", 1)) )
    {
        ERRWRAP2(cv::rgbd::warpFrame(image, depth, mask, Rt, cameraMatrix, distCoeff, warpedImage, warpedDepth, warpedMask));
        return Py_BuildValue("(NNN)", pyopencv_from(warpedImage), pyopencv_from(warpedDepth), pyopencv_from(warpedMask));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_depth = NULL;
    Mat depth;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    PyObject* pyobj_Rt = NULL;
    Mat Rt;
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_distCoeff = NULL;
    Mat distCoeff;
    PyObject* pyobj_warpedImage = NULL;
    UMat warpedImage;
    PyObject* pyobj_warpedDepth = NULL;
    UMat warpedDepth;
    PyObject* pyobj_warpedMask = NULL;
    UMat warpedMask;

    const char* keywords[] = { "image", "depth", "mask", "Rt", "cameraMatrix", "distCoeff", "warpedImage", "warpedDepth", "warpedMask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOOO|OOO:warpFrame", (char**)keywords, &pyobj_image, &pyobj_depth, &pyobj_mask, &pyobj_Rt, &pyobj_cameraMatrix, &pyobj_distCoeff, &pyobj_warpedImage, &pyobj_warpedDepth, &pyobj_warpedMask) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_depth, depth, ArgInfo("depth", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) &&
        pyopencv_to(pyobj_Rt, Rt, ArgInfo("Rt", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeff, distCoeff, ArgInfo("distCoeff", 0)) &&
        pyopencv_to(pyobj_warpedImage, warpedImage, ArgInfo("warpedImage", 1)) &&
        pyopencv_to(pyobj_warpedDepth, warpedDepth, ArgInfo("warpedDepth", 1)) &&
        pyopencv_to(pyobj_warpedMask, warpedMask, ArgInfo("warpedMask", 1)) )
    {
        ERRWRAP2(cv::rgbd::warpFrame(image, depth, mask, Rt, cameraMatrix, distCoeff, warpedImage, warpedDepth, warpedMask));
        return Py_BuildValue("(NNN)", pyopencv_from(warpedImage), pyopencv_from(warpedDepth), pyopencv_from(warpedMask));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_saliency_MotionSaliencyBinWangApr2014_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::saliency;

    Ptr<MotionSaliencyBinWangApr2014> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::saliency::MotionSaliencyBinWangApr2014::create());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_saliency_ObjectnessBING_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::saliency;

    Ptr<ObjectnessBING> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::saliency::ObjectnessBING::create());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_saliency_StaticSaliencyFineGrained_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::saliency;

    Ptr<StaticSaliencyFineGrained> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::saliency::StaticSaliencyFineGrained::create());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_saliency_StaticSaliencySpectralResidual_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::saliency;

    Ptr<StaticSaliencySpectralResidual> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::saliency::StaticSaliencySpectralResidual::create());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_structured_light_GrayCodePattern_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::structured_light;

    int width=0;
    int height=0;
    Ptr<GrayCodePattern> retval;

    const char* keywords[] = { "width", "height", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "ii:GrayCodePattern_create", (char**)keywords, &width, &height) )
    {
        ERRWRAP2(retval = cv::structured_light::GrayCodePattern::create(width, height));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_text_OCRBeamSearchDecoder_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::text;

    {
    PyObject* pyobj_classifier = NULL;
    Ptr<OCRBeamSearchDecoder::ClassifierCallback> classifier;
    PyObject* pyobj_vocabulary = NULL;
    String vocabulary;
    PyObject* pyobj_transition_probabilities_table = NULL;
    Mat transition_probabilities_table;
    PyObject* pyobj_emission_probabilities_table = NULL;
    Mat emission_probabilities_table;
    int mode=OCR_DECODER_VITERBI;
    int beam_size=500;
    Ptr<OCRBeamSearchDecoder> retval;

    const char* keywords[] = { "classifier", "vocabulary", "transition_probabilities_table", "emission_probabilities_table", "mode", "beam_size", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|ii:OCRBeamSearchDecoder_create", (char**)keywords, &pyobj_classifier, &pyobj_vocabulary, &pyobj_transition_probabilities_table, &pyobj_emission_probabilities_table, &mode, &beam_size) &&
        pyopencv_to(pyobj_classifier, classifier, ArgInfo("classifier", 0)) &&
        pyopencv_to(pyobj_vocabulary, vocabulary, ArgInfo("vocabulary", 0)) &&
        pyopencv_to(pyobj_transition_probabilities_table, transition_probabilities_table, ArgInfo("transition_probabilities_table", 0)) &&
        pyopencv_to(pyobj_emission_probabilities_table, emission_probabilities_table, ArgInfo("emission_probabilities_table", 0)) )
    {
        ERRWRAP2(retval = cv::text::OCRBeamSearchDecoder::create(classifier, vocabulary, transition_probabilities_table, emission_probabilities_table, mode, beam_size));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_classifier = NULL;
    Ptr<OCRBeamSearchDecoder::ClassifierCallback> classifier;
    PyObject* pyobj_vocabulary = NULL;
    String vocabulary;
    PyObject* pyobj_transition_probabilities_table = NULL;
    UMat transition_probabilities_table;
    PyObject* pyobj_emission_probabilities_table = NULL;
    UMat emission_probabilities_table;
    int mode=OCR_DECODER_VITERBI;
    int beam_size=500;
    Ptr<OCRBeamSearchDecoder> retval;

    const char* keywords[] = { "classifier", "vocabulary", "transition_probabilities_table", "emission_probabilities_table", "mode", "beam_size", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|ii:OCRBeamSearchDecoder_create", (char**)keywords, &pyobj_classifier, &pyobj_vocabulary, &pyobj_transition_probabilities_table, &pyobj_emission_probabilities_table, &mode, &beam_size) &&
        pyopencv_to(pyobj_classifier, classifier, ArgInfo("classifier", 0)) &&
        pyopencv_to(pyobj_vocabulary, vocabulary, ArgInfo("vocabulary", 0)) &&
        pyopencv_to(pyobj_transition_probabilities_table, transition_probabilities_table, ArgInfo("transition_probabilities_table", 0)) &&
        pyopencv_to(pyobj_emission_probabilities_table, emission_probabilities_table, ArgInfo("emission_probabilities_table", 0)) )
    {
        ERRWRAP2(retval = cv::text::OCRBeamSearchDecoder::create(classifier, vocabulary, transition_probabilities_table, emission_probabilities_table, mode, beam_size));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_text_OCRHMMDecoder_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::text;

    {
    PyObject* pyobj_classifier = NULL;
    Ptr<OCRHMMDecoder::ClassifierCallback> classifier;
    PyObject* pyobj_vocabulary = NULL;
    String vocabulary;
    PyObject* pyobj_transition_probabilities_table = NULL;
    Mat transition_probabilities_table;
    PyObject* pyobj_emission_probabilities_table = NULL;
    Mat emission_probabilities_table;
    int mode=OCR_DECODER_VITERBI;
    Ptr<OCRHMMDecoder> retval;

    const char* keywords[] = { "classifier", "vocabulary", "transition_probabilities_table", "emission_probabilities_table", "mode", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|i:OCRHMMDecoder_create", (char**)keywords, &pyobj_classifier, &pyobj_vocabulary, &pyobj_transition_probabilities_table, &pyobj_emission_probabilities_table, &mode) &&
        pyopencv_to(pyobj_classifier, classifier, ArgInfo("classifier", 0)) &&
        pyopencv_to(pyobj_vocabulary, vocabulary, ArgInfo("vocabulary", 0)) &&
        pyopencv_to(pyobj_transition_probabilities_table, transition_probabilities_table, ArgInfo("transition_probabilities_table", 0)) &&
        pyopencv_to(pyobj_emission_probabilities_table, emission_probabilities_table, ArgInfo("emission_probabilities_table", 0)) )
    {
        ERRWRAP2(retval = cv::text::OCRHMMDecoder::create(classifier, vocabulary, transition_probabilities_table, emission_probabilities_table, mode));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_classifier = NULL;
    Ptr<OCRHMMDecoder::ClassifierCallback> classifier;
    PyObject* pyobj_vocabulary = NULL;
    String vocabulary;
    PyObject* pyobj_transition_probabilities_table = NULL;
    UMat transition_probabilities_table;
    PyObject* pyobj_emission_probabilities_table = NULL;
    UMat emission_probabilities_table;
    int mode=OCR_DECODER_VITERBI;
    Ptr<OCRHMMDecoder> retval;

    const char* keywords[] = { "classifier", "vocabulary", "transition_probabilities_table", "emission_probabilities_table", "mode", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|i:OCRHMMDecoder_create", (char**)keywords, &pyobj_classifier, &pyobj_vocabulary, &pyobj_transition_probabilities_table, &pyobj_emission_probabilities_table, &mode) &&
        pyopencv_to(pyobj_classifier, classifier, ArgInfo("classifier", 0)) &&
        pyopencv_to(pyobj_vocabulary, vocabulary, ArgInfo("vocabulary", 0)) &&
        pyopencv_to(pyobj_transition_probabilities_table, transition_probabilities_table, ArgInfo("transition_probabilities_table", 0)) &&
        pyopencv_to(pyobj_emission_probabilities_table, emission_probabilities_table, ArgInfo("emission_probabilities_table", 0)) )
    {
        ERRWRAP2(retval = cv::text::OCRHMMDecoder::create(classifier, vocabulary, transition_probabilities_table, emission_probabilities_table, mode));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_text_OCRTesseract_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::text;

    char* datapath=0;
    char* language=0;
    char* char_whitelist=0;
    int oem=3;
    int psmode=3;
    Ptr<OCRTesseract> retval;

    const char* keywords[] = { "datapath", "language", "char_whitelist", "oem", "psmode", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|sssii:OCRTesseract_create", (char**)keywords, &datapath, &language, &char_whitelist, &oem, &psmode) )
    {
        ERRWRAP2(retval = cv::text::OCRTesseract::create(datapath, language, char_whitelist, oem, psmode));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_text_computeNMChannels(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::text;

    {
    PyObject* pyobj__src = NULL;
    Mat _src;
    PyObject* pyobj__channels = NULL;
    vector_Mat _channels;
    int _mode=ERFILTER_NM_RGBLGrad;

    const char* keywords[] = { "_src", "_channels", "_mode", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oi:computeNMChannels", (char**)keywords, &pyobj__src, &pyobj__channels, &_mode) &&
        pyopencv_to(pyobj__src, _src, ArgInfo("_src", 0)) &&
        pyopencv_to(pyobj__channels, _channels, ArgInfo("_channels", 1)) )
    {
        ERRWRAP2(cv::text::computeNMChannels(_src, _channels, _mode));
        return pyopencv_from(_channels);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj__src = NULL;
    UMat _src;
    PyObject* pyobj__channels = NULL;
    vector_Mat _channels;
    int _mode=ERFILTER_NM_RGBLGrad;

    const char* keywords[] = { "_src", "_channels", "_mode", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oi:computeNMChannels", (char**)keywords, &pyobj__src, &pyobj__channels, &_mode) &&
        pyopencv_to(pyobj__src, _src, ArgInfo("_src", 0)) &&
        pyopencv_to(pyobj__channels, _channels, ArgInfo("_channels", 1)) )
    {
        ERRWRAP2(cv::text::computeNMChannels(_src, _channels, _mode));
        return pyopencv_from(_channels);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_text_createERFilterNM1(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::text;

    PyObject* pyobj_cb = NULL;
    Ptr<ERFilter::Callback> cb;
    int thresholdDelta=1;
    float minArea=(float)0.00025;
    float maxArea=(float)0.13;
    float minProbability=(float)0.4;
    bool nonMaxSuppression=true;
    float minProbabilityDiff=(float)0.1;
    Ptr<ERFilter> retval;

    const char* keywords[] = { "cb", "thresholdDelta", "minArea", "maxArea", "minProbability", "nonMaxSuppression", "minProbabilityDiff", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|ifffbf:createERFilterNM1", (char**)keywords, &pyobj_cb, &thresholdDelta, &minArea, &maxArea, &minProbability, &nonMaxSuppression, &minProbabilityDiff) &&
        pyopencv_to(pyobj_cb, cb, ArgInfo("cb", 0)) )
    {
        ERRWRAP2(retval = cv::text::createERFilterNM1(cb, thresholdDelta, minArea, maxArea, minProbability, nonMaxSuppression, minProbabilityDiff));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_text_createERFilterNM2(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::text;

    PyObject* pyobj_cb = NULL;
    Ptr<ERFilter::Callback> cb;
    float minProbability=(float)0.3;
    Ptr<ERFilter> retval;

    const char* keywords[] = { "cb", "minProbability", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|f:createERFilterNM2", (char**)keywords, &pyobj_cb, &minProbability) &&
        pyopencv_to(pyobj_cb, cb, ArgInfo("cb", 0)) )
    {
        ERRWRAP2(retval = cv::text::createERFilterNM2(cb, minProbability));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_text_createOCRHMMTransitionsTable(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::text;

    PyObject* pyobj_vocabulary = NULL;
    String vocabulary;
    PyObject* pyobj_lexicon = NULL;
    vector_String lexicon;
    Mat retval;

    const char* keywords[] = { "vocabulary", "lexicon", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:createOCRHMMTransitionsTable", (char**)keywords, &pyobj_vocabulary, &pyobj_lexicon) &&
        pyopencv_to(pyobj_vocabulary, vocabulary, ArgInfo("vocabulary", 0)) &&
        pyopencv_to(pyobj_lexicon, lexicon, ArgInfo("lexicon", 0)) )
    {
        ERRWRAP2(retval = cv::text::createOCRHMMTransitionsTable(vocabulary, lexicon));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_text_detectRegions(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::text;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_er_filter1 = NULL;
    Ptr<ERFilter> er_filter1;
    PyObject* pyobj_er_filter2 = NULL;
    Ptr<ERFilter> er_filter2;
    vector_vector_Point regions;

    const char* keywords[] = { "image", "er_filter1", "er_filter2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:detectRegions", (char**)keywords, &pyobj_image, &pyobj_er_filter1, &pyobj_er_filter2) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_er_filter1, er_filter1, ArgInfo("er_filter1", 0)) &&
        pyopencv_to(pyobj_er_filter2, er_filter2, ArgInfo("er_filter2", 0)) )
    {
        ERRWRAP2(cv::text::detectRegions(image, er_filter1, er_filter2, regions));
        return pyopencv_from(regions);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_er_filter1 = NULL;
    Ptr<ERFilter> er_filter1;
    PyObject* pyobj_er_filter2 = NULL;
    Ptr<ERFilter> er_filter2;
    vector_vector_Point regions;

    const char* keywords[] = { "image", "er_filter1", "er_filter2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:detectRegions", (char**)keywords, &pyobj_image, &pyobj_er_filter1, &pyobj_er_filter2) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_er_filter1, er_filter1, ArgInfo("er_filter1", 0)) &&
        pyopencv_to(pyobj_er_filter2, er_filter2, ArgInfo("er_filter2", 0)) )
    {
        ERRWRAP2(cv::text::detectRegions(image, er_filter1, er_filter2, regions));
        return pyopencv_from(regions);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_text_erGrouping(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::text;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_channel = NULL;
    Mat channel;
    PyObject* pyobj_regions = NULL;
    vector_vector_Point regions;
    vector_Rect groups_rects;
    int method=ERGROUPING_ORIENTATION_HORIZ;
    PyObject* pyobj_filename = NULL;
    String filename;
    float minProbablity=(float)0.5;

    const char* keywords[] = { "image", "channel", "regions", "method", "filename", "minProbablity", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|iOf:erGrouping", (char**)keywords, &pyobj_image, &pyobj_channel, &pyobj_regions, &method, &pyobj_filename, &minProbablity) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_channel, channel, ArgInfo("channel", 0)) &&
        pyopencv_to(pyobj_regions, regions, ArgInfo("regions", 0)) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) )
    {
        ERRWRAP2(cv::text::erGrouping(image, channel, regions, groups_rects, method, filename, minProbablity));
        return pyopencv_from(groups_rects);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_channel = NULL;
    UMat channel;
    PyObject* pyobj_regions = NULL;
    vector_vector_Point regions;
    vector_Rect groups_rects;
    int method=ERGROUPING_ORIENTATION_HORIZ;
    PyObject* pyobj_filename = NULL;
    String filename;
    float minProbablity=(float)0.5;

    const char* keywords[] = { "image", "channel", "regions", "method", "filename", "minProbablity", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|iOf:erGrouping", (char**)keywords, &pyobj_image, &pyobj_channel, &pyobj_regions, &method, &pyobj_filename, &minProbablity) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_channel, channel, ArgInfo("channel", 0)) &&
        pyopencv_to(pyobj_regions, regions, ArgInfo("regions", 0)) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) )
    {
        ERRWRAP2(cv::text::erGrouping(image, channel, regions, groups_rects, method, filename, minProbablity));
        return pyopencv_from(groups_rects);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_text_loadClassifierNM1(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::text;

    PyObject* pyobj_filename = NULL;
    String filename;
    Ptr<ERFilter::Callback> retval;

    const char* keywords[] = { "filename", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:loadClassifierNM1", (char**)keywords, &pyobj_filename) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) )
    {
        ERRWRAP2(retval = cv::text::loadClassifierNM1(filename));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_text_loadClassifierNM2(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::text;

    PyObject* pyobj_filename = NULL;
    String filename;
    Ptr<ERFilter::Callback> retval;

    const char* keywords[] = { "filename", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:loadClassifierNM2", (char**)keywords, &pyobj_filename) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) )
    {
        ERRWRAP2(retval = cv::text::loadClassifierNM2(filename));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_text_loadOCRBeamSearchClassifierCNN(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::text;

    PyObject* pyobj_filename = NULL;
    String filename;
    Ptr<OCRBeamSearchDecoder::ClassifierCallback> retval;

    const char* keywords[] = { "filename", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:loadOCRBeamSearchClassifierCNN", (char**)keywords, &pyobj_filename) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) )
    {
        ERRWRAP2(retval = cv::text::loadOCRBeamSearchClassifierCNN(filename));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_text_loadOCRHMMClassifierCNN(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::text;

    PyObject* pyobj_filename = NULL;
    String filename;
    Ptr<OCRHMMDecoder::ClassifierCallback> retval;

    const char* keywords[] = { "filename", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:loadOCRHMMClassifierCNN", (char**)keywords, &pyobj_filename) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) )
    {
        ERRWRAP2(retval = cv::text::loadOCRHMMClassifierCNN(filename));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_text_loadOCRHMMClassifierNM(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::text;

    PyObject* pyobj_filename = NULL;
    String filename;
    Ptr<OCRHMMDecoder::ClassifierCallback> retval;

    const char* keywords[] = { "filename", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:loadOCRHMMClassifierNM", (char**)keywords, &pyobj_filename) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) )
    {
        ERRWRAP2(retval = cv::text::loadOCRHMMClassifierNM(filename));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_xfeatures2d_BoostDesc_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::xfeatures2d;

    int desc=BoostDesc::BINBOOST_256;
    bool use_scale_orientation=true;
    float scale_factor=6.25f;
    Ptr<BoostDesc> retval;

    const char* keywords[] = { "desc", "use_scale_orientation", "scale_factor", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|ibf:BoostDesc_create", (char**)keywords, &desc, &use_scale_orientation, &scale_factor) )
    {
        ERRWRAP2(retval = cv::xfeatures2d::BoostDesc::create(desc, use_scale_orientation, scale_factor));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_xfeatures2d_BriefDescriptorExtractor_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::xfeatures2d;

    int bytes=32;
    bool use_orientation=false;
    Ptr<BriefDescriptorExtractor> retval;

    const char* keywords[] = { "bytes", "use_orientation", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|ib:BriefDescriptorExtractor_create", (char**)keywords, &bytes, &use_orientation) )
    {
        ERRWRAP2(retval = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, use_orientation));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_xfeatures2d_DAISY_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::xfeatures2d;

    {
    float radius=15;
    int q_radius=3;
    int q_theta=8;
    int q_hist=8;
    int norm=DAISY::NRM_NONE;
    PyObject* pyobj_H = NULL;
    Mat H;
    bool interpolation=true;
    bool use_orientation=false;
    Ptr<DAISY> retval;

    const char* keywords[] = { "radius", "q_radius", "q_theta", "q_hist", "norm", "H", "interpolation", "use_orientation", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|fiiiiObb:DAISY_create", (char**)keywords, &radius, &q_radius, &q_theta, &q_hist, &norm, &pyobj_H, &interpolation, &use_orientation) &&
        pyopencv_to(pyobj_H, H, ArgInfo("H", 0)) )
    {
        ERRWRAP2(retval = cv::xfeatures2d::DAISY::create(radius, q_radius, q_theta, q_hist, norm, H, interpolation, use_orientation));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    float radius=15;
    int q_radius=3;
    int q_theta=8;
    int q_hist=8;
    int norm=DAISY::NRM_NONE;
    PyObject* pyobj_H = NULL;
    UMat H;
    bool interpolation=true;
    bool use_orientation=false;
    Ptr<DAISY> retval;

    const char* keywords[] = { "radius", "q_radius", "q_theta", "q_hist", "norm", "H", "interpolation", "use_orientation", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|fiiiiObb:DAISY_create", (char**)keywords, &radius, &q_radius, &q_theta, &q_hist, &norm, &pyobj_H, &interpolation, &use_orientation) &&
        pyopencv_to(pyobj_H, H, ArgInfo("H", 0)) )
    {
        ERRWRAP2(retval = cv::xfeatures2d::DAISY::create(radius, q_radius, q_theta, q_hist, norm, H, interpolation, use_orientation));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_xfeatures2d_FREAK_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::xfeatures2d;

    bool orientationNormalized=true;
    bool scaleNormalized=true;
    float patternScale=22.0f;
    int nOctaves=4;
    PyObject* pyobj_selectedPairs = NULL;
    vector_int selectedPairs=std::vector<int>();
    Ptr<FREAK> retval;

    const char* keywords[] = { "orientationNormalized", "scaleNormalized", "patternScale", "nOctaves", "selectedPairs", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|bbfiO:FREAK_create", (char**)keywords, &orientationNormalized, &scaleNormalized, &patternScale, &nOctaves, &pyobj_selectedPairs) &&
        pyopencv_to(pyobj_selectedPairs, selectedPairs, ArgInfo("selectedPairs", 0)) )
    {
        ERRWRAP2(retval = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale, nOctaves, selectedPairs));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_xfeatures2d_HarrisLaplaceFeatureDetector_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::xfeatures2d;

    int numOctaves=6;
    float corn_thresh=0.01f;
    float DOG_thresh=0.01f;
    int maxCorners=5000;
    int num_layers=4;
    Ptr<HarrisLaplaceFeatureDetector> retval;

    const char* keywords[] = { "numOctaves", "corn_thresh", "DOG_thresh", "maxCorners", "num_layers", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|iffii:HarrisLaplaceFeatureDetector_create", (char**)keywords, &numOctaves, &corn_thresh, &DOG_thresh, &maxCorners, &num_layers) )
    {
        ERRWRAP2(retval = cv::xfeatures2d::HarrisLaplaceFeatureDetector::create(numOctaves, corn_thresh, DOG_thresh, maxCorners, num_layers));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_xfeatures2d_LATCH_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::xfeatures2d;

    int bytes=32;
    bool rotationInvariance=true;
    int half_ssd_size=3;
    Ptr<LATCH> retval;

    const char* keywords[] = { "bytes", "rotationInvariance", "half_ssd_size", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|ibi:LATCH_create", (char**)keywords, &bytes, &rotationInvariance, &half_ssd_size) )
    {
        ERRWRAP2(retval = cv::xfeatures2d::LATCH::create(bytes, rotationInvariance, half_ssd_size));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_xfeatures2d_LUCID_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::xfeatures2d;

    int lucid_kernel=1;
    int blur_kernel=2;
    Ptr<LUCID> retval;

    const char* keywords[] = { "lucid_kernel", "blur_kernel", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|ii:LUCID_create", (char**)keywords, &lucid_kernel, &blur_kernel) )
    {
        ERRWRAP2(retval = cv::xfeatures2d::LUCID::create(lucid_kernel, blur_kernel));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_xfeatures2d_PCTSignaturesSQFD_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::xfeatures2d;

    int distanceFunction=3;
    int similarityFunction=2;
    float similarityParameter=1.0f;
    Ptr<PCTSignaturesSQFD> retval;

    const char* keywords[] = { "distanceFunction", "similarityFunction", "similarityParameter", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|iif:PCTSignaturesSQFD_create", (char**)keywords, &distanceFunction, &similarityFunction, &similarityParameter) )
    {
        ERRWRAP2(retval = cv::xfeatures2d::PCTSignaturesSQFD::create(distanceFunction, similarityFunction, similarityParameter));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_xfeatures2d_PCTSignatures_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::xfeatures2d;

    {
    int initSampleCount=2000;
    int initSeedCount=400;
    int pointDistribution=0;
    Ptr<PCTSignatures> retval;

    const char* keywords[] = { "initSampleCount", "initSeedCount", "pointDistribution", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|iii:PCTSignatures_create", (char**)keywords, &initSampleCount, &initSeedCount, &pointDistribution) )
    {
        ERRWRAP2(retval = cv::xfeatures2d::PCTSignatures::create(initSampleCount, initSeedCount, pointDistribution));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_initSamplingPoints = NULL;
    vector_Point2f initSamplingPoints;
    int initSeedCount=0;
    Ptr<PCTSignatures> retval;

    const char* keywords[] = { "initSamplingPoints", "initSeedCount", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi:PCTSignatures_create", (char**)keywords, &pyobj_initSamplingPoints, &initSeedCount) &&
        pyopencv_to(pyobj_initSamplingPoints, initSamplingPoints, ArgInfo("initSamplingPoints", 0)) )
    {
        ERRWRAP2(retval = cv::xfeatures2d::PCTSignatures::create(initSamplingPoints, initSeedCount));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_initSamplingPoints = NULL;
    vector_Point2f initSamplingPoints;
    PyObject* pyobj_initClusterSeedIndexes = NULL;
    vector_int initClusterSeedIndexes;
    Ptr<PCTSignatures> retval;

    const char* keywords[] = { "initSamplingPoints", "initClusterSeedIndexes", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:PCTSignatures_create", (char**)keywords, &pyobj_initSamplingPoints, &pyobj_initClusterSeedIndexes) &&
        pyopencv_to(pyobj_initSamplingPoints, initSamplingPoints, ArgInfo("initSamplingPoints", 0)) &&
        pyopencv_to(pyobj_initClusterSeedIndexes, initClusterSeedIndexes, ArgInfo("initClusterSeedIndexes", 0)) )
    {
        ERRWRAP2(retval = cv::xfeatures2d::PCTSignatures::create(initSamplingPoints, initClusterSeedIndexes));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_xfeatures2d_PCTSignatures_drawSignature(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::xfeatures2d;

    {
    PyObject* pyobj_source = NULL;
    Mat source;
    PyObject* pyobj_signature = NULL;
    Mat signature;
    PyObject* pyobj_result = NULL;
    Mat result;
    float radiusToShorterSideRatio=1.0 / 8;
    int borderThickness=1;

    const char* keywords[] = { "source", "signature", "result", "radiusToShorterSideRatio", "borderThickness", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Ofi:PCTSignatures_drawSignature", (char**)keywords, &pyobj_source, &pyobj_signature, &pyobj_result, &radiusToShorterSideRatio, &borderThickness) &&
        pyopencv_to(pyobj_source, source, ArgInfo("source", 0)) &&
        pyopencv_to(pyobj_signature, signature, ArgInfo("signature", 0)) &&
        pyopencv_to(pyobj_result, result, ArgInfo("result", 1)) )
    {
        ERRWRAP2(cv::xfeatures2d::PCTSignatures::drawSignature(source, signature, result, radiusToShorterSideRatio, borderThickness));
        return pyopencv_from(result);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_source = NULL;
    UMat source;
    PyObject* pyobj_signature = NULL;
    UMat signature;
    PyObject* pyobj_result = NULL;
    UMat result;
    float radiusToShorterSideRatio=1.0 / 8;
    int borderThickness=1;

    const char* keywords[] = { "source", "signature", "result", "radiusToShorterSideRatio", "borderThickness", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Ofi:PCTSignatures_drawSignature", (char**)keywords, &pyobj_source, &pyobj_signature, &pyobj_result, &radiusToShorterSideRatio, &borderThickness) &&
        pyopencv_to(pyobj_source, source, ArgInfo("source", 0)) &&
        pyopencv_to(pyobj_signature, signature, ArgInfo("signature", 0)) &&
        pyopencv_to(pyobj_result, result, ArgInfo("result", 1)) )
    {
        ERRWRAP2(cv::xfeatures2d::PCTSignatures::drawSignature(source, signature, result, radiusToShorterSideRatio, borderThickness));
        return pyopencv_from(result);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_xfeatures2d_PCTSignatures_generateInitPoints(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::xfeatures2d;

    PyObject* pyobj_initPoints = NULL;
    vector_Point2f initPoints;
    int count=0;
    int pointDistribution=0;

    const char* keywords[] = { "initPoints", "count", "pointDistribution", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii:PCTSignatures_generateInitPoints", (char**)keywords, &pyobj_initPoints, &count, &pointDistribution) &&
        pyopencv_to(pyobj_initPoints, initPoints, ArgInfo("initPoints", 0)) )
    {
        ERRWRAP2(cv::xfeatures2d::PCTSignatures::generateInitPoints(initPoints, count, pointDistribution));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_xfeatures2d_SIFT_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::xfeatures2d;

    int nfeatures=0;
    int nOctaveLayers=3;
    double contrastThreshold=0.04;
    double edgeThreshold=10;
    double sigma=1.6;
    Ptr<SIFT> retval;

    const char* keywords[] = { "nfeatures", "nOctaveLayers", "contrastThreshold", "edgeThreshold", "sigma", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|iiddd:SIFT_create", (char**)keywords, &nfeatures, &nOctaveLayers, &contrastThreshold, &edgeThreshold, &sigma) )
    {
        ERRWRAP2(retval = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_xfeatures2d_SURF_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::xfeatures2d;

    double hessianThreshold=100;
    int nOctaves=4;
    int nOctaveLayers=3;
    bool extended=false;
    bool upright=false;
    Ptr<SURF> retval;

    const char* keywords[] = { "hessianThreshold", "nOctaves", "nOctaveLayers", "extended", "upright", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|diibb:SURF_create", (char**)keywords, &hessianThreshold, &nOctaves, &nOctaveLayers, &extended, &upright) )
    {
        ERRWRAP2(retval = cv::xfeatures2d::SURF::create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_xfeatures2d_StarDetector_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::xfeatures2d;

    int maxSize=45;
    int responseThreshold=30;
    int lineThresholdProjected=10;
    int lineThresholdBinarized=8;
    int suppressNonmaxSize=5;
    Ptr<StarDetector> retval;

    const char* keywords[] = { "maxSize", "responseThreshold", "lineThresholdProjected", "lineThresholdBinarized", "suppressNonmaxSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|iiiii:StarDetector_create", (char**)keywords, &maxSize, &responseThreshold, &lineThresholdProjected, &lineThresholdBinarized, &suppressNonmaxSize) )
    {
        ERRWRAP2(retval = cv::xfeatures2d::StarDetector::create(maxSize, responseThreshold, lineThresholdProjected, lineThresholdBinarized, suppressNonmaxSize));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_xfeatures2d_VGG_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::xfeatures2d;

    int desc=VGG::VGG_120;
    float isigma=1.4f;
    bool img_normalize=true;
    bool use_scale_orientation=true;
    float scale_factor=6.25f;
    bool dsc_normalize=false;
    Ptr<VGG> retval;

    const char* keywords[] = { "desc", "isigma", "img_normalize", "use_scale_orientation", "scale_factor", "dsc_normalize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|ifbbfb:VGG_create", (char**)keywords, &desc, &isigma, &img_normalize, &use_scale_orientation, &scale_factor, &dsc_normalize) )
    {
        ERRWRAP2(retval = cv::xfeatures2d::VGG::create(desc, isigma, img_normalize, use_scale_orientation, scale_factor, dsc_normalize));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_AdaptiveManifoldFilter_create(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    Ptr<AdaptiveManifoldFilter> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::ximgproc::AdaptiveManifoldFilter::create());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_amFilter(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    {
    PyObject* pyobj_joint = NULL;
    Mat joint;
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    double sigma_s=0;
    double sigma_r=0;
    bool adjust_outliers=false;

    const char* keywords[] = { "joint", "src", "sigma_s", "sigma_r", "dst", "adjust_outliers", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOdd|Ob:amFilter", (char**)keywords, &pyobj_joint, &pyobj_src, &sigma_s, &sigma_r, &pyobj_dst, &adjust_outliers) &&
        pyopencv_to(pyobj_joint, joint, ArgInfo("joint", 0)) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::ximgproc::amFilter(joint, src, dst, sigma_s, sigma_r, adjust_outliers));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_joint = NULL;
    UMat joint;
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    double sigma_s=0;
    double sigma_r=0;
    bool adjust_outliers=false;

    const char* keywords[] = { "joint", "src", "sigma_s", "sigma_r", "dst", "adjust_outliers", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOdd|Ob:amFilter", (char**)keywords, &pyobj_joint, &pyobj_src, &sigma_s, &sigma_r, &pyobj_dst, &adjust_outliers) &&
        pyopencv_to(pyobj_joint, joint, ArgInfo("joint", 0)) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::ximgproc::amFilter(joint, src, dst, sigma_s, sigma_r, adjust_outliers));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_bilateralTextureFilter(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int fr=3;
    int numIter=1;
    double sigmaAlpha=-1.;
    double sigmaAvg=-1.;

    const char* keywords[] = { "src", "dst", "fr", "numIter", "sigmaAlpha", "sigmaAvg", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oiidd:bilateralTextureFilter", (char**)keywords, &pyobj_src, &pyobj_dst, &fr, &numIter, &sigmaAlpha, &sigmaAvg) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::ximgproc::bilateralTextureFilter(src, dst, fr, numIter, sigmaAlpha, sigmaAvg));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int fr=3;
    int numIter=1;
    double sigmaAlpha=-1.;
    double sigmaAvg=-1.;

    const char* keywords[] = { "src", "dst", "fr", "numIter", "sigmaAlpha", "sigmaAvg", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oiidd:bilateralTextureFilter", (char**)keywords, &pyobj_src, &pyobj_dst, &fr, &numIter, &sigmaAlpha, &sigmaAvg) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::ximgproc::bilateralTextureFilter(src, dst, fr, numIter, sigmaAlpha, sigmaAvg));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_covarianceEstimation(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int windowRows=0;
    int windowCols=0;

    const char* keywords[] = { "src", "windowRows", "windowCols", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii|O:covarianceEstimation", (char**)keywords, &pyobj_src, &windowRows, &windowCols, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::ximgproc::covarianceEstimation(src, dst, windowRows, windowCols));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int windowRows=0;
    int windowCols=0;

    const char* keywords[] = { "src", "windowRows", "windowCols", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii|O:covarianceEstimation", (char**)keywords, &pyobj_src, &windowRows, &windowCols, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::ximgproc::covarianceEstimation(src, dst, windowRows, windowCols));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_createAMFilter(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    double sigma_s=0;
    double sigma_r=0;
    bool adjust_outliers=false;
    Ptr<AdaptiveManifoldFilter> retval;

    const char* keywords[] = { "sigma_s", "sigma_r", "adjust_outliers", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "dd|b:createAMFilter", (char**)keywords, &sigma_s, &sigma_r, &adjust_outliers) )
    {
        ERRWRAP2(retval = cv::ximgproc::createAMFilter(sigma_s, sigma_r, adjust_outliers));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_createDTFilter(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    {
    PyObject* pyobj_guide = NULL;
    Mat guide;
    double sigmaSpatial=0;
    double sigmaColor=0;
    int mode=DTF_NC;
    int numIters=3;
    Ptr<DTFilter> retval;

    const char* keywords[] = { "guide", "sigmaSpatial", "sigmaColor", "mode", "numIters", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odd|ii:createDTFilter", (char**)keywords, &pyobj_guide, &sigmaSpatial, &sigmaColor, &mode, &numIters) &&
        pyopencv_to(pyobj_guide, guide, ArgInfo("guide", 0)) )
    {
        ERRWRAP2(retval = cv::ximgproc::createDTFilter(guide, sigmaSpatial, sigmaColor, mode, numIters));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_guide = NULL;
    UMat guide;
    double sigmaSpatial=0;
    double sigmaColor=0;
    int mode=DTF_NC;
    int numIters=3;
    Ptr<DTFilter> retval;

    const char* keywords[] = { "guide", "sigmaSpatial", "sigmaColor", "mode", "numIters", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odd|ii:createDTFilter", (char**)keywords, &pyobj_guide, &sigmaSpatial, &sigmaColor, &mode, &numIters) &&
        pyopencv_to(pyobj_guide, guide, ArgInfo("guide", 0)) )
    {
        ERRWRAP2(retval = cv::ximgproc::createDTFilter(guide, sigmaSpatial, sigmaColor, mode, numIters));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_createDisparityWLSFilter(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    PyObject* pyobj_matcher_left = NULL;
    Ptr<StereoMatcher> matcher_left;
    Ptr<DisparityWLSFilter> retval;

    const char* keywords[] = { "matcher_left", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:createDisparityWLSFilter", (char**)keywords, &pyobj_matcher_left) &&
        pyopencv_to(pyobj_matcher_left, matcher_left, ArgInfo("matcher_left", 0)) )
    {
        ERRWRAP2(retval = cv::ximgproc::createDisparityWLSFilter(matcher_left));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_createDisparityWLSFilterGeneric(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    bool use_confidence=0;
    Ptr<DisparityWLSFilter> retval;

    const char* keywords[] = { "use_confidence", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:createDisparityWLSFilterGeneric", (char**)keywords, &use_confidence) )
    {
        ERRWRAP2(retval = cv::ximgproc::createDisparityWLSFilterGeneric(use_confidence));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_createEdgeAwareInterpolator(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    Ptr<EdgeAwareInterpolator> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::ximgproc::createEdgeAwareInterpolator());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_createFastGlobalSmootherFilter(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    {
    PyObject* pyobj_guide = NULL;
    Mat guide;
    double lambda=0;
    double sigma_color=0;
    double lambda_attenuation=0.25;
    int num_iter=3;
    Ptr<FastGlobalSmootherFilter> retval;

    const char* keywords[] = { "guide", "lambda", "sigma_color", "lambda_attenuation", "num_iter", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odd|di:createFastGlobalSmootherFilter", (char**)keywords, &pyobj_guide, &lambda, &sigma_color, &lambda_attenuation, &num_iter) &&
        pyopencv_to(pyobj_guide, guide, ArgInfo("guide", 0)) )
    {
        ERRWRAP2(retval = cv::ximgproc::createFastGlobalSmootherFilter(guide, lambda, sigma_color, lambda_attenuation, num_iter));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_guide = NULL;
    UMat guide;
    double lambda=0;
    double sigma_color=0;
    double lambda_attenuation=0.25;
    int num_iter=3;
    Ptr<FastGlobalSmootherFilter> retval;

    const char* keywords[] = { "guide", "lambda", "sigma_color", "lambda_attenuation", "num_iter", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odd|di:createFastGlobalSmootherFilter", (char**)keywords, &pyobj_guide, &lambda, &sigma_color, &lambda_attenuation, &num_iter) &&
        pyopencv_to(pyobj_guide, guide, ArgInfo("guide", 0)) )
    {
        ERRWRAP2(retval = cv::ximgproc::createFastGlobalSmootherFilter(guide, lambda, sigma_color, lambda_attenuation, num_iter));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_createFastLineDetector(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    int _length_threshold=10;
    float _distance_threshold=1.414213562f;
    double _canny_th1=50.0;
    double _canny_th2=50.0;
    int _canny_aperture_size=3;
    bool _do_merge=false;
    Ptr<FastLineDetector> retval;

    const char* keywords[] = { "_length_threshold", "_distance_threshold", "_canny_th1", "_canny_th2", "_canny_aperture_size", "_do_merge", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|ifddib:createFastLineDetector", (char**)keywords, &_length_threshold, &_distance_threshold, &_canny_th1, &_canny_th2, &_canny_aperture_size, &_do_merge) )
    {
        ERRWRAP2(retval = cv::ximgproc::createFastLineDetector(_length_threshold, _distance_threshold, _canny_th1, _canny_th2, _canny_aperture_size, _do_merge));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_createGuidedFilter(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    {
    PyObject* pyobj_guide = NULL;
    Mat guide;
    int radius=0;
    double eps=0;
    Ptr<GuidedFilter> retval;

    const char* keywords[] = { "guide", "radius", "eps", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oid:createGuidedFilter", (char**)keywords, &pyobj_guide, &radius, &eps) &&
        pyopencv_to(pyobj_guide, guide, ArgInfo("guide", 0)) )
    {
        ERRWRAP2(retval = cv::ximgproc::createGuidedFilter(guide, radius, eps));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_guide = NULL;
    UMat guide;
    int radius=0;
    double eps=0;
    Ptr<GuidedFilter> retval;

    const char* keywords[] = { "guide", "radius", "eps", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oid:createGuidedFilter", (char**)keywords, &pyobj_guide, &radius, &eps) &&
        pyopencv_to(pyobj_guide, guide, ArgInfo("guide", 0)) )
    {
        ERRWRAP2(retval = cv::ximgproc::createGuidedFilter(guide, radius, eps));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_createRFFeatureGetter(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    Ptr<RFFeatureGetter> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::ximgproc::createRFFeatureGetter());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_createRightMatcher(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    PyObject* pyobj_matcher_left = NULL;
    Ptr<StereoMatcher> matcher_left;
    Ptr<StereoMatcher> retval;

    const char* keywords[] = { "matcher_left", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:createRightMatcher", (char**)keywords, &pyobj_matcher_left) &&
        pyopencv_to(pyobj_matcher_left, matcher_left, ArgInfo("matcher_left", 0)) )
    {
        ERRWRAP2(retval = cv::ximgproc::createRightMatcher(matcher_left));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_createStructuredEdgeDetection(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    PyObject* pyobj_model = NULL;
    String model;
    PyObject* pyobj_howToGetFeatures = NULL;
    Ptr<RFFeatureGetter> howToGetFeatures;
    Ptr<StructuredEdgeDetection> retval;

    const char* keywords[] = { "model", "howToGetFeatures", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:createStructuredEdgeDetection", (char**)keywords, &pyobj_model, &pyobj_howToGetFeatures) &&
        pyopencv_to(pyobj_model, model, ArgInfo("model", 0)) &&
        pyopencv_to(pyobj_howToGetFeatures, howToGetFeatures, ArgInfo("howToGetFeatures", 0)) )
    {
        ERRWRAP2(retval = cv::ximgproc::createStructuredEdgeDetection(model, howToGetFeatures));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_createSuperpixelLSC(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    int region_size=10;
    float ratio=0.075f;
    Ptr<SuperpixelLSC> retval;

    const char* keywords[] = { "image", "region_size", "ratio", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|if:createSuperpixelLSC", (char**)keywords, &pyobj_image, &region_size, &ratio) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) )
    {
        ERRWRAP2(retval = cv::ximgproc::createSuperpixelLSC(image, region_size, ratio));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    int region_size=10;
    float ratio=0.075f;
    Ptr<SuperpixelLSC> retval;

    const char* keywords[] = { "image", "region_size", "ratio", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|if:createSuperpixelLSC", (char**)keywords, &pyobj_image, &region_size, &ratio) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) )
    {
        ERRWRAP2(retval = cv::ximgproc::createSuperpixelLSC(image, region_size, ratio));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_createSuperpixelSEEDS(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    int image_width=0;
    int image_height=0;
    int image_channels=0;
    int num_superpixels=0;
    int num_levels=0;
    int prior=2;
    int histogram_bins=5;
    bool double_step=false;
    Ptr<SuperpixelSEEDS> retval;

    const char* keywords[] = { "image_width", "image_height", "image_channels", "num_superpixels", "num_levels", "prior", "histogram_bins", "double_step", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "iiiii|iib:createSuperpixelSEEDS", (char**)keywords, &image_width, &image_height, &image_channels, &num_superpixels, &num_levels, &prior, &histogram_bins, &double_step) )
    {
        ERRWRAP2(retval = cv::ximgproc::createSuperpixelSEEDS(image_width, image_height, image_channels, num_superpixels, num_levels, prior, histogram_bins, double_step));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_createSuperpixelSLIC(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    int algorithm=SLICO;
    int region_size=10;
    float ruler=10.0f;
    Ptr<SuperpixelSLIC> retval;

    const char* keywords[] = { "image", "algorithm", "region_size", "ruler", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|iif:createSuperpixelSLIC", (char**)keywords, &pyobj_image, &algorithm, &region_size, &ruler) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) )
    {
        ERRWRAP2(retval = cv::ximgproc::createSuperpixelSLIC(image, algorithm, region_size, ruler));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    int algorithm=SLICO;
    int region_size=10;
    float ruler=10.0f;
    Ptr<SuperpixelSLIC> retval;

    const char* keywords[] = { "image", "algorithm", "region_size", "ruler", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|iif:createSuperpixelSLIC", (char**)keywords, &pyobj_image, &algorithm, &region_size, &ruler) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) )
    {
        ERRWRAP2(retval = cv::ximgproc::createSuperpixelSLIC(image, algorithm, region_size, ruler));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_dtFilter(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    {
    PyObject* pyobj_guide = NULL;
    Mat guide;
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    double sigmaSpatial=0;
    double sigmaColor=0;
    int mode=DTF_NC;
    int numIters=3;

    const char* keywords[] = { "guide", "src", "sigmaSpatial", "sigmaColor", "dst", "mode", "numIters", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOdd|Oii:dtFilter", (char**)keywords, &pyobj_guide, &pyobj_src, &sigmaSpatial, &sigmaColor, &pyobj_dst, &mode, &numIters) &&
        pyopencv_to(pyobj_guide, guide, ArgInfo("guide", 0)) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::ximgproc::dtFilter(guide, src, dst, sigmaSpatial, sigmaColor, mode, numIters));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_guide = NULL;
    UMat guide;
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    double sigmaSpatial=0;
    double sigmaColor=0;
    int mode=DTF_NC;
    int numIters=3;

    const char* keywords[] = { "guide", "src", "sigmaSpatial", "sigmaColor", "dst", "mode", "numIters", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOdd|Oii:dtFilter", (char**)keywords, &pyobj_guide, &pyobj_src, &sigmaSpatial, &sigmaColor, &pyobj_dst, &mode, &numIters) &&
        pyopencv_to(pyobj_guide, guide, ArgInfo("guide", 0)) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::ximgproc::dtFilter(guide, src, dst, sigmaSpatial, sigmaColor, mode, numIters));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_fastGlobalSmootherFilter(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    {
    PyObject* pyobj_guide = NULL;
    Mat guide;
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    double lambda=0;
    double sigma_color=0;
    double lambda_attenuation=0.25;
    int num_iter=3;

    const char* keywords[] = { "guide", "src", "lambda", "sigma_color", "dst", "lambda_attenuation", "num_iter", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOdd|Odi:fastGlobalSmootherFilter", (char**)keywords, &pyobj_guide, &pyobj_src, &lambda, &sigma_color, &pyobj_dst, &lambda_attenuation, &num_iter) &&
        pyopencv_to(pyobj_guide, guide, ArgInfo("guide", 0)) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::ximgproc::fastGlobalSmootherFilter(guide, src, dst, lambda, sigma_color, lambda_attenuation, num_iter));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_guide = NULL;
    UMat guide;
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    double lambda=0;
    double sigma_color=0;
    double lambda_attenuation=0.25;
    int num_iter=3;

    const char* keywords[] = { "guide", "src", "lambda", "sigma_color", "dst", "lambda_attenuation", "num_iter", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOdd|Odi:fastGlobalSmootherFilter", (char**)keywords, &pyobj_guide, &pyobj_src, &lambda, &sigma_color, &pyobj_dst, &lambda_attenuation, &num_iter) &&
        pyopencv_to(pyobj_guide, guide, ArgInfo("guide", 0)) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::ximgproc::fastGlobalSmootherFilter(guide, src, dst, lambda, sigma_color, lambda_attenuation, num_iter));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_guidedFilter(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    {
    PyObject* pyobj_guide = NULL;
    Mat guide;
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int radius=0;
    double eps=0;
    int dDepth=-1;

    const char* keywords[] = { "guide", "src", "radius", "eps", "dst", "dDepth", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOid|Oi:guidedFilter", (char**)keywords, &pyobj_guide, &pyobj_src, &radius, &eps, &pyobj_dst, &dDepth) &&
        pyopencv_to(pyobj_guide, guide, ArgInfo("guide", 0)) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::ximgproc::guidedFilter(guide, src, dst, radius, eps, dDepth));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_guide = NULL;
    UMat guide;
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int radius=0;
    double eps=0;
    int dDepth=-1;

    const char* keywords[] = { "guide", "src", "radius", "eps", "dst", "dDepth", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOid|Oi:guidedFilter", (char**)keywords, &pyobj_guide, &pyobj_src, &radius, &eps, &pyobj_dst, &dDepth) &&
        pyopencv_to(pyobj_guide, guide, ArgInfo("guide", 0)) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::ximgproc::guidedFilter(guide, src, dst, radius, eps, dDepth));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_jointBilateralFilter(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    {
    PyObject* pyobj_joint = NULL;
    Mat joint;
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int d=0;
    double sigmaColor=0;
    double sigmaSpace=0;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "joint", "src", "d", "sigmaColor", "sigmaSpace", "dst", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOidd|Oi:jointBilateralFilter", (char**)keywords, &pyobj_joint, &pyobj_src, &d, &sigmaColor, &sigmaSpace, &pyobj_dst, &borderType) &&
        pyopencv_to(pyobj_joint, joint, ArgInfo("joint", 0)) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::ximgproc::jointBilateralFilter(joint, src, dst, d, sigmaColor, sigmaSpace, borderType));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_joint = NULL;
    UMat joint;
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int d=0;
    double sigmaColor=0;
    double sigmaSpace=0;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "joint", "src", "d", "sigmaColor", "sigmaSpace", "dst", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOidd|Oi:jointBilateralFilter", (char**)keywords, &pyobj_joint, &pyobj_src, &d, &sigmaColor, &sigmaSpace, &pyobj_dst, &borderType) &&
        pyopencv_to(pyobj_joint, joint, ArgInfo("joint", 0)) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::ximgproc::jointBilateralFilter(joint, src, dst, d, sigmaColor, sigmaSpace, borderType));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_l0Smooth(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    double lambda=0.02;
    double kappa=2.0;

    const char* keywords[] = { "src", "dst", "lambda", "kappa", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Odd:l0Smooth", (char**)keywords, &pyobj_src, &pyobj_dst, &lambda, &kappa) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::ximgproc::l0Smooth(src, dst, lambda, kappa));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    double lambda=0.02;
    double kappa=2.0;

    const char* keywords[] = { "src", "dst", "lambda", "kappa", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Odd:l0Smooth", (char**)keywords, &pyobj_src, &pyobj_dst, &lambda, &kappa) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::ximgproc::l0Smooth(src, dst, lambda, kappa));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_niBlackThreshold(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    {
    PyObject* pyobj__src = NULL;
    Mat _src;
    PyObject* pyobj__dst = NULL;
    Mat _dst;
    double maxValue=0;
    int type=0;
    int blockSize=0;
    double delta=0;

    const char* keywords[] = { "_src", "maxValue", "type", "blockSize", "delta", "_dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odiid|O:niBlackThreshold", (char**)keywords, &pyobj__src, &maxValue, &type, &blockSize, &delta, &pyobj__dst) &&
        pyopencv_to(pyobj__src, _src, ArgInfo("_src", 0)) &&
        pyopencv_to(pyobj__dst, _dst, ArgInfo("_dst", 1)) )
    {
        ERRWRAP2(cv::ximgproc::niBlackThreshold(_src, _dst, maxValue, type, blockSize, delta));
        return pyopencv_from(_dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj__src = NULL;
    UMat _src;
    PyObject* pyobj__dst = NULL;
    UMat _dst;
    double maxValue=0;
    int type=0;
    int blockSize=0;
    double delta=0;

    const char* keywords[] = { "_src", "maxValue", "type", "blockSize", "delta", "_dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odiid|O:niBlackThreshold", (char**)keywords, &pyobj__src, &maxValue, &type, &blockSize, &delta, &pyobj__dst) &&
        pyopencv_to(pyobj__src, _src, ArgInfo("_src", 0)) &&
        pyopencv_to(pyobj__dst, _dst, ArgInfo("_dst", 1)) )
    {
        ERRWRAP2(cv::ximgproc::niBlackThreshold(_src, _dst, maxValue, type, blockSize, delta));
        return pyopencv_from(_dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_rollingGuidanceFilter(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int d=-1;
    double sigmaColor=25;
    double sigmaSpace=3;
    int numOfIter=4;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "dst", "d", "sigmaColor", "sigmaSpace", "numOfIter", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oiddii:rollingGuidanceFilter", (char**)keywords, &pyobj_src, &pyobj_dst, &d, &sigmaColor, &sigmaSpace, &numOfIter, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::ximgproc::rollingGuidanceFilter(src, dst, d, sigmaColor, sigmaSpace, numOfIter, borderType));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int d=-1;
    double sigmaColor=25;
    double sigmaSpace=3;
    int numOfIter=4;
    int borderType=BORDER_DEFAULT;

    const char* keywords[] = { "src", "dst", "d", "sigmaColor", "sigmaSpace", "numOfIter", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oiddii:rollingGuidanceFilter", (char**)keywords, &pyobj_src, &pyobj_dst, &d, &sigmaColor, &sigmaSpace, &numOfIter, &borderType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::ximgproc::rollingGuidanceFilter(src, dst, d, sigmaColor, sigmaSpace, numOfIter, borderType));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_thinning(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int thinningType=THINNING_ZHANGSUEN;

    const char* keywords[] = { "src", "dst", "thinningType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oi:thinning", (char**)keywords, &pyobj_src, &pyobj_dst, &thinningType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::ximgproc::thinning(src, dst, thinningType));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    int thinningType=THINNING_ZHANGSUEN;

    const char* keywords[] = { "src", "dst", "thinningType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oi:thinning", (char**)keywords, &pyobj_src, &pyobj_dst, &thinningType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::ximgproc::thinning(src, dst, thinningType));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_segmentation_createGraphSegmentation(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc::segmentation;

    double sigma=0.5;
    float k=300;
    int min_size=100;
    Ptr<GraphSegmentation> retval;

    const char* keywords[] = { "sigma", "k", "min_size", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|dfi:createGraphSegmentation", (char**)keywords, &sigma, &k, &min_size) )
    {
        ERRWRAP2(retval = cv::ximgproc::segmentation::createGraphSegmentation(sigma, k, min_size));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_segmentation_createSelectiveSearchSegmentation(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc::segmentation;

    Ptr<SelectiveSearchSegmentation> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::ximgproc::segmentation::createSelectiveSearchSegmentation());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_segmentation_createSelectiveSearchSegmentationStrategyColor(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc::segmentation;

    Ptr<SelectiveSearchSegmentationStrategyColor> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::ximgproc::segmentation::createSelectiveSearchSegmentationStrategyColor());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_segmentation_createSelectiveSearchSegmentationStrategyFill(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc::segmentation;

    Ptr<SelectiveSearchSegmentationStrategyFill> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::ximgproc::segmentation::createSelectiveSearchSegmentationStrategyFill());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_segmentation_createSelectiveSearchSegmentationStrategyMultiple(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc::segmentation;

    {
    Ptr<SelectiveSearchSegmentationStrategyMultiple> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::ximgproc::segmentation::createSelectiveSearchSegmentationStrategyMultiple());
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_s1 = NULL;
    Ptr<SelectiveSearchSegmentationStrategy> s1;
    Ptr<SelectiveSearchSegmentationStrategyMultiple> retval;

    const char* keywords[] = { "s1", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:createSelectiveSearchSegmentationStrategyMultiple", (char**)keywords, &pyobj_s1) &&
        pyopencv_to(pyobj_s1, s1, ArgInfo("s1", 0)) )
    {
        ERRWRAP2(retval = cv::ximgproc::segmentation::createSelectiveSearchSegmentationStrategyMultiple(s1));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_s1 = NULL;
    Ptr<SelectiveSearchSegmentationStrategy> s1;
    PyObject* pyobj_s2 = NULL;
    Ptr<SelectiveSearchSegmentationStrategy> s2;
    Ptr<SelectiveSearchSegmentationStrategyMultiple> retval;

    const char* keywords[] = { "s1", "s2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:createSelectiveSearchSegmentationStrategyMultiple", (char**)keywords, &pyobj_s1, &pyobj_s2) &&
        pyopencv_to(pyobj_s1, s1, ArgInfo("s1", 0)) &&
        pyopencv_to(pyobj_s2, s2, ArgInfo("s2", 0)) )
    {
        ERRWRAP2(retval = cv::ximgproc::segmentation::createSelectiveSearchSegmentationStrategyMultiple(s1, s2));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_s1 = NULL;
    Ptr<SelectiveSearchSegmentationStrategy> s1;
    PyObject* pyobj_s2 = NULL;
    Ptr<SelectiveSearchSegmentationStrategy> s2;
    PyObject* pyobj_s3 = NULL;
    Ptr<SelectiveSearchSegmentationStrategy> s3;
    Ptr<SelectiveSearchSegmentationStrategyMultiple> retval;

    const char* keywords[] = { "s1", "s2", "s3", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:createSelectiveSearchSegmentationStrategyMultiple", (char**)keywords, &pyobj_s1, &pyobj_s2, &pyobj_s3) &&
        pyopencv_to(pyobj_s1, s1, ArgInfo("s1", 0)) &&
        pyopencv_to(pyobj_s2, s2, ArgInfo("s2", 0)) &&
        pyopencv_to(pyobj_s3, s3, ArgInfo("s3", 0)) )
    {
        ERRWRAP2(retval = cv::ximgproc::segmentation::createSelectiveSearchSegmentationStrategyMultiple(s1, s2, s3));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_s1 = NULL;
    Ptr<SelectiveSearchSegmentationStrategy> s1;
    PyObject* pyobj_s2 = NULL;
    Ptr<SelectiveSearchSegmentationStrategy> s2;
    PyObject* pyobj_s3 = NULL;
    Ptr<SelectiveSearchSegmentationStrategy> s3;
    PyObject* pyobj_s4 = NULL;
    Ptr<SelectiveSearchSegmentationStrategy> s4;
    Ptr<SelectiveSearchSegmentationStrategyMultiple> retval;

    const char* keywords[] = { "s1", "s2", "s3", "s4", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO:createSelectiveSearchSegmentationStrategyMultiple", (char**)keywords, &pyobj_s1, &pyobj_s2, &pyobj_s3, &pyobj_s4) &&
        pyopencv_to(pyobj_s1, s1, ArgInfo("s1", 0)) &&
        pyopencv_to(pyobj_s2, s2, ArgInfo("s2", 0)) &&
        pyopencv_to(pyobj_s3, s3, ArgInfo("s3", 0)) &&
        pyopencv_to(pyobj_s4, s4, ArgInfo("s4", 0)) )
    {
        ERRWRAP2(retval = cv::ximgproc::segmentation::createSelectiveSearchSegmentationStrategyMultiple(s1, s2, s3, s4));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_segmentation_createSelectiveSearchSegmentationStrategySize(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc::segmentation;

    Ptr<SelectiveSearchSegmentationStrategySize> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::ximgproc::segmentation::createSelectiveSearchSegmentationStrategySize());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ximgproc_segmentation_createSelectiveSearchSegmentationStrategyTexture(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::ximgproc::segmentation;

    Ptr<SelectiveSearchSegmentationStrategyTexture> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::ximgproc::segmentation::createSelectiveSearchSegmentationStrategyTexture());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_xphoto_applyChannelGains(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::xphoto;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    float gainB=0.f;
    float gainG=0.f;
    float gainR=0.f;

    const char* keywords[] = { "src", "gainB", "gainG", "gainR", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Offf|O:applyChannelGains", (char**)keywords, &pyobj_src, &gainB, &gainG, &gainR, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::xphoto::applyChannelGains(src, dst, gainB, gainG, gainR));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    float gainB=0.f;
    float gainG=0.f;
    float gainR=0.f;

    const char* keywords[] = { "src", "gainB", "gainG", "gainR", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Offf|O:applyChannelGains", (char**)keywords, &pyobj_src, &gainB, &gainG, &gainR, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::xphoto::applyChannelGains(src, dst, gainB, gainG, gainR));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_xphoto_bm3dDenoising(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::xphoto;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dstStep1 = NULL;
    Mat dstStep1;
    PyObject* pyobj_dstStep2 = NULL;
    Mat dstStep2;
    float h=1;
    int templateWindowSize=4;
    int searchWindowSize=16;
    int blockMatchingStep1=2500;
    int blockMatchingStep2=400;
    int groupSize=8;
    int slidingStep=1;
    float beta=2.0f;
    int normType=cv::NORM_L2;
    int step=cv::xphoto::BM3D_STEPALL;
    int transformType=cv::xphoto::HAAR;

    const char* keywords[] = { "src", "dstStep1", "dstStep2", "h", "templateWindowSize", "searchWindowSize", "blockMatchingStep1", "blockMatchingStep2", "groupSize", "slidingStep", "beta", "normType", "step", "transformType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Ofiiiiiifiii:bm3dDenoising", (char**)keywords, &pyobj_src, &pyobj_dstStep1, &pyobj_dstStep2, &h, &templateWindowSize, &searchWindowSize, &blockMatchingStep1, &blockMatchingStep2, &groupSize, &slidingStep, &beta, &normType, &step, &transformType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dstStep1, dstStep1, ArgInfo("dstStep1", 1)) &&
        pyopencv_to(pyobj_dstStep2, dstStep2, ArgInfo("dstStep2", 1)) )
    {
        ERRWRAP2(cv::xphoto::bm3dDenoising(src, dstStep1, dstStep2, h, templateWindowSize, searchWindowSize, blockMatchingStep1, blockMatchingStep2, groupSize, slidingStep, beta, normType, step, transformType));
        return Py_BuildValue("(NN)", pyopencv_from(dstStep1), pyopencv_from(dstStep2));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dstStep1 = NULL;
    UMat dstStep1;
    PyObject* pyobj_dstStep2 = NULL;
    UMat dstStep2;
    float h=1;
    int templateWindowSize=4;
    int searchWindowSize=16;
    int blockMatchingStep1=2500;
    int blockMatchingStep2=400;
    int groupSize=8;
    int slidingStep=1;
    float beta=2.0f;
    int normType=cv::NORM_L2;
    int step=cv::xphoto::BM3D_STEPALL;
    int transformType=cv::xphoto::HAAR;

    const char* keywords[] = { "src", "dstStep1", "dstStep2", "h", "templateWindowSize", "searchWindowSize", "blockMatchingStep1", "blockMatchingStep2", "groupSize", "slidingStep", "beta", "normType", "step", "transformType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Ofiiiiiifiii:bm3dDenoising", (char**)keywords, &pyobj_src, &pyobj_dstStep1, &pyobj_dstStep2, &h, &templateWindowSize, &searchWindowSize, &blockMatchingStep1, &blockMatchingStep2, &groupSize, &slidingStep, &beta, &normType, &step, &transformType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dstStep1, dstStep1, ArgInfo("dstStep1", 1)) &&
        pyopencv_to(pyobj_dstStep2, dstStep2, ArgInfo("dstStep2", 1)) )
    {
        ERRWRAP2(cv::xphoto::bm3dDenoising(src, dstStep1, dstStep2, h, templateWindowSize, searchWindowSize, blockMatchingStep1, blockMatchingStep2, groupSize, slidingStep, beta, normType, step, transformType));
        return Py_BuildValue("(NN)", pyopencv_from(dstStep1), pyopencv_from(dstStep2));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    float h=1;
    int templateWindowSize=4;
    int searchWindowSize=16;
    int blockMatchingStep1=2500;
    int blockMatchingStep2=400;
    int groupSize=8;
    int slidingStep=1;
    float beta=2.0f;
    int normType=cv::NORM_L2;
    int step=cv::xphoto::BM3D_STEPALL;
    int transformType=cv::xphoto::HAAR;

    const char* keywords[] = { "src", "dst", "h", "templateWindowSize", "searchWindowSize", "blockMatchingStep1", "blockMatchingStep2", "groupSize", "slidingStep", "beta", "normType", "step", "transformType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Ofiiiiiifiii:bm3dDenoising", (char**)keywords, &pyobj_src, &pyobj_dst, &h, &templateWindowSize, &searchWindowSize, &blockMatchingStep1, &blockMatchingStep2, &groupSize, &slidingStep, &beta, &normType, &step, &transformType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::xphoto::bm3dDenoising(src, dst, h, templateWindowSize, searchWindowSize, blockMatchingStep1, blockMatchingStep2, groupSize, slidingStep, beta, normType, step, transformType));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    float h=1;
    int templateWindowSize=4;
    int searchWindowSize=16;
    int blockMatchingStep1=2500;
    int blockMatchingStep2=400;
    int groupSize=8;
    int slidingStep=1;
    float beta=2.0f;
    int normType=cv::NORM_L2;
    int step=cv::xphoto::BM3D_STEPALL;
    int transformType=cv::xphoto::HAAR;

    const char* keywords[] = { "src", "dst", "h", "templateWindowSize", "searchWindowSize", "blockMatchingStep1", "blockMatchingStep2", "groupSize", "slidingStep", "beta", "normType", "step", "transformType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Ofiiiiiifiii:bm3dDenoising", (char**)keywords, &pyobj_src, &pyobj_dst, &h, &templateWindowSize, &searchWindowSize, &blockMatchingStep1, &blockMatchingStep2, &groupSize, &slidingStep, &beta, &normType, &step, &transformType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(cv::xphoto::bm3dDenoising(src, dst, h, templateWindowSize, searchWindowSize, blockMatchingStep1, blockMatchingStep2, groupSize, slidingStep, beta, normType, step, transformType));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_xphoto_createGrayworldWB(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::xphoto;

    Ptr<GrayworldWB> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::xphoto::createGrayworldWB());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_xphoto_createLearningBasedWB(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::xphoto;

    PyObject* pyobj_path_to_model = NULL;
    String path_to_model;
    Ptr<LearningBasedWB> retval;

    const char* keywords[] = { "path_to_model", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|O:createLearningBasedWB", (char**)keywords, &pyobj_path_to_model) &&
        pyopencv_to(pyobj_path_to_model, path_to_model, ArgInfo("path_to_model", 0)) )
    {
        ERRWRAP2(retval = cv::xphoto::createLearningBasedWB(path_to_model));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_xphoto_createSimpleWB(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::xphoto;

    Ptr<SimpleWB> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = cv::xphoto::createSimpleWB());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_xphoto_dctDenoising(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::xphoto;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    double sigma=0;
    int psize=16;

    const char* keywords[] = { "src", "dst", "sigma", "psize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOd|i:dctDenoising", (char**)keywords, &pyobj_src, &pyobj_dst, &sigma, &psize) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 0)) )
    {
        ERRWRAP2(cv::xphoto::dctDenoising(src, dst, sigma, psize));
        Py_RETURN_NONE;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    double sigma=0;
    int psize=16;

    const char* keywords[] = { "src", "dst", "sigma", "psize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOd|i:dctDenoising", (char**)keywords, &pyobj_src, &pyobj_dst, &sigma, &psize) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 0)) )
    {
        ERRWRAP2(cv::xphoto::dctDenoising(src, dst, sigma, psize));
        Py_RETURN_NONE;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_xphoto_inpaint(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace cv::xphoto;

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int algorithmType=0;

    const char* keywords[] = { "src", "mask", "dst", "algorithmType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOi:inpaint", (char**)keywords, &pyobj_src, &pyobj_mask, &pyobj_dst, &algorithmType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 0)) )
    {
        ERRWRAP2(cv::xphoto::inpaint(src, mask, dst, algorithmType));
        Py_RETURN_NONE;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int algorithmType=0;

    const char* keywords[] = { "src", "mask", "dst", "algorithmType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOi:inpaint", (char**)keywords, &pyobj_src, &pyobj_mask, &pyobj_dst, &algorithmType) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 0)) )
    {
        ERRWRAP2(cv::xphoto::inpaint(src, mask, dst, algorithmType));
        Py_RETURN_NONE;
    }
    }

    return NULL;
}

