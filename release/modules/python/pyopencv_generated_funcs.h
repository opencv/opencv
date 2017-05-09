static PyObject* pyopencv_Algorithm__create(PyObject* , PyObject* args, PyObject* kw)
{
    Ptr_Algorithm retval;
    PyObject* pyobj_name = NULL;
    string name;

    const char* keywords[] = { "name", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:Algorithm__create", (char**)keywords, &pyobj_name) &&
        pyopencv_to(pyobj_name, name, ArgInfo("name", 0)) )
    {
        ERRWRAP2( retval = Algorithm::_create(name));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_Algorithm_getList(PyObject* , PyObject* args, PyObject* kw)
{
    vector_string algorithms;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2( Algorithm::getList(algorithms));
        return pyopencv_from(algorithms);
    }

    return NULL;
}

static PyObject* pyopencv_BFMatcher_BFMatcher(PyObject* , PyObject* args, PyObject* kw)
{
    pyopencv_BFMatcher_t* self = 0;
    int normType=0;
    bool crossCheck=false;

    const char* keywords[] = { "normType", "crossCheck", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i|b:BFMatcher", (char**)keywords, &normType, &crossCheck) )
    {
        self = PyObject_NEW(pyopencv_BFMatcher_t, &pyopencv_BFMatcher_Type);
        new (&(self->v)) Ptr<cv::BFMatcher>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::BFMatcher(normType, crossCheck));
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_BRISK_BRISK(PyObject* , PyObject* args, PyObject* kw)
{
    {
    pyopencv_BRISK_t* self = 0;
    int thresh=30;
    int octaves=3;
    float patternScale=1.0f;

    const char* keywords[] = { "thresh", "octaves", "patternScale", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|iif:BRISK", (char**)keywords, &thresh, &octaves, &patternScale) )
    {
        self = PyObject_NEW(pyopencv_BRISK_t, &pyopencv_BRISK_Type);
        new (&(self->v)) Ptr<cv::BRISK>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::BRISK(thresh, octaves, patternScale));
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_BRISK_t* self = 0;
    PyObject* pyobj_radiusList = NULL;
    vector_float radiusList;
    PyObject* pyobj_numberList = NULL;
    vector_int numberList;
    float dMax=5.85f;
    float dMin=8.2f;
    PyObject* pyobj_indexChange = NULL;
    vector_int indexChange=std::vector<int>();

    const char* keywords[] = { "radiusList", "numberList", "dMax", "dMin", "indexChange", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|ffO:BRISK", (char**)keywords, &pyobj_radiusList, &pyobj_numberList, &dMax, &dMin, &pyobj_indexChange) &&
        pyopencv_to(pyobj_radiusList, radiusList, ArgInfo("radiusList", 0)) &&
        pyopencv_to(pyobj_numberList, numberList, ArgInfo("numberList", 0)) &&
        pyopencv_to(pyobj_indexChange, indexChange, ArgInfo("indexChange", 0)) )
    {
        self = PyObject_NEW(pyopencv_BRISK_t, &pyopencv_BRISK_Type);
        new (&(self->v)) Ptr<cv::BRISK>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::BRISK(radiusList, numberList, dMax, dMin, indexChange));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_BackgroundSubtractorMOG_BackgroundSubtractorMOG(PyObject* , PyObject* args, PyObject* kw)
{
    {
    pyopencv_BackgroundSubtractorMOG_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_BackgroundSubtractorMOG_t, &pyopencv_BackgroundSubtractorMOG_Type);
        new (&(self->v)) Ptr<cv::BackgroundSubtractorMOG>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::BackgroundSubtractorMOG());
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_BackgroundSubtractorMOG_t* self = 0;
    int history=0;
    int nmixtures=0;
    double backgroundRatio=0;
    double noiseSigma=0;

    const char* keywords[] = { "history", "nmixtures", "backgroundRatio", "noiseSigma", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "iid|d:BackgroundSubtractorMOG", (char**)keywords, &history, &nmixtures, &backgroundRatio, &noiseSigma) )
    {
        self = PyObject_NEW(pyopencv_BackgroundSubtractorMOG_t, &pyopencv_BackgroundSubtractorMOG_Type);
        new (&(self->v)) Ptr<cv::BackgroundSubtractorMOG>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::BackgroundSubtractorMOG(history, nmixtures, backgroundRatio, noiseSigma));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_CamShift(PyObject* , PyObject* args, PyObject* kw)
{
    RotatedRect retval;
    PyObject* pyobj_probImage = NULL;
    Mat probImage;
    PyObject* pyobj_window = NULL;
    Rect window;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria;

    const char* keywords[] = { "probImage", "window", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:CamShift", (char**)keywords, &pyobj_probImage, &pyobj_window, &pyobj_criteria) &&
        pyopencv_to(pyobj_probImage, probImage, ArgInfo("probImage", 0)) &&
        pyopencv_to(pyobj_window, window, ArgInfo("window", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2( retval = cv::CamShift(probImage, window, criteria));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(window));
    }

    return NULL;
}

static PyObject* pyopencv_Canny(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::Canny(image, edges, threshold1, threshold2, apertureSize, L2gradient));
        return pyopencv_from(edges);
    }

    return NULL;
}

static PyObject* pyopencv_CascadeClassifier_CascadeClassifier(PyObject* , PyObject* args, PyObject* kw)
{
    {
    pyopencv_CascadeClassifier_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_CascadeClassifier_t, &pyopencv_CascadeClassifier_Type);
        new (&(self->v)) Ptr<cv::CascadeClassifier>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::CascadeClassifier());
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_CascadeClassifier_t* self = 0;
    PyObject* pyobj_filename = NULL;
    string filename;

    const char* keywords[] = { "filename", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:CascadeClassifier", (char**)keywords, &pyobj_filename) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) )
    {
        self = PyObject_NEW(pyopencv_CascadeClassifier_t, &pyopencv_CascadeClassifier_Type);
        new (&(self->v)) Ptr<cv::CascadeClassifier>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::CascadeClassifier(filename));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_CvANN_MLP_CvANN_MLP(PyObject* , PyObject* args, PyObject* kw)
{
    {
    pyopencv_CvANN_MLP_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_CvANN_MLP_t, &pyopencv_CvANN_MLP_Type);
        new (&(self->v)) Ptr<CvANN_MLP>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new CvANN_MLP());
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_CvANN_MLP_t* self = 0;
    PyObject* pyobj_layerSizes = NULL;
    Mat layerSizes;
    int activateFunc=CvANN_MLP::SIGMOID_SYM;
    double fparam1=0;
    double fparam2=0;

    const char* keywords[] = { "layerSizes", "activateFunc", "fparam1", "fparam2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|idd:CvANN_MLP", (char**)keywords, &pyobj_layerSizes, &activateFunc, &fparam1, &fparam2) &&
        pyopencv_to(pyobj_layerSizes, layerSizes, ArgInfo("layerSizes", 0)) )
    {
        self = PyObject_NEW(pyopencv_CvANN_MLP_t, &pyopencv_CvANN_MLP_Type);
        new (&(self->v)) Ptr<CvANN_MLP>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new CvANN_MLP(layerSizes, activateFunc, fparam1, fparam2));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_CvBoost_CvBoost(PyObject* , PyObject* args, PyObject* kw)
{
    {
    pyopencv_CvBoost_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_CvBoost_t, &pyopencv_CvBoost_Type);
        new (&(self->v)) Ptr<CvBoost>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new CvBoost());
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_CvBoost_t* self = 0;
    PyObject* pyobj_trainData = NULL;
    Mat trainData;
    int tflag=0;
    PyObject* pyobj_responses = NULL;
    Mat responses;
    PyObject* pyobj_varIdx = NULL;
    Mat varIdx=cv::Mat();
    PyObject* pyobj_sampleIdx = NULL;
    Mat sampleIdx=cv::Mat();
    PyObject* pyobj_varType = NULL;
    Mat varType=cv::Mat();
    PyObject* pyobj_missingDataMask = NULL;
    Mat missingDataMask=cv::Mat();
    PyObject* pyobj_params = NULL;
    CvBoostParams params;

    const char* keywords[] = { "trainData", "tflag", "responses", "varIdx", "sampleIdx", "varType", "missingDataMask", "params", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OiO|OOOOO:CvBoost", (char**)keywords, &pyobj_trainData, &tflag, &pyobj_responses, &pyobj_varIdx, &pyobj_sampleIdx, &pyobj_varType, &pyobj_missingDataMask, &pyobj_params) &&
        pyopencv_to(pyobj_trainData, trainData, ArgInfo("trainData", 0)) &&
        pyopencv_to(pyobj_responses, responses, ArgInfo("responses", 0)) &&
        pyopencv_to(pyobj_varIdx, varIdx, ArgInfo("varIdx", 0)) &&
        pyopencv_to(pyobj_sampleIdx, sampleIdx, ArgInfo("sampleIdx", 0)) &&
        pyopencv_to(pyobj_varType, varType, ArgInfo("varType", 0)) &&
        pyopencv_to(pyobj_missingDataMask, missingDataMask, ArgInfo("missingDataMask", 0)) &&
        pyopencv_to(pyobj_params, params, ArgInfo("params", 0)) )
    {
        self = PyObject_NEW(pyopencv_CvBoost_t, &pyopencv_CvBoost_Type);
        new (&(self->v)) Ptr<CvBoost>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new CvBoost(trainData, tflag, responses, varIdx, sampleIdx, varType, missingDataMask, params));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_CvDTree_CvDTree(PyObject* , PyObject* args, PyObject* kw)
{
    pyopencv_CvDTree_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_CvDTree_t, &pyopencv_CvDTree_Type);
        new (&(self->v)) Ptr<CvDTree>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new CvDTree());
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_CvERTrees_CvERTrees(PyObject* , PyObject* args, PyObject* kw)
{
    pyopencv_CvERTrees_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_CvERTrees_t, &pyopencv_CvERTrees_Type);
        new (&(self->v)) Ptr<CvERTrees>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new CvERTrees());
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_CvGBTrees_CvGBTrees(PyObject* , PyObject* args, PyObject* kw)
{
    {
    pyopencv_CvGBTrees_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_CvGBTrees_t, &pyopencv_CvGBTrees_Type);
        new (&(self->v)) Ptr<CvGBTrees>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new CvGBTrees());
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_CvGBTrees_t* self = 0;
    PyObject* pyobj_trainData = NULL;
    Mat trainData;
    int tflag=0;
    PyObject* pyobj_responses = NULL;
    Mat responses;
    PyObject* pyobj_varIdx = NULL;
    Mat varIdx=cv::Mat();
    PyObject* pyobj_sampleIdx = NULL;
    Mat sampleIdx=cv::Mat();
    PyObject* pyobj_varType = NULL;
    Mat varType=cv::Mat();
    PyObject* pyobj_missingDataMask = NULL;
    Mat missingDataMask=cv::Mat();
    PyObject* pyobj_params = NULL;
    CvGBTreesParams params;

    const char* keywords[] = { "trainData", "tflag", "responses", "varIdx", "sampleIdx", "varType", "missingDataMask", "params", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OiO|OOOOO:CvGBTrees", (char**)keywords, &pyobj_trainData, &tflag, &pyobj_responses, &pyobj_varIdx, &pyobj_sampleIdx, &pyobj_varType, &pyobj_missingDataMask, &pyobj_params) &&
        pyopencv_to(pyobj_trainData, trainData, ArgInfo("trainData", 0)) &&
        pyopencv_to(pyobj_responses, responses, ArgInfo("responses", 0)) &&
        pyopencv_to(pyobj_varIdx, varIdx, ArgInfo("varIdx", 0)) &&
        pyopencv_to(pyobj_sampleIdx, sampleIdx, ArgInfo("sampleIdx", 0)) &&
        pyopencv_to(pyobj_varType, varType, ArgInfo("varType", 0)) &&
        pyopencv_to(pyobj_missingDataMask, missingDataMask, ArgInfo("missingDataMask", 0)) &&
        pyopencv_to(pyobj_params, params, ArgInfo("params", 0)) )
    {
        self = PyObject_NEW(pyopencv_CvGBTrees_t, &pyopencv_CvGBTrees_Type);
        new (&(self->v)) Ptr<CvGBTrees>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new CvGBTrees(trainData, tflag, responses, varIdx, sampleIdx, varType, missingDataMask, params));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_CvKNearest_CvKNearest(PyObject* , PyObject* args, PyObject* kw)
{
    {
    pyopencv_CvKNearest_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_CvKNearest_t, &pyopencv_CvKNearest_Type);
        new (&(self->v)) Ptr<CvKNearest>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new CvKNearest());
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_CvKNearest_t* self = 0;
    PyObject* pyobj_trainData = NULL;
    Mat trainData;
    PyObject* pyobj_responses = NULL;
    Mat responses;
    PyObject* pyobj_sampleIdx = NULL;
    Mat sampleIdx=cv::Mat();
    bool isRegression=false;
    int max_k=32;

    const char* keywords[] = { "trainData", "responses", "sampleIdx", "isRegression", "max_k", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Obi:CvKNearest", (char**)keywords, &pyobj_trainData, &pyobj_responses, &pyobj_sampleIdx, &isRegression, &max_k) &&
        pyopencv_to(pyobj_trainData, trainData, ArgInfo("trainData", 0)) &&
        pyopencv_to(pyobj_responses, responses, ArgInfo("responses", 0)) &&
        pyopencv_to(pyobj_sampleIdx, sampleIdx, ArgInfo("sampleIdx", 0)) )
    {
        self = PyObject_NEW(pyopencv_CvKNearest_t, &pyopencv_CvKNearest_Type);
        new (&(self->v)) Ptr<CvKNearest>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new CvKNearest(trainData, responses, sampleIdx, isRegression, max_k));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_CvNormalBayesClassifier_CvNormalBayesClassifier(PyObject* , PyObject* args, PyObject* kw)
{
    {
    pyopencv_CvNormalBayesClassifier_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_CvNormalBayesClassifier_t, &pyopencv_CvNormalBayesClassifier_Type);
        new (&(self->v)) Ptr<CvNormalBayesClassifier>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new CvNormalBayesClassifier());
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_CvNormalBayesClassifier_t* self = 0;
    PyObject* pyobj_trainData = NULL;
    Mat trainData;
    PyObject* pyobj_responses = NULL;
    Mat responses;
    PyObject* pyobj_varIdx = NULL;
    Mat varIdx=cv::Mat();
    PyObject* pyobj_sampleIdx = NULL;
    Mat sampleIdx=cv::Mat();

    const char* keywords[] = { "trainData", "responses", "varIdx", "sampleIdx", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OO:CvNormalBayesClassifier", (char**)keywords, &pyobj_trainData, &pyobj_responses, &pyobj_varIdx, &pyobj_sampleIdx) &&
        pyopencv_to(pyobj_trainData, trainData, ArgInfo("trainData", 0)) &&
        pyopencv_to(pyobj_responses, responses, ArgInfo("responses", 0)) &&
        pyopencv_to(pyobj_varIdx, varIdx, ArgInfo("varIdx", 0)) &&
        pyopencv_to(pyobj_sampleIdx, sampleIdx, ArgInfo("sampleIdx", 0)) )
    {
        self = PyObject_NEW(pyopencv_CvNormalBayesClassifier_t, &pyopencv_CvNormalBayesClassifier_Type);
        new (&(self->v)) Ptr<CvNormalBayesClassifier>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new CvNormalBayesClassifier(trainData, responses, varIdx, sampleIdx));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_CvRTrees_CvRTrees(PyObject* , PyObject* args, PyObject* kw)
{
    pyopencv_CvRTrees_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_CvRTrees_t, &pyopencv_CvRTrees_Type);
        new (&(self->v)) Ptr<CvRTrees>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new CvRTrees());
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_CvSVM_CvSVM(PyObject* , PyObject* args, PyObject* kw)
{
    {
    pyopencv_CvSVM_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_CvSVM_t, &pyopencv_CvSVM_Type);
        new (&(self->v)) Ptr<CvSVM>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new CvSVM());
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_CvSVM_t* self = 0;
    PyObject* pyobj_trainData = NULL;
    Mat trainData;
    PyObject* pyobj_responses = NULL;
    Mat responses;
    PyObject* pyobj_varIdx = NULL;
    Mat varIdx=cv::Mat();
    PyObject* pyobj_sampleIdx = NULL;
    Mat sampleIdx=cv::Mat();
    PyObject* pyobj_params = NULL;
    CvSVMParams params;

    const char* keywords[] = { "trainData", "responses", "varIdx", "sampleIdx", "params", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOO:CvSVM", (char**)keywords, &pyobj_trainData, &pyobj_responses, &pyobj_varIdx, &pyobj_sampleIdx, &pyobj_params) &&
        pyopencv_to(pyobj_trainData, trainData, ArgInfo("trainData", 0)) &&
        pyopencv_to(pyobj_responses, responses, ArgInfo("responses", 0)) &&
        pyopencv_to(pyobj_varIdx, varIdx, ArgInfo("varIdx", 0)) &&
        pyopencv_to(pyobj_sampleIdx, sampleIdx, ArgInfo("sampleIdx", 0)) &&
        pyopencv_to(pyobj_params, params, ArgInfo("params", 0)) )
    {
        self = PyObject_NEW(pyopencv_CvSVM_t, &pyopencv_CvSVM_Type);
        new (&(self->v)) Ptr<CvSVM>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new CvSVM(trainData, responses, varIdx, sampleIdx, params));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_DMatch_DMatch(PyObject* , PyObject* args, PyObject* kw)
{
    {
    pyopencv_DMatch_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_DMatch_t, &pyopencv_DMatch_Type);
        if(self) ERRWRAP2(self->v = cv::DMatch());
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_DMatch_t* self = 0;
    int _queryIdx=0;
    int _trainIdx=0;
    float _distance=0.f;

    const char* keywords[] = { "_queryIdx", "_trainIdx", "_distance", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "iif:DMatch", (char**)keywords, &_queryIdx, &_trainIdx, &_distance) )
    {
        self = PyObject_NEW(pyopencv_DMatch_t, &pyopencv_DMatch_Type);
        if(self) ERRWRAP2(self->v = cv::DMatch(_queryIdx, _trainIdx, _distance));
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_DMatch_t* self = 0;
    int _queryIdx=0;
    int _trainIdx=0;
    int _imgIdx=0;
    float _distance=0.f;

    const char* keywords[] = { "_queryIdx", "_trainIdx", "_imgIdx", "_distance", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "iiif:DMatch", (char**)keywords, &_queryIdx, &_trainIdx, &_imgIdx, &_distance) )
    {
        self = PyObject_NEW(pyopencv_DMatch_t, &pyopencv_DMatch_Type);
        if(self) ERRWRAP2(self->v = cv::DMatch(_queryIdx, _trainIdx, _imgIdx, _distance));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_DescriptorExtractor_create(PyObject* , PyObject* args, PyObject* kw)
{
    Ptr_DescriptorExtractor retval;
    PyObject* pyobj_descriptorExtractorType = NULL;
    string descriptorExtractorType;

    const char* keywords[] = { "descriptorExtractorType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:DescriptorExtractor_create", (char**)keywords, &pyobj_descriptorExtractorType) &&
        pyopencv_to(pyobj_descriptorExtractorType, descriptorExtractorType, ArgInfo("descriptorExtractorType", 0)) )
    {
        ERRWRAP2( retval = DescriptorExtractor::create(descriptorExtractorType));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_DescriptorMatcher_create(PyObject* , PyObject* args, PyObject* kw)
{
    Ptr_DescriptorMatcher retval;
    PyObject* pyobj_descriptorMatcherType = NULL;
    string descriptorMatcherType;

    const char* keywords[] = { "descriptorMatcherType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:DescriptorMatcher_create", (char**)keywords, &pyobj_descriptorMatcherType) &&
        pyopencv_to(pyobj_descriptorMatcherType, descriptorMatcherType, ArgInfo("descriptorMatcherType", 0)) )
    {
        ERRWRAP2( retval = DescriptorMatcher::create(descriptorMatcherType));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_EM_EM(PyObject* , PyObject* args, PyObject* kw)
{
    pyopencv_EM_t* self = 0;
    int nclusters=EM::DEFAULT_NCLUSTERS;
    int covMatType=EM::COV_MAT_DIAGONAL;
    PyObject* pyobj_termCrit = NULL;
    TermCriteria termCrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, EM::DEFAULT_MAX_ITERS, FLT_EPSILON);

    const char* keywords[] = { "nclusters", "covMatType", "termCrit", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|iiO:EM", (char**)keywords, &nclusters, &covMatType, &pyobj_termCrit) &&
        pyopencv_to(pyobj_termCrit, termCrit, ArgInfo("termCrit", 0)) )
    {
        self = PyObject_NEW(pyopencv_EM_t, &pyopencv_EM_Type);
        new (&(self->v)) Ptr<cv::EM>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::EM(nclusters, covMatType, termCrit));
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_FastFeatureDetector_FastFeatureDetector(PyObject* , PyObject* args, PyObject* kw)
{
    {
    pyopencv_FastFeatureDetector_t* self = 0;
    int threshold=10;
    bool nonmaxSuppression=true;

    const char* keywords[] = { "threshold", "nonmaxSuppression", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|ib:FastFeatureDetector", (char**)keywords, &threshold, &nonmaxSuppression) )
    {
        self = PyObject_NEW(pyopencv_FastFeatureDetector_t, &pyopencv_FastFeatureDetector_Type);
        new (&(self->v)) Ptr<cv::FastFeatureDetector>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::FastFeatureDetector(threshold, nonmaxSuppression));
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_FastFeatureDetector_t* self = 0;
    int threshold=0;
    bool nonmaxSuppression=0;
    int type=0;

    const char* keywords[] = { "threshold", "nonmaxSuppression", "type", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "ibi:FastFeatureDetector", (char**)keywords, &threshold, &nonmaxSuppression, &type) )
    {
        self = PyObject_NEW(pyopencv_FastFeatureDetector_t, &pyopencv_FastFeatureDetector_Type);
        new (&(self->v)) Ptr<cv::FastFeatureDetector>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::FastFeatureDetector(threshold, nonmaxSuppression, type));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_Feature2D_create(PyObject* , PyObject* args, PyObject* kw)
{
    Ptr_Feature2D retval;
    PyObject* pyobj_name = NULL;
    string name;

    const char* keywords[] = { "name", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:Feature2D_create", (char**)keywords, &pyobj_name) &&
        pyopencv_to(pyobj_name, name, ArgInfo("name", 0)) )
    {
        ERRWRAP2( retval = Feature2D::create(name));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_FeatureDetector_create(PyObject* , PyObject* args, PyObject* kw)
{
    Ptr_FeatureDetector retval;
    PyObject* pyobj_detectorType = NULL;
    string detectorType;

    const char* keywords[] = { "detectorType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:FeatureDetector_create", (char**)keywords, &pyobj_detectorType) &&
        pyopencv_to(pyobj_detectorType, detectorType, ArgInfo("detectorType", 0)) )
    {
        ERRWRAP2( retval = FeatureDetector::create(detectorType));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_FileNode_FileNode(PyObject* , PyObject* args, PyObject* kw)
{
    pyopencv_FileNode_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_FileNode_t, &pyopencv_FileNode_Type);
        if(self) ERRWRAP2(self->v = cv::FileNode());
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_FileStorage_FileStorage(PyObject* , PyObject* args, PyObject* kw)
{
    {
    pyopencv_FileStorage_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_FileStorage_t, &pyopencv_FileStorage_Type);
        new (&(self->v)) Ptr<cv::FileStorage>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::FileStorage());
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_FileStorage_t* self = 0;
    PyObject* pyobj_source = NULL;
    string source;
    int flags=0;
    PyObject* pyobj_encoding = NULL;
    string encoding;

    const char* keywords[] = { "source", "flags", "encoding", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|O:FileStorage", (char**)keywords, &pyobj_source, &flags, &pyobj_encoding) &&
        pyopencv_to(pyobj_source, source, ArgInfo("source", 0)) &&
        pyopencv_to(pyobj_encoding, encoding, ArgInfo("encoding", 0)) )
    {
        self = PyObject_NEW(pyopencv_FileStorage_t, &pyopencv_FileStorage_Type);
        new (&(self->v)) Ptr<cv::FileStorage>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::FileStorage(source, flags, encoding));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_FlannBasedMatcher_FlannBasedMatcher(PyObject* , PyObject* args, PyObject* kw)
{
    pyopencv_FlannBasedMatcher_t* self = 0;
    PyObject* pyobj_indexParams = NULL;
    Ptr_flann_IndexParams indexParams=new flann::KDTreeIndexParams();
    PyObject* pyobj_searchParams = NULL;
    Ptr_flann_SearchParams searchParams=new flann::SearchParams();

    const char* keywords[] = { "indexParams", "searchParams", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|OO:FlannBasedMatcher", (char**)keywords, &pyobj_indexParams, &pyobj_searchParams) &&
        pyopencv_to(pyobj_indexParams, indexParams, ArgInfo("indexParams", 0)) &&
        pyopencv_to(pyobj_searchParams, searchParams, ArgInfo("searchParams", 0)) )
    {
        self = PyObject_NEW(pyopencv_FlannBasedMatcher_t, &pyopencv_FlannBasedMatcher_Type);
        new (&(self->v)) Ptr<cv::FlannBasedMatcher>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::FlannBasedMatcher(indexParams, searchParams));
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_GaussianBlur(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::GaussianBlur(src, dst, ksize, sigmaX, sigmaY, borderType));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_GridAdaptedFeatureDetector_GridAdaptedFeatureDetector(PyObject* , PyObject* args, PyObject* kw)
{
    pyopencv_GridAdaptedFeatureDetector_t* self = 0;
    PyObject* pyobj_detector = NULL;
    Ptr_FeatureDetector detector=0;
    int maxTotalKeypoints=1000;
    int gridRows=4;
    int gridCols=4;

    const char* keywords[] = { "detector", "maxTotalKeypoints", "gridRows", "gridCols", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|Oiii:GridAdaptedFeatureDetector", (char**)keywords, &pyobj_detector, &maxTotalKeypoints, &gridRows, &gridCols) &&
        pyopencv_to(pyobj_detector, detector, ArgInfo("detector", 0)) )
    {
        self = PyObject_NEW(pyopencv_GridAdaptedFeatureDetector_t, &pyopencv_GridAdaptedFeatureDetector_Type);
        new (&(self->v)) Ptr<cv::GridAdaptedFeatureDetector>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::GridAdaptedFeatureDetector(detector, maxTotalKeypoints, gridRows, gridCols));
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_HOGDescriptor_HOGDescriptor(PyObject* , PyObject* args, PyObject* kw)
{
    {
    pyopencv_HOGDescriptor_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_HOGDescriptor_t, &pyopencv_HOGDescriptor_Type);
        new (&(self->v)) Ptr<cv::HOGDescriptor>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::HOGDescriptor());
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_HOGDescriptor_t* self = 0;
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

    const char* keywords[] = { "_winSize", "_blockSize", "_blockStride", "_cellSize", "_nbins", "_derivAperture", "_winSigma", "_histogramNormType", "_L2HysThreshold", "_gammaCorrection", "_nlevels", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOOi|ididbi:HOGDescriptor", (char**)keywords, &pyobj__winSize, &pyobj__blockSize, &pyobj__blockStride, &pyobj__cellSize, &_nbins, &_derivAperture, &_winSigma, &_histogramNormType, &_L2HysThreshold, &_gammaCorrection, &_nlevels) &&
        pyopencv_to(pyobj__winSize, _winSize, ArgInfo("_winSize", 0)) &&
        pyopencv_to(pyobj__blockSize, _blockSize, ArgInfo("_blockSize", 0)) &&
        pyopencv_to(pyobj__blockStride, _blockStride, ArgInfo("_blockStride", 0)) &&
        pyopencv_to(pyobj__cellSize, _cellSize, ArgInfo("_cellSize", 0)) )
    {
        self = PyObject_NEW(pyopencv_HOGDescriptor_t, &pyopencv_HOGDescriptor_Type);
        new (&(self->v)) Ptr<cv::HOGDescriptor>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::HOGDescriptor(_winSize, _blockSize, _blockStride, _cellSize, _nbins, _derivAperture, _winSigma, _histogramNormType, _L2HysThreshold, _gammaCorrection, _nlevels));
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_HOGDescriptor_t* self = 0;
    PyObject* pyobj_filename = NULL;
    String filename;

    const char* keywords[] = { "filename", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:HOGDescriptor", (char**)keywords, &pyobj_filename) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) )
    {
        self = PyObject_NEW(pyopencv_HOGDescriptor_t, &pyopencv_HOGDescriptor_Type);
        new (&(self->v)) Ptr<cv::HOGDescriptor>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::HOGDescriptor(filename));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_HOGDescriptor_getDaimlerPeopleDetector(PyObject* , PyObject* args, PyObject* kw)
{
    vector_float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2( retval = HOGDescriptor::getDaimlerPeopleDetector());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_HOGDescriptor_getDefaultPeopleDetector(PyObject* , PyObject* args, PyObject* kw)
{
    vector_float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2( retval = HOGDescriptor::getDefaultPeopleDetector());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_HoughCircles(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::HoughCircles(image, circles, method, dp, minDist, param1, param2, minRadius, maxRadius));
        return pyopencv_from(circles);
    }

    return NULL;
}

static PyObject* pyopencv_HoughLines(PyObject* , PyObject* args, PyObject* kw)
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

    const char* keywords[] = { "image", "rho", "theta", "threshold", "lines", "srn", "stn", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oddi|Odd:HoughLines", (char**)keywords, &pyobj_image, &rho, &theta, &threshold, &pyobj_lines, &srn, &stn) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_lines, lines, ArgInfo("lines", 1)) )
    {
        ERRWRAP2( cv::HoughLines(image, lines, rho, theta, threshold, srn, stn));
        return pyopencv_from(lines);
    }

    return NULL;
}

static PyObject* pyopencv_HoughLinesP(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::HoughLinesP(image, lines, rho, theta, threshold, minLineLength, maxLineGap));
        return pyopencv_from(lines);
    }

    return NULL;
}

static PyObject* pyopencv_HuMoments(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::HuMoments(m, hu));
        return pyopencv_from(hu);
    }

    return NULL;
}

static PyObject* pyopencv_flann_Index_Index(PyObject* , PyObject* args, PyObject* kw)
{
    {
    pyopencv_flann_Index_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_flann_Index_t, &pyopencv_flann_Index_Type);
        new (&(self->v)) Ptr<cv::flann::Index>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::flann::Index());
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_flann_Index_t* self = 0;
    PyObject* pyobj_features = NULL;
    Mat features;
    PyObject* pyobj_params = NULL;
    IndexParams params;
    PyObject* pyobj_distType = NULL;
    cvflann_flann_distance_t distType=cvflann::FLANN_DIST_L2;

    const char* keywords[] = { "features", "params", "distType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:Index", (char**)keywords, &pyobj_features, &pyobj_params, &pyobj_distType) &&
        pyopencv_to(pyobj_features, features, ArgInfo("features", 0)) &&
        pyopencv_to(pyobj_params, params, ArgInfo("params", 0)) &&
        pyopencv_to(pyobj_distType, distType, ArgInfo("distType", 0)) )
    {
        self = PyObject_NEW(pyopencv_flann_Index_t, &pyopencv_flann_Index_Type);
        new (&(self->v)) Ptr<cv::flann::Index>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::flann::Index(features, params, distType));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_KDTree_KDTree(PyObject* , PyObject* args, PyObject* kw)
{
    {
    pyopencv_KDTree_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_KDTree_t, &pyopencv_KDTree_Type);
        new (&(self->v)) Ptr<cv::KDTree>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::KDTree());
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_KDTree_t* self = 0;
    PyObject* pyobj_points = NULL;
    Mat points;
    bool copyAndReorderPoints=false;

    const char* keywords[] = { "points", "copyAndReorderPoints", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|b:KDTree", (char**)keywords, &pyobj_points, &copyAndReorderPoints) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 0)) )
    {
        self = PyObject_NEW(pyopencv_KDTree_t, &pyopencv_KDTree_Type);
        new (&(self->v)) Ptr<cv::KDTree>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::KDTree(points, copyAndReorderPoints));
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_KDTree_t* self = 0;
    PyObject* pyobj_points = NULL;
    Mat points;
    PyObject* pyobj__labels = NULL;
    Mat _labels;
    bool copyAndReorderPoints=false;

    const char* keywords[] = { "points", "_labels", "copyAndReorderPoints", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|b:KDTree", (char**)keywords, &pyobj_points, &pyobj__labels, &copyAndReorderPoints) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 0)) &&
        pyopencv_to(pyobj__labels, _labels, ArgInfo("_labels", 0)) )
    {
        self = PyObject_NEW(pyopencv_KDTree_t, &pyopencv_KDTree_Type);
        new (&(self->v)) Ptr<cv::KDTree>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::KDTree(points, _labels, copyAndReorderPoints));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_KalmanFilter_KalmanFilter(PyObject* , PyObject* args, PyObject* kw)
{
    {
    pyopencv_KalmanFilter_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_KalmanFilter_t, &pyopencv_KalmanFilter_Type);
        new (&(self->v)) Ptr<cv::KalmanFilter>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::KalmanFilter());
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_KalmanFilter_t* self = 0;
    int dynamParams=0;
    int measureParams=0;
    int controlParams=0;
    int type=CV_32F;

    const char* keywords[] = { "dynamParams", "measureParams", "controlParams", "type", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "ii|ii:KalmanFilter", (char**)keywords, &dynamParams, &measureParams, &controlParams, &type) )
    {
        self = PyObject_NEW(pyopencv_KalmanFilter_t, &pyopencv_KalmanFilter_Type);
        new (&(self->v)) Ptr<cv::KalmanFilter>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::KalmanFilter(dynamParams, measureParams, controlParams, type));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_KeyPoint_KeyPoint(PyObject* , PyObject* args, PyObject* kw)
{
    {
    pyopencv_KeyPoint_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_KeyPoint_t, &pyopencv_KeyPoint_Type);
        if(self) ERRWRAP2(self->v = cv::KeyPoint());
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_KeyPoint_t* self = 0;
    float x=0.f;
    float y=0.f;
    float _size=0.f;
    float _angle=-1;
    float _response=0;
    int _octave=0;
    int _class_id=-1;

    const char* keywords[] = { "x", "y", "_size", "_angle", "_response", "_octave", "_class_id", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "fff|ffii:KeyPoint", (char**)keywords, &x, &y, &_size, &_angle, &_response, &_octave, &_class_id) )
    {
        self = PyObject_NEW(pyopencv_KeyPoint_t, &pyopencv_KeyPoint_Type);
        if(self) ERRWRAP2(self->v = cv::KeyPoint(x, y, _size, _angle, _response, _octave, _class_id));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_LUT(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_lut = NULL;
    Mat lut;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int interpolation=0;

    const char* keywords[] = { "src", "lut", "dst", "interpolation", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Oi:LUT", (char**)keywords, &pyobj_src, &pyobj_lut, &pyobj_dst, &interpolation) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_lut, lut, ArgInfo("lut", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2( cv::LUT(src, lut, dst, interpolation));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_Laplacian(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::Laplacian(src, dst, ddepth, ksize, scale, delta, borderType));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_MSER_MSER(PyObject* , PyObject* args, PyObject* kw)
{
    pyopencv_MSER_t* self = 0;
    int _delta=5;
    int _min_area=60;
    int _max_area=14400;
    double _max_variation=0.25;
    double _min_diversity=.2;
    int _max_evolution=200;
    double _area_threshold=1.01;
    double _min_margin=0.003;
    int _edge_blur_size=5;

    const char* keywords[] = { "_delta", "_min_area", "_max_area", "_max_variation", "_min_diversity", "_max_evolution", "_area_threshold", "_min_margin", "_edge_blur_size", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|iiiddiddi:MSER", (char**)keywords, &_delta, &_min_area, &_max_area, &_max_variation, &_min_diversity, &_max_evolution, &_area_threshold, &_min_margin, &_edge_blur_size) )
    {
        self = PyObject_NEW(pyopencv_MSER_t, &pyopencv_MSER_Type);
        new (&(self->v)) Ptr<cv::MSER>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::MSER(_delta, _min_area, _max_area, _max_variation, _min_diversity, _max_evolution, _area_threshold, _min_margin, _edge_blur_size));
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_Mahalanobis(PyObject* , PyObject* args, PyObject* kw)
{
    double retval;
    PyObject* pyobj_v1 = NULL;
    Mat v1;
    PyObject* pyobj_v2 = NULL;
    Mat v2;
    PyObject* pyobj_icovar = NULL;
    Mat icovar;

    const char* keywords[] = { "v1", "v2", "icovar", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:Mahalanobis", (char**)keywords, &pyobj_v1, &pyobj_v2, &pyobj_icovar) &&
        pyopencv_to(pyobj_v1, v1, ArgInfo("v1", 0)) &&
        pyopencv_to(pyobj_v2, v2, ArgInfo("v2", 0)) &&
        pyopencv_to(pyobj_icovar, icovar, ArgInfo("icovar", 0)) )
    {
        ERRWRAP2( retval = cv::Mahalanobis(v1, v2, icovar));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_ORB_ORB(PyObject* , PyObject* args, PyObject* kw)
{
    pyopencv_ORB_t* self = 0;
    int nfeatures=500;
    float scaleFactor=1.2f;
    int nlevels=8;
    int edgeThreshold=31;
    int firstLevel=0;
    int WTA_K=2;
    int scoreType=ORB::HARRIS_SCORE;
    int patchSize=31;

    const char* keywords[] = { "nfeatures", "scaleFactor", "nlevels", "edgeThreshold", "firstLevel", "WTA_K", "scoreType", "patchSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|ifiiiiii:ORB", (char**)keywords, &nfeatures, &scaleFactor, &nlevels, &edgeThreshold, &firstLevel, &WTA_K, &scoreType, &patchSize) )
    {
        self = PyObject_NEW(pyopencv_ORB_t, &pyopencv_ORB_Type);
        new (&(self->v)) Ptr<cv::ORB>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::ORB(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize));
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_PCABackProject(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::PCABackProject(data, mean, eigenvectors, result));
        return pyopencv_from(result);
    }

    return NULL;
}

static PyObject* pyopencv_PCACompute(PyObject* , PyObject* args, PyObject* kw)
{
    {
    PyObject* pyobj_data = NULL;
    Mat data;
    PyObject* pyobj_mean = NULL;
    Mat mean;
    PyObject* pyobj_eigenvectors = NULL;
    Mat eigenvectors;
    int maxComponents=0;

    const char* keywords[] = { "data", "mean", "eigenvectors", "maxComponents", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOi:PCACompute", (char**)keywords, &pyobj_data, &pyobj_mean, &pyobj_eigenvectors, &maxComponents) &&
        pyopencv_to(pyobj_data, data, ArgInfo("data", 0)) &&
        pyopencv_to(pyobj_mean, mean, ArgInfo("mean", 1)) &&
        pyopencv_to(pyobj_eigenvectors, eigenvectors, ArgInfo("eigenvectors", 1)) )
    {
        ERRWRAP2( cv::PCACompute(data, mean, eigenvectors, maxComponents));
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

    const char* keywords[] = { "data", "retainedVariance", "mean", "eigenvectors", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Od|OO:PCACompute", (char**)keywords, &pyobj_data, &retainedVariance, &pyobj_mean, &pyobj_eigenvectors) &&
        pyopencv_to(pyobj_data, data, ArgInfo("data", 0)) &&
        pyopencv_to(pyobj_mean, mean, ArgInfo("mean", 1)) &&
        pyopencv_to(pyobj_eigenvectors, eigenvectors, ArgInfo("eigenvectors", 1)) )
    {
        ERRWRAP2( cv::PCACompute(data, mean, eigenvectors, retainedVariance));
        return Py_BuildValue("(NN)", pyopencv_from(mean), pyopencv_from(eigenvectors));
    }
    }

    return NULL;
}

static PyObject* pyopencv_PCAProject(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::PCAProject(data, mean, eigenvectors, result));
        return pyopencv_from(result);
    }

    return NULL;
}

static PyObject* pyopencv_PSNR(PyObject* , PyObject* args, PyObject* kw)
{
    double retval;
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    PyObject* pyobj_src2 = NULL;
    Mat src2;

    const char* keywords[] = { "src1", "src2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:PSNR", (char**)keywords, &pyobj_src1, &pyobj_src2) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) )
    {
        ERRWRAP2( retval = cv::PSNR(src1, src2));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_SimpleBlobDetector_Params_Params(PyObject* , PyObject* args, PyObject* kw)
{
    pyopencv_SimpleBlobDetector_Params_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_SimpleBlobDetector_Params_t, &pyopencv_SimpleBlobDetector_Params_Type);
        if(self) ERRWRAP2(self->v = cv::SimpleBlobDetector::Params());
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_PyramidAdaptedFeatureDetector_PyramidAdaptedFeatureDetector(PyObject* , PyObject* args, PyObject* kw)
{
    pyopencv_PyramidAdaptedFeatureDetector_t* self = 0;
    PyObject* pyobj_detector = NULL;
    Ptr_FeatureDetector detector;
    int maxLevel=2;

    const char* keywords[] = { "detector", "maxLevel", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|i:PyramidAdaptedFeatureDetector", (char**)keywords, &pyobj_detector, &maxLevel) &&
        pyopencv_to(pyobj_detector, detector, ArgInfo("detector", 0)) )
    {
        self = PyObject_NEW(pyopencv_PyramidAdaptedFeatureDetector_t, &pyopencv_PyramidAdaptedFeatureDetector_Type);
        new (&(self->v)) Ptr<cv::PyramidAdaptedFeatureDetector>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::PyramidAdaptedFeatureDetector(detector, maxLevel));
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_RQDecomp3x3(PyObject* , PyObject* args, PyObject* kw)
{
    Vec3d retval;
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

    const char* keywords[] = { "src", "mtxR", "mtxQ", "Qx", "Qy", "Qz", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOOOO:RQDecomp3x3", (char**)keywords, &pyobj_src, &pyobj_mtxR, &pyobj_mtxQ, &pyobj_Qx, &pyobj_Qy, &pyobj_Qz) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_mtxR, mtxR, ArgInfo("mtxR", 1)) &&
        pyopencv_to(pyobj_mtxQ, mtxQ, ArgInfo("mtxQ", 1)) &&
        pyopencv_to(pyobj_Qx, Qx, ArgInfo("Qx", 1)) &&
        pyopencv_to(pyobj_Qy, Qy, ArgInfo("Qy", 1)) &&
        pyopencv_to(pyobj_Qz, Qz, ArgInfo("Qz", 1)) )
    {
        ERRWRAP2( retval = cv::RQDecomp3x3(src, mtxR, mtxQ, Qx, Qy, Qz));
        return Py_BuildValue("(NNNNNN)", pyopencv_from(retval), pyopencv_from(mtxR), pyopencv_from(mtxQ), pyopencv_from(Qx), pyopencv_from(Qy), pyopencv_from(Qz));
    }

    return NULL;
}

static PyObject* pyopencv_Rodrigues(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::Rodrigues(src, dst, jacobian));
        return Py_BuildValue("(NN)", pyopencv_from(dst), pyopencv_from(jacobian));
    }

    return NULL;
}

static PyObject* pyopencv_SCascade_SCascade(PyObject* , PyObject* args, PyObject* kw)
{
    pyopencv_SCascade_t* self = 0;
    double minScale=0.4;
    double maxScale=5.;
    int scales=55;
    int rejfactor=1;

    const char* keywords[] = { "minScale", "maxScale", "scales", "rejfactor", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|ddii:SCascade", (char**)keywords, &minScale, &maxScale, &scales, &rejfactor) )
    {
        self = PyObject_NEW(pyopencv_SCascade_t, &pyopencv_SCascade_Type);
        new (&(self->v)) Ptr<cv::SCascade>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::SCascade(minScale, maxScale, scales, rejfactor));
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_SIFT_SIFT(PyObject* , PyObject* args, PyObject* kw)
{
    pyopencv_SIFT_t* self = 0;
    int nfeatures=0;
    int nOctaveLayers=3;
    double contrastThreshold=0.04;
    double edgeThreshold=10;
    double sigma=1.6;

    const char* keywords[] = { "nfeatures", "nOctaveLayers", "contrastThreshold", "edgeThreshold", "sigma", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|iiddd:SIFT", (char**)keywords, &nfeatures, &nOctaveLayers, &contrastThreshold, &edgeThreshold, &sigma) )
    {
        self = PyObject_NEW(pyopencv_SIFT_t, &pyopencv_SIFT_Type);
        new (&(self->v)) Ptr<cv::SIFT>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::SIFT(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma));
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_SURF_SURF(PyObject* , PyObject* args, PyObject* kw)
{
    {
    pyopencv_SURF_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_SURF_t, &pyopencv_SURF_Type);
        new (&(self->v)) Ptr<cv::SURF>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::SURF());
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_SURF_t* self = 0;
    double hessianThreshold=0;
    int nOctaves=4;
    int nOctaveLayers=2;
    bool extended=true;
    bool upright=false;

    const char* keywords[] = { "hessianThreshold", "nOctaves", "nOctaveLayers", "extended", "upright", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d|iibb:SURF", (char**)keywords, &hessianThreshold, &nOctaves, &nOctaveLayers, &extended, &upright) )
    {
        self = PyObject_NEW(pyopencv_SURF_t, &pyopencv_SURF_Type);
        new (&(self->v)) Ptr<cv::SURF>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::SURF(hessianThreshold, nOctaves, nOctaveLayers, extended, upright));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_SVBackSubst(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::SVBackSubst(w, u, vt, rhs, dst));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_SVDecomp(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::SVDecomp(src, w, u, vt, flags));
        return Py_BuildValue("(NNN)", pyopencv_from(w), pyopencv_from(u), pyopencv_from(vt));
    }

    return NULL;
}

static PyObject* pyopencv_Scharr(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::Scharr(src, dst, ddepth, dx, dy, scale, delta, borderType));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_SimpleBlobDetector_SimpleBlobDetector(PyObject* , PyObject* args, PyObject* kw)
{
    pyopencv_SimpleBlobDetector_t* self = 0;
    PyObject* pyobj_parameters = NULL;
    SimpleBlobDetector_Params parameters=SimpleBlobDetector::Params();

    const char* keywords[] = { "parameters", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|O:SimpleBlobDetector", (char**)keywords, &pyobj_parameters) &&
        pyopencv_to(pyobj_parameters, parameters, ArgInfo("parameters", 0)) )
    {
        self = PyObject_NEW(pyopencv_SimpleBlobDetector_t, &pyopencv_SimpleBlobDetector_Type);
        new (&(self->v)) Ptr<cv::SimpleBlobDetector>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::SimpleBlobDetector(parameters));
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_Sobel(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::Sobel(src, dst, ddepth, dx, dy, ksize, scale, delta, borderType));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_StarDetector_StarDetector(PyObject* , PyObject* args, PyObject* kw)
{
    pyopencv_StarDetector_t* self = 0;
    int _maxSize=45;
    int _responseThreshold=30;
    int _lineThresholdProjected=10;
    int _lineThresholdBinarized=8;
    int _suppressNonmaxSize=5;

    const char* keywords[] = { "_maxSize", "_responseThreshold", "_lineThresholdProjected", "_lineThresholdBinarized", "_suppressNonmaxSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|iiiii:StarDetector", (char**)keywords, &_maxSize, &_responseThreshold, &_lineThresholdProjected, &_lineThresholdBinarized, &_suppressNonmaxSize) )
    {
        self = PyObject_NEW(pyopencv_StarDetector_t, &pyopencv_StarDetector_Type);
        new (&(self->v)) Ptr<cv::StarDetector>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::StarDetector(_maxSize, _responseThreshold, _lineThresholdProjected, _lineThresholdBinarized, _suppressNonmaxSize));
        return (PyObject*)self;
    }

    return NULL;
}

static PyObject* pyopencv_StereoBM_StereoBM(PyObject* , PyObject* args, PyObject* kw)
{
    {
    pyopencv_StereoBM_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_StereoBM_t, &pyopencv_StereoBM_Type);
        new (&(self->v)) Ptr<cv::StereoBM>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::StereoBM());
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_StereoBM_t* self = 0;
    int preset=0;
    int ndisparities=0;
    int SADWindowSize=21;

    const char* keywords[] = { "preset", "ndisparities", "SADWindowSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i|ii:StereoBM", (char**)keywords, &preset, &ndisparities, &SADWindowSize) )
    {
        self = PyObject_NEW(pyopencv_StereoBM_t, &pyopencv_StereoBM_Type);
        new (&(self->v)) Ptr<cv::StereoBM>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::StereoBM(preset, ndisparities, SADWindowSize));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_StereoSGBM_StereoSGBM(PyObject* , PyObject* args, PyObject* kw)
{
    {
    pyopencv_StereoSGBM_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_StereoSGBM_t, &pyopencv_StereoSGBM_Type);
        new (&(self->v)) Ptr<cv::StereoSGBM>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::StereoSGBM());
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_StereoSGBM_t* self = 0;
    int minDisparity=0;
    int numDisparities=0;
    int SADWindowSize=0;
    int P1=0;
    int P2=0;
    int disp12MaxDiff=0;
    int preFilterCap=0;
    int uniquenessRatio=0;
    int speckleWindowSize=0;
    int speckleRange=0;
    bool fullDP=false;

    const char* keywords[] = { "minDisparity", "numDisparities", "SADWindowSize", "P1", "P2", "disp12MaxDiff", "preFilterCap", "uniquenessRatio", "speckleWindowSize", "speckleRange", "fullDP", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "iii|iiiiiiib:StereoSGBM", (char**)keywords, &minDisparity, &numDisparities, &SADWindowSize, &P1, &P2, &disp12MaxDiff, &preFilterCap, &uniquenessRatio, &speckleWindowSize, &speckleRange, &fullDP) )
    {
        self = PyObject_NEW(pyopencv_StereoSGBM_t, &pyopencv_StereoSGBM_Type);
        new (&(self->v)) Ptr<cv::StereoSGBM>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::StereoSGBM(minDisparity, numDisparities, SADWindowSize, P1, P2, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, fullDP));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_StereoVar_StereoVar(PyObject* , PyObject* args, PyObject* kw)
{
    {
    pyopencv_StereoVar_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_StereoVar_t, &pyopencv_StereoVar_Type);
        new (&(self->v)) Ptr<cv::StereoVar>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::StereoVar());
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_StereoVar_t* self = 0;
    int levels=0;
    double pyrScale=0;
    int nIt=0;
    int minDisp=0;
    int maxDisp=0;
    int poly_n=0;
    double poly_sigma=0;
    float fi=0.f;
    float lambda=0.f;
    int penalization=0;
    int cycle=0;
    int flags=0;

    const char* keywords[] = { "levels", "pyrScale", "nIt", "minDisp", "maxDisp", "poly_n", "poly_sigma", "fi", "lambda", "penalization", "cycle", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "idiiiidffiii:StereoVar", (char**)keywords, &levels, &pyrScale, &nIt, &minDisp, &maxDisp, &poly_n, &poly_sigma, &fi, &lambda, &penalization, &cycle, &flags) )
    {
        self = PyObject_NEW(pyopencv_StereoVar_t, &pyopencv_StereoVar_Type);
        new (&(self->v)) Ptr<cv::StereoVar>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::StereoVar(levels, pyrScale, nIt, minDisp, maxDisp, poly_n, poly_sigma, fi, lambda, penalization, cycle, flags));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_Subdiv2D_Subdiv2D(PyObject* , PyObject* args, PyObject* kw)
{
    {
    pyopencv_Subdiv2D_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_Subdiv2D_t, &pyopencv_Subdiv2D_Type);
        new (&(self->v)) Ptr<cv::Subdiv2D>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::Subdiv2D());
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_Subdiv2D_t* self = 0;
    PyObject* pyobj_rect = NULL;
    Rect rect;

    const char* keywords[] = { "rect", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:Subdiv2D", (char**)keywords, &pyobj_rect) &&
        pyopencv_to(pyobj_rect, rect, ArgInfo("rect", 0)) )
    {
        self = PyObject_NEW(pyopencv_Subdiv2D_t, &pyopencv_Subdiv2D_Type);
        new (&(self->v)) Ptr<cv::Subdiv2D>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::Subdiv2D(rect));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_VideoCapture_VideoCapture(PyObject* , PyObject* args, PyObject* kw)
{
    {
    pyopencv_VideoCapture_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_VideoCapture_t, &pyopencv_VideoCapture_Type);
        new (&(self->v)) Ptr<cv::VideoCapture>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::VideoCapture());
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_VideoCapture_t* self = 0;
    PyObject* pyobj_filename = NULL;
    string filename;

    const char* keywords[] = { "filename", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:VideoCapture", (char**)keywords, &pyobj_filename) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) )
    {
        self = PyObject_NEW(pyopencv_VideoCapture_t, &pyopencv_VideoCapture_Type);
        new (&(self->v)) Ptr<cv::VideoCapture>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::VideoCapture(filename));
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_VideoCapture_t* self = 0;
    int device=0;

    const char* keywords[] = { "device", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:VideoCapture", (char**)keywords, &device) )
    {
        self = PyObject_NEW(pyopencv_VideoCapture_t, &pyopencv_VideoCapture_Type);
        new (&(self->v)) Ptr<cv::VideoCapture>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::VideoCapture(device));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_VideoWriter_VideoWriter(PyObject* , PyObject* args, PyObject* kw)
{
    {
    pyopencv_VideoWriter_t* self = 0;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        self = PyObject_NEW(pyopencv_VideoWriter_t, &pyopencv_VideoWriter_Type);
        new (&(self->v)) Ptr<cv::VideoWriter>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::VideoWriter());
        return (PyObject*)self;
    }
    }
    PyErr_Clear();

    {
    pyopencv_VideoWriter_t* self = 0;
    PyObject* pyobj_filename = NULL;
    string filename;
    int fourcc=0;
    double fps=0;
    PyObject* pyobj_frameSize = NULL;
    Size frameSize;
    bool isColor=true;

    const char* keywords[] = { "filename", "fourcc", "fps", "frameSize", "isColor", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OidO|b:VideoWriter", (char**)keywords, &pyobj_filename, &fourcc, &fps, &pyobj_frameSize, &isColor) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) &&
        pyopencv_to(pyobj_frameSize, frameSize, ArgInfo("frameSize", 0)) )
    {
        self = PyObject_NEW(pyopencv_VideoWriter_t, &pyopencv_VideoWriter_Type);
        new (&(self->v)) Ptr<cv::VideoWriter>(); // init Ptr with placement new
        if(self) ERRWRAP2(self->v = new cv::VideoWriter(filename, fourcc, fps, frameSize, isColor));
        return (PyObject*)self;
    }
    }

    return NULL;
}

static PyObject* pyopencv_absdiff(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::absdiff(src1, src2, dst));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_accumulate(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::accumulate(src, dst, mask));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_accumulateProduct(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::accumulateProduct(src1, src2, dst, mask));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_accumulateSquare(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::accumulateSquare(src, dst, mask));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_accumulateWeighted(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::accumulateWeighted(src, dst, alpha, mask));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_adaptiveThreshold(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::adaptiveThreshold(src, dst, maxValue, adaptiveMethod, thresholdType, blockSize, C));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_add(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::add(src1, src2, dst, mask, dtype));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_addWeighted(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::addWeighted(src1, alpha, src2, beta, gamma, dst, dtype));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_applyColorMap(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::applyColorMap(src, dst, colormap));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_approxPolyDP(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::approxPolyDP(curve, approxCurve, epsilon, closed));
        return pyopencv_from(approxCurve);
    }

    return NULL;
}

static PyObject* pyopencv_arcLength(PyObject* , PyObject* args, PyObject* kw)
{
    double retval;
    PyObject* pyobj_curve = NULL;
    Mat curve;
    bool closed=0;

    const char* keywords[] = { "curve", "closed", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Ob:arcLength", (char**)keywords, &pyobj_curve, &closed) &&
        pyopencv_to(pyobj_curve, curve, ArgInfo("curve", 0)) )
    {
        ERRWRAP2( retval = cv::arcLength(curve, closed));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_batchDistance(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::batchDistance(src1, src2, dist, dtype, nidx, normType, K, mask, update, crosscheck));
        return Py_BuildValue("(NN)", pyopencv_from(dist), pyopencv_from(nidx));
    }

    return NULL;
}

static PyObject* pyopencv_bilateralFilter(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::bilateralFilter(src, dst, d, sigmaColor, sigmaSpace, borderType));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_bitwise_and(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::bitwise_and(src1, src2, dst, mask));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_bitwise_not(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::bitwise_not(src, dst, mask));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_bitwise_or(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::bitwise_or(src1, src2, dst, mask));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_bitwise_xor(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::bitwise_xor(src1, src2, dst, mask));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_blur(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::blur(src, dst, ksize, anchor, borderType));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_borderInterpolate(PyObject* , PyObject* args, PyObject* kw)
{
    int retval;
    int p=0;
    int len=0;
    int borderType=0;

    const char* keywords[] = { "p", "len", "borderType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "iii:borderInterpolate", (char**)keywords, &p, &len, &borderType) )
    {
        ERRWRAP2( retval = cv::borderInterpolate(p, len, borderType));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_boundingRect(PyObject* , PyObject* args, PyObject* kw)
{
    Rect retval;
    PyObject* pyobj_points = NULL;
    Mat points;

    const char* keywords[] = { "points", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:boundingRect", (char**)keywords, &pyobj_points) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 0)) )
    {
        ERRWRAP2( retval = cv::boundingRect(points));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_boxFilter(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::boxFilter(src, dst, ddepth, ksize, anchor, normalize, borderType));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_buildOpticalFlowPyramid(PyObject* , PyObject* args, PyObject* kw)
{
    int retval;
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

    const char* keywords[] = { "img", "winSize", "maxLevel", "pyramid", "withDerivatives", "pyrBorder", "derivBorder", "tryReuseInputImage", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi|Obiib:buildOpticalFlowPyramid", (char**)keywords, &pyobj_img, &pyobj_winSize, &maxLevel, &pyobj_pyramid, &withDerivatives, &pyrBorder, &derivBorder, &tryReuseInputImage) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_pyramid, pyramid, ArgInfo("pyramid", 1)) &&
        pyopencv_to(pyobj_winSize, winSize, ArgInfo("winSize", 0)) )
    {
        ERRWRAP2( retval = cv::buildOpticalFlowPyramid(img, pyramid, winSize, maxLevel, withDerivatives, pyrBorder, derivBorder, tryReuseInputImage));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(pyramid));
    }

    return NULL;
}

static PyObject* pyopencv_calcBackProject(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::calcBackProject(images, channels, hist, dst, ranges, scale));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_calcCovarMatrix(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_samples = NULL;
    Mat samples;
    PyObject* pyobj_covar = NULL;
    Mat covar;
    PyObject* pyobj_mean = NULL;
    Mat mean;
    int flags=0;
    int ctype=CV_64F;

    const char* keywords[] = { "samples", "flags", "covar", "mean", "ctype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|OOi:calcCovarMatrix", (char**)keywords, &pyobj_samples, &flags, &pyobj_covar, &pyobj_mean, &ctype) &&
        pyopencv_to(pyobj_samples, samples, ArgInfo("samples", 0)) &&
        pyopencv_to(pyobj_covar, covar, ArgInfo("covar", 1)) &&
        pyopencv_to(pyobj_mean, mean, ArgInfo("mean", 1)) )
    {
        ERRWRAP2( cv::calcCovarMatrix(samples, covar, mean, flags, ctype));
        return Py_BuildValue("(NN)", pyopencv_from(covar), pyopencv_from(mean));
    }

    return NULL;
}

static PyObject* pyopencv_calcGlobalOrientation(PyObject* , PyObject* args, PyObject* kw)
{
    double retval;
    PyObject* pyobj_orientation = NULL;
    Mat orientation;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    PyObject* pyobj_mhi = NULL;
    Mat mhi;
    double timestamp=0;
    double duration=0;

    const char* keywords[] = { "orientation", "mask", "mhi", "timestamp", "duration", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOdd:calcGlobalOrientation", (char**)keywords, &pyobj_orientation, &pyobj_mask, &pyobj_mhi, &timestamp, &duration) &&
        pyopencv_to(pyobj_orientation, orientation, ArgInfo("orientation", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) &&
        pyopencv_to(pyobj_mhi, mhi, ArgInfo("mhi", 0)) )
    {
        ERRWRAP2( retval = cv::calcGlobalOrientation(orientation, mask, mhi, timestamp, duration));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_calcHist(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::calcHist(images, channels, mask, hist, histSize, ranges, accumulate));
        return pyopencv_from(hist);
    }

    return NULL;
}

static PyObject* pyopencv_calcMotionGradient(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::calcMotionGradient(mhi, mask, orientation, delta1, delta2, apertureSize));
        return Py_BuildValue("(NN)", pyopencv_from(mask), pyopencv_from(orientation));
    }

    return NULL;
}

static PyObject* pyopencv_calcOpticalFlowFarneback(PyObject* , PyObject* args, PyObject* kw)
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

    const char* keywords[] = { "prev", "next", "pyr_scale", "levels", "winsize", "iterations", "poly_n", "poly_sigma", "flags", "flow", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOdiiiidi|O:calcOpticalFlowFarneback", (char**)keywords, &pyobj_prev, &pyobj_next, &pyr_scale, &levels, &winsize, &iterations, &poly_n, &poly_sigma, &flags, &pyobj_flow) &&
        pyopencv_to(pyobj_prev, prev, ArgInfo("prev", 0)) &&
        pyopencv_to(pyobj_next, next, ArgInfo("next", 0)) &&
        pyopencv_to(pyobj_flow, flow, ArgInfo("flow", 1)) )
    {
        ERRWRAP2( cv::calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags));
        return pyopencv_from(flow);
    }

    return NULL;
}

static PyObject* pyopencv_calcOpticalFlowPyrLK(PyObject* , PyObject* args, PyObject* kw)
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
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|OOOOiOid:calcOpticalFlowPyrLK", (char**)keywords, &pyobj_prevImg, &pyobj_nextImg, &pyobj_prevPts, &pyobj_nextPts, &pyobj_status, &pyobj_err, &pyobj_winSize, &maxLevel, &pyobj_criteria, &flags, &minEigThreshold) &&
        pyopencv_to(pyobj_prevImg, prevImg, ArgInfo("prevImg", 0)) &&
        pyopencv_to(pyobj_nextImg, nextImg, ArgInfo("nextImg", 0)) &&
        pyopencv_to(pyobj_prevPts, prevPts, ArgInfo("prevPts", 0)) &&
        pyopencv_to(pyobj_nextPts, nextPts, ArgInfo("nextPts", 1)) &&
        pyopencv_to(pyobj_status, status, ArgInfo("status", 1)) &&
        pyopencv_to(pyobj_err, err, ArgInfo("err", 1)) &&
        pyopencv_to(pyobj_winSize, winSize, ArgInfo("winSize", 0)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2( cv::calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, status, err, winSize, maxLevel, criteria, flags, minEigThreshold));
        return Py_BuildValue("(NNN)", pyopencv_from(nextPts), pyopencv_from(status), pyopencv_from(err));
    }

    return NULL;
}

static PyObject* pyopencv_calcOpticalFlowSF(PyObject* , PyObject* args, PyObject* kw)
{
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
        ERRWRAP2( cv::calcOpticalFlowSF(from, to, flow, layers, averaging_block_size, max_flow));
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
        ERRWRAP2( cv::calcOpticalFlowSF(from, to, flow, layers, averaging_block_size, max_flow, sigma_dist, sigma_color, postprocess_window, sigma_dist_fix, sigma_color_fix, occ_thr, upscale_averaging_radius, upscale_sigma_dist, upscale_sigma_color, speed_up_thr));
        return pyopencv_from(flow);
    }
    }

    return NULL;
}

static PyObject* pyopencv_calibrateCamera(PyObject* , PyObject* args, PyObject* kw)
{
    double retval;
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
    TermCriteria criteria=TermCriteria( TermCriteria::COUNT+TermCriteria::EPS, 30, DBL_EPSILON);

    const char* keywords[] = { "objectPoints", "imagePoints", "imageSize", "cameraMatrix", "distCoeffs", "rvecs", "tvecs", "flags", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|OOOOiO:calibrateCamera", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints, &pyobj_imageSize, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvecs, &pyobj_tvecs, &flags, &pyobj_criteria) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints, imagePoints, ArgInfo("imagePoints", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 1)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 1)) &&
        pyopencv_to(pyobj_rvecs, rvecs, ArgInfo("rvecs", 1)) &&
        pyopencv_to(pyobj_tvecs, tvecs, ArgInfo("tvecs", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2( retval = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags, criteria));
        return Py_BuildValue("(NNNNN)", pyopencv_from(retval), pyopencv_from(cameraMatrix), pyopencv_from(distCoeffs), pyopencv_from(rvecs), pyopencv_from(tvecs));
    }

    return NULL;
}

static PyObject* pyopencv_calibrationMatrixValues(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::calibrationMatrixValues(cameraMatrix, imageSize, apertureWidth, apertureHeight, fovx, fovy, focalLength, principalPoint, aspectRatio));
        return Py_BuildValue("(NNNNN)", pyopencv_from(fovx), pyopencv_from(fovy), pyopencv_from(focalLength), pyopencv_from(principalPoint), pyopencv_from(aspectRatio));
    }

    return NULL;
}

static PyObject* pyopencv_cartToPolar(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::cartToPolar(x, y, magnitude, angle, angleInDegrees));
        return Py_BuildValue("(NN)", pyopencv_from(magnitude), pyopencv_from(angle));
    }

    return NULL;
}

static PyObject* pyopencv_chamerMatching(PyObject* , PyObject* args, PyObject* kw)
{
    int retval;
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_templ = NULL;
    Mat templ;
    vector_vector_Point results;
    vector_float cost;
    double templScale=1;
    int maxMatches=20;
    double minMatchDistance=1.0;
    int padX=3;
    int padY=3;
    int scales=5;
    double minScale=0.6;
    double maxScale=1.6;
    double orientationWeight=0.5;
    double truncate=20;

    const char* keywords[] = { "img", "templ", "templScale", "maxMatches", "minMatchDistance", "padX", "padY", "scales", "minScale", "maxScale", "orientationWeight", "truncate", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|didiiidddd:chamerMatching", (char**)keywords, &pyobj_img, &pyobj_templ, &templScale, &maxMatches, &minMatchDistance, &padX, &padY, &scales, &minScale, &maxScale, &orientationWeight, &truncate) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_templ, templ, ArgInfo("templ", 0)) )
    {
        ERRWRAP2( retval = cv::chamerMatching(img, templ, results, cost, templScale, maxMatches, minMatchDistance, padX, padY, scales, minScale, maxScale, orientationWeight, truncate));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(results), pyopencv_from(cost));
    }

    return NULL;
}

static PyObject* pyopencv_checkHardwareSupport(PyObject* , PyObject* args, PyObject* kw)
{
    bool retval;
    int feature=0;

    const char* keywords[] = { "feature", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:checkHardwareSupport", (char**)keywords, &feature) )
    {
        ERRWRAP2( retval = cv::checkHardwareSupport(feature));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_checkRange(PyObject* , PyObject* args, PyObject* kw)
{
    bool retval;
    PyObject* pyobj_a = NULL;
    Mat a;
    bool quiet=true;
    Point pos;
    double minVal=-DBL_MAX;
    double maxVal=DBL_MAX;

    const char* keywords[] = { "a", "quiet", "minVal", "maxVal", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|bdd:checkRange", (char**)keywords, &pyobj_a, &quiet, &minVal, &maxVal) &&
        pyopencv_to(pyobj_a, a, ArgInfo("a", 0)) )
    {
        ERRWRAP2( retval = cv::checkRange(a, quiet, &pos, minVal, maxVal));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(pos));
    }

    return NULL;
}

static PyObject* pyopencv_circle(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_center = NULL;
    Point center;
    int radius=0;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int thickness=1;
    int lineType=8;
    int shift=0;

    const char* keywords[] = { "img", "center", "radius", "color", "thickness", "lineType", "shift", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOiO|iii:circle", (char**)keywords, &pyobj_img, &pyobj_center, &radius, &pyobj_color, &thickness, &lineType, &shift) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_center, center, ArgInfo("center", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2( cv::circle(img, center, radius, color, thickness, lineType, shift));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_clipLine(PyObject* , PyObject* args, PyObject* kw)
{
    bool retval;
    PyObject* pyobj_imgRect = NULL;
    Rect imgRect;
    PyObject* pyobj_pt1 = NULL;
    Point pt1;
    PyObject* pyobj_pt2 = NULL;
    Point pt2;

    const char* keywords[] = { "imgRect", "pt1", "pt2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:clipLine", (char**)keywords, &pyobj_imgRect, &pyobj_pt1, &pyobj_pt2) &&
        pyopencv_to(pyobj_imgRect, imgRect, ArgInfo("imgRect", 0)) &&
        pyopencv_to(pyobj_pt1, pt1, ArgInfo("pt1", 1)) &&
        pyopencv_to(pyobj_pt2, pt2, ArgInfo("pt2", 1)) )
    {
        ERRWRAP2( retval = cv::clipLine(imgRect, pt1, pt2));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(pt1), pyopencv_from(pt2));
    }

    return NULL;
}

static PyObject* pyopencv_compare(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::compare(src1, src2, dst, cmpop));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_compareHist(PyObject* , PyObject* args, PyObject* kw)
{
    double retval;
    PyObject* pyobj_H1 = NULL;
    Mat H1;
    PyObject* pyobj_H2 = NULL;
    Mat H2;
    int method=0;

    const char* keywords[] = { "H1", "H2", "method", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi:compareHist", (char**)keywords, &pyobj_H1, &pyobj_H2, &method) &&
        pyopencv_to(pyobj_H1, H1, ArgInfo("H1", 0)) &&
        pyopencv_to(pyobj_H2, H2, ArgInfo("H2", 0)) )
    {
        ERRWRAP2( retval = cv::compareHist(H1, H2, method));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_completeSymm(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_mtx = NULL;
    Mat mtx;
    bool lowerToUpper=false;

    const char* keywords[] = { "mtx", "lowerToUpper", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|b:completeSymm", (char**)keywords, &pyobj_mtx, &lowerToUpper) &&
        pyopencv_to(pyobj_mtx, mtx, ArgInfo("mtx", 1)) )
    {
        ERRWRAP2( cv::completeSymm(mtx, lowerToUpper));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_composeRT(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::composeRT(rvec1, tvec1, rvec2, tvec2, rvec3, tvec3, dr3dr1, dr3dt1, dr3dr2, dr3dt2, dt3dr1, dt3dt1, dt3dr2, dt3dt2));
        return Py_BuildValue("(NNNNNNNNNN)", pyopencv_from(rvec3), pyopencv_from(tvec3), pyopencv_from(dr3dr1), pyopencv_from(dr3dt1), pyopencv_from(dr3dr2), pyopencv_from(dr3dt2), pyopencv_from(dt3dr1), pyopencv_from(dt3dt1), pyopencv_from(dt3dr2), pyopencv_from(dt3dt2));
    }

    return NULL;
}

static PyObject* pyopencv_contourArea(PyObject* , PyObject* args, PyObject* kw)
{
    double retval;
    PyObject* pyobj_contour = NULL;
    Mat contour;
    bool oriented=false;

    const char* keywords[] = { "contour", "oriented", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|b:contourArea", (char**)keywords, &pyobj_contour, &oriented) &&
        pyopencv_to(pyobj_contour, contour, ArgInfo("contour", 0)) )
    {
        ERRWRAP2( retval = cv::contourArea(contour, oriented));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_convertMaps(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::convertMaps(map1, map2, dstmap1, dstmap2, dstmap1type, nninterpolation));
        return Py_BuildValue("(NN)", pyopencv_from(dstmap1), pyopencv_from(dstmap2));
    }

    return NULL;
}

static PyObject* pyopencv_convertPointsFromHomogeneous(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::convertPointsFromHomogeneous(src, dst));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_convertPointsToHomogeneous(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::convertPointsToHomogeneous(src, dst));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_convertScaleAbs(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::convertScaleAbs(src, dst, alpha, beta));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_convexHull(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::convexHull(points, hull, clockwise, returnPoints));
        return pyopencv_from(hull);
    }

    return NULL;
}

static PyObject* pyopencv_convexityDefects(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::convexityDefects(contour, convexhull, convexityDefects));
        return pyopencv_from(convexityDefects);
    }

    return NULL;
}

static PyObject* pyopencv_copyMakeBorder(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::copyMakeBorder(src, dst, top, bottom, left, right, borderType, value));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_cornerEigenValsAndVecs(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::cornerEigenValsAndVecs(src, dst, blockSize, ksize, borderType));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_cornerHarris(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::cornerHarris(src, dst, blockSize, ksize, k, borderType));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_cornerMinEigenVal(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::cornerMinEigenVal(src, dst, blockSize, ksize, borderType));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_cornerSubPix(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::cornerSubPix(image, corners, winSize, zeroZone, criteria));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_correctMatches(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::correctMatches(F, points1, points2, newPoints1, newPoints2));
        return Py_BuildValue("(NN)", pyopencv_from(newPoints1), pyopencv_from(newPoints2));
    }

    return NULL;
}

static PyObject* pyopencv_countNonZero(PyObject* , PyObject* args, PyObject* kw)
{
    int retval;
    PyObject* pyobj_src = NULL;
    Mat src;

    const char* keywords[] = { "src", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:countNonZero", (char**)keywords, &pyobj_src) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) )
    {
        ERRWRAP2( retval = cv::countNonZero(src));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_createEigenFaceRecognizer(PyObject* , PyObject* args, PyObject* kw)
{
    Ptr_FaceRecognizer retval;
    int num_components=0;
    double threshold=DBL_MAX;

    const char* keywords[] = { "num_components", "threshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|id:createEigenFaceRecognizer", (char**)keywords, &num_components, &threshold) )
    {
        ERRWRAP2( retval = cv::createEigenFaceRecognizer(num_components, threshold));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_createFisherFaceRecognizer(PyObject* , PyObject* args, PyObject* kw)
{
    Ptr_FaceRecognizer retval;
    int num_components=0;
    double threshold=DBL_MAX;

    const char* keywords[] = { "num_components", "threshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|id:createFisherFaceRecognizer", (char**)keywords, &num_components, &threshold) )
    {
        ERRWRAP2( retval = cv::createFisherFaceRecognizer(num_components, threshold));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_createHanningWindow(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::createHanningWindow(dst, winSize, type));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_createLBPHFaceRecognizer(PyObject* , PyObject* args, PyObject* kw)
{
    Ptr_FaceRecognizer retval;
    int radius=1;
    int neighbors=8;
    int grid_x=8;
    int grid_y=8;
    double threshold=DBL_MAX;

    const char* keywords[] = { "radius", "neighbors", "grid_x", "grid_y", "threshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|iiiid:createLBPHFaceRecognizer", (char**)keywords, &radius, &neighbors, &grid_x, &grid_y, &threshold) )
    {
        ERRWRAP2( retval = cv::createLBPHFaceRecognizer(radius, neighbors, grid_x, grid_y, threshold));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cubeRoot(PyObject* , PyObject* args, PyObject* kw)
{
    float retval;
    float val=0.f;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:cubeRoot", (char**)keywords, &val) )
    {
        ERRWRAP2( retval = cv::cubeRoot(val));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cvtColor(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::cvtColor(src, dst, code, dstCn));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_dct(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::dct(src, dst, flags));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_decomposeProjectionMatrix(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::decomposeProjectionMatrix(projMatrix, cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles));
        return Py_BuildValue("(NNNNNNN)", pyopencv_from(cameraMatrix), pyopencv_from(rotMatrix), pyopencv_from(transVect), pyopencv_from(rotMatrixX), pyopencv_from(rotMatrixY), pyopencv_from(rotMatrixZ), pyopencv_from(eulerAngles));
    }

    return NULL;
}

static PyObject* pyopencv_destroyAllWindows(PyObject* , PyObject* args, PyObject* kw)
{

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2( cv::destroyAllWindows());
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_destroyWindow(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_winname = NULL;
    string winname;

    const char* keywords[] = { "winname", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:destroyWindow", (char**)keywords, &pyobj_winname) &&
        pyopencv_to(pyobj_winname, winname, ArgInfo("winname", 0)) )
    {
        ERRWRAP2( cv::destroyWindow(winname));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_determinant(PyObject* , PyObject* args, PyObject* kw)
{
    double retval;
    PyObject* pyobj_mtx = NULL;
    Mat mtx;

    const char* keywords[] = { "mtx", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:determinant", (char**)keywords, &pyobj_mtx) &&
        pyopencv_to(pyobj_mtx, mtx, ArgInfo("mtx", 0)) )
    {
        ERRWRAP2( retval = cv::determinant(mtx));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_dft(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::dft(src, dst, flags, nonzeroRows));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_dilate(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::dilate(src, dst, kernel, anchor, iterations, borderType, borderValue));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_distanceTransform(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int distanceType=0;
    int maskSize=0;

    const char* keywords[] = { "src", "distanceType", "maskSize", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii|O:distanceTransform", (char**)keywords, &pyobj_src, &distanceType, &maskSize, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2( cv::distanceTransform(src, dst, distanceType, maskSize));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_distanceTransformWithLabels(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::distanceTransform(src, dst, labels, distanceType, maskSize, labelType));
        return Py_BuildValue("(NN)", pyopencv_from(dst), pyopencv_from(labels));
    }

    return NULL;
}

static PyObject* pyopencv_divide(PyObject* , PyObject* args, PyObject* kw)
{
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
        ERRWRAP2( cv::divide(src1, src2, dst, scale, dtype));
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
        ERRWRAP2( cv::divide(scale, src2, dst, dtype));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_drawChessboardCorners(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::drawChessboardCorners(image, patternSize, corners, patternWasFound));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_drawContours(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_contours = NULL;
    vector_Mat contours;
    int contourIdx=0;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int thickness=1;
    int lineType=8;
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
        ERRWRAP2( cv::drawContours(image, contours, contourIdx, color, thickness, lineType, hierarchy, maxLevel, offset));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_drawDataMatrixCodes(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_codes = NULL;
    vector_string codes;
    PyObject* pyobj_corners = NULL;
    Mat corners;

    const char* keywords[] = { "image", "codes", "corners", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:drawDataMatrixCodes", (char**)keywords, &pyobj_image, &pyobj_codes, &pyobj_corners) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 1)) &&
        pyopencv_to(pyobj_codes, codes, ArgInfo("codes", 0)) &&
        pyopencv_to(pyobj_corners, corners, ArgInfo("corners", 0)) )
    {
        ERRWRAP2( cv::drawDataMatrixCodes(image, codes, corners));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_drawKeypoints(PyObject* , PyObject* args, PyObject* kw)
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
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOi:drawKeypoints", (char**)keywords, &pyobj_image, &pyobj_keypoints, &pyobj_outImage, &pyobj_color, &flags) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_keypoints, keypoints, ArgInfo("keypoints", 0)) &&
        pyopencv_to(pyobj_outImage, outImage, ArgInfo("outImage", 1)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2( cv::drawKeypoints(image, keypoints, outImage, color, flags));
        return pyopencv_from(outImage);
    }

    return NULL;
}

static PyObject* pyopencv_eigen(PyObject* , PyObject* args, PyObject* kw)
{
    bool retval;
    PyObject* pyobj_src = NULL;
    Mat src;
    bool computeEigenvectors=0;
    PyObject* pyobj_eigenvalues = NULL;
    Mat eigenvalues;
    PyObject* pyobj_eigenvectors = NULL;
    Mat eigenvectors;

    const char* keywords[] = { "src", "computeEigenvectors", "eigenvalues", "eigenvectors", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Ob|OO:eigen", (char**)keywords, &pyobj_src, &computeEigenvectors, &pyobj_eigenvalues, &pyobj_eigenvectors) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_eigenvalues, eigenvalues, ArgInfo("eigenvalues", 1)) &&
        pyopencv_to(pyobj_eigenvectors, eigenvectors, ArgInfo("eigenvectors", 1)) )
    {
        ERRWRAP2( retval = cv::eigen(src, computeEigenvectors, eigenvalues, eigenvectors));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(eigenvalues), pyopencv_from(eigenvectors));
    }

    return NULL;
}

static PyObject* pyopencv_ellipse(PyObject* , PyObject* args, PyObject* kw)
{
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
    int lineType=8;
    int shift=0;

    const char* keywords[] = { "img", "center", "axes", "angle", "startAngle", "endAngle", "color", "thickness", "lineType", "shift", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOdddO|iii:ellipse", (char**)keywords, &pyobj_img, &pyobj_center, &pyobj_axes, &angle, &startAngle, &endAngle, &pyobj_color, &thickness, &lineType, &shift) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_center, center, ArgInfo("center", 0)) &&
        pyopencv_to(pyobj_axes, axes, ArgInfo("axes", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2( cv::ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness, lineType, shift));
        Py_RETURN_NONE;
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
    int lineType=8;

    const char* keywords[] = { "img", "box", "color", "thickness", "lineType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|ii:ellipse", (char**)keywords, &pyobj_img, &pyobj_box, &pyobj_color, &thickness, &lineType) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_box, box, ArgInfo("box", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2( cv::ellipse(img, box, color, thickness, lineType));
        Py_RETURN_NONE;
    }
    }

    return NULL;
}

static PyObject* pyopencv_ellipse2Poly(PyObject* , PyObject* args, PyObject* kw)
{
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
        ERRWRAP2( cv::ellipse2Poly(center, axes, angle, arcStart, arcEnd, delta, pts));
        return pyopencv_from(pts);
    }

    return NULL;
}

static PyObject* pyopencv_equalizeHist(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::equalizeHist(src, dst));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_erode(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::erode(src, dst, kernel, anchor, iterations, borderType, borderValue));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_estimateAffine3D(PyObject* , PyObject* args, PyObject* kw)
{
    int retval;
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

    const char* keywords[] = { "src", "dst", "out", "inliers", "ransacThreshold", "confidence", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOdd:estimateAffine3D", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_out, &pyobj_inliers, &ransacThreshold, &confidence) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 0)) &&
        pyopencv_to(pyobj_out, out, ArgInfo("out", 1)) &&
        pyopencv_to(pyobj_inliers, inliers, ArgInfo("inliers", 1)) )
    {
        ERRWRAP2( retval = cv::estimateAffine3D(src, dst, out, inliers, ransacThreshold, confidence));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(out), pyopencv_from(inliers));
    }

    return NULL;
}

static PyObject* pyopencv_estimateRigidTransform(PyObject* , PyObject* args, PyObject* kw)
{
    Mat retval;
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    bool fullAffine=0;

    const char* keywords[] = { "src", "dst", "fullAffine", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOb:estimateRigidTransform", (char**)keywords, &pyobj_src, &pyobj_dst, &fullAffine) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 0)) )
    {
        ERRWRAP2( retval = cv::estimateRigidTransform(src, dst, fullAffine));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_exp(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::exp(src, dst));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_extractChannel(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::extractChannel(src, dst, coi));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_fastAtan2(PyObject* , PyObject* args, PyObject* kw)
{
    float retval;
    float y=0.f;
    float x=0.f;

    const char* keywords[] = { "y", "x", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "ff:fastAtan2", (char**)keywords, &y, &x) )
    {
        ERRWRAP2( retval = cv::fastAtan2(y, x));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_fastNlMeansDenoising(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::fastNlMeansDenoising(src, dst, h, templateWindowSize, searchWindowSize));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_fastNlMeansDenoisingColored(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::fastNlMeansDenoisingColored(src, dst, h, hColor, templateWindowSize, searchWindowSize));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_fastNlMeansDenoisingColoredMulti(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::fastNlMeansDenoisingColoredMulti(srcImgs, dst, imgToDenoiseIndex, temporalWindowSize, h, hColor, templateWindowSize, searchWindowSize));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_fastNlMeansDenoisingMulti(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::fastNlMeansDenoisingMulti(srcImgs, dst, imgToDenoiseIndex, temporalWindowSize, h, templateWindowSize, searchWindowSize));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_fillConvexPoly(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_points = NULL;
    Mat points;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int lineType=8;
    int shift=0;

    const char* keywords[] = { "img", "points", "color", "lineType", "shift", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|ii:fillConvexPoly", (char**)keywords, &pyobj_img, &pyobj_points, &pyobj_color, &lineType, &shift) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2( cv::fillConvexPoly(img, points, color, lineType, shift));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_fillPoly(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_pts = NULL;
    vector_Mat pts;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int lineType=8;
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
        ERRWRAP2( cv::fillPoly(img, pts, color, lineType, shift, offset));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_filter2D(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::filter2D(src, dst, ddepth, kernel, anchor, delta, borderType));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_filterSpeckles(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::filterSpeckles(img, newVal, maxSpeckleSize, maxDiff, buf));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_findChessboardCorners(PyObject* , PyObject* args, PyObject* kw)
{
    bool retval;
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_patternSize = NULL;
    Size patternSize;
    PyObject* pyobj_corners = NULL;
    Mat corners;
    int flags=CALIB_CB_ADAPTIVE_THRESH+CALIB_CB_NORMALIZE_IMAGE;

    const char* keywords[] = { "image", "patternSize", "corners", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Oi:findChessboardCorners", (char**)keywords, &pyobj_image, &pyobj_patternSize, &pyobj_corners, &flags) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_patternSize, patternSize, ArgInfo("patternSize", 0)) &&
        pyopencv_to(pyobj_corners, corners, ArgInfo("corners", 1)) )
    {
        ERRWRAP2( retval = cv::findChessboardCorners(image, patternSize, corners, flags));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(corners));
    }

    return NULL;
}

static PyObject* pyopencv_findCirclesGrid(PyObject* , PyObject* args, PyObject* kw)
{
    bool retval;
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_patternSize = NULL;
    Size patternSize;
    PyObject* pyobj_centers = NULL;
    Mat centers;
    int flags=CALIB_CB_SYMMETRIC_GRID;
    PyObject* pyobj_blobDetector = NULL;
    Ptr_FeatureDetector blobDetector=new SimpleBlobDetector();

    const char* keywords[] = { "image", "patternSize", "centers", "flags", "blobDetector", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OiO:findCirclesGrid", (char**)keywords, &pyobj_image, &pyobj_patternSize, &pyobj_centers, &flags, &pyobj_blobDetector) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_patternSize, patternSize, ArgInfo("patternSize", 0)) &&
        pyopencv_to(pyobj_centers, centers, ArgInfo("centers", 1)) &&
        pyopencv_to(pyobj_blobDetector, blobDetector, ArgInfo("blobDetector", 0)) )
    {
        ERRWRAP2( retval = cv::findCirclesGrid(image, patternSize, centers, flags, blobDetector));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(centers));
    }

    return NULL;
}

static PyObject* pyopencv_findCirclesGridDefault(PyObject* , PyObject* args, PyObject* kw)
{
    bool retval;
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_patternSize = NULL;
    Size patternSize;
    PyObject* pyobj_centers = NULL;
    Mat centers;
    int flags=CALIB_CB_SYMMETRIC_GRID;

    const char* keywords[] = { "image", "patternSize", "centers", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Oi:findCirclesGridDefault", (char**)keywords, &pyobj_image, &pyobj_patternSize, &pyobj_centers, &flags) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_patternSize, patternSize, ArgInfo("patternSize", 0)) &&
        pyopencv_to(pyobj_centers, centers, ArgInfo("centers", 1)) )
    {
        ERRWRAP2( retval = cv::findCirclesGridDefault(image, patternSize, centers, flags));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(centers));
    }

    return NULL;
}

static PyObject* pyopencv_findContours(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::findContours(image, contours, hierarchy, mode, method, offset));
        return Py_BuildValue("(NN)", pyopencv_from(contours), pyopencv_from(hierarchy));
    }

    return NULL;
}

static PyObject* pyopencv_findDataMatrix(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_image = NULL;
    Mat image;
    vector_string codes;
    PyObject* pyobj_corners = NULL;
    Mat corners;
    PyObject* pyobj_dmtx = NULL;
    vector_Mat dmtx;

    const char* keywords[] = { "image", "corners", "dmtx", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OO:findDataMatrix", (char**)keywords, &pyobj_image, &pyobj_corners, &pyobj_dmtx) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_corners, corners, ArgInfo("corners", 1)) &&
        pyopencv_to(pyobj_dmtx, dmtx, ArgInfo("dmtx", 1)) )
    {
        ERRWRAP2( cv::findDataMatrix(image, codes, corners, dmtx));
        return Py_BuildValue("(NNN)", pyopencv_from(codes), pyopencv_from(corners), pyopencv_from(dmtx));
    }

    return NULL;
}

static PyObject* pyopencv_findFundamentalMat(PyObject* , PyObject* args, PyObject* kw)
{
    Mat retval;
    PyObject* pyobj_points1 = NULL;
    Mat points1;
    PyObject* pyobj_points2 = NULL;
    Mat points2;
    int method=FM_RANSAC;
    double param1=3.;
    double param2=0.99;
    PyObject* pyobj_mask = NULL;
    Mat mask;

    const char* keywords[] = { "points1", "points2", "method", "param1", "param2", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|iddO:findFundamentalMat", (char**)keywords, &pyobj_points1, &pyobj_points2, &method, &param1, &param2, &pyobj_mask) &&
        pyopencv_to(pyobj_points1, points1, ArgInfo("points1", 0)) &&
        pyopencv_to(pyobj_points2, points2, ArgInfo("points2", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 1)) )
    {
        ERRWRAP2( retval = cv::findFundamentalMat(points1, points2, method, param1, param2, mask));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(mask));
    }

    return NULL;
}

static PyObject* pyopencv_findHomography(PyObject* , PyObject* args, PyObject* kw)
{
    Mat retval;
    PyObject* pyobj_srcPoints = NULL;
    Mat srcPoints;
    PyObject* pyobj_dstPoints = NULL;
    Mat dstPoints;
    int method=0;
    double ransacReprojThreshold=3;
    PyObject* pyobj_mask = NULL;
    Mat mask;

    const char* keywords[] = { "srcPoints", "dstPoints", "method", "ransacReprojThreshold", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|idO:findHomography", (char**)keywords, &pyobj_srcPoints, &pyobj_dstPoints, &method, &ransacReprojThreshold, &pyobj_mask) &&
        pyopencv_to(pyobj_srcPoints, srcPoints, ArgInfo("srcPoints", 0)) &&
        pyopencv_to(pyobj_dstPoints, dstPoints, ArgInfo("dstPoints", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 1)) )
    {
        ERRWRAP2( retval = cv::findHomography(srcPoints, dstPoints, method, ransacReprojThreshold, mask));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(mask));
    }

    return NULL;
}

static PyObject* pyopencv_findNonZero(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::findNonZero(src, idx));
        return pyopencv_from(idx);
    }

    return NULL;
}

static PyObject* pyopencv_fitEllipse(PyObject* , PyObject* args, PyObject* kw)
{
    RotatedRect retval;
    PyObject* pyobj_points = NULL;
    Mat points;

    const char* keywords[] = { "points", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:fitEllipse", (char**)keywords, &pyobj_points) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 0)) )
    {
        ERRWRAP2( retval = cv::fitEllipse(points));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_fitLine(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::fitLine(points, line, distType, param, reps, aeps));
        return pyopencv_from(line);
    }

    return NULL;
}

static PyObject* pyopencv_flip(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::flip(src, dst, flipCode));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_floodFill(PyObject* , PyObject* args, PyObject* kw)
{
    int retval;
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

    const char* keywords[] = { "image", "mask", "seedPoint", "newVal", "loDiff", "upDiff", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OOi:floodFill", (char**)keywords, &pyobj_image, &pyobj_mask, &pyobj_seedPoint, &pyobj_newVal, &pyobj_loDiff, &pyobj_upDiff, &flags) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 1)) &&
        pyopencv_to(pyobj_seedPoint, seedPoint, ArgInfo("seedPoint", 0)) &&
        pyopencv_to(pyobj_newVal, newVal, ArgInfo("newVal", 0)) &&
        pyopencv_to(pyobj_loDiff, loDiff, ArgInfo("loDiff", 0)) &&
        pyopencv_to(pyobj_upDiff, upDiff, ArgInfo("upDiff", 0)) )
    {
        ERRWRAP2( retval = cv::floodFill(image, mask, seedPoint, newVal, &rect, loDiff, upDiff, flags));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(rect));
    }

    return NULL;
}

static PyObject* pyopencv_gemm(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    PyObject* pyobj_src2 = NULL;
    Mat src2;
    double alpha=0;
    PyObject* pyobj_src3 = NULL;
    Mat src3;
    double gamma=0;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int flags=0;

    const char* keywords[] = { "src1", "src2", "alpha", "src3", "gamma", "dst", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOdOd|Oi:gemm", (char**)keywords, &pyobj_src1, &pyobj_src2, &alpha, &pyobj_src3, &gamma, &pyobj_dst, &flags) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_src3, src3, ArgInfo("src3", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2( cv::gemm(src1, src2, alpha, src3, gamma, dst, flags));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_getAffineTransform(PyObject* , PyObject* args, PyObject* kw)
{
    Mat retval;
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:getAffineTransform", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 0)) )
    {
        ERRWRAP2( retval = cv::getAffineTransform(src, dst));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_getBuildInformation(PyObject* , PyObject* args, PyObject* kw)
{
    string retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2( retval = cv::getBuildInformation());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_getCPUTickCount(PyObject* , PyObject* args, PyObject* kw)
{
    int64 retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2( retval = cv::getCPUTickCount());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_getDefaultNewCameraMatrix(PyObject* , PyObject* args, PyObject* kw)
{
    Mat retval;
    PyObject* pyobj_cameraMatrix = NULL;
    Mat cameraMatrix;
    PyObject* pyobj_imgsize = NULL;
    Size imgsize;
    bool centerPrincipalPoint=false;

    const char* keywords[] = { "cameraMatrix", "imgsize", "centerPrincipalPoint", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Ob:getDefaultNewCameraMatrix", (char**)keywords, &pyobj_cameraMatrix, &pyobj_imgsize, &centerPrincipalPoint) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_imgsize, imgsize, ArgInfo("imgsize", 0)) )
    {
        ERRWRAP2( retval = cv::getDefaultNewCameraMatrix(cameraMatrix, imgsize, centerPrincipalPoint));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_getDerivKernels(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::getDerivKernels(kx, ky, dx, dy, ksize, normalize, ktype));
        return Py_BuildValue("(NN)", pyopencv_from(kx), pyopencv_from(ky));
    }

    return NULL;
}

static PyObject* pyopencv_getGaborKernel(PyObject* , PyObject* args, PyObject* kw)
{
    Mat retval;
    PyObject* pyobj_ksize = NULL;
    Size ksize;
    double sigma=0;
    double theta=0;
    double lambd=0;
    double gamma=0;
    double psi=CV_PI*0.5;
    int ktype=CV_64F;

    const char* keywords[] = { "ksize", "sigma", "theta", "lambd", "gamma", "psi", "ktype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odddd|di:getGaborKernel", (char**)keywords, &pyobj_ksize, &sigma, &theta, &lambd, &gamma, &psi, &ktype) &&
        pyopencv_to(pyobj_ksize, ksize, ArgInfo("ksize", 0)) )
    {
        ERRWRAP2( retval = cv::getGaborKernel(ksize, sigma, theta, lambd, gamma, psi, ktype));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_getGaussianKernel(PyObject* , PyObject* args, PyObject* kw)
{
    Mat retval;
    int ksize=0;
    double sigma=0;
    int ktype=CV_64F;

    const char* keywords[] = { "ksize", "sigma", "ktype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "id|i:getGaussianKernel", (char**)keywords, &ksize, &sigma, &ktype) )
    {
        ERRWRAP2( retval = cv::getGaussianKernel(ksize, sigma, ktype));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_getNumberOfCPUs(PyObject* , PyObject* args, PyObject* kw)
{
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2( retval = cv::getNumberOfCPUs());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_getOptimalDFTSize(PyObject* , PyObject* args, PyObject* kw)
{
    int retval;
    int vecsize=0;

    const char* keywords[] = { "vecsize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:getOptimalDFTSize", (char**)keywords, &vecsize) )
    {
        ERRWRAP2( retval = cv::getOptimalDFTSize(vecsize));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_getOptimalNewCameraMatrix(PyObject* , PyObject* args, PyObject* kw)
{
    Mat retval;
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

    const char* keywords[] = { "cameraMatrix", "distCoeffs", "imageSize", "alpha", "newImgSize", "centerPrincipalPoint", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOd|Ob:getOptimalNewCameraMatrix", (char**)keywords, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_imageSize, &alpha, &pyobj_newImgSize, &centerPrincipalPoint) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_newImgSize, newImgSize, ArgInfo("newImgSize", 0)) )
    {
        ERRWRAP2( retval = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, alpha, newImgSize, &validPixROI, centerPrincipalPoint));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(validPixROI));
    }

    return NULL;
}

static PyObject* pyopencv_getPerspectiveTransform(PyObject* , PyObject* args, PyObject* kw)
{
    Mat retval;
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:getPerspectiveTransform", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 0)) )
    {
        ERRWRAP2( retval = cv::getPerspectiveTransform(src, dst));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_getRectSubPix(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::getRectSubPix(image, patchSize, center, patch, patchType));
        return pyopencv_from(patch);
    }

    return NULL;
}

static PyObject* pyopencv_getRotationMatrix2D(PyObject* , PyObject* args, PyObject* kw)
{
    Mat retval;
    PyObject* pyobj_center = NULL;
    Point2f center;
    double angle=0;
    double scale=0;

    const char* keywords[] = { "center", "angle", "scale", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odd:getRotationMatrix2D", (char**)keywords, &pyobj_center, &angle, &scale) &&
        pyopencv_to(pyobj_center, center, ArgInfo("center", 0)) )
    {
        ERRWRAP2( retval = cv::getRotationMatrix2D(center, angle, scale));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_getStructuringElement(PyObject* , PyObject* args, PyObject* kw)
{
    Mat retval;
    int shape=0;
    PyObject* pyobj_ksize = NULL;
    Size ksize;
    PyObject* pyobj_anchor = NULL;
    Point anchor=Point(-1,-1);

    const char* keywords[] = { "shape", "ksize", "anchor", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "iO|O:getStructuringElement", (char**)keywords, &shape, &pyobj_ksize, &pyobj_anchor) &&
        pyopencv_to(pyobj_ksize, ksize, ArgInfo("ksize", 0)) &&
        pyopencv_to(pyobj_anchor, anchor, ArgInfo("anchor", 0)) )
    {
        ERRWRAP2( retval = cv::getStructuringElement(shape, ksize, anchor));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_getTextSize(PyObject* , PyObject* args, PyObject* kw)
{
    Size retval;
    PyObject* pyobj_text = NULL;
    string text;
    int fontFace=0;
    double fontScale=0;
    int thickness=0;
    int baseLine;

    const char* keywords[] = { "text", "fontFace", "fontScale", "thickness", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oidi:getTextSize", (char**)keywords, &pyobj_text, &fontFace, &fontScale, &thickness) &&
        pyopencv_to(pyobj_text, text, ArgInfo("text", 0)) )
    {
        ERRWRAP2( retval = cv::getTextSize(text, fontFace, fontScale, thickness, &baseLine));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(baseLine));
    }

    return NULL;
}

static PyObject* pyopencv_getTickCount(PyObject* , PyObject* args, PyObject* kw)
{
    int64 retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2( retval = cv::getTickCount());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_getTickFrequency(PyObject* , PyObject* args, PyObject* kw)
{
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2( retval = cv::getTickFrequency());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_getTrackbarPos(PyObject* , PyObject* args, PyObject* kw)
{
    int retval;
    PyObject* pyobj_trackbarname = NULL;
    string trackbarname;
    PyObject* pyobj_winname = NULL;
    string winname;

    const char* keywords[] = { "trackbarname", "winname", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:getTrackbarPos", (char**)keywords, &pyobj_trackbarname, &pyobj_winname) &&
        pyopencv_to(pyobj_trackbarname, trackbarname, ArgInfo("trackbarname", 0)) &&
        pyopencv_to(pyobj_winname, winname, ArgInfo("winname", 0)) )
    {
        ERRWRAP2( retval = cv::getTrackbarPos(trackbarname, winname));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_getValidDisparityROI(PyObject* , PyObject* args, PyObject* kw)
{
    Rect retval;
    PyObject* pyobj_roi1 = NULL;
    Rect roi1;
    PyObject* pyobj_roi2 = NULL;
    Rect roi2;
    int minDisparity=0;
    int numberOfDisparities=0;
    int SADWindowSize=0;

    const char* keywords[] = { "roi1", "roi2", "minDisparity", "numberOfDisparities", "SADWindowSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOiii:getValidDisparityROI", (char**)keywords, &pyobj_roi1, &pyobj_roi2, &minDisparity, &numberOfDisparities, &SADWindowSize) &&
        pyopencv_to(pyobj_roi1, roi1, ArgInfo("roi1", 0)) &&
        pyopencv_to(pyobj_roi2, roi2, ArgInfo("roi2", 0)) )
    {
        ERRWRAP2( retval = cv::getValidDisparityROI(roi1, roi2, minDisparity, numberOfDisparities, SADWindowSize));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_getWindowProperty(PyObject* , PyObject* args, PyObject* kw)
{
    double retval;
    PyObject* pyobj_winname = NULL;
    string winname;
    int prop_id=0;

    const char* keywords[] = { "winname", "prop_id", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi:getWindowProperty", (char**)keywords, &pyobj_winname, &prop_id) &&
        pyopencv_to(pyobj_winname, winname, ArgInfo("winname", 0)) )
    {
        ERRWRAP2( retval = cv::getWindowProperty(winname, prop_id));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_goodFeaturesToTrack(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::goodFeaturesToTrack(image, corners, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k));
        return pyopencv_from(corners);
    }

    return NULL;
}

static PyObject* pyopencv_grabCut(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_groupRectangles(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_rectList = NULL;
    vector_Rect rectList;
    vector_int weights;
    int groupThreshold=0;
    double eps=0.2;

    const char* keywords[] = { "rectList", "groupThreshold", "eps", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|d:groupRectangles", (char**)keywords, &pyobj_rectList, &groupThreshold, &eps) &&
        pyopencv_to(pyobj_rectList, rectList, ArgInfo("rectList", 1)) )
    {
        ERRWRAP2( cv::groupRectangles(rectList, weights, groupThreshold, eps));
        return Py_BuildValue("(NN)", pyopencv_from(rectList), pyopencv_from(weights));
    }

    return NULL;
}

static PyObject* pyopencv_hconcat(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::hconcat(src, dst));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_idct(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::idct(src, dst, flags));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_idft(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::idft(src, dst, flags, nonzeroRows));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_imdecode(PyObject* , PyObject* args, PyObject* kw)
{
    Mat retval;
    PyObject* pyobj_buf = NULL;
    Mat buf;
    int flags=0;

    const char* keywords[] = { "buf", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi:imdecode", (char**)keywords, &pyobj_buf, &flags) &&
        pyopencv_to(pyobj_buf, buf, ArgInfo("buf", 0)) )
    {
        ERRWRAP2( retval = cv::imdecode(buf, flags));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_imencode(PyObject* , PyObject* args, PyObject* kw)
{
    bool retval;
    PyObject* pyobj_ext = NULL;
    string ext;
    PyObject* pyobj_img = NULL;
    Mat img;
    vector_uchar buf;
    PyObject* pyobj_params = NULL;
    vector_int params=vector<int>();

    const char* keywords[] = { "ext", "img", "params", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:imencode", (char**)keywords, &pyobj_ext, &pyobj_img, &pyobj_params) &&
        pyopencv_to(pyobj_ext, ext, ArgInfo("ext", 0)) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_params, params, ArgInfo("params", 0)) )
    {
        ERRWRAP2( retval = cv::imencode(ext, img, buf, params));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(buf));
    }

    return NULL;
}

static PyObject* pyopencv_imread(PyObject* , PyObject* args, PyObject* kw)
{
    Mat retval;
    PyObject* pyobj_filename = NULL;
    string filename;
    int flags=1;

    const char* keywords[] = { "filename", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|i:imread", (char**)keywords, &pyobj_filename, &flags) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) )
    {
        ERRWRAP2( retval = cv::imread(filename, flags));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_imshow(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_winname = NULL;
    string winname;
    PyObject* pyobj_mat = NULL;
    Mat mat;

    const char* keywords[] = { "winname", "mat", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:imshow", (char**)keywords, &pyobj_winname, &pyobj_mat) &&
        pyopencv_to(pyobj_winname, winname, ArgInfo("winname", 0)) &&
        pyopencv_to(pyobj_mat, mat, ArgInfo("mat", 0)) )
    {
        ERRWRAP2( cv::imshow(winname, mat));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_imwrite(PyObject* , PyObject* args, PyObject* kw)
{
    bool retval;
    PyObject* pyobj_filename = NULL;
    string filename;
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_params = NULL;
    vector_int params=vector<int>();

    const char* keywords[] = { "filename", "img", "params", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:imwrite", (char**)keywords, &pyobj_filename, &pyobj_img, &pyobj_params) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_params, params, ArgInfo("params", 0)) )
    {
        ERRWRAP2( retval = cv::imwrite(filename, img, params));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_inRange(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::inRange(src, lowerb, upperb, dst));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_initCameraMatrix2D(PyObject* , PyObject* args, PyObject* kw)
{
    Mat retval;
    PyObject* pyobj_objectPoints = NULL;
    vector_Mat objectPoints;
    PyObject* pyobj_imagePoints = NULL;
    vector_Mat imagePoints;
    PyObject* pyobj_imageSize = NULL;
    Size imageSize;
    double aspectRatio=1.;

    const char* keywords[] = { "objectPoints", "imagePoints", "imageSize", "aspectRatio", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|d:initCameraMatrix2D", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints, &pyobj_imageSize, &aspectRatio) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints, imagePoints, ArgInfo("imagePoints", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) )
    {
        ERRWRAP2( retval = cv::initCameraMatrix2D(objectPoints, imagePoints, imageSize, aspectRatio));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_initUndistortRectifyMap(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type, map1, map2));
        return Py_BuildValue("(NN)", pyopencv_from(map1), pyopencv_from(map2));
    }

    return NULL;
}

static PyObject* pyopencv_initWideAngleProjMap(PyObject* , PyObject* args, PyObject* kw)
{
    float retval;
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

    const char* keywords[] = { "cameraMatrix", "distCoeffs", "imageSize", "destImageWidth", "m1type", "map1", "map2", "projType", "alpha", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOii|OOid:initWideAngleProjMap", (char**)keywords, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_imageSize, &destImageWidth, &m1type, &pyobj_map1, &pyobj_map2, &projType, &alpha) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_imageSize, imageSize, ArgInfo("imageSize", 0)) &&
        pyopencv_to(pyobj_map1, map1, ArgInfo("map1", 1)) &&
        pyopencv_to(pyobj_map2, map2, ArgInfo("map2", 1)) )
    {
        ERRWRAP2( retval = cv::initWideAngleProjMap(cameraMatrix, distCoeffs, imageSize, destImageWidth, m1type, map1, map2, projType, alpha));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(map1), pyopencv_from(map2));
    }

    return NULL;
}

static PyObject* pyopencv_inpaint(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::inpaint(src, inpaintMask, dst, inpaintRadius, flags));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_insertChannel(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::insertChannel(src, dst, coi));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_integral(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::integral(src, sum, sdepth));
        return pyopencv_from(sum);
    }

    return NULL;
}

static PyObject* pyopencv_integral2(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_sum = NULL;
    Mat sum;
    PyObject* pyobj_sqsum = NULL;
    Mat sqsum;
    int sdepth=-1;

    const char* keywords[] = { "src", "sum", "sqsum", "sdepth", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOi:integral2", (char**)keywords, &pyobj_src, &pyobj_sum, &pyobj_sqsum, &sdepth) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_sum, sum, ArgInfo("sum", 1)) &&
        pyopencv_to(pyobj_sqsum, sqsum, ArgInfo("sqsum", 1)) )
    {
        ERRWRAP2( cv::integral(src, sum, sqsum, sdepth));
        return Py_BuildValue("(NN)", pyopencv_from(sum), pyopencv_from(sqsum));
    }

    return NULL;
}

static PyObject* pyopencv_integral3(PyObject* , PyObject* args, PyObject* kw)
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

    const char* keywords[] = { "src", "sum", "sqsum", "tilted", "sdepth", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOOi:integral3", (char**)keywords, &pyobj_src, &pyobj_sum, &pyobj_sqsum, &pyobj_tilted, &sdepth) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_sum, sum, ArgInfo("sum", 1)) &&
        pyopencv_to(pyobj_sqsum, sqsum, ArgInfo("sqsum", 1)) &&
        pyopencv_to(pyobj_tilted, tilted, ArgInfo("tilted", 1)) )
    {
        ERRWRAP2( cv::integral(src, sum, sqsum, tilted, sdepth));
        return Py_BuildValue("(NNN)", pyopencv_from(sum), pyopencv_from(sqsum), pyopencv_from(tilted));
    }

    return NULL;
}

static PyObject* pyopencv_intersectConvexConvex(PyObject* , PyObject* args, PyObject* kw)
{
    float retval;
    PyObject* pyobj__p1 = NULL;
    Mat _p1;
    PyObject* pyobj__p2 = NULL;
    Mat _p2;
    PyObject* pyobj__p12 = NULL;
    Mat _p12;
    bool handleNested=true;

    const char* keywords[] = { "_p1", "_p2", "_p12", "handleNested", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Ob:intersectConvexConvex", (char**)keywords, &pyobj__p1, &pyobj__p2, &pyobj__p12, &handleNested) &&
        pyopencv_to(pyobj__p1, _p1, ArgInfo("_p1", 0)) &&
        pyopencv_to(pyobj__p2, _p2, ArgInfo("_p2", 0)) &&
        pyopencv_to(pyobj__p12, _p12, ArgInfo("_p12", 1)) )
    {
        ERRWRAP2( retval = cv::intersectConvexConvex(_p1, _p2, _p12, handleNested));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(_p12));
    }

    return NULL;
}

static PyObject* pyopencv_invert(PyObject* , PyObject* args, PyObject* kw)
{
    double retval;
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int flags=DECOMP_LU;

    const char* keywords[] = { "src", "dst", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oi:invert", (char**)keywords, &pyobj_src, &pyobj_dst, &flags) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2( retval = cv::invert(src, dst, flags));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(dst));
    }

    return NULL;
}

static PyObject* pyopencv_invertAffineTransform(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::invertAffineTransform(M, iM));
        return pyopencv_from(iM);
    }

    return NULL;
}

static PyObject* pyopencv_isContourConvex(PyObject* , PyObject* args, PyObject* kw)
{
    bool retval;
    PyObject* pyobj_contour = NULL;
    Mat contour;

    const char* keywords[] = { "contour", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:isContourConvex", (char**)keywords, &pyobj_contour) &&
        pyopencv_to(pyobj_contour, contour, ArgInfo("contour", 0)) )
    {
        ERRWRAP2( retval = cv::isContourConvex(contour));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_kmeans(PyObject* , PyObject* args, PyObject* kw)
{
    double retval;
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

    const char* keywords[] = { "data", "K", "criteria", "attempts", "flags", "bestLabels", "centers", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OiOii|OO:kmeans", (char**)keywords, &pyobj_data, &K, &pyobj_criteria, &attempts, &flags, &pyobj_bestLabels, &pyobj_centers) &&
        pyopencv_to(pyobj_data, data, ArgInfo("data", 0)) &&
        pyopencv_to(pyobj_bestLabels, bestLabels, ArgInfo("bestLabels", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) &&
        pyopencv_to(pyobj_centers, centers, ArgInfo("centers", 1)) )
    {
        ERRWRAP2( retval = cv::kmeans(data, K, bestLabels, criteria, attempts, flags, centers));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(bestLabels), pyopencv_from(centers));
    }

    return NULL;
}

static PyObject* pyopencv_line(PyObject* , PyObject* args, PyObject* kw)
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
    int lineType=8;
    int shift=0;

    const char* keywords[] = { "img", "pt1", "pt2", "color", "thickness", "lineType", "shift", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|iii:line", (char**)keywords, &pyobj_img, &pyobj_pt1, &pyobj_pt2, &pyobj_color, &thickness, &lineType, &shift) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_pt1, pt1, ArgInfo("pt1", 0)) &&
        pyopencv_to(pyobj_pt2, pt2, ArgInfo("pt2", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2( cv::line(img, pt1, pt2, color, thickness, lineType, shift));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_log(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::log(src, dst));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_magnitude(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::magnitude(x, y, magnitude));
        return pyopencv_from(magnitude);
    }

    return NULL;
}

static PyObject* pyopencv_matMulDeriv(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::matMulDeriv(A, B, dABdA, dABdB));
        return Py_BuildValue("(NN)", pyopencv_from(dABdA), pyopencv_from(dABdB));
    }

    return NULL;
}

static PyObject* pyopencv_matchShapes(PyObject* , PyObject* args, PyObject* kw)
{
    double retval;
    PyObject* pyobj_contour1 = NULL;
    Mat contour1;
    PyObject* pyobj_contour2 = NULL;
    Mat contour2;
    int method=0;
    double parameter=0;

    const char* keywords[] = { "contour1", "contour2", "method", "parameter", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOid:matchShapes", (char**)keywords, &pyobj_contour1, &pyobj_contour2, &method, &parameter) &&
        pyopencv_to(pyobj_contour1, contour1, ArgInfo("contour1", 0)) &&
        pyopencv_to(pyobj_contour2, contour2, ArgInfo("contour2", 0)) )
    {
        ERRWRAP2( retval = cv::matchShapes(contour1, contour2, method, parameter));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_matchTemplate(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_templ = NULL;
    Mat templ;
    PyObject* pyobj_result = NULL;
    Mat result;
    int method=0;

    const char* keywords[] = { "image", "templ", "method", "result", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi|O:matchTemplate", (char**)keywords, &pyobj_image, &pyobj_templ, &method, &pyobj_result) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_templ, templ, ArgInfo("templ", 0)) &&
        pyopencv_to(pyobj_result, result, ArgInfo("result", 1)) )
    {
        ERRWRAP2( cv::matchTemplate(image, templ, result, method));
        return pyopencv_from(result);
    }

    return NULL;
}

static PyObject* pyopencv_max(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::max(src1, src2, dst));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_mean(PyObject* , PyObject* args, PyObject* kw)
{
    Scalar retval;
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_mask = NULL;
    Mat mask;

    const char* keywords[] = { "src", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:mean", (char**)keywords, &pyobj_src, &pyobj_mask) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2( retval = cv::mean(src, mask));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_meanShift(PyObject* , PyObject* args, PyObject* kw)
{
    int retval;
    PyObject* pyobj_probImage = NULL;
    Mat probImage;
    PyObject* pyobj_window = NULL;
    Rect window;
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria;

    const char* keywords[] = { "probImage", "window", "criteria", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:meanShift", (char**)keywords, &pyobj_probImage, &pyobj_window, &pyobj_criteria) &&
        pyopencv_to(pyobj_probImage, probImage, ArgInfo("probImage", 0)) &&
        pyopencv_to(pyobj_window, window, ArgInfo("window", 1)) &&
        pyopencv_to(pyobj_criteria, criteria, ArgInfo("criteria", 0)) )
    {
        ERRWRAP2( retval = cv::meanShift(probImage, window, criteria));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(window));
    }

    return NULL;
}

static PyObject* pyopencv_meanStdDev(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::meanStdDev(src, mean, stddev, mask));
        return Py_BuildValue("(NN)", pyopencv_from(mean), pyopencv_from(stddev));
    }

    return NULL;
}

static PyObject* pyopencv_medianBlur(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::medianBlur(src, dst, ksize));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_merge(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::merge(mv, dst));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_min(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::min(src1, src2, dst));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_minAreaRect(PyObject* , PyObject* args, PyObject* kw)
{
    RotatedRect retval;
    PyObject* pyobj_points = NULL;
    Mat points;

    const char* keywords[] = { "points", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:minAreaRect", (char**)keywords, &pyobj_points) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 0)) )
    {
        ERRWRAP2( retval = cv::minAreaRect(points));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_minEnclosingCircle(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_points = NULL;
    Mat points;
    Point2f center;
    float radius;

    const char* keywords[] = { "points", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:minEnclosingCircle", (char**)keywords, &pyobj_points) &&
        pyopencv_to(pyobj_points, points, ArgInfo("points", 0)) )
    {
        ERRWRAP2( cv::minEnclosingCircle(points, center, radius));
        return Py_BuildValue("(NN)", pyopencv_from(center), pyopencv_from(radius));
    }

    return NULL;
}

static PyObject* pyopencv_minMaxLoc(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc, mask));
        return Py_BuildValue("(NNNN)", pyopencv_from(minVal), pyopencv_from(maxVal), pyopencv_from(minLoc), pyopencv_from(maxLoc));
    }

    return NULL;
}

static PyObject* pyopencv_mixChannels(PyObject* , PyObject* args, PyObject* kw)
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
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 0)) &&
        pyopencv_to(pyobj_fromTo, fromTo, ArgInfo("fromTo", 0)) )
    {
        ERRWRAP2( cv::mixChannels(src, dst, fromTo));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_moments(PyObject* , PyObject* args, PyObject* kw)
{
    Moments retval;
    PyObject* pyobj_array = NULL;
    Mat array;
    bool binaryImage=false;

    const char* keywords[] = { "array", "binaryImage", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|b:moments", (char**)keywords, &pyobj_array, &binaryImage) &&
        pyopencv_to(pyobj_array, array, ArgInfo("array", 0)) )
    {
        ERRWRAP2( retval = cv::moments(array, binaryImage));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_morphologyEx(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::morphologyEx(src, dst, op, kernel, anchor, iterations, borderType, borderValue));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_moveWindow(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_winname = NULL;
    string winname;
    int x=0;
    int y=0;

    const char* keywords[] = { "winname", "x", "y", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii:moveWindow", (char**)keywords, &pyobj_winname, &x, &y) &&
        pyopencv_to(pyobj_winname, winname, ArgInfo("winname", 0)) )
    {
        ERRWRAP2( cv::moveWindow(winname, x, y));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_mulSpectrums(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::mulSpectrums(a, b, c, flags, conjB));
        return pyopencv_from(c);
    }

    return NULL;
}

static PyObject* pyopencv_mulTransposed(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::mulTransposed(src, dst, aTa, delta, scale, dtype));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_multiply(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::multiply(src1, src2, dst, scale, dtype));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_namedWindow(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_winname = NULL;
    string winname;
    int flags=WINDOW_AUTOSIZE;

    const char* keywords[] = { "winname", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|i:namedWindow", (char**)keywords, &pyobj_winname, &flags) &&
        pyopencv_to(pyobj_winname, winname, ArgInfo("winname", 0)) )
    {
        ERRWRAP2( cv::namedWindow(winname, flags));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_norm(PyObject* , PyObject* args, PyObject* kw)
{
    {
    double retval;
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    int normType=NORM_L2;
    PyObject* pyobj_mask = NULL;
    Mat mask;

    const char* keywords[] = { "src1", "normType", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|iO:norm", (char**)keywords, &pyobj_src1, &normType, &pyobj_mask) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2( retval = cv::norm(src1, normType, mask));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    double retval;
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    PyObject* pyobj_src2 = NULL;
    Mat src2;
    int normType=NORM_L2;
    PyObject* pyobj_mask = NULL;
    Mat mask;

    const char* keywords[] = { "src1", "src2", "normType", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|iO:norm", (char**)keywords, &pyobj_src1, &pyobj_src2, &normType, &pyobj_mask) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2( retval = cv::norm(src1, src2, normType, mask));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_normalize(PyObject* , PyObject* args, PyObject* kw)
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
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OddiiO:normalize", (char**)keywords, &pyobj_src, &pyobj_dst, &alpha, &beta, &norm_type, &dtype, &pyobj_mask) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2( cv::normalize(src, dst, alpha, beta, norm_type, dtype, mask));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_patchNaNs(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_a = NULL;
    Mat a;
    double val=0;

    const char* keywords[] = { "a", "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|d:patchNaNs", (char**)keywords, &pyobj_a, &val) &&
        pyopencv_to(pyobj_a, a, ArgInfo("a", 1)) )
    {
        ERRWRAP2( cv::patchNaNs(a, val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_perspectiveTransform(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::perspectiveTransform(src, dst, m));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_phase(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::phase(x, y, angle, angleInDegrees));
        return pyopencv_from(angle);
    }

    return NULL;
}

static PyObject* pyopencv_phaseCorrelate(PyObject* , PyObject* args, PyObject* kw)
{
    Point2d retval;
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    PyObject* pyobj_src2 = NULL;
    Mat src2;
    PyObject* pyobj_window = NULL;
    Mat window;
    double response;

    const char* keywords[] = { "src1", "src2", "window", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:phaseCorrelate", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_window) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_window, window, ArgInfo("window", 0)) )
    {
        ERRWRAP2( retval = cv::phaseCorrelate(src1, src2, window, &response));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(response));
    }

    return NULL;
}

static PyObject* pyopencv_pointPolygonTest(PyObject* , PyObject* args, PyObject* kw)
{
    double retval;
    PyObject* pyobj_contour = NULL;
    Mat contour;
    PyObject* pyobj_pt = NULL;
    Point2f pt;
    bool measureDist=0;

    const char* keywords[] = { "contour", "pt", "measureDist", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOb:pointPolygonTest", (char**)keywords, &pyobj_contour, &pyobj_pt, &measureDist) &&
        pyopencv_to(pyobj_contour, contour, ArgInfo("contour", 0)) &&
        pyopencv_to(pyobj_pt, pt, ArgInfo("pt", 0)) )
    {
        ERRWRAP2( retval = cv::pointPolygonTest(contour, pt, measureDist));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_polarToCart(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::polarToCart(magnitude, angle, x, y, angleInDegrees));
        return Py_BuildValue("(NN)", pyopencv_from(x), pyopencv_from(y));
    }

    return NULL;
}

static PyObject* pyopencv_polylines(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_pts = NULL;
    vector_Mat pts;
    bool isClosed=0;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int thickness=1;
    int lineType=8;
    int shift=0;

    const char* keywords[] = { "img", "pts", "isClosed", "color", "thickness", "lineType", "shift", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OObO|iii:polylines", (char**)keywords, &pyobj_img, &pyobj_pts, &isClosed, &pyobj_color, &thickness, &lineType, &shift) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_pts, pts, ArgInfo("pts", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2( cv::polylines(img, pts, isClosed, color, thickness, lineType, shift));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_pow(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::pow(src, power, dst));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_preCornerDetect(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::preCornerDetect(src, dst, ksize, borderType));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_projectPoints(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints, jacobian, aspectRatio));
        return Py_BuildValue("(NN)", pyopencv_from(imagePoints), pyopencv_from(jacobian));
    }

    return NULL;
}

static PyObject* pyopencv_putText(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_text = NULL;
    string text;
    PyObject* pyobj_org = NULL;
    Point org;
    int fontFace=0;
    double fontScale=0;
    PyObject* pyobj_color = NULL;
    Scalar color;
    int thickness=1;
    int lineType=8;
    bool bottomLeftOrigin=false;

    const char* keywords[] = { "img", "text", "org", "fontFace", "fontScale", "color", "thickness", "lineType", "bottomLeftOrigin", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOidO|iib:putText", (char**)keywords, &pyobj_img, &pyobj_text, &pyobj_org, &fontFace, &fontScale, &pyobj_color, &thickness, &lineType, &bottomLeftOrigin) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_text, text, ArgInfo("text", 0)) &&
        pyopencv_to(pyobj_org, org, ArgInfo("org", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2( cv::putText(img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_pyrDown(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::pyrDown(src, dst, dstsize, borderType));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_pyrMeanShiftFiltering(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    double sp=0;
    double sr=0;
    int maxLevel=1;
    PyObject* pyobj_termcrit = NULL;
    TermCriteria termcrit=TermCriteria( TermCriteria::MAX_ITER+TermCriteria::EPS,5,1);

    const char* keywords[] = { "src", "sp", "sr", "dst", "maxLevel", "termcrit", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odd|OiO:pyrMeanShiftFiltering", (char**)keywords, &pyobj_src, &sp, &sr, &pyobj_dst, &maxLevel, &pyobj_termcrit) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_termcrit, termcrit, ArgInfo("termcrit", 0)) )
    {
        ERRWRAP2( cv::pyrMeanShiftFiltering(src, dst, sp, sr, maxLevel, termcrit));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_pyrUp(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::pyrUp(src, dst, dstsize, borderType));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_randShuffle(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_dst = NULL;
    Mat dst;
    double iterFactor=1.;

    const char* keywords[] = { "dst", "iterFactor", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|d:randShuffle", (char**)keywords, &pyobj_dst, &iterFactor) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2( cv::randShuffle_(dst, iterFactor));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_randn(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::randn(dst, mean, stddev));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_randu(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::randu(dst, low, high));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_rectangle(PyObject* , PyObject* args, PyObject* kw)
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
    int lineType=8;
    int shift=0;

    const char* keywords[] = { "img", "pt1", "pt2", "color", "thickness", "lineType", "shift", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|iii:rectangle", (char**)keywords, &pyobj_img, &pyobj_pt1, &pyobj_pt2, &pyobj_color, &thickness, &lineType, &shift) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 1)) &&
        pyopencv_to(pyobj_pt1, pt1, ArgInfo("pt1", 0)) &&
        pyopencv_to(pyobj_pt2, pt2, ArgInfo("pt2", 0)) &&
        pyopencv_to(pyobj_color, color, ArgInfo("color", 0)) )
    {
        ERRWRAP2( cv::rectangle(img, pt1, pt2, color, thickness, lineType, shift));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_rectify3Collinear(PyObject* , PyObject* args, PyObject* kw)
{
    float retval;
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
        ERRWRAP2( retval = cv::rectify3Collinear(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, cameraMatrix3, distCoeffs3, imgpt1, imgpt3, imageSize, R12, T12, R13, T13, R1, R2, R3, P1, P2, P3, Q, alpha, newImgSize, &roi1, &roi2, flags));
        return Py_BuildValue("(NNNNNNNNNN)", pyopencv_from(retval), pyopencv_from(R1), pyopencv_from(R2), pyopencv_from(R3), pyopencv_from(P1), pyopencv_from(P2), pyopencv_from(P3), pyopencv_from(Q), pyopencv_from(roi1), pyopencv_from(roi2));
    }

    return NULL;
}

static PyObject* pyopencv_reduce(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::reduce(src, dst, dim, rtype, dtype));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_remap(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::remap(src, dst, map1, map2, interpolation, borderMode, borderValue));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_repeat(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::repeat(src, ny, nx, dst));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_reprojectImageTo3D(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::reprojectImageTo3D(disparity, _3dImage, Q, handleMissingValues, ddepth));
        return pyopencv_from(_3dImage);
    }

    return NULL;
}

static PyObject* pyopencv_resize(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::resize(src, dst, dsize, fx, fy, interpolation));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_resizeWindow(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_winname = NULL;
    string winname;
    int width=0;
    int height=0;

    const char* keywords[] = { "winname", "width", "height", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oii:resizeWindow", (char**)keywords, &pyobj_winname, &width, &height) &&
        pyopencv_to(pyobj_winname, winname, ArgInfo("winname", 0)) )
    {
        ERRWRAP2( cv::resizeWindow(winname, width, height));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_scaleAdd(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::scaleAdd(src1, alpha, src2, dst));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_segmentMotion(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::segmentMotion(mhi, segmask, boundingRects, timestamp, segThresh));
        return Py_BuildValue("(NN)", pyopencv_from(segmask), pyopencv_from(boundingRects));
    }

    return NULL;
}

static PyObject* pyopencv_sepFilter2D(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::sepFilter2D(src, dst, ddepth, kernelX, kernelY, anchor, delta, borderType));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_setIdentity(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::setIdentity(mtx, s));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_setTrackbarPos(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_trackbarname = NULL;
    string trackbarname;
    PyObject* pyobj_winname = NULL;
    string winname;
    int pos=0;

    const char* keywords[] = { "trackbarname", "winname", "pos", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi:setTrackbarPos", (char**)keywords, &pyobj_trackbarname, &pyobj_winname, &pos) &&
        pyopencv_to(pyobj_trackbarname, trackbarname, ArgInfo("trackbarname", 0)) &&
        pyopencv_to(pyobj_winname, winname, ArgInfo("winname", 0)) )
    {
        ERRWRAP2( cv::setTrackbarPos(trackbarname, winname, pos));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_setUseOptimized(PyObject* , PyObject* args, PyObject* kw)
{
    bool onoff=0;

    const char* keywords[] = { "onoff", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:setUseOptimized", (char**)keywords, &onoff) )
    {
        ERRWRAP2( cv::setUseOptimized(onoff));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_setWindowProperty(PyObject* , PyObject* args, PyObject* kw)
{
    PyObject* pyobj_winname = NULL;
    string winname;
    int prop_id=0;
    double prop_value=0;

    const char* keywords[] = { "winname", "prop_id", "prop_value", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oid:setWindowProperty", (char**)keywords, &pyobj_winname, &prop_id, &prop_value) &&
        pyopencv_to(pyobj_winname, winname, ArgInfo("winname", 0)) )
    {
        ERRWRAP2( cv::setWindowProperty(winname, prop_id, prop_value));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_solve(PyObject* , PyObject* args, PyObject* kw)
{
    bool retval;
    PyObject* pyobj_src1 = NULL;
    Mat src1;
    PyObject* pyobj_src2 = NULL;
    Mat src2;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    int flags=DECOMP_LU;

    const char* keywords[] = { "src1", "src2", "dst", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Oi:solve", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &flags) &&
        pyopencv_to(pyobj_src1, src1, ArgInfo("src1", 0)) &&
        pyopencv_to(pyobj_src2, src2, ArgInfo("src2", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2( retval = cv::solve(src1, src2, dst, flags));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(dst));
    }

    return NULL;
}

static PyObject* pyopencv_solveCubic(PyObject* , PyObject* args, PyObject* kw)
{
    int retval;
    PyObject* pyobj_coeffs = NULL;
    Mat coeffs;
    PyObject* pyobj_roots = NULL;
    Mat roots;

    const char* keywords[] = { "coeffs", "roots", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:solveCubic", (char**)keywords, &pyobj_coeffs, &pyobj_roots) &&
        pyopencv_to(pyobj_coeffs, coeffs, ArgInfo("coeffs", 0)) &&
        pyopencv_to(pyobj_roots, roots, ArgInfo("roots", 1)) )
    {
        ERRWRAP2( retval = cv::solveCubic(coeffs, roots));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(roots));
    }

    return NULL;
}

static PyObject* pyopencv_solvePnP(PyObject* , PyObject* args, PyObject* kw)
{
    bool retval;
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
    int flags=ITERATIVE;

    const char* keywords[] = { "objectPoints", "imagePoints", "cameraMatrix", "distCoeffs", "rvec", "tvec", "useExtrinsicGuess", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OObi:solvePnP", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvec, &pyobj_tvec, &useExtrinsicGuess, &flags) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints, imagePoints, ArgInfo("imagePoints", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_rvec, rvec, ArgInfo("rvec", 1)) &&
        pyopencv_to(pyobj_tvec, tvec, ArgInfo("tvec", 1)) )
    {
        ERRWRAP2( retval = cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess, flags));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(rvec), pyopencv_from(tvec));
    }

    return NULL;
}

static PyObject* pyopencv_solvePnPRansac(PyObject* , PyObject* args, PyObject* kw)
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
    int minInliersCount=100;
    PyObject* pyobj_inliers = NULL;
    Mat inliers;
    int flags=ITERATIVE;

    const char* keywords[] = { "objectPoints", "imagePoints", "cameraMatrix", "distCoeffs", "rvec", "tvec", "useExtrinsicGuess", "iterationsCount", "reprojectionError", "minInliersCount", "inliers", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OObifiOi:solvePnPRansac", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvec, &pyobj_tvec, &useExtrinsicGuess, &iterationsCount, &reprojectionError, &minInliersCount, &pyobj_inliers, &flags) &&
        pyopencv_to(pyobj_objectPoints, objectPoints, ArgInfo("objectPoints", 0)) &&
        pyopencv_to(pyobj_imagePoints, imagePoints, ArgInfo("imagePoints", 0)) &&
        pyopencv_to(pyobj_cameraMatrix, cameraMatrix, ArgInfo("cameraMatrix", 0)) &&
        pyopencv_to(pyobj_distCoeffs, distCoeffs, ArgInfo("distCoeffs", 0)) &&
        pyopencv_to(pyobj_rvec, rvec, ArgInfo("rvec", 1)) &&
        pyopencv_to(pyobj_tvec, tvec, ArgInfo("tvec", 1)) &&
        pyopencv_to(pyobj_inliers, inliers, ArgInfo("inliers", 1)) )
    {
        ERRWRAP2( cv::solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess, iterationsCount, reprojectionError, minInliersCount, inliers, flags));
        return Py_BuildValue("(NNN)", pyopencv_from(rvec), pyopencv_from(tvec), pyopencv_from(inliers));
    }

    return NULL;
}

static PyObject* pyopencv_solvePoly(PyObject* , PyObject* args, PyObject* kw)
{
    double retval;
    PyObject* pyobj_coeffs = NULL;
    Mat coeffs;
    PyObject* pyobj_roots = NULL;
    Mat roots;
    int maxIters=300;

    const char* keywords[] = { "coeffs", "roots", "maxIters", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oi:solvePoly", (char**)keywords, &pyobj_coeffs, &pyobj_roots, &maxIters) &&
        pyopencv_to(pyobj_coeffs, coeffs, ArgInfo("coeffs", 0)) &&
        pyopencv_to(pyobj_roots, roots, ArgInfo("roots", 1)) )
    {
        ERRWRAP2( retval = cv::solvePoly(coeffs, roots, maxIters));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(roots));
    }

    return NULL;
}

static PyObject* pyopencv_sort(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::sort(src, dst, flags));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_sortIdx(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::sortIdx(src, dst, flags));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_split(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::split(m, mv));
        return pyopencv_from(mv);
    }

    return NULL;
}

static PyObject* pyopencv_sqrt(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::sqrt(src, dst));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_startWindowThread(PyObject* , PyObject* args, PyObject* kw)
{
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2( retval = cv::startWindowThread());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_stereoCalibrate(PyObject* , PyObject* args, PyObject* kw)
{
    double retval;
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
    PyObject* pyobj_criteria = NULL;
    TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6);
    int flags=CALIB_FIX_INTRINSIC;

    const char* keywords[] = { "objectPoints", "imagePoints1", "imagePoints2", "imageSize", "cameraMatrix1", "distCoeffs1", "cameraMatrix2", "distCoeffs2", "R", "T", "E", "F", "criteria", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OOOOOOOOOi:stereoCalibrate", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints1, &pyobj_imagePoints2, &pyobj_imageSize, &pyobj_cameraMatrix1, &pyobj_distCoeffs1, &pyobj_cameraMatrix2, &pyobj_distCoeffs2, &pyobj_R, &pyobj_T, &pyobj_E, &pyobj_F, &pyobj_criteria, &flags) &&
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
        ERRWRAP2( retval = cv::stereoCalibrate(objectPoints, imagePoints1, imagePoints2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, E, F, criteria, flags));
        return Py_BuildValue("(NNNNNNNNN)", pyopencv_from(retval), pyopencv_from(cameraMatrix1), pyopencv_from(distCoeffs1), pyopencv_from(cameraMatrix2), pyopencv_from(distCoeffs2), pyopencv_from(R), pyopencv_from(T), pyopencv_from(E), pyopencv_from(F));
    }

    return NULL;
}

static PyObject* pyopencv_stereoRectify(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, R1, R2, P1, P2, Q, flags, alpha, newImageSize, &validPixROI1, &validPixROI2));
        return Py_BuildValue("(NNNNNNN)", pyopencv_from(R1), pyopencv_from(R2), pyopencv_from(P1), pyopencv_from(P2), pyopencv_from(Q), pyopencv_from(validPixROI1), pyopencv_from(validPixROI2));
    }

    return NULL;
}

static PyObject* pyopencv_stereoRectifyUncalibrated(PyObject* , PyObject* args, PyObject* kw)
{
    bool retval;
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

    const char* keywords[] = { "points1", "points2", "F", "imgSize", "H1", "H2", "threshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OOd:stereoRectifyUncalibrated", (char**)keywords, &pyobj_points1, &pyobj_points2, &pyobj_F, &pyobj_imgSize, &pyobj_H1, &pyobj_H2, &threshold) &&
        pyopencv_to(pyobj_points1, points1, ArgInfo("points1", 0)) &&
        pyopencv_to(pyobj_points2, points2, ArgInfo("points2", 0)) &&
        pyopencv_to(pyobj_F, F, ArgInfo("F", 0)) &&
        pyopencv_to(pyobj_imgSize, imgSize, ArgInfo("imgSize", 0)) &&
        pyopencv_to(pyobj_H1, H1, ArgInfo("H1", 1)) &&
        pyopencv_to(pyobj_H2, H2, ArgInfo("H2", 1)) )
    {
        ERRWRAP2( retval = cv::stereoRectifyUncalibrated(points1, points2, F, imgSize, H1, H2, threshold));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(H1), pyopencv_from(H2));
    }

    return NULL;
}

static PyObject* pyopencv_subtract(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::subtract(src1, src2, dst, mask, dtype));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_sumElems(PyObject* , PyObject* args, PyObject* kw)
{
    Scalar retval;
    PyObject* pyobj_src = NULL;
    Mat src;

    const char* keywords[] = { "src", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:sumElems", (char**)keywords, &pyobj_src) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) )
    {
        ERRWRAP2( retval = cv::sum(src));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_threshold(PyObject* , PyObject* args, PyObject* kw)
{
    double retval;
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    double thresh=0;
    double maxval=0;
    int type=0;

    const char* keywords[] = { "src", "thresh", "maxval", "type", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oddi|O:threshold", (char**)keywords, &pyobj_src, &thresh, &maxval, &type, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2( retval = cv::threshold(src, dst, thresh, maxval, type));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(dst));
    }

    return NULL;
}

static PyObject* pyopencv_trace(PyObject* , PyObject* args, PyObject* kw)
{
    Scalar retval;
    PyObject* pyobj_mtx = NULL;
    Mat mtx;

    const char* keywords[] = { "mtx", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:trace", (char**)keywords, &pyobj_mtx) &&
        pyopencv_to(pyobj_mtx, mtx, ArgInfo("mtx", 0)) )
    {
        ERRWRAP2( retval = cv::trace(mtx));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_transform(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::transform(src, dst, m));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_transpose(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::transpose(src, dst));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_triangulatePoints(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::triangulatePoints(projMatr1, projMatr2, projPoints1, projPoints2, points4D));
        return pyopencv_from(points4D);
    }

    return NULL;
}

static PyObject* pyopencv_undistort(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::undistort(src, dst, cameraMatrix, distCoeffs, newCameraMatrix));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_undistortPoints(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::undistortPoints(src, dst, cameraMatrix, distCoeffs, R, P));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_updateMotionHistory(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::updateMotionHistory(silhouette, mhi, timestamp, duration));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_useOptimized(PyObject* , PyObject* args, PyObject* kw)
{
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2( retval = cv::useOptimized());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_validateDisparity(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::validateDisparity(disparity, cost, minDisparity, numberOfDisparities, disp12MaxDisp));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_vconcat(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::vconcat(src, dst));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_waitKey(PyObject* , PyObject* args, PyObject* kw)
{
    int retval;
    int delay=0;

    const char* keywords[] = { "delay", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|i:waitKey", (char**)keywords, &delay) )
    {
        ERRWRAP2( retval = cv::waitKey(delay));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_warpAffine(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::warpAffine(src, dst, M, dsize, flags, borderMode, borderValue));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_warpPerspective(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::warpPerspective(src, dst, M, dsize, flags, borderMode, borderValue));
        return pyopencv_from(dst);
    }

    return NULL;
}

static PyObject* pyopencv_watershed(PyObject* , PyObject* args, PyObject* kw)
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
        ERRWRAP2( cv::watershed(image, markers));
        Py_RETURN_NONE;
    }

    return NULL;
}

