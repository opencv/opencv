
struct pyopencv_AKAZE_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_AKAZE_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".AKAZE",
    sizeof(pyopencv_AKAZE_t),
};

static void pyopencv_AKAZE_dealloc(PyObject* self)
{
    ((pyopencv_AKAZE_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::AKAZE>& r)
{
    pyopencv_AKAZE_t *m = PyObject_NEW(pyopencv_AKAZE_t, &pyopencv_AKAZE_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::AKAZE>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_AKAZE_Type))
    {
        failmsg("Expected cv::AKAZE for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_AKAZE_t*)src)->v.dynamicCast<cv::AKAZE>();
    return true;
}


struct pyopencv_AffineTransformer_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_AffineTransformer_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".AffineTransformer",
    sizeof(pyopencv_AffineTransformer_t),
};

static void pyopencv_AffineTransformer_dealloc(PyObject* self)
{
    ((pyopencv_AffineTransformer_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::AffineTransformer>& r)
{
    pyopencv_AffineTransformer_t *m = PyObject_NEW(pyopencv_AffineTransformer_t, &pyopencv_AffineTransformer_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::AffineTransformer>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_AffineTransformer_Type))
    {
        failmsg("Expected cv::AffineTransformer for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_AffineTransformer_t*)src)->v.dynamicCast<cv::AffineTransformer>();
    return true;
}


struct pyopencv_AgastFeatureDetector_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_AgastFeatureDetector_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".AgastFeatureDetector",
    sizeof(pyopencv_AgastFeatureDetector_t),
};

static void pyopencv_AgastFeatureDetector_dealloc(PyObject* self)
{
    ((pyopencv_AgastFeatureDetector_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::AgastFeatureDetector>& r)
{
    pyopencv_AgastFeatureDetector_t *m = PyObject_NEW(pyopencv_AgastFeatureDetector_t, &pyopencv_AgastFeatureDetector_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::AgastFeatureDetector>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_AgastFeatureDetector_Type))
    {
        failmsg("Expected cv::AgastFeatureDetector for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_AgastFeatureDetector_t*)src)->v.dynamicCast<cv::AgastFeatureDetector>();
    return true;
}


struct pyopencv_Algorithm_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_Algorithm_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".Algorithm",
    sizeof(pyopencv_Algorithm_t),
};

static void pyopencv_Algorithm_dealloc(PyObject* self)
{
    ((pyopencv_Algorithm_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::Algorithm>& r)
{
    pyopencv_Algorithm_t *m = PyObject_NEW(pyopencv_Algorithm_t, &pyopencv_Algorithm_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::Algorithm>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_Algorithm_Type))
    {
        failmsg("Expected cv::Algorithm for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_Algorithm_t*)src)->v.dynamicCast<cv::Algorithm>();
    return true;
}


struct pyopencv_AlignExposures_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_AlignExposures_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".AlignExposures",
    sizeof(pyopencv_AlignExposures_t),
};

static void pyopencv_AlignExposures_dealloc(PyObject* self)
{
    ((pyopencv_AlignExposures_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::AlignExposures>& r)
{
    pyopencv_AlignExposures_t *m = PyObject_NEW(pyopencv_AlignExposures_t, &pyopencv_AlignExposures_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::AlignExposures>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_AlignExposures_Type))
    {
        failmsg("Expected cv::AlignExposures for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_AlignExposures_t*)src)->v.dynamicCast<cv::AlignExposures>();
    return true;
}


struct pyopencv_AlignMTB_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_AlignMTB_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".AlignMTB",
    sizeof(pyopencv_AlignMTB_t),
};

static void pyopencv_AlignMTB_dealloc(PyObject* self)
{
    ((pyopencv_AlignMTB_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::AlignMTB>& r)
{
    pyopencv_AlignMTB_t *m = PyObject_NEW(pyopencv_AlignMTB_t, &pyopencv_AlignMTB_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::AlignMTB>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_AlignMTB_Type))
    {
        failmsg("Expected cv::AlignMTB for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_AlignMTB_t*)src)->v.dynamicCast<cv::AlignMTB>();
    return true;
}


struct pyopencv_BFMatcher_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_BFMatcher_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".BFMatcher",
    sizeof(pyopencv_BFMatcher_t),
};

static void pyopencv_BFMatcher_dealloc(PyObject* self)
{
    ((pyopencv_BFMatcher_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::BFMatcher>& r)
{
    pyopencv_BFMatcher_t *m = PyObject_NEW(pyopencv_BFMatcher_t, &pyopencv_BFMatcher_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::BFMatcher>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_BFMatcher_Type))
    {
        failmsg("Expected cv::BFMatcher for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_BFMatcher_t*)src)->v.dynamicCast<cv::BFMatcher>();
    return true;
}


struct pyopencv_BOWImgDescriptorExtractor_t
{
    PyObject_HEAD
    Ptr<cv::BOWImgDescriptorExtractor> v;
};

static PyTypeObject pyopencv_BOWImgDescriptorExtractor_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".BOWImgDescriptorExtractor",
    sizeof(pyopencv_BOWImgDescriptorExtractor_t),
};

static void pyopencv_BOWImgDescriptorExtractor_dealloc(PyObject* self)
{
    ((pyopencv_BOWImgDescriptorExtractor_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::BOWImgDescriptorExtractor>& r)
{
    pyopencv_BOWImgDescriptorExtractor_t *m = PyObject_NEW(pyopencv_BOWImgDescriptorExtractor_t, &pyopencv_BOWImgDescriptorExtractor_Type);
    new (&(m->v)) Ptr<cv::BOWImgDescriptorExtractor>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::BOWImgDescriptorExtractor>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_BOWImgDescriptorExtractor_Type))
    {
        failmsg("Expected cv::BOWImgDescriptorExtractor for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_BOWImgDescriptorExtractor_t*)src)->v.dynamicCast<cv::BOWImgDescriptorExtractor>();
    return true;
}


struct pyopencv_BOWKMeansTrainer_t
{
    PyObject_HEAD
    Ptr<cv::BOWKMeansTrainer> v;
};

static PyTypeObject pyopencv_BOWKMeansTrainer_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".BOWKMeansTrainer",
    sizeof(pyopencv_BOWKMeansTrainer_t),
};

static void pyopencv_BOWKMeansTrainer_dealloc(PyObject* self)
{
    ((pyopencv_BOWKMeansTrainer_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::BOWKMeansTrainer>& r)
{
    pyopencv_BOWKMeansTrainer_t *m = PyObject_NEW(pyopencv_BOWKMeansTrainer_t, &pyopencv_BOWKMeansTrainer_Type);
    new (&(m->v)) Ptr<cv::BOWKMeansTrainer>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::BOWKMeansTrainer>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_BOWKMeansTrainer_Type))
    {
        failmsg("Expected cv::BOWKMeansTrainer for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_BOWKMeansTrainer_t*)src)->v.dynamicCast<cv::BOWKMeansTrainer>();
    return true;
}


struct pyopencv_BOWTrainer_t
{
    PyObject_HEAD
    Ptr<cv::BOWTrainer> v;
};

static PyTypeObject pyopencv_BOWTrainer_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".BOWTrainer",
    sizeof(pyopencv_BOWTrainer_t),
};

static void pyopencv_BOWTrainer_dealloc(PyObject* self)
{
    ((pyopencv_BOWTrainer_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::BOWTrainer>& r)
{
    pyopencv_BOWTrainer_t *m = PyObject_NEW(pyopencv_BOWTrainer_t, &pyopencv_BOWTrainer_Type);
    new (&(m->v)) Ptr<cv::BOWTrainer>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::BOWTrainer>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_BOWTrainer_Type))
    {
        failmsg("Expected cv::BOWTrainer for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_BOWTrainer_t*)src)->v.dynamicCast<cv::BOWTrainer>();
    return true;
}


struct pyopencv_BRISK_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_BRISK_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".BRISK",
    sizeof(pyopencv_BRISK_t),
};

static void pyopencv_BRISK_dealloc(PyObject* self)
{
    ((pyopencv_BRISK_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::BRISK>& r)
{
    pyopencv_BRISK_t *m = PyObject_NEW(pyopencv_BRISK_t, &pyopencv_BRISK_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::BRISK>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_BRISK_Type))
    {
        failmsg("Expected cv::BRISK for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_BRISK_t*)src)->v.dynamicCast<cv::BRISK>();
    return true;
}


struct pyopencv_BackgroundSubtractor_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_BackgroundSubtractor_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".BackgroundSubtractor",
    sizeof(pyopencv_BackgroundSubtractor_t),
};

static void pyopencv_BackgroundSubtractor_dealloc(PyObject* self)
{
    ((pyopencv_BackgroundSubtractor_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::BackgroundSubtractor>& r)
{
    pyopencv_BackgroundSubtractor_t *m = PyObject_NEW(pyopencv_BackgroundSubtractor_t, &pyopencv_BackgroundSubtractor_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::BackgroundSubtractor>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_BackgroundSubtractor_Type))
    {
        failmsg("Expected cv::BackgroundSubtractor for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_BackgroundSubtractor_t*)src)->v.dynamicCast<cv::BackgroundSubtractor>();
    return true;
}


struct pyopencv_BackgroundSubtractorKNN_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_BackgroundSubtractorKNN_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".BackgroundSubtractorKNN",
    sizeof(pyopencv_BackgroundSubtractorKNN_t),
};

static void pyopencv_BackgroundSubtractorKNN_dealloc(PyObject* self)
{
    ((pyopencv_BackgroundSubtractorKNN_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::BackgroundSubtractorKNN>& r)
{
    pyopencv_BackgroundSubtractorKNN_t *m = PyObject_NEW(pyopencv_BackgroundSubtractorKNN_t, &pyopencv_BackgroundSubtractorKNN_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::BackgroundSubtractorKNN>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_BackgroundSubtractorKNN_Type))
    {
        failmsg("Expected cv::BackgroundSubtractorKNN for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_BackgroundSubtractorKNN_t*)src)->v.dynamicCast<cv::BackgroundSubtractorKNN>();
    return true;
}


struct pyopencv_BackgroundSubtractorMOG2_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_BackgroundSubtractorMOG2_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".BackgroundSubtractorMOG2",
    sizeof(pyopencv_BackgroundSubtractorMOG2_t),
};

static void pyopencv_BackgroundSubtractorMOG2_dealloc(PyObject* self)
{
    ((pyopencv_BackgroundSubtractorMOG2_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::BackgroundSubtractorMOG2>& r)
{
    pyopencv_BackgroundSubtractorMOG2_t *m = PyObject_NEW(pyopencv_BackgroundSubtractorMOG2_t, &pyopencv_BackgroundSubtractorMOG2_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::BackgroundSubtractorMOG2>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_BackgroundSubtractorMOG2_Type))
    {
        failmsg("Expected cv::BackgroundSubtractorMOG2 for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_BackgroundSubtractorMOG2_t*)src)->v.dynamicCast<cv::BackgroundSubtractorMOG2>();
    return true;
}


struct pyopencv_BaseCascadeClassifier_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_BaseCascadeClassifier_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".BaseCascadeClassifier",
    sizeof(pyopencv_BaseCascadeClassifier_t),
};

static void pyopencv_BaseCascadeClassifier_dealloc(PyObject* self)
{
    ((pyopencv_BaseCascadeClassifier_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::BaseCascadeClassifier>& r)
{
    pyopencv_BaseCascadeClassifier_t *m = PyObject_NEW(pyopencv_BaseCascadeClassifier_t, &pyopencv_BaseCascadeClassifier_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::BaseCascadeClassifier>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_BaseCascadeClassifier_Type))
    {
        failmsg("Expected cv::BaseCascadeClassifier for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_BaseCascadeClassifier_t*)src)->v.dynamicCast<cv::BaseCascadeClassifier>();
    return true;
}


struct pyopencv_CLAHE_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_CLAHE_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".CLAHE",
    sizeof(pyopencv_CLAHE_t),
};

static void pyopencv_CLAHE_dealloc(PyObject* self)
{
    ((pyopencv_CLAHE_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::CLAHE>& r)
{
    pyopencv_CLAHE_t *m = PyObject_NEW(pyopencv_CLAHE_t, &pyopencv_CLAHE_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::CLAHE>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_CLAHE_Type))
    {
        failmsg("Expected cv::CLAHE for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_CLAHE_t*)src)->v.dynamicCast<cv::CLAHE>();
    return true;
}


struct pyopencv_CalibrateCRF_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_CalibrateCRF_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".CalibrateCRF",
    sizeof(pyopencv_CalibrateCRF_t),
};

static void pyopencv_CalibrateCRF_dealloc(PyObject* self)
{
    ((pyopencv_CalibrateCRF_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::CalibrateCRF>& r)
{
    pyopencv_CalibrateCRF_t *m = PyObject_NEW(pyopencv_CalibrateCRF_t, &pyopencv_CalibrateCRF_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::CalibrateCRF>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_CalibrateCRF_Type))
    {
        failmsg("Expected cv::CalibrateCRF for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_CalibrateCRF_t*)src)->v.dynamicCast<cv::CalibrateCRF>();
    return true;
}


struct pyopencv_CalibrateDebevec_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_CalibrateDebevec_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".CalibrateDebevec",
    sizeof(pyopencv_CalibrateDebevec_t),
};

static void pyopencv_CalibrateDebevec_dealloc(PyObject* self)
{
    ((pyopencv_CalibrateDebevec_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::CalibrateDebevec>& r)
{
    pyopencv_CalibrateDebevec_t *m = PyObject_NEW(pyopencv_CalibrateDebevec_t, &pyopencv_CalibrateDebevec_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::CalibrateDebevec>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_CalibrateDebevec_Type))
    {
        failmsg("Expected cv::CalibrateDebevec for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_CalibrateDebevec_t*)src)->v.dynamicCast<cv::CalibrateDebevec>();
    return true;
}


struct pyopencv_CalibrateRobertson_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_CalibrateRobertson_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".CalibrateRobertson",
    sizeof(pyopencv_CalibrateRobertson_t),
};

static void pyopencv_CalibrateRobertson_dealloc(PyObject* self)
{
    ((pyopencv_CalibrateRobertson_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::CalibrateRobertson>& r)
{
    pyopencv_CalibrateRobertson_t *m = PyObject_NEW(pyopencv_CalibrateRobertson_t, &pyopencv_CalibrateRobertson_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::CalibrateRobertson>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_CalibrateRobertson_Type))
    {
        failmsg("Expected cv::CalibrateRobertson for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_CalibrateRobertson_t*)src)->v.dynamicCast<cv::CalibrateRobertson>();
    return true;
}


struct pyopencv_CascadeClassifier_t
{
    PyObject_HEAD
    Ptr<cv::CascadeClassifier> v;
};

static PyTypeObject pyopencv_CascadeClassifier_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".CascadeClassifier",
    sizeof(pyopencv_CascadeClassifier_t),
};

static void pyopencv_CascadeClassifier_dealloc(PyObject* self)
{
    ((pyopencv_CascadeClassifier_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::CascadeClassifier>& r)
{
    pyopencv_CascadeClassifier_t *m = PyObject_NEW(pyopencv_CascadeClassifier_t, &pyopencv_CascadeClassifier_Type);
    new (&(m->v)) Ptr<cv::CascadeClassifier>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::CascadeClassifier>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_CascadeClassifier_Type))
    {
        failmsg("Expected cv::CascadeClassifier for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_CascadeClassifier_t*)src)->v.dynamicCast<cv::CascadeClassifier>();
    return true;
}


struct pyopencv_ChiHistogramCostExtractor_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_ChiHistogramCostExtractor_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".ChiHistogramCostExtractor",
    sizeof(pyopencv_ChiHistogramCostExtractor_t),
};

static void pyopencv_ChiHistogramCostExtractor_dealloc(PyObject* self)
{
    ((pyopencv_ChiHistogramCostExtractor_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::ChiHistogramCostExtractor>& r)
{
    pyopencv_ChiHistogramCostExtractor_t *m = PyObject_NEW(pyopencv_ChiHistogramCostExtractor_t, &pyopencv_ChiHistogramCostExtractor_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::ChiHistogramCostExtractor>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_ChiHistogramCostExtractor_Type))
    {
        failmsg("Expected cv::ChiHistogramCostExtractor for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_ChiHistogramCostExtractor_t*)src)->v.dynamicCast<cv::ChiHistogramCostExtractor>();
    return true;
}


struct pyopencv_DMatch_t
{
    PyObject_HEAD
    cv::DMatch v;
};

static PyTypeObject pyopencv_DMatch_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".DMatch",
    sizeof(pyopencv_DMatch_t),
};

static void pyopencv_DMatch_dealloc(PyObject* self)
{
    ((pyopencv_DMatch_t*)self)->v.cv::DMatch::~DMatch();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const cv::DMatch& r)
{
    pyopencv_DMatch_t *m = PyObject_NEW(pyopencv_DMatch_t, &pyopencv_DMatch_Type);
    new (&m->v) cv::DMatch(r); //Copy constructor
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, cv::DMatch& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_DMatch_Type))
    {
        failmsg("Expected cv::DMatch for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_DMatch_t*)src)->v;
    return true;
}

struct pyopencv_DenseOpticalFlow_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_DenseOpticalFlow_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".DenseOpticalFlow",
    sizeof(pyopencv_DenseOpticalFlow_t),
};

static void pyopencv_DenseOpticalFlow_dealloc(PyObject* self)
{
    ((pyopencv_DenseOpticalFlow_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::DenseOpticalFlow>& r)
{
    pyopencv_DenseOpticalFlow_t *m = PyObject_NEW(pyopencv_DenseOpticalFlow_t, &pyopencv_DenseOpticalFlow_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::DenseOpticalFlow>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_DenseOpticalFlow_Type))
    {
        failmsg("Expected cv::DenseOpticalFlow for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_DenseOpticalFlow_t*)src)->v.dynamicCast<cv::DenseOpticalFlow>();
    return true;
}


struct pyopencv_DescriptorMatcher_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_DescriptorMatcher_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".DescriptorMatcher",
    sizeof(pyopencv_DescriptorMatcher_t),
};

static void pyopencv_DescriptorMatcher_dealloc(PyObject* self)
{
    ((pyopencv_DescriptorMatcher_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::DescriptorMatcher>& r)
{
    pyopencv_DescriptorMatcher_t *m = PyObject_NEW(pyopencv_DescriptorMatcher_t, &pyopencv_DescriptorMatcher_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::DescriptorMatcher>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_DescriptorMatcher_Type))
    {
        failmsg("Expected cv::DescriptorMatcher for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_DescriptorMatcher_t*)src)->v.dynamicCast<cv::DescriptorMatcher>();
    return true;
}


struct pyopencv_DualTVL1OpticalFlow_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_DualTVL1OpticalFlow_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".DualTVL1OpticalFlow",
    sizeof(pyopencv_DualTVL1OpticalFlow_t),
};

static void pyopencv_DualTVL1OpticalFlow_dealloc(PyObject* self)
{
    ((pyopencv_DualTVL1OpticalFlow_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::DualTVL1OpticalFlow>& r)
{
    pyopencv_DualTVL1OpticalFlow_t *m = PyObject_NEW(pyopencv_DualTVL1OpticalFlow_t, &pyopencv_DualTVL1OpticalFlow_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::DualTVL1OpticalFlow>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_DualTVL1OpticalFlow_Type))
    {
        failmsg("Expected cv::DualTVL1OpticalFlow for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_DualTVL1OpticalFlow_t*)src)->v.dynamicCast<cv::DualTVL1OpticalFlow>();
    return true;
}


struct pyopencv_EMDHistogramCostExtractor_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_EMDHistogramCostExtractor_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".EMDHistogramCostExtractor",
    sizeof(pyopencv_EMDHistogramCostExtractor_t),
};

static void pyopencv_EMDHistogramCostExtractor_dealloc(PyObject* self)
{
    ((pyopencv_EMDHistogramCostExtractor_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::EMDHistogramCostExtractor>& r)
{
    pyopencv_EMDHistogramCostExtractor_t *m = PyObject_NEW(pyopencv_EMDHistogramCostExtractor_t, &pyopencv_EMDHistogramCostExtractor_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::EMDHistogramCostExtractor>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_EMDHistogramCostExtractor_Type))
    {
        failmsg("Expected cv::EMDHistogramCostExtractor for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_EMDHistogramCostExtractor_t*)src)->v.dynamicCast<cv::EMDHistogramCostExtractor>();
    return true;
}


struct pyopencv_EMDL1HistogramCostExtractor_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_EMDL1HistogramCostExtractor_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".EMDL1HistogramCostExtractor",
    sizeof(pyopencv_EMDL1HistogramCostExtractor_t),
};

static void pyopencv_EMDL1HistogramCostExtractor_dealloc(PyObject* self)
{
    ((pyopencv_EMDL1HistogramCostExtractor_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::EMDL1HistogramCostExtractor>& r)
{
    pyopencv_EMDL1HistogramCostExtractor_t *m = PyObject_NEW(pyopencv_EMDL1HistogramCostExtractor_t, &pyopencv_EMDL1HistogramCostExtractor_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::EMDL1HistogramCostExtractor>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_EMDL1HistogramCostExtractor_Type))
    {
        failmsg("Expected cv::EMDL1HistogramCostExtractor for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_EMDL1HistogramCostExtractor_t*)src)->v.dynamicCast<cv::EMDL1HistogramCostExtractor>();
    return true;
}


struct pyopencv_FarnebackOpticalFlow_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_FarnebackOpticalFlow_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".FarnebackOpticalFlow",
    sizeof(pyopencv_FarnebackOpticalFlow_t),
};

static void pyopencv_FarnebackOpticalFlow_dealloc(PyObject* self)
{
    ((pyopencv_FarnebackOpticalFlow_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::FarnebackOpticalFlow>& r)
{
    pyopencv_FarnebackOpticalFlow_t *m = PyObject_NEW(pyopencv_FarnebackOpticalFlow_t, &pyopencv_FarnebackOpticalFlow_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::FarnebackOpticalFlow>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_FarnebackOpticalFlow_Type))
    {
        failmsg("Expected cv::FarnebackOpticalFlow for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_FarnebackOpticalFlow_t*)src)->v.dynamicCast<cv::FarnebackOpticalFlow>();
    return true;
}


struct pyopencv_FastFeatureDetector_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_FastFeatureDetector_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".FastFeatureDetector",
    sizeof(pyopencv_FastFeatureDetector_t),
};

static void pyopencv_FastFeatureDetector_dealloc(PyObject* self)
{
    ((pyopencv_FastFeatureDetector_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::FastFeatureDetector>& r)
{
    pyopencv_FastFeatureDetector_t *m = PyObject_NEW(pyopencv_FastFeatureDetector_t, &pyopencv_FastFeatureDetector_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::FastFeatureDetector>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_FastFeatureDetector_Type))
    {
        failmsg("Expected cv::FastFeatureDetector for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_FastFeatureDetector_t*)src)->v.dynamicCast<cv::FastFeatureDetector>();
    return true;
}


struct pyopencv_Feature2D_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_Feature2D_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".Feature2D",
    sizeof(pyopencv_Feature2D_t),
};

static void pyopencv_Feature2D_dealloc(PyObject* self)
{
    ((pyopencv_Feature2D_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::Feature2D>& r)
{
    pyopencv_Feature2D_t *m = PyObject_NEW(pyopencv_Feature2D_t, &pyopencv_Feature2D_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::Feature2D>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_Feature2D_Type))
    {
        failmsg("Expected cv::Feature2D for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_Feature2D_t*)src)->v.dynamicCast<cv::Feature2D>();
    return true;
}


struct pyopencv_FileNode_t
{
    PyObject_HEAD
    cv::FileNode v;
};

static PyTypeObject pyopencv_FileNode_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".FileNode",
    sizeof(pyopencv_FileNode_t),
};

static void pyopencv_FileNode_dealloc(PyObject* self)
{
    ((pyopencv_FileNode_t*)self)->v.cv::FileNode::~FileNode();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const cv::FileNode& r)
{
    pyopencv_FileNode_t *m = PyObject_NEW(pyopencv_FileNode_t, &pyopencv_FileNode_Type);
    new (&m->v) cv::FileNode(r); //Copy constructor
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, cv::FileNode& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_FileNode_Type))
    {
        failmsg("Expected cv::FileNode for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_FileNode_t*)src)->v;
    return true;
}

struct pyopencv_FileStorage_t
{
    PyObject_HEAD
    Ptr<cv::FileStorage> v;
};

static PyTypeObject pyopencv_FileStorage_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".FileStorage",
    sizeof(pyopencv_FileStorage_t),
};

static void pyopencv_FileStorage_dealloc(PyObject* self)
{
    ((pyopencv_FileStorage_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::FileStorage>& r)
{
    pyopencv_FileStorage_t *m = PyObject_NEW(pyopencv_FileStorage_t, &pyopencv_FileStorage_Type);
    new (&(m->v)) Ptr<cv::FileStorage>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::FileStorage>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_FileStorage_Type))
    {
        failmsg("Expected cv::FileStorage for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_FileStorage_t*)src)->v.dynamicCast<cv::FileStorage>();
    return true;
}


struct pyopencv_FlannBasedMatcher_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_FlannBasedMatcher_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".FlannBasedMatcher",
    sizeof(pyopencv_FlannBasedMatcher_t),
};

static void pyopencv_FlannBasedMatcher_dealloc(PyObject* self)
{
    ((pyopencv_FlannBasedMatcher_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::FlannBasedMatcher>& r)
{
    pyopencv_FlannBasedMatcher_t *m = PyObject_NEW(pyopencv_FlannBasedMatcher_t, &pyopencv_FlannBasedMatcher_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::FlannBasedMatcher>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_FlannBasedMatcher_Type))
    {
        failmsg("Expected cv::FlannBasedMatcher for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_FlannBasedMatcher_t*)src)->v.dynamicCast<cv::FlannBasedMatcher>();
    return true;
}


struct pyopencv_GFTTDetector_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_GFTTDetector_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".GFTTDetector",
    sizeof(pyopencv_GFTTDetector_t),
};

static void pyopencv_GFTTDetector_dealloc(PyObject* self)
{
    ((pyopencv_GFTTDetector_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::GFTTDetector>& r)
{
    pyopencv_GFTTDetector_t *m = PyObject_NEW(pyopencv_GFTTDetector_t, &pyopencv_GFTTDetector_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::GFTTDetector>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_GFTTDetector_Type))
    {
        failmsg("Expected cv::GFTTDetector for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_GFTTDetector_t*)src)->v.dynamicCast<cv::GFTTDetector>();
    return true;
}


struct pyopencv_HOGDescriptor_t
{
    PyObject_HEAD
    Ptr<cv::HOGDescriptor> v;
};

static PyTypeObject pyopencv_HOGDescriptor_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".HOGDescriptor",
    sizeof(pyopencv_HOGDescriptor_t),
};

static void pyopencv_HOGDescriptor_dealloc(PyObject* self)
{
    ((pyopencv_HOGDescriptor_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::HOGDescriptor>& r)
{
    pyopencv_HOGDescriptor_t *m = PyObject_NEW(pyopencv_HOGDescriptor_t, &pyopencv_HOGDescriptor_Type);
    new (&(m->v)) Ptr<cv::HOGDescriptor>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::HOGDescriptor>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_HOGDescriptor_Type))
    {
        failmsg("Expected cv::HOGDescriptor for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_HOGDescriptor_t*)src)->v.dynamicCast<cv::HOGDescriptor>();
    return true;
}


struct pyopencv_HausdorffDistanceExtractor_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_HausdorffDistanceExtractor_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".HausdorffDistanceExtractor",
    sizeof(pyopencv_HausdorffDistanceExtractor_t),
};

static void pyopencv_HausdorffDistanceExtractor_dealloc(PyObject* self)
{
    ((pyopencv_HausdorffDistanceExtractor_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::HausdorffDistanceExtractor>& r)
{
    pyopencv_HausdorffDistanceExtractor_t *m = PyObject_NEW(pyopencv_HausdorffDistanceExtractor_t, &pyopencv_HausdorffDistanceExtractor_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::HausdorffDistanceExtractor>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_HausdorffDistanceExtractor_Type))
    {
        failmsg("Expected cv::HausdorffDistanceExtractor for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_HausdorffDistanceExtractor_t*)src)->v.dynamicCast<cv::HausdorffDistanceExtractor>();
    return true;
}


struct pyopencv_HistogramCostExtractor_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_HistogramCostExtractor_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".HistogramCostExtractor",
    sizeof(pyopencv_HistogramCostExtractor_t),
};

static void pyopencv_HistogramCostExtractor_dealloc(PyObject* self)
{
    ((pyopencv_HistogramCostExtractor_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::HistogramCostExtractor>& r)
{
    pyopencv_HistogramCostExtractor_t *m = PyObject_NEW(pyopencv_HistogramCostExtractor_t, &pyopencv_HistogramCostExtractor_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::HistogramCostExtractor>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_HistogramCostExtractor_Type))
    {
        failmsg("Expected cv::HistogramCostExtractor for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_HistogramCostExtractor_t*)src)->v.dynamicCast<cv::HistogramCostExtractor>();
    return true;
}


struct pyopencv_KAZE_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_KAZE_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".KAZE",
    sizeof(pyopencv_KAZE_t),
};

static void pyopencv_KAZE_dealloc(PyObject* self)
{
    ((pyopencv_KAZE_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::KAZE>& r)
{
    pyopencv_KAZE_t *m = PyObject_NEW(pyopencv_KAZE_t, &pyopencv_KAZE_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::KAZE>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_KAZE_Type))
    {
        failmsg("Expected cv::KAZE for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_KAZE_t*)src)->v.dynamicCast<cv::KAZE>();
    return true;
}


struct pyopencv_KalmanFilter_t
{
    PyObject_HEAD
    Ptr<cv::KalmanFilter> v;
};

static PyTypeObject pyopencv_KalmanFilter_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".KalmanFilter",
    sizeof(pyopencv_KalmanFilter_t),
};

static void pyopencv_KalmanFilter_dealloc(PyObject* self)
{
    ((pyopencv_KalmanFilter_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::KalmanFilter>& r)
{
    pyopencv_KalmanFilter_t *m = PyObject_NEW(pyopencv_KalmanFilter_t, &pyopencv_KalmanFilter_Type);
    new (&(m->v)) Ptr<cv::KalmanFilter>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::KalmanFilter>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_KalmanFilter_Type))
    {
        failmsg("Expected cv::KalmanFilter for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_KalmanFilter_t*)src)->v.dynamicCast<cv::KalmanFilter>();
    return true;
}


struct pyopencv_KeyPoint_t
{
    PyObject_HEAD
    cv::KeyPoint v;
};

static PyTypeObject pyopencv_KeyPoint_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".KeyPoint",
    sizeof(pyopencv_KeyPoint_t),
};

static void pyopencv_KeyPoint_dealloc(PyObject* self)
{
    ((pyopencv_KeyPoint_t*)self)->v.cv::KeyPoint::~KeyPoint();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const cv::KeyPoint& r)
{
    pyopencv_KeyPoint_t *m = PyObject_NEW(pyopencv_KeyPoint_t, &pyopencv_KeyPoint_Type);
    new (&m->v) cv::KeyPoint(r); //Copy constructor
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, cv::KeyPoint& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_KeyPoint_Type))
    {
        failmsg("Expected cv::KeyPoint for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_KeyPoint_t*)src)->v;
    return true;
}

struct pyopencv_LineSegmentDetector_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_LineSegmentDetector_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".LineSegmentDetector",
    sizeof(pyopencv_LineSegmentDetector_t),
};

static void pyopencv_LineSegmentDetector_dealloc(PyObject* self)
{
    ((pyopencv_LineSegmentDetector_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::LineSegmentDetector>& r)
{
    pyopencv_LineSegmentDetector_t *m = PyObject_NEW(pyopencv_LineSegmentDetector_t, &pyopencv_LineSegmentDetector_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::LineSegmentDetector>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_LineSegmentDetector_Type))
    {
        failmsg("Expected cv::LineSegmentDetector for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_LineSegmentDetector_t*)src)->v.dynamicCast<cv::LineSegmentDetector>();
    return true;
}


struct pyopencv_MSER_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_MSER_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".MSER",
    sizeof(pyopencv_MSER_t),
};

static void pyopencv_MSER_dealloc(PyObject* self)
{
    ((pyopencv_MSER_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::MSER>& r)
{
    pyopencv_MSER_t *m = PyObject_NEW(pyopencv_MSER_t, &pyopencv_MSER_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::MSER>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_MSER_Type))
    {
        failmsg("Expected cv::MSER for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_MSER_t*)src)->v.dynamicCast<cv::MSER>();
    return true;
}


struct pyopencv_MergeDebevec_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_MergeDebevec_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".MergeDebevec",
    sizeof(pyopencv_MergeDebevec_t),
};

static void pyopencv_MergeDebevec_dealloc(PyObject* self)
{
    ((pyopencv_MergeDebevec_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::MergeDebevec>& r)
{
    pyopencv_MergeDebevec_t *m = PyObject_NEW(pyopencv_MergeDebevec_t, &pyopencv_MergeDebevec_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::MergeDebevec>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_MergeDebevec_Type))
    {
        failmsg("Expected cv::MergeDebevec for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_MergeDebevec_t*)src)->v.dynamicCast<cv::MergeDebevec>();
    return true;
}


struct pyopencv_MergeExposures_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_MergeExposures_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".MergeExposures",
    sizeof(pyopencv_MergeExposures_t),
};

static void pyopencv_MergeExposures_dealloc(PyObject* self)
{
    ((pyopencv_MergeExposures_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::MergeExposures>& r)
{
    pyopencv_MergeExposures_t *m = PyObject_NEW(pyopencv_MergeExposures_t, &pyopencv_MergeExposures_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::MergeExposures>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_MergeExposures_Type))
    {
        failmsg("Expected cv::MergeExposures for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_MergeExposures_t*)src)->v.dynamicCast<cv::MergeExposures>();
    return true;
}


struct pyopencv_MergeMertens_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_MergeMertens_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".MergeMertens",
    sizeof(pyopencv_MergeMertens_t),
};

static void pyopencv_MergeMertens_dealloc(PyObject* self)
{
    ((pyopencv_MergeMertens_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::MergeMertens>& r)
{
    pyopencv_MergeMertens_t *m = PyObject_NEW(pyopencv_MergeMertens_t, &pyopencv_MergeMertens_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::MergeMertens>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_MergeMertens_Type))
    {
        failmsg("Expected cv::MergeMertens for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_MergeMertens_t*)src)->v.dynamicCast<cv::MergeMertens>();
    return true;
}


struct pyopencv_MergeRobertson_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_MergeRobertson_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".MergeRobertson",
    sizeof(pyopencv_MergeRobertson_t),
};

static void pyopencv_MergeRobertson_dealloc(PyObject* self)
{
    ((pyopencv_MergeRobertson_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::MergeRobertson>& r)
{
    pyopencv_MergeRobertson_t *m = PyObject_NEW(pyopencv_MergeRobertson_t, &pyopencv_MergeRobertson_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::MergeRobertson>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_MergeRobertson_Type))
    {
        failmsg("Expected cv::MergeRobertson for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_MergeRobertson_t*)src)->v.dynamicCast<cv::MergeRobertson>();
    return true;
}


template<> bool pyopencv_to(PyObject* src, cv::Moments& dst, const char* name);

struct pyopencv_NormHistogramCostExtractor_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_NormHistogramCostExtractor_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".NormHistogramCostExtractor",
    sizeof(pyopencv_NormHistogramCostExtractor_t),
};

static void pyopencv_NormHistogramCostExtractor_dealloc(PyObject* self)
{
    ((pyopencv_NormHistogramCostExtractor_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::NormHistogramCostExtractor>& r)
{
    pyopencv_NormHistogramCostExtractor_t *m = PyObject_NEW(pyopencv_NormHistogramCostExtractor_t, &pyopencv_NormHistogramCostExtractor_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::NormHistogramCostExtractor>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_NormHistogramCostExtractor_Type))
    {
        failmsg("Expected cv::NormHistogramCostExtractor for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_NormHistogramCostExtractor_t*)src)->v.dynamicCast<cv::NormHistogramCostExtractor>();
    return true;
}


struct pyopencv_ORB_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_ORB_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".ORB",
    sizeof(pyopencv_ORB_t),
};

static void pyopencv_ORB_dealloc(PyObject* self)
{
    ((pyopencv_ORB_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::ORB>& r)
{
    pyopencv_ORB_t *m = PyObject_NEW(pyopencv_ORB_t, &pyopencv_ORB_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::ORB>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_ORB_Type))
    {
        failmsg("Expected cv::ORB for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_ORB_t*)src)->v.dynamicCast<cv::ORB>();
    return true;
}


struct pyopencv_ShapeContextDistanceExtractor_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_ShapeContextDistanceExtractor_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".ShapeContextDistanceExtractor",
    sizeof(pyopencv_ShapeContextDistanceExtractor_t),
};

static void pyopencv_ShapeContextDistanceExtractor_dealloc(PyObject* self)
{
    ((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::ShapeContextDistanceExtractor>& r)
{
    pyopencv_ShapeContextDistanceExtractor_t *m = PyObject_NEW(pyopencv_ShapeContextDistanceExtractor_t, &pyopencv_ShapeContextDistanceExtractor_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::ShapeContextDistanceExtractor>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_ShapeContextDistanceExtractor_Type))
    {
        failmsg("Expected cv::ShapeContextDistanceExtractor for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_ShapeContextDistanceExtractor_t*)src)->v.dynamicCast<cv::ShapeContextDistanceExtractor>();
    return true;
}


struct pyopencv_ShapeDistanceExtractor_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_ShapeDistanceExtractor_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".ShapeDistanceExtractor",
    sizeof(pyopencv_ShapeDistanceExtractor_t),
};

static void pyopencv_ShapeDistanceExtractor_dealloc(PyObject* self)
{
    ((pyopencv_ShapeDistanceExtractor_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::ShapeDistanceExtractor>& r)
{
    pyopencv_ShapeDistanceExtractor_t *m = PyObject_NEW(pyopencv_ShapeDistanceExtractor_t, &pyopencv_ShapeDistanceExtractor_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::ShapeDistanceExtractor>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_ShapeDistanceExtractor_Type))
    {
        failmsg("Expected cv::ShapeDistanceExtractor for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_ShapeDistanceExtractor_t*)src)->v.dynamicCast<cv::ShapeDistanceExtractor>();
    return true;
}


struct pyopencv_ShapeTransformer_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_ShapeTransformer_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".ShapeTransformer",
    sizeof(pyopencv_ShapeTransformer_t),
};

static void pyopencv_ShapeTransformer_dealloc(PyObject* self)
{
    ((pyopencv_ShapeTransformer_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::ShapeTransformer>& r)
{
    pyopencv_ShapeTransformer_t *m = PyObject_NEW(pyopencv_ShapeTransformer_t, &pyopencv_ShapeTransformer_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::ShapeTransformer>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_ShapeTransformer_Type))
    {
        failmsg("Expected cv::ShapeTransformer for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_ShapeTransformer_t*)src)->v.dynamicCast<cv::ShapeTransformer>();
    return true;
}


struct pyopencv_SimpleBlobDetector_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_SimpleBlobDetector_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".SimpleBlobDetector",
    sizeof(pyopencv_SimpleBlobDetector_t),
};

static void pyopencv_SimpleBlobDetector_dealloc(PyObject* self)
{
    ((pyopencv_SimpleBlobDetector_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::SimpleBlobDetector>& r)
{
    pyopencv_SimpleBlobDetector_t *m = PyObject_NEW(pyopencv_SimpleBlobDetector_t, &pyopencv_SimpleBlobDetector_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::SimpleBlobDetector>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_SimpleBlobDetector_Type))
    {
        failmsg("Expected cv::SimpleBlobDetector for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_SimpleBlobDetector_t*)src)->v.dynamicCast<cv::SimpleBlobDetector>();
    return true;
}


struct pyopencv_SimpleBlobDetector_Params_t
{
    PyObject_HEAD
    cv::SimpleBlobDetector::Params v;
};

static PyTypeObject pyopencv_SimpleBlobDetector_Params_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".SimpleBlobDetector_Params",
    sizeof(pyopencv_SimpleBlobDetector_Params_t),
};

static void pyopencv_SimpleBlobDetector_Params_dealloc(PyObject* self)
{
    ((pyopencv_SimpleBlobDetector_Params_t*)self)->v.cv::SimpleBlobDetector::Params::~Params();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const cv::SimpleBlobDetector::Params& r)
{
    pyopencv_SimpleBlobDetector_Params_t *m = PyObject_NEW(pyopencv_SimpleBlobDetector_Params_t, &pyopencv_SimpleBlobDetector_Params_Type);
    new (&m->v) cv::SimpleBlobDetector::Params(r); //Copy constructor
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, cv::SimpleBlobDetector::Params& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_SimpleBlobDetector_Params_Type))
    {
        failmsg("Expected cv::SimpleBlobDetector::Params for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_SimpleBlobDetector_Params_t*)src)->v;
    return true;
}

struct pyopencv_SparseOpticalFlow_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_SparseOpticalFlow_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".SparseOpticalFlow",
    sizeof(pyopencv_SparseOpticalFlow_t),
};

static void pyopencv_SparseOpticalFlow_dealloc(PyObject* self)
{
    ((pyopencv_SparseOpticalFlow_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::SparseOpticalFlow>& r)
{
    pyopencv_SparseOpticalFlow_t *m = PyObject_NEW(pyopencv_SparseOpticalFlow_t, &pyopencv_SparseOpticalFlow_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::SparseOpticalFlow>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_SparseOpticalFlow_Type))
    {
        failmsg("Expected cv::SparseOpticalFlow for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_SparseOpticalFlow_t*)src)->v.dynamicCast<cv::SparseOpticalFlow>();
    return true;
}


struct pyopencv_SparsePyrLKOpticalFlow_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_SparsePyrLKOpticalFlow_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".SparsePyrLKOpticalFlow",
    sizeof(pyopencv_SparsePyrLKOpticalFlow_t),
};

static void pyopencv_SparsePyrLKOpticalFlow_dealloc(PyObject* self)
{
    ((pyopencv_SparsePyrLKOpticalFlow_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::SparsePyrLKOpticalFlow>& r)
{
    pyopencv_SparsePyrLKOpticalFlow_t *m = PyObject_NEW(pyopencv_SparsePyrLKOpticalFlow_t, &pyopencv_SparsePyrLKOpticalFlow_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::SparsePyrLKOpticalFlow>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_SparsePyrLKOpticalFlow_Type))
    {
        failmsg("Expected cv::SparsePyrLKOpticalFlow for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_SparsePyrLKOpticalFlow_t*)src)->v.dynamicCast<cv::SparsePyrLKOpticalFlow>();
    return true;
}


struct pyopencv_StereoBM_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_StereoBM_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".StereoBM",
    sizeof(pyopencv_StereoBM_t),
};

static void pyopencv_StereoBM_dealloc(PyObject* self)
{
    ((pyopencv_StereoBM_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::StereoBM>& r)
{
    pyopencv_StereoBM_t *m = PyObject_NEW(pyopencv_StereoBM_t, &pyopencv_StereoBM_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::StereoBM>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_StereoBM_Type))
    {
        failmsg("Expected cv::StereoBM for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_StereoBM_t*)src)->v.dynamicCast<cv::StereoBM>();
    return true;
}


struct pyopencv_StereoMatcher_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_StereoMatcher_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".StereoMatcher",
    sizeof(pyopencv_StereoMatcher_t),
};

static void pyopencv_StereoMatcher_dealloc(PyObject* self)
{
    ((pyopencv_StereoMatcher_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::StereoMatcher>& r)
{
    pyopencv_StereoMatcher_t *m = PyObject_NEW(pyopencv_StereoMatcher_t, &pyopencv_StereoMatcher_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::StereoMatcher>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_StereoMatcher_Type))
    {
        failmsg("Expected cv::StereoMatcher for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_StereoMatcher_t*)src)->v.dynamicCast<cv::StereoMatcher>();
    return true;
}


struct pyopencv_StereoSGBM_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_StereoSGBM_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".StereoSGBM",
    sizeof(pyopencv_StereoSGBM_t),
};

static void pyopencv_StereoSGBM_dealloc(PyObject* self)
{
    ((pyopencv_StereoSGBM_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::StereoSGBM>& r)
{
    pyopencv_StereoSGBM_t *m = PyObject_NEW(pyopencv_StereoSGBM_t, &pyopencv_StereoSGBM_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::StereoSGBM>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_StereoSGBM_Type))
    {
        failmsg("Expected cv::StereoSGBM for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_StereoSGBM_t*)src)->v.dynamicCast<cv::StereoSGBM>();
    return true;
}


struct pyopencv_Stitcher_t
{
    PyObject_HEAD
    Ptr<cv::Stitcher> v;
};

static PyTypeObject pyopencv_Stitcher_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".Stitcher",
    sizeof(pyopencv_Stitcher_t),
};

static void pyopencv_Stitcher_dealloc(PyObject* self)
{
    ((pyopencv_Stitcher_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::Stitcher>& r)
{
    pyopencv_Stitcher_t *m = PyObject_NEW(pyopencv_Stitcher_t, &pyopencv_Stitcher_Type);
    new (&(m->v)) Ptr<cv::Stitcher>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::Stitcher>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_Stitcher_Type))
    {
        failmsg("Expected cv::Stitcher for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_Stitcher_t*)src)->v.dynamicCast<cv::Stitcher>();
    return true;
}


struct pyopencv_Subdiv2D_t
{
    PyObject_HEAD
    Ptr<cv::Subdiv2D> v;
};

static PyTypeObject pyopencv_Subdiv2D_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".Subdiv2D",
    sizeof(pyopencv_Subdiv2D_t),
};

static void pyopencv_Subdiv2D_dealloc(PyObject* self)
{
    ((pyopencv_Subdiv2D_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::Subdiv2D>& r)
{
    pyopencv_Subdiv2D_t *m = PyObject_NEW(pyopencv_Subdiv2D_t, &pyopencv_Subdiv2D_Type);
    new (&(m->v)) Ptr<cv::Subdiv2D>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::Subdiv2D>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_Subdiv2D_Type))
    {
        failmsg("Expected cv::Subdiv2D for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_Subdiv2D_t*)src)->v.dynamicCast<cv::Subdiv2D>();
    return true;
}


struct pyopencv_ThinPlateSplineShapeTransformer_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_ThinPlateSplineShapeTransformer_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".ThinPlateSplineShapeTransformer",
    sizeof(pyopencv_ThinPlateSplineShapeTransformer_t),
};

static void pyopencv_ThinPlateSplineShapeTransformer_dealloc(PyObject* self)
{
    ((pyopencv_ThinPlateSplineShapeTransformer_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::ThinPlateSplineShapeTransformer>& r)
{
    pyopencv_ThinPlateSplineShapeTransformer_t *m = PyObject_NEW(pyopencv_ThinPlateSplineShapeTransformer_t, &pyopencv_ThinPlateSplineShapeTransformer_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::ThinPlateSplineShapeTransformer>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_ThinPlateSplineShapeTransformer_Type))
    {
        failmsg("Expected cv::ThinPlateSplineShapeTransformer for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_ThinPlateSplineShapeTransformer_t*)src)->v.dynamicCast<cv::ThinPlateSplineShapeTransformer>();
    return true;
}


struct pyopencv_TickMeter_t
{
    PyObject_HEAD
    Ptr<cv::TickMeter> v;
};

static PyTypeObject pyopencv_TickMeter_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".TickMeter",
    sizeof(pyopencv_TickMeter_t),
};

static void pyopencv_TickMeter_dealloc(PyObject* self)
{
    ((pyopencv_TickMeter_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::TickMeter>& r)
{
    pyopencv_TickMeter_t *m = PyObject_NEW(pyopencv_TickMeter_t, &pyopencv_TickMeter_Type);
    new (&(m->v)) Ptr<cv::TickMeter>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::TickMeter>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_TickMeter_Type))
    {
        failmsg("Expected cv::TickMeter for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_TickMeter_t*)src)->v.dynamicCast<cv::TickMeter>();
    return true;
}


struct pyopencv_Tonemap_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_Tonemap_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".Tonemap",
    sizeof(pyopencv_Tonemap_t),
};

static void pyopencv_Tonemap_dealloc(PyObject* self)
{
    ((pyopencv_Tonemap_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::Tonemap>& r)
{
    pyopencv_Tonemap_t *m = PyObject_NEW(pyopencv_Tonemap_t, &pyopencv_Tonemap_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::Tonemap>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_Tonemap_Type))
    {
        failmsg("Expected cv::Tonemap for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_Tonemap_t*)src)->v.dynamicCast<cv::Tonemap>();
    return true;
}


struct pyopencv_TonemapDrago_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_TonemapDrago_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".TonemapDrago",
    sizeof(pyopencv_TonemapDrago_t),
};

static void pyopencv_TonemapDrago_dealloc(PyObject* self)
{
    ((pyopencv_TonemapDrago_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::TonemapDrago>& r)
{
    pyopencv_TonemapDrago_t *m = PyObject_NEW(pyopencv_TonemapDrago_t, &pyopencv_TonemapDrago_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::TonemapDrago>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_TonemapDrago_Type))
    {
        failmsg("Expected cv::TonemapDrago for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_TonemapDrago_t*)src)->v.dynamicCast<cv::TonemapDrago>();
    return true;
}


struct pyopencv_TonemapDurand_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_TonemapDurand_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".TonemapDurand",
    sizeof(pyopencv_TonemapDurand_t),
};

static void pyopencv_TonemapDurand_dealloc(PyObject* self)
{
    ((pyopencv_TonemapDurand_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::TonemapDurand>& r)
{
    pyopencv_TonemapDurand_t *m = PyObject_NEW(pyopencv_TonemapDurand_t, &pyopencv_TonemapDurand_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::TonemapDurand>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_TonemapDurand_Type))
    {
        failmsg("Expected cv::TonemapDurand for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_TonemapDurand_t*)src)->v.dynamicCast<cv::TonemapDurand>();
    return true;
}


struct pyopencv_TonemapMantiuk_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_TonemapMantiuk_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".TonemapMantiuk",
    sizeof(pyopencv_TonemapMantiuk_t),
};

static void pyopencv_TonemapMantiuk_dealloc(PyObject* self)
{
    ((pyopencv_TonemapMantiuk_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::TonemapMantiuk>& r)
{
    pyopencv_TonemapMantiuk_t *m = PyObject_NEW(pyopencv_TonemapMantiuk_t, &pyopencv_TonemapMantiuk_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::TonemapMantiuk>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_TonemapMantiuk_Type))
    {
        failmsg("Expected cv::TonemapMantiuk for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_TonemapMantiuk_t*)src)->v.dynamicCast<cv::TonemapMantiuk>();
    return true;
}


struct pyopencv_TonemapReinhard_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_TonemapReinhard_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".TonemapReinhard",
    sizeof(pyopencv_TonemapReinhard_t),
};

static void pyopencv_TonemapReinhard_dealloc(PyObject* self)
{
    ((pyopencv_TonemapReinhard_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::TonemapReinhard>& r)
{
    pyopencv_TonemapReinhard_t *m = PyObject_NEW(pyopencv_TonemapReinhard_t, &pyopencv_TonemapReinhard_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::TonemapReinhard>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_TonemapReinhard_Type))
    {
        failmsg("Expected cv::TonemapReinhard for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_TonemapReinhard_t*)src)->v.dynamicCast<cv::TonemapReinhard>();
    return true;
}


struct pyopencv_VideoCapture_t
{
    PyObject_HEAD
    Ptr<cv::VideoCapture> v;
};

static PyTypeObject pyopencv_VideoCapture_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".VideoCapture",
    sizeof(pyopencv_VideoCapture_t),
};

static void pyopencv_VideoCapture_dealloc(PyObject* self)
{
    ((pyopencv_VideoCapture_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::VideoCapture>& r)
{
    pyopencv_VideoCapture_t *m = PyObject_NEW(pyopencv_VideoCapture_t, &pyopencv_VideoCapture_Type);
    new (&(m->v)) Ptr<cv::VideoCapture>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::VideoCapture>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_VideoCapture_Type))
    {
        failmsg("Expected cv::VideoCapture for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_VideoCapture_t*)src)->v.dynamicCast<cv::VideoCapture>();
    return true;
}


struct pyopencv_VideoWriter_t
{
    PyObject_HEAD
    Ptr<cv::VideoWriter> v;
};

static PyTypeObject pyopencv_VideoWriter_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".VideoWriter",
    sizeof(pyopencv_VideoWriter_t),
};

static void pyopencv_VideoWriter_dealloc(PyObject* self)
{
    ((pyopencv_VideoWriter_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::VideoWriter>& r)
{
    pyopencv_VideoWriter_t *m = PyObject_NEW(pyopencv_VideoWriter_t, &pyopencv_VideoWriter_Type);
    new (&(m->v)) Ptr<cv::VideoWriter>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::VideoWriter>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_VideoWriter_Type))
    {
        failmsg("Expected cv::VideoWriter for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_VideoWriter_t*)src)->v.dynamicCast<cv::VideoWriter>();
    return true;
}


struct pyopencv_flann_Index_t
{
    PyObject_HEAD
    Ptr<cv::flann::Index> v;
};

static PyTypeObject pyopencv_flann_Index_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".flann_Index",
    sizeof(pyopencv_flann_Index_t),
};

static void pyopencv_flann_Index_dealloc(PyObject* self)
{
    ((pyopencv_flann_Index_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::flann::Index>& r)
{
    pyopencv_flann_Index_t *m = PyObject_NEW(pyopencv_flann_Index_t, &pyopencv_flann_Index_Type);
    new (&(m->v)) Ptr<cv::flann::Index>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::flann::Index>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_flann_Index_Type))
    {
        failmsg("Expected cv::flann::Index for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_flann_Index_t*)src)->v.dynamicCast<cv::flann::Index>();
    return true;
}


struct pyopencv_ml_ANN_MLP_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_ml_ANN_MLP_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".ml_ANN_MLP",
    sizeof(pyopencv_ml_ANN_MLP_t),
};

static void pyopencv_ml_ANN_MLP_dealloc(PyObject* self)
{
    ((pyopencv_ml_ANN_MLP_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::ml::ANN_MLP>& r)
{
    pyopencv_ml_ANN_MLP_t *m = PyObject_NEW(pyopencv_ml_ANN_MLP_t, &pyopencv_ml_ANN_MLP_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::ml::ANN_MLP>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_ml_ANN_MLP_Type))
    {
        failmsg("Expected cv::ml::ANN_MLP for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_ml_ANN_MLP_t*)src)->v.dynamicCast<cv::ml::ANN_MLP>();
    return true;
}


struct pyopencv_ml_Boost_t
{
    PyObject_HEAD
    Ptr<cv::ml::Boost> v;
};

static PyTypeObject pyopencv_ml_Boost_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".ml_Boost",
    sizeof(pyopencv_ml_Boost_t),
};

static void pyopencv_ml_Boost_dealloc(PyObject* self)
{
    ((pyopencv_ml_Boost_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::ml::Boost>& r)
{
    pyopencv_ml_Boost_t *m = PyObject_NEW(pyopencv_ml_Boost_t, &pyopencv_ml_Boost_Type);
    new (&(m->v)) Ptr<cv::ml::Boost>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::ml::Boost>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_ml_Boost_Type))
    {
        failmsg("Expected cv::ml::Boost for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_ml_Boost_t*)src)->v.dynamicCast<cv::ml::Boost>();
    return true;
}


struct pyopencv_ml_DTrees_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_ml_DTrees_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".ml_DTrees",
    sizeof(pyopencv_ml_DTrees_t),
};

static void pyopencv_ml_DTrees_dealloc(PyObject* self)
{
    ((pyopencv_ml_DTrees_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::ml::DTrees>& r)
{
    pyopencv_ml_DTrees_t *m = PyObject_NEW(pyopencv_ml_DTrees_t, &pyopencv_ml_DTrees_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::ml::DTrees>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_ml_DTrees_Type))
    {
        failmsg("Expected cv::ml::DTrees for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_ml_DTrees_t*)src)->v.dynamicCast<cv::ml::DTrees>();
    return true;
}


struct pyopencv_ml_EM_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_ml_EM_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".ml_EM",
    sizeof(pyopencv_ml_EM_t),
};

static void pyopencv_ml_EM_dealloc(PyObject* self)
{
    ((pyopencv_ml_EM_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::ml::EM>& r)
{
    pyopencv_ml_EM_t *m = PyObject_NEW(pyopencv_ml_EM_t, &pyopencv_ml_EM_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::ml::EM>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_ml_EM_Type))
    {
        failmsg("Expected cv::ml::EM for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_ml_EM_t*)src)->v.dynamicCast<cv::ml::EM>();
    return true;
}


struct pyopencv_ml_KNearest_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_ml_KNearest_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".ml_KNearest",
    sizeof(pyopencv_ml_KNearest_t),
};

static void pyopencv_ml_KNearest_dealloc(PyObject* self)
{
    ((pyopencv_ml_KNearest_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::ml::KNearest>& r)
{
    pyopencv_ml_KNearest_t *m = PyObject_NEW(pyopencv_ml_KNearest_t, &pyopencv_ml_KNearest_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::ml::KNearest>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_ml_KNearest_Type))
    {
        failmsg("Expected cv::ml::KNearest for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_ml_KNearest_t*)src)->v.dynamicCast<cv::ml::KNearest>();
    return true;
}


struct pyopencv_ml_LogisticRegression_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_ml_LogisticRegression_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".ml_LogisticRegression",
    sizeof(pyopencv_ml_LogisticRegression_t),
};

static void pyopencv_ml_LogisticRegression_dealloc(PyObject* self)
{
    ((pyopencv_ml_LogisticRegression_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::ml::LogisticRegression>& r)
{
    pyopencv_ml_LogisticRegression_t *m = PyObject_NEW(pyopencv_ml_LogisticRegression_t, &pyopencv_ml_LogisticRegression_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::ml::LogisticRegression>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_ml_LogisticRegression_Type))
    {
        failmsg("Expected cv::ml::LogisticRegression for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_ml_LogisticRegression_t*)src)->v.dynamicCast<cv::ml::LogisticRegression>();
    return true;
}


struct pyopencv_ml_NormalBayesClassifier_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_ml_NormalBayesClassifier_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".ml_NormalBayesClassifier",
    sizeof(pyopencv_ml_NormalBayesClassifier_t),
};

static void pyopencv_ml_NormalBayesClassifier_dealloc(PyObject* self)
{
    ((pyopencv_ml_NormalBayesClassifier_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::ml::NormalBayesClassifier>& r)
{
    pyopencv_ml_NormalBayesClassifier_t *m = PyObject_NEW(pyopencv_ml_NormalBayesClassifier_t, &pyopencv_ml_NormalBayesClassifier_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::ml::NormalBayesClassifier>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_ml_NormalBayesClassifier_Type))
    {
        failmsg("Expected cv::ml::NormalBayesClassifier for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_ml_NormalBayesClassifier_t*)src)->v.dynamicCast<cv::ml::NormalBayesClassifier>();
    return true;
}


struct pyopencv_ml_RTrees_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_ml_RTrees_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".ml_RTrees",
    sizeof(pyopencv_ml_RTrees_t),
};

static void pyopencv_ml_RTrees_dealloc(PyObject* self)
{
    ((pyopencv_ml_RTrees_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::ml::RTrees>& r)
{
    pyopencv_ml_RTrees_t *m = PyObject_NEW(pyopencv_ml_RTrees_t, &pyopencv_ml_RTrees_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::ml::RTrees>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_ml_RTrees_Type))
    {
        failmsg("Expected cv::ml::RTrees for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_ml_RTrees_t*)src)->v.dynamicCast<cv::ml::RTrees>();
    return true;
}


struct pyopencv_ml_SVM_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_ml_SVM_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".ml_SVM",
    sizeof(pyopencv_ml_SVM_t),
};

static void pyopencv_ml_SVM_dealloc(PyObject* self)
{
    ((pyopencv_ml_SVM_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::ml::SVM>& r)
{
    pyopencv_ml_SVM_t *m = PyObject_NEW(pyopencv_ml_SVM_t, &pyopencv_ml_SVM_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::ml::SVM>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_ml_SVM_Type))
    {
        failmsg("Expected cv::ml::SVM for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_ml_SVM_t*)src)->v.dynamicCast<cv::ml::SVM>();
    return true;
}


struct pyopencv_ml_SVMSGD_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_ml_SVMSGD_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".ml_SVMSGD",
    sizeof(pyopencv_ml_SVMSGD_t),
};

static void pyopencv_ml_SVMSGD_dealloc(PyObject* self)
{
    ((pyopencv_ml_SVMSGD_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::ml::SVMSGD>& r)
{
    pyopencv_ml_SVMSGD_t *m = PyObject_NEW(pyopencv_ml_SVMSGD_t, &pyopencv_ml_SVMSGD_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::ml::SVMSGD>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_ml_SVMSGD_Type))
    {
        failmsg("Expected cv::ml::SVMSGD for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_ml_SVMSGD_t*)src)->v.dynamicCast<cv::ml::SVMSGD>();
    return true;
}


struct pyopencv_ml_StatModel_t
{
    PyObject_HEAD
    Ptr<cv::Algorithm> v;
};

static PyTypeObject pyopencv_ml_StatModel_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".ml_StatModel",
    sizeof(pyopencv_ml_StatModel_t),
};

static void pyopencv_ml_StatModel_dealloc(PyObject* self)
{
    ((pyopencv_ml_StatModel_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::ml::StatModel>& r)
{
    pyopencv_ml_StatModel_t *m = PyObject_NEW(pyopencv_ml_StatModel_t, &pyopencv_ml_StatModel_Type);
    new (&(m->v)) Ptr<cv::Algorithm>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::ml::StatModel>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_ml_StatModel_Type))
    {
        failmsg("Expected cv::ml::StatModel for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_ml_StatModel_t*)src)->v.dynamicCast<cv::ml::StatModel>();
    return true;
}


struct pyopencv_ml_TrainData_t
{
    PyObject_HEAD
    Ptr<cv::ml::TrainData> v;
};

static PyTypeObject pyopencv_ml_TrainData_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
0,
    MODULESTR".ml_TrainData",
    sizeof(pyopencv_ml_TrainData_t),
};

static void pyopencv_ml_TrainData_dealloc(PyObject* self)
{
    ((pyopencv_ml_TrainData_t*)self)->v.release();
    PyObject_Del(self);
}

template<> PyObject* pyopencv_from(const Ptr<cv::ml::TrainData>& r)
{
    pyopencv_ml_TrainData_t *m = PyObject_NEW(pyopencv_ml_TrainData_t, &pyopencv_ml_TrainData_Type);
    new (&(m->v)) Ptr<cv::ml::TrainData>(); // init Ptr with placement new
    m->v = r;
    return (PyObject*)m;
}

template<> bool pyopencv_to(PyObject* src, Ptr<cv::ml::TrainData>& dst, const char* name)
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_ml_TrainData_Type))
    {
        failmsg("Expected cv::ml::TrainData for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_ml_TrainData_t*)src)->v.dynamicCast<cv::ml::TrainData>();
    return true;
}


static PyObject* pyopencv_Algorithm_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<Algorithm %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_Algorithm_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_Algorithm_clear(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Algorithm_Type))
        return failmsgp("Incorrect type of self (must be 'Algorithm' or its derivative)");
    cv::Algorithm* _self_ = ((pyopencv_Algorithm_t*)self)->v.get();

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(_self_->clear());
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_Algorithm_getDefaultName(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Algorithm_Type))
        return failmsgp("Incorrect type of self (must be 'Algorithm' or its derivative)");
    cv::Algorithm* _self_ = ((pyopencv_Algorithm_t*)self)->v.get();
    String retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getDefaultName());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_Algorithm_save(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Algorithm_Type))
        return failmsgp("Incorrect type of self (must be 'Algorithm' or its derivative)");
    cv::Algorithm* _self_ = ((pyopencv_Algorithm_t*)self)->v.get();
    PyObject* pyobj_filename = NULL;
    String filename;

    const char* keywords[] = { "filename", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:Algorithm.save", (char**)keywords, &pyobj_filename) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) )
    {
        ERRWRAP2(_self_->save(filename));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_Algorithm_methods[] =
{
    {"clear", (PyCFunction)pyopencv_cv_Algorithm_clear, METH_VARARGS | METH_KEYWORDS, "clear() -> None"},
    {"getDefaultName", (PyCFunction)pyopencv_cv_Algorithm_getDefaultName, METH_VARARGS | METH_KEYWORDS, "getDefaultName() -> retval"},
    {"save", (PyCFunction)pyopencv_cv_Algorithm_save, METH_VARARGS | METH_KEYWORDS, "save(filename) -> None"},

    {NULL,          NULL}
};

static void pyopencv_Algorithm_specials(void)
{
    pyopencv_Algorithm_Type.tp_base = NULL;
    pyopencv_Algorithm_Type.tp_dealloc = pyopencv_Algorithm_dealloc;
    pyopencv_Algorithm_Type.tp_repr = pyopencv_Algorithm_repr;
    pyopencv_Algorithm_Type.tp_getset = pyopencv_Algorithm_getseters;
    pyopencv_Algorithm_Type.tp_methods = pyopencv_Algorithm_methods;
}

static PyObject* pyopencv_FileStorage_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<FileStorage %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_FileStorage_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_FileStorage_getFirstTopLevelNode(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FileStorage_Type))
        return failmsgp("Incorrect type of self (must be 'FileStorage' or its derivative)");
    cv::FileStorage* _self_ = ((pyopencv_FileStorage_t*)self)->v.get();
    FileNode retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getFirstTopLevelNode());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FileStorage_getNode(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FileStorage_Type))
        return failmsgp("Incorrect type of self (must be 'FileStorage' or its derivative)");
    cv::FileStorage* _self_ = ((pyopencv_FileStorage_t*)self)->v.get();
    char* nodename=(char*)"";
    FileNode retval;

    const char* keywords[] = { "nodename", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "s:FileStorage.getNode", (char**)keywords, &nodename) )
    {
        ERRWRAP2(retval = _self_->operator[](nodename));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FileStorage_isOpened(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FileStorage_Type))
        return failmsgp("Incorrect type of self (must be 'FileStorage' or its derivative)");
    cv::FileStorage* _self_ = ((pyopencv_FileStorage_t*)self)->v.get();
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->isOpened());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FileStorage_open(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FileStorage_Type))
        return failmsgp("Incorrect type of self (must be 'FileStorage' or its derivative)");
    cv::FileStorage* _self_ = ((pyopencv_FileStorage_t*)self)->v.get();
    PyObject* pyobj_filename = NULL;
    String filename;
    int flags=0;
    PyObject* pyobj_encoding = NULL;
    String encoding;
    bool retval;

    const char* keywords[] = { "filename", "flags", "encoding", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|O:FileStorage.open", (char**)keywords, &pyobj_filename, &flags, &pyobj_encoding) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) &&
        pyopencv_to(pyobj_encoding, encoding, ArgInfo("encoding", 0)) )
    {
        ERRWRAP2(retval = _self_->open(filename, flags, encoding));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FileStorage_release(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FileStorage_Type))
        return failmsgp("Incorrect type of self (must be 'FileStorage' or its derivative)");
    cv::FileStorage* _self_ = ((pyopencv_FileStorage_t*)self)->v.get();

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(_self_->release());
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_FileStorage_releaseAndGetString(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FileStorage_Type))
        return failmsgp("Incorrect type of self (must be 'FileStorage' or its derivative)");
    cv::FileStorage* _self_ = ((pyopencv_FileStorage_t*)self)->v.get();
    String retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->releaseAndGetString());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FileStorage_root(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FileStorage_Type))
        return failmsgp("Incorrect type of self (must be 'FileStorage' or its derivative)");
    cv::FileStorage* _self_ = ((pyopencv_FileStorage_t*)self)->v.get();
    int streamidx=0;
    FileNode retval;

    const char* keywords[] = { "streamidx", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|i:FileStorage.root", (char**)keywords, &streamidx) )
    {
        ERRWRAP2(retval = _self_->root(streamidx));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FileStorage_write(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FileStorage_Type))
        return failmsgp("Incorrect type of self (must be 'FileStorage' or its derivative)");
    cv::FileStorage* _self_ = ((pyopencv_FileStorage_t*)self)->v.get();
    {
    PyObject* pyobj_name = NULL;
    String name;
    double val=0;

    const char* keywords[] = { "name", "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Od:FileStorage.write", (char**)keywords, &pyobj_name, &val) &&
        pyopencv_to(pyobj_name, name, ArgInfo("name", 0)) )
    {
        ERRWRAP2(_self_->write(name, val));
        Py_RETURN_NONE;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_name = NULL;
    String name;
    PyObject* pyobj_val = NULL;
    String val;

    const char* keywords[] = { "name", "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:FileStorage.write", (char**)keywords, &pyobj_name, &pyobj_val) &&
        pyopencv_to(pyobj_name, name, ArgInfo("name", 0)) &&
        pyopencv_to(pyobj_val, val, ArgInfo("val", 0)) )
    {
        ERRWRAP2(_self_->write(name, val));
        Py_RETURN_NONE;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_name = NULL;
    String name;
    PyObject* pyobj_val = NULL;
    Mat val;

    const char* keywords[] = { "name", "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:FileStorage.write", (char**)keywords, &pyobj_name, &pyobj_val) &&
        pyopencv_to(pyobj_name, name, ArgInfo("name", 0)) &&
        pyopencv_to(pyobj_val, val, ArgInfo("val", 0)) )
    {
        ERRWRAP2(_self_->write(name, val));
        Py_RETURN_NONE;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_name = NULL;
    String name;
    PyObject* pyobj_val = NULL;
    UMat val;

    const char* keywords[] = { "name", "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:FileStorage.write", (char**)keywords, &pyobj_name, &pyobj_val) &&
        pyopencv_to(pyobj_name, name, ArgInfo("name", 0)) &&
        pyopencv_to(pyobj_val, val, ArgInfo("val", 0)) )
    {
        ERRWRAP2(_self_->write(name, val));
        Py_RETURN_NONE;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_FileStorage_writeComment(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FileStorage_Type))
        return failmsgp("Incorrect type of self (must be 'FileStorage' or its derivative)");
    cv::FileStorage* _self_ = ((pyopencv_FileStorage_t*)self)->v.get();
    PyObject* pyobj_comment = NULL;
    String comment;
    bool append=false;

    const char* keywords[] = { "comment", "append", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|b:FileStorage.writeComment", (char**)keywords, &pyobj_comment, &append) &&
        pyopencv_to(pyobj_comment, comment, ArgInfo("comment", 0)) )
    {
        ERRWRAP2(_self_->writeComment(comment, append));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_FileStorage_methods[] =
{
    {"getFirstTopLevelNode", (PyCFunction)pyopencv_cv_FileStorage_getFirstTopLevelNode, METH_VARARGS | METH_KEYWORDS, "getFirstTopLevelNode() -> retval"},
    {"getNode", (PyCFunction)pyopencv_cv_FileStorage_getNode, METH_VARARGS | METH_KEYWORDS, "getNode(nodename) -> retval"},
    {"isOpened", (PyCFunction)pyopencv_cv_FileStorage_isOpened, METH_VARARGS | METH_KEYWORDS, "isOpened() -> retval"},
    {"open", (PyCFunction)pyopencv_cv_FileStorage_open, METH_VARARGS | METH_KEYWORDS, "open(filename, flags[, encoding]) -> retval"},
    {"release", (PyCFunction)pyopencv_cv_FileStorage_release, METH_VARARGS | METH_KEYWORDS, "release() -> None"},
    {"releaseAndGetString", (PyCFunction)pyopencv_cv_FileStorage_releaseAndGetString, METH_VARARGS | METH_KEYWORDS, "releaseAndGetString() -> retval"},
    {"root", (PyCFunction)pyopencv_cv_FileStorage_root, METH_VARARGS | METH_KEYWORDS, "root([, streamidx]) -> retval"},
    {"write", (PyCFunction)pyopencv_cv_FileStorage_write, METH_VARARGS | METH_KEYWORDS, "write(name, val) -> None"},
    {"writeComment", (PyCFunction)pyopencv_cv_FileStorage_writeComment, METH_VARARGS | METH_KEYWORDS, "writeComment(comment[, append]) -> None"},

    {NULL,          NULL}
};

static void pyopencv_FileStorage_specials(void)
{
    pyopencv_FileStorage_Type.tp_base = NULL;
    pyopencv_FileStorage_Type.tp_dealloc = pyopencv_FileStorage_dealloc;
    pyopencv_FileStorage_Type.tp_repr = pyopencv_FileStorage_repr;
    pyopencv_FileStorage_Type.tp_getset = pyopencv_FileStorage_getseters;
    pyopencv_FileStorage_Type.tp_methods = pyopencv_FileStorage_methods;
}

static PyObject* pyopencv_FileNode_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<FileNode %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_FileNode_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_FileNode_at(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FileNode_Type))
        return failmsgp("Incorrect type of self (must be 'FileNode' or its derivative)");
    cv::FileNode* _self_ = &((pyopencv_FileNode_t*)self)->v;
    int i=0;
    FileNode retval;

    const char* keywords[] = { "i", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:FileNode.at", (char**)keywords, &i) )
    {
        ERRWRAP2(retval = _self_->operator[](i));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FileNode_empty(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FileNode_Type))
        return failmsgp("Incorrect type of self (must be 'FileNode' or its derivative)");
    cv::FileNode* _self_ = &((pyopencv_FileNode_t*)self)->v;
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->empty());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FileNode_getNode(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FileNode_Type))
        return failmsgp("Incorrect type of self (must be 'FileNode' or its derivative)");
    cv::FileNode* _self_ = &((pyopencv_FileNode_t*)self)->v;
    char* nodename=(char*)"";
    FileNode retval;

    const char* keywords[] = { "nodename", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "s:FileNode.getNode", (char**)keywords, &nodename) )
    {
        ERRWRAP2(retval = _self_->operator[](nodename));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FileNode_isInt(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FileNode_Type))
        return failmsgp("Incorrect type of self (must be 'FileNode' or its derivative)");
    cv::FileNode* _self_ = &((pyopencv_FileNode_t*)self)->v;
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->isInt());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FileNode_isMap(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FileNode_Type))
        return failmsgp("Incorrect type of self (must be 'FileNode' or its derivative)");
    cv::FileNode* _self_ = &((pyopencv_FileNode_t*)self)->v;
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->isMap());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FileNode_isNamed(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FileNode_Type))
        return failmsgp("Incorrect type of self (must be 'FileNode' or its derivative)");
    cv::FileNode* _self_ = &((pyopencv_FileNode_t*)self)->v;
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->isNamed());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FileNode_isNone(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FileNode_Type))
        return failmsgp("Incorrect type of self (must be 'FileNode' or its derivative)");
    cv::FileNode* _self_ = &((pyopencv_FileNode_t*)self)->v;
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->isNone());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FileNode_isReal(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FileNode_Type))
        return failmsgp("Incorrect type of self (must be 'FileNode' or its derivative)");
    cv::FileNode* _self_ = &((pyopencv_FileNode_t*)self)->v;
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->isReal());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FileNode_isSeq(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FileNode_Type))
        return failmsgp("Incorrect type of self (must be 'FileNode' or its derivative)");
    cv::FileNode* _self_ = &((pyopencv_FileNode_t*)self)->v;
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->isSeq());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FileNode_isString(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FileNode_Type))
        return failmsgp("Incorrect type of self (must be 'FileNode' or its derivative)");
    cv::FileNode* _self_ = &((pyopencv_FileNode_t*)self)->v;
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->isString());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FileNode_mat(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FileNode_Type))
        return failmsgp("Incorrect type of self (must be 'FileNode' or its derivative)");
    cv::FileNode* _self_ = &((pyopencv_FileNode_t*)self)->v;
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->mat());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FileNode_name(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FileNode_Type))
        return failmsgp("Incorrect type of self (must be 'FileNode' or its derivative)");
    cv::FileNode* _self_ = &((pyopencv_FileNode_t*)self)->v;
    String retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->name());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FileNode_real(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FileNode_Type))
        return failmsgp("Incorrect type of self (must be 'FileNode' or its derivative)");
    cv::FileNode* _self_ = &((pyopencv_FileNode_t*)self)->v;
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->real());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FileNode_size(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FileNode_Type))
        return failmsgp("Incorrect type of self (must be 'FileNode' or its derivative)");
    cv::FileNode* _self_ = &((pyopencv_FileNode_t*)self)->v;
    size_t retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->size());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FileNode_string(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FileNode_Type))
        return failmsgp("Incorrect type of self (must be 'FileNode' or its derivative)");
    cv::FileNode* _self_ = &((pyopencv_FileNode_t*)self)->v;
    String retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->string());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FileNode_type(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FileNode_Type))
        return failmsgp("Incorrect type of self (must be 'FileNode' or its derivative)");
    cv::FileNode* _self_ = &((pyopencv_FileNode_t*)self)->v;
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->type());
        return pyopencv_from(retval);
    }

    return NULL;
}



static PyMethodDef pyopencv_FileNode_methods[] =
{
    {"at", (PyCFunction)pyopencv_cv_FileNode_at, METH_VARARGS | METH_KEYWORDS, "at(i) -> retval"},
    {"empty", (PyCFunction)pyopencv_cv_FileNode_empty, METH_VARARGS | METH_KEYWORDS, "empty() -> retval"},
    {"getNode", (PyCFunction)pyopencv_cv_FileNode_getNode, METH_VARARGS | METH_KEYWORDS, "getNode(nodename) -> retval"},
    {"isInt", (PyCFunction)pyopencv_cv_FileNode_isInt, METH_VARARGS | METH_KEYWORDS, "isInt() -> retval"},
    {"isMap", (PyCFunction)pyopencv_cv_FileNode_isMap, METH_VARARGS | METH_KEYWORDS, "isMap() -> retval"},
    {"isNamed", (PyCFunction)pyopencv_cv_FileNode_isNamed, METH_VARARGS | METH_KEYWORDS, "isNamed() -> retval"},
    {"isNone", (PyCFunction)pyopencv_cv_FileNode_isNone, METH_VARARGS | METH_KEYWORDS, "isNone() -> retval"},
    {"isReal", (PyCFunction)pyopencv_cv_FileNode_isReal, METH_VARARGS | METH_KEYWORDS, "isReal() -> retval"},
    {"isSeq", (PyCFunction)pyopencv_cv_FileNode_isSeq, METH_VARARGS | METH_KEYWORDS, "isSeq() -> retval"},
    {"isString", (PyCFunction)pyopencv_cv_FileNode_isString, METH_VARARGS | METH_KEYWORDS, "isString() -> retval"},
    {"mat", (PyCFunction)pyopencv_cv_FileNode_mat, METH_VARARGS | METH_KEYWORDS, "mat() -> retval"},
    {"name", (PyCFunction)pyopencv_cv_FileNode_name, METH_VARARGS | METH_KEYWORDS, "name() -> retval"},
    {"real", (PyCFunction)pyopencv_cv_FileNode_real, METH_VARARGS | METH_KEYWORDS, "real() -> retval"},
    {"size", (PyCFunction)pyopencv_cv_FileNode_size, METH_VARARGS | METH_KEYWORDS, "size() -> retval"},
    {"string", (PyCFunction)pyopencv_cv_FileNode_string, METH_VARARGS | METH_KEYWORDS, "string() -> retval"},
    {"type", (PyCFunction)pyopencv_cv_FileNode_type, METH_VARARGS | METH_KEYWORDS, "type() -> retval"},

    {NULL,          NULL}
};

static void pyopencv_FileNode_specials(void)
{
    pyopencv_FileNode_Type.tp_base = NULL;
    pyopencv_FileNode_Type.tp_dealloc = pyopencv_FileNode_dealloc;
    pyopencv_FileNode_Type.tp_repr = pyopencv_FileNode_repr;
    pyopencv_FileNode_Type.tp_getset = pyopencv_FileNode_getseters;
    pyopencv_FileNode_Type.tp_methods = pyopencv_FileNode_methods;
}

static PyObject* pyopencv_TickMeter_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<TickMeter %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_TickMeter_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_TickMeter_getCounter(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TickMeter_Type))
        return failmsgp("Incorrect type of self (must be 'TickMeter' or its derivative)");
    cv::TickMeter* _self_ = ((pyopencv_TickMeter_t*)self)->v.get();
    int64 retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getCounter());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_TickMeter_getTimeMicro(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TickMeter_Type))
        return failmsgp("Incorrect type of self (must be 'TickMeter' or its derivative)");
    cv::TickMeter* _self_ = ((pyopencv_TickMeter_t*)self)->v.get();
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTimeMicro());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_TickMeter_getTimeMilli(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TickMeter_Type))
        return failmsgp("Incorrect type of self (must be 'TickMeter' or its derivative)");
    cv::TickMeter* _self_ = ((pyopencv_TickMeter_t*)self)->v.get();
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTimeMilli());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_TickMeter_getTimeSec(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TickMeter_Type))
        return failmsgp("Incorrect type of self (must be 'TickMeter' or its derivative)");
    cv::TickMeter* _self_ = ((pyopencv_TickMeter_t*)self)->v.get();
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTimeSec());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_TickMeter_getTimeTicks(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TickMeter_Type))
        return failmsgp("Incorrect type of self (must be 'TickMeter' or its derivative)");
    cv::TickMeter* _self_ = ((pyopencv_TickMeter_t*)self)->v.get();
    int64 retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTimeTicks());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_TickMeter_reset(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TickMeter_Type))
        return failmsgp("Incorrect type of self (must be 'TickMeter' or its derivative)");
    cv::TickMeter* _self_ = ((pyopencv_TickMeter_t*)self)->v.get();

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(_self_->reset());
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_TickMeter_start(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TickMeter_Type))
        return failmsgp("Incorrect type of self (must be 'TickMeter' or its derivative)");
    cv::TickMeter* _self_ = ((pyopencv_TickMeter_t*)self)->v.get();

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(_self_->start());
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_TickMeter_stop(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TickMeter_Type))
        return failmsgp("Incorrect type of self (must be 'TickMeter' or its derivative)");
    cv::TickMeter* _self_ = ((pyopencv_TickMeter_t*)self)->v.get();

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(_self_->stop());
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_TickMeter_methods[] =
{
    {"getCounter", (PyCFunction)pyopencv_cv_TickMeter_getCounter, METH_VARARGS | METH_KEYWORDS, "getCounter() -> retval"},
    {"getTimeMicro", (PyCFunction)pyopencv_cv_TickMeter_getTimeMicro, METH_VARARGS | METH_KEYWORDS, "getTimeMicro() -> retval"},
    {"getTimeMilli", (PyCFunction)pyopencv_cv_TickMeter_getTimeMilli, METH_VARARGS | METH_KEYWORDS, "getTimeMilli() -> retval"},
    {"getTimeSec", (PyCFunction)pyopencv_cv_TickMeter_getTimeSec, METH_VARARGS | METH_KEYWORDS, "getTimeSec() -> retval"},
    {"getTimeTicks", (PyCFunction)pyopencv_cv_TickMeter_getTimeTicks, METH_VARARGS | METH_KEYWORDS, "getTimeTicks() -> retval"},
    {"reset", (PyCFunction)pyopencv_cv_TickMeter_reset, METH_VARARGS | METH_KEYWORDS, "reset() -> None"},
    {"start", (PyCFunction)pyopencv_cv_TickMeter_start, METH_VARARGS | METH_KEYWORDS, "start() -> None"},
    {"stop", (PyCFunction)pyopencv_cv_TickMeter_stop, METH_VARARGS | METH_KEYWORDS, "stop() -> None"},

    {NULL,          NULL}
};

static void pyopencv_TickMeter_specials(void)
{
    pyopencv_TickMeter_Type.tp_base = NULL;
    pyopencv_TickMeter_Type.tp_dealloc = pyopencv_TickMeter_dealloc;
    pyopencv_TickMeter_Type.tp_repr = pyopencv_TickMeter_repr;
    pyopencv_TickMeter_Type.tp_getset = pyopencv_TickMeter_getseters;
    pyopencv_TickMeter_Type.tp_methods = pyopencv_TickMeter_methods;
}

static PyObject* pyopencv_KeyPoint_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<KeyPoint %p>", self);
    return PyString_FromString(str);
}


static PyObject* pyopencv_KeyPoint_get_angle(pyopencv_KeyPoint_t* p, void *closure)
{
    return pyopencv_from(p->v.angle);
}

static int pyopencv_KeyPoint_set_angle(pyopencv_KeyPoint_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the angle attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.angle) ? 0 : -1;
}

static PyObject* pyopencv_KeyPoint_get_class_id(pyopencv_KeyPoint_t* p, void *closure)
{
    return pyopencv_from(p->v.class_id);
}

static int pyopencv_KeyPoint_set_class_id(pyopencv_KeyPoint_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the class_id attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.class_id) ? 0 : -1;
}

static PyObject* pyopencv_KeyPoint_get_octave(pyopencv_KeyPoint_t* p, void *closure)
{
    return pyopencv_from(p->v.octave);
}

static int pyopencv_KeyPoint_set_octave(pyopencv_KeyPoint_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the octave attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.octave) ? 0 : -1;
}

static PyObject* pyopencv_KeyPoint_get_pt(pyopencv_KeyPoint_t* p, void *closure)
{
    return pyopencv_from(p->v.pt);
}

static int pyopencv_KeyPoint_set_pt(pyopencv_KeyPoint_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the pt attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.pt) ? 0 : -1;
}

static PyObject* pyopencv_KeyPoint_get_response(pyopencv_KeyPoint_t* p, void *closure)
{
    return pyopencv_from(p->v.response);
}

static int pyopencv_KeyPoint_set_response(pyopencv_KeyPoint_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the response attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.response) ? 0 : -1;
}

static PyObject* pyopencv_KeyPoint_get_size(pyopencv_KeyPoint_t* p, void *closure)
{
    return pyopencv_from(p->v.size);
}

static int pyopencv_KeyPoint_set_size(pyopencv_KeyPoint_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the size attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.size) ? 0 : -1;
}


static PyGetSetDef pyopencv_KeyPoint_getseters[] =
{
    {(char*)"angle", (getter)pyopencv_KeyPoint_get_angle, (setter)pyopencv_KeyPoint_set_angle, (char*)"angle", NULL},
    {(char*)"class_id", (getter)pyopencv_KeyPoint_get_class_id, (setter)pyopencv_KeyPoint_set_class_id, (char*)"class_id", NULL},
    {(char*)"octave", (getter)pyopencv_KeyPoint_get_octave, (setter)pyopencv_KeyPoint_set_octave, (char*)"octave", NULL},
    {(char*)"pt", (getter)pyopencv_KeyPoint_get_pt, (setter)pyopencv_KeyPoint_set_pt, (char*)"pt", NULL},
    {(char*)"response", (getter)pyopencv_KeyPoint_get_response, (setter)pyopencv_KeyPoint_set_response, (char*)"response", NULL},
    {(char*)"size", (getter)pyopencv_KeyPoint_get_size, (setter)pyopencv_KeyPoint_set_size, (char*)"size", NULL},
    {NULL}  /* Sentinel */
};



static PyMethodDef pyopencv_KeyPoint_methods[] =
{

    {NULL,          NULL}
};

static void pyopencv_KeyPoint_specials(void)
{
    pyopencv_KeyPoint_Type.tp_base = NULL;
    pyopencv_KeyPoint_Type.tp_dealloc = pyopencv_KeyPoint_dealloc;
    pyopencv_KeyPoint_Type.tp_repr = pyopencv_KeyPoint_repr;
    pyopencv_KeyPoint_Type.tp_getset = pyopencv_KeyPoint_getseters;
    pyopencv_KeyPoint_Type.tp_methods = pyopencv_KeyPoint_methods;
}

static PyObject* pyopencv_DMatch_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<DMatch %p>", self);
    return PyString_FromString(str);
}


static PyObject* pyopencv_DMatch_get_distance(pyopencv_DMatch_t* p, void *closure)
{
    return pyopencv_from(p->v.distance);
}

static int pyopencv_DMatch_set_distance(pyopencv_DMatch_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the distance attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.distance) ? 0 : -1;
}

static PyObject* pyopencv_DMatch_get_imgIdx(pyopencv_DMatch_t* p, void *closure)
{
    return pyopencv_from(p->v.imgIdx);
}

static int pyopencv_DMatch_set_imgIdx(pyopencv_DMatch_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the imgIdx attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.imgIdx) ? 0 : -1;
}

static PyObject* pyopencv_DMatch_get_queryIdx(pyopencv_DMatch_t* p, void *closure)
{
    return pyopencv_from(p->v.queryIdx);
}

static int pyopencv_DMatch_set_queryIdx(pyopencv_DMatch_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the queryIdx attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.queryIdx) ? 0 : -1;
}

static PyObject* pyopencv_DMatch_get_trainIdx(pyopencv_DMatch_t* p, void *closure)
{
    return pyopencv_from(p->v.trainIdx);
}

static int pyopencv_DMatch_set_trainIdx(pyopencv_DMatch_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the trainIdx attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.trainIdx) ? 0 : -1;
}


static PyGetSetDef pyopencv_DMatch_getseters[] =
{
    {(char*)"distance", (getter)pyopencv_DMatch_get_distance, (setter)pyopencv_DMatch_set_distance, (char*)"distance", NULL},
    {(char*)"imgIdx", (getter)pyopencv_DMatch_get_imgIdx, (setter)pyopencv_DMatch_set_imgIdx, (char*)"imgIdx", NULL},
    {(char*)"queryIdx", (getter)pyopencv_DMatch_get_queryIdx, (setter)pyopencv_DMatch_set_queryIdx, (char*)"queryIdx", NULL},
    {(char*)"trainIdx", (getter)pyopencv_DMatch_get_trainIdx, (setter)pyopencv_DMatch_set_trainIdx, (char*)"trainIdx", NULL},
    {NULL}  /* Sentinel */
};



static PyMethodDef pyopencv_DMatch_methods[] =
{

    {NULL,          NULL}
};

static void pyopencv_DMatch_specials(void)
{
    pyopencv_DMatch_Type.tp_base = NULL;
    pyopencv_DMatch_Type.tp_dealloc = pyopencv_DMatch_dealloc;
    pyopencv_DMatch_Type.tp_repr = pyopencv_DMatch_repr;
    pyopencv_DMatch_Type.tp_getset = pyopencv_DMatch_getseters;
    pyopencv_DMatch_Type.tp_methods = pyopencv_DMatch_methods;
}
static bool pyopencv_to(PyObject* src, cv::Moments& dst, const char* name)
{
    PyObject* tmp;
    bool ok;

    if( PyMapping_HasKeyString(src, (char*)"m00") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"m00");
        ok = tmp && pyopencv_to(tmp, dst.m00);
        Py_DECREF(tmp);
        if(!ok) return false;
    }
    if( PyMapping_HasKeyString(src, (char*)"m10") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"m10");
        ok = tmp && pyopencv_to(tmp, dst.m10);
        Py_DECREF(tmp);
        if(!ok) return false;
    }
    if( PyMapping_HasKeyString(src, (char*)"m01") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"m01");
        ok = tmp && pyopencv_to(tmp, dst.m01);
        Py_DECREF(tmp);
        if(!ok) return false;
    }
    if( PyMapping_HasKeyString(src, (char*)"m20") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"m20");
        ok = tmp && pyopencv_to(tmp, dst.m20);
        Py_DECREF(tmp);
        if(!ok) return false;
    }
    if( PyMapping_HasKeyString(src, (char*)"m11") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"m11");
        ok = tmp && pyopencv_to(tmp, dst.m11);
        Py_DECREF(tmp);
        if(!ok) return false;
    }
    if( PyMapping_HasKeyString(src, (char*)"m02") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"m02");
        ok = tmp && pyopencv_to(tmp, dst.m02);
        Py_DECREF(tmp);
        if(!ok) return false;
    }
    if( PyMapping_HasKeyString(src, (char*)"m30") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"m30");
        ok = tmp && pyopencv_to(tmp, dst.m30);
        Py_DECREF(tmp);
        if(!ok) return false;
    }
    if( PyMapping_HasKeyString(src, (char*)"m21") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"m21");
        ok = tmp && pyopencv_to(tmp, dst.m21);
        Py_DECREF(tmp);
        if(!ok) return false;
    }
    if( PyMapping_HasKeyString(src, (char*)"m12") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"m12");
        ok = tmp && pyopencv_to(tmp, dst.m12);
        Py_DECREF(tmp);
        if(!ok) return false;
    }
    if( PyMapping_HasKeyString(src, (char*)"m03") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"m03");
        ok = tmp && pyopencv_to(tmp, dst.m03);
        Py_DECREF(tmp);
        if(!ok) return false;
    }
    if( PyMapping_HasKeyString(src, (char*)"mu20") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"mu20");
        ok = tmp && pyopencv_to(tmp, dst.mu20);
        Py_DECREF(tmp);
        if(!ok) return false;
    }
    if( PyMapping_HasKeyString(src, (char*)"mu11") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"mu11");
        ok = tmp && pyopencv_to(tmp, dst.mu11);
        Py_DECREF(tmp);
        if(!ok) return false;
    }
    if( PyMapping_HasKeyString(src, (char*)"mu02") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"mu02");
        ok = tmp && pyopencv_to(tmp, dst.mu02);
        Py_DECREF(tmp);
        if(!ok) return false;
    }
    if( PyMapping_HasKeyString(src, (char*)"mu30") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"mu30");
        ok = tmp && pyopencv_to(tmp, dst.mu30);
        Py_DECREF(tmp);
        if(!ok) return false;
    }
    if( PyMapping_HasKeyString(src, (char*)"mu21") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"mu21");
        ok = tmp && pyopencv_to(tmp, dst.mu21);
        Py_DECREF(tmp);
        if(!ok) return false;
    }
    if( PyMapping_HasKeyString(src, (char*)"mu12") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"mu12");
        ok = tmp && pyopencv_to(tmp, dst.mu12);
        Py_DECREF(tmp);
        if(!ok) return false;
    }
    if( PyMapping_HasKeyString(src, (char*)"mu03") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"mu03");
        ok = tmp && pyopencv_to(tmp, dst.mu03);
        Py_DECREF(tmp);
        if(!ok) return false;
    }
    if( PyMapping_HasKeyString(src, (char*)"nu20") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"nu20");
        ok = tmp && pyopencv_to(tmp, dst.nu20);
        Py_DECREF(tmp);
        if(!ok) return false;
    }
    if( PyMapping_HasKeyString(src, (char*)"nu11") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"nu11");
        ok = tmp && pyopencv_to(tmp, dst.nu11);
        Py_DECREF(tmp);
        if(!ok) return false;
    }
    if( PyMapping_HasKeyString(src, (char*)"nu02") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"nu02");
        ok = tmp && pyopencv_to(tmp, dst.nu02);
        Py_DECREF(tmp);
        if(!ok) return false;
    }
    if( PyMapping_HasKeyString(src, (char*)"nu30") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"nu30");
        ok = tmp && pyopencv_to(tmp, dst.nu30);
        Py_DECREF(tmp);
        if(!ok) return false;
    }
    if( PyMapping_HasKeyString(src, (char*)"nu21") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"nu21");
        ok = tmp && pyopencv_to(tmp, dst.nu21);
        Py_DECREF(tmp);
        if(!ok) return false;
    }
    if( PyMapping_HasKeyString(src, (char*)"nu12") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"nu12");
        ok = tmp && pyopencv_to(tmp, dst.nu12);
        Py_DECREF(tmp);
        if(!ok) return false;
    }
    if( PyMapping_HasKeyString(src, (char*)"nu03") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"nu03");
        ok = tmp && pyopencv_to(tmp, dst.nu03);
        Py_DECREF(tmp);
        if(!ok) return false;
    }
    return true;
}

static PyObject* pyopencv_flann_Index_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<flann_Index %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_flann_Index_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_flann_flann_Index_build(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::flann;

    if(!PyObject_TypeCheck(self, &pyopencv_flann_Index_Type))
        return failmsgp("Incorrect type of self (must be 'flann_Index' or its derivative)");
    cv::flann::Index* _self_ = ((pyopencv_flann_Index_t*)self)->v.get();
    {
    PyObject* pyobj_features = NULL;
    Mat features;
    PyObject* pyobj_params = NULL;
    IndexParams params;
    PyObject* pyobj_distType = NULL;
    cvflann_flann_distance_t distType=cvflann::FLANN_DIST_L2;

    const char* keywords[] = { "features", "params", "distType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:flann_Index.build", (char**)keywords, &pyobj_features, &pyobj_params, &pyobj_distType) &&
        pyopencv_to(pyobj_features, features, ArgInfo("features", 0)) &&
        pyopencv_to(pyobj_params, params, ArgInfo("params", 0)) &&
        pyopencv_to(pyobj_distType, distType, ArgInfo("distType", 0)) )
    {
        ERRWRAP2(_self_->build(features, params, distType));
        Py_RETURN_NONE;
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

    const char* keywords[] = { "features", "params", "distType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:flann_Index.build", (char**)keywords, &pyobj_features, &pyobj_params, &pyobj_distType) &&
        pyopencv_to(pyobj_features, features, ArgInfo("features", 0)) &&
        pyopencv_to(pyobj_params, params, ArgInfo("params", 0)) &&
        pyopencv_to(pyobj_distType, distType, ArgInfo("distType", 0)) )
    {
        ERRWRAP2(_self_->build(features, params, distType));
        Py_RETURN_NONE;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_flann_flann_Index_getAlgorithm(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::flann;

    if(!PyObject_TypeCheck(self, &pyopencv_flann_Index_Type))
        return failmsgp("Incorrect type of self (must be 'flann_Index' or its derivative)");
    cv::flann::Index* _self_ = ((pyopencv_flann_Index_t*)self)->v.get();
    cvflann::flann_algorithm_t retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getAlgorithm());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_flann_flann_Index_getDistance(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::flann;

    if(!PyObject_TypeCheck(self, &pyopencv_flann_Index_Type))
        return failmsgp("Incorrect type of self (must be 'flann_Index' or its derivative)");
    cv::flann::Index* _self_ = ((pyopencv_flann_Index_t*)self)->v.get();
    cvflann::flann_distance_t retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getDistance());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_flann_flann_Index_knnSearch(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::flann;

    if(!PyObject_TypeCheck(self, &pyopencv_flann_Index_Type))
        return failmsgp("Incorrect type of self (must be 'flann_Index' or its derivative)");
    cv::flann::Index* _self_ = ((pyopencv_flann_Index_t*)self)->v.get();
    {
    PyObject* pyobj_query = NULL;
    Mat query;
    PyObject* pyobj_indices = NULL;
    Mat indices;
    PyObject* pyobj_dists = NULL;
    Mat dists;
    int knn=0;
    PyObject* pyobj_params = NULL;
    SearchParams params;

    const char* keywords[] = { "query", "knn", "indices", "dists", "params", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|OOO:flann_Index.knnSearch", (char**)keywords, &pyobj_query, &knn, &pyobj_indices, &pyobj_dists, &pyobj_params) &&
        pyopencv_to(pyobj_query, query, ArgInfo("query", 0)) &&
        pyopencv_to(pyobj_indices, indices, ArgInfo("indices", 1)) &&
        pyopencv_to(pyobj_dists, dists, ArgInfo("dists", 1)) &&
        pyopencv_to(pyobj_params, params, ArgInfo("params", 0)) )
    {
        ERRWRAP2(_self_->knnSearch(query, indices, dists, knn, params));
        return Py_BuildValue("(NN)", pyopencv_from(indices), pyopencv_from(dists));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_query = NULL;
    UMat query;
    PyObject* pyobj_indices = NULL;
    UMat indices;
    PyObject* pyobj_dists = NULL;
    UMat dists;
    int knn=0;
    PyObject* pyobj_params = NULL;
    SearchParams params;

    const char* keywords[] = { "query", "knn", "indices", "dists", "params", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|OOO:flann_Index.knnSearch", (char**)keywords, &pyobj_query, &knn, &pyobj_indices, &pyobj_dists, &pyobj_params) &&
        pyopencv_to(pyobj_query, query, ArgInfo("query", 0)) &&
        pyopencv_to(pyobj_indices, indices, ArgInfo("indices", 1)) &&
        pyopencv_to(pyobj_dists, dists, ArgInfo("dists", 1)) &&
        pyopencv_to(pyobj_params, params, ArgInfo("params", 0)) )
    {
        ERRWRAP2(_self_->knnSearch(query, indices, dists, knn, params));
        return Py_BuildValue("(NN)", pyopencv_from(indices), pyopencv_from(dists));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_flann_flann_Index_load(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::flann;

    if(!PyObject_TypeCheck(self, &pyopencv_flann_Index_Type))
        return failmsgp("Incorrect type of self (must be 'flann_Index' or its derivative)");
    cv::flann::Index* _self_ = ((pyopencv_flann_Index_t*)self)->v.get();
    {
    PyObject* pyobj_features = NULL;
    Mat features;
    PyObject* pyobj_filename = NULL;
    String filename;
    bool retval;

    const char* keywords[] = { "features", "filename", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:flann_Index.load", (char**)keywords, &pyobj_features, &pyobj_filename) &&
        pyopencv_to(pyobj_features, features, ArgInfo("features", 0)) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) )
    {
        ERRWRAP2(retval = _self_->load(features, filename));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_features = NULL;
    UMat features;
    PyObject* pyobj_filename = NULL;
    String filename;
    bool retval;

    const char* keywords[] = { "features", "filename", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:flann_Index.load", (char**)keywords, &pyobj_features, &pyobj_filename) &&
        pyopencv_to(pyobj_features, features, ArgInfo("features", 0)) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) )
    {
        ERRWRAP2(retval = _self_->load(features, filename));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_flann_flann_Index_radiusSearch(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::flann;

    if(!PyObject_TypeCheck(self, &pyopencv_flann_Index_Type))
        return failmsgp("Incorrect type of self (must be 'flann_Index' or its derivative)");
    cv::flann::Index* _self_ = ((pyopencv_flann_Index_t*)self)->v.get();
    {
    PyObject* pyobj_query = NULL;
    Mat query;
    PyObject* pyobj_indices = NULL;
    Mat indices;
    PyObject* pyobj_dists = NULL;
    Mat dists;
    double radius=0;
    int maxResults=0;
    PyObject* pyobj_params = NULL;
    SearchParams params;
    int retval;

    const char* keywords[] = { "query", "radius", "maxResults", "indices", "dists", "params", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odi|OOO:flann_Index.radiusSearch", (char**)keywords, &pyobj_query, &radius, &maxResults, &pyobj_indices, &pyobj_dists, &pyobj_params) &&
        pyopencv_to(pyobj_query, query, ArgInfo("query", 0)) &&
        pyopencv_to(pyobj_indices, indices, ArgInfo("indices", 1)) &&
        pyopencv_to(pyobj_dists, dists, ArgInfo("dists", 1)) &&
        pyopencv_to(pyobj_params, params, ArgInfo("params", 0)) )
    {
        ERRWRAP2(retval = _self_->radiusSearch(query, indices, dists, radius, maxResults, params));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(indices), pyopencv_from(dists));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_query = NULL;
    UMat query;
    PyObject* pyobj_indices = NULL;
    UMat indices;
    PyObject* pyobj_dists = NULL;
    UMat dists;
    double radius=0;
    int maxResults=0;
    PyObject* pyobj_params = NULL;
    SearchParams params;
    int retval;

    const char* keywords[] = { "query", "radius", "maxResults", "indices", "dists", "params", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Odi|OOO:flann_Index.radiusSearch", (char**)keywords, &pyobj_query, &radius, &maxResults, &pyobj_indices, &pyobj_dists, &pyobj_params) &&
        pyopencv_to(pyobj_query, query, ArgInfo("query", 0)) &&
        pyopencv_to(pyobj_indices, indices, ArgInfo("indices", 1)) &&
        pyopencv_to(pyobj_dists, dists, ArgInfo("dists", 1)) &&
        pyopencv_to(pyobj_params, params, ArgInfo("params", 0)) )
    {
        ERRWRAP2(retval = _self_->radiusSearch(query, indices, dists, radius, maxResults, params));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(indices), pyopencv_from(dists));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_flann_flann_Index_release(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::flann;

    if(!PyObject_TypeCheck(self, &pyopencv_flann_Index_Type))
        return failmsgp("Incorrect type of self (must be 'flann_Index' or its derivative)");
    cv::flann::Index* _self_ = ((pyopencv_flann_Index_t*)self)->v.get();

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(_self_->release());
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_flann_flann_Index_save(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::flann;

    if(!PyObject_TypeCheck(self, &pyopencv_flann_Index_Type))
        return failmsgp("Incorrect type of self (must be 'flann_Index' or its derivative)");
    cv::flann::Index* _self_ = ((pyopencv_flann_Index_t*)self)->v.get();
    PyObject* pyobj_filename = NULL;
    String filename;

    const char* keywords[] = { "filename", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:flann_Index.save", (char**)keywords, &pyobj_filename) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) )
    {
        ERRWRAP2(_self_->save(filename));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_flann_Index_methods[] =
{
    {"build", (PyCFunction)pyopencv_cv_flann_flann_Index_build, METH_VARARGS | METH_KEYWORDS, "build(features, params[, distType]) -> None"},
    {"getAlgorithm", (PyCFunction)pyopencv_cv_flann_flann_Index_getAlgorithm, METH_VARARGS | METH_KEYWORDS, "getAlgorithm() -> retval"},
    {"getDistance", (PyCFunction)pyopencv_cv_flann_flann_Index_getDistance, METH_VARARGS | METH_KEYWORDS, "getDistance() -> retval"},
    {"knnSearch", (PyCFunction)pyopencv_cv_flann_flann_Index_knnSearch, METH_VARARGS | METH_KEYWORDS, "knnSearch(query, knn[, indices[, dists[, params]]]) -> indices, dists"},
    {"load", (PyCFunction)pyopencv_cv_flann_flann_Index_load, METH_VARARGS | METH_KEYWORDS, "load(features, filename) -> retval"},
    {"radiusSearch", (PyCFunction)pyopencv_cv_flann_flann_Index_radiusSearch, METH_VARARGS | METH_KEYWORDS, "radiusSearch(query, radius, maxResults[, indices[, dists[, params]]]) -> retval, indices, dists"},
    {"release", (PyCFunction)pyopencv_cv_flann_flann_Index_release, METH_VARARGS | METH_KEYWORDS, "release() -> None"},
    {"save", (PyCFunction)pyopencv_cv_flann_flann_Index_save, METH_VARARGS | METH_KEYWORDS, "save(filename) -> None"},

    {NULL,          NULL}
};

static void pyopencv_flann_Index_specials(void)
{
    pyopencv_flann_Index_Type.tp_base = NULL;
    pyopencv_flann_Index_Type.tp_dealloc = pyopencv_flann_Index_dealloc;
    pyopencv_flann_Index_Type.tp_repr = pyopencv_flann_Index_repr;
    pyopencv_flann_Index_Type.tp_getset = pyopencv_flann_Index_getseters;
    pyopencv_flann_Index_Type.tp_methods = pyopencv_flann_Index_methods;
}

static PyObject* pyopencv_CLAHE_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<CLAHE %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_CLAHE_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_CLAHE_apply(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CLAHE_Type))
        return failmsgp("Incorrect type of self (must be 'CLAHE' or its derivative)");
    cv::CLAHE* _self_ = dynamic_cast<cv::CLAHE*>(((pyopencv_CLAHE_t*)self)->v.get());
    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:CLAHE.apply", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(_self_->apply(src, dst));
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
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:CLAHE.apply", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(_self_->apply(src, dst));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_CLAHE_collectGarbage(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CLAHE_Type))
        return failmsgp("Incorrect type of self (must be 'CLAHE' or its derivative)");
    cv::CLAHE* _self_ = dynamic_cast<cv::CLAHE*>(((pyopencv_CLAHE_t*)self)->v.get());

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(_self_->collectGarbage());
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_CLAHE_getClipLimit(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CLAHE_Type))
        return failmsgp("Incorrect type of self (must be 'CLAHE' or its derivative)");
    cv::CLAHE* _self_ = dynamic_cast<cv::CLAHE*>(((pyopencv_CLAHE_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getClipLimit());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_CLAHE_getTilesGridSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CLAHE_Type))
        return failmsgp("Incorrect type of self (must be 'CLAHE' or its derivative)");
    cv::CLAHE* _self_ = dynamic_cast<cv::CLAHE*>(((pyopencv_CLAHE_t*)self)->v.get());
    Size retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTilesGridSize());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_CLAHE_setClipLimit(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CLAHE_Type))
        return failmsgp("Incorrect type of self (must be 'CLAHE' or its derivative)");
    cv::CLAHE* _self_ = dynamic_cast<cv::CLAHE*>(((pyopencv_CLAHE_t*)self)->v.get());
    double clipLimit=0;

    const char* keywords[] = { "clipLimit", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:CLAHE.setClipLimit", (char**)keywords, &clipLimit) )
    {
        ERRWRAP2(_self_->setClipLimit(clipLimit));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_CLAHE_setTilesGridSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CLAHE_Type))
        return failmsgp("Incorrect type of self (must be 'CLAHE' or its derivative)");
    cv::CLAHE* _self_ = dynamic_cast<cv::CLAHE*>(((pyopencv_CLAHE_t*)self)->v.get());
    PyObject* pyobj_tileGridSize = NULL;
    Size tileGridSize;

    const char* keywords[] = { "tileGridSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:CLAHE.setTilesGridSize", (char**)keywords, &pyobj_tileGridSize) &&
        pyopencv_to(pyobj_tileGridSize, tileGridSize, ArgInfo("tileGridSize", 0)) )
    {
        ERRWRAP2(_self_->setTilesGridSize(tileGridSize));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_CLAHE_methods[] =
{
    {"apply", (PyCFunction)pyopencv_cv_CLAHE_apply, METH_VARARGS | METH_KEYWORDS, "apply(src[, dst]) -> dst"},
    {"collectGarbage", (PyCFunction)pyopencv_cv_CLAHE_collectGarbage, METH_VARARGS | METH_KEYWORDS, "collectGarbage() -> None"},
    {"getClipLimit", (PyCFunction)pyopencv_cv_CLAHE_getClipLimit, METH_VARARGS | METH_KEYWORDS, "getClipLimit() -> retval"},
    {"getTilesGridSize", (PyCFunction)pyopencv_cv_CLAHE_getTilesGridSize, METH_VARARGS | METH_KEYWORDS, "getTilesGridSize() -> retval"},
    {"setClipLimit", (PyCFunction)pyopencv_cv_CLAHE_setClipLimit, METH_VARARGS | METH_KEYWORDS, "setClipLimit(clipLimit) -> None"},
    {"setTilesGridSize", (PyCFunction)pyopencv_cv_CLAHE_setTilesGridSize, METH_VARARGS | METH_KEYWORDS, "setTilesGridSize(tileGridSize) -> None"},

    {NULL,          NULL}
};

static void pyopencv_CLAHE_specials(void)
{
    pyopencv_CLAHE_Type.tp_base = &pyopencv_Algorithm_Type;
    pyopencv_CLAHE_Type.tp_dealloc = pyopencv_CLAHE_dealloc;
    pyopencv_CLAHE_Type.tp_repr = pyopencv_CLAHE_repr;
    pyopencv_CLAHE_Type.tp_getset = pyopencv_CLAHE_getseters;
    pyopencv_CLAHE_Type.tp_methods = pyopencv_CLAHE_methods;
}

static PyObject* pyopencv_Subdiv2D_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<Subdiv2D %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_Subdiv2D_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_Subdiv2D_edgeDst(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Subdiv2D_Type))
        return failmsgp("Incorrect type of self (must be 'Subdiv2D' or its derivative)");
    cv::Subdiv2D* _self_ = ((pyopencv_Subdiv2D_t*)self)->v.get();
    int edge=0;
    Point2f dstpt;
    int retval;

    const char* keywords[] = { "edge", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:Subdiv2D.edgeDst", (char**)keywords, &edge) )
    {
        ERRWRAP2(retval = _self_->edgeDst(edge, &dstpt));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(dstpt));
    }

    return NULL;
}

static PyObject* pyopencv_cv_Subdiv2D_edgeOrg(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Subdiv2D_Type))
        return failmsgp("Incorrect type of self (must be 'Subdiv2D' or its derivative)");
    cv::Subdiv2D* _self_ = ((pyopencv_Subdiv2D_t*)self)->v.get();
    int edge=0;
    Point2f orgpt;
    int retval;

    const char* keywords[] = { "edge", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:Subdiv2D.edgeOrg", (char**)keywords, &edge) )
    {
        ERRWRAP2(retval = _self_->edgeOrg(edge, &orgpt));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(orgpt));
    }

    return NULL;
}

static PyObject* pyopencv_cv_Subdiv2D_findNearest(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Subdiv2D_Type))
        return failmsgp("Incorrect type of self (must be 'Subdiv2D' or its derivative)");
    cv::Subdiv2D* _self_ = ((pyopencv_Subdiv2D_t*)self)->v.get();
    PyObject* pyobj_pt = NULL;
    Point2f pt;
    Point2f nearestPt;
    int retval;

    const char* keywords[] = { "pt", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:Subdiv2D.findNearest", (char**)keywords, &pyobj_pt) &&
        pyopencv_to(pyobj_pt, pt, ArgInfo("pt", 0)) )
    {
        ERRWRAP2(retval = _self_->findNearest(pt, &nearestPt));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(nearestPt));
    }

    return NULL;
}

static PyObject* pyopencv_cv_Subdiv2D_getEdge(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Subdiv2D_Type))
        return failmsgp("Incorrect type of self (must be 'Subdiv2D' or its derivative)");
    cv::Subdiv2D* _self_ = ((pyopencv_Subdiv2D_t*)self)->v.get();
    int edge=0;
    int nextEdgeType=0;
    int retval;

    const char* keywords[] = { "edge", "nextEdgeType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "ii:Subdiv2D.getEdge", (char**)keywords, &edge, &nextEdgeType) )
    {
        ERRWRAP2(retval = _self_->getEdge(edge, nextEdgeType));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_Subdiv2D_getEdgeList(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Subdiv2D_Type))
        return failmsgp("Incorrect type of self (must be 'Subdiv2D' or its derivative)");
    cv::Subdiv2D* _self_ = ((pyopencv_Subdiv2D_t*)self)->v.get();
    vector_Vec4f edgeList;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(_self_->getEdgeList(edgeList));
        return pyopencv_from(edgeList);
    }

    return NULL;
}

static PyObject* pyopencv_cv_Subdiv2D_getLeadingEdgeList(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Subdiv2D_Type))
        return failmsgp("Incorrect type of self (must be 'Subdiv2D' or its derivative)");
    cv::Subdiv2D* _self_ = ((pyopencv_Subdiv2D_t*)self)->v.get();
    vector_int leadingEdgeList;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(_self_->getLeadingEdgeList(leadingEdgeList));
        return pyopencv_from(leadingEdgeList);
    }

    return NULL;
}

static PyObject* pyopencv_cv_Subdiv2D_getTriangleList(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Subdiv2D_Type))
        return failmsgp("Incorrect type of self (must be 'Subdiv2D' or its derivative)");
    cv::Subdiv2D* _self_ = ((pyopencv_Subdiv2D_t*)self)->v.get();
    vector_Vec6f triangleList;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(_self_->getTriangleList(triangleList));
        return pyopencv_from(triangleList);
    }

    return NULL;
}

static PyObject* pyopencv_cv_Subdiv2D_getVertex(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Subdiv2D_Type))
        return failmsgp("Incorrect type of self (must be 'Subdiv2D' or its derivative)");
    cv::Subdiv2D* _self_ = ((pyopencv_Subdiv2D_t*)self)->v.get();
    int vertex=0;
    int firstEdge;
    Point2f retval;

    const char* keywords[] = { "vertex", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:Subdiv2D.getVertex", (char**)keywords, &vertex) )
    {
        ERRWRAP2(retval = _self_->getVertex(vertex, &firstEdge));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(firstEdge));
    }

    return NULL;
}

static PyObject* pyopencv_cv_Subdiv2D_getVoronoiFacetList(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Subdiv2D_Type))
        return failmsgp("Incorrect type of self (must be 'Subdiv2D' or its derivative)");
    cv::Subdiv2D* _self_ = ((pyopencv_Subdiv2D_t*)self)->v.get();
    PyObject* pyobj_idx = NULL;
    vector_int idx;
    vector_vector_Point2f facetList;
    vector_Point2f facetCenters;

    const char* keywords[] = { "idx", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:Subdiv2D.getVoronoiFacetList", (char**)keywords, &pyobj_idx) &&
        pyopencv_to(pyobj_idx, idx, ArgInfo("idx", 0)) )
    {
        ERRWRAP2(_self_->getVoronoiFacetList(idx, facetList, facetCenters));
        return Py_BuildValue("(NN)", pyopencv_from(facetList), pyopencv_from(facetCenters));
    }

    return NULL;
}

static PyObject* pyopencv_cv_Subdiv2D_initDelaunay(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Subdiv2D_Type))
        return failmsgp("Incorrect type of self (must be 'Subdiv2D' or its derivative)");
    cv::Subdiv2D* _self_ = ((pyopencv_Subdiv2D_t*)self)->v.get();
    PyObject* pyobj_rect = NULL;
    Rect rect;

    const char* keywords[] = { "rect", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:Subdiv2D.initDelaunay", (char**)keywords, &pyobj_rect) &&
        pyopencv_to(pyobj_rect, rect, ArgInfo("rect", 0)) )
    {
        ERRWRAP2(_self_->initDelaunay(rect));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_Subdiv2D_insert(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Subdiv2D_Type))
        return failmsgp("Incorrect type of self (must be 'Subdiv2D' or its derivative)");
    cv::Subdiv2D* _self_ = ((pyopencv_Subdiv2D_t*)self)->v.get();
    {
    PyObject* pyobj_pt = NULL;
    Point2f pt;
    int retval;

    const char* keywords[] = { "pt", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:Subdiv2D.insert", (char**)keywords, &pyobj_pt) &&
        pyopencv_to(pyobj_pt, pt, ArgInfo("pt", 0)) )
    {
        ERRWRAP2(retval = _self_->insert(pt));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_ptvec = NULL;
    vector_Point2f ptvec;

    const char* keywords[] = { "ptvec", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:Subdiv2D.insert", (char**)keywords, &pyobj_ptvec) &&
        pyopencv_to(pyobj_ptvec, ptvec, ArgInfo("ptvec", 0)) )
    {
        ERRWRAP2(_self_->insert(ptvec));
        Py_RETURN_NONE;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_Subdiv2D_locate(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Subdiv2D_Type))
        return failmsgp("Incorrect type of self (must be 'Subdiv2D' or its derivative)");
    cv::Subdiv2D* _self_ = ((pyopencv_Subdiv2D_t*)self)->v.get();
    PyObject* pyobj_pt = NULL;
    Point2f pt;
    int edge;
    int vertex;
    int retval;

    const char* keywords[] = { "pt", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:Subdiv2D.locate", (char**)keywords, &pyobj_pt) &&
        pyopencv_to(pyobj_pt, pt, ArgInfo("pt", 0)) )
    {
        ERRWRAP2(retval = _self_->locate(pt, edge, vertex));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(edge), pyopencv_from(vertex));
    }

    return NULL;
}

static PyObject* pyopencv_cv_Subdiv2D_nextEdge(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Subdiv2D_Type))
        return failmsgp("Incorrect type of self (must be 'Subdiv2D' or its derivative)");
    cv::Subdiv2D* _self_ = ((pyopencv_Subdiv2D_t*)self)->v.get();
    int edge=0;
    int retval;

    const char* keywords[] = { "edge", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:Subdiv2D.nextEdge", (char**)keywords, &edge) )
    {
        ERRWRAP2(retval = _self_->nextEdge(edge));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_Subdiv2D_rotateEdge(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Subdiv2D_Type))
        return failmsgp("Incorrect type of self (must be 'Subdiv2D' or its derivative)");
    cv::Subdiv2D* _self_ = ((pyopencv_Subdiv2D_t*)self)->v.get();
    int edge=0;
    int rotate=0;
    int retval;

    const char* keywords[] = { "edge", "rotate", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "ii:Subdiv2D.rotateEdge", (char**)keywords, &edge, &rotate) )
    {
        ERRWRAP2(retval = _self_->rotateEdge(edge, rotate));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_Subdiv2D_symEdge(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Subdiv2D_Type))
        return failmsgp("Incorrect type of self (must be 'Subdiv2D' or its derivative)");
    cv::Subdiv2D* _self_ = ((pyopencv_Subdiv2D_t*)self)->v.get();
    int edge=0;
    int retval;

    const char* keywords[] = { "edge", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:Subdiv2D.symEdge", (char**)keywords, &edge) )
    {
        ERRWRAP2(retval = _self_->symEdge(edge));
        return pyopencv_from(retval);
    }

    return NULL;
}



static PyMethodDef pyopencv_Subdiv2D_methods[] =
{
    {"edgeDst", (PyCFunction)pyopencv_cv_Subdiv2D_edgeDst, METH_VARARGS | METH_KEYWORDS, "edgeDst(edge) -> retval, dstpt"},
    {"edgeOrg", (PyCFunction)pyopencv_cv_Subdiv2D_edgeOrg, METH_VARARGS | METH_KEYWORDS, "edgeOrg(edge) -> retval, orgpt"},
    {"findNearest", (PyCFunction)pyopencv_cv_Subdiv2D_findNearest, METH_VARARGS | METH_KEYWORDS, "findNearest(pt) -> retval, nearestPt"},
    {"getEdge", (PyCFunction)pyopencv_cv_Subdiv2D_getEdge, METH_VARARGS | METH_KEYWORDS, "getEdge(edge, nextEdgeType) -> retval"},
    {"getEdgeList", (PyCFunction)pyopencv_cv_Subdiv2D_getEdgeList, METH_VARARGS | METH_KEYWORDS, "getEdgeList() -> edgeList"},
    {"getLeadingEdgeList", (PyCFunction)pyopencv_cv_Subdiv2D_getLeadingEdgeList, METH_VARARGS | METH_KEYWORDS, "getLeadingEdgeList() -> leadingEdgeList"},
    {"getTriangleList", (PyCFunction)pyopencv_cv_Subdiv2D_getTriangleList, METH_VARARGS | METH_KEYWORDS, "getTriangleList() -> triangleList"},
    {"getVertex", (PyCFunction)pyopencv_cv_Subdiv2D_getVertex, METH_VARARGS | METH_KEYWORDS, "getVertex(vertex) -> retval, firstEdge"},
    {"getVoronoiFacetList", (PyCFunction)pyopencv_cv_Subdiv2D_getVoronoiFacetList, METH_VARARGS | METH_KEYWORDS, "getVoronoiFacetList(idx) -> facetList, facetCenters"},
    {"initDelaunay", (PyCFunction)pyopencv_cv_Subdiv2D_initDelaunay, METH_VARARGS | METH_KEYWORDS, "initDelaunay(rect) -> None"},
    {"insert", (PyCFunction)pyopencv_cv_Subdiv2D_insert, METH_VARARGS | METH_KEYWORDS, "insert(pt) -> retval  or  insert(ptvec) -> None"},
    {"locate", (PyCFunction)pyopencv_cv_Subdiv2D_locate, METH_VARARGS | METH_KEYWORDS, "locate(pt) -> retval, edge, vertex"},
    {"nextEdge", (PyCFunction)pyopencv_cv_Subdiv2D_nextEdge, METH_VARARGS | METH_KEYWORDS, "nextEdge(edge) -> retval"},
    {"rotateEdge", (PyCFunction)pyopencv_cv_Subdiv2D_rotateEdge, METH_VARARGS | METH_KEYWORDS, "rotateEdge(edge, rotate) -> retval"},
    {"symEdge", (PyCFunction)pyopencv_cv_Subdiv2D_symEdge, METH_VARARGS | METH_KEYWORDS, "symEdge(edge) -> retval"},

    {NULL,          NULL}
};

static void pyopencv_Subdiv2D_specials(void)
{
    pyopencv_Subdiv2D_Type.tp_base = NULL;
    pyopencv_Subdiv2D_Type.tp_dealloc = pyopencv_Subdiv2D_dealloc;
    pyopencv_Subdiv2D_Type.tp_repr = pyopencv_Subdiv2D_repr;
    pyopencv_Subdiv2D_Type.tp_getset = pyopencv_Subdiv2D_getseters;
    pyopencv_Subdiv2D_Type.tp_methods = pyopencv_Subdiv2D_methods;
}

static PyObject* pyopencv_LineSegmentDetector_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<LineSegmentDetector %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_LineSegmentDetector_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_LineSegmentDetector_compareSegments(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_LineSegmentDetector_Type))
        return failmsgp("Incorrect type of self (must be 'LineSegmentDetector' or its derivative)");
    cv::LineSegmentDetector* _self_ = dynamic_cast<cv::LineSegmentDetector*>(((pyopencv_LineSegmentDetector_t*)self)->v.get());
    {
    PyObject* pyobj_size = NULL;
    Size size;
    PyObject* pyobj_lines1 = NULL;
    Mat lines1;
    PyObject* pyobj_lines2 = NULL;
    Mat lines2;
    PyObject* pyobj__image = NULL;
    Mat _image;
    int retval;

    const char* keywords[] = { "size", "lines1", "lines2", "_image", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|O:LineSegmentDetector.compareSegments", (char**)keywords, &pyobj_size, &pyobj_lines1, &pyobj_lines2, &pyobj__image) &&
        pyopencv_to(pyobj_size, size, ArgInfo("size", 0)) &&
        pyopencv_to(pyobj_lines1, lines1, ArgInfo("lines1", 0)) &&
        pyopencv_to(pyobj_lines2, lines2, ArgInfo("lines2", 0)) &&
        pyopencv_to(pyobj__image, _image, ArgInfo("_image", 1)) )
    {
        ERRWRAP2(retval = _self_->compareSegments(size, lines1, lines2, _image));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(_image));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_size = NULL;
    Size size;
    PyObject* pyobj_lines1 = NULL;
    UMat lines1;
    PyObject* pyobj_lines2 = NULL;
    UMat lines2;
    PyObject* pyobj__image = NULL;
    UMat _image;
    int retval;

    const char* keywords[] = { "size", "lines1", "lines2", "_image", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|O:LineSegmentDetector.compareSegments", (char**)keywords, &pyobj_size, &pyobj_lines1, &pyobj_lines2, &pyobj__image) &&
        pyopencv_to(pyobj_size, size, ArgInfo("size", 0)) &&
        pyopencv_to(pyobj_lines1, lines1, ArgInfo("lines1", 0)) &&
        pyopencv_to(pyobj_lines2, lines2, ArgInfo("lines2", 0)) &&
        pyopencv_to(pyobj__image, _image, ArgInfo("_image", 1)) )
    {
        ERRWRAP2(retval = _self_->compareSegments(size, lines1, lines2, _image));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(_image));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_LineSegmentDetector_detect(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_LineSegmentDetector_Type))
        return failmsgp("Incorrect type of self (must be 'LineSegmentDetector' or its derivative)");
    cv::LineSegmentDetector* _self_ = dynamic_cast<cv::LineSegmentDetector*>(((pyopencv_LineSegmentDetector_t*)self)->v.get());
    {
    PyObject* pyobj__image = NULL;
    Mat _image;
    PyObject* pyobj__lines = NULL;
    Mat _lines;
    PyObject* pyobj_width = NULL;
    Mat width;
    PyObject* pyobj_prec = NULL;
    Mat prec;
    PyObject* pyobj_nfa = NULL;
    Mat nfa;

    const char* keywords[] = { "_image", "_lines", "width", "prec", "nfa", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOOO:LineSegmentDetector.detect", (char**)keywords, &pyobj__image, &pyobj__lines, &pyobj_width, &pyobj_prec, &pyobj_nfa) &&
        pyopencv_to(pyobj__image, _image, ArgInfo("_image", 0)) &&
        pyopencv_to(pyobj__lines, _lines, ArgInfo("_lines", 1)) &&
        pyopencv_to(pyobj_width, width, ArgInfo("width", 1)) &&
        pyopencv_to(pyobj_prec, prec, ArgInfo("prec", 1)) &&
        pyopencv_to(pyobj_nfa, nfa, ArgInfo("nfa", 1)) )
    {
        ERRWRAP2(_self_->detect(_image, _lines, width, prec, nfa));
        return Py_BuildValue("(NNNN)", pyopencv_from(_lines), pyopencv_from(width), pyopencv_from(prec), pyopencv_from(nfa));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj__image = NULL;
    UMat _image;
    PyObject* pyobj__lines = NULL;
    UMat _lines;
    PyObject* pyobj_width = NULL;
    UMat width;
    PyObject* pyobj_prec = NULL;
    UMat prec;
    PyObject* pyobj_nfa = NULL;
    UMat nfa;

    const char* keywords[] = { "_image", "_lines", "width", "prec", "nfa", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOOO:LineSegmentDetector.detect", (char**)keywords, &pyobj__image, &pyobj__lines, &pyobj_width, &pyobj_prec, &pyobj_nfa) &&
        pyopencv_to(pyobj__image, _image, ArgInfo("_image", 0)) &&
        pyopencv_to(pyobj__lines, _lines, ArgInfo("_lines", 1)) &&
        pyopencv_to(pyobj_width, width, ArgInfo("width", 1)) &&
        pyopencv_to(pyobj_prec, prec, ArgInfo("prec", 1)) &&
        pyopencv_to(pyobj_nfa, nfa, ArgInfo("nfa", 1)) )
    {
        ERRWRAP2(_self_->detect(_image, _lines, width, prec, nfa));
        return Py_BuildValue("(NNNN)", pyopencv_from(_lines), pyopencv_from(width), pyopencv_from(prec), pyopencv_from(nfa));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_LineSegmentDetector_drawSegments(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_LineSegmentDetector_Type))
        return failmsgp("Incorrect type of self (must be 'LineSegmentDetector' or its derivative)");
    cv::LineSegmentDetector* _self_ = dynamic_cast<cv::LineSegmentDetector*>(((pyopencv_LineSegmentDetector_t*)self)->v.get());
    {
    PyObject* pyobj__image = NULL;
    Mat _image;
    PyObject* pyobj_lines = NULL;
    Mat lines;

    const char* keywords[] = { "_image", "lines", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:LineSegmentDetector.drawSegments", (char**)keywords, &pyobj__image, &pyobj_lines) &&
        pyopencv_to(pyobj__image, _image, ArgInfo("_image", 1)) &&
        pyopencv_to(pyobj_lines, lines, ArgInfo("lines", 0)) )
    {
        ERRWRAP2(_self_->drawSegments(_image, lines));
        return pyopencv_from(_image);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj__image = NULL;
    UMat _image;
    PyObject* pyobj_lines = NULL;
    UMat lines;

    const char* keywords[] = { "_image", "lines", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:LineSegmentDetector.drawSegments", (char**)keywords, &pyobj__image, &pyobj_lines) &&
        pyopencv_to(pyobj__image, _image, ArgInfo("_image", 1)) &&
        pyopencv_to(pyobj_lines, lines, ArgInfo("lines", 0)) )
    {
        ERRWRAP2(_self_->drawSegments(_image, lines));
        return pyopencv_from(_image);
    }
    }

    return NULL;
}



static PyMethodDef pyopencv_LineSegmentDetector_methods[] =
{
    {"compareSegments", (PyCFunction)pyopencv_cv_LineSegmentDetector_compareSegments, METH_VARARGS | METH_KEYWORDS, "compareSegments(size, lines1, lines2[, _image]) -> retval, _image"},
    {"detect", (PyCFunction)pyopencv_cv_LineSegmentDetector_detect, METH_VARARGS | METH_KEYWORDS, "detect(_image[, _lines[, width[, prec[, nfa]]]]) -> _lines, width, prec, nfa"},
    {"drawSegments", (PyCFunction)pyopencv_cv_LineSegmentDetector_drawSegments, METH_VARARGS | METH_KEYWORDS, "drawSegments(_image, lines) -> _image"},

    {NULL,          NULL}
};

static void pyopencv_LineSegmentDetector_specials(void)
{
    pyopencv_LineSegmentDetector_Type.tp_base = &pyopencv_Algorithm_Type;
    pyopencv_LineSegmentDetector_Type.tp_dealloc = pyopencv_LineSegmentDetector_dealloc;
    pyopencv_LineSegmentDetector_Type.tp_repr = pyopencv_LineSegmentDetector_repr;
    pyopencv_LineSegmentDetector_Type.tp_getset = pyopencv_LineSegmentDetector_getseters;
    pyopencv_LineSegmentDetector_Type.tp_methods = pyopencv_LineSegmentDetector_methods;
}

static PyObject* pyopencv_ml_TrainData_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<ml_TrainData %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_ml_TrainData_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_ml_ml_TrainData_getCatCount(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    int vi=0;
    int retval;

    const char* keywords[] = { "vi", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ml_TrainData.getCatCount", (char**)keywords, &vi) )
    {
        ERRWRAP2(retval = _self_->getCatCount(vi));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getCatMap(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getCatMap());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getCatOfs(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getCatOfs());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getClassLabels(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getClassLabels());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getDefaultSubstValues(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getDefaultSubstValues());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getLayout(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getLayout());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getMissing(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getMissing());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getNAllVars(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getNAllVars());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getNSamples(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getNSamples());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getNTestSamples(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getNTestSamples());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getNTrainSamples(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getNTrainSamples());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getNVars(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getNVars());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getNames(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    PyObject* pyobj_names = NULL;
    vector_String names;

    const char* keywords[] = { "names", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:ml_TrainData.getNames", (char**)keywords, &pyobj_names) &&
        pyopencv_to(pyobj_names, names, ArgInfo("names", 0)) )
    {
        ERRWRAP2(_self_->getNames(names));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getNormCatResponses(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getNormCatResponses());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getResponseType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getResponseType());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getResponses(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getResponses());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getSample(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    {
    PyObject* pyobj_varIdx = NULL;
    Mat varIdx;
    int sidx=0;
    float buf=0.f;

    const char* keywords[] = { "varIdx", "sidx", "buf", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oif:ml_TrainData.getSample", (char**)keywords, &pyobj_varIdx, &sidx, &buf) &&
        pyopencv_to(pyobj_varIdx, varIdx, ArgInfo("varIdx", 0)) )
    {
        ERRWRAP2(_self_->getSample(varIdx, sidx, &buf));
        Py_RETURN_NONE;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_varIdx = NULL;
    UMat varIdx;
    int sidx=0;
    float buf=0.f;

    const char* keywords[] = { "varIdx", "sidx", "buf", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oif:ml_TrainData.getSample", (char**)keywords, &pyobj_varIdx, &sidx, &buf) &&
        pyopencv_to(pyobj_varIdx, varIdx, ArgInfo("varIdx", 0)) )
    {
        ERRWRAP2(_self_->getSample(varIdx, sidx, &buf));
        Py_RETURN_NONE;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getSampleWeights(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getSampleWeights());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getSamples(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getSamples());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getTestNormCatResponses(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTestNormCatResponses());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getTestResponses(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTestResponses());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getTestSampleIdx(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTestSampleIdx());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getTestSampleWeights(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTestSampleWeights());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getTestSamples(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTestSamples());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getTrainNormCatResponses(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTrainNormCatResponses());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getTrainResponses(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTrainResponses());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getTrainSampleIdx(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTrainSampleIdx());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getTrainSampleWeights(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTrainSampleWeights());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getTrainSamples(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    int layout=ROW_SAMPLE;
    bool compressSamples=true;
    bool compressVars=true;
    Mat retval;

    const char* keywords[] = { "layout", "compressSamples", "compressVars", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|ibb:ml_TrainData.getTrainSamples", (char**)keywords, &layout, &compressSamples, &compressVars) )
    {
        ERRWRAP2(retval = _self_->getTrainSamples(layout, compressSamples, compressVars));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getValues(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    {
    int vi=0;
    PyObject* pyobj_sidx = NULL;
    Mat sidx;
    float values=0.f;

    const char* keywords[] = { "vi", "sidx", "values", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "iOf:ml_TrainData.getValues", (char**)keywords, &vi, &pyobj_sidx, &values) &&
        pyopencv_to(pyobj_sidx, sidx, ArgInfo("sidx", 0)) )
    {
        ERRWRAP2(_self_->getValues(vi, sidx, &values));
        Py_RETURN_NONE;
    }
    }
    PyErr_Clear();

    {
    int vi=0;
    PyObject* pyobj_sidx = NULL;
    UMat sidx;
    float values=0.f;

    const char* keywords[] = { "vi", "sidx", "values", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "iOf:ml_TrainData.getValues", (char**)keywords, &vi, &pyobj_sidx, &values) &&
        pyopencv_to(pyobj_sidx, sidx, ArgInfo("sidx", 0)) )
    {
        ERRWRAP2(_self_->getValues(vi, sidx, &values));
        Py_RETURN_NONE;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getVarIdx(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getVarIdx());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getVarSymbolFlags(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getVarSymbolFlags());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_getVarType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getVarType());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_setTrainTestSplit(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    int count=0;
    bool shuffle=true;

    const char* keywords[] = { "count", "shuffle", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i|b:ml_TrainData.setTrainTestSplit", (char**)keywords, &count, &shuffle) )
    {
        ERRWRAP2(_self_->setTrainTestSplit(count, shuffle));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_setTrainTestSplitRatio(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();
    double ratio=0;
    bool shuffle=true;

    const char* keywords[] = { "ratio", "shuffle", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d|b:ml_TrainData.setTrainTestSplitRatio", (char**)keywords, &ratio, &shuffle) )
    {
        ERRWRAP2(_self_->setTrainTestSplitRatio(ratio, shuffle));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_TrainData_shuffleTrainTest(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_TrainData_Type))
        return failmsgp("Incorrect type of self (must be 'ml_TrainData' or its derivative)");
    cv::ml::TrainData* _self_ = ((pyopencv_ml_TrainData_t*)self)->v.get();

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(_self_->shuffleTrainTest());
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_ml_TrainData_methods[] =
{
    {"getCatCount", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getCatCount, METH_VARARGS | METH_KEYWORDS, "getCatCount(vi) -> retval"},
    {"getCatMap", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getCatMap, METH_VARARGS | METH_KEYWORDS, "getCatMap() -> retval"},
    {"getCatOfs", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getCatOfs, METH_VARARGS | METH_KEYWORDS, "getCatOfs() -> retval"},
    {"getClassLabels", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getClassLabels, METH_VARARGS | METH_KEYWORDS, "getClassLabels() -> retval"},
    {"getDefaultSubstValues", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getDefaultSubstValues, METH_VARARGS | METH_KEYWORDS, "getDefaultSubstValues() -> retval"},
    {"getLayout", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getLayout, METH_VARARGS | METH_KEYWORDS, "getLayout() -> retval"},
    {"getMissing", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getMissing, METH_VARARGS | METH_KEYWORDS, "getMissing() -> retval"},
    {"getNAllVars", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getNAllVars, METH_VARARGS | METH_KEYWORDS, "getNAllVars() -> retval"},
    {"getNSamples", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getNSamples, METH_VARARGS | METH_KEYWORDS, "getNSamples() -> retval"},
    {"getNTestSamples", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getNTestSamples, METH_VARARGS | METH_KEYWORDS, "getNTestSamples() -> retval"},
    {"getNTrainSamples", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getNTrainSamples, METH_VARARGS | METH_KEYWORDS, "getNTrainSamples() -> retval"},
    {"getNVars", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getNVars, METH_VARARGS | METH_KEYWORDS, "getNVars() -> retval"},
    {"getNames", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getNames, METH_VARARGS | METH_KEYWORDS, "getNames(names) -> None"},
    {"getNormCatResponses", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getNormCatResponses, METH_VARARGS | METH_KEYWORDS, "getNormCatResponses() -> retval"},
    {"getResponseType", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getResponseType, METH_VARARGS | METH_KEYWORDS, "getResponseType() -> retval"},
    {"getResponses", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getResponses, METH_VARARGS | METH_KEYWORDS, "getResponses() -> retval"},
    {"getSample", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getSample, METH_VARARGS | METH_KEYWORDS, "getSample(varIdx, sidx, buf) -> None"},
    {"getSampleWeights", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getSampleWeights, METH_VARARGS | METH_KEYWORDS, "getSampleWeights() -> retval"},
    {"getSamples", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getSamples, METH_VARARGS | METH_KEYWORDS, "getSamples() -> retval"},
    {"getTestNormCatResponses", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getTestNormCatResponses, METH_VARARGS | METH_KEYWORDS, "getTestNormCatResponses() -> retval"},
    {"getTestResponses", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getTestResponses, METH_VARARGS | METH_KEYWORDS, "getTestResponses() -> retval"},
    {"getTestSampleIdx", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getTestSampleIdx, METH_VARARGS | METH_KEYWORDS, "getTestSampleIdx() -> retval"},
    {"getTestSampleWeights", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getTestSampleWeights, METH_VARARGS | METH_KEYWORDS, "getTestSampleWeights() -> retval"},
    {"getTestSamples", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getTestSamples, METH_VARARGS | METH_KEYWORDS, "getTestSamples() -> retval"},
    {"getTrainNormCatResponses", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getTrainNormCatResponses, METH_VARARGS | METH_KEYWORDS, "getTrainNormCatResponses() -> retval"},
    {"getTrainResponses", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getTrainResponses, METH_VARARGS | METH_KEYWORDS, "getTrainResponses() -> retval"},
    {"getTrainSampleIdx", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getTrainSampleIdx, METH_VARARGS | METH_KEYWORDS, "getTrainSampleIdx() -> retval"},
    {"getTrainSampleWeights", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getTrainSampleWeights, METH_VARARGS | METH_KEYWORDS, "getTrainSampleWeights() -> retval"},
    {"getTrainSamples", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getTrainSamples, METH_VARARGS | METH_KEYWORDS, "getTrainSamples([, layout[, compressSamples[, compressVars]]]) -> retval"},
    {"getValues", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getValues, METH_VARARGS | METH_KEYWORDS, "getValues(vi, sidx, values) -> None"},
    {"getVarIdx", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getVarIdx, METH_VARARGS | METH_KEYWORDS, "getVarIdx() -> retval"},
    {"getVarSymbolFlags", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getVarSymbolFlags, METH_VARARGS | METH_KEYWORDS, "getVarSymbolFlags() -> retval"},
    {"getVarType", (PyCFunction)pyopencv_cv_ml_ml_TrainData_getVarType, METH_VARARGS | METH_KEYWORDS, "getVarType() -> retval"},
    {"setTrainTestSplit", (PyCFunction)pyopencv_cv_ml_ml_TrainData_setTrainTestSplit, METH_VARARGS | METH_KEYWORDS, "setTrainTestSplit(count[, shuffle]) -> None"},
    {"setTrainTestSplitRatio", (PyCFunction)pyopencv_cv_ml_ml_TrainData_setTrainTestSplitRatio, METH_VARARGS | METH_KEYWORDS, "setTrainTestSplitRatio(ratio[, shuffle]) -> None"},
    {"shuffleTrainTest", (PyCFunction)pyopencv_cv_ml_ml_TrainData_shuffleTrainTest, METH_VARARGS | METH_KEYWORDS, "shuffleTrainTest() -> None"},

    {NULL,          NULL}
};

static void pyopencv_ml_TrainData_specials(void)
{
    pyopencv_ml_TrainData_Type.tp_base = NULL;
    pyopencv_ml_TrainData_Type.tp_dealloc = pyopencv_ml_TrainData_dealloc;
    pyopencv_ml_TrainData_Type.tp_repr = pyopencv_ml_TrainData_repr;
    pyopencv_ml_TrainData_Type.tp_getset = pyopencv_ml_TrainData_getseters;
    pyopencv_ml_TrainData_Type.tp_methods = pyopencv_ml_TrainData_methods;
}

static PyObject* pyopencv_ml_StatModel_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<ml_StatModel %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_ml_StatModel_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_ml_ml_StatModel_calcError(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_StatModel_Type))
        return failmsgp("Incorrect type of self (must be 'ml_StatModel' or its derivative)");
    cv::ml::StatModel* _self_ = dynamic_cast<cv::ml::StatModel*>(((pyopencv_ml_StatModel_t*)self)->v.get());
    {
    PyObject* pyobj_data = NULL;
    Ptr<TrainData> data;
    bool test=0;
    PyObject* pyobj_resp = NULL;
    Mat resp;
    float retval;

    const char* keywords[] = { "data", "test", "resp", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Ob|O:ml_StatModel.calcError", (char**)keywords, &pyobj_data, &test, &pyobj_resp) &&
        pyopencv_to(pyobj_data, data, ArgInfo("data", 0)) &&
        pyopencv_to(pyobj_resp, resp, ArgInfo("resp", 1)) )
    {
        ERRWRAP2(retval = _self_->calcError(data, test, resp));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(resp));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_data = NULL;
    Ptr<TrainData> data;
    bool test=0;
    PyObject* pyobj_resp = NULL;
    UMat resp;
    float retval;

    const char* keywords[] = { "data", "test", "resp", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Ob|O:ml_StatModel.calcError", (char**)keywords, &pyobj_data, &test, &pyobj_resp) &&
        pyopencv_to(pyobj_data, data, ArgInfo("data", 0)) &&
        pyopencv_to(pyobj_resp, resp, ArgInfo("resp", 1)) )
    {
        ERRWRAP2(retval = _self_->calcError(data, test, resp));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(resp));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_StatModel_empty(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_StatModel_Type))
        return failmsgp("Incorrect type of self (must be 'ml_StatModel' or its derivative)");
    cv::ml::StatModel* _self_ = dynamic_cast<cv::ml::StatModel*>(((pyopencv_ml_StatModel_t*)self)->v.get());
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->empty());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_StatModel_getVarCount(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_StatModel_Type))
        return failmsgp("Incorrect type of self (must be 'ml_StatModel' or its derivative)");
    cv::ml::StatModel* _self_ = dynamic_cast<cv::ml::StatModel*>(((pyopencv_ml_StatModel_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getVarCount());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_StatModel_isClassifier(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_StatModel_Type))
        return failmsgp("Incorrect type of self (must be 'ml_StatModel' or its derivative)");
    cv::ml::StatModel* _self_ = dynamic_cast<cv::ml::StatModel*>(((pyopencv_ml_StatModel_t*)self)->v.get());
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->isClassifier());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_StatModel_isTrained(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_StatModel_Type))
        return failmsgp("Incorrect type of self (must be 'ml_StatModel' or its derivative)");
    cv::ml::StatModel* _self_ = dynamic_cast<cv::ml::StatModel*>(((pyopencv_ml_StatModel_t*)self)->v.get());
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->isTrained());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_StatModel_predict(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_StatModel_Type))
        return failmsgp("Incorrect type of self (must be 'ml_StatModel' or its derivative)");
    cv::ml::StatModel* _self_ = dynamic_cast<cv::ml::StatModel*>(((pyopencv_ml_StatModel_t*)self)->v.get());
    {
    PyObject* pyobj_samples = NULL;
    Mat samples;
    PyObject* pyobj_results = NULL;
    Mat results;
    int flags=0;
    float retval;

    const char* keywords[] = { "samples", "results", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oi:ml_StatModel.predict", (char**)keywords, &pyobj_samples, &pyobj_results, &flags) &&
        pyopencv_to(pyobj_samples, samples, ArgInfo("samples", 0)) &&
        pyopencv_to(pyobj_results, results, ArgInfo("results", 1)) )
    {
        ERRWRAP2(retval = _self_->predict(samples, results, flags));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(results));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_samples = NULL;
    UMat samples;
    PyObject* pyobj_results = NULL;
    UMat results;
    int flags=0;
    float retval;

    const char* keywords[] = { "samples", "results", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oi:ml_StatModel.predict", (char**)keywords, &pyobj_samples, &pyobj_results, &flags) &&
        pyopencv_to(pyobj_samples, samples, ArgInfo("samples", 0)) &&
        pyopencv_to(pyobj_results, results, ArgInfo("results", 1)) )
    {
        ERRWRAP2(retval = _self_->predict(samples, results, flags));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(results));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_StatModel_train(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_StatModel_Type))
        return failmsgp("Incorrect type of self (must be 'ml_StatModel' or its derivative)");
    cv::ml::StatModel* _self_ = dynamic_cast<cv::ml::StatModel*>(((pyopencv_ml_StatModel_t*)self)->v.get());
    {
    PyObject* pyobj_trainData = NULL;
    Ptr<TrainData> trainData;
    int flags=0;
    bool retval;

    const char* keywords[] = { "trainData", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|i:ml_StatModel.train", (char**)keywords, &pyobj_trainData, &flags) &&
        pyopencv_to(pyobj_trainData, trainData, ArgInfo("trainData", 0)) )
    {
        ERRWRAP2(retval = _self_->train(trainData, flags));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_samples = NULL;
    Mat samples;
    int layout=0;
    PyObject* pyobj_responses = NULL;
    Mat responses;
    bool retval;

    const char* keywords[] = { "samples", "layout", "responses", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OiO:ml_StatModel.train", (char**)keywords, &pyobj_samples, &layout, &pyobj_responses) &&
        pyopencv_to(pyobj_samples, samples, ArgInfo("samples", 0)) &&
        pyopencv_to(pyobj_responses, responses, ArgInfo("responses", 0)) )
    {
        ERRWRAP2(retval = _self_->train(samples, layout, responses));
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
    bool retval;

    const char* keywords[] = { "samples", "layout", "responses", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OiO:ml_StatModel.train", (char**)keywords, &pyobj_samples, &layout, &pyobj_responses) &&
        pyopencv_to(pyobj_samples, samples, ArgInfo("samples", 0)) &&
        pyopencv_to(pyobj_responses, responses, ArgInfo("responses", 0)) )
    {
        ERRWRAP2(retval = _self_->train(samples, layout, responses));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}



static PyMethodDef pyopencv_ml_StatModel_methods[] =
{
    {"calcError", (PyCFunction)pyopencv_cv_ml_ml_StatModel_calcError, METH_VARARGS | METH_KEYWORDS, "calcError(data, test[, resp]) -> retval, resp"},
    {"empty", (PyCFunction)pyopencv_cv_ml_ml_StatModel_empty, METH_VARARGS | METH_KEYWORDS, "empty() -> retval"},
    {"getVarCount", (PyCFunction)pyopencv_cv_ml_ml_StatModel_getVarCount, METH_VARARGS | METH_KEYWORDS, "getVarCount() -> retval"},
    {"isClassifier", (PyCFunction)pyopencv_cv_ml_ml_StatModel_isClassifier, METH_VARARGS | METH_KEYWORDS, "isClassifier() -> retval"},
    {"isTrained", (PyCFunction)pyopencv_cv_ml_ml_StatModel_isTrained, METH_VARARGS | METH_KEYWORDS, "isTrained() -> retval"},
    {"predict", (PyCFunction)pyopencv_cv_ml_ml_StatModel_predict, METH_VARARGS | METH_KEYWORDS, "predict(samples[, results[, flags]]) -> retval, results"},
    {"train", (PyCFunction)pyopencv_cv_ml_ml_StatModel_train, METH_VARARGS | METH_KEYWORDS, "train(trainData[, flags]) -> retval  or  train(samples, layout, responses) -> retval"},

    {NULL,          NULL}
};

static void pyopencv_ml_StatModel_specials(void)
{
    pyopencv_ml_StatModel_Type.tp_base = &pyopencv_Algorithm_Type;
    pyopencv_ml_StatModel_Type.tp_dealloc = pyopencv_ml_StatModel_dealloc;
    pyopencv_ml_StatModel_Type.tp_repr = pyopencv_ml_StatModel_repr;
    pyopencv_ml_StatModel_Type.tp_getset = pyopencv_ml_StatModel_getseters;
    pyopencv_ml_StatModel_Type.tp_methods = pyopencv_ml_StatModel_methods;
}

static PyObject* pyopencv_ml_NormalBayesClassifier_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<ml_NormalBayesClassifier %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_ml_NormalBayesClassifier_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_ml_ml_NormalBayesClassifier_predictProb(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_NormalBayesClassifier_Type))
        return failmsgp("Incorrect type of self (must be 'ml_NormalBayesClassifier' or its derivative)");
    cv::ml::NormalBayesClassifier* _self_ = dynamic_cast<cv::ml::NormalBayesClassifier*>(((pyopencv_ml_NormalBayesClassifier_t*)self)->v.get());
    {
    PyObject* pyobj_inputs = NULL;
    Mat inputs;
    PyObject* pyobj_outputs = NULL;
    Mat outputs;
    PyObject* pyobj_outputProbs = NULL;
    Mat outputProbs;
    int flags=0;
    float retval;

    const char* keywords[] = { "inputs", "outputs", "outputProbs", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOi:ml_NormalBayesClassifier.predictProb", (char**)keywords, &pyobj_inputs, &pyobj_outputs, &pyobj_outputProbs, &flags) &&
        pyopencv_to(pyobj_inputs, inputs, ArgInfo("inputs", 0)) &&
        pyopencv_to(pyobj_outputs, outputs, ArgInfo("outputs", 1)) &&
        pyopencv_to(pyobj_outputProbs, outputProbs, ArgInfo("outputProbs", 1)) )
    {
        ERRWRAP2(retval = _self_->predictProb(inputs, outputs, outputProbs, flags));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(outputs), pyopencv_from(outputProbs));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_inputs = NULL;
    UMat inputs;
    PyObject* pyobj_outputs = NULL;
    UMat outputs;
    PyObject* pyobj_outputProbs = NULL;
    UMat outputProbs;
    int flags=0;
    float retval;

    const char* keywords[] = { "inputs", "outputs", "outputProbs", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOi:ml_NormalBayesClassifier.predictProb", (char**)keywords, &pyobj_inputs, &pyobj_outputs, &pyobj_outputProbs, &flags) &&
        pyopencv_to(pyobj_inputs, inputs, ArgInfo("inputs", 0)) &&
        pyopencv_to(pyobj_outputs, outputs, ArgInfo("outputs", 1)) &&
        pyopencv_to(pyobj_outputProbs, outputProbs, ArgInfo("outputProbs", 1)) )
    {
        ERRWRAP2(retval = _self_->predictProb(inputs, outputs, outputProbs, flags));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(outputs), pyopencv_from(outputProbs));
    }
    }

    return NULL;
}



static PyMethodDef pyopencv_ml_NormalBayesClassifier_methods[] =
{
    {"predictProb", (PyCFunction)pyopencv_cv_ml_ml_NormalBayesClassifier_predictProb, METH_VARARGS | METH_KEYWORDS, "predictProb(inputs[, outputs[, outputProbs[, flags]]]) -> retval, outputs, outputProbs"},

    {NULL,          NULL}
};

static void pyopencv_ml_NormalBayesClassifier_specials(void)
{
    pyopencv_ml_NormalBayesClassifier_Type.tp_base = &pyopencv_ml_StatModel_Type;
    pyopencv_ml_NormalBayesClassifier_Type.tp_dealloc = pyopencv_ml_NormalBayesClassifier_dealloc;
    pyopencv_ml_NormalBayesClassifier_Type.tp_repr = pyopencv_ml_NormalBayesClassifier_repr;
    pyopencv_ml_NormalBayesClassifier_Type.tp_getset = pyopencv_ml_NormalBayesClassifier_getseters;
    pyopencv_ml_NormalBayesClassifier_Type.tp_methods = pyopencv_ml_NormalBayesClassifier_methods;
}

static PyObject* pyopencv_ml_KNearest_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<ml_KNearest %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_ml_KNearest_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_ml_ml_KNearest_findNearest(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_KNearest_Type))
        return failmsgp("Incorrect type of self (must be 'ml_KNearest' or its derivative)");
    cv::ml::KNearest* _self_ = dynamic_cast<cv::ml::KNearest*>(((pyopencv_ml_KNearest_t*)self)->v.get());
    {
    PyObject* pyobj_samples = NULL;
    Mat samples;
    int k=0;
    PyObject* pyobj_results = NULL;
    Mat results;
    PyObject* pyobj_neighborResponses = NULL;
    Mat neighborResponses;
    PyObject* pyobj_dist = NULL;
    Mat dist;
    float retval;

    const char* keywords[] = { "samples", "k", "results", "neighborResponses", "dist", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|OOO:ml_KNearest.findNearest", (char**)keywords, &pyobj_samples, &k, &pyobj_results, &pyobj_neighborResponses, &pyobj_dist) &&
        pyopencv_to(pyobj_samples, samples, ArgInfo("samples", 0)) &&
        pyopencv_to(pyobj_results, results, ArgInfo("results", 1)) &&
        pyopencv_to(pyobj_neighborResponses, neighborResponses, ArgInfo("neighborResponses", 1)) &&
        pyopencv_to(pyobj_dist, dist, ArgInfo("dist", 1)) )
    {
        ERRWRAP2(retval = _self_->findNearest(samples, k, results, neighborResponses, dist));
        return Py_BuildValue("(NNNN)", pyopencv_from(retval), pyopencv_from(results), pyopencv_from(neighborResponses), pyopencv_from(dist));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_samples = NULL;
    UMat samples;
    int k=0;
    PyObject* pyobj_results = NULL;
    UMat results;
    PyObject* pyobj_neighborResponses = NULL;
    UMat neighborResponses;
    PyObject* pyobj_dist = NULL;
    UMat dist;
    float retval;

    const char* keywords[] = { "samples", "k", "results", "neighborResponses", "dist", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|OOO:ml_KNearest.findNearest", (char**)keywords, &pyobj_samples, &k, &pyobj_results, &pyobj_neighborResponses, &pyobj_dist) &&
        pyopencv_to(pyobj_samples, samples, ArgInfo("samples", 0)) &&
        pyopencv_to(pyobj_results, results, ArgInfo("results", 1)) &&
        pyopencv_to(pyobj_neighborResponses, neighborResponses, ArgInfo("neighborResponses", 1)) &&
        pyopencv_to(pyobj_dist, dist, ArgInfo("dist", 1)) )
    {
        ERRWRAP2(retval = _self_->findNearest(samples, k, results, neighborResponses, dist));
        return Py_BuildValue("(NNNN)", pyopencv_from(retval), pyopencv_from(results), pyopencv_from(neighborResponses), pyopencv_from(dist));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_KNearest_getAlgorithmType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_KNearest_Type))
        return failmsgp("Incorrect type of self (must be 'ml_KNearest' or its derivative)");
    cv::ml::KNearest* _self_ = dynamic_cast<cv::ml::KNearest*>(((pyopencv_ml_KNearest_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getAlgorithmType());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_KNearest_getDefaultK(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_KNearest_Type))
        return failmsgp("Incorrect type of self (must be 'ml_KNearest' or its derivative)");
    cv::ml::KNearest* _self_ = dynamic_cast<cv::ml::KNearest*>(((pyopencv_ml_KNearest_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getDefaultK());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_KNearest_getEmax(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_KNearest_Type))
        return failmsgp("Incorrect type of self (must be 'ml_KNearest' or its derivative)");
    cv::ml::KNearest* _self_ = dynamic_cast<cv::ml::KNearest*>(((pyopencv_ml_KNearest_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getEmax());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_KNearest_getIsClassifier(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_KNearest_Type))
        return failmsgp("Incorrect type of self (must be 'ml_KNearest' or its derivative)");
    cv::ml::KNearest* _self_ = dynamic_cast<cv::ml::KNearest*>(((pyopencv_ml_KNearest_t*)self)->v.get());
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getIsClassifier());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_KNearest_setAlgorithmType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_KNearest_Type))
        return failmsgp("Incorrect type of self (must be 'ml_KNearest' or its derivative)");
    cv::ml::KNearest* _self_ = dynamic_cast<cv::ml::KNearest*>(((pyopencv_ml_KNearest_t*)self)->v.get());
    int val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ml_KNearest.setAlgorithmType", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setAlgorithmType(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_KNearest_setDefaultK(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_KNearest_Type))
        return failmsgp("Incorrect type of self (must be 'ml_KNearest' or its derivative)");
    cv::ml::KNearest* _self_ = dynamic_cast<cv::ml::KNearest*>(((pyopencv_ml_KNearest_t*)self)->v.get());
    int val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ml_KNearest.setDefaultK", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setDefaultK(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_KNearest_setEmax(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_KNearest_Type))
        return failmsgp("Incorrect type of self (must be 'ml_KNearest' or its derivative)");
    cv::ml::KNearest* _self_ = dynamic_cast<cv::ml::KNearest*>(((pyopencv_ml_KNearest_t*)self)->v.get());
    int val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ml_KNearest.setEmax", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setEmax(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_KNearest_setIsClassifier(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_KNearest_Type))
        return failmsgp("Incorrect type of self (must be 'ml_KNearest' or its derivative)");
    cv::ml::KNearest* _self_ = dynamic_cast<cv::ml::KNearest*>(((pyopencv_ml_KNearest_t*)self)->v.get());
    bool val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:ml_KNearest.setIsClassifier", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setIsClassifier(val));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_ml_KNearest_methods[] =
{
    {"findNearest", (PyCFunction)pyopencv_cv_ml_ml_KNearest_findNearest, METH_VARARGS | METH_KEYWORDS, "findNearest(samples, k[, results[, neighborResponses[, dist]]]) -> retval, results, neighborResponses, dist"},
    {"getAlgorithmType", (PyCFunction)pyopencv_cv_ml_ml_KNearest_getAlgorithmType, METH_VARARGS | METH_KEYWORDS, "getAlgorithmType() -> retval"},
    {"getDefaultK", (PyCFunction)pyopencv_cv_ml_ml_KNearest_getDefaultK, METH_VARARGS | METH_KEYWORDS, "getDefaultK() -> retval"},
    {"getEmax", (PyCFunction)pyopencv_cv_ml_ml_KNearest_getEmax, METH_VARARGS | METH_KEYWORDS, "getEmax() -> retval"},
    {"getIsClassifier", (PyCFunction)pyopencv_cv_ml_ml_KNearest_getIsClassifier, METH_VARARGS | METH_KEYWORDS, "getIsClassifier() -> retval"},
    {"setAlgorithmType", (PyCFunction)pyopencv_cv_ml_ml_KNearest_setAlgorithmType, METH_VARARGS | METH_KEYWORDS, "setAlgorithmType(val) -> None"},
    {"setDefaultK", (PyCFunction)pyopencv_cv_ml_ml_KNearest_setDefaultK, METH_VARARGS | METH_KEYWORDS, "setDefaultK(val) -> None"},
    {"setEmax", (PyCFunction)pyopencv_cv_ml_ml_KNearest_setEmax, METH_VARARGS | METH_KEYWORDS, "setEmax(val) -> None"},
    {"setIsClassifier", (PyCFunction)pyopencv_cv_ml_ml_KNearest_setIsClassifier, METH_VARARGS | METH_KEYWORDS, "setIsClassifier(val) -> None"},

    {NULL,          NULL}
};

static void pyopencv_ml_KNearest_specials(void)
{
    pyopencv_ml_KNearest_Type.tp_base = &pyopencv_ml_StatModel_Type;
    pyopencv_ml_KNearest_Type.tp_dealloc = pyopencv_ml_KNearest_dealloc;
    pyopencv_ml_KNearest_Type.tp_repr = pyopencv_ml_KNearest_repr;
    pyopencv_ml_KNearest_Type.tp_getset = pyopencv_ml_KNearest_getseters;
    pyopencv_ml_KNearest_Type.tp_methods = pyopencv_ml_KNearest_methods;
}

static PyObject* pyopencv_ml_SVM_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<ml_SVM %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_ml_SVM_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_ml_ml_SVM_getC(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVM' or its derivative)");
    cv::ml::SVM* _self_ = dynamic_cast<cv::ml::SVM*>(((pyopencv_ml_SVM_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getC());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVM_getClassWeights(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVM' or its derivative)");
    cv::ml::SVM* _self_ = dynamic_cast<cv::ml::SVM*>(((pyopencv_ml_SVM_t*)self)->v.get());
    cv::Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getClassWeights());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVM_getCoef0(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVM' or its derivative)");
    cv::ml::SVM* _self_ = dynamic_cast<cv::ml::SVM*>(((pyopencv_ml_SVM_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getCoef0());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVM_getDecisionFunction(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVM' or its derivative)");
    cv::ml::SVM* _self_ = dynamic_cast<cv::ml::SVM*>(((pyopencv_ml_SVM_t*)self)->v.get());
    {
    int i=0;
    PyObject* pyobj_alpha = NULL;
    Mat alpha;
    PyObject* pyobj_svidx = NULL;
    Mat svidx;
    double retval;

    const char* keywords[] = { "i", "alpha", "svidx", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i|OO:ml_SVM.getDecisionFunction", (char**)keywords, &i, &pyobj_alpha, &pyobj_svidx) &&
        pyopencv_to(pyobj_alpha, alpha, ArgInfo("alpha", 1)) &&
        pyopencv_to(pyobj_svidx, svidx, ArgInfo("svidx", 1)) )
    {
        ERRWRAP2(retval = _self_->getDecisionFunction(i, alpha, svidx));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(alpha), pyopencv_from(svidx));
    }
    }
    PyErr_Clear();

    {
    int i=0;
    PyObject* pyobj_alpha = NULL;
    UMat alpha;
    PyObject* pyobj_svidx = NULL;
    UMat svidx;
    double retval;

    const char* keywords[] = { "i", "alpha", "svidx", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i|OO:ml_SVM.getDecisionFunction", (char**)keywords, &i, &pyobj_alpha, &pyobj_svidx) &&
        pyopencv_to(pyobj_alpha, alpha, ArgInfo("alpha", 1)) &&
        pyopencv_to(pyobj_svidx, svidx, ArgInfo("svidx", 1)) )
    {
        ERRWRAP2(retval = _self_->getDecisionFunction(i, alpha, svidx));
        return Py_BuildValue("(NNN)", pyopencv_from(retval), pyopencv_from(alpha), pyopencv_from(svidx));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVM_getDegree(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVM' or its derivative)");
    cv::ml::SVM* _self_ = dynamic_cast<cv::ml::SVM*>(((pyopencv_ml_SVM_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getDegree());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVM_getGamma(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVM' or its derivative)");
    cv::ml::SVM* _self_ = dynamic_cast<cv::ml::SVM*>(((pyopencv_ml_SVM_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getGamma());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVM_getKernelType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVM' or its derivative)");
    cv::ml::SVM* _self_ = dynamic_cast<cv::ml::SVM*>(((pyopencv_ml_SVM_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getKernelType());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVM_getNu(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVM' or its derivative)");
    cv::ml::SVM* _self_ = dynamic_cast<cv::ml::SVM*>(((pyopencv_ml_SVM_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getNu());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVM_getP(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVM' or its derivative)");
    cv::ml::SVM* _self_ = dynamic_cast<cv::ml::SVM*>(((pyopencv_ml_SVM_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getP());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVM_getSupportVectors(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVM' or its derivative)");
    cv::ml::SVM* _self_ = dynamic_cast<cv::ml::SVM*>(((pyopencv_ml_SVM_t*)self)->v.get());
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getSupportVectors());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVM_getTermCriteria(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVM' or its derivative)");
    cv::ml::SVM* _self_ = dynamic_cast<cv::ml::SVM*>(((pyopencv_ml_SVM_t*)self)->v.get());
    cv::TermCriteria retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTermCriteria());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVM_getType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVM' or its derivative)");
    cv::ml::SVM* _self_ = dynamic_cast<cv::ml::SVM*>(((pyopencv_ml_SVM_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getType());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVM_getUncompressedSupportVectors(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVM' or its derivative)");
    cv::ml::SVM* _self_ = dynamic_cast<cv::ml::SVM*>(((pyopencv_ml_SVM_t*)self)->v.get());
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getUncompressedSupportVectors());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVM_setC(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVM' or its derivative)");
    cv::ml::SVM* _self_ = dynamic_cast<cv::ml::SVM*>(((pyopencv_ml_SVM_t*)self)->v.get());
    double val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:ml_SVM.setC", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setC(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVM_setClassWeights(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVM' or its derivative)");
    cv::ml::SVM* _self_ = dynamic_cast<cv::ml::SVM*>(((pyopencv_ml_SVM_t*)self)->v.get());
    {
    PyObject* pyobj_val = NULL;
    Mat val;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:ml_SVM.setClassWeights", (char**)keywords, &pyobj_val) &&
        pyopencv_to(pyobj_val, val, ArgInfo("val", 0)) )
    {
        ERRWRAP2(_self_->setClassWeights(val));
        Py_RETURN_NONE;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_val = NULL;
    Mat val;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:ml_SVM.setClassWeights", (char**)keywords, &pyobj_val) &&
        pyopencv_to(pyobj_val, val, ArgInfo("val", 0)) )
    {
        ERRWRAP2(_self_->setClassWeights(val));
        Py_RETURN_NONE;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVM_setCoef0(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVM' or its derivative)");
    cv::ml::SVM* _self_ = dynamic_cast<cv::ml::SVM*>(((pyopencv_ml_SVM_t*)self)->v.get());
    double val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:ml_SVM.setCoef0", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setCoef0(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVM_setDegree(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVM' or its derivative)");
    cv::ml::SVM* _self_ = dynamic_cast<cv::ml::SVM*>(((pyopencv_ml_SVM_t*)self)->v.get());
    double val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:ml_SVM.setDegree", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setDegree(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVM_setGamma(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVM' or its derivative)");
    cv::ml::SVM* _self_ = dynamic_cast<cv::ml::SVM*>(((pyopencv_ml_SVM_t*)self)->v.get());
    double val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:ml_SVM.setGamma", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setGamma(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVM_setKernel(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVM' or its derivative)");
    cv::ml::SVM* _self_ = dynamic_cast<cv::ml::SVM*>(((pyopencv_ml_SVM_t*)self)->v.get());
    int kernelType=0;

    const char* keywords[] = { "kernelType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ml_SVM.setKernel", (char**)keywords, &kernelType) )
    {
        ERRWRAP2(_self_->setKernel(kernelType));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVM_setNu(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVM' or its derivative)");
    cv::ml::SVM* _self_ = dynamic_cast<cv::ml::SVM*>(((pyopencv_ml_SVM_t*)self)->v.get());
    double val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:ml_SVM.setNu", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setNu(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVM_setP(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVM' or its derivative)");
    cv::ml::SVM* _self_ = dynamic_cast<cv::ml::SVM*>(((pyopencv_ml_SVM_t*)self)->v.get());
    double val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:ml_SVM.setP", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setP(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVM_setTermCriteria(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVM' or its derivative)");
    cv::ml::SVM* _self_ = dynamic_cast<cv::ml::SVM*>(((pyopencv_ml_SVM_t*)self)->v.get());
    PyObject* pyobj_val = NULL;
    TermCriteria val;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:ml_SVM.setTermCriteria", (char**)keywords, &pyobj_val) &&
        pyopencv_to(pyobj_val, val, ArgInfo("val", 0)) )
    {
        ERRWRAP2(_self_->setTermCriteria(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVM_setType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVM' or its derivative)");
    cv::ml::SVM* _self_ = dynamic_cast<cv::ml::SVM*>(((pyopencv_ml_SVM_t*)self)->v.get());
    int val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ml_SVM.setType", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setType(val));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_ml_SVM_methods[] =
{
    {"getC", (PyCFunction)pyopencv_cv_ml_ml_SVM_getC, METH_VARARGS | METH_KEYWORDS, "getC() -> retval"},
    {"getClassWeights", (PyCFunction)pyopencv_cv_ml_ml_SVM_getClassWeights, METH_VARARGS | METH_KEYWORDS, "getClassWeights() -> retval"},
    {"getCoef0", (PyCFunction)pyopencv_cv_ml_ml_SVM_getCoef0, METH_VARARGS | METH_KEYWORDS, "getCoef0() -> retval"},
    {"getDecisionFunction", (PyCFunction)pyopencv_cv_ml_ml_SVM_getDecisionFunction, METH_VARARGS | METH_KEYWORDS, "getDecisionFunction(i[, alpha[, svidx]]) -> retval, alpha, svidx"},
    {"getDegree", (PyCFunction)pyopencv_cv_ml_ml_SVM_getDegree, METH_VARARGS | METH_KEYWORDS, "getDegree() -> retval"},
    {"getGamma", (PyCFunction)pyopencv_cv_ml_ml_SVM_getGamma, METH_VARARGS | METH_KEYWORDS, "getGamma() -> retval"},
    {"getKernelType", (PyCFunction)pyopencv_cv_ml_ml_SVM_getKernelType, METH_VARARGS | METH_KEYWORDS, "getKernelType() -> retval"},
    {"getNu", (PyCFunction)pyopencv_cv_ml_ml_SVM_getNu, METH_VARARGS | METH_KEYWORDS, "getNu() -> retval"},
    {"getP", (PyCFunction)pyopencv_cv_ml_ml_SVM_getP, METH_VARARGS | METH_KEYWORDS, "getP() -> retval"},
    {"getSupportVectors", (PyCFunction)pyopencv_cv_ml_ml_SVM_getSupportVectors, METH_VARARGS | METH_KEYWORDS, "getSupportVectors() -> retval"},
    {"getTermCriteria", (PyCFunction)pyopencv_cv_ml_ml_SVM_getTermCriteria, METH_VARARGS | METH_KEYWORDS, "getTermCriteria() -> retval"},
    {"getType", (PyCFunction)pyopencv_cv_ml_ml_SVM_getType, METH_VARARGS | METH_KEYWORDS, "getType() -> retval"},
    {"getUncompressedSupportVectors", (PyCFunction)pyopencv_cv_ml_ml_SVM_getUncompressedSupportVectors, METH_VARARGS | METH_KEYWORDS, "getUncompressedSupportVectors() -> retval"},
    {"setC", (PyCFunction)pyopencv_cv_ml_ml_SVM_setC, METH_VARARGS | METH_KEYWORDS, "setC(val) -> None"},
    {"setClassWeights", (PyCFunction)pyopencv_cv_ml_ml_SVM_setClassWeights, METH_VARARGS | METH_KEYWORDS, "setClassWeights(val) -> None"},
    {"setCoef0", (PyCFunction)pyopencv_cv_ml_ml_SVM_setCoef0, METH_VARARGS | METH_KEYWORDS, "setCoef0(val) -> None"},
    {"setDegree", (PyCFunction)pyopencv_cv_ml_ml_SVM_setDegree, METH_VARARGS | METH_KEYWORDS, "setDegree(val) -> None"},
    {"setGamma", (PyCFunction)pyopencv_cv_ml_ml_SVM_setGamma, METH_VARARGS | METH_KEYWORDS, "setGamma(val) -> None"},
    {"setKernel", (PyCFunction)pyopencv_cv_ml_ml_SVM_setKernel, METH_VARARGS | METH_KEYWORDS, "setKernel(kernelType) -> None"},
    {"setNu", (PyCFunction)pyopencv_cv_ml_ml_SVM_setNu, METH_VARARGS | METH_KEYWORDS, "setNu(val) -> None"},
    {"setP", (PyCFunction)pyopencv_cv_ml_ml_SVM_setP, METH_VARARGS | METH_KEYWORDS, "setP(val) -> None"},
    {"setTermCriteria", (PyCFunction)pyopencv_cv_ml_ml_SVM_setTermCriteria, METH_VARARGS | METH_KEYWORDS, "setTermCriteria(val) -> None"},
    {"setType", (PyCFunction)pyopencv_cv_ml_ml_SVM_setType, METH_VARARGS | METH_KEYWORDS, "setType(val) -> None"},

    {NULL,          NULL}
};

static void pyopencv_ml_SVM_specials(void)
{
    pyopencv_ml_SVM_Type.tp_base = &pyopencv_ml_StatModel_Type;
    pyopencv_ml_SVM_Type.tp_dealloc = pyopencv_ml_SVM_dealloc;
    pyopencv_ml_SVM_Type.tp_repr = pyopencv_ml_SVM_repr;
    pyopencv_ml_SVM_Type.tp_getset = pyopencv_ml_SVM_getseters;
    pyopencv_ml_SVM_Type.tp_methods = pyopencv_ml_SVM_methods;
}

static PyObject* pyopencv_ml_EM_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<ml_EM %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_ml_EM_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_ml_ml_EM_getClustersNumber(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_EM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_EM' or its derivative)");
    cv::ml::EM* _self_ = dynamic_cast<cv::ml::EM*>(((pyopencv_ml_EM_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getClustersNumber());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_EM_getCovarianceMatrixType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_EM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_EM' or its derivative)");
    cv::ml::EM* _self_ = dynamic_cast<cv::ml::EM*>(((pyopencv_ml_EM_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getCovarianceMatrixType());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_EM_getCovs(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_EM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_EM' or its derivative)");
    cv::ml::EM* _self_ = dynamic_cast<cv::ml::EM*>(((pyopencv_ml_EM_t*)self)->v.get());
    {
    PyObject* pyobj_covs = NULL;
    vector_Mat covs;

    const char* keywords[] = { "covs", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|O:ml_EM.getCovs", (char**)keywords, &pyobj_covs) &&
        pyopencv_to(pyobj_covs, covs, ArgInfo("covs", 1)) )
    {
        ERRWRAP2(_self_->getCovs(covs));
        return pyopencv_from(covs);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_covs = NULL;
    vector_Mat covs;

    const char* keywords[] = { "covs", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|O:ml_EM.getCovs", (char**)keywords, &pyobj_covs) &&
        pyopencv_to(pyobj_covs, covs, ArgInfo("covs", 1)) )
    {
        ERRWRAP2(_self_->getCovs(covs));
        return pyopencv_from(covs);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_EM_getMeans(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_EM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_EM' or its derivative)");
    cv::ml::EM* _self_ = dynamic_cast<cv::ml::EM*>(((pyopencv_ml_EM_t*)self)->v.get());
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getMeans());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_EM_getTermCriteria(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_EM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_EM' or its derivative)");
    cv::ml::EM* _self_ = dynamic_cast<cv::ml::EM*>(((pyopencv_ml_EM_t*)self)->v.get());
    TermCriteria retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTermCriteria());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_EM_getWeights(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_EM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_EM' or its derivative)");
    cv::ml::EM* _self_ = dynamic_cast<cv::ml::EM*>(((pyopencv_ml_EM_t*)self)->v.get());
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getWeights());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_EM_predict2(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_EM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_EM' or its derivative)");
    cv::ml::EM* _self_ = dynamic_cast<cv::ml::EM*>(((pyopencv_ml_EM_t*)self)->v.get());
    {
    PyObject* pyobj_sample = NULL;
    Mat sample;
    PyObject* pyobj_probs = NULL;
    Mat probs;
    Vec2d retval;

    const char* keywords[] = { "sample", "probs", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:ml_EM.predict2", (char**)keywords, &pyobj_sample, &pyobj_probs) &&
        pyopencv_to(pyobj_sample, sample, ArgInfo("sample", 0)) &&
        pyopencv_to(pyobj_probs, probs, ArgInfo("probs", 1)) )
    {
        ERRWRAP2(retval = _self_->predict2(sample, probs));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(probs));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_sample = NULL;
    UMat sample;
    PyObject* pyobj_probs = NULL;
    UMat probs;
    Vec2d retval;

    const char* keywords[] = { "sample", "probs", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:ml_EM.predict2", (char**)keywords, &pyobj_sample, &pyobj_probs) &&
        pyopencv_to(pyobj_sample, sample, ArgInfo("sample", 0)) &&
        pyopencv_to(pyobj_probs, probs, ArgInfo("probs", 1)) )
    {
        ERRWRAP2(retval = _self_->predict2(sample, probs));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(probs));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_EM_setClustersNumber(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_EM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_EM' or its derivative)");
    cv::ml::EM* _self_ = dynamic_cast<cv::ml::EM*>(((pyopencv_ml_EM_t*)self)->v.get());
    int val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ml_EM.setClustersNumber", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setClustersNumber(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_EM_setCovarianceMatrixType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_EM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_EM' or its derivative)");
    cv::ml::EM* _self_ = dynamic_cast<cv::ml::EM*>(((pyopencv_ml_EM_t*)self)->v.get());
    int val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ml_EM.setCovarianceMatrixType", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setCovarianceMatrixType(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_EM_setTermCriteria(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_EM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_EM' or its derivative)");
    cv::ml::EM* _self_ = dynamic_cast<cv::ml::EM*>(((pyopencv_ml_EM_t*)self)->v.get());
    PyObject* pyobj_val = NULL;
    TermCriteria val;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:ml_EM.setTermCriteria", (char**)keywords, &pyobj_val) &&
        pyopencv_to(pyobj_val, val, ArgInfo("val", 0)) )
    {
        ERRWRAP2(_self_->setTermCriteria(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_EM_trainE(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_EM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_EM' or its derivative)");
    cv::ml::EM* _self_ = dynamic_cast<cv::ml::EM*>(((pyopencv_ml_EM_t*)self)->v.get());
    {
    PyObject* pyobj_samples = NULL;
    Mat samples;
    PyObject* pyobj_means0 = NULL;
    Mat means0;
    PyObject* pyobj_covs0 = NULL;
    Mat covs0;
    PyObject* pyobj_weights0 = NULL;
    Mat weights0;
    PyObject* pyobj_logLikelihoods = NULL;
    Mat logLikelihoods;
    PyObject* pyobj_labels = NULL;
    Mat labels;
    PyObject* pyobj_probs = NULL;
    Mat probs;
    bool retval;

    const char* keywords[] = { "samples", "means0", "covs0", "weights0", "logLikelihoods", "labels", "probs", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOOOO:ml_EM.trainE", (char**)keywords, &pyobj_samples, &pyobj_means0, &pyobj_covs0, &pyobj_weights0, &pyobj_logLikelihoods, &pyobj_labels, &pyobj_probs) &&
        pyopencv_to(pyobj_samples, samples, ArgInfo("samples", 0)) &&
        pyopencv_to(pyobj_means0, means0, ArgInfo("means0", 0)) &&
        pyopencv_to(pyobj_covs0, covs0, ArgInfo("covs0", 0)) &&
        pyopencv_to(pyobj_weights0, weights0, ArgInfo("weights0", 0)) &&
        pyopencv_to(pyobj_logLikelihoods, logLikelihoods, ArgInfo("logLikelihoods", 1)) &&
        pyopencv_to(pyobj_labels, labels, ArgInfo("labels", 1)) &&
        pyopencv_to(pyobj_probs, probs, ArgInfo("probs", 1)) )
    {
        ERRWRAP2(retval = _self_->trainE(samples, means0, covs0, weights0, logLikelihoods, labels, probs));
        return Py_BuildValue("(NNNN)", pyopencv_from(retval), pyopencv_from(logLikelihoods), pyopencv_from(labels), pyopencv_from(probs));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_samples = NULL;
    UMat samples;
    PyObject* pyobj_means0 = NULL;
    UMat means0;
    PyObject* pyobj_covs0 = NULL;
    UMat covs0;
    PyObject* pyobj_weights0 = NULL;
    UMat weights0;
    PyObject* pyobj_logLikelihoods = NULL;
    UMat logLikelihoods;
    PyObject* pyobj_labels = NULL;
    UMat labels;
    PyObject* pyobj_probs = NULL;
    UMat probs;
    bool retval;

    const char* keywords[] = { "samples", "means0", "covs0", "weights0", "logLikelihoods", "labels", "probs", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOOOO:ml_EM.trainE", (char**)keywords, &pyobj_samples, &pyobj_means0, &pyobj_covs0, &pyobj_weights0, &pyobj_logLikelihoods, &pyobj_labels, &pyobj_probs) &&
        pyopencv_to(pyobj_samples, samples, ArgInfo("samples", 0)) &&
        pyopencv_to(pyobj_means0, means0, ArgInfo("means0", 0)) &&
        pyopencv_to(pyobj_covs0, covs0, ArgInfo("covs0", 0)) &&
        pyopencv_to(pyobj_weights0, weights0, ArgInfo("weights0", 0)) &&
        pyopencv_to(pyobj_logLikelihoods, logLikelihoods, ArgInfo("logLikelihoods", 1)) &&
        pyopencv_to(pyobj_labels, labels, ArgInfo("labels", 1)) &&
        pyopencv_to(pyobj_probs, probs, ArgInfo("probs", 1)) )
    {
        ERRWRAP2(retval = _self_->trainE(samples, means0, covs0, weights0, logLikelihoods, labels, probs));
        return Py_BuildValue("(NNNN)", pyopencv_from(retval), pyopencv_from(logLikelihoods), pyopencv_from(labels), pyopencv_from(probs));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_EM_trainEM(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_EM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_EM' or its derivative)");
    cv::ml::EM* _self_ = dynamic_cast<cv::ml::EM*>(((pyopencv_ml_EM_t*)self)->v.get());
    {
    PyObject* pyobj_samples = NULL;
    Mat samples;
    PyObject* pyobj_logLikelihoods = NULL;
    Mat logLikelihoods;
    PyObject* pyobj_labels = NULL;
    Mat labels;
    PyObject* pyobj_probs = NULL;
    Mat probs;
    bool retval;

    const char* keywords[] = { "samples", "logLikelihoods", "labels", "probs", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOO:ml_EM.trainEM", (char**)keywords, &pyobj_samples, &pyobj_logLikelihoods, &pyobj_labels, &pyobj_probs) &&
        pyopencv_to(pyobj_samples, samples, ArgInfo("samples", 0)) &&
        pyopencv_to(pyobj_logLikelihoods, logLikelihoods, ArgInfo("logLikelihoods", 1)) &&
        pyopencv_to(pyobj_labels, labels, ArgInfo("labels", 1)) &&
        pyopencv_to(pyobj_probs, probs, ArgInfo("probs", 1)) )
    {
        ERRWRAP2(retval = _self_->trainEM(samples, logLikelihoods, labels, probs));
        return Py_BuildValue("(NNNN)", pyopencv_from(retval), pyopencv_from(logLikelihoods), pyopencv_from(labels), pyopencv_from(probs));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_samples = NULL;
    UMat samples;
    PyObject* pyobj_logLikelihoods = NULL;
    UMat logLikelihoods;
    PyObject* pyobj_labels = NULL;
    UMat labels;
    PyObject* pyobj_probs = NULL;
    UMat probs;
    bool retval;

    const char* keywords[] = { "samples", "logLikelihoods", "labels", "probs", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOO:ml_EM.trainEM", (char**)keywords, &pyobj_samples, &pyobj_logLikelihoods, &pyobj_labels, &pyobj_probs) &&
        pyopencv_to(pyobj_samples, samples, ArgInfo("samples", 0)) &&
        pyopencv_to(pyobj_logLikelihoods, logLikelihoods, ArgInfo("logLikelihoods", 1)) &&
        pyopencv_to(pyobj_labels, labels, ArgInfo("labels", 1)) &&
        pyopencv_to(pyobj_probs, probs, ArgInfo("probs", 1)) )
    {
        ERRWRAP2(retval = _self_->trainEM(samples, logLikelihoods, labels, probs));
        return Py_BuildValue("(NNNN)", pyopencv_from(retval), pyopencv_from(logLikelihoods), pyopencv_from(labels), pyopencv_from(probs));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_EM_trainM(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_EM_Type))
        return failmsgp("Incorrect type of self (must be 'ml_EM' or its derivative)");
    cv::ml::EM* _self_ = dynamic_cast<cv::ml::EM*>(((pyopencv_ml_EM_t*)self)->v.get());
    {
    PyObject* pyobj_samples = NULL;
    Mat samples;
    PyObject* pyobj_probs0 = NULL;
    Mat probs0;
    PyObject* pyobj_logLikelihoods = NULL;
    Mat logLikelihoods;
    PyObject* pyobj_labels = NULL;
    Mat labels;
    PyObject* pyobj_probs = NULL;
    Mat probs;
    bool retval;

    const char* keywords[] = { "samples", "probs0", "logLikelihoods", "labels", "probs", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOO:ml_EM.trainM", (char**)keywords, &pyobj_samples, &pyobj_probs0, &pyobj_logLikelihoods, &pyobj_labels, &pyobj_probs) &&
        pyopencv_to(pyobj_samples, samples, ArgInfo("samples", 0)) &&
        pyopencv_to(pyobj_probs0, probs0, ArgInfo("probs0", 0)) &&
        pyopencv_to(pyobj_logLikelihoods, logLikelihoods, ArgInfo("logLikelihoods", 1)) &&
        pyopencv_to(pyobj_labels, labels, ArgInfo("labels", 1)) &&
        pyopencv_to(pyobj_probs, probs, ArgInfo("probs", 1)) )
    {
        ERRWRAP2(retval = _self_->trainM(samples, probs0, logLikelihoods, labels, probs));
        return Py_BuildValue("(NNNN)", pyopencv_from(retval), pyopencv_from(logLikelihoods), pyopencv_from(labels), pyopencv_from(probs));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_samples = NULL;
    UMat samples;
    PyObject* pyobj_probs0 = NULL;
    UMat probs0;
    PyObject* pyobj_logLikelihoods = NULL;
    UMat logLikelihoods;
    PyObject* pyobj_labels = NULL;
    UMat labels;
    PyObject* pyobj_probs = NULL;
    UMat probs;
    bool retval;

    const char* keywords[] = { "samples", "probs0", "logLikelihoods", "labels", "probs", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|OOO:ml_EM.trainM", (char**)keywords, &pyobj_samples, &pyobj_probs0, &pyobj_logLikelihoods, &pyobj_labels, &pyobj_probs) &&
        pyopencv_to(pyobj_samples, samples, ArgInfo("samples", 0)) &&
        pyopencv_to(pyobj_probs0, probs0, ArgInfo("probs0", 0)) &&
        pyopencv_to(pyobj_logLikelihoods, logLikelihoods, ArgInfo("logLikelihoods", 1)) &&
        pyopencv_to(pyobj_labels, labels, ArgInfo("labels", 1)) &&
        pyopencv_to(pyobj_probs, probs, ArgInfo("probs", 1)) )
    {
        ERRWRAP2(retval = _self_->trainM(samples, probs0, logLikelihoods, labels, probs));
        return Py_BuildValue("(NNNN)", pyopencv_from(retval), pyopencv_from(logLikelihoods), pyopencv_from(labels), pyopencv_from(probs));
    }
    }

    return NULL;
}



static PyMethodDef pyopencv_ml_EM_methods[] =
{
    {"getClustersNumber", (PyCFunction)pyopencv_cv_ml_ml_EM_getClustersNumber, METH_VARARGS | METH_KEYWORDS, "getClustersNumber() -> retval"},
    {"getCovarianceMatrixType", (PyCFunction)pyopencv_cv_ml_ml_EM_getCovarianceMatrixType, METH_VARARGS | METH_KEYWORDS, "getCovarianceMatrixType() -> retval"},
    {"getCovs", (PyCFunction)pyopencv_cv_ml_ml_EM_getCovs, METH_VARARGS | METH_KEYWORDS, "getCovs([, covs]) -> covs"},
    {"getMeans", (PyCFunction)pyopencv_cv_ml_ml_EM_getMeans, METH_VARARGS | METH_KEYWORDS, "getMeans() -> retval"},
    {"getTermCriteria", (PyCFunction)pyopencv_cv_ml_ml_EM_getTermCriteria, METH_VARARGS | METH_KEYWORDS, "getTermCriteria() -> retval"},
    {"getWeights", (PyCFunction)pyopencv_cv_ml_ml_EM_getWeights, METH_VARARGS | METH_KEYWORDS, "getWeights() -> retval"},
    {"predict2", (PyCFunction)pyopencv_cv_ml_ml_EM_predict2, METH_VARARGS | METH_KEYWORDS, "predict2(sample[, probs]) -> retval, probs"},
    {"setClustersNumber", (PyCFunction)pyopencv_cv_ml_ml_EM_setClustersNumber, METH_VARARGS | METH_KEYWORDS, "setClustersNumber(val) -> None"},
    {"setCovarianceMatrixType", (PyCFunction)pyopencv_cv_ml_ml_EM_setCovarianceMatrixType, METH_VARARGS | METH_KEYWORDS, "setCovarianceMatrixType(val) -> None"},
    {"setTermCriteria", (PyCFunction)pyopencv_cv_ml_ml_EM_setTermCriteria, METH_VARARGS | METH_KEYWORDS, "setTermCriteria(val) -> None"},
    {"trainE", (PyCFunction)pyopencv_cv_ml_ml_EM_trainE, METH_VARARGS | METH_KEYWORDS, "trainE(samples, means0[, covs0[, weights0[, logLikelihoods[, labels[, probs]]]]]) -> retval, logLikelihoods, labels, probs"},
    {"trainEM", (PyCFunction)pyopencv_cv_ml_ml_EM_trainEM, METH_VARARGS | METH_KEYWORDS, "trainEM(samples[, logLikelihoods[, labels[, probs]]]) -> retval, logLikelihoods, labels, probs"},
    {"trainM", (PyCFunction)pyopencv_cv_ml_ml_EM_trainM, METH_VARARGS | METH_KEYWORDS, "trainM(samples, probs0[, logLikelihoods[, labels[, probs]]]) -> retval, logLikelihoods, labels, probs"},

    {NULL,          NULL}
};

static void pyopencv_ml_EM_specials(void)
{
    pyopencv_ml_EM_Type.tp_base = &pyopencv_ml_StatModel_Type;
    pyopencv_ml_EM_Type.tp_dealloc = pyopencv_ml_EM_dealloc;
    pyopencv_ml_EM_Type.tp_repr = pyopencv_ml_EM_repr;
    pyopencv_ml_EM_Type.tp_getset = pyopencv_ml_EM_getseters;
    pyopencv_ml_EM_Type.tp_methods = pyopencv_ml_EM_methods;
}

static PyObject* pyopencv_ml_DTrees_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<ml_DTrees %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_ml_DTrees_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_ml_ml_DTrees_getCVFolds(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_DTrees_Type))
        return failmsgp("Incorrect type of self (must be 'ml_DTrees' or its derivative)");
    cv::ml::DTrees* _self_ = dynamic_cast<cv::ml::DTrees*>(((pyopencv_ml_DTrees_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getCVFolds());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_DTrees_getMaxCategories(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_DTrees_Type))
        return failmsgp("Incorrect type of self (must be 'ml_DTrees' or its derivative)");
    cv::ml::DTrees* _self_ = dynamic_cast<cv::ml::DTrees*>(((pyopencv_ml_DTrees_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getMaxCategories());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_DTrees_getMaxDepth(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_DTrees_Type))
        return failmsgp("Incorrect type of self (must be 'ml_DTrees' or its derivative)");
    cv::ml::DTrees* _self_ = dynamic_cast<cv::ml::DTrees*>(((pyopencv_ml_DTrees_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getMaxDepth());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_DTrees_getMinSampleCount(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_DTrees_Type))
        return failmsgp("Incorrect type of self (must be 'ml_DTrees' or its derivative)");
    cv::ml::DTrees* _self_ = dynamic_cast<cv::ml::DTrees*>(((pyopencv_ml_DTrees_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getMinSampleCount());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_DTrees_getPriors(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_DTrees_Type))
        return failmsgp("Incorrect type of self (must be 'ml_DTrees' or its derivative)");
    cv::ml::DTrees* _self_ = dynamic_cast<cv::ml::DTrees*>(((pyopencv_ml_DTrees_t*)self)->v.get());
    cv::Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getPriors());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_DTrees_getRegressionAccuracy(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_DTrees_Type))
        return failmsgp("Incorrect type of self (must be 'ml_DTrees' or its derivative)");
    cv::ml::DTrees* _self_ = dynamic_cast<cv::ml::DTrees*>(((pyopencv_ml_DTrees_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getRegressionAccuracy());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_DTrees_getTruncatePrunedTree(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_DTrees_Type))
        return failmsgp("Incorrect type of self (must be 'ml_DTrees' or its derivative)");
    cv::ml::DTrees* _self_ = dynamic_cast<cv::ml::DTrees*>(((pyopencv_ml_DTrees_t*)self)->v.get());
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTruncatePrunedTree());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_DTrees_getUse1SERule(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_DTrees_Type))
        return failmsgp("Incorrect type of self (must be 'ml_DTrees' or its derivative)");
    cv::ml::DTrees* _self_ = dynamic_cast<cv::ml::DTrees*>(((pyopencv_ml_DTrees_t*)self)->v.get());
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getUse1SERule());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_DTrees_getUseSurrogates(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_DTrees_Type))
        return failmsgp("Incorrect type of self (must be 'ml_DTrees' or its derivative)");
    cv::ml::DTrees* _self_ = dynamic_cast<cv::ml::DTrees*>(((pyopencv_ml_DTrees_t*)self)->v.get());
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getUseSurrogates());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_DTrees_setCVFolds(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_DTrees_Type))
        return failmsgp("Incorrect type of self (must be 'ml_DTrees' or its derivative)");
    cv::ml::DTrees* _self_ = dynamic_cast<cv::ml::DTrees*>(((pyopencv_ml_DTrees_t*)self)->v.get());
    int val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ml_DTrees.setCVFolds", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setCVFolds(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_DTrees_setMaxCategories(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_DTrees_Type))
        return failmsgp("Incorrect type of self (must be 'ml_DTrees' or its derivative)");
    cv::ml::DTrees* _self_ = dynamic_cast<cv::ml::DTrees*>(((pyopencv_ml_DTrees_t*)self)->v.get());
    int val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ml_DTrees.setMaxCategories", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setMaxCategories(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_DTrees_setMaxDepth(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_DTrees_Type))
        return failmsgp("Incorrect type of self (must be 'ml_DTrees' or its derivative)");
    cv::ml::DTrees* _self_ = dynamic_cast<cv::ml::DTrees*>(((pyopencv_ml_DTrees_t*)self)->v.get());
    int val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ml_DTrees.setMaxDepth", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setMaxDepth(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_DTrees_setMinSampleCount(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_DTrees_Type))
        return failmsgp("Incorrect type of self (must be 'ml_DTrees' or its derivative)");
    cv::ml::DTrees* _self_ = dynamic_cast<cv::ml::DTrees*>(((pyopencv_ml_DTrees_t*)self)->v.get());
    int val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ml_DTrees.setMinSampleCount", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setMinSampleCount(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_DTrees_setPriors(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_DTrees_Type))
        return failmsgp("Incorrect type of self (must be 'ml_DTrees' or its derivative)");
    cv::ml::DTrees* _self_ = dynamic_cast<cv::ml::DTrees*>(((pyopencv_ml_DTrees_t*)self)->v.get());
    {
    PyObject* pyobj_val = NULL;
    Mat val;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:ml_DTrees.setPriors", (char**)keywords, &pyobj_val) &&
        pyopencv_to(pyobj_val, val, ArgInfo("val", 0)) )
    {
        ERRWRAP2(_self_->setPriors(val));
        Py_RETURN_NONE;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_val = NULL;
    Mat val;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:ml_DTrees.setPriors", (char**)keywords, &pyobj_val) &&
        pyopencv_to(pyobj_val, val, ArgInfo("val", 0)) )
    {
        ERRWRAP2(_self_->setPriors(val));
        Py_RETURN_NONE;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_DTrees_setRegressionAccuracy(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_DTrees_Type))
        return failmsgp("Incorrect type of self (must be 'ml_DTrees' or its derivative)");
    cv::ml::DTrees* _self_ = dynamic_cast<cv::ml::DTrees*>(((pyopencv_ml_DTrees_t*)self)->v.get());
    float val=0.f;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:ml_DTrees.setRegressionAccuracy", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setRegressionAccuracy(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_DTrees_setTruncatePrunedTree(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_DTrees_Type))
        return failmsgp("Incorrect type of self (must be 'ml_DTrees' or its derivative)");
    cv::ml::DTrees* _self_ = dynamic_cast<cv::ml::DTrees*>(((pyopencv_ml_DTrees_t*)self)->v.get());
    bool val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:ml_DTrees.setTruncatePrunedTree", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setTruncatePrunedTree(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_DTrees_setUse1SERule(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_DTrees_Type))
        return failmsgp("Incorrect type of self (must be 'ml_DTrees' or its derivative)");
    cv::ml::DTrees* _self_ = dynamic_cast<cv::ml::DTrees*>(((pyopencv_ml_DTrees_t*)self)->v.get());
    bool val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:ml_DTrees.setUse1SERule", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setUse1SERule(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_DTrees_setUseSurrogates(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_DTrees_Type))
        return failmsgp("Incorrect type of self (must be 'ml_DTrees' or its derivative)");
    cv::ml::DTrees* _self_ = dynamic_cast<cv::ml::DTrees*>(((pyopencv_ml_DTrees_t*)self)->v.get());
    bool val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:ml_DTrees.setUseSurrogates", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setUseSurrogates(val));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_ml_DTrees_methods[] =
{
    {"getCVFolds", (PyCFunction)pyopencv_cv_ml_ml_DTrees_getCVFolds, METH_VARARGS | METH_KEYWORDS, "getCVFolds() -> retval"},
    {"getMaxCategories", (PyCFunction)pyopencv_cv_ml_ml_DTrees_getMaxCategories, METH_VARARGS | METH_KEYWORDS, "getMaxCategories() -> retval"},
    {"getMaxDepth", (PyCFunction)pyopencv_cv_ml_ml_DTrees_getMaxDepth, METH_VARARGS | METH_KEYWORDS, "getMaxDepth() -> retval"},
    {"getMinSampleCount", (PyCFunction)pyopencv_cv_ml_ml_DTrees_getMinSampleCount, METH_VARARGS | METH_KEYWORDS, "getMinSampleCount() -> retval"},
    {"getPriors", (PyCFunction)pyopencv_cv_ml_ml_DTrees_getPriors, METH_VARARGS | METH_KEYWORDS, "getPriors() -> retval"},
    {"getRegressionAccuracy", (PyCFunction)pyopencv_cv_ml_ml_DTrees_getRegressionAccuracy, METH_VARARGS | METH_KEYWORDS, "getRegressionAccuracy() -> retval"},
    {"getTruncatePrunedTree", (PyCFunction)pyopencv_cv_ml_ml_DTrees_getTruncatePrunedTree, METH_VARARGS | METH_KEYWORDS, "getTruncatePrunedTree() -> retval"},
    {"getUse1SERule", (PyCFunction)pyopencv_cv_ml_ml_DTrees_getUse1SERule, METH_VARARGS | METH_KEYWORDS, "getUse1SERule() -> retval"},
    {"getUseSurrogates", (PyCFunction)pyopencv_cv_ml_ml_DTrees_getUseSurrogates, METH_VARARGS | METH_KEYWORDS, "getUseSurrogates() -> retval"},
    {"setCVFolds", (PyCFunction)pyopencv_cv_ml_ml_DTrees_setCVFolds, METH_VARARGS | METH_KEYWORDS, "setCVFolds(val) -> None"},
    {"setMaxCategories", (PyCFunction)pyopencv_cv_ml_ml_DTrees_setMaxCategories, METH_VARARGS | METH_KEYWORDS, "setMaxCategories(val) -> None"},
    {"setMaxDepth", (PyCFunction)pyopencv_cv_ml_ml_DTrees_setMaxDepth, METH_VARARGS | METH_KEYWORDS, "setMaxDepth(val) -> None"},
    {"setMinSampleCount", (PyCFunction)pyopencv_cv_ml_ml_DTrees_setMinSampleCount, METH_VARARGS | METH_KEYWORDS, "setMinSampleCount(val) -> None"},
    {"setPriors", (PyCFunction)pyopencv_cv_ml_ml_DTrees_setPriors, METH_VARARGS | METH_KEYWORDS, "setPriors(val) -> None"},
    {"setRegressionAccuracy", (PyCFunction)pyopencv_cv_ml_ml_DTrees_setRegressionAccuracy, METH_VARARGS | METH_KEYWORDS, "setRegressionAccuracy(val) -> None"},
    {"setTruncatePrunedTree", (PyCFunction)pyopencv_cv_ml_ml_DTrees_setTruncatePrunedTree, METH_VARARGS | METH_KEYWORDS, "setTruncatePrunedTree(val) -> None"},
    {"setUse1SERule", (PyCFunction)pyopencv_cv_ml_ml_DTrees_setUse1SERule, METH_VARARGS | METH_KEYWORDS, "setUse1SERule(val) -> None"},
    {"setUseSurrogates", (PyCFunction)pyopencv_cv_ml_ml_DTrees_setUseSurrogates, METH_VARARGS | METH_KEYWORDS, "setUseSurrogates(val) -> None"},

    {NULL,          NULL}
};

static void pyopencv_ml_DTrees_specials(void)
{
    pyopencv_ml_DTrees_Type.tp_base = &pyopencv_ml_StatModel_Type;
    pyopencv_ml_DTrees_Type.tp_dealloc = pyopencv_ml_DTrees_dealloc;
    pyopencv_ml_DTrees_Type.tp_repr = pyopencv_ml_DTrees_repr;
    pyopencv_ml_DTrees_Type.tp_getset = pyopencv_ml_DTrees_getseters;
    pyopencv_ml_DTrees_Type.tp_methods = pyopencv_ml_DTrees_methods;
}

static PyObject* pyopencv_ml_RTrees_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<ml_RTrees %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_ml_RTrees_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_ml_ml_RTrees_getActiveVarCount(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_RTrees_Type))
        return failmsgp("Incorrect type of self (must be 'ml_RTrees' or its derivative)");
    cv::ml::RTrees* _self_ = dynamic_cast<cv::ml::RTrees*>(((pyopencv_ml_RTrees_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getActiveVarCount());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_RTrees_getCalculateVarImportance(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_RTrees_Type))
        return failmsgp("Incorrect type of self (must be 'ml_RTrees' or its derivative)");
    cv::ml::RTrees* _self_ = dynamic_cast<cv::ml::RTrees*>(((pyopencv_ml_RTrees_t*)self)->v.get());
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getCalculateVarImportance());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_RTrees_getTermCriteria(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_RTrees_Type))
        return failmsgp("Incorrect type of self (must be 'ml_RTrees' or its derivative)");
    cv::ml::RTrees* _self_ = dynamic_cast<cv::ml::RTrees*>(((pyopencv_ml_RTrees_t*)self)->v.get());
    TermCriteria retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTermCriteria());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_RTrees_getVarImportance(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_RTrees_Type))
        return failmsgp("Incorrect type of self (must be 'ml_RTrees' or its derivative)");
    cv::ml::RTrees* _self_ = dynamic_cast<cv::ml::RTrees*>(((pyopencv_ml_RTrees_t*)self)->v.get());
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getVarImportance());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_RTrees_setActiveVarCount(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_RTrees_Type))
        return failmsgp("Incorrect type of self (must be 'ml_RTrees' or its derivative)");
    cv::ml::RTrees* _self_ = dynamic_cast<cv::ml::RTrees*>(((pyopencv_ml_RTrees_t*)self)->v.get());
    int val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ml_RTrees.setActiveVarCount", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setActiveVarCount(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_RTrees_setCalculateVarImportance(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_RTrees_Type))
        return failmsgp("Incorrect type of self (must be 'ml_RTrees' or its derivative)");
    cv::ml::RTrees* _self_ = dynamic_cast<cv::ml::RTrees*>(((pyopencv_ml_RTrees_t*)self)->v.get());
    bool val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:ml_RTrees.setCalculateVarImportance", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setCalculateVarImportance(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_RTrees_setTermCriteria(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_RTrees_Type))
        return failmsgp("Incorrect type of self (must be 'ml_RTrees' or its derivative)");
    cv::ml::RTrees* _self_ = dynamic_cast<cv::ml::RTrees*>(((pyopencv_ml_RTrees_t*)self)->v.get());
    PyObject* pyobj_val = NULL;
    TermCriteria val;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:ml_RTrees.setTermCriteria", (char**)keywords, &pyobj_val) &&
        pyopencv_to(pyobj_val, val, ArgInfo("val", 0)) )
    {
        ERRWRAP2(_self_->setTermCriteria(val));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_ml_RTrees_methods[] =
{
    {"getActiveVarCount", (PyCFunction)pyopencv_cv_ml_ml_RTrees_getActiveVarCount, METH_VARARGS | METH_KEYWORDS, "getActiveVarCount() -> retval"},
    {"getCalculateVarImportance", (PyCFunction)pyopencv_cv_ml_ml_RTrees_getCalculateVarImportance, METH_VARARGS | METH_KEYWORDS, "getCalculateVarImportance() -> retval"},
    {"getTermCriteria", (PyCFunction)pyopencv_cv_ml_ml_RTrees_getTermCriteria, METH_VARARGS | METH_KEYWORDS, "getTermCriteria() -> retval"},
    {"getVarImportance", (PyCFunction)pyopencv_cv_ml_ml_RTrees_getVarImportance, METH_VARARGS | METH_KEYWORDS, "getVarImportance() -> retval"},
    {"setActiveVarCount", (PyCFunction)pyopencv_cv_ml_ml_RTrees_setActiveVarCount, METH_VARARGS | METH_KEYWORDS, "setActiveVarCount(val) -> None"},
    {"setCalculateVarImportance", (PyCFunction)pyopencv_cv_ml_ml_RTrees_setCalculateVarImportance, METH_VARARGS | METH_KEYWORDS, "setCalculateVarImportance(val) -> None"},
    {"setTermCriteria", (PyCFunction)pyopencv_cv_ml_ml_RTrees_setTermCriteria, METH_VARARGS | METH_KEYWORDS, "setTermCriteria(val) -> None"},

    {NULL,          NULL}
};

static void pyopencv_ml_RTrees_specials(void)
{
    pyopencv_ml_RTrees_Type.tp_base = &pyopencv_ml_DTrees_Type;
    pyopencv_ml_RTrees_Type.tp_dealloc = pyopencv_ml_RTrees_dealloc;
    pyopencv_ml_RTrees_Type.tp_repr = pyopencv_ml_RTrees_repr;
    pyopencv_ml_RTrees_Type.tp_getset = pyopencv_ml_RTrees_getseters;
    pyopencv_ml_RTrees_Type.tp_methods = pyopencv_ml_RTrees_methods;
}

static PyObject* pyopencv_ml_Boost_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<ml_Boost %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_ml_Boost_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_ml_ml_Boost_getBoostType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_Boost_Type))
        return failmsgp("Incorrect type of self (must be 'ml_Boost' or its derivative)");
    cv::ml::Boost* _self_ = ((pyopencv_ml_Boost_t*)self)->v.get();
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getBoostType());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_Boost_getWeakCount(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_Boost_Type))
        return failmsgp("Incorrect type of self (must be 'ml_Boost' or its derivative)");
    cv::ml::Boost* _self_ = ((pyopencv_ml_Boost_t*)self)->v.get();
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getWeakCount());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_Boost_getWeightTrimRate(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_Boost_Type))
        return failmsgp("Incorrect type of self (must be 'ml_Boost' or its derivative)");
    cv::ml::Boost* _self_ = ((pyopencv_ml_Boost_t*)self)->v.get();
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getWeightTrimRate());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_Boost_setBoostType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_Boost_Type))
        return failmsgp("Incorrect type of self (must be 'ml_Boost' or its derivative)");
    cv::ml::Boost* _self_ = ((pyopencv_ml_Boost_t*)self)->v.get();
    int val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ml_Boost.setBoostType", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setBoostType(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_Boost_setWeakCount(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_Boost_Type))
        return failmsgp("Incorrect type of self (must be 'ml_Boost' or its derivative)");
    cv::ml::Boost* _self_ = ((pyopencv_ml_Boost_t*)self)->v.get();
    int val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ml_Boost.setWeakCount", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setWeakCount(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_Boost_setWeightTrimRate(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_Boost_Type))
        return failmsgp("Incorrect type of self (must be 'ml_Boost' or its derivative)");
    cv::ml::Boost* _self_ = ((pyopencv_ml_Boost_t*)self)->v.get();
    double val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:ml_Boost.setWeightTrimRate", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setWeightTrimRate(val));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_ml_Boost_methods[] =
{
    {"getBoostType", (PyCFunction)pyopencv_cv_ml_ml_Boost_getBoostType, METH_VARARGS | METH_KEYWORDS, "getBoostType() -> retval"},
    {"getWeakCount", (PyCFunction)pyopencv_cv_ml_ml_Boost_getWeakCount, METH_VARARGS | METH_KEYWORDS, "getWeakCount() -> retval"},
    {"getWeightTrimRate", (PyCFunction)pyopencv_cv_ml_ml_Boost_getWeightTrimRate, METH_VARARGS | METH_KEYWORDS, "getWeightTrimRate() -> retval"},
    {"setBoostType", (PyCFunction)pyopencv_cv_ml_ml_Boost_setBoostType, METH_VARARGS | METH_KEYWORDS, "setBoostType(val) -> None"},
    {"setWeakCount", (PyCFunction)pyopencv_cv_ml_ml_Boost_setWeakCount, METH_VARARGS | METH_KEYWORDS, "setWeakCount(val) -> None"},
    {"setWeightTrimRate", (PyCFunction)pyopencv_cv_ml_ml_Boost_setWeightTrimRate, METH_VARARGS | METH_KEYWORDS, "setWeightTrimRate(val) -> None"},

    {NULL,          NULL}
};

static void pyopencv_ml_Boost_specials(void)
{
    pyopencv_ml_Boost_Type.tp_base = &pyopencv_ml_DTrees_Type;
    pyopencv_ml_Boost_Type.tp_dealloc = pyopencv_ml_Boost_dealloc;
    pyopencv_ml_Boost_Type.tp_repr = pyopencv_ml_Boost_repr;
    pyopencv_ml_Boost_Type.tp_getset = pyopencv_ml_Boost_getseters;
    pyopencv_ml_Boost_Type.tp_methods = pyopencv_ml_Boost_methods;
}

static PyObject* pyopencv_ml_ANN_MLP_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<ml_ANN_MLP %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_ml_ANN_MLP_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_ml_ml_ANN_MLP_getBackpropMomentumScale(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_ANN_MLP_Type))
        return failmsgp("Incorrect type of self (must be 'ml_ANN_MLP' or its derivative)");
    cv::ml::ANN_MLP* _self_ = dynamic_cast<cv::ml::ANN_MLP*>(((pyopencv_ml_ANN_MLP_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getBackpropMomentumScale());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_ANN_MLP_getBackpropWeightScale(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_ANN_MLP_Type))
        return failmsgp("Incorrect type of self (must be 'ml_ANN_MLP' or its derivative)");
    cv::ml::ANN_MLP* _self_ = dynamic_cast<cv::ml::ANN_MLP*>(((pyopencv_ml_ANN_MLP_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getBackpropWeightScale());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_ANN_MLP_getLayerSizes(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_ANN_MLP_Type))
        return failmsgp("Incorrect type of self (must be 'ml_ANN_MLP' or its derivative)");
    cv::ml::ANN_MLP* _self_ = dynamic_cast<cv::ml::ANN_MLP*>(((pyopencv_ml_ANN_MLP_t*)self)->v.get());
    cv::Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getLayerSizes());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_ANN_MLP_getRpropDW0(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_ANN_MLP_Type))
        return failmsgp("Incorrect type of self (must be 'ml_ANN_MLP' or its derivative)");
    cv::ml::ANN_MLP* _self_ = dynamic_cast<cv::ml::ANN_MLP*>(((pyopencv_ml_ANN_MLP_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getRpropDW0());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_ANN_MLP_getRpropDWMax(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_ANN_MLP_Type))
        return failmsgp("Incorrect type of self (must be 'ml_ANN_MLP' or its derivative)");
    cv::ml::ANN_MLP* _self_ = dynamic_cast<cv::ml::ANN_MLP*>(((pyopencv_ml_ANN_MLP_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getRpropDWMax());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_ANN_MLP_getRpropDWMin(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_ANN_MLP_Type))
        return failmsgp("Incorrect type of self (must be 'ml_ANN_MLP' or its derivative)");
    cv::ml::ANN_MLP* _self_ = dynamic_cast<cv::ml::ANN_MLP*>(((pyopencv_ml_ANN_MLP_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getRpropDWMin());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_ANN_MLP_getRpropDWMinus(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_ANN_MLP_Type))
        return failmsgp("Incorrect type of self (must be 'ml_ANN_MLP' or its derivative)");
    cv::ml::ANN_MLP* _self_ = dynamic_cast<cv::ml::ANN_MLP*>(((pyopencv_ml_ANN_MLP_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getRpropDWMinus());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_ANN_MLP_getRpropDWPlus(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_ANN_MLP_Type))
        return failmsgp("Incorrect type of self (must be 'ml_ANN_MLP' or its derivative)");
    cv::ml::ANN_MLP* _self_ = dynamic_cast<cv::ml::ANN_MLP*>(((pyopencv_ml_ANN_MLP_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getRpropDWPlus());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_ANN_MLP_getTermCriteria(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_ANN_MLP_Type))
        return failmsgp("Incorrect type of self (must be 'ml_ANN_MLP' or its derivative)");
    cv::ml::ANN_MLP* _self_ = dynamic_cast<cv::ml::ANN_MLP*>(((pyopencv_ml_ANN_MLP_t*)self)->v.get());
    TermCriteria retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTermCriteria());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_ANN_MLP_getTrainMethod(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_ANN_MLP_Type))
        return failmsgp("Incorrect type of self (must be 'ml_ANN_MLP' or its derivative)");
    cv::ml::ANN_MLP* _self_ = dynamic_cast<cv::ml::ANN_MLP*>(((pyopencv_ml_ANN_MLP_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTrainMethod());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_ANN_MLP_getWeights(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_ANN_MLP_Type))
        return failmsgp("Incorrect type of self (must be 'ml_ANN_MLP' or its derivative)");
    cv::ml::ANN_MLP* _self_ = dynamic_cast<cv::ml::ANN_MLP*>(((pyopencv_ml_ANN_MLP_t*)self)->v.get());
    int layerIdx=0;
    Mat retval;

    const char* keywords[] = { "layerIdx", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ml_ANN_MLP.getWeights", (char**)keywords, &layerIdx) )
    {
        ERRWRAP2(retval = _self_->getWeights(layerIdx));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_ANN_MLP_setActivationFunction(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_ANN_MLP_Type))
        return failmsgp("Incorrect type of self (must be 'ml_ANN_MLP' or its derivative)");
    cv::ml::ANN_MLP* _self_ = dynamic_cast<cv::ml::ANN_MLP*>(((pyopencv_ml_ANN_MLP_t*)self)->v.get());
    int type=0;
    double param1=0;
    double param2=0;

    const char* keywords[] = { "type", "param1", "param2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i|dd:ml_ANN_MLP.setActivationFunction", (char**)keywords, &type, &param1, &param2) )
    {
        ERRWRAP2(_self_->setActivationFunction(type, param1, param2));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_ANN_MLP_setBackpropMomentumScale(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_ANN_MLP_Type))
        return failmsgp("Incorrect type of self (must be 'ml_ANN_MLP' or its derivative)");
    cv::ml::ANN_MLP* _self_ = dynamic_cast<cv::ml::ANN_MLP*>(((pyopencv_ml_ANN_MLP_t*)self)->v.get());
    double val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:ml_ANN_MLP.setBackpropMomentumScale", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setBackpropMomentumScale(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_ANN_MLP_setBackpropWeightScale(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_ANN_MLP_Type))
        return failmsgp("Incorrect type of self (must be 'ml_ANN_MLP' or its derivative)");
    cv::ml::ANN_MLP* _self_ = dynamic_cast<cv::ml::ANN_MLP*>(((pyopencv_ml_ANN_MLP_t*)self)->v.get());
    double val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:ml_ANN_MLP.setBackpropWeightScale", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setBackpropWeightScale(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_ANN_MLP_setLayerSizes(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_ANN_MLP_Type))
        return failmsgp("Incorrect type of self (must be 'ml_ANN_MLP' or its derivative)");
    cv::ml::ANN_MLP* _self_ = dynamic_cast<cv::ml::ANN_MLP*>(((pyopencv_ml_ANN_MLP_t*)self)->v.get());
    {
    PyObject* pyobj__layer_sizes = NULL;
    Mat _layer_sizes;

    const char* keywords[] = { "_layer_sizes", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:ml_ANN_MLP.setLayerSizes", (char**)keywords, &pyobj__layer_sizes) &&
        pyopencv_to(pyobj__layer_sizes, _layer_sizes, ArgInfo("_layer_sizes", 0)) )
    {
        ERRWRAP2(_self_->setLayerSizes(_layer_sizes));
        Py_RETURN_NONE;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj__layer_sizes = NULL;
    UMat _layer_sizes;

    const char* keywords[] = { "_layer_sizes", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:ml_ANN_MLP.setLayerSizes", (char**)keywords, &pyobj__layer_sizes) &&
        pyopencv_to(pyobj__layer_sizes, _layer_sizes, ArgInfo("_layer_sizes", 0)) )
    {
        ERRWRAP2(_self_->setLayerSizes(_layer_sizes));
        Py_RETURN_NONE;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_ANN_MLP_setRpropDW0(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_ANN_MLP_Type))
        return failmsgp("Incorrect type of self (must be 'ml_ANN_MLP' or its derivative)");
    cv::ml::ANN_MLP* _self_ = dynamic_cast<cv::ml::ANN_MLP*>(((pyopencv_ml_ANN_MLP_t*)self)->v.get());
    double val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:ml_ANN_MLP.setRpropDW0", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setRpropDW0(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_ANN_MLP_setRpropDWMax(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_ANN_MLP_Type))
        return failmsgp("Incorrect type of self (must be 'ml_ANN_MLP' or its derivative)");
    cv::ml::ANN_MLP* _self_ = dynamic_cast<cv::ml::ANN_MLP*>(((pyopencv_ml_ANN_MLP_t*)self)->v.get());
    double val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:ml_ANN_MLP.setRpropDWMax", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setRpropDWMax(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_ANN_MLP_setRpropDWMin(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_ANN_MLP_Type))
        return failmsgp("Incorrect type of self (must be 'ml_ANN_MLP' or its derivative)");
    cv::ml::ANN_MLP* _self_ = dynamic_cast<cv::ml::ANN_MLP*>(((pyopencv_ml_ANN_MLP_t*)self)->v.get());
    double val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:ml_ANN_MLP.setRpropDWMin", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setRpropDWMin(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_ANN_MLP_setRpropDWMinus(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_ANN_MLP_Type))
        return failmsgp("Incorrect type of self (must be 'ml_ANN_MLP' or its derivative)");
    cv::ml::ANN_MLP* _self_ = dynamic_cast<cv::ml::ANN_MLP*>(((pyopencv_ml_ANN_MLP_t*)self)->v.get());
    double val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:ml_ANN_MLP.setRpropDWMinus", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setRpropDWMinus(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_ANN_MLP_setRpropDWPlus(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_ANN_MLP_Type))
        return failmsgp("Incorrect type of self (must be 'ml_ANN_MLP' or its derivative)");
    cv::ml::ANN_MLP* _self_ = dynamic_cast<cv::ml::ANN_MLP*>(((pyopencv_ml_ANN_MLP_t*)self)->v.get());
    double val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:ml_ANN_MLP.setRpropDWPlus", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setRpropDWPlus(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_ANN_MLP_setTermCriteria(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_ANN_MLP_Type))
        return failmsgp("Incorrect type of self (must be 'ml_ANN_MLP' or its derivative)");
    cv::ml::ANN_MLP* _self_ = dynamic_cast<cv::ml::ANN_MLP*>(((pyopencv_ml_ANN_MLP_t*)self)->v.get());
    PyObject* pyobj_val = NULL;
    TermCriteria val;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:ml_ANN_MLP.setTermCriteria", (char**)keywords, &pyobj_val) &&
        pyopencv_to(pyobj_val, val, ArgInfo("val", 0)) )
    {
        ERRWRAP2(_self_->setTermCriteria(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_ANN_MLP_setTrainMethod(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_ANN_MLP_Type))
        return failmsgp("Incorrect type of self (must be 'ml_ANN_MLP' or its derivative)");
    cv::ml::ANN_MLP* _self_ = dynamic_cast<cv::ml::ANN_MLP*>(((pyopencv_ml_ANN_MLP_t*)self)->v.get());
    int method=0;
    double param1=0;
    double param2=0;

    const char* keywords[] = { "method", "param1", "param2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i|dd:ml_ANN_MLP.setTrainMethod", (char**)keywords, &method, &param1, &param2) )
    {
        ERRWRAP2(_self_->setTrainMethod(method, param1, param2));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_ml_ANN_MLP_methods[] =
{
    {"getBackpropMomentumScale", (PyCFunction)pyopencv_cv_ml_ml_ANN_MLP_getBackpropMomentumScale, METH_VARARGS | METH_KEYWORDS, "getBackpropMomentumScale() -> retval"},
    {"getBackpropWeightScale", (PyCFunction)pyopencv_cv_ml_ml_ANN_MLP_getBackpropWeightScale, METH_VARARGS | METH_KEYWORDS, "getBackpropWeightScale() -> retval"},
    {"getLayerSizes", (PyCFunction)pyopencv_cv_ml_ml_ANN_MLP_getLayerSizes, METH_VARARGS | METH_KEYWORDS, "getLayerSizes() -> retval"},
    {"getRpropDW0", (PyCFunction)pyopencv_cv_ml_ml_ANN_MLP_getRpropDW0, METH_VARARGS | METH_KEYWORDS, "getRpropDW0() -> retval"},
    {"getRpropDWMax", (PyCFunction)pyopencv_cv_ml_ml_ANN_MLP_getRpropDWMax, METH_VARARGS | METH_KEYWORDS, "getRpropDWMax() -> retval"},
    {"getRpropDWMin", (PyCFunction)pyopencv_cv_ml_ml_ANN_MLP_getRpropDWMin, METH_VARARGS | METH_KEYWORDS, "getRpropDWMin() -> retval"},
    {"getRpropDWMinus", (PyCFunction)pyopencv_cv_ml_ml_ANN_MLP_getRpropDWMinus, METH_VARARGS | METH_KEYWORDS, "getRpropDWMinus() -> retval"},
    {"getRpropDWPlus", (PyCFunction)pyopencv_cv_ml_ml_ANN_MLP_getRpropDWPlus, METH_VARARGS | METH_KEYWORDS, "getRpropDWPlus() -> retval"},
    {"getTermCriteria", (PyCFunction)pyopencv_cv_ml_ml_ANN_MLP_getTermCriteria, METH_VARARGS | METH_KEYWORDS, "getTermCriteria() -> retval"},
    {"getTrainMethod", (PyCFunction)pyopencv_cv_ml_ml_ANN_MLP_getTrainMethod, METH_VARARGS | METH_KEYWORDS, "getTrainMethod() -> retval"},
    {"getWeights", (PyCFunction)pyopencv_cv_ml_ml_ANN_MLP_getWeights, METH_VARARGS | METH_KEYWORDS, "getWeights(layerIdx) -> retval"},
    {"setActivationFunction", (PyCFunction)pyopencv_cv_ml_ml_ANN_MLP_setActivationFunction, METH_VARARGS | METH_KEYWORDS, "setActivationFunction(type[, param1[, param2]]) -> None"},
    {"setBackpropMomentumScale", (PyCFunction)pyopencv_cv_ml_ml_ANN_MLP_setBackpropMomentumScale, METH_VARARGS | METH_KEYWORDS, "setBackpropMomentumScale(val) -> None"},
    {"setBackpropWeightScale", (PyCFunction)pyopencv_cv_ml_ml_ANN_MLP_setBackpropWeightScale, METH_VARARGS | METH_KEYWORDS, "setBackpropWeightScale(val) -> None"},
    {"setLayerSizes", (PyCFunction)pyopencv_cv_ml_ml_ANN_MLP_setLayerSizes, METH_VARARGS | METH_KEYWORDS, "setLayerSizes(_layer_sizes) -> None"},
    {"setRpropDW0", (PyCFunction)pyopencv_cv_ml_ml_ANN_MLP_setRpropDW0, METH_VARARGS | METH_KEYWORDS, "setRpropDW0(val) -> None"},
    {"setRpropDWMax", (PyCFunction)pyopencv_cv_ml_ml_ANN_MLP_setRpropDWMax, METH_VARARGS | METH_KEYWORDS, "setRpropDWMax(val) -> None"},
    {"setRpropDWMin", (PyCFunction)pyopencv_cv_ml_ml_ANN_MLP_setRpropDWMin, METH_VARARGS | METH_KEYWORDS, "setRpropDWMin(val) -> None"},
    {"setRpropDWMinus", (PyCFunction)pyopencv_cv_ml_ml_ANN_MLP_setRpropDWMinus, METH_VARARGS | METH_KEYWORDS, "setRpropDWMinus(val) -> None"},
    {"setRpropDWPlus", (PyCFunction)pyopencv_cv_ml_ml_ANN_MLP_setRpropDWPlus, METH_VARARGS | METH_KEYWORDS, "setRpropDWPlus(val) -> None"},
    {"setTermCriteria", (PyCFunction)pyopencv_cv_ml_ml_ANN_MLP_setTermCriteria, METH_VARARGS | METH_KEYWORDS, "setTermCriteria(val) -> None"},
    {"setTrainMethod", (PyCFunction)pyopencv_cv_ml_ml_ANN_MLP_setTrainMethod, METH_VARARGS | METH_KEYWORDS, "setTrainMethod(method[, param1[, param2]]) -> None"},

    {NULL,          NULL}
};

static void pyopencv_ml_ANN_MLP_specials(void)
{
    pyopencv_ml_ANN_MLP_Type.tp_base = &pyopencv_ml_StatModel_Type;
    pyopencv_ml_ANN_MLP_Type.tp_dealloc = pyopencv_ml_ANN_MLP_dealloc;
    pyopencv_ml_ANN_MLP_Type.tp_repr = pyopencv_ml_ANN_MLP_repr;
    pyopencv_ml_ANN_MLP_Type.tp_getset = pyopencv_ml_ANN_MLP_getseters;
    pyopencv_ml_ANN_MLP_Type.tp_methods = pyopencv_ml_ANN_MLP_methods;
}

static PyObject* pyopencv_ml_LogisticRegression_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<ml_LogisticRegression %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_ml_LogisticRegression_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_ml_ml_LogisticRegression_getIterations(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_LogisticRegression_Type))
        return failmsgp("Incorrect type of self (must be 'ml_LogisticRegression' or its derivative)");
    cv::ml::LogisticRegression* _self_ = dynamic_cast<cv::ml::LogisticRegression*>(((pyopencv_ml_LogisticRegression_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getIterations());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_LogisticRegression_getLearningRate(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_LogisticRegression_Type))
        return failmsgp("Incorrect type of self (must be 'ml_LogisticRegression' or its derivative)");
    cv::ml::LogisticRegression* _self_ = dynamic_cast<cv::ml::LogisticRegression*>(((pyopencv_ml_LogisticRegression_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getLearningRate());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_LogisticRegression_getMiniBatchSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_LogisticRegression_Type))
        return failmsgp("Incorrect type of self (must be 'ml_LogisticRegression' or its derivative)");
    cv::ml::LogisticRegression* _self_ = dynamic_cast<cv::ml::LogisticRegression*>(((pyopencv_ml_LogisticRegression_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getMiniBatchSize());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_LogisticRegression_getRegularization(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_LogisticRegression_Type))
        return failmsgp("Incorrect type of self (must be 'ml_LogisticRegression' or its derivative)");
    cv::ml::LogisticRegression* _self_ = dynamic_cast<cv::ml::LogisticRegression*>(((pyopencv_ml_LogisticRegression_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getRegularization());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_LogisticRegression_getTermCriteria(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_LogisticRegression_Type))
        return failmsgp("Incorrect type of self (must be 'ml_LogisticRegression' or its derivative)");
    cv::ml::LogisticRegression* _self_ = dynamic_cast<cv::ml::LogisticRegression*>(((pyopencv_ml_LogisticRegression_t*)self)->v.get());
    TermCriteria retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTermCriteria());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_LogisticRegression_getTrainMethod(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_LogisticRegression_Type))
        return failmsgp("Incorrect type of self (must be 'ml_LogisticRegression' or its derivative)");
    cv::ml::LogisticRegression* _self_ = dynamic_cast<cv::ml::LogisticRegression*>(((pyopencv_ml_LogisticRegression_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTrainMethod());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_LogisticRegression_get_learnt_thetas(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_LogisticRegression_Type))
        return failmsgp("Incorrect type of self (must be 'ml_LogisticRegression' or its derivative)");
    cv::ml::LogisticRegression* _self_ = dynamic_cast<cv::ml::LogisticRegression*>(((pyopencv_ml_LogisticRegression_t*)self)->v.get());
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->get_learnt_thetas());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_LogisticRegression_predict(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_LogisticRegression_Type))
        return failmsgp("Incorrect type of self (must be 'ml_LogisticRegression' or its derivative)");
    cv::ml::LogisticRegression* _self_ = dynamic_cast<cv::ml::LogisticRegression*>(((pyopencv_ml_LogisticRegression_t*)self)->v.get());
    {
    PyObject* pyobj_samples = NULL;
    Mat samples;
    PyObject* pyobj_results = NULL;
    Mat results;
    int flags=0;
    float retval;

    const char* keywords[] = { "samples", "results", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oi:ml_LogisticRegression.predict", (char**)keywords, &pyobj_samples, &pyobj_results, &flags) &&
        pyopencv_to(pyobj_samples, samples, ArgInfo("samples", 0)) &&
        pyopencv_to(pyobj_results, results, ArgInfo("results", 1)) )
    {
        ERRWRAP2(retval = _self_->predict(samples, results, flags));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(results));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_samples = NULL;
    UMat samples;
    PyObject* pyobj_results = NULL;
    UMat results;
    int flags=0;
    float retval;

    const char* keywords[] = { "samples", "results", "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Oi:ml_LogisticRegression.predict", (char**)keywords, &pyobj_samples, &pyobj_results, &flags) &&
        pyopencv_to(pyobj_samples, samples, ArgInfo("samples", 0)) &&
        pyopencv_to(pyobj_results, results, ArgInfo("results", 1)) )
    {
        ERRWRAP2(retval = _self_->predict(samples, results, flags));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(results));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_LogisticRegression_setIterations(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_LogisticRegression_Type))
        return failmsgp("Incorrect type of self (must be 'ml_LogisticRegression' or its derivative)");
    cv::ml::LogisticRegression* _self_ = dynamic_cast<cv::ml::LogisticRegression*>(((pyopencv_ml_LogisticRegression_t*)self)->v.get());
    int val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ml_LogisticRegression.setIterations", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setIterations(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_LogisticRegression_setLearningRate(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_LogisticRegression_Type))
        return failmsgp("Incorrect type of self (must be 'ml_LogisticRegression' or its derivative)");
    cv::ml::LogisticRegression* _self_ = dynamic_cast<cv::ml::LogisticRegression*>(((pyopencv_ml_LogisticRegression_t*)self)->v.get());
    double val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:ml_LogisticRegression.setLearningRate", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setLearningRate(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_LogisticRegression_setMiniBatchSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_LogisticRegression_Type))
        return failmsgp("Incorrect type of self (must be 'ml_LogisticRegression' or its derivative)");
    cv::ml::LogisticRegression* _self_ = dynamic_cast<cv::ml::LogisticRegression*>(((pyopencv_ml_LogisticRegression_t*)self)->v.get());
    int val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ml_LogisticRegression.setMiniBatchSize", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setMiniBatchSize(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_LogisticRegression_setRegularization(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_LogisticRegression_Type))
        return failmsgp("Incorrect type of self (must be 'ml_LogisticRegression' or its derivative)");
    cv::ml::LogisticRegression* _self_ = dynamic_cast<cv::ml::LogisticRegression*>(((pyopencv_ml_LogisticRegression_t*)self)->v.get());
    int val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ml_LogisticRegression.setRegularization", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setRegularization(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_LogisticRegression_setTermCriteria(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_LogisticRegression_Type))
        return failmsgp("Incorrect type of self (must be 'ml_LogisticRegression' or its derivative)");
    cv::ml::LogisticRegression* _self_ = dynamic_cast<cv::ml::LogisticRegression*>(((pyopencv_ml_LogisticRegression_t*)self)->v.get());
    PyObject* pyobj_val = NULL;
    TermCriteria val;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:ml_LogisticRegression.setTermCriteria", (char**)keywords, &pyobj_val) &&
        pyopencv_to(pyobj_val, val, ArgInfo("val", 0)) )
    {
        ERRWRAP2(_self_->setTermCriteria(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_LogisticRegression_setTrainMethod(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_LogisticRegression_Type))
        return failmsgp("Incorrect type of self (must be 'ml_LogisticRegression' or its derivative)");
    cv::ml::LogisticRegression* _self_ = dynamic_cast<cv::ml::LogisticRegression*>(((pyopencv_ml_LogisticRegression_t*)self)->v.get());
    int val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ml_LogisticRegression.setTrainMethod", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setTrainMethod(val));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_ml_LogisticRegression_methods[] =
{
    {"getIterations", (PyCFunction)pyopencv_cv_ml_ml_LogisticRegression_getIterations, METH_VARARGS | METH_KEYWORDS, "getIterations() -> retval"},
    {"getLearningRate", (PyCFunction)pyopencv_cv_ml_ml_LogisticRegression_getLearningRate, METH_VARARGS | METH_KEYWORDS, "getLearningRate() -> retval"},
    {"getMiniBatchSize", (PyCFunction)pyopencv_cv_ml_ml_LogisticRegression_getMiniBatchSize, METH_VARARGS | METH_KEYWORDS, "getMiniBatchSize() -> retval"},
    {"getRegularization", (PyCFunction)pyopencv_cv_ml_ml_LogisticRegression_getRegularization, METH_VARARGS | METH_KEYWORDS, "getRegularization() -> retval"},
    {"getTermCriteria", (PyCFunction)pyopencv_cv_ml_ml_LogisticRegression_getTermCriteria, METH_VARARGS | METH_KEYWORDS, "getTermCriteria() -> retval"},
    {"getTrainMethod", (PyCFunction)pyopencv_cv_ml_ml_LogisticRegression_getTrainMethod, METH_VARARGS | METH_KEYWORDS, "getTrainMethod() -> retval"},
    {"get_learnt_thetas", (PyCFunction)pyopencv_cv_ml_ml_LogisticRegression_get_learnt_thetas, METH_VARARGS | METH_KEYWORDS, "get_learnt_thetas() -> retval"},
    {"predict", (PyCFunction)pyopencv_cv_ml_ml_LogisticRegression_predict, METH_VARARGS | METH_KEYWORDS, "predict(samples[, results[, flags]]) -> retval, results"},
    {"setIterations", (PyCFunction)pyopencv_cv_ml_ml_LogisticRegression_setIterations, METH_VARARGS | METH_KEYWORDS, "setIterations(val) -> None"},
    {"setLearningRate", (PyCFunction)pyopencv_cv_ml_ml_LogisticRegression_setLearningRate, METH_VARARGS | METH_KEYWORDS, "setLearningRate(val) -> None"},
    {"setMiniBatchSize", (PyCFunction)pyopencv_cv_ml_ml_LogisticRegression_setMiniBatchSize, METH_VARARGS | METH_KEYWORDS, "setMiniBatchSize(val) -> None"},
    {"setRegularization", (PyCFunction)pyopencv_cv_ml_ml_LogisticRegression_setRegularization, METH_VARARGS | METH_KEYWORDS, "setRegularization(val) -> None"},
    {"setTermCriteria", (PyCFunction)pyopencv_cv_ml_ml_LogisticRegression_setTermCriteria, METH_VARARGS | METH_KEYWORDS, "setTermCriteria(val) -> None"},
    {"setTrainMethod", (PyCFunction)pyopencv_cv_ml_ml_LogisticRegression_setTrainMethod, METH_VARARGS | METH_KEYWORDS, "setTrainMethod(val) -> None"},

    {NULL,          NULL}
};

static void pyopencv_ml_LogisticRegression_specials(void)
{
    pyopencv_ml_LogisticRegression_Type.tp_base = &pyopencv_ml_StatModel_Type;
    pyopencv_ml_LogisticRegression_Type.tp_dealloc = pyopencv_ml_LogisticRegression_dealloc;
    pyopencv_ml_LogisticRegression_Type.tp_repr = pyopencv_ml_LogisticRegression_repr;
    pyopencv_ml_LogisticRegression_Type.tp_getset = pyopencv_ml_LogisticRegression_getseters;
    pyopencv_ml_LogisticRegression_Type.tp_methods = pyopencv_ml_LogisticRegression_methods;
}

static PyObject* pyopencv_ml_SVMSGD_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<ml_SVMSGD %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_ml_SVMSGD_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_ml_ml_SVMSGD_getInitialStepSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVMSGD_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVMSGD' or its derivative)");
    cv::ml::SVMSGD* _self_ = dynamic_cast<cv::ml::SVMSGD*>(((pyopencv_ml_SVMSGD_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getInitialStepSize());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVMSGD_getMarginRegularization(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVMSGD_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVMSGD' or its derivative)");
    cv::ml::SVMSGD* _self_ = dynamic_cast<cv::ml::SVMSGD*>(((pyopencv_ml_SVMSGD_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getMarginRegularization());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVMSGD_getMarginType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVMSGD_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVMSGD' or its derivative)");
    cv::ml::SVMSGD* _self_ = dynamic_cast<cv::ml::SVMSGD*>(((pyopencv_ml_SVMSGD_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getMarginType());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVMSGD_getShift(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVMSGD_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVMSGD' or its derivative)");
    cv::ml::SVMSGD* _self_ = dynamic_cast<cv::ml::SVMSGD*>(((pyopencv_ml_SVMSGD_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getShift());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVMSGD_getStepDecreasingPower(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVMSGD_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVMSGD' or its derivative)");
    cv::ml::SVMSGD* _self_ = dynamic_cast<cv::ml::SVMSGD*>(((pyopencv_ml_SVMSGD_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getStepDecreasingPower());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVMSGD_getSvmsgdType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVMSGD_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVMSGD' or its derivative)");
    cv::ml::SVMSGD* _self_ = dynamic_cast<cv::ml::SVMSGD*>(((pyopencv_ml_SVMSGD_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getSvmsgdType());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVMSGD_getTermCriteria(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVMSGD_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVMSGD' or its derivative)");
    cv::ml::SVMSGD* _self_ = dynamic_cast<cv::ml::SVMSGD*>(((pyopencv_ml_SVMSGD_t*)self)->v.get());
    TermCriteria retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTermCriteria());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVMSGD_getWeights(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVMSGD_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVMSGD' or its derivative)");
    cv::ml::SVMSGD* _self_ = dynamic_cast<cv::ml::SVMSGD*>(((pyopencv_ml_SVMSGD_t*)self)->v.get());
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getWeights());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVMSGD_setInitialStepSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVMSGD_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVMSGD' or its derivative)");
    cv::ml::SVMSGD* _self_ = dynamic_cast<cv::ml::SVMSGD*>(((pyopencv_ml_SVMSGD_t*)self)->v.get());
    float InitialStepSize=0.f;

    const char* keywords[] = { "InitialStepSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:ml_SVMSGD.setInitialStepSize", (char**)keywords, &InitialStepSize) )
    {
        ERRWRAP2(_self_->setInitialStepSize(InitialStepSize));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVMSGD_setMarginRegularization(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVMSGD_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVMSGD' or its derivative)");
    cv::ml::SVMSGD* _self_ = dynamic_cast<cv::ml::SVMSGD*>(((pyopencv_ml_SVMSGD_t*)self)->v.get());
    float marginRegularization=0.f;

    const char* keywords[] = { "marginRegularization", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:ml_SVMSGD.setMarginRegularization", (char**)keywords, &marginRegularization) )
    {
        ERRWRAP2(_self_->setMarginRegularization(marginRegularization));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVMSGD_setMarginType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVMSGD_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVMSGD' or its derivative)");
    cv::ml::SVMSGD* _self_ = dynamic_cast<cv::ml::SVMSGD*>(((pyopencv_ml_SVMSGD_t*)self)->v.get());
    int marginType=0;

    const char* keywords[] = { "marginType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ml_SVMSGD.setMarginType", (char**)keywords, &marginType) )
    {
        ERRWRAP2(_self_->setMarginType(marginType));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVMSGD_setOptimalParameters(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVMSGD_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVMSGD' or its derivative)");
    cv::ml::SVMSGD* _self_ = dynamic_cast<cv::ml::SVMSGD*>(((pyopencv_ml_SVMSGD_t*)self)->v.get());
    int svmsgdType=SVMSGD::ASGD;
    int marginType=SVMSGD::SOFT_MARGIN;

    const char* keywords[] = { "svmsgdType", "marginType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|ii:ml_SVMSGD.setOptimalParameters", (char**)keywords, &svmsgdType, &marginType) )
    {
        ERRWRAP2(_self_->setOptimalParameters(svmsgdType, marginType));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVMSGD_setStepDecreasingPower(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVMSGD_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVMSGD' or its derivative)");
    cv::ml::SVMSGD* _self_ = dynamic_cast<cv::ml::SVMSGD*>(((pyopencv_ml_SVMSGD_t*)self)->v.get());
    float stepDecreasingPower=0.f;

    const char* keywords[] = { "stepDecreasingPower", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:ml_SVMSGD.setStepDecreasingPower", (char**)keywords, &stepDecreasingPower) )
    {
        ERRWRAP2(_self_->setStepDecreasingPower(stepDecreasingPower));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVMSGD_setSvmsgdType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVMSGD_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVMSGD' or its derivative)");
    cv::ml::SVMSGD* _self_ = dynamic_cast<cv::ml::SVMSGD*>(((pyopencv_ml_SVMSGD_t*)self)->v.get());
    int svmsgdType=0;

    const char* keywords[] = { "svmsgdType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ml_SVMSGD.setSvmsgdType", (char**)keywords, &svmsgdType) )
    {
        ERRWRAP2(_self_->setSvmsgdType(svmsgdType));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ml_ml_SVMSGD_setTermCriteria(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv::ml;

    if(!PyObject_TypeCheck(self, &pyopencv_ml_SVMSGD_Type))
        return failmsgp("Incorrect type of self (must be 'ml_SVMSGD' or its derivative)");
    cv::ml::SVMSGD* _self_ = dynamic_cast<cv::ml::SVMSGD*>(((pyopencv_ml_SVMSGD_t*)self)->v.get());
    PyObject* pyobj_val = NULL;
    TermCriteria val;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:ml_SVMSGD.setTermCriteria", (char**)keywords, &pyobj_val) &&
        pyopencv_to(pyobj_val, val, ArgInfo("val", 0)) )
    {
        ERRWRAP2(_self_->setTermCriteria(val));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_ml_SVMSGD_methods[] =
{
    {"getInitialStepSize", (PyCFunction)pyopencv_cv_ml_ml_SVMSGD_getInitialStepSize, METH_VARARGS | METH_KEYWORDS, "getInitialStepSize() -> retval"},
    {"getMarginRegularization", (PyCFunction)pyopencv_cv_ml_ml_SVMSGD_getMarginRegularization, METH_VARARGS | METH_KEYWORDS, "getMarginRegularization() -> retval"},
    {"getMarginType", (PyCFunction)pyopencv_cv_ml_ml_SVMSGD_getMarginType, METH_VARARGS | METH_KEYWORDS, "getMarginType() -> retval"},
    {"getShift", (PyCFunction)pyopencv_cv_ml_ml_SVMSGD_getShift, METH_VARARGS | METH_KEYWORDS, "getShift() -> retval"},
    {"getStepDecreasingPower", (PyCFunction)pyopencv_cv_ml_ml_SVMSGD_getStepDecreasingPower, METH_VARARGS | METH_KEYWORDS, "getStepDecreasingPower() -> retval"},
    {"getSvmsgdType", (PyCFunction)pyopencv_cv_ml_ml_SVMSGD_getSvmsgdType, METH_VARARGS | METH_KEYWORDS, "getSvmsgdType() -> retval"},
    {"getTermCriteria", (PyCFunction)pyopencv_cv_ml_ml_SVMSGD_getTermCriteria, METH_VARARGS | METH_KEYWORDS, "getTermCriteria() -> retval"},
    {"getWeights", (PyCFunction)pyopencv_cv_ml_ml_SVMSGD_getWeights, METH_VARARGS | METH_KEYWORDS, "getWeights() -> retval"},
    {"setInitialStepSize", (PyCFunction)pyopencv_cv_ml_ml_SVMSGD_setInitialStepSize, METH_VARARGS | METH_KEYWORDS, "setInitialStepSize(InitialStepSize) -> None"},
    {"setMarginRegularization", (PyCFunction)pyopencv_cv_ml_ml_SVMSGD_setMarginRegularization, METH_VARARGS | METH_KEYWORDS, "setMarginRegularization(marginRegularization) -> None"},
    {"setMarginType", (PyCFunction)pyopencv_cv_ml_ml_SVMSGD_setMarginType, METH_VARARGS | METH_KEYWORDS, "setMarginType(marginType) -> None"},
    {"setOptimalParameters", (PyCFunction)pyopencv_cv_ml_ml_SVMSGD_setOptimalParameters, METH_VARARGS | METH_KEYWORDS, "setOptimalParameters([, svmsgdType[, marginType]]) -> None"},
    {"setStepDecreasingPower", (PyCFunction)pyopencv_cv_ml_ml_SVMSGD_setStepDecreasingPower, METH_VARARGS | METH_KEYWORDS, "setStepDecreasingPower(stepDecreasingPower) -> None"},
    {"setSvmsgdType", (PyCFunction)pyopencv_cv_ml_ml_SVMSGD_setSvmsgdType, METH_VARARGS | METH_KEYWORDS, "setSvmsgdType(svmsgdType) -> None"},
    {"setTermCriteria", (PyCFunction)pyopencv_cv_ml_ml_SVMSGD_setTermCriteria, METH_VARARGS | METH_KEYWORDS, "setTermCriteria(val) -> None"},

    {NULL,          NULL}
};

static void pyopencv_ml_SVMSGD_specials(void)
{
    pyopencv_ml_SVMSGD_Type.tp_base = &pyopencv_ml_StatModel_Type;
    pyopencv_ml_SVMSGD_Type.tp_dealloc = pyopencv_ml_SVMSGD_dealloc;
    pyopencv_ml_SVMSGD_Type.tp_repr = pyopencv_ml_SVMSGD_repr;
    pyopencv_ml_SVMSGD_Type.tp_getset = pyopencv_ml_SVMSGD_getseters;
    pyopencv_ml_SVMSGD_Type.tp_methods = pyopencv_ml_SVMSGD_methods;
}

static PyObject* pyopencv_Tonemap_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<Tonemap %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_Tonemap_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_Tonemap_getGamma(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Tonemap_Type))
        return failmsgp("Incorrect type of self (must be 'Tonemap' or its derivative)");
    cv::Tonemap* _self_ = dynamic_cast<cv::Tonemap*>(((pyopencv_Tonemap_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getGamma());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_Tonemap_process(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Tonemap_Type))
        return failmsgp("Incorrect type of self (must be 'Tonemap' or its derivative)");
    cv::Tonemap* _self_ = dynamic_cast<cv::Tonemap*>(((pyopencv_Tonemap_t*)self)->v.get());
    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:Tonemap.process", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(_self_->process(src, dst));
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
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:Tonemap.process", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(_self_->process(src, dst));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_Tonemap_setGamma(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Tonemap_Type))
        return failmsgp("Incorrect type of self (must be 'Tonemap' or its derivative)");
    cv::Tonemap* _self_ = dynamic_cast<cv::Tonemap*>(((pyopencv_Tonemap_t*)self)->v.get());
    float gamma=0.f;

    const char* keywords[] = { "gamma", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:Tonemap.setGamma", (char**)keywords, &gamma) )
    {
        ERRWRAP2(_self_->setGamma(gamma));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_Tonemap_methods[] =
{
    {"getGamma", (PyCFunction)pyopencv_cv_Tonemap_getGamma, METH_VARARGS | METH_KEYWORDS, "getGamma() -> retval"},
    {"process", (PyCFunction)pyopencv_cv_Tonemap_process, METH_VARARGS | METH_KEYWORDS, "process(src[, dst]) -> dst"},
    {"setGamma", (PyCFunction)pyopencv_cv_Tonemap_setGamma, METH_VARARGS | METH_KEYWORDS, "setGamma(gamma) -> None"},

    {NULL,          NULL}
};

static void pyopencv_Tonemap_specials(void)
{
    pyopencv_Tonemap_Type.tp_base = &pyopencv_Algorithm_Type;
    pyopencv_Tonemap_Type.tp_dealloc = pyopencv_Tonemap_dealloc;
    pyopencv_Tonemap_Type.tp_repr = pyopencv_Tonemap_repr;
    pyopencv_Tonemap_Type.tp_getset = pyopencv_Tonemap_getseters;
    pyopencv_Tonemap_Type.tp_methods = pyopencv_Tonemap_methods;
}

static PyObject* pyopencv_TonemapDrago_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<TonemapDrago %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_TonemapDrago_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_TonemapDrago_getBias(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TonemapDrago_Type))
        return failmsgp("Incorrect type of self (must be 'TonemapDrago' or its derivative)");
    cv::TonemapDrago* _self_ = dynamic_cast<cv::TonemapDrago*>(((pyopencv_TonemapDrago_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getBias());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_TonemapDrago_getSaturation(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TonemapDrago_Type))
        return failmsgp("Incorrect type of self (must be 'TonemapDrago' or its derivative)");
    cv::TonemapDrago* _self_ = dynamic_cast<cv::TonemapDrago*>(((pyopencv_TonemapDrago_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getSaturation());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_TonemapDrago_setBias(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TonemapDrago_Type))
        return failmsgp("Incorrect type of self (must be 'TonemapDrago' or its derivative)");
    cv::TonemapDrago* _self_ = dynamic_cast<cv::TonemapDrago*>(((pyopencv_TonemapDrago_t*)self)->v.get());
    float bias=0.f;

    const char* keywords[] = { "bias", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:TonemapDrago.setBias", (char**)keywords, &bias) )
    {
        ERRWRAP2(_self_->setBias(bias));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_TonemapDrago_setSaturation(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TonemapDrago_Type))
        return failmsgp("Incorrect type of self (must be 'TonemapDrago' or its derivative)");
    cv::TonemapDrago* _self_ = dynamic_cast<cv::TonemapDrago*>(((pyopencv_TonemapDrago_t*)self)->v.get());
    float saturation=0.f;

    const char* keywords[] = { "saturation", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:TonemapDrago.setSaturation", (char**)keywords, &saturation) )
    {
        ERRWRAP2(_self_->setSaturation(saturation));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_TonemapDrago_methods[] =
{
    {"getBias", (PyCFunction)pyopencv_cv_TonemapDrago_getBias, METH_VARARGS | METH_KEYWORDS, "getBias() -> retval"},
    {"getSaturation", (PyCFunction)pyopencv_cv_TonemapDrago_getSaturation, METH_VARARGS | METH_KEYWORDS, "getSaturation() -> retval"},
    {"setBias", (PyCFunction)pyopencv_cv_TonemapDrago_setBias, METH_VARARGS | METH_KEYWORDS, "setBias(bias) -> None"},
    {"setSaturation", (PyCFunction)pyopencv_cv_TonemapDrago_setSaturation, METH_VARARGS | METH_KEYWORDS, "setSaturation(saturation) -> None"},

    {NULL,          NULL}
};

static void pyopencv_TonemapDrago_specials(void)
{
    pyopencv_TonemapDrago_Type.tp_base = &pyopencv_Tonemap_Type;
    pyopencv_TonemapDrago_Type.tp_dealloc = pyopencv_TonemapDrago_dealloc;
    pyopencv_TonemapDrago_Type.tp_repr = pyopencv_TonemapDrago_repr;
    pyopencv_TonemapDrago_Type.tp_getset = pyopencv_TonemapDrago_getseters;
    pyopencv_TonemapDrago_Type.tp_methods = pyopencv_TonemapDrago_methods;
}

static PyObject* pyopencv_TonemapDurand_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<TonemapDurand %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_TonemapDurand_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_TonemapDurand_getContrast(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TonemapDurand_Type))
        return failmsgp("Incorrect type of self (must be 'TonemapDurand' or its derivative)");
    cv::TonemapDurand* _self_ = dynamic_cast<cv::TonemapDurand*>(((pyopencv_TonemapDurand_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getContrast());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_TonemapDurand_getSaturation(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TonemapDurand_Type))
        return failmsgp("Incorrect type of self (must be 'TonemapDurand' or its derivative)");
    cv::TonemapDurand* _self_ = dynamic_cast<cv::TonemapDurand*>(((pyopencv_TonemapDurand_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getSaturation());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_TonemapDurand_getSigmaColor(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TonemapDurand_Type))
        return failmsgp("Incorrect type of self (must be 'TonemapDurand' or its derivative)");
    cv::TonemapDurand* _self_ = dynamic_cast<cv::TonemapDurand*>(((pyopencv_TonemapDurand_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getSigmaColor());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_TonemapDurand_getSigmaSpace(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TonemapDurand_Type))
        return failmsgp("Incorrect type of self (must be 'TonemapDurand' or its derivative)");
    cv::TonemapDurand* _self_ = dynamic_cast<cv::TonemapDurand*>(((pyopencv_TonemapDurand_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getSigmaSpace());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_TonemapDurand_setContrast(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TonemapDurand_Type))
        return failmsgp("Incorrect type of self (must be 'TonemapDurand' or its derivative)");
    cv::TonemapDurand* _self_ = dynamic_cast<cv::TonemapDurand*>(((pyopencv_TonemapDurand_t*)self)->v.get());
    float contrast=0.f;

    const char* keywords[] = { "contrast", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:TonemapDurand.setContrast", (char**)keywords, &contrast) )
    {
        ERRWRAP2(_self_->setContrast(contrast));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_TonemapDurand_setSaturation(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TonemapDurand_Type))
        return failmsgp("Incorrect type of self (must be 'TonemapDurand' or its derivative)");
    cv::TonemapDurand* _self_ = dynamic_cast<cv::TonemapDurand*>(((pyopencv_TonemapDurand_t*)self)->v.get());
    float saturation=0.f;

    const char* keywords[] = { "saturation", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:TonemapDurand.setSaturation", (char**)keywords, &saturation) )
    {
        ERRWRAP2(_self_->setSaturation(saturation));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_TonemapDurand_setSigmaColor(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TonemapDurand_Type))
        return failmsgp("Incorrect type of self (must be 'TonemapDurand' or its derivative)");
    cv::TonemapDurand* _self_ = dynamic_cast<cv::TonemapDurand*>(((pyopencv_TonemapDurand_t*)self)->v.get());
    float sigma_color=0.f;

    const char* keywords[] = { "sigma_color", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:TonemapDurand.setSigmaColor", (char**)keywords, &sigma_color) )
    {
        ERRWRAP2(_self_->setSigmaColor(sigma_color));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_TonemapDurand_setSigmaSpace(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TonemapDurand_Type))
        return failmsgp("Incorrect type of self (must be 'TonemapDurand' or its derivative)");
    cv::TonemapDurand* _self_ = dynamic_cast<cv::TonemapDurand*>(((pyopencv_TonemapDurand_t*)self)->v.get());
    float sigma_space=0.f;

    const char* keywords[] = { "sigma_space", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:TonemapDurand.setSigmaSpace", (char**)keywords, &sigma_space) )
    {
        ERRWRAP2(_self_->setSigmaSpace(sigma_space));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_TonemapDurand_methods[] =
{
    {"getContrast", (PyCFunction)pyopencv_cv_TonemapDurand_getContrast, METH_VARARGS | METH_KEYWORDS, "getContrast() -> retval"},
    {"getSaturation", (PyCFunction)pyopencv_cv_TonemapDurand_getSaturation, METH_VARARGS | METH_KEYWORDS, "getSaturation() -> retval"},
    {"getSigmaColor", (PyCFunction)pyopencv_cv_TonemapDurand_getSigmaColor, METH_VARARGS | METH_KEYWORDS, "getSigmaColor() -> retval"},
    {"getSigmaSpace", (PyCFunction)pyopencv_cv_TonemapDurand_getSigmaSpace, METH_VARARGS | METH_KEYWORDS, "getSigmaSpace() -> retval"},
    {"setContrast", (PyCFunction)pyopencv_cv_TonemapDurand_setContrast, METH_VARARGS | METH_KEYWORDS, "setContrast(contrast) -> None"},
    {"setSaturation", (PyCFunction)pyopencv_cv_TonemapDurand_setSaturation, METH_VARARGS | METH_KEYWORDS, "setSaturation(saturation) -> None"},
    {"setSigmaColor", (PyCFunction)pyopencv_cv_TonemapDurand_setSigmaColor, METH_VARARGS | METH_KEYWORDS, "setSigmaColor(sigma_color) -> None"},
    {"setSigmaSpace", (PyCFunction)pyopencv_cv_TonemapDurand_setSigmaSpace, METH_VARARGS | METH_KEYWORDS, "setSigmaSpace(sigma_space) -> None"},

    {NULL,          NULL}
};

static void pyopencv_TonemapDurand_specials(void)
{
    pyopencv_TonemapDurand_Type.tp_base = &pyopencv_Tonemap_Type;
    pyopencv_TonemapDurand_Type.tp_dealloc = pyopencv_TonemapDurand_dealloc;
    pyopencv_TonemapDurand_Type.tp_repr = pyopencv_TonemapDurand_repr;
    pyopencv_TonemapDurand_Type.tp_getset = pyopencv_TonemapDurand_getseters;
    pyopencv_TonemapDurand_Type.tp_methods = pyopencv_TonemapDurand_methods;
}

static PyObject* pyopencv_TonemapReinhard_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<TonemapReinhard %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_TonemapReinhard_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_TonemapReinhard_getColorAdaptation(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TonemapReinhard_Type))
        return failmsgp("Incorrect type of self (must be 'TonemapReinhard' or its derivative)");
    cv::TonemapReinhard* _self_ = dynamic_cast<cv::TonemapReinhard*>(((pyopencv_TonemapReinhard_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getColorAdaptation());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_TonemapReinhard_getIntensity(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TonemapReinhard_Type))
        return failmsgp("Incorrect type of self (must be 'TonemapReinhard' or its derivative)");
    cv::TonemapReinhard* _self_ = dynamic_cast<cv::TonemapReinhard*>(((pyopencv_TonemapReinhard_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getIntensity());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_TonemapReinhard_getLightAdaptation(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TonemapReinhard_Type))
        return failmsgp("Incorrect type of self (must be 'TonemapReinhard' or its derivative)");
    cv::TonemapReinhard* _self_ = dynamic_cast<cv::TonemapReinhard*>(((pyopencv_TonemapReinhard_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getLightAdaptation());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_TonemapReinhard_setColorAdaptation(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TonemapReinhard_Type))
        return failmsgp("Incorrect type of self (must be 'TonemapReinhard' or its derivative)");
    cv::TonemapReinhard* _self_ = dynamic_cast<cv::TonemapReinhard*>(((pyopencv_TonemapReinhard_t*)self)->v.get());
    float color_adapt=0.f;

    const char* keywords[] = { "color_adapt", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:TonemapReinhard.setColorAdaptation", (char**)keywords, &color_adapt) )
    {
        ERRWRAP2(_self_->setColorAdaptation(color_adapt));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_TonemapReinhard_setIntensity(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TonemapReinhard_Type))
        return failmsgp("Incorrect type of self (must be 'TonemapReinhard' or its derivative)");
    cv::TonemapReinhard* _self_ = dynamic_cast<cv::TonemapReinhard*>(((pyopencv_TonemapReinhard_t*)self)->v.get());
    float intensity=0.f;

    const char* keywords[] = { "intensity", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:TonemapReinhard.setIntensity", (char**)keywords, &intensity) )
    {
        ERRWRAP2(_self_->setIntensity(intensity));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_TonemapReinhard_setLightAdaptation(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TonemapReinhard_Type))
        return failmsgp("Incorrect type of self (must be 'TonemapReinhard' or its derivative)");
    cv::TonemapReinhard* _self_ = dynamic_cast<cv::TonemapReinhard*>(((pyopencv_TonemapReinhard_t*)self)->v.get());
    float light_adapt=0.f;

    const char* keywords[] = { "light_adapt", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:TonemapReinhard.setLightAdaptation", (char**)keywords, &light_adapt) )
    {
        ERRWRAP2(_self_->setLightAdaptation(light_adapt));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_TonemapReinhard_methods[] =
{
    {"getColorAdaptation", (PyCFunction)pyopencv_cv_TonemapReinhard_getColorAdaptation, METH_VARARGS | METH_KEYWORDS, "getColorAdaptation() -> retval"},
    {"getIntensity", (PyCFunction)pyopencv_cv_TonemapReinhard_getIntensity, METH_VARARGS | METH_KEYWORDS, "getIntensity() -> retval"},
    {"getLightAdaptation", (PyCFunction)pyopencv_cv_TonemapReinhard_getLightAdaptation, METH_VARARGS | METH_KEYWORDS, "getLightAdaptation() -> retval"},
    {"setColorAdaptation", (PyCFunction)pyopencv_cv_TonemapReinhard_setColorAdaptation, METH_VARARGS | METH_KEYWORDS, "setColorAdaptation(color_adapt) -> None"},
    {"setIntensity", (PyCFunction)pyopencv_cv_TonemapReinhard_setIntensity, METH_VARARGS | METH_KEYWORDS, "setIntensity(intensity) -> None"},
    {"setLightAdaptation", (PyCFunction)pyopencv_cv_TonemapReinhard_setLightAdaptation, METH_VARARGS | METH_KEYWORDS, "setLightAdaptation(light_adapt) -> None"},

    {NULL,          NULL}
};

static void pyopencv_TonemapReinhard_specials(void)
{
    pyopencv_TonemapReinhard_Type.tp_base = &pyopencv_Tonemap_Type;
    pyopencv_TonemapReinhard_Type.tp_dealloc = pyopencv_TonemapReinhard_dealloc;
    pyopencv_TonemapReinhard_Type.tp_repr = pyopencv_TonemapReinhard_repr;
    pyopencv_TonemapReinhard_Type.tp_getset = pyopencv_TonemapReinhard_getseters;
    pyopencv_TonemapReinhard_Type.tp_methods = pyopencv_TonemapReinhard_methods;
}

static PyObject* pyopencv_TonemapMantiuk_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<TonemapMantiuk %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_TonemapMantiuk_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_TonemapMantiuk_getSaturation(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TonemapMantiuk_Type))
        return failmsgp("Incorrect type of self (must be 'TonemapMantiuk' or its derivative)");
    cv::TonemapMantiuk* _self_ = dynamic_cast<cv::TonemapMantiuk*>(((pyopencv_TonemapMantiuk_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getSaturation());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_TonemapMantiuk_getScale(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TonemapMantiuk_Type))
        return failmsgp("Incorrect type of self (must be 'TonemapMantiuk' or its derivative)");
    cv::TonemapMantiuk* _self_ = dynamic_cast<cv::TonemapMantiuk*>(((pyopencv_TonemapMantiuk_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getScale());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_TonemapMantiuk_setSaturation(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TonemapMantiuk_Type))
        return failmsgp("Incorrect type of self (must be 'TonemapMantiuk' or its derivative)");
    cv::TonemapMantiuk* _self_ = dynamic_cast<cv::TonemapMantiuk*>(((pyopencv_TonemapMantiuk_t*)self)->v.get());
    float saturation=0.f;

    const char* keywords[] = { "saturation", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:TonemapMantiuk.setSaturation", (char**)keywords, &saturation) )
    {
        ERRWRAP2(_self_->setSaturation(saturation));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_TonemapMantiuk_setScale(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_TonemapMantiuk_Type))
        return failmsgp("Incorrect type of self (must be 'TonemapMantiuk' or its derivative)");
    cv::TonemapMantiuk* _self_ = dynamic_cast<cv::TonemapMantiuk*>(((pyopencv_TonemapMantiuk_t*)self)->v.get());
    float scale=0.f;

    const char* keywords[] = { "scale", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:TonemapMantiuk.setScale", (char**)keywords, &scale) )
    {
        ERRWRAP2(_self_->setScale(scale));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_TonemapMantiuk_methods[] =
{
    {"getSaturation", (PyCFunction)pyopencv_cv_TonemapMantiuk_getSaturation, METH_VARARGS | METH_KEYWORDS, "getSaturation() -> retval"},
    {"getScale", (PyCFunction)pyopencv_cv_TonemapMantiuk_getScale, METH_VARARGS | METH_KEYWORDS, "getScale() -> retval"},
    {"setSaturation", (PyCFunction)pyopencv_cv_TonemapMantiuk_setSaturation, METH_VARARGS | METH_KEYWORDS, "setSaturation(saturation) -> None"},
    {"setScale", (PyCFunction)pyopencv_cv_TonemapMantiuk_setScale, METH_VARARGS | METH_KEYWORDS, "setScale(scale) -> None"},

    {NULL,          NULL}
};

static void pyopencv_TonemapMantiuk_specials(void)
{
    pyopencv_TonemapMantiuk_Type.tp_base = &pyopencv_Tonemap_Type;
    pyopencv_TonemapMantiuk_Type.tp_dealloc = pyopencv_TonemapMantiuk_dealloc;
    pyopencv_TonemapMantiuk_Type.tp_repr = pyopencv_TonemapMantiuk_repr;
    pyopencv_TonemapMantiuk_Type.tp_getset = pyopencv_TonemapMantiuk_getseters;
    pyopencv_TonemapMantiuk_Type.tp_methods = pyopencv_TonemapMantiuk_methods;
}

static PyObject* pyopencv_AlignExposures_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<AlignExposures %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_AlignExposures_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_AlignExposures_process(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AlignExposures_Type))
        return failmsgp("Incorrect type of self (must be 'AlignExposures' or its derivative)");
    cv::AlignExposures* _self_ = dynamic_cast<cv::AlignExposures*>(((pyopencv_AlignExposures_t*)self)->v.get());
    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    vector_Mat dst;
    PyObject* pyobj_times = NULL;
    Mat times;
    PyObject* pyobj_response = NULL;
    Mat response;

    const char* keywords[] = { "src", "dst", "times", "response", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO:AlignExposures.process", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_times, &pyobj_response) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 0)) &&
        pyopencv_to(pyobj_times, times, ArgInfo("times", 0)) &&
        pyopencv_to(pyobj_response, response, ArgInfo("response", 0)) )
    {
        ERRWRAP2(_self_->process(src, dst, times, response));
        Py_RETURN_NONE;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    vector_Mat dst;
    PyObject* pyobj_times = NULL;
    UMat times;
    PyObject* pyobj_response = NULL;
    UMat response;

    const char* keywords[] = { "src", "dst", "times", "response", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO:AlignExposures.process", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_times, &pyobj_response) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 0)) &&
        pyopencv_to(pyobj_times, times, ArgInfo("times", 0)) &&
        pyopencv_to(pyobj_response, response, ArgInfo("response", 0)) )
    {
        ERRWRAP2(_self_->process(src, dst, times, response));
        Py_RETURN_NONE;
    }
    }

    return NULL;
}



static PyMethodDef pyopencv_AlignExposures_methods[] =
{
    {"process", (PyCFunction)pyopencv_cv_AlignExposures_process, METH_VARARGS | METH_KEYWORDS, "process(src, dst, times, response) -> None"},

    {NULL,          NULL}
};

static void pyopencv_AlignExposures_specials(void)
{
    pyopencv_AlignExposures_Type.tp_base = &pyopencv_Algorithm_Type;
    pyopencv_AlignExposures_Type.tp_dealloc = pyopencv_AlignExposures_dealloc;
    pyopencv_AlignExposures_Type.tp_repr = pyopencv_AlignExposures_repr;
    pyopencv_AlignExposures_Type.tp_getset = pyopencv_AlignExposures_getseters;
    pyopencv_AlignExposures_Type.tp_methods = pyopencv_AlignExposures_methods;
}

static PyObject* pyopencv_AlignMTB_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<AlignMTB %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_AlignMTB_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_AlignMTB_calculateShift(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AlignMTB_Type))
        return failmsgp("Incorrect type of self (must be 'AlignMTB' or its derivative)");
    cv::AlignMTB* _self_ = dynamic_cast<cv::AlignMTB*>(((pyopencv_AlignMTB_t*)self)->v.get());
    {
    PyObject* pyobj_img0 = NULL;
    Mat img0;
    PyObject* pyobj_img1 = NULL;
    Mat img1;
    Point retval;

    const char* keywords[] = { "img0", "img1", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:AlignMTB.calculateShift", (char**)keywords, &pyobj_img0, &pyobj_img1) &&
        pyopencv_to(pyobj_img0, img0, ArgInfo("img0", 0)) &&
        pyopencv_to(pyobj_img1, img1, ArgInfo("img1", 0)) )
    {
        ERRWRAP2(retval = _self_->calculateShift(img0, img1));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_img0 = NULL;
    UMat img0;
    PyObject* pyobj_img1 = NULL;
    UMat img1;
    Point retval;

    const char* keywords[] = { "img0", "img1", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:AlignMTB.calculateShift", (char**)keywords, &pyobj_img0, &pyobj_img1) &&
        pyopencv_to(pyobj_img0, img0, ArgInfo("img0", 0)) &&
        pyopencv_to(pyobj_img1, img1, ArgInfo("img1", 0)) )
    {
        ERRWRAP2(retval = _self_->calculateShift(img0, img1));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_AlignMTB_computeBitmaps(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AlignMTB_Type))
        return failmsgp("Incorrect type of self (must be 'AlignMTB' or its derivative)");
    cv::AlignMTB* _self_ = dynamic_cast<cv::AlignMTB*>(((pyopencv_AlignMTB_t*)self)->v.get());
    {
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_tb = NULL;
    Mat tb;
    PyObject* pyobj_eb = NULL;
    Mat eb;

    const char* keywords[] = { "img", "tb", "eb", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OO:AlignMTB.computeBitmaps", (char**)keywords, &pyobj_img, &pyobj_tb, &pyobj_eb) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_tb, tb, ArgInfo("tb", 1)) &&
        pyopencv_to(pyobj_eb, eb, ArgInfo("eb", 1)) )
    {
        ERRWRAP2(_self_->computeBitmaps(img, tb, eb));
        return Py_BuildValue("(NN)", pyopencv_from(tb), pyopencv_from(eb));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_img = NULL;
    UMat img;
    PyObject* pyobj_tb = NULL;
    UMat tb;
    PyObject* pyobj_eb = NULL;
    UMat eb;

    const char* keywords[] = { "img", "tb", "eb", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OO:AlignMTB.computeBitmaps", (char**)keywords, &pyobj_img, &pyobj_tb, &pyobj_eb) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_tb, tb, ArgInfo("tb", 1)) &&
        pyopencv_to(pyobj_eb, eb, ArgInfo("eb", 1)) )
    {
        ERRWRAP2(_self_->computeBitmaps(img, tb, eb));
        return Py_BuildValue("(NN)", pyopencv_from(tb), pyopencv_from(eb));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_AlignMTB_getCut(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AlignMTB_Type))
        return failmsgp("Incorrect type of self (must be 'AlignMTB' or its derivative)");
    cv::AlignMTB* _self_ = dynamic_cast<cv::AlignMTB*>(((pyopencv_AlignMTB_t*)self)->v.get());
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getCut());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_AlignMTB_getExcludeRange(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AlignMTB_Type))
        return failmsgp("Incorrect type of self (must be 'AlignMTB' or its derivative)");
    cv::AlignMTB* _self_ = dynamic_cast<cv::AlignMTB*>(((pyopencv_AlignMTB_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getExcludeRange());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_AlignMTB_getMaxBits(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AlignMTB_Type))
        return failmsgp("Incorrect type of self (must be 'AlignMTB' or its derivative)");
    cv::AlignMTB* _self_ = dynamic_cast<cv::AlignMTB*>(((pyopencv_AlignMTB_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getMaxBits());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_AlignMTB_process(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AlignMTB_Type))
        return failmsgp("Incorrect type of self (must be 'AlignMTB' or its derivative)");
    cv::AlignMTB* _self_ = dynamic_cast<cv::AlignMTB*>(((pyopencv_AlignMTB_t*)self)->v.get());
    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    vector_Mat dst;
    PyObject* pyobj_times = NULL;
    Mat times;
    PyObject* pyobj_response = NULL;
    Mat response;

    const char* keywords[] = { "src", "dst", "times", "response", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO:AlignMTB.process", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_times, &pyobj_response) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 0)) &&
        pyopencv_to(pyobj_times, times, ArgInfo("times", 0)) &&
        pyopencv_to(pyobj_response, response, ArgInfo("response", 0)) )
    {
        ERRWRAP2(_self_->process(src, dst, times, response));
        Py_RETURN_NONE;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    vector_Mat dst;
    PyObject* pyobj_times = NULL;
    UMat times;
    PyObject* pyobj_response = NULL;
    UMat response;

    const char* keywords[] = { "src", "dst", "times", "response", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO:AlignMTB.process", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_times, &pyobj_response) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 0)) &&
        pyopencv_to(pyobj_times, times, ArgInfo("times", 0)) &&
        pyopencv_to(pyobj_response, response, ArgInfo("response", 0)) )
    {
        ERRWRAP2(_self_->process(src, dst, times, response));
        Py_RETURN_NONE;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    vector_Mat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:AlignMTB.process", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 0)) )
    {
        ERRWRAP2(_self_->process(src, dst));
        Py_RETURN_NONE;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    vector_Mat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:AlignMTB.process", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 0)) )
    {
        ERRWRAP2(_self_->process(src, dst));
        Py_RETURN_NONE;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_AlignMTB_setCut(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AlignMTB_Type))
        return failmsgp("Incorrect type of self (must be 'AlignMTB' or its derivative)");
    cv::AlignMTB* _self_ = dynamic_cast<cv::AlignMTB*>(((pyopencv_AlignMTB_t*)self)->v.get());
    bool value=0;

    const char* keywords[] = { "value", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:AlignMTB.setCut", (char**)keywords, &value) )
    {
        ERRWRAP2(_self_->setCut(value));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_AlignMTB_setExcludeRange(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AlignMTB_Type))
        return failmsgp("Incorrect type of self (must be 'AlignMTB' or its derivative)");
    cv::AlignMTB* _self_ = dynamic_cast<cv::AlignMTB*>(((pyopencv_AlignMTB_t*)self)->v.get());
    int exclude_range=0;

    const char* keywords[] = { "exclude_range", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:AlignMTB.setExcludeRange", (char**)keywords, &exclude_range) )
    {
        ERRWRAP2(_self_->setExcludeRange(exclude_range));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_AlignMTB_setMaxBits(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AlignMTB_Type))
        return failmsgp("Incorrect type of self (must be 'AlignMTB' or its derivative)");
    cv::AlignMTB* _self_ = dynamic_cast<cv::AlignMTB*>(((pyopencv_AlignMTB_t*)self)->v.get());
    int max_bits=0;

    const char* keywords[] = { "max_bits", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:AlignMTB.setMaxBits", (char**)keywords, &max_bits) )
    {
        ERRWRAP2(_self_->setMaxBits(max_bits));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_AlignMTB_shiftMat(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AlignMTB_Type))
        return failmsgp("Incorrect type of self (must be 'AlignMTB' or its derivative)");
    cv::AlignMTB* _self_ = dynamic_cast<cv::AlignMTB*>(((pyopencv_AlignMTB_t*)self)->v.get());
    {
    PyObject* pyobj_src = NULL;
    Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_shift = NULL;
    Point shift;

    const char* keywords[] = { "src", "shift", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:AlignMTB.shiftMat", (char**)keywords, &pyobj_src, &pyobj_shift, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_shift, shift, ArgInfo("shift", 0)) )
    {
        ERRWRAP2(_self_->shiftMat(src, dst, shift));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    UMat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_shift = NULL;
    Point shift;

    const char* keywords[] = { "src", "shift", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:AlignMTB.shiftMat", (char**)keywords, &pyobj_src, &pyobj_shift, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_shift, shift, ArgInfo("shift", 0)) )
    {
        ERRWRAP2(_self_->shiftMat(src, dst, shift));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}



static PyMethodDef pyopencv_AlignMTB_methods[] =
{
    {"calculateShift", (PyCFunction)pyopencv_cv_AlignMTB_calculateShift, METH_VARARGS | METH_KEYWORDS, "calculateShift(img0, img1) -> retval"},
    {"computeBitmaps", (PyCFunction)pyopencv_cv_AlignMTB_computeBitmaps, METH_VARARGS | METH_KEYWORDS, "computeBitmaps(img[, tb[, eb]]) -> tb, eb"},
    {"getCut", (PyCFunction)pyopencv_cv_AlignMTB_getCut, METH_VARARGS | METH_KEYWORDS, "getCut() -> retval"},
    {"getExcludeRange", (PyCFunction)pyopencv_cv_AlignMTB_getExcludeRange, METH_VARARGS | METH_KEYWORDS, "getExcludeRange() -> retval"},
    {"getMaxBits", (PyCFunction)pyopencv_cv_AlignMTB_getMaxBits, METH_VARARGS | METH_KEYWORDS, "getMaxBits() -> retval"},
    {"process", (PyCFunction)pyopencv_cv_AlignMTB_process, METH_VARARGS | METH_KEYWORDS, "process(src, dst, times, response) -> None  or  process(src, dst) -> None"},
    {"setCut", (PyCFunction)pyopencv_cv_AlignMTB_setCut, METH_VARARGS | METH_KEYWORDS, "setCut(value) -> None"},
    {"setExcludeRange", (PyCFunction)pyopencv_cv_AlignMTB_setExcludeRange, METH_VARARGS | METH_KEYWORDS, "setExcludeRange(exclude_range) -> None"},
    {"setMaxBits", (PyCFunction)pyopencv_cv_AlignMTB_setMaxBits, METH_VARARGS | METH_KEYWORDS, "setMaxBits(max_bits) -> None"},
    {"shiftMat", (PyCFunction)pyopencv_cv_AlignMTB_shiftMat, METH_VARARGS | METH_KEYWORDS, "shiftMat(src, shift[, dst]) -> dst"},

    {NULL,          NULL}
};

static void pyopencv_AlignMTB_specials(void)
{
    pyopencv_AlignMTB_Type.tp_base = &pyopencv_AlignExposures_Type;
    pyopencv_AlignMTB_Type.tp_dealloc = pyopencv_AlignMTB_dealloc;
    pyopencv_AlignMTB_Type.tp_repr = pyopencv_AlignMTB_repr;
    pyopencv_AlignMTB_Type.tp_getset = pyopencv_AlignMTB_getseters;
    pyopencv_AlignMTB_Type.tp_methods = pyopencv_AlignMTB_methods;
}

static PyObject* pyopencv_CalibrateCRF_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<CalibrateCRF %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_CalibrateCRF_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_CalibrateCRF_process(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CalibrateCRF_Type))
        return failmsgp("Incorrect type of self (must be 'CalibrateCRF' or its derivative)");
    cv::CalibrateCRF* _self_ = dynamic_cast<cv::CalibrateCRF*>(((pyopencv_CalibrateCRF_t*)self)->v.get());
    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_times = NULL;
    Mat times;

    const char* keywords[] = { "src", "times", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:CalibrateCRF.process", (char**)keywords, &pyobj_src, &pyobj_times, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_times, times, ArgInfo("times", 0)) )
    {
        ERRWRAP2(_self_->process(src, dst, times));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_times = NULL;
    UMat times;

    const char* keywords[] = { "src", "times", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:CalibrateCRF.process", (char**)keywords, &pyobj_src, &pyobj_times, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_times, times, ArgInfo("times", 0)) )
    {
        ERRWRAP2(_self_->process(src, dst, times));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}



static PyMethodDef pyopencv_CalibrateCRF_methods[] =
{
    {"process", (PyCFunction)pyopencv_cv_CalibrateCRF_process, METH_VARARGS | METH_KEYWORDS, "process(src, times[, dst]) -> dst"},

    {NULL,          NULL}
};

static void pyopencv_CalibrateCRF_specials(void)
{
    pyopencv_CalibrateCRF_Type.tp_base = &pyopencv_Algorithm_Type;
    pyopencv_CalibrateCRF_Type.tp_dealloc = pyopencv_CalibrateCRF_dealloc;
    pyopencv_CalibrateCRF_Type.tp_repr = pyopencv_CalibrateCRF_repr;
    pyopencv_CalibrateCRF_Type.tp_getset = pyopencv_CalibrateCRF_getseters;
    pyopencv_CalibrateCRF_Type.tp_methods = pyopencv_CalibrateCRF_methods;
}

static PyObject* pyopencv_CalibrateDebevec_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<CalibrateDebevec %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_CalibrateDebevec_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_CalibrateDebevec_getLambda(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CalibrateDebevec_Type))
        return failmsgp("Incorrect type of self (must be 'CalibrateDebevec' or its derivative)");
    cv::CalibrateDebevec* _self_ = dynamic_cast<cv::CalibrateDebevec*>(((pyopencv_CalibrateDebevec_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getLambda());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_CalibrateDebevec_getRandom(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CalibrateDebevec_Type))
        return failmsgp("Incorrect type of self (must be 'CalibrateDebevec' or its derivative)");
    cv::CalibrateDebevec* _self_ = dynamic_cast<cv::CalibrateDebevec*>(((pyopencv_CalibrateDebevec_t*)self)->v.get());
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getRandom());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_CalibrateDebevec_getSamples(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CalibrateDebevec_Type))
        return failmsgp("Incorrect type of self (must be 'CalibrateDebevec' or its derivative)");
    cv::CalibrateDebevec* _self_ = dynamic_cast<cv::CalibrateDebevec*>(((pyopencv_CalibrateDebevec_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getSamples());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_CalibrateDebevec_setLambda(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CalibrateDebevec_Type))
        return failmsgp("Incorrect type of self (must be 'CalibrateDebevec' or its derivative)");
    cv::CalibrateDebevec* _self_ = dynamic_cast<cv::CalibrateDebevec*>(((pyopencv_CalibrateDebevec_t*)self)->v.get());
    float lambda=0.f;

    const char* keywords[] = { "lambda", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:CalibrateDebevec.setLambda", (char**)keywords, &lambda) )
    {
        ERRWRAP2(_self_->setLambda(lambda));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_CalibrateDebevec_setRandom(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CalibrateDebevec_Type))
        return failmsgp("Incorrect type of self (must be 'CalibrateDebevec' or its derivative)");
    cv::CalibrateDebevec* _self_ = dynamic_cast<cv::CalibrateDebevec*>(((pyopencv_CalibrateDebevec_t*)self)->v.get());
    bool random=0;

    const char* keywords[] = { "random", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:CalibrateDebevec.setRandom", (char**)keywords, &random) )
    {
        ERRWRAP2(_self_->setRandom(random));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_CalibrateDebevec_setSamples(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CalibrateDebevec_Type))
        return failmsgp("Incorrect type of self (must be 'CalibrateDebevec' or its derivative)");
    cv::CalibrateDebevec* _self_ = dynamic_cast<cv::CalibrateDebevec*>(((pyopencv_CalibrateDebevec_t*)self)->v.get());
    int samples=0;

    const char* keywords[] = { "samples", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:CalibrateDebevec.setSamples", (char**)keywords, &samples) )
    {
        ERRWRAP2(_self_->setSamples(samples));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_CalibrateDebevec_methods[] =
{
    {"getLambda", (PyCFunction)pyopencv_cv_CalibrateDebevec_getLambda, METH_VARARGS | METH_KEYWORDS, "getLambda() -> retval"},
    {"getRandom", (PyCFunction)pyopencv_cv_CalibrateDebevec_getRandom, METH_VARARGS | METH_KEYWORDS, "getRandom() -> retval"},
    {"getSamples", (PyCFunction)pyopencv_cv_CalibrateDebevec_getSamples, METH_VARARGS | METH_KEYWORDS, "getSamples() -> retval"},
    {"setLambda", (PyCFunction)pyopencv_cv_CalibrateDebevec_setLambda, METH_VARARGS | METH_KEYWORDS, "setLambda(lambda) -> None"},
    {"setRandom", (PyCFunction)pyopencv_cv_CalibrateDebevec_setRandom, METH_VARARGS | METH_KEYWORDS, "setRandom(random) -> None"},
    {"setSamples", (PyCFunction)pyopencv_cv_CalibrateDebevec_setSamples, METH_VARARGS | METH_KEYWORDS, "setSamples(samples) -> None"},

    {NULL,          NULL}
};

static void pyopencv_CalibrateDebevec_specials(void)
{
    pyopencv_CalibrateDebevec_Type.tp_base = &pyopencv_CalibrateCRF_Type;
    pyopencv_CalibrateDebevec_Type.tp_dealloc = pyopencv_CalibrateDebevec_dealloc;
    pyopencv_CalibrateDebevec_Type.tp_repr = pyopencv_CalibrateDebevec_repr;
    pyopencv_CalibrateDebevec_Type.tp_getset = pyopencv_CalibrateDebevec_getseters;
    pyopencv_CalibrateDebevec_Type.tp_methods = pyopencv_CalibrateDebevec_methods;
}

static PyObject* pyopencv_CalibrateRobertson_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<CalibrateRobertson %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_CalibrateRobertson_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_CalibrateRobertson_getMaxIter(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CalibrateRobertson_Type))
        return failmsgp("Incorrect type of self (must be 'CalibrateRobertson' or its derivative)");
    cv::CalibrateRobertson* _self_ = dynamic_cast<cv::CalibrateRobertson*>(((pyopencv_CalibrateRobertson_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getMaxIter());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_CalibrateRobertson_getRadiance(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CalibrateRobertson_Type))
        return failmsgp("Incorrect type of self (must be 'CalibrateRobertson' or its derivative)");
    cv::CalibrateRobertson* _self_ = dynamic_cast<cv::CalibrateRobertson*>(((pyopencv_CalibrateRobertson_t*)self)->v.get());
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getRadiance());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_CalibrateRobertson_getThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CalibrateRobertson_Type))
        return failmsgp("Incorrect type of self (must be 'CalibrateRobertson' or its derivative)");
    cv::CalibrateRobertson* _self_ = dynamic_cast<cv::CalibrateRobertson*>(((pyopencv_CalibrateRobertson_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getThreshold());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_CalibrateRobertson_setMaxIter(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CalibrateRobertson_Type))
        return failmsgp("Incorrect type of self (must be 'CalibrateRobertson' or its derivative)");
    cv::CalibrateRobertson* _self_ = dynamic_cast<cv::CalibrateRobertson*>(((pyopencv_CalibrateRobertson_t*)self)->v.get());
    int max_iter=0;

    const char* keywords[] = { "max_iter", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:CalibrateRobertson.setMaxIter", (char**)keywords, &max_iter) )
    {
        ERRWRAP2(_self_->setMaxIter(max_iter));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_CalibrateRobertson_setThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CalibrateRobertson_Type))
        return failmsgp("Incorrect type of self (must be 'CalibrateRobertson' or its derivative)");
    cv::CalibrateRobertson* _self_ = dynamic_cast<cv::CalibrateRobertson*>(((pyopencv_CalibrateRobertson_t*)self)->v.get());
    float threshold=0.f;

    const char* keywords[] = { "threshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:CalibrateRobertson.setThreshold", (char**)keywords, &threshold) )
    {
        ERRWRAP2(_self_->setThreshold(threshold));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_CalibrateRobertson_methods[] =
{
    {"getMaxIter", (PyCFunction)pyopencv_cv_CalibrateRobertson_getMaxIter, METH_VARARGS | METH_KEYWORDS, "getMaxIter() -> retval"},
    {"getRadiance", (PyCFunction)pyopencv_cv_CalibrateRobertson_getRadiance, METH_VARARGS | METH_KEYWORDS, "getRadiance() -> retval"},
    {"getThreshold", (PyCFunction)pyopencv_cv_CalibrateRobertson_getThreshold, METH_VARARGS | METH_KEYWORDS, "getThreshold() -> retval"},
    {"setMaxIter", (PyCFunction)pyopencv_cv_CalibrateRobertson_setMaxIter, METH_VARARGS | METH_KEYWORDS, "setMaxIter(max_iter) -> None"},
    {"setThreshold", (PyCFunction)pyopencv_cv_CalibrateRobertson_setThreshold, METH_VARARGS | METH_KEYWORDS, "setThreshold(threshold) -> None"},

    {NULL,          NULL}
};

static void pyopencv_CalibrateRobertson_specials(void)
{
    pyopencv_CalibrateRobertson_Type.tp_base = &pyopencv_CalibrateCRF_Type;
    pyopencv_CalibrateRobertson_Type.tp_dealloc = pyopencv_CalibrateRobertson_dealloc;
    pyopencv_CalibrateRobertson_Type.tp_repr = pyopencv_CalibrateRobertson_repr;
    pyopencv_CalibrateRobertson_Type.tp_getset = pyopencv_CalibrateRobertson_getseters;
    pyopencv_CalibrateRobertson_Type.tp_methods = pyopencv_CalibrateRobertson_methods;
}

static PyObject* pyopencv_MergeExposures_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<MergeExposures %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_MergeExposures_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_MergeExposures_process(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_MergeExposures_Type))
        return failmsgp("Incorrect type of self (must be 'MergeExposures' or its derivative)");
    cv::MergeExposures* _self_ = dynamic_cast<cv::MergeExposures*>(((pyopencv_MergeExposures_t*)self)->v.get());
    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_times = NULL;
    Mat times;
    PyObject* pyobj_response = NULL;
    Mat response;

    const char* keywords[] = { "src", "times", "response", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|O:MergeExposures.process", (char**)keywords, &pyobj_src, &pyobj_times, &pyobj_response, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_times, times, ArgInfo("times", 0)) &&
        pyopencv_to(pyobj_response, response, ArgInfo("response", 0)) )
    {
        ERRWRAP2(_self_->process(src, dst, times, response));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_times = NULL;
    UMat times;
    PyObject* pyobj_response = NULL;
    UMat response;

    const char* keywords[] = { "src", "times", "response", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|O:MergeExposures.process", (char**)keywords, &pyobj_src, &pyobj_times, &pyobj_response, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_times, times, ArgInfo("times", 0)) &&
        pyopencv_to(pyobj_response, response, ArgInfo("response", 0)) )
    {
        ERRWRAP2(_self_->process(src, dst, times, response));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}



static PyMethodDef pyopencv_MergeExposures_methods[] =
{
    {"process", (PyCFunction)pyopencv_cv_MergeExposures_process, METH_VARARGS | METH_KEYWORDS, "process(src, times, response[, dst]) -> dst"},

    {NULL,          NULL}
};

static void pyopencv_MergeExposures_specials(void)
{
    pyopencv_MergeExposures_Type.tp_base = &pyopencv_Algorithm_Type;
    pyopencv_MergeExposures_Type.tp_dealloc = pyopencv_MergeExposures_dealloc;
    pyopencv_MergeExposures_Type.tp_repr = pyopencv_MergeExposures_repr;
    pyopencv_MergeExposures_Type.tp_getset = pyopencv_MergeExposures_getseters;
    pyopencv_MergeExposures_Type.tp_methods = pyopencv_MergeExposures_methods;
}

static PyObject* pyopencv_MergeDebevec_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<MergeDebevec %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_MergeDebevec_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_MergeDebevec_process(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_MergeDebevec_Type))
        return failmsgp("Incorrect type of self (must be 'MergeDebevec' or its derivative)");
    cv::MergeDebevec* _self_ = dynamic_cast<cv::MergeDebevec*>(((pyopencv_MergeDebevec_t*)self)->v.get());
    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_times = NULL;
    Mat times;
    PyObject* pyobj_response = NULL;
    Mat response;

    const char* keywords[] = { "src", "times", "response", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|O:MergeDebevec.process", (char**)keywords, &pyobj_src, &pyobj_times, &pyobj_response, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_times, times, ArgInfo("times", 0)) &&
        pyopencv_to(pyobj_response, response, ArgInfo("response", 0)) )
    {
        ERRWRAP2(_self_->process(src, dst, times, response));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_times = NULL;
    UMat times;
    PyObject* pyobj_response = NULL;
    UMat response;

    const char* keywords[] = { "src", "times", "response", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|O:MergeDebevec.process", (char**)keywords, &pyobj_src, &pyobj_times, &pyobj_response, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_times, times, ArgInfo("times", 0)) &&
        pyopencv_to(pyobj_response, response, ArgInfo("response", 0)) )
    {
        ERRWRAP2(_self_->process(src, dst, times, response));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_times = NULL;
    Mat times;

    const char* keywords[] = { "src", "times", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:MergeDebevec.process", (char**)keywords, &pyobj_src, &pyobj_times, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_times, times, ArgInfo("times", 0)) )
    {
        ERRWRAP2(_self_->process(src, dst, times));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_times = NULL;
    UMat times;

    const char* keywords[] = { "src", "times", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:MergeDebevec.process", (char**)keywords, &pyobj_src, &pyobj_times, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_times, times, ArgInfo("times", 0)) )
    {
        ERRWRAP2(_self_->process(src, dst, times));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}



static PyMethodDef pyopencv_MergeDebevec_methods[] =
{
    {"process", (PyCFunction)pyopencv_cv_MergeDebevec_process, METH_VARARGS | METH_KEYWORDS, "process(src, times, response[, dst]) -> dst  or  process(src, times[, dst]) -> dst"},

    {NULL,          NULL}
};

static void pyopencv_MergeDebevec_specials(void)
{
    pyopencv_MergeDebevec_Type.tp_base = &pyopencv_MergeExposures_Type;
    pyopencv_MergeDebevec_Type.tp_dealloc = pyopencv_MergeDebevec_dealloc;
    pyopencv_MergeDebevec_Type.tp_repr = pyopencv_MergeDebevec_repr;
    pyopencv_MergeDebevec_Type.tp_getset = pyopencv_MergeDebevec_getseters;
    pyopencv_MergeDebevec_Type.tp_methods = pyopencv_MergeDebevec_methods;
}

static PyObject* pyopencv_MergeMertens_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<MergeMertens %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_MergeMertens_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_MergeMertens_getContrastWeight(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_MergeMertens_Type))
        return failmsgp("Incorrect type of self (must be 'MergeMertens' or its derivative)");
    cv::MergeMertens* _self_ = dynamic_cast<cv::MergeMertens*>(((pyopencv_MergeMertens_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getContrastWeight());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_MergeMertens_getExposureWeight(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_MergeMertens_Type))
        return failmsgp("Incorrect type of self (must be 'MergeMertens' or its derivative)");
    cv::MergeMertens* _self_ = dynamic_cast<cv::MergeMertens*>(((pyopencv_MergeMertens_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getExposureWeight());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_MergeMertens_getSaturationWeight(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_MergeMertens_Type))
        return failmsgp("Incorrect type of self (must be 'MergeMertens' or its derivative)");
    cv::MergeMertens* _self_ = dynamic_cast<cv::MergeMertens*>(((pyopencv_MergeMertens_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getSaturationWeight());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_MergeMertens_process(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_MergeMertens_Type))
        return failmsgp("Incorrect type of self (must be 'MergeMertens' or its derivative)");
    cv::MergeMertens* _self_ = dynamic_cast<cv::MergeMertens*>(((pyopencv_MergeMertens_t*)self)->v.get());
    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_times = NULL;
    Mat times;
    PyObject* pyobj_response = NULL;
    Mat response;

    const char* keywords[] = { "src", "times", "response", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|O:MergeMertens.process", (char**)keywords, &pyobj_src, &pyobj_times, &pyobj_response, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_times, times, ArgInfo("times", 0)) &&
        pyopencv_to(pyobj_response, response, ArgInfo("response", 0)) )
    {
        ERRWRAP2(_self_->process(src, dst, times, response));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_times = NULL;
    UMat times;
    PyObject* pyobj_response = NULL;
    UMat response;

    const char* keywords[] = { "src", "times", "response", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|O:MergeMertens.process", (char**)keywords, &pyobj_src, &pyobj_times, &pyobj_response, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_times, times, ArgInfo("times", 0)) &&
        pyopencv_to(pyobj_response, response, ArgInfo("response", 0)) )
    {
        ERRWRAP2(_self_->process(src, dst, times, response));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;

    const char* keywords[] = { "src", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:MergeMertens.process", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(_self_->process(src, dst));
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
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:MergeMertens.process", (char**)keywords, &pyobj_src, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) )
    {
        ERRWRAP2(_self_->process(src, dst));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_MergeMertens_setContrastWeight(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_MergeMertens_Type))
        return failmsgp("Incorrect type of self (must be 'MergeMertens' or its derivative)");
    cv::MergeMertens* _self_ = dynamic_cast<cv::MergeMertens*>(((pyopencv_MergeMertens_t*)self)->v.get());
    float contrast_weiht=0.f;

    const char* keywords[] = { "contrast_weiht", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:MergeMertens.setContrastWeight", (char**)keywords, &contrast_weiht) )
    {
        ERRWRAP2(_self_->setContrastWeight(contrast_weiht));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_MergeMertens_setExposureWeight(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_MergeMertens_Type))
        return failmsgp("Incorrect type of self (must be 'MergeMertens' or its derivative)");
    cv::MergeMertens* _self_ = dynamic_cast<cv::MergeMertens*>(((pyopencv_MergeMertens_t*)self)->v.get());
    float exposure_weight=0.f;

    const char* keywords[] = { "exposure_weight", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:MergeMertens.setExposureWeight", (char**)keywords, &exposure_weight) )
    {
        ERRWRAP2(_self_->setExposureWeight(exposure_weight));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_MergeMertens_setSaturationWeight(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_MergeMertens_Type))
        return failmsgp("Incorrect type of self (must be 'MergeMertens' or its derivative)");
    cv::MergeMertens* _self_ = dynamic_cast<cv::MergeMertens*>(((pyopencv_MergeMertens_t*)self)->v.get());
    float saturation_weight=0.f;

    const char* keywords[] = { "saturation_weight", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:MergeMertens.setSaturationWeight", (char**)keywords, &saturation_weight) )
    {
        ERRWRAP2(_self_->setSaturationWeight(saturation_weight));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_MergeMertens_methods[] =
{
    {"getContrastWeight", (PyCFunction)pyopencv_cv_MergeMertens_getContrastWeight, METH_VARARGS | METH_KEYWORDS, "getContrastWeight() -> retval"},
    {"getExposureWeight", (PyCFunction)pyopencv_cv_MergeMertens_getExposureWeight, METH_VARARGS | METH_KEYWORDS, "getExposureWeight() -> retval"},
    {"getSaturationWeight", (PyCFunction)pyopencv_cv_MergeMertens_getSaturationWeight, METH_VARARGS | METH_KEYWORDS, "getSaturationWeight() -> retval"},
    {"process", (PyCFunction)pyopencv_cv_MergeMertens_process, METH_VARARGS | METH_KEYWORDS, "process(src, times, response[, dst]) -> dst  or  process(src[, dst]) -> dst"},
    {"setContrastWeight", (PyCFunction)pyopencv_cv_MergeMertens_setContrastWeight, METH_VARARGS | METH_KEYWORDS, "setContrastWeight(contrast_weiht) -> None"},
    {"setExposureWeight", (PyCFunction)pyopencv_cv_MergeMertens_setExposureWeight, METH_VARARGS | METH_KEYWORDS, "setExposureWeight(exposure_weight) -> None"},
    {"setSaturationWeight", (PyCFunction)pyopencv_cv_MergeMertens_setSaturationWeight, METH_VARARGS | METH_KEYWORDS, "setSaturationWeight(saturation_weight) -> None"},

    {NULL,          NULL}
};

static void pyopencv_MergeMertens_specials(void)
{
    pyopencv_MergeMertens_Type.tp_base = &pyopencv_MergeExposures_Type;
    pyopencv_MergeMertens_Type.tp_dealloc = pyopencv_MergeMertens_dealloc;
    pyopencv_MergeMertens_Type.tp_repr = pyopencv_MergeMertens_repr;
    pyopencv_MergeMertens_Type.tp_getset = pyopencv_MergeMertens_getseters;
    pyopencv_MergeMertens_Type.tp_methods = pyopencv_MergeMertens_methods;
}

static PyObject* pyopencv_MergeRobertson_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<MergeRobertson %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_MergeRobertson_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_MergeRobertson_process(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_MergeRobertson_Type))
        return failmsgp("Incorrect type of self (must be 'MergeRobertson' or its derivative)");
    cv::MergeRobertson* _self_ = dynamic_cast<cv::MergeRobertson*>(((pyopencv_MergeRobertson_t*)self)->v.get());
    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_times = NULL;
    Mat times;
    PyObject* pyobj_response = NULL;
    Mat response;

    const char* keywords[] = { "src", "times", "response", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|O:MergeRobertson.process", (char**)keywords, &pyobj_src, &pyobj_times, &pyobj_response, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_times, times, ArgInfo("times", 0)) &&
        pyopencv_to(pyobj_response, response, ArgInfo("response", 0)) )
    {
        ERRWRAP2(_self_->process(src, dst, times, response));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_times = NULL;
    UMat times;
    PyObject* pyobj_response = NULL;
    UMat response;

    const char* keywords[] = { "src", "times", "response", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO|O:MergeRobertson.process", (char**)keywords, &pyobj_src, &pyobj_times, &pyobj_response, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_times, times, ArgInfo("times", 0)) &&
        pyopencv_to(pyobj_response, response, ArgInfo("response", 0)) )
    {
        ERRWRAP2(_self_->process(src, dst, times, response));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    Mat dst;
    PyObject* pyobj_times = NULL;
    Mat times;

    const char* keywords[] = { "src", "times", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:MergeRobertson.process", (char**)keywords, &pyobj_src, &pyobj_times, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_times, times, ArgInfo("times", 0)) )
    {
        ERRWRAP2(_self_->process(src, dst, times));
        return pyopencv_from(dst);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_src = NULL;
    vector_Mat src;
    PyObject* pyobj_dst = NULL;
    UMat dst;
    PyObject* pyobj_times = NULL;
    UMat times;

    const char* keywords[] = { "src", "times", "dst", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:MergeRobertson.process", (char**)keywords, &pyobj_src, &pyobj_times, &pyobj_dst) &&
        pyopencv_to(pyobj_src, src, ArgInfo("src", 0)) &&
        pyopencv_to(pyobj_dst, dst, ArgInfo("dst", 1)) &&
        pyopencv_to(pyobj_times, times, ArgInfo("times", 0)) )
    {
        ERRWRAP2(_self_->process(src, dst, times));
        return pyopencv_from(dst);
    }
    }

    return NULL;
}



static PyMethodDef pyopencv_MergeRobertson_methods[] =
{
    {"process", (PyCFunction)pyopencv_cv_MergeRobertson_process, METH_VARARGS | METH_KEYWORDS, "process(src, times, response[, dst]) -> dst  or  process(src, times[, dst]) -> dst"},

    {NULL,          NULL}
};

static void pyopencv_MergeRobertson_specials(void)
{
    pyopencv_MergeRobertson_Type.tp_base = &pyopencv_MergeExposures_Type;
    pyopencv_MergeRobertson_Type.tp_dealloc = pyopencv_MergeRobertson_dealloc;
    pyopencv_MergeRobertson_Type.tp_repr = pyopencv_MergeRobertson_repr;
    pyopencv_MergeRobertson_Type.tp_getset = pyopencv_MergeRobertson_getseters;
    pyopencv_MergeRobertson_Type.tp_methods = pyopencv_MergeRobertson_methods;
}

static PyObject* pyopencv_KalmanFilter_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<KalmanFilter %p>", self);
    return PyString_FromString(str);
}


static PyObject* pyopencv_KalmanFilter_get_controlMatrix(pyopencv_KalmanFilter_t* p, void *closure)
{
    return pyopencv_from(p->v->controlMatrix);
}

static int pyopencv_KalmanFilter_set_controlMatrix(pyopencv_KalmanFilter_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the controlMatrix attribute");
        return -1;
    }
    return pyopencv_to(value, p->v->controlMatrix) ? 0 : -1;
}

static PyObject* pyopencv_KalmanFilter_get_errorCovPost(pyopencv_KalmanFilter_t* p, void *closure)
{
    return pyopencv_from(p->v->errorCovPost);
}

static int pyopencv_KalmanFilter_set_errorCovPost(pyopencv_KalmanFilter_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the errorCovPost attribute");
        return -1;
    }
    return pyopencv_to(value, p->v->errorCovPost) ? 0 : -1;
}

static PyObject* pyopencv_KalmanFilter_get_errorCovPre(pyopencv_KalmanFilter_t* p, void *closure)
{
    return pyopencv_from(p->v->errorCovPre);
}

static int pyopencv_KalmanFilter_set_errorCovPre(pyopencv_KalmanFilter_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the errorCovPre attribute");
        return -1;
    }
    return pyopencv_to(value, p->v->errorCovPre) ? 0 : -1;
}

static PyObject* pyopencv_KalmanFilter_get_gain(pyopencv_KalmanFilter_t* p, void *closure)
{
    return pyopencv_from(p->v->gain);
}

static int pyopencv_KalmanFilter_set_gain(pyopencv_KalmanFilter_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the gain attribute");
        return -1;
    }
    return pyopencv_to(value, p->v->gain) ? 0 : -1;
}

static PyObject* pyopencv_KalmanFilter_get_measurementMatrix(pyopencv_KalmanFilter_t* p, void *closure)
{
    return pyopencv_from(p->v->measurementMatrix);
}

static int pyopencv_KalmanFilter_set_measurementMatrix(pyopencv_KalmanFilter_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the measurementMatrix attribute");
        return -1;
    }
    return pyopencv_to(value, p->v->measurementMatrix) ? 0 : -1;
}

static PyObject* pyopencv_KalmanFilter_get_measurementNoiseCov(pyopencv_KalmanFilter_t* p, void *closure)
{
    return pyopencv_from(p->v->measurementNoiseCov);
}

static int pyopencv_KalmanFilter_set_measurementNoiseCov(pyopencv_KalmanFilter_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the measurementNoiseCov attribute");
        return -1;
    }
    return pyopencv_to(value, p->v->measurementNoiseCov) ? 0 : -1;
}

static PyObject* pyopencv_KalmanFilter_get_processNoiseCov(pyopencv_KalmanFilter_t* p, void *closure)
{
    return pyopencv_from(p->v->processNoiseCov);
}

static int pyopencv_KalmanFilter_set_processNoiseCov(pyopencv_KalmanFilter_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the processNoiseCov attribute");
        return -1;
    }
    return pyopencv_to(value, p->v->processNoiseCov) ? 0 : -1;
}

static PyObject* pyopencv_KalmanFilter_get_statePost(pyopencv_KalmanFilter_t* p, void *closure)
{
    return pyopencv_from(p->v->statePost);
}

static int pyopencv_KalmanFilter_set_statePost(pyopencv_KalmanFilter_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the statePost attribute");
        return -1;
    }
    return pyopencv_to(value, p->v->statePost) ? 0 : -1;
}

static PyObject* pyopencv_KalmanFilter_get_statePre(pyopencv_KalmanFilter_t* p, void *closure)
{
    return pyopencv_from(p->v->statePre);
}

static int pyopencv_KalmanFilter_set_statePre(pyopencv_KalmanFilter_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the statePre attribute");
        return -1;
    }
    return pyopencv_to(value, p->v->statePre) ? 0 : -1;
}

static PyObject* pyopencv_KalmanFilter_get_transitionMatrix(pyopencv_KalmanFilter_t* p, void *closure)
{
    return pyopencv_from(p->v->transitionMatrix);
}

static int pyopencv_KalmanFilter_set_transitionMatrix(pyopencv_KalmanFilter_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the transitionMatrix attribute");
        return -1;
    }
    return pyopencv_to(value, p->v->transitionMatrix) ? 0 : -1;
}


static PyGetSetDef pyopencv_KalmanFilter_getseters[] =
{
    {(char*)"controlMatrix", (getter)pyopencv_KalmanFilter_get_controlMatrix, (setter)pyopencv_KalmanFilter_set_controlMatrix, (char*)"controlMatrix", NULL},
    {(char*)"errorCovPost", (getter)pyopencv_KalmanFilter_get_errorCovPost, (setter)pyopencv_KalmanFilter_set_errorCovPost, (char*)"errorCovPost", NULL},
    {(char*)"errorCovPre", (getter)pyopencv_KalmanFilter_get_errorCovPre, (setter)pyopencv_KalmanFilter_set_errorCovPre, (char*)"errorCovPre", NULL},
    {(char*)"gain", (getter)pyopencv_KalmanFilter_get_gain, (setter)pyopencv_KalmanFilter_set_gain, (char*)"gain", NULL},
    {(char*)"measurementMatrix", (getter)pyopencv_KalmanFilter_get_measurementMatrix, (setter)pyopencv_KalmanFilter_set_measurementMatrix, (char*)"measurementMatrix", NULL},
    {(char*)"measurementNoiseCov", (getter)pyopencv_KalmanFilter_get_measurementNoiseCov, (setter)pyopencv_KalmanFilter_set_measurementNoiseCov, (char*)"measurementNoiseCov", NULL},
    {(char*)"processNoiseCov", (getter)pyopencv_KalmanFilter_get_processNoiseCov, (setter)pyopencv_KalmanFilter_set_processNoiseCov, (char*)"processNoiseCov", NULL},
    {(char*)"statePost", (getter)pyopencv_KalmanFilter_get_statePost, (setter)pyopencv_KalmanFilter_set_statePost, (char*)"statePost", NULL},
    {(char*)"statePre", (getter)pyopencv_KalmanFilter_get_statePre, (setter)pyopencv_KalmanFilter_set_statePre, (char*)"statePre", NULL},
    {(char*)"transitionMatrix", (getter)pyopencv_KalmanFilter_get_transitionMatrix, (setter)pyopencv_KalmanFilter_set_transitionMatrix, (char*)"transitionMatrix", NULL},
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_KalmanFilter_correct(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_KalmanFilter_Type))
        return failmsgp("Incorrect type of self (must be 'KalmanFilter' or its derivative)");
    cv::KalmanFilter* _self_ = ((pyopencv_KalmanFilter_t*)self)->v.get();
    {
    PyObject* pyobj_measurement = NULL;
    Mat measurement;
    Mat retval;

    const char* keywords[] = { "measurement", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:KalmanFilter.correct", (char**)keywords, &pyobj_measurement) &&
        pyopencv_to(pyobj_measurement, measurement, ArgInfo("measurement", 0)) )
    {
        ERRWRAP2(retval = _self_->correct(measurement));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_measurement = NULL;
    Mat measurement;
    Mat retval;

    const char* keywords[] = { "measurement", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:KalmanFilter.correct", (char**)keywords, &pyobj_measurement) &&
        pyopencv_to(pyobj_measurement, measurement, ArgInfo("measurement", 0)) )
    {
        ERRWRAP2(retval = _self_->correct(measurement));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_KalmanFilter_predict(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_KalmanFilter_Type))
        return failmsgp("Incorrect type of self (must be 'KalmanFilter' or its derivative)");
    cv::KalmanFilter* _self_ = ((pyopencv_KalmanFilter_t*)self)->v.get();
    {
    PyObject* pyobj_control = NULL;
    Mat control;
    Mat retval;

    const char* keywords[] = { "control", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|O:KalmanFilter.predict", (char**)keywords, &pyobj_control) &&
        pyopencv_to(pyobj_control, control, ArgInfo("control", 0)) )
    {
        ERRWRAP2(retval = _self_->predict(control));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_control = NULL;
    Mat control;
    Mat retval;

    const char* keywords[] = { "control", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|O:KalmanFilter.predict", (char**)keywords, &pyobj_control) &&
        pyopencv_to(pyobj_control, control, ArgInfo("control", 0)) )
    {
        ERRWRAP2(retval = _self_->predict(control));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}



static PyMethodDef pyopencv_KalmanFilter_methods[] =
{
    {"correct", (PyCFunction)pyopencv_cv_KalmanFilter_correct, METH_VARARGS | METH_KEYWORDS, "correct(measurement) -> retval"},
    {"predict", (PyCFunction)pyopencv_cv_KalmanFilter_predict, METH_VARARGS | METH_KEYWORDS, "predict([, control]) -> retval"},

    {NULL,          NULL}
};

static void pyopencv_KalmanFilter_specials(void)
{
    pyopencv_KalmanFilter_Type.tp_base = NULL;
    pyopencv_KalmanFilter_Type.tp_dealloc = pyopencv_KalmanFilter_dealloc;
    pyopencv_KalmanFilter_Type.tp_repr = pyopencv_KalmanFilter_repr;
    pyopencv_KalmanFilter_Type.tp_getset = pyopencv_KalmanFilter_getseters;
    pyopencv_KalmanFilter_Type.tp_methods = pyopencv_KalmanFilter_methods;
}

static PyObject* pyopencv_DenseOpticalFlow_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<DenseOpticalFlow %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_DenseOpticalFlow_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_DenseOpticalFlow_calc(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DenseOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DenseOpticalFlow' or its derivative)");
    cv::DenseOpticalFlow* _self_ = dynamic_cast<cv::DenseOpticalFlow*>(((pyopencv_DenseOpticalFlow_t*)self)->v.get());
    {
    PyObject* pyobj_I0 = NULL;
    Mat I0;
    PyObject* pyobj_I1 = NULL;
    Mat I1;
    PyObject* pyobj_flow = NULL;
    Mat flow;

    const char* keywords[] = { "I0", "I1", "flow", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:DenseOpticalFlow.calc", (char**)keywords, &pyobj_I0, &pyobj_I1, &pyobj_flow) &&
        pyopencv_to(pyobj_I0, I0, ArgInfo("I0", 0)) &&
        pyopencv_to(pyobj_I1, I1, ArgInfo("I1", 0)) &&
        pyopencv_to(pyobj_flow, flow, ArgInfo("flow", 1)) )
    {
        ERRWRAP2(_self_->calc(I0, I1, flow));
        return pyopencv_from(flow);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_I0 = NULL;
    UMat I0;
    PyObject* pyobj_I1 = NULL;
    UMat I1;
    PyObject* pyobj_flow = NULL;
    UMat flow;

    const char* keywords[] = { "I0", "I1", "flow", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:DenseOpticalFlow.calc", (char**)keywords, &pyobj_I0, &pyobj_I1, &pyobj_flow) &&
        pyopencv_to(pyobj_I0, I0, ArgInfo("I0", 0)) &&
        pyopencv_to(pyobj_I1, I1, ArgInfo("I1", 0)) &&
        pyopencv_to(pyobj_flow, flow, ArgInfo("flow", 1)) )
    {
        ERRWRAP2(_self_->calc(I0, I1, flow));
        return pyopencv_from(flow);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_DenseOpticalFlow_collectGarbage(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DenseOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DenseOpticalFlow' or its derivative)");
    cv::DenseOpticalFlow* _self_ = dynamic_cast<cv::DenseOpticalFlow*>(((pyopencv_DenseOpticalFlow_t*)self)->v.get());

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(_self_->collectGarbage());
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_DenseOpticalFlow_methods[] =
{
    {"calc", (PyCFunction)pyopencv_cv_DenseOpticalFlow_calc, METH_VARARGS | METH_KEYWORDS, "calc(I0, I1, flow) -> flow"},
    {"collectGarbage", (PyCFunction)pyopencv_cv_DenseOpticalFlow_collectGarbage, METH_VARARGS | METH_KEYWORDS, "collectGarbage() -> None"},

    {NULL,          NULL}
};

static void pyopencv_DenseOpticalFlow_specials(void)
{
    pyopencv_DenseOpticalFlow_Type.tp_base = &pyopencv_Algorithm_Type;
    pyopencv_DenseOpticalFlow_Type.tp_dealloc = pyopencv_DenseOpticalFlow_dealloc;
    pyopencv_DenseOpticalFlow_Type.tp_repr = pyopencv_DenseOpticalFlow_repr;
    pyopencv_DenseOpticalFlow_Type.tp_getset = pyopencv_DenseOpticalFlow_getseters;
    pyopencv_DenseOpticalFlow_Type.tp_methods = pyopencv_DenseOpticalFlow_methods;
}

static PyObject* pyopencv_SparseOpticalFlow_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<SparseOpticalFlow %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_SparseOpticalFlow_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_SparseOpticalFlow_calc(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_SparseOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'SparseOpticalFlow' or its derivative)");
    cv::SparseOpticalFlow* _self_ = dynamic_cast<cv::SparseOpticalFlow*>(((pyopencv_SparseOpticalFlow_t*)self)->v.get());
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

    const char* keywords[] = { "prevImg", "nextImg", "prevPts", "nextPts", "status", "err", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OO:SparseOpticalFlow.calc", (char**)keywords, &pyobj_prevImg, &pyobj_nextImg, &pyobj_prevPts, &pyobj_nextPts, &pyobj_status, &pyobj_err) &&
        pyopencv_to(pyobj_prevImg, prevImg, ArgInfo("prevImg", 0)) &&
        pyopencv_to(pyobj_nextImg, nextImg, ArgInfo("nextImg", 0)) &&
        pyopencv_to(pyobj_prevPts, prevPts, ArgInfo("prevPts", 0)) &&
        pyopencv_to(pyobj_nextPts, nextPts, ArgInfo("nextPts", 1)) &&
        pyopencv_to(pyobj_status, status, ArgInfo("status", 1)) &&
        pyopencv_to(pyobj_err, err, ArgInfo("err", 1)) )
    {
        ERRWRAP2(_self_->calc(prevImg, nextImg, prevPts, nextPts, status, err));
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

    const char* keywords[] = { "prevImg", "nextImg", "prevPts", "nextPts", "status", "err", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OO:SparseOpticalFlow.calc", (char**)keywords, &pyobj_prevImg, &pyobj_nextImg, &pyobj_prevPts, &pyobj_nextPts, &pyobj_status, &pyobj_err) &&
        pyopencv_to(pyobj_prevImg, prevImg, ArgInfo("prevImg", 0)) &&
        pyopencv_to(pyobj_nextImg, nextImg, ArgInfo("nextImg", 0)) &&
        pyopencv_to(pyobj_prevPts, prevPts, ArgInfo("prevPts", 0)) &&
        pyopencv_to(pyobj_nextPts, nextPts, ArgInfo("nextPts", 1)) &&
        pyopencv_to(pyobj_status, status, ArgInfo("status", 1)) &&
        pyopencv_to(pyobj_err, err, ArgInfo("err", 1)) )
    {
        ERRWRAP2(_self_->calc(prevImg, nextImg, prevPts, nextPts, status, err));
        return Py_BuildValue("(NNN)", pyopencv_from(nextPts), pyopencv_from(status), pyopencv_from(err));
    }
    }

    return NULL;
}



static PyMethodDef pyopencv_SparseOpticalFlow_methods[] =
{
    {"calc", (PyCFunction)pyopencv_cv_SparseOpticalFlow_calc, METH_VARARGS | METH_KEYWORDS, "calc(prevImg, nextImg, prevPts, nextPts[, status[, err]]) -> nextPts, status, err"},

    {NULL,          NULL}
};

static void pyopencv_SparseOpticalFlow_specials(void)
{
    pyopencv_SparseOpticalFlow_Type.tp_base = &pyopencv_Algorithm_Type;
    pyopencv_SparseOpticalFlow_Type.tp_dealloc = pyopencv_SparseOpticalFlow_dealloc;
    pyopencv_SparseOpticalFlow_Type.tp_repr = pyopencv_SparseOpticalFlow_repr;
    pyopencv_SparseOpticalFlow_Type.tp_getset = pyopencv_SparseOpticalFlow_getseters;
    pyopencv_SparseOpticalFlow_Type.tp_methods = pyopencv_SparseOpticalFlow_methods;
}

static PyObject* pyopencv_DualTVL1OpticalFlow_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<DualTVL1OpticalFlow %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_DualTVL1OpticalFlow_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_DualTVL1OpticalFlow_getEpsilon(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DualTVL1OpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DualTVL1OpticalFlow' or its derivative)");
    cv::DualTVL1OpticalFlow* _self_ = dynamic_cast<cv::DualTVL1OpticalFlow*>(((pyopencv_DualTVL1OpticalFlow_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getEpsilon());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_DualTVL1OpticalFlow_getGamma(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DualTVL1OpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DualTVL1OpticalFlow' or its derivative)");
    cv::DualTVL1OpticalFlow* _self_ = dynamic_cast<cv::DualTVL1OpticalFlow*>(((pyopencv_DualTVL1OpticalFlow_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getGamma());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_DualTVL1OpticalFlow_getInnerIterations(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DualTVL1OpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DualTVL1OpticalFlow' or its derivative)");
    cv::DualTVL1OpticalFlow* _self_ = dynamic_cast<cv::DualTVL1OpticalFlow*>(((pyopencv_DualTVL1OpticalFlow_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getInnerIterations());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_DualTVL1OpticalFlow_getLambda(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DualTVL1OpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DualTVL1OpticalFlow' or its derivative)");
    cv::DualTVL1OpticalFlow* _self_ = dynamic_cast<cv::DualTVL1OpticalFlow*>(((pyopencv_DualTVL1OpticalFlow_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getLambda());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_DualTVL1OpticalFlow_getMedianFiltering(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DualTVL1OpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DualTVL1OpticalFlow' or its derivative)");
    cv::DualTVL1OpticalFlow* _self_ = dynamic_cast<cv::DualTVL1OpticalFlow*>(((pyopencv_DualTVL1OpticalFlow_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getMedianFiltering());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_DualTVL1OpticalFlow_getOuterIterations(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DualTVL1OpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DualTVL1OpticalFlow' or its derivative)");
    cv::DualTVL1OpticalFlow* _self_ = dynamic_cast<cv::DualTVL1OpticalFlow*>(((pyopencv_DualTVL1OpticalFlow_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getOuterIterations());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_DualTVL1OpticalFlow_getScaleStep(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DualTVL1OpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DualTVL1OpticalFlow' or its derivative)");
    cv::DualTVL1OpticalFlow* _self_ = dynamic_cast<cv::DualTVL1OpticalFlow*>(((pyopencv_DualTVL1OpticalFlow_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getScaleStep());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_DualTVL1OpticalFlow_getScalesNumber(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DualTVL1OpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DualTVL1OpticalFlow' or its derivative)");
    cv::DualTVL1OpticalFlow* _self_ = dynamic_cast<cv::DualTVL1OpticalFlow*>(((pyopencv_DualTVL1OpticalFlow_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getScalesNumber());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_DualTVL1OpticalFlow_getTau(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DualTVL1OpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DualTVL1OpticalFlow' or its derivative)");
    cv::DualTVL1OpticalFlow* _self_ = dynamic_cast<cv::DualTVL1OpticalFlow*>(((pyopencv_DualTVL1OpticalFlow_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTau());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_DualTVL1OpticalFlow_getTheta(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DualTVL1OpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DualTVL1OpticalFlow' or its derivative)");
    cv::DualTVL1OpticalFlow* _self_ = dynamic_cast<cv::DualTVL1OpticalFlow*>(((pyopencv_DualTVL1OpticalFlow_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTheta());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_DualTVL1OpticalFlow_getUseInitialFlow(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DualTVL1OpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DualTVL1OpticalFlow' or its derivative)");
    cv::DualTVL1OpticalFlow* _self_ = dynamic_cast<cv::DualTVL1OpticalFlow*>(((pyopencv_DualTVL1OpticalFlow_t*)self)->v.get());
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getUseInitialFlow());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_DualTVL1OpticalFlow_getWarpingsNumber(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DualTVL1OpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DualTVL1OpticalFlow' or its derivative)");
    cv::DualTVL1OpticalFlow* _self_ = dynamic_cast<cv::DualTVL1OpticalFlow*>(((pyopencv_DualTVL1OpticalFlow_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getWarpingsNumber());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_DualTVL1OpticalFlow_setEpsilon(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DualTVL1OpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DualTVL1OpticalFlow' or its derivative)");
    cv::DualTVL1OpticalFlow* _self_ = dynamic_cast<cv::DualTVL1OpticalFlow*>(((pyopencv_DualTVL1OpticalFlow_t*)self)->v.get());
    double val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:DualTVL1OpticalFlow.setEpsilon", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setEpsilon(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_DualTVL1OpticalFlow_setGamma(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DualTVL1OpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DualTVL1OpticalFlow' or its derivative)");
    cv::DualTVL1OpticalFlow* _self_ = dynamic_cast<cv::DualTVL1OpticalFlow*>(((pyopencv_DualTVL1OpticalFlow_t*)self)->v.get());
    double val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:DualTVL1OpticalFlow.setGamma", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setGamma(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_DualTVL1OpticalFlow_setInnerIterations(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DualTVL1OpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DualTVL1OpticalFlow' or its derivative)");
    cv::DualTVL1OpticalFlow* _self_ = dynamic_cast<cv::DualTVL1OpticalFlow*>(((pyopencv_DualTVL1OpticalFlow_t*)self)->v.get());
    int val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:DualTVL1OpticalFlow.setInnerIterations", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setInnerIterations(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_DualTVL1OpticalFlow_setLambda(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DualTVL1OpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DualTVL1OpticalFlow' or its derivative)");
    cv::DualTVL1OpticalFlow* _self_ = dynamic_cast<cv::DualTVL1OpticalFlow*>(((pyopencv_DualTVL1OpticalFlow_t*)self)->v.get());
    double val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:DualTVL1OpticalFlow.setLambda", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setLambda(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_DualTVL1OpticalFlow_setMedianFiltering(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DualTVL1OpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DualTVL1OpticalFlow' or its derivative)");
    cv::DualTVL1OpticalFlow* _self_ = dynamic_cast<cv::DualTVL1OpticalFlow*>(((pyopencv_DualTVL1OpticalFlow_t*)self)->v.get());
    int val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:DualTVL1OpticalFlow.setMedianFiltering", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setMedianFiltering(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_DualTVL1OpticalFlow_setOuterIterations(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DualTVL1OpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DualTVL1OpticalFlow' or its derivative)");
    cv::DualTVL1OpticalFlow* _self_ = dynamic_cast<cv::DualTVL1OpticalFlow*>(((pyopencv_DualTVL1OpticalFlow_t*)self)->v.get());
    int val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:DualTVL1OpticalFlow.setOuterIterations", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setOuterIterations(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_DualTVL1OpticalFlow_setScaleStep(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DualTVL1OpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DualTVL1OpticalFlow' or its derivative)");
    cv::DualTVL1OpticalFlow* _self_ = dynamic_cast<cv::DualTVL1OpticalFlow*>(((pyopencv_DualTVL1OpticalFlow_t*)self)->v.get());
    double val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:DualTVL1OpticalFlow.setScaleStep", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setScaleStep(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_DualTVL1OpticalFlow_setScalesNumber(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DualTVL1OpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DualTVL1OpticalFlow' or its derivative)");
    cv::DualTVL1OpticalFlow* _self_ = dynamic_cast<cv::DualTVL1OpticalFlow*>(((pyopencv_DualTVL1OpticalFlow_t*)self)->v.get());
    int val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:DualTVL1OpticalFlow.setScalesNumber", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setScalesNumber(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_DualTVL1OpticalFlow_setTau(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DualTVL1OpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DualTVL1OpticalFlow' or its derivative)");
    cv::DualTVL1OpticalFlow* _self_ = dynamic_cast<cv::DualTVL1OpticalFlow*>(((pyopencv_DualTVL1OpticalFlow_t*)self)->v.get());
    double val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:DualTVL1OpticalFlow.setTau", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setTau(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_DualTVL1OpticalFlow_setTheta(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DualTVL1OpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DualTVL1OpticalFlow' or its derivative)");
    cv::DualTVL1OpticalFlow* _self_ = dynamic_cast<cv::DualTVL1OpticalFlow*>(((pyopencv_DualTVL1OpticalFlow_t*)self)->v.get());
    double val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:DualTVL1OpticalFlow.setTheta", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setTheta(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_DualTVL1OpticalFlow_setUseInitialFlow(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DualTVL1OpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DualTVL1OpticalFlow' or its derivative)");
    cv::DualTVL1OpticalFlow* _self_ = dynamic_cast<cv::DualTVL1OpticalFlow*>(((pyopencv_DualTVL1OpticalFlow_t*)self)->v.get());
    bool val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:DualTVL1OpticalFlow.setUseInitialFlow", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setUseInitialFlow(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_DualTVL1OpticalFlow_setWarpingsNumber(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DualTVL1OpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'DualTVL1OpticalFlow' or its derivative)");
    cv::DualTVL1OpticalFlow* _self_ = dynamic_cast<cv::DualTVL1OpticalFlow*>(((pyopencv_DualTVL1OpticalFlow_t*)self)->v.get());
    int val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:DualTVL1OpticalFlow.setWarpingsNumber", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setWarpingsNumber(val));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_DualTVL1OpticalFlow_methods[] =
{
    {"getEpsilon", (PyCFunction)pyopencv_cv_DualTVL1OpticalFlow_getEpsilon, METH_VARARGS | METH_KEYWORDS, "getEpsilon() -> retval"},
    {"getGamma", (PyCFunction)pyopencv_cv_DualTVL1OpticalFlow_getGamma, METH_VARARGS | METH_KEYWORDS, "getGamma() -> retval"},
    {"getInnerIterations", (PyCFunction)pyopencv_cv_DualTVL1OpticalFlow_getInnerIterations, METH_VARARGS | METH_KEYWORDS, "getInnerIterations() -> retval"},
    {"getLambda", (PyCFunction)pyopencv_cv_DualTVL1OpticalFlow_getLambda, METH_VARARGS | METH_KEYWORDS, "getLambda() -> retval"},
    {"getMedianFiltering", (PyCFunction)pyopencv_cv_DualTVL1OpticalFlow_getMedianFiltering, METH_VARARGS | METH_KEYWORDS, "getMedianFiltering() -> retval"},
    {"getOuterIterations", (PyCFunction)pyopencv_cv_DualTVL1OpticalFlow_getOuterIterations, METH_VARARGS | METH_KEYWORDS, "getOuterIterations() -> retval"},
    {"getScaleStep", (PyCFunction)pyopencv_cv_DualTVL1OpticalFlow_getScaleStep, METH_VARARGS | METH_KEYWORDS, "getScaleStep() -> retval"},
    {"getScalesNumber", (PyCFunction)pyopencv_cv_DualTVL1OpticalFlow_getScalesNumber, METH_VARARGS | METH_KEYWORDS, "getScalesNumber() -> retval"},
    {"getTau", (PyCFunction)pyopencv_cv_DualTVL1OpticalFlow_getTau, METH_VARARGS | METH_KEYWORDS, "getTau() -> retval"},
    {"getTheta", (PyCFunction)pyopencv_cv_DualTVL1OpticalFlow_getTheta, METH_VARARGS | METH_KEYWORDS, "getTheta() -> retval"},
    {"getUseInitialFlow", (PyCFunction)pyopencv_cv_DualTVL1OpticalFlow_getUseInitialFlow, METH_VARARGS | METH_KEYWORDS, "getUseInitialFlow() -> retval"},
    {"getWarpingsNumber", (PyCFunction)pyopencv_cv_DualTVL1OpticalFlow_getWarpingsNumber, METH_VARARGS | METH_KEYWORDS, "getWarpingsNumber() -> retval"},
    {"setEpsilon", (PyCFunction)pyopencv_cv_DualTVL1OpticalFlow_setEpsilon, METH_VARARGS | METH_KEYWORDS, "setEpsilon(val) -> None"},
    {"setGamma", (PyCFunction)pyopencv_cv_DualTVL1OpticalFlow_setGamma, METH_VARARGS | METH_KEYWORDS, "setGamma(val) -> None"},
    {"setInnerIterations", (PyCFunction)pyopencv_cv_DualTVL1OpticalFlow_setInnerIterations, METH_VARARGS | METH_KEYWORDS, "setInnerIterations(val) -> None"},
    {"setLambda", (PyCFunction)pyopencv_cv_DualTVL1OpticalFlow_setLambda, METH_VARARGS | METH_KEYWORDS, "setLambda(val) -> None"},
    {"setMedianFiltering", (PyCFunction)pyopencv_cv_DualTVL1OpticalFlow_setMedianFiltering, METH_VARARGS | METH_KEYWORDS, "setMedianFiltering(val) -> None"},
    {"setOuterIterations", (PyCFunction)pyopencv_cv_DualTVL1OpticalFlow_setOuterIterations, METH_VARARGS | METH_KEYWORDS, "setOuterIterations(val) -> None"},
    {"setScaleStep", (PyCFunction)pyopencv_cv_DualTVL1OpticalFlow_setScaleStep, METH_VARARGS | METH_KEYWORDS, "setScaleStep(val) -> None"},
    {"setScalesNumber", (PyCFunction)pyopencv_cv_DualTVL1OpticalFlow_setScalesNumber, METH_VARARGS | METH_KEYWORDS, "setScalesNumber(val) -> None"},
    {"setTau", (PyCFunction)pyopencv_cv_DualTVL1OpticalFlow_setTau, METH_VARARGS | METH_KEYWORDS, "setTau(val) -> None"},
    {"setTheta", (PyCFunction)pyopencv_cv_DualTVL1OpticalFlow_setTheta, METH_VARARGS | METH_KEYWORDS, "setTheta(val) -> None"},
    {"setUseInitialFlow", (PyCFunction)pyopencv_cv_DualTVL1OpticalFlow_setUseInitialFlow, METH_VARARGS | METH_KEYWORDS, "setUseInitialFlow(val) -> None"},
    {"setWarpingsNumber", (PyCFunction)pyopencv_cv_DualTVL1OpticalFlow_setWarpingsNumber, METH_VARARGS | METH_KEYWORDS, "setWarpingsNumber(val) -> None"},

    {NULL,          NULL}
};

static void pyopencv_DualTVL1OpticalFlow_specials(void)
{
    pyopencv_DualTVL1OpticalFlow_Type.tp_base = &pyopencv_DenseOpticalFlow_Type;
    pyopencv_DualTVL1OpticalFlow_Type.tp_dealloc = pyopencv_DualTVL1OpticalFlow_dealloc;
    pyopencv_DualTVL1OpticalFlow_Type.tp_repr = pyopencv_DualTVL1OpticalFlow_repr;
    pyopencv_DualTVL1OpticalFlow_Type.tp_getset = pyopencv_DualTVL1OpticalFlow_getseters;
    pyopencv_DualTVL1OpticalFlow_Type.tp_methods = pyopencv_DualTVL1OpticalFlow_methods;
}

static PyObject* pyopencv_FarnebackOpticalFlow_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<FarnebackOpticalFlow %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_FarnebackOpticalFlow_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_FarnebackOpticalFlow_getFastPyramids(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FarnebackOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'FarnebackOpticalFlow' or its derivative)");
    cv::FarnebackOpticalFlow* _self_ = dynamic_cast<cv::FarnebackOpticalFlow*>(((pyopencv_FarnebackOpticalFlow_t*)self)->v.get());
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getFastPyramids());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FarnebackOpticalFlow_getFlags(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FarnebackOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'FarnebackOpticalFlow' or its derivative)");
    cv::FarnebackOpticalFlow* _self_ = dynamic_cast<cv::FarnebackOpticalFlow*>(((pyopencv_FarnebackOpticalFlow_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getFlags());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FarnebackOpticalFlow_getNumIters(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FarnebackOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'FarnebackOpticalFlow' or its derivative)");
    cv::FarnebackOpticalFlow* _self_ = dynamic_cast<cv::FarnebackOpticalFlow*>(((pyopencv_FarnebackOpticalFlow_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getNumIters());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FarnebackOpticalFlow_getNumLevels(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FarnebackOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'FarnebackOpticalFlow' or its derivative)");
    cv::FarnebackOpticalFlow* _self_ = dynamic_cast<cv::FarnebackOpticalFlow*>(((pyopencv_FarnebackOpticalFlow_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getNumLevels());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FarnebackOpticalFlow_getPolyN(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FarnebackOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'FarnebackOpticalFlow' or its derivative)");
    cv::FarnebackOpticalFlow* _self_ = dynamic_cast<cv::FarnebackOpticalFlow*>(((pyopencv_FarnebackOpticalFlow_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getPolyN());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FarnebackOpticalFlow_getPolySigma(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FarnebackOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'FarnebackOpticalFlow' or its derivative)");
    cv::FarnebackOpticalFlow* _self_ = dynamic_cast<cv::FarnebackOpticalFlow*>(((pyopencv_FarnebackOpticalFlow_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getPolySigma());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FarnebackOpticalFlow_getPyrScale(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FarnebackOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'FarnebackOpticalFlow' or its derivative)");
    cv::FarnebackOpticalFlow* _self_ = dynamic_cast<cv::FarnebackOpticalFlow*>(((pyopencv_FarnebackOpticalFlow_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getPyrScale());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FarnebackOpticalFlow_getWinSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FarnebackOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'FarnebackOpticalFlow' or its derivative)");
    cv::FarnebackOpticalFlow* _self_ = dynamic_cast<cv::FarnebackOpticalFlow*>(((pyopencv_FarnebackOpticalFlow_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getWinSize());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FarnebackOpticalFlow_setFastPyramids(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FarnebackOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'FarnebackOpticalFlow' or its derivative)");
    cv::FarnebackOpticalFlow* _self_ = dynamic_cast<cv::FarnebackOpticalFlow*>(((pyopencv_FarnebackOpticalFlow_t*)self)->v.get());
    bool fastPyramids=0;

    const char* keywords[] = { "fastPyramids", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:FarnebackOpticalFlow.setFastPyramids", (char**)keywords, &fastPyramids) )
    {
        ERRWRAP2(_self_->setFastPyramids(fastPyramids));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_FarnebackOpticalFlow_setFlags(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FarnebackOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'FarnebackOpticalFlow' or its derivative)");
    cv::FarnebackOpticalFlow* _self_ = dynamic_cast<cv::FarnebackOpticalFlow*>(((pyopencv_FarnebackOpticalFlow_t*)self)->v.get());
    int flags=0;

    const char* keywords[] = { "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:FarnebackOpticalFlow.setFlags", (char**)keywords, &flags) )
    {
        ERRWRAP2(_self_->setFlags(flags));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_FarnebackOpticalFlow_setNumIters(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FarnebackOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'FarnebackOpticalFlow' or its derivative)");
    cv::FarnebackOpticalFlow* _self_ = dynamic_cast<cv::FarnebackOpticalFlow*>(((pyopencv_FarnebackOpticalFlow_t*)self)->v.get());
    int numIters=0;

    const char* keywords[] = { "numIters", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:FarnebackOpticalFlow.setNumIters", (char**)keywords, &numIters) )
    {
        ERRWRAP2(_self_->setNumIters(numIters));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_FarnebackOpticalFlow_setNumLevels(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FarnebackOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'FarnebackOpticalFlow' or its derivative)");
    cv::FarnebackOpticalFlow* _self_ = dynamic_cast<cv::FarnebackOpticalFlow*>(((pyopencv_FarnebackOpticalFlow_t*)self)->v.get());
    int numLevels=0;

    const char* keywords[] = { "numLevels", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:FarnebackOpticalFlow.setNumLevels", (char**)keywords, &numLevels) )
    {
        ERRWRAP2(_self_->setNumLevels(numLevels));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_FarnebackOpticalFlow_setPolyN(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FarnebackOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'FarnebackOpticalFlow' or its derivative)");
    cv::FarnebackOpticalFlow* _self_ = dynamic_cast<cv::FarnebackOpticalFlow*>(((pyopencv_FarnebackOpticalFlow_t*)self)->v.get());
    int polyN=0;

    const char* keywords[] = { "polyN", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:FarnebackOpticalFlow.setPolyN", (char**)keywords, &polyN) )
    {
        ERRWRAP2(_self_->setPolyN(polyN));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_FarnebackOpticalFlow_setPolySigma(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FarnebackOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'FarnebackOpticalFlow' or its derivative)");
    cv::FarnebackOpticalFlow* _self_ = dynamic_cast<cv::FarnebackOpticalFlow*>(((pyopencv_FarnebackOpticalFlow_t*)self)->v.get());
    double polySigma=0;

    const char* keywords[] = { "polySigma", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:FarnebackOpticalFlow.setPolySigma", (char**)keywords, &polySigma) )
    {
        ERRWRAP2(_self_->setPolySigma(polySigma));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_FarnebackOpticalFlow_setPyrScale(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FarnebackOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'FarnebackOpticalFlow' or its derivative)");
    cv::FarnebackOpticalFlow* _self_ = dynamic_cast<cv::FarnebackOpticalFlow*>(((pyopencv_FarnebackOpticalFlow_t*)self)->v.get());
    double pyrScale=0;

    const char* keywords[] = { "pyrScale", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:FarnebackOpticalFlow.setPyrScale", (char**)keywords, &pyrScale) )
    {
        ERRWRAP2(_self_->setPyrScale(pyrScale));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_FarnebackOpticalFlow_setWinSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FarnebackOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'FarnebackOpticalFlow' or its derivative)");
    cv::FarnebackOpticalFlow* _self_ = dynamic_cast<cv::FarnebackOpticalFlow*>(((pyopencv_FarnebackOpticalFlow_t*)self)->v.get());
    int winSize=0;

    const char* keywords[] = { "winSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:FarnebackOpticalFlow.setWinSize", (char**)keywords, &winSize) )
    {
        ERRWRAP2(_self_->setWinSize(winSize));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_FarnebackOpticalFlow_methods[] =
{
    {"getFastPyramids", (PyCFunction)pyopencv_cv_FarnebackOpticalFlow_getFastPyramids, METH_VARARGS | METH_KEYWORDS, "getFastPyramids() -> retval"},
    {"getFlags", (PyCFunction)pyopencv_cv_FarnebackOpticalFlow_getFlags, METH_VARARGS | METH_KEYWORDS, "getFlags() -> retval"},
    {"getNumIters", (PyCFunction)pyopencv_cv_FarnebackOpticalFlow_getNumIters, METH_VARARGS | METH_KEYWORDS, "getNumIters() -> retval"},
    {"getNumLevels", (PyCFunction)pyopencv_cv_FarnebackOpticalFlow_getNumLevels, METH_VARARGS | METH_KEYWORDS, "getNumLevels() -> retval"},
    {"getPolyN", (PyCFunction)pyopencv_cv_FarnebackOpticalFlow_getPolyN, METH_VARARGS | METH_KEYWORDS, "getPolyN() -> retval"},
    {"getPolySigma", (PyCFunction)pyopencv_cv_FarnebackOpticalFlow_getPolySigma, METH_VARARGS | METH_KEYWORDS, "getPolySigma() -> retval"},
    {"getPyrScale", (PyCFunction)pyopencv_cv_FarnebackOpticalFlow_getPyrScale, METH_VARARGS | METH_KEYWORDS, "getPyrScale() -> retval"},
    {"getWinSize", (PyCFunction)pyopencv_cv_FarnebackOpticalFlow_getWinSize, METH_VARARGS | METH_KEYWORDS, "getWinSize() -> retval"},
    {"setFastPyramids", (PyCFunction)pyopencv_cv_FarnebackOpticalFlow_setFastPyramids, METH_VARARGS | METH_KEYWORDS, "setFastPyramids(fastPyramids) -> None"},
    {"setFlags", (PyCFunction)pyopencv_cv_FarnebackOpticalFlow_setFlags, METH_VARARGS | METH_KEYWORDS, "setFlags(flags) -> None"},
    {"setNumIters", (PyCFunction)pyopencv_cv_FarnebackOpticalFlow_setNumIters, METH_VARARGS | METH_KEYWORDS, "setNumIters(numIters) -> None"},
    {"setNumLevels", (PyCFunction)pyopencv_cv_FarnebackOpticalFlow_setNumLevels, METH_VARARGS | METH_KEYWORDS, "setNumLevels(numLevels) -> None"},
    {"setPolyN", (PyCFunction)pyopencv_cv_FarnebackOpticalFlow_setPolyN, METH_VARARGS | METH_KEYWORDS, "setPolyN(polyN) -> None"},
    {"setPolySigma", (PyCFunction)pyopencv_cv_FarnebackOpticalFlow_setPolySigma, METH_VARARGS | METH_KEYWORDS, "setPolySigma(polySigma) -> None"},
    {"setPyrScale", (PyCFunction)pyopencv_cv_FarnebackOpticalFlow_setPyrScale, METH_VARARGS | METH_KEYWORDS, "setPyrScale(pyrScale) -> None"},
    {"setWinSize", (PyCFunction)pyopencv_cv_FarnebackOpticalFlow_setWinSize, METH_VARARGS | METH_KEYWORDS, "setWinSize(winSize) -> None"},

    {NULL,          NULL}
};

static void pyopencv_FarnebackOpticalFlow_specials(void)
{
    pyopencv_FarnebackOpticalFlow_Type.tp_base = &pyopencv_DenseOpticalFlow_Type;
    pyopencv_FarnebackOpticalFlow_Type.tp_dealloc = pyopencv_FarnebackOpticalFlow_dealloc;
    pyopencv_FarnebackOpticalFlow_Type.tp_repr = pyopencv_FarnebackOpticalFlow_repr;
    pyopencv_FarnebackOpticalFlow_Type.tp_getset = pyopencv_FarnebackOpticalFlow_getseters;
    pyopencv_FarnebackOpticalFlow_Type.tp_methods = pyopencv_FarnebackOpticalFlow_methods;
}

static PyObject* pyopencv_SparsePyrLKOpticalFlow_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<SparsePyrLKOpticalFlow %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_SparsePyrLKOpticalFlow_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_SparsePyrLKOpticalFlow_getFlags(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_SparsePyrLKOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'SparsePyrLKOpticalFlow' or its derivative)");
    cv::SparsePyrLKOpticalFlow* _self_ = dynamic_cast<cv::SparsePyrLKOpticalFlow*>(((pyopencv_SparsePyrLKOpticalFlow_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getFlags());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_SparsePyrLKOpticalFlow_getMaxLevel(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_SparsePyrLKOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'SparsePyrLKOpticalFlow' or its derivative)");
    cv::SparsePyrLKOpticalFlow* _self_ = dynamic_cast<cv::SparsePyrLKOpticalFlow*>(((pyopencv_SparsePyrLKOpticalFlow_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getMaxLevel());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_SparsePyrLKOpticalFlow_getMinEigThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_SparsePyrLKOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'SparsePyrLKOpticalFlow' or its derivative)");
    cv::SparsePyrLKOpticalFlow* _self_ = dynamic_cast<cv::SparsePyrLKOpticalFlow*>(((pyopencv_SparsePyrLKOpticalFlow_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getMinEigThreshold());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_SparsePyrLKOpticalFlow_getTermCriteria(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_SparsePyrLKOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'SparsePyrLKOpticalFlow' or its derivative)");
    cv::SparsePyrLKOpticalFlow* _self_ = dynamic_cast<cv::SparsePyrLKOpticalFlow*>(((pyopencv_SparsePyrLKOpticalFlow_t*)self)->v.get());
    TermCriteria retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTermCriteria());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_SparsePyrLKOpticalFlow_getWinSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_SparsePyrLKOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'SparsePyrLKOpticalFlow' or its derivative)");
    cv::SparsePyrLKOpticalFlow* _self_ = dynamic_cast<cv::SparsePyrLKOpticalFlow*>(((pyopencv_SparsePyrLKOpticalFlow_t*)self)->v.get());
    Size retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getWinSize());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_SparsePyrLKOpticalFlow_setFlags(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_SparsePyrLKOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'SparsePyrLKOpticalFlow' or its derivative)");
    cv::SparsePyrLKOpticalFlow* _self_ = dynamic_cast<cv::SparsePyrLKOpticalFlow*>(((pyopencv_SparsePyrLKOpticalFlow_t*)self)->v.get());
    int flags=0;

    const char* keywords[] = { "flags", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:SparsePyrLKOpticalFlow.setFlags", (char**)keywords, &flags) )
    {
        ERRWRAP2(_self_->setFlags(flags));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_SparsePyrLKOpticalFlow_setMaxLevel(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_SparsePyrLKOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'SparsePyrLKOpticalFlow' or its derivative)");
    cv::SparsePyrLKOpticalFlow* _self_ = dynamic_cast<cv::SparsePyrLKOpticalFlow*>(((pyopencv_SparsePyrLKOpticalFlow_t*)self)->v.get());
    int maxLevel=0;

    const char* keywords[] = { "maxLevel", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:SparsePyrLKOpticalFlow.setMaxLevel", (char**)keywords, &maxLevel) )
    {
        ERRWRAP2(_self_->setMaxLevel(maxLevel));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_SparsePyrLKOpticalFlow_setMinEigThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_SparsePyrLKOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'SparsePyrLKOpticalFlow' or its derivative)");
    cv::SparsePyrLKOpticalFlow* _self_ = dynamic_cast<cv::SparsePyrLKOpticalFlow*>(((pyopencv_SparsePyrLKOpticalFlow_t*)self)->v.get());
    double minEigThreshold=0;

    const char* keywords[] = { "minEigThreshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:SparsePyrLKOpticalFlow.setMinEigThreshold", (char**)keywords, &minEigThreshold) )
    {
        ERRWRAP2(_self_->setMinEigThreshold(minEigThreshold));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_SparsePyrLKOpticalFlow_setTermCriteria(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_SparsePyrLKOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'SparsePyrLKOpticalFlow' or its derivative)");
    cv::SparsePyrLKOpticalFlow* _self_ = dynamic_cast<cv::SparsePyrLKOpticalFlow*>(((pyopencv_SparsePyrLKOpticalFlow_t*)self)->v.get());
    PyObject* pyobj_crit = NULL;
    TermCriteria crit;

    const char* keywords[] = { "crit", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:SparsePyrLKOpticalFlow.setTermCriteria", (char**)keywords, &pyobj_crit) &&
        pyopencv_to(pyobj_crit, crit, ArgInfo("crit", 0)) )
    {
        ERRWRAP2(_self_->setTermCriteria(crit));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_SparsePyrLKOpticalFlow_setWinSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_SparsePyrLKOpticalFlow_Type))
        return failmsgp("Incorrect type of self (must be 'SparsePyrLKOpticalFlow' or its derivative)");
    cv::SparsePyrLKOpticalFlow* _self_ = dynamic_cast<cv::SparsePyrLKOpticalFlow*>(((pyopencv_SparsePyrLKOpticalFlow_t*)self)->v.get());
    PyObject* pyobj_winSize = NULL;
    Size winSize;

    const char* keywords[] = { "winSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:SparsePyrLKOpticalFlow.setWinSize", (char**)keywords, &pyobj_winSize) &&
        pyopencv_to(pyobj_winSize, winSize, ArgInfo("winSize", 0)) )
    {
        ERRWRAP2(_self_->setWinSize(winSize));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_SparsePyrLKOpticalFlow_methods[] =
{
    {"getFlags", (PyCFunction)pyopencv_cv_SparsePyrLKOpticalFlow_getFlags, METH_VARARGS | METH_KEYWORDS, "getFlags() -> retval"},
    {"getMaxLevel", (PyCFunction)pyopencv_cv_SparsePyrLKOpticalFlow_getMaxLevel, METH_VARARGS | METH_KEYWORDS, "getMaxLevel() -> retval"},
    {"getMinEigThreshold", (PyCFunction)pyopencv_cv_SparsePyrLKOpticalFlow_getMinEigThreshold, METH_VARARGS | METH_KEYWORDS, "getMinEigThreshold() -> retval"},
    {"getTermCriteria", (PyCFunction)pyopencv_cv_SparsePyrLKOpticalFlow_getTermCriteria, METH_VARARGS | METH_KEYWORDS, "getTermCriteria() -> retval"},
    {"getWinSize", (PyCFunction)pyopencv_cv_SparsePyrLKOpticalFlow_getWinSize, METH_VARARGS | METH_KEYWORDS, "getWinSize() -> retval"},
    {"setFlags", (PyCFunction)pyopencv_cv_SparsePyrLKOpticalFlow_setFlags, METH_VARARGS | METH_KEYWORDS, "setFlags(flags) -> None"},
    {"setMaxLevel", (PyCFunction)pyopencv_cv_SparsePyrLKOpticalFlow_setMaxLevel, METH_VARARGS | METH_KEYWORDS, "setMaxLevel(maxLevel) -> None"},
    {"setMinEigThreshold", (PyCFunction)pyopencv_cv_SparsePyrLKOpticalFlow_setMinEigThreshold, METH_VARARGS | METH_KEYWORDS, "setMinEigThreshold(minEigThreshold) -> None"},
    {"setTermCriteria", (PyCFunction)pyopencv_cv_SparsePyrLKOpticalFlow_setTermCriteria, METH_VARARGS | METH_KEYWORDS, "setTermCriteria(crit) -> None"},
    {"setWinSize", (PyCFunction)pyopencv_cv_SparsePyrLKOpticalFlow_setWinSize, METH_VARARGS | METH_KEYWORDS, "setWinSize(winSize) -> None"},

    {NULL,          NULL}
};

static void pyopencv_SparsePyrLKOpticalFlow_specials(void)
{
    pyopencv_SparsePyrLKOpticalFlow_Type.tp_base = &pyopencv_SparseOpticalFlow_Type;
    pyopencv_SparsePyrLKOpticalFlow_Type.tp_dealloc = pyopencv_SparsePyrLKOpticalFlow_dealloc;
    pyopencv_SparsePyrLKOpticalFlow_Type.tp_repr = pyopencv_SparsePyrLKOpticalFlow_repr;
    pyopencv_SparsePyrLKOpticalFlow_Type.tp_getset = pyopencv_SparsePyrLKOpticalFlow_getseters;
    pyopencv_SparsePyrLKOpticalFlow_Type.tp_methods = pyopencv_SparsePyrLKOpticalFlow_methods;
}

static PyObject* pyopencv_BackgroundSubtractor_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<BackgroundSubtractor %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_BackgroundSubtractor_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_BackgroundSubtractor_apply(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractor_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractor' or its derivative)");
    cv::BackgroundSubtractor* _self_ = dynamic_cast<cv::BackgroundSubtractor*>(((pyopencv_BackgroundSubtractor_t*)self)->v.get());
    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_fgmask = NULL;
    Mat fgmask;
    double learningRate=-1;

    const char* keywords[] = { "image", "fgmask", "learningRate", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Od:BackgroundSubtractor.apply", (char**)keywords, &pyobj_image, &pyobj_fgmask, &learningRate) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_fgmask, fgmask, ArgInfo("fgmask", 1)) )
    {
        ERRWRAP2(_self_->apply(image, fgmask, learningRate));
        return pyopencv_from(fgmask);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_fgmask = NULL;
    UMat fgmask;
    double learningRate=-1;

    const char* keywords[] = { "image", "fgmask", "learningRate", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|Od:BackgroundSubtractor.apply", (char**)keywords, &pyobj_image, &pyobj_fgmask, &learningRate) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_fgmask, fgmask, ArgInfo("fgmask", 1)) )
    {
        ERRWRAP2(_self_->apply(image, fgmask, learningRate));
        return pyopencv_from(fgmask);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractor_getBackgroundImage(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractor_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractor' or its derivative)");
    cv::BackgroundSubtractor* _self_ = dynamic_cast<cv::BackgroundSubtractor*>(((pyopencv_BackgroundSubtractor_t*)self)->v.get());
    {
    PyObject* pyobj_backgroundImage = NULL;
    Mat backgroundImage;

    const char* keywords[] = { "backgroundImage", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|O:BackgroundSubtractor.getBackgroundImage", (char**)keywords, &pyobj_backgroundImage) &&
        pyopencv_to(pyobj_backgroundImage, backgroundImage, ArgInfo("backgroundImage", 1)) )
    {
        ERRWRAP2(_self_->getBackgroundImage(backgroundImage));
        return pyopencv_from(backgroundImage);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_backgroundImage = NULL;
    UMat backgroundImage;

    const char* keywords[] = { "backgroundImage", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|O:BackgroundSubtractor.getBackgroundImage", (char**)keywords, &pyobj_backgroundImage) &&
        pyopencv_to(pyobj_backgroundImage, backgroundImage, ArgInfo("backgroundImage", 1)) )
    {
        ERRWRAP2(_self_->getBackgroundImage(backgroundImage));
        return pyopencv_from(backgroundImage);
    }
    }

    return NULL;
}



static PyMethodDef pyopencv_BackgroundSubtractor_methods[] =
{
    {"apply", (PyCFunction)pyopencv_cv_BackgroundSubtractor_apply, METH_VARARGS | METH_KEYWORDS, "apply(image[, fgmask[, learningRate]]) -> fgmask"},
    {"getBackgroundImage", (PyCFunction)pyopencv_cv_BackgroundSubtractor_getBackgroundImage, METH_VARARGS | METH_KEYWORDS, "getBackgroundImage([, backgroundImage]) -> backgroundImage"},

    {NULL,          NULL}
};

static void pyopencv_BackgroundSubtractor_specials(void)
{
    pyopencv_BackgroundSubtractor_Type.tp_base = &pyopencv_Algorithm_Type;
    pyopencv_BackgroundSubtractor_Type.tp_dealloc = pyopencv_BackgroundSubtractor_dealloc;
    pyopencv_BackgroundSubtractor_Type.tp_repr = pyopencv_BackgroundSubtractor_repr;
    pyopencv_BackgroundSubtractor_Type.tp_getset = pyopencv_BackgroundSubtractor_getseters;
    pyopencv_BackgroundSubtractor_Type.tp_methods = pyopencv_BackgroundSubtractor_methods;
}

static PyObject* pyopencv_BackgroundSubtractorMOG2_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<BackgroundSubtractorMOG2 %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_BackgroundSubtractorMOG2_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_BackgroundSubtractorMOG2_getBackgroundRatio(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorMOG2_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorMOG2' or its derivative)");
    cv::BackgroundSubtractorMOG2* _self_ = dynamic_cast<cv::BackgroundSubtractorMOG2*>(((pyopencv_BackgroundSubtractorMOG2_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getBackgroundRatio());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorMOG2_getComplexityReductionThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorMOG2_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorMOG2' or its derivative)");
    cv::BackgroundSubtractorMOG2* _self_ = dynamic_cast<cv::BackgroundSubtractorMOG2*>(((pyopencv_BackgroundSubtractorMOG2_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getComplexityReductionThreshold());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorMOG2_getDetectShadows(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorMOG2_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorMOG2' or its derivative)");
    cv::BackgroundSubtractorMOG2* _self_ = dynamic_cast<cv::BackgroundSubtractorMOG2*>(((pyopencv_BackgroundSubtractorMOG2_t*)self)->v.get());
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getDetectShadows());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorMOG2_getHistory(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorMOG2_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorMOG2' or its derivative)");
    cv::BackgroundSubtractorMOG2* _self_ = dynamic_cast<cv::BackgroundSubtractorMOG2*>(((pyopencv_BackgroundSubtractorMOG2_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getHistory());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorMOG2_getNMixtures(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorMOG2_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorMOG2' or its derivative)");
    cv::BackgroundSubtractorMOG2* _self_ = dynamic_cast<cv::BackgroundSubtractorMOG2*>(((pyopencv_BackgroundSubtractorMOG2_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getNMixtures());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorMOG2_getShadowThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorMOG2_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorMOG2' or its derivative)");
    cv::BackgroundSubtractorMOG2* _self_ = dynamic_cast<cv::BackgroundSubtractorMOG2*>(((pyopencv_BackgroundSubtractorMOG2_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getShadowThreshold());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorMOG2_getShadowValue(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorMOG2_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorMOG2' or its derivative)");
    cv::BackgroundSubtractorMOG2* _self_ = dynamic_cast<cv::BackgroundSubtractorMOG2*>(((pyopencv_BackgroundSubtractorMOG2_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getShadowValue());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorMOG2_getVarInit(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorMOG2_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorMOG2' or its derivative)");
    cv::BackgroundSubtractorMOG2* _self_ = dynamic_cast<cv::BackgroundSubtractorMOG2*>(((pyopencv_BackgroundSubtractorMOG2_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getVarInit());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorMOG2_getVarMax(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorMOG2_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorMOG2' or its derivative)");
    cv::BackgroundSubtractorMOG2* _self_ = dynamic_cast<cv::BackgroundSubtractorMOG2*>(((pyopencv_BackgroundSubtractorMOG2_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getVarMax());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorMOG2_getVarMin(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorMOG2_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorMOG2' or its derivative)");
    cv::BackgroundSubtractorMOG2* _self_ = dynamic_cast<cv::BackgroundSubtractorMOG2*>(((pyopencv_BackgroundSubtractorMOG2_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getVarMin());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorMOG2_getVarThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorMOG2_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorMOG2' or its derivative)");
    cv::BackgroundSubtractorMOG2* _self_ = dynamic_cast<cv::BackgroundSubtractorMOG2*>(((pyopencv_BackgroundSubtractorMOG2_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getVarThreshold());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorMOG2_getVarThresholdGen(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorMOG2_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorMOG2' or its derivative)");
    cv::BackgroundSubtractorMOG2* _self_ = dynamic_cast<cv::BackgroundSubtractorMOG2*>(((pyopencv_BackgroundSubtractorMOG2_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getVarThresholdGen());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorMOG2_setBackgroundRatio(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorMOG2_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorMOG2' or its derivative)");
    cv::BackgroundSubtractorMOG2* _self_ = dynamic_cast<cv::BackgroundSubtractorMOG2*>(((pyopencv_BackgroundSubtractorMOG2_t*)self)->v.get());
    double ratio=0;

    const char* keywords[] = { "ratio", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:BackgroundSubtractorMOG2.setBackgroundRatio", (char**)keywords, &ratio) )
    {
        ERRWRAP2(_self_->setBackgroundRatio(ratio));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorMOG2_setComplexityReductionThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorMOG2_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorMOG2' or its derivative)");
    cv::BackgroundSubtractorMOG2* _self_ = dynamic_cast<cv::BackgroundSubtractorMOG2*>(((pyopencv_BackgroundSubtractorMOG2_t*)self)->v.get());
    double ct=0;

    const char* keywords[] = { "ct", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:BackgroundSubtractorMOG2.setComplexityReductionThreshold", (char**)keywords, &ct) )
    {
        ERRWRAP2(_self_->setComplexityReductionThreshold(ct));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorMOG2_setDetectShadows(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorMOG2_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorMOG2' or its derivative)");
    cv::BackgroundSubtractorMOG2* _self_ = dynamic_cast<cv::BackgroundSubtractorMOG2*>(((pyopencv_BackgroundSubtractorMOG2_t*)self)->v.get());
    bool detectShadows=0;

    const char* keywords[] = { "detectShadows", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:BackgroundSubtractorMOG2.setDetectShadows", (char**)keywords, &detectShadows) )
    {
        ERRWRAP2(_self_->setDetectShadows(detectShadows));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorMOG2_setHistory(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorMOG2_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorMOG2' or its derivative)");
    cv::BackgroundSubtractorMOG2* _self_ = dynamic_cast<cv::BackgroundSubtractorMOG2*>(((pyopencv_BackgroundSubtractorMOG2_t*)self)->v.get());
    int history=0;

    const char* keywords[] = { "history", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:BackgroundSubtractorMOG2.setHistory", (char**)keywords, &history) )
    {
        ERRWRAP2(_self_->setHistory(history));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorMOG2_setNMixtures(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorMOG2_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorMOG2' or its derivative)");
    cv::BackgroundSubtractorMOG2* _self_ = dynamic_cast<cv::BackgroundSubtractorMOG2*>(((pyopencv_BackgroundSubtractorMOG2_t*)self)->v.get());
    int nmixtures=0;

    const char* keywords[] = { "nmixtures", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:BackgroundSubtractorMOG2.setNMixtures", (char**)keywords, &nmixtures) )
    {
        ERRWRAP2(_self_->setNMixtures(nmixtures));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorMOG2_setShadowThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorMOG2_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorMOG2' or its derivative)");
    cv::BackgroundSubtractorMOG2* _self_ = dynamic_cast<cv::BackgroundSubtractorMOG2*>(((pyopencv_BackgroundSubtractorMOG2_t*)self)->v.get());
    double threshold=0;

    const char* keywords[] = { "threshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:BackgroundSubtractorMOG2.setShadowThreshold", (char**)keywords, &threshold) )
    {
        ERRWRAP2(_self_->setShadowThreshold(threshold));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorMOG2_setShadowValue(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorMOG2_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorMOG2' or its derivative)");
    cv::BackgroundSubtractorMOG2* _self_ = dynamic_cast<cv::BackgroundSubtractorMOG2*>(((pyopencv_BackgroundSubtractorMOG2_t*)self)->v.get());
    int value=0;

    const char* keywords[] = { "value", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:BackgroundSubtractorMOG2.setShadowValue", (char**)keywords, &value) )
    {
        ERRWRAP2(_self_->setShadowValue(value));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorMOG2_setVarInit(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorMOG2_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorMOG2' or its derivative)");
    cv::BackgroundSubtractorMOG2* _self_ = dynamic_cast<cv::BackgroundSubtractorMOG2*>(((pyopencv_BackgroundSubtractorMOG2_t*)self)->v.get());
    double varInit=0;

    const char* keywords[] = { "varInit", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:BackgroundSubtractorMOG2.setVarInit", (char**)keywords, &varInit) )
    {
        ERRWRAP2(_self_->setVarInit(varInit));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorMOG2_setVarMax(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorMOG2_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorMOG2' or its derivative)");
    cv::BackgroundSubtractorMOG2* _self_ = dynamic_cast<cv::BackgroundSubtractorMOG2*>(((pyopencv_BackgroundSubtractorMOG2_t*)self)->v.get());
    double varMax=0;

    const char* keywords[] = { "varMax", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:BackgroundSubtractorMOG2.setVarMax", (char**)keywords, &varMax) )
    {
        ERRWRAP2(_self_->setVarMax(varMax));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorMOG2_setVarMin(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorMOG2_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorMOG2' or its derivative)");
    cv::BackgroundSubtractorMOG2* _self_ = dynamic_cast<cv::BackgroundSubtractorMOG2*>(((pyopencv_BackgroundSubtractorMOG2_t*)self)->v.get());
    double varMin=0;

    const char* keywords[] = { "varMin", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:BackgroundSubtractorMOG2.setVarMin", (char**)keywords, &varMin) )
    {
        ERRWRAP2(_self_->setVarMin(varMin));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorMOG2_setVarThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorMOG2_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorMOG2' or its derivative)");
    cv::BackgroundSubtractorMOG2* _self_ = dynamic_cast<cv::BackgroundSubtractorMOG2*>(((pyopencv_BackgroundSubtractorMOG2_t*)self)->v.get());
    double varThreshold=0;

    const char* keywords[] = { "varThreshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:BackgroundSubtractorMOG2.setVarThreshold", (char**)keywords, &varThreshold) )
    {
        ERRWRAP2(_self_->setVarThreshold(varThreshold));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorMOG2_setVarThresholdGen(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorMOG2_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorMOG2' or its derivative)");
    cv::BackgroundSubtractorMOG2* _self_ = dynamic_cast<cv::BackgroundSubtractorMOG2*>(((pyopencv_BackgroundSubtractorMOG2_t*)self)->v.get());
    double varThresholdGen=0;

    const char* keywords[] = { "varThresholdGen", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:BackgroundSubtractorMOG2.setVarThresholdGen", (char**)keywords, &varThresholdGen) )
    {
        ERRWRAP2(_self_->setVarThresholdGen(varThresholdGen));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_BackgroundSubtractorMOG2_methods[] =
{
    {"getBackgroundRatio", (PyCFunction)pyopencv_cv_BackgroundSubtractorMOG2_getBackgroundRatio, METH_VARARGS | METH_KEYWORDS, "getBackgroundRatio() -> retval"},
    {"getComplexityReductionThreshold", (PyCFunction)pyopencv_cv_BackgroundSubtractorMOG2_getComplexityReductionThreshold, METH_VARARGS | METH_KEYWORDS, "getComplexityReductionThreshold() -> retval"},
    {"getDetectShadows", (PyCFunction)pyopencv_cv_BackgroundSubtractorMOG2_getDetectShadows, METH_VARARGS | METH_KEYWORDS, "getDetectShadows() -> retval"},
    {"getHistory", (PyCFunction)pyopencv_cv_BackgroundSubtractorMOG2_getHistory, METH_VARARGS | METH_KEYWORDS, "getHistory() -> retval"},
    {"getNMixtures", (PyCFunction)pyopencv_cv_BackgroundSubtractorMOG2_getNMixtures, METH_VARARGS | METH_KEYWORDS, "getNMixtures() -> retval"},
    {"getShadowThreshold", (PyCFunction)pyopencv_cv_BackgroundSubtractorMOG2_getShadowThreshold, METH_VARARGS | METH_KEYWORDS, "getShadowThreshold() -> retval"},
    {"getShadowValue", (PyCFunction)pyopencv_cv_BackgroundSubtractorMOG2_getShadowValue, METH_VARARGS | METH_KEYWORDS, "getShadowValue() -> retval"},
    {"getVarInit", (PyCFunction)pyopencv_cv_BackgroundSubtractorMOG2_getVarInit, METH_VARARGS | METH_KEYWORDS, "getVarInit() -> retval"},
    {"getVarMax", (PyCFunction)pyopencv_cv_BackgroundSubtractorMOG2_getVarMax, METH_VARARGS | METH_KEYWORDS, "getVarMax() -> retval"},
    {"getVarMin", (PyCFunction)pyopencv_cv_BackgroundSubtractorMOG2_getVarMin, METH_VARARGS | METH_KEYWORDS, "getVarMin() -> retval"},
    {"getVarThreshold", (PyCFunction)pyopencv_cv_BackgroundSubtractorMOG2_getVarThreshold, METH_VARARGS | METH_KEYWORDS, "getVarThreshold() -> retval"},
    {"getVarThresholdGen", (PyCFunction)pyopencv_cv_BackgroundSubtractorMOG2_getVarThresholdGen, METH_VARARGS | METH_KEYWORDS, "getVarThresholdGen() -> retval"},
    {"setBackgroundRatio", (PyCFunction)pyopencv_cv_BackgroundSubtractorMOG2_setBackgroundRatio, METH_VARARGS | METH_KEYWORDS, "setBackgroundRatio(ratio) -> None"},
    {"setComplexityReductionThreshold", (PyCFunction)pyopencv_cv_BackgroundSubtractorMOG2_setComplexityReductionThreshold, METH_VARARGS | METH_KEYWORDS, "setComplexityReductionThreshold(ct) -> None"},
    {"setDetectShadows", (PyCFunction)pyopencv_cv_BackgroundSubtractorMOG2_setDetectShadows, METH_VARARGS | METH_KEYWORDS, "setDetectShadows(detectShadows) -> None"},
    {"setHistory", (PyCFunction)pyopencv_cv_BackgroundSubtractorMOG2_setHistory, METH_VARARGS | METH_KEYWORDS, "setHistory(history) -> None"},
    {"setNMixtures", (PyCFunction)pyopencv_cv_BackgroundSubtractorMOG2_setNMixtures, METH_VARARGS | METH_KEYWORDS, "setNMixtures(nmixtures) -> None"},
    {"setShadowThreshold", (PyCFunction)pyopencv_cv_BackgroundSubtractorMOG2_setShadowThreshold, METH_VARARGS | METH_KEYWORDS, "setShadowThreshold(threshold) -> None"},
    {"setShadowValue", (PyCFunction)pyopencv_cv_BackgroundSubtractorMOG2_setShadowValue, METH_VARARGS | METH_KEYWORDS, "setShadowValue(value) -> None"},
    {"setVarInit", (PyCFunction)pyopencv_cv_BackgroundSubtractorMOG2_setVarInit, METH_VARARGS | METH_KEYWORDS, "setVarInit(varInit) -> None"},
    {"setVarMax", (PyCFunction)pyopencv_cv_BackgroundSubtractorMOG2_setVarMax, METH_VARARGS | METH_KEYWORDS, "setVarMax(varMax) -> None"},
    {"setVarMin", (PyCFunction)pyopencv_cv_BackgroundSubtractorMOG2_setVarMin, METH_VARARGS | METH_KEYWORDS, "setVarMin(varMin) -> None"},
    {"setVarThreshold", (PyCFunction)pyopencv_cv_BackgroundSubtractorMOG2_setVarThreshold, METH_VARARGS | METH_KEYWORDS, "setVarThreshold(varThreshold) -> None"},
    {"setVarThresholdGen", (PyCFunction)pyopencv_cv_BackgroundSubtractorMOG2_setVarThresholdGen, METH_VARARGS | METH_KEYWORDS, "setVarThresholdGen(varThresholdGen) -> None"},

    {NULL,          NULL}
};

static void pyopencv_BackgroundSubtractorMOG2_specials(void)
{
    pyopencv_BackgroundSubtractorMOG2_Type.tp_base = &pyopencv_BackgroundSubtractor_Type;
    pyopencv_BackgroundSubtractorMOG2_Type.tp_dealloc = pyopencv_BackgroundSubtractorMOG2_dealloc;
    pyopencv_BackgroundSubtractorMOG2_Type.tp_repr = pyopencv_BackgroundSubtractorMOG2_repr;
    pyopencv_BackgroundSubtractorMOG2_Type.tp_getset = pyopencv_BackgroundSubtractorMOG2_getseters;
    pyopencv_BackgroundSubtractorMOG2_Type.tp_methods = pyopencv_BackgroundSubtractorMOG2_methods;
}

static PyObject* pyopencv_BackgroundSubtractorKNN_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<BackgroundSubtractorKNN %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_BackgroundSubtractorKNN_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_BackgroundSubtractorKNN_getDetectShadows(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorKNN_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorKNN' or its derivative)");
    cv::BackgroundSubtractorKNN* _self_ = dynamic_cast<cv::BackgroundSubtractorKNN*>(((pyopencv_BackgroundSubtractorKNN_t*)self)->v.get());
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getDetectShadows());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorKNN_getDist2Threshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorKNN_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorKNN' or its derivative)");
    cv::BackgroundSubtractorKNN* _self_ = dynamic_cast<cv::BackgroundSubtractorKNN*>(((pyopencv_BackgroundSubtractorKNN_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getDist2Threshold());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorKNN_getHistory(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorKNN_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorKNN' or its derivative)");
    cv::BackgroundSubtractorKNN* _self_ = dynamic_cast<cv::BackgroundSubtractorKNN*>(((pyopencv_BackgroundSubtractorKNN_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getHistory());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorKNN_getNSamples(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorKNN_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorKNN' or its derivative)");
    cv::BackgroundSubtractorKNN* _self_ = dynamic_cast<cv::BackgroundSubtractorKNN*>(((pyopencv_BackgroundSubtractorKNN_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getNSamples());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorKNN_getShadowThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorKNN_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorKNN' or its derivative)");
    cv::BackgroundSubtractorKNN* _self_ = dynamic_cast<cv::BackgroundSubtractorKNN*>(((pyopencv_BackgroundSubtractorKNN_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getShadowThreshold());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorKNN_getShadowValue(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorKNN_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorKNN' or its derivative)");
    cv::BackgroundSubtractorKNN* _self_ = dynamic_cast<cv::BackgroundSubtractorKNN*>(((pyopencv_BackgroundSubtractorKNN_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getShadowValue());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorKNN_getkNNSamples(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorKNN_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorKNN' or its derivative)");
    cv::BackgroundSubtractorKNN* _self_ = dynamic_cast<cv::BackgroundSubtractorKNN*>(((pyopencv_BackgroundSubtractorKNN_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getkNNSamples());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorKNN_setDetectShadows(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorKNN_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorKNN' or its derivative)");
    cv::BackgroundSubtractorKNN* _self_ = dynamic_cast<cv::BackgroundSubtractorKNN*>(((pyopencv_BackgroundSubtractorKNN_t*)self)->v.get());
    bool detectShadows=0;

    const char* keywords[] = { "detectShadows", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:BackgroundSubtractorKNN.setDetectShadows", (char**)keywords, &detectShadows) )
    {
        ERRWRAP2(_self_->setDetectShadows(detectShadows));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorKNN_setDist2Threshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorKNN_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorKNN' or its derivative)");
    cv::BackgroundSubtractorKNN* _self_ = dynamic_cast<cv::BackgroundSubtractorKNN*>(((pyopencv_BackgroundSubtractorKNN_t*)self)->v.get());
    double _dist2Threshold=0;

    const char* keywords[] = { "_dist2Threshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:BackgroundSubtractorKNN.setDist2Threshold", (char**)keywords, &_dist2Threshold) )
    {
        ERRWRAP2(_self_->setDist2Threshold(_dist2Threshold));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorKNN_setHistory(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorKNN_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorKNN' or its derivative)");
    cv::BackgroundSubtractorKNN* _self_ = dynamic_cast<cv::BackgroundSubtractorKNN*>(((pyopencv_BackgroundSubtractorKNN_t*)self)->v.get());
    int history=0;

    const char* keywords[] = { "history", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:BackgroundSubtractorKNN.setHistory", (char**)keywords, &history) )
    {
        ERRWRAP2(_self_->setHistory(history));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorKNN_setNSamples(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorKNN_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorKNN' or its derivative)");
    cv::BackgroundSubtractorKNN* _self_ = dynamic_cast<cv::BackgroundSubtractorKNN*>(((pyopencv_BackgroundSubtractorKNN_t*)self)->v.get());
    int _nN=0;

    const char* keywords[] = { "_nN", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:BackgroundSubtractorKNN.setNSamples", (char**)keywords, &_nN) )
    {
        ERRWRAP2(_self_->setNSamples(_nN));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorKNN_setShadowThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorKNN_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorKNN' or its derivative)");
    cv::BackgroundSubtractorKNN* _self_ = dynamic_cast<cv::BackgroundSubtractorKNN*>(((pyopencv_BackgroundSubtractorKNN_t*)self)->v.get());
    double threshold=0;

    const char* keywords[] = { "threshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:BackgroundSubtractorKNN.setShadowThreshold", (char**)keywords, &threshold) )
    {
        ERRWRAP2(_self_->setShadowThreshold(threshold));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorKNN_setShadowValue(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorKNN_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorKNN' or its derivative)");
    cv::BackgroundSubtractorKNN* _self_ = dynamic_cast<cv::BackgroundSubtractorKNN*>(((pyopencv_BackgroundSubtractorKNN_t*)self)->v.get());
    int value=0;

    const char* keywords[] = { "value", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:BackgroundSubtractorKNN.setShadowValue", (char**)keywords, &value) )
    {
        ERRWRAP2(_self_->setShadowValue(value));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_BackgroundSubtractorKNN_setkNNSamples(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BackgroundSubtractorKNN_Type))
        return failmsgp("Incorrect type of self (must be 'BackgroundSubtractorKNN' or its derivative)");
    cv::BackgroundSubtractorKNN* _self_ = dynamic_cast<cv::BackgroundSubtractorKNN*>(((pyopencv_BackgroundSubtractorKNN_t*)self)->v.get());
    int _nkNN=0;

    const char* keywords[] = { "_nkNN", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:BackgroundSubtractorKNN.setkNNSamples", (char**)keywords, &_nkNN) )
    {
        ERRWRAP2(_self_->setkNNSamples(_nkNN));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_BackgroundSubtractorKNN_methods[] =
{
    {"getDetectShadows", (PyCFunction)pyopencv_cv_BackgroundSubtractorKNN_getDetectShadows, METH_VARARGS | METH_KEYWORDS, "getDetectShadows() -> retval"},
    {"getDist2Threshold", (PyCFunction)pyopencv_cv_BackgroundSubtractorKNN_getDist2Threshold, METH_VARARGS | METH_KEYWORDS, "getDist2Threshold() -> retval"},
    {"getHistory", (PyCFunction)pyopencv_cv_BackgroundSubtractorKNN_getHistory, METH_VARARGS | METH_KEYWORDS, "getHistory() -> retval"},
    {"getNSamples", (PyCFunction)pyopencv_cv_BackgroundSubtractorKNN_getNSamples, METH_VARARGS | METH_KEYWORDS, "getNSamples() -> retval"},
    {"getShadowThreshold", (PyCFunction)pyopencv_cv_BackgroundSubtractorKNN_getShadowThreshold, METH_VARARGS | METH_KEYWORDS, "getShadowThreshold() -> retval"},
    {"getShadowValue", (PyCFunction)pyopencv_cv_BackgroundSubtractorKNN_getShadowValue, METH_VARARGS | METH_KEYWORDS, "getShadowValue() -> retval"},
    {"getkNNSamples", (PyCFunction)pyopencv_cv_BackgroundSubtractorKNN_getkNNSamples, METH_VARARGS | METH_KEYWORDS, "getkNNSamples() -> retval"},
    {"setDetectShadows", (PyCFunction)pyopencv_cv_BackgroundSubtractorKNN_setDetectShadows, METH_VARARGS | METH_KEYWORDS, "setDetectShadows(detectShadows) -> None"},
    {"setDist2Threshold", (PyCFunction)pyopencv_cv_BackgroundSubtractorKNN_setDist2Threshold, METH_VARARGS | METH_KEYWORDS, "setDist2Threshold(_dist2Threshold) -> None"},
    {"setHistory", (PyCFunction)pyopencv_cv_BackgroundSubtractorKNN_setHistory, METH_VARARGS | METH_KEYWORDS, "setHistory(history) -> None"},
    {"setNSamples", (PyCFunction)pyopencv_cv_BackgroundSubtractorKNN_setNSamples, METH_VARARGS | METH_KEYWORDS, "setNSamples(_nN) -> None"},
    {"setShadowThreshold", (PyCFunction)pyopencv_cv_BackgroundSubtractorKNN_setShadowThreshold, METH_VARARGS | METH_KEYWORDS, "setShadowThreshold(threshold) -> None"},
    {"setShadowValue", (PyCFunction)pyopencv_cv_BackgroundSubtractorKNN_setShadowValue, METH_VARARGS | METH_KEYWORDS, "setShadowValue(value) -> None"},
    {"setkNNSamples", (PyCFunction)pyopencv_cv_BackgroundSubtractorKNN_setkNNSamples, METH_VARARGS | METH_KEYWORDS, "setkNNSamples(_nkNN) -> None"},

    {NULL,          NULL}
};

static void pyopencv_BackgroundSubtractorKNN_specials(void)
{
    pyopencv_BackgroundSubtractorKNN_Type.tp_base = &pyopencv_BackgroundSubtractor_Type;
    pyopencv_BackgroundSubtractorKNN_Type.tp_dealloc = pyopencv_BackgroundSubtractorKNN_dealloc;
    pyopencv_BackgroundSubtractorKNN_Type.tp_repr = pyopencv_BackgroundSubtractorKNN_repr;
    pyopencv_BackgroundSubtractorKNN_Type.tp_getset = pyopencv_BackgroundSubtractorKNN_getseters;
    pyopencv_BackgroundSubtractorKNN_Type.tp_methods = pyopencv_BackgroundSubtractorKNN_methods;
}

static PyObject* pyopencv_ShapeDistanceExtractor_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<ShapeDistanceExtractor %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_ShapeDistanceExtractor_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_ShapeDistanceExtractor_computeDistance(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeDistanceExtractor' or its derivative)");
    cv::ShapeDistanceExtractor* _self_ = dynamic_cast<cv::ShapeDistanceExtractor*>(((pyopencv_ShapeDistanceExtractor_t*)self)->v.get());
    {
    PyObject* pyobj_contour1 = NULL;
    Mat contour1;
    PyObject* pyobj_contour2 = NULL;
    Mat contour2;
    float retval;

    const char* keywords[] = { "contour1", "contour2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:ShapeDistanceExtractor.computeDistance", (char**)keywords, &pyobj_contour1, &pyobj_contour2) &&
        pyopencv_to(pyobj_contour1, contour1, ArgInfo("contour1", 0)) &&
        pyopencv_to(pyobj_contour2, contour2, ArgInfo("contour2", 0)) )
    {
        ERRWRAP2(retval = _self_->computeDistance(contour1, contour2));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_contour1 = NULL;
    UMat contour1;
    PyObject* pyobj_contour2 = NULL;
    UMat contour2;
    float retval;

    const char* keywords[] = { "contour1", "contour2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:ShapeDistanceExtractor.computeDistance", (char**)keywords, &pyobj_contour1, &pyobj_contour2) &&
        pyopencv_to(pyobj_contour1, contour1, ArgInfo("contour1", 0)) &&
        pyopencv_to(pyobj_contour2, contour2, ArgInfo("contour2", 0)) )
    {
        ERRWRAP2(retval = _self_->computeDistance(contour1, contour2));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}



static PyMethodDef pyopencv_ShapeDistanceExtractor_methods[] =
{
    {"computeDistance", (PyCFunction)pyopencv_cv_ShapeDistanceExtractor_computeDistance, METH_VARARGS | METH_KEYWORDS, "computeDistance(contour1, contour2) -> retval"},

    {NULL,          NULL}
};

static void pyopencv_ShapeDistanceExtractor_specials(void)
{
    pyopencv_ShapeDistanceExtractor_Type.tp_base = &pyopencv_Algorithm_Type;
    pyopencv_ShapeDistanceExtractor_Type.tp_dealloc = pyopencv_ShapeDistanceExtractor_dealloc;
    pyopencv_ShapeDistanceExtractor_Type.tp_repr = pyopencv_ShapeDistanceExtractor_repr;
    pyopencv_ShapeDistanceExtractor_Type.tp_getset = pyopencv_ShapeDistanceExtractor_getseters;
    pyopencv_ShapeDistanceExtractor_Type.tp_methods = pyopencv_ShapeDistanceExtractor_methods;
}

static PyObject* pyopencv_ShapeContextDistanceExtractor_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<ShapeContextDistanceExtractor %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_ShapeContextDistanceExtractor_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_getAngularBins(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getAngularBins());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_getBendingEnergyWeight(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getBendingEnergyWeight());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_getCostExtractor(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    Ptr<HistogramCostExtractor> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getCostExtractor());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_getImageAppearanceWeight(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getImageAppearanceWeight());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_getImages(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    {
    PyObject* pyobj_image1 = NULL;
    Mat image1;
    PyObject* pyobj_image2 = NULL;
    Mat image2;

    const char* keywords[] = { "image1", "image2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|OO:ShapeContextDistanceExtractor.getImages", (char**)keywords, &pyobj_image1, &pyobj_image2) &&
        pyopencv_to(pyobj_image1, image1, ArgInfo("image1", 1)) &&
        pyopencv_to(pyobj_image2, image2, ArgInfo("image2", 1)) )
    {
        ERRWRAP2(_self_->getImages(image1, image2));
        return Py_BuildValue("(NN)", pyopencv_from(image1), pyopencv_from(image2));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image1 = NULL;
    UMat image1;
    PyObject* pyobj_image2 = NULL;
    UMat image2;

    const char* keywords[] = { "image1", "image2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|OO:ShapeContextDistanceExtractor.getImages", (char**)keywords, &pyobj_image1, &pyobj_image2) &&
        pyopencv_to(pyobj_image1, image1, ArgInfo("image1", 1)) &&
        pyopencv_to(pyobj_image2, image2, ArgInfo("image2", 1)) )
    {
        ERRWRAP2(_self_->getImages(image1, image2));
        return Py_BuildValue("(NN)", pyopencv_from(image1), pyopencv_from(image2));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_getInnerRadius(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getInnerRadius());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_getIterations(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getIterations());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_getOuterRadius(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getOuterRadius());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_getRadialBins(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getRadialBins());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_getRotationInvariant(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getRotationInvariant());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_getShapeContextWeight(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getShapeContextWeight());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_getStdDev(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getStdDev());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_getTransformAlgorithm(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    Ptr<ShapeTransformer> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTransformAlgorithm());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_setAngularBins(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    int nAngularBins=0;

    const char* keywords[] = { "nAngularBins", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ShapeContextDistanceExtractor.setAngularBins", (char**)keywords, &nAngularBins) )
    {
        ERRWRAP2(_self_->setAngularBins(nAngularBins));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_setBendingEnergyWeight(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    float bendingEnergyWeight=0.f;

    const char* keywords[] = { "bendingEnergyWeight", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:ShapeContextDistanceExtractor.setBendingEnergyWeight", (char**)keywords, &bendingEnergyWeight) )
    {
        ERRWRAP2(_self_->setBendingEnergyWeight(bendingEnergyWeight));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_setCostExtractor(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    PyObject* pyobj_comparer = NULL;
    Ptr<HistogramCostExtractor> comparer;

    const char* keywords[] = { "comparer", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:ShapeContextDistanceExtractor.setCostExtractor", (char**)keywords, &pyobj_comparer) &&
        pyopencv_to(pyobj_comparer, comparer, ArgInfo("comparer", 0)) )
    {
        ERRWRAP2(_self_->setCostExtractor(comparer));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_setImageAppearanceWeight(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    float imageAppearanceWeight=0.f;

    const char* keywords[] = { "imageAppearanceWeight", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:ShapeContextDistanceExtractor.setImageAppearanceWeight", (char**)keywords, &imageAppearanceWeight) )
    {
        ERRWRAP2(_self_->setImageAppearanceWeight(imageAppearanceWeight));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_setImages(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    {
    PyObject* pyobj_image1 = NULL;
    Mat image1;
    PyObject* pyobj_image2 = NULL;
    Mat image2;

    const char* keywords[] = { "image1", "image2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:ShapeContextDistanceExtractor.setImages", (char**)keywords, &pyobj_image1, &pyobj_image2) &&
        pyopencv_to(pyobj_image1, image1, ArgInfo("image1", 0)) &&
        pyopencv_to(pyobj_image2, image2, ArgInfo("image2", 0)) )
    {
        ERRWRAP2(_self_->setImages(image1, image2));
        Py_RETURN_NONE;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image1 = NULL;
    UMat image1;
    PyObject* pyobj_image2 = NULL;
    UMat image2;

    const char* keywords[] = { "image1", "image2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO:ShapeContextDistanceExtractor.setImages", (char**)keywords, &pyobj_image1, &pyobj_image2) &&
        pyopencv_to(pyobj_image1, image1, ArgInfo("image1", 0)) &&
        pyopencv_to(pyobj_image2, image2, ArgInfo("image2", 0)) )
    {
        ERRWRAP2(_self_->setImages(image1, image2));
        Py_RETURN_NONE;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_setInnerRadius(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    float innerRadius=0.f;

    const char* keywords[] = { "innerRadius", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:ShapeContextDistanceExtractor.setInnerRadius", (char**)keywords, &innerRadius) )
    {
        ERRWRAP2(_self_->setInnerRadius(innerRadius));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_setIterations(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    int iterations=0;

    const char* keywords[] = { "iterations", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ShapeContextDistanceExtractor.setIterations", (char**)keywords, &iterations) )
    {
        ERRWRAP2(_self_->setIterations(iterations));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_setOuterRadius(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    float outerRadius=0.f;

    const char* keywords[] = { "outerRadius", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:ShapeContextDistanceExtractor.setOuterRadius", (char**)keywords, &outerRadius) )
    {
        ERRWRAP2(_self_->setOuterRadius(outerRadius));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_setRadialBins(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    int nRadialBins=0;

    const char* keywords[] = { "nRadialBins", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ShapeContextDistanceExtractor.setRadialBins", (char**)keywords, &nRadialBins) )
    {
        ERRWRAP2(_self_->setRadialBins(nRadialBins));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_setRotationInvariant(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    bool rotationInvariant=0;

    const char* keywords[] = { "rotationInvariant", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:ShapeContextDistanceExtractor.setRotationInvariant", (char**)keywords, &rotationInvariant) )
    {
        ERRWRAP2(_self_->setRotationInvariant(rotationInvariant));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_setShapeContextWeight(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    float shapeContextWeight=0.f;

    const char* keywords[] = { "shapeContextWeight", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:ShapeContextDistanceExtractor.setShapeContextWeight", (char**)keywords, &shapeContextWeight) )
    {
        ERRWRAP2(_self_->setShapeContextWeight(shapeContextWeight));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_setStdDev(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    float sigma=0.f;

    const char* keywords[] = { "sigma", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:ShapeContextDistanceExtractor.setStdDev", (char**)keywords, &sigma) )
    {
        ERRWRAP2(_self_->setStdDev(sigma));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeContextDistanceExtractor_setTransformAlgorithm(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeContextDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeContextDistanceExtractor' or its derivative)");
    cv::ShapeContextDistanceExtractor* _self_ = dynamic_cast<cv::ShapeContextDistanceExtractor*>(((pyopencv_ShapeContextDistanceExtractor_t*)self)->v.get());
    PyObject* pyobj_transformer = NULL;
    Ptr<ShapeTransformer> transformer;

    const char* keywords[] = { "transformer", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:ShapeContextDistanceExtractor.setTransformAlgorithm", (char**)keywords, &pyobj_transformer) &&
        pyopencv_to(pyobj_transformer, transformer, ArgInfo("transformer", 0)) )
    {
        ERRWRAP2(_self_->setTransformAlgorithm(transformer));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_ShapeContextDistanceExtractor_methods[] =
{
    {"getAngularBins", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_getAngularBins, METH_VARARGS | METH_KEYWORDS, "getAngularBins() -> retval"},
    {"getBendingEnergyWeight", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_getBendingEnergyWeight, METH_VARARGS | METH_KEYWORDS, "getBendingEnergyWeight() -> retval"},
    {"getCostExtractor", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_getCostExtractor, METH_VARARGS | METH_KEYWORDS, "getCostExtractor() -> retval"},
    {"getImageAppearanceWeight", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_getImageAppearanceWeight, METH_VARARGS | METH_KEYWORDS, "getImageAppearanceWeight() -> retval"},
    {"getImages", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_getImages, METH_VARARGS | METH_KEYWORDS, "getImages([, image1[, image2]]) -> image1, image2"},
    {"getInnerRadius", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_getInnerRadius, METH_VARARGS | METH_KEYWORDS, "getInnerRadius() -> retval"},
    {"getIterations", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_getIterations, METH_VARARGS | METH_KEYWORDS, "getIterations() -> retval"},
    {"getOuterRadius", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_getOuterRadius, METH_VARARGS | METH_KEYWORDS, "getOuterRadius() -> retval"},
    {"getRadialBins", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_getRadialBins, METH_VARARGS | METH_KEYWORDS, "getRadialBins() -> retval"},
    {"getRotationInvariant", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_getRotationInvariant, METH_VARARGS | METH_KEYWORDS, "getRotationInvariant() -> retval"},
    {"getShapeContextWeight", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_getShapeContextWeight, METH_VARARGS | METH_KEYWORDS, "getShapeContextWeight() -> retval"},
    {"getStdDev", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_getStdDev, METH_VARARGS | METH_KEYWORDS, "getStdDev() -> retval"},
    {"getTransformAlgorithm", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_getTransformAlgorithm, METH_VARARGS | METH_KEYWORDS, "getTransformAlgorithm() -> retval"},
    {"setAngularBins", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_setAngularBins, METH_VARARGS | METH_KEYWORDS, "setAngularBins(nAngularBins) -> None"},
    {"setBendingEnergyWeight", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_setBendingEnergyWeight, METH_VARARGS | METH_KEYWORDS, "setBendingEnergyWeight(bendingEnergyWeight) -> None"},
    {"setCostExtractor", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_setCostExtractor, METH_VARARGS | METH_KEYWORDS, "setCostExtractor(comparer) -> None"},
    {"setImageAppearanceWeight", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_setImageAppearanceWeight, METH_VARARGS | METH_KEYWORDS, "setImageAppearanceWeight(imageAppearanceWeight) -> None"},
    {"setImages", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_setImages, METH_VARARGS | METH_KEYWORDS, "setImages(image1, image2) -> None"},
    {"setInnerRadius", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_setInnerRadius, METH_VARARGS | METH_KEYWORDS, "setInnerRadius(innerRadius) -> None"},
    {"setIterations", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_setIterations, METH_VARARGS | METH_KEYWORDS, "setIterations(iterations) -> None"},
    {"setOuterRadius", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_setOuterRadius, METH_VARARGS | METH_KEYWORDS, "setOuterRadius(outerRadius) -> None"},
    {"setRadialBins", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_setRadialBins, METH_VARARGS | METH_KEYWORDS, "setRadialBins(nRadialBins) -> None"},
    {"setRotationInvariant", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_setRotationInvariant, METH_VARARGS | METH_KEYWORDS, "setRotationInvariant(rotationInvariant) -> None"},
    {"setShapeContextWeight", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_setShapeContextWeight, METH_VARARGS | METH_KEYWORDS, "setShapeContextWeight(shapeContextWeight) -> None"},
    {"setStdDev", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_setStdDev, METH_VARARGS | METH_KEYWORDS, "setStdDev(sigma) -> None"},
    {"setTransformAlgorithm", (PyCFunction)pyopencv_cv_ShapeContextDistanceExtractor_setTransformAlgorithm, METH_VARARGS | METH_KEYWORDS, "setTransformAlgorithm(transformer) -> None"},

    {NULL,          NULL}
};

static void pyopencv_ShapeContextDistanceExtractor_specials(void)
{
    pyopencv_ShapeContextDistanceExtractor_Type.tp_base = &pyopencv_ShapeDistanceExtractor_Type;
    pyopencv_ShapeContextDistanceExtractor_Type.tp_dealloc = pyopencv_ShapeContextDistanceExtractor_dealloc;
    pyopencv_ShapeContextDistanceExtractor_Type.tp_repr = pyopencv_ShapeContextDistanceExtractor_repr;
    pyopencv_ShapeContextDistanceExtractor_Type.tp_getset = pyopencv_ShapeContextDistanceExtractor_getseters;
    pyopencv_ShapeContextDistanceExtractor_Type.tp_methods = pyopencv_ShapeContextDistanceExtractor_methods;
}

static PyObject* pyopencv_HausdorffDistanceExtractor_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<HausdorffDistanceExtractor %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_HausdorffDistanceExtractor_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_HausdorffDistanceExtractor_getDistanceFlag(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_HausdorffDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'HausdorffDistanceExtractor' or its derivative)");
    cv::HausdorffDistanceExtractor* _self_ = dynamic_cast<cv::HausdorffDistanceExtractor*>(((pyopencv_HausdorffDistanceExtractor_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getDistanceFlag());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_HausdorffDistanceExtractor_getRankProportion(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_HausdorffDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'HausdorffDistanceExtractor' or its derivative)");
    cv::HausdorffDistanceExtractor* _self_ = dynamic_cast<cv::HausdorffDistanceExtractor*>(((pyopencv_HausdorffDistanceExtractor_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getRankProportion());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_HausdorffDistanceExtractor_setDistanceFlag(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_HausdorffDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'HausdorffDistanceExtractor' or its derivative)");
    cv::HausdorffDistanceExtractor* _self_ = dynamic_cast<cv::HausdorffDistanceExtractor*>(((pyopencv_HausdorffDistanceExtractor_t*)self)->v.get());
    int distanceFlag=0;

    const char* keywords[] = { "distanceFlag", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:HausdorffDistanceExtractor.setDistanceFlag", (char**)keywords, &distanceFlag) )
    {
        ERRWRAP2(_self_->setDistanceFlag(distanceFlag));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_HausdorffDistanceExtractor_setRankProportion(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_HausdorffDistanceExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'HausdorffDistanceExtractor' or its derivative)");
    cv::HausdorffDistanceExtractor* _self_ = dynamic_cast<cv::HausdorffDistanceExtractor*>(((pyopencv_HausdorffDistanceExtractor_t*)self)->v.get());
    float rankProportion=0.f;

    const char* keywords[] = { "rankProportion", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:HausdorffDistanceExtractor.setRankProportion", (char**)keywords, &rankProportion) )
    {
        ERRWRAP2(_self_->setRankProportion(rankProportion));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_HausdorffDistanceExtractor_methods[] =
{
    {"getDistanceFlag", (PyCFunction)pyopencv_cv_HausdorffDistanceExtractor_getDistanceFlag, METH_VARARGS | METH_KEYWORDS, "getDistanceFlag() -> retval"},
    {"getRankProportion", (PyCFunction)pyopencv_cv_HausdorffDistanceExtractor_getRankProportion, METH_VARARGS | METH_KEYWORDS, "getRankProportion() -> retval"},
    {"setDistanceFlag", (PyCFunction)pyopencv_cv_HausdorffDistanceExtractor_setDistanceFlag, METH_VARARGS | METH_KEYWORDS, "setDistanceFlag(distanceFlag) -> None"},
    {"setRankProportion", (PyCFunction)pyopencv_cv_HausdorffDistanceExtractor_setRankProportion, METH_VARARGS | METH_KEYWORDS, "setRankProportion(rankProportion) -> None"},

    {NULL,          NULL}
};

static void pyopencv_HausdorffDistanceExtractor_specials(void)
{
    pyopencv_HausdorffDistanceExtractor_Type.tp_base = &pyopencv_ShapeDistanceExtractor_Type;
    pyopencv_HausdorffDistanceExtractor_Type.tp_dealloc = pyopencv_HausdorffDistanceExtractor_dealloc;
    pyopencv_HausdorffDistanceExtractor_Type.tp_repr = pyopencv_HausdorffDistanceExtractor_repr;
    pyopencv_HausdorffDistanceExtractor_Type.tp_getset = pyopencv_HausdorffDistanceExtractor_getseters;
    pyopencv_HausdorffDistanceExtractor_Type.tp_methods = pyopencv_HausdorffDistanceExtractor_methods;
}

static PyObject* pyopencv_HistogramCostExtractor_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<HistogramCostExtractor %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_HistogramCostExtractor_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_HistogramCostExtractor_buildCostMatrix(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_HistogramCostExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'HistogramCostExtractor' or its derivative)");
    cv::HistogramCostExtractor* _self_ = dynamic_cast<cv::HistogramCostExtractor*>(((pyopencv_HistogramCostExtractor_t*)self)->v.get());
    {
    PyObject* pyobj_descriptors1 = NULL;
    Mat descriptors1;
    PyObject* pyobj_descriptors2 = NULL;
    Mat descriptors2;
    PyObject* pyobj_costMatrix = NULL;
    Mat costMatrix;

    const char* keywords[] = { "descriptors1", "descriptors2", "costMatrix", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:HistogramCostExtractor.buildCostMatrix", (char**)keywords, &pyobj_descriptors1, &pyobj_descriptors2, &pyobj_costMatrix) &&
        pyopencv_to(pyobj_descriptors1, descriptors1, ArgInfo("descriptors1", 0)) &&
        pyopencv_to(pyobj_descriptors2, descriptors2, ArgInfo("descriptors2", 0)) &&
        pyopencv_to(pyobj_costMatrix, costMatrix, ArgInfo("costMatrix", 1)) )
    {
        ERRWRAP2(_self_->buildCostMatrix(descriptors1, descriptors2, costMatrix));
        return pyopencv_from(costMatrix);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_descriptors1 = NULL;
    UMat descriptors1;
    PyObject* pyobj_descriptors2 = NULL;
    UMat descriptors2;
    PyObject* pyobj_costMatrix = NULL;
    UMat costMatrix;

    const char* keywords[] = { "descriptors1", "descriptors2", "costMatrix", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:HistogramCostExtractor.buildCostMatrix", (char**)keywords, &pyobj_descriptors1, &pyobj_descriptors2, &pyobj_costMatrix) &&
        pyopencv_to(pyobj_descriptors1, descriptors1, ArgInfo("descriptors1", 0)) &&
        pyopencv_to(pyobj_descriptors2, descriptors2, ArgInfo("descriptors2", 0)) &&
        pyopencv_to(pyobj_costMatrix, costMatrix, ArgInfo("costMatrix", 1)) )
    {
        ERRWRAP2(_self_->buildCostMatrix(descriptors1, descriptors2, costMatrix));
        return pyopencv_from(costMatrix);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_HistogramCostExtractor_getDefaultCost(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_HistogramCostExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'HistogramCostExtractor' or its derivative)");
    cv::HistogramCostExtractor* _self_ = dynamic_cast<cv::HistogramCostExtractor*>(((pyopencv_HistogramCostExtractor_t*)self)->v.get());
    float retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getDefaultCost());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_HistogramCostExtractor_getNDummies(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_HistogramCostExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'HistogramCostExtractor' or its derivative)");
    cv::HistogramCostExtractor* _self_ = dynamic_cast<cv::HistogramCostExtractor*>(((pyopencv_HistogramCostExtractor_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getNDummies());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_HistogramCostExtractor_setDefaultCost(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_HistogramCostExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'HistogramCostExtractor' or its derivative)");
    cv::HistogramCostExtractor* _self_ = dynamic_cast<cv::HistogramCostExtractor*>(((pyopencv_HistogramCostExtractor_t*)self)->v.get());
    float defaultCost=0.f;

    const char* keywords[] = { "defaultCost", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "f:HistogramCostExtractor.setDefaultCost", (char**)keywords, &defaultCost) )
    {
        ERRWRAP2(_self_->setDefaultCost(defaultCost));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_HistogramCostExtractor_setNDummies(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_HistogramCostExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'HistogramCostExtractor' or its derivative)");
    cv::HistogramCostExtractor* _self_ = dynamic_cast<cv::HistogramCostExtractor*>(((pyopencv_HistogramCostExtractor_t*)self)->v.get());
    int nDummies=0;

    const char* keywords[] = { "nDummies", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:HistogramCostExtractor.setNDummies", (char**)keywords, &nDummies) )
    {
        ERRWRAP2(_self_->setNDummies(nDummies));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_HistogramCostExtractor_methods[] =
{
    {"buildCostMatrix", (PyCFunction)pyopencv_cv_HistogramCostExtractor_buildCostMatrix, METH_VARARGS | METH_KEYWORDS, "buildCostMatrix(descriptors1, descriptors2[, costMatrix]) -> costMatrix"},
    {"getDefaultCost", (PyCFunction)pyopencv_cv_HistogramCostExtractor_getDefaultCost, METH_VARARGS | METH_KEYWORDS, "getDefaultCost() -> retval"},
    {"getNDummies", (PyCFunction)pyopencv_cv_HistogramCostExtractor_getNDummies, METH_VARARGS | METH_KEYWORDS, "getNDummies() -> retval"},
    {"setDefaultCost", (PyCFunction)pyopencv_cv_HistogramCostExtractor_setDefaultCost, METH_VARARGS | METH_KEYWORDS, "setDefaultCost(defaultCost) -> None"},
    {"setNDummies", (PyCFunction)pyopencv_cv_HistogramCostExtractor_setNDummies, METH_VARARGS | METH_KEYWORDS, "setNDummies(nDummies) -> None"},

    {NULL,          NULL}
};

static void pyopencv_HistogramCostExtractor_specials(void)
{
    pyopencv_HistogramCostExtractor_Type.tp_base = &pyopencv_Algorithm_Type;
    pyopencv_HistogramCostExtractor_Type.tp_dealloc = pyopencv_HistogramCostExtractor_dealloc;
    pyopencv_HistogramCostExtractor_Type.tp_repr = pyopencv_HistogramCostExtractor_repr;
    pyopencv_HistogramCostExtractor_Type.tp_getset = pyopencv_HistogramCostExtractor_getseters;
    pyopencv_HistogramCostExtractor_Type.tp_methods = pyopencv_HistogramCostExtractor_methods;
}

static PyObject* pyopencv_NormHistogramCostExtractor_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<NormHistogramCostExtractor %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_NormHistogramCostExtractor_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_NormHistogramCostExtractor_getNormFlag(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_NormHistogramCostExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'NormHistogramCostExtractor' or its derivative)");
    cv::NormHistogramCostExtractor* _self_ = dynamic_cast<cv::NormHistogramCostExtractor*>(((pyopencv_NormHistogramCostExtractor_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getNormFlag());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_NormHistogramCostExtractor_setNormFlag(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_NormHistogramCostExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'NormHistogramCostExtractor' or its derivative)");
    cv::NormHistogramCostExtractor* _self_ = dynamic_cast<cv::NormHistogramCostExtractor*>(((pyopencv_NormHistogramCostExtractor_t*)self)->v.get());
    int flag=0;

    const char* keywords[] = { "flag", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:NormHistogramCostExtractor.setNormFlag", (char**)keywords, &flag) )
    {
        ERRWRAP2(_self_->setNormFlag(flag));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_NormHistogramCostExtractor_methods[] =
{
    {"getNormFlag", (PyCFunction)pyopencv_cv_NormHistogramCostExtractor_getNormFlag, METH_VARARGS | METH_KEYWORDS, "getNormFlag() -> retval"},
    {"setNormFlag", (PyCFunction)pyopencv_cv_NormHistogramCostExtractor_setNormFlag, METH_VARARGS | METH_KEYWORDS, "setNormFlag(flag) -> None"},

    {NULL,          NULL}
};

static void pyopencv_NormHistogramCostExtractor_specials(void)
{
    pyopencv_NormHistogramCostExtractor_Type.tp_base = &pyopencv_HistogramCostExtractor_Type;
    pyopencv_NormHistogramCostExtractor_Type.tp_dealloc = pyopencv_NormHistogramCostExtractor_dealloc;
    pyopencv_NormHistogramCostExtractor_Type.tp_repr = pyopencv_NormHistogramCostExtractor_repr;
    pyopencv_NormHistogramCostExtractor_Type.tp_getset = pyopencv_NormHistogramCostExtractor_getseters;
    pyopencv_NormHistogramCostExtractor_Type.tp_methods = pyopencv_NormHistogramCostExtractor_methods;
}

static PyObject* pyopencv_EMDHistogramCostExtractor_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<EMDHistogramCostExtractor %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_EMDHistogramCostExtractor_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_EMDHistogramCostExtractor_getNormFlag(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_EMDHistogramCostExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'EMDHistogramCostExtractor' or its derivative)");
    cv::EMDHistogramCostExtractor* _self_ = dynamic_cast<cv::EMDHistogramCostExtractor*>(((pyopencv_EMDHistogramCostExtractor_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getNormFlag());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_EMDHistogramCostExtractor_setNormFlag(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_EMDHistogramCostExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'EMDHistogramCostExtractor' or its derivative)");
    cv::EMDHistogramCostExtractor* _self_ = dynamic_cast<cv::EMDHistogramCostExtractor*>(((pyopencv_EMDHistogramCostExtractor_t*)self)->v.get());
    int flag=0;

    const char* keywords[] = { "flag", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:EMDHistogramCostExtractor.setNormFlag", (char**)keywords, &flag) )
    {
        ERRWRAP2(_self_->setNormFlag(flag));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_EMDHistogramCostExtractor_methods[] =
{
    {"getNormFlag", (PyCFunction)pyopencv_cv_EMDHistogramCostExtractor_getNormFlag, METH_VARARGS | METH_KEYWORDS, "getNormFlag() -> retval"},
    {"setNormFlag", (PyCFunction)pyopencv_cv_EMDHistogramCostExtractor_setNormFlag, METH_VARARGS | METH_KEYWORDS, "setNormFlag(flag) -> None"},

    {NULL,          NULL}
};

static void pyopencv_EMDHistogramCostExtractor_specials(void)
{
    pyopencv_EMDHistogramCostExtractor_Type.tp_base = &pyopencv_HistogramCostExtractor_Type;
    pyopencv_EMDHistogramCostExtractor_Type.tp_dealloc = pyopencv_EMDHistogramCostExtractor_dealloc;
    pyopencv_EMDHistogramCostExtractor_Type.tp_repr = pyopencv_EMDHistogramCostExtractor_repr;
    pyopencv_EMDHistogramCostExtractor_Type.tp_getset = pyopencv_EMDHistogramCostExtractor_getseters;
    pyopencv_EMDHistogramCostExtractor_Type.tp_methods = pyopencv_EMDHistogramCostExtractor_methods;
}

static PyObject* pyopencv_ChiHistogramCostExtractor_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<ChiHistogramCostExtractor %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_ChiHistogramCostExtractor_getseters[] =
{
    {NULL}  /* Sentinel */
};



static PyMethodDef pyopencv_ChiHistogramCostExtractor_methods[] =
{

    {NULL,          NULL}
};

static void pyopencv_ChiHistogramCostExtractor_specials(void)
{
    pyopencv_ChiHistogramCostExtractor_Type.tp_base = &pyopencv_HistogramCostExtractor_Type;
    pyopencv_ChiHistogramCostExtractor_Type.tp_dealloc = pyopencv_ChiHistogramCostExtractor_dealloc;
    pyopencv_ChiHistogramCostExtractor_Type.tp_repr = pyopencv_ChiHistogramCostExtractor_repr;
    pyopencv_ChiHistogramCostExtractor_Type.tp_getset = pyopencv_ChiHistogramCostExtractor_getseters;
    pyopencv_ChiHistogramCostExtractor_Type.tp_methods = pyopencv_ChiHistogramCostExtractor_methods;
}

static PyObject* pyopencv_EMDL1HistogramCostExtractor_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<EMDL1HistogramCostExtractor %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_EMDL1HistogramCostExtractor_getseters[] =
{
    {NULL}  /* Sentinel */
};



static PyMethodDef pyopencv_EMDL1HistogramCostExtractor_methods[] =
{

    {NULL,          NULL}
};

static void pyopencv_EMDL1HistogramCostExtractor_specials(void)
{
    pyopencv_EMDL1HistogramCostExtractor_Type.tp_base = &pyopencv_HistogramCostExtractor_Type;
    pyopencv_EMDL1HistogramCostExtractor_Type.tp_dealloc = pyopencv_EMDL1HistogramCostExtractor_dealloc;
    pyopencv_EMDL1HistogramCostExtractor_Type.tp_repr = pyopencv_EMDL1HistogramCostExtractor_repr;
    pyopencv_EMDL1HistogramCostExtractor_Type.tp_getset = pyopencv_EMDL1HistogramCostExtractor_getseters;
    pyopencv_EMDL1HistogramCostExtractor_Type.tp_methods = pyopencv_EMDL1HistogramCostExtractor_methods;
}

static PyObject* pyopencv_ShapeTransformer_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<ShapeTransformer %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_ShapeTransformer_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_ShapeTransformer_applyTransformation(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeTransformer_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeTransformer' or its derivative)");
    cv::ShapeTransformer* _self_ = dynamic_cast<cv::ShapeTransformer*>(((pyopencv_ShapeTransformer_t*)self)->v.get());
    {
    PyObject* pyobj_input = NULL;
    Mat input;
    PyObject* pyobj_output = NULL;
    Mat output;
    float retval;

    const char* keywords[] = { "input", "output", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:ShapeTransformer.applyTransformation", (char**)keywords, &pyobj_input, &pyobj_output) &&
        pyopencv_to(pyobj_input, input, ArgInfo("input", 0)) &&
        pyopencv_to(pyobj_output, output, ArgInfo("output", 1)) )
    {
        ERRWRAP2(retval = _self_->applyTransformation(input, output));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(output));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_input = NULL;
    UMat input;
    PyObject* pyobj_output = NULL;
    UMat output;
    float retval;

    const char* keywords[] = { "input", "output", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:ShapeTransformer.applyTransformation", (char**)keywords, &pyobj_input, &pyobj_output) &&
        pyopencv_to(pyobj_input, input, ArgInfo("input", 0)) &&
        pyopencv_to(pyobj_output, output, ArgInfo("output", 1)) )
    {
        ERRWRAP2(retval = _self_->applyTransformation(input, output));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(output));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeTransformer_estimateTransformation(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeTransformer_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeTransformer' or its derivative)");
    cv::ShapeTransformer* _self_ = dynamic_cast<cv::ShapeTransformer*>(((pyopencv_ShapeTransformer_t*)self)->v.get());
    {
    PyObject* pyobj_transformingShape = NULL;
    Mat transformingShape;
    PyObject* pyobj_targetShape = NULL;
    Mat targetShape;
    PyObject* pyobj_matches = NULL;
    vector_DMatch matches;

    const char* keywords[] = { "transformingShape", "targetShape", "matches", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:ShapeTransformer.estimateTransformation", (char**)keywords, &pyobj_transformingShape, &pyobj_targetShape, &pyobj_matches) &&
        pyopencv_to(pyobj_transformingShape, transformingShape, ArgInfo("transformingShape", 0)) &&
        pyopencv_to(pyobj_targetShape, targetShape, ArgInfo("targetShape", 0)) &&
        pyopencv_to(pyobj_matches, matches, ArgInfo("matches", 0)) )
    {
        ERRWRAP2(_self_->estimateTransformation(transformingShape, targetShape, matches));
        Py_RETURN_NONE;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_transformingShape = NULL;
    UMat transformingShape;
    PyObject* pyobj_targetShape = NULL;
    UMat targetShape;
    PyObject* pyobj_matches = NULL;
    vector_DMatch matches;

    const char* keywords[] = { "transformingShape", "targetShape", "matches", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOO:ShapeTransformer.estimateTransformation", (char**)keywords, &pyobj_transformingShape, &pyobj_targetShape, &pyobj_matches) &&
        pyopencv_to(pyobj_transformingShape, transformingShape, ArgInfo("transformingShape", 0)) &&
        pyopencv_to(pyobj_targetShape, targetShape, ArgInfo("targetShape", 0)) &&
        pyopencv_to(pyobj_matches, matches, ArgInfo("matches", 0)) )
    {
        ERRWRAP2(_self_->estimateTransformation(transformingShape, targetShape, matches));
        Py_RETURN_NONE;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_ShapeTransformer_warpImage(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ShapeTransformer_Type))
        return failmsgp("Incorrect type of self (must be 'ShapeTransformer' or its derivative)");
    cv::ShapeTransformer* _self_ = dynamic_cast<cv::ShapeTransformer*>(((pyopencv_ShapeTransformer_t*)self)->v.get());
    {
    PyObject* pyobj_transformingImage = NULL;
    Mat transformingImage;
    PyObject* pyobj_output = NULL;
    Mat output;
    int flags=INTER_LINEAR;
    int borderMode=BORDER_CONSTANT;
    PyObject* pyobj_borderValue = NULL;
    Scalar borderValue;

    const char* keywords[] = { "transformingImage", "output", "flags", "borderMode", "borderValue", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OiiO:ShapeTransformer.warpImage", (char**)keywords, &pyobj_transformingImage, &pyobj_output, &flags, &borderMode, &pyobj_borderValue) &&
        pyopencv_to(pyobj_transformingImage, transformingImage, ArgInfo("transformingImage", 0)) &&
        pyopencv_to(pyobj_output, output, ArgInfo("output", 1)) &&
        pyopencv_to(pyobj_borderValue, borderValue, ArgInfo("borderValue", 0)) )
    {
        ERRWRAP2(_self_->warpImage(transformingImage, output, flags, borderMode, borderValue));
        return pyopencv_from(output);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_transformingImage = NULL;
    UMat transformingImage;
    PyObject* pyobj_output = NULL;
    UMat output;
    int flags=INTER_LINEAR;
    int borderMode=BORDER_CONSTANT;
    PyObject* pyobj_borderValue = NULL;
    Scalar borderValue;

    const char* keywords[] = { "transformingImage", "output", "flags", "borderMode", "borderValue", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OiiO:ShapeTransformer.warpImage", (char**)keywords, &pyobj_transformingImage, &pyobj_output, &flags, &borderMode, &pyobj_borderValue) &&
        pyopencv_to(pyobj_transformingImage, transformingImage, ArgInfo("transformingImage", 0)) &&
        pyopencv_to(pyobj_output, output, ArgInfo("output", 1)) &&
        pyopencv_to(pyobj_borderValue, borderValue, ArgInfo("borderValue", 0)) )
    {
        ERRWRAP2(_self_->warpImage(transformingImage, output, flags, borderMode, borderValue));
        return pyopencv_from(output);
    }
    }

    return NULL;
}



static PyMethodDef pyopencv_ShapeTransformer_methods[] =
{
    {"applyTransformation", (PyCFunction)pyopencv_cv_ShapeTransformer_applyTransformation, METH_VARARGS | METH_KEYWORDS, "applyTransformation(input[, output]) -> retval, output"},
    {"estimateTransformation", (PyCFunction)pyopencv_cv_ShapeTransformer_estimateTransformation, METH_VARARGS | METH_KEYWORDS, "estimateTransformation(transformingShape, targetShape, matches) -> None"},
    {"warpImage", (PyCFunction)pyopencv_cv_ShapeTransformer_warpImage, METH_VARARGS | METH_KEYWORDS, "warpImage(transformingImage[, output[, flags[, borderMode[, borderValue]]]]) -> output"},

    {NULL,          NULL}
};

static void pyopencv_ShapeTransformer_specials(void)
{
    pyopencv_ShapeTransformer_Type.tp_base = &pyopencv_Algorithm_Type;
    pyopencv_ShapeTransformer_Type.tp_dealloc = pyopencv_ShapeTransformer_dealloc;
    pyopencv_ShapeTransformer_Type.tp_repr = pyopencv_ShapeTransformer_repr;
    pyopencv_ShapeTransformer_Type.tp_getset = pyopencv_ShapeTransformer_getseters;
    pyopencv_ShapeTransformer_Type.tp_methods = pyopencv_ShapeTransformer_methods;
}

static PyObject* pyopencv_ThinPlateSplineShapeTransformer_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<ThinPlateSplineShapeTransformer %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_ThinPlateSplineShapeTransformer_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_ThinPlateSplineShapeTransformer_getRegularizationParameter(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ThinPlateSplineShapeTransformer_Type))
        return failmsgp("Incorrect type of self (must be 'ThinPlateSplineShapeTransformer' or its derivative)");
    cv::ThinPlateSplineShapeTransformer* _self_ = dynamic_cast<cv::ThinPlateSplineShapeTransformer*>(((pyopencv_ThinPlateSplineShapeTransformer_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getRegularizationParameter());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ThinPlateSplineShapeTransformer_setRegularizationParameter(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ThinPlateSplineShapeTransformer_Type))
        return failmsgp("Incorrect type of self (must be 'ThinPlateSplineShapeTransformer' or its derivative)");
    cv::ThinPlateSplineShapeTransformer* _self_ = dynamic_cast<cv::ThinPlateSplineShapeTransformer*>(((pyopencv_ThinPlateSplineShapeTransformer_t*)self)->v.get());
    double beta=0;

    const char* keywords[] = { "beta", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:ThinPlateSplineShapeTransformer.setRegularizationParameter", (char**)keywords, &beta) )
    {
        ERRWRAP2(_self_->setRegularizationParameter(beta));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_ThinPlateSplineShapeTransformer_methods[] =
{
    {"getRegularizationParameter", (PyCFunction)pyopencv_cv_ThinPlateSplineShapeTransformer_getRegularizationParameter, METH_VARARGS | METH_KEYWORDS, "getRegularizationParameter() -> retval"},
    {"setRegularizationParameter", (PyCFunction)pyopencv_cv_ThinPlateSplineShapeTransformer_setRegularizationParameter, METH_VARARGS | METH_KEYWORDS, "setRegularizationParameter(beta) -> None"},

    {NULL,          NULL}
};

static void pyopencv_ThinPlateSplineShapeTransformer_specials(void)
{
    pyopencv_ThinPlateSplineShapeTransformer_Type.tp_base = &pyopencv_ShapeTransformer_Type;
    pyopencv_ThinPlateSplineShapeTransformer_Type.tp_dealloc = pyopencv_ThinPlateSplineShapeTransformer_dealloc;
    pyopencv_ThinPlateSplineShapeTransformer_Type.tp_repr = pyopencv_ThinPlateSplineShapeTransformer_repr;
    pyopencv_ThinPlateSplineShapeTransformer_Type.tp_getset = pyopencv_ThinPlateSplineShapeTransformer_getseters;
    pyopencv_ThinPlateSplineShapeTransformer_Type.tp_methods = pyopencv_ThinPlateSplineShapeTransformer_methods;
}

static PyObject* pyopencv_AffineTransformer_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<AffineTransformer %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_AffineTransformer_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_AffineTransformer_getFullAffine(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AffineTransformer_Type))
        return failmsgp("Incorrect type of self (must be 'AffineTransformer' or its derivative)");
    cv::AffineTransformer* _self_ = dynamic_cast<cv::AffineTransformer*>(((pyopencv_AffineTransformer_t*)self)->v.get());
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getFullAffine());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_AffineTransformer_setFullAffine(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AffineTransformer_Type))
        return failmsgp("Incorrect type of self (must be 'AffineTransformer' or its derivative)");
    cv::AffineTransformer* _self_ = dynamic_cast<cv::AffineTransformer*>(((pyopencv_AffineTransformer_t*)self)->v.get());
    bool fullAffine=0;

    const char* keywords[] = { "fullAffine", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:AffineTransformer.setFullAffine", (char**)keywords, &fullAffine) )
    {
        ERRWRAP2(_self_->setFullAffine(fullAffine));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_AffineTransformer_methods[] =
{
    {"getFullAffine", (PyCFunction)pyopencv_cv_AffineTransformer_getFullAffine, METH_VARARGS | METH_KEYWORDS, "getFullAffine() -> retval"},
    {"setFullAffine", (PyCFunction)pyopencv_cv_AffineTransformer_setFullAffine, METH_VARARGS | METH_KEYWORDS, "setFullAffine(fullAffine) -> None"},

    {NULL,          NULL}
};

static void pyopencv_AffineTransformer_specials(void)
{
    pyopencv_AffineTransformer_Type.tp_base = &pyopencv_ShapeTransformer_Type;
    pyopencv_AffineTransformer_Type.tp_dealloc = pyopencv_AffineTransformer_dealloc;
    pyopencv_AffineTransformer_Type.tp_repr = pyopencv_AffineTransformer_repr;
    pyopencv_AffineTransformer_Type.tp_getset = pyopencv_AffineTransformer_getseters;
    pyopencv_AffineTransformer_Type.tp_methods = pyopencv_AffineTransformer_methods;
}

static PyObject* pyopencv_VideoCapture_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<VideoCapture %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_VideoCapture_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_VideoCapture_get(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_VideoCapture_Type))
        return failmsgp("Incorrect type of self (must be 'VideoCapture' or its derivative)");
    cv::VideoCapture* _self_ = ((pyopencv_VideoCapture_t*)self)->v.get();
    int propId=0;
    double retval;

    const char* keywords[] = { "propId", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:VideoCapture.get", (char**)keywords, &propId) )
    {
        ERRWRAP2(retval = _self_->get(propId));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_VideoCapture_grab(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_VideoCapture_Type))
        return failmsgp("Incorrect type of self (must be 'VideoCapture' or its derivative)");
    cv::VideoCapture* _self_ = ((pyopencv_VideoCapture_t*)self)->v.get();
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->grab());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_VideoCapture_isOpened(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_VideoCapture_Type))
        return failmsgp("Incorrect type of self (must be 'VideoCapture' or its derivative)");
    cv::VideoCapture* _self_ = ((pyopencv_VideoCapture_t*)self)->v.get();
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->isOpened());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_VideoCapture_open(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_VideoCapture_Type))
        return failmsgp("Incorrect type of self (must be 'VideoCapture' or its derivative)");
    cv::VideoCapture* _self_ = ((pyopencv_VideoCapture_t*)self)->v.get();
    {
    PyObject* pyobj_filename = NULL;
    String filename;
    bool retval;

    const char* keywords[] = { "filename", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:VideoCapture.open", (char**)keywords, &pyobj_filename) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) )
    {
        ERRWRAP2(retval = _self_->open(filename));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    int index=0;
    bool retval;

    const char* keywords[] = { "index", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:VideoCapture.open", (char**)keywords, &index) )
    {
        ERRWRAP2(retval = _self_->open(index));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    int cameraNum=0;
    int apiPreference=0;
    bool retval;

    const char* keywords[] = { "cameraNum", "apiPreference", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "ii:VideoCapture.open", (char**)keywords, &cameraNum, &apiPreference) )
    {
        ERRWRAP2(retval = _self_->open(cameraNum, apiPreference));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_filename = NULL;
    String filename;
    int apiPreference=0;
    bool retval;

    const char* keywords[] = { "filename", "apiPreference", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi:VideoCapture.open", (char**)keywords, &pyobj_filename, &apiPreference) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) )
    {
        ERRWRAP2(retval = _self_->open(filename, apiPreference));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_VideoCapture_read(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_VideoCapture_Type))
        return failmsgp("Incorrect type of self (must be 'VideoCapture' or its derivative)");
    cv::VideoCapture* _self_ = ((pyopencv_VideoCapture_t*)self)->v.get();
    {
    PyObject* pyobj_image = NULL;
    Mat image;
    bool retval;

    const char* keywords[] = { "image", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|O:VideoCapture.read", (char**)keywords, &pyobj_image) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 1)) )
    {
        ERRWRAP2(retval = _self_->read(image));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(image));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    bool retval;

    const char* keywords[] = { "image", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|O:VideoCapture.read", (char**)keywords, &pyobj_image) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 1)) )
    {
        ERRWRAP2(retval = _self_->read(image));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(image));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_VideoCapture_release(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_VideoCapture_Type))
        return failmsgp("Incorrect type of self (must be 'VideoCapture' or its derivative)");
    cv::VideoCapture* _self_ = ((pyopencv_VideoCapture_t*)self)->v.get();

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(_self_->release());
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_VideoCapture_retrieve(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_VideoCapture_Type))
        return failmsgp("Incorrect type of self (must be 'VideoCapture' or its derivative)");
    cv::VideoCapture* _self_ = ((pyopencv_VideoCapture_t*)self)->v.get();
    {
    PyObject* pyobj_image = NULL;
    Mat image;
    int flag=0;
    bool retval;

    const char* keywords[] = { "image", "flag", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|Oi:VideoCapture.retrieve", (char**)keywords, &pyobj_image, &flag) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 1)) )
    {
        ERRWRAP2(retval = _self_->retrieve(image, flag));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(image));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    int flag=0;
    bool retval;

    const char* keywords[] = { "image", "flag", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|Oi:VideoCapture.retrieve", (char**)keywords, &pyobj_image, &flag) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 1)) )
    {
        ERRWRAP2(retval = _self_->retrieve(image, flag));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(image));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_VideoCapture_set(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_VideoCapture_Type))
        return failmsgp("Incorrect type of self (must be 'VideoCapture' or its derivative)");
    cv::VideoCapture* _self_ = ((pyopencv_VideoCapture_t*)self)->v.get();
    int propId=0;
    double value=0;
    bool retval;

    const char* keywords[] = { "propId", "value", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "id:VideoCapture.set", (char**)keywords, &propId, &value) )
    {
        ERRWRAP2(retval = _self_->set(propId, value));
        return pyopencv_from(retval);
    }

    return NULL;
}



static PyMethodDef pyopencv_VideoCapture_methods[] =
{
    {"get", (PyCFunction)pyopencv_cv_VideoCapture_get, METH_VARARGS | METH_KEYWORDS, "get(propId) -> retval"},
    {"grab", (PyCFunction)pyopencv_cv_VideoCapture_grab, METH_VARARGS | METH_KEYWORDS, "grab() -> retval"},
    {"isOpened", (PyCFunction)pyopencv_cv_VideoCapture_isOpened, METH_VARARGS | METH_KEYWORDS, "isOpened() -> retval"},
    {"open", (PyCFunction)pyopencv_cv_VideoCapture_open, METH_VARARGS | METH_KEYWORDS, "open(filename) -> retval  or  open(index) -> retval  or  open(cameraNum, apiPreference) -> retval  or  open(filename, apiPreference) -> retval"},
    {"read", (PyCFunction)pyopencv_cv_VideoCapture_read, METH_VARARGS | METH_KEYWORDS, "read([, image]) -> retval, image"},
    {"release", (PyCFunction)pyopencv_cv_VideoCapture_release, METH_VARARGS | METH_KEYWORDS, "release() -> None"},
    {"retrieve", (PyCFunction)pyopencv_cv_VideoCapture_retrieve, METH_VARARGS | METH_KEYWORDS, "retrieve([, image[, flag]]) -> retval, image"},
    {"set", (PyCFunction)pyopencv_cv_VideoCapture_set, METH_VARARGS | METH_KEYWORDS, "set(propId, value) -> retval"},

    {NULL,          NULL}
};

static void pyopencv_VideoCapture_specials(void)
{
    pyopencv_VideoCapture_Type.tp_base = NULL;
    pyopencv_VideoCapture_Type.tp_dealloc = pyopencv_VideoCapture_dealloc;
    pyopencv_VideoCapture_Type.tp_repr = pyopencv_VideoCapture_repr;
    pyopencv_VideoCapture_Type.tp_getset = pyopencv_VideoCapture_getseters;
    pyopencv_VideoCapture_Type.tp_methods = pyopencv_VideoCapture_methods;
}

static PyObject* pyopencv_VideoWriter_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<VideoWriter %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_VideoWriter_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_VideoWriter_get(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_VideoWriter_Type))
        return failmsgp("Incorrect type of self (must be 'VideoWriter' or its derivative)");
    cv::VideoWriter* _self_ = ((pyopencv_VideoWriter_t*)self)->v.get();
    int propId=0;
    double retval;

    const char* keywords[] = { "propId", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:VideoWriter.get", (char**)keywords, &propId) )
    {
        ERRWRAP2(retval = _self_->get(propId));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_VideoWriter_isOpened(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_VideoWriter_Type))
        return failmsgp("Incorrect type of self (must be 'VideoWriter' or its derivative)");
    cv::VideoWriter* _self_ = ((pyopencv_VideoWriter_t*)self)->v.get();
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->isOpened());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_VideoWriter_open(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_VideoWriter_Type))
        return failmsgp("Incorrect type of self (must be 'VideoWriter' or its derivative)");
    cv::VideoWriter* _self_ = ((pyopencv_VideoWriter_t*)self)->v.get();
    PyObject* pyobj_filename = NULL;
    String filename;
    int fourcc=0;
    double fps=0;
    PyObject* pyobj_frameSize = NULL;
    Size frameSize;
    bool isColor=true;
    bool retval;

    const char* keywords[] = { "filename", "fourcc", "fps", "frameSize", "isColor", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OidO|b:VideoWriter.open", (char**)keywords, &pyobj_filename, &fourcc, &fps, &pyobj_frameSize, &isColor) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) &&
        pyopencv_to(pyobj_frameSize, frameSize, ArgInfo("frameSize", 0)) )
    {
        ERRWRAP2(retval = _self_->open(filename, fourcc, fps, frameSize, isColor));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_VideoWriter_release(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_VideoWriter_Type))
        return failmsgp("Incorrect type of self (must be 'VideoWriter' or its derivative)");
    cv::VideoWriter* _self_ = ((pyopencv_VideoWriter_t*)self)->v.get();

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(_self_->release());
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_VideoWriter_set(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_VideoWriter_Type))
        return failmsgp("Incorrect type of self (must be 'VideoWriter' or its derivative)");
    cv::VideoWriter* _self_ = ((pyopencv_VideoWriter_t*)self)->v.get();
    int propId=0;
    double value=0;
    bool retval;

    const char* keywords[] = { "propId", "value", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "id:VideoWriter.set", (char**)keywords, &propId, &value) )
    {
        ERRWRAP2(retval = _self_->set(propId, value));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_VideoWriter_write(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_VideoWriter_Type))
        return failmsgp("Incorrect type of self (must be 'VideoWriter' or its derivative)");
    cv::VideoWriter* _self_ = ((pyopencv_VideoWriter_t*)self)->v.get();
    {
    PyObject* pyobj_image = NULL;
    Mat image;

    const char* keywords[] = { "image", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:VideoWriter.write", (char**)keywords, &pyobj_image) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) )
    {
        ERRWRAP2(_self_->write(image));
        Py_RETURN_NONE;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    Mat image;

    const char* keywords[] = { "image", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:VideoWriter.write", (char**)keywords, &pyobj_image) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) )
    {
        ERRWRAP2(_self_->write(image));
        Py_RETURN_NONE;
    }
    }

    return NULL;
}



static PyMethodDef pyopencv_VideoWriter_methods[] =
{
    {"get", (PyCFunction)pyopencv_cv_VideoWriter_get, METH_VARARGS | METH_KEYWORDS, "get(propId) -> retval"},
    {"isOpened", (PyCFunction)pyopencv_cv_VideoWriter_isOpened, METH_VARARGS | METH_KEYWORDS, "isOpened() -> retval"},
    {"open", (PyCFunction)pyopencv_cv_VideoWriter_open, METH_VARARGS | METH_KEYWORDS, "open(filename, fourcc, fps, frameSize[, isColor]) -> retval"},
    {"release", (PyCFunction)pyopencv_cv_VideoWriter_release, METH_VARARGS | METH_KEYWORDS, "release() -> None"},
    {"set", (PyCFunction)pyopencv_cv_VideoWriter_set, METH_VARARGS | METH_KEYWORDS, "set(propId, value) -> retval"},
    {"write", (PyCFunction)pyopencv_cv_VideoWriter_write, METH_VARARGS | METH_KEYWORDS, "write(image) -> None"},

    {NULL,          NULL}
};

static void pyopencv_VideoWriter_specials(void)
{
    pyopencv_VideoWriter_Type.tp_base = NULL;
    pyopencv_VideoWriter_Type.tp_dealloc = pyopencv_VideoWriter_dealloc;
    pyopencv_VideoWriter_Type.tp_repr = pyopencv_VideoWriter_repr;
    pyopencv_VideoWriter_Type.tp_getset = pyopencv_VideoWriter_getseters;
    pyopencv_VideoWriter_Type.tp_methods = pyopencv_VideoWriter_methods;
}

static PyObject* pyopencv_BaseCascadeClassifier_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<BaseCascadeClassifier %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_BaseCascadeClassifier_getseters[] =
{
    {NULL}  /* Sentinel */
};



static PyMethodDef pyopencv_BaseCascadeClassifier_methods[] =
{

    {NULL,          NULL}
};

static void pyopencv_BaseCascadeClassifier_specials(void)
{
    pyopencv_BaseCascadeClassifier_Type.tp_base = &pyopencv_Algorithm_Type;
    pyopencv_BaseCascadeClassifier_Type.tp_dealloc = pyopencv_BaseCascadeClassifier_dealloc;
    pyopencv_BaseCascadeClassifier_Type.tp_repr = pyopencv_BaseCascadeClassifier_repr;
    pyopencv_BaseCascadeClassifier_Type.tp_getset = pyopencv_BaseCascadeClassifier_getseters;
    pyopencv_BaseCascadeClassifier_Type.tp_methods = pyopencv_BaseCascadeClassifier_methods;
}

static PyObject* pyopencv_CascadeClassifier_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<CascadeClassifier %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_CascadeClassifier_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_CascadeClassifier_detectMultiScale(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CascadeClassifier_Type))
        return failmsgp("Incorrect type of self (must be 'CascadeClassifier' or its derivative)");
    cv::CascadeClassifier* _self_ = ((pyopencv_CascadeClassifier_t*)self)->v.get();
    {
    PyObject* pyobj_image = NULL;
    Mat image;
    vector_Rect objects;
    double scaleFactor=1.1;
    int minNeighbors=3;
    int flags=0;
    PyObject* pyobj_minSize = NULL;
    Size minSize;
    PyObject* pyobj_maxSize = NULL;
    Size maxSize;

    const char* keywords[] = { "image", "scaleFactor", "minNeighbors", "flags", "minSize", "maxSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|diiOO:CascadeClassifier.detectMultiScale", (char**)keywords, &pyobj_image, &scaleFactor, &minNeighbors, &flags, &pyobj_minSize, &pyobj_maxSize) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_minSize, minSize, ArgInfo("minSize", 0)) &&
        pyopencv_to(pyobj_maxSize, maxSize, ArgInfo("maxSize", 0)) )
    {
        ERRWRAP2(_self_->detectMultiScale(image, objects, scaleFactor, minNeighbors, flags, minSize, maxSize));
        return pyopencv_from(objects);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    vector_Rect objects;
    double scaleFactor=1.1;
    int minNeighbors=3;
    int flags=0;
    PyObject* pyobj_minSize = NULL;
    Size minSize;
    PyObject* pyobj_maxSize = NULL;
    Size maxSize;

    const char* keywords[] = { "image", "scaleFactor", "minNeighbors", "flags", "minSize", "maxSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|diiOO:CascadeClassifier.detectMultiScale", (char**)keywords, &pyobj_image, &scaleFactor, &minNeighbors, &flags, &pyobj_minSize, &pyobj_maxSize) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_minSize, minSize, ArgInfo("minSize", 0)) &&
        pyopencv_to(pyobj_maxSize, maxSize, ArgInfo("maxSize", 0)) )
    {
        ERRWRAP2(_self_->detectMultiScale(image, objects, scaleFactor, minNeighbors, flags, minSize, maxSize));
        return pyopencv_from(objects);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_CascadeClassifier_detectMultiScale2(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CascadeClassifier_Type))
        return failmsgp("Incorrect type of self (must be 'CascadeClassifier' or its derivative)");
    cv::CascadeClassifier* _self_ = ((pyopencv_CascadeClassifier_t*)self)->v.get();
    {
    PyObject* pyobj_image = NULL;
    Mat image;
    vector_Rect objects;
    vector_int numDetections;
    double scaleFactor=1.1;
    int minNeighbors=3;
    int flags=0;
    PyObject* pyobj_minSize = NULL;
    Size minSize;
    PyObject* pyobj_maxSize = NULL;
    Size maxSize;

    const char* keywords[] = { "image", "scaleFactor", "minNeighbors", "flags", "minSize", "maxSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|diiOO:CascadeClassifier.detectMultiScale2", (char**)keywords, &pyobj_image, &scaleFactor, &minNeighbors, &flags, &pyobj_minSize, &pyobj_maxSize) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_minSize, minSize, ArgInfo("minSize", 0)) &&
        pyopencv_to(pyobj_maxSize, maxSize, ArgInfo("maxSize", 0)) )
    {
        ERRWRAP2(_self_->detectMultiScale(image, objects, numDetections, scaleFactor, minNeighbors, flags, minSize, maxSize));
        return Py_BuildValue("(NN)", pyopencv_from(objects), pyopencv_from(numDetections));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    vector_Rect objects;
    vector_int numDetections;
    double scaleFactor=1.1;
    int minNeighbors=3;
    int flags=0;
    PyObject* pyobj_minSize = NULL;
    Size minSize;
    PyObject* pyobj_maxSize = NULL;
    Size maxSize;

    const char* keywords[] = { "image", "scaleFactor", "minNeighbors", "flags", "minSize", "maxSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|diiOO:CascadeClassifier.detectMultiScale2", (char**)keywords, &pyobj_image, &scaleFactor, &minNeighbors, &flags, &pyobj_minSize, &pyobj_maxSize) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_minSize, minSize, ArgInfo("minSize", 0)) &&
        pyopencv_to(pyobj_maxSize, maxSize, ArgInfo("maxSize", 0)) )
    {
        ERRWRAP2(_self_->detectMultiScale(image, objects, numDetections, scaleFactor, minNeighbors, flags, minSize, maxSize));
        return Py_BuildValue("(NN)", pyopencv_from(objects), pyopencv_from(numDetections));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_CascadeClassifier_detectMultiScale3(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CascadeClassifier_Type))
        return failmsgp("Incorrect type of self (must be 'CascadeClassifier' or its derivative)");
    cv::CascadeClassifier* _self_ = ((pyopencv_CascadeClassifier_t*)self)->v.get();
    {
    PyObject* pyobj_image = NULL;
    Mat image;
    vector_Rect objects;
    vector_int rejectLevels;
    vector_double levelWeights;
    double scaleFactor=1.1;
    int minNeighbors=3;
    int flags=0;
    PyObject* pyobj_minSize = NULL;
    Size minSize;
    PyObject* pyobj_maxSize = NULL;
    Size maxSize;
    bool outputRejectLevels=false;

    const char* keywords[] = { "image", "scaleFactor", "minNeighbors", "flags", "minSize", "maxSize", "outputRejectLevels", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|diiOOb:CascadeClassifier.detectMultiScale3", (char**)keywords, &pyobj_image, &scaleFactor, &minNeighbors, &flags, &pyobj_minSize, &pyobj_maxSize, &outputRejectLevels) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_minSize, minSize, ArgInfo("minSize", 0)) &&
        pyopencv_to(pyobj_maxSize, maxSize, ArgInfo("maxSize", 0)) )
    {
        ERRWRAP2(_self_->detectMultiScale(image, objects, rejectLevels, levelWeights, scaleFactor, minNeighbors, flags, minSize, maxSize, outputRejectLevels));
        return Py_BuildValue("(NNN)", pyopencv_from(objects), pyopencv_from(rejectLevels), pyopencv_from(levelWeights));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    vector_Rect objects;
    vector_int rejectLevels;
    vector_double levelWeights;
    double scaleFactor=1.1;
    int minNeighbors=3;
    int flags=0;
    PyObject* pyobj_minSize = NULL;
    Size minSize;
    PyObject* pyobj_maxSize = NULL;
    Size maxSize;
    bool outputRejectLevels=false;

    const char* keywords[] = { "image", "scaleFactor", "minNeighbors", "flags", "minSize", "maxSize", "outputRejectLevels", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|diiOOb:CascadeClassifier.detectMultiScale3", (char**)keywords, &pyobj_image, &scaleFactor, &minNeighbors, &flags, &pyobj_minSize, &pyobj_maxSize, &outputRejectLevels) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_minSize, minSize, ArgInfo("minSize", 0)) &&
        pyopencv_to(pyobj_maxSize, maxSize, ArgInfo("maxSize", 0)) )
    {
        ERRWRAP2(_self_->detectMultiScale(image, objects, rejectLevels, levelWeights, scaleFactor, minNeighbors, flags, minSize, maxSize, outputRejectLevels));
        return Py_BuildValue("(NNN)", pyopencv_from(objects), pyopencv_from(rejectLevels), pyopencv_from(levelWeights));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_CascadeClassifier_empty(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CascadeClassifier_Type))
        return failmsgp("Incorrect type of self (must be 'CascadeClassifier' or its derivative)");
    cv::CascadeClassifier* _self_ = ((pyopencv_CascadeClassifier_t*)self)->v.get();
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->empty());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_CascadeClassifier_getFeatureType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CascadeClassifier_Type))
        return failmsgp("Incorrect type of self (must be 'CascadeClassifier' or its derivative)");
    cv::CascadeClassifier* _self_ = ((pyopencv_CascadeClassifier_t*)self)->v.get();
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getFeatureType());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_CascadeClassifier_getOriginalWindowSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CascadeClassifier_Type))
        return failmsgp("Incorrect type of self (must be 'CascadeClassifier' or its derivative)");
    cv::CascadeClassifier* _self_ = ((pyopencv_CascadeClassifier_t*)self)->v.get();
    Size retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getOriginalWindowSize());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_CascadeClassifier_isOldFormatCascade(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CascadeClassifier_Type))
        return failmsgp("Incorrect type of self (must be 'CascadeClassifier' or its derivative)");
    cv::CascadeClassifier* _self_ = ((pyopencv_CascadeClassifier_t*)self)->v.get();
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->isOldFormatCascade());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_CascadeClassifier_load(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CascadeClassifier_Type))
        return failmsgp("Incorrect type of self (must be 'CascadeClassifier' or its derivative)");
    cv::CascadeClassifier* _self_ = ((pyopencv_CascadeClassifier_t*)self)->v.get();
    PyObject* pyobj_filename = NULL;
    String filename;
    bool retval;

    const char* keywords[] = { "filename", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:CascadeClassifier.load", (char**)keywords, &pyobj_filename) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) )
    {
        ERRWRAP2(retval = _self_->load(filename));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_CascadeClassifier_read(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_CascadeClassifier_Type))
        return failmsgp("Incorrect type of self (must be 'CascadeClassifier' or its derivative)");
    cv::CascadeClassifier* _self_ = ((pyopencv_CascadeClassifier_t*)self)->v.get();
    PyObject* pyobj_node = NULL;
    FileNode node;
    bool retval;

    const char* keywords[] = { "node", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:CascadeClassifier.read", (char**)keywords, &pyobj_node) &&
        pyopencv_to(pyobj_node, node, ArgInfo("node", 0)) )
    {
        ERRWRAP2(retval = _self_->read(node));
        return pyopencv_from(retval);
    }

    return NULL;
}



static PyMethodDef pyopencv_CascadeClassifier_methods[] =
{
    {"detectMultiScale", (PyCFunction)pyopencv_cv_CascadeClassifier_detectMultiScale, METH_VARARGS | METH_KEYWORDS, "detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]]) -> objects"},
    {"detectMultiScale2", (PyCFunction)pyopencv_cv_CascadeClassifier_detectMultiScale2, METH_VARARGS | METH_KEYWORDS, "detectMultiScale2(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]]) -> objects, numDetections"},
    {"detectMultiScale3", (PyCFunction)pyopencv_cv_CascadeClassifier_detectMultiScale3, METH_VARARGS | METH_KEYWORDS, "detectMultiScale3(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize[, outputRejectLevels]]]]]]) -> objects, rejectLevels, levelWeights"},
    {"empty", (PyCFunction)pyopencv_cv_CascadeClassifier_empty, METH_VARARGS | METH_KEYWORDS, "empty() -> retval"},
    {"getFeatureType", (PyCFunction)pyopencv_cv_CascadeClassifier_getFeatureType, METH_VARARGS | METH_KEYWORDS, "getFeatureType() -> retval"},
    {"getOriginalWindowSize", (PyCFunction)pyopencv_cv_CascadeClassifier_getOriginalWindowSize, METH_VARARGS | METH_KEYWORDS, "getOriginalWindowSize() -> retval"},
    {"isOldFormatCascade", (PyCFunction)pyopencv_cv_CascadeClassifier_isOldFormatCascade, METH_VARARGS | METH_KEYWORDS, "isOldFormatCascade() -> retval"},
    {"load", (PyCFunction)pyopencv_cv_CascadeClassifier_load, METH_VARARGS | METH_KEYWORDS, "load(filename) -> retval"},
    {"read", (PyCFunction)pyopencv_cv_CascadeClassifier_read, METH_VARARGS | METH_KEYWORDS, "read(node) -> retval"},

    {NULL,          NULL}
};

static void pyopencv_CascadeClassifier_specials(void)
{
    pyopencv_CascadeClassifier_Type.tp_base = NULL;
    pyopencv_CascadeClassifier_Type.tp_dealloc = pyopencv_CascadeClassifier_dealloc;
    pyopencv_CascadeClassifier_Type.tp_repr = pyopencv_CascadeClassifier_repr;
    pyopencv_CascadeClassifier_Type.tp_getset = pyopencv_CascadeClassifier_getseters;
    pyopencv_CascadeClassifier_Type.tp_methods = pyopencv_CascadeClassifier_methods;
}

static PyObject* pyopencv_HOGDescriptor_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<HOGDescriptor %p>", self);
    return PyString_FromString(str);
}


static PyObject* pyopencv_HOGDescriptor_get_L2HysThreshold(pyopencv_HOGDescriptor_t* p, void *closure)
{
    return pyopencv_from(p->v->L2HysThreshold);
}

static PyObject* pyopencv_HOGDescriptor_get_blockSize(pyopencv_HOGDescriptor_t* p, void *closure)
{
    return pyopencv_from(p->v->blockSize);
}

static PyObject* pyopencv_HOGDescriptor_get_blockStride(pyopencv_HOGDescriptor_t* p, void *closure)
{
    return pyopencv_from(p->v->blockStride);
}

static PyObject* pyopencv_HOGDescriptor_get_cellSize(pyopencv_HOGDescriptor_t* p, void *closure)
{
    return pyopencv_from(p->v->cellSize);
}

static PyObject* pyopencv_HOGDescriptor_get_derivAperture(pyopencv_HOGDescriptor_t* p, void *closure)
{
    return pyopencv_from(p->v->derivAperture);
}

static PyObject* pyopencv_HOGDescriptor_get_gammaCorrection(pyopencv_HOGDescriptor_t* p, void *closure)
{
    return pyopencv_from(p->v->gammaCorrection);
}

static PyObject* pyopencv_HOGDescriptor_get_histogramNormType(pyopencv_HOGDescriptor_t* p, void *closure)
{
    return pyopencv_from(p->v->histogramNormType);
}

static PyObject* pyopencv_HOGDescriptor_get_nbins(pyopencv_HOGDescriptor_t* p, void *closure)
{
    return pyopencv_from(p->v->nbins);
}

static PyObject* pyopencv_HOGDescriptor_get_nlevels(pyopencv_HOGDescriptor_t* p, void *closure)
{
    return pyopencv_from(p->v->nlevels);
}

static PyObject* pyopencv_HOGDescriptor_get_signedGradient(pyopencv_HOGDescriptor_t* p, void *closure)
{
    return pyopencv_from(p->v->signedGradient);
}

static PyObject* pyopencv_HOGDescriptor_get_svmDetector(pyopencv_HOGDescriptor_t* p, void *closure)
{
    return pyopencv_from(p->v->svmDetector);
}

static PyObject* pyopencv_HOGDescriptor_get_winSigma(pyopencv_HOGDescriptor_t* p, void *closure)
{
    return pyopencv_from(p->v->winSigma);
}

static PyObject* pyopencv_HOGDescriptor_get_winSize(pyopencv_HOGDescriptor_t* p, void *closure)
{
    return pyopencv_from(p->v->winSize);
}


static PyGetSetDef pyopencv_HOGDescriptor_getseters[] =
{
    {(char*)"L2HysThreshold", (getter)pyopencv_HOGDescriptor_get_L2HysThreshold, NULL, (char*)"L2HysThreshold", NULL},
    {(char*)"blockSize", (getter)pyopencv_HOGDescriptor_get_blockSize, NULL, (char*)"blockSize", NULL},
    {(char*)"blockStride", (getter)pyopencv_HOGDescriptor_get_blockStride, NULL, (char*)"blockStride", NULL},
    {(char*)"cellSize", (getter)pyopencv_HOGDescriptor_get_cellSize, NULL, (char*)"cellSize", NULL},
    {(char*)"derivAperture", (getter)pyopencv_HOGDescriptor_get_derivAperture, NULL, (char*)"derivAperture", NULL},
    {(char*)"gammaCorrection", (getter)pyopencv_HOGDescriptor_get_gammaCorrection, NULL, (char*)"gammaCorrection", NULL},
    {(char*)"histogramNormType", (getter)pyopencv_HOGDescriptor_get_histogramNormType, NULL, (char*)"histogramNormType", NULL},
    {(char*)"nbins", (getter)pyopencv_HOGDescriptor_get_nbins, NULL, (char*)"nbins", NULL},
    {(char*)"nlevels", (getter)pyopencv_HOGDescriptor_get_nlevels, NULL, (char*)"nlevels", NULL},
    {(char*)"signedGradient", (getter)pyopencv_HOGDescriptor_get_signedGradient, NULL, (char*)"signedGradient", NULL},
    {(char*)"svmDetector", (getter)pyopencv_HOGDescriptor_get_svmDetector, NULL, (char*)"svmDetector", NULL},
    {(char*)"winSigma", (getter)pyopencv_HOGDescriptor_get_winSigma, NULL, (char*)"winSigma", NULL},
    {(char*)"winSize", (getter)pyopencv_HOGDescriptor_get_winSize, NULL, (char*)"winSize", NULL},
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_HOGDescriptor_checkDetectorSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_HOGDescriptor_Type))
        return failmsgp("Incorrect type of self (must be 'HOGDescriptor' or its derivative)");
    cv::HOGDescriptor* _self_ = ((pyopencv_HOGDescriptor_t*)self)->v.get();
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->checkDetectorSize());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_HOGDescriptor_compute(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_HOGDescriptor_Type))
        return failmsgp("Incorrect type of self (must be 'HOGDescriptor' or its derivative)");
    cv::HOGDescriptor* _self_ = ((pyopencv_HOGDescriptor_t*)self)->v.get();
    {
    PyObject* pyobj_img = NULL;
    Mat img;
    vector_float descriptors;
    PyObject* pyobj_winStride = NULL;
    Size winStride;
    PyObject* pyobj_padding = NULL;
    Size padding;
    PyObject* pyobj_locations = NULL;
    vector_Point locations=std::vector<Point>();

    const char* keywords[] = { "img", "winStride", "padding", "locations", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOO:HOGDescriptor.compute", (char**)keywords, &pyobj_img, &pyobj_winStride, &pyobj_padding, &pyobj_locations) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_winStride, winStride, ArgInfo("winStride", 0)) &&
        pyopencv_to(pyobj_padding, padding, ArgInfo("padding", 0)) &&
        pyopencv_to(pyobj_locations, locations, ArgInfo("locations", 0)) )
    {
        ERRWRAP2(_self_->compute(img, descriptors, winStride, padding, locations));
        return pyopencv_from(descriptors);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_img = NULL;
    UMat img;
    vector_float descriptors;
    PyObject* pyobj_winStride = NULL;
    Size winStride;
    PyObject* pyobj_padding = NULL;
    Size padding;
    PyObject* pyobj_locations = NULL;
    vector_Point locations=std::vector<Point>();

    const char* keywords[] = { "img", "winStride", "padding", "locations", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOO:HOGDescriptor.compute", (char**)keywords, &pyobj_img, &pyobj_winStride, &pyobj_padding, &pyobj_locations) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_winStride, winStride, ArgInfo("winStride", 0)) &&
        pyopencv_to(pyobj_padding, padding, ArgInfo("padding", 0)) &&
        pyopencv_to(pyobj_locations, locations, ArgInfo("locations", 0)) )
    {
        ERRWRAP2(_self_->compute(img, descriptors, winStride, padding, locations));
        return pyopencv_from(descriptors);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_HOGDescriptor_computeGradient(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_HOGDescriptor_Type))
        return failmsgp("Incorrect type of self (must be 'HOGDescriptor' or its derivative)");
    cv::HOGDescriptor* _self_ = ((pyopencv_HOGDescriptor_t*)self)->v.get();
    {
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_grad = NULL;
    Mat grad;
    PyObject* pyobj_angleOfs = NULL;
    Mat angleOfs;
    PyObject* pyobj_paddingTL = NULL;
    Size paddingTL;
    PyObject* pyobj_paddingBR = NULL;
    Size paddingBR;

    const char* keywords[] = { "img", "grad", "angleOfs", "paddingTL", "paddingBR", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOOO:HOGDescriptor.computeGradient", (char**)keywords, &pyobj_img, &pyobj_grad, &pyobj_angleOfs, &pyobj_paddingTL, &pyobj_paddingBR) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_grad, grad, ArgInfo("grad", 1)) &&
        pyopencv_to(pyobj_angleOfs, angleOfs, ArgInfo("angleOfs", 1)) &&
        pyopencv_to(pyobj_paddingTL, paddingTL, ArgInfo("paddingTL", 0)) &&
        pyopencv_to(pyobj_paddingBR, paddingBR, ArgInfo("paddingBR", 0)) )
    {
        ERRWRAP2(_self_->computeGradient(img, grad, angleOfs, paddingTL, paddingBR));
        return Py_BuildValue("(NN)", pyopencv_from(grad), pyopencv_from(angleOfs));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_img = NULL;
    Mat img;
    PyObject* pyobj_grad = NULL;
    Mat grad;
    PyObject* pyobj_angleOfs = NULL;
    Mat angleOfs;
    PyObject* pyobj_paddingTL = NULL;
    Size paddingTL;
    PyObject* pyobj_paddingBR = NULL;
    Size paddingBR;

    const char* keywords[] = { "img", "grad", "angleOfs", "paddingTL", "paddingBR", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|OOOO:HOGDescriptor.computeGradient", (char**)keywords, &pyobj_img, &pyobj_grad, &pyobj_angleOfs, &pyobj_paddingTL, &pyobj_paddingBR) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_grad, grad, ArgInfo("grad", 1)) &&
        pyopencv_to(pyobj_angleOfs, angleOfs, ArgInfo("angleOfs", 1)) &&
        pyopencv_to(pyobj_paddingTL, paddingTL, ArgInfo("paddingTL", 0)) &&
        pyopencv_to(pyobj_paddingBR, paddingBR, ArgInfo("paddingBR", 0)) )
    {
        ERRWRAP2(_self_->computeGradient(img, grad, angleOfs, paddingTL, paddingBR));
        return Py_BuildValue("(NN)", pyopencv_from(grad), pyopencv_from(angleOfs));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_HOGDescriptor_detect(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_HOGDescriptor_Type))
        return failmsgp("Incorrect type of self (must be 'HOGDescriptor' or its derivative)");
    cv::HOGDescriptor* _self_ = ((pyopencv_HOGDescriptor_t*)self)->v.get();
    {
    PyObject* pyobj_img = NULL;
    Mat img;
    vector_Point foundLocations;
    vector_double weights;
    double hitThreshold=0;
    PyObject* pyobj_winStride = NULL;
    Size winStride;
    PyObject* pyobj_padding = NULL;
    Size padding;
    PyObject* pyobj_searchLocations = NULL;
    vector_Point searchLocations=std::vector<Point>();

    const char* keywords[] = { "img", "hitThreshold", "winStride", "padding", "searchLocations", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|dOOO:HOGDescriptor.detect", (char**)keywords, &pyobj_img, &hitThreshold, &pyobj_winStride, &pyobj_padding, &pyobj_searchLocations) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_winStride, winStride, ArgInfo("winStride", 0)) &&
        pyopencv_to(pyobj_padding, padding, ArgInfo("padding", 0)) &&
        pyopencv_to(pyobj_searchLocations, searchLocations, ArgInfo("searchLocations", 0)) )
    {
        ERRWRAP2(_self_->detect(img, foundLocations, weights, hitThreshold, winStride, padding, searchLocations));
        return Py_BuildValue("(NN)", pyopencv_from(foundLocations), pyopencv_from(weights));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_img = NULL;
    Mat img;
    vector_Point foundLocations;
    vector_double weights;
    double hitThreshold=0;
    PyObject* pyobj_winStride = NULL;
    Size winStride;
    PyObject* pyobj_padding = NULL;
    Size padding;
    PyObject* pyobj_searchLocations = NULL;
    vector_Point searchLocations=std::vector<Point>();

    const char* keywords[] = { "img", "hitThreshold", "winStride", "padding", "searchLocations", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|dOOO:HOGDescriptor.detect", (char**)keywords, &pyobj_img, &hitThreshold, &pyobj_winStride, &pyobj_padding, &pyobj_searchLocations) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_winStride, winStride, ArgInfo("winStride", 0)) &&
        pyopencv_to(pyobj_padding, padding, ArgInfo("padding", 0)) &&
        pyopencv_to(pyobj_searchLocations, searchLocations, ArgInfo("searchLocations", 0)) )
    {
        ERRWRAP2(_self_->detect(img, foundLocations, weights, hitThreshold, winStride, padding, searchLocations));
        return Py_BuildValue("(NN)", pyopencv_from(foundLocations), pyopencv_from(weights));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_HOGDescriptor_detectMultiScale(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_HOGDescriptor_Type))
        return failmsgp("Incorrect type of self (must be 'HOGDescriptor' or its derivative)");
    cv::HOGDescriptor* _self_ = ((pyopencv_HOGDescriptor_t*)self)->v.get();
    {
    PyObject* pyobj_img = NULL;
    Mat img;
    vector_Rect foundLocations;
    vector_double foundWeights;
    double hitThreshold=0;
    PyObject* pyobj_winStride = NULL;
    Size winStride;
    PyObject* pyobj_padding = NULL;
    Size padding;
    double scale=1.05;
    double finalThreshold=2.0;
    bool useMeanshiftGrouping=false;

    const char* keywords[] = { "img", "hitThreshold", "winStride", "padding", "scale", "finalThreshold", "useMeanshiftGrouping", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|dOOddb:HOGDescriptor.detectMultiScale", (char**)keywords, &pyobj_img, &hitThreshold, &pyobj_winStride, &pyobj_padding, &scale, &finalThreshold, &useMeanshiftGrouping) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_winStride, winStride, ArgInfo("winStride", 0)) &&
        pyopencv_to(pyobj_padding, padding, ArgInfo("padding", 0)) )
    {
        ERRWRAP2(_self_->detectMultiScale(img, foundLocations, foundWeights, hitThreshold, winStride, padding, scale, finalThreshold, useMeanshiftGrouping));
        return Py_BuildValue("(NN)", pyopencv_from(foundLocations), pyopencv_from(foundWeights));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_img = NULL;
    UMat img;
    vector_Rect foundLocations;
    vector_double foundWeights;
    double hitThreshold=0;
    PyObject* pyobj_winStride = NULL;
    Size winStride;
    PyObject* pyobj_padding = NULL;
    Size padding;
    double scale=1.05;
    double finalThreshold=2.0;
    bool useMeanshiftGrouping=false;

    const char* keywords[] = { "img", "hitThreshold", "winStride", "padding", "scale", "finalThreshold", "useMeanshiftGrouping", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|dOOddb:HOGDescriptor.detectMultiScale", (char**)keywords, &pyobj_img, &hitThreshold, &pyobj_winStride, &pyobj_padding, &scale, &finalThreshold, &useMeanshiftGrouping) &&
        pyopencv_to(pyobj_img, img, ArgInfo("img", 0)) &&
        pyopencv_to(pyobj_winStride, winStride, ArgInfo("winStride", 0)) &&
        pyopencv_to(pyobj_padding, padding, ArgInfo("padding", 0)) )
    {
        ERRWRAP2(_self_->detectMultiScale(img, foundLocations, foundWeights, hitThreshold, winStride, padding, scale, finalThreshold, useMeanshiftGrouping));
        return Py_BuildValue("(NN)", pyopencv_from(foundLocations), pyopencv_from(foundWeights));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_HOGDescriptor_getDescriptorSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_HOGDescriptor_Type))
        return failmsgp("Incorrect type of self (must be 'HOGDescriptor' or its derivative)");
    cv::HOGDescriptor* _self_ = ((pyopencv_HOGDescriptor_t*)self)->v.get();
    size_t retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getDescriptorSize());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_HOGDescriptor_getWinSigma(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_HOGDescriptor_Type))
        return failmsgp("Incorrect type of self (must be 'HOGDescriptor' or its derivative)");
    cv::HOGDescriptor* _self_ = ((pyopencv_HOGDescriptor_t*)self)->v.get();
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getWinSigma());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_HOGDescriptor_load(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_HOGDescriptor_Type))
        return failmsgp("Incorrect type of self (must be 'HOGDescriptor' or its derivative)");
    cv::HOGDescriptor* _self_ = ((pyopencv_HOGDescriptor_t*)self)->v.get();
    PyObject* pyobj_filename = NULL;
    String filename;
    PyObject* pyobj_objname = NULL;
    String objname;
    bool retval;

    const char* keywords[] = { "filename", "objname", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:HOGDescriptor.load", (char**)keywords, &pyobj_filename, &pyobj_objname) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) &&
        pyopencv_to(pyobj_objname, objname, ArgInfo("objname", 0)) )
    {
        ERRWRAP2(retval = _self_->load(filename, objname));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_HOGDescriptor_save(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_HOGDescriptor_Type))
        return failmsgp("Incorrect type of self (must be 'HOGDescriptor' or its derivative)");
    cv::HOGDescriptor* _self_ = ((pyopencv_HOGDescriptor_t*)self)->v.get();
    PyObject* pyobj_filename = NULL;
    String filename;
    PyObject* pyobj_objname = NULL;
    String objname;

    const char* keywords[] = { "filename", "objname", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:HOGDescriptor.save", (char**)keywords, &pyobj_filename, &pyobj_objname) &&
        pyopencv_to(pyobj_filename, filename, ArgInfo("filename", 0)) &&
        pyopencv_to(pyobj_objname, objname, ArgInfo("objname", 0)) )
    {
        ERRWRAP2(_self_->save(filename, objname));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_HOGDescriptor_setSVMDetector(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_HOGDescriptor_Type))
        return failmsgp("Incorrect type of self (must be 'HOGDescriptor' or its derivative)");
    cv::HOGDescriptor* _self_ = ((pyopencv_HOGDescriptor_t*)self)->v.get();
    {
    PyObject* pyobj__svmdetector = NULL;
    Mat _svmdetector;

    const char* keywords[] = { "_svmdetector", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:HOGDescriptor.setSVMDetector", (char**)keywords, &pyobj__svmdetector) &&
        pyopencv_to(pyobj__svmdetector, _svmdetector, ArgInfo("_svmdetector", 0)) )
    {
        ERRWRAP2(_self_->setSVMDetector(_svmdetector));
        Py_RETURN_NONE;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj__svmdetector = NULL;
    UMat _svmdetector;

    const char* keywords[] = { "_svmdetector", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:HOGDescriptor.setSVMDetector", (char**)keywords, &pyobj__svmdetector) &&
        pyopencv_to(pyobj__svmdetector, _svmdetector, ArgInfo("_svmdetector", 0)) )
    {
        ERRWRAP2(_self_->setSVMDetector(_svmdetector));
        Py_RETURN_NONE;
    }
    }

    return NULL;
}



static PyMethodDef pyopencv_HOGDescriptor_methods[] =
{
    {"checkDetectorSize", (PyCFunction)pyopencv_cv_HOGDescriptor_checkDetectorSize, METH_VARARGS | METH_KEYWORDS, "checkDetectorSize() -> retval"},
    {"compute", (PyCFunction)pyopencv_cv_HOGDescriptor_compute, METH_VARARGS | METH_KEYWORDS, "compute(img[, winStride[, padding[, locations]]]) -> descriptors"},
    {"computeGradient", (PyCFunction)pyopencv_cv_HOGDescriptor_computeGradient, METH_VARARGS | METH_KEYWORDS, "computeGradient(img[, grad[, angleOfs[, paddingTL[, paddingBR]]]]) -> grad, angleOfs"},
    {"detect", (PyCFunction)pyopencv_cv_HOGDescriptor_detect, METH_VARARGS | METH_KEYWORDS, "detect(img[, hitThreshold[, winStride[, padding[, searchLocations]]]]) -> foundLocations, weights"},
    {"detectMultiScale", (PyCFunction)pyopencv_cv_HOGDescriptor_detectMultiScale, METH_VARARGS | METH_KEYWORDS, "detectMultiScale(img[, hitThreshold[, winStride[, padding[, scale[, finalThreshold[, useMeanshiftGrouping]]]]]]) -> foundLocations, foundWeights"},
    {"getDescriptorSize", (PyCFunction)pyopencv_cv_HOGDescriptor_getDescriptorSize, METH_VARARGS | METH_KEYWORDS, "getDescriptorSize() -> retval"},
    {"getWinSigma", (PyCFunction)pyopencv_cv_HOGDescriptor_getWinSigma, METH_VARARGS | METH_KEYWORDS, "getWinSigma() -> retval"},
    {"load", (PyCFunction)pyopencv_cv_HOGDescriptor_load, METH_VARARGS | METH_KEYWORDS, "load(filename[, objname]) -> retval"},
    {"save", (PyCFunction)pyopencv_cv_HOGDescriptor_save, METH_VARARGS | METH_KEYWORDS, "save(filename[, objname]) -> None"},
    {"setSVMDetector", (PyCFunction)pyopencv_cv_HOGDescriptor_setSVMDetector, METH_VARARGS | METH_KEYWORDS, "setSVMDetector(_svmdetector) -> None"},

    {NULL,          NULL}
};

static void pyopencv_HOGDescriptor_specials(void)
{
    pyopencv_HOGDescriptor_Type.tp_base = NULL;
    pyopencv_HOGDescriptor_Type.tp_dealloc = pyopencv_HOGDescriptor_dealloc;
    pyopencv_HOGDescriptor_Type.tp_repr = pyopencv_HOGDescriptor_repr;
    pyopencv_HOGDescriptor_Type.tp_getset = pyopencv_HOGDescriptor_getseters;
    pyopencv_HOGDescriptor_Type.tp_methods = pyopencv_HOGDescriptor_methods;
}

static PyObject* pyopencv_Feature2D_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<Feature2D %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_Feature2D_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_Feature2D_compute(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Feature2D_Type))
        return failmsgp("Incorrect type of self (must be 'Feature2D' or its derivative)");
    cv::Feature2D* _self_ = dynamic_cast<cv::Feature2D*>(((pyopencv_Feature2D_t*)self)->v.get());
    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_keypoints = NULL;
    vector_KeyPoint keypoints;
    PyObject* pyobj_descriptors = NULL;
    Mat descriptors;

    const char* keywords[] = { "image", "keypoints", "descriptors", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:Feature2D.compute", (char**)keywords, &pyobj_image, &pyobj_keypoints, &pyobj_descriptors) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_keypoints, keypoints, ArgInfo("keypoints", 1)) &&
        pyopencv_to(pyobj_descriptors, descriptors, ArgInfo("descriptors", 1)) )
    {
        ERRWRAP2(_self_->compute(image, keypoints, descriptors));
        return Py_BuildValue("(NN)", pyopencv_from(keypoints), pyopencv_from(descriptors));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_keypoints = NULL;
    vector_KeyPoint keypoints;
    PyObject* pyobj_descriptors = NULL;
    UMat descriptors;

    const char* keywords[] = { "image", "keypoints", "descriptors", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:Feature2D.compute", (char**)keywords, &pyobj_image, &pyobj_keypoints, &pyobj_descriptors) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_keypoints, keypoints, ArgInfo("keypoints", 1)) &&
        pyopencv_to(pyobj_descriptors, descriptors, ArgInfo("descriptors", 1)) )
    {
        ERRWRAP2(_self_->compute(image, keypoints, descriptors));
        return Py_BuildValue("(NN)", pyopencv_from(keypoints), pyopencv_from(descriptors));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_images = NULL;
    vector_Mat images;
    PyObject* pyobj_keypoints = NULL;
    vector_vector_KeyPoint keypoints;
    PyObject* pyobj_descriptors = NULL;
    vector_Mat descriptors;

    const char* keywords[] = { "images", "keypoints", "descriptors", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:Feature2D.compute", (char**)keywords, &pyobj_images, &pyobj_keypoints, &pyobj_descriptors) &&
        pyopencv_to(pyobj_images, images, ArgInfo("images", 0)) &&
        pyopencv_to(pyobj_keypoints, keypoints, ArgInfo("keypoints", 1)) &&
        pyopencv_to(pyobj_descriptors, descriptors, ArgInfo("descriptors", 1)) )
    {
        ERRWRAP2(_self_->compute(images, keypoints, descriptors));
        return Py_BuildValue("(NN)", pyopencv_from(keypoints), pyopencv_from(descriptors));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_images = NULL;
    vector_Mat images;
    PyObject* pyobj_keypoints = NULL;
    vector_vector_KeyPoint keypoints;
    PyObject* pyobj_descriptors = NULL;
    vector_Mat descriptors;

    const char* keywords[] = { "images", "keypoints", "descriptors", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:Feature2D.compute", (char**)keywords, &pyobj_images, &pyobj_keypoints, &pyobj_descriptors) &&
        pyopencv_to(pyobj_images, images, ArgInfo("images", 0)) &&
        pyopencv_to(pyobj_keypoints, keypoints, ArgInfo("keypoints", 1)) &&
        pyopencv_to(pyobj_descriptors, descriptors, ArgInfo("descriptors", 1)) )
    {
        ERRWRAP2(_self_->compute(images, keypoints, descriptors));
        return Py_BuildValue("(NN)", pyopencv_from(keypoints), pyopencv_from(descriptors));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_Feature2D_defaultNorm(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Feature2D_Type))
        return failmsgp("Incorrect type of self (must be 'Feature2D' or its derivative)");
    cv::Feature2D* _self_ = dynamic_cast<cv::Feature2D*>(((pyopencv_Feature2D_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->defaultNorm());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_Feature2D_descriptorSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Feature2D_Type))
        return failmsgp("Incorrect type of self (must be 'Feature2D' or its derivative)");
    cv::Feature2D* _self_ = dynamic_cast<cv::Feature2D*>(((pyopencv_Feature2D_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->descriptorSize());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_Feature2D_descriptorType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Feature2D_Type))
        return failmsgp("Incorrect type of self (must be 'Feature2D' or its derivative)");
    cv::Feature2D* _self_ = dynamic_cast<cv::Feature2D*>(((pyopencv_Feature2D_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->descriptorType());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_Feature2D_detect(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Feature2D_Type))
        return failmsgp("Incorrect type of self (must be 'Feature2D' or its derivative)");
    cv::Feature2D* _self_ = dynamic_cast<cv::Feature2D*>(((pyopencv_Feature2D_t*)self)->v.get());
    {
    PyObject* pyobj_image = NULL;
    Mat image;
    vector_KeyPoint keypoints;
    PyObject* pyobj_mask = NULL;
    Mat mask;

    const char* keywords[] = { "image", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:Feature2D.detect", (char**)keywords, &pyobj_image, &pyobj_mask) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(_self_->detect(image, keypoints, mask));
        return pyopencv_from(keypoints);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    vector_KeyPoint keypoints;
    PyObject* pyobj_mask = NULL;
    UMat mask;

    const char* keywords[] = { "image", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:Feature2D.detect", (char**)keywords, &pyobj_image, &pyobj_mask) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(_self_->detect(image, keypoints, mask));
        return pyopencv_from(keypoints);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_images = NULL;
    vector_Mat images;
    vector_vector_KeyPoint keypoints;
    PyObject* pyobj_masks = NULL;
    vector_Mat masks;

    const char* keywords[] = { "images", "masks", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:Feature2D.detect", (char**)keywords, &pyobj_images, &pyobj_masks) &&
        pyopencv_to(pyobj_images, images, ArgInfo("images", 0)) &&
        pyopencv_to(pyobj_masks, masks, ArgInfo("masks", 0)) )
    {
        ERRWRAP2(_self_->detect(images, keypoints, masks));
        return pyopencv_from(keypoints);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_images = NULL;
    vector_Mat images;
    vector_vector_KeyPoint keypoints;
    PyObject* pyobj_masks = NULL;
    vector_Mat masks;

    const char* keywords[] = { "images", "masks", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:Feature2D.detect", (char**)keywords, &pyobj_images, &pyobj_masks) &&
        pyopencv_to(pyobj_images, images, ArgInfo("images", 0)) &&
        pyopencv_to(pyobj_masks, masks, ArgInfo("masks", 0)) )
    {
        ERRWRAP2(_self_->detect(images, keypoints, masks));
        return pyopencv_from(keypoints);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_Feature2D_detectAndCompute(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Feature2D_Type))
        return failmsgp("Incorrect type of self (must be 'Feature2D' or its derivative)");
    cv::Feature2D* _self_ = dynamic_cast<cv::Feature2D*>(((pyopencv_Feature2D_t*)self)->v.get());
    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    vector_KeyPoint keypoints;
    PyObject* pyobj_descriptors = NULL;
    Mat descriptors;
    bool useProvidedKeypoints=false;

    const char* keywords[] = { "image", "mask", "descriptors", "useProvidedKeypoints", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Ob:Feature2D.detectAndCompute", (char**)keywords, &pyobj_image, &pyobj_mask, &pyobj_descriptors, &useProvidedKeypoints) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) &&
        pyopencv_to(pyobj_descriptors, descriptors, ArgInfo("descriptors", 1)) )
    {
        ERRWRAP2(_self_->detectAndCompute(image, mask, keypoints, descriptors, useProvidedKeypoints));
        return Py_BuildValue("(NN)", pyopencv_from(keypoints), pyopencv_from(descriptors));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    vector_KeyPoint keypoints;
    PyObject* pyobj_descriptors = NULL;
    UMat descriptors;
    bool useProvidedKeypoints=false;

    const char* keywords[] = { "image", "mask", "descriptors", "useProvidedKeypoints", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|Ob:Feature2D.detectAndCompute", (char**)keywords, &pyobj_image, &pyobj_mask, &pyobj_descriptors, &useProvidedKeypoints) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) &&
        pyopencv_to(pyobj_descriptors, descriptors, ArgInfo("descriptors", 1)) )
    {
        ERRWRAP2(_self_->detectAndCompute(image, mask, keypoints, descriptors, useProvidedKeypoints));
        return Py_BuildValue("(NN)", pyopencv_from(keypoints), pyopencv_from(descriptors));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_Feature2D_empty(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Feature2D_Type))
        return failmsgp("Incorrect type of self (must be 'Feature2D' or its derivative)");
    cv::Feature2D* _self_ = dynamic_cast<cv::Feature2D*>(((pyopencv_Feature2D_t*)self)->v.get());
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->empty());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_Feature2D_read(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Feature2D_Type))
        return failmsgp("Incorrect type of self (must be 'Feature2D' or its derivative)");
    cv::Feature2D* _self_ = dynamic_cast<cv::Feature2D*>(((pyopencv_Feature2D_t*)self)->v.get());
    PyObject* pyobj_fileName = NULL;
    String fileName;

    const char* keywords[] = { "fileName", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:Feature2D.read", (char**)keywords, &pyobj_fileName) &&
        pyopencv_to(pyobj_fileName, fileName, ArgInfo("fileName", 0)) )
    {
        ERRWRAP2(_self_->read(fileName));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_Feature2D_write(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Feature2D_Type))
        return failmsgp("Incorrect type of self (must be 'Feature2D' or its derivative)");
    cv::Feature2D* _self_ = dynamic_cast<cv::Feature2D*>(((pyopencv_Feature2D_t*)self)->v.get());
    PyObject* pyobj_fileName = NULL;
    String fileName;

    const char* keywords[] = { "fileName", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:Feature2D.write", (char**)keywords, &pyobj_fileName) &&
        pyopencv_to(pyobj_fileName, fileName, ArgInfo("fileName", 0)) )
    {
        ERRWRAP2(_self_->write(fileName));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_Feature2D_methods[] =
{
    {"compute", (PyCFunction)pyopencv_cv_Feature2D_compute, METH_VARARGS | METH_KEYWORDS, "compute(image, keypoints[, descriptors]) -> keypoints, descriptors  or  compute(images, keypoints[, descriptors]) -> keypoints, descriptors"},
    {"defaultNorm", (PyCFunction)pyopencv_cv_Feature2D_defaultNorm, METH_VARARGS | METH_KEYWORDS, "defaultNorm() -> retval"},
    {"descriptorSize", (PyCFunction)pyopencv_cv_Feature2D_descriptorSize, METH_VARARGS | METH_KEYWORDS, "descriptorSize() -> retval"},
    {"descriptorType", (PyCFunction)pyopencv_cv_Feature2D_descriptorType, METH_VARARGS | METH_KEYWORDS, "descriptorType() -> retval"},
    {"detect", (PyCFunction)pyopencv_cv_Feature2D_detect, METH_VARARGS | METH_KEYWORDS, "detect(image[, mask]) -> keypoints  or  detect(images[, masks]) -> keypoints"},
    {"detectAndCompute", (PyCFunction)pyopencv_cv_Feature2D_detectAndCompute, METH_VARARGS | METH_KEYWORDS, "detectAndCompute(image, mask[, descriptors[, useProvidedKeypoints]]) -> keypoints, descriptors"},
    {"empty", (PyCFunction)pyopencv_cv_Feature2D_empty, METH_VARARGS | METH_KEYWORDS, "empty() -> retval"},
    {"read", (PyCFunction)pyopencv_cv_Feature2D_read, METH_VARARGS | METH_KEYWORDS, "read(fileName) -> None"},
    {"write", (PyCFunction)pyopencv_cv_Feature2D_write, METH_VARARGS | METH_KEYWORDS, "write(fileName) -> None"},

    {NULL,          NULL}
};

static void pyopencv_Feature2D_specials(void)
{
    pyopencv_Feature2D_Type.tp_base = &pyopencv_Algorithm_Type;
    pyopencv_Feature2D_Type.tp_dealloc = pyopencv_Feature2D_dealloc;
    pyopencv_Feature2D_Type.tp_repr = pyopencv_Feature2D_repr;
    pyopencv_Feature2D_Type.tp_getset = pyopencv_Feature2D_getseters;
    pyopencv_Feature2D_Type.tp_methods = pyopencv_Feature2D_methods;
}

static PyObject* pyopencv_BRISK_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<BRISK %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_BRISK_getseters[] =
{
    {NULL}  /* Sentinel */
};



static PyMethodDef pyopencv_BRISK_methods[] =
{

    {NULL,          NULL}
};

static void pyopencv_BRISK_specials(void)
{
    pyopencv_BRISK_Type.tp_base = &pyopencv_Feature2D_Type;
    pyopencv_BRISK_Type.tp_dealloc = pyopencv_BRISK_dealloc;
    pyopencv_BRISK_Type.tp_repr = pyopencv_BRISK_repr;
    pyopencv_BRISK_Type.tp_getset = pyopencv_BRISK_getseters;
    pyopencv_BRISK_Type.tp_methods = pyopencv_BRISK_methods;
}

static PyObject* pyopencv_ORB_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<ORB %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_ORB_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_ORB_getEdgeThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ORB_Type))
        return failmsgp("Incorrect type of self (must be 'ORB' or its derivative)");
    cv::ORB* _self_ = dynamic_cast<cv::ORB*>(((pyopencv_ORB_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getEdgeThreshold());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ORB_getFastThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ORB_Type))
        return failmsgp("Incorrect type of self (must be 'ORB' or its derivative)");
    cv::ORB* _self_ = dynamic_cast<cv::ORB*>(((pyopencv_ORB_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getFastThreshold());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ORB_getFirstLevel(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ORB_Type))
        return failmsgp("Incorrect type of self (must be 'ORB' or its derivative)");
    cv::ORB* _self_ = dynamic_cast<cv::ORB*>(((pyopencv_ORB_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getFirstLevel());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ORB_getMaxFeatures(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ORB_Type))
        return failmsgp("Incorrect type of self (must be 'ORB' or its derivative)");
    cv::ORB* _self_ = dynamic_cast<cv::ORB*>(((pyopencv_ORB_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getMaxFeatures());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ORB_getNLevels(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ORB_Type))
        return failmsgp("Incorrect type of self (must be 'ORB' or its derivative)");
    cv::ORB* _self_ = dynamic_cast<cv::ORB*>(((pyopencv_ORB_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getNLevels());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ORB_getPatchSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ORB_Type))
        return failmsgp("Incorrect type of self (must be 'ORB' or its derivative)");
    cv::ORB* _self_ = dynamic_cast<cv::ORB*>(((pyopencv_ORB_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getPatchSize());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ORB_getScaleFactor(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ORB_Type))
        return failmsgp("Incorrect type of self (must be 'ORB' or its derivative)");
    cv::ORB* _self_ = dynamic_cast<cv::ORB*>(((pyopencv_ORB_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getScaleFactor());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ORB_getScoreType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ORB_Type))
        return failmsgp("Incorrect type of self (must be 'ORB' or its derivative)");
    cv::ORB* _self_ = dynamic_cast<cv::ORB*>(((pyopencv_ORB_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getScoreType());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ORB_getWTA_K(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ORB_Type))
        return failmsgp("Incorrect type of self (must be 'ORB' or its derivative)");
    cv::ORB* _self_ = dynamic_cast<cv::ORB*>(((pyopencv_ORB_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getWTA_K());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_ORB_setEdgeThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ORB_Type))
        return failmsgp("Incorrect type of self (must be 'ORB' or its derivative)");
    cv::ORB* _self_ = dynamic_cast<cv::ORB*>(((pyopencv_ORB_t*)self)->v.get());
    int edgeThreshold=0;

    const char* keywords[] = { "edgeThreshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ORB.setEdgeThreshold", (char**)keywords, &edgeThreshold) )
    {
        ERRWRAP2(_self_->setEdgeThreshold(edgeThreshold));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ORB_setFastThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ORB_Type))
        return failmsgp("Incorrect type of self (must be 'ORB' or its derivative)");
    cv::ORB* _self_ = dynamic_cast<cv::ORB*>(((pyopencv_ORB_t*)self)->v.get());
    int fastThreshold=0;

    const char* keywords[] = { "fastThreshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ORB.setFastThreshold", (char**)keywords, &fastThreshold) )
    {
        ERRWRAP2(_self_->setFastThreshold(fastThreshold));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ORB_setFirstLevel(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ORB_Type))
        return failmsgp("Incorrect type of self (must be 'ORB' or its derivative)");
    cv::ORB* _self_ = dynamic_cast<cv::ORB*>(((pyopencv_ORB_t*)self)->v.get());
    int firstLevel=0;

    const char* keywords[] = { "firstLevel", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ORB.setFirstLevel", (char**)keywords, &firstLevel) )
    {
        ERRWRAP2(_self_->setFirstLevel(firstLevel));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ORB_setMaxFeatures(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ORB_Type))
        return failmsgp("Incorrect type of self (must be 'ORB' or its derivative)");
    cv::ORB* _self_ = dynamic_cast<cv::ORB*>(((pyopencv_ORB_t*)self)->v.get());
    int maxFeatures=0;

    const char* keywords[] = { "maxFeatures", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ORB.setMaxFeatures", (char**)keywords, &maxFeatures) )
    {
        ERRWRAP2(_self_->setMaxFeatures(maxFeatures));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ORB_setNLevels(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ORB_Type))
        return failmsgp("Incorrect type of self (must be 'ORB' or its derivative)");
    cv::ORB* _self_ = dynamic_cast<cv::ORB*>(((pyopencv_ORB_t*)self)->v.get());
    int nlevels=0;

    const char* keywords[] = { "nlevels", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ORB.setNLevels", (char**)keywords, &nlevels) )
    {
        ERRWRAP2(_self_->setNLevels(nlevels));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ORB_setPatchSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ORB_Type))
        return failmsgp("Incorrect type of self (must be 'ORB' or its derivative)");
    cv::ORB* _self_ = dynamic_cast<cv::ORB*>(((pyopencv_ORB_t*)self)->v.get());
    int patchSize=0;

    const char* keywords[] = { "patchSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ORB.setPatchSize", (char**)keywords, &patchSize) )
    {
        ERRWRAP2(_self_->setPatchSize(patchSize));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ORB_setScaleFactor(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ORB_Type))
        return failmsgp("Incorrect type of self (must be 'ORB' or its derivative)");
    cv::ORB* _self_ = dynamic_cast<cv::ORB*>(((pyopencv_ORB_t*)self)->v.get());
    double scaleFactor=0;

    const char* keywords[] = { "scaleFactor", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:ORB.setScaleFactor", (char**)keywords, &scaleFactor) )
    {
        ERRWRAP2(_self_->setScaleFactor(scaleFactor));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ORB_setScoreType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ORB_Type))
        return failmsgp("Incorrect type of self (must be 'ORB' or its derivative)");
    cv::ORB* _self_ = dynamic_cast<cv::ORB*>(((pyopencv_ORB_t*)self)->v.get());
    int scoreType=0;

    const char* keywords[] = { "scoreType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ORB.setScoreType", (char**)keywords, &scoreType) )
    {
        ERRWRAP2(_self_->setScoreType(scoreType));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_ORB_setWTA_K(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_ORB_Type))
        return failmsgp("Incorrect type of self (must be 'ORB' or its derivative)");
    cv::ORB* _self_ = dynamic_cast<cv::ORB*>(((pyopencv_ORB_t*)self)->v.get());
    int wta_k=0;

    const char* keywords[] = { "wta_k", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:ORB.setWTA_K", (char**)keywords, &wta_k) )
    {
        ERRWRAP2(_self_->setWTA_K(wta_k));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_ORB_methods[] =
{
    {"getEdgeThreshold", (PyCFunction)pyopencv_cv_ORB_getEdgeThreshold, METH_VARARGS | METH_KEYWORDS, "getEdgeThreshold() -> retval"},
    {"getFastThreshold", (PyCFunction)pyopencv_cv_ORB_getFastThreshold, METH_VARARGS | METH_KEYWORDS, "getFastThreshold() -> retval"},
    {"getFirstLevel", (PyCFunction)pyopencv_cv_ORB_getFirstLevel, METH_VARARGS | METH_KEYWORDS, "getFirstLevel() -> retval"},
    {"getMaxFeatures", (PyCFunction)pyopencv_cv_ORB_getMaxFeatures, METH_VARARGS | METH_KEYWORDS, "getMaxFeatures() -> retval"},
    {"getNLevels", (PyCFunction)pyopencv_cv_ORB_getNLevels, METH_VARARGS | METH_KEYWORDS, "getNLevels() -> retval"},
    {"getPatchSize", (PyCFunction)pyopencv_cv_ORB_getPatchSize, METH_VARARGS | METH_KEYWORDS, "getPatchSize() -> retval"},
    {"getScaleFactor", (PyCFunction)pyopencv_cv_ORB_getScaleFactor, METH_VARARGS | METH_KEYWORDS, "getScaleFactor() -> retval"},
    {"getScoreType", (PyCFunction)pyopencv_cv_ORB_getScoreType, METH_VARARGS | METH_KEYWORDS, "getScoreType() -> retval"},
    {"getWTA_K", (PyCFunction)pyopencv_cv_ORB_getWTA_K, METH_VARARGS | METH_KEYWORDS, "getWTA_K() -> retval"},
    {"setEdgeThreshold", (PyCFunction)pyopencv_cv_ORB_setEdgeThreshold, METH_VARARGS | METH_KEYWORDS, "setEdgeThreshold(edgeThreshold) -> None"},
    {"setFastThreshold", (PyCFunction)pyopencv_cv_ORB_setFastThreshold, METH_VARARGS | METH_KEYWORDS, "setFastThreshold(fastThreshold) -> None"},
    {"setFirstLevel", (PyCFunction)pyopencv_cv_ORB_setFirstLevel, METH_VARARGS | METH_KEYWORDS, "setFirstLevel(firstLevel) -> None"},
    {"setMaxFeatures", (PyCFunction)pyopencv_cv_ORB_setMaxFeatures, METH_VARARGS | METH_KEYWORDS, "setMaxFeatures(maxFeatures) -> None"},
    {"setNLevels", (PyCFunction)pyopencv_cv_ORB_setNLevels, METH_VARARGS | METH_KEYWORDS, "setNLevels(nlevels) -> None"},
    {"setPatchSize", (PyCFunction)pyopencv_cv_ORB_setPatchSize, METH_VARARGS | METH_KEYWORDS, "setPatchSize(patchSize) -> None"},
    {"setScaleFactor", (PyCFunction)pyopencv_cv_ORB_setScaleFactor, METH_VARARGS | METH_KEYWORDS, "setScaleFactor(scaleFactor) -> None"},
    {"setScoreType", (PyCFunction)pyopencv_cv_ORB_setScoreType, METH_VARARGS | METH_KEYWORDS, "setScoreType(scoreType) -> None"},
    {"setWTA_K", (PyCFunction)pyopencv_cv_ORB_setWTA_K, METH_VARARGS | METH_KEYWORDS, "setWTA_K(wta_k) -> None"},

    {NULL,          NULL}
};

static void pyopencv_ORB_specials(void)
{
    pyopencv_ORB_Type.tp_base = &pyopencv_Feature2D_Type;
    pyopencv_ORB_Type.tp_dealloc = pyopencv_ORB_dealloc;
    pyopencv_ORB_Type.tp_repr = pyopencv_ORB_repr;
    pyopencv_ORB_Type.tp_getset = pyopencv_ORB_getseters;
    pyopencv_ORB_Type.tp_methods = pyopencv_ORB_methods;
}

static PyObject* pyopencv_MSER_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<MSER %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_MSER_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_MSER_detectRegions(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_MSER_Type))
        return failmsgp("Incorrect type of self (must be 'MSER' or its derivative)");
    cv::MSER* _self_ = dynamic_cast<cv::MSER*>(((pyopencv_MSER_t*)self)->v.get());
    {
    PyObject* pyobj_image = NULL;
    Mat image;
    vector_vector_Point msers;
    vector_Rect bboxes;

    const char* keywords[] = { "image", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:MSER.detectRegions", (char**)keywords, &pyobj_image) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) )
    {
        ERRWRAP2(_self_->detectRegions(image, msers, bboxes));
        return Py_BuildValue("(NN)", pyopencv_from(msers), pyopencv_from(bboxes));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    UMat image;
    vector_vector_Point msers;
    vector_Rect bboxes;

    const char* keywords[] = { "image", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:MSER.detectRegions", (char**)keywords, &pyobj_image) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) )
    {
        ERRWRAP2(_self_->detectRegions(image, msers, bboxes));
        return Py_BuildValue("(NN)", pyopencv_from(msers), pyopencv_from(bboxes));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_MSER_getDelta(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_MSER_Type))
        return failmsgp("Incorrect type of self (must be 'MSER' or its derivative)");
    cv::MSER* _self_ = dynamic_cast<cv::MSER*>(((pyopencv_MSER_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getDelta());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_MSER_getMaxArea(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_MSER_Type))
        return failmsgp("Incorrect type of self (must be 'MSER' or its derivative)");
    cv::MSER* _self_ = dynamic_cast<cv::MSER*>(((pyopencv_MSER_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getMaxArea());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_MSER_getMinArea(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_MSER_Type))
        return failmsgp("Incorrect type of self (must be 'MSER' or its derivative)");
    cv::MSER* _self_ = dynamic_cast<cv::MSER*>(((pyopencv_MSER_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getMinArea());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_MSER_getPass2Only(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_MSER_Type))
        return failmsgp("Incorrect type of self (must be 'MSER' or its derivative)");
    cv::MSER* _self_ = dynamic_cast<cv::MSER*>(((pyopencv_MSER_t*)self)->v.get());
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getPass2Only());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_MSER_setDelta(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_MSER_Type))
        return failmsgp("Incorrect type of self (must be 'MSER' or its derivative)");
    cv::MSER* _self_ = dynamic_cast<cv::MSER*>(((pyopencv_MSER_t*)self)->v.get());
    int delta=0;

    const char* keywords[] = { "delta", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:MSER.setDelta", (char**)keywords, &delta) )
    {
        ERRWRAP2(_self_->setDelta(delta));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_MSER_setMaxArea(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_MSER_Type))
        return failmsgp("Incorrect type of self (must be 'MSER' or its derivative)");
    cv::MSER* _self_ = dynamic_cast<cv::MSER*>(((pyopencv_MSER_t*)self)->v.get());
    int maxArea=0;

    const char* keywords[] = { "maxArea", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:MSER.setMaxArea", (char**)keywords, &maxArea) )
    {
        ERRWRAP2(_self_->setMaxArea(maxArea));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_MSER_setMinArea(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_MSER_Type))
        return failmsgp("Incorrect type of self (must be 'MSER' or its derivative)");
    cv::MSER* _self_ = dynamic_cast<cv::MSER*>(((pyopencv_MSER_t*)self)->v.get());
    int minArea=0;

    const char* keywords[] = { "minArea", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:MSER.setMinArea", (char**)keywords, &minArea) )
    {
        ERRWRAP2(_self_->setMinArea(minArea));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_MSER_setPass2Only(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_MSER_Type))
        return failmsgp("Incorrect type of self (must be 'MSER' or its derivative)");
    cv::MSER* _self_ = dynamic_cast<cv::MSER*>(((pyopencv_MSER_t*)self)->v.get());
    bool f=0;

    const char* keywords[] = { "f", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:MSER.setPass2Only", (char**)keywords, &f) )
    {
        ERRWRAP2(_self_->setPass2Only(f));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_MSER_methods[] =
{
    {"detectRegions", (PyCFunction)pyopencv_cv_MSER_detectRegions, METH_VARARGS | METH_KEYWORDS, "detectRegions(image) -> msers, bboxes"},
    {"getDelta", (PyCFunction)pyopencv_cv_MSER_getDelta, METH_VARARGS | METH_KEYWORDS, "getDelta() -> retval"},
    {"getMaxArea", (PyCFunction)pyopencv_cv_MSER_getMaxArea, METH_VARARGS | METH_KEYWORDS, "getMaxArea() -> retval"},
    {"getMinArea", (PyCFunction)pyopencv_cv_MSER_getMinArea, METH_VARARGS | METH_KEYWORDS, "getMinArea() -> retval"},
    {"getPass2Only", (PyCFunction)pyopencv_cv_MSER_getPass2Only, METH_VARARGS | METH_KEYWORDS, "getPass2Only() -> retval"},
    {"setDelta", (PyCFunction)pyopencv_cv_MSER_setDelta, METH_VARARGS | METH_KEYWORDS, "setDelta(delta) -> None"},
    {"setMaxArea", (PyCFunction)pyopencv_cv_MSER_setMaxArea, METH_VARARGS | METH_KEYWORDS, "setMaxArea(maxArea) -> None"},
    {"setMinArea", (PyCFunction)pyopencv_cv_MSER_setMinArea, METH_VARARGS | METH_KEYWORDS, "setMinArea(minArea) -> None"},
    {"setPass2Only", (PyCFunction)pyopencv_cv_MSER_setPass2Only, METH_VARARGS | METH_KEYWORDS, "setPass2Only(f) -> None"},

    {NULL,          NULL}
};

static void pyopencv_MSER_specials(void)
{
    pyopencv_MSER_Type.tp_base = &pyopencv_Feature2D_Type;
    pyopencv_MSER_Type.tp_dealloc = pyopencv_MSER_dealloc;
    pyopencv_MSER_Type.tp_repr = pyopencv_MSER_repr;
    pyopencv_MSER_Type.tp_getset = pyopencv_MSER_getseters;
    pyopencv_MSER_Type.tp_methods = pyopencv_MSER_methods;
}

static PyObject* pyopencv_FastFeatureDetector_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<FastFeatureDetector %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_FastFeatureDetector_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_FastFeatureDetector_getNonmaxSuppression(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FastFeatureDetector_Type))
        return failmsgp("Incorrect type of self (must be 'FastFeatureDetector' or its derivative)");
    cv::FastFeatureDetector* _self_ = dynamic_cast<cv::FastFeatureDetector*>(((pyopencv_FastFeatureDetector_t*)self)->v.get());
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getNonmaxSuppression());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FastFeatureDetector_getThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FastFeatureDetector_Type))
        return failmsgp("Incorrect type of self (must be 'FastFeatureDetector' or its derivative)");
    cv::FastFeatureDetector* _self_ = dynamic_cast<cv::FastFeatureDetector*>(((pyopencv_FastFeatureDetector_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getThreshold());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FastFeatureDetector_getType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FastFeatureDetector_Type))
        return failmsgp("Incorrect type of self (must be 'FastFeatureDetector' or its derivative)");
    cv::FastFeatureDetector* _self_ = dynamic_cast<cv::FastFeatureDetector*>(((pyopencv_FastFeatureDetector_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getType());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_FastFeatureDetector_setNonmaxSuppression(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FastFeatureDetector_Type))
        return failmsgp("Incorrect type of self (must be 'FastFeatureDetector' or its derivative)");
    cv::FastFeatureDetector* _self_ = dynamic_cast<cv::FastFeatureDetector*>(((pyopencv_FastFeatureDetector_t*)self)->v.get());
    bool f=0;

    const char* keywords[] = { "f", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:FastFeatureDetector.setNonmaxSuppression", (char**)keywords, &f) )
    {
        ERRWRAP2(_self_->setNonmaxSuppression(f));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_FastFeatureDetector_setThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FastFeatureDetector_Type))
        return failmsgp("Incorrect type of self (must be 'FastFeatureDetector' or its derivative)");
    cv::FastFeatureDetector* _self_ = dynamic_cast<cv::FastFeatureDetector*>(((pyopencv_FastFeatureDetector_t*)self)->v.get());
    int threshold=0;

    const char* keywords[] = { "threshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:FastFeatureDetector.setThreshold", (char**)keywords, &threshold) )
    {
        ERRWRAP2(_self_->setThreshold(threshold));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_FastFeatureDetector_setType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_FastFeatureDetector_Type))
        return failmsgp("Incorrect type of self (must be 'FastFeatureDetector' or its derivative)");
    cv::FastFeatureDetector* _self_ = dynamic_cast<cv::FastFeatureDetector*>(((pyopencv_FastFeatureDetector_t*)self)->v.get());
    int type=0;

    const char* keywords[] = { "type", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:FastFeatureDetector.setType", (char**)keywords, &type) )
    {
        ERRWRAP2(_self_->setType(type));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_FastFeatureDetector_methods[] =
{
    {"getNonmaxSuppression", (PyCFunction)pyopencv_cv_FastFeatureDetector_getNonmaxSuppression, METH_VARARGS | METH_KEYWORDS, "getNonmaxSuppression() -> retval"},
    {"getThreshold", (PyCFunction)pyopencv_cv_FastFeatureDetector_getThreshold, METH_VARARGS | METH_KEYWORDS, "getThreshold() -> retval"},
    {"getType", (PyCFunction)pyopencv_cv_FastFeatureDetector_getType, METH_VARARGS | METH_KEYWORDS, "getType() -> retval"},
    {"setNonmaxSuppression", (PyCFunction)pyopencv_cv_FastFeatureDetector_setNonmaxSuppression, METH_VARARGS | METH_KEYWORDS, "setNonmaxSuppression(f) -> None"},
    {"setThreshold", (PyCFunction)pyopencv_cv_FastFeatureDetector_setThreshold, METH_VARARGS | METH_KEYWORDS, "setThreshold(threshold) -> None"},
    {"setType", (PyCFunction)pyopencv_cv_FastFeatureDetector_setType, METH_VARARGS | METH_KEYWORDS, "setType(type) -> None"},

    {NULL,          NULL}
};

static void pyopencv_FastFeatureDetector_specials(void)
{
    pyopencv_FastFeatureDetector_Type.tp_base = &pyopencv_Feature2D_Type;
    pyopencv_FastFeatureDetector_Type.tp_dealloc = pyopencv_FastFeatureDetector_dealloc;
    pyopencv_FastFeatureDetector_Type.tp_repr = pyopencv_FastFeatureDetector_repr;
    pyopencv_FastFeatureDetector_Type.tp_getset = pyopencv_FastFeatureDetector_getseters;
    pyopencv_FastFeatureDetector_Type.tp_methods = pyopencv_FastFeatureDetector_methods;
}

static PyObject* pyopencv_AgastFeatureDetector_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<AgastFeatureDetector %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_AgastFeatureDetector_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_AgastFeatureDetector_getNonmaxSuppression(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AgastFeatureDetector_Type))
        return failmsgp("Incorrect type of self (must be 'AgastFeatureDetector' or its derivative)");
    cv::AgastFeatureDetector* _self_ = dynamic_cast<cv::AgastFeatureDetector*>(((pyopencv_AgastFeatureDetector_t*)self)->v.get());
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getNonmaxSuppression());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_AgastFeatureDetector_getThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AgastFeatureDetector_Type))
        return failmsgp("Incorrect type of self (must be 'AgastFeatureDetector' or its derivative)");
    cv::AgastFeatureDetector* _self_ = dynamic_cast<cv::AgastFeatureDetector*>(((pyopencv_AgastFeatureDetector_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getThreshold());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_AgastFeatureDetector_getType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AgastFeatureDetector_Type))
        return failmsgp("Incorrect type of self (must be 'AgastFeatureDetector' or its derivative)");
    cv::AgastFeatureDetector* _self_ = dynamic_cast<cv::AgastFeatureDetector*>(((pyopencv_AgastFeatureDetector_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getType());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_AgastFeatureDetector_setNonmaxSuppression(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AgastFeatureDetector_Type))
        return failmsgp("Incorrect type of self (must be 'AgastFeatureDetector' or its derivative)");
    cv::AgastFeatureDetector* _self_ = dynamic_cast<cv::AgastFeatureDetector*>(((pyopencv_AgastFeatureDetector_t*)self)->v.get());
    bool f=0;

    const char* keywords[] = { "f", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:AgastFeatureDetector.setNonmaxSuppression", (char**)keywords, &f) )
    {
        ERRWRAP2(_self_->setNonmaxSuppression(f));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_AgastFeatureDetector_setThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AgastFeatureDetector_Type))
        return failmsgp("Incorrect type of self (must be 'AgastFeatureDetector' or its derivative)");
    cv::AgastFeatureDetector* _self_ = dynamic_cast<cv::AgastFeatureDetector*>(((pyopencv_AgastFeatureDetector_t*)self)->v.get());
    int threshold=0;

    const char* keywords[] = { "threshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:AgastFeatureDetector.setThreshold", (char**)keywords, &threshold) )
    {
        ERRWRAP2(_self_->setThreshold(threshold));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_AgastFeatureDetector_setType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AgastFeatureDetector_Type))
        return failmsgp("Incorrect type of self (must be 'AgastFeatureDetector' or its derivative)");
    cv::AgastFeatureDetector* _self_ = dynamic_cast<cv::AgastFeatureDetector*>(((pyopencv_AgastFeatureDetector_t*)self)->v.get());
    int type=0;

    const char* keywords[] = { "type", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:AgastFeatureDetector.setType", (char**)keywords, &type) )
    {
        ERRWRAP2(_self_->setType(type));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_AgastFeatureDetector_methods[] =
{
    {"getNonmaxSuppression", (PyCFunction)pyopencv_cv_AgastFeatureDetector_getNonmaxSuppression, METH_VARARGS | METH_KEYWORDS, "getNonmaxSuppression() -> retval"},
    {"getThreshold", (PyCFunction)pyopencv_cv_AgastFeatureDetector_getThreshold, METH_VARARGS | METH_KEYWORDS, "getThreshold() -> retval"},
    {"getType", (PyCFunction)pyopencv_cv_AgastFeatureDetector_getType, METH_VARARGS | METH_KEYWORDS, "getType() -> retval"},
    {"setNonmaxSuppression", (PyCFunction)pyopencv_cv_AgastFeatureDetector_setNonmaxSuppression, METH_VARARGS | METH_KEYWORDS, "setNonmaxSuppression(f) -> None"},
    {"setThreshold", (PyCFunction)pyopencv_cv_AgastFeatureDetector_setThreshold, METH_VARARGS | METH_KEYWORDS, "setThreshold(threshold) -> None"},
    {"setType", (PyCFunction)pyopencv_cv_AgastFeatureDetector_setType, METH_VARARGS | METH_KEYWORDS, "setType(type) -> None"},

    {NULL,          NULL}
};

static void pyopencv_AgastFeatureDetector_specials(void)
{
    pyopencv_AgastFeatureDetector_Type.tp_base = &pyopencv_Feature2D_Type;
    pyopencv_AgastFeatureDetector_Type.tp_dealloc = pyopencv_AgastFeatureDetector_dealloc;
    pyopencv_AgastFeatureDetector_Type.tp_repr = pyopencv_AgastFeatureDetector_repr;
    pyopencv_AgastFeatureDetector_Type.tp_getset = pyopencv_AgastFeatureDetector_getseters;
    pyopencv_AgastFeatureDetector_Type.tp_methods = pyopencv_AgastFeatureDetector_methods;
}

static PyObject* pyopencv_GFTTDetector_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<GFTTDetector %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_GFTTDetector_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_GFTTDetector_getBlockSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_GFTTDetector_Type))
        return failmsgp("Incorrect type of self (must be 'GFTTDetector' or its derivative)");
    cv::GFTTDetector* _self_ = dynamic_cast<cv::GFTTDetector*>(((pyopencv_GFTTDetector_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getBlockSize());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_GFTTDetector_getHarrisDetector(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_GFTTDetector_Type))
        return failmsgp("Incorrect type of self (must be 'GFTTDetector' or its derivative)");
    cv::GFTTDetector* _self_ = dynamic_cast<cv::GFTTDetector*>(((pyopencv_GFTTDetector_t*)self)->v.get());
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getHarrisDetector());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_GFTTDetector_getK(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_GFTTDetector_Type))
        return failmsgp("Incorrect type of self (must be 'GFTTDetector' or its derivative)");
    cv::GFTTDetector* _self_ = dynamic_cast<cv::GFTTDetector*>(((pyopencv_GFTTDetector_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getK());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_GFTTDetector_getMaxFeatures(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_GFTTDetector_Type))
        return failmsgp("Incorrect type of self (must be 'GFTTDetector' or its derivative)");
    cv::GFTTDetector* _self_ = dynamic_cast<cv::GFTTDetector*>(((pyopencv_GFTTDetector_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getMaxFeatures());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_GFTTDetector_getMinDistance(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_GFTTDetector_Type))
        return failmsgp("Incorrect type of self (must be 'GFTTDetector' or its derivative)");
    cv::GFTTDetector* _self_ = dynamic_cast<cv::GFTTDetector*>(((pyopencv_GFTTDetector_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getMinDistance());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_GFTTDetector_getQualityLevel(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_GFTTDetector_Type))
        return failmsgp("Incorrect type of self (must be 'GFTTDetector' or its derivative)");
    cv::GFTTDetector* _self_ = dynamic_cast<cv::GFTTDetector*>(((pyopencv_GFTTDetector_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getQualityLevel());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_GFTTDetector_setBlockSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_GFTTDetector_Type))
        return failmsgp("Incorrect type of self (must be 'GFTTDetector' or its derivative)");
    cv::GFTTDetector* _self_ = dynamic_cast<cv::GFTTDetector*>(((pyopencv_GFTTDetector_t*)self)->v.get());
    int blockSize=0;

    const char* keywords[] = { "blockSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:GFTTDetector.setBlockSize", (char**)keywords, &blockSize) )
    {
        ERRWRAP2(_self_->setBlockSize(blockSize));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_GFTTDetector_setHarrisDetector(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_GFTTDetector_Type))
        return failmsgp("Incorrect type of self (must be 'GFTTDetector' or its derivative)");
    cv::GFTTDetector* _self_ = dynamic_cast<cv::GFTTDetector*>(((pyopencv_GFTTDetector_t*)self)->v.get());
    bool val=0;

    const char* keywords[] = { "val", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:GFTTDetector.setHarrisDetector", (char**)keywords, &val) )
    {
        ERRWRAP2(_self_->setHarrisDetector(val));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_GFTTDetector_setK(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_GFTTDetector_Type))
        return failmsgp("Incorrect type of self (must be 'GFTTDetector' or its derivative)");
    cv::GFTTDetector* _self_ = dynamic_cast<cv::GFTTDetector*>(((pyopencv_GFTTDetector_t*)self)->v.get());
    double k=0;

    const char* keywords[] = { "k", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:GFTTDetector.setK", (char**)keywords, &k) )
    {
        ERRWRAP2(_self_->setK(k));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_GFTTDetector_setMaxFeatures(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_GFTTDetector_Type))
        return failmsgp("Incorrect type of self (must be 'GFTTDetector' or its derivative)");
    cv::GFTTDetector* _self_ = dynamic_cast<cv::GFTTDetector*>(((pyopencv_GFTTDetector_t*)self)->v.get());
    int maxFeatures=0;

    const char* keywords[] = { "maxFeatures", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:GFTTDetector.setMaxFeatures", (char**)keywords, &maxFeatures) )
    {
        ERRWRAP2(_self_->setMaxFeatures(maxFeatures));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_GFTTDetector_setMinDistance(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_GFTTDetector_Type))
        return failmsgp("Incorrect type of self (must be 'GFTTDetector' or its derivative)");
    cv::GFTTDetector* _self_ = dynamic_cast<cv::GFTTDetector*>(((pyopencv_GFTTDetector_t*)self)->v.get());
    double minDistance=0;

    const char* keywords[] = { "minDistance", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:GFTTDetector.setMinDistance", (char**)keywords, &minDistance) )
    {
        ERRWRAP2(_self_->setMinDistance(minDistance));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_GFTTDetector_setQualityLevel(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_GFTTDetector_Type))
        return failmsgp("Incorrect type of self (must be 'GFTTDetector' or its derivative)");
    cv::GFTTDetector* _self_ = dynamic_cast<cv::GFTTDetector*>(((pyopencv_GFTTDetector_t*)self)->v.get());
    double qlevel=0;

    const char* keywords[] = { "qlevel", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:GFTTDetector.setQualityLevel", (char**)keywords, &qlevel) )
    {
        ERRWRAP2(_self_->setQualityLevel(qlevel));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_GFTTDetector_methods[] =
{
    {"getBlockSize", (PyCFunction)pyopencv_cv_GFTTDetector_getBlockSize, METH_VARARGS | METH_KEYWORDS, "getBlockSize() -> retval"},
    {"getHarrisDetector", (PyCFunction)pyopencv_cv_GFTTDetector_getHarrisDetector, METH_VARARGS | METH_KEYWORDS, "getHarrisDetector() -> retval"},
    {"getK", (PyCFunction)pyopencv_cv_GFTTDetector_getK, METH_VARARGS | METH_KEYWORDS, "getK() -> retval"},
    {"getMaxFeatures", (PyCFunction)pyopencv_cv_GFTTDetector_getMaxFeatures, METH_VARARGS | METH_KEYWORDS, "getMaxFeatures() -> retval"},
    {"getMinDistance", (PyCFunction)pyopencv_cv_GFTTDetector_getMinDistance, METH_VARARGS | METH_KEYWORDS, "getMinDistance() -> retval"},
    {"getQualityLevel", (PyCFunction)pyopencv_cv_GFTTDetector_getQualityLevel, METH_VARARGS | METH_KEYWORDS, "getQualityLevel() -> retval"},
    {"setBlockSize", (PyCFunction)pyopencv_cv_GFTTDetector_setBlockSize, METH_VARARGS | METH_KEYWORDS, "setBlockSize(blockSize) -> None"},
    {"setHarrisDetector", (PyCFunction)pyopencv_cv_GFTTDetector_setHarrisDetector, METH_VARARGS | METH_KEYWORDS, "setHarrisDetector(val) -> None"},
    {"setK", (PyCFunction)pyopencv_cv_GFTTDetector_setK, METH_VARARGS | METH_KEYWORDS, "setK(k) -> None"},
    {"setMaxFeatures", (PyCFunction)pyopencv_cv_GFTTDetector_setMaxFeatures, METH_VARARGS | METH_KEYWORDS, "setMaxFeatures(maxFeatures) -> None"},
    {"setMinDistance", (PyCFunction)pyopencv_cv_GFTTDetector_setMinDistance, METH_VARARGS | METH_KEYWORDS, "setMinDistance(minDistance) -> None"},
    {"setQualityLevel", (PyCFunction)pyopencv_cv_GFTTDetector_setQualityLevel, METH_VARARGS | METH_KEYWORDS, "setQualityLevel(qlevel) -> None"},

    {NULL,          NULL}
};

static void pyopencv_GFTTDetector_specials(void)
{
    pyopencv_GFTTDetector_Type.tp_base = &pyopencv_Feature2D_Type;
    pyopencv_GFTTDetector_Type.tp_dealloc = pyopencv_GFTTDetector_dealloc;
    pyopencv_GFTTDetector_Type.tp_repr = pyopencv_GFTTDetector_repr;
    pyopencv_GFTTDetector_Type.tp_getset = pyopencv_GFTTDetector_getseters;
    pyopencv_GFTTDetector_Type.tp_methods = pyopencv_GFTTDetector_methods;
}

static PyObject* pyopencv_SimpleBlobDetector_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<SimpleBlobDetector %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_SimpleBlobDetector_getseters[] =
{
    {NULL}  /* Sentinel */
};



static PyMethodDef pyopencv_SimpleBlobDetector_methods[] =
{

    {NULL,          NULL}
};

static void pyopencv_SimpleBlobDetector_specials(void)
{
    pyopencv_SimpleBlobDetector_Type.tp_base = &pyopencv_Feature2D_Type;
    pyopencv_SimpleBlobDetector_Type.tp_dealloc = pyopencv_SimpleBlobDetector_dealloc;
    pyopencv_SimpleBlobDetector_Type.tp_repr = pyopencv_SimpleBlobDetector_repr;
    pyopencv_SimpleBlobDetector_Type.tp_getset = pyopencv_SimpleBlobDetector_getseters;
    pyopencv_SimpleBlobDetector_Type.tp_methods = pyopencv_SimpleBlobDetector_methods;
}

static PyObject* pyopencv_SimpleBlobDetector_Params_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<SimpleBlobDetector_Params %p>", self);
    return PyString_FromString(str);
}


static PyObject* pyopencv_SimpleBlobDetector_Params_get_blobColor(pyopencv_SimpleBlobDetector_Params_t* p, void *closure)
{
    return pyopencv_from(p->v.blobColor);
}

static int pyopencv_SimpleBlobDetector_Params_set_blobColor(pyopencv_SimpleBlobDetector_Params_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the blobColor attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.blobColor) ? 0 : -1;
}

static PyObject* pyopencv_SimpleBlobDetector_Params_get_filterByArea(pyopencv_SimpleBlobDetector_Params_t* p, void *closure)
{
    return pyopencv_from(p->v.filterByArea);
}

static int pyopencv_SimpleBlobDetector_Params_set_filterByArea(pyopencv_SimpleBlobDetector_Params_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the filterByArea attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.filterByArea) ? 0 : -1;
}

static PyObject* pyopencv_SimpleBlobDetector_Params_get_filterByCircularity(pyopencv_SimpleBlobDetector_Params_t* p, void *closure)
{
    return pyopencv_from(p->v.filterByCircularity);
}

static int pyopencv_SimpleBlobDetector_Params_set_filterByCircularity(pyopencv_SimpleBlobDetector_Params_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the filterByCircularity attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.filterByCircularity) ? 0 : -1;
}

static PyObject* pyopencv_SimpleBlobDetector_Params_get_filterByColor(pyopencv_SimpleBlobDetector_Params_t* p, void *closure)
{
    return pyopencv_from(p->v.filterByColor);
}

static int pyopencv_SimpleBlobDetector_Params_set_filterByColor(pyopencv_SimpleBlobDetector_Params_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the filterByColor attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.filterByColor) ? 0 : -1;
}

static PyObject* pyopencv_SimpleBlobDetector_Params_get_filterByConvexity(pyopencv_SimpleBlobDetector_Params_t* p, void *closure)
{
    return pyopencv_from(p->v.filterByConvexity);
}

static int pyopencv_SimpleBlobDetector_Params_set_filterByConvexity(pyopencv_SimpleBlobDetector_Params_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the filterByConvexity attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.filterByConvexity) ? 0 : -1;
}

static PyObject* pyopencv_SimpleBlobDetector_Params_get_filterByInertia(pyopencv_SimpleBlobDetector_Params_t* p, void *closure)
{
    return pyopencv_from(p->v.filterByInertia);
}

static int pyopencv_SimpleBlobDetector_Params_set_filterByInertia(pyopencv_SimpleBlobDetector_Params_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the filterByInertia attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.filterByInertia) ? 0 : -1;
}

static PyObject* pyopencv_SimpleBlobDetector_Params_get_maxArea(pyopencv_SimpleBlobDetector_Params_t* p, void *closure)
{
    return pyopencv_from(p->v.maxArea);
}

static int pyopencv_SimpleBlobDetector_Params_set_maxArea(pyopencv_SimpleBlobDetector_Params_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the maxArea attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.maxArea) ? 0 : -1;
}

static PyObject* pyopencv_SimpleBlobDetector_Params_get_maxCircularity(pyopencv_SimpleBlobDetector_Params_t* p, void *closure)
{
    return pyopencv_from(p->v.maxCircularity);
}

static int pyopencv_SimpleBlobDetector_Params_set_maxCircularity(pyopencv_SimpleBlobDetector_Params_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the maxCircularity attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.maxCircularity) ? 0 : -1;
}

static PyObject* pyopencv_SimpleBlobDetector_Params_get_maxConvexity(pyopencv_SimpleBlobDetector_Params_t* p, void *closure)
{
    return pyopencv_from(p->v.maxConvexity);
}

static int pyopencv_SimpleBlobDetector_Params_set_maxConvexity(pyopencv_SimpleBlobDetector_Params_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the maxConvexity attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.maxConvexity) ? 0 : -1;
}

static PyObject* pyopencv_SimpleBlobDetector_Params_get_maxInertiaRatio(pyopencv_SimpleBlobDetector_Params_t* p, void *closure)
{
    return pyopencv_from(p->v.maxInertiaRatio);
}

static int pyopencv_SimpleBlobDetector_Params_set_maxInertiaRatio(pyopencv_SimpleBlobDetector_Params_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the maxInertiaRatio attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.maxInertiaRatio) ? 0 : -1;
}

static PyObject* pyopencv_SimpleBlobDetector_Params_get_maxThreshold(pyopencv_SimpleBlobDetector_Params_t* p, void *closure)
{
    return pyopencv_from(p->v.maxThreshold);
}

static int pyopencv_SimpleBlobDetector_Params_set_maxThreshold(pyopencv_SimpleBlobDetector_Params_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the maxThreshold attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.maxThreshold) ? 0 : -1;
}

static PyObject* pyopencv_SimpleBlobDetector_Params_get_minArea(pyopencv_SimpleBlobDetector_Params_t* p, void *closure)
{
    return pyopencv_from(p->v.minArea);
}

static int pyopencv_SimpleBlobDetector_Params_set_minArea(pyopencv_SimpleBlobDetector_Params_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the minArea attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.minArea) ? 0 : -1;
}

static PyObject* pyopencv_SimpleBlobDetector_Params_get_minCircularity(pyopencv_SimpleBlobDetector_Params_t* p, void *closure)
{
    return pyopencv_from(p->v.minCircularity);
}

static int pyopencv_SimpleBlobDetector_Params_set_minCircularity(pyopencv_SimpleBlobDetector_Params_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the minCircularity attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.minCircularity) ? 0 : -1;
}

static PyObject* pyopencv_SimpleBlobDetector_Params_get_minConvexity(pyopencv_SimpleBlobDetector_Params_t* p, void *closure)
{
    return pyopencv_from(p->v.minConvexity);
}

static int pyopencv_SimpleBlobDetector_Params_set_minConvexity(pyopencv_SimpleBlobDetector_Params_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the minConvexity attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.minConvexity) ? 0 : -1;
}

static PyObject* pyopencv_SimpleBlobDetector_Params_get_minDistBetweenBlobs(pyopencv_SimpleBlobDetector_Params_t* p, void *closure)
{
    return pyopencv_from(p->v.minDistBetweenBlobs);
}

static int pyopencv_SimpleBlobDetector_Params_set_minDistBetweenBlobs(pyopencv_SimpleBlobDetector_Params_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the minDistBetweenBlobs attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.minDistBetweenBlobs) ? 0 : -1;
}

static PyObject* pyopencv_SimpleBlobDetector_Params_get_minInertiaRatio(pyopencv_SimpleBlobDetector_Params_t* p, void *closure)
{
    return pyopencv_from(p->v.minInertiaRatio);
}

static int pyopencv_SimpleBlobDetector_Params_set_minInertiaRatio(pyopencv_SimpleBlobDetector_Params_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the minInertiaRatio attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.minInertiaRatio) ? 0 : -1;
}

static PyObject* pyopencv_SimpleBlobDetector_Params_get_minRepeatability(pyopencv_SimpleBlobDetector_Params_t* p, void *closure)
{
    return pyopencv_from(p->v.minRepeatability);
}

static int pyopencv_SimpleBlobDetector_Params_set_minRepeatability(pyopencv_SimpleBlobDetector_Params_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the minRepeatability attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.minRepeatability) ? 0 : -1;
}

static PyObject* pyopencv_SimpleBlobDetector_Params_get_minThreshold(pyopencv_SimpleBlobDetector_Params_t* p, void *closure)
{
    return pyopencv_from(p->v.minThreshold);
}

static int pyopencv_SimpleBlobDetector_Params_set_minThreshold(pyopencv_SimpleBlobDetector_Params_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the minThreshold attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.minThreshold) ? 0 : -1;
}

static PyObject* pyopencv_SimpleBlobDetector_Params_get_thresholdStep(pyopencv_SimpleBlobDetector_Params_t* p, void *closure)
{
    return pyopencv_from(p->v.thresholdStep);
}

static int pyopencv_SimpleBlobDetector_Params_set_thresholdStep(pyopencv_SimpleBlobDetector_Params_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the thresholdStep attribute");
        return -1;
    }
    return pyopencv_to(value, p->v.thresholdStep) ? 0 : -1;
}


static PyGetSetDef pyopencv_SimpleBlobDetector_Params_getseters[] =
{
    {(char*)"blobColor", (getter)pyopencv_SimpleBlobDetector_Params_get_blobColor, (setter)pyopencv_SimpleBlobDetector_Params_set_blobColor, (char*)"blobColor", NULL},
    {(char*)"filterByArea", (getter)pyopencv_SimpleBlobDetector_Params_get_filterByArea, (setter)pyopencv_SimpleBlobDetector_Params_set_filterByArea, (char*)"filterByArea", NULL},
    {(char*)"filterByCircularity", (getter)pyopencv_SimpleBlobDetector_Params_get_filterByCircularity, (setter)pyopencv_SimpleBlobDetector_Params_set_filterByCircularity, (char*)"filterByCircularity", NULL},
    {(char*)"filterByColor", (getter)pyopencv_SimpleBlobDetector_Params_get_filterByColor, (setter)pyopencv_SimpleBlobDetector_Params_set_filterByColor, (char*)"filterByColor", NULL},
    {(char*)"filterByConvexity", (getter)pyopencv_SimpleBlobDetector_Params_get_filterByConvexity, (setter)pyopencv_SimpleBlobDetector_Params_set_filterByConvexity, (char*)"filterByConvexity", NULL},
    {(char*)"filterByInertia", (getter)pyopencv_SimpleBlobDetector_Params_get_filterByInertia, (setter)pyopencv_SimpleBlobDetector_Params_set_filterByInertia, (char*)"filterByInertia", NULL},
    {(char*)"maxArea", (getter)pyopencv_SimpleBlobDetector_Params_get_maxArea, (setter)pyopencv_SimpleBlobDetector_Params_set_maxArea, (char*)"maxArea", NULL},
    {(char*)"maxCircularity", (getter)pyopencv_SimpleBlobDetector_Params_get_maxCircularity, (setter)pyopencv_SimpleBlobDetector_Params_set_maxCircularity, (char*)"maxCircularity", NULL},
    {(char*)"maxConvexity", (getter)pyopencv_SimpleBlobDetector_Params_get_maxConvexity, (setter)pyopencv_SimpleBlobDetector_Params_set_maxConvexity, (char*)"maxConvexity", NULL},
    {(char*)"maxInertiaRatio", (getter)pyopencv_SimpleBlobDetector_Params_get_maxInertiaRatio, (setter)pyopencv_SimpleBlobDetector_Params_set_maxInertiaRatio, (char*)"maxInertiaRatio", NULL},
    {(char*)"maxThreshold", (getter)pyopencv_SimpleBlobDetector_Params_get_maxThreshold, (setter)pyopencv_SimpleBlobDetector_Params_set_maxThreshold, (char*)"maxThreshold", NULL},
    {(char*)"minArea", (getter)pyopencv_SimpleBlobDetector_Params_get_minArea, (setter)pyopencv_SimpleBlobDetector_Params_set_minArea, (char*)"minArea", NULL},
    {(char*)"minCircularity", (getter)pyopencv_SimpleBlobDetector_Params_get_minCircularity, (setter)pyopencv_SimpleBlobDetector_Params_set_minCircularity, (char*)"minCircularity", NULL},
    {(char*)"minConvexity", (getter)pyopencv_SimpleBlobDetector_Params_get_minConvexity, (setter)pyopencv_SimpleBlobDetector_Params_set_minConvexity, (char*)"minConvexity", NULL},
    {(char*)"minDistBetweenBlobs", (getter)pyopencv_SimpleBlobDetector_Params_get_minDistBetweenBlobs, (setter)pyopencv_SimpleBlobDetector_Params_set_minDistBetweenBlobs, (char*)"minDistBetweenBlobs", NULL},
    {(char*)"minInertiaRatio", (getter)pyopencv_SimpleBlobDetector_Params_get_minInertiaRatio, (setter)pyopencv_SimpleBlobDetector_Params_set_minInertiaRatio, (char*)"minInertiaRatio", NULL},
    {(char*)"minRepeatability", (getter)pyopencv_SimpleBlobDetector_Params_get_minRepeatability, (setter)pyopencv_SimpleBlobDetector_Params_set_minRepeatability, (char*)"minRepeatability", NULL},
    {(char*)"minThreshold", (getter)pyopencv_SimpleBlobDetector_Params_get_minThreshold, (setter)pyopencv_SimpleBlobDetector_Params_set_minThreshold, (char*)"minThreshold", NULL},
    {(char*)"thresholdStep", (getter)pyopencv_SimpleBlobDetector_Params_get_thresholdStep, (setter)pyopencv_SimpleBlobDetector_Params_set_thresholdStep, (char*)"thresholdStep", NULL},
    {NULL}  /* Sentinel */
};



static PyMethodDef pyopencv_SimpleBlobDetector_Params_methods[] =
{

    {NULL,          NULL}
};

static void pyopencv_SimpleBlobDetector_Params_specials(void)
{
    pyopencv_SimpleBlobDetector_Params_Type.tp_base = NULL;
    pyopencv_SimpleBlobDetector_Params_Type.tp_dealloc = pyopencv_SimpleBlobDetector_Params_dealloc;
    pyopencv_SimpleBlobDetector_Params_Type.tp_repr = pyopencv_SimpleBlobDetector_Params_repr;
    pyopencv_SimpleBlobDetector_Params_Type.tp_getset = pyopencv_SimpleBlobDetector_Params_getseters;
    pyopencv_SimpleBlobDetector_Params_Type.tp_methods = pyopencv_SimpleBlobDetector_Params_methods;
}

static PyObject* pyopencv_KAZE_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<KAZE %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_KAZE_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_KAZE_getDiffusivity(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_KAZE_Type))
        return failmsgp("Incorrect type of self (must be 'KAZE' or its derivative)");
    cv::KAZE* _self_ = dynamic_cast<cv::KAZE*>(((pyopencv_KAZE_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getDiffusivity());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_KAZE_getExtended(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_KAZE_Type))
        return failmsgp("Incorrect type of self (must be 'KAZE' or its derivative)");
    cv::KAZE* _self_ = dynamic_cast<cv::KAZE*>(((pyopencv_KAZE_t*)self)->v.get());
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getExtended());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_KAZE_getNOctaveLayers(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_KAZE_Type))
        return failmsgp("Incorrect type of self (must be 'KAZE' or its derivative)");
    cv::KAZE* _self_ = dynamic_cast<cv::KAZE*>(((pyopencv_KAZE_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getNOctaveLayers());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_KAZE_getNOctaves(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_KAZE_Type))
        return failmsgp("Incorrect type of self (must be 'KAZE' or its derivative)");
    cv::KAZE* _self_ = dynamic_cast<cv::KAZE*>(((pyopencv_KAZE_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getNOctaves());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_KAZE_getThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_KAZE_Type))
        return failmsgp("Incorrect type of self (must be 'KAZE' or its derivative)");
    cv::KAZE* _self_ = dynamic_cast<cv::KAZE*>(((pyopencv_KAZE_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getThreshold());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_KAZE_getUpright(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_KAZE_Type))
        return failmsgp("Incorrect type of self (must be 'KAZE' or its derivative)");
    cv::KAZE* _self_ = dynamic_cast<cv::KAZE*>(((pyopencv_KAZE_t*)self)->v.get());
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getUpright());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_KAZE_setDiffusivity(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_KAZE_Type))
        return failmsgp("Incorrect type of self (must be 'KAZE' or its derivative)");
    cv::KAZE* _self_ = dynamic_cast<cv::KAZE*>(((pyopencv_KAZE_t*)self)->v.get());
    int diff=0;

    const char* keywords[] = { "diff", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:KAZE.setDiffusivity", (char**)keywords, &diff) )
    {
        ERRWRAP2(_self_->setDiffusivity(diff));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_KAZE_setExtended(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_KAZE_Type))
        return failmsgp("Incorrect type of self (must be 'KAZE' or its derivative)");
    cv::KAZE* _self_ = dynamic_cast<cv::KAZE*>(((pyopencv_KAZE_t*)self)->v.get());
    bool extended=0;

    const char* keywords[] = { "extended", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:KAZE.setExtended", (char**)keywords, &extended) )
    {
        ERRWRAP2(_self_->setExtended(extended));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_KAZE_setNOctaveLayers(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_KAZE_Type))
        return failmsgp("Incorrect type of self (must be 'KAZE' or its derivative)");
    cv::KAZE* _self_ = dynamic_cast<cv::KAZE*>(((pyopencv_KAZE_t*)self)->v.get());
    int octaveLayers=0;

    const char* keywords[] = { "octaveLayers", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:KAZE.setNOctaveLayers", (char**)keywords, &octaveLayers) )
    {
        ERRWRAP2(_self_->setNOctaveLayers(octaveLayers));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_KAZE_setNOctaves(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_KAZE_Type))
        return failmsgp("Incorrect type of self (must be 'KAZE' or its derivative)");
    cv::KAZE* _self_ = dynamic_cast<cv::KAZE*>(((pyopencv_KAZE_t*)self)->v.get());
    int octaves=0;

    const char* keywords[] = { "octaves", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:KAZE.setNOctaves", (char**)keywords, &octaves) )
    {
        ERRWRAP2(_self_->setNOctaves(octaves));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_KAZE_setThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_KAZE_Type))
        return failmsgp("Incorrect type of self (must be 'KAZE' or its derivative)");
    cv::KAZE* _self_ = dynamic_cast<cv::KAZE*>(((pyopencv_KAZE_t*)self)->v.get());
    double threshold=0;

    const char* keywords[] = { "threshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:KAZE.setThreshold", (char**)keywords, &threshold) )
    {
        ERRWRAP2(_self_->setThreshold(threshold));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_KAZE_setUpright(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_KAZE_Type))
        return failmsgp("Incorrect type of self (must be 'KAZE' or its derivative)");
    cv::KAZE* _self_ = dynamic_cast<cv::KAZE*>(((pyopencv_KAZE_t*)self)->v.get());
    bool upright=0;

    const char* keywords[] = { "upright", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:KAZE.setUpright", (char**)keywords, &upright) )
    {
        ERRWRAP2(_self_->setUpright(upright));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_KAZE_methods[] =
{
    {"getDiffusivity", (PyCFunction)pyopencv_cv_KAZE_getDiffusivity, METH_VARARGS | METH_KEYWORDS, "getDiffusivity() -> retval"},
    {"getExtended", (PyCFunction)pyopencv_cv_KAZE_getExtended, METH_VARARGS | METH_KEYWORDS, "getExtended() -> retval"},
    {"getNOctaveLayers", (PyCFunction)pyopencv_cv_KAZE_getNOctaveLayers, METH_VARARGS | METH_KEYWORDS, "getNOctaveLayers() -> retval"},
    {"getNOctaves", (PyCFunction)pyopencv_cv_KAZE_getNOctaves, METH_VARARGS | METH_KEYWORDS, "getNOctaves() -> retval"},
    {"getThreshold", (PyCFunction)pyopencv_cv_KAZE_getThreshold, METH_VARARGS | METH_KEYWORDS, "getThreshold() -> retval"},
    {"getUpright", (PyCFunction)pyopencv_cv_KAZE_getUpright, METH_VARARGS | METH_KEYWORDS, "getUpright() -> retval"},
    {"setDiffusivity", (PyCFunction)pyopencv_cv_KAZE_setDiffusivity, METH_VARARGS | METH_KEYWORDS, "setDiffusivity(diff) -> None"},
    {"setExtended", (PyCFunction)pyopencv_cv_KAZE_setExtended, METH_VARARGS | METH_KEYWORDS, "setExtended(extended) -> None"},
    {"setNOctaveLayers", (PyCFunction)pyopencv_cv_KAZE_setNOctaveLayers, METH_VARARGS | METH_KEYWORDS, "setNOctaveLayers(octaveLayers) -> None"},
    {"setNOctaves", (PyCFunction)pyopencv_cv_KAZE_setNOctaves, METH_VARARGS | METH_KEYWORDS, "setNOctaves(octaves) -> None"},
    {"setThreshold", (PyCFunction)pyopencv_cv_KAZE_setThreshold, METH_VARARGS | METH_KEYWORDS, "setThreshold(threshold) -> None"},
    {"setUpright", (PyCFunction)pyopencv_cv_KAZE_setUpright, METH_VARARGS | METH_KEYWORDS, "setUpright(upright) -> None"},

    {NULL,          NULL}
};

static void pyopencv_KAZE_specials(void)
{
    pyopencv_KAZE_Type.tp_base = &pyopencv_Feature2D_Type;
    pyopencv_KAZE_Type.tp_dealloc = pyopencv_KAZE_dealloc;
    pyopencv_KAZE_Type.tp_repr = pyopencv_KAZE_repr;
    pyopencv_KAZE_Type.tp_getset = pyopencv_KAZE_getseters;
    pyopencv_KAZE_Type.tp_methods = pyopencv_KAZE_methods;
}

static PyObject* pyopencv_AKAZE_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<AKAZE %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_AKAZE_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_AKAZE_getDescriptorChannels(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AKAZE_Type))
        return failmsgp("Incorrect type of self (must be 'AKAZE' or its derivative)");
    cv::AKAZE* _self_ = dynamic_cast<cv::AKAZE*>(((pyopencv_AKAZE_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getDescriptorChannels());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_AKAZE_getDescriptorSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AKAZE_Type))
        return failmsgp("Incorrect type of self (must be 'AKAZE' or its derivative)");
    cv::AKAZE* _self_ = dynamic_cast<cv::AKAZE*>(((pyopencv_AKAZE_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getDescriptorSize());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_AKAZE_getDescriptorType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AKAZE_Type))
        return failmsgp("Incorrect type of self (must be 'AKAZE' or its derivative)");
    cv::AKAZE* _self_ = dynamic_cast<cv::AKAZE*>(((pyopencv_AKAZE_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getDescriptorType());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_AKAZE_getDiffusivity(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AKAZE_Type))
        return failmsgp("Incorrect type of self (must be 'AKAZE' or its derivative)");
    cv::AKAZE* _self_ = dynamic_cast<cv::AKAZE*>(((pyopencv_AKAZE_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getDiffusivity());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_AKAZE_getNOctaveLayers(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AKAZE_Type))
        return failmsgp("Incorrect type of self (must be 'AKAZE' or its derivative)");
    cv::AKAZE* _self_ = dynamic_cast<cv::AKAZE*>(((pyopencv_AKAZE_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getNOctaveLayers());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_AKAZE_getNOctaves(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AKAZE_Type))
        return failmsgp("Incorrect type of self (must be 'AKAZE' or its derivative)");
    cv::AKAZE* _self_ = dynamic_cast<cv::AKAZE*>(((pyopencv_AKAZE_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getNOctaves());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_AKAZE_getThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AKAZE_Type))
        return failmsgp("Incorrect type of self (must be 'AKAZE' or its derivative)");
    cv::AKAZE* _self_ = dynamic_cast<cv::AKAZE*>(((pyopencv_AKAZE_t*)self)->v.get());
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getThreshold());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_AKAZE_setDescriptorChannels(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AKAZE_Type))
        return failmsgp("Incorrect type of self (must be 'AKAZE' or its derivative)");
    cv::AKAZE* _self_ = dynamic_cast<cv::AKAZE*>(((pyopencv_AKAZE_t*)self)->v.get());
    int dch=0;

    const char* keywords[] = { "dch", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:AKAZE.setDescriptorChannels", (char**)keywords, &dch) )
    {
        ERRWRAP2(_self_->setDescriptorChannels(dch));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_AKAZE_setDescriptorSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AKAZE_Type))
        return failmsgp("Incorrect type of self (must be 'AKAZE' or its derivative)");
    cv::AKAZE* _self_ = dynamic_cast<cv::AKAZE*>(((pyopencv_AKAZE_t*)self)->v.get());
    int dsize=0;

    const char* keywords[] = { "dsize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:AKAZE.setDescriptorSize", (char**)keywords, &dsize) )
    {
        ERRWRAP2(_self_->setDescriptorSize(dsize));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_AKAZE_setDescriptorType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AKAZE_Type))
        return failmsgp("Incorrect type of self (must be 'AKAZE' or its derivative)");
    cv::AKAZE* _self_ = dynamic_cast<cv::AKAZE*>(((pyopencv_AKAZE_t*)self)->v.get());
    int dtype=0;

    const char* keywords[] = { "dtype", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:AKAZE.setDescriptorType", (char**)keywords, &dtype) )
    {
        ERRWRAP2(_self_->setDescriptorType(dtype));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_AKAZE_setDiffusivity(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AKAZE_Type))
        return failmsgp("Incorrect type of self (must be 'AKAZE' or its derivative)");
    cv::AKAZE* _self_ = dynamic_cast<cv::AKAZE*>(((pyopencv_AKAZE_t*)self)->v.get());
    int diff=0;

    const char* keywords[] = { "diff", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:AKAZE.setDiffusivity", (char**)keywords, &diff) )
    {
        ERRWRAP2(_self_->setDiffusivity(diff));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_AKAZE_setNOctaveLayers(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AKAZE_Type))
        return failmsgp("Incorrect type of self (must be 'AKAZE' or its derivative)");
    cv::AKAZE* _self_ = dynamic_cast<cv::AKAZE*>(((pyopencv_AKAZE_t*)self)->v.get());
    int octaveLayers=0;

    const char* keywords[] = { "octaveLayers", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:AKAZE.setNOctaveLayers", (char**)keywords, &octaveLayers) )
    {
        ERRWRAP2(_self_->setNOctaveLayers(octaveLayers));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_AKAZE_setNOctaves(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AKAZE_Type))
        return failmsgp("Incorrect type of self (must be 'AKAZE' or its derivative)");
    cv::AKAZE* _self_ = dynamic_cast<cv::AKAZE*>(((pyopencv_AKAZE_t*)self)->v.get());
    int octaves=0;

    const char* keywords[] = { "octaves", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:AKAZE.setNOctaves", (char**)keywords, &octaves) )
    {
        ERRWRAP2(_self_->setNOctaves(octaves));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_AKAZE_setThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_AKAZE_Type))
        return failmsgp("Incorrect type of self (must be 'AKAZE' or its derivative)");
    cv::AKAZE* _self_ = dynamic_cast<cv::AKAZE*>(((pyopencv_AKAZE_t*)self)->v.get());
    double threshold=0;

    const char* keywords[] = { "threshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:AKAZE.setThreshold", (char**)keywords, &threshold) )
    {
        ERRWRAP2(_self_->setThreshold(threshold));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_AKAZE_methods[] =
{
    {"getDescriptorChannels", (PyCFunction)pyopencv_cv_AKAZE_getDescriptorChannels, METH_VARARGS | METH_KEYWORDS, "getDescriptorChannels() -> retval"},
    {"getDescriptorSize", (PyCFunction)pyopencv_cv_AKAZE_getDescriptorSize, METH_VARARGS | METH_KEYWORDS, "getDescriptorSize() -> retval"},
    {"getDescriptorType", (PyCFunction)pyopencv_cv_AKAZE_getDescriptorType, METH_VARARGS | METH_KEYWORDS, "getDescriptorType() -> retval"},
    {"getDiffusivity", (PyCFunction)pyopencv_cv_AKAZE_getDiffusivity, METH_VARARGS | METH_KEYWORDS, "getDiffusivity() -> retval"},
    {"getNOctaveLayers", (PyCFunction)pyopencv_cv_AKAZE_getNOctaveLayers, METH_VARARGS | METH_KEYWORDS, "getNOctaveLayers() -> retval"},
    {"getNOctaves", (PyCFunction)pyopencv_cv_AKAZE_getNOctaves, METH_VARARGS | METH_KEYWORDS, "getNOctaves() -> retval"},
    {"getThreshold", (PyCFunction)pyopencv_cv_AKAZE_getThreshold, METH_VARARGS | METH_KEYWORDS, "getThreshold() -> retval"},
    {"setDescriptorChannels", (PyCFunction)pyopencv_cv_AKAZE_setDescriptorChannels, METH_VARARGS | METH_KEYWORDS, "setDescriptorChannels(dch) -> None"},
    {"setDescriptorSize", (PyCFunction)pyopencv_cv_AKAZE_setDescriptorSize, METH_VARARGS | METH_KEYWORDS, "setDescriptorSize(dsize) -> None"},
    {"setDescriptorType", (PyCFunction)pyopencv_cv_AKAZE_setDescriptorType, METH_VARARGS | METH_KEYWORDS, "setDescriptorType(dtype) -> None"},
    {"setDiffusivity", (PyCFunction)pyopencv_cv_AKAZE_setDiffusivity, METH_VARARGS | METH_KEYWORDS, "setDiffusivity(diff) -> None"},
    {"setNOctaveLayers", (PyCFunction)pyopencv_cv_AKAZE_setNOctaveLayers, METH_VARARGS | METH_KEYWORDS, "setNOctaveLayers(octaveLayers) -> None"},
    {"setNOctaves", (PyCFunction)pyopencv_cv_AKAZE_setNOctaves, METH_VARARGS | METH_KEYWORDS, "setNOctaves(octaves) -> None"},
    {"setThreshold", (PyCFunction)pyopencv_cv_AKAZE_setThreshold, METH_VARARGS | METH_KEYWORDS, "setThreshold(threshold) -> None"},

    {NULL,          NULL}
};

static void pyopencv_AKAZE_specials(void)
{
    pyopencv_AKAZE_Type.tp_base = &pyopencv_Feature2D_Type;
    pyopencv_AKAZE_Type.tp_dealloc = pyopencv_AKAZE_dealloc;
    pyopencv_AKAZE_Type.tp_repr = pyopencv_AKAZE_repr;
    pyopencv_AKAZE_Type.tp_getset = pyopencv_AKAZE_getseters;
    pyopencv_AKAZE_Type.tp_methods = pyopencv_AKAZE_methods;
}

static PyObject* pyopencv_DescriptorMatcher_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<DescriptorMatcher %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_DescriptorMatcher_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_DescriptorMatcher_add(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DescriptorMatcher_Type))
        return failmsgp("Incorrect type of self (must be 'DescriptorMatcher' or its derivative)");
    cv::DescriptorMatcher* _self_ = dynamic_cast<cv::DescriptorMatcher*>(((pyopencv_DescriptorMatcher_t*)self)->v.get());
    {
    PyObject* pyobj_descriptors = NULL;
    vector_Mat descriptors;

    const char* keywords[] = { "descriptors", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:DescriptorMatcher.add", (char**)keywords, &pyobj_descriptors) &&
        pyopencv_to(pyobj_descriptors, descriptors, ArgInfo("descriptors", 0)) )
    {
        ERRWRAP2(_self_->add(descriptors));
        Py_RETURN_NONE;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_descriptors = NULL;
    vector_Mat descriptors;

    const char* keywords[] = { "descriptors", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:DescriptorMatcher.add", (char**)keywords, &pyobj_descriptors) &&
        pyopencv_to(pyobj_descriptors, descriptors, ArgInfo("descriptors", 0)) )
    {
        ERRWRAP2(_self_->add(descriptors));
        Py_RETURN_NONE;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_DescriptorMatcher_clear(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DescriptorMatcher_Type))
        return failmsgp("Incorrect type of self (must be 'DescriptorMatcher' or its derivative)");
    cv::DescriptorMatcher* _self_ = dynamic_cast<cv::DescriptorMatcher*>(((pyopencv_DescriptorMatcher_t*)self)->v.get());

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(_self_->clear());
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_DescriptorMatcher_clone(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DescriptorMatcher_Type))
        return failmsgp("Incorrect type of self (must be 'DescriptorMatcher' or its derivative)");
    cv::DescriptorMatcher* _self_ = dynamic_cast<cv::DescriptorMatcher*>(((pyopencv_DescriptorMatcher_t*)self)->v.get());
    bool emptyTrainData=false;
    Ptr<DescriptorMatcher> retval;

    const char* keywords[] = { "emptyTrainData", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|b:DescriptorMatcher.clone", (char**)keywords, &emptyTrainData) )
    {
        ERRWRAP2(retval = _self_->clone(emptyTrainData));
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_DescriptorMatcher_empty(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DescriptorMatcher_Type))
        return failmsgp("Incorrect type of self (must be 'DescriptorMatcher' or its derivative)");
    cv::DescriptorMatcher* _self_ = dynamic_cast<cv::DescriptorMatcher*>(((pyopencv_DescriptorMatcher_t*)self)->v.get());
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->empty());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_DescriptorMatcher_getTrainDescriptors(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DescriptorMatcher_Type))
        return failmsgp("Incorrect type of self (must be 'DescriptorMatcher' or its derivative)");
    cv::DescriptorMatcher* _self_ = dynamic_cast<cv::DescriptorMatcher*>(((pyopencv_DescriptorMatcher_t*)self)->v.get());
    std::vector<Mat> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTrainDescriptors());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_DescriptorMatcher_isMaskSupported(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DescriptorMatcher_Type))
        return failmsgp("Incorrect type of self (must be 'DescriptorMatcher' or its derivative)");
    cv::DescriptorMatcher* _self_ = dynamic_cast<cv::DescriptorMatcher*>(((pyopencv_DescriptorMatcher_t*)self)->v.get());
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->isMaskSupported());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_DescriptorMatcher_knnMatch(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DescriptorMatcher_Type))
        return failmsgp("Incorrect type of self (must be 'DescriptorMatcher' or its derivative)");
    cv::DescriptorMatcher* _self_ = dynamic_cast<cv::DescriptorMatcher*>(((pyopencv_DescriptorMatcher_t*)self)->v.get());
    {
    PyObject* pyobj_queryDescriptors = NULL;
    Mat queryDescriptors;
    PyObject* pyobj_trainDescriptors = NULL;
    Mat trainDescriptors;
    vector_vector_DMatch matches;
    int k=0;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    bool compactResult=false;

    const char* keywords[] = { "queryDescriptors", "trainDescriptors", "k", "mask", "compactResult", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi|Ob:DescriptorMatcher.knnMatch", (char**)keywords, &pyobj_queryDescriptors, &pyobj_trainDescriptors, &k, &pyobj_mask, &compactResult) &&
        pyopencv_to(pyobj_queryDescriptors, queryDescriptors, ArgInfo("queryDescriptors", 0)) &&
        pyopencv_to(pyobj_trainDescriptors, trainDescriptors, ArgInfo("trainDescriptors", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(_self_->knnMatch(queryDescriptors, trainDescriptors, matches, k, mask, compactResult));
        return pyopencv_from(matches);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_queryDescriptors = NULL;
    UMat queryDescriptors;
    PyObject* pyobj_trainDescriptors = NULL;
    UMat trainDescriptors;
    vector_vector_DMatch matches;
    int k=0;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    bool compactResult=false;

    const char* keywords[] = { "queryDescriptors", "trainDescriptors", "k", "mask", "compactResult", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOi|Ob:DescriptorMatcher.knnMatch", (char**)keywords, &pyobj_queryDescriptors, &pyobj_trainDescriptors, &k, &pyobj_mask, &compactResult) &&
        pyopencv_to(pyobj_queryDescriptors, queryDescriptors, ArgInfo("queryDescriptors", 0)) &&
        pyopencv_to(pyobj_trainDescriptors, trainDescriptors, ArgInfo("trainDescriptors", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(_self_->knnMatch(queryDescriptors, trainDescriptors, matches, k, mask, compactResult));
        return pyopencv_from(matches);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_queryDescriptors = NULL;
    Mat queryDescriptors;
    vector_vector_DMatch matches;
    int k=0;
    PyObject* pyobj_masks = NULL;
    vector_Mat masks;
    bool compactResult=false;

    const char* keywords[] = { "queryDescriptors", "k", "masks", "compactResult", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|Ob:DescriptorMatcher.knnMatch", (char**)keywords, &pyobj_queryDescriptors, &k, &pyobj_masks, &compactResult) &&
        pyopencv_to(pyobj_queryDescriptors, queryDescriptors, ArgInfo("queryDescriptors", 0)) &&
        pyopencv_to(pyobj_masks, masks, ArgInfo("masks", 0)) )
    {
        ERRWRAP2(_self_->knnMatch(queryDescriptors, matches, k, masks, compactResult));
        return pyopencv_from(matches);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_queryDescriptors = NULL;
    UMat queryDescriptors;
    vector_vector_DMatch matches;
    int k=0;
    PyObject* pyobj_masks = NULL;
    vector_Mat masks;
    bool compactResult=false;

    const char* keywords[] = { "queryDescriptors", "k", "masks", "compactResult", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Oi|Ob:DescriptorMatcher.knnMatch", (char**)keywords, &pyobj_queryDescriptors, &k, &pyobj_masks, &compactResult) &&
        pyopencv_to(pyobj_queryDescriptors, queryDescriptors, ArgInfo("queryDescriptors", 0)) &&
        pyopencv_to(pyobj_masks, masks, ArgInfo("masks", 0)) )
    {
        ERRWRAP2(_self_->knnMatch(queryDescriptors, matches, k, masks, compactResult));
        return pyopencv_from(matches);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_DescriptorMatcher_match(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DescriptorMatcher_Type))
        return failmsgp("Incorrect type of self (must be 'DescriptorMatcher' or its derivative)");
    cv::DescriptorMatcher* _self_ = dynamic_cast<cv::DescriptorMatcher*>(((pyopencv_DescriptorMatcher_t*)self)->v.get());
    {
    PyObject* pyobj_queryDescriptors = NULL;
    Mat queryDescriptors;
    PyObject* pyobj_trainDescriptors = NULL;
    Mat trainDescriptors;
    vector_DMatch matches;
    PyObject* pyobj_mask = NULL;
    Mat mask;

    const char* keywords[] = { "queryDescriptors", "trainDescriptors", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:DescriptorMatcher.match", (char**)keywords, &pyobj_queryDescriptors, &pyobj_trainDescriptors, &pyobj_mask) &&
        pyopencv_to(pyobj_queryDescriptors, queryDescriptors, ArgInfo("queryDescriptors", 0)) &&
        pyopencv_to(pyobj_trainDescriptors, trainDescriptors, ArgInfo("trainDescriptors", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(_self_->match(queryDescriptors, trainDescriptors, matches, mask));
        return pyopencv_from(matches);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_queryDescriptors = NULL;
    UMat queryDescriptors;
    PyObject* pyobj_trainDescriptors = NULL;
    UMat trainDescriptors;
    vector_DMatch matches;
    PyObject* pyobj_mask = NULL;
    UMat mask;

    const char* keywords[] = { "queryDescriptors", "trainDescriptors", "mask", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:DescriptorMatcher.match", (char**)keywords, &pyobj_queryDescriptors, &pyobj_trainDescriptors, &pyobj_mask) &&
        pyopencv_to(pyobj_queryDescriptors, queryDescriptors, ArgInfo("queryDescriptors", 0)) &&
        pyopencv_to(pyobj_trainDescriptors, trainDescriptors, ArgInfo("trainDescriptors", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(_self_->match(queryDescriptors, trainDescriptors, matches, mask));
        return pyopencv_from(matches);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_queryDescriptors = NULL;
    Mat queryDescriptors;
    vector_DMatch matches;
    PyObject* pyobj_masks = NULL;
    vector_Mat masks;

    const char* keywords[] = { "queryDescriptors", "masks", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:DescriptorMatcher.match", (char**)keywords, &pyobj_queryDescriptors, &pyobj_masks) &&
        pyopencv_to(pyobj_queryDescriptors, queryDescriptors, ArgInfo("queryDescriptors", 0)) &&
        pyopencv_to(pyobj_masks, masks, ArgInfo("masks", 0)) )
    {
        ERRWRAP2(_self_->match(queryDescriptors, matches, masks));
        return pyopencv_from(matches);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_queryDescriptors = NULL;
    UMat queryDescriptors;
    vector_DMatch matches;
    PyObject* pyobj_masks = NULL;
    vector_Mat masks;

    const char* keywords[] = { "queryDescriptors", "masks", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:DescriptorMatcher.match", (char**)keywords, &pyobj_queryDescriptors, &pyobj_masks) &&
        pyopencv_to(pyobj_queryDescriptors, queryDescriptors, ArgInfo("queryDescriptors", 0)) &&
        pyopencv_to(pyobj_masks, masks, ArgInfo("masks", 0)) )
    {
        ERRWRAP2(_self_->match(queryDescriptors, matches, masks));
        return pyopencv_from(matches);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_DescriptorMatcher_radiusMatch(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DescriptorMatcher_Type))
        return failmsgp("Incorrect type of self (must be 'DescriptorMatcher' or its derivative)");
    cv::DescriptorMatcher* _self_ = dynamic_cast<cv::DescriptorMatcher*>(((pyopencv_DescriptorMatcher_t*)self)->v.get());
    {
    PyObject* pyobj_queryDescriptors = NULL;
    Mat queryDescriptors;
    PyObject* pyobj_trainDescriptors = NULL;
    Mat trainDescriptors;
    vector_vector_DMatch matches;
    float maxDistance=0.f;
    PyObject* pyobj_mask = NULL;
    Mat mask;
    bool compactResult=false;

    const char* keywords[] = { "queryDescriptors", "trainDescriptors", "maxDistance", "mask", "compactResult", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOf|Ob:DescriptorMatcher.radiusMatch", (char**)keywords, &pyobj_queryDescriptors, &pyobj_trainDescriptors, &maxDistance, &pyobj_mask, &compactResult) &&
        pyopencv_to(pyobj_queryDescriptors, queryDescriptors, ArgInfo("queryDescriptors", 0)) &&
        pyopencv_to(pyobj_trainDescriptors, trainDescriptors, ArgInfo("trainDescriptors", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(_self_->radiusMatch(queryDescriptors, trainDescriptors, matches, maxDistance, mask, compactResult));
        return pyopencv_from(matches);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_queryDescriptors = NULL;
    UMat queryDescriptors;
    PyObject* pyobj_trainDescriptors = NULL;
    UMat trainDescriptors;
    vector_vector_DMatch matches;
    float maxDistance=0.f;
    PyObject* pyobj_mask = NULL;
    UMat mask;
    bool compactResult=false;

    const char* keywords[] = { "queryDescriptors", "trainDescriptors", "maxDistance", "mask", "compactResult", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OOf|Ob:DescriptorMatcher.radiusMatch", (char**)keywords, &pyobj_queryDescriptors, &pyobj_trainDescriptors, &maxDistance, &pyobj_mask, &compactResult) &&
        pyopencv_to(pyobj_queryDescriptors, queryDescriptors, ArgInfo("queryDescriptors", 0)) &&
        pyopencv_to(pyobj_trainDescriptors, trainDescriptors, ArgInfo("trainDescriptors", 0)) &&
        pyopencv_to(pyobj_mask, mask, ArgInfo("mask", 0)) )
    {
        ERRWRAP2(_self_->radiusMatch(queryDescriptors, trainDescriptors, matches, maxDistance, mask, compactResult));
        return pyopencv_from(matches);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_queryDescriptors = NULL;
    Mat queryDescriptors;
    vector_vector_DMatch matches;
    float maxDistance=0.f;
    PyObject* pyobj_masks = NULL;
    vector_Mat masks;
    bool compactResult=false;

    const char* keywords[] = { "queryDescriptors", "maxDistance", "masks", "compactResult", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Of|Ob:DescriptorMatcher.radiusMatch", (char**)keywords, &pyobj_queryDescriptors, &maxDistance, &pyobj_masks, &compactResult) &&
        pyopencv_to(pyobj_queryDescriptors, queryDescriptors, ArgInfo("queryDescriptors", 0)) &&
        pyopencv_to(pyobj_masks, masks, ArgInfo("masks", 0)) )
    {
        ERRWRAP2(_self_->radiusMatch(queryDescriptors, matches, maxDistance, masks, compactResult));
        return pyopencv_from(matches);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_queryDescriptors = NULL;
    UMat queryDescriptors;
    vector_vector_DMatch matches;
    float maxDistance=0.f;
    PyObject* pyobj_masks = NULL;
    vector_Mat masks;
    bool compactResult=false;

    const char* keywords[] = { "queryDescriptors", "maxDistance", "masks", "compactResult", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "Of|Ob:DescriptorMatcher.radiusMatch", (char**)keywords, &pyobj_queryDescriptors, &maxDistance, &pyobj_masks, &compactResult) &&
        pyopencv_to(pyobj_queryDescriptors, queryDescriptors, ArgInfo("queryDescriptors", 0)) &&
        pyopencv_to(pyobj_masks, masks, ArgInfo("masks", 0)) )
    {
        ERRWRAP2(_self_->radiusMatch(queryDescriptors, matches, maxDistance, masks, compactResult));
        return pyopencv_from(matches);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_DescriptorMatcher_read(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DescriptorMatcher_Type))
        return failmsgp("Incorrect type of self (must be 'DescriptorMatcher' or its derivative)");
    cv::DescriptorMatcher* _self_ = dynamic_cast<cv::DescriptorMatcher*>(((pyopencv_DescriptorMatcher_t*)self)->v.get());
    PyObject* pyobj_fileName = NULL;
    String fileName;

    const char* keywords[] = { "fileName", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:DescriptorMatcher.read", (char**)keywords, &pyobj_fileName) &&
        pyopencv_to(pyobj_fileName, fileName, ArgInfo("fileName", 0)) )
    {
        ERRWRAP2(_self_->read(fileName));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_DescriptorMatcher_train(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DescriptorMatcher_Type))
        return failmsgp("Incorrect type of self (must be 'DescriptorMatcher' or its derivative)");
    cv::DescriptorMatcher* _self_ = dynamic_cast<cv::DescriptorMatcher*>(((pyopencv_DescriptorMatcher_t*)self)->v.get());

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(_self_->train());
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_DescriptorMatcher_write(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_DescriptorMatcher_Type))
        return failmsgp("Incorrect type of self (must be 'DescriptorMatcher' or its derivative)");
    cv::DescriptorMatcher* _self_ = dynamic_cast<cv::DescriptorMatcher*>(((pyopencv_DescriptorMatcher_t*)self)->v.get());
    PyObject* pyobj_fileName = NULL;
    String fileName;

    const char* keywords[] = { "fileName", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:DescriptorMatcher.write", (char**)keywords, &pyobj_fileName) &&
        pyopencv_to(pyobj_fileName, fileName, ArgInfo("fileName", 0)) )
    {
        ERRWRAP2(_self_->write(fileName));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_DescriptorMatcher_methods[] =
{
    {"add", (PyCFunction)pyopencv_cv_DescriptorMatcher_add, METH_VARARGS | METH_KEYWORDS, "add(descriptors) -> None"},
    {"clear", (PyCFunction)pyopencv_cv_DescriptorMatcher_clear, METH_VARARGS | METH_KEYWORDS, "clear() -> None"},
    {"clone", (PyCFunction)pyopencv_cv_DescriptorMatcher_clone, METH_VARARGS | METH_KEYWORDS, "clone([, emptyTrainData]) -> retval"},
    {"empty", (PyCFunction)pyopencv_cv_DescriptorMatcher_empty, METH_VARARGS | METH_KEYWORDS, "empty() -> retval"},
    {"getTrainDescriptors", (PyCFunction)pyopencv_cv_DescriptorMatcher_getTrainDescriptors, METH_VARARGS | METH_KEYWORDS, "getTrainDescriptors() -> retval"},
    {"isMaskSupported", (PyCFunction)pyopencv_cv_DescriptorMatcher_isMaskSupported, METH_VARARGS | METH_KEYWORDS, "isMaskSupported() -> retval"},
    {"knnMatch", (PyCFunction)pyopencv_cv_DescriptorMatcher_knnMatch, METH_VARARGS | METH_KEYWORDS, "knnMatch(queryDescriptors, trainDescriptors, k[, mask[, compactResult]]) -> matches  or  knnMatch(queryDescriptors, k[, masks[, compactResult]]) -> matches"},
    {"match", (PyCFunction)pyopencv_cv_DescriptorMatcher_match, METH_VARARGS | METH_KEYWORDS, "match(queryDescriptors, trainDescriptors[, mask]) -> matches  or  match(queryDescriptors[, masks]) -> matches"},
    {"radiusMatch", (PyCFunction)pyopencv_cv_DescriptorMatcher_radiusMatch, METH_VARARGS | METH_KEYWORDS, "radiusMatch(queryDescriptors, trainDescriptors, maxDistance[, mask[, compactResult]]) -> matches  or  radiusMatch(queryDescriptors, maxDistance[, masks[, compactResult]]) -> matches"},
    {"read", (PyCFunction)pyopencv_cv_DescriptorMatcher_read, METH_VARARGS | METH_KEYWORDS, "read(fileName) -> None"},
    {"train", (PyCFunction)pyopencv_cv_DescriptorMatcher_train, METH_VARARGS | METH_KEYWORDS, "train() -> None"},
    {"write", (PyCFunction)pyopencv_cv_DescriptorMatcher_write, METH_VARARGS | METH_KEYWORDS, "write(fileName) -> None"},

    {NULL,          NULL}
};

static void pyopencv_DescriptorMatcher_specials(void)
{
    pyopencv_DescriptorMatcher_Type.tp_base = &pyopencv_Algorithm_Type;
    pyopencv_DescriptorMatcher_Type.tp_dealloc = pyopencv_DescriptorMatcher_dealloc;
    pyopencv_DescriptorMatcher_Type.tp_repr = pyopencv_DescriptorMatcher_repr;
    pyopencv_DescriptorMatcher_Type.tp_getset = pyopencv_DescriptorMatcher_getseters;
    pyopencv_DescriptorMatcher_Type.tp_methods = pyopencv_DescriptorMatcher_methods;
}

static PyObject* pyopencv_BFMatcher_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<BFMatcher %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_BFMatcher_getseters[] =
{
    {NULL}  /* Sentinel */
};



static PyMethodDef pyopencv_BFMatcher_methods[] =
{

    {NULL,          NULL}
};

static void pyopencv_BFMatcher_specials(void)
{
    pyopencv_BFMatcher_Type.tp_base = &pyopencv_DescriptorMatcher_Type;
    pyopencv_BFMatcher_Type.tp_dealloc = pyopencv_BFMatcher_dealloc;
    pyopencv_BFMatcher_Type.tp_repr = pyopencv_BFMatcher_repr;
    pyopencv_BFMatcher_Type.tp_getset = pyopencv_BFMatcher_getseters;
    pyopencv_BFMatcher_Type.tp_methods = pyopencv_BFMatcher_methods;
}

static PyObject* pyopencv_FlannBasedMatcher_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<FlannBasedMatcher %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_FlannBasedMatcher_getseters[] =
{
    {NULL}  /* Sentinel */
};



static PyMethodDef pyopencv_FlannBasedMatcher_methods[] =
{

    {NULL,          NULL}
};

static void pyopencv_FlannBasedMatcher_specials(void)
{
    pyopencv_FlannBasedMatcher_Type.tp_base = &pyopencv_DescriptorMatcher_Type;
    pyopencv_FlannBasedMatcher_Type.tp_dealloc = pyopencv_FlannBasedMatcher_dealloc;
    pyopencv_FlannBasedMatcher_Type.tp_repr = pyopencv_FlannBasedMatcher_repr;
    pyopencv_FlannBasedMatcher_Type.tp_getset = pyopencv_FlannBasedMatcher_getseters;
    pyopencv_FlannBasedMatcher_Type.tp_methods = pyopencv_FlannBasedMatcher_methods;
}

static PyObject* pyopencv_BOWTrainer_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<BOWTrainer %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_BOWTrainer_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_BOWTrainer_add(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BOWTrainer_Type))
        return failmsgp("Incorrect type of self (must be 'BOWTrainer' or its derivative)");
    cv::BOWTrainer* _self_ = ((pyopencv_BOWTrainer_t*)self)->v.get();
    {
    PyObject* pyobj_descriptors = NULL;
    Mat descriptors;

    const char* keywords[] = { "descriptors", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:BOWTrainer.add", (char**)keywords, &pyobj_descriptors) &&
        pyopencv_to(pyobj_descriptors, descriptors, ArgInfo("descriptors", 0)) )
    {
        ERRWRAP2(_self_->add(descriptors));
        Py_RETURN_NONE;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_descriptors = NULL;
    Mat descriptors;

    const char* keywords[] = { "descriptors", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:BOWTrainer.add", (char**)keywords, &pyobj_descriptors) &&
        pyopencv_to(pyobj_descriptors, descriptors, ArgInfo("descriptors", 0)) )
    {
        ERRWRAP2(_self_->add(descriptors));
        Py_RETURN_NONE;
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_BOWTrainer_clear(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BOWTrainer_Type))
        return failmsgp("Incorrect type of self (must be 'BOWTrainer' or its derivative)");
    cv::BOWTrainer* _self_ = ((pyopencv_BOWTrainer_t*)self)->v.get();

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(_self_->clear());
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_BOWTrainer_cluster(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BOWTrainer_Type))
        return failmsgp("Incorrect type of self (must be 'BOWTrainer' or its derivative)");
    cv::BOWTrainer* _self_ = ((pyopencv_BOWTrainer_t*)self)->v.get();
    {
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->cluster());
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_descriptors = NULL;
    Mat descriptors;
    Mat retval;

    const char* keywords[] = { "descriptors", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:BOWTrainer.cluster", (char**)keywords, &pyobj_descriptors) &&
        pyopencv_to(pyobj_descriptors, descriptors, ArgInfo("descriptors", 0)) )
    {
        ERRWRAP2(retval = _self_->cluster(descriptors));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_descriptors = NULL;
    Mat descriptors;
    Mat retval;

    const char* keywords[] = { "descriptors", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:BOWTrainer.cluster", (char**)keywords, &pyobj_descriptors) &&
        pyopencv_to(pyobj_descriptors, descriptors, ArgInfo("descriptors", 0)) )
    {
        ERRWRAP2(retval = _self_->cluster(descriptors));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_BOWTrainer_descriptorsCount(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BOWTrainer_Type))
        return failmsgp("Incorrect type of self (must be 'BOWTrainer' or its derivative)");
    cv::BOWTrainer* _self_ = ((pyopencv_BOWTrainer_t*)self)->v.get();
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->descriptorsCount());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_BOWTrainer_getDescriptors(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BOWTrainer_Type))
        return failmsgp("Incorrect type of self (must be 'BOWTrainer' or its derivative)");
    cv::BOWTrainer* _self_ = ((pyopencv_BOWTrainer_t*)self)->v.get();
    std::vector<Mat> retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getDescriptors());
        return pyopencv_from(retval);
    }

    return NULL;
}



static PyMethodDef pyopencv_BOWTrainer_methods[] =
{
    {"add", (PyCFunction)pyopencv_cv_BOWTrainer_add, METH_VARARGS | METH_KEYWORDS, "add(descriptors) -> None"},
    {"clear", (PyCFunction)pyopencv_cv_BOWTrainer_clear, METH_VARARGS | METH_KEYWORDS, "clear() -> None"},
    {"cluster", (PyCFunction)pyopencv_cv_BOWTrainer_cluster, METH_VARARGS | METH_KEYWORDS, "cluster() -> retval  or  cluster(descriptors) -> retval"},
    {"descriptorsCount", (PyCFunction)pyopencv_cv_BOWTrainer_descriptorsCount, METH_VARARGS | METH_KEYWORDS, "descriptorsCount() -> retval"},
    {"getDescriptors", (PyCFunction)pyopencv_cv_BOWTrainer_getDescriptors, METH_VARARGS | METH_KEYWORDS, "getDescriptors() -> retval"},

    {NULL,          NULL}
};

static void pyopencv_BOWTrainer_specials(void)
{
    pyopencv_BOWTrainer_Type.tp_base = NULL;
    pyopencv_BOWTrainer_Type.tp_dealloc = pyopencv_BOWTrainer_dealloc;
    pyopencv_BOWTrainer_Type.tp_repr = pyopencv_BOWTrainer_repr;
    pyopencv_BOWTrainer_Type.tp_getset = pyopencv_BOWTrainer_getseters;
    pyopencv_BOWTrainer_Type.tp_methods = pyopencv_BOWTrainer_methods;
}

static PyObject* pyopencv_BOWKMeansTrainer_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<BOWKMeansTrainer %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_BOWKMeansTrainer_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_BOWKMeansTrainer_cluster(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BOWKMeansTrainer_Type))
        return failmsgp("Incorrect type of self (must be 'BOWKMeansTrainer' or its derivative)");
    cv::BOWKMeansTrainer* _self_ = ((pyopencv_BOWKMeansTrainer_t*)self)->v.get();
    {
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->cluster());
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_descriptors = NULL;
    Mat descriptors;
    Mat retval;

    const char* keywords[] = { "descriptors", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:BOWKMeansTrainer.cluster", (char**)keywords, &pyobj_descriptors) &&
        pyopencv_to(pyobj_descriptors, descriptors, ArgInfo("descriptors", 0)) )
    {
        ERRWRAP2(retval = _self_->cluster(descriptors));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_descriptors = NULL;
    Mat descriptors;
    Mat retval;

    const char* keywords[] = { "descriptors", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:BOWKMeansTrainer.cluster", (char**)keywords, &pyobj_descriptors) &&
        pyopencv_to(pyobj_descriptors, descriptors, ArgInfo("descriptors", 0)) )
    {
        ERRWRAP2(retval = _self_->cluster(descriptors));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}



static PyMethodDef pyopencv_BOWKMeansTrainer_methods[] =
{
    {"cluster", (PyCFunction)pyopencv_cv_BOWKMeansTrainer_cluster, METH_VARARGS | METH_KEYWORDS, "cluster() -> retval  or  cluster(descriptors) -> retval"},

    {NULL,          NULL}
};

static void pyopencv_BOWKMeansTrainer_specials(void)
{
    pyopencv_BOWKMeansTrainer_Type.tp_base = &pyopencv_BOWTrainer_Type;
    pyopencv_BOWKMeansTrainer_Type.tp_dealloc = pyopencv_BOWKMeansTrainer_dealloc;
    pyopencv_BOWKMeansTrainer_Type.tp_repr = pyopencv_BOWKMeansTrainer_repr;
    pyopencv_BOWKMeansTrainer_Type.tp_getset = pyopencv_BOWKMeansTrainer_getseters;
    pyopencv_BOWKMeansTrainer_Type.tp_methods = pyopencv_BOWKMeansTrainer_methods;
}

static PyObject* pyopencv_BOWImgDescriptorExtractor_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<BOWImgDescriptorExtractor %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_BOWImgDescriptorExtractor_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_BOWImgDescriptorExtractor_compute(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BOWImgDescriptorExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'BOWImgDescriptorExtractor' or its derivative)");
    cv::BOWImgDescriptorExtractor* _self_ = ((pyopencv_BOWImgDescriptorExtractor_t*)self)->v.get();
    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_keypoints = NULL;
    vector_KeyPoint keypoints;
    PyObject* pyobj_imgDescriptor = NULL;
    Mat imgDescriptor;

    const char* keywords[] = { "image", "keypoints", "imgDescriptor", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:BOWImgDescriptorExtractor.compute", (char**)keywords, &pyobj_image, &pyobj_keypoints, &pyobj_imgDescriptor) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_keypoints, keypoints, ArgInfo("keypoints", 0)) &&
        pyopencv_to(pyobj_imgDescriptor, imgDescriptor, ArgInfo("imgDescriptor", 1)) )
    {
        ERRWRAP2(_self_->compute2(image, keypoints, imgDescriptor));
        return pyopencv_from(imgDescriptor);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_image = NULL;
    Mat image;
    PyObject* pyobj_keypoints = NULL;
    vector_KeyPoint keypoints;
    PyObject* pyobj_imgDescriptor = NULL;
    Mat imgDescriptor;

    const char* keywords[] = { "image", "keypoints", "imgDescriptor", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:BOWImgDescriptorExtractor.compute", (char**)keywords, &pyobj_image, &pyobj_keypoints, &pyobj_imgDescriptor) &&
        pyopencv_to(pyobj_image, image, ArgInfo("image", 0)) &&
        pyopencv_to(pyobj_keypoints, keypoints, ArgInfo("keypoints", 0)) &&
        pyopencv_to(pyobj_imgDescriptor, imgDescriptor, ArgInfo("imgDescriptor", 1)) )
    {
        ERRWRAP2(_self_->compute2(image, keypoints, imgDescriptor));
        return pyopencv_from(imgDescriptor);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_BOWImgDescriptorExtractor_descriptorSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BOWImgDescriptorExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'BOWImgDescriptorExtractor' or its derivative)");
    cv::BOWImgDescriptorExtractor* _self_ = ((pyopencv_BOWImgDescriptorExtractor_t*)self)->v.get();
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->descriptorSize());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_BOWImgDescriptorExtractor_descriptorType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BOWImgDescriptorExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'BOWImgDescriptorExtractor' or its derivative)");
    cv::BOWImgDescriptorExtractor* _self_ = ((pyopencv_BOWImgDescriptorExtractor_t*)self)->v.get();
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->descriptorType());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_BOWImgDescriptorExtractor_getVocabulary(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BOWImgDescriptorExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'BOWImgDescriptorExtractor' or its derivative)");
    cv::BOWImgDescriptorExtractor* _self_ = ((pyopencv_BOWImgDescriptorExtractor_t*)self)->v.get();
    Mat retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getVocabulary());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_BOWImgDescriptorExtractor_setVocabulary(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_BOWImgDescriptorExtractor_Type))
        return failmsgp("Incorrect type of self (must be 'BOWImgDescriptorExtractor' or its derivative)");
    cv::BOWImgDescriptorExtractor* _self_ = ((pyopencv_BOWImgDescriptorExtractor_t*)self)->v.get();
    {
    PyObject* pyobj_vocabulary = NULL;
    Mat vocabulary;

    const char* keywords[] = { "vocabulary", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:BOWImgDescriptorExtractor.setVocabulary", (char**)keywords, &pyobj_vocabulary) &&
        pyopencv_to(pyobj_vocabulary, vocabulary, ArgInfo("vocabulary", 0)) )
    {
        ERRWRAP2(_self_->setVocabulary(vocabulary));
        Py_RETURN_NONE;
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_vocabulary = NULL;
    Mat vocabulary;

    const char* keywords[] = { "vocabulary", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:BOWImgDescriptorExtractor.setVocabulary", (char**)keywords, &pyobj_vocabulary) &&
        pyopencv_to(pyobj_vocabulary, vocabulary, ArgInfo("vocabulary", 0)) )
    {
        ERRWRAP2(_self_->setVocabulary(vocabulary));
        Py_RETURN_NONE;
    }
    }

    return NULL;
}



static PyMethodDef pyopencv_BOWImgDescriptorExtractor_methods[] =
{
    {"compute", (PyCFunction)pyopencv_cv_BOWImgDescriptorExtractor_compute, METH_VARARGS | METH_KEYWORDS, "compute(image, keypoints[, imgDescriptor]) -> imgDescriptor"},
    {"descriptorSize", (PyCFunction)pyopencv_cv_BOWImgDescriptorExtractor_descriptorSize, METH_VARARGS | METH_KEYWORDS, "descriptorSize() -> retval"},
    {"descriptorType", (PyCFunction)pyopencv_cv_BOWImgDescriptorExtractor_descriptorType, METH_VARARGS | METH_KEYWORDS, "descriptorType() -> retval"},
    {"getVocabulary", (PyCFunction)pyopencv_cv_BOWImgDescriptorExtractor_getVocabulary, METH_VARARGS | METH_KEYWORDS, "getVocabulary() -> retval"},
    {"setVocabulary", (PyCFunction)pyopencv_cv_BOWImgDescriptorExtractor_setVocabulary, METH_VARARGS | METH_KEYWORDS, "setVocabulary(vocabulary) -> None"},

    {NULL,          NULL}
};

static void pyopencv_BOWImgDescriptorExtractor_specials(void)
{
    pyopencv_BOWImgDescriptorExtractor_Type.tp_base = NULL;
    pyopencv_BOWImgDescriptorExtractor_Type.tp_dealloc = pyopencv_BOWImgDescriptorExtractor_dealloc;
    pyopencv_BOWImgDescriptorExtractor_Type.tp_repr = pyopencv_BOWImgDescriptorExtractor_repr;
    pyopencv_BOWImgDescriptorExtractor_Type.tp_getset = pyopencv_BOWImgDescriptorExtractor_getseters;
    pyopencv_BOWImgDescriptorExtractor_Type.tp_methods = pyopencv_BOWImgDescriptorExtractor_methods;
}

static PyObject* pyopencv_StereoMatcher_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<StereoMatcher %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_StereoMatcher_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_StereoMatcher_compute(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoMatcher_Type))
        return failmsgp("Incorrect type of self (must be 'StereoMatcher' or its derivative)");
    cv::StereoMatcher* _self_ = dynamic_cast<cv::StereoMatcher*>(((pyopencv_StereoMatcher_t*)self)->v.get());
    {
    PyObject* pyobj_left = NULL;
    Mat left;
    PyObject* pyobj_right = NULL;
    Mat right;
    PyObject* pyobj_disparity = NULL;
    Mat disparity;

    const char* keywords[] = { "left", "right", "disparity", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:StereoMatcher.compute", (char**)keywords, &pyobj_left, &pyobj_right, &pyobj_disparity) &&
        pyopencv_to(pyobj_left, left, ArgInfo("left", 0)) &&
        pyopencv_to(pyobj_right, right, ArgInfo("right", 0)) &&
        pyopencv_to(pyobj_disparity, disparity, ArgInfo("disparity", 1)) )
    {
        ERRWRAP2(_self_->compute(left, right, disparity));
        return pyopencv_from(disparity);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_left = NULL;
    UMat left;
    PyObject* pyobj_right = NULL;
    UMat right;
    PyObject* pyobj_disparity = NULL;
    UMat disparity;

    const char* keywords[] = { "left", "right", "disparity", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "OO|O:StereoMatcher.compute", (char**)keywords, &pyobj_left, &pyobj_right, &pyobj_disparity) &&
        pyopencv_to(pyobj_left, left, ArgInfo("left", 0)) &&
        pyopencv_to(pyobj_right, right, ArgInfo("right", 0)) &&
        pyopencv_to(pyobj_disparity, disparity, ArgInfo("disparity", 1)) )
    {
        ERRWRAP2(_self_->compute(left, right, disparity));
        return pyopencv_from(disparity);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoMatcher_getBlockSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoMatcher_Type))
        return failmsgp("Incorrect type of self (must be 'StereoMatcher' or its derivative)");
    cv::StereoMatcher* _self_ = dynamic_cast<cv::StereoMatcher*>(((pyopencv_StereoMatcher_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getBlockSize());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoMatcher_getDisp12MaxDiff(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoMatcher_Type))
        return failmsgp("Incorrect type of self (must be 'StereoMatcher' or its derivative)");
    cv::StereoMatcher* _self_ = dynamic_cast<cv::StereoMatcher*>(((pyopencv_StereoMatcher_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getDisp12MaxDiff());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoMatcher_getMinDisparity(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoMatcher_Type))
        return failmsgp("Incorrect type of self (must be 'StereoMatcher' or its derivative)");
    cv::StereoMatcher* _self_ = dynamic_cast<cv::StereoMatcher*>(((pyopencv_StereoMatcher_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getMinDisparity());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoMatcher_getNumDisparities(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoMatcher_Type))
        return failmsgp("Incorrect type of self (must be 'StereoMatcher' or its derivative)");
    cv::StereoMatcher* _self_ = dynamic_cast<cv::StereoMatcher*>(((pyopencv_StereoMatcher_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getNumDisparities());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoMatcher_getSpeckleRange(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoMatcher_Type))
        return failmsgp("Incorrect type of self (must be 'StereoMatcher' or its derivative)");
    cv::StereoMatcher* _self_ = dynamic_cast<cv::StereoMatcher*>(((pyopencv_StereoMatcher_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getSpeckleRange());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoMatcher_getSpeckleWindowSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoMatcher_Type))
        return failmsgp("Incorrect type of self (must be 'StereoMatcher' or its derivative)");
    cv::StereoMatcher* _self_ = dynamic_cast<cv::StereoMatcher*>(((pyopencv_StereoMatcher_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getSpeckleWindowSize());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoMatcher_setBlockSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoMatcher_Type))
        return failmsgp("Incorrect type of self (must be 'StereoMatcher' or its derivative)");
    cv::StereoMatcher* _self_ = dynamic_cast<cv::StereoMatcher*>(((pyopencv_StereoMatcher_t*)self)->v.get());
    int blockSize=0;

    const char* keywords[] = { "blockSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:StereoMatcher.setBlockSize", (char**)keywords, &blockSize) )
    {
        ERRWRAP2(_self_->setBlockSize(blockSize));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoMatcher_setDisp12MaxDiff(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoMatcher_Type))
        return failmsgp("Incorrect type of self (must be 'StereoMatcher' or its derivative)");
    cv::StereoMatcher* _self_ = dynamic_cast<cv::StereoMatcher*>(((pyopencv_StereoMatcher_t*)self)->v.get());
    int disp12MaxDiff=0;

    const char* keywords[] = { "disp12MaxDiff", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:StereoMatcher.setDisp12MaxDiff", (char**)keywords, &disp12MaxDiff) )
    {
        ERRWRAP2(_self_->setDisp12MaxDiff(disp12MaxDiff));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoMatcher_setMinDisparity(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoMatcher_Type))
        return failmsgp("Incorrect type of self (must be 'StereoMatcher' or its derivative)");
    cv::StereoMatcher* _self_ = dynamic_cast<cv::StereoMatcher*>(((pyopencv_StereoMatcher_t*)self)->v.get());
    int minDisparity=0;

    const char* keywords[] = { "minDisparity", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:StereoMatcher.setMinDisparity", (char**)keywords, &minDisparity) )
    {
        ERRWRAP2(_self_->setMinDisparity(minDisparity));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoMatcher_setNumDisparities(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoMatcher_Type))
        return failmsgp("Incorrect type of self (must be 'StereoMatcher' or its derivative)");
    cv::StereoMatcher* _self_ = dynamic_cast<cv::StereoMatcher*>(((pyopencv_StereoMatcher_t*)self)->v.get());
    int numDisparities=0;

    const char* keywords[] = { "numDisparities", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:StereoMatcher.setNumDisparities", (char**)keywords, &numDisparities) )
    {
        ERRWRAP2(_self_->setNumDisparities(numDisparities));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoMatcher_setSpeckleRange(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoMatcher_Type))
        return failmsgp("Incorrect type of self (must be 'StereoMatcher' or its derivative)");
    cv::StereoMatcher* _self_ = dynamic_cast<cv::StereoMatcher*>(((pyopencv_StereoMatcher_t*)self)->v.get());
    int speckleRange=0;

    const char* keywords[] = { "speckleRange", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:StereoMatcher.setSpeckleRange", (char**)keywords, &speckleRange) )
    {
        ERRWRAP2(_self_->setSpeckleRange(speckleRange));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoMatcher_setSpeckleWindowSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoMatcher_Type))
        return failmsgp("Incorrect type of self (must be 'StereoMatcher' or its derivative)");
    cv::StereoMatcher* _self_ = dynamic_cast<cv::StereoMatcher*>(((pyopencv_StereoMatcher_t*)self)->v.get());
    int speckleWindowSize=0;

    const char* keywords[] = { "speckleWindowSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:StereoMatcher.setSpeckleWindowSize", (char**)keywords, &speckleWindowSize) )
    {
        ERRWRAP2(_self_->setSpeckleWindowSize(speckleWindowSize));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_StereoMatcher_methods[] =
{
    {"compute", (PyCFunction)pyopencv_cv_StereoMatcher_compute, METH_VARARGS | METH_KEYWORDS, "compute(left, right[, disparity]) -> disparity"},
    {"getBlockSize", (PyCFunction)pyopencv_cv_StereoMatcher_getBlockSize, METH_VARARGS | METH_KEYWORDS, "getBlockSize() -> retval"},
    {"getDisp12MaxDiff", (PyCFunction)pyopencv_cv_StereoMatcher_getDisp12MaxDiff, METH_VARARGS | METH_KEYWORDS, "getDisp12MaxDiff() -> retval"},
    {"getMinDisparity", (PyCFunction)pyopencv_cv_StereoMatcher_getMinDisparity, METH_VARARGS | METH_KEYWORDS, "getMinDisparity() -> retval"},
    {"getNumDisparities", (PyCFunction)pyopencv_cv_StereoMatcher_getNumDisparities, METH_VARARGS | METH_KEYWORDS, "getNumDisparities() -> retval"},
    {"getSpeckleRange", (PyCFunction)pyopencv_cv_StereoMatcher_getSpeckleRange, METH_VARARGS | METH_KEYWORDS, "getSpeckleRange() -> retval"},
    {"getSpeckleWindowSize", (PyCFunction)pyopencv_cv_StereoMatcher_getSpeckleWindowSize, METH_VARARGS | METH_KEYWORDS, "getSpeckleWindowSize() -> retval"},
    {"setBlockSize", (PyCFunction)pyopencv_cv_StereoMatcher_setBlockSize, METH_VARARGS | METH_KEYWORDS, "setBlockSize(blockSize) -> None"},
    {"setDisp12MaxDiff", (PyCFunction)pyopencv_cv_StereoMatcher_setDisp12MaxDiff, METH_VARARGS | METH_KEYWORDS, "setDisp12MaxDiff(disp12MaxDiff) -> None"},
    {"setMinDisparity", (PyCFunction)pyopencv_cv_StereoMatcher_setMinDisparity, METH_VARARGS | METH_KEYWORDS, "setMinDisparity(minDisparity) -> None"},
    {"setNumDisparities", (PyCFunction)pyopencv_cv_StereoMatcher_setNumDisparities, METH_VARARGS | METH_KEYWORDS, "setNumDisparities(numDisparities) -> None"},
    {"setSpeckleRange", (PyCFunction)pyopencv_cv_StereoMatcher_setSpeckleRange, METH_VARARGS | METH_KEYWORDS, "setSpeckleRange(speckleRange) -> None"},
    {"setSpeckleWindowSize", (PyCFunction)pyopencv_cv_StereoMatcher_setSpeckleWindowSize, METH_VARARGS | METH_KEYWORDS, "setSpeckleWindowSize(speckleWindowSize) -> None"},

    {NULL,          NULL}
};

static void pyopencv_StereoMatcher_specials(void)
{
    pyopencv_StereoMatcher_Type.tp_base = &pyopencv_Algorithm_Type;
    pyopencv_StereoMatcher_Type.tp_dealloc = pyopencv_StereoMatcher_dealloc;
    pyopencv_StereoMatcher_Type.tp_repr = pyopencv_StereoMatcher_repr;
    pyopencv_StereoMatcher_Type.tp_getset = pyopencv_StereoMatcher_getseters;
    pyopencv_StereoMatcher_Type.tp_methods = pyopencv_StereoMatcher_methods;
}

static PyObject* pyopencv_StereoBM_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<StereoBM %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_StereoBM_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_StereoBM_getPreFilterCap(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoBM' or its derivative)");
    cv::StereoBM* _self_ = dynamic_cast<cv::StereoBM*>(((pyopencv_StereoBM_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getPreFilterCap());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoBM_getPreFilterSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoBM' or its derivative)");
    cv::StereoBM* _self_ = dynamic_cast<cv::StereoBM*>(((pyopencv_StereoBM_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getPreFilterSize());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoBM_getPreFilterType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoBM' or its derivative)");
    cv::StereoBM* _self_ = dynamic_cast<cv::StereoBM*>(((pyopencv_StereoBM_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getPreFilterType());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoBM_getROI1(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoBM' or its derivative)");
    cv::StereoBM* _self_ = dynamic_cast<cv::StereoBM*>(((pyopencv_StereoBM_t*)self)->v.get());
    Rect retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getROI1());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoBM_getROI2(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoBM' or its derivative)");
    cv::StereoBM* _self_ = dynamic_cast<cv::StereoBM*>(((pyopencv_StereoBM_t*)self)->v.get());
    Rect retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getROI2());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoBM_getSmallerBlockSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoBM' or its derivative)");
    cv::StereoBM* _self_ = dynamic_cast<cv::StereoBM*>(((pyopencv_StereoBM_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getSmallerBlockSize());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoBM_getTextureThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoBM' or its derivative)");
    cv::StereoBM* _self_ = dynamic_cast<cv::StereoBM*>(((pyopencv_StereoBM_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getTextureThreshold());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoBM_getUniquenessRatio(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoBM' or its derivative)");
    cv::StereoBM* _self_ = dynamic_cast<cv::StereoBM*>(((pyopencv_StereoBM_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getUniquenessRatio());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoBM_setPreFilterCap(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoBM' or its derivative)");
    cv::StereoBM* _self_ = dynamic_cast<cv::StereoBM*>(((pyopencv_StereoBM_t*)self)->v.get());
    int preFilterCap=0;

    const char* keywords[] = { "preFilterCap", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:StereoBM.setPreFilterCap", (char**)keywords, &preFilterCap) )
    {
        ERRWRAP2(_self_->setPreFilterCap(preFilterCap));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoBM_setPreFilterSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoBM' or its derivative)");
    cv::StereoBM* _self_ = dynamic_cast<cv::StereoBM*>(((pyopencv_StereoBM_t*)self)->v.get());
    int preFilterSize=0;

    const char* keywords[] = { "preFilterSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:StereoBM.setPreFilterSize", (char**)keywords, &preFilterSize) )
    {
        ERRWRAP2(_self_->setPreFilterSize(preFilterSize));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoBM_setPreFilterType(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoBM' or its derivative)");
    cv::StereoBM* _self_ = dynamic_cast<cv::StereoBM*>(((pyopencv_StereoBM_t*)self)->v.get());
    int preFilterType=0;

    const char* keywords[] = { "preFilterType", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:StereoBM.setPreFilterType", (char**)keywords, &preFilterType) )
    {
        ERRWRAP2(_self_->setPreFilterType(preFilterType));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoBM_setROI1(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoBM' or its derivative)");
    cv::StereoBM* _self_ = dynamic_cast<cv::StereoBM*>(((pyopencv_StereoBM_t*)self)->v.get());
    PyObject* pyobj_roi1 = NULL;
    Rect roi1;

    const char* keywords[] = { "roi1", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:StereoBM.setROI1", (char**)keywords, &pyobj_roi1) &&
        pyopencv_to(pyobj_roi1, roi1, ArgInfo("roi1", 0)) )
    {
        ERRWRAP2(_self_->setROI1(roi1));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoBM_setROI2(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoBM' or its derivative)");
    cv::StereoBM* _self_ = dynamic_cast<cv::StereoBM*>(((pyopencv_StereoBM_t*)self)->v.get());
    PyObject* pyobj_roi2 = NULL;
    Rect roi2;

    const char* keywords[] = { "roi2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:StereoBM.setROI2", (char**)keywords, &pyobj_roi2) &&
        pyopencv_to(pyobj_roi2, roi2, ArgInfo("roi2", 0)) )
    {
        ERRWRAP2(_self_->setROI2(roi2));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoBM_setSmallerBlockSize(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoBM' or its derivative)");
    cv::StereoBM* _self_ = dynamic_cast<cv::StereoBM*>(((pyopencv_StereoBM_t*)self)->v.get());
    int blockSize=0;

    const char* keywords[] = { "blockSize", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:StereoBM.setSmallerBlockSize", (char**)keywords, &blockSize) )
    {
        ERRWRAP2(_self_->setSmallerBlockSize(blockSize));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoBM_setTextureThreshold(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoBM' or its derivative)");
    cv::StereoBM* _self_ = dynamic_cast<cv::StereoBM*>(((pyopencv_StereoBM_t*)self)->v.get());
    int textureThreshold=0;

    const char* keywords[] = { "textureThreshold", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:StereoBM.setTextureThreshold", (char**)keywords, &textureThreshold) )
    {
        ERRWRAP2(_self_->setTextureThreshold(textureThreshold));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoBM_setUniquenessRatio(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoBM' or its derivative)");
    cv::StereoBM* _self_ = dynamic_cast<cv::StereoBM*>(((pyopencv_StereoBM_t*)self)->v.get());
    int uniquenessRatio=0;

    const char* keywords[] = { "uniquenessRatio", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:StereoBM.setUniquenessRatio", (char**)keywords, &uniquenessRatio) )
    {
        ERRWRAP2(_self_->setUniquenessRatio(uniquenessRatio));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_StereoBM_methods[] =
{
    {"getPreFilterCap", (PyCFunction)pyopencv_cv_StereoBM_getPreFilterCap, METH_VARARGS | METH_KEYWORDS, "getPreFilterCap() -> retval"},
    {"getPreFilterSize", (PyCFunction)pyopencv_cv_StereoBM_getPreFilterSize, METH_VARARGS | METH_KEYWORDS, "getPreFilterSize() -> retval"},
    {"getPreFilterType", (PyCFunction)pyopencv_cv_StereoBM_getPreFilterType, METH_VARARGS | METH_KEYWORDS, "getPreFilterType() -> retval"},
    {"getROI1", (PyCFunction)pyopencv_cv_StereoBM_getROI1, METH_VARARGS | METH_KEYWORDS, "getROI1() -> retval"},
    {"getROI2", (PyCFunction)pyopencv_cv_StereoBM_getROI2, METH_VARARGS | METH_KEYWORDS, "getROI2() -> retval"},
    {"getSmallerBlockSize", (PyCFunction)pyopencv_cv_StereoBM_getSmallerBlockSize, METH_VARARGS | METH_KEYWORDS, "getSmallerBlockSize() -> retval"},
    {"getTextureThreshold", (PyCFunction)pyopencv_cv_StereoBM_getTextureThreshold, METH_VARARGS | METH_KEYWORDS, "getTextureThreshold() -> retval"},
    {"getUniquenessRatio", (PyCFunction)pyopencv_cv_StereoBM_getUniquenessRatio, METH_VARARGS | METH_KEYWORDS, "getUniquenessRatio() -> retval"},
    {"setPreFilterCap", (PyCFunction)pyopencv_cv_StereoBM_setPreFilterCap, METH_VARARGS | METH_KEYWORDS, "setPreFilterCap(preFilterCap) -> None"},
    {"setPreFilterSize", (PyCFunction)pyopencv_cv_StereoBM_setPreFilterSize, METH_VARARGS | METH_KEYWORDS, "setPreFilterSize(preFilterSize) -> None"},
    {"setPreFilterType", (PyCFunction)pyopencv_cv_StereoBM_setPreFilterType, METH_VARARGS | METH_KEYWORDS, "setPreFilterType(preFilterType) -> None"},
    {"setROI1", (PyCFunction)pyopencv_cv_StereoBM_setROI1, METH_VARARGS | METH_KEYWORDS, "setROI1(roi1) -> None"},
    {"setROI2", (PyCFunction)pyopencv_cv_StereoBM_setROI2, METH_VARARGS | METH_KEYWORDS, "setROI2(roi2) -> None"},
    {"setSmallerBlockSize", (PyCFunction)pyopencv_cv_StereoBM_setSmallerBlockSize, METH_VARARGS | METH_KEYWORDS, "setSmallerBlockSize(blockSize) -> None"},
    {"setTextureThreshold", (PyCFunction)pyopencv_cv_StereoBM_setTextureThreshold, METH_VARARGS | METH_KEYWORDS, "setTextureThreshold(textureThreshold) -> None"},
    {"setUniquenessRatio", (PyCFunction)pyopencv_cv_StereoBM_setUniquenessRatio, METH_VARARGS | METH_KEYWORDS, "setUniquenessRatio(uniquenessRatio) -> None"},

    {NULL,          NULL}
};

static void pyopencv_StereoBM_specials(void)
{
    pyopencv_StereoBM_Type.tp_base = &pyopencv_StereoMatcher_Type;
    pyopencv_StereoBM_Type.tp_dealloc = pyopencv_StereoBM_dealloc;
    pyopencv_StereoBM_Type.tp_repr = pyopencv_StereoBM_repr;
    pyopencv_StereoBM_Type.tp_getset = pyopencv_StereoBM_getseters;
    pyopencv_StereoBM_Type.tp_methods = pyopencv_StereoBM_methods;
}

static PyObject* pyopencv_StereoSGBM_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<StereoSGBM %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_StereoSGBM_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_StereoSGBM_getMode(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoSGBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoSGBM' or its derivative)");
    cv::StereoSGBM* _self_ = dynamic_cast<cv::StereoSGBM*>(((pyopencv_StereoSGBM_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getMode());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoSGBM_getP1(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoSGBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoSGBM' or its derivative)");
    cv::StereoSGBM* _self_ = dynamic_cast<cv::StereoSGBM*>(((pyopencv_StereoSGBM_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getP1());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoSGBM_getP2(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoSGBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoSGBM' or its derivative)");
    cv::StereoSGBM* _self_ = dynamic_cast<cv::StereoSGBM*>(((pyopencv_StereoSGBM_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getP2());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoSGBM_getPreFilterCap(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoSGBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoSGBM' or its derivative)");
    cv::StereoSGBM* _self_ = dynamic_cast<cv::StereoSGBM*>(((pyopencv_StereoSGBM_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getPreFilterCap());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoSGBM_getUniquenessRatio(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoSGBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoSGBM' or its derivative)");
    cv::StereoSGBM* _self_ = dynamic_cast<cv::StereoSGBM*>(((pyopencv_StereoSGBM_t*)self)->v.get());
    int retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->getUniquenessRatio());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoSGBM_setMode(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoSGBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoSGBM' or its derivative)");
    cv::StereoSGBM* _self_ = dynamic_cast<cv::StereoSGBM*>(((pyopencv_StereoSGBM_t*)self)->v.get());
    int mode=0;

    const char* keywords[] = { "mode", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:StereoSGBM.setMode", (char**)keywords, &mode) )
    {
        ERRWRAP2(_self_->setMode(mode));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoSGBM_setP1(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoSGBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoSGBM' or its derivative)");
    cv::StereoSGBM* _self_ = dynamic_cast<cv::StereoSGBM*>(((pyopencv_StereoSGBM_t*)self)->v.get());
    int P1=0;

    const char* keywords[] = { "P1", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:StereoSGBM.setP1", (char**)keywords, &P1) )
    {
        ERRWRAP2(_self_->setP1(P1));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoSGBM_setP2(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoSGBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoSGBM' or its derivative)");
    cv::StereoSGBM* _self_ = dynamic_cast<cv::StereoSGBM*>(((pyopencv_StereoSGBM_t*)self)->v.get());
    int P2=0;

    const char* keywords[] = { "P2", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:StereoSGBM.setP2", (char**)keywords, &P2) )
    {
        ERRWRAP2(_self_->setP2(P2));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoSGBM_setPreFilterCap(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoSGBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoSGBM' or its derivative)");
    cv::StereoSGBM* _self_ = dynamic_cast<cv::StereoSGBM*>(((pyopencv_StereoSGBM_t*)self)->v.get());
    int preFilterCap=0;

    const char* keywords[] = { "preFilterCap", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:StereoSGBM.setPreFilterCap", (char**)keywords, &preFilterCap) )
    {
        ERRWRAP2(_self_->setPreFilterCap(preFilterCap));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_StereoSGBM_setUniquenessRatio(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_StereoSGBM_Type))
        return failmsgp("Incorrect type of self (must be 'StereoSGBM' or its derivative)");
    cv::StereoSGBM* _self_ = dynamic_cast<cv::StereoSGBM*>(((pyopencv_StereoSGBM_t*)self)->v.get());
    int uniquenessRatio=0;

    const char* keywords[] = { "uniquenessRatio", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "i:StereoSGBM.setUniquenessRatio", (char**)keywords, &uniquenessRatio) )
    {
        ERRWRAP2(_self_->setUniquenessRatio(uniquenessRatio));
        Py_RETURN_NONE;
    }

    return NULL;
}



static PyMethodDef pyopencv_StereoSGBM_methods[] =
{
    {"getMode", (PyCFunction)pyopencv_cv_StereoSGBM_getMode, METH_VARARGS | METH_KEYWORDS, "getMode() -> retval"},
    {"getP1", (PyCFunction)pyopencv_cv_StereoSGBM_getP1, METH_VARARGS | METH_KEYWORDS, "getP1() -> retval"},
    {"getP2", (PyCFunction)pyopencv_cv_StereoSGBM_getP2, METH_VARARGS | METH_KEYWORDS, "getP2() -> retval"},
    {"getPreFilterCap", (PyCFunction)pyopencv_cv_StereoSGBM_getPreFilterCap, METH_VARARGS | METH_KEYWORDS, "getPreFilterCap() -> retval"},
    {"getUniquenessRatio", (PyCFunction)pyopencv_cv_StereoSGBM_getUniquenessRatio, METH_VARARGS | METH_KEYWORDS, "getUniquenessRatio() -> retval"},
    {"setMode", (PyCFunction)pyopencv_cv_StereoSGBM_setMode, METH_VARARGS | METH_KEYWORDS, "setMode(mode) -> None"},
    {"setP1", (PyCFunction)pyopencv_cv_StereoSGBM_setP1, METH_VARARGS | METH_KEYWORDS, "setP1(P1) -> None"},
    {"setP2", (PyCFunction)pyopencv_cv_StereoSGBM_setP2, METH_VARARGS | METH_KEYWORDS, "setP2(P2) -> None"},
    {"setPreFilterCap", (PyCFunction)pyopencv_cv_StereoSGBM_setPreFilterCap, METH_VARARGS | METH_KEYWORDS, "setPreFilterCap(preFilterCap) -> None"},
    {"setUniquenessRatio", (PyCFunction)pyopencv_cv_StereoSGBM_setUniquenessRatio, METH_VARARGS | METH_KEYWORDS, "setUniquenessRatio(uniquenessRatio) -> None"},

    {NULL,          NULL}
};

static void pyopencv_StereoSGBM_specials(void)
{
    pyopencv_StereoSGBM_Type.tp_base = &pyopencv_StereoMatcher_Type;
    pyopencv_StereoSGBM_Type.tp_dealloc = pyopencv_StereoSGBM_dealloc;
    pyopencv_StereoSGBM_Type.tp_repr = pyopencv_StereoSGBM_repr;
    pyopencv_StereoSGBM_Type.tp_getset = pyopencv_StereoSGBM_getseters;
    pyopencv_StereoSGBM_Type.tp_methods = pyopencv_StereoSGBM_methods;
}

static PyObject* pyopencv_Stitcher_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<Stitcher %p>", self);
    return PyString_FromString(str);
}



static PyGetSetDef pyopencv_Stitcher_getseters[] =
{
    {NULL}  /* Sentinel */
};

static PyObject* pyopencv_cv_Stitcher_composePanorama(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Stitcher_Type))
        return failmsgp("Incorrect type of self (must be 'Stitcher' or its derivative)");
    cv::Stitcher* _self_ = ((pyopencv_Stitcher_t*)self)->v.get();
    {
    PyObject* pyobj_pano = NULL;
    Mat pano;
    Status retval;

    const char* keywords[] = { "pano", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|O:Stitcher.composePanorama", (char**)keywords, &pyobj_pano) &&
        pyopencv_to(pyobj_pano, pano, ArgInfo("pano", 1)) )
    {
        ERRWRAP2(retval = _self_->composePanorama(pano));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(pano));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_pano = NULL;
    UMat pano;
    Status retval;

    const char* keywords[] = { "pano", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "|O:Stitcher.composePanorama", (char**)keywords, &pyobj_pano) &&
        pyopencv_to(pyobj_pano, pano, ArgInfo("pano", 1)) )
    {
        ERRWRAP2(retval = _self_->composePanorama(pano));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(pano));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_Stitcher_compositingResol(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Stitcher_Type))
        return failmsgp("Incorrect type of self (must be 'Stitcher' or its derivative)");
    cv::Stitcher* _self_ = ((pyopencv_Stitcher_t*)self)->v.get();
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->compositingResol());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_Stitcher_estimateTransform(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Stitcher_Type))
        return failmsgp("Incorrect type of self (must be 'Stitcher' or its derivative)");
    cv::Stitcher* _self_ = ((pyopencv_Stitcher_t*)self)->v.get();
    {
    PyObject* pyobj_images = NULL;
    vector_Mat images;
    Status retval;

    const char* keywords[] = { "images", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:Stitcher.estimateTransform", (char**)keywords, &pyobj_images) &&
        pyopencv_to(pyobj_images, images, ArgInfo("images", 0)) )
    {
        ERRWRAP2(retval = _self_->estimateTransform(images));
        return pyopencv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_images = NULL;
    vector_Mat images;
    Status retval;

    const char* keywords[] = { "images", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:Stitcher.estimateTransform", (char**)keywords, &pyobj_images) &&
        pyopencv_to(pyobj_images, images, ArgInfo("images", 0)) )
    {
        ERRWRAP2(retval = _self_->estimateTransform(images));
        return pyopencv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_Stitcher_panoConfidenceThresh(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Stitcher_Type))
        return failmsgp("Incorrect type of self (must be 'Stitcher' or its derivative)");
    cv::Stitcher* _self_ = ((pyopencv_Stitcher_t*)self)->v.get();
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->panoConfidenceThresh());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_Stitcher_registrationResol(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Stitcher_Type))
        return failmsgp("Incorrect type of self (must be 'Stitcher' or its derivative)");
    cv::Stitcher* _self_ = ((pyopencv_Stitcher_t*)self)->v.get();
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->registrationResol());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_Stitcher_seamEstimationResol(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Stitcher_Type))
        return failmsgp("Incorrect type of self (must be 'Stitcher' or its derivative)");
    cv::Stitcher* _self_ = ((pyopencv_Stitcher_t*)self)->v.get();
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->seamEstimationResol());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_Stitcher_setCompositingResol(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Stitcher_Type))
        return failmsgp("Incorrect type of self (must be 'Stitcher' or its derivative)");
    cv::Stitcher* _self_ = ((pyopencv_Stitcher_t*)self)->v.get();
    double resol_mpx=0;

    const char* keywords[] = { "resol_mpx", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:Stitcher.setCompositingResol", (char**)keywords, &resol_mpx) )
    {
        ERRWRAP2(_self_->setCompositingResol(resol_mpx));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_Stitcher_setPanoConfidenceThresh(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Stitcher_Type))
        return failmsgp("Incorrect type of self (must be 'Stitcher' or its derivative)");
    cv::Stitcher* _self_ = ((pyopencv_Stitcher_t*)self)->v.get();
    double conf_thresh=0;

    const char* keywords[] = { "conf_thresh", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:Stitcher.setPanoConfidenceThresh", (char**)keywords, &conf_thresh) )
    {
        ERRWRAP2(_self_->setPanoConfidenceThresh(conf_thresh));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_Stitcher_setRegistrationResol(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Stitcher_Type))
        return failmsgp("Incorrect type of self (must be 'Stitcher' or its derivative)");
    cv::Stitcher* _self_ = ((pyopencv_Stitcher_t*)self)->v.get();
    double resol_mpx=0;

    const char* keywords[] = { "resol_mpx", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:Stitcher.setRegistrationResol", (char**)keywords, &resol_mpx) )
    {
        ERRWRAP2(_self_->setRegistrationResol(resol_mpx));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_Stitcher_setSeamEstimationResol(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Stitcher_Type))
        return failmsgp("Incorrect type of self (must be 'Stitcher' or its derivative)");
    cv::Stitcher* _self_ = ((pyopencv_Stitcher_t*)self)->v.get();
    double resol_mpx=0;

    const char* keywords[] = { "resol_mpx", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "d:Stitcher.setSeamEstimationResol", (char**)keywords, &resol_mpx) )
    {
        ERRWRAP2(_self_->setSeamEstimationResol(resol_mpx));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_Stitcher_setWaveCorrection(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Stitcher_Type))
        return failmsgp("Incorrect type of self (must be 'Stitcher' or its derivative)");
    cv::Stitcher* _self_ = ((pyopencv_Stitcher_t*)self)->v.get();
    bool flag=0;

    const char* keywords[] = { "flag", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "b:Stitcher.setWaveCorrection", (char**)keywords, &flag) )
    {
        ERRWRAP2(_self_->setWaveCorrection(flag));
        Py_RETURN_NONE;
    }

    return NULL;
}

static PyObject* pyopencv_cv_Stitcher_stitch(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Stitcher_Type))
        return failmsgp("Incorrect type of self (must be 'Stitcher' or its derivative)");
    cv::Stitcher* _self_ = ((pyopencv_Stitcher_t*)self)->v.get();
    {
    PyObject* pyobj_images = NULL;
    vector_Mat images;
    PyObject* pyobj_pano = NULL;
    Mat pano;
    Status retval;

    const char* keywords[] = { "images", "pano", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:Stitcher.stitch", (char**)keywords, &pyobj_images, &pyobj_pano) &&
        pyopencv_to(pyobj_images, images, ArgInfo("images", 0)) &&
        pyopencv_to(pyobj_pano, pano, ArgInfo("pano", 1)) )
    {
        ERRWRAP2(retval = _self_->stitch(images, pano));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(pano));
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_images = NULL;
    vector_Mat images;
    PyObject* pyobj_pano = NULL;
    UMat pano;
    Status retval;

    const char* keywords[] = { "images", "pano", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O|O:Stitcher.stitch", (char**)keywords, &pyobj_images, &pyobj_pano) &&
        pyopencv_to(pyobj_images, images, ArgInfo("images", 0)) &&
        pyopencv_to(pyobj_pano, pano, ArgInfo("pano", 1)) )
    {
        ERRWRAP2(retval = _self_->stitch(images, pano));
        return Py_BuildValue("(NN)", pyopencv_from(retval), pyopencv_from(pano));
    }
    }

    return NULL;
}

static PyObject* pyopencv_cv_Stitcher_waveCorrection(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Stitcher_Type))
        return failmsgp("Incorrect type of self (must be 'Stitcher' or its derivative)");
    cv::Stitcher* _self_ = ((pyopencv_Stitcher_t*)self)->v.get();
    bool retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->waveCorrection());
        return pyopencv_from(retval);
    }

    return NULL;
}

static PyObject* pyopencv_cv_Stitcher_workScale(PyObject* self, PyObject* args, PyObject* kw)
{
    using namespace cv;

    if(!PyObject_TypeCheck(self, &pyopencv_Stitcher_Type))
        return failmsgp("Incorrect type of self (must be 'Stitcher' or its derivative)");
    cv::Stitcher* _self_ = ((pyopencv_Stitcher_t*)self)->v.get();
    double retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = _self_->workScale());
        return pyopencv_from(retval);
    }

    return NULL;
}



static PyMethodDef pyopencv_Stitcher_methods[] =
{
    {"composePanorama", (PyCFunction)pyopencv_cv_Stitcher_composePanorama, METH_VARARGS | METH_KEYWORDS, "composePanorama([, pano]) -> retval, pano"},
    {"compositingResol", (PyCFunction)pyopencv_cv_Stitcher_compositingResol, METH_VARARGS | METH_KEYWORDS, "compositingResol() -> retval"},
    {"estimateTransform", (PyCFunction)pyopencv_cv_Stitcher_estimateTransform, METH_VARARGS | METH_KEYWORDS, "estimateTransform(images) -> retval"},
    {"panoConfidenceThresh", (PyCFunction)pyopencv_cv_Stitcher_panoConfidenceThresh, METH_VARARGS | METH_KEYWORDS, "panoConfidenceThresh() -> retval"},
    {"registrationResol", (PyCFunction)pyopencv_cv_Stitcher_registrationResol, METH_VARARGS | METH_KEYWORDS, "registrationResol() -> retval"},
    {"seamEstimationResol", (PyCFunction)pyopencv_cv_Stitcher_seamEstimationResol, METH_VARARGS | METH_KEYWORDS, "seamEstimationResol() -> retval"},
    {"setCompositingResol", (PyCFunction)pyopencv_cv_Stitcher_setCompositingResol, METH_VARARGS | METH_KEYWORDS, "setCompositingResol(resol_mpx) -> None"},
    {"setPanoConfidenceThresh", (PyCFunction)pyopencv_cv_Stitcher_setPanoConfidenceThresh, METH_VARARGS | METH_KEYWORDS, "setPanoConfidenceThresh(conf_thresh) -> None"},
    {"setRegistrationResol", (PyCFunction)pyopencv_cv_Stitcher_setRegistrationResol, METH_VARARGS | METH_KEYWORDS, "setRegistrationResol(resol_mpx) -> None"},
    {"setSeamEstimationResol", (PyCFunction)pyopencv_cv_Stitcher_setSeamEstimationResol, METH_VARARGS | METH_KEYWORDS, "setSeamEstimationResol(resol_mpx) -> None"},
    {"setWaveCorrection", (PyCFunction)pyopencv_cv_Stitcher_setWaveCorrection, METH_VARARGS | METH_KEYWORDS, "setWaveCorrection(flag) -> None"},
    {"stitch", (PyCFunction)pyopencv_cv_Stitcher_stitch, METH_VARARGS | METH_KEYWORDS, "stitch(images[, pano]) -> retval, pano"},
    {"waveCorrection", (PyCFunction)pyopencv_cv_Stitcher_waveCorrection, METH_VARARGS | METH_KEYWORDS, "waveCorrection() -> retval"},
    {"workScale", (PyCFunction)pyopencv_cv_Stitcher_workScale, METH_VARARGS | METH_KEYWORDS, "workScale() -> retval"},

    {NULL,          NULL}
};

static void pyopencv_Stitcher_specials(void)
{
    pyopencv_Stitcher_Type.tp_base = NULL;
    pyopencv_Stitcher_Type.tp_dealloc = pyopencv_Stitcher_dealloc;
    pyopencv_Stitcher_Type.tp_repr = pyopencv_Stitcher_repr;
    pyopencv_Stitcher_Type.tp_getset = pyopencv_Stitcher_getseters;
    pyopencv_Stitcher_Type.tp_methods = pyopencv_Stitcher_methods;
}
