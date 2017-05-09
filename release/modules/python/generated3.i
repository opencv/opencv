
/*
  IplConvKernel is the OpenCV C struct
  IplConvKernel_t is the Python object
*/

struct IplConvKernel_t {
  PyObject_HEAD
  IplConvKernel *v;
};

static void IplConvKernel_dealloc(PyObject *self)
{
  IplConvKernel_t *p = (IplConvKernel_t*)self;
  cvReleaseIplConvKernel(&p->v);
  PyObject_Del(self);
}

static PyObject *IplConvKernel_repr(PyObject *self)
{
  IplConvKernel_t *p = (IplConvKernel_t*)self;
  char str[1000];
  sprintf(str, "<IplConvKernel %p>", p);
  return PyString_FromString(str);
}


static PyObject *IplConvKernel_get_nRows(IplConvKernel_t *p, void *closure)
{
  return PyInt_FromLong(p->v->nRows);
}

static int IplConvKernel_set_nRows(IplConvKernel_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the nRows attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The nRows attribute value must be a integer");
    return -1;
  }

  p->v->nRows = PyInt_AsLong(value);
  return 0;
}


static PyObject *IplConvKernel_get_anchorX(IplConvKernel_t *p, void *closure)
{
  return PyInt_FromLong(p->v->anchorX);
}

static int IplConvKernel_set_anchorX(IplConvKernel_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the anchorX attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The anchorX attribute value must be a integer");
    return -1;
  }

  p->v->anchorX = PyInt_AsLong(value);
  return 0;
}


static PyObject *IplConvKernel_get_nCols(IplConvKernel_t *p, void *closure)
{
  return PyInt_FromLong(p->v->nCols);
}

static int IplConvKernel_set_nCols(IplConvKernel_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the nCols attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The nCols attribute value must be a integer");
    return -1;
  }

  p->v->nCols = PyInt_AsLong(value);
  return 0;
}


static PyObject *IplConvKernel_get_anchorY(IplConvKernel_t *p, void *closure)
{
  return PyInt_FromLong(p->v->anchorY);
}

static int IplConvKernel_set_anchorY(IplConvKernel_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the anchorY attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The anchorY attribute value must be a integer");
    return -1;
  }

  p->v->anchorY = PyInt_AsLong(value);
  return 0;
}



static PyGetSetDef IplConvKernel_getseters[] = {

  
  {(char*)"nRows", (getter)IplConvKernel_get_nRows, (setter)IplConvKernel_set_nRows, (char*)"nRows", NULL},

  {(char*)"anchorX", (getter)IplConvKernel_get_anchorX, (setter)IplConvKernel_set_anchorX, (char*)"anchorX", NULL},

  {(char*)"nCols", (getter)IplConvKernel_get_nCols, (setter)IplConvKernel_set_nCols, (char*)"nCols", NULL},

  {(char*)"anchorY", (getter)IplConvKernel_get_anchorY, (setter)IplConvKernel_set_anchorY, (char*)"anchorY", NULL},

  {NULL}  /* Sentinel */
};

static PyTypeObject IplConvKernel_Type = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                                      /*size*/
  MODULESTR".IplConvKernel",              /*name*/
  sizeof(IplConvKernel_t),                /*basicsize*/
};

static void IplConvKernel_specials(void)
{
  IplConvKernel_Type.tp_dealloc = IplConvKernel_dealloc;
  IplConvKernel_Type.tp_repr = IplConvKernel_repr;
  IplConvKernel_Type.tp_getset = IplConvKernel_getseters;
}

static PyObject *FROM_IplConvKernelPTR(IplConvKernel *r)
{
  IplConvKernel_t *m = PyObject_NEW(IplConvKernel_t, &IplConvKernel_Type);
  m->v = r;
  return (PyObject*)m;
}

static int convert_to_IplConvKernelPTR(PyObject *o, IplConvKernel** dst, const char *name = "no_name")
{
  if (o == Py_None) { *dst = (IplConvKernel*)NULL; return 1; }
  if (PyType_IsSubtype(o->ob_type, &IplConvKernel_Type)) {
    *dst = ((IplConvKernel_t*)o)->v;
    return 1;
  } else {
    (*dst) = (IplConvKernel*)NULL;
    return failmsg("Expected IplConvKernel for argument '%s'", name);
  }
}



/*
  CvCapture is the OpenCV C struct
  Capture_t is the Python object
*/

struct Capture_t {
  PyObject_HEAD
  CvCapture *v;
};

static void Capture_dealloc(PyObject *self)
{
  Capture_t *p = (Capture_t*)self;
  cvReleaseCapture(&p->v);
  PyObject_Del(self);
}

static PyObject *Capture_repr(PyObject *self)
{
  Capture_t *p = (Capture_t*)self;
  char str[1000];
  sprintf(str, "<Capture %p>", p);
  return PyString_FromString(str);
}



static PyGetSetDef Capture_getseters[] = {

  
  {NULL}  /* Sentinel */
};

static PyTypeObject Capture_Type = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                                      /*size*/
  MODULESTR".Capture",              /*name*/
  sizeof(Capture_t),                /*basicsize*/
};

static void Capture_specials(void)
{
  Capture_Type.tp_dealloc = Capture_dealloc;
  Capture_Type.tp_repr = Capture_repr;
  Capture_Type.tp_getset = Capture_getseters;
}

static PyObject *FROM_CvCapturePTR(CvCapture *r)
{
  Capture_t *m = PyObject_NEW(Capture_t, &Capture_Type);
  m->v = r;
  return (PyObject*)m;
}

static int convert_to_CvCapturePTR(PyObject *o, CvCapture** dst, const char *name = "no_name")
{
  
  if (PyType_IsSubtype(o->ob_type, &Capture_Type)) {
    *dst = ((Capture_t*)o)->v;
    return 1;
  } else {
    (*dst) = (CvCapture*)NULL;
    return failmsg("Expected CvCapture for argument '%s'", name);
  }
}



/*
  CvHaarClassifierCascade is the OpenCV C struct
  HaarClassifierCascade_t is the Python object
*/

struct HaarClassifierCascade_t {
  PyObject_HEAD
  CvHaarClassifierCascade *v;
};

static void HaarClassifierCascade_dealloc(PyObject *self)
{
  HaarClassifierCascade_t *p = (HaarClassifierCascade_t*)self;
  cvReleaseHaarClassifierCascade(&p->v);
  PyObject_Del(self);
}

static PyObject *HaarClassifierCascade_repr(PyObject *self)
{
  HaarClassifierCascade_t *p = (HaarClassifierCascade_t*)self;
  char str[1000];
  sprintf(str, "<HaarClassifierCascade %p>", p);
  return PyString_FromString(str);
}



static PyGetSetDef HaarClassifierCascade_getseters[] = {

  
  {NULL}  /* Sentinel */
};

static PyTypeObject HaarClassifierCascade_Type = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                                      /*size*/
  MODULESTR".HaarClassifierCascade",              /*name*/
  sizeof(HaarClassifierCascade_t),                /*basicsize*/
};

static void HaarClassifierCascade_specials(void)
{
  HaarClassifierCascade_Type.tp_dealloc = HaarClassifierCascade_dealloc;
  HaarClassifierCascade_Type.tp_repr = HaarClassifierCascade_repr;
  HaarClassifierCascade_Type.tp_getset = HaarClassifierCascade_getseters;
}

static PyObject *FROM_CvHaarClassifierCascadePTR(CvHaarClassifierCascade *r)
{
  HaarClassifierCascade_t *m = PyObject_NEW(HaarClassifierCascade_t, &HaarClassifierCascade_Type);
  m->v = r;
  return (PyObject*)m;
}

static int convert_to_CvHaarClassifierCascadePTR(PyObject *o, CvHaarClassifierCascade** dst, const char *name = "no_name")
{
  
  if (PyType_IsSubtype(o->ob_type, &HaarClassifierCascade_Type)) {
    *dst = ((HaarClassifierCascade_t*)o)->v;
    return 1;
  } else {
    (*dst) = (CvHaarClassifierCascade*)NULL;
    return failmsg("Expected CvHaarClassifierCascade for argument '%s'", name);
  }
}



/*
  CvPOSITObject is the OpenCV C struct
  POSITObject_t is the Python object
*/

struct POSITObject_t {
  PyObject_HEAD
  CvPOSITObject *v;
};

static void POSITObject_dealloc(PyObject *self)
{
  POSITObject_t *p = (POSITObject_t*)self;
  cvReleasePOSITObject(&p->v);
  PyObject_Del(self);
}

static PyObject *POSITObject_repr(PyObject *self)
{
  POSITObject_t *p = (POSITObject_t*)self;
  char str[1000];
  sprintf(str, "<POSITObject %p>", p);
  return PyString_FromString(str);
}



static PyGetSetDef POSITObject_getseters[] = {

  
  {NULL}  /* Sentinel */
};

static PyTypeObject POSITObject_Type = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                                      /*size*/
  MODULESTR".POSITObject",              /*name*/
  sizeof(POSITObject_t),                /*basicsize*/
};

static void POSITObject_specials(void)
{
  POSITObject_Type.tp_dealloc = POSITObject_dealloc;
  POSITObject_Type.tp_repr = POSITObject_repr;
  POSITObject_Type.tp_getset = POSITObject_getseters;
}

static PyObject *FROM_CvPOSITObjectPTR(CvPOSITObject *r)
{
  POSITObject_t *m = PyObject_NEW(POSITObject_t, &POSITObject_Type);
  m->v = r;
  return (PyObject*)m;
}

static int convert_to_CvPOSITObjectPTR(PyObject *o, CvPOSITObject** dst, const char *name = "no_name")
{
  
  if (PyType_IsSubtype(o->ob_type, &POSITObject_Type)) {
    *dst = ((POSITObject_t*)o)->v;
    return 1;
  } else {
    (*dst) = (CvPOSITObject*)NULL;
    return failmsg("Expected CvPOSITObject for argument '%s'", name);
  }
}



/*
  CvVideoWriter is the OpenCV C struct
  VideoWriter_t is the Python object
*/

struct VideoWriter_t {
  PyObject_HEAD
  CvVideoWriter *v;
};

static void VideoWriter_dealloc(PyObject *self)
{
  VideoWriter_t *p = (VideoWriter_t*)self;
  cvReleaseVideoWriter(&p->v);
  PyObject_Del(self);
}

static PyObject *VideoWriter_repr(PyObject *self)
{
  VideoWriter_t *p = (VideoWriter_t*)self;
  char str[1000];
  sprintf(str, "<VideoWriter %p>", p);
  return PyString_FromString(str);
}



static PyGetSetDef VideoWriter_getseters[] = {

  
  {NULL}  /* Sentinel */
};

static PyTypeObject VideoWriter_Type = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                                      /*size*/
  MODULESTR".VideoWriter",              /*name*/
  sizeof(VideoWriter_t),                /*basicsize*/
};

static void VideoWriter_specials(void)
{
  VideoWriter_Type.tp_dealloc = VideoWriter_dealloc;
  VideoWriter_Type.tp_repr = VideoWriter_repr;
  VideoWriter_Type.tp_getset = VideoWriter_getseters;
}

static PyObject *FROM_CvVideoWriterPTR(CvVideoWriter *r)
{
  VideoWriter_t *m = PyObject_NEW(VideoWriter_t, &VideoWriter_Type);
  m->v = r;
  return (PyObject*)m;
}

static int convert_to_CvVideoWriterPTR(PyObject *o, CvVideoWriter** dst, const char *name = "no_name")
{
  
  if (PyType_IsSubtype(o->ob_type, &VideoWriter_Type)) {
    *dst = ((VideoWriter_t*)o)->v;
    return 1;
  } else {
    (*dst) = (CvVideoWriter*)NULL;
    return failmsg("Expected CvVideoWriter for argument '%s'", name);
  }
}



/*
  CvStereoBMState is the OpenCV C struct
  StereoBMState_t is the Python object
*/

struct StereoBMState_t {
  PyObject_HEAD
  CvStereoBMState *v;
};

static void StereoBMState_dealloc(PyObject *self)
{
  StereoBMState_t *p = (StereoBMState_t*)self;
  cvReleaseStereoBMState(&p->v);
  PyObject_Del(self);
}

static PyObject *StereoBMState_repr(PyObject *self)
{
  StereoBMState_t *p = (StereoBMState_t*)self;
  char str[1000];
  sprintf(str, "<StereoBMState %p>", p);
  return PyString_FromString(str);
}


static PyObject *StereoBMState_get_textureThreshold(StereoBMState_t *p, void *closure)
{
  return PyInt_FromLong(p->v->textureThreshold);
}

static int StereoBMState_set_textureThreshold(StereoBMState_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the textureThreshold attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The textureThreshold attribute value must be a integer");
    return -1;
  }

  p->v->textureThreshold = PyInt_AsLong(value);
  return 0;
}


static PyObject *StereoBMState_get_preFilterSize(StereoBMState_t *p, void *closure)
{
  return PyInt_FromLong(p->v->preFilterSize);
}

static int StereoBMState_set_preFilterSize(StereoBMState_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the preFilterSize attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The preFilterSize attribute value must be a integer");
    return -1;
  }

  p->v->preFilterSize = PyInt_AsLong(value);
  return 0;
}


static PyObject *StereoBMState_get_speckleRange(StereoBMState_t *p, void *closure)
{
  return PyInt_FromLong(p->v->speckleRange);
}

static int StereoBMState_set_speckleRange(StereoBMState_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the speckleRange attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The speckleRange attribute value must be a integer");
    return -1;
  }

  p->v->speckleRange = PyInt_AsLong(value);
  return 0;
}


static PyObject *StereoBMState_get_uniquenessRatio(StereoBMState_t *p, void *closure)
{
  return PyInt_FromLong(p->v->uniquenessRatio);
}

static int StereoBMState_set_uniquenessRatio(StereoBMState_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the uniquenessRatio attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The uniquenessRatio attribute value must be a integer");
    return -1;
  }

  p->v->uniquenessRatio = PyInt_AsLong(value);
  return 0;
}


static PyObject *StereoBMState_get_preFilterCap(StereoBMState_t *p, void *closure)
{
  return PyInt_FromLong(p->v->preFilterCap);
}

static int StereoBMState_set_preFilterCap(StereoBMState_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the preFilterCap attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The preFilterCap attribute value must be a integer");
    return -1;
  }

  p->v->preFilterCap = PyInt_AsLong(value);
  return 0;
}


static PyObject *StereoBMState_get_numberOfDisparities(StereoBMState_t *p, void *closure)
{
  return PyInt_FromLong(p->v->numberOfDisparities);
}

static int StereoBMState_set_numberOfDisparities(StereoBMState_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the numberOfDisparities attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The numberOfDisparities attribute value must be a integer");
    return -1;
  }

  p->v->numberOfDisparities = PyInt_AsLong(value);
  return 0;
}


static PyObject *StereoBMState_get_minDisparity(StereoBMState_t *p, void *closure)
{
  return PyInt_FromLong(p->v->minDisparity);
}

static int StereoBMState_set_minDisparity(StereoBMState_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the minDisparity attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The minDisparity attribute value must be a integer");
    return -1;
  }

  p->v->minDisparity = PyInt_AsLong(value);
  return 0;
}


static PyObject *StereoBMState_get_preFilterType(StereoBMState_t *p, void *closure)
{
  return PyInt_FromLong(p->v->preFilterType);
}

static int StereoBMState_set_preFilterType(StereoBMState_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the preFilterType attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The preFilterType attribute value must be a integer");
    return -1;
  }

  p->v->preFilterType = PyInt_AsLong(value);
  return 0;
}


static PyObject *StereoBMState_get_SADWindowSize(StereoBMState_t *p, void *closure)
{
  return PyInt_FromLong(p->v->SADWindowSize);
}

static int StereoBMState_set_SADWindowSize(StereoBMState_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the SADWindowSize attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The SADWindowSize attribute value must be a integer");
    return -1;
  }

  p->v->SADWindowSize = PyInt_AsLong(value);
  return 0;
}


static PyObject *StereoBMState_get_speckleWindowSize(StereoBMState_t *p, void *closure)
{
  return PyInt_FromLong(p->v->speckleWindowSize);
}

static int StereoBMState_set_speckleWindowSize(StereoBMState_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the speckleWindowSize attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The speckleWindowSize attribute value must be a integer");
    return -1;
  }

  p->v->speckleWindowSize = PyInt_AsLong(value);
  return 0;
}



static PyGetSetDef StereoBMState_getseters[] = {

  
  {(char*)"textureThreshold", (getter)StereoBMState_get_textureThreshold, (setter)StereoBMState_set_textureThreshold, (char*)"textureThreshold", NULL},

  {(char*)"preFilterSize", (getter)StereoBMState_get_preFilterSize, (setter)StereoBMState_set_preFilterSize, (char*)"preFilterSize", NULL},

  {(char*)"speckleRange", (getter)StereoBMState_get_speckleRange, (setter)StereoBMState_set_speckleRange, (char*)"speckleRange", NULL},

  {(char*)"uniquenessRatio", (getter)StereoBMState_get_uniquenessRatio, (setter)StereoBMState_set_uniquenessRatio, (char*)"uniquenessRatio", NULL},

  {(char*)"preFilterCap", (getter)StereoBMState_get_preFilterCap, (setter)StereoBMState_set_preFilterCap, (char*)"preFilterCap", NULL},

  {(char*)"numberOfDisparities", (getter)StereoBMState_get_numberOfDisparities, (setter)StereoBMState_set_numberOfDisparities, (char*)"numberOfDisparities", NULL},

  {(char*)"minDisparity", (getter)StereoBMState_get_minDisparity, (setter)StereoBMState_set_minDisparity, (char*)"minDisparity", NULL},

  {(char*)"preFilterType", (getter)StereoBMState_get_preFilterType, (setter)StereoBMState_set_preFilterType, (char*)"preFilterType", NULL},

  {(char*)"SADWindowSize", (getter)StereoBMState_get_SADWindowSize, (setter)StereoBMState_set_SADWindowSize, (char*)"SADWindowSize", NULL},

  {(char*)"speckleWindowSize", (getter)StereoBMState_get_speckleWindowSize, (setter)StereoBMState_set_speckleWindowSize, (char*)"speckleWindowSize", NULL},

  {NULL}  /* Sentinel */
};

static PyTypeObject StereoBMState_Type = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                                      /*size*/
  MODULESTR".StereoBMState",              /*name*/
  sizeof(StereoBMState_t),                /*basicsize*/
};

static void StereoBMState_specials(void)
{
  StereoBMState_Type.tp_dealloc = StereoBMState_dealloc;
  StereoBMState_Type.tp_repr = StereoBMState_repr;
  StereoBMState_Type.tp_getset = StereoBMState_getseters;
}

static PyObject *FROM_CvStereoBMStatePTR(CvStereoBMState *r)
{
  StereoBMState_t *m = PyObject_NEW(StereoBMState_t, &StereoBMState_Type);
  m->v = r;
  return (PyObject*)m;
}

static int convert_to_CvStereoBMStatePTR(PyObject *o, CvStereoBMState** dst, const char *name = "no_name")
{
  
  if (PyType_IsSubtype(o->ob_type, &StereoBMState_Type)) {
    *dst = ((StereoBMState_t*)o)->v;
    return 1;
  } else {
    (*dst) = (CvStereoBMState*)NULL;
    return failmsg("Expected CvStereoBMState for argument '%s'", name);
  }
}



/*
  CvStereoGCState is the OpenCV C struct
  StereoGCState_t is the Python object
*/

struct StereoGCState_t {
  PyObject_HEAD
  CvStereoGCState *v;
};

static void StereoGCState_dealloc(PyObject *self)
{
  StereoGCState_t *p = (StereoGCState_t*)self;
  cvReleaseStereoGCState(&p->v);
  PyObject_Del(self);
}

static PyObject *StereoGCState_repr(PyObject *self)
{
  StereoGCState_t *p = (StereoGCState_t*)self;
  char str[1000];
  sprintf(str, "<StereoGCState %p>", p);
  return PyString_FromString(str);
}


static PyObject *StereoGCState_get_numberOfDisparities(StereoGCState_t *p, void *closure)
{
  return PyInt_FromLong(p->v->numberOfDisparities);
}

static int StereoGCState_set_numberOfDisparities(StereoGCState_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the numberOfDisparities attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The numberOfDisparities attribute value must be a integer");
    return -1;
  }

  p->v->numberOfDisparities = PyInt_AsLong(value);
  return 0;
}


static PyObject *StereoGCState_get_minDisparity(StereoGCState_t *p, void *closure)
{
  return PyInt_FromLong(p->v->minDisparity);
}

static int StereoGCState_set_minDisparity(StereoGCState_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the minDisparity attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The minDisparity attribute value must be a integer");
    return -1;
  }

  p->v->minDisparity = PyInt_AsLong(value);
  return 0;
}


static PyObject *StereoGCState_get_maxIters(StereoGCState_t *p, void *closure)
{
  return PyInt_FromLong(p->v->maxIters);
}

static int StereoGCState_set_maxIters(StereoGCState_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the maxIters attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The maxIters attribute value must be a integer");
    return -1;
  }

  p->v->maxIters = PyInt_AsLong(value);
  return 0;
}


static PyObject *StereoGCState_get_Ithreshold(StereoGCState_t *p, void *closure)
{
  return PyInt_FromLong(p->v->Ithreshold);
}

static int StereoGCState_set_Ithreshold(StereoGCState_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the Ithreshold attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The Ithreshold attribute value must be a integer");
    return -1;
  }

  p->v->Ithreshold = PyInt_AsLong(value);
  return 0;
}


static PyObject *StereoGCState_get_occlusionCost(StereoGCState_t *p, void *closure)
{
  return PyInt_FromLong(p->v->occlusionCost);
}

static int StereoGCState_set_occlusionCost(StereoGCState_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the occlusionCost attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The occlusionCost attribute value must be a integer");
    return -1;
  }

  p->v->occlusionCost = PyInt_AsLong(value);
  return 0;
}


static PyObject *StereoGCState_get_K(StereoGCState_t *p, void *closure)
{
  return PyFloat_FromDouble(p->v->K);
}

static int StereoGCState_set_K(StereoGCState_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the K attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The K attribute value must be a float");
    return -1;
  }

  p->v->K = PyFloat_AsDouble(value);
  return 0;
}


static PyObject *StereoGCState_get_interactionRadius(StereoGCState_t *p, void *closure)
{
  return PyInt_FromLong(p->v->interactionRadius);
}

static int StereoGCState_set_interactionRadius(StereoGCState_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the interactionRadius attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The interactionRadius attribute value must be a integer");
    return -1;
  }

  p->v->interactionRadius = PyInt_AsLong(value);
  return 0;
}


static PyObject *StereoGCState_get_lambda1(StereoGCState_t *p, void *closure)
{
  return PyFloat_FromDouble(p->v->lambda1);
}

static int StereoGCState_set_lambda1(StereoGCState_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the lambda1 attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The lambda1 attribute value must be a float");
    return -1;
  }

  p->v->lambda1 = PyFloat_AsDouble(value);
  return 0;
}


static PyObject *StereoGCState_get_lambda2(StereoGCState_t *p, void *closure)
{
  return PyFloat_FromDouble(p->v->lambda2);
}

static int StereoGCState_set_lambda2(StereoGCState_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the lambda2 attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The lambda2 attribute value must be a float");
    return -1;
  }

  p->v->lambda2 = PyFloat_AsDouble(value);
  return 0;
}


static PyObject *StereoGCState_get_lambda(StereoGCState_t *p, void *closure)
{
  return PyFloat_FromDouble(p->v->lambda);
}

static int StereoGCState_set_lambda(StereoGCState_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the lambda attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The lambda attribute value must be a float");
    return -1;
  }

  p->v->lambda = PyFloat_AsDouble(value);
  return 0;
}



static PyGetSetDef StereoGCState_getseters[] = {

  
  {(char*)"numberOfDisparities", (getter)StereoGCState_get_numberOfDisparities, (setter)StereoGCState_set_numberOfDisparities, (char*)"numberOfDisparities", NULL},

  {(char*)"minDisparity", (getter)StereoGCState_get_minDisparity, (setter)StereoGCState_set_minDisparity, (char*)"minDisparity", NULL},

  {(char*)"maxIters", (getter)StereoGCState_get_maxIters, (setter)StereoGCState_set_maxIters, (char*)"maxIters", NULL},

  {(char*)"Ithreshold", (getter)StereoGCState_get_Ithreshold, (setter)StereoGCState_set_Ithreshold, (char*)"Ithreshold", NULL},

  {(char*)"occlusionCost", (getter)StereoGCState_get_occlusionCost, (setter)StereoGCState_set_occlusionCost, (char*)"occlusionCost", NULL},

  {(char*)"K", (getter)StereoGCState_get_K, (setter)StereoGCState_set_K, (char*)"K", NULL},

  {(char*)"interactionRadius", (getter)StereoGCState_get_interactionRadius, (setter)StereoGCState_set_interactionRadius, (char*)"interactionRadius", NULL},

  {(char*)"lambda1", (getter)StereoGCState_get_lambda1, (setter)StereoGCState_set_lambda1, (char*)"lambda1", NULL},

  {(char*)"lambda2", (getter)StereoGCState_get_lambda2, (setter)StereoGCState_set_lambda2, (char*)"lambda2", NULL},

  {(char*)"lambda", (getter)StereoGCState_get_lambda, (setter)StereoGCState_set_lambda, (char*)"lambda", NULL},

  {NULL}  /* Sentinel */
};

static PyTypeObject StereoGCState_Type = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                                      /*size*/
  MODULESTR".StereoGCState",              /*name*/
  sizeof(StereoGCState_t),                /*basicsize*/
};

static void StereoGCState_specials(void)
{
  StereoGCState_Type.tp_dealloc = StereoGCState_dealloc;
  StereoGCState_Type.tp_repr = StereoGCState_repr;
  StereoGCState_Type.tp_getset = StereoGCState_getseters;
}

static PyObject *FROM_CvStereoGCStatePTR(CvStereoGCState *r)
{
  StereoGCState_t *m = PyObject_NEW(StereoGCState_t, &StereoGCState_Type);
  m->v = r;
  return (PyObject*)m;
}

static int convert_to_CvStereoGCStatePTR(PyObject *o, CvStereoGCState** dst, const char *name = "no_name")
{
  
  if (PyType_IsSubtype(o->ob_type, &StereoGCState_Type)) {
    *dst = ((StereoGCState_t*)o)->v;
    return 1;
  } else {
    (*dst) = (CvStereoGCState*)NULL;
    return failmsg("Expected CvStereoGCState for argument '%s'", name);
  }
}



/*
  CvKalman is the OpenCV C struct
  Kalman_t is the Python object
*/

struct Kalman_t {
  PyObject_HEAD
  CvKalman *v;
};

static void Kalman_dealloc(PyObject *self)
{
  Kalman_t *p = (Kalman_t*)self;
  cvReleaseKalman(&p->v);
  PyObject_Del(self);
}

static PyObject *Kalman_repr(PyObject *self)
{
  Kalman_t *p = (Kalman_t*)self;
  char str[1000];
  sprintf(str, "<Kalman %p>", p);
  return PyString_FromString(str);
}


static PyObject *Kalman_get_measurement_matrix(Kalman_t *p, void *closure)
{
  return FROM_ROCvMatPTR(p->v->measurement_matrix);
}

static int Kalman_set_measurement_matrix(Kalman_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the measurement_matrix attribute");
    return -1;
  }

  if (! is_cvmat(value)) {
    PyErr_SetString(PyExc_TypeError, "The measurement_matrix attribute value must be a list of CvMat");
    return -1;
  }

  p->v->measurement_matrix = PyCvMat_AsCvMat(value);
  return 0;
}


static PyObject *Kalman_get_error_cov_post(Kalman_t *p, void *closure)
{
  return FROM_ROCvMatPTR(p->v->error_cov_post);
}

static int Kalman_set_error_cov_post(Kalman_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the error_cov_post attribute");
    return -1;
  }

  if (! is_cvmat(value)) {
    PyErr_SetString(PyExc_TypeError, "The error_cov_post attribute value must be a list of CvMat");
    return -1;
  }

  p->v->error_cov_post = PyCvMat_AsCvMat(value);
  return 0;
}


static PyObject *Kalman_get_state_pre(Kalman_t *p, void *closure)
{
  return FROM_ROCvMatPTR(p->v->state_pre);
}

static int Kalman_set_state_pre(Kalman_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the state_pre attribute");
    return -1;
  }

  if (! is_cvmat(value)) {
    PyErr_SetString(PyExc_TypeError, "The state_pre attribute value must be a list of CvMat");
    return -1;
  }

  p->v->state_pre = PyCvMat_AsCvMat(value);
  return 0;
}


static PyObject *Kalman_get_gain(Kalman_t *p, void *closure)
{
  return FROM_ROCvMatPTR(p->v->gain);
}

static int Kalman_set_gain(Kalman_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the gain attribute");
    return -1;
  }

  if (! is_cvmat(value)) {
    PyErr_SetString(PyExc_TypeError, "The gain attribute value must be a list of CvMat");
    return -1;
  }

  p->v->gain = PyCvMat_AsCvMat(value);
  return 0;
}


static PyObject *Kalman_get_CP(Kalman_t *p, void *closure)
{
  return PyInt_FromLong(p->v->CP);
}

static int Kalman_set_CP(Kalman_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the CP attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The CP attribute value must be a integer");
    return -1;
  }

  p->v->CP = PyInt_AsLong(value);
  return 0;
}


static PyObject *Kalman_get_DP(Kalman_t *p, void *closure)
{
  return PyInt_FromLong(p->v->DP);
}

static int Kalman_set_DP(Kalman_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the DP attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The DP attribute value must be a integer");
    return -1;
  }

  p->v->DP = PyInt_AsLong(value);
  return 0;
}


static PyObject *Kalman_get_measurement_noise_cov(Kalman_t *p, void *closure)
{
  return FROM_ROCvMatPTR(p->v->measurement_noise_cov);
}

static int Kalman_set_measurement_noise_cov(Kalman_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the measurement_noise_cov attribute");
    return -1;
  }

  if (! is_cvmat(value)) {
    PyErr_SetString(PyExc_TypeError, "The measurement_noise_cov attribute value must be a list of CvMat");
    return -1;
  }

  p->v->measurement_noise_cov = PyCvMat_AsCvMat(value);
  return 0;
}


static PyObject *Kalman_get_error_cov_pre(Kalman_t *p, void *closure)
{
  return FROM_ROCvMatPTR(p->v->error_cov_pre);
}

static int Kalman_set_error_cov_pre(Kalman_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the error_cov_pre attribute");
    return -1;
  }

  if (! is_cvmat(value)) {
    PyErr_SetString(PyExc_TypeError, "The error_cov_pre attribute value must be a list of CvMat");
    return -1;
  }

  p->v->error_cov_pre = PyCvMat_AsCvMat(value);
  return 0;
}


static PyObject *Kalman_get_control_matrix(Kalman_t *p, void *closure)
{
  return FROM_ROCvMatPTR(p->v->control_matrix);
}

static int Kalman_set_control_matrix(Kalman_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the control_matrix attribute");
    return -1;
  }

  if (! is_cvmat(value)) {
    PyErr_SetString(PyExc_TypeError, "The control_matrix attribute value must be a list of CvMat");
    return -1;
  }

  p->v->control_matrix = PyCvMat_AsCvMat(value);
  return 0;
}


static PyObject *Kalman_get_process_noise_cov(Kalman_t *p, void *closure)
{
  return FROM_ROCvMatPTR(p->v->process_noise_cov);
}

static int Kalman_set_process_noise_cov(Kalman_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the process_noise_cov attribute");
    return -1;
  }

  if (! is_cvmat(value)) {
    PyErr_SetString(PyExc_TypeError, "The process_noise_cov attribute value must be a list of CvMat");
    return -1;
  }

  p->v->process_noise_cov = PyCvMat_AsCvMat(value);
  return 0;
}


static PyObject *Kalman_get_transition_matrix(Kalman_t *p, void *closure)
{
  return FROM_ROCvMatPTR(p->v->transition_matrix);
}

static int Kalman_set_transition_matrix(Kalman_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the transition_matrix attribute");
    return -1;
  }

  if (! is_cvmat(value)) {
    PyErr_SetString(PyExc_TypeError, "The transition_matrix attribute value must be a list of CvMat");
    return -1;
  }

  p->v->transition_matrix = PyCvMat_AsCvMat(value);
  return 0;
}


static PyObject *Kalman_get_MP(Kalman_t *p, void *closure)
{
  return PyInt_FromLong(p->v->MP);
}

static int Kalman_set_MP(Kalman_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the MP attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The MP attribute value must be a integer");
    return -1;
  }

  p->v->MP = PyInt_AsLong(value);
  return 0;
}


static PyObject *Kalman_get_state_post(Kalman_t *p, void *closure)
{
  return FROM_ROCvMatPTR(p->v->state_post);
}

static int Kalman_set_state_post(Kalman_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the state_post attribute");
    return -1;
  }

  if (! is_cvmat(value)) {
    PyErr_SetString(PyExc_TypeError, "The state_post attribute value must be a list of CvMat");
    return -1;
  }

  p->v->state_post = PyCvMat_AsCvMat(value);
  return 0;
}



static PyGetSetDef Kalman_getseters[] = {

  
  {(char*)"measurement_matrix", (getter)Kalman_get_measurement_matrix, (setter)Kalman_set_measurement_matrix, (char*)"measurement_matrix", NULL},

  {(char*)"error_cov_post", (getter)Kalman_get_error_cov_post, (setter)Kalman_set_error_cov_post, (char*)"error_cov_post", NULL},

  {(char*)"state_pre", (getter)Kalman_get_state_pre, (setter)Kalman_set_state_pre, (char*)"state_pre", NULL},

  {(char*)"gain", (getter)Kalman_get_gain, (setter)Kalman_set_gain, (char*)"gain", NULL},

  {(char*)"CP", (getter)Kalman_get_CP, (setter)Kalman_set_CP, (char*)"CP", NULL},

  {(char*)"DP", (getter)Kalman_get_DP, (setter)Kalman_set_DP, (char*)"DP", NULL},

  {(char*)"measurement_noise_cov", (getter)Kalman_get_measurement_noise_cov, (setter)Kalman_set_measurement_noise_cov, (char*)"measurement_noise_cov", NULL},

  {(char*)"error_cov_pre", (getter)Kalman_get_error_cov_pre, (setter)Kalman_set_error_cov_pre, (char*)"error_cov_pre", NULL},

  {(char*)"control_matrix", (getter)Kalman_get_control_matrix, (setter)Kalman_set_control_matrix, (char*)"control_matrix", NULL},

  {(char*)"process_noise_cov", (getter)Kalman_get_process_noise_cov, (setter)Kalman_set_process_noise_cov, (char*)"process_noise_cov", NULL},

  {(char*)"transition_matrix", (getter)Kalman_get_transition_matrix, (setter)Kalman_set_transition_matrix, (char*)"transition_matrix", NULL},

  {(char*)"MP", (getter)Kalman_get_MP, (setter)Kalman_set_MP, (char*)"MP", NULL},

  {(char*)"state_post", (getter)Kalman_get_state_post, (setter)Kalman_set_state_post, (char*)"state_post", NULL},

  {NULL}  /* Sentinel */
};

static PyTypeObject Kalman_Type = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                                      /*size*/
  MODULESTR".Kalman",              /*name*/
  sizeof(Kalman_t),                /*basicsize*/
};

static void Kalman_specials(void)
{
  Kalman_Type.tp_dealloc = Kalman_dealloc;
  Kalman_Type.tp_repr = Kalman_repr;
  Kalman_Type.tp_getset = Kalman_getseters;
}

static PyObject *FROM_CvKalmanPTR(CvKalman *r)
{
  Kalman_t *m = PyObject_NEW(Kalman_t, &Kalman_Type);
  m->v = r;
  return (PyObject*)m;
}

static int convert_to_CvKalmanPTR(PyObject *o, CvKalman** dst, const char *name = "no_name")
{
  
  if (PyType_IsSubtype(o->ob_type, &Kalman_Type)) {
    *dst = ((Kalman_t*)o)->v;
    return 1;
  } else {
    (*dst) = (CvKalman*)NULL;
    return failmsg("Expected CvKalman for argument '%s'", name);
  }
}



/*
  CvMoments is the OpenCV C struct
  Moments_t is the Python object
*/

struct Moments_t {
  PyObject_HEAD
  CvMoments v;
};

static PyObject *Moments_repr(PyObject *self)
{
  Moments_t *p = (Moments_t*)self;
  char str[1000];
  sprintf(str, "<Moments %p>", p);
  return PyString_FromString(str);
}


static PyObject *Moments_get_mu11(Moments_t *p, void *closure)
{
  return PyFloat_FromDouble(p->v.mu11);
}

static int Moments_set_mu11(Moments_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the mu11 attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The mu11 attribute value must be a float");
    return -1;
  }

  p->v.mu11 = PyFloat_AsDouble(value);
  return 0;
}


static PyObject *Moments_get_mu12(Moments_t *p, void *closure)
{
  return PyFloat_FromDouble(p->v.mu12);
}

static int Moments_set_mu12(Moments_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the mu12 attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The mu12 attribute value must be a float");
    return -1;
  }

  p->v.mu12 = PyFloat_AsDouble(value);
  return 0;
}


static PyObject *Moments_get_mu02(Moments_t *p, void *closure)
{
  return PyFloat_FromDouble(p->v.mu02);
}

static int Moments_set_mu02(Moments_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the mu02 attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The mu02 attribute value must be a float");
    return -1;
  }

  p->v.mu02 = PyFloat_AsDouble(value);
  return 0;
}


static PyObject *Moments_get_mu03(Moments_t *p, void *closure)
{
  return PyFloat_FromDouble(p->v.mu03);
}

static int Moments_set_mu03(Moments_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the mu03 attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The mu03 attribute value must be a float");
    return -1;
  }

  p->v.mu03 = PyFloat_AsDouble(value);
  return 0;
}


static PyObject *Moments_get_inv_sqrt_m00(Moments_t *p, void *closure)
{
  return PyFloat_FromDouble(p->v.inv_sqrt_m00);
}

static int Moments_set_inv_sqrt_m00(Moments_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the inv_sqrt_m00 attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The inv_sqrt_m00 attribute value must be a float");
    return -1;
  }

  p->v.inv_sqrt_m00 = PyFloat_AsDouble(value);
  return 0;
}


static PyObject *Moments_get_m20(Moments_t *p, void *closure)
{
  return PyFloat_FromDouble(p->v.m20);
}

static int Moments_set_m20(Moments_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the m20 attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The m20 attribute value must be a float");
    return -1;
  }

  p->v.m20 = PyFloat_AsDouble(value);
  return 0;
}


static PyObject *Moments_get_m21(Moments_t *p, void *closure)
{
  return PyFloat_FromDouble(p->v.m21);
}

static int Moments_set_m21(Moments_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the m21 attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The m21 attribute value must be a float");
    return -1;
  }

  p->v.m21 = PyFloat_AsDouble(value);
  return 0;
}


static PyObject *Moments_get_m30(Moments_t *p, void *closure)
{
  return PyFloat_FromDouble(p->v.m30);
}

static int Moments_set_m30(Moments_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the m30 attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The m30 attribute value must be a float");
    return -1;
  }

  p->v.m30 = PyFloat_AsDouble(value);
  return 0;
}


static PyObject *Moments_get_m11(Moments_t *p, void *closure)
{
  return PyFloat_FromDouble(p->v.m11);
}

static int Moments_set_m11(Moments_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the m11 attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The m11 attribute value must be a float");
    return -1;
  }

  p->v.m11 = PyFloat_AsDouble(value);
  return 0;
}


static PyObject *Moments_get_m10(Moments_t *p, void *closure)
{
  return PyFloat_FromDouble(p->v.m10);
}

static int Moments_set_m10(Moments_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the m10 attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The m10 attribute value must be a float");
    return -1;
  }

  p->v.m10 = PyFloat_AsDouble(value);
  return 0;
}


static PyObject *Moments_get_m12(Moments_t *p, void *closure)
{
  return PyFloat_FromDouble(p->v.m12);
}

static int Moments_set_m12(Moments_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the m12 attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The m12 attribute value must be a float");
    return -1;
  }

  p->v.m12 = PyFloat_AsDouble(value);
  return 0;
}


static PyObject *Moments_get_m02(Moments_t *p, void *closure)
{
  return PyFloat_FromDouble(p->v.m02);
}

static int Moments_set_m02(Moments_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the m02 attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The m02 attribute value must be a float");
    return -1;
  }

  p->v.m02 = PyFloat_AsDouble(value);
  return 0;
}


static PyObject *Moments_get_m03(Moments_t *p, void *closure)
{
  return PyFloat_FromDouble(p->v.m03);
}

static int Moments_set_m03(Moments_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the m03 attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The m03 attribute value must be a float");
    return -1;
  }

  p->v.m03 = PyFloat_AsDouble(value);
  return 0;
}


static PyObject *Moments_get_m00(Moments_t *p, void *closure)
{
  return PyFloat_FromDouble(p->v.m00);
}

static int Moments_set_m00(Moments_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the m00 attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The m00 attribute value must be a float");
    return -1;
  }

  p->v.m00 = PyFloat_AsDouble(value);
  return 0;
}


static PyObject *Moments_get_m01(Moments_t *p, void *closure)
{
  return PyFloat_FromDouble(p->v.m01);
}

static int Moments_set_m01(Moments_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the m01 attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The m01 attribute value must be a float");
    return -1;
  }

  p->v.m01 = PyFloat_AsDouble(value);
  return 0;
}


static PyObject *Moments_get_mu20(Moments_t *p, void *closure)
{
  return PyFloat_FromDouble(p->v.mu20);
}

static int Moments_set_mu20(Moments_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the mu20 attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The mu20 attribute value must be a float");
    return -1;
  }

  p->v.mu20 = PyFloat_AsDouble(value);
  return 0;
}


static PyObject *Moments_get_mu21(Moments_t *p, void *closure)
{
  return PyFloat_FromDouble(p->v.mu21);
}

static int Moments_set_mu21(Moments_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the mu21 attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The mu21 attribute value must be a float");
    return -1;
  }

  p->v.mu21 = PyFloat_AsDouble(value);
  return 0;
}


static PyObject *Moments_get_mu30(Moments_t *p, void *closure)
{
  return PyFloat_FromDouble(p->v.mu30);
}

static int Moments_set_mu30(Moments_t *p, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the mu30 attribute");
    return -1;
  }

  if (! PyNumber_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The mu30 attribute value must be a float");
    return -1;
  }

  p->v.mu30 = PyFloat_AsDouble(value);
  return 0;
}



static PyGetSetDef Moments_getseters[] = {

  
  {(char*)"mu11", (getter)Moments_get_mu11, (setter)Moments_set_mu11, (char*)"mu11", NULL},

  {(char*)"mu12", (getter)Moments_get_mu12, (setter)Moments_set_mu12, (char*)"mu12", NULL},

  {(char*)"mu02", (getter)Moments_get_mu02, (setter)Moments_set_mu02, (char*)"mu02", NULL},

  {(char*)"mu03", (getter)Moments_get_mu03, (setter)Moments_set_mu03, (char*)"mu03", NULL},

  {(char*)"inv_sqrt_m00", (getter)Moments_get_inv_sqrt_m00, (setter)Moments_set_inv_sqrt_m00, (char*)"inv_sqrt_m00", NULL},

  {(char*)"m20", (getter)Moments_get_m20, (setter)Moments_set_m20, (char*)"m20", NULL},

  {(char*)"m21", (getter)Moments_get_m21, (setter)Moments_set_m21, (char*)"m21", NULL},

  {(char*)"m30", (getter)Moments_get_m30, (setter)Moments_set_m30, (char*)"m30", NULL},

  {(char*)"m11", (getter)Moments_get_m11, (setter)Moments_set_m11, (char*)"m11", NULL},

  {(char*)"m10", (getter)Moments_get_m10, (setter)Moments_set_m10, (char*)"m10", NULL},

  {(char*)"m12", (getter)Moments_get_m12, (setter)Moments_set_m12, (char*)"m12", NULL},

  {(char*)"m02", (getter)Moments_get_m02, (setter)Moments_set_m02, (char*)"m02", NULL},

  {(char*)"m03", (getter)Moments_get_m03, (setter)Moments_set_m03, (char*)"m03", NULL},

  {(char*)"m00", (getter)Moments_get_m00, (setter)Moments_set_m00, (char*)"m00", NULL},

  {(char*)"m01", (getter)Moments_get_m01, (setter)Moments_set_m01, (char*)"m01", NULL},

  {(char*)"mu20", (getter)Moments_get_mu20, (setter)Moments_set_mu20, (char*)"mu20", NULL},

  {(char*)"mu21", (getter)Moments_get_mu21, (setter)Moments_set_mu21, (char*)"mu21", NULL},

  {(char*)"mu30", (getter)Moments_get_mu30, (setter)Moments_set_mu30, (char*)"mu30", NULL},

  {NULL}  /* Sentinel */
};

static PyTypeObject Moments_Type = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                                      /*size*/
  MODULESTR".Moments",              /*name*/
  sizeof(Moments_t),                /*basicsize*/
};

static void Moments_specials(void)
{
  Moments_Type.tp_repr = Moments_repr;
  Moments_Type.tp_getset = Moments_getseters;
}

static PyObject *FROM_CvMoments(CvMoments r)
{
  Moments_t *m = PyObject_NEW(Moments_t, &Moments_Type);
  m->v = r;
  return (PyObject*)m;
}

static int convert_to_CvMomentsPTR(PyObject *o, CvMoments** dst, const char *name = "no_name")
{
  
  if (PyType_IsSubtype(o->ob_type, &Moments_Type)) {
    *dst = &(((Moments_t*)o)->v);
    return 1;
  } else {
    (*dst) = (CvMoments*)NULL;
    return failmsg("Expected CvMoments for argument '%s'", name);
  }
}


