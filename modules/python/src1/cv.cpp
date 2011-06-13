#include <Python.h>

#include <assert.h>

#include "opencv/cxcore.h"
#include "opencv/cv.h"
#include "opencv/cvaux.h"
#include "opencv/cvwimage.h"
#include "opencv/highgui.h"

#define MODULESTR "cv"

static PyObject *opencv_error;

struct memtrack_t {
  PyObject_HEAD
  int owner;
  void *ptr;
  int freeptr;
  Py_ssize_t size;
  PyObject *backing;
  CvArr *backingmat;
};

struct iplimage_t {
  PyObject_HEAD
  IplImage *a;
  PyObject *data;
  size_t offset;
};

struct cvmat_t {
  PyObject_HEAD
  CvMat *a;
  PyObject *data;
  size_t offset;
};

struct cvmatnd_t {
  PyObject_HEAD
  CvMatND *a;
  PyObject *data;
  size_t offset;
};

struct cvhistogram_t {
  PyObject_HEAD
  CvHistogram h;
  PyObject *bins;
};

struct cvmemstorage_t {
  PyObject_HEAD
  CvMemStorage *a;
};

struct cvseq_t {
  PyObject_HEAD
  CvSeq *a;
  PyObject *container;  // Containing cvmemstorage_t
};

struct cvset_t {
  PyObject_HEAD
  CvSet *a;
  PyObject *container;  // Containing cvmemstorage_t
  int i;
};

struct cvsubdiv2d_t {
  PyObject_HEAD
  CvSubdiv2D *a;
  PyObject *container;  // Containing cvmemstorage_t
};

struct cvsubdiv2dpoint_t {
  PyObject_HEAD
  CvSubdiv2DPoint *a;
  PyObject *container;  // Containing cvmemstorage_t
};

struct cvsubdiv2dedge_t {
  PyObject_HEAD
  CvSubdiv2DEdge a;
  PyObject *container;  // Containing cvmemstorage_t
};

struct cvlineiterator_t {
  PyObject_HEAD
  CvLineIterator iter;
  int count;
  int type;
};

typedef IplImage ROIplImage;
typedef const CvMat ROCvMat;
typedef PyObject PyCallableObject;

struct cvfont_t {
  PyObject_HEAD
  CvFont a;
};

struct cvcontourtree_t {
  PyObject_HEAD
  CvContourTree *a;
};

struct cvrng_t {
  PyObject_HEAD
  CvRNG a;
};

static int is_iplimage(PyObject *o);
static int is_cvmat(PyObject *o);
static int is_cvmatnd(PyObject *o);
static int convert_to_CvArr(PyObject *o, CvArr **dst, const char *name = "no_name");
static int convert_to_IplImage(PyObject *o, IplImage **dst, const char *name = "no_name");
static int convert_to_CvMat(PyObject *o, CvMat **dst, const char *name = "no_name");
static int convert_to_CvMatND(PyObject *o, CvMatND **dst, const char *name = "no_name");
static PyObject *what_data(PyObject *o);
static PyObject *FROM_CvMat(CvMat *r);
static PyObject *FROM_ROCvMatPTR(ROCvMat *r);
static PyObject *shareDataND(PyObject *donor, CvMatND *pdonor, CvMatND *precipient);

#define FROM_double(r)  PyFloat_FromDouble(r)
#define FROM_float(r)  PyFloat_FromDouble(r)
#define FROM_int(r)  PyInt_FromLong(r)
#define FROM_int64(r)  PyLong_FromLongLong(r)
#define FROM_unsigned(r)  PyLong_FromUnsignedLong(r)
#define FROM_CvBox2D(r) Py_BuildValue("(ff)(ff)f", r.center.x, r.center.y, r.size.width, r.size.height, r.angle)
#define FROM_CvScalar(r)  Py_BuildValue("(ffff)", r.val[0], r.val[1], r.val[2], r.val[3])
#define FROM_CvPoint(r)  Py_BuildValue("(ii)", r.x, r.y)
#define FROM_CvPoint2D32f(r) Py_BuildValue("(ff)", r.x, r.y)
#define FROM_CvPoint3D64f(r) Py_BuildValue("(fff)", r.x, r.y, r.z)
#define FROM_CvSize(r) Py_BuildValue("(ii)", r.width, r.height)
#define FROM_CvRect(r) Py_BuildValue("(iiii)", r.x, r.y, r.width, r.height)
#define FROM_CvSeqPTR(r) _FROM_CvSeqPTR(r, pyobj_storage)
#define FROM_CvSubdiv2DPTR(r) _FROM_CvSubdiv2DPTR(r, pyobj_storage)
#define FROM_CvPoint2D64f(r) Py_BuildValue("(ff)", r.x, r.y)
#define FROM_CvConnectedComp(r) Py_BuildValue("(fNN)", (r).area, FROM_CvScalar((r).value), FROM_CvRect((r).rect))

#if PYTHON_USE_NUMPY
static PyObject *fromarray(PyObject *o, int allowND);
#endif

static void translate_error_to_exception(void)
{
  PyErr_SetString(opencv_error, cvErrorStr(cvGetErrStatus()));
  cvSetErrStatus(0);
}

#define ERRCHK do { if (cvGetErrStatus() != 0) { translate_error_to_exception(); return NULL; } } while (0)
#define ERRWRAPN(F, N) \
    do { \
        try \
        { \
            F; \
        } \
        catch (const cv::Exception &e) \
        { \
           PyErr_SetString(opencv_error, e.err.c_str()); \
           return N; \
        } \
        ERRCHK; \
    } while(0)
#define ERRWRAP(F) ERRWRAPN(F, NULL) // for most functions, exception -> NULL return

/************************************************************************/

static int failmsg(const char *fmt, ...)
{
  char str[1000];

  va_list ap;
  va_start(ap, fmt);
  vsnprintf(str, sizeof(str), fmt, ap);
  va_end(ap);

  PyErr_SetString(PyExc_TypeError, str);
  return 0;
}

/************************************************************************/

/* These get/setters are polymorphic, used in both iplimage and cvmat */

static PyObject *PyObject_FromCvScalar(CvScalar s, int type)
{
  int i, spe = CV_MAT_CN(type);
  PyObject *r;
  if (spe > 1) {
    r = PyTuple_New(spe);
    for (i = 0; i < spe; i++)
      PyTuple_SET_ITEM(r, i, PyFloat_FromDouble(s.val[i]));
  } else {
    r = PyFloat_FromDouble(s.val[0]);
  }
  return r;
}

static PyObject *cvarr_GetItem(PyObject *o, PyObject *key);
static int cvarr_SetItem(PyObject *o, PyObject *key, PyObject *v);

// o is a Python string or buffer object.  Return its size.

static Py_ssize_t what_size(PyObject *o)
{
  void *buffer;
  Py_ssize_t buffer_len;

  if (PyString_Check(o)) {
    return PyString_Size(o);
  } else if (PyObject_AsWriteBuffer(o, &buffer, &buffer_len) == 0) {
    return buffer_len;
  } else {
    assert(0);  // argument must be string or buffer.
    return 0;
  }
}


/************************************************************************/

CvMat *PyCvMat_AsCvMat(PyObject *o)
{
  assert(0); // not yet implemented: reference counting for CvMat in Kalman is unclear...
  return NULL;
}

#define cvReleaseIplConvKernel(x) cvReleaseStructuringElement(x)
#include "generated3.i"

/* iplimage */

static void iplimage_dealloc(PyObject *self)
{
  iplimage_t *pc = (iplimage_t*)self;
  cvReleaseImageHeader((IplImage**)&pc->a);
  Py_DECREF(pc->data);
  PyObject_Del(self);
}

static PyObject *iplimage_repr(PyObject *self)
{
  iplimage_t *cva = (iplimage_t*)self;
  IplImage* ipl = (IplImage*)(cva->a);
  char str[1000];
  sprintf(str, "<iplimage(");
  char *d = str + strlen(str);
  sprintf(d, "nChannels=%d ", ipl->nChannels);
  d += strlen(d);
  sprintf(d, "width=%d ", ipl->width);
  d += strlen(d);
  sprintf(d, "height=%d ", ipl->height);
  d += strlen(d);
  sprintf(d, "widthStep=%d ", ipl->widthStep);
  d += strlen(d);
  sprintf(d, ")>");
  return PyString_FromString(str);
}

static PyObject *iplimage_tostring(PyObject *self, PyObject *args)
{
  iplimage_t *pc = (iplimage_t*)self;
  IplImage *i;
  if (!convert_to_IplImage(self, &i, "self"))
    return NULL;
  if (i == NULL)
    return NULL;
  int bps;
  switch (i->depth) {
  case IPL_DEPTH_8U:
  case IPL_DEPTH_8S:
    bps = 1;
    break;
  case IPL_DEPTH_16U:
  case IPL_DEPTH_16S:
    bps = 2;
    break;
  case IPL_DEPTH_32S:
  case IPL_DEPTH_32F:
    bps = 4;
    break;
  case IPL_DEPTH_64F:
    bps = 8;
    break;
  default:
    return (PyObject*)failmsg("Unrecognised depth %d", i->depth);
  }
  int bpl = i->width * i->nChannels * bps;
  if (PyString_Check(pc->data) && bpl == i->widthStep && pc->offset == 0 && ((bpl * i->height) == what_size(pc->data))) {
    Py_INCREF(pc->data);
    return pc->data;
  } else {
    int l = bpl * i->height;
    char *s = new char[l];
    int y;
    for (y = 0; y < i->height; y++) {
      memcpy(s + y * bpl, i->imageData + y * i->widthStep, bpl);
    }
    PyObject *r = PyString_FromStringAndSize(s, l);
    delete s;
    return r;
  }
}

static struct PyMethodDef iplimage_methods[] =
{
  {"tostring", iplimage_tostring, METH_VARARGS},
  {NULL,          NULL}
};

static PyObject *iplimage_getnChannels(iplimage_t *cva)
{
  return PyInt_FromLong(((IplImage*)(cva->a))->nChannels);
}
static PyObject *iplimage_getwidth(iplimage_t *cva)
{
  return PyInt_FromLong(((IplImage*)(cva->a))->width);
}
static PyObject *iplimage_getheight(iplimage_t *cva)
{
  return PyInt_FromLong(((IplImage*)(cva->a))->height);
}
static PyObject *iplimage_getdepth(iplimage_t *cva)
{
  return PyLong_FromUnsignedLong((unsigned)((IplImage*)(cva->a))->depth);
}
static PyObject *iplimage_getorigin(iplimage_t *cva)
{
  return PyInt_FromLong(((IplImage*)(cva->a))->origin);
}
static void iplimage_setorigin(iplimage_t *cva, PyObject *v)
{
  ((IplImage*)(cva->a))->origin = PyInt_AsLong(v);
}

static PyGetSetDef iplimage_getseters[] = {
  {(char*)"nChannels", (getter)iplimage_getnChannels, (setter)NULL, (char*)"nChannels", NULL},
  {(char*)"channels", (getter)iplimage_getnChannels, (setter)NULL, (char*)"nChannels", NULL},
  {(char*)"width", (getter)iplimage_getwidth, (setter)NULL, (char*)"width", NULL},
  {(char*)"height", (getter)iplimage_getheight, (setter)NULL, (char*)"height", NULL},
  {(char*)"depth", (getter)iplimage_getdepth, (setter)NULL, (char*)"depth", NULL},
  {(char*)"origin", (getter)iplimage_getorigin, (setter)iplimage_setorigin, (char*)"origin", NULL},
  {NULL}  /* Sentinel */
};

static PyMappingMethods iplimage_as_map = {
  NULL,
  &cvarr_GetItem,
  &cvarr_SetItem,
};

static PyTypeObject iplimage_Type = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                                      /*size*/
  MODULESTR".iplimage",                          /*name*/
  sizeof(iplimage_t),                        /*basicsize*/
};

static void iplimage_specials(void)
{
  iplimage_Type.tp_dealloc = iplimage_dealloc;
  iplimage_Type.tp_as_mapping = &iplimage_as_map;
  iplimage_Type.tp_repr = iplimage_repr;
  iplimage_Type.tp_methods = iplimage_methods;
  iplimage_Type.tp_getset = iplimage_getseters;
}

static int is_iplimage(PyObject *o)
{
  return PyType_IsSubtype(o->ob_type, &iplimage_Type);
}

/************************************************************************/

/* cvmat */

static void cvmat_dealloc(PyObject *self)
{
  cvmat_t *pc = (cvmat_t*)self;
  if (pc->data) {
    Py_DECREF(pc->data);
  }
  cvFree(&pc->a);
  PyObject_Del(self);
}

static PyObject *cvmat_repr(PyObject *self)
{
  CvMat *m = ((cvmat_t*)self)->a;
  char str[1000];
  sprintf(str, "<cvmat(");
  char *d = str + strlen(str);
  sprintf(d, "type=%08x ", m->type);
  d += strlen(d);
  switch (CV_MAT_DEPTH(m->type)) {
  case CV_8U: strcpy(d, "8U"); break;
  case CV_8S: strcpy(d, "8S"); break;
  case CV_16U: strcpy(d, "16U"); break;
  case CV_16S: strcpy(d, "16S"); break;
  case CV_32S: strcpy(d, "32S"); break;
  case CV_32F: strcpy(d, "32F"); break;
  case CV_64F: strcpy(d, "64F"); break;
  }
  d += strlen(d);
  sprintf(d, "C%d ", CV_MAT_CN(m->type));
  d += strlen(d);
  sprintf(d, "rows=%d ", m->rows);
  d += strlen(d);
  sprintf(d, "cols=%d ", m->cols);
  d += strlen(d);
  sprintf(d, "step=%d ", m->step);
  d += strlen(d);
  sprintf(d, ")>");
  return PyString_FromString(str);
}

static PyObject *cvmat_tostring(PyObject *self, PyObject *args)
{
  CvMat *m;
  if (!convert_to_CvMat(self, &m, "self"))
    return NULL;

  int bps;                     // bytes per sample

  switch (CV_MAT_DEPTH(m->type)) {
  case CV_8U:
  case CV_8S:
    bps = CV_MAT_CN(m->type) * 1;
    break;
  case CV_16U:
  case CV_16S:
    bps = CV_MAT_CN(m->type) * 2;
    break;
  case CV_32S:
  case CV_32F:
    bps = CV_MAT_CN(m->type) * 4;
    break;
  case CV_64F:
    bps = CV_MAT_CN(m->type) * 8;
    break;
  default:
    return (PyObject*)failmsg("Unrecognised depth %d", CV_MAT_DEPTH(m->type));
  }

  int bpl = m->cols * bps; // bytes per line
  cvmat_t *pc = (cvmat_t*)self;
  if (PyString_Check(pc->data) && bpl == m->step && pc->offset == 0 && ((bpl * m->rows) == what_size(pc->data))) {
    Py_INCREF(pc->data);
    return pc->data;
  } else {
    int l = bpl * m->rows;
    char *s = new char[l];
    int y;
    for (y = 0; y < m->rows; y++) {
      memcpy(s + y * bpl, m->data.ptr + y * m->step, bpl);
    }
    PyObject *r = PyString_FromStringAndSize(s, l);
    delete s;
    return r;
  }
}

static struct PyMethodDef cvmat_methods[] =
{
  {"tostring", cvmat_tostring, METH_VARARGS},
  {NULL,          NULL}
};

static PyObject *cvmat_gettype(cvmat_t *cva)
{
  return PyInt_FromLong(cvGetElemType(cva->a));
}

static PyObject *cvmat_getstep(cvmat_t *cva)
{
  return PyInt_FromLong(cva->a->step);
}

static PyObject *cvmat_getrows(cvmat_t *cva)
{
  return PyInt_FromLong(cva->a->rows);
}

static PyObject *cvmat_getcols(cvmat_t *cva)
{
  return PyInt_FromLong(cva->a->cols);
}

static PyObject *cvmat_getchannels(cvmat_t *cva)
{
  return PyInt_FromLong(CV_MAT_CN(cva->a->type));
}

#if PYTHON_USE_NUMPY
#include "numpy/ndarrayobject.h"

// A PyArrayInterface, with an associated python object that should be DECREF'ed on release
struct arrayTrack {
  PyArrayInterface s;
  PyObject *o;
};

static void arrayTrackDtor(void *p)
{
  struct arrayTrack *at = (struct arrayTrack *)p;
  delete at->s.shape;
  delete at->s.strides;
  if (at->s.descr)
    Py_DECREF(at->s.descr);
  Py_DECREF(at->o);
}

// Fill in fields of PyArrayInterface s using mtype.  This code is common
// to cvmat and cvmatnd

static void arrayinterface_common(PyArrayInterface *s, int mtype)
{
  s->two = 2;

  switch (CV_MAT_DEPTH(mtype)) {
  case CV_8U:
    s->typekind = 'u';
    s->itemsize = 1;
    break;
  case CV_8S:
    s->typekind = 'i';
    s->itemsize = 1;
    break;
  case CV_16U:
    s->typekind = 'u';
    s->itemsize = 2;
    break;
  case CV_16S:
    s->typekind = 'i';
    s->itemsize = 2;
    break;
  case CV_32S:
    s->typekind = 'i';
    s->itemsize = 4;
    break;
  case CV_32F:
    s->typekind = 'f';
    s->itemsize = 4;
    break;
  case CV_64F:
    s->typekind = 'f';
    s->itemsize = 8;
    break;
  default:
    assert(0);
  }

  s->flags = NPY_WRITEABLE | NPY_NOTSWAPPED;
}

static PyObject *cvmat_array_struct(cvmat_t *cva)
{
  CvMat *m;
  convert_to_CvMat((PyObject *)cva, &m, "");

  arrayTrack *at = new arrayTrack;
  PyArrayInterface *s = &at->s;

  at->o = cva->data;
  Py_INCREF(at->o);

  arrayinterface_common(s, m->type);

  if (CV_MAT_CN(m->type) == 1) {
    s->nd = 2;
    s->shape = new npy_intp[2];
    s->shape[0] = m->rows;
    s->shape[1] = m->cols;
    s->strides = new npy_intp[2];
    s->strides[0] = m->step;
    s->strides[1] = s->itemsize;
  } else {
    s->nd = 3;
    s->shape = new npy_intp[3];
    s->shape[0] = m->rows;
    s->shape[1] = m->cols;
    s->shape[2] = CV_MAT_CN(m->type);
    s->strides = new npy_intp[3];
    s->strides[0] = m->step;
    s->strides[1] = s->itemsize * CV_MAT_CN(m->type);
    s->strides[2] = s->itemsize;
  }
  s->data = (void*)(m->data.ptr);
  s->descr = PyList_New(1);
  char typestr[10];
  sprintf(typestr, "<%c%d", s->typekind, s->itemsize);
  PyList_SetItem(s->descr, 0, Py_BuildValue("(ss)", "x", typestr));

  return PyCObject_FromVoidPtr(s, arrayTrackDtor);
}

static PyObject *cvmatnd_array_struct(cvmatnd_t *cva)
{
  CvMatND *m;
  convert_to_CvMatND((PyObject *)cva, &m, "");

  arrayTrack *at = new arrayTrack;
  PyArrayInterface *s = &at->s;

  at->o = cva->data;
  Py_INCREF(at->o);

  arrayinterface_common(s, m->type);

  int i;
  if (CV_MAT_CN(m->type) == 1) {
    s->nd = m->dims;
    s->shape = new npy_intp[s->nd];
    for (i = 0; i < s->nd; i++)
      s->shape[i] = m->dim[i].size;
    s->strides = new npy_intp[s->nd];
    for (i = 0; i < (s->nd - 1); i++)
      s->strides[i] = m->dim[i].step;
    s->strides[s->nd - 1] = s->itemsize;
  } else {
    s->nd = m->dims + 1;
    s->shape = new npy_intp[s->nd];
    for (i = 0; i < (s->nd - 1); i++)
      s->shape[i] = m->dim[i].size;
    s->shape[s->nd - 1] = CV_MAT_CN(m->type);

    s->strides = new npy_intp[s->nd];
    for (i = 0; i < (s->nd - 2); i++)
      s->strides[i] = m->dim[i].step;
    s->strides[s->nd - 2] = s->itemsize * CV_MAT_CN(m->type);
    s->strides[s->nd - 1] = s->itemsize;
  }
  s->data = (void*)(m->data.ptr);
  s->descr = PyList_New(1);
  char typestr[10];
  sprintf(typestr, "<%c%d", s->typekind, s->itemsize);
  PyList_SetItem(s->descr, 0, Py_BuildValue("(ss)", "x", typestr));

  return PyCObject_FromVoidPtr(s, arrayTrackDtor);
}
#endif

static PyGetSetDef cvmat_getseters[] = {
  {(char*)"type",   (getter)cvmat_gettype, (setter)NULL, (char*)"type",   NULL},
  {(char*)"step",   (getter)cvmat_getstep, (setter)NULL, (char*)"step",   NULL},
  {(char*)"rows",   (getter)cvmat_getrows, (setter)NULL, (char*)"rows",   NULL},
  {(char*)"cols",   (getter)cvmat_getcols, (setter)NULL, (char*)"cols",   NULL},
  {(char*)"channels",(getter)cvmat_getchannels, (setter)NULL, (char*)"channels",   NULL},
  {(char*)"width",  (getter)cvmat_getcols, (setter)NULL, (char*)"width",  NULL},
  {(char*)"height", (getter)cvmat_getrows, (setter)NULL, (char*)"height", NULL},
#if PYTHON_USE_NUMPY
  {(char*)"__array_struct__", (getter)cvmat_array_struct, (setter)NULL, (char*)"__array_struct__", NULL},
#endif
  {NULL}  /* Sentinel */
};

static PyMappingMethods cvmat_as_map = {
  NULL,
  &cvarr_GetItem,
  &cvarr_SetItem,
};

static PyTypeObject cvmat_Type = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                                      /*size*/
  MODULESTR".cvmat",                      /*name*/
  sizeof(cvmat_t),                        /*basicsize*/
};

static int illegal_init(PyObject *self, PyObject *args, PyObject *kwds)
{
  PyErr_SetString(opencv_error, "Cannot create cvmat directly; use CreateMat() instead");
  return -1;
}

static void cvmat_specials(void)
{
  cvmat_Type.tp_dealloc = cvmat_dealloc;
  cvmat_Type.tp_as_mapping = &cvmat_as_map;
  cvmat_Type.tp_repr = cvmat_repr;
  cvmat_Type.tp_methods = cvmat_methods;
  cvmat_Type.tp_getset = cvmat_getseters;
  cvmat_Type.tp_init = illegal_init;
}

static int is_cvmat(PyObject *o)
{
  return PyType_IsSubtype(o->ob_type, &cvmat_Type);
}

/************************************************************************/

/* cvmatnd */

static void cvmatnd_dealloc(PyObject *self)
{
  cvmatnd_t *pc = (cvmatnd_t*)self;
  Py_DECREF(pc->data);
  cvFree(&pc->a);
  PyObject_Del(self);
}

static PyObject *cvmatnd_repr(PyObject *self)
{
  CvMatND *m = ((cvmatnd_t*)self)->a;
  char str[1000];
  sprintf(str, "<cvmatnd(");
  char *d = str + strlen(str);
  sprintf(d, "type=%08x ", m->type);
  d += strlen(d);
  sprintf(d, ")>");
  return PyString_FromString(str);
}

static size_t cvmatnd_size(CvMatND *m)
{
  int bps = 1;
  switch (CV_MAT_DEPTH(m->type)) {
  case CV_8U:
  case CV_8S:
    bps = CV_MAT_CN(m->type) * 1;
    break;
  case CV_16U:
  case CV_16S:
    bps = CV_MAT_CN(m->type) * 2;
    break;
  case CV_32S:
  case CV_32F:
    bps = CV_MAT_CN(m->type) * 4;
    break;
  case CV_64F:
    bps = CV_MAT_CN(m->type) * 8;
    break;
  default:
    assert(0);
  }
  size_t l = bps;
  for (int d = 0; d < m->dims; d++) {
    l *= m->dim[d].size;
  }
  return l;
}

static PyObject *cvmatnd_tostring(PyObject *self, PyObject *args)
{
  CvMatND *m;
  if (!convert_to_CvMatND(self, &m, "self"))
    return NULL;

  int bps;
  switch (CV_MAT_DEPTH(m->type)) {
  case CV_8U:
  case CV_8S:
    bps = CV_MAT_CN(m->type) * 1;
    break;
  case CV_16U:
  case CV_16S:
    bps = CV_MAT_CN(m->type) * 2;
    break;
  case CV_32S:
  case CV_32F:
    bps = CV_MAT_CN(m->type) * 4;
    break;
  case CV_64F:
    bps = CV_MAT_CN(m->type) * 8;
    break;
  default:
    return (PyObject*)failmsg("Unrecognised depth %d", CV_MAT_DEPTH(m->type));
  }

  int l = bps;
  for (int d = 0; d < m->dims; d++) {
    l *= m->dim[d].size;
  }
  int i[CV_MAX_DIM];
  int d;
  for (d = 0; d < m->dims; d++) {
    i[d] = 0;
  }
  int rowsize = m->dim[m->dims-1].size * bps;
  char *s = new char[l];
  char *ps = s;

  int finished = 0;
  while (!finished) {
    memcpy(ps, cvPtrND(m, i), rowsize);
    ps += rowsize;
    for (d = m->dims - 2; 0 <= d; d--) {
      if (++i[d] < cvGetDimSize(m, d)) {
        break;
      } else {
        i[d] = 0;
      }
    }
    if (d < 0)
      finished = 1;
  }

  return PyString_FromStringAndSize(s, ps - s);
}

static struct PyMethodDef cvmatnd_methods[] =
{
  {"tostring", cvmatnd_tostring, METH_VARARGS},
  {NULL,          NULL}
};

static PyObject *cvmatnd_getchannels(cvmatnd_t *cva)
{
  return PyInt_FromLong(CV_MAT_CN(cva->a->type));
}

static PyGetSetDef cvmatnd_getseters[] = {
#if PYTHON_USE_NUMPY
  {(char*)"__array_struct__", (getter)cvmatnd_array_struct, (setter)NULL, (char*)"__array_struct__", NULL},
#endif
  {(char*)"channels",(getter)cvmatnd_getchannels, (setter)NULL, (char*)"channels",   NULL},
  {NULL}  /* Sentinel */
};

static PyMappingMethods cvmatnd_as_map = {
  NULL,
  &cvarr_GetItem,
  &cvarr_SetItem,
};

static PyTypeObject cvmatnd_Type = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                                      /*size*/
  MODULESTR".cvmatnd",                          /*name*/
  sizeof(cvmatnd_t),                        /*basicsize*/
};

static void cvmatnd_specials(void)
{
  cvmatnd_Type.tp_dealloc = cvmatnd_dealloc;
  cvmatnd_Type.tp_as_mapping = &cvmatnd_as_map;
  cvmatnd_Type.tp_repr = cvmatnd_repr;
  cvmatnd_Type.tp_methods = cvmatnd_methods;
  cvmatnd_Type.tp_getset = cvmatnd_getseters;
}

static int is_cvmatnd(PyObject *o)
{
  return PyType_IsSubtype(o->ob_type, &cvmatnd_Type);
}

/************************************************************************/

/* cvhistogram */

static void cvhistogram_dealloc(PyObject *self)
{
  cvhistogram_t *cvh = (cvhistogram_t*)self;
  Py_DECREF(cvh->bins);
  PyObject_Del(self);
}

static PyTypeObject cvhistogram_Type = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                                      /*size*/
  MODULESTR".cvhistogram",                /*name*/
  sizeof(cvhistogram_t),                  /*basicsize*/
};

static PyObject *cvhistogram_getbins(cvhistogram_t *cvh)
{
  Py_INCREF(cvh->bins);
  return cvh->bins;
}

static PyGetSetDef cvhistogram_getseters[] = {
  {(char*)"bins", (getter)cvhistogram_getbins, (setter)NULL, (char*)"bins", NULL},
  {NULL}  /* Sentinel */
};

static void cvhistogram_specials(void)
{
  cvhistogram_Type.tp_dealloc = cvhistogram_dealloc;
  cvhistogram_Type.tp_getset = cvhistogram_getseters;
}

/************************************************************************/

/* cvlineiterator */

static PyObject *cvlineiterator_iter(PyObject *o)
{
  Py_INCREF(o);
  return o;
}

static PyObject *cvlineiterator_next(PyObject *o)
{
  cvlineiterator_t *pi = (cvlineiterator_t*)o;

  if (pi->count) {
      pi->count--;

      CvScalar r;
      cvRawDataToScalar( (void*)(pi->iter.ptr), pi->type, &r);
      PyObject *pr = PyObject_FromCvScalar(r, pi->type);

      CV_NEXT_LINE_POINT(pi->iter);

      return pr;
  } else {
    return NULL;
  }
}

static PyTypeObject cvlineiterator_Type = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                                      /*size*/
  MODULESTR".cvlineiterator",             /*name*/
  sizeof(cvlineiterator_t),               /*basicsize*/
};

static void cvlineiterator_specials(void)
{
  cvlineiterator_Type.tp_iter = cvlineiterator_iter;
  cvlineiterator_Type.tp_iternext = cvlineiterator_next;
}

/************************************************************************/

/* memtrack */

/* Motivation for memtrack is when the storage for a Mat is an array or buffer
object.  By setting 'data' to be a memtrack, can deallocate the storage at
object destruction.

For array objects, 'backing' is the actual storage object.  memtrack holds the reference,
then DECREF's it at dealloc.

For MatND's, we need to cvDecRefData() on release, and this is what field 'backingmat' is for.

If freeptr is true, then a straight cvFree() of ptr happens.

*/


static void memtrack_dealloc(PyObject *self)
{
  memtrack_t *pi = (memtrack_t*)self;
  if (pi->backing)
    Py_DECREF(pi->backing);
  if (pi->backingmat)
    cvDecRefData(pi->backingmat);
  if (pi->freeptr)
    cvFree(&pi->ptr);
  PyObject_Del(self);
}

static PyTypeObject memtrack_Type = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                                      /*size*/
  MODULESTR".memtrack",                          /*name*/
  sizeof(memtrack_t),                        /*basicsize*/
};

Py_ssize_t memtrack_getreadbuffer(PyObject *self, Py_ssize_t segment, void **ptrptr)
{
  *ptrptr = &((memtrack_t*)self)->ptr;
  return ((memtrack_t*)self)->size;
}

Py_ssize_t memtrack_getwritebuffer(PyObject *self, Py_ssize_t segment, void **ptrptr)
{
  *ptrptr = ((memtrack_t*)self)->ptr;
  return ((memtrack_t*)self)->size;
}

Py_ssize_t memtrack_getsegcount(PyObject *self, Py_ssize_t *lenp)
{
  return (Py_ssize_t)1;
}

PyBufferProcs memtrack_as_buffer = {
  memtrack_getreadbuffer,
  memtrack_getwritebuffer,
  memtrack_getsegcount
};

static void memtrack_specials(void)
{
  memtrack_Type.tp_dealloc = memtrack_dealloc;
  memtrack_Type.tp_as_buffer = &memtrack_as_buffer;
}

/************************************************************************/

/* cvmemstorage */

static void cvmemstorage_dealloc(PyObject *self)
{
  cvmemstorage_t *ps = (cvmemstorage_t*)self;
  cvReleaseMemStorage(&(ps->a));
  PyObject_Del(self);
}

static PyTypeObject cvmemstorage_Type = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                                      /*size*/
  MODULESTR".cvmemstorage",               /*name*/
  sizeof(cvmemstorage_t),                 /*basicsize*/
};

static void cvmemstorage_specials(void)
{
  cvmemstorage_Type.tp_dealloc = cvmemstorage_dealloc;
}

/************************************************************************/

/* cvfont */

static PyTypeObject cvfont_Type = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                                      /*size*/
  MODULESTR".cvfont",                     /*name*/
  sizeof(cvfont_t),                       /*basicsize*/
};

static void cvfont_specials(void) { }

/************************************************************************/

/* cvrng */

static PyTypeObject cvrng_Type = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                                      /*size*/
  MODULESTR".cvrng",                     /*name*/
  sizeof(cvrng_t),                       /*basicsize*/
};

static void cvrng_specials(void)
{
}

/************************************************************************/

/* cvcontourtree */

static PyTypeObject cvcontourtree_Type = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                                      /*size*/
  MODULESTR".cvcontourtree",                     /*name*/
  sizeof(cvcontourtree_t),                       /*basicsize*/
};

static void cvcontourtree_specials(void) { }


/************************************************************************/

/* cvsubdiv2dedge */

static PyTypeObject cvsubdiv2dedge_Type = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                                      /*size*/
  MODULESTR".cvsubdiv2dedge",                     /*name*/
  sizeof(cvsubdiv2dedge_t),                       /*basicsize*/
};

static int cvsubdiv2dedge_compare(PyObject *o1, PyObject *o2)
{
  cvsubdiv2dedge_t *e1 = (cvsubdiv2dedge_t*)o1;
  cvsubdiv2dedge_t *e2 = (cvsubdiv2dedge_t*)o2;
  if (e1->a < e2->a)
    return -1;
  else if (e1->a > e2->a)
    return 1;
  else
    return 0;
}

static PyObject *cvquadedge_repr(PyObject *self)
{
  CvSubdiv2DEdge m = ((cvsubdiv2dedge_t*)self)->a;
  char str[1000];
  sprintf(str, "<cvsubdiv2dedge(");
  char *d = str + strlen(str);
  sprintf(d, "%zx.%d", m & ~3, (int)(m & 3));
  d += strlen(d);
  sprintf(d, ")>");
  return PyString_FromString(str);
}

static void cvsubdiv2dedge_specials(void) {
  cvsubdiv2dedge_Type.tp_compare = cvsubdiv2dedge_compare;
  cvsubdiv2dedge_Type.tp_repr = cvquadedge_repr;
}

/************************************************************************/

/* cvseq */

static void cvseq_dealloc(PyObject *self)
{
  cvseq_t *ps = (cvseq_t*)self;
  Py_DECREF(ps->container);
  PyObject_Del(self);
}

static PyObject *cvseq_h_next(PyObject *self, PyObject *args);
static PyObject *cvseq_h_prev(PyObject *self, PyObject *args);
static PyObject *cvseq_v_next(PyObject *self, PyObject *args);
static PyObject *cvseq_v_prev(PyObject *self, PyObject *args);

static struct PyMethodDef cvseq_methods[] =
{
  {"h_next", cvseq_h_next, METH_VARARGS},
  {"h_prev", cvseq_h_prev, METH_VARARGS},
  {"v_next", cvseq_v_next, METH_VARARGS},
  {"v_prev", cvseq_v_prev, METH_VARARGS},
  {NULL,          NULL}
};

static Py_ssize_t cvseq_seq_length(PyObject *o)
{
  cvseq_t *ps = (cvseq_t*)o;
  if (ps->a == NULL)
    return (Py_ssize_t)0;
  else
    return (Py_ssize_t)(ps->a->total);
}

static PyObject* cvseq_seq_getitem(PyObject *o, Py_ssize_t i)
{
  cvseq_t *ps = (cvseq_t*)o;
  CvPoint *pt;
  struct pointpair{
    CvPoint a, b;
  } *pp;
  CvPoint2D32f *pt2;
  CvPoint3D32f *pt3;

  if (i < cvseq_seq_length(o)) {
    switch (CV_SEQ_ELTYPE(ps->a)) {

    case CV_SEQ_ELTYPE_POINT:
      pt = CV_GET_SEQ_ELEM(CvPoint, ps->a, i);
      return Py_BuildValue("ii", pt->x, pt->y);

    case CV_SEQ_ELTYPE_GENERIC:
      switch (ps->a->elem_size) {
      case sizeof(CvQuadEdge2D):
        {
          cvsubdiv2dedge_t *r = PyObject_NEW(cvsubdiv2dedge_t, &cvsubdiv2dedge_Type);
          r->a = (CvSubdiv2DEdge)CV_GET_SEQ_ELEM(CvQuadEdge2D, ps->a, i);
          r->container = ps->container;
          Py_INCREF(r->container);
          return (PyObject*)r;
        }
      case sizeof(CvConnectedComp):
        {
          CvConnectedComp *cc = CV_GET_SEQ_ELEM(CvConnectedComp, ps->a, i);
          return FROM_CvConnectedComp(*cc);
        }
      default:
        printf("seq elem size is %d\n", ps->a->elem_size);
        printf("KIND %d\n", CV_SEQ_KIND(ps->a));
        assert(0);
      }
      return PyInt_FromLong(*CV_GET_SEQ_ELEM(unsigned char, ps->a, i));

    case CV_SEQ_ELTYPE_PTR:
    case CV_SEQ_ELTYPE_INDEX:
      return PyInt_FromLong(*CV_GET_SEQ_ELEM(int, ps->a, i));

    case CV_32SC4:
      pp = CV_GET_SEQ_ELEM(pointpair, ps->a, i);
      return Py_BuildValue("(ii),(ii)", pp->a.x, pp->a.y, pp->b.x, pp->b.y);

    case CV_32FC2:
      pt2 = CV_GET_SEQ_ELEM(CvPoint2D32f, ps->a, i);
      return Py_BuildValue("ff", pt2->x, pt2->y);

    case CV_SEQ_ELTYPE_POINT3D:
      pt3 = CV_GET_SEQ_ELEM(CvPoint3D32f, ps->a, i);
      return Py_BuildValue("fff", pt3->x, pt3->y, pt3->z);

    default:
      printf("Unknown element type %08x\n", CV_SEQ_ELTYPE(ps->a));
      assert(0);
      return NULL;
    }
  } else
    return NULL;
}

static PyObject* cvseq_map_getitem(PyObject *o, PyObject *item)
{
  if (PyInt_Check(item)) {
    long i = PyInt_AS_LONG(item);
    if (i < 0)
      i += cvseq_seq_length(o);
    return cvseq_seq_getitem(o, i);
  } else if (PySlice_Check(item)) {
    Py_ssize_t start, stop, step, slicelength, cur, i;
    PyObject* result;

    if (PySlice_GetIndicesEx((PySliceObject*)item, cvseq_seq_length(o),
         &start, &stop, &step, &slicelength) < 0) {
      return NULL;
    }

    if (slicelength <= 0) {
      return PyList_New(0);
    } else {
      result = PyList_New(slicelength);
      if (!result) return NULL;

      for (cur = start, i = 0; i < slicelength;
           cur += step, i++) {
        PyList_SET_ITEM(result, i, cvseq_seq_getitem(o, cur));
      }

      return result;
    }
  } else {
    PyErr_SetString(PyExc_TypeError, "CvSeq indices must be integers");
    return NULL;
  }
}

static 
PySequenceMethods cvseq_sequence = {
  cvseq_seq_length,
  NULL,
  NULL,
  cvseq_seq_getitem
};

static PyMappingMethods cvseq_mapping = {
  cvseq_seq_length,
  cvseq_map_getitem,
  NULL,
};

static PyTypeObject cvseq_Type = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                                      /*size*/
  MODULESTR".cvseq",                          /*name*/
  sizeof(cvseq_t),                        /*basicsize*/
};

static void cvseq_specials(void)
{
  cvseq_Type.tp_dealloc = cvseq_dealloc;
  cvseq_Type.tp_as_sequence = &cvseq_sequence;
  cvseq_Type.tp_as_mapping = &cvseq_mapping;
  cvseq_Type.tp_methods = cvseq_methods;
}

#define MK_ACCESSOR(FIELD) \
static PyObject *cvseq_##FIELD(PyObject *self, PyObject *args) \
{ \
  cvseq_t *ps = (cvseq_t*)self; \
  CvSeq *s = ps->a; \
  if (s->FIELD == NULL) { \
    Py_RETURN_NONE; \
  } else { \
    cvseq_t *r = PyObject_NEW(cvseq_t, &cvseq_Type); \
    r->a = s->FIELD; \
    r->container = ps->container; \
    Py_INCREF(r->container); \
    return (PyObject*)r; \
  } \
}

MK_ACCESSOR(h_next)
MK_ACCESSOR(h_prev)
MK_ACCESSOR(v_next)
MK_ACCESSOR(v_prev)
#undef MK_ACCESSOR

/************************************************************************/

/* cvset */

static void cvset_dealloc(PyObject *self)
{
  cvset_t *ps = (cvset_t*)self;
  Py_DECREF(ps->container);
  PyObject_Del(self);
}

static PyTypeObject cvset_Type = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                                      /*size*/
  MODULESTR".cvset",                          /*name*/
  sizeof(cvset_t),                        /*basicsize*/
};

static PyObject *cvset_iter(PyObject *o)
{
  Py_INCREF(o);
  cvset_t *ps = (cvset_t*)o;
  ps->i = 0;
  return o;
}

static PyObject *cvset_next(PyObject *o)
{
  cvset_t *ps = (cvset_t*)o;

  while (ps->i < ps->a->total) {
    CvSetElem *e = cvGetSetElem(ps->a, ps->i);
    int prev_i = ps->i++;
    if (e != NULL) {
      return cvseq_seq_getitem(o, prev_i);
    }
  }
  return NULL;
}

static void cvset_specials(void)
{
  cvset_Type.tp_dealloc = cvset_dealloc;
  cvset_Type.tp_iter = cvset_iter;
  cvset_Type.tp_iternext = cvset_next;
}

/************************************************************************/

/* cvsubdiv2d */

static PyTypeObject cvsubdiv2d_Type = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                                          /*size*/
  MODULESTR".cvsubdiv2d",                     /*name*/
  sizeof(cvsubdiv2d_t),                       /*basicsize*/
};

static PyObject *cvsubdiv2d_getattro(PyObject *o, PyObject *name)
{
  cvsubdiv2d_t *p = (cvsubdiv2d_t*)o;
  if (strcmp(PyString_AsString(name), "edges") == 0) {
    cvset_t *r = PyObject_NEW(cvset_t, &cvset_Type);
    r->a = p->a->edges;
    r->container = p->container;
    Py_INCREF(r->container);
    return (PyObject*)r;
  } else {
    PyErr_SetString(PyExc_TypeError, "cvsubdiv2d has no such attribute");
    return NULL;
  }
}

static void cvsubdiv2d_specials(void)
{
  cvsubdiv2d_Type.tp_getattro = cvsubdiv2d_getattro;
}

/************************************************************************/

/* cvsubdiv2dpoint */

static PyTypeObject cvsubdiv2dpoint_Type = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,                                      /*size*/
  MODULESTR".cvsubdiv2dpoint",                     /*name*/
  sizeof(cvsubdiv2dpoint_t),                       /*basicsize*/
};

static PyObject *cvsubdiv2dpoint_getattro(PyObject *o, PyObject *name)
{
  cvsubdiv2dpoint_t *p = (cvsubdiv2dpoint_t*)o;
  if (strcmp(PyString_AsString(name), "first") == 0) {
    cvsubdiv2dedge_t *r = PyObject_NEW(cvsubdiv2dedge_t, &cvsubdiv2dedge_Type);
    r->a = p->a->first;
    r->container = p->container;
    Py_INCREF(r->container);
    return (PyObject*)r;
  } else if (strcmp(PyString_AsString(name), "pt") == 0) {
    return Py_BuildValue("(ff)", p->a->pt.x, p->a->pt.y);
  } else {
    PyErr_SetString(PyExc_TypeError, "cvsubdiv2dpoint has no such attribute");
    return NULL;
  }
}

static void cvsubdiv2dpoint_specials(void)
{
  cvsubdiv2dpoint_Type.tp_getattro = cvsubdiv2dpoint_getattro;
}

/************************************************************************/
/* convert_to_X: used after PyArg_ParseTuple in the generated code  */

/*static int convert_to_PyObjectPTR(PyObject *o, PyObject **dst, const char *name = "no_name")
{
  *dst = o;
  return 1;
}

static int convert_to_PyCallableObjectPTR(PyObject *o, PyObject **dst, const char *name = "no_name")
{
  *dst = o;
  return 1;
}*/

static int convert_to_char(PyObject *o, char *dst, const char *name = "no_name")
{
  if (PyString_Check(o) && PyString_Size(o) == 1) {
    *dst = PyString_AsString(o)[0];
    return 1;
  } else {
    (*dst) = 0;
    return failmsg("Expected single character string for argument '%s'", name);
  }
}

static int convert_to_CvMemStorage(PyObject *o, CvMemStorage **dst, const char *name = "no_name")
{
  if (PyType_IsSubtype(o->ob_type, &cvmemstorage_Type)) {
    (*dst) = (((cvmemstorage_t*)o)->a);
    return 1;
  } else {
    (*dst) = (CvMemStorage*)NULL;
    return failmsg("Expected CvMemStorage for argument '%s'", name);
  }
}

static int convert_to_CvSeq(PyObject *o, CvSeq **dst, const char *name = "no_name")
{
  if (PyType_IsSubtype(o->ob_type, &cvseq_Type)) {
    (*dst) = (((cvseq_t*)o)->a);
    return 1;
  } else {
    (*dst) = (CvSeq*)NULL;
    return failmsg("Expected CvSeq for argument '%s'", name);
  }
}

static int convert_to_CvSize(PyObject *o, CvSize *dst, const char *name = "no_name")
{
  if (!PyArg_ParseTuple(o, "ii", &dst->width, &dst->height))
    return failmsg("CvSize argument '%s' expects two integers", name);
  else
    return 1;
}

static int convert_to_CvScalar(PyObject *o, CvScalar *s, const char *name = "no_name")
{
  if (PySequence_Check(o)) {
    PyObject *fi = PySequence_Fast(o, name);
    if (fi == NULL)
      return 0;
    if (4 < PySequence_Fast_GET_SIZE(fi))
        return failmsg("CvScalar value for argument '%s' is longer than 4", name);
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fi); i++) {
      PyObject *item = PySequence_Fast_GET_ITEM(fi, i);
      if (PyFloat_Check(item) || PyInt_Check(item)) {
        s->val[i] = PyFloat_AsDouble(item);
      } else {
        return failmsg("CvScalar value for argument '%s' is not numeric", name);
      }
    }
    Py_DECREF(fi);
  } else {
    if (PyFloat_Check(o) || PyInt_Check(o)) {
      s->val[0] = PyFloat_AsDouble(o);
    } else {
      return failmsg("CvScalar value for argument '%s' is not numeric", name);
    }
  }
  return 1;
}

static int convert_to_CvPointPTR(PyObject *o, CvPoint **p, const char *name = "no_name")
{
  if (!PySequence_Check(o))
    return failmsg("Expected sequence for point list argument '%s'", name);
  PyObject *fi = PySequence_Fast(o, name);
  if (fi == NULL)
    return 0;
  *p = new CvPoint[PySequence_Fast_GET_SIZE(fi)];
  for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fi); i++) {
    PyObject *item = PySequence_Fast_GET_ITEM(fi, i);
    if (!PyTuple_Check(item))
      return failmsg("Expected tuple for element in point list argument '%s'", name);
    if (!PyArg_ParseTuple(item, "ii", &((*p)[i].x), &((*p)[i].y))) {
      return 0;
    }
  }
  Py_DECREF(fi);
  return 1;
}

static int convert_to_CvPoint2D32fPTR(PyObject *o, CvPoint2D32f **p, const char *name = "no_name")
{
  PyObject *fi = PySequence_Fast(o, name);
  if (fi == NULL)
    return 0;
  *p = new CvPoint2D32f[PySequence_Fast_GET_SIZE(fi)];
  for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fi); i++) {
    PyObject *item = PySequence_Fast_GET_ITEM(fi, i);
    if (!PyTuple_Check(item))
      return failmsg("Expected tuple for CvPoint2D32f argument '%s'", name);
    if (!PyArg_ParseTuple(item, "ff", &((*p)[i].x), &((*p)[i].y))) {
      return 0;
    }
  }
  Py_DECREF(fi);
  return 1;
}

#if 0 // not used
static int convert_to_CvPoint3D32fPTR(PyObject *o, CvPoint3D32f **p, const char *name = "no_name")
{
  PyObject *fi = PySequence_Fast(o, name);
  if (fi == NULL)
    return 0;
  *p = new CvPoint3D32f[PySequence_Fast_GET_SIZE(fi)];
  for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fi); i++) {
    PyObject *item = PySequence_Fast_GET_ITEM(fi, i);
    if (!PyTuple_Check(item))
      return failmsg("Expected tuple for CvPoint3D32f argument '%s'", name);
    if (!PyArg_ParseTuple(item, "fff", &((*p)[i].x), &((*p)[i].y), &((*p)[i].z))) {
      return 0;
    }
  }
  Py_DECREF(fi);
  return 1;
}
#endif

static int convert_to_CvStarDetectorParams(PyObject *o, CvStarDetectorParams *dst, const char *name = "no_name")
{
  if (!PyArg_ParseTuple(o,
                        "iiiii",
                        &dst->maxSize,
                        &dst->responseThreshold,
                        &dst->lineThresholdProjected,
                        &dst->lineThresholdBinarized,
                        &dst->suppressNonmaxSize))
    return failmsg("CvRect argument '%s' expects four integers", name);
  else
    return 1;
}

static int convert_to_CvRect(PyObject *o, CvRect *dst, const char *name = "no_name")
{
  if (!PyArg_ParseTuple(o, "iiii", &dst->x, &dst->y, &dst->width, &dst->height))
    return failmsg("CvRect argument '%s' expects four integers", name);
  else
    return 1;
}

static int convert_to_CvRectPTR(PyObject *o, CvRect **dst, const char *name = "no_name")
{
  *dst = new CvRect;
  if (!PyArg_ParseTuple(o, "iiii", &(*dst)->x, &(*dst)->y, &(*dst)->width, &(*dst)->height))
    return failmsg("CvRect argument '%s' expects four integers", name);
  else
    return 1;
}

static int convert_to_CvSlice(PyObject *o, CvSlice *dst, const char *name = "no_name")
{
  if (!PyArg_ParseTuple(o, "ii", &dst->start_index, &dst->end_index))
    return failmsg("CvSlice argument '%s' expects two integers", name);
  else
    return 1;
}

static int convert_to_CvPoint(PyObject *o, CvPoint *dst, const char *name = "no_name")
{
  if (!PyArg_ParseTuple(o, "ii", &dst->x, &dst->y))
    return failmsg("CvPoint argument '%s' expects two integers", name);
  else
    return 1;
}

static int convert_to_CvPoint2D32f(PyObject *o, CvPoint2D32f *dst, const char *name = "no_name")
{
  if (!PyArg_ParseTuple(o, "ff", &dst->x, &dst->y))
    return failmsg("CvPoint2D32f argument '%s' expects two floats", name);
  else
    return 1;
}

static int convert_to_CvPoint3D32f(PyObject *o, CvPoint3D32f *dst, const char *name = "no_name")
{
  if (!PyArg_ParseTuple(o, "fff", &dst->x, &dst->y, &dst->z))
    return failmsg("CvPoint3D32f argument '%s' expects three floats", name);
  else
    return 1;
}

static int convert_to_IplImage(PyObject *o, IplImage **dst, const char *name)
{
  iplimage_t *ipl = (iplimage_t*)o;
  void *buffer;
  Py_ssize_t buffer_len;

  if (!is_iplimage(o)) {
    return failmsg("Argument '%s' must be IplImage", name);
  } else if (PyString_Check(ipl->data)) {
    cvSetData(ipl->a, PyString_AsString(ipl->data) + ipl->offset, ipl->a->widthStep);
    assert(cvGetErrStatus() == 0);
    *dst = ipl->a;
    return 1;
  } else if (ipl->data && PyObject_AsWriteBuffer(ipl->data, &buffer, &buffer_len) == 0) {
    cvSetData(ipl->a, (void*)((char*)buffer + ipl->offset), ipl->a->widthStep);
    assert(cvGetErrStatus() == 0);
    *dst = ipl->a;
    return 1;
  } else {
    return failmsg("IplImage argument '%s' has no data", name);
  }
}

static int convert_to_CvMat(PyObject *o, CvMat **dst, const char *name)
{
  cvmat_t *m = (cvmat_t*)o;
  void *buffer;
  Py_ssize_t buffer_len;

  if (!is_cvmat(o)) {
#if !PYTHON_USE_NUMPY
    return failmsg("Argument '%s' must be CvMat", name);
#else
    PyObject *asmat = fromarray(o, 0);
    if (asmat == NULL)
      return failmsg("Argument '%s' must be CvMat", name);
    // now have the array obect as a cvmat, can use regular conversion
    return convert_to_CvMat(asmat, dst, name);
#endif
  } else {
    m->a->refcount = NULL;
    if (m->data && PyString_Check(m->data)) {
      assert(cvGetErrStatus() == 0);
      char *ptr = PyString_AsString(m->data) + m->offset;
      cvSetData(m->a, ptr, m->a->step);
      assert(cvGetErrStatus() == 0);
      *dst = m->a;
      return 1;
    } else if (m->data && PyObject_AsWriteBuffer(m->data, &buffer, &buffer_len) == 0) {
      cvSetData(m->a, (void*)((char*)buffer + m->offset), m->a->step);
      assert(cvGetErrStatus() == 0);
      *dst = m->a;
      return 1;
    } else {
      return failmsg("CvMat argument '%s' has no data", name);
    }
  }
}

static int convert_to_CvMatND(PyObject *o, CvMatND **dst, const char *name)
{
  cvmatnd_t *m = (cvmatnd_t*)o;
  void *buffer;
  Py_ssize_t buffer_len;

  if (!is_cvmatnd(o)) {
    return failmsg("Argument '%s' must be CvMatND", name);
  } else if (m->data && PyString_Check(m->data)) {
    m->a->data.ptr = ((uchar*)PyString_AsString(m->data)) + m->offset;
    *dst = m->a;
    return 1;
  } else if (m->data && PyObject_AsWriteBuffer(m->data, &buffer, &buffer_len) == 0) {
    m->a->data.ptr = ((uchar*)buffer + m->offset);
    *dst = m->a;
    return 1;
  } else {
    return failmsg("CvMatND argument '%s' has no data", name);
  }
}

static int convert_to_CvArr(PyObject *o, CvArr **dst, const char *name)
{
  if (o == Py_None) {
    *dst = (void*)NULL;
    return 1;
  } else if (is_iplimage(o)) {
    return convert_to_IplImage(o, (IplImage**)dst, name);
  } else if (is_cvmat(o)) {
    return convert_to_CvMat(o, (CvMat**)dst, name);
  } else if (is_cvmatnd(o)) {
    return convert_to_CvMatND(o, (CvMatND**)dst, name);
  } else {
#if !PYTHON_USE_NUMPY
    return failmsg("CvArr argument '%s' must be IplImage, CvMat or CvMatND", name);
#else
    PyObject *asmat = fromarray(o, 0);
    if (asmat == NULL)
      return failmsg("CvArr argument '%s' must be IplImage, CvMat, CvMatND, or support the array interface", name);
    // now have the array obect as a cvmat, can use regular conversion
    return convert_to_CvArr(asmat, dst, name);
#endif
  }
}

static int convert_to_CvHistogram(PyObject *o, CvHistogram **dst, const char *name = "no_name")
{
  if (PyType_IsSubtype(o->ob_type, &cvhistogram_Type)) {
    cvhistogram_t *ht = (cvhistogram_t*)o;
    *dst = &ht->h;
    return convert_to_CvArr(ht->bins, &(ht->h.bins), "bins");
  } else {
    *dst = (CvHistogram *)NULL;
    return failmsg("Expected CvHistogram for argument '%s'", name);
  }
}

// Used by FillPoly, FillConvexPoly, PolyLine
struct pts_npts_contours {
  CvPoint** pts;
  int* npts;
  int contours;
};

static int convert_to_pts_npts_contours(PyObject *o, pts_npts_contours *dst, const char *name = "no_name")
{
  PyObject *fi = PySequence_Fast(o, name);
  if (fi == NULL)
    return 0;
  dst->contours = PySequence_Fast_GET_SIZE(fi);
  dst->pts = new CvPoint*[dst->contours];
  dst->npts = new int[dst->contours];
  for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fi); i++) {
    if (!convert_to_CvPointPTR(PySequence_Fast_GET_ITEM(fi, i), &dst->pts[i], name))
      return 0;
    dst->npts[i] = PySequence_Size(PySequence_Fast_GET_ITEM(fi, i)); // safe because convert_ just succeeded
  }
  Py_DECREF(fi);
  return 1;
}

class cvarrseq {
public:
  union {
    CvSeq *seq;
    CvArr *mat;
  };
  int freemat;
  cvarrseq() {
    freemat = false;
  }
  ~cvarrseq() {
    if (freemat) {
      cvReleaseMat((CvMat**)&mat);
    }
  }
};

static int is_convertible_to_mat(PyObject *o)
{
#if PYTHON_USE_NUMPY
  if (PyObject_HasAttrString(o, "__array_struct__")) {
    PyObject *ao = PyObject_GetAttrString(o, "__array_struct__");
    if (ao != NULL &&
        PyCObject_Check(ao) &&
        ((PyArrayInterface*)PyCObject_AsVoidPtr(ao))->two == 2) {
      return 1;
    }
  }
#endif
  return is_iplimage(o) && is_cvmat(o) && is_cvmatnd(o);
}

static int convert_to_cvarrseq(PyObject *o, cvarrseq *dst, const char *name = "no_name")
{
  if (PyType_IsSubtype(o->ob_type, &cvseq_Type)) {
    return convert_to_CvSeq(o, &(dst->seq), name);
  } else if (is_convertible_to_mat(o)) {
    int r = convert_to_CvArr(o, &(dst->mat), name);
    return r;
  } else if (PySequence_Check(o)) {
    PyObject *fi = PySequence_Fast(o, name);
    if (fi == NULL)
      return 0;
    Py_ssize_t size = -1;
    // Make a pass through the sequence, checking that each element is
    // a sequence and that they are all the same size
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fi); i++) {
      PyObject *e = PySequence_Fast_GET_ITEM(fi, i);

      if (!PySequence_Check(e))
        return failmsg("Sequence '%s' must contain sequences", name);
      if (i == 0)
        size = (int)PySequence_Size(e);
      else if (size != PySequence_Size(e))
        return failmsg("All elements of sequence '%s' must be same size", name);
    }
    assert(size != -1);
    CvMat *mt = cvCreateMat((int)PySequence_Fast_GET_SIZE(fi), 1, CV_32SC(size));
    dst->freemat = true; // dealloc this mat when done
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fi); i++) {
      PyObject *e = PySequence_Fast_GET_ITEM(fi, i);
      PyObject *fe = PySequence_Fast(e, name);
      assert(fe != NULL);
      int *pdst = (int*)cvPtr2D(mt, i, 0);
      for (Py_ssize_t j = 0; j < size; j++) {
        PyObject *num = PySequence_Fast_GET_ITEM(fe, j);
        if (!PyNumber_Check(num)) {
          return failmsg("Sequence must contain numbers", name);
        }
        *pdst++ = PyInt_AsLong(num);
      }
      Py_DECREF(fe);
    }
    Py_DECREF(fi);
    dst->mat = mt;
    return 1;
  } else {
    return failmsg("Argument '%s' must be CvSeq, CvArr, or a sequence of numbers");
  }
}

struct cvarr_count {
  CvArr **cvarr;
  int count;
};

static int convert_to_cvarr_count(PyObject *o, cvarr_count *dst, const char *name = "no_name")
{
  PyObject *fi = PySequence_Fast(o, name);
  if (fi == NULL)
    return 0;
  dst->count = PySequence_Fast_GET_SIZE(fi);
  dst->cvarr = new CvArr*[dst->count];
  for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fi); i++) {
    if (!convert_to_CvArr(PySequence_Fast_GET_ITEM(fi, i), &dst->cvarr[i], name))
      return 0;
  }
  Py_DECREF(fi);
  return 1;
}

struct intpair
{
  int *pairs;
  int count;
};

static int convert_to_intpair(PyObject *o, intpair *dst, const char *name = "no_name")
{
  PyObject *fi = PySequence_Fast(o, name);
  if (fi == NULL)
    return 0;
  dst->count = PySequence_Fast_GET_SIZE(fi);
  dst->pairs = new int[2 * dst->count];
  for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fi); i++) {
    PyObject *item = PySequence_Fast_GET_ITEM(fi, i);
    if (!PyArg_ParseTuple(item, "ii", &dst->pairs[2 * i], &dst->pairs[2 * i + 1])) {
      return 0;
    }
  }
  Py_DECREF(fi);
  return 1;
}

struct cvpoint2d32f_count {
  CvPoint2D32f* points;
  int count;
};

static int convert_to_cvpoint2d32f_count(PyObject *o, cvpoint2d32f_count *dst, const char *name = "no_name")
{
  if (PyInt_Check(o)) {
    dst->count = PyInt_AsLong(o);
    dst->points = new CvPoint2D32f[dst->count];
    return 1;
  } else {
    return failmsg("Expected integer for CvPoint2D32f count");
  }
}

struct floats {
  float *f;
  int count;
};
static int convert_to_floats(PyObject *o, floats *dst, const char *name = "no_name")
{
  if (PySequence_Check(o)) {
    PyObject *fi = PySequence_Fast(o, name);
    if (fi == NULL)
      return 0;
    dst->count = PySequence_Fast_GET_SIZE(fi);
    dst->f = new float[dst->count];
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fi); i++) {
      PyObject *item = PySequence_Fast_GET_ITEM(fi, i);
      dst->f[i] = (float)PyFloat_AsDouble(item);
    }
    Py_DECREF(fi);
  } else if (PyNumber_Check(o)) {
    dst->count = 1;
    dst->f = new float[1];
    dst->f[0] = (float)PyFloat_AsDouble(o);
  } else {
    return failmsg("Expected list of floats, or float for argument '%s'", name);
  }
  return 1;
}

struct chars {
  char *f;
  int count;
};
/// convert_to_chars not used

struct CvPoints {
  CvPoint *p;
  int count;
};
static int convert_to_CvPoints(PyObject *o, CvPoints *dst, const char *name = "no_name")
{
  PyObject *fi = PySequence_Fast(o, name);
  if (fi == NULL)
    return 0;
  dst->count = PySequence_Fast_GET_SIZE(fi);
  dst->p = new CvPoint[dst->count];
  for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fi); i++) {
    PyObject *item = PySequence_Fast_GET_ITEM(fi, i);
    convert_to_CvPoint(item, &dst->p[i], name);
  }
  Py_DECREF(fi);
  return 1;
}

struct CvPoint3D32fs {
  CvPoint3D32f *p;
  int count;
};
static int convert_to_CvPoint3D32fs(PyObject *o, CvPoint3D32fs *dst, const char *name = "no_name")
{
  PyObject *fi = PySequence_Fast(o, name);
  if (fi == NULL)
    return 0;
  dst->count = PySequence_Fast_GET_SIZE(fi);
  dst->p = new CvPoint3D32f[dst->count];
  for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fi); i++) {
    PyObject *item = PySequence_Fast_GET_ITEM(fi, i);
    convert_to_CvPoint3D32f(item, &dst->p[i], name);
  }
  Py_DECREF(fi);
  return 1;
}

struct CvPoint2D32fs {
  CvPoint2D32f *p;
  int count;
};
static int convert_to_CvPoint2D32fs(PyObject *o, CvPoint2D32fs *dst, const char *name = "no_name")
{
  PyObject *fi = PySequence_Fast(o, name);
  if (fi == NULL)
    return 0;
  dst->count = PySequence_Fast_GET_SIZE(fi);
  dst->p = new CvPoint2D32f[dst->count];
  for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fi); i++) {
    PyObject *item = PySequence_Fast_GET_ITEM(fi, i);
    convert_to_CvPoint2D32f(item, &dst->p[i], name);
  }
  Py_DECREF(fi);
  return 1;
}

struct ints {
  int *i;
  int count;
};
static int convert_to_ints(PyObject *o, ints *dst, const char *name = "no_name")
{
  PyObject *fi = PySequence_Fast(o, name);
  if (fi == NULL)
    return 0;
  dst->count = PySequence_Fast_GET_SIZE(fi);
  dst->i = new int[dst->count];
  for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fi); i++) {
    PyObject *item = PySequence_Fast_GET_ITEM(fi, i);
    dst->i[i] = PyInt_AsLong(item);
  }
  Py_DECREF(fi);
  return 1;
}

struct ints0 {
  int *i;
  int count;
};
static int convert_to_ints0(PyObject *o, ints0 *dst, const char *name = "no_name")
{
  PyObject *fi = PySequence_Fast(o, name);
  if (fi == NULL)
    return 0;
  dst->count = PySequence_Fast_GET_SIZE(fi);
  dst->i = new int[dst->count + 1];
  for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fi); i++) {
    PyObject *item = PySequence_Fast_GET_ITEM(fi, i);
    dst->i[i] = PyInt_AsLong(item);
  }
  dst->i[dst->count] = 0;
  Py_DECREF(fi);
  return 1;
}

struct dims
{
  int count;
  int i[CV_MAX_DIM];
  int step[CV_MAX_DIM];
  int length[CV_MAX_DIM];
};

static int convert_to_dim(PyObject *item, int i, dims *dst, CvArr *cva, const char *name = "no_name")
{
  if (PySlice_Check(item)) {
    Py_ssize_t start, stop, step, slicelength;
    PySlice_GetIndicesEx((PySliceObject*)item, cvGetDimSize(cva, i), &start, &stop, &step, &slicelength);
    dst->i[i] = start;
    dst->step[i] = step;
    dst->length[i] = slicelength;
  } else {
    int index = PyInt_AsLong(item);
    if (0 <= index)
      dst->i[i] = index;
    else
      dst->i[i] = cvGetDimSize(cva, i) + index;
    dst->step[i] = 0;
    dst->length[i] = 1;
  }
  return 1;
}

static int convert_to_dims(PyObject *o, dims *dst, CvArr *cva, const char *name = "no_name")
{
  if (!PyTuple_Check(o)) {
    dst->count = 1;
    return convert_to_dim(o, 0, dst, cva, name);
  } else {
    PyObject *fi = PySequence_Fast(o, name);
    if (fi == NULL) {
      PyErr_SetString(PyExc_TypeError, "Expected tuple for index");
      return 0;
    }
    dst->count = PySequence_Fast_GET_SIZE(fi);
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fi); i++) {
      if (i >= cvGetDims(cva)) {
        return failmsg("Access specifies %d dimensions, but array only has %d", PySequence_Fast_GET_SIZE(fi), cvGetDims(cva));
      }
      PyObject *item = PySequence_Fast_GET_ITEM(fi, i);
      if (!convert_to_dim(item, i, dst, cva, name))
        return 0;
    }
    Py_DECREF(fi);
    return 1;
  }
}

struct IplImages {
  IplImage **ims;
  int count;
};
static int convert_to_IplImages(PyObject *o, IplImages *dst, const char *name = "no_name")
{
  PyObject *fi = PySequence_Fast(o, name);
  if (fi == NULL)
    return 0;
  dst->count = PySequence_Fast_GET_SIZE(fi);
  dst->ims = new IplImage*[dst->count];
  for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fi); i++) {
    PyObject *item = PySequence_Fast_GET_ITEM(fi, i);
    if (!convert_to_IplImage(item, &dst->ims[i]))
      return 0;
  }
  Py_DECREF(fi);
  return 1;
}

struct CvArrs {
  CvArr **ims;
  int count;
};
static int convert_to_CvArrs(PyObject *o, CvArrs *dst, const char *name = "no_name")
{
  PyObject *fi = PySequence_Fast(o, name);
  if (fi == NULL)
    return 0;
  dst->count = PySequence_Fast_GET_SIZE(fi);
  dst->ims = new CvArr*[dst->count];
  for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fi); i++) {
    PyObject *item = PySequence_Fast_GET_ITEM(fi, i);
    if (!convert_to_CvArr(item, &dst->ims[i]))
      return 0;
  }
  Py_DECREF(fi);
  return 1;
}

/*static int convert_to_floatPTRPTR(PyObject *o, float*** dst, const char *name = "no_name")
{
  PyObject *fi = PySequence_Fast(o, name);
  if (fi == NULL)
    return 0;
  Py_ssize_t sz = PySequence_Fast_GET_SIZE(fi);
  float **r = new float*[sz];
  for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fi); i++) {
    PyObject *item = PySequence_Fast_GET_ITEM(fi, i);
    floats ff;
    if (!convert_to_floats(item, &ff))
      return 0;
    r[i] = ff.f;
  }
  *dst = r;
  return 1;
}*/

static int convert_to_CvFontPTR(PyObject *o, CvFont** dst, const char *name = "no_name")
{
  if (PyType_IsSubtype(o->ob_type, &cvfont_Type)) {
    (*dst) = &(((cvfont_t*)o)->a);
    return 1;
  } else {
    (*dst) = (CvFont*)NULL;
    return failmsg("Expected CvFont for argument '%s'", name);
  }
}

/*static int convert_to_CvContourTreePTR(PyObject *o, CvContourTree** dst, const char *name = "no_name")
{
  if (PyType_IsSubtype(o->ob_type, &cvcontourtree_Type)) {
    (*dst) = ((cvcontourtree_t*)o)->a;
    return 1;
  } else {
    (*dst) = NULL;
    return failmsg("Expected CvContourTree for argument '%s'", name);
  }
}*/

static int convert_to_CvRNGPTR(PyObject *o, CvRNG** dst, const char *name = "no_name")
{
  if (PyType_IsSubtype(o->ob_type, &cvrng_Type)) {
    (*dst) = &(((cvrng_t*)o)->a);
    return 1;
  } else {
    (*dst) = (CvRNG*)NULL;
    return failmsg("Expected CvRNG for argument '%s'", name);
  }
}

typedef void* generic;
static int convert_to_generic(PyObject *o, generic *dst, const char *name = "no_name")
{
  if (PyType_IsSubtype(o->ob_type, &iplimage_Type))
    return convert_to_IplImage(o, (IplImage**)dst, name);
  else if (PyType_IsSubtype(o->ob_type, &cvmat_Type))
    return convert_to_CvMat(o, (CvMat**)dst, name);
  else if (PyType_IsSubtype(o->ob_type, &cvmatnd_Type))
    return convert_to_CvMatND(o, (CvMatND**)dst, name);
  else {
    return failmsg("Cannot identify type of '%s'", name);
  }
}

static int convert_to_CvTermCriteria(PyObject *o, CvTermCriteria* dst, const char *name = "no_name")
{
  if (!PyArg_ParseTuple(o, "iid", &dst->type, &dst->max_iter, &dst->epsilon))
    return 0;
  return 1;
}

static int convert_to_CvBox2D(PyObject *o, CvBox2D* dst, const char *name = "no_name")
{
  if (!PyArg_ParseTuple(o, "(ff)(ff)f", &dst->center.x, &dst->center.y, &dst->size.width, &dst->size.height, &dst->angle))
    return 0;
  return 1;
}

static int convert_to_CvSubdiv2DPTR(PyObject *o, CvSubdiv2D** dst, const char *name = "no_name")
{
  if (PyType_IsSubtype(o->ob_type, &cvsubdiv2d_Type)) {
    (*dst) = (((cvsubdiv2d_t*)o)->a);
    return 1;
  } else {
    (*dst) = (CvSubdiv2D*)NULL;
    return failmsg("Expected CvSubdiv2D for argument '%s'", name);
  }
}

static int convert_to_CvNextEdgeType(PyObject *o, CvNextEdgeType *dst, const char *name = "no_name")
{
  if (!PyInt_Check(o)) {
    *dst = (CvNextEdgeType)NULL;
    return failmsg("Expected number for CvNextEdgeType argument '%s'", name);
  } else {
    *dst = (CvNextEdgeType)PyInt_AsLong(o);
    return 1;
  }
}

static int convert_to_CvSubdiv2DEdge(PyObject *o, CvSubdiv2DEdge *dst, const char *name = "no_name")
{
  if (PyType_IsSubtype(o->ob_type, &cvsubdiv2dedge_Type)) {
    (*dst) = (((cvsubdiv2dedge_t*)o)->a);
    return 1;
  } else {
    *dst = 0L;
    return failmsg("Expected CvSubdiv2DEdge for argument '%s'", name);
  }
}

/************************************************************************/

static PyObject *pythonize_CvMat(cvmat_t *m)
{
  // Need to make this CvMat look like any other, with a Python 
  // buffer object as its data.
  CvMat *mat = m->a;
  assert(mat->step != 0);
#if 0
  PyObject *data = PyString_FromStringAndSize((char*)(mat->data.ptr), mat->rows * mat->step);
#else
  memtrack_t *o = PyObject_NEW(memtrack_t, &memtrack_Type);
  size_t gap = mat->data.ptr - (uchar*)mat->refcount;
  o->ptr = mat->refcount;
  o->owner = __LINE__;
  o->freeptr = true;
  o->size = gap + mat->rows * mat->step;
  o->backing = NULL;
  o->backingmat = NULL;
  PyObject *data = PyBuffer_FromReadWriteObject((PyObject*)o, (size_t)gap, mat->rows * mat->step);
  if (data == NULL)
    return NULL;
#endif
  m->data = data;
  m->offset = 0;
  Py_DECREF(o);

  // Now m has a reference to data, which has a reference to o.

  return (PyObject*)m;
}

static PyObject *pythonize_foreign_CvMat(cvmat_t *m)
{
  // Need to make this CvMat look like any other, with a Python 
  // buffer object as its data.
  // Difference here is that the buffer is 'foreign' (from NumPy, for example)
  CvMat *mat = m->a;
  assert(mat->step != 0);
#if 0
  PyObject *data = PyString_FromStringAndSize((char*)(mat->data.ptr), mat->rows * mat->step);
#else
  memtrack_t *o = PyObject_NEW(memtrack_t, &memtrack_Type);
  o->ptr = mat->data.ptr;
  o->owner = __LINE__;
  o->freeptr = false;
  o->size = mat->rows * mat->step;
  o->backing = NULL;
  o->backingmat = mat;
  PyObject *data = PyBuffer_FromReadWriteObject((PyObject*)o, (size_t)0, mat->rows * mat->step);
  if (data == NULL)
    return NULL;
#endif
  m->data = data;
  m->offset = 0;
  Py_DECREF(o);

  // Now m has a reference to data, which has a reference to o.

  return (PyObject*)m;
}

static PyObject *pythonize_IplImage(iplimage_t *cva)
{
  // Need to make this iplimage look like any other, with a Python 
  // string as its data.
  // So copy the image data into a Python string object, then release 
  // it.

  IplImage *ipl = (IplImage*)(cva->a);
  // PyObject *data = PyString_FromStringAndSize(ipl->imageData, ipl->imageSize);

  memtrack_t *o = PyObject_NEW(memtrack_t, &memtrack_Type);
  assert(ipl->imageDataOrigin == ipl->imageData);
  o->ptr = ipl->imageDataOrigin;
  o->owner = __LINE__;
  o->freeptr = true;
  o->size = ipl->height * ipl->widthStep;
  o->backing = NULL;
  o->backingmat = NULL;
  PyObject *data = PyBuffer_FromReadWriteObject((PyObject*)o, (size_t)0, o->size);
  if (data == NULL)
    return NULL;
  Py_DECREF(o);
  cva->data = data;
  cva->offset = 0;

  return (PyObject*)cva;
}

static PyObject *pythonize_CvMatND(cvmatnd_t *m, PyObject *backing = NULL)
{
  //
  // Need to make this CvMatND look like any other, with a Python 
  // buffer object as its data.
  //

  CvMatND *mat = m->a;
  assert(mat->dim[0].step != 0);
#if 0
  PyObject *data = PyString_FromStringAndSize((char*)(mat->data.ptr), mat->dim[0].size * mat->dim[0].step);
#else
  memtrack_t *o = PyObject_NEW(memtrack_t, &memtrack_Type);
  o->ptr = mat->data.ptr;
  o->owner = __LINE__;
  o->freeptr = false;
  o->size = cvmatnd_size(mat);
  Py_XINCREF(backing);
  o->backing = backing;
  o->backingmat = mat;
  PyObject *data = PyBuffer_FromReadWriteObject((PyObject*)o, (size_t)0, o->size);
  Py_DECREF(o); // Now 'data' holds the only reference to 'o'
  if (data == NULL)
    return NULL;
#endif
  m->data = data;
  m->offset = 0;

  return (PyObject*)m;
}

/************************************************************************/
/* FROM_xxx:   C -> Python converters.
 *
 * Turn various OpenCV types (and some aggregate types above)
 * into Python objects.  Used by the generated code.
 *
 * All these functions and macros return a new reference.
 */


static PyObject *_FROM_CvSeqPTR(CvSeq *s, PyObject *storage)
{
  cvseq_t *ps = PyObject_NEW(cvseq_t, &cvseq_Type);
  ps->a = s;
  ps->container = storage;
  Py_INCREF(ps->container);
  return (PyObject*)ps;
}

static PyObject *_FROM_CvSubdiv2DPTR(CvSubdiv2D *s, PyObject *storage)
{
  cvsubdiv2d_t *ps = PyObject_NEW(cvsubdiv2d_t, &cvsubdiv2d_Type);
  ps->a = s;
  ps->container = storage;
  Py_INCREF(ps->container);
  return (PyObject*)ps;
}

static PyObject *FROM_floats(floats r)
{
  PyObject *pr;

  pr = PyList_New(r.count);
  for (Py_ssize_t i = 0; i < (Py_ssize_t)r.count; i++) {
    PyList_SetItem(pr, i, PyFloat_FromDouble(r.f[i]));
  }
  return pr;
}

static PyObject *FROM_chars(chars r)
{
  PyObject *pr;

  pr = PyList_New(r.count);
  for (Py_ssize_t i = 0; i < (Py_ssize_t)r.count; i++) {
    PyList_SetItem(pr, i, PyInt_FromLong(r.f[i]));
  }
  return pr;
}

static PyObject *FROM_cvpoint2d32f_count(cvpoint2d32f_count r)
{
  PyObject *pr;

  pr = PyList_New(r.count);
  for (Py_ssize_t i = 0; i < (Py_ssize_t)r.count; i++) {
    PyList_SetItem(pr, i, FROM_CvPoint2D32f(r.points[i]));
  }
  return pr;
}

static PyObject *FROM_CvPoint2D32fs(CvPoint2D32fs r)
{
  PyObject *pr;

  pr = PyList_New(r.count);
  for (Py_ssize_t i = 0; i < (Py_ssize_t)r.count; i++) {
    PyList_SetItem(pr, i, FROM_CvPoint2D32f(r.p[i]));
  }
  return pr;
}

typedef CvSeq CvSeqOfCvConvexityDefect;
static PyObject *FROM_CvSeqOfCvConvexityDefectPTR(CvSeqOfCvConvexityDefect *r)
{
  PyObject *pr;
  pr = PyList_New(r->total);
  for (int i = 0; i < r->total; i++) {
    CvConvexityDefect *pd = CV_GET_SEQ_ELEM(CvConvexityDefect, r, i);
    PyList_SetItem(pr, i, Py_BuildValue("(ii)(ii)(ii)f",
                                        pd->start->x, pd->start->y, 
                                        pd->end->x, pd->end->y, 
                                        pd->depth_point->x, pd->depth_point->y, 
                                        pd->depth));
  }
  // This function has copied the CvSeq data into a list.  Hence the
  // CvSeq is not being returned to the caller.  Hence, no reference
  // count increase for the storage, unlike _FROM_CvSeqPTR.
  return pr;
}

typedef CvSeq CvSeqOfCvAvgComp;
static PyObject *FROM_CvSeqOfCvAvgCompPTR(CvSeqOfCvAvgComp *r)
{
  PyObject *pr;
  pr = PyList_New(r->total);
  for (int i = 0; i < r->total; i++) {
    CvAvgComp *pd = CV_GET_SEQ_ELEM(CvAvgComp, r, i);
    PyList_SetItem(pr, i, Py_BuildValue("(iiii)i",
                                        pd->rect.x, pd->rect.y, 
                                        pd->rect.width, pd->rect.height, 
                                        pd->neighbors));
  }
  // This function has copied the CvSeq data into a list.  Hence the
  // CvSeq is not being returned to the caller.  Hence, no reference
  // count increase for the storage, unlike _FROM_CvSeqPTR.
  return pr;
}

typedef CvSeq CvSeqOfCvStarKeypoint;
static PyObject *FROM_CvSeqOfCvStarKeypointPTR(CvSeqOfCvStarKeypoint *r)
{
  PyObject *pr;
  pr = PyList_New(r->total);
  for (int i = 0; i < r->total; i++) {
    CvStarKeypoint *pd = CV_GET_SEQ_ELEM(CvStarKeypoint, r, i);
    PyList_SetItem(pr, i, Py_BuildValue("(ii)if",
                                        pd->pt.x, pd->pt.y, 
                                        pd->size,
                                        pd->response));
  }
  // This function has copied the CvSeq data into a list.  Hence the
  // CvSeq is not being returned to the caller.  Hence, no reference
  // count increase for the storage, unlike _FROM_CvSeqPTR.
  return pr;
}

typedef CvSeq CvSeqOfCvSURFPoint;
static PyObject *FROM_CvSeqOfCvSURFPointPTR(CvSeqOfCvSURFPoint *r)
{
  PyObject *pr;
  pr = PyList_New(r->total);
  for (int i = 0; i < r->total; i++) {
    CvSURFPoint *pd = CV_GET_SEQ_ELEM(CvSURFPoint, r, i);
    PyList_SetItem(pr, i, Py_BuildValue("(ff)iiff",
                                        pd->pt.x, pd->pt.y, 
                                        pd->laplacian,
                                        pd->size,
                                        pd->dir,
                                        pd->hessian));
  }
  // This function has copied the CvSeq data into a list.  Hence the
  // CvSeq is not being returned to the caller.  Hence, no reference
  // count increase for the storage, unlike _FROM_CvSeqPTR.
  return pr;
}

typedef CvSeq CvSeqOfCvSURFDescriptor;
static PyObject *FROM_CvSeqOfCvSURFDescriptorPTR(CvSeqOfCvSURFDescriptor *r)
{
  PyObject *pr;
  pr = PyList_New(r->total);
  for (int i = 0; i < r->total; i++) {
    float *pd = (float*)cvGetSeqElem(r, i);
    int count = r->elem_size / sizeof(float);
    PyObject *oi = PyList_New(count);
    for (int j = 0; j < count; j++) {
      PyList_SetItem(oi, j, PyFloat_FromDouble(pd[j]));
    }
    PyList_SetItem(pr, i, oi);
  }
  // This function has copied the CvSeq data into a list.  Hence the
  // CvSeq is not being returned to the caller.  Hence, no reference
  // count increase for the storage, unlike _FROM_CvSeqPTR.
  return pr;
}

typedef CvPoint2D32f CvPoint2D32f_4[4];
static PyObject *FROM_CvPoint2D32f_4(CvPoint2D32f* r)
{
  return Py_BuildValue("(ff)(ff)(ff)(ff)",
                       r[0].x, r[0].y,
                       r[1].x, r[1].y,
                       r[2].x, r[2].y,
                       r[3].x, r[3].y);
}

typedef float CvMatr32f_i[9];

static PyObject *FROM_CvMatr32f_i(CvMatr32f_i r)
{
  return Py_BuildValue("(fff)(fff)(fff)",
    r[0], r[1], r[2],
    r[3], r[4], r[5],
    r[6], r[7], r[8]);
}

typedef float CvVect32f_i[3];
static PyObject *FROM_CvVect32f_i(CvVect32f_i r)
{
  return Py_BuildValue("fff",
    r[0], r[1], r[2]);
}

static PyObject *FROM_CvFont(CvFont r)
{
  cvfont_t *cf = PyObject_NEW(cvfont_t, &cvfont_Type);
  cf->a = r;
  return (PyObject*)cf;
}

static PyObject *FROM_CvSubdiv2DPointPTR(CvSubdiv2DPoint* r)
{
  if (r != NULL) {
    cvsubdiv2dpoint_t *cf = PyObject_NEW(cvsubdiv2dpoint_t, &cvsubdiv2dpoint_Type);
    cf->a = r;
    return (PyObject*)cf;
  } else {
    Py_INCREF(Py_None);
    return Py_None;
  }
}

static PyObject *FROM_IplImagePTR(IplImage *r)
{
  iplimage_t *cva = PyObject_NEW(iplimage_t, &iplimage_Type);
  cva->a = r;
  return pythonize_IplImage(cva);
}

static PyObject *FROM_ROIplImagePTR(ROIplImage *r)
{
  if (r != NULL) {
    iplimage_t *cva = PyObject_NEW(iplimage_t, &iplimage_Type);
    cva->a = cvCreateImageHeader(cvSize(100,100), 8, 1);
    *(cva->a) = *r;
    cva->data = PyBuffer_FromReadWriteMemory(r->imageData, r->height * r->widthStep);
    cva->offset = 0;
    return (PyObject*)cva;
  } else {
    Py_RETURN_NONE;
  }
}

static PyObject *FROM_ROCvMatPTR(ROCvMat *r)
{
  if (r != NULL) {
    cvmat_t *cva = PyObject_NEW(cvmat_t, &cvmat_Type);
    cva->a = cvCreateMatHeader(100, 100, CV_8U);
    *(cva->a) = *r;
    cva->data = PyBuffer_FromReadWriteMemory(r->data.ptr, r->rows * r->step);
    cva->offset = 0;
    return (PyObject*)cva;
  } else {
    Py_RETURN_NONE;
  }
}

static PyObject *FROM_CvMatPTR(CvMat *r)
{
  cvmat_t *cvm = PyObject_NEW(cvmat_t, &cvmat_Type);
  cvm->a = r;

  return pythonize_CvMat(cvm);
}

static PyObject *FROM_CvMat(CvMat *r)
{
  cvmat_t *m = PyObject_NEW(cvmat_t, &cvmat_Type);
  m->a = r;
  return pythonize_CvMat(m);
}

static PyObject *FROM_CvMatNDPTR(CvMatND *r)
{
  cvmatnd_t *m = PyObject_NEW(cvmatnd_t, &cvmatnd_Type);
  m->a = r;
  return pythonize_CvMatND(m);
}

static PyObject *FROM_CvRNG(CvRNG r)
{
  cvrng_t *m = PyObject_NEW(cvrng_t, &cvrng_Type);
  m->a = r;
  return (PyObject*)m;
}

/*static PyObject *FROM_CvContourTreePTR(CvContourTree *r)
{
  cvcontourtree_t *m = PyObject_NEW(cvcontourtree_t, &cvcontourtree_Type);
  m->a = r;
  return (PyObject*)m;
}*/

static PyObject *FROM_generic(generic r)
{
  if (r == NULL) {
    failmsg("OpenCV returned NULL");
    return NULL;
  }
  CvTypeInfo* t = cvTypeOf(r);
  if (strcmp(t->type_name, "opencv-image") == 0)
    return FROM_IplImagePTR((IplImage*)r);
  else if (strcmp(t->type_name, "opencv-matrix") == 0)
    return FROM_CvMat((CvMat*)r);
  else if (strcmp(t->type_name, "opencv-nd-matrix") == 0)
    return FROM_CvMatNDPTR((CvMatND*)r);
  else if (strcmp(t->type_name, "opencv-haar-classifier") == 0)
    return FROM_CvHaarClassifierCascadePTR((CvHaarClassifierCascade*)r);
  else {
    failmsg("Unknown OpenCV type '%s'", t->type_name);
    return NULL;
  }
}

static PyObject *FROM_CvSubdiv2DEdge(CvSubdiv2DEdge r)
{
  cvsubdiv2dedge_t *m = PyObject_NEW(cvsubdiv2dedge_t, &cvsubdiv2dedge_Type);
  m->a = r;
  m->container = Py_None; // XXX
  Py_INCREF(m->container);
  return (PyObject*)m;
}

static PyObject *FROM_CvPoints(CvPoints src)
{
  PyObject *pr;
  pr = PyList_New(src.count);
  for (int i = 0; i < src.count; i++) {
    PyList_SetItem(pr, i, FROM_CvPoint(src.p[i]));
  }
  return pr;
}

/************************************************************************/

/* A few functions are too odd to be generated, 
 * so are handwritten here */

static PyObject *pycvWaitKey(PyObject *self, PyObject *args, PyObject *kw)
{
  int delay = 0;

  const char *keywords[] = { "delay", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "|i", (char**)keywords, &delay))
    return NULL;
  int r;
  Py_BEGIN_ALLOW_THREADS
  r = cvWaitKey(delay);
  Py_END_ALLOW_THREADS
  return FROM_int(r);
}

static PyObject *pycvLoadImage(PyObject *self, PyObject *args, PyObject *kw)
{
  const char *keywords[] = { "filename", "iscolor", NULL };
  char *filename;
  int iscolor = CV_LOAD_IMAGE_COLOR;

  if (!PyArg_ParseTupleAndKeywords(args, kw, "s|i", (char**)keywords, &filename, &iscolor))
    return NULL;

  // Inside ALLOW_THREADS, must not reference 'filename' because it might move.
  // So make a local copy 'filename_copy'.
  char filename_copy[2048];
  strncpy(filename_copy, filename, sizeof(filename_copy));

  IplImage *r;
  Py_BEGIN_ALLOW_THREADS
  r = cvLoadImage(filename_copy, iscolor);
  Py_END_ALLOW_THREADS

  if (r == NULL) {
    PyErr_SetFromErrnoWithFilename(PyExc_IOError, filename);
    return NULL;
  } else {
    return FROM_IplImagePTR(r);
  }
}

static PyObject *pycvLoadImageM(PyObject *self, PyObject *args, PyObject *kw)
{
  const char *keywords[] = { "filename", "iscolor", NULL };
  char *filename;
  int iscolor = CV_LOAD_IMAGE_COLOR;

  if (!PyArg_ParseTupleAndKeywords(args, kw, "s|i", (char**)keywords, &filename, &iscolor))
    return NULL;

  // Inside ALLOW_THREADS, must not reference 'filename' because it might move.
  // So make a local copy 'filename_copy'.
  char filename_copy[2048];
  strncpy(filename_copy, filename, sizeof(filename_copy));

  CvMat *r;
  Py_BEGIN_ALLOW_THREADS
  r = cvLoadImageM(filename_copy, iscolor);
  Py_END_ALLOW_THREADS

  if (r == NULL) {
    PyErr_SetFromErrnoWithFilename(PyExc_IOError, filename);
    return NULL;
  } else {
    return FROM_CvMatPTR(r);
  }
}

static PyObject *pycvCreateImageHeader(PyObject *self, PyObject *args)
{
  int w, h, depth, channels;
  if (!PyArg_ParseTuple(args, "(ii)Ii", &w, &h, &depth, &channels))
    return NULL;
  iplimage_t *cva = PyObject_NEW(iplimage_t, &iplimage_Type);
  cva->a = cvCreateImageHeader(cvSize(w, h), depth, channels);
  if (cva->a == NULL) {
    PyErr_SetString(PyExc_TypeError, "CreateImage failed");
    return NULL;
  } else {
    cva->data = Py_None;
    Py_INCREF(cva->data);
    cva->offset = 0;

    return (PyObject*)cva;
  }
}

static PyObject *pycvCreateImage(PyObject *self, PyObject *args)
{
  int w, h, depth, channels;
  if (!PyArg_ParseTuple(args, "(ii)Ii:CreateImage", &w, &h, &depth, &channels))
    return NULL;
  iplimage_t *cva = PyObject_NEW(iplimage_t, &iplimage_Type);
  ERRWRAP(cva->a = cvCreateImage(cvSize(w, h), depth, channels));
  if (cva->a == NULL) {
    PyErr_SetString(PyExc_TypeError, "CreateImage failed");
    return NULL;
  } else {
    return pythonize_IplImage(cva);
  }
}

static PyObject *pycvCreateMatHeader(PyObject *self, PyObject *args)
{
  int rows, cols, type;
  if (!PyArg_ParseTuple(args, "iii", &rows, &cols, &type))
    return NULL;
  cvmat_t *m = PyObject_NEW(cvmat_t, &cvmat_Type);
  ERRWRAP(m->a = cvCreateMatHeader(rows, cols, type));
  if (m->a == NULL) {
    PyErr_SetString(PyExc_TypeError, "CreateMat failed");
    return NULL;
  } else {
    m->data = Py_None;
    Py_INCREF(m->data);
    m->offset = 0;
    return (PyObject*)m;
  }
}

static PyObject *pycvCreateMat(PyObject *self, PyObject *args)
{
  int rows, cols, type;
  if (!PyArg_ParseTuple(args, "iii", &rows, &cols, &type))
    return NULL;
  cvmat_t *m = PyObject_NEW(cvmat_t, &cvmat_Type);
  ERRWRAP(m->a = cvCreateMat(rows, cols, type));
  if (m->a == NULL) {
    PyErr_SetString(PyExc_TypeError, "CreateMat failed");
    return NULL;
  } else {
    return pythonize_CvMat(m);
  }
}

static PyObject *pycvCreateMatNDHeader(PyObject *self, PyObject *args)
{
  ints dims;
  int type;

  if (!PyArg_ParseTuple(args, "O&i", convert_to_ints, (void*)&dims, &type))
    return NULL;
  cvmatnd_t *m = PyObject_NEW(cvmatnd_t, &cvmatnd_Type);
  ERRWRAP(m->a = cvCreateMatNDHeader(dims.count, dims.i, type));

  m->data = Py_None;
  Py_INCREF(m->data);
  delete [] dims.i;
  return (PyObject*)m;
}


static PyObject *pycvCreateMatND(PyObject *self, PyObject *args)
{
  ints dims;
  int type;

  if (!PyArg_ParseTuple(args, "O&i", convert_to_ints, (void*)&dims, &type))
    return NULL;
  cvmatnd_t *m = PyObject_NEW(cvmatnd_t, &cvmatnd_Type);
  ERRWRAP(m->a = cvCreateMatND(dims.count, dims.i, type));
  delete [] dims.i;
  return pythonize_CvMatND(m);
}

#if PYTHON_USE_NUMPY
static PyObject *pycvfromarray(PyObject *self, PyObject *args, PyObject *kw)
{
  const char *keywords[] = { "arr", "allowND", NULL };
  PyObject *o;
  int allowND = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kw, "O|i", (char**)keywords, &o, &allowND))
    return NULL;
  return fromarray(o, allowND);
}

static PyObject *fromarray(PyObject *o, int allowND)
{
  PyObject *ao = PyObject_GetAttrString(o, "__array_struct__");
  PyObject *retval;

  if ((ao == NULL) || !PyCObject_Check(ao)) {
    PyErr_SetString(PyExc_TypeError, "object does not have array interface");
    return NULL;
  }
  PyArrayInterface *pai = (PyArrayInterface*)PyCObject_AsVoidPtr(ao);
  if (pai->two != 2) {
    PyErr_SetString(PyExc_TypeError, "object does not have array interface");
    return NULL;
  }

  int type = -1;

  switch (pai->typekind) {
  case 'i':
    if (pai->itemsize == 1)
      type = CV_8SC1;
    else if (pai->itemsize == 2)
      type = CV_16SC1;
    else if (pai->itemsize == 4)
      type = CV_32SC1;
    break;

  case 'u':
    if (pai->itemsize == 1)
      type = CV_8UC1;
    else if (pai->itemsize == 2)
      type = CV_16UC1;
    break;

  case 'f':
    if (pai->itemsize == 4)
      type = CV_32FC1;
    else if (pai->itemsize == 8)
      type = CV_64FC1;
    break;
    
  }
  if (type == -1) {
     PyErr_SetString(PyExc_TypeError, "the array type is not supported by OpenCV");
     return NULL;
  }

  if (!allowND) {
    cvmat_t *m = PyObject_NEW(cvmat_t, &cvmat_Type);
    if (pai->nd == 2) {
      if (pai->strides[1] != pai->itemsize) {
        return (PyObject*)failmsg("cv.fromarray array can only accept arrays with contiguous data");
      }
      ERRWRAP(m->a = cvCreateMatHeader(pai->shape[0], pai->shape[1], type));
      m->a->step = pai->strides[0];
    } else if (pai->nd == 3) {
      if (pai->shape[2] > CV_CN_MAX)
        return (PyObject*)failmsg("cv.fromarray too many channels, see allowND argument");
      ERRWRAP(m->a = cvCreateMatHeader(pai->shape[0], pai->shape[1], type + ((pai->shape[2] - 1) << CV_CN_SHIFT)));
      m->a->step = pai->strides[0];
    } else {
      return (PyObject*)failmsg("cv.fromarray array can be 2D or 3D only, see allowND argument");
    }
    m->a->data.ptr = (uchar*)pai->data;
    retval = pythonize_foreign_CvMat(m);
  } else {
    int dims[CV_MAX_DIM];
    int i;
    for (i = 0; i < pai->nd; i++)
      dims[i] = pai->shape[i];
    cvmatnd_t *m = PyObject_NEW(cvmatnd_t, &cvmatnd_Type);
    ERRWRAP(m->a = cvCreateMatND(pai->nd, dims, type));
    m->a->data.ptr = (uchar*)pai->data;
    
    retval = pythonize_CvMatND(m, ao);
  }
  Py_DECREF(ao);
  return retval;
}
#endif

class ranges {
public:
  Py_ssize_t len;
  float **rr;
  ranges() {
    len = 0;
    rr = NULL;
  }
  int fromobj(PyObject *o, const char *name = "no_name") {
    PyObject *fi = PySequence_Fast(o, name);
    if (fi == NULL)
      return 0;
    len = PySequence_Fast_GET_SIZE(fi);
    rr = new float*[len];
    for (Py_ssize_t i = 0; i < len; i++) {
      PyObject *item = PySequence_Fast_GET_ITEM(fi, i);
      floats ff;
      if (!convert_to_floats(item, &ff))
        return 0;
      rr[i] = ff.f;
    }
    Py_DECREF(fi);
    return 1;
  }
  ~ranges() {
    for (Py_ssize_t i = 0; i < len; i++)
      delete rr[i];
   delete rr;
  }
};

static int ranges_converter(PyObject *o, ranges* dst)
{
  return dst->fromobj(o);
}

static PyObject *pycvCreateHist(PyObject *self, PyObject *args, PyObject *kw)
{
  const char *keywords[] = { "dims", "type", "ranges", "uniform", NULL };
  PyObject *dims;
  int type;
  int uniform = 1;
  ranges r;
  if (!PyArg_ParseTupleAndKeywords(args, kw, "Oi|O&i", (char**)keywords, &dims, &type, ranges_converter, (void*)&r, &uniform)) {
    return NULL;
  }
  cvhistogram_t *h = PyObject_NEW(cvhistogram_t, &cvhistogram_Type);
  args = Py_BuildValue("Oi", dims, CV_32FC1);
  memset(&h->h, 0, sizeof(h->h));
  h->bins = pycvCreateMatND(self, args);
  Py_DECREF(args);
  if (h->bins == NULL) {
    return NULL;
  }
  h->h.type = CV_HIST_MAGIC_VAL + CV_HIST_UNIFORM_FLAG;
  if (!convert_to_CvArr(h->bins, &(h->h.bins), "bins"))
    return NULL;

  if(r.rr)
  {
      ERRWRAP(cvSetHistBinRanges(&(h->h), r.rr, uniform));
  }

  return (PyObject*)h;
}

static PyObject *pycvInitLineIterator(PyObject *self, PyObject *args, PyObject *kw)
{
  const char *keywords[] = { "image", "pt1", "pt2", "connectivity", "left_to_right", NULL };
  CvArr *image;
  CvPoint pt1;
  CvPoint pt2;
  int connectivity = 8;
  int left_to_right = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kw, "O&O&O&|ii", (char**)keywords,
                        convert_to_CvArr, &image,
                        convert_to_CvPoint, &pt1,
                        convert_to_CvPoint, &pt2,
                        &connectivity,
                        &left_to_right))
    return NULL;

  cvlineiterator_t *pi = PyObject_NEW(cvlineiterator_t, &cvlineiterator_Type);
  pi->count = cvInitLineIterator(image, pt1, pt2, &pi->iter, connectivity, left_to_right);
  ERRWRAP(pi->type = cvGetElemType(image));
  return (PyObject*)pi;
}

static PyObject *pycvCreateMemStorage(PyObject *self, PyObject *args)
{
  int block_size = 0;
  if (!PyArg_ParseTuple(args, "|i", &block_size))
    return NULL;
  cvmemstorage_t *pm = PyObject_NEW(cvmemstorage_t, &cvmemstorage_Type);
  pm->a = cvCreateMemStorage(block_size);
  return (PyObject*)pm;
}

// single index: return row
// 2 indices: row, column
// both row and column can be slices.  column slice must have a step of 1.
//
// returns a scalar when all dimensions are specified and all are integers.  Otherwise returns a CvMat.
//
static PyObject *cvarr_GetItem(PyObject *o, PyObject *key)
{
  dims dd;

  CvArr *cva;
  if (!convert_to_CvArr(o, &cva, "src"))
    return NULL;

  if (!convert_to_dims(key, &dd, cva, "key")) {
    return NULL;
  }

  // Figure out if all supplied indices have a stride of zero - means they are not slices
  // and if all indices are positive
  int all0 = 1;
  for (int i = 0; i < dd.count; i++) {
    all0 &= (dd.step[i] == 0) && (0 <= dd.i[i]);
  }

  // if every dimension supplied, and none are slices, return the scalar
  if ((cvGetDims(cva) == dd.count) && all0) {
    CvScalar s;
    ERRWRAP(s = cvGetND(cva, dd.i));
    return PyObject_FromCvScalar(s, cvGetElemType(cva));
  } else {
    // pad missing dimensions
    for (int i = dd.count; i < cvGetDims(cva); i++) {
      dd.i[i] = 0;
      dd.step[i] = 1;
      dd.length[i] = cvGetDimSize(cva, i);
    }
    dd.count = cvGetDims(cva);

    // negative steps are illegal for OpenCV
    for (int i = 0; i < dd.count; i++) {
      if (dd.step[i] < 0)
        return (PyObject*)failmsg("Negative step is illegal");
    }

    // zero length illegal for OpenCV
    for (int i = 0; i < dd.count; i++) {
      if (dd.length[i] == 0)
        return (PyObject*)failmsg("Zero sized dimension is illegal");
    }

    // column step can only be 0 or 1
    if ((dd.step[dd.count-1] != 0) && (dd.step[dd.count-1] != 1))
        return (PyObject*)failmsg("Column step is illegal");

    if (is_cvmat(o) || is_iplimage(o)) {
      cvmat_t *sub = PyObject_NEW(cvmat_t, &cvmat_Type);
      sub->a = cvCreateMatHeader(dd.length[0], dd.length[1], cvGetElemType(cva));
      uchar *old0;  // pointer to first element in old mat
      int oldstep;
      cvGetRawData(cva, &old0, &oldstep);
      uchar *new0;  // pointer to first element in new mat
      ERRWRAP(new0 = cvPtrND(cva, dd.i));

      sub->a->step = oldstep * dd.step[0];
      sub->data = what_data(o);
      Py_INCREF(sub->data);
      sub->offset = new0 - old0;
      return (PyObject*)sub;
    } else {
      cvmatnd_t *sub = PyObject_NEW(cvmatnd_t, &cvmatnd_Type);
      sub->a = cvCreateMatNDHeader(dd.count, dd.length, cvGetElemType(cva));
      uchar *old0;  // pointer to first element in old mat
      cvGetRawData(cva, &old0);
      uchar *new0;  // pointer to first element in new mat
      ERRWRAP(new0 = cvPtrND(cva, dd.i));

      for (int d = 0; d < dd.count; d++) {
        int stp = dd.step[d];
        sub->a->dim[d].step = ((CvMatND*)cva)->dim[d].step * ((stp == 0) ? 1 : stp);
        sub->a->dim[d].size = dd.length[d];
      }
      sub->data = what_data(o);
      Py_INCREF(sub->data);
      sub->offset = new0 - old0;
      return (PyObject*)sub;
    }
  }
}

static int cvarr_SetItem(PyObject *o, PyObject *key, PyObject *v)
{
  dims dd;

  CvArr *cva;
  if (!convert_to_CvArr(o, &cva, "src"))
    return -1;

  if (!convert_to_dims(key, &dd, cva, "key")) {
    return -1;
  }

  if (cvGetDims(cva) != dd.count) {
    PyErr_SetString(PyExc_TypeError, "key length does not match array dimension");
    return -1;
  }

  CvScalar s;
  if (PySequence_Check(v)) {
    PyObject *fi = PySequence_Fast(v, "v");
    if (fi == NULL)
      return -1;
    if (PySequence_Fast_GET_SIZE(fi) != CV_MAT_CN(cvGetElemType(cva))) {
      PyErr_SetString(PyExc_TypeError, "sequence size must be same as channel count");
      return -1;
    }
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fi); i++)
      s.val[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(fi, i));
    Py_DECREF(fi);
  } else {
    if (1 != CV_MAT_CN(cvGetElemType(cva))) {
      PyErr_SetString(PyExc_TypeError, "scalar supplied but channel count does not equal 1");
      return -1;
    }
    s.val[0] = PyFloat_AsDouble(v);
  }
  switch (dd.count) {
  case 1:
    ERRWRAPN(cvSet1D(cva, dd.i[0], s), -1);
    break;
  case 2:
    ERRWRAPN(cvSet2D(cva, dd.i[0], dd.i[1], s), -1);
    break;
  case 3:
    ERRWRAPN(cvSet3D(cva, dd.i[0], dd.i[1], dd.i[2], s), -1);
    break;
  default:
    ERRWRAPN(cvSetND(cva, dd.i, s), -1);
    // XXX - OpenCV bug? - seems as if an error in cvSetND does not set error status?
    break;
  }
  if (cvGetErrStatus() != 0) {
    translate_error_to_exception();
    return -1;
  }

  return 0;
}


static PyObject *pycvSetData(PyObject *self, PyObject *args)
{
  PyObject *o, *s;
  int step = CV_AUTO_STEP;

  if (!PyArg_ParseTuple(args, "OO|i", &o, &s, &step))
    return NULL;
  if (is_iplimage(o)) {
    iplimage_t *ipl = (iplimage_t*)o;
    ipl->a->widthStep = step;
    Py_DECREF(ipl->data);
    ipl->data = s;
    Py_INCREF(ipl->data);
  } else if (is_cvmat(o)) {
    cvmat_t *m = (cvmat_t*)o;
    m->a->step = step;
    Py_DECREF(m->data);
    m->data = s;
    Py_INCREF(m->data);
  } else if (is_cvmatnd(o)) {
    cvmatnd_t *m = (cvmatnd_t*)o;
    Py_DECREF(m->data);
    m->data = s;
    Py_INCREF(m->data);
  } else {
    PyErr_SetString(PyExc_TypeError, "SetData argument must be either IplImage, CvMat or CvMatND");
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyObject *what_data(PyObject *o)
{
  if (is_iplimage(o)) {
    iplimage_t *ipl = (iplimage_t*)o;
    return ipl->data;
  } else if (is_cvmat(o)) {
    cvmat_t *m = (cvmat_t*)o;
    return m->data;
  } else if (is_cvmatnd(o)) {
    cvmatnd_t *m = (cvmatnd_t*)o;
    return m->data;
  } else {
    assert(0);
    return NULL;
  }
}

static PyObject *pycvCreateData(PyObject *self, PyObject *args)
{
  PyObject *o;

  if (!PyArg_ParseTuple(args, "O", &o))
    return NULL;

  CvArr *a;
  if (!convert_to_CvArr(o, &a, "arr"))
    return NULL;
  ERRWRAP(cvCreateData(a));

  Py_DECREF(what_data(o));
  if (is_iplimage(o)) {
    iplimage_t *ipl = (iplimage_t*)o;
    pythonize_IplImage(ipl);
  } else if (is_cvmat(o)) {
    cvmat_t *m = (cvmat_t*)o;
    pythonize_CvMat(m);
  } else if (is_cvmatnd(o)) {
    cvmatnd_t *m = (cvmatnd_t*)o;
    pythonize_CvMatND(m);
  } else {
    PyErr_SetString(PyExc_TypeError, "CreateData argument must be either IplImage, CvMat or CvMatND");
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyObject *pycvGetDims(PyObject *self, PyObject *args)
{
  PyObject *o;

  if (!PyArg_ParseTuple(args, "O", &o))
    return NULL;
  CvArr *cva;
  if (!convert_to_CvArr(o, &cva, "src"))
    return NULL;

  int i, nd;
  ERRWRAP(nd = cvGetDims(cva));
  PyObject *r = PyTuple_New(nd);
  for (i = 0; i < nd; i++)
    PyTuple_SetItem(r, i, PyInt_FromLong(cvGetDimSize(cva, i)));
  return r;
}

static PyObject *pycvGetImage(PyObject *self, PyObject *args)
{
  PyObject *o, *r;

  if (!PyArg_ParseTuple(args, "O", &o))
    return NULL;
  if (is_iplimage(o)) {
    r = o;
    Py_INCREF(o);
  } else {
    IplImage *ipl = cvCreateImageHeader(cvSize(100,100), 8, 1); // these args do not matter, because overwritten
    CvArr *cva;
    if (!convert_to_CvArr(o, &cva, "src"))
      return NULL;
    ERRWRAP(cvGetImage(cva, ipl));

    iplimage_t *oipl = PyObject_NEW(iplimage_t, &iplimage_Type);
    oipl->a = ipl;
    oipl->data = what_data(o);
    Py_INCREF(oipl->data);
    oipl->offset = 0;

    r = (PyObject*)oipl;
  }
  return r;
}

static PyObject *pycvGetMat(PyObject *self, PyObject *args, PyObject *kw)
{
  const char *keywords[] = { "arr", "allowND", NULL };
  PyObject *o, *r;
  int allowND = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kw, "O|i", (char**)keywords, &o, &allowND))
    return NULL;
  if (is_cvmat(o)) {
    r = o;
    Py_INCREF(o);
  } else {
    CvMat *m = cvCreateMatHeader(100,100, 1); // these args do not matter, because overwritten
    CvArr *cva;
    if (!convert_to_CvArr(o, &cva, "src"))
      return NULL;
    ERRWRAP(cvGetMat(cva, m, NULL, allowND));

    cvmat_t *om = PyObject_NEW(cvmat_t, &cvmat_Type);
    om->a = m;
    om->data = what_data(o);
    Py_INCREF(om->data);
    om->offset = 0;

    r = (PyObject*)om;
  }
  return r;
}

static PyObject *pycvReshape(PyObject *self, PyObject *args)
{
  PyObject *o;
  int new_cn;
  int new_rows = 0;

  if (!PyArg_ParseTuple(args, "Oi|i", &o, &new_cn, &new_rows))
    return NULL;

  CvMat *m = cvCreateMatHeader(100,100, 1); // these args do not matter, because overwritten
  CvArr *cva;
  if (!convert_to_CvArr(o, &cva, "src"))
    return NULL;
  ERRWRAP(cvReshape(cva, m, new_cn, new_rows));

  cvmat_t *om = PyObject_NEW(cvmat_t, &cvmat_Type);
  om->a = m;
  om->data = what_data(o);
  Py_INCREF(om->data);
  om->offset = 0;

  return (PyObject*)om;
}

static PyObject *pycvReshapeMatND(PyObject *self, PyObject *args)
{
  PyObject *o;
  int new_cn = 0;
  PyObject *new_dims = NULL;

  if (!PyArg_ParseTuple(args, "OiO", &o, &new_cn, &new_dims))
    return NULL;

  CvMatND *cva;
  if (!convert_to_CvMatND(o, &cva, "src"))
    return NULL;
  ints dims;
  if (new_dims != NULL) {
    if (!convert_to_ints(new_dims, &dims, "new_dims"))
      return NULL;
  }

  if (new_cn == 0)
    new_cn = CV_MAT_CN(cvGetElemType(cva));

  int i;
  int count = CV_MAT_CN(cvGetElemType(cva));
  for (i = 0; i < cva->dims; i++)
    count *= cva->dim[i].size;

  int newcount = new_cn;
  for (i = 0; i < dims.count; i++)
    newcount *= dims.i[i];

  if (count != newcount) {
    PyErr_SetString(PyExc_TypeError, "Total number of elements must be unchanged");
    return NULL;
  }

  CvMatND *pn = cvCreateMatNDHeader(dims.count, dims.i, CV_MAKETYPE(CV_MAT_TYPE(cva->type), new_cn));
  return shareDataND(o, cva, pn);
}

static void OnMouse(int event, int x, int y, int flags, void* param)
{
  PyGILState_STATE gstate;
  gstate = PyGILState_Ensure();

  PyObject *o = (PyObject*)param;
  PyObject *args = Py_BuildValue("iiiiO", event, x, y, flags, PyTuple_GetItem(o, 1));

  PyObject *r = PyObject_Call(PyTuple_GetItem(o, 0), args, NULL);
  if (r == NULL)
    PyErr_Print();
  else
    Py_DECREF(r);
  Py_DECREF(args);
  PyGILState_Release(gstate);
}

static PyObject *pycvSetMouseCallback(PyObject *self, PyObject *args, PyObject *kw)
{
  const char *keywords[] = { "window_name", "on_mouse", "param", NULL };
  char* name;
  PyObject *on_mouse;
  PyObject *param = NULL;

  if (!PyArg_ParseTupleAndKeywords(args, kw, "sO|O", (char**)keywords, &name, &on_mouse, &param))
    return NULL;
  if (!PyCallable_Check(on_mouse)) {
    PyErr_SetString(PyExc_TypeError, "on_mouse must be callable");
    return NULL;
  }
  if (param == NULL) {
    param = Py_None;
  }
  ERRWRAP(cvSetMouseCallback(name, OnMouse, Py_BuildValue("OO", on_mouse, param)));
  Py_RETURN_NONE;
}

void OnChange(int pos, void *param)
{
  PyGILState_STATE gstate;
  gstate = PyGILState_Ensure();

  PyObject *o = (PyObject*)param;
  PyObject *args = Py_BuildValue("(i)", pos);
  PyObject *r = PyObject_Call(PyTuple_GetItem(o, 0), args, NULL);
  if (r == NULL)
    PyErr_Print();
  Py_DECREF(args);
  PyGILState_Release(gstate);
}

static PyObject *pycvCreateTrackbar(PyObject *self, PyObject *args)
{
  PyObject *on_change;
  char* trackbar_name;
  char* window_name;
  int *value = new int;
  int count;

  if (!PyArg_ParseTuple(args, "ssiiO", &trackbar_name, &window_name, value, &count, &on_change))
    return NULL;
  if (!PyCallable_Check(on_change)) {
    PyErr_SetString(PyExc_TypeError, "on_change must be callable");
    return NULL;
  }
  ERRWRAP(cvCreateTrackbar2(trackbar_name, window_name, value, count, OnChange, Py_BuildValue("OO", on_change, Py_None)));
  Py_RETURN_NONE;
}

static PyObject *pycvFindContours(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* image;
  PyObject *pyobj_image = NULL;
  CvMemStorage* storage;
  PyObject *pyobj_storage = NULL;
  CvSeq* first_contour;
  int header_size = sizeof(CvContour);
  int mode = CV_RETR_LIST;
  int method = CV_CHAIN_APPROX_SIMPLE;
  CvPoint offset = cvPoint(0,0);
  PyObject *pyobj_offset = NULL;

  const char *keywords[] = { "image", "storage", "mode", "method", "offset", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|iiO", (char**)keywords, &pyobj_image, &pyobj_storage, &mode, &method, &pyobj_offset))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvMemStorage(pyobj_storage, &storage, "storage")) return NULL;
  if ((pyobj_offset != NULL) && !convert_to_CvPoint(pyobj_offset, &offset, "offset")) return NULL;
  ERRWRAP(cvFindContours(image, storage, &first_contour, header_size, mode, method, offset));
  cvseq_t *ps = PyObject_NEW(cvseq_t, &cvseq_Type);
  ps->a = first_contour;
  ps->container = PyTuple_GetItem(args, 1); // storage
  Py_INCREF(ps->container);
  return (PyObject*)ps;
}

static PyObject *pycvApproxPoly(PyObject *self, PyObject *args, PyObject *kw)
{
  cvarrseq src_seq;
  PyObject *pyobj_src_seq = NULL;
  int header_size = sizeof(CvContour);
  CvMemStorage* storage;
  PyObject *pyobj_storage = NULL;
  int method;
  double parameter = 0;
  int parameter2 = 0;

  const char *keywords[] = { "src_seq", "storage", "method", "parameter", "parameter2", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOi|di", (char**)keywords, &pyobj_src_seq, &pyobj_storage, &method, &parameter, &parameter2))
    return NULL;
  if (!convert_to_cvarrseq(pyobj_src_seq, &src_seq, "src_seq")) return NULL;
  if (!convert_to_CvMemStorage(pyobj_storage, &storage, "storage")) return NULL;
  CvSeq* r;
  ERRWRAP(r = cvApproxPoly(src_seq.mat, header_size, storage, method, parameter, parameter2));
  return FROM_CvSeqPTR(r);
}

static float distance_function_glue( const float* a, const float* b, void* user_param )
{
  PyObject *o = (PyObject*)user_param;
  PyObject *args = Py_BuildValue("(ff)(ff)O", a[0], a[1], b[0], b[1], PyTuple_GetItem(o, 1));
  PyObject *r = PyObject_Call(PyTuple_GetItem(o, 0), args, NULL);
  Py_DECREF(args);
  return (float)PyFloat_AsDouble(r);
}

static PyObject *pycvCalcEMD2(PyObject *self, PyObject *args, PyObject *kw)
{
  const char *keywords[] = { "signature1", "signature2", "distance_type", "distance_func", "cost_matrix", "flow", "lower_bound", "userdata", NULL };
  CvArr* signature1;
  PyObject *pyobj_signature1;
  CvArr* signature2;
  PyObject *pyobj_signature2;
  int distance_type;
  PyObject *distance_func = NULL;
  CvArr* cost_matrix=NULL;
  PyObject *pyobj_cost_matrix = NULL;
  CvArr* flow=NULL;
  PyObject *pyobj_flow = NULL;
  float lower_bound = 0.0;
  PyObject *userdata = NULL;

  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOi|OOOfO", (char**)keywords,
                                   &pyobj_signature1,
                                   &pyobj_signature2,
                                   &distance_type,
                                   &distance_func,
                                   &pyobj_cost_matrix,
                                   &pyobj_flow,
                                   &lower_bound,
                                   &userdata))
    return NULL;
  if (!convert_to_CvArr(pyobj_signature1, &signature1, "signature1")) return NULL;
  if (!convert_to_CvArr(pyobj_signature2, &signature2, "signature2")) return NULL;
  if (pyobj_cost_matrix && !convert_to_CvArr(pyobj_cost_matrix, &cost_matrix, "cost_matrix")) return NULL;
  if (pyobj_flow && !convert_to_CvArr(pyobj_flow, &flow, "flow")) return NULL;

  if (distance_func == NULL) {
    distance_func = Py_None;
  }
  if (userdata == NULL) {
    userdata = Py_None;
  }

  PyObject *ud = Py_BuildValue("OO", distance_func, userdata);
  float r;
  ERRWRAP(r = cvCalcEMD2(signature1, signature2, distance_type, distance_function_glue, cost_matrix, flow, &lower_bound, (void*)ud));
  Py_DECREF(ud);

  return PyFloat_FromDouble(r);
}

static PyObject *pycvSubdiv2DLocate(PyObject *self, PyObject *args)
{
  PyObject *pyobj_subdiv;
  PyObject *pyobj_pt;
  CvSubdiv2D *subdiv;
  CvPoint2D32f pt;
  CvSubdiv2DEdge edge;
  CvSubdiv2DPoint* vertex;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_subdiv, &pyobj_pt))
    return NULL;
  if (!convert_to_CvSubdiv2DPTR(pyobj_subdiv, &subdiv, "subdiv"))
    return NULL;
  if (!convert_to_CvPoint2D32f(pyobj_pt, &pt, "pt"))
    return NULL;

  CvSubdiv2DPointLocation loc = cvSubdiv2DLocate(subdiv, pt, &edge, &vertex);
  PyObject *r;
  switch (loc) {
  case CV_PTLOC_INSIDE:
  case CV_PTLOC_ON_EDGE:
    r = FROM_CvSubdiv2DEdge(edge);
    break;
  case CV_PTLOC_VERTEX:
    r = FROM_CvSubdiv2DPointPTR(vertex);
    break;
  case CV_PTLOC_OUTSIDE_RECT:
    r = Py_None;
    Py_INCREF(Py_None);
    break;
  default:
    return (PyObject*)failmsg("Unexpected loc from cvSubdiv2DLocate");
  }
  return Py_BuildValue("iO", (int)loc, r);
}

static PyObject *pycvCalcOpticalFlowPyrLK(PyObject *self, PyObject *args)
{
  CvArr* prev;
  PyObject *pyobj_prev = NULL;
  CvArr* curr;
  PyObject *pyobj_curr = NULL;
  CvArr* prev_pyr;
  PyObject *pyobj_prev_pyr = NULL;
  CvArr* curr_pyr;
  PyObject *pyobj_curr_pyr = NULL;
  CvPoint2D32f* prev_features;
  PyObject *pyobj_prev_features = NULL;
  PyObject *pyobj_curr_features = NULL;
  CvPoint2D32f* curr_features;
  CvSize win_size;
  int level;
  CvTermCriteria criteria;
  int flags;

  if (!PyArg_ParseTuple(args, "OOOOO(ii)i(iif)i|O",
    &pyobj_prev, &pyobj_curr, &pyobj_prev_pyr, &pyobj_curr_pyr,
    &pyobj_prev_features,
    &win_size.width, &win_size.height, &level,
    &criteria.type, &criteria.max_iter, &criteria.epsilon,
    &flags,
    &pyobj_curr_features))
    return NULL;
  if (!convert_to_CvArr(pyobj_prev, &prev, "prev")) return NULL;
  if (!convert_to_CvArr(pyobj_curr, &curr, "curr")) return NULL;
  if (!convert_to_CvArr(pyobj_prev_pyr, &prev_pyr, "prev_pyr")) return NULL;
  if (!convert_to_CvArr(pyobj_curr_pyr, &curr_pyr, "curr_pyr")) return NULL;
  if (!convert_to_CvPoint2D32fPTR(pyobj_prev_features, &prev_features, "prev_features")) return NULL;
  int count = (int)PySequence_Length(pyobj_prev_features);
  if (flags & CV_LKFLOW_INITIAL_GUESSES) {
    failmsg("flag CV_LKFLOW_INITIAL_GUESSES is determined automatically from function arguments - it is not required");
    return NULL;
  }
  if (!pyobj_curr_features) {
    curr_features = new CvPoint2D32f[count];
  } else {
    if (PySequence_Length(pyobj_curr_features) != count) {
      failmsg("curr_features must have same length as prev_features");
      return NULL;
    }
    if (!convert_to_CvPoint2D32fPTR(pyobj_curr_features, &curr_features, "curr_features")) return NULL;
    flags |= CV_LKFLOW_INITIAL_GUESSES;
  }
  float *track_error = new float[count];
  char* status = new char[count];
  ERRWRAP(cvCalcOpticalFlowPyrLK(prev, curr, prev_pyr, curr_pyr, prev_features, curr_features, count, win_size, level, status, track_error, criteria, flags));

  cvpoint2d32f_count r0;
  r0.points = curr_features;
  r0.count = count;

  chars r1;
  r1.f = status;
  r1.count = count;

  floats r2;
  r2.f = track_error;
  r2.count = count;

  return Py_BuildValue("NNN", FROM_cvpoint2d32f_count(r0), FROM_chars(r1), FROM_floats(r2));
}

// pt1,pt2 are input and output arguments here

static PyObject *pycvClipLine(PyObject *self, PyObject *args)
{
  CvSize img_size;
  PyObject *pyobj_img_size = NULL;
  CvPoint pt1;
  PyObject *pyobj_pt1 = NULL;
  CvPoint pt2;
  PyObject *pyobj_pt2 = NULL;

  if (!PyArg_ParseTuple(args, "OOO", &pyobj_img_size, &pyobj_pt1, &pyobj_pt2))
    return NULL;
  if (!convert_to_CvSize(pyobj_img_size, &img_size, "img_size")) return NULL;
  if (!convert_to_CvPoint(pyobj_pt1, &pt1, "pt1")) return NULL;
  if (!convert_to_CvPoint(pyobj_pt2, &pt2, "pt2")) return NULL;
  int r;
  ERRWRAP(r = cvClipLine(img_size, &pt1, &pt2));
  if (r == 0) {
    Py_RETURN_NONE;
  } else {
    return Py_BuildValue("NN", FROM_CvPoint(pt1), FROM_CvPoint(pt2));
  }
}

static PyObject *pyfinddatamatrix(PyObject *self, PyObject *args)
{
  PyObject *pyim;
  if (!PyArg_ParseTuple(args, "O", &pyim))
    return NULL;

  CvMat *image;
  if (!convert_to_CvMat(pyim, &image, "image")) return NULL;

  std::deque <CvDataMatrixCode> codes;
  ERRWRAP(codes = cvFindDataMatrix(image));

  PyObject *pycodes = PyList_New(codes.size());
  for (size_t i = 0; i < codes.size(); i++) {
    CvDataMatrixCode *pc = &codes[i];
    PyList_SetItem(pycodes, i, Py_BuildValue("(sOO)", pc->msg, FROM_CvMat(pc->corners), FROM_CvMat(pc->original)));
  }

  return pycodes;
}

static PyObject *temp_test(PyObject *self, PyObject *args)
{
#if 0
  CvArr *im = cvLoadImage("../samples/c/lena.jpg", 0);
  printf("im=%p\n", im);
  CvMat *m = cvEncodeImage(".jpeg", im);
#endif
#if 0
  CvArr *im = cvLoadImage("lena.jpg", 0);
  float r0[] = { 0, 255 };
  float *ranges[] = { r0 };
  int hist_size[] = { 256 };
  CvHistogram *hist = cvCreateHist(1, hist_size, CV_HIST_ARRAY, ranges, 1);
  cvCalcHist(im, hist, 0, 0);
#endif

#if 0
  CvMat* mat = cvCreateMat( 3, 3, CV_32F );
  CvMat row_header, *row;
  row = cvReshape( mat, &row_header, 0, 1 );
  printf("%d,%d\n", row_header.rows, row_header.cols);
  printf("ge %08x\n", cvGetElemType(mat));
#endif

#if 0
  CvMat *m = cvCreateMat(1, 10, CV_8UC1);
  printf("CvMat stride ===> %d\n", m->step);
#endif

#if 0
  CvPoint2D32f src[3] = { { 0,0 }, { 1,0 }, { 0,1 } };
  CvPoint2D32f dst[3] = { { 0,0 }, { 17,0 }, { 0,17 } };

  CvMat* mapping = cvCreateMat(2, 3, CV_32FC1);
  cvGetAffineTransform(src, dst, mapping);
  printf("===> %f\n", cvGetReal2D(mapping, 0, 0));
#endif

#if 0
  CvArr *im = cvLoadImage("checker77.png");
  CvPoint2D32f corners[49];
  int count;
  cvFindChessboardCorners(im, cvSize(7,7), corners, &count, 0);
  printf("count=%d\n", count);
#endif

#if 0
  CvMat *src = cvCreateMat(512, 512, CV_8UC3);
  CvMat *dst = cvCreateMat(512, 512, CV_8UC3);
  cvPyrMeanShiftFiltering(src, dst, 5, 5);
  return FROM_CvMat(src);
#endif

  return PyFloat_FromDouble(0.0);
}

static PyObject *pycvFindChessboardCorners(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* image;
  PyObject *pyobj_image = NULL;
  CvSize pattern_size;
  PyObject *pyobj_pattern_size = NULL;
  cvpoint2d32f_count corners;
  int flags = CV_CALIB_CB_ADAPTIVE_THRESH;

  const char *keywords[] = { "image", "pattern_size", "flags", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|i", (char**)keywords, &pyobj_image, &pyobj_pattern_size, &flags))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvSize(pyobj_pattern_size, &pattern_size, "pattern_size")) return NULL;
  int r;
  corners.points = new CvPoint2D32f[pattern_size.width * pattern_size.height];
  ERRWRAP(r = cvFindChessboardCorners(image, pattern_size, corners.points,&corners.count, flags));
  return Py_BuildValue("NN", FROM_int(r), FROM_cvpoint2d32f_count(corners));
}

// For functions GetSubRect, GetRow, GetCol.
// recipient has a view into donor's data, and needs to share it.
// make recipient use the donor's data, compute the offset,
// and manage reference counts.

static void preShareData(CvArr *donor, CvMat **recipient)
{
  *recipient = cvCreateMatHeader(4, 4, cvGetElemType(donor));
}

static PyObject *shareData(PyObject *donor, CvArr *pdonor, CvMat *precipient)
{
  PyObject *recipient = (PyObject*)PyObject_NEW(cvmat_t, &cvmat_Type);
  ((cvmat_t*)recipient)->a = precipient;
  ((cvmat_t*)recipient)->offset = cvPtr1D(precipient, 0) - cvPtr1D(pdonor, 0);

  PyObject *arr_data;
  if (is_cvmat(donor)) {
    arr_data = ((cvmat_t*)donor)->data;
    ((cvmat_t*)recipient)->offset += ((cvmat_t*)donor)->offset;
  } else if (is_iplimage(donor)) {
    arr_data = ((iplimage_t*)donor)->data;
    ((cvmat_t*)recipient)->offset += ((iplimage_t*)donor)->offset;
  } else {
    return (PyObject*)failmsg("Argument 'mat' must be either IplImage or CvMat");
  }
  ((cvmat_t*)recipient)->data = arr_data;
  Py_INCREF(arr_data);
  return recipient;
}

static PyObject *shareDataND(PyObject *donor, CvMatND *pdonor, CvMatND *precipient)
{
  PyObject *recipient = (PyObject*)PyObject_NEW(cvmatnd_t, &cvmatnd_Type);
  ((cvmatnd_t*)recipient)->a = precipient;
  ((cvmatnd_t*)recipient)->offset = 0;

  PyObject *arr_data;
  arr_data = ((cvmatnd_t*)donor)->data;
  ((cvmatnd_t*)recipient)->data = arr_data;
  Py_INCREF(arr_data);
  return recipient;
}

static PyObject *pycvGetHuMoments(PyObject *self, PyObject *args)
{
  CvMoments* moments;
  PyObject *pyobj_moments = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_moments))
    return NULL;
  if (!convert_to_CvMomentsPTR(pyobj_moments, &moments, "moments")) return NULL;
  CvHuMoments r;
  ERRWRAP(cvGetHuMoments(moments, &r));
  return Py_BuildValue("ddddddd", r.hu1, r.hu2, r.hu3, r.hu4, r.hu5, r.hu6, r.hu7);
}

static PyObject *pycvFitLine(PyObject *self, PyObject *args)
{
  cvarrseq points;
  PyObject *pyobj_points = NULL;
  int dist_type;
  float param;
  float reps;
  float aeps;
  float r[6];

  if (!PyArg_ParseTuple(args, "Oifff", &pyobj_points, &dist_type, &param, &reps, &aeps))
    return NULL;
  if (!convert_to_cvarrseq(pyobj_points, &points, "points")) return NULL;
  ERRWRAP(cvFitLine(points.mat, dist_type, param, reps, aeps, r));
  int dimension;
  if (strcmp("opencv-matrix", cvTypeOf(points.mat)->type_name) == 0)
    dimension = CV_MAT_CN(cvGetElemType(points.mat));
  else {
    // sequence case... don't think there is a sequence of 3d points,
    // so assume 2D
    dimension = 2;
  }
  if (dimension == 2)
    return Py_BuildValue("dddd", r[0], r[1], r[2], r[3]);
  else
    return Py_BuildValue("dddddd", r[0], r[1], r[2], r[3], r[4], r[5]);
}

static PyObject *pycvGetMinMaxHistValue(PyObject *self, PyObject *args)
{
  CvHistogram* hist;
  PyObject *pyobj_hist = NULL;
  float min_val;
  float max_val;
  int min_loc[CV_MAX_DIM];
  int max_loc[CV_MAX_DIM];

  if (!PyArg_ParseTuple(args, "O", &pyobj_hist))
    return NULL;
  if (!convert_to_CvHistogram(pyobj_hist, &hist, "hist")) return NULL;
  ERRWRAP(cvGetMinMaxHistValue(hist, &min_val, &max_val, min_loc, max_loc));
  int d = cvGetDims(hist->bins);
  PyObject *pminloc = PyTuple_New(d), *pmaxloc = PyTuple_New(d);
  for (int i = 0; i < d; i++) {
    PyTuple_SetItem(pminloc, i, PyInt_FromLong(min_loc[i]));
    PyTuple_SetItem(pmaxloc, i, PyInt_FromLong(max_loc[i]));
  }
  return Py_BuildValue("ffNN", min_val, max_val, pminloc, pmaxloc);
}

static CvSeq* cvHOGDetectMultiScale( const CvArr* image, CvMemStorage* storage,
  const CvArr* svm_classifier=NULL, CvSize win_stride=cvSize(0,0),
  double hit_threshold=0, double scale=1.05,
  int group_threshold=2, CvSize padding=cvSize(0,0),
  CvSize win_size=cvSize(64,128), CvSize block_size=cvSize(16,16),
  CvSize block_stride=cvSize(8,8), CvSize cell_size=cvSize(8,8),
  int nbins=9, int gammaCorrection=1 )
{
    cv::HOGDescriptor hog(win_size, block_size, block_stride, cell_size, nbins, 1, -1, cv::HOGDescriptor::L2Hys, 0.2, gammaCorrection!=0);
    if(win_stride.width == 0 && win_stride.height == 0)
        win_stride = block_stride;
    cv::Mat img = cv::cvarrToMat(image);
    std::vector<cv::Rect> found;
    if(svm_classifier)
    {
        CvMat stub, *m = cvGetMat(svm_classifier, &stub);
        int sz = m->cols*m->rows;
        CV_Assert(CV_IS_MAT_CONT(m->type) && (m->cols == 1 || m->rows == 1) && CV_MAT_TYPE(m->type) == CV_32FC1);
        std::vector<float> w(sz);
        std::copy(m->data.fl, m->data.fl + sz, w.begin());
        hog.setSVMDetector(w);
    }
    else
        hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    hog.detectMultiScale(img, found, hit_threshold, win_stride, padding, scale, group_threshold);
    CvSeq* seq = cvCreateSeq(cv::DataType<cv::Rect>::type, sizeof(CvSeq), sizeof(cv::Rect), storage);
    if(found.size())
        cvSeqPushMulti(seq, &found[0], (int)found.size());
    return seq;
}

static void cvGrabCut(CvArr *image,
                      CvArr *mask,
                      CvRect rect,
                      CvArr *bgdModel,
                      CvArr *fgdModel,
                      int iterCount,
                      int mode)
{
  cv::Mat _image = cv::cvarrToMat(image);
  cv::Mat _mask = cv::cvarrToMat(mask);
  cv::Mat _bgdModel = cv::cvarrToMat(bgdModel);
  cv::Mat _fgdModel = cv::cvarrToMat(fgdModel);
  grabCut(_image, _mask, rect, _bgdModel, _fgdModel, iterCount, mode);
}

static int zero = 0;

/************************************************************************/
/* Custom Validators */

#define CVPY_VALIDATE_DrawChessboardCorners() do { \
  if ((patternSize.width * patternSize.height) != corners.count) \
    return (PyObject*)failmsg("Size is %dx%d, but corner list is length %d", patternSize.width, patternSize.height, corners.count); \
  } while (0)

#define cvGetRotationMatrix2D cv2DRotationMatrix

/************************************************************************/
/* Generated functions */

#define constCvMat const CvMat
#define FROM_constCvMatPTR(x) FROM_CvMatPTR((CvMat*)x)

#define cvSnakeImage(image, points, length, a, b, g, win, criteria, calc_gradient) \
  do { \
    int coeff_usage; \
    if ((alpha.count == 1) && (beta.count == 1) && (gamma.count == 1)) \
      coeff_usage = CV_VALUE; \
    else if ((length == alpha.count) && (alpha.count == beta.count) && (beta.count == gamma.count)) \
      coeff_usage = CV_ARRAY; \
    else \
      return (PyObject*)failmsg("SnakeImage weights invalid"); \
    cvSnakeImage(image, points, length, a, b, g, coeff_usage, win, criteria, calc_gradient); \
  } while (0)

static double cppKMeans(const CvArr* _samples, int cluster_count, CvArr* _labels,
           CvTermCriteria termcrit, int attempts, int flags, CvArr* _centers)
{
    cv::Mat data = cv::cvarrToMat(_samples), labels = cv::cvarrToMat(_labels), centers;
    if( _centers )
        centers = cv::cvarrToMat(_centers);
    CV_Assert( labels.isContinuous() && labels.type() == CV_32S &&
        (labels.cols == 1 || labels.rows == 1) &&
        labels.cols + labels.rows - 1 == data.rows );
    return cv::kmeans(data, cluster_count, labels, termcrit, attempts,
                        flags, _centers ? cv::_OutputArray(centers) : cv::_OutputArray() );
}

#define cvKMeans2(samples, nclusters, labels, termcrit, attempts, flags, centers) \
    cppKMeans(samples, nclusters, labels, termcrit, attempts, flags, centers)

#include "generated0.i"

static PyMethodDef methods[] = {

#if PYTHON_USE_NUMPY
    {"fromarray", (PyCFunction)pycvfromarray, METH_KEYWORDS, "fromarray(array) -> cvmatnd"},
#endif

  //{"CalcOpticalFlowFarneback", (PyCFunction)pycvCalcOpticalFlowFarneback, METH_KEYWORDS, "CalcOpticalFlowFarneback(prev, next, flow, pyr_scale=0.5, levels=3, win_size=15, iterations=3, poly_n=7, poly_sigma=1.5, flags=0) -> None"},
  //{"_HOGComputeDescriptors", (PyCFunction)pycvHOGComputeDescriptors, METH_KEYWORDS, "_HOGComputeDescriptors(image, win_stride=block_stride, locations=None, padding=(0,0), win_size=(64,128), block_size=(16,16), block_stride=(8,8), cell_size=(8,8), nbins=9, gammaCorrection=true) -> list_of_descriptors"},
  //{"_HOGDetect", (PyCFunction)pycvHOGDetect, METH_KEYWORDS, "_HOGDetect(image, svm_classifier, win_stride=block_stride, locations=None, padding=(0,0), win_size=(64,128), block_size=(16,16), block_stride=(8,8), cell_size=(8,8), nbins=9, gammaCorrection=true) -> list_of_points"},
  //{"_HOGDetectMultiScale", (PyCFunction)pycvHOGDetectMultiScale, METH_KEYWORDS, "_HOGDetectMultiScale(image, svm_classifier, win_stride=block_stride, scale=1.05, group_threshold=2, padding=(0,0), win_size=(64,128), block_size=(16,16), block_stride=(8,8), cell_size=(8,8), nbins=9, gammaCorrection=true) -> list_of_points"},

  {"FindDataMatrix", pyfinddatamatrix, METH_VARARGS},
  {"temp_test", temp_test, METH_VARARGS},

#include "generated1.i"

  {NULL, NULL},
};

/************************************************************************/
/* Module init */

static int to_ok(PyTypeObject *to)
{
  to->tp_alloc = PyType_GenericAlloc;
  to->tp_new = PyType_GenericNew;
  to->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  return (PyType_Ready(to) == 0);
}

#define MKTYPE(NAME)  NAME##_specials(); if (!to_ok(&NAME##_Type)) return

using namespace cv;

extern "C"
#if defined WIN32 || defined _WIN32
__declspec(dllexport)
#endif

void initcv()
{
#if PYTHON_USE_NUMPY
    import_array();
#endif
    
  PyObject *m, *d;

  cvSetErrMode(CV_ErrModeParent);

  MKTYPE(cvcontourtree);
  MKTYPE(cvfont);
  MKTYPE(cvhistogram);
  MKTYPE(cvlineiterator);
  MKTYPE(cvmat);
  MKTYPE(cvmatnd);
  MKTYPE(cvmemstorage);
  MKTYPE(cvsubdiv2dedge);
  MKTYPE(cvrng);
  MKTYPE(cvseq);
  MKTYPE(cvset);
  MKTYPE(cvsubdiv2d);
  MKTYPE(cvsubdiv2dpoint);
  MKTYPE(iplimage);
  MKTYPE(memtrack);

#include "generated4.i"

  m = Py_InitModule(MODULESTR"", methods);
  d = PyModule_GetDict(m);

  PyDict_SetItemString(d, "__version__", PyString_FromString("$Rev: 4557 $"));

  opencv_error = PyErr_NewException((char*)MODULESTR".error", NULL, NULL);
  PyDict_SetItemString(d, "error", opencv_error);

  // Couple of warnings about strict aliasing here.  Not clear how to fix.
  union {
    PyObject *o;
    PyTypeObject *to;
  } convert;
  convert.to = &iplimage_Type;
  PyDict_SetItemString(d, "iplimage", convert.o);
  convert.to = &cvmat_Type;
  PyDict_SetItemString(d, "cvmat", convert.o);

  // AFAIK the only floating-point constant
  PyDict_SetItemString(d, "CV_PI", PyFloat_FromDouble(CV_PI));

#define PUBLISH(I) PyDict_SetItemString(d, #I, PyInt_FromLong(I))
#define PUBLISHU(I) PyDict_SetItemString(d, #I, PyLong_FromUnsignedLong(I))
#define PUBLISH2(I, value) PyDict_SetItemString(d, #I, PyLong_FromLong(value))

  PUBLISHU(IPL_DEPTH_8U);
  PUBLISHU(IPL_DEPTH_8S);
  PUBLISHU(IPL_DEPTH_16U);
  PUBLISHU(IPL_DEPTH_16S);
  PUBLISHU(IPL_DEPTH_32S);
  PUBLISHU(IPL_DEPTH_32F);
  PUBLISHU(IPL_DEPTH_64F);

  PUBLISH(CV_LOAD_IMAGE_COLOR);
  PUBLISH(CV_LOAD_IMAGE_GRAYSCALE);
  PUBLISH(CV_LOAD_IMAGE_UNCHANGED);
  PUBLISH(CV_HIST_ARRAY);
  PUBLISH(CV_HIST_SPARSE);
  PUBLISH(CV_8U);
  PUBLISH(CV_8UC1);
  PUBLISH(CV_8UC2);
  PUBLISH(CV_8UC3);
  PUBLISH(CV_8UC4);
  PUBLISH(CV_8S);
  PUBLISH(CV_8SC1);
  PUBLISH(CV_8SC2);
  PUBLISH(CV_8SC3);
  PUBLISH(CV_8SC4);
  PUBLISH(CV_16U);
  PUBLISH(CV_16UC1);
  PUBLISH(CV_16UC2);
  PUBLISH(CV_16UC3);
  PUBLISH(CV_16UC4);
  PUBLISH(CV_16S);
  PUBLISH(CV_16SC1);
  PUBLISH(CV_16SC2);
  PUBLISH(CV_16SC3);
  PUBLISH(CV_16SC4);
  PUBLISH(CV_32S);
  PUBLISH(CV_32SC1);
  PUBLISH(CV_32SC2);
  PUBLISH(CV_32SC3);
  PUBLISH(CV_32SC4);
  PUBLISH(CV_32F);
  PUBLISH(CV_32FC1);
  PUBLISH(CV_32FC2);
  PUBLISH(CV_32FC3);
  PUBLISH(CV_32FC4);
  PUBLISH(CV_64F);
  PUBLISH(CV_64FC1);
  PUBLISH(CV_64FC2);
  PUBLISH(CV_64FC3);
  PUBLISH(CV_64FC4);
  PUBLISH(CV_NEXT_AROUND_ORG);
  PUBLISH(CV_NEXT_AROUND_DST);
  PUBLISH(CV_PREV_AROUND_ORG);
  PUBLISH(CV_PREV_AROUND_DST);
  PUBLISH(CV_NEXT_AROUND_LEFT);
  PUBLISH(CV_NEXT_AROUND_RIGHT);
  PUBLISH(CV_PREV_AROUND_LEFT);
  PUBLISH(CV_PREV_AROUND_RIGHT);

  PUBLISH(CV_WINDOW_AUTOSIZE);

  PUBLISH(CV_PTLOC_INSIDE);
  PUBLISH(CV_PTLOC_ON_EDGE);
  PUBLISH(CV_PTLOC_VERTEX);
  PUBLISH(CV_PTLOC_OUTSIDE_RECT);

  PUBLISH(GC_BGD);
  PUBLISH(GC_FGD);
  PUBLISH(GC_PR_BGD);
  PUBLISH(GC_PR_FGD);
  PUBLISH(GC_INIT_WITH_RECT);
  PUBLISH(GC_INIT_WITH_MASK);
  PUBLISH(GC_EVAL);

#include "generated2.i"
}

