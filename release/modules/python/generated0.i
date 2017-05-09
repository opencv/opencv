
static PyObject *pycvAbs(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_src, &pyobj_dst))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_Abs
CVPY_VALIDATE_Abs();
#endif
  ERRWRAP(cvAbs(src, dst));
  Py_RETURN_NONE;
}

static PyObject *pycvAbsDiff(PyObject *self, PyObject *args)
{
  CvArr* src1;
  PyObject *pyobj_src1 = NULL;
  CvArr* src2;
  PyObject *pyobj_src2 = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;

  if (!PyArg_ParseTuple(args, "OOO", &pyobj_src1, &pyobj_src2, &pyobj_dst))
    return NULL;
  if (!convert_to_CvArr(pyobj_src1, &src1, "src1")) return NULL;
  if (!convert_to_CvArr(pyobj_src2, &src2, "src2")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_AbsDiff
CVPY_VALIDATE_AbsDiff();
#endif
  ERRWRAP(cvAbsDiff(src1, src2, dst));
  Py_RETURN_NONE;
}

static PyObject *pycvAbsDiffS(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvScalar value;
  PyObject *pyobj_value = NULL;

  if (!PyArg_ParseTuple(args, "OOO", &pyobj_src, &pyobj_dst, &pyobj_value))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if (!convert_to_CvScalar(pyobj_value, &value, "value")) return NULL;
#ifdef CVPY_VALIDATE_AbsDiffS
CVPY_VALIDATE_AbsDiffS();
#endif
  ERRWRAP(cvAbsDiffS(src, dst, value));
  Py_RETURN_NONE;
}

static PyObject *pycvAcc(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* image;
  PyObject *pyobj_image = NULL;
  CvArr* sum;
  PyObject *pyobj_sum = NULL;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;

  const char *keywords[] = { "image", "sum", "mask", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|O", (char**)keywords, &pyobj_image, &pyobj_sum, &pyobj_mask))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvArr(pyobj_sum, &sum, "sum")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_Acc
CVPY_VALIDATE_Acc();
#endif
  ERRWRAP(cvAcc(image, sum, mask));
  Py_RETURN_NONE;
}

static PyObject *pycvAdaptiveThreshold(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  double maxValue;
  int adaptive_method = CV_ADAPTIVE_THRESH_MEAN_C;
  int thresholdType = CV_THRESH_BINARY;
  int blockSize = 3;
  double param1 = 5;

  const char *keywords[] = { "src", "dst", "maxValue", "adaptive_method", "thresholdType", "blockSize", "param1", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOd|iiid", (char**)keywords, &pyobj_src, &pyobj_dst, &maxValue, &adaptive_method, &thresholdType, &blockSize, &param1))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_AdaptiveThreshold
CVPY_VALIDATE_AdaptiveThreshold();
#endif
  ERRWRAP(cvAdaptiveThreshold(src, dst, maxValue, adaptive_method, thresholdType, blockSize, param1));
  Py_RETURN_NONE;
}

static PyObject *pycvAdd(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src1;
  PyObject *pyobj_src1 = NULL;
  CvArr* src2;
  PyObject *pyobj_src2 = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;

  const char *keywords[] = { "src1", "src2", "dst", "mask", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|O", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &pyobj_mask))
    return NULL;
  if (!convert_to_CvArr(pyobj_src1, &src1, "src1")) return NULL;
  if (!convert_to_CvArr(pyobj_src2, &src2, "src2")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_Add
CVPY_VALIDATE_Add();
#endif
  ERRWRAP(cvAdd(src1, src2, dst, mask));
  Py_RETURN_NONE;
}

static PyObject *pycvAddS(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvScalar value;
  PyObject *pyobj_value = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;

  const char *keywords[] = { "src", "value", "dst", "mask", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|O", (char**)keywords, &pyobj_src, &pyobj_value, &pyobj_dst, &pyobj_mask))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvScalar(pyobj_value, &value, "value")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_AddS
CVPY_VALIDATE_AddS();
#endif
  ERRWRAP(cvAddS(src, value, dst, mask));
  Py_RETURN_NONE;
}

static PyObject *pycvAddWeighted(PyObject *self, PyObject *args)
{
  CvArr* src1;
  PyObject *pyobj_src1 = NULL;
  double alpha;
  CvArr* src2;
  PyObject *pyobj_src2 = NULL;
  double beta;
  double gamma;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;

  if (!PyArg_ParseTuple(args, "OdOddO", &pyobj_src1, &alpha, &pyobj_src2, &beta, &gamma, &pyobj_dst))
    return NULL;
  if (!convert_to_CvArr(pyobj_src1, &src1, "src1")) return NULL;
  if (!convert_to_CvArr(pyobj_src2, &src2, "src2")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_AddWeighted
CVPY_VALIDATE_AddWeighted();
#endif
  ERRWRAP(cvAddWeighted(src1, alpha, src2, beta, gamma, dst));
  Py_RETURN_NONE;
}

static PyObject *pycvAnd(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src1;
  PyObject *pyobj_src1 = NULL;
  CvArr* src2;
  PyObject *pyobj_src2 = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;

  const char *keywords[] = { "src1", "src2", "dst", "mask", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|O", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &pyobj_mask))
    return NULL;
  if (!convert_to_CvArr(pyobj_src1, &src1, "src1")) return NULL;
  if (!convert_to_CvArr(pyobj_src2, &src2, "src2")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_And
CVPY_VALIDATE_And();
#endif
  ERRWRAP(cvAnd(src1, src2, dst, mask));
  Py_RETURN_NONE;
}

static PyObject *pycvAndS(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvScalar value;
  PyObject *pyobj_value = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;

  const char *keywords[] = { "src", "value", "dst", "mask", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|O", (char**)keywords, &pyobj_src, &pyobj_value, &pyobj_dst, &pyobj_mask))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvScalar(pyobj_value, &value, "value")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_AndS
CVPY_VALIDATE_AndS();
#endif
  ERRWRAP(cvAndS(src, value, dst, mask));
  Py_RETURN_NONE;
}

static PyObject *pycvApproxChains(PyObject *self, PyObject *args, PyObject *kw)
{
  CvSeq* src_seq;
  PyObject *pyobj_src_seq = NULL;
  CvMemStorage* storage;
  PyObject *pyobj_storage = NULL;
  int method = CV_CHAIN_APPROX_SIMPLE;
  double parameter = 0;
  int minimal_perimeter = 0;
  int recursive = 0;

  const char *keywords[] = { "src_seq", "storage", "method", "parameter", "minimal_perimeter", "recursive", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|idii", (char**)keywords, &pyobj_src_seq, &pyobj_storage, &method, &parameter, &minimal_perimeter, &recursive))
    return NULL;
  if (!convert_to_CvSeq(pyobj_src_seq, &src_seq, "src_seq")) return NULL;
  if (!convert_to_CvMemStorage(pyobj_storage, &storage, "storage")) return NULL;
#ifdef CVPY_VALIDATE_ApproxChains
CVPY_VALIDATE_ApproxChains();
#endif
  CvSeq* r;
  ERRWRAP(r = cvApproxChains(src_seq, storage, method, parameter, minimal_perimeter, recursive));
  return FROM_CvSeqPTR(r);
}

static PyObject *pycvApproxPoly(PyObject *self, PyObject *args, PyObject *kw)
;

static PyObject *pycvArcLength(PyObject *self, PyObject *args, PyObject *kw)
{
  cvarrseq curve;
  PyObject *pyobj_curve = NULL;
  CvSlice slice = CV_WHOLE_SEQ;
  PyObject *pyobj_slice = NULL;
  int isClosed = -1;

  const char *keywords[] = { "curve", "slice", "isClosed", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O|Oi", (char**)keywords, &pyobj_curve, &pyobj_slice, &isClosed))
    return NULL;
  if (!convert_to_cvarrseq(pyobj_curve, &curve, "curve")) return NULL;
  if ((pyobj_slice != NULL) && !convert_to_CvSlice(pyobj_slice, &slice, "slice")) return NULL;
#ifdef CVPY_VALIDATE_ArcLength
CVPY_VALIDATE_ArcLength();
#endif
  double r;
  ERRWRAP(r = cvArcLength(curve.seq, slice, isClosed));
  return FROM_double(r);
}

static PyObject *pycvAvg(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;

  const char *keywords[] = { "arr", "mask", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O|O", (char**)keywords, &pyobj_arr, &pyobj_mask))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_Avg
CVPY_VALIDATE_Avg();
#endif
  CvScalar r;
  ERRWRAP(r = cvAvg(arr, mask));
  return FROM_CvScalar(r);
}

static PyObject *pycvAvgSdv(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  CvScalar mean;
  CvScalar stdDev;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;

  const char *keywords[] = { "arr", "mask", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O|O", (char**)keywords, &pyobj_arr, &pyobj_mask))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_AvgSdv
CVPY_VALIDATE_AvgSdv();
#endif
  ERRWRAP(cvAvgSdv(arr, &mean, &stdDev, mask));
  return Py_BuildValue("NN", FROM_CvScalar(mean), FROM_CvScalar(stdDev));
}

static PyObject *pycvBackProjectPCA(PyObject *self, PyObject *args)
{
  CvArr* proj;
  PyObject *pyobj_proj = NULL;
  CvArr* avg;
  PyObject *pyobj_avg = NULL;
  CvArr* eigenvects;
  PyObject *pyobj_eigenvects = NULL;
  CvArr* result;
  PyObject *pyobj_result = NULL;

  if (!PyArg_ParseTuple(args, "OOOO", &pyobj_proj, &pyobj_avg, &pyobj_eigenvects, &pyobj_result))
    return NULL;
  if (!convert_to_CvArr(pyobj_proj, &proj, "proj")) return NULL;
  if (!convert_to_CvArr(pyobj_avg, &avg, "avg")) return NULL;
  if (!convert_to_CvArr(pyobj_eigenvects, &eigenvects, "eigenvects")) return NULL;
  if (!convert_to_CvArr(pyobj_result, &result, "result")) return NULL;
#ifdef CVPY_VALIDATE_BackProjectPCA
CVPY_VALIDATE_BackProjectPCA();
#endif
  ERRWRAP(cvBackProjectPCA(proj, avg, eigenvects, result));
  Py_RETURN_NONE;
}

static PyObject *pycvBoundingRect(PyObject *self, PyObject *args, PyObject *kw)
{
  cvarrseq points;
  PyObject *pyobj_points = NULL;
  int update = 0;

  const char *keywords[] = { "points", "update", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O|i", (char**)keywords, &pyobj_points, &update))
    return NULL;
  if (!convert_to_cvarrseq(pyobj_points, &points, "points")) return NULL;
#ifdef CVPY_VALIDATE_BoundingRect
CVPY_VALIDATE_BoundingRect();
#endif
  CvRect r;
  ERRWRAP(r = cvBoundingRect(points.seq, update));
  return FROM_CvRect(r);
}

static PyObject *pycvBoxPoints(PyObject *self, PyObject *args)
{
  CvBox2D box;
  PyObject *pyobj_box = NULL;
  CvPoint2D32f_4 points;

  if (!PyArg_ParseTuple(args, "O", &pyobj_box))
    return NULL;
  if (!convert_to_CvBox2D(pyobj_box, &box, "box")) return NULL;
#ifdef CVPY_VALIDATE_BoxPoints
CVPY_VALIDATE_BoxPoints();
#endif
  ERRWRAP(cvBoxPoints(box, points));
  return FROM_CvPoint2D32f_4(points);
}

static PyObject *pycv_CV_16SC(PyObject *self, PyObject *args)
{
  int n;

  if (!PyArg_ParseTuple(args, "i", &n))
    return NULL;
#ifdef CVPY_VALIDATE_CV_16SC
CVPY_VALIDATE_CV_16SC();
#endif
  int r;
  ERRWRAP(r = CV_16SC(n));
  return FROM_int(r);
}

static PyObject *pycv_CV_16UC(PyObject *self, PyObject *args)
{
  int n;

  if (!PyArg_ParseTuple(args, "i", &n))
    return NULL;
#ifdef CVPY_VALIDATE_CV_16UC
CVPY_VALIDATE_CV_16UC();
#endif
  int r;
  ERRWRAP(r = CV_16UC(n));
  return FROM_int(r);
}

static PyObject *pycv_CV_32FC(PyObject *self, PyObject *args)
{
  int n;

  if (!PyArg_ParseTuple(args, "i", &n))
    return NULL;
#ifdef CVPY_VALIDATE_CV_32FC
CVPY_VALIDATE_CV_32FC();
#endif
  int r;
  ERRWRAP(r = CV_32FC(n));
  return FROM_int(r);
}

static PyObject *pycv_CV_32SC(PyObject *self, PyObject *args)
{
  int n;

  if (!PyArg_ParseTuple(args, "i", &n))
    return NULL;
#ifdef CVPY_VALIDATE_CV_32SC
CVPY_VALIDATE_CV_32SC();
#endif
  int r;
  ERRWRAP(r = CV_32SC(n));
  return FROM_int(r);
}

static PyObject *pycv_CV_64FC(PyObject *self, PyObject *args)
{
  int n;

  if (!PyArg_ParseTuple(args, "i", &n))
    return NULL;
#ifdef CVPY_VALIDATE_CV_64FC
CVPY_VALIDATE_CV_64FC();
#endif
  int r;
  ERRWRAP(r = CV_64FC(n));
  return FROM_int(r);
}

static PyObject *pycv_CV_8SC(PyObject *self, PyObject *args)
{
  int n;

  if (!PyArg_ParseTuple(args, "i", &n))
    return NULL;
#ifdef CVPY_VALIDATE_CV_8SC
CVPY_VALIDATE_CV_8SC();
#endif
  int r;
  ERRWRAP(r = CV_8SC(n));
  return FROM_int(r);
}

static PyObject *pycv_CV_8UC(PyObject *self, PyObject *args)
{
  int n;

  if (!PyArg_ParseTuple(args, "i", &n))
    return NULL;
#ifdef CVPY_VALIDATE_CV_8UC
CVPY_VALIDATE_CV_8UC();
#endif
  int r;
  ERRWRAP(r = CV_8UC(n));
  return FROM_int(r);
}

static PyObject *pycv_CV_CMP(PyObject *self, PyObject *args)
{
  int a;
  int b;

  if (!PyArg_ParseTuple(args, "ii", &a, &b))
    return NULL;
#ifdef CVPY_VALIDATE_CV_CMP
CVPY_VALIDATE_CV_CMP();
#endif
  int r;
  ERRWRAP(r = CV_CMP(a, b));
  return FROM_int(r);
}

static PyObject *pycv_CV_FOURCC(PyObject *self, PyObject *args)
{
  char c1;
  PyObject *pyobj_c1 = NULL;
  char c2;
  PyObject *pyobj_c2 = NULL;
  char c3;
  PyObject *pyobj_c3 = NULL;
  char c4;
  PyObject *pyobj_c4 = NULL;

  if (!PyArg_ParseTuple(args, "OOOO", &pyobj_c1, &pyobj_c2, &pyobj_c3, &pyobj_c4))
    return NULL;
  if (!convert_to_char(pyobj_c1, &c1, "c1")) return NULL;
  if (!convert_to_char(pyobj_c2, &c2, "c2")) return NULL;
  if (!convert_to_char(pyobj_c3, &c3, "c3")) return NULL;
  if (!convert_to_char(pyobj_c4, &c4, "c4")) return NULL;
#ifdef CVPY_VALIDATE_CV_FOURCC
CVPY_VALIDATE_CV_FOURCC();
#endif
  int r;
  ERRWRAP(r = CV_FOURCC(c1, c2, c3, c4));
  return FROM_int(r);
}

static PyObject *pycv_CV_IABS(PyObject *self, PyObject *args)
{
  int a;

  if (!PyArg_ParseTuple(args, "i", &a))
    return NULL;
#ifdef CVPY_VALIDATE_CV_IABS
CVPY_VALIDATE_CV_IABS();
#endif
  int r;
  ERRWRAP(r = CV_IABS(a));
  return FROM_int(r);
}

static PyObject *pycv_CV_IS_SEQ_CLOSED(PyObject *self, PyObject *args)
{
  CvSeq* s;
  PyObject *pyobj_s = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_s))
    return NULL;
  if (!convert_to_CvSeq(pyobj_s, &s, "s")) return NULL;
#ifdef CVPY_VALIDATE_CV_IS_SEQ_CLOSED
CVPY_VALIDATE_CV_IS_SEQ_CLOSED();
#endif
  int r;
  ERRWRAP(r = CV_IS_SEQ_CLOSED(s));
  return FROM_int(r);
}

static PyObject *pycv_CV_IS_SEQ_CONVEX(PyObject *self, PyObject *args)
{
  CvSeq* s;
  PyObject *pyobj_s = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_s))
    return NULL;
  if (!convert_to_CvSeq(pyobj_s, &s, "s")) return NULL;
#ifdef CVPY_VALIDATE_CV_IS_SEQ_CONVEX
CVPY_VALIDATE_CV_IS_SEQ_CONVEX();
#endif
  int r;
  ERRWRAP(r = CV_IS_SEQ_CONVEX(s));
  return FROM_int(r);
}

static PyObject *pycv_CV_IS_SEQ_CURVE(PyObject *self, PyObject *args)
{
  CvSeq* s;
  PyObject *pyobj_s = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_s))
    return NULL;
  if (!convert_to_CvSeq(pyobj_s, &s, "s")) return NULL;
#ifdef CVPY_VALIDATE_CV_IS_SEQ_CURVE
CVPY_VALIDATE_CV_IS_SEQ_CURVE();
#endif
  int r;
  ERRWRAP(r = CV_IS_SEQ_CURVE(s));
  return FROM_int(r);
}

static PyObject *pycv_CV_IS_SEQ_HOLE(PyObject *self, PyObject *args)
{
  CvSeq* s;
  PyObject *pyobj_s = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_s))
    return NULL;
  if (!convert_to_CvSeq(pyobj_s, &s, "s")) return NULL;
#ifdef CVPY_VALIDATE_CV_IS_SEQ_HOLE
CVPY_VALIDATE_CV_IS_SEQ_HOLE();
#endif
  int r;
  ERRWRAP(r = CV_IS_SEQ_HOLE(s));
  return FROM_int(r);
}

static PyObject *pycv_CV_IS_SEQ_INDEX(PyObject *self, PyObject *args)
{
  CvSeq* s;
  PyObject *pyobj_s = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_s))
    return NULL;
  if (!convert_to_CvSeq(pyobj_s, &s, "s")) return NULL;
#ifdef CVPY_VALIDATE_CV_IS_SEQ_INDEX
CVPY_VALIDATE_CV_IS_SEQ_INDEX();
#endif
  int r;
  ERRWRAP(r = CV_IS_SEQ_INDEX(s));
  return FROM_int(r);
}

static PyObject *pycv_CV_IS_SEQ_SIMPLE(PyObject *self, PyObject *args)
{
  CvSeq* s;
  PyObject *pyobj_s = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_s))
    return NULL;
  if (!convert_to_CvSeq(pyobj_s, &s, "s")) return NULL;
#ifdef CVPY_VALIDATE_CV_IS_SEQ_SIMPLE
CVPY_VALIDATE_CV_IS_SEQ_SIMPLE();
#endif
  int r;
  ERRWRAP(r = CV_IS_SEQ_SIMPLE(s));
  return FROM_int(r);
}

static PyObject *pycv_CV_MAKETYPE(PyObject *self, PyObject *args)
{
  int depth;
  int cn;

  if (!PyArg_ParseTuple(args, "ii", &depth, &cn))
    return NULL;
#ifdef CVPY_VALIDATE_CV_MAKETYPE
CVPY_VALIDATE_CV_MAKETYPE();
#endif
  int r;
  ERRWRAP(r = CV_MAKETYPE(depth, cn));
  return FROM_int(r);
}

static PyObject *pycv_CV_MAT_CN(PyObject *self, PyObject *args)
{
  int i;

  if (!PyArg_ParseTuple(args, "i", &i))
    return NULL;
#ifdef CVPY_VALIDATE_CV_MAT_CN
CVPY_VALIDATE_CV_MAT_CN();
#endif
  int r;
  ERRWRAP(r = CV_MAT_CN(i));
  return FROM_int(r);
}

static PyObject *pycv_CV_MAT_DEPTH(PyObject *self, PyObject *args)
{
  int i;

  if (!PyArg_ParseTuple(args, "i", &i))
    return NULL;
#ifdef CVPY_VALIDATE_CV_MAT_DEPTH
CVPY_VALIDATE_CV_MAT_DEPTH();
#endif
  int r;
  ERRWRAP(r = CV_MAT_DEPTH(i));
  return FROM_int(r);
}

static PyObject *pycv_CV_RGB(PyObject *self, PyObject *args)
{
  double red;
  double grn;
  double blu;

  if (!PyArg_ParseTuple(args, "ddd", &red, &grn, &blu))
    return NULL;
#ifdef CVPY_VALIDATE_CV_RGB
CVPY_VALIDATE_CV_RGB();
#endif
  CvScalar r;
  ERRWRAP(r = CV_RGB(red, grn, blu));
  return FROM_CvScalar(r);
}

static PyObject *pycv_CV_SIGN(PyObject *self, PyObject *args)
{
  int a;

  if (!PyArg_ParseTuple(args, "i", &a))
    return NULL;
#ifdef CVPY_VALIDATE_CV_SIGN
CVPY_VALIDATE_CV_SIGN();
#endif
  int r;
  ERRWRAP(r = CV_SIGN(a));
  return FROM_int(r);
}

static PyObject *pycvCalcArrBackProject(PyObject *self, PyObject *args)
{
  CvArrs image;
  PyObject *pyobj_image = NULL;
  CvArr* back_project;
  PyObject *pyobj_back_project = NULL;
  CvHistogram* hist;
  PyObject *pyobj_hist = NULL;

  if (!PyArg_ParseTuple(args, "OOO", &pyobj_image, &pyobj_back_project, &pyobj_hist))
    return NULL;
  if (!convert_to_CvArrs(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvArr(pyobj_back_project, &back_project, "back_project")) return NULL;
  if (!convert_to_CvHistogram(pyobj_hist, &hist, "hist")) return NULL;
#ifdef CVPY_VALIDATE_CalcArrBackProject
CVPY_VALIDATE_CalcArrBackProject();
#endif
  ERRWRAP(cvCalcArrBackProject(image.ims, back_project, hist));
  Py_RETURN_NONE;
}

static PyObject *pycvCalcArrHist(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArrs image;
  PyObject *pyobj_image = NULL;
  CvHistogram* hist;
  PyObject *pyobj_hist = NULL;
  int accumulate = 0;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;

  const char *keywords[] = { "image", "hist", "accumulate", "mask", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|iO", (char**)keywords, &pyobj_image, &pyobj_hist, &accumulate, &pyobj_mask))
    return NULL;
  if (!convert_to_CvArrs(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvHistogram(pyobj_hist, &hist, "hist")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_CalcArrHist
CVPY_VALIDATE_CalcArrHist();
#endif
  ERRWRAP(cvCalcArrHist(image.ims, hist, accumulate, mask));
  Py_RETURN_NONE;
}

static PyObject *pycvCalcBackProject(PyObject *self, PyObject *args)
{
  IplImages image;
  PyObject *pyobj_image = NULL;
  CvArr* back_project;
  PyObject *pyobj_back_project = NULL;
  CvHistogram* hist;
  PyObject *pyobj_hist = NULL;

  if (!PyArg_ParseTuple(args, "OOO", &pyobj_image, &pyobj_back_project, &pyobj_hist))
    return NULL;
  if (!convert_to_IplImages(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvArr(pyobj_back_project, &back_project, "back_project")) return NULL;
  if (!convert_to_CvHistogram(pyobj_hist, &hist, "hist")) return NULL;
#ifdef CVPY_VALIDATE_CalcBackProject
CVPY_VALIDATE_CalcBackProject();
#endif
  ERRWRAP(cvCalcBackProject(image.ims, back_project, hist));
  Py_RETURN_NONE;
}

static PyObject *pycvCalcBackProjectPatch(PyObject *self, PyObject *args)
{
  IplImages images;
  PyObject *pyobj_images = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvSize patch_size;
  PyObject *pyobj_patch_size = NULL;
  CvHistogram* hist;
  PyObject *pyobj_hist = NULL;
  int method;
  float factor;

  if (!PyArg_ParseTuple(args, "OOOOif", &pyobj_images, &pyobj_dst, &pyobj_patch_size, &pyobj_hist, &method, &factor))
    return NULL;
  if (!convert_to_IplImages(pyobj_images, &images, "images")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if (!convert_to_CvSize(pyobj_patch_size, &patch_size, "patch_size")) return NULL;
  if (!convert_to_CvHistogram(pyobj_hist, &hist, "hist")) return NULL;
#ifdef CVPY_VALIDATE_CalcBackProjectPatch
CVPY_VALIDATE_CalcBackProjectPatch();
#endif
  ERRWRAP(cvCalcBackProjectPatch(images.ims, dst, patch_size, hist, method, factor));
  Py_RETURN_NONE;
}

static PyObject *pycvCalcCovarMatrix(PyObject *self, PyObject *args)
{
  cvarr_count vects;
  PyObject *pyobj_vects = NULL;
  CvArr* covMat;
  PyObject *pyobj_covMat = NULL;
  CvArr* avg;
  PyObject *pyobj_avg = NULL;
  int flags;

  if (!PyArg_ParseTuple(args, "OOOi", &pyobj_vects, &pyobj_covMat, &pyobj_avg, &flags))
    return NULL;
  if (!convert_to_cvarr_count(pyobj_vects, &vects, "vects")) return NULL;
  if (!convert_to_CvArr(pyobj_covMat, &covMat, "covMat")) return NULL;
  if (!convert_to_CvArr(pyobj_avg, &avg, "avg")) return NULL;
#ifdef CVPY_VALIDATE_CalcCovarMatrix
CVPY_VALIDATE_CalcCovarMatrix();
#endif
  ERRWRAP(cvCalcCovarMatrix((const CvArr **)vects.cvarr,vects.count, covMat, avg, flags));
  Py_RETURN_NONE;
}

static PyObject *pycvCalcEMD2(PyObject *self, PyObject *args, PyObject *kw)
;

static PyObject *pycvCalcGlobalOrientation(PyObject *self, PyObject *args)
{
  CvArr* orientation;
  PyObject *pyobj_orientation = NULL;
  CvArr* mask;
  PyObject *pyobj_mask = NULL;
  CvArr* mhi;
  PyObject *pyobj_mhi = NULL;
  double timestamp;
  double duration;

  if (!PyArg_ParseTuple(args, "OOOdd", &pyobj_orientation, &pyobj_mask, &pyobj_mhi, &timestamp, &duration))
    return NULL;
  if (!convert_to_CvArr(pyobj_orientation, &orientation, "orientation")) return NULL;
  if (!convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
  if (!convert_to_CvArr(pyobj_mhi, &mhi, "mhi")) return NULL;
#ifdef CVPY_VALIDATE_CalcGlobalOrientation
CVPY_VALIDATE_CalcGlobalOrientation();
#endif
  double r;
  ERRWRAP(r = cvCalcGlobalOrientation(orientation, mask, mhi, timestamp, duration));
  return FROM_double(r);
}

static PyObject *pycvCalcHist(PyObject *self, PyObject *args, PyObject *kw)
{
  IplImages image;
  PyObject *pyobj_image = NULL;
  CvHistogram* hist;
  PyObject *pyobj_hist = NULL;
  int accumulate = 0;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;

  const char *keywords[] = { "image", "hist", "accumulate", "mask", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|iO", (char**)keywords, &pyobj_image, &pyobj_hist, &accumulate, &pyobj_mask))
    return NULL;
  if (!convert_to_IplImages(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvHistogram(pyobj_hist, &hist, "hist")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_CalcHist
CVPY_VALIDATE_CalcHist();
#endif
  ERRWRAP(cvCalcHist(image.ims, hist, accumulate, mask));
  Py_RETURN_NONE;
}

static PyObject *pycvCalcMotionGradient(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* mhi;
  PyObject *pyobj_mhi = NULL;
  CvArr* mask;
  PyObject *pyobj_mask = NULL;
  CvArr* orientation;
  PyObject *pyobj_orientation = NULL;
  double delta1;
  double delta2;
  int apertureSize = 3;

  const char *keywords[] = { "mhi", "mask", "orientation", "delta1", "delta2", "apertureSize", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOdd|i", (char**)keywords, &pyobj_mhi, &pyobj_mask, &pyobj_orientation, &delta1, &delta2, &apertureSize))
    return NULL;
  if (!convert_to_CvArr(pyobj_mhi, &mhi, "mhi")) return NULL;
  if (!convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
  if (!convert_to_CvArr(pyobj_orientation, &orientation, "orientation")) return NULL;
#ifdef CVPY_VALIDATE_CalcMotionGradient
CVPY_VALIDATE_CalcMotionGradient();
#endif
  ERRWRAP(cvCalcMotionGradient(mhi, mask, orientation, delta1, delta2, apertureSize));
  Py_RETURN_NONE;
}

static PyObject *pycvCalcOpticalFlowBM(PyObject *self, PyObject *args)
{
  CvArr* prev;
  PyObject *pyobj_prev = NULL;
  CvArr* curr;
  PyObject *pyobj_curr = NULL;
  CvSize blockSize;
  PyObject *pyobj_blockSize = NULL;
  CvSize shiftSize;
  PyObject *pyobj_shiftSize = NULL;
  CvSize max_range;
  PyObject *pyobj_max_range = NULL;
  int usePrevious;
  CvArr* velx;
  PyObject *pyobj_velx = NULL;
  CvArr* vely;
  PyObject *pyobj_vely = NULL;

  if (!PyArg_ParseTuple(args, "OOOOOiOO", &pyobj_prev, &pyobj_curr, &pyobj_blockSize, &pyobj_shiftSize, &pyobj_max_range, &usePrevious, &pyobj_velx, &pyobj_vely))
    return NULL;
  if (!convert_to_CvArr(pyobj_prev, &prev, "prev")) return NULL;
  if (!convert_to_CvArr(pyobj_curr, &curr, "curr")) return NULL;
  if (!convert_to_CvSize(pyobj_blockSize, &blockSize, "blockSize")) return NULL;
  if (!convert_to_CvSize(pyobj_shiftSize, &shiftSize, "shiftSize")) return NULL;
  if (!convert_to_CvSize(pyobj_max_range, &max_range, "max_range")) return NULL;
  if (!convert_to_CvArr(pyobj_velx, &velx, "velx")) return NULL;
  if (!convert_to_CvArr(pyobj_vely, &vely, "vely")) return NULL;
#ifdef CVPY_VALIDATE_CalcOpticalFlowBM
CVPY_VALIDATE_CalcOpticalFlowBM();
#endif
  ERRWRAP(cvCalcOpticalFlowBM(prev, curr, blockSize, shiftSize, max_range, usePrevious, velx, vely));
  Py_RETURN_NONE;
}

static PyObject *pycvCalcOpticalFlowFarneback(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* prev;
  PyObject *pyobj_prev = NULL;
  CvArr* curr;
  PyObject *pyobj_curr = NULL;
  CvArr* flow;
  PyObject *pyobj_flow = NULL;
  double pyr_scale = 0.5;
  int levels = 3;
  int winsize = 15;
  int iterations = 3;
  int poly_n = 7;
  double poly_sigma = 1.5;
  int flags = 0;

  const char *keywords[] = { "prev", "curr", "flow", "pyr_scale", "levels", "winsize", "iterations", "poly_n", "poly_sigma", "flags", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|diiiidi", (char**)keywords, &pyobj_prev, &pyobj_curr, &pyobj_flow, &pyr_scale, &levels, &winsize, &iterations, &poly_n, &poly_sigma, &flags))
    return NULL;
  if (!convert_to_CvArr(pyobj_prev, &prev, "prev")) return NULL;
  if (!convert_to_CvArr(pyobj_curr, &curr, "curr")) return NULL;
  if (!convert_to_CvArr(pyobj_flow, &flow, "flow")) return NULL;
#ifdef CVPY_VALIDATE_CalcOpticalFlowFarneback
CVPY_VALIDATE_CalcOpticalFlowFarneback();
#endif
  ERRWRAP(cvCalcOpticalFlowFarneback(prev, curr, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags));
  Py_RETURN_NONE;
}

static PyObject *pycvCalcOpticalFlowHS(PyObject *self, PyObject *args)
{
  CvArr* prev;
  PyObject *pyobj_prev = NULL;
  CvArr* curr;
  PyObject *pyobj_curr = NULL;
  int usePrevious;
  CvArr* velx;
  PyObject *pyobj_velx = NULL;
  CvArr* vely;
  PyObject *pyobj_vely = NULL;
  double lambda;
  CvTermCriteria criteria;
  PyObject *pyobj_criteria = NULL;

  if (!PyArg_ParseTuple(args, "OOiOOdO", &pyobj_prev, &pyobj_curr, &usePrevious, &pyobj_velx, &pyobj_vely, &lambda, &pyobj_criteria))
    return NULL;
  if (!convert_to_CvArr(pyobj_prev, &prev, "prev")) return NULL;
  if (!convert_to_CvArr(pyobj_curr, &curr, "curr")) return NULL;
  if (!convert_to_CvArr(pyobj_velx, &velx, "velx")) return NULL;
  if (!convert_to_CvArr(pyobj_vely, &vely, "vely")) return NULL;
  if (!convert_to_CvTermCriteria(pyobj_criteria, &criteria, "criteria")) return NULL;
#ifdef CVPY_VALIDATE_CalcOpticalFlowHS
CVPY_VALIDATE_CalcOpticalFlowHS();
#endif
  ERRWRAP(cvCalcOpticalFlowHS(prev, curr, usePrevious, velx, vely, lambda, criteria));
  Py_RETURN_NONE;
}

static PyObject *pycvCalcOpticalFlowLK(PyObject *self, PyObject *args)
{
  CvArr* prev;
  PyObject *pyobj_prev = NULL;
  CvArr* curr;
  PyObject *pyobj_curr = NULL;
  CvSize winSize;
  PyObject *pyobj_winSize = NULL;
  CvArr* velx;
  PyObject *pyobj_velx = NULL;
  CvArr* vely;
  PyObject *pyobj_vely = NULL;

  if (!PyArg_ParseTuple(args, "OOOOO", &pyobj_prev, &pyobj_curr, &pyobj_winSize, &pyobj_velx, &pyobj_vely))
    return NULL;
  if (!convert_to_CvArr(pyobj_prev, &prev, "prev")) return NULL;
  if (!convert_to_CvArr(pyobj_curr, &curr, "curr")) return NULL;
  if (!convert_to_CvSize(pyobj_winSize, &winSize, "winSize")) return NULL;
  if (!convert_to_CvArr(pyobj_velx, &velx, "velx")) return NULL;
  if (!convert_to_CvArr(pyobj_vely, &vely, "vely")) return NULL;
#ifdef CVPY_VALIDATE_CalcOpticalFlowLK
CVPY_VALIDATE_CalcOpticalFlowLK();
#endif
  ERRWRAP(cvCalcOpticalFlowLK(prev, curr, winSize, velx, vely));
  Py_RETURN_NONE;
}

static PyObject *pycvCalcOpticalFlowPyrLK(PyObject *self, PyObject *args, PyObject *kw)
;

static PyObject *pycvCalcPCA(PyObject *self, PyObject *args)
{
  CvArr* data;
  PyObject *pyobj_data = NULL;
  CvArr* avg;
  PyObject *pyobj_avg = NULL;
  CvArr* eigenvalues;
  PyObject *pyobj_eigenvalues = NULL;
  CvArr* eigenvectors;
  PyObject *pyobj_eigenvectors = NULL;
  int flags;

  if (!PyArg_ParseTuple(args, "OOOOi", &pyobj_data, &pyobj_avg, &pyobj_eigenvalues, &pyobj_eigenvectors, &flags))
    return NULL;
  if (!convert_to_CvArr(pyobj_data, &data, "data")) return NULL;
  if (!convert_to_CvArr(pyobj_avg, &avg, "avg")) return NULL;
  if (!convert_to_CvArr(pyobj_eigenvalues, &eigenvalues, "eigenvalues")) return NULL;
  if (!convert_to_CvArr(pyobj_eigenvectors, &eigenvectors, "eigenvectors")) return NULL;
#ifdef CVPY_VALIDATE_CalcPCA
CVPY_VALIDATE_CalcPCA();
#endif
  ERRWRAP(cvCalcPCA(data, avg, eigenvalues, eigenvectors, flags));
  Py_RETURN_NONE;
}

static PyObject *pycvCalcProbDensity(PyObject *self, PyObject *args, PyObject *kw)
{
  CvHistogram* hist1;
  PyObject *pyobj_hist1 = NULL;
  CvHistogram* hist2;
  PyObject *pyobj_hist2 = NULL;
  CvHistogram* dst_hist;
  PyObject *pyobj_dst_hist = NULL;
  double scale = 255;

  const char *keywords[] = { "hist1", "hist2", "dst_hist", "scale", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|d", (char**)keywords, &pyobj_hist1, &pyobj_hist2, &pyobj_dst_hist, &scale))
    return NULL;
  if (!convert_to_CvHistogram(pyobj_hist1, &hist1, "hist1")) return NULL;
  if (!convert_to_CvHistogram(pyobj_hist2, &hist2, "hist2")) return NULL;
  if (!convert_to_CvHistogram(pyobj_dst_hist, &dst_hist, "dst_hist")) return NULL;
#ifdef CVPY_VALIDATE_CalcProbDensity
CVPY_VALIDATE_CalcProbDensity();
#endif
  ERRWRAP(cvCalcProbDensity(hist1, hist2, dst_hist, scale));
  Py_RETURN_NONE;
}

static PyObject *pycvCalcSubdivVoronoi2D(PyObject *self, PyObject *args)
{
  CvSubdiv2D* subdiv;
  PyObject *pyobj_subdiv = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_subdiv))
    return NULL;
  if (!convert_to_CvSubdiv2DPTR(pyobj_subdiv, &subdiv, "subdiv")) return NULL;
#ifdef CVPY_VALIDATE_CalcSubdivVoronoi2D
CVPY_VALIDATE_CalcSubdivVoronoi2D();
#endif
  ERRWRAP(cvCalcSubdivVoronoi2D(subdiv));
  Py_RETURN_NONE;
}

static PyObject *pycvCalibrateCamera2(PyObject *self, PyObject *args, PyObject *kw)
{
  CvMat* objectPoints;
  PyObject *pyobj_objectPoints = NULL;
  CvMat* imagePoints;
  PyObject *pyobj_imagePoints = NULL;
  CvMat* pointCounts;
  PyObject *pyobj_pointCounts = NULL;
  CvSize imageSize;
  PyObject *pyobj_imageSize = NULL;
  CvMat* cameraMatrix;
  PyObject *pyobj_cameraMatrix = NULL;
  CvMat* distCoeffs;
  PyObject *pyobj_distCoeffs = NULL;
  CvMat* rvecs;
  PyObject *pyobj_rvecs = NULL;
  CvMat* tvecs;
  PyObject *pyobj_tvecs = NULL;
  int flags = 0;

  const char *keywords[] = { "objectPoints", "imagePoints", "pointCounts", "imageSize", "cameraMatrix", "distCoeffs", "rvecs", "tvecs", "flags", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOOO|i", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints, &pyobj_pointCounts, &pyobj_imageSize, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvecs, &pyobj_tvecs, &flags))
    return NULL;
  if (!convert_to_CvMat(pyobj_objectPoints, &objectPoints, "objectPoints")) return NULL;
  if (!convert_to_CvMat(pyobj_imagePoints, &imagePoints, "imagePoints")) return NULL;
  if (!convert_to_CvMat(pyobj_pointCounts, &pointCounts, "pointCounts")) return NULL;
  if (!convert_to_CvSize(pyobj_imageSize, &imageSize, "imageSize")) return NULL;
  if (!convert_to_CvMat(pyobj_cameraMatrix, &cameraMatrix, "cameraMatrix")) return NULL;
  if (!convert_to_CvMat(pyobj_distCoeffs, &distCoeffs, "distCoeffs")) return NULL;
  if (!convert_to_CvMat(pyobj_rvecs, &rvecs, "rvecs")) return NULL;
  if (!convert_to_CvMat(pyobj_tvecs, &tvecs, "tvecs")) return NULL;
#ifdef CVPY_VALIDATE_CalibrateCamera2
CVPY_VALIDATE_CalibrateCamera2();
#endif
  ERRWRAP(cvCalibrateCamera2(objectPoints, imagePoints, pointCounts, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags));
  Py_RETURN_NONE;
}

static PyObject *pycvCalibrationMatrixValues(PyObject *self, PyObject *args, PyObject *kw)
{
  CvMat* calibMatr;
  PyObject *pyobj_calibMatr = NULL;
  CvSize image_size;
  PyObject *pyobj_image_size = NULL;
  double apertureWidth = 0;
  double apertureHeight = 0;
  double fovx;
  double fovy;
  double focalLength;
  CvPoint2D64f principalPoint;
  double pixelAspectRatio;

  const char *keywords[] = { "calibMatr", "image_size", "apertureWidth", "apertureHeight", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|dd", (char**)keywords, &pyobj_calibMatr, &pyobj_image_size, &apertureWidth, &apertureHeight))
    return NULL;
  if (!convert_to_CvMat(pyobj_calibMatr, &calibMatr, "calibMatr")) return NULL;
  if (!convert_to_CvSize(pyobj_image_size, &image_size, "image_size")) return NULL;
#ifdef CVPY_VALIDATE_CalibrationMatrixValues
CVPY_VALIDATE_CalibrationMatrixValues();
#endif
  ERRWRAP(cvCalibrationMatrixValues(calibMatr, image_size, apertureWidth, apertureHeight, &fovx, &fovy, &focalLength, &principalPoint, &pixelAspectRatio));
  return Py_BuildValue("NNNNN", FROM_double(fovx), FROM_double(fovy), FROM_double(focalLength), FROM_CvPoint2D64f(principalPoint), FROM_double(pixelAspectRatio));
}

static PyObject *pycvCamShift(PyObject *self, PyObject *args)
{
  CvArr* prob_image;
  PyObject *pyobj_prob_image = NULL;
  CvRect window;
  PyObject *pyobj_window = NULL;
  CvTermCriteria criteria;
  PyObject *pyobj_criteria = NULL;
  CvConnectedComp comp;
  CvBox2D box;

  if (!PyArg_ParseTuple(args, "OOO", &pyobj_prob_image, &pyobj_window, &pyobj_criteria))
    return NULL;
  if (!convert_to_CvArr(pyobj_prob_image, &prob_image, "prob_image")) return NULL;
  if (!convert_to_CvRect(pyobj_window, &window, "window")) return NULL;
  if (!convert_to_CvTermCriteria(pyobj_criteria, &criteria, "criteria")) return NULL;
#ifdef CVPY_VALIDATE_CamShift
CVPY_VALIDATE_CamShift();
#endif
  int r;
  ERRWRAP(r = cvCamShift(prob_image, window, criteria, &comp, &box));
  return Py_BuildValue("NNN", FROM_int(r), FROM_CvConnectedComp(comp), FROM_CvBox2D(box));
}

static PyObject *pycvCanny(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* image;
  PyObject *pyobj_image = NULL;
  CvArr* edges;
  PyObject *pyobj_edges = NULL;
  double threshold1;
  double threshold2;
  int aperture_size = 3;

  const char *keywords[] = { "image", "edges", "threshold1", "threshold2", "aperture_size", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOdd|i", (char**)keywords, &pyobj_image, &pyobj_edges, &threshold1, &threshold2, &aperture_size))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvArr(pyobj_edges, &edges, "edges")) return NULL;
#ifdef CVPY_VALIDATE_Canny
CVPY_VALIDATE_Canny();
#endif
  ERRWRAP(cvCanny(image, edges, threshold1, threshold2, aperture_size));
  Py_RETURN_NONE;
}

static PyObject *pycvCaptureFromCAM(PyObject *self, PyObject *args)
{
  int index;

  if (!PyArg_ParseTuple(args, "i", &index))
    return NULL;
#ifdef CVPY_VALIDATE_CaptureFromCAM
CVPY_VALIDATE_CaptureFromCAM();
#endif
  CvCapture* r;
  ERRWRAP(r = cvCaptureFromCAM(index));
  return FROM_CvCapturePTR(r);
}

static PyObject *pycvCaptureFromFile(PyObject *self, PyObject *args)
{
  char* filename;

  if (!PyArg_ParseTuple(args, "s", &filename))
    return NULL;
#ifdef CVPY_VALIDATE_CaptureFromFile
CVPY_VALIDATE_CaptureFromFile();
#endif
  CvCapture* r;
  ERRWRAP(r = cvCaptureFromFile(filename));
  return FROM_CvCapturePTR(r);
}

static PyObject *pycvCartToPolar(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* x;
  PyObject *pyobj_x = NULL;
  CvArr* y;
  PyObject *pyobj_y = NULL;
  CvArr* magnitude;
  PyObject *pyobj_magnitude = NULL;
  CvArr* angle = NULL;
  PyObject *pyobj_angle = NULL;
  int angleInDegrees = 0;

  const char *keywords[] = { "x", "y", "magnitude", "angle", "angleInDegrees", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|Oi", (char**)keywords, &pyobj_x, &pyobj_y, &pyobj_magnitude, &pyobj_angle, &angleInDegrees))
    return NULL;
  if (!convert_to_CvArr(pyobj_x, &x, "x")) return NULL;
  if (!convert_to_CvArr(pyobj_y, &y, "y")) return NULL;
  if (!convert_to_CvArr(pyobj_magnitude, &magnitude, "magnitude")) return NULL;
  if ((pyobj_angle != NULL) && !convert_to_CvArr(pyobj_angle, &angle, "angle")) return NULL;
#ifdef CVPY_VALIDATE_CartToPolar
CVPY_VALIDATE_CartToPolar();
#endif
  ERRWRAP(cvCartToPolar(x, y, magnitude, angle, angleInDegrees));
  Py_RETURN_NONE;
}

static PyObject *pycvCbrt(PyObject *self, PyObject *args)
{
  float value;

  if (!PyArg_ParseTuple(args, "f", &value))
    return NULL;
#ifdef CVPY_VALIDATE_Cbrt
CVPY_VALIDATE_Cbrt();
#endif
  float r;
  ERRWRAP(r = cvCbrt(value));
  return FROM_float(r);
}

static PyObject *pycvCeil(PyObject *self, PyObject *args)
{
  double value;

  if (!PyArg_ParseTuple(args, "d", &value))
    return NULL;
#ifdef CVPY_VALIDATE_Ceil
CVPY_VALIDATE_Ceil();
#endif
  int r;
  ERRWRAP(r = cvCeil(value));
  return FROM_int(r);
}

static PyObject *pycvCheckArr(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  int flags = 0;
  double min_val = 0;
  double max_val = 0;

  const char *keywords[] = { "arr", "flags", "min_val", "max_val", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O|idd", (char**)keywords, &pyobj_arr, &flags, &min_val, &max_val))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
#ifdef CVPY_VALIDATE_CheckArr
CVPY_VALIDATE_CheckArr();
#endif
  int r;
  ERRWRAP(r = cvCheckArr(arr, flags, min_val, max_val));
  return FROM_int(r);
}

static PyObject *pycvCheckContourConvexity(PyObject *self, PyObject *args)
{
  cvarrseq contour;
  PyObject *pyobj_contour = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_contour))
    return NULL;
  if (!convert_to_cvarrseq(pyobj_contour, &contour, "contour")) return NULL;
#ifdef CVPY_VALIDATE_CheckContourConvexity
CVPY_VALIDATE_CheckContourConvexity();
#endif
  int r;
  ERRWRAP(r = cvCheckContourConvexity(contour.seq));
  return FROM_int(r);
}

static PyObject *pycvCircle(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* img;
  PyObject *pyobj_img = NULL;
  CvPoint center;
  PyObject *pyobj_center = NULL;
  int radius;
  CvScalar color;
  PyObject *pyobj_color = NULL;
  int thickness = 1;
  int lineType = 8;
  int shift = 0;

  const char *keywords[] = { "img", "center", "radius", "color", "thickness", "lineType", "shift", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOiO|iii", (char**)keywords, &pyobj_img, &pyobj_center, &radius, &pyobj_color, &thickness, &lineType, &shift))
    return NULL;
  if (!convert_to_CvArr(pyobj_img, &img, "img")) return NULL;
  if (!convert_to_CvPoint(pyobj_center, &center, "center")) return NULL;
  if (!convert_to_CvScalar(pyobj_color, &color, "color")) return NULL;
#ifdef CVPY_VALIDATE_Circle
CVPY_VALIDATE_Circle();
#endif
  ERRWRAP(cvCircle(img, center, radius, color, thickness, lineType, shift));
  Py_RETURN_NONE;
}

static PyObject *pycvClearHist(PyObject *self, PyObject *args)
{
  CvHistogram* hist;
  PyObject *pyobj_hist = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_hist))
    return NULL;
  if (!convert_to_CvHistogram(pyobj_hist, &hist, "hist")) return NULL;
#ifdef CVPY_VALIDATE_ClearHist
CVPY_VALIDATE_ClearHist();
#endif
  ERRWRAP(cvClearHist(hist));
  Py_RETURN_NONE;
}

static PyObject *pycvClearND(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  ints idx;
  PyObject *pyobj_idx = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_arr, &pyobj_idx))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
  if (!convert_to_ints(pyobj_idx, &idx, "idx")) return NULL;
#ifdef CVPY_VALIDATE_ClearND
CVPY_VALIDATE_ClearND();
#endif
  ERRWRAP(cvClearND(arr, idx.i));
  Py_RETURN_NONE;
}

static PyObject *pycvClearSeq(PyObject *self, PyObject *args)
{
  CvSeq* seq;
  PyObject *pyobj_seq = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_seq))
    return NULL;
  if (!convert_to_CvSeq(pyobj_seq, &seq, "seq")) return NULL;
#ifdef CVPY_VALIDATE_ClearSeq
CVPY_VALIDATE_ClearSeq();
#endif
  ERRWRAP(cvClearSeq(seq));
  Py_RETURN_NONE;
}

static PyObject *pycvClearSubdivVoronoi2D(PyObject *self, PyObject *args)
{
  CvSubdiv2D* subdiv;
  PyObject *pyobj_subdiv = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_subdiv))
    return NULL;
  if (!convert_to_CvSubdiv2DPTR(pyobj_subdiv, &subdiv, "subdiv")) return NULL;
#ifdef CVPY_VALIDATE_ClearSubdivVoronoi2D
CVPY_VALIDATE_ClearSubdivVoronoi2D();
#endif
  ERRWRAP(cvClearSubdivVoronoi2D(subdiv));
  Py_RETURN_NONE;
}

static PyObject *pycvClipLine(PyObject *self, PyObject *args)
;

static PyObject *pycvCloneImage(PyObject *self, PyObject *args)
{
  IplImage* image;
  PyObject *pyobj_image = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_image))
    return NULL;
  if (!convert_to_IplImage(pyobj_image, &image, "image")) return NULL;
#ifdef CVPY_VALIDATE_CloneImage
CVPY_VALIDATE_CloneImage();
#endif
  IplImage* r;
  ERRWRAP(r = cvCloneImage(image));
  return FROM_IplImagePTR(r);
}

static PyObject *pycvCloneMat(PyObject *self, PyObject *args)
{
  CvMat* mat;
  PyObject *pyobj_mat = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_mat))
    return NULL;
  if (!convert_to_CvMat(pyobj_mat, &mat, "mat")) return NULL;
#ifdef CVPY_VALIDATE_CloneMat
CVPY_VALIDATE_CloneMat();
#endif
  CvMat* r;
  ERRWRAP(r = cvCloneMat(mat));
  return FROM_CvMatPTR(r);
}

static PyObject *pycvCloneMatND(PyObject *self, PyObject *args)
{
  CvMatND* mat;
  PyObject *pyobj_mat = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_mat))
    return NULL;
  if (!convert_to_CvMatND(pyobj_mat, &mat, "mat")) return NULL;
#ifdef CVPY_VALIDATE_CloneMatND
CVPY_VALIDATE_CloneMatND();
#endif
  CvMatND* r;
  ERRWRAP(r = cvCloneMatND(mat));
  return FROM_CvMatNDPTR(r);
}

static PyObject *pycvCloneSeq(PyObject *self, PyObject *args)
{
  CvSeq* seq;
  PyObject *pyobj_seq = NULL;
  CvMemStorage* storage;
  PyObject *pyobj_storage = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_seq, &pyobj_storage))
    return NULL;
  if (!convert_to_CvSeq(pyobj_seq, &seq, "seq")) return NULL;
  if (!convert_to_CvMemStorage(pyobj_storage, &storage, "storage")) return NULL;
#ifdef CVPY_VALIDATE_CloneSeq
CVPY_VALIDATE_CloneSeq();
#endif
  ERRWRAP(cvCloneSeq(seq, storage));
  Py_RETURN_NONE;
}

static PyObject *pycvCmp(PyObject *self, PyObject *args)
{
  CvArr* src1;
  PyObject *pyobj_src1 = NULL;
  CvArr* src2;
  PyObject *pyobj_src2 = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  int cmpOp;

  if (!PyArg_ParseTuple(args, "OOOi", &pyobj_src1, &pyobj_src2, &pyobj_dst, &cmpOp))
    return NULL;
  if (!convert_to_CvArr(pyobj_src1, &src1, "src1")) return NULL;
  if (!convert_to_CvArr(pyobj_src2, &src2, "src2")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_Cmp
CVPY_VALIDATE_Cmp();
#endif
  ERRWRAP(cvCmp(src1, src2, dst, cmpOp));
  Py_RETURN_NONE;
}

static PyObject *pycvCmpS(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  double value;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  int cmpOp;

  if (!PyArg_ParseTuple(args, "OdOi", &pyobj_src, &value, &pyobj_dst, &cmpOp))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_CmpS
CVPY_VALIDATE_CmpS();
#endif
  ERRWRAP(cvCmpS(src, value, dst, cmpOp));
  Py_RETURN_NONE;
}

static PyObject *pycvCompareHist(PyObject *self, PyObject *args)
{
  CvHistogram* hist1;
  PyObject *pyobj_hist1 = NULL;
  CvHistogram* hist2;
  PyObject *pyobj_hist2 = NULL;
  int method;

  if (!PyArg_ParseTuple(args, "OOi", &pyobj_hist1, &pyobj_hist2, &method))
    return NULL;
  if (!convert_to_CvHistogram(pyobj_hist1, &hist1, "hist1")) return NULL;
  if (!convert_to_CvHistogram(pyobj_hist2, &hist2, "hist2")) return NULL;
#ifdef CVPY_VALIDATE_CompareHist
CVPY_VALIDATE_CompareHist();
#endif
  double r;
  ERRWRAP(r = cvCompareHist(hist1, hist2, method));
  return FROM_double(r);
}

static PyObject *pycvComputeCorrespondEpilines(PyObject *self, PyObject *args)
{
  CvMat* points;
  PyObject *pyobj_points = NULL;
  int whichImage;
  CvMat* F;
  PyObject *pyobj_F = NULL;
  CvMat* lines;
  PyObject *pyobj_lines = NULL;

  if (!PyArg_ParseTuple(args, "OiOO", &pyobj_points, &whichImage, &pyobj_F, &pyobj_lines))
    return NULL;
  if (!convert_to_CvMat(pyobj_points, &points, "points")) return NULL;
  if (!convert_to_CvMat(pyobj_F, &F, "F")) return NULL;
  if (!convert_to_CvMat(pyobj_lines, &lines, "lines")) return NULL;
#ifdef CVPY_VALIDATE_ComputeCorrespondEpilines
CVPY_VALIDATE_ComputeCorrespondEpilines();
#endif
  ERRWRAP(cvComputeCorrespondEpilines(points, whichImage, F, lines));
  Py_RETURN_NONE;
}

static PyObject *pycvContourArea(PyObject *self, PyObject *args, PyObject *kw)
{
  cvarrseq contour;
  PyObject *pyobj_contour = NULL;
  CvSlice slice = CV_WHOLE_SEQ;
  PyObject *pyobj_slice = NULL;

  const char *keywords[] = { "contour", "slice", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O|O", (char**)keywords, &pyobj_contour, &pyobj_slice))
    return NULL;
  if (!convert_to_cvarrseq(pyobj_contour, &contour, "contour")) return NULL;
  if ((pyobj_slice != NULL) && !convert_to_CvSlice(pyobj_slice, &slice, "slice")) return NULL;
#ifdef CVPY_VALIDATE_ContourArea
CVPY_VALIDATE_ContourArea();
#endif
  double r;
  ERRWRAP(r = cvContourArea(contour.seq, slice));
  return FROM_double(r);
}

static PyObject *pycvConvert(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_src, &pyobj_dst))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_Convert
CVPY_VALIDATE_Convert();
#endif
  ERRWRAP(cvConvert(src, dst));
  Py_RETURN_NONE;
}

static PyObject *pycvConvertImage(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  int flags = 0;

  const char *keywords[] = { "src", "dst", "flags", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|i", (char**)keywords, &pyobj_src, &pyobj_dst, &flags))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_ConvertImage
CVPY_VALIDATE_ConvertImage();
#endif
  ERRWRAP(cvConvertImage(src, dst, flags));
  Py_RETURN_NONE;
}

static PyObject *pycvConvertMaps(PyObject *self, PyObject *args)
{
  CvArr* mapx;
  PyObject *pyobj_mapx = NULL;
  CvArr* mapy;
  PyObject *pyobj_mapy = NULL;
  CvArr* mapxy;
  PyObject *pyobj_mapxy = NULL;
  CvArr* mapalpha;
  PyObject *pyobj_mapalpha = NULL;

  if (!PyArg_ParseTuple(args, "OOOO", &pyobj_mapx, &pyobj_mapy, &pyobj_mapxy, &pyobj_mapalpha))
    return NULL;
  if (!convert_to_CvArr(pyobj_mapx, &mapx, "mapx")) return NULL;
  if (!convert_to_CvArr(pyobj_mapy, &mapy, "mapy")) return NULL;
  if (!convert_to_CvArr(pyobj_mapxy, &mapxy, "mapxy")) return NULL;
  if (!convert_to_CvArr(pyobj_mapalpha, &mapalpha, "mapalpha")) return NULL;
#ifdef CVPY_VALIDATE_ConvertMaps
CVPY_VALIDATE_ConvertMaps();
#endif
  ERRWRAP(cvConvertMaps(mapx, mapy, mapxy, mapalpha));
  Py_RETURN_NONE;
}

static PyObject *pycvConvertPointsHomogeneous(PyObject *self, PyObject *args)
{
  CvMat* src;
  PyObject *pyobj_src = NULL;
  CvMat* dst;
  PyObject *pyobj_dst = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_src, &pyobj_dst))
    return NULL;
  if (!convert_to_CvMat(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvMat(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_ConvertPointsHomogeneous
CVPY_VALIDATE_ConvertPointsHomogeneous();
#endif
  ERRWRAP(cvConvertPointsHomogeneous(src, dst));
  Py_RETURN_NONE;
}

static PyObject *pycvConvertScale(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  double scale = 1.0;
  double shift = 0.0;

  const char *keywords[] = { "src", "dst", "scale", "shift", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|dd", (char**)keywords, &pyobj_src, &pyobj_dst, &scale, &shift))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_ConvertScale
CVPY_VALIDATE_ConvertScale();
#endif
  ERRWRAP(cvConvertScale(src, dst, scale, shift));
  Py_RETURN_NONE;
}

static PyObject *pycvConvertScaleAbs(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  double scale = 1.0;
  double shift = 0.0;

  const char *keywords[] = { "src", "dst", "scale", "shift", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|dd", (char**)keywords, &pyobj_src, &pyobj_dst, &scale, &shift))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_ConvertScaleAbs
CVPY_VALIDATE_ConvertScaleAbs();
#endif
  ERRWRAP(cvConvertScaleAbs(src, dst, scale, shift));
  Py_RETURN_NONE;
}

static PyObject *pycvConvexHull2(PyObject *self, PyObject *args, PyObject *kw)
{
  cvarrseq points;
  PyObject *pyobj_points = NULL;
  CvMemStorage* storage;
  PyObject *pyobj_storage = NULL;
  int orientation = CV_CLOCKWISE;
  int return_points = 0;

  const char *keywords[] = { "points", "storage", "orientation", "return_points", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|ii", (char**)keywords, &pyobj_points, &pyobj_storage, &orientation, &return_points))
    return NULL;
  if (!convert_to_cvarrseq(pyobj_points, &points, "points")) return NULL;
  if (!convert_to_CvMemStorage(pyobj_storage, &storage, "storage")) return NULL;
#ifdef CVPY_VALIDATE_ConvexHull2
CVPY_VALIDATE_ConvexHull2();
#endif
  CvSeq* r;
  ERRWRAP(r = cvConvexHull2(points.seq, storage, orientation, return_points));
  return FROM_CvSeqPTR(r);
}

static PyObject *pycvConvexityDefects(PyObject *self, PyObject *args)
{
  cvarrseq contour;
  PyObject *pyobj_contour = NULL;
  CvSeq* convexhull;
  PyObject *pyobj_convexhull = NULL;
  CvMemStorage* storage;
  PyObject *pyobj_storage = NULL;

  if (!PyArg_ParseTuple(args, "OOO", &pyobj_contour, &pyobj_convexhull, &pyobj_storage))
    return NULL;
  if (!convert_to_cvarrseq(pyobj_contour, &contour, "contour")) return NULL;
  if (!convert_to_CvSeq(pyobj_convexhull, &convexhull, "convexhull")) return NULL;
  if (!convert_to_CvMemStorage(pyobj_storage, &storage, "storage")) return NULL;
#ifdef CVPY_VALIDATE_ConvexityDefects
CVPY_VALIDATE_ConvexityDefects();
#endif
  CvSeqOfCvConvexityDefect* r;
  ERRWRAP(r = cvConvexityDefects(contour.seq, convexhull, storage));
  return FROM_CvSeqOfCvConvexityDefectPTR(r);
}

static PyObject *pycvCopy(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;

  const char *keywords[] = { "src", "dst", "mask", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|O", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_mask))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_Copy
CVPY_VALIDATE_Copy();
#endif
  ERRWRAP(cvCopy(src, dst, mask));
  Py_RETURN_NONE;
}

static PyObject *pycvCopyMakeBorder(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvPoint offset;
  PyObject *pyobj_offset = NULL;
  int bordertype;
  CvScalar value = cvScalarAll(0);
  PyObject *pyobj_value = NULL;

  const char *keywords[] = { "src", "dst", "offset", "bordertype", "value", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOi|O", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_offset, &bordertype, &pyobj_value))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if (!convert_to_CvPoint(pyobj_offset, &offset, "offset")) return NULL;
  if ((pyobj_value != NULL) && !convert_to_CvScalar(pyobj_value, &value, "value")) return NULL;
#ifdef CVPY_VALIDATE_CopyMakeBorder
CVPY_VALIDATE_CopyMakeBorder();
#endif
  ERRWRAP(cvCopyMakeBorder(src, dst, offset, bordertype, value));
  Py_RETURN_NONE;
}

static PyObject *pycvCornerEigenValsAndVecs(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* image;
  PyObject *pyobj_image = NULL;
  CvArr* eigenvv;
  PyObject *pyobj_eigenvv = NULL;
  int blockSize;
  int aperture_size = 3;

  const char *keywords[] = { "image", "eigenvv", "blockSize", "aperture_size", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOi|i", (char**)keywords, &pyobj_image, &pyobj_eigenvv, &blockSize, &aperture_size))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvArr(pyobj_eigenvv, &eigenvv, "eigenvv")) return NULL;
#ifdef CVPY_VALIDATE_CornerEigenValsAndVecs
CVPY_VALIDATE_CornerEigenValsAndVecs();
#endif
  ERRWRAP(cvCornerEigenValsAndVecs(image, eigenvv, blockSize, aperture_size));
  Py_RETURN_NONE;
}

static PyObject *pycvCornerHarris(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* image;
  PyObject *pyobj_image = NULL;
  CvArr* harris_dst;
  PyObject *pyobj_harris_dst = NULL;
  int blockSize;
  int aperture_size = 3;
  double k = 0.04;

  const char *keywords[] = { "image", "harris_dst", "blockSize", "aperture_size", "k", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOi|id", (char**)keywords, &pyobj_image, &pyobj_harris_dst, &blockSize, &aperture_size, &k))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvArr(pyobj_harris_dst, &harris_dst, "harris_dst")) return NULL;
#ifdef CVPY_VALIDATE_CornerHarris
CVPY_VALIDATE_CornerHarris();
#endif
  ERRWRAP(cvCornerHarris(image, harris_dst, blockSize, aperture_size, k));
  Py_RETURN_NONE;
}

static PyObject *pycvCornerMinEigenVal(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* image;
  PyObject *pyobj_image = NULL;
  CvArr* eigenval;
  PyObject *pyobj_eigenval = NULL;
  int blockSize;
  int aperture_size = 3;

  const char *keywords[] = { "image", "eigenval", "blockSize", "aperture_size", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOi|i", (char**)keywords, &pyobj_image, &pyobj_eigenval, &blockSize, &aperture_size))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvArr(pyobj_eigenval, &eigenval, "eigenval")) return NULL;
#ifdef CVPY_VALIDATE_CornerMinEigenVal
CVPY_VALIDATE_CornerMinEigenVal();
#endif
  ERRWRAP(cvCornerMinEigenVal(image, eigenval, blockSize, aperture_size));
  Py_RETURN_NONE;
}

static PyObject *pycvCountNonZero(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_arr))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
#ifdef CVPY_VALIDATE_CountNonZero
CVPY_VALIDATE_CountNonZero();
#endif
  int r;
  ERRWRAP(r = cvCountNonZero(arr));
  return FROM_int(r);
}

static PyObject *pycvCreateCameraCapture(PyObject *self, PyObject *args)
{
  int index;

  if (!PyArg_ParseTuple(args, "i", &index))
    return NULL;
#ifdef CVPY_VALIDATE_CreateCameraCapture
CVPY_VALIDATE_CreateCameraCapture();
#endif
  CvCapture* r;
  ERRWRAP(r = cvCreateCameraCapture(index));
  return FROM_CvCapturePTR(r);
}

static PyObject *pycvCreateData(PyObject *self, PyObject *args)
;

static PyObject *pycvCreateFileCapture(PyObject *self, PyObject *args)
{
  char* filename;

  if (!PyArg_ParseTuple(args, "s", &filename))
    return NULL;
#ifdef CVPY_VALIDATE_CreateFileCapture
CVPY_VALIDATE_CreateFileCapture();
#endif
  CvCapture* r;
  ERRWRAP(r = cvCreateFileCapture(filename));
  return FROM_CvCapturePTR(r);
}

static PyObject *pycvCreateHist(PyObject *self, PyObject *args, PyObject *kw)
;

static PyObject *pycvCreateImage(PyObject *self, PyObject *args)
;

static PyObject *pycvCreateImageHeader(PyObject *self, PyObject *args)
;

static PyObject *pycvCreateKalman(PyObject *self, PyObject *args, PyObject *kw)
{
  int dynam_params;
  int measure_params;
  int control_params = 0;

  const char *keywords[] = { "dynam_params", "measure_params", "control_params", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "ii|i", (char**)keywords, &dynam_params, &measure_params, &control_params))
    return NULL;
#ifdef CVPY_VALIDATE_CreateKalman
CVPY_VALIDATE_CreateKalman();
#endif
  CvKalman* r;
  ERRWRAP(r = cvCreateKalman(dynam_params, measure_params, control_params));
  return FROM_CvKalmanPTR(r);
}

static PyObject *pycvCreateMat(PyObject *self, PyObject *args)
;

static PyObject *pycvCreateMatHeader(PyObject *self, PyObject *args)
;

static PyObject *pycvCreateMatND(PyObject *self, PyObject *args)
;

static PyObject *pycvCreateMatNDHeader(PyObject *self, PyObject *args)
;

static PyObject *pycvCreateMemStorage(PyObject *self, PyObject *args, PyObject *kw)
;

static PyObject *pycvCreatePOSITObject(PyObject *self, PyObject *args)
{
  CvPoint3D32fs points;
  PyObject *pyobj_points = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_points))
    return NULL;
  if (!convert_to_CvPoint3D32fs(pyobj_points, &points, "points")) return NULL;
#ifdef CVPY_VALIDATE_CreatePOSITObject
CVPY_VALIDATE_CreatePOSITObject();
#endif
  CvPOSITObject* r;
  ERRWRAP(r = cvCreatePOSITObject(points.p,points.count));
  return FROM_CvPOSITObjectPTR(r);
}

static PyObject *pycvCreateStereoBMState(PyObject *self, PyObject *args, PyObject *kw)
{
  int preset = CV_STEREO_BM_BASIC;
  int numberOfDisparities = 0;

  const char *keywords[] = { "preset", "numberOfDisparities", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "|ii", (char**)keywords, &preset, &numberOfDisparities))
    return NULL;
#ifdef CVPY_VALIDATE_CreateStereoBMState
CVPY_VALIDATE_CreateStereoBMState();
#endif
  CvStereoBMState* r;
  ERRWRAP(r = cvCreateStereoBMState(preset, numberOfDisparities));
  return FROM_CvStereoBMStatePTR(r);
}

static PyObject *pycvCreateStereoGCState(PyObject *self, PyObject *args)
{
  int numberOfDisparities;
  int maxIters;

  if (!PyArg_ParseTuple(args, "ii", &numberOfDisparities, &maxIters))
    return NULL;
#ifdef CVPY_VALIDATE_CreateStereoGCState
CVPY_VALIDATE_CreateStereoGCState();
#endif
  CvStereoGCState* r;
  ERRWRAP(r = cvCreateStereoGCState(numberOfDisparities, maxIters));
  return FROM_CvStereoGCStatePTR(r);
}

static PyObject *pycvCreateStructuringElementEx(PyObject *self, PyObject *args, PyObject *kw)
{
  int cols;
  int rows;
  int anchorX;
  int anchorY;
  int shape;
  ints values = {NULL,0};
  PyObject *pyobj_values = NULL;

  const char *keywords[] = { "cols", "rows", "anchorX", "anchorY", "shape", "values", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "iiiii|O", (char**)keywords, &cols, &rows, &anchorX, &anchorY, &shape, &pyobj_values))
    return NULL;
  if ((pyobj_values != NULL) && !convert_to_ints(pyobj_values, &values, "values")) return NULL;
#ifdef CVPY_VALIDATE_CreateStructuringElementEx
CVPY_VALIDATE_CreateStructuringElementEx();
#endif
  IplConvKernel* r;
  ERRWRAP(r = cvCreateStructuringElementEx(cols, rows, anchorX, anchorY, shape, values.i));
  return FROM_IplConvKernelPTR(r);
}

static PyObject *pycvCreateSubdivDelaunay2D(PyObject *self, PyObject *args)
{
  CvRect rect;
  PyObject *pyobj_rect = NULL;
  CvMemStorage* storage;
  PyObject *pyobj_storage = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_rect, &pyobj_storage))
    return NULL;
  if (!convert_to_CvRect(pyobj_rect, &rect, "rect")) return NULL;
  if (!convert_to_CvMemStorage(pyobj_storage, &storage, "storage")) return NULL;
#ifdef CVPY_VALIDATE_CreateSubdivDelaunay2D
CVPY_VALIDATE_CreateSubdivDelaunay2D();
#endif
  CvSubdiv2D* r;
  ERRWRAP(r = cvCreateSubdivDelaunay2D(rect, storage));
  return FROM_CvSubdiv2DPTR(r);
}

static PyObject *pycvCreateTrackbar(PyObject *self, PyObject *args)
;

static PyObject *pycvCreateVideoWriter(PyObject *self, PyObject *args, PyObject *kw)
{
  char* filename;
  int fourcc;
  double fps;
  CvSize frame_size;
  PyObject *pyobj_frame_size = NULL;
  int is_color = 1;

  const char *keywords[] = { "filename", "fourcc", "fps", "frame_size", "is_color", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "sidO|i", (char**)keywords, &filename, &fourcc, &fps, &pyobj_frame_size, &is_color))
    return NULL;
  if (!convert_to_CvSize(pyobj_frame_size, &frame_size, "frame_size")) return NULL;
#ifdef CVPY_VALIDATE_CreateVideoWriter
CVPY_VALIDATE_CreateVideoWriter();
#endif
  CvVideoWriter* r;
  ERRWRAP(r = cvCreateVideoWriter(filename, fourcc, fps, frame_size, is_color));
  return FROM_CvVideoWriterPTR(r);
}

static PyObject *pycvCrossProduct(PyObject *self, PyObject *args)
{
  CvArr* src1;
  PyObject *pyobj_src1 = NULL;
  CvArr* src2;
  PyObject *pyobj_src2 = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;

  if (!PyArg_ParseTuple(args, "OOO", &pyobj_src1, &pyobj_src2, &pyobj_dst))
    return NULL;
  if (!convert_to_CvArr(pyobj_src1, &src1, "src1")) return NULL;
  if (!convert_to_CvArr(pyobj_src2, &src2, "src2")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_CrossProduct
CVPY_VALIDATE_CrossProduct();
#endif
  ERRWRAP(cvCrossProduct(src1, src2, dst));
  Py_RETURN_NONE;
}

static PyObject *pycvCvtColor(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  int code;

  if (!PyArg_ParseTuple(args, "OOi", &pyobj_src, &pyobj_dst, &code))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_CvtColor
CVPY_VALIDATE_CvtColor();
#endif
  ERRWRAP(cvCvtColor(src, dst, code));
  Py_RETURN_NONE;
}

static PyObject *pycvCvtPixToPlane(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst0;
  PyObject *pyobj_dst0 = NULL;
  CvArr* dst1;
  PyObject *pyobj_dst1 = NULL;
  CvArr* dst2;
  PyObject *pyobj_dst2 = NULL;
  CvArr* dst3;
  PyObject *pyobj_dst3 = NULL;

  if (!PyArg_ParseTuple(args, "OOOOO", &pyobj_src, &pyobj_dst0, &pyobj_dst1, &pyobj_dst2, &pyobj_dst3))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst0, &dst0, "dst0")) return NULL;
  if (!convert_to_CvArr(pyobj_dst1, &dst1, "dst1")) return NULL;
  if (!convert_to_CvArr(pyobj_dst2, &dst2, "dst2")) return NULL;
  if (!convert_to_CvArr(pyobj_dst3, &dst3, "dst3")) return NULL;
#ifdef CVPY_VALIDATE_CvtPixToPlane
CVPY_VALIDATE_CvtPixToPlane();
#endif
  ERRWRAP(cvCvtPixToPlane(src, dst0, dst1, dst2, dst3));
  Py_RETURN_NONE;
}

static PyObject *pycvCvtScale(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  double scale = 1.0;
  double shift = 0.0;

  const char *keywords[] = { "src", "dst", "scale", "shift", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|dd", (char**)keywords, &pyobj_src, &pyobj_dst, &scale, &shift))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_CvtScale
CVPY_VALIDATE_CvtScale();
#endif
  ERRWRAP(cvCvtScale(src, dst, scale, shift));
  Py_RETURN_NONE;
}

static PyObject *pycvDCT(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  int flags;

  if (!PyArg_ParseTuple(args, "OOi", &pyobj_src, &pyobj_dst, &flags))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_DCT
CVPY_VALIDATE_DCT();
#endif
  ERRWRAP(cvDCT(src, dst, flags));
  Py_RETURN_NONE;
}

static PyObject *pycvDFT(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  int flags;
  int nonzeroRows = 0;

  const char *keywords[] = { "src", "dst", "flags", "nonzeroRows", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOi|i", (char**)keywords, &pyobj_src, &pyobj_dst, &flags, &nonzeroRows))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_DFT
CVPY_VALIDATE_DFT();
#endif
  ERRWRAP(cvDFT(src, dst, flags, nonzeroRows));
  Py_RETURN_NONE;
}

static PyObject *pycvDecodeImage(PyObject *self, PyObject *args, PyObject *kw)
{
  CvMat* buf;
  PyObject *pyobj_buf = NULL;
  int iscolor = CV_LOAD_IMAGE_COLOR;

  const char *keywords[] = { "buf", "iscolor", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O|i", (char**)keywords, &pyobj_buf, &iscolor))
    return NULL;
  if (!convert_to_CvMat(pyobj_buf, &buf, "buf")) return NULL;
#ifdef CVPY_VALIDATE_DecodeImage
CVPY_VALIDATE_DecodeImage();
#endif
  IplImage* r;
  ERRWRAP(r = cvDecodeImage(buf, iscolor));
  return FROM_IplImagePTR(r);
}

static PyObject *pycvDecodeImageM(PyObject *self, PyObject *args, PyObject *kw)
{
  CvMat* buf;
  PyObject *pyobj_buf = NULL;
  int iscolor = CV_LOAD_IMAGE_COLOR;

  const char *keywords[] = { "buf", "iscolor", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O|i", (char**)keywords, &pyobj_buf, &iscolor))
    return NULL;
  if (!convert_to_CvMat(pyobj_buf, &buf, "buf")) return NULL;
#ifdef CVPY_VALIDATE_DecodeImageM
CVPY_VALIDATE_DecodeImageM();
#endif
  CvMat* r;
  ERRWRAP(r = cvDecodeImageM(buf, iscolor));
  return FROM_CvMatPTR(r);
}

static PyObject *pycvDecomposeProjectionMatrix(PyObject *self, PyObject *args, PyObject *kw)
{
  CvMat* projMatrix;
  PyObject *pyobj_projMatrix = NULL;
  CvMat* cameraMatrix;
  PyObject *pyobj_cameraMatrix = NULL;
  CvMat* rotMatrix;
  PyObject *pyobj_rotMatrix = NULL;
  CvMat* transVect;
  PyObject *pyobj_transVect = NULL;
  CvMat* rotMatrX = NULL;
  PyObject *pyobj_rotMatrX = NULL;
  CvMat* rotMatrY = NULL;
  PyObject *pyobj_rotMatrY = NULL;
  CvMat* rotMatrZ = NULL;
  PyObject *pyobj_rotMatrZ = NULL;
  CvPoint3D64f eulerAngles;

  const char *keywords[] = { "projMatrix", "cameraMatrix", "rotMatrix", "transVect", "rotMatrX", "rotMatrY", "rotMatrZ", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OOO", (char**)keywords, &pyobj_projMatrix, &pyobj_cameraMatrix, &pyobj_rotMatrix, &pyobj_transVect, &pyobj_rotMatrX, &pyobj_rotMatrY, &pyobj_rotMatrZ))
    return NULL;
  if (!convert_to_CvMat(pyobj_projMatrix, &projMatrix, "projMatrix")) return NULL;
  if (!convert_to_CvMat(pyobj_cameraMatrix, &cameraMatrix, "cameraMatrix")) return NULL;
  if (!convert_to_CvMat(pyobj_rotMatrix, &rotMatrix, "rotMatrix")) return NULL;
  if (!convert_to_CvMat(pyobj_transVect, &transVect, "transVect")) return NULL;
  if ((pyobj_rotMatrX != NULL) && !convert_to_CvMat(pyobj_rotMatrX, &rotMatrX, "rotMatrX")) return NULL;
  if ((pyobj_rotMatrY != NULL) && !convert_to_CvMat(pyobj_rotMatrY, &rotMatrY, "rotMatrY")) return NULL;
  if ((pyobj_rotMatrZ != NULL) && !convert_to_CvMat(pyobj_rotMatrZ, &rotMatrZ, "rotMatrZ")) return NULL;
#ifdef CVPY_VALIDATE_DecomposeProjectionMatrix
CVPY_VALIDATE_DecomposeProjectionMatrix();
#endif
  ERRWRAP(cvDecomposeProjectionMatrix(projMatrix, cameraMatrix, rotMatrix, transVect, rotMatrX, rotMatrY, rotMatrZ, &eulerAngles));
  return FROM_CvPoint3D64f(eulerAngles);
}

static PyObject *pycvDestroyAllWindows(PyObject *self, PyObject *args)
{

#ifdef CVPY_VALIDATE_DestroyAllWindows
CVPY_VALIDATE_DestroyAllWindows();
#endif
  ERRWRAP(cvDestroyAllWindows());
  Py_RETURN_NONE;
}

static PyObject *pycvDestroyWindow(PyObject *self, PyObject *args)
{
  char* name;

  if (!PyArg_ParseTuple(args, "s", &name))
    return NULL;
#ifdef CVPY_VALIDATE_DestroyWindow
CVPY_VALIDATE_DestroyWindow();
#endif
  ERRWRAP(cvDestroyWindow(name));
  Py_RETURN_NONE;
}

static PyObject *pycvDet(PyObject *self, PyObject *args)
{
  CvArr* mat;
  PyObject *pyobj_mat = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_mat))
    return NULL;
  if (!convert_to_CvArr(pyobj_mat, &mat, "mat")) return NULL;
#ifdef CVPY_VALIDATE_Det
CVPY_VALIDATE_Det();
#endif
  double r;
  ERRWRAP(r = cvDet(mat));
  return FROM_double(r);
}

static PyObject *pycvDilate(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  IplConvKernel* element = NULL;
  PyObject *pyobj_element = NULL;
  int iterations = 1;

  const char *keywords[] = { "src", "dst", "element", "iterations", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|Oi", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_element, &iterations))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if ((pyobj_element != NULL) && !convert_to_IplConvKernelPTR(pyobj_element, &element, "element")) return NULL;
#ifdef CVPY_VALIDATE_Dilate
CVPY_VALIDATE_Dilate();
#endif
  ERRWRAP(cvDilate(src, dst, element, iterations));
  Py_RETURN_NONE;
}

static PyObject *pycvDistTransform(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  int distance_type = CV_DIST_L2;
  int mask_size = 3;
  floats mask = {NULL,0};
  PyObject *pyobj_mask = NULL;
  CvArr* labels = NULL;
  PyObject *pyobj_labels = NULL;

  const char *keywords[] = { "src", "dst", "distance_type", "mask_size", "mask", "labels", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|iiOO", (char**)keywords, &pyobj_src, &pyobj_dst, &distance_type, &mask_size, &pyobj_mask, &pyobj_labels))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_floats(pyobj_mask, &mask, "mask")) return NULL;
  if ((pyobj_labels != NULL) && !convert_to_CvArr(pyobj_labels, &labels, "labels")) return NULL;
#ifdef CVPY_VALIDATE_DistTransform
CVPY_VALIDATE_DistTransform();
#endif
  ERRWRAP(cvDistTransform(src, dst, distance_type, mask_size, mask.f, labels));
  Py_RETURN_NONE;
}

static PyObject *pycvDiv(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src1;
  PyObject *pyobj_src1 = NULL;
  CvArr* src2;
  PyObject *pyobj_src2 = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  double scale = 1.0;

  const char *keywords[] = { "src1", "src2", "dst", "scale", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|d", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &scale))
    return NULL;
  if (!convert_to_CvArr(pyobj_src1, &src1, "src1")) return NULL;
  if (!convert_to_CvArr(pyobj_src2, &src2, "src2")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_Div
CVPY_VALIDATE_Div();
#endif
  ERRWRAP(cvDiv(src1, src2, dst, scale));
  Py_RETURN_NONE;
}

static PyObject *pycvDotProduct(PyObject *self, PyObject *args)
{
  CvArr* src1;
  PyObject *pyobj_src1 = NULL;
  CvArr* src2;
  PyObject *pyobj_src2 = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_src1, &pyobj_src2))
    return NULL;
  if (!convert_to_CvArr(pyobj_src1, &src1, "src1")) return NULL;
  if (!convert_to_CvArr(pyobj_src2, &src2, "src2")) return NULL;
#ifdef CVPY_VALIDATE_DotProduct
CVPY_VALIDATE_DotProduct();
#endif
  double r;
  ERRWRAP(r = cvDotProduct(src1, src2));
  return FROM_double(r);
}

static PyObject *pycvDrawChessboardCorners(PyObject *self, PyObject *args)
{
  CvArr* image;
  PyObject *pyobj_image = NULL;
  CvSize patternSize;
  PyObject *pyobj_patternSize = NULL;
  CvPoint2D32fs corners;
  PyObject *pyobj_corners = NULL;
  int patternWasFound;

  if (!PyArg_ParseTuple(args, "OOOi", &pyobj_image, &pyobj_patternSize, &pyobj_corners, &patternWasFound))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvSize(pyobj_patternSize, &patternSize, "patternSize")) return NULL;
  if (!convert_to_CvPoint2D32fs(pyobj_corners, &corners, "corners")) return NULL;
#ifdef CVPY_VALIDATE_DrawChessboardCorners
CVPY_VALIDATE_DrawChessboardCorners();
#endif
  ERRWRAP(cvDrawChessboardCorners(image, patternSize, corners.p,corners.count, patternWasFound));
  Py_RETURN_NONE;
}

static PyObject *pycvDrawContours(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* img;
  PyObject *pyobj_img = NULL;
  CvSeq* contour;
  PyObject *pyobj_contour = NULL;
  CvScalar external_color;
  PyObject *pyobj_external_color = NULL;
  CvScalar hole_color;
  PyObject *pyobj_hole_color = NULL;
  int max_level;
  int thickness = 1;
  int lineType = 8;
  CvPoint offset = cvPoint(0,0);
  PyObject *pyobj_offset = NULL;

  const char *keywords[] = { "img", "contour", "external_color", "hole_color", "max_level", "thickness", "lineType", "offset", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOOi|iiO", (char**)keywords, &pyobj_img, &pyobj_contour, &pyobj_external_color, &pyobj_hole_color, &max_level, &thickness, &lineType, &pyobj_offset))
    return NULL;
  if (!convert_to_CvArr(pyobj_img, &img, "img")) return NULL;
  if (!convert_to_CvSeq(pyobj_contour, &contour, "contour")) return NULL;
  if (!convert_to_CvScalar(pyobj_external_color, &external_color, "external_color")) return NULL;
  if (!convert_to_CvScalar(pyobj_hole_color, &hole_color, "hole_color")) return NULL;
  if ((pyobj_offset != NULL) && !convert_to_CvPoint(pyobj_offset, &offset, "offset")) return NULL;
#ifdef CVPY_VALIDATE_DrawContours
CVPY_VALIDATE_DrawContours();
#endif
  ERRWRAP(cvDrawContours(img, contour, external_color, hole_color, max_level, thickness, lineType, offset));
  Py_RETURN_NONE;
}

static PyObject *pycvEigenVV(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* mat;
  PyObject *pyobj_mat = NULL;
  CvArr* evects;
  PyObject *pyobj_evects = NULL;
  CvArr* evals;
  PyObject *pyobj_evals = NULL;
  double eps;
  int lowindex = 0;
  int highindex = 0;

  const char *keywords[] = { "mat", "evects", "evals", "eps", "lowindex", "highindex", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOd|ii", (char**)keywords, &pyobj_mat, &pyobj_evects, &pyobj_evals, &eps, &lowindex, &highindex))
    return NULL;
  if (!convert_to_CvArr(pyobj_mat, &mat, "mat")) return NULL;
  if (!convert_to_CvArr(pyobj_evects, &evects, "evects")) return NULL;
  if (!convert_to_CvArr(pyobj_evals, &evals, "evals")) return NULL;
#ifdef CVPY_VALIDATE_EigenVV
CVPY_VALIDATE_EigenVV();
#endif
  ERRWRAP(cvEigenVV(mat, evects, evals, eps, lowindex, highindex));
  Py_RETURN_NONE;
}

static PyObject *pycvEllipse(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* img;
  PyObject *pyobj_img = NULL;
  CvPoint center;
  PyObject *pyobj_center = NULL;
  CvSize axes;
  PyObject *pyobj_axes = NULL;
  double angle;
  double start_angle;
  double end_angle;
  CvScalar color;
  PyObject *pyobj_color = NULL;
  int thickness = 1;
  int lineType = 8;
  int shift = 0;

  const char *keywords[] = { "img", "center", "axes", "angle", "start_angle", "end_angle", "color", "thickness", "lineType", "shift", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOdddO|iii", (char**)keywords, &pyobj_img, &pyobj_center, &pyobj_axes, &angle, &start_angle, &end_angle, &pyobj_color, &thickness, &lineType, &shift))
    return NULL;
  if (!convert_to_CvArr(pyobj_img, &img, "img")) return NULL;
  if (!convert_to_CvPoint(pyobj_center, &center, "center")) return NULL;
  if (!convert_to_CvSize(pyobj_axes, &axes, "axes")) return NULL;
  if (!convert_to_CvScalar(pyobj_color, &color, "color")) return NULL;
#ifdef CVPY_VALIDATE_Ellipse
CVPY_VALIDATE_Ellipse();
#endif
  ERRWRAP(cvEllipse(img, center, axes, angle, start_angle, end_angle, color, thickness, lineType, shift));
  Py_RETURN_NONE;
}

static PyObject *pycvEllipseBox(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* img;
  PyObject *pyobj_img = NULL;
  CvBox2D box;
  PyObject *pyobj_box = NULL;
  CvScalar color;
  PyObject *pyobj_color = NULL;
  int thickness = 1;
  int lineType = 8;
  int shift = 0;

  const char *keywords[] = { "img", "box", "color", "thickness", "lineType", "shift", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|iii", (char**)keywords, &pyobj_img, &pyobj_box, &pyobj_color, &thickness, &lineType, &shift))
    return NULL;
  if (!convert_to_CvArr(pyobj_img, &img, "img")) return NULL;
  if (!convert_to_CvBox2D(pyobj_box, &box, "box")) return NULL;
  if (!convert_to_CvScalar(pyobj_color, &color, "color")) return NULL;
#ifdef CVPY_VALIDATE_EllipseBox
CVPY_VALIDATE_EllipseBox();
#endif
  ERRWRAP(cvEllipseBox(img, box, color, thickness, lineType, shift));
  Py_RETURN_NONE;
}

static PyObject *pycvEncodeImage(PyObject *self, PyObject *args, PyObject *kw)
{
  char* ext;
  CvArr* image;
  PyObject *pyobj_image = NULL;
  ints0 params = {&zero,1};
  PyObject *pyobj_params = NULL;

  const char *keywords[] = { "ext", "image", "params", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "sO|O", (char**)keywords, &ext, &pyobj_image, &pyobj_params))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
  if ((pyobj_params != NULL) && !convert_to_ints0(pyobj_params, &params, "params")) return NULL;
#ifdef CVPY_VALIDATE_EncodeImage
CVPY_VALIDATE_EncodeImage();
#endif
  CvMat* r;
  ERRWRAP(r = cvEncodeImage(ext, image, params.i));
  return FROM_CvMatPTR(r);
}

static PyObject *pycvEqualizeHist(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_src, &pyobj_dst))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_EqualizeHist
CVPY_VALIDATE_EqualizeHist();
#endif
  ERRWRAP(cvEqualizeHist(src, dst));
  Py_RETURN_NONE;
}

static PyObject *pycvErode(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  IplConvKernel* element = NULL;
  PyObject *pyobj_element = NULL;
  int iterations = 1;

  const char *keywords[] = { "src", "dst", "element", "iterations", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|Oi", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_element, &iterations))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if ((pyobj_element != NULL) && !convert_to_IplConvKernelPTR(pyobj_element, &element, "element")) return NULL;
#ifdef CVPY_VALIDATE_Erode
CVPY_VALIDATE_Erode();
#endif
  ERRWRAP(cvErode(src, dst, element, iterations));
  Py_RETURN_NONE;
}

static PyObject *pycvEstimateRigidTransform(PyObject *self, PyObject *args)
{
  CvArr* A;
  PyObject *pyobj_A = NULL;
  CvArr* B;
  PyObject *pyobj_B = NULL;
  CvMat* M;
  PyObject *pyobj_M = NULL;
  int full_affine;

  if (!PyArg_ParseTuple(args, "OOOi", &pyobj_A, &pyobj_B, &pyobj_M, &full_affine))
    return NULL;
  if (!convert_to_CvArr(pyobj_A, &A, "A")) return NULL;
  if (!convert_to_CvArr(pyobj_B, &B, "B")) return NULL;
  if (!convert_to_CvMat(pyobj_M, &M, "M")) return NULL;
#ifdef CVPY_VALIDATE_EstimateRigidTransform
CVPY_VALIDATE_EstimateRigidTransform();
#endif
  ERRWRAP(cvEstimateRigidTransform(A, B, M, full_affine));
  Py_RETURN_NONE;
}

static PyObject *pycvExp(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_src, &pyobj_dst))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_Exp
CVPY_VALIDATE_Exp();
#endif
  ERRWRAP(cvExp(src, dst));
  Py_RETURN_NONE;
}

static PyObject *pycvExtractSURF(PyObject *self, PyObject *args)
{
  CvArr* image;
  PyObject *pyobj_image = NULL;
  CvArr* mask;
  PyObject *pyobj_mask = NULL;
  CvSeqOfCvSURFPoint* keypoints;
  CvSeqOfCvSURFDescriptor* descriptors;
  CvMemStorage* storage;
  PyObject *pyobj_storage = NULL;
  CvSURFParams params;

  if (!PyArg_ParseTuple(args, "OOO(idii)", &pyobj_image, &pyobj_mask, &pyobj_storage, &params.extended, &params.hessianThreshold, &params.nOctaves, &params.nOctaveLayers))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
  if (!convert_to_CvMemStorage(pyobj_storage, &storage, "storage")) return NULL;
#ifdef CVPY_VALIDATE_ExtractSURF
CVPY_VALIDATE_ExtractSURF();
#endif
  ERRWRAP(cvExtractSURF(image, mask, &keypoints, &descriptors, storage, params));
  return Py_BuildValue("NN", FROM_CvSeqOfCvSURFPointPTR(keypoints), FROM_CvSeqOfCvSURFDescriptorPTR(descriptors));
}

static PyObject *pycvFastArctan(PyObject *self, PyObject *args)
{
  float y;
  float x;

  if (!PyArg_ParseTuple(args, "ff", &y, &x))
    return NULL;
#ifdef CVPY_VALIDATE_FastArctan
CVPY_VALIDATE_FastArctan();
#endif
  float r;
  ERRWRAP(r = cvFastArctan(y, x));
  return FROM_float(r);
}

static PyObject *pycvFillConvexPoly(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* img;
  PyObject *pyobj_img = NULL;
  CvPoints pn;
  PyObject *pyobj_pn = NULL;
  CvScalar color;
  PyObject *pyobj_color = NULL;
  int lineType = 8;
  int shift = 0;

  const char *keywords[] = { "img", "pn", "color", "lineType", "shift", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|ii", (char**)keywords, &pyobj_img, &pyobj_pn, &pyobj_color, &lineType, &shift))
    return NULL;
  if (!convert_to_CvArr(pyobj_img, &img, "img")) return NULL;
  if (!convert_to_CvPoints(pyobj_pn, &pn, "pn")) return NULL;
  if (!convert_to_CvScalar(pyobj_color, &color, "color")) return NULL;
#ifdef CVPY_VALIDATE_FillConvexPoly
CVPY_VALIDATE_FillConvexPoly();
#endif
  ERRWRAP(cvFillConvexPoly(img, pn.p,pn.count, color, lineType, shift));
  Py_RETURN_NONE;
}

static PyObject *pycvFillPoly(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* img;
  PyObject *pyobj_img = NULL;
  pts_npts_contours polys;
  PyObject *pyobj_polys = NULL;
  CvScalar color;
  PyObject *pyobj_color = NULL;
  int lineType = 8;
  int shift = 0;

  const char *keywords[] = { "img", "polys", "color", "lineType", "shift", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|ii", (char**)keywords, &pyobj_img, &pyobj_polys, &pyobj_color, &lineType, &shift))
    return NULL;
  if (!convert_to_CvArr(pyobj_img, &img, "img")) return NULL;
  if (!convert_to_pts_npts_contours(pyobj_polys, &polys, "polys")) return NULL;
  if (!convert_to_CvScalar(pyobj_color, &color, "color")) return NULL;
#ifdef CVPY_VALIDATE_FillPoly
CVPY_VALIDATE_FillPoly();
#endif
  ERRWRAP(cvFillPoly(img, polys.pts,polys.npts,polys.contours, color, lineType, shift));
  Py_RETURN_NONE;
}

static PyObject *pycvFilter2D(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvMat* kernel;
  PyObject *pyobj_kernel = NULL;
  CvPoint anchor = cvPoint(-1,-1);
  PyObject *pyobj_anchor = NULL;

  const char *keywords[] = { "src", "dst", "kernel", "anchor", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|O", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_kernel, &pyobj_anchor))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if (!convert_to_CvMat(pyobj_kernel, &kernel, "kernel")) return NULL;
  if ((pyobj_anchor != NULL) && !convert_to_CvPoint(pyobj_anchor, &anchor, "anchor")) return NULL;
#ifdef CVPY_VALIDATE_Filter2D
CVPY_VALIDATE_Filter2D();
#endif
  ERRWRAP(cvFilter2D(src, dst, kernel, anchor));
  Py_RETURN_NONE;
}

static PyObject *pycvFindChessboardCorners(PyObject *self, PyObject *args, PyObject *kw)
;

static PyObject *pycvFindContours(PyObject *self, PyObject *args, PyObject *kw)
;

static PyObject *pycvFindCornerSubPix(PyObject *self, PyObject *args)
{
  CvArr* image;
  PyObject *pyobj_image = NULL;
  CvPoint2D32fs corners;
  PyObject *pyobj_corners = NULL;
  CvSize win;
  PyObject *pyobj_win = NULL;
  CvSize zero_zone;
  PyObject *pyobj_zero_zone = NULL;
  CvTermCriteria criteria;
  PyObject *pyobj_criteria = NULL;

  if (!PyArg_ParseTuple(args, "OOOOO", &pyobj_image, &pyobj_corners, &pyobj_win, &pyobj_zero_zone, &pyobj_criteria))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvPoint2D32fs(pyobj_corners, &corners, "corners")) return NULL;
  if (!convert_to_CvSize(pyobj_win, &win, "win")) return NULL;
  if (!convert_to_CvSize(pyobj_zero_zone, &zero_zone, "zero_zone")) return NULL;
  if (!convert_to_CvTermCriteria(pyobj_criteria, &criteria, "criteria")) return NULL;
#ifdef CVPY_VALIDATE_FindCornerSubPix
CVPY_VALIDATE_FindCornerSubPix();
#endif
  ERRWRAP(cvFindCornerSubPix(image, corners.p,corners.count, win, zero_zone, criteria));
  return FROM_CvPoint2D32fs(corners);
}

static PyObject *pycvFindExtrinsicCameraParams2(PyObject *self, PyObject *args, PyObject *kw)
{
  CvMat* objectPoints;
  PyObject *pyobj_objectPoints = NULL;
  CvMat* imagePoints;
  PyObject *pyobj_imagePoints = NULL;
  CvMat* cameraMatrix;
  PyObject *pyobj_cameraMatrix = NULL;
  CvMat* distCoeffs;
  PyObject *pyobj_distCoeffs = NULL;
  CvMat* rvec;
  PyObject *pyobj_rvec = NULL;
  CvMat* tvec;
  PyObject *pyobj_tvec = NULL;
  int useExtrinsicGuess = 0;

  const char *keywords[] = { "objectPoints", "imagePoints", "cameraMatrix", "distCoeffs", "rvec", "tvec", "useExtrinsicGuess", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOOOO|i", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_rvec, &pyobj_tvec, &useExtrinsicGuess))
    return NULL;
  if (!convert_to_CvMat(pyobj_objectPoints, &objectPoints, "objectPoints")) return NULL;
  if (!convert_to_CvMat(pyobj_imagePoints, &imagePoints, "imagePoints")) return NULL;
  if (!convert_to_CvMat(pyobj_cameraMatrix, &cameraMatrix, "cameraMatrix")) return NULL;
  if (!convert_to_CvMat(pyobj_distCoeffs, &distCoeffs, "distCoeffs")) return NULL;
  if (!convert_to_CvMat(pyobj_rvec, &rvec, "rvec")) return NULL;
  if (!convert_to_CvMat(pyobj_tvec, &tvec, "tvec")) return NULL;
#ifdef CVPY_VALIDATE_FindExtrinsicCameraParams2
CVPY_VALIDATE_FindExtrinsicCameraParams2();
#endif
  ERRWRAP(cvFindExtrinsicCameraParams2(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess));
  Py_RETURN_NONE;
}

static PyObject *pycvFindFundamentalMat(PyObject *self, PyObject *args, PyObject *kw)
{
  CvMat* points1;
  PyObject *pyobj_points1 = NULL;
  CvMat* points2;
  PyObject *pyobj_points2 = NULL;
  CvMat* fundamentalMatrix;
  PyObject *pyobj_fundamentalMatrix = NULL;
  int method = CV_FM_RANSAC;
  double param1 = 1.;
  double param2 = 0.99;
  CvMat* status = NULL;
  PyObject *pyobj_status = NULL;

  const char *keywords[] = { "points1", "points2", "fundamentalMatrix", "method", "param1", "param2", "status", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|iddO", (char**)keywords, &pyobj_points1, &pyobj_points2, &pyobj_fundamentalMatrix, &method, &param1, &param2, &pyobj_status))
    return NULL;
  if (!convert_to_CvMat(pyobj_points1, &points1, "points1")) return NULL;
  if (!convert_to_CvMat(pyobj_points2, &points2, "points2")) return NULL;
  if (!convert_to_CvMat(pyobj_fundamentalMatrix, &fundamentalMatrix, "fundamentalMatrix")) return NULL;
  if ((pyobj_status != NULL) && !convert_to_CvMat(pyobj_status, &status, "status")) return NULL;
#ifdef CVPY_VALIDATE_FindFundamentalMat
CVPY_VALIDATE_FindFundamentalMat();
#endif
  int r;
  ERRWRAP(r = cvFindFundamentalMat(points1, points2, fundamentalMatrix, method, param1, param2, status));
  return FROM_int(r);
}

static PyObject *pycvFindHomography(PyObject *self, PyObject *args, PyObject *kw)
{
  CvMat* srcPoints;
  PyObject *pyobj_srcPoints = NULL;
  CvMat* dstPoints;
  PyObject *pyobj_dstPoints = NULL;
  CvMat* H;
  PyObject *pyobj_H = NULL;
  int method = 0;
  double ransacReprojThreshold = 3.0;
  CvMat* status = NULL;
  PyObject *pyobj_status = NULL;

  const char *keywords[] = { "srcPoints", "dstPoints", "H", "method", "ransacReprojThreshold", "status", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|idO", (char**)keywords, &pyobj_srcPoints, &pyobj_dstPoints, &pyobj_H, &method, &ransacReprojThreshold, &pyobj_status))
    return NULL;
  if (!convert_to_CvMat(pyobj_srcPoints, &srcPoints, "srcPoints")) return NULL;
  if (!convert_to_CvMat(pyobj_dstPoints, &dstPoints, "dstPoints")) return NULL;
  if (!convert_to_CvMat(pyobj_H, &H, "H")) return NULL;
  if ((pyobj_status != NULL) && !convert_to_CvMat(pyobj_status, &status, "status")) return NULL;
#ifdef CVPY_VALIDATE_FindHomography
CVPY_VALIDATE_FindHomography();
#endif
  ERRWRAP(cvFindHomography(srcPoints, dstPoints, H, method, ransacReprojThreshold, status));
  Py_RETURN_NONE;
}

static PyObject *pycvFindNearestPoint2D(PyObject *self, PyObject *args)
{
  CvSubdiv2D* subdiv;
  PyObject *pyobj_subdiv = NULL;
  CvPoint2D32f pt;
  PyObject *pyobj_pt = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_subdiv, &pyobj_pt))
    return NULL;
  if (!convert_to_CvSubdiv2DPTR(pyobj_subdiv, &subdiv, "subdiv")) return NULL;
  if (!convert_to_CvPoint2D32f(pyobj_pt, &pt, "pt")) return NULL;
#ifdef CVPY_VALIDATE_FindNearestPoint2D
CVPY_VALIDATE_FindNearestPoint2D();
#endif
  CvSubdiv2DPoint* r;
  ERRWRAP(r = cvFindNearestPoint2D(subdiv, pt));
  return FROM_CvSubdiv2DPointPTR(r);
}

static PyObject *pycvFindStereoCorrespondenceBM(PyObject *self, PyObject *args)
{
  CvArr* left;
  PyObject *pyobj_left = NULL;
  CvArr* right;
  PyObject *pyobj_right = NULL;
  CvArr* disparity;
  PyObject *pyobj_disparity = NULL;
  CvStereoBMState* state;
  PyObject *pyobj_state = NULL;

  if (!PyArg_ParseTuple(args, "OOOO", &pyobj_left, &pyobj_right, &pyobj_disparity, &pyobj_state))
    return NULL;
  if (!convert_to_CvArr(pyobj_left, &left, "left")) return NULL;
  if (!convert_to_CvArr(pyobj_right, &right, "right")) return NULL;
  if (!convert_to_CvArr(pyobj_disparity, &disparity, "disparity")) return NULL;
  if (!convert_to_CvStereoBMStatePTR(pyobj_state, &state, "state")) return NULL;
#ifdef CVPY_VALIDATE_FindStereoCorrespondenceBM
CVPY_VALIDATE_FindStereoCorrespondenceBM();
#endif
  ERRWRAP(cvFindStereoCorrespondenceBM(left, right, disparity, state));
  Py_RETURN_NONE;
}

static PyObject *pycvFindStereoCorrespondenceGC(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* left;
  PyObject *pyobj_left = NULL;
  CvArr* right;
  PyObject *pyobj_right = NULL;
  CvArr* dispLeft;
  PyObject *pyobj_dispLeft = NULL;
  CvArr* dispRight;
  PyObject *pyobj_dispRight = NULL;
  CvStereoGCState* state;
  PyObject *pyobj_state = NULL;
  int useDisparityGuess = 0;

  const char *keywords[] = { "left", "right", "dispLeft", "dispRight", "state", "useDisparityGuess", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOOO|i", (char**)keywords, &pyobj_left, &pyobj_right, &pyobj_dispLeft, &pyobj_dispRight, &pyobj_state, &useDisparityGuess))
    return NULL;
  if (!convert_to_CvArr(pyobj_left, &left, "left")) return NULL;
  if (!convert_to_CvArr(pyobj_right, &right, "right")) return NULL;
  if (!convert_to_CvArr(pyobj_dispLeft, &dispLeft, "dispLeft")) return NULL;
  if (!convert_to_CvArr(pyobj_dispRight, &dispRight, "dispRight")) return NULL;
  if (!convert_to_CvStereoGCStatePTR(pyobj_state, &state, "state")) return NULL;
#ifdef CVPY_VALIDATE_FindStereoCorrespondenceGC
CVPY_VALIDATE_FindStereoCorrespondenceGC();
#endif
  ERRWRAP(cvFindStereoCorrespondenceGC(left, right, dispLeft, dispRight, state, useDisparityGuess));
  Py_RETURN_NONE;
}

static PyObject *pycvFitEllipse2(PyObject *self, PyObject *args)
{
  CvArr* points;
  PyObject *pyobj_points = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_points))
    return NULL;
  if (!convert_to_CvArr(pyobj_points, &points, "points")) return NULL;
#ifdef CVPY_VALIDATE_FitEllipse2
CVPY_VALIDATE_FitEllipse2();
#endif
  CvBox2D r;
  ERRWRAP(r = cvFitEllipse2(points));
  return FROM_CvBox2D(r);
}

static PyObject *pycvFitLine(PyObject *self, PyObject *args)
;

static PyObject *pycvFlip(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst = NULL;
  PyObject *pyobj_dst = NULL;
  int flipMode = 0;

  const char *keywords[] = { "src", "dst", "flipMode", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O|Oi", (char**)keywords, &pyobj_src, &pyobj_dst, &flipMode))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if ((pyobj_dst != NULL) && !convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_Flip
CVPY_VALIDATE_Flip();
#endif
  ERRWRAP(cvFlip(src, dst, flipMode));
  Py_RETURN_NONE;
}

static PyObject *pycvFloodFill(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* image;
  PyObject *pyobj_image = NULL;
  CvPoint seed_point;
  PyObject *pyobj_seed_point = NULL;
  CvScalar new_val;
  PyObject *pyobj_new_val = NULL;
  CvScalar lo_diff = cvScalarAll(0);
  PyObject *pyobj_lo_diff = NULL;
  CvScalar up_diff = cvScalarAll(0);
  PyObject *pyobj_up_diff = NULL;
  CvConnectedComp comp;
  int flags = 4;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;

  const char *keywords[] = { "image", "seed_point", "new_val", "lo_diff", "up_diff", "flags", "mask", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|OOiO", (char**)keywords, &pyobj_image, &pyobj_seed_point, &pyobj_new_val, &pyobj_lo_diff, &pyobj_up_diff, &flags, &pyobj_mask))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvPoint(pyobj_seed_point, &seed_point, "seed_point")) return NULL;
  if (!convert_to_CvScalar(pyobj_new_val, &new_val, "new_val")) return NULL;
  if ((pyobj_lo_diff != NULL) && !convert_to_CvScalar(pyobj_lo_diff, &lo_diff, "lo_diff")) return NULL;
  if ((pyobj_up_diff != NULL) && !convert_to_CvScalar(pyobj_up_diff, &up_diff, "up_diff")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_FloodFill
CVPY_VALIDATE_FloodFill();
#endif
  ERRWRAP(cvFloodFill(image, seed_point, new_val, lo_diff, up_diff, &comp, flags, mask));
  return FROM_CvConnectedComp(comp);
}

static PyObject *pycvFloor(PyObject *self, PyObject *args)
{
  double value;

  if (!PyArg_ParseTuple(args, "d", &value))
    return NULL;
#ifdef CVPY_VALIDATE_Floor
CVPY_VALIDATE_Floor();
#endif
  int r;
  ERRWRAP(r = cvFloor(value));
  return FROM_int(r);
}

static PyObject *pycvGEMM(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src1;
  PyObject *pyobj_src1 = NULL;
  CvArr* src2;
  PyObject *pyobj_src2 = NULL;
  double alpha;
  CvArr* src3;
  PyObject *pyobj_src3 = NULL;
  double beta;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  int tABC = 0;

  const char *keywords[] = { "src1", "src2", "alpha", "src3", "beta", "dst", "tABC", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOdOdO|i", (char**)keywords, &pyobj_src1, &pyobj_src2, &alpha, &pyobj_src3, &beta, &pyobj_dst, &tABC))
    return NULL;
  if (!convert_to_CvArr(pyobj_src1, &src1, "src1")) return NULL;
  if (!convert_to_CvArr(pyobj_src2, &src2, "src2")) return NULL;
  if (!convert_to_CvArr(pyobj_src3, &src3, "src3")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_GEMM
CVPY_VALIDATE_GEMM();
#endif
  ERRWRAP(cvGEMM(src1, src2, alpha, src3, beta, dst, tABC));
  Py_RETURN_NONE;
}

static PyObject *pycvGet1D(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  int idx;

  if (!PyArg_ParseTuple(args, "Oi", &pyobj_arr, &idx))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
#ifdef CVPY_VALIDATE_Get1D
CVPY_VALIDATE_Get1D();
#endif
  CvScalar r;
  ERRWRAP(r = cvGet1D(arr, idx));
  return FROM_CvScalar(r);
}

static PyObject *pycvGet2D(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  int idx0;
  int idx1;

  if (!PyArg_ParseTuple(args, "Oii", &pyobj_arr, &idx0, &idx1))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
#ifdef CVPY_VALIDATE_Get2D
CVPY_VALIDATE_Get2D();
#endif
  CvScalar r;
  ERRWRAP(r = cvGet2D(arr, idx0, idx1));
  return FROM_CvScalar(r);
}

static PyObject *pycvGet3D(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  int idx0;
  int idx1;
  int idx2;

  if (!PyArg_ParseTuple(args, "Oiii", &pyobj_arr, &idx0, &idx1, &idx2))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
#ifdef CVPY_VALIDATE_Get3D
CVPY_VALIDATE_Get3D();
#endif
  CvScalar r;
  ERRWRAP(r = cvGet3D(arr, idx0, idx1, idx2));
  return FROM_CvScalar(r);
}

static PyObject *pycvGetAffineTransform(PyObject *self, PyObject *args)
{
  CvPoint2D32f* src;
  PyObject *pyobj_src = NULL;
  CvPoint2D32f* dst;
  PyObject *pyobj_dst = NULL;
  CvMat* mapMatrix;
  PyObject *pyobj_mapMatrix = NULL;

  if (!PyArg_ParseTuple(args, "OOO", &pyobj_src, &pyobj_dst, &pyobj_mapMatrix))
    return NULL;
  if (!convert_to_CvPoint2D32fPTR(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvPoint2D32fPTR(pyobj_dst, &dst, "dst")) return NULL;
  if (!convert_to_CvMat(pyobj_mapMatrix, &mapMatrix, "mapMatrix")) return NULL;
#ifdef CVPY_VALIDATE_GetAffineTransform
CVPY_VALIDATE_GetAffineTransform();
#endif
  ERRWRAP(cvGetAffineTransform(src, dst, mapMatrix));
  Py_RETURN_NONE;
}

static PyObject *pycvGetCaptureProperty(PyObject *self, PyObject *args)
{
  CvCapture* capture;
  PyObject *pyobj_capture = NULL;
  int property_id;

  if (!PyArg_ParseTuple(args, "Oi", &pyobj_capture, &property_id))
    return NULL;
  if (!convert_to_CvCapturePTR(pyobj_capture, &capture, "capture")) return NULL;
#ifdef CVPY_VALIDATE_GetCaptureProperty
CVPY_VALIDATE_GetCaptureProperty();
#endif
  double r;
  ERRWRAP(r = cvGetCaptureProperty(capture, property_id));
  return FROM_double(r);
}

static PyObject *pycvGetCentralMoment(PyObject *self, PyObject *args)
{
  CvMoments* moments;
  PyObject *pyobj_moments = NULL;
  int x_order;
  int y_order;

  if (!PyArg_ParseTuple(args, "Oii", &pyobj_moments, &x_order, &y_order))
    return NULL;
  if (!convert_to_CvMomentsPTR(pyobj_moments, &moments, "moments")) return NULL;
#ifdef CVPY_VALIDATE_GetCentralMoment
CVPY_VALIDATE_GetCentralMoment();
#endif
  double r;
  ERRWRAP(r = cvGetCentralMoment(moments, x_order, y_order));
  return FROM_double(r);
}

static PyObject *pycvGetCol(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  CvMat* submat;
  int col;

  if (!PyArg_ParseTuple(args, "Oi", &pyobj_arr, &col))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
preShareData(arr, &submat);
#ifdef CVPY_VALIDATE_GetCol
CVPY_VALIDATE_GetCol();
#endif
  ERRWRAP(cvGetCol(arr, submat, col));
  return shareData(pyobj_arr, arr, submat);
}

static PyObject *pycvGetCols(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  CvMat* submat;
  int startCol;
  int endCol;

  if (!PyArg_ParseTuple(args, "Oii", &pyobj_arr, &startCol, &endCol))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
preShareData(arr, &submat);
#ifdef CVPY_VALIDATE_GetCols
CVPY_VALIDATE_GetCols();
#endif
  ERRWRAP(cvGetCols(arr, submat, startCol, endCol));
  return shareData(pyobj_arr, arr, submat);
}

static PyObject *pycvGetDiag(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  CvMat* submat;
  int diag = 0;

  const char *keywords[] = { "arr", "diag", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O|i", (char**)keywords, &pyobj_arr, &diag))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
preShareData(arr, &submat);
#ifdef CVPY_VALIDATE_GetDiag
CVPY_VALIDATE_GetDiag();
#endif
  ERRWRAP(cvGetDiag(arr, submat, diag));
  return shareData(pyobj_arr, arr, submat);
}

static PyObject *pycvGetDims(PyObject *self, PyObject *args)
;

static PyObject *pycvGetElemType(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_arr))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
#ifdef CVPY_VALIDATE_GetElemType
CVPY_VALIDATE_GetElemType();
#endif
  int r;
  ERRWRAP(r = cvGetElemType(arr));
  return FROM_int(r);
}

static PyObject *pycvGetHuMoments(PyObject *self, PyObject *args)
;

static PyObject *pycvGetImage(PyObject *self, PyObject *args)
;

static PyObject *pycvGetImageCOI(PyObject *self, PyObject *args)
{
  IplImage* image;
  PyObject *pyobj_image = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_image))
    return NULL;
  if (!convert_to_IplImage(pyobj_image, &image, "image")) return NULL;
#ifdef CVPY_VALIDATE_GetImageCOI
CVPY_VALIDATE_GetImageCOI();
#endif
  int r;
  ERRWRAP(r = cvGetImageCOI(image));
  return FROM_int(r);
}

static PyObject *pycvGetImageROI(PyObject *self, PyObject *args)
{
  IplImage* image;
  PyObject *pyobj_image = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_image))
    return NULL;
  if (!convert_to_IplImage(pyobj_image, &image, "image")) return NULL;
#ifdef CVPY_VALIDATE_GetImageROI
CVPY_VALIDATE_GetImageROI();
#endif
  CvRect r;
  ERRWRAP(r = cvGetImageROI(image));
  return FROM_CvRect(r);
}

static PyObject *pycvGetMat(PyObject *self, PyObject *args, PyObject *kw)
;

static PyObject *pycvGetMinMaxHistValue(PyObject *self, PyObject *args)
;

static PyObject *pycvGetND(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  ints indices;
  PyObject *pyobj_indices = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_arr, &pyobj_indices))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
  if (!convert_to_ints(pyobj_indices, &indices, "indices")) return NULL;
#ifdef CVPY_VALIDATE_GetND
CVPY_VALIDATE_GetND();
#endif
  CvScalar r;
  ERRWRAP(r = cvGetND(arr, indices.i));
  return FROM_CvScalar(r);
}

static PyObject *pycvGetNormalizedCentralMoment(PyObject *self, PyObject *args)
{
  CvMoments* moments;
  PyObject *pyobj_moments = NULL;
  int x_order;
  int y_order;

  if (!PyArg_ParseTuple(args, "Oii", &pyobj_moments, &x_order, &y_order))
    return NULL;
  if (!convert_to_CvMomentsPTR(pyobj_moments, &moments, "moments")) return NULL;
#ifdef CVPY_VALIDATE_GetNormalizedCentralMoment
CVPY_VALIDATE_GetNormalizedCentralMoment();
#endif
  double r;
  ERRWRAP(r = cvGetNormalizedCentralMoment(moments, x_order, y_order));
  return FROM_double(r);
}

static PyObject *pycvGetOptimalDFTSize(PyObject *self, PyObject *args)
{
  int size0;

  if (!PyArg_ParseTuple(args, "i", &size0))
    return NULL;
#ifdef CVPY_VALIDATE_GetOptimalDFTSize
CVPY_VALIDATE_GetOptimalDFTSize();
#endif
  int r;
  ERRWRAP(r = cvGetOptimalDFTSize(size0));
  return FROM_int(r);
}

static PyObject *pycvGetOptimalNewCameraMatrix(PyObject *self, PyObject *args, PyObject *kw)
{
  CvMat* cameraMatrix;
  PyObject *pyobj_cameraMatrix = NULL;
  CvMat* distCoeffs;
  PyObject *pyobj_distCoeffs = NULL;
  CvSize imageSize;
  PyObject *pyobj_imageSize = NULL;
  double alpha;
  CvMat* newCameraMatrix;
  PyObject *pyobj_newCameraMatrix = NULL;
  CvSize newImageSize = cvSize(0,0);
  PyObject *pyobj_newImageSize = NULL;
  CvRect* validPixROI = NULL;
  PyObject *pyobj_validPixROI = NULL;
  int centerPrincipalPoint = 0;

  const char *keywords[] = { "cameraMatrix", "distCoeffs", "imageSize", "alpha", "newCameraMatrix", "newImageSize", "validPixROI", "centerPrincipalPoint", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOdO|OOi", (char**)keywords, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_imageSize, &alpha, &pyobj_newCameraMatrix, &pyobj_newImageSize, &pyobj_validPixROI, &centerPrincipalPoint))
    return NULL;
  if (!convert_to_CvMat(pyobj_cameraMatrix, &cameraMatrix, "cameraMatrix")) return NULL;
  if (!convert_to_CvMat(pyobj_distCoeffs, &distCoeffs, "distCoeffs")) return NULL;
  if (!convert_to_CvSize(pyobj_imageSize, &imageSize, "imageSize")) return NULL;
  if (!convert_to_CvMat(pyobj_newCameraMatrix, &newCameraMatrix, "newCameraMatrix")) return NULL;
  if ((pyobj_newImageSize != NULL) && !convert_to_CvSize(pyobj_newImageSize, &newImageSize, "newImageSize")) return NULL;
  if ((pyobj_validPixROI != NULL) && !convert_to_CvRectPTR(pyobj_validPixROI, &validPixROI, "validPixROI")) return NULL;
#ifdef CVPY_VALIDATE_GetOptimalNewCameraMatrix
CVPY_VALIDATE_GetOptimalNewCameraMatrix();
#endif
  ERRWRAP(cvGetOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, alpha, newCameraMatrix, newImageSize, validPixROI, centerPrincipalPoint));
  Py_RETURN_NONE;
}

static PyObject *pycvGetPerspectiveTransform(PyObject *self, PyObject *args)
{
  CvPoint2D32f* src;
  PyObject *pyobj_src = NULL;
  CvPoint2D32f* dst;
  PyObject *pyobj_dst = NULL;
  CvMat* mapMatrix;
  PyObject *pyobj_mapMatrix = NULL;

  if (!PyArg_ParseTuple(args, "OOO", &pyobj_src, &pyobj_dst, &pyobj_mapMatrix))
    return NULL;
  if (!convert_to_CvPoint2D32fPTR(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvPoint2D32fPTR(pyobj_dst, &dst, "dst")) return NULL;
  if (!convert_to_CvMat(pyobj_mapMatrix, &mapMatrix, "mapMatrix")) return NULL;
#ifdef CVPY_VALIDATE_GetPerspectiveTransform
CVPY_VALIDATE_GetPerspectiveTransform();
#endif
  ERRWRAP(cvGetPerspectiveTransform(src, dst, mapMatrix));
  Py_RETURN_NONE;
}

static PyObject *pycvGetQuadrangleSubPix(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvMat* mapMatrix;
  PyObject *pyobj_mapMatrix = NULL;

  if (!PyArg_ParseTuple(args, "OOO", &pyobj_src, &pyobj_dst, &pyobj_mapMatrix))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if (!convert_to_CvMat(pyobj_mapMatrix, &mapMatrix, "mapMatrix")) return NULL;
#ifdef CVPY_VALIDATE_GetQuadrangleSubPix
CVPY_VALIDATE_GetQuadrangleSubPix();
#endif
  ERRWRAP(cvGetQuadrangleSubPix(src, dst, mapMatrix));
  Py_RETURN_NONE;
}

static PyObject *pycvGetReal1D(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  int idx0;

  if (!PyArg_ParseTuple(args, "Oi", &pyobj_arr, &idx0))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
#ifdef CVPY_VALIDATE_GetReal1D
CVPY_VALIDATE_GetReal1D();
#endif
  double r;
  ERRWRAP(r = cvGetReal1D(arr, idx0));
  return FROM_double(r);
}

static PyObject *pycvGetReal2D(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  int idx0;
  int idx1;

  if (!PyArg_ParseTuple(args, "Oii", &pyobj_arr, &idx0, &idx1))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
#ifdef CVPY_VALIDATE_GetReal2D
CVPY_VALIDATE_GetReal2D();
#endif
  double r;
  ERRWRAP(r = cvGetReal2D(arr, idx0, idx1));
  return FROM_double(r);
}

static PyObject *pycvGetReal3D(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  int idx0;
  int idx1;
  int idx2;

  if (!PyArg_ParseTuple(args, "Oiii", &pyobj_arr, &idx0, &idx1, &idx2))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
#ifdef CVPY_VALIDATE_GetReal3D
CVPY_VALIDATE_GetReal3D();
#endif
  double r;
  ERRWRAP(r = cvGetReal3D(arr, idx0, idx1, idx2));
  return FROM_double(r);
}

static PyObject *pycvGetRealND(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  ints idx;
  PyObject *pyobj_idx = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_arr, &pyobj_idx))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
  if (!convert_to_ints(pyobj_idx, &idx, "idx")) return NULL;
#ifdef CVPY_VALIDATE_GetRealND
CVPY_VALIDATE_GetRealND();
#endif
  double r;
  ERRWRAP(r = cvGetRealND(arr, idx.i));
  return FROM_double(r);
}

static PyObject *pycvGetRectSubPix(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvPoint2D32f center;
  PyObject *pyobj_center = NULL;

  if (!PyArg_ParseTuple(args, "OOO", &pyobj_src, &pyobj_dst, &pyobj_center))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if (!convert_to_CvPoint2D32f(pyobj_center, &center, "center")) return NULL;
#ifdef CVPY_VALIDATE_GetRectSubPix
CVPY_VALIDATE_GetRectSubPix();
#endif
  ERRWRAP(cvGetRectSubPix(src, dst, center));
  Py_RETURN_NONE;
}

static PyObject *pycvGetRotationMatrix2D(PyObject *self, PyObject *args)
{
  CvPoint2D32f center;
  PyObject *pyobj_center = NULL;
  double angle;
  double scale;
  CvMat* mapMatrix;
  PyObject *pyobj_mapMatrix = NULL;

  if (!PyArg_ParseTuple(args, "OddO", &pyobj_center, &angle, &scale, &pyobj_mapMatrix))
    return NULL;
  if (!convert_to_CvPoint2D32f(pyobj_center, &center, "center")) return NULL;
  if (!convert_to_CvMat(pyobj_mapMatrix, &mapMatrix, "mapMatrix")) return NULL;
#ifdef CVPY_VALIDATE_GetRotationMatrix2D
CVPY_VALIDATE_GetRotationMatrix2D();
#endif
  ERRWRAP(cvGetRotationMatrix2D(center, angle, scale, mapMatrix));
  Py_RETURN_NONE;
}

static PyObject *pycvGetRow(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  CvMat* submat;
  int row;

  if (!PyArg_ParseTuple(args, "Oi", &pyobj_arr, &row))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
preShareData(arr, &submat);
#ifdef CVPY_VALIDATE_GetRow
CVPY_VALIDATE_GetRow();
#endif
  ERRWRAP(cvGetRow(arr, submat, row));
  return shareData(pyobj_arr, arr, submat);
}

static PyObject *pycvGetRows(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  CvMat* submat;
  int startRow;
  int endRow;
  int deltaRow = 1;

  const char *keywords[] = { "arr", "startRow", "endRow", "deltaRow", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "Oii|i", (char**)keywords, &pyobj_arr, &startRow, &endRow, &deltaRow))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
preShareData(arr, &submat);
#ifdef CVPY_VALIDATE_GetRows
CVPY_VALIDATE_GetRows();
#endif
  ERRWRAP(cvGetRows(arr, submat, startRow, endRow, deltaRow));
  return shareData(pyobj_arr, arr, submat);
}

static PyObject *pycvGetSize(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_arr))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
#ifdef CVPY_VALIDATE_GetSize
CVPY_VALIDATE_GetSize();
#endif
  CvSize r;
  ERRWRAP(r = cvGetSize(arr));
  return FROM_CvSize(r);
}

static PyObject *pycvGetSpatialMoment(PyObject *self, PyObject *args)
{
  CvMoments* moments;
  PyObject *pyobj_moments = NULL;
  int x_order;
  int y_order;

  if (!PyArg_ParseTuple(args, "Oii", &pyobj_moments, &x_order, &y_order))
    return NULL;
  if (!convert_to_CvMomentsPTR(pyobj_moments, &moments, "moments")) return NULL;
#ifdef CVPY_VALIDATE_GetSpatialMoment
CVPY_VALIDATE_GetSpatialMoment();
#endif
  double r;
  ERRWRAP(r = cvGetSpatialMoment(moments, x_order, y_order));
  return FROM_double(r);
}

static PyObject *pycvGetStarKeypoints(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* image;
  PyObject *pyobj_image = NULL;
  CvMemStorage* storage;
  PyObject *pyobj_storage = NULL;
  CvStarDetectorParams params = cvStarDetectorParams();
  PyObject *pyobj_params = NULL;

  const char *keywords[] = { "image", "storage", "params", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|O", (char**)keywords, &pyobj_image, &pyobj_storage, &pyobj_params))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvMemStorage(pyobj_storage, &storage, "storage")) return NULL;
  if ((pyobj_params != NULL) && !convert_to_CvStarDetectorParams(pyobj_params, &params, "params")) return NULL;
#ifdef CVPY_VALIDATE_GetStarKeypoints
CVPY_VALIDATE_GetStarKeypoints();
#endif
  CvSeqOfCvStarKeypoint* r;
  ERRWRAP(r = cvGetStarKeypoints(image, storage, params));
  return FROM_CvSeqOfCvStarKeypointPTR(r);
}

static PyObject *pycvGetSubRect(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  CvMat* submat;
  CvRect rect;
  PyObject *pyobj_rect = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_arr, &pyobj_rect))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
preShareData(arr, &submat);
  if (!convert_to_CvRect(pyobj_rect, &rect, "rect")) return NULL;
#ifdef CVPY_VALIDATE_GetSubRect
CVPY_VALIDATE_GetSubRect();
#endif
  ERRWRAP(cvGetSubRect(arr, submat, rect));
  return shareData(pyobj_arr, arr, submat);
}

static PyObject *pycvGetTextSize(PyObject *self, PyObject *args)
{
  char* textString;
  CvFont* font;
  PyObject *pyobj_font = NULL;
  CvSize textSize;
  int baseline;

  if (!PyArg_ParseTuple(args, "sO", &textString, &pyobj_font))
    return NULL;
  if (!convert_to_CvFontPTR(pyobj_font, &font, "font")) return NULL;
#ifdef CVPY_VALIDATE_GetTextSize
CVPY_VALIDATE_GetTextSize();
#endif
  ERRWRAP(cvGetTextSize(textString, font, &textSize, &baseline));
  return Py_BuildValue("NN", FROM_CvSize(textSize), FROM_int(baseline));
}

static PyObject *pycvGetTickCount(PyObject *self, PyObject *args)
{

#ifdef CVPY_VALIDATE_GetTickCount
CVPY_VALIDATE_GetTickCount();
#endif
  int64 r;
  ERRWRAP(r = cvGetTickCount());
  return FROM_int64(r);
}

static PyObject *pycvGetTickFrequency(PyObject *self, PyObject *args)
{

#ifdef CVPY_VALIDATE_GetTickFrequency
CVPY_VALIDATE_GetTickFrequency();
#endif
  int64 r;
  ERRWRAP(r = cvGetTickFrequency());
  return FROM_int64(r);
}

static PyObject *pycvGetTrackbarPos(PyObject *self, PyObject *args)
{
  char* trackbarName;
  char* windowName;

  if (!PyArg_ParseTuple(args, "ss", &trackbarName, &windowName))
    return NULL;
#ifdef CVPY_VALIDATE_GetTrackbarPos
CVPY_VALIDATE_GetTrackbarPos();
#endif
  int r;
  ERRWRAP(r = cvGetTrackbarPos(trackbarName, windowName));
  return FROM_int(r);
}

static PyObject *pycvGetWindowProperty(PyObject *self, PyObject *args)
{
  char* name;
  int prop_id;

  if (!PyArg_ParseTuple(args, "si", &name, &prop_id))
    return NULL;
#ifdef CVPY_VALIDATE_GetWindowProperty
CVPY_VALIDATE_GetWindowProperty();
#endif
  double r;
  ERRWRAP(r = cvGetWindowProperty(name, prop_id));
  return FROM_double(r);
}

static PyObject *pycvGoodFeaturesToTrack(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* image;
  PyObject *pyobj_image = NULL;
  CvArr* eigImage;
  PyObject *pyobj_eigImage = NULL;
  CvArr* tempImage;
  PyObject *pyobj_tempImage = NULL;
  cvpoint2d32f_count cornerCount;
  PyObject *pyobj_cornerCount = NULL;
  double qualityLevel;
  double minDistance;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;
  int blockSize = 3;
  int useHarris = 0;
  double k = 0.04;

  const char *keywords[] = { "image", "eigImage", "tempImage", "cornerCount", "qualityLevel", "minDistance", "mask", "blockSize", "useHarris", "k", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOOdd|Oiid", (char**)keywords, &pyobj_image, &pyobj_eigImage, &pyobj_tempImage, &pyobj_cornerCount, &qualityLevel, &minDistance, &pyobj_mask, &blockSize, &useHarris, &k))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvArr(pyobj_eigImage, &eigImage, "eigImage")) return NULL;
  if (!convert_to_CvArr(pyobj_tempImage, &tempImage, "tempImage")) return NULL;
  if (!convert_to_cvpoint2d32f_count(pyobj_cornerCount, &cornerCount, "cornerCount")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_GoodFeaturesToTrack
CVPY_VALIDATE_GoodFeaturesToTrack();
#endif
  ERRWRAP(cvGoodFeaturesToTrack(image, eigImage, tempImage, cornerCount.points,&cornerCount.count, qualityLevel, minDistance, mask, blockSize, useHarris, k));
  return FROM_cvpoint2d32f_count(cornerCount);
}

static PyObject *pycvGrabCut(PyObject *self, PyObject *args)
{
  CvArr* image;
  PyObject *pyobj_image = NULL;
  CvArr* mask;
  PyObject *pyobj_mask = NULL;
  CvRect rect;
  PyObject *pyobj_rect = NULL;
  CvArr* bgdModel;
  PyObject *pyobj_bgdModel = NULL;
  CvArr* fgdModel;
  PyObject *pyobj_fgdModel = NULL;
  int iterCount;
  int mode;

  if (!PyArg_ParseTuple(args, "OOOOOii", &pyobj_image, &pyobj_mask, &pyobj_rect, &pyobj_bgdModel, &pyobj_fgdModel, &iterCount, &mode))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
  if (!convert_to_CvRect(pyobj_rect, &rect, "rect")) return NULL;
  if (!convert_to_CvArr(pyobj_bgdModel, &bgdModel, "bgdModel")) return NULL;
  if (!convert_to_CvArr(pyobj_fgdModel, &fgdModel, "fgdModel")) return NULL;
#ifdef CVPY_VALIDATE_GrabCut
CVPY_VALIDATE_GrabCut();
#endif
  ERRWRAP(cvGrabCut(image, mask, rect, bgdModel, fgdModel, iterCount, mode));
  Py_RETURN_NONE;
}

static PyObject *pycvGrabFrame(PyObject *self, PyObject *args)
{
  CvCapture* capture;
  PyObject *pyobj_capture = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_capture))
    return NULL;
  if (!convert_to_CvCapturePTR(pyobj_capture, &capture, "capture")) return NULL;
#ifdef CVPY_VALIDATE_GrabFrame
CVPY_VALIDATE_GrabFrame();
#endif
  int r;
  ERRWRAP(r = cvGrabFrame(capture));
  return FROM_int(r);
}

static PyObject *pycvHOGDetectMultiScale(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* image;
  PyObject *pyobj_image = NULL;
  CvMemStorage* storage;
  PyObject *pyobj_storage = NULL;
  CvArr* svm_classifier = NULL;
  PyObject *pyobj_svm_classifier = NULL;
  CvSize win_stride = cvSize(0,0);
  PyObject *pyobj_win_stride = NULL;
  double hit_threshold = 0;
  double scale = 1.05;
  int group_threshold = 2;
  CvSize padding = cvSize(0,0);
  PyObject *pyobj_padding = NULL;
  CvSize win_size = cvSize(64,128);
  PyObject *pyobj_win_size = NULL;
  CvSize block_size = cvSize(16,16);
  PyObject *pyobj_block_size = NULL;
  CvSize block_stride = cvSize(8,8);
  PyObject *pyobj_block_stride = NULL;
  CvSize cell_size = cvSize(8,8);
  PyObject *pyobj_cell_size = NULL;
  int nbins = 9;
  int gammaCorrection = 1;

  const char *keywords[] = { "image", "storage", "svm_classifier", "win_stride", "hit_threshold", "scale", "group_threshold", "padding", "win_size", "block_size", "block_stride", "cell_size", "nbins", "gammaCorrection", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|OOddiOOOOOii", (char**)keywords, &pyobj_image, &pyobj_storage, &pyobj_svm_classifier, &pyobj_win_stride, &hit_threshold, &scale, &group_threshold, &pyobj_padding, &pyobj_win_size, &pyobj_block_size, &pyobj_block_stride, &pyobj_cell_size, &nbins, &gammaCorrection))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvMemStorage(pyobj_storage, &storage, "storage")) return NULL;
  if ((pyobj_svm_classifier != NULL) && !convert_to_CvArr(pyobj_svm_classifier, &svm_classifier, "svm_classifier")) return NULL;
  if ((pyobj_win_stride != NULL) && !convert_to_CvSize(pyobj_win_stride, &win_stride, "win_stride")) return NULL;
  if ((pyobj_padding != NULL) && !convert_to_CvSize(pyobj_padding, &padding, "padding")) return NULL;
  if ((pyobj_win_size != NULL) && !convert_to_CvSize(pyobj_win_size, &win_size, "win_size")) return NULL;
  if ((pyobj_block_size != NULL) && !convert_to_CvSize(pyobj_block_size, &block_size, "block_size")) return NULL;
  if ((pyobj_block_stride != NULL) && !convert_to_CvSize(pyobj_block_stride, &block_stride, "block_stride")) return NULL;
  if ((pyobj_cell_size != NULL) && !convert_to_CvSize(pyobj_cell_size, &cell_size, "cell_size")) return NULL;
#ifdef CVPY_VALIDATE_HOGDetectMultiScale
CVPY_VALIDATE_HOGDetectMultiScale();
#endif
  CvSeq* r;
  ERRWRAP(r = cvHOGDetectMultiScale(image, storage, svm_classifier, win_stride, hit_threshold, scale, group_threshold, padding, win_size, block_size, block_stride, cell_size, nbins, gammaCorrection));
  return FROM_CvSeqPTR(r);
}

static PyObject *pycvHaarDetectObjects(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* image;
  PyObject *pyobj_image = NULL;
  CvHaarClassifierCascade* cascade;
  PyObject *pyobj_cascade = NULL;
  CvMemStorage* storage;
  PyObject *pyobj_storage = NULL;
  double scale_factor = 1.1;
  int min_neighbors = 3;
  int flags = 0;
  CvSize min_size = cvSize(0,0);
  PyObject *pyobj_min_size = NULL;

  const char *keywords[] = { "image", "cascade", "storage", "scale_factor", "min_neighbors", "flags", "min_size", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|diiO", (char**)keywords, &pyobj_image, &pyobj_cascade, &pyobj_storage, &scale_factor, &min_neighbors, &flags, &pyobj_min_size))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvHaarClassifierCascadePTR(pyobj_cascade, &cascade, "cascade")) return NULL;
  if (!convert_to_CvMemStorage(pyobj_storage, &storage, "storage")) return NULL;
  if ((pyobj_min_size != NULL) && !convert_to_CvSize(pyobj_min_size, &min_size, "min_size")) return NULL;
#ifdef CVPY_VALIDATE_HaarDetectObjects
CVPY_VALIDATE_HaarDetectObjects();
#endif
  CvSeqOfCvAvgComp* r;
  ERRWRAP(r = cvHaarDetectObjects(image, cascade, storage, scale_factor, min_neighbors, flags, min_size));
  return FROM_CvSeqOfCvAvgCompPTR(r);
}

static PyObject *pycvHoughCircles(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* image;
  PyObject *pyobj_image = NULL;
  CvMat* circle_storage;
  PyObject *pyobj_circle_storage = NULL;
  int method;
  double dp;
  double min_dist;
  double param1 = 100;
  double param2 = 100;
  int min_radius = 0;
  int max_radius = 0;

  const char *keywords[] = { "image", "circle_storage", "method", "dp", "min_dist", "param1", "param2", "min_radius", "max_radius", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOidd|ddii", (char**)keywords, &pyobj_image, &pyobj_circle_storage, &method, &dp, &min_dist, &param1, &param2, &min_radius, &max_radius))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvMat(pyobj_circle_storage, &circle_storage, "circle_storage")) return NULL;
#ifdef CVPY_VALIDATE_HoughCircles
CVPY_VALIDATE_HoughCircles();
#endif
  ERRWRAP(cvHoughCircles(image, circle_storage, method, dp, min_dist, param1, param2, min_radius, max_radius));
  Py_RETURN_NONE;
}

static PyObject *pycvHoughLines2(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* image;
  PyObject *pyobj_image = NULL;
  CvMemStorage* storage;
  PyObject *pyobj_storage = NULL;
  int method;
  double rho;
  double theta;
  int threshold;
  double param1 = 0;
  double param2 = 0;

  const char *keywords[] = { "image", "storage", "method", "rho", "theta", "threshold", "param1", "param2", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOiddi|dd", (char**)keywords, &pyobj_image, &pyobj_storage, &method, &rho, &theta, &threshold, &param1, &param2))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvMemStorage(pyobj_storage, &storage, "storage")) return NULL;
#ifdef CVPY_VALIDATE_HoughLines2
CVPY_VALIDATE_HoughLines2();
#endif
  CvSeq* r;
  ERRWRAP(r = cvHoughLines2(image, storage, method, rho, theta, threshold, param1, param2));
  return FROM_CvSeqPTR(r);
}

static PyObject *pycvInRange(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* lower;
  PyObject *pyobj_lower = NULL;
  CvArr* upper;
  PyObject *pyobj_upper = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;

  if (!PyArg_ParseTuple(args, "OOOO", &pyobj_src, &pyobj_lower, &pyobj_upper, &pyobj_dst))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_lower, &lower, "lower")) return NULL;
  if (!convert_to_CvArr(pyobj_upper, &upper, "upper")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_InRange
CVPY_VALIDATE_InRange();
#endif
  ERRWRAP(cvInRange(src, lower, upper, dst));
  Py_RETURN_NONE;
}

static PyObject *pycvInRangeS(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvScalar lower;
  PyObject *pyobj_lower = NULL;
  CvScalar upper;
  PyObject *pyobj_upper = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;

  if (!PyArg_ParseTuple(args, "OOOO", &pyobj_src, &pyobj_lower, &pyobj_upper, &pyobj_dst))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvScalar(pyobj_lower, &lower, "lower")) return NULL;
  if (!convert_to_CvScalar(pyobj_upper, &upper, "upper")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_InRangeS
CVPY_VALIDATE_InRangeS();
#endif
  ERRWRAP(cvInRangeS(src, lower, upper, dst));
  Py_RETURN_NONE;
}

static PyObject *pycvInitFont(PyObject *self, PyObject *args, PyObject *kw)
{
  CvFont font;
  int fontFace;
  double hscale;
  double vscale;
  double shear = 0;
  int thickness = 1;
  int lineType = 8;

  const char *keywords[] = { "fontFace", "hscale", "vscale", "shear", "thickness", "lineType", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "idd|dii", (char**)keywords, &fontFace, &hscale, &vscale, &shear, &thickness, &lineType))
    return NULL;
#ifdef CVPY_VALIDATE_InitFont
CVPY_VALIDATE_InitFont();
#endif
  ERRWRAP(cvInitFont(&font, fontFace, hscale, vscale, shear, thickness, lineType));
  return FROM_CvFont(font);
}

static PyObject *pycvInitIntrinsicParams2D(PyObject *self, PyObject *args, PyObject *kw)
{
  CvMat* objectPoints;
  PyObject *pyobj_objectPoints = NULL;
  CvMat* imagePoints;
  PyObject *pyobj_imagePoints = NULL;
  CvMat* npoints;
  PyObject *pyobj_npoints = NULL;
  CvSize imageSize;
  PyObject *pyobj_imageSize = NULL;
  CvMat* cameraMatrix;
  PyObject *pyobj_cameraMatrix = NULL;
  double aspectRatio = 1.;

  const char *keywords[] = { "objectPoints", "imagePoints", "npoints", "imageSize", "cameraMatrix", "aspectRatio", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOOO|d", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints, &pyobj_npoints, &pyobj_imageSize, &pyobj_cameraMatrix, &aspectRatio))
    return NULL;
  if (!convert_to_CvMat(pyobj_objectPoints, &objectPoints, "objectPoints")) return NULL;
  if (!convert_to_CvMat(pyobj_imagePoints, &imagePoints, "imagePoints")) return NULL;
  if (!convert_to_CvMat(pyobj_npoints, &npoints, "npoints")) return NULL;
  if (!convert_to_CvSize(pyobj_imageSize, &imageSize, "imageSize")) return NULL;
  if (!convert_to_CvMat(pyobj_cameraMatrix, &cameraMatrix, "cameraMatrix")) return NULL;
#ifdef CVPY_VALIDATE_InitIntrinsicParams2D
CVPY_VALIDATE_InitIntrinsicParams2D();
#endif
  ERRWRAP(cvInitIntrinsicParams2D(objectPoints, imagePoints, npoints, imageSize, cameraMatrix, aspectRatio));
  Py_RETURN_NONE;
}

static PyObject *pycvInitLineIterator(PyObject *self, PyObject *args, PyObject *kw)
;

static PyObject *pycvInitUndistortMap(PyObject *self, PyObject *args)
{
  CvMat* cameraMatrix;
  PyObject *pyobj_cameraMatrix = NULL;
  CvMat* distCoeffs;
  PyObject *pyobj_distCoeffs = NULL;
  CvArr* map1;
  PyObject *pyobj_map1 = NULL;
  CvArr* map2;
  PyObject *pyobj_map2 = NULL;

  if (!PyArg_ParseTuple(args, "OOOO", &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_map1, &pyobj_map2))
    return NULL;
  if (!convert_to_CvMat(pyobj_cameraMatrix, &cameraMatrix, "cameraMatrix")) return NULL;
  if (!convert_to_CvMat(pyobj_distCoeffs, &distCoeffs, "distCoeffs")) return NULL;
  if (!convert_to_CvArr(pyobj_map1, &map1, "map1")) return NULL;
  if (!convert_to_CvArr(pyobj_map2, &map2, "map2")) return NULL;
#ifdef CVPY_VALIDATE_InitUndistortMap
CVPY_VALIDATE_InitUndistortMap();
#endif
  ERRWRAP(cvInitUndistortMap(cameraMatrix, distCoeffs, map1, map2));
  Py_RETURN_NONE;
}

static PyObject *pycvInitUndistortRectifyMap(PyObject *self, PyObject *args)
{
  CvMat* cameraMatrix;
  PyObject *pyobj_cameraMatrix = NULL;
  CvMat* distCoeffs;
  PyObject *pyobj_distCoeffs = NULL;
  CvMat* R;
  PyObject *pyobj_R = NULL;
  CvMat* newCameraMatrix;
  PyObject *pyobj_newCameraMatrix = NULL;
  CvArr* map1;
  PyObject *pyobj_map1 = NULL;
  CvArr* map2;
  PyObject *pyobj_map2 = NULL;

  if (!PyArg_ParseTuple(args, "OOOOOO", &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_R, &pyobj_newCameraMatrix, &pyobj_map1, &pyobj_map2))
    return NULL;
  if (!convert_to_CvMat(pyobj_cameraMatrix, &cameraMatrix, "cameraMatrix")) return NULL;
  if (!convert_to_CvMat(pyobj_distCoeffs, &distCoeffs, "distCoeffs")) return NULL;
  if (!convert_to_CvMat(pyobj_R, &R, "R")) return NULL;
  if (!convert_to_CvMat(pyobj_newCameraMatrix, &newCameraMatrix, "newCameraMatrix")) return NULL;
  if (!convert_to_CvArr(pyobj_map1, &map1, "map1")) return NULL;
  if (!convert_to_CvArr(pyobj_map2, &map2, "map2")) return NULL;
#ifdef CVPY_VALIDATE_InitUndistortRectifyMap
CVPY_VALIDATE_InitUndistortRectifyMap();
#endif
  ERRWRAP(cvInitUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, map1, map2));
  Py_RETURN_NONE;
}

static PyObject *pycvInpaint(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* mask;
  PyObject *pyobj_mask = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  double inpaintRadius;
  int flags;

  if (!PyArg_ParseTuple(args, "OOOdi", &pyobj_src, &pyobj_mask, &pyobj_dst, &inpaintRadius, &flags))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_Inpaint
CVPY_VALIDATE_Inpaint();
#endif
  ERRWRAP(cvInpaint(src, mask, dst, inpaintRadius, flags));
  Py_RETURN_NONE;
}

static PyObject *pycvIntegral(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* image;
  PyObject *pyobj_image = NULL;
  CvArr* sum;
  PyObject *pyobj_sum = NULL;
  CvArr* sqsum = NULL;
  PyObject *pyobj_sqsum = NULL;
  CvArr* tiltedSum = NULL;
  PyObject *pyobj_tiltedSum = NULL;

  const char *keywords[] = { "image", "sum", "sqsum", "tiltedSum", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|OO", (char**)keywords, &pyobj_image, &pyobj_sum, &pyobj_sqsum, &pyobj_tiltedSum))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvArr(pyobj_sum, &sum, "sum")) return NULL;
  if ((pyobj_sqsum != NULL) && !convert_to_CvArr(pyobj_sqsum, &sqsum, "sqsum")) return NULL;
  if ((pyobj_tiltedSum != NULL) && !convert_to_CvArr(pyobj_tiltedSum, &tiltedSum, "tiltedSum")) return NULL;
#ifdef CVPY_VALIDATE_Integral
CVPY_VALIDATE_Integral();
#endif
  ERRWRAP(cvIntegral(image, sum, sqsum, tiltedSum));
  Py_RETURN_NONE;
}

static PyObject *pycvInvSqrt(PyObject *self, PyObject *args)
{
  float value;

  if (!PyArg_ParseTuple(args, "f", &value))
    return NULL;
#ifdef CVPY_VALIDATE_InvSqrt
CVPY_VALIDATE_InvSqrt();
#endif
  float r;
  ERRWRAP(r = cvInvSqrt(value));
  return FROM_float(r);
}

static PyObject *pycvInvert(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  int method = CV_LU;

  const char *keywords[] = { "src", "dst", "method", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|i", (char**)keywords, &pyobj_src, &pyobj_dst, &method))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_Invert
CVPY_VALIDATE_Invert();
#endif
  double r;
  ERRWRAP(r = cvInvert(src, dst, method));
  return FROM_double(r);
}

static PyObject *pycvIsInf(PyObject *self, PyObject *args)
{
  double value;

  if (!PyArg_ParseTuple(args, "d", &value))
    return NULL;
#ifdef CVPY_VALIDATE_IsInf
CVPY_VALIDATE_IsInf();
#endif
  int r;
  ERRWRAP(r = cvIsInf(value));
  return FROM_int(r);
}

static PyObject *pycvIsNaN(PyObject *self, PyObject *args)
{
  double value;

  if (!PyArg_ParseTuple(args, "d", &value))
    return NULL;
#ifdef CVPY_VALIDATE_IsNaN
CVPY_VALIDATE_IsNaN();
#endif
  int r;
  ERRWRAP(r = cvIsNaN(value));
  return FROM_int(r);
}

static PyObject *pycvKMeans2(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* samples;
  PyObject *pyobj_samples = NULL;
  int nclusters;
  CvArr* labels;
  PyObject *pyobj_labels = NULL;
  CvTermCriteria termcrit;
  PyObject *pyobj_termcrit = NULL;
  int attempts = 1;
  int flags = 0;
  CvArr* centers = NULL;
  PyObject *pyobj_centers = NULL;

  const char *keywords[] = { "samples", "nclusters", "labels", "termcrit", "attempts", "flags", "centers", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OiOO|iiO", (char**)keywords, &pyobj_samples, &nclusters, &pyobj_labels, &pyobj_termcrit, &attempts, &flags, &pyobj_centers))
    return NULL;
  if (!convert_to_CvArr(pyobj_samples, &samples, "samples")) return NULL;
  if (!convert_to_CvArr(pyobj_labels, &labels, "labels")) return NULL;
  if (!convert_to_CvTermCriteria(pyobj_termcrit, &termcrit, "termcrit")) return NULL;
  if ((pyobj_centers != NULL) && !convert_to_CvArr(pyobj_centers, &centers, "centers")) return NULL;
#ifdef CVPY_VALIDATE_KMeans2
CVPY_VALIDATE_KMeans2();
#endif
  double r;
  ERRWRAP(r = cvKMeans2(samples, nclusters, labels, termcrit, attempts, flags, centers));
  return FROM_double(r);
}

static PyObject *pycvKalmanCorrect(PyObject *self, PyObject *args)
{
  CvKalman* kalman;
  PyObject *pyobj_kalman = NULL;
  CvMat* measurement;
  PyObject *pyobj_measurement = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_kalman, &pyobj_measurement))
    return NULL;
  if (!convert_to_CvKalmanPTR(pyobj_kalman, &kalman, "kalman")) return NULL;
  if (!convert_to_CvMat(pyobj_measurement, &measurement, "measurement")) return NULL;
#ifdef CVPY_VALIDATE_KalmanCorrect
CVPY_VALIDATE_KalmanCorrect();
#endif
  ROCvMat* r;
  ERRWRAP(r = cvKalmanCorrect(kalman, measurement));
  return FROM_ROCvMatPTR(r);
}

static PyObject *pycvKalmanPredict(PyObject *self, PyObject *args, PyObject *kw)
{
  CvKalman* kalman;
  PyObject *pyobj_kalman = NULL;
  CvMat* control = NULL;
  PyObject *pyobj_control = NULL;

  const char *keywords[] = { "kalman", "control", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O|O", (char**)keywords, &pyobj_kalman, &pyobj_control))
    return NULL;
  if (!convert_to_CvKalmanPTR(pyobj_kalman, &kalman, "kalman")) return NULL;
  if ((pyobj_control != NULL) && !convert_to_CvMat(pyobj_control, &control, "control")) return NULL;
#ifdef CVPY_VALIDATE_KalmanPredict
CVPY_VALIDATE_KalmanPredict();
#endif
  ROCvMat* r;
  ERRWRAP(r = cvKalmanPredict(kalman, control));
  return FROM_ROCvMatPTR(r);
}

static PyObject *pycvLUT(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvArr* lut;
  PyObject *pyobj_lut = NULL;

  if (!PyArg_ParseTuple(args, "OOO", &pyobj_src, &pyobj_dst, &pyobj_lut))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if (!convert_to_CvArr(pyobj_lut, &lut, "lut")) return NULL;
#ifdef CVPY_VALIDATE_LUT
CVPY_VALIDATE_LUT();
#endif
  ERRWRAP(cvLUT(src, dst, lut));
  Py_RETURN_NONE;
}

static PyObject *pycvLaplace(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  int apertureSize = 3;

  const char *keywords[] = { "src", "dst", "apertureSize", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|i", (char**)keywords, &pyobj_src, &pyobj_dst, &apertureSize))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_Laplace
CVPY_VALIDATE_Laplace();
#endif
  ERRWRAP(cvLaplace(src, dst, apertureSize));
  Py_RETURN_NONE;
}

static PyObject *pycvLine(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* img;
  PyObject *pyobj_img = NULL;
  CvPoint pt1;
  PyObject *pyobj_pt1 = NULL;
  CvPoint pt2;
  PyObject *pyobj_pt2 = NULL;
  CvScalar color;
  PyObject *pyobj_color = NULL;
  int thickness = 1;
  int lineType = 8;
  int shift = 0;

  const char *keywords[] = { "img", "pt1", "pt2", "color", "thickness", "lineType", "shift", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOO|iii", (char**)keywords, &pyobj_img, &pyobj_pt1, &pyobj_pt2, &pyobj_color, &thickness, &lineType, &shift))
    return NULL;
  if (!convert_to_CvArr(pyobj_img, &img, "img")) return NULL;
  if (!convert_to_CvPoint(pyobj_pt1, &pt1, "pt1")) return NULL;
  if (!convert_to_CvPoint(pyobj_pt2, &pt2, "pt2")) return NULL;
  if (!convert_to_CvScalar(pyobj_color, &color, "color")) return NULL;
#ifdef CVPY_VALIDATE_Line
CVPY_VALIDATE_Line();
#endif
  ERRWRAP(cvLine(img, pt1, pt2, color, thickness, lineType, shift));
  Py_RETURN_NONE;
}

static PyObject *pycvLoad(PyObject *self, PyObject *args, PyObject *kw)
{
  char* filename;
  CvMemStorage* storage = NULL;
  PyObject *pyobj_storage = NULL;
  char* name = NULL;

  const char *keywords[] = { "filename", "storage", "name", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "s|Os", (char**)keywords, &filename, &pyobj_storage, &name))
    return NULL;
  if ((pyobj_storage != NULL) && !convert_to_CvMemStorage(pyobj_storage, &storage, "storage")) return NULL;
#ifdef CVPY_VALIDATE_Load
CVPY_VALIDATE_Load();
#endif
  generic r;
  ERRWRAP(r = cvLoad(filename, storage, name));
  return FROM_generic(r);
}

static PyObject *pycvLoadImage(PyObject *self, PyObject *args, PyObject *kw)
;

static PyObject *pycvLoadImageM(PyObject *self, PyObject *args, PyObject *kw)
;

static PyObject *pycvLog(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_src, &pyobj_dst))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_Log
CVPY_VALIDATE_Log();
#endif
  ERRWRAP(cvLog(src, dst));
  Py_RETURN_NONE;
}

static PyObject *pycvLogPolar(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvPoint2D32f center;
  PyObject *pyobj_center = NULL;
  double M;
  int flags = CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS;

  const char *keywords[] = { "src", "dst", "center", "M", "flags", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOd|i", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_center, &M, &flags))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if (!convert_to_CvPoint2D32f(pyobj_center, &center, "center")) return NULL;
#ifdef CVPY_VALIDATE_LogPolar
CVPY_VALIDATE_LogPolar();
#endif
  ERRWRAP(cvLogPolar(src, dst, center, M, flags));
  Py_RETURN_NONE;
}

static PyObject *pycvMahalonobis(PyObject *self, PyObject *args)
{
  CvArr* vec1;
  PyObject *pyobj_vec1 = NULL;
  CvArr* vec2;
  PyObject *pyobj_vec2 = NULL;
  CvArr* mat;
  PyObject *pyobj_mat = NULL;

  if (!PyArg_ParseTuple(args, "OOO", &pyobj_vec1, &pyobj_vec2, &pyobj_mat))
    return NULL;
  if (!convert_to_CvArr(pyobj_vec1, &vec1, "vec1")) return NULL;
  if (!convert_to_CvArr(pyobj_vec2, &vec2, "vec2")) return NULL;
  if (!convert_to_CvArr(pyobj_mat, &mat, "mat")) return NULL;
#ifdef CVPY_VALIDATE_Mahalonobis
CVPY_VALIDATE_Mahalonobis();
#endif
  ERRWRAP(cvMahalonobis(vec1, vec2, mat));
  Py_RETURN_NONE;
}

static PyObject *pycvMatMul(PyObject *self, PyObject *args)
{
  CvArr* src1;
  PyObject *pyobj_src1 = NULL;
  CvArr* src2;
  PyObject *pyobj_src2 = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;

  if (!PyArg_ParseTuple(args, "OOO", &pyobj_src1, &pyobj_src2, &pyobj_dst))
    return NULL;
  if (!convert_to_CvArr(pyobj_src1, &src1, "src1")) return NULL;
  if (!convert_to_CvArr(pyobj_src2, &src2, "src2")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_MatMul
CVPY_VALIDATE_MatMul();
#endif
  ERRWRAP(cvMatMul(src1, src2, dst));
  Py_RETURN_NONE;
}

static PyObject *pycvMatMulAdd(PyObject *self, PyObject *args)
{
  CvArr* src1;
  PyObject *pyobj_src1 = NULL;
  CvArr* src2;
  PyObject *pyobj_src2 = NULL;
  CvArr* src3;
  PyObject *pyobj_src3 = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;

  if (!PyArg_ParseTuple(args, "OOOO", &pyobj_src1, &pyobj_src2, &pyobj_src3, &pyobj_dst))
    return NULL;
  if (!convert_to_CvArr(pyobj_src1, &src1, "src1")) return NULL;
  if (!convert_to_CvArr(pyobj_src2, &src2, "src2")) return NULL;
  if (!convert_to_CvArr(pyobj_src3, &src3, "src3")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_MatMulAdd
CVPY_VALIDATE_MatMulAdd();
#endif
  ERRWRAP(cvMatMulAdd(src1, src2, src3, dst));
  Py_RETURN_NONE;
}

static PyObject *pycvMatchShapes(PyObject *self, PyObject *args, PyObject *kw)
{
  CvSeq* object1;
  PyObject *pyobj_object1 = NULL;
  CvSeq* object2;
  PyObject *pyobj_object2 = NULL;
  int method;
  double parameter = 0;

  const char *keywords[] = { "object1", "object2", "method", "parameter", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOi|d", (char**)keywords, &pyobj_object1, &pyobj_object2, &method, &parameter))
    return NULL;
  if (!convert_to_CvSeq(pyobj_object1, &object1, "object1")) return NULL;
  if (!convert_to_CvSeq(pyobj_object2, &object2, "object2")) return NULL;
#ifdef CVPY_VALIDATE_MatchShapes
CVPY_VALIDATE_MatchShapes();
#endif
  double r;
  ERRWRAP(r = cvMatchShapes(object1, object2, method, parameter));
  return FROM_double(r);
}

static PyObject *pycvMatchTemplate(PyObject *self, PyObject *args)
{
  CvArr* image;
  PyObject *pyobj_image = NULL;
  CvArr* templ;
  PyObject *pyobj_templ = NULL;
  CvArr* result;
  PyObject *pyobj_result = NULL;
  int method;

  if (!PyArg_ParseTuple(args, "OOOi", &pyobj_image, &pyobj_templ, &pyobj_result, &method))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvArr(pyobj_templ, &templ, "templ")) return NULL;
  if (!convert_to_CvArr(pyobj_result, &result, "result")) return NULL;
#ifdef CVPY_VALIDATE_MatchTemplate
CVPY_VALIDATE_MatchTemplate();
#endif
  ERRWRAP(cvMatchTemplate(image, templ, result, method));
  Py_RETURN_NONE;
}

static PyObject *pycvMax(PyObject *self, PyObject *args)
{
  CvArr* src1;
  PyObject *pyobj_src1 = NULL;
  CvArr* src2;
  PyObject *pyobj_src2 = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;

  if (!PyArg_ParseTuple(args, "OOO", &pyobj_src1, &pyobj_src2, &pyobj_dst))
    return NULL;
  if (!convert_to_CvArr(pyobj_src1, &src1, "src1")) return NULL;
  if (!convert_to_CvArr(pyobj_src2, &src2, "src2")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_Max
CVPY_VALIDATE_Max();
#endif
  ERRWRAP(cvMax(src1, src2, dst));
  Py_RETURN_NONE;
}

static PyObject *pycvMaxRect(PyObject *self, PyObject *args)
{
  CvRect* rect1;
  PyObject *pyobj_rect1 = NULL;
  CvRect* rect2;
  PyObject *pyobj_rect2 = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_rect1, &pyobj_rect2))
    return NULL;
  if (!convert_to_CvRectPTR(pyobj_rect1, &rect1, "rect1")) return NULL;
  if (!convert_to_CvRectPTR(pyobj_rect2, &rect2, "rect2")) return NULL;
#ifdef CVPY_VALIDATE_MaxRect
CVPY_VALIDATE_MaxRect();
#endif
  CvRect r;
  ERRWRAP(r = cvMaxRect(rect1, rect2));
  return FROM_CvRect(r);
}

static PyObject *pycvMaxS(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  double value;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;

  if (!PyArg_ParseTuple(args, "OdO", &pyobj_src, &value, &pyobj_dst))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_MaxS
CVPY_VALIDATE_MaxS();
#endif
  ERRWRAP(cvMaxS(src, value, dst));
  Py_RETURN_NONE;
}

static PyObject *pycvMeanShift(PyObject *self, PyObject *args)
{
  CvArr* prob_image;
  PyObject *pyobj_prob_image = NULL;
  CvRect window;
  PyObject *pyobj_window = NULL;
  CvTermCriteria criteria;
  PyObject *pyobj_criteria = NULL;
  CvConnectedComp comp;

  if (!PyArg_ParseTuple(args, "OOO", &pyobj_prob_image, &pyobj_window, &pyobj_criteria))
    return NULL;
  if (!convert_to_CvArr(pyobj_prob_image, &prob_image, "prob_image")) return NULL;
  if (!convert_to_CvRect(pyobj_window, &window, "window")) return NULL;
  if (!convert_to_CvTermCriteria(pyobj_criteria, &criteria, "criteria")) return NULL;
#ifdef CVPY_VALIDATE_MeanShift
CVPY_VALIDATE_MeanShift();
#endif
  ERRWRAP(cvMeanShift(prob_image, window, criteria, &comp));
  return FROM_CvConnectedComp(comp);
}

static PyObject *pycvMerge(PyObject *self, PyObject *args)
{
  CvArr* src0;
  PyObject *pyobj_src0 = NULL;
  CvArr* src1;
  PyObject *pyobj_src1 = NULL;
  CvArr* src2;
  PyObject *pyobj_src2 = NULL;
  CvArr* src3;
  PyObject *pyobj_src3 = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;

  if (!PyArg_ParseTuple(args, "OOOOO", &pyobj_src0, &pyobj_src1, &pyobj_src2, &pyobj_src3, &pyobj_dst))
    return NULL;
  if (!convert_to_CvArr(pyobj_src0, &src0, "src0")) return NULL;
  if (!convert_to_CvArr(pyobj_src1, &src1, "src1")) return NULL;
  if (!convert_to_CvArr(pyobj_src2, &src2, "src2")) return NULL;
  if (!convert_to_CvArr(pyobj_src3, &src3, "src3")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_Merge
CVPY_VALIDATE_Merge();
#endif
  ERRWRAP(cvMerge(src0, src1, src2, src3, dst));
  Py_RETURN_NONE;
}

static PyObject *pycvMin(PyObject *self, PyObject *args)
{
  CvArr* src1;
  PyObject *pyobj_src1 = NULL;
  CvArr* src2;
  PyObject *pyobj_src2 = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;

  if (!PyArg_ParseTuple(args, "OOO", &pyobj_src1, &pyobj_src2, &pyobj_dst))
    return NULL;
  if (!convert_to_CvArr(pyobj_src1, &src1, "src1")) return NULL;
  if (!convert_to_CvArr(pyobj_src2, &src2, "src2")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_Min
CVPY_VALIDATE_Min();
#endif
  ERRWRAP(cvMin(src1, src2, dst));
  Py_RETURN_NONE;
}

static PyObject *pycvMinAreaRect2(PyObject *self, PyObject *args, PyObject *kw)
{
  cvarrseq points;
  PyObject *pyobj_points = NULL;
  CvMemStorage* storage = NULL;
  PyObject *pyobj_storage = NULL;

  const char *keywords[] = { "points", "storage", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O|O", (char**)keywords, &pyobj_points, &pyobj_storage))
    return NULL;
  if (!convert_to_cvarrseq(pyobj_points, &points, "points")) return NULL;
  if ((pyobj_storage != NULL) && !convert_to_CvMemStorage(pyobj_storage, &storage, "storage")) return NULL;
#ifdef CVPY_VALIDATE_MinAreaRect2
CVPY_VALIDATE_MinAreaRect2();
#endif
  CvBox2D r;
  ERRWRAP(r = cvMinAreaRect2(points.seq, storage));
  return FROM_CvBox2D(r);
}

static PyObject *pycvMinEnclosingCircle(PyObject *self, PyObject *args)
{
  cvarrseq points;
  PyObject *pyobj_points = NULL;
  CvPoint2D32f center;
  float radius;

  if (!PyArg_ParseTuple(args, "O", &pyobj_points))
    return NULL;
  if (!convert_to_cvarrseq(pyobj_points, &points, "points")) return NULL;
#ifdef CVPY_VALIDATE_MinEnclosingCircle
CVPY_VALIDATE_MinEnclosingCircle();
#endif
  int r;
  ERRWRAP(r = cvMinEnclosingCircle(points.seq, &center, &radius));
  return Py_BuildValue("NNN", FROM_int(r), FROM_CvPoint2D32f(center), FROM_float(radius));
}

static PyObject *pycvMinMaxLoc(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  double minVal;
  double maxVal;
  CvPoint minLoc;
  CvPoint maxLoc;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;

  const char *keywords[] = { "arr", "mask", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O|O", (char**)keywords, &pyobj_arr, &pyobj_mask))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_MinMaxLoc
CVPY_VALIDATE_MinMaxLoc();
#endif
  ERRWRAP(cvMinMaxLoc(arr, &minVal, &maxVal, &minLoc, &maxLoc, mask));
  return Py_BuildValue("NNNN", FROM_double(minVal), FROM_double(maxVal), FROM_CvPoint(minLoc), FROM_CvPoint(maxLoc));
}

static PyObject *pycvMinS(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  double value;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;

  if (!PyArg_ParseTuple(args, "OdO", &pyobj_src, &value, &pyobj_dst))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_MinS
CVPY_VALIDATE_MinS();
#endif
  ERRWRAP(cvMinS(src, value, dst));
  Py_RETURN_NONE;
}

static PyObject *pycvMixChannels(PyObject *self, PyObject *args)
{
  cvarr_count src;
  PyObject *pyobj_src = NULL;
  cvarr_count dst;
  PyObject *pyobj_dst = NULL;
  intpair fromTo;
  PyObject *pyobj_fromTo = NULL;

  if (!PyArg_ParseTuple(args, "OOO", &pyobj_src, &pyobj_dst, &pyobj_fromTo))
    return NULL;
  if (!convert_to_cvarr_count(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_cvarr_count(pyobj_dst, &dst, "dst")) return NULL;
  if (!convert_to_intpair(pyobj_fromTo, &fromTo, "fromTo")) return NULL;
#ifdef CVPY_VALIDATE_MixChannels
CVPY_VALIDATE_MixChannels();
#endif
  ERRWRAP(cvMixChannels((const CvArr **)src.cvarr,src.count, dst.cvarr,dst.count, fromTo.pairs,fromTo.count));
  Py_RETURN_NONE;
}

static PyObject *pycvMoments(PyObject *self, PyObject *args, PyObject *kw)
{
  cvarrseq arr;
  PyObject *pyobj_arr = NULL;
  CvMoments moments;
  int binary = 0;

  const char *keywords[] = { "arr", "binary", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O|i", (char**)keywords, &pyobj_arr, &binary))
    return NULL;
  if (!convert_to_cvarrseq(pyobj_arr, &arr, "arr")) return NULL;
#ifdef CVPY_VALIDATE_Moments
CVPY_VALIDATE_Moments();
#endif
  ERRWRAP(cvMoments(arr.seq, &moments, binary));
  return FROM_CvMoments(moments);
}

static PyObject *pycvMorphologyEx(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvArr* temp;
  PyObject *pyobj_temp = NULL;
  IplConvKernel* element;
  PyObject *pyobj_element = NULL;
  int operation;
  int iterations = 1;

  const char *keywords[] = { "src", "dst", "temp", "element", "operation", "iterations", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOOi|i", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_temp, &pyobj_element, &operation, &iterations))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if (!convert_to_CvArr(pyobj_temp, &temp, "temp")) return NULL;
  if (!convert_to_IplConvKernelPTR(pyobj_element, &element, "element")) return NULL;
#ifdef CVPY_VALIDATE_MorphologyEx
CVPY_VALIDATE_MorphologyEx();
#endif
  ERRWRAP(cvMorphologyEx(src, dst, temp, element, operation, iterations));
  Py_RETURN_NONE;
}

static PyObject *pycvMoveWindow(PyObject *self, PyObject *args)
{
  char* name;
  int x;
  int y;

  if (!PyArg_ParseTuple(args, "sii", &name, &x, &y))
    return NULL;
#ifdef CVPY_VALIDATE_MoveWindow
CVPY_VALIDATE_MoveWindow();
#endif
  ERRWRAP(cvMoveWindow(name, x, y));
  Py_RETURN_NONE;
}

static PyObject *pycvMul(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src1;
  PyObject *pyobj_src1 = NULL;
  CvArr* src2;
  PyObject *pyobj_src2 = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  double scale = 1.0;

  const char *keywords[] = { "src1", "src2", "dst", "scale", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|d", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &scale))
    return NULL;
  if (!convert_to_CvArr(pyobj_src1, &src1, "src1")) return NULL;
  if (!convert_to_CvArr(pyobj_src2, &src2, "src2")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_Mul
CVPY_VALIDATE_Mul();
#endif
  ERRWRAP(cvMul(src1, src2, dst, scale));
  Py_RETURN_NONE;
}

static PyObject *pycvMulSpectrums(PyObject *self, PyObject *args)
{
  CvArr* src1;
  PyObject *pyobj_src1 = NULL;
  CvArr* src2;
  PyObject *pyobj_src2 = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  int flags;

  if (!PyArg_ParseTuple(args, "OOOi", &pyobj_src1, &pyobj_src2, &pyobj_dst, &flags))
    return NULL;
  if (!convert_to_CvArr(pyobj_src1, &src1, "src1")) return NULL;
  if (!convert_to_CvArr(pyobj_src2, &src2, "src2")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_MulSpectrums
CVPY_VALIDATE_MulSpectrums();
#endif
  ERRWRAP(cvMulSpectrums(src1, src2, dst, flags));
  Py_RETURN_NONE;
}

static PyObject *pycvMulTransposed(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  int order;
  CvArr* delta = NULL;
  PyObject *pyobj_delta = NULL;
  double scale = 1.0;

  const char *keywords[] = { "src", "dst", "order", "delta", "scale", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOi|Od", (char**)keywords, &pyobj_src, &pyobj_dst, &order, &pyobj_delta, &scale))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if ((pyobj_delta != NULL) && !convert_to_CvArr(pyobj_delta, &delta, "delta")) return NULL;
#ifdef CVPY_VALIDATE_MulTransposed
CVPY_VALIDATE_MulTransposed();
#endif
  ERRWRAP(cvMulTransposed(src, dst, order, delta, scale));
  Py_RETURN_NONE;
}

static PyObject *pycvMultiplyAcc(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* image1;
  PyObject *pyobj_image1 = NULL;
  CvArr* image2;
  PyObject *pyobj_image2 = NULL;
  CvArr* acc;
  PyObject *pyobj_acc = NULL;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;

  const char *keywords[] = { "image1", "image2", "acc", "mask", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|O", (char**)keywords, &pyobj_image1, &pyobj_image2, &pyobj_acc, &pyobj_mask))
    return NULL;
  if (!convert_to_CvArr(pyobj_image1, &image1, "image1")) return NULL;
  if (!convert_to_CvArr(pyobj_image2, &image2, "image2")) return NULL;
  if (!convert_to_CvArr(pyobj_acc, &acc, "acc")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_MultiplyAcc
CVPY_VALIDATE_MultiplyAcc();
#endif
  ERRWRAP(cvMultiplyAcc(image1, image2, acc, mask));
  Py_RETURN_NONE;
}

static PyObject *pycvNamedWindow(PyObject *self, PyObject *args, PyObject *kw)
{
  char* name;
  int flags = CV_WINDOW_AUTOSIZE;

  const char *keywords[] = { "name", "flags", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "s|i", (char**)keywords, &name, &flags))
    return NULL;
#ifdef CVPY_VALIDATE_NamedWindow
CVPY_VALIDATE_NamedWindow();
#endif
  ERRWRAP(cvNamedWindow(name, flags));
  Py_RETURN_NONE;
}

static PyObject *pycvNorm(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* arr1;
  PyObject *pyobj_arr1 = NULL;
  CvArr* arr2;
  PyObject *pyobj_arr2 = NULL;
  int normType = CV_L2;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;

  const char *keywords[] = { "arr1", "arr2", "normType", "mask", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|iO", (char**)keywords, &pyobj_arr1, &pyobj_arr2, &normType, &pyobj_mask))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr1, &arr1, "arr1")) return NULL;
  if (!convert_to_CvArr(pyobj_arr2, &arr2, "arr2")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_Norm
CVPY_VALIDATE_Norm();
#endif
  double r;
  ERRWRAP(r = cvNorm(arr1, arr2, normType, mask));
  return FROM_double(r);
}

static PyObject *pycvNormalize(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  double a = 1.0;
  double b = 0.0;
  int norm_type = CV_L2;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;

  const char *keywords[] = { "src", "dst", "a", "b", "norm_type", "mask", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|ddiO", (char**)keywords, &pyobj_src, &pyobj_dst, &a, &b, &norm_type, &pyobj_mask))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_Normalize
CVPY_VALIDATE_Normalize();
#endif
  ERRWRAP(cvNormalize(src, dst, a, b, norm_type, mask));
  Py_RETURN_NONE;
}

static PyObject *pycvNormalizeHist(PyObject *self, PyObject *args)
{
  CvHistogram* hist;
  PyObject *pyobj_hist = NULL;
  double factor;

  if (!PyArg_ParseTuple(args, "Od", &pyobj_hist, &factor))
    return NULL;
  if (!convert_to_CvHistogram(pyobj_hist, &hist, "hist")) return NULL;
#ifdef CVPY_VALIDATE_NormalizeHist
CVPY_VALIDATE_NormalizeHist();
#endif
  ERRWRAP(cvNormalizeHist(hist, factor));
  Py_RETURN_NONE;
}

static PyObject *pycvNot(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_src, &pyobj_dst))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_Not
CVPY_VALIDATE_Not();
#endif
  ERRWRAP(cvNot(src, dst));
  Py_RETURN_NONE;
}

static PyObject *pycvOr(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src1;
  PyObject *pyobj_src1 = NULL;
  CvArr* src2;
  PyObject *pyobj_src2 = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;

  const char *keywords[] = { "src1", "src2", "dst", "mask", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|O", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &pyobj_mask))
    return NULL;
  if (!convert_to_CvArr(pyobj_src1, &src1, "src1")) return NULL;
  if (!convert_to_CvArr(pyobj_src2, &src2, "src2")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_Or
CVPY_VALIDATE_Or();
#endif
  ERRWRAP(cvOr(src1, src2, dst, mask));
  Py_RETURN_NONE;
}

static PyObject *pycvOrS(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvScalar value;
  PyObject *pyobj_value = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;

  const char *keywords[] = { "src", "value", "dst", "mask", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|O", (char**)keywords, &pyobj_src, &pyobj_value, &pyobj_dst, &pyobj_mask))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvScalar(pyobj_value, &value, "value")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_OrS
CVPY_VALIDATE_OrS();
#endif
  ERRWRAP(cvOrS(src, value, dst, mask));
  Py_RETURN_NONE;
}

static PyObject *pycvPOSIT(PyObject *self, PyObject *args)
{
  CvPOSITObject* posit_object;
  PyObject *pyobj_posit_object = NULL;
  CvPoint2D32f* imagePoints;
  PyObject *pyobj_imagePoints = NULL;
  double focal_length;
  CvTermCriteria criteria;
  PyObject *pyobj_criteria = NULL;
  CvMatr32f_i rotationMatrix;
  CvVect32f_i translation_vector;

  if (!PyArg_ParseTuple(args, "OOdO", &pyobj_posit_object, &pyobj_imagePoints, &focal_length, &pyobj_criteria))
    return NULL;
  if (!convert_to_CvPOSITObjectPTR(pyobj_posit_object, &posit_object, "posit_object")) return NULL;
  if (!convert_to_CvPoint2D32fPTR(pyobj_imagePoints, &imagePoints, "imagePoints")) return NULL;
  if (!convert_to_CvTermCriteria(pyobj_criteria, &criteria, "criteria")) return NULL;
#ifdef CVPY_VALIDATE_POSIT
CVPY_VALIDATE_POSIT();
#endif
  ERRWRAP(cvPOSIT(posit_object, imagePoints, focal_length, criteria, rotationMatrix, translation_vector));
  return Py_BuildValue("NN", FROM_CvMatr32f_i(rotationMatrix), FROM_CvVect32f_i(translation_vector));
}

static PyObject *pycvPerspectiveTransform(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvMat* mat;
  PyObject *pyobj_mat = NULL;

  if (!PyArg_ParseTuple(args, "OOO", &pyobj_src, &pyobj_dst, &pyobj_mat))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if (!convert_to_CvMat(pyobj_mat, &mat, "mat")) return NULL;
#ifdef CVPY_VALIDATE_PerspectiveTransform
CVPY_VALIDATE_PerspectiveTransform();
#endif
  ERRWRAP(cvPerspectiveTransform(src, dst, mat));
  Py_RETURN_NONE;
}

static PyObject *pycvPointPolygonTest(PyObject *self, PyObject *args)
{
  cvarrseq contour;
  PyObject *pyobj_contour = NULL;
  CvPoint2D32f pt;
  PyObject *pyobj_pt = NULL;
  int measure_dist;

  if (!PyArg_ParseTuple(args, "OOi", &pyobj_contour, &pyobj_pt, &measure_dist))
    return NULL;
  if (!convert_to_cvarrseq(pyobj_contour, &contour, "contour")) return NULL;
  if (!convert_to_CvPoint2D32f(pyobj_pt, &pt, "pt")) return NULL;
#ifdef CVPY_VALIDATE_PointPolygonTest
CVPY_VALIDATE_PointPolygonTest();
#endif
  double r;
  ERRWRAP(r = cvPointPolygonTest(contour.seq, pt, measure_dist));
  return FROM_double(r);
}

static PyObject *pycvPolarToCart(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* magnitude;
  PyObject *pyobj_magnitude = NULL;
  CvArr* angle;
  PyObject *pyobj_angle = NULL;
  CvArr* x;
  PyObject *pyobj_x = NULL;
  CvArr* y;
  PyObject *pyobj_y = NULL;
  int angleInDegrees = 0;

  const char *keywords[] = { "magnitude", "angle", "x", "y", "angleInDegrees", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOO|i", (char**)keywords, &pyobj_magnitude, &pyobj_angle, &pyobj_x, &pyobj_y, &angleInDegrees))
    return NULL;
  if (!convert_to_CvArr(pyobj_magnitude, &magnitude, "magnitude")) return NULL;
  if (!convert_to_CvArr(pyobj_angle, &angle, "angle")) return NULL;
  if (!convert_to_CvArr(pyobj_x, &x, "x")) return NULL;
  if (!convert_to_CvArr(pyobj_y, &y, "y")) return NULL;
#ifdef CVPY_VALIDATE_PolarToCart
CVPY_VALIDATE_PolarToCart();
#endif
  ERRWRAP(cvPolarToCart(magnitude, angle, x, y, angleInDegrees));
  Py_RETURN_NONE;
}

static PyObject *pycvPolyLine(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* img;
  PyObject *pyobj_img = NULL;
  pts_npts_contours polys;
  PyObject *pyobj_polys = NULL;
  int is_closed;
  CvScalar color;
  PyObject *pyobj_color = NULL;
  int thickness = 1;
  int lineType = 8;
  int shift = 0;

  const char *keywords[] = { "img", "polys", "is_closed", "color", "thickness", "lineType", "shift", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOiO|iii", (char**)keywords, &pyobj_img, &pyobj_polys, &is_closed, &pyobj_color, &thickness, &lineType, &shift))
    return NULL;
  if (!convert_to_CvArr(pyobj_img, &img, "img")) return NULL;
  if (!convert_to_pts_npts_contours(pyobj_polys, &polys, "polys")) return NULL;
  if (!convert_to_CvScalar(pyobj_color, &color, "color")) return NULL;
#ifdef CVPY_VALIDATE_PolyLine
CVPY_VALIDATE_PolyLine();
#endif
  ERRWRAP(cvPolyLine(img, polys.pts,polys.npts,polys.contours, is_closed, color, thickness, lineType, shift));
  Py_RETURN_NONE;
}

static PyObject *pycvPow(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  double power;

  if (!PyArg_ParseTuple(args, "OOd", &pyobj_src, &pyobj_dst, &power))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_Pow
CVPY_VALIDATE_Pow();
#endif
  ERRWRAP(cvPow(src, dst, power));
  Py_RETURN_NONE;
}

static PyObject *pycvPreCornerDetect(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* image;
  PyObject *pyobj_image = NULL;
  CvArr* corners;
  PyObject *pyobj_corners = NULL;
  int apertureSize = 3;

  const char *keywords[] = { "image", "corners", "apertureSize", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|i", (char**)keywords, &pyobj_image, &pyobj_corners, &apertureSize))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvArr(pyobj_corners, &corners, "corners")) return NULL;
#ifdef CVPY_VALIDATE_PreCornerDetect
CVPY_VALIDATE_PreCornerDetect();
#endif
  ERRWRAP(cvPreCornerDetect(image, corners, apertureSize));
  Py_RETURN_NONE;
}

static PyObject *pycvProjectPCA(PyObject *self, PyObject *args)
{
  CvArr* data;
  PyObject *pyobj_data = NULL;
  CvArr* avg;
  PyObject *pyobj_avg = NULL;
  CvArr* eigenvectors;
  PyObject *pyobj_eigenvectors = NULL;
  CvArr* result;
  PyObject *pyobj_result = NULL;

  if (!PyArg_ParseTuple(args, "OOOO", &pyobj_data, &pyobj_avg, &pyobj_eigenvectors, &pyobj_result))
    return NULL;
  if (!convert_to_CvArr(pyobj_data, &data, "data")) return NULL;
  if (!convert_to_CvArr(pyobj_avg, &avg, "avg")) return NULL;
  if (!convert_to_CvArr(pyobj_eigenvectors, &eigenvectors, "eigenvectors")) return NULL;
  if (!convert_to_CvArr(pyobj_result, &result, "result")) return NULL;
#ifdef CVPY_VALIDATE_ProjectPCA
CVPY_VALIDATE_ProjectPCA();
#endif
  ERRWRAP(cvProjectPCA(data, avg, eigenvectors, result));
  Py_RETURN_NONE;
}

static PyObject *pycvProjectPoints2(PyObject *self, PyObject *args, PyObject *kw)
{
  CvMat* objectPoints;
  PyObject *pyobj_objectPoints = NULL;
  CvMat* rvec;
  PyObject *pyobj_rvec = NULL;
  CvMat* tvec;
  PyObject *pyobj_tvec = NULL;
  CvMat* cameraMatrix;
  PyObject *pyobj_cameraMatrix = NULL;
  CvMat* distCoeffs;
  PyObject *pyobj_distCoeffs = NULL;
  CvMat* imagePoints;
  PyObject *pyobj_imagePoints = NULL;
  CvMat* dpdrot = NULL;
  PyObject *pyobj_dpdrot = NULL;
  CvMat* dpdt = NULL;
  PyObject *pyobj_dpdt = NULL;
  CvMat* dpdf = NULL;
  PyObject *pyobj_dpdf = NULL;
  CvMat* dpdc = NULL;
  PyObject *pyobj_dpdc = NULL;
  CvMat* dpddist = NULL;
  PyObject *pyobj_dpddist = NULL;

  const char *keywords[] = { "objectPoints", "rvec", "tvec", "cameraMatrix", "distCoeffs", "imagePoints", "dpdrot", "dpdt", "dpdf", "dpdc", "dpddist", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOOOO|OOOOO", (char**)keywords, &pyobj_objectPoints, &pyobj_rvec, &pyobj_tvec, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_imagePoints, &pyobj_dpdrot, &pyobj_dpdt, &pyobj_dpdf, &pyobj_dpdc, &pyobj_dpddist))
    return NULL;
  if (!convert_to_CvMat(pyobj_objectPoints, &objectPoints, "objectPoints")) return NULL;
  if (!convert_to_CvMat(pyobj_rvec, &rvec, "rvec")) return NULL;
  if (!convert_to_CvMat(pyobj_tvec, &tvec, "tvec")) return NULL;
  if (!convert_to_CvMat(pyobj_cameraMatrix, &cameraMatrix, "cameraMatrix")) return NULL;
  if (!convert_to_CvMat(pyobj_distCoeffs, &distCoeffs, "distCoeffs")) return NULL;
  if (!convert_to_CvMat(pyobj_imagePoints, &imagePoints, "imagePoints")) return NULL;
  if ((pyobj_dpdrot != NULL) && !convert_to_CvMat(pyobj_dpdrot, &dpdrot, "dpdrot")) return NULL;
  if ((pyobj_dpdt != NULL) && !convert_to_CvMat(pyobj_dpdt, &dpdt, "dpdt")) return NULL;
  if ((pyobj_dpdf != NULL) && !convert_to_CvMat(pyobj_dpdf, &dpdf, "dpdf")) return NULL;
  if ((pyobj_dpdc != NULL) && !convert_to_CvMat(pyobj_dpdc, &dpdc, "dpdc")) return NULL;
  if ((pyobj_dpddist != NULL) && !convert_to_CvMat(pyobj_dpddist, &dpddist, "dpddist")) return NULL;
#ifdef CVPY_VALIDATE_ProjectPoints2
CVPY_VALIDATE_ProjectPoints2();
#endif
  ERRWRAP(cvProjectPoints2(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints, dpdrot, dpdt, dpdf, dpdc, dpddist));
  Py_RETURN_NONE;
}

static PyObject *pycvPutText(PyObject *self, PyObject *args)
{
  CvArr* img;
  PyObject *pyobj_img = NULL;
  char* text;
  CvPoint org;
  PyObject *pyobj_org = NULL;
  CvFont* font;
  PyObject *pyobj_font = NULL;
  CvScalar color;
  PyObject *pyobj_color = NULL;

  if (!PyArg_ParseTuple(args, "OsOOO", &pyobj_img, &text, &pyobj_org, &pyobj_font, &pyobj_color))
    return NULL;
  if (!convert_to_CvArr(pyobj_img, &img, "img")) return NULL;
  if (!convert_to_CvPoint(pyobj_org, &org, "org")) return NULL;
  if (!convert_to_CvFontPTR(pyobj_font, &font, "font")) return NULL;
  if (!convert_to_CvScalar(pyobj_color, &color, "color")) return NULL;
#ifdef CVPY_VALIDATE_PutText
CVPY_VALIDATE_PutText();
#endif
  ERRWRAP(cvPutText(img, text, org, font, color));
  Py_RETURN_NONE;
}

static PyObject *pycvPyrDown(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  int filter = CV_GAUSSIAN_5x5;

  const char *keywords[] = { "src", "dst", "filter", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|i", (char**)keywords, &pyobj_src, &pyobj_dst, &filter))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_PyrDown
CVPY_VALIDATE_PyrDown();
#endif
  ERRWRAP(cvPyrDown(src, dst, filter));
  Py_RETURN_NONE;
}

static PyObject *pycvPyrMeanShiftFiltering(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  double sp;
  double sr;
  int max_level = 1;
  CvTermCriteria termcrit = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,5,1);
  PyObject *pyobj_termcrit = NULL;

  const char *keywords[] = { "src", "dst", "sp", "sr", "max_level", "termcrit", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOdd|iO", (char**)keywords, &pyobj_src, &pyobj_dst, &sp, &sr, &max_level, &pyobj_termcrit))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if ((pyobj_termcrit != NULL) && !convert_to_CvTermCriteria(pyobj_termcrit, &termcrit, "termcrit")) return NULL;
#ifdef CVPY_VALIDATE_PyrMeanShiftFiltering
CVPY_VALIDATE_PyrMeanShiftFiltering();
#endif
  ERRWRAP(cvPyrMeanShiftFiltering(src, dst, sp, sr, max_level, termcrit));
  Py_RETURN_NONE;
}

static PyObject *pycvPyrSegmentation(PyObject *self, PyObject *args)
{
  IplImage* src;
  PyObject *pyobj_src = NULL;
  IplImage* dst;
  PyObject *pyobj_dst = NULL;
  CvMemStorage* storage;
  PyObject *pyobj_storage = NULL;
  CvSeq* comp;
  int level;
  double threshold1;
  double threshold2;

  if (!PyArg_ParseTuple(args, "OOOidd", &pyobj_src, &pyobj_dst, &pyobj_storage, &level, &threshold1, &threshold2))
    return NULL;
  if (!convert_to_IplImage(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_IplImage(pyobj_dst, &dst, "dst")) return NULL;
  if (!convert_to_CvMemStorage(pyobj_storage, &storage, "storage")) return NULL;
#ifdef CVPY_VALIDATE_PyrSegmentation
CVPY_VALIDATE_PyrSegmentation();
#endif
  ERRWRAP(cvPyrSegmentation(src, dst, storage, &comp, level, threshold1, threshold2));
  return FROM_CvSeqPTR(comp);
}

static PyObject *pycvPyrUp(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  int filter = CV_GAUSSIAN_5x5;

  const char *keywords[] = { "src", "dst", "filter", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|i", (char**)keywords, &pyobj_src, &pyobj_dst, &filter))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_PyrUp
CVPY_VALIDATE_PyrUp();
#endif
  ERRWRAP(cvPyrUp(src, dst, filter));
  Py_RETURN_NONE;
}

static PyObject *pycvQueryFrame(PyObject *self, PyObject *args)
{
  CvCapture* capture;
  PyObject *pyobj_capture = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_capture))
    return NULL;
  if (!convert_to_CvCapturePTR(pyobj_capture, &capture, "capture")) return NULL;
#ifdef CVPY_VALIDATE_QueryFrame
CVPY_VALIDATE_QueryFrame();
#endif
  ROIplImage* r;
  ERRWRAP(r = cvQueryFrame(capture));
  return FROM_ROIplImagePTR(r);
}

static PyObject *pycvQueryHistValue_1D(PyObject *self, PyObject *args)
{
  CvHistogram* hist;
  PyObject *pyobj_hist = NULL;
  int idx0;

  if (!PyArg_ParseTuple(args, "Oi", &pyobj_hist, &idx0))
    return NULL;
  if (!convert_to_CvHistogram(pyobj_hist, &hist, "hist")) return NULL;
#ifdef CVPY_VALIDATE_QueryHistValue_1D
CVPY_VALIDATE_QueryHistValue_1D();
#endif
  double r;
  ERRWRAP(r = cvQueryHistValue_1D(hist, idx0));
  return FROM_double(r);
}

static PyObject *pycvQueryHistValue_2D(PyObject *self, PyObject *args)
{
  CvHistogram* hist;
  PyObject *pyobj_hist = NULL;
  int idx0;
  int idx1;

  if (!PyArg_ParseTuple(args, "Oii", &pyobj_hist, &idx0, &idx1))
    return NULL;
  if (!convert_to_CvHistogram(pyobj_hist, &hist, "hist")) return NULL;
#ifdef CVPY_VALIDATE_QueryHistValue_2D
CVPY_VALIDATE_QueryHistValue_2D();
#endif
  double r;
  ERRWRAP(r = cvQueryHistValue_2D(hist, idx0, idx1));
  return FROM_double(r);
}

static PyObject *pycvQueryHistValue_3D(PyObject *self, PyObject *args)
{
  CvHistogram* hist;
  PyObject *pyobj_hist = NULL;
  int idx0;
  int idx1;
  int idx2;

  if (!PyArg_ParseTuple(args, "Oiii", &pyobj_hist, &idx0, &idx1, &idx2))
    return NULL;
  if (!convert_to_CvHistogram(pyobj_hist, &hist, "hist")) return NULL;
#ifdef CVPY_VALIDATE_QueryHistValue_3D
CVPY_VALIDATE_QueryHistValue_3D();
#endif
  double r;
  ERRWRAP(r = cvQueryHistValue_3D(hist, idx0, idx1, idx2));
  return FROM_double(r);
}

static PyObject *pycvQueryHistValue_nD(PyObject *self, PyObject *args)
{
  CvHistogram* hist;
  PyObject *pyobj_hist = NULL;
  ints idx;
  PyObject *pyobj_idx = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_hist, &pyobj_idx))
    return NULL;
  if (!convert_to_CvHistogram(pyobj_hist, &hist, "hist")) return NULL;
  if (!convert_to_ints(pyobj_idx, &idx, "idx")) return NULL;
#ifdef CVPY_VALIDATE_QueryHistValue_nD
CVPY_VALIDATE_QueryHistValue_nD();
#endif
  double r;
  ERRWRAP(r = cvQueryHistValue_nD(hist, idx.i));
  return FROM_double(r);
}

static PyObject *pycvRNG(PyObject *self, PyObject *args, PyObject *kw)
{
  int64 seed = -1LL;

  const char *keywords[] = { "seed", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "|L", (char**)keywords, &seed))
    return NULL;
#ifdef CVPY_VALIDATE_RNG
CVPY_VALIDATE_RNG();
#endif
  CvRNG r;
  ERRWRAP(r = cvRNG(seed));
  return FROM_CvRNG(r);
}

static PyObject *pycvRQDecomp3x3(PyObject *self, PyObject *args, PyObject *kw)
{
  CvMat* M;
  PyObject *pyobj_M = NULL;
  CvMat* R;
  PyObject *pyobj_R = NULL;
  CvMat* Q;
  PyObject *pyobj_Q = NULL;
  CvMat* Qx = NULL;
  PyObject *pyobj_Qx = NULL;
  CvMat* Qy = NULL;
  PyObject *pyobj_Qy = NULL;
  CvMat* Qz = NULL;
  PyObject *pyobj_Qz = NULL;
  CvPoint3D64f eulerAngles;

  const char *keywords[] = { "M", "R", "Q", "Qx", "Qy", "Qz", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|OOO", (char**)keywords, &pyobj_M, &pyobj_R, &pyobj_Q, &pyobj_Qx, &pyobj_Qy, &pyobj_Qz))
    return NULL;
  if (!convert_to_CvMat(pyobj_M, &M, "M")) return NULL;
  if (!convert_to_CvMat(pyobj_R, &R, "R")) return NULL;
  if (!convert_to_CvMat(pyobj_Q, &Q, "Q")) return NULL;
  if ((pyobj_Qx != NULL) && !convert_to_CvMat(pyobj_Qx, &Qx, "Qx")) return NULL;
  if ((pyobj_Qy != NULL) && !convert_to_CvMat(pyobj_Qy, &Qy, "Qy")) return NULL;
  if ((pyobj_Qz != NULL) && !convert_to_CvMat(pyobj_Qz, &Qz, "Qz")) return NULL;
#ifdef CVPY_VALIDATE_RQDecomp3x3
CVPY_VALIDATE_RQDecomp3x3();
#endif
  ERRWRAP(cvRQDecomp3x3(M, R, Q, Qx, Qy, Qz, &eulerAngles));
  return FROM_CvPoint3D64f(eulerAngles);
}

static PyObject *pycvRandArr(PyObject *self, PyObject *args)
{
  CvRNG* rng;
  PyObject *pyobj_rng = NULL;
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  int distType;
  CvScalar param1;
  PyObject *pyobj_param1 = NULL;
  CvScalar param2;
  PyObject *pyobj_param2 = NULL;

  if (!PyArg_ParseTuple(args, "OOiOO", &pyobj_rng, &pyobj_arr, &distType, &pyobj_param1, &pyobj_param2))
    return NULL;
  if (!convert_to_CvRNGPTR(pyobj_rng, &rng, "rng")) return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
  if (!convert_to_CvScalar(pyobj_param1, &param1, "param1")) return NULL;
  if (!convert_to_CvScalar(pyobj_param2, &param2, "param2")) return NULL;
#ifdef CVPY_VALIDATE_RandArr
CVPY_VALIDATE_RandArr();
#endif
  ERRWRAP(cvRandArr(rng, arr, distType, param1, param2));
  Py_RETURN_NONE;
}

static PyObject *pycvRandInt(PyObject *self, PyObject *args)
{
  CvRNG* rng;
  PyObject *pyobj_rng = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_rng))
    return NULL;
  if (!convert_to_CvRNGPTR(pyobj_rng, &rng, "rng")) return NULL;
#ifdef CVPY_VALIDATE_RandInt
CVPY_VALIDATE_RandInt();
#endif
  unsigned r;
  ERRWRAP(r = cvRandInt(rng));
  return FROM_unsigned(r);
}

static PyObject *pycvRandReal(PyObject *self, PyObject *args)
{
  CvRNG* rng;
  PyObject *pyobj_rng = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_rng))
    return NULL;
  if (!convert_to_CvRNGPTR(pyobj_rng, &rng, "rng")) return NULL;
#ifdef CVPY_VALIDATE_RandReal
CVPY_VALIDATE_RandReal();
#endif
  double r;
  ERRWRAP(r = cvRandReal(rng));
  return FROM_double(r);
}

static PyObject *pycvRandShuffle(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* mat;
  PyObject *pyobj_mat = NULL;
  CvRNG* rng;
  PyObject *pyobj_rng = NULL;
  double iter_factor = 1.0;

  const char *keywords[] = { "mat", "rng", "iter_factor", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|d", (char**)keywords, &pyobj_mat, &pyobj_rng, &iter_factor))
    return NULL;
  if (!convert_to_CvArr(pyobj_mat, &mat, "mat")) return NULL;
  if (!convert_to_CvRNGPTR(pyobj_rng, &rng, "rng")) return NULL;
#ifdef CVPY_VALIDATE_RandShuffle
CVPY_VALIDATE_RandShuffle();
#endif
  ERRWRAP(cvRandShuffle(mat, rng, iter_factor));
  Py_RETURN_NONE;
}

static PyObject *pycvRange(PyObject *self, PyObject *args)
{
  CvArr* mat;
  PyObject *pyobj_mat = NULL;
  double start;
  double end;

  if (!PyArg_ParseTuple(args, "Odd", &pyobj_mat, &start, &end))
    return NULL;
  if (!convert_to_CvArr(pyobj_mat, &mat, "mat")) return NULL;
#ifdef CVPY_VALIDATE_Range
CVPY_VALIDATE_Range();
#endif
  ERRWRAP(cvRange(mat, start, end));
  Py_RETURN_NONE;
}

static PyObject *pycvRealScalar(PyObject *self, PyObject *args)
{
  double val0;

  if (!PyArg_ParseTuple(args, "d", &val0))
    return NULL;
#ifdef CVPY_VALIDATE_RealScalar
CVPY_VALIDATE_RealScalar();
#endif
  CvScalar r;
  ERRWRAP(r = cvRealScalar(val0));
  return FROM_CvScalar(r);
}

static PyObject *pycvRectangle(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* img;
  PyObject *pyobj_img = NULL;
  CvPoint pt1;
  PyObject *pyobj_pt1 = NULL;
  CvPoint pt2;
  PyObject *pyobj_pt2 = NULL;
  CvScalar color;
  PyObject *pyobj_color = NULL;
  int thickness = 1;
  int lineType = 8;
  int shift = 0;

  const char *keywords[] = { "img", "pt1", "pt2", "color", "thickness", "lineType", "shift", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOO|iii", (char**)keywords, &pyobj_img, &pyobj_pt1, &pyobj_pt2, &pyobj_color, &thickness, &lineType, &shift))
    return NULL;
  if (!convert_to_CvArr(pyobj_img, &img, "img")) return NULL;
  if (!convert_to_CvPoint(pyobj_pt1, &pt1, "pt1")) return NULL;
  if (!convert_to_CvPoint(pyobj_pt2, &pt2, "pt2")) return NULL;
  if (!convert_to_CvScalar(pyobj_color, &color, "color")) return NULL;
#ifdef CVPY_VALIDATE_Rectangle
CVPY_VALIDATE_Rectangle();
#endif
  ERRWRAP(cvRectangle(img, pt1, pt2, color, thickness, lineType, shift));
  Py_RETURN_NONE;
}

static PyObject *pycvReduce(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  int dim = -1;
  int op = CV_REDUCE_SUM;

  const char *keywords[] = { "src", "dst", "dim", "op", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|ii", (char**)keywords, &pyobj_src, &pyobj_dst, &dim, &op))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_Reduce
CVPY_VALIDATE_Reduce();
#endif
  ERRWRAP(cvReduce(src, dst, dim, op));
  Py_RETURN_NONE;
}

static PyObject *pycvRemap(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvArr* mapx;
  PyObject *pyobj_mapx = NULL;
  CvArr* mapy;
  PyObject *pyobj_mapy = NULL;
  int flags = CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS;
  CvScalar fillval = cvScalarAll(0);
  PyObject *pyobj_fillval = NULL;

  const char *keywords[] = { "src", "dst", "mapx", "mapy", "flags", "fillval", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOO|iO", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_mapx, &pyobj_mapy, &flags, &pyobj_fillval))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if (!convert_to_CvArr(pyobj_mapx, &mapx, "mapx")) return NULL;
  if (!convert_to_CvArr(pyobj_mapy, &mapy, "mapy")) return NULL;
  if ((pyobj_fillval != NULL) && !convert_to_CvScalar(pyobj_fillval, &fillval, "fillval")) return NULL;
#ifdef CVPY_VALIDATE_Remap
CVPY_VALIDATE_Remap();
#endif
  ERRWRAP(cvRemap(src, dst, mapx, mapy, flags, fillval));
  Py_RETURN_NONE;
}

static PyObject *pycvRepeat(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_src, &pyobj_dst))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_Repeat
CVPY_VALIDATE_Repeat();
#endif
  ERRWRAP(cvRepeat(src, dst));
  Py_RETURN_NONE;
}

static PyObject *pycvReprojectImageTo3D(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* disparity;
  PyObject *pyobj_disparity = NULL;
  CvArr* _3dImage;
  PyObject *pyobj__3dImage = NULL;
  CvMat* Q;
  PyObject *pyobj_Q = NULL;
  int handleMissingValues = 0;

  const char *keywords[] = { "disparity", "_3dImage", "Q", "handleMissingValues", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|i", (char**)keywords, &pyobj_disparity, &pyobj__3dImage, &pyobj_Q, &handleMissingValues))
    return NULL;
  if (!convert_to_CvArr(pyobj_disparity, &disparity, "disparity")) return NULL;
  if (!convert_to_CvArr(pyobj__3dImage, &_3dImage, "_3dImage")) return NULL;
  if (!convert_to_CvMat(pyobj_Q, &Q, "Q")) return NULL;
#ifdef CVPY_VALIDATE_ReprojectImageTo3D
CVPY_VALIDATE_ReprojectImageTo3D();
#endif
  ERRWRAP(cvReprojectImageTo3D(disparity, _3dImage, Q, handleMissingValues));
  Py_RETURN_NONE;
}

static PyObject *pycvResetImageROI(PyObject *self, PyObject *args)
{
  IplImage* image;
  PyObject *pyobj_image = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_image))
    return NULL;
  if (!convert_to_IplImage(pyobj_image, &image, "image")) return NULL;
#ifdef CVPY_VALIDATE_ResetImageROI
CVPY_VALIDATE_ResetImageROI();
#endif
  ERRWRAP(cvResetImageROI(image));
  Py_RETURN_NONE;
}

static PyObject *pycvReshape(PyObject *self, PyObject *args, PyObject *kw)
;

static PyObject *pycvReshapeMatND(PyObject *self, PyObject *args)
;

static PyObject *pycvResize(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  int interpolation = CV_INTER_LINEAR;

  const char *keywords[] = { "src", "dst", "interpolation", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|i", (char**)keywords, &pyobj_src, &pyobj_dst, &interpolation))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_Resize
CVPY_VALIDATE_Resize();
#endif
  ERRWRAP(cvResize(src, dst, interpolation));
  Py_RETURN_NONE;
}

static PyObject *pycvResizeWindow(PyObject *self, PyObject *args)
{
  char* name;
  int width;
  int height;

  if (!PyArg_ParseTuple(args, "sii", &name, &width, &height))
    return NULL;
#ifdef CVPY_VALIDATE_ResizeWindow
CVPY_VALIDATE_ResizeWindow();
#endif
  ERRWRAP(cvResizeWindow(name, width, height));
  Py_RETURN_NONE;
}

static PyObject *pycvRetrieveFrame(PyObject *self, PyObject *args)
{
  CvCapture* capture;
  PyObject *pyobj_capture = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_capture))
    return NULL;
  if (!convert_to_CvCapturePTR(pyobj_capture, &capture, "capture")) return NULL;
#ifdef CVPY_VALIDATE_RetrieveFrame
CVPY_VALIDATE_RetrieveFrame();
#endif
  ROIplImage* r;
  ERRWRAP(r = cvRetrieveFrame(capture));
  return FROM_ROIplImagePTR(r);
}

static PyObject *pycvRodrigues2(PyObject *self, PyObject *args, PyObject *kw)
{
  CvMat* src;
  PyObject *pyobj_src = NULL;
  CvMat* dst;
  PyObject *pyobj_dst = NULL;
  CvMat* jacobian = 0;
  PyObject *pyobj_jacobian = NULL;

  const char *keywords[] = { "src", "dst", "jacobian", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|O", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_jacobian))
    return NULL;
  if (!convert_to_CvMat(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvMat(pyobj_dst, &dst, "dst")) return NULL;
  if ((pyobj_jacobian != NULL) && !convert_to_CvMat(pyobj_jacobian, &jacobian, "jacobian")) return NULL;
#ifdef CVPY_VALIDATE_Rodrigues2
CVPY_VALIDATE_Rodrigues2();
#endif
  ERRWRAP(cvRodrigues2(src, dst, jacobian));
  Py_RETURN_NONE;
}

static PyObject *pycvRound(PyObject *self, PyObject *args)
{
  double value;

  if (!PyArg_ParseTuple(args, "d", &value))
    return NULL;
#ifdef CVPY_VALIDATE_Round
CVPY_VALIDATE_Round();
#endif
  int r;
  ERRWRAP(r = cvRound(value));
  return FROM_int(r);
}

static PyObject *pycvRunningAvg(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* image;
  PyObject *pyobj_image = NULL;
  CvArr* acc;
  PyObject *pyobj_acc = NULL;
  double alpha;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;

  const char *keywords[] = { "image", "acc", "alpha", "mask", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOd|O", (char**)keywords, &pyobj_image, &pyobj_acc, &alpha, &pyobj_mask))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvArr(pyobj_acc, &acc, "acc")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_RunningAvg
CVPY_VALIDATE_RunningAvg();
#endif
  ERRWRAP(cvRunningAvg(image, acc, alpha, mask));
  Py_RETURN_NONE;
}

static PyObject *pycvSVBkSb(PyObject *self, PyObject *args)
{
  CvArr* W;
  PyObject *pyobj_W = NULL;
  CvArr* U;
  PyObject *pyobj_U = NULL;
  CvArr* V;
  PyObject *pyobj_V = NULL;
  CvArr* B;
  PyObject *pyobj_B = NULL;
  CvArr* X;
  PyObject *pyobj_X = NULL;
  int flags;

  if (!PyArg_ParseTuple(args, "OOOOOi", &pyobj_W, &pyobj_U, &pyobj_V, &pyobj_B, &pyobj_X, &flags))
    return NULL;
  if (!convert_to_CvArr(pyobj_W, &W, "W")) return NULL;
  if (!convert_to_CvArr(pyobj_U, &U, "U")) return NULL;
  if (!convert_to_CvArr(pyobj_V, &V, "V")) return NULL;
  if (!convert_to_CvArr(pyobj_B, &B, "B")) return NULL;
  if (!convert_to_CvArr(pyobj_X, &X, "X")) return NULL;
#ifdef CVPY_VALIDATE_SVBkSb
CVPY_VALIDATE_SVBkSb();
#endif
  ERRWRAP(cvSVBkSb(W, U, V, B, X, flags));
  Py_RETURN_NONE;
}

static PyObject *pycvSVD(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* A;
  PyObject *pyobj_A = NULL;
  CvArr* W;
  PyObject *pyobj_W = NULL;
  CvArr* U = NULL;
  PyObject *pyobj_U = NULL;
  CvArr* V = NULL;
  PyObject *pyobj_V = NULL;
  int flags = 0;

  const char *keywords[] = { "A", "W", "U", "V", "flags", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|OOi", (char**)keywords, &pyobj_A, &pyobj_W, &pyobj_U, &pyobj_V, &flags))
    return NULL;
  if (!convert_to_CvArr(pyobj_A, &A, "A")) return NULL;
  if (!convert_to_CvArr(pyobj_W, &W, "W")) return NULL;
  if ((pyobj_U != NULL) && !convert_to_CvArr(pyobj_U, &U, "U")) return NULL;
  if ((pyobj_V != NULL) && !convert_to_CvArr(pyobj_V, &V, "V")) return NULL;
#ifdef CVPY_VALIDATE_SVD
CVPY_VALIDATE_SVD();
#endif
  ERRWRAP(cvSVD(A, W, U, V, flags));
  Py_RETURN_NONE;
}

static PyObject *pycvSave(PyObject *self, PyObject *args, PyObject *kw)
{
  char* filename;
  generic structPtr;
  PyObject *pyobj_structPtr = NULL;
  char* name = NULL;
  char* comment = NULL;

  const char *keywords[] = { "filename", "structPtr", "name", "comment", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "sO|ss", (char**)keywords, &filename, &pyobj_structPtr, &name, &comment))
    return NULL;
  if (!convert_to_generic(pyobj_structPtr, &structPtr, "structPtr")) return NULL;
#ifdef CVPY_VALIDATE_Save
CVPY_VALIDATE_Save();
#endif
  ERRWRAP(cvSave(filename, structPtr, name, comment));
  Py_RETURN_NONE;
}

static PyObject *pycvSaveImage(PyObject *self, PyObject *args)
{
  char* filename;
  CvArr* image;
  PyObject *pyobj_image = NULL;

  if (!PyArg_ParseTuple(args, "sO", &filename, &pyobj_image))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
#ifdef CVPY_VALIDATE_SaveImage
CVPY_VALIDATE_SaveImage();
#endif
  ERRWRAP(cvSaveImage(filename, image));
  Py_RETURN_NONE;
}

static PyObject *pycvScalar(PyObject *self, PyObject *args, PyObject *kw)
{
  double val0;
  double val1 = 0;
  double val2 = 0;
  double val3 = 0;

  const char *keywords[] = { "val0", "val1", "val2", "val3", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "d|ddd", (char**)keywords, &val0, &val1, &val2, &val3))
    return NULL;
#ifdef CVPY_VALIDATE_Scalar
CVPY_VALIDATE_Scalar();
#endif
  CvScalar r;
  ERRWRAP(r = cvScalar(val0, val1, val2, val3));
  return FROM_CvScalar(r);
}

static PyObject *pycvScalarAll(PyObject *self, PyObject *args)
{
  double val0123;

  if (!PyArg_ParseTuple(args, "d", &val0123))
    return NULL;
#ifdef CVPY_VALIDATE_ScalarAll
CVPY_VALIDATE_ScalarAll();
#endif
  CvScalar r;
  ERRWRAP(r = cvScalarAll(val0123));
  return FROM_CvScalar(r);
}

static PyObject *pycvScale(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  double scale = 1.0;
  double shift = 0.0;

  const char *keywords[] = { "src", "dst", "scale", "shift", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|dd", (char**)keywords, &pyobj_src, &pyobj_dst, &scale, &shift))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_Scale
CVPY_VALIDATE_Scale();
#endif
  ERRWRAP(cvScale(src, dst, scale, shift));
  Py_RETURN_NONE;
}

static PyObject *pycvScaleAdd(PyObject *self, PyObject *args)
{
  CvArr* src1;
  PyObject *pyobj_src1 = NULL;
  CvScalar scale;
  PyObject *pyobj_scale = NULL;
  CvArr* src2;
  PyObject *pyobj_src2 = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;

  if (!PyArg_ParseTuple(args, "OOOO", &pyobj_src1, &pyobj_scale, &pyobj_src2, &pyobj_dst))
    return NULL;
  if (!convert_to_CvArr(pyobj_src1, &src1, "src1")) return NULL;
  if (!convert_to_CvScalar(pyobj_scale, &scale, "scale")) return NULL;
  if (!convert_to_CvArr(pyobj_src2, &src2, "src2")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_ScaleAdd
CVPY_VALIDATE_ScaleAdd();
#endif
  ERRWRAP(cvScaleAdd(src1, scale, src2, dst));
  Py_RETURN_NONE;
}

static PyObject *pycvSegmentMotion(PyObject *self, PyObject *args)
{
  CvArr* mhi;
  PyObject *pyobj_mhi = NULL;
  CvArr* seg_mask;
  PyObject *pyobj_seg_mask = NULL;
  CvMemStorage* storage;
  PyObject *pyobj_storage = NULL;
  double timestamp;
  double seg_thresh;

  if (!PyArg_ParseTuple(args, "OOOdd", &pyobj_mhi, &pyobj_seg_mask, &pyobj_storage, &timestamp, &seg_thresh))
    return NULL;
  if (!convert_to_CvArr(pyobj_mhi, &mhi, "mhi")) return NULL;
  if (!convert_to_CvArr(pyobj_seg_mask, &seg_mask, "seg_mask")) return NULL;
  if (!convert_to_CvMemStorage(pyobj_storage, &storage, "storage")) return NULL;
#ifdef CVPY_VALIDATE_SegmentMotion
CVPY_VALIDATE_SegmentMotion();
#endif
  CvSeq* r;
  ERRWRAP(r = cvSegmentMotion(mhi, seg_mask, storage, timestamp, seg_thresh));
  return FROM_CvSeqPTR(r);
}

static PyObject *pycvSeqInvert(PyObject *self, PyObject *args)
{
  CvSeq* seq;
  PyObject *pyobj_seq = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_seq))
    return NULL;
  if (!convert_to_CvSeq(pyobj_seq, &seq, "seq")) return NULL;
#ifdef CVPY_VALIDATE_SeqInvert
CVPY_VALIDATE_SeqInvert();
#endif
  ERRWRAP(cvSeqInvert(seq));
  Py_RETURN_NONE;
}

static PyObject *pycvSeqRemove(PyObject *self, PyObject *args)
{
  CvSeq* seq;
  PyObject *pyobj_seq = NULL;
  int index;

  if (!PyArg_ParseTuple(args, "Oi", &pyobj_seq, &index))
    return NULL;
  if (!convert_to_CvSeq(pyobj_seq, &seq, "seq")) return NULL;
#ifdef CVPY_VALIDATE_SeqRemove
CVPY_VALIDATE_SeqRemove();
#endif
  ERRWRAP(cvSeqRemove(seq, index));
  Py_RETURN_NONE;
}

static PyObject *pycvSeqRemoveSlice(PyObject *self, PyObject *args)
{
  CvSeq* seq;
  PyObject *pyobj_seq = NULL;
  CvSlice slice;
  PyObject *pyobj_slice = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_seq, &pyobj_slice))
    return NULL;
  if (!convert_to_CvSeq(pyobj_seq, &seq, "seq")) return NULL;
  if (!convert_to_CvSlice(pyobj_slice, &slice, "slice")) return NULL;
#ifdef CVPY_VALIDATE_SeqRemoveSlice
CVPY_VALIDATE_SeqRemoveSlice();
#endif
  ERRWRAP(cvSeqRemoveSlice(seq, slice));
  Py_RETURN_NONE;
}

static PyObject *pycvSet(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  CvScalar value;
  PyObject *pyobj_value = NULL;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;

  const char *keywords[] = { "arr", "value", "mask", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|O", (char**)keywords, &pyobj_arr, &pyobj_value, &pyobj_mask))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
  if (!convert_to_CvScalar(pyobj_value, &value, "value")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_Set
CVPY_VALIDATE_Set();
#endif
  ERRWRAP(cvSet(arr, value, mask));
  Py_RETURN_NONE;
}

static PyObject *pycvSet1D(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  int idx;
  CvScalar value;
  PyObject *pyobj_value = NULL;

  if (!PyArg_ParseTuple(args, "OiO", &pyobj_arr, &idx, &pyobj_value))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
  if (!convert_to_CvScalar(pyobj_value, &value, "value")) return NULL;
#ifdef CVPY_VALIDATE_Set1D
CVPY_VALIDATE_Set1D();
#endif
  ERRWRAP(cvSet1D(arr, idx, value));
  Py_RETURN_NONE;
}

static PyObject *pycvSet2D(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  int idx0;
  int idx1;
  CvScalar value;
  PyObject *pyobj_value = NULL;

  if (!PyArg_ParseTuple(args, "OiiO", &pyobj_arr, &idx0, &idx1, &pyobj_value))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
  if (!convert_to_CvScalar(pyobj_value, &value, "value")) return NULL;
#ifdef CVPY_VALIDATE_Set2D
CVPY_VALIDATE_Set2D();
#endif
  ERRWRAP(cvSet2D(arr, idx0, idx1, value));
  Py_RETURN_NONE;
}

static PyObject *pycvSet3D(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  int idx0;
  int idx1;
  int idx2;
  CvScalar value;
  PyObject *pyobj_value = NULL;

  if (!PyArg_ParseTuple(args, "OiiiO", &pyobj_arr, &idx0, &idx1, &idx2, &pyobj_value))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
  if (!convert_to_CvScalar(pyobj_value, &value, "value")) return NULL;
#ifdef CVPY_VALIDATE_Set3D
CVPY_VALIDATE_Set3D();
#endif
  ERRWRAP(cvSet3D(arr, idx0, idx1, idx2, value));
  Py_RETURN_NONE;
}

static PyObject *pycvSetCaptureProperty(PyObject *self, PyObject *args)
{
  CvCapture* capture;
  PyObject *pyobj_capture = NULL;
  int property_id;
  double value;

  if (!PyArg_ParseTuple(args, "Oid", &pyobj_capture, &property_id, &value))
    return NULL;
  if (!convert_to_CvCapturePTR(pyobj_capture, &capture, "capture")) return NULL;
#ifdef CVPY_VALIDATE_SetCaptureProperty
CVPY_VALIDATE_SetCaptureProperty();
#endif
  int r;
  ERRWRAP(r = cvSetCaptureProperty(capture, property_id, value));
  return FROM_int(r);
}

static PyObject *pycvSetData(PyObject *self, PyObject *args)
;

static PyObject *pycvSetIdentity(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* mat;
  PyObject *pyobj_mat = NULL;
  CvScalar value = cvRealScalar(1);
  PyObject *pyobj_value = NULL;

  const char *keywords[] = { "mat", "value", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O|O", (char**)keywords, &pyobj_mat, &pyobj_value))
    return NULL;
  if (!convert_to_CvArr(pyobj_mat, &mat, "mat")) return NULL;
  if ((pyobj_value != NULL) && !convert_to_CvScalar(pyobj_value, &value, "value")) return NULL;
#ifdef CVPY_VALIDATE_SetIdentity
CVPY_VALIDATE_SetIdentity();
#endif
  ERRWRAP(cvSetIdentity(mat, value));
  Py_RETURN_NONE;
}

static PyObject *pycvSetImageCOI(PyObject *self, PyObject *args)
{
  IplImage* image;
  PyObject *pyobj_image = NULL;
  int coi;

  if (!PyArg_ParseTuple(args, "Oi", &pyobj_image, &coi))
    return NULL;
  if (!convert_to_IplImage(pyobj_image, &image, "image")) return NULL;
#ifdef CVPY_VALIDATE_SetImageCOI
CVPY_VALIDATE_SetImageCOI();
#endif
  ERRWRAP(cvSetImageCOI(image, coi));
  Py_RETURN_NONE;
}

static PyObject *pycvSetImageROI(PyObject *self, PyObject *args)
{
  IplImage* image;
  PyObject *pyobj_image = NULL;
  CvRect rect;
  PyObject *pyobj_rect = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_image, &pyobj_rect))
    return NULL;
  if (!convert_to_IplImage(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvRect(pyobj_rect, &rect, "rect")) return NULL;
#ifdef CVPY_VALIDATE_SetImageROI
CVPY_VALIDATE_SetImageROI();
#endif
  ERRWRAP(cvSetImageROI(image, rect));
  Py_RETURN_NONE;
}

static PyObject *pycvSetMouseCallback(PyObject *self, PyObject *args, PyObject *kw)
;

static PyObject *pycvSetND(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  ints indices;
  PyObject *pyobj_indices = NULL;
  CvScalar value;
  PyObject *pyobj_value = NULL;

  if (!PyArg_ParseTuple(args, "OOO", &pyobj_arr, &pyobj_indices, &pyobj_value))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
  if (!convert_to_ints(pyobj_indices, &indices, "indices")) return NULL;
  if (!convert_to_CvScalar(pyobj_value, &value, "value")) return NULL;
#ifdef CVPY_VALIDATE_SetND
CVPY_VALIDATE_SetND();
#endif
  ERRWRAP(cvSetND(arr, indices.i, value));
  Py_RETURN_NONE;
}

static PyObject *pycvSetReal1D(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  int idx;
  double value;

  if (!PyArg_ParseTuple(args, "Oid", &pyobj_arr, &idx, &value))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
#ifdef CVPY_VALIDATE_SetReal1D
CVPY_VALIDATE_SetReal1D();
#endif
  ERRWRAP(cvSetReal1D(arr, idx, value));
  Py_RETURN_NONE;
}

static PyObject *pycvSetReal2D(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  int idx0;
  int idx1;
  double value;

  if (!PyArg_ParseTuple(args, "Oiid", &pyobj_arr, &idx0, &idx1, &value))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
#ifdef CVPY_VALIDATE_SetReal2D
CVPY_VALIDATE_SetReal2D();
#endif
  ERRWRAP(cvSetReal2D(arr, idx0, idx1, value));
  Py_RETURN_NONE;
}

static PyObject *pycvSetReal3D(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  int idx0;
  int idx1;
  int idx2;
  double value;

  if (!PyArg_ParseTuple(args, "Oiiid", &pyobj_arr, &idx0, &idx1, &idx2, &value))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
#ifdef CVPY_VALIDATE_SetReal3D
CVPY_VALIDATE_SetReal3D();
#endif
  ERRWRAP(cvSetReal3D(arr, idx0, idx1, idx2, value));
  Py_RETURN_NONE;
}

static PyObject *pycvSetRealND(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;
  ints indices;
  PyObject *pyobj_indices = NULL;
  double value;

  if (!PyArg_ParseTuple(args, "OOd", &pyobj_arr, &pyobj_indices, &value))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
  if (!convert_to_ints(pyobj_indices, &indices, "indices")) return NULL;
#ifdef CVPY_VALIDATE_SetRealND
CVPY_VALIDATE_SetRealND();
#endif
  ERRWRAP(cvSetRealND(arr, indices.i, value));
  Py_RETURN_NONE;
}

static PyObject *pycvSetTrackbarPos(PyObject *self, PyObject *args)
{
  char* trackbarName;
  char* windowName;
  int pos;

  if (!PyArg_ParseTuple(args, "ssi", &trackbarName, &windowName, &pos))
    return NULL;
#ifdef CVPY_VALIDATE_SetTrackbarPos
CVPY_VALIDATE_SetTrackbarPos();
#endif
  ERRWRAP(cvSetTrackbarPos(trackbarName, windowName, pos));
  Py_RETURN_NONE;
}

static PyObject *pycvSetWindowProperty(PyObject *self, PyObject *args)
{
  char* name;
  int prop_id;
  double prop_value;

  if (!PyArg_ParseTuple(args, "sid", &name, &prop_id, &prop_value))
    return NULL;
#ifdef CVPY_VALIDATE_SetWindowProperty
CVPY_VALIDATE_SetWindowProperty();
#endif
  ERRWRAP(cvSetWindowProperty(name, prop_id, prop_value));
  Py_RETURN_NONE;
}

static PyObject *pycvSetZero(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_arr))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
#ifdef CVPY_VALIDATE_SetZero
CVPY_VALIDATE_SetZero();
#endif
  ERRWRAP(cvSetZero(arr));
  Py_RETURN_NONE;
}

static PyObject *pycvShowImage(PyObject *self, PyObject *args)
{
  char* name;
  CvArr* image;
  PyObject *pyobj_image = NULL;

  if (!PyArg_ParseTuple(args, "sO", &name, &pyobj_image))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
#ifdef CVPY_VALIDATE_ShowImage
CVPY_VALIDATE_ShowImage();
#endif
  ERRWRAP(cvShowImage(name, image));
  Py_RETURN_NONE;
}

static PyObject *pycvSmooth(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  int smoothtype = CV_GAUSSIAN;
  int param1 = 3;
  int param2 = 0;
  double param3 = 0;
  double param4 = 0;

  const char *keywords[] = { "src", "dst", "smoothtype", "param1", "param2", "param3", "param4", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|iiidd", (char**)keywords, &pyobj_src, &pyobj_dst, &smoothtype, &param1, &param2, &param3, &param4))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_Smooth
CVPY_VALIDATE_Smooth();
#endif
  ERRWRAP(cvSmooth(src, dst, smoothtype, param1, param2, param3, param4));
  Py_RETURN_NONE;
}

static PyObject *pycvSnakeImage(PyObject *self, PyObject *args, PyObject *kw)
{
  IplImage* image;
  PyObject *pyobj_image = NULL;
  CvPoints points;
  PyObject *pyobj_points = NULL;
  floats alpha;
  PyObject *pyobj_alpha = NULL;
  floats beta;
  PyObject *pyobj_beta = NULL;
  floats gamma;
  PyObject *pyobj_gamma = NULL;
  CvSize win;
  PyObject *pyobj_win = NULL;
  CvTermCriteria criteria;
  PyObject *pyobj_criteria = NULL;
  int calc_gradient = 1;

  const char *keywords[] = { "image", "points", "alpha", "beta", "gamma", "win", "criteria", "calc_gradient", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOO|i", (char**)keywords, &pyobj_image, &pyobj_points, &pyobj_alpha, &pyobj_beta, &pyobj_gamma, &pyobj_win, &pyobj_criteria, &calc_gradient))
    return NULL;
  if (!convert_to_IplImage(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvPoints(pyobj_points, &points, "points")) return NULL;
  if (!convert_to_floats(pyobj_alpha, &alpha, "alpha")) return NULL;
  if (!convert_to_floats(pyobj_beta, &beta, "beta")) return NULL;
  if (!convert_to_floats(pyobj_gamma, &gamma, "gamma")) return NULL;
  if (!convert_to_CvSize(pyobj_win, &win, "win")) return NULL;
  if (!convert_to_CvTermCriteria(pyobj_criteria, &criteria, "criteria")) return NULL;
#ifdef CVPY_VALIDATE_SnakeImage
CVPY_VALIDATE_SnakeImage();
#endif
  ERRWRAP(cvSnakeImage(image, points.p,points.count, alpha.f, beta.f, gamma.f, win, criteria, calc_gradient));
  return FROM_CvPoints(points);
}

static PyObject *pycvSobel(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  int xorder;
  int yorder;
  int apertureSize = 3;

  const char *keywords[] = { "src", "dst", "xorder", "yorder", "apertureSize", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOii|i", (char**)keywords, &pyobj_src, &pyobj_dst, &xorder, &yorder, &apertureSize))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_Sobel
CVPY_VALIDATE_Sobel();
#endif
  ERRWRAP(cvSobel(src, dst, xorder, yorder, apertureSize));
  Py_RETURN_NONE;
}

static PyObject *pycvSolve(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* A;
  PyObject *pyobj_A = NULL;
  CvArr* B;
  PyObject *pyobj_B = NULL;
  CvArr* X;
  PyObject *pyobj_X = NULL;
  int method = CV_LU;

  const char *keywords[] = { "A", "B", "X", "method", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|i", (char**)keywords, &pyobj_A, &pyobj_B, &pyobj_X, &method))
    return NULL;
  if (!convert_to_CvArr(pyobj_A, &A, "A")) return NULL;
  if (!convert_to_CvArr(pyobj_B, &B, "B")) return NULL;
  if (!convert_to_CvArr(pyobj_X, &X, "X")) return NULL;
#ifdef CVPY_VALIDATE_Solve
CVPY_VALIDATE_Solve();
#endif
  ERRWRAP(cvSolve(A, B, X, method));
  Py_RETURN_NONE;
}

static PyObject *pycvSolveCubic(PyObject *self, PyObject *args)
{
  CvMat* coeffs;
  PyObject *pyobj_coeffs = NULL;
  CvMat* roots;
  PyObject *pyobj_roots = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_coeffs, &pyobj_roots))
    return NULL;
  if (!convert_to_CvMat(pyobj_coeffs, &coeffs, "coeffs")) return NULL;
  if (!convert_to_CvMat(pyobj_roots, &roots, "roots")) return NULL;
#ifdef CVPY_VALIDATE_SolveCubic
CVPY_VALIDATE_SolveCubic();
#endif
  ERRWRAP(cvSolveCubic(coeffs, roots));
  Py_RETURN_NONE;
}

static PyObject *pycvSolvePoly(PyObject *self, PyObject *args, PyObject *kw)
{
  CvMat* coeffs;
  PyObject *pyobj_coeffs = NULL;
  CvMat* roots;
  PyObject *pyobj_roots = NULL;
  int maxiter = 10;
  int fig = 10;

  const char *keywords[] = { "coeffs", "roots", "maxiter", "fig", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|ii", (char**)keywords, &pyobj_coeffs, &pyobj_roots, &maxiter, &fig))
    return NULL;
  if (!convert_to_CvMat(pyobj_coeffs, &coeffs, "coeffs")) return NULL;
  if (!convert_to_CvMat(pyobj_roots, &roots, "roots")) return NULL;
#ifdef CVPY_VALIDATE_SolvePoly
CVPY_VALIDATE_SolvePoly();
#endif
  ERRWRAP(cvSolvePoly(coeffs, roots, maxiter, fig));
  Py_RETURN_NONE;
}

static PyObject *pycvSort(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvArr* idxmat;
  PyObject *pyobj_idxmat = NULL;
  int flags = 0;

  const char *keywords[] = { "src", "dst", "idxmat", "flags", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|i", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_idxmat, &flags))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if (!convert_to_CvArr(pyobj_idxmat, &idxmat, "idxmat")) return NULL;
#ifdef CVPY_VALIDATE_Sort
CVPY_VALIDATE_Sort();
#endif
  ERRWRAP(cvSort(src, dst, idxmat, flags));
  Py_RETURN_NONE;
}

static PyObject *pycvSplit(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst0;
  PyObject *pyobj_dst0 = NULL;
  CvArr* dst1;
  PyObject *pyobj_dst1 = NULL;
  CvArr* dst2;
  PyObject *pyobj_dst2 = NULL;
  CvArr* dst3;
  PyObject *pyobj_dst3 = NULL;

  if (!PyArg_ParseTuple(args, "OOOOO", &pyobj_src, &pyobj_dst0, &pyobj_dst1, &pyobj_dst2, &pyobj_dst3))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst0, &dst0, "dst0")) return NULL;
  if (!convert_to_CvArr(pyobj_dst1, &dst1, "dst1")) return NULL;
  if (!convert_to_CvArr(pyobj_dst2, &dst2, "dst2")) return NULL;
  if (!convert_to_CvArr(pyobj_dst3, &dst3, "dst3")) return NULL;
#ifdef CVPY_VALIDATE_Split
CVPY_VALIDATE_Split();
#endif
  ERRWRAP(cvSplit(src, dst0, dst1, dst2, dst3));
  Py_RETURN_NONE;
}

static PyObject *pycvSqrt(PyObject *self, PyObject *args)
{
  float value;

  if (!PyArg_ParseTuple(args, "f", &value))
    return NULL;
#ifdef CVPY_VALIDATE_Sqrt
CVPY_VALIDATE_Sqrt();
#endif
  float r;
  ERRWRAP(r = cvSqrt(value));
  return FROM_float(r);
}

static PyObject *pycvSquareAcc(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* image;
  PyObject *pyobj_image = NULL;
  CvArr* sqsum;
  PyObject *pyobj_sqsum = NULL;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;

  const char *keywords[] = { "image", "sqsum", "mask", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|O", (char**)keywords, &pyobj_image, &pyobj_sqsum, &pyobj_mask))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvArr(pyobj_sqsum, &sqsum, "sqsum")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_SquareAcc
CVPY_VALIDATE_SquareAcc();
#endif
  ERRWRAP(cvSquareAcc(image, sqsum, mask));
  Py_RETURN_NONE;
}

static PyObject *pycvStartWindowThread(PyObject *self, PyObject *args)
{

#ifdef CVPY_VALIDATE_StartWindowThread
CVPY_VALIDATE_StartWindowThread();
#endif
  ERRWRAP(cvStartWindowThread());
  Py_RETURN_NONE;
}

static PyObject *pycvStereoCalibrate(PyObject *self, PyObject *args, PyObject *kw)
{
  CvMat* objectPoints;
  PyObject *pyobj_objectPoints = NULL;
  CvMat* imagePoints1;
  PyObject *pyobj_imagePoints1 = NULL;
  CvMat* imagePoints2;
  PyObject *pyobj_imagePoints2 = NULL;
  CvMat* pointCounts;
  PyObject *pyobj_pointCounts = NULL;
  CvMat* cameraMatrix1;
  PyObject *pyobj_cameraMatrix1 = NULL;
  CvMat* distCoeffs1;
  PyObject *pyobj_distCoeffs1 = NULL;
  CvMat* cameraMatrix2;
  PyObject *pyobj_cameraMatrix2 = NULL;
  CvMat* distCoeffs2;
  PyObject *pyobj_distCoeffs2 = NULL;
  CvSize imageSize;
  PyObject *pyobj_imageSize = NULL;
  CvMat* R;
  PyObject *pyobj_R = NULL;
  CvMat* T;
  PyObject *pyobj_T = NULL;
  CvMat* E = NULL;
  PyObject *pyobj_E = NULL;
  CvMat* F = NULL;
  PyObject *pyobj_F = NULL;
  CvTermCriteria term_crit = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,30,1e-6);
  PyObject *pyobj_term_crit = NULL;
  int flags = CV_CALIB_FIX_INTRINSIC;

  const char *keywords[] = { "objectPoints", "imagePoints1", "imagePoints2", "pointCounts", "cameraMatrix1", "distCoeffs1", "cameraMatrix2", "distCoeffs2", "imageSize", "R", "T", "E", "F", "term_crit", "flags", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOOOOOO|OOOi", (char**)keywords, &pyobj_objectPoints, &pyobj_imagePoints1, &pyobj_imagePoints2, &pyobj_pointCounts, &pyobj_cameraMatrix1, &pyobj_distCoeffs1, &pyobj_cameraMatrix2, &pyobj_distCoeffs2, &pyobj_imageSize, &pyobj_R, &pyobj_T, &pyobj_E, &pyobj_F, &pyobj_term_crit, &flags))
    return NULL;
  if (!convert_to_CvMat(pyobj_objectPoints, &objectPoints, "objectPoints")) return NULL;
  if (!convert_to_CvMat(pyobj_imagePoints1, &imagePoints1, "imagePoints1")) return NULL;
  if (!convert_to_CvMat(pyobj_imagePoints2, &imagePoints2, "imagePoints2")) return NULL;
  if (!convert_to_CvMat(pyobj_pointCounts, &pointCounts, "pointCounts")) return NULL;
  if (!convert_to_CvMat(pyobj_cameraMatrix1, &cameraMatrix1, "cameraMatrix1")) return NULL;
  if (!convert_to_CvMat(pyobj_distCoeffs1, &distCoeffs1, "distCoeffs1")) return NULL;
  if (!convert_to_CvMat(pyobj_cameraMatrix2, &cameraMatrix2, "cameraMatrix2")) return NULL;
  if (!convert_to_CvMat(pyobj_distCoeffs2, &distCoeffs2, "distCoeffs2")) return NULL;
  if (!convert_to_CvSize(pyobj_imageSize, &imageSize, "imageSize")) return NULL;
  if (!convert_to_CvMat(pyobj_R, &R, "R")) return NULL;
  if (!convert_to_CvMat(pyobj_T, &T, "T")) return NULL;
  if ((pyobj_E != NULL) && !convert_to_CvMat(pyobj_E, &E, "E")) return NULL;
  if ((pyobj_F != NULL) && !convert_to_CvMat(pyobj_F, &F, "F")) return NULL;
  if ((pyobj_term_crit != NULL) && !convert_to_CvTermCriteria(pyobj_term_crit, &term_crit, "term_crit")) return NULL;
#ifdef CVPY_VALIDATE_StereoCalibrate
CVPY_VALIDATE_StereoCalibrate();
#endif
  ERRWRAP(cvStereoCalibrate(objectPoints, imagePoints1, imagePoints2, pointCounts, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, E, F, term_crit, flags));
  Py_RETURN_NONE;
}

static PyObject *pycvStereoRectify(PyObject *self, PyObject *args, PyObject *kw)
{
  CvMat* cameraMatrix1;
  PyObject *pyobj_cameraMatrix1 = NULL;
  CvMat* cameraMatrix2;
  PyObject *pyobj_cameraMatrix2 = NULL;
  CvMat* distCoeffs1;
  PyObject *pyobj_distCoeffs1 = NULL;
  CvMat* distCoeffs2;
  PyObject *pyobj_distCoeffs2 = NULL;
  CvSize imageSize;
  PyObject *pyobj_imageSize = NULL;
  CvMat* R;
  PyObject *pyobj_R = NULL;
  CvMat* T;
  PyObject *pyobj_T = NULL;
  CvMat* R1;
  PyObject *pyobj_R1 = NULL;
  CvMat* R2;
  PyObject *pyobj_R2 = NULL;
  CvMat* P1;
  PyObject *pyobj_P1 = NULL;
  CvMat* P2;
  PyObject *pyobj_P2 = NULL;
  CvMat* Q = NULL;
  PyObject *pyobj_Q = NULL;
  int flags = CV_CALIB_ZERO_DISPARITY;
  double alpha = -1;
  CvSize newImageSize = cvSize(0,0);
  PyObject *pyobj_newImageSize = NULL;
  CvRect roi1;
  CvRect roi2;

  const char *keywords[] = { "cameraMatrix1", "cameraMatrix2", "distCoeffs1", "distCoeffs2", "imageSize", "R", "T", "R1", "R2", "P1", "P2", "Q", "flags", "alpha", "newImageSize", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOOOOOO|OidO", (char**)keywords, &pyobj_cameraMatrix1, &pyobj_cameraMatrix2, &pyobj_distCoeffs1, &pyobj_distCoeffs2, &pyobj_imageSize, &pyobj_R, &pyobj_T, &pyobj_R1, &pyobj_R2, &pyobj_P1, &pyobj_P2, &pyobj_Q, &flags, &alpha, &pyobj_newImageSize))
    return NULL;
  if (!convert_to_CvMat(pyobj_cameraMatrix1, &cameraMatrix1, "cameraMatrix1")) return NULL;
  if (!convert_to_CvMat(pyobj_cameraMatrix2, &cameraMatrix2, "cameraMatrix2")) return NULL;
  if (!convert_to_CvMat(pyobj_distCoeffs1, &distCoeffs1, "distCoeffs1")) return NULL;
  if (!convert_to_CvMat(pyobj_distCoeffs2, &distCoeffs2, "distCoeffs2")) return NULL;
  if (!convert_to_CvSize(pyobj_imageSize, &imageSize, "imageSize")) return NULL;
  if (!convert_to_CvMat(pyobj_R, &R, "R")) return NULL;
  if (!convert_to_CvMat(pyobj_T, &T, "T")) return NULL;
  if (!convert_to_CvMat(pyobj_R1, &R1, "R1")) return NULL;
  if (!convert_to_CvMat(pyobj_R2, &R2, "R2")) return NULL;
  if (!convert_to_CvMat(pyobj_P1, &P1, "P1")) return NULL;
  if (!convert_to_CvMat(pyobj_P2, &P2, "P2")) return NULL;
  if ((pyobj_Q != NULL) && !convert_to_CvMat(pyobj_Q, &Q, "Q")) return NULL;
  if ((pyobj_newImageSize != NULL) && !convert_to_CvSize(pyobj_newImageSize, &newImageSize, "newImageSize")) return NULL;
#ifdef CVPY_VALIDATE_StereoRectify
CVPY_VALIDATE_StereoRectify();
#endif
  ERRWRAP(cvStereoRectify(cameraMatrix1, cameraMatrix2, distCoeffs1, distCoeffs2, imageSize, R, T, R1, R2, P1, P2, Q, flags, alpha, newImageSize, &roi1, &roi2));
  return Py_BuildValue("NN", FROM_CvRect(roi1), FROM_CvRect(roi2));
}

static PyObject *pycvStereoRectifyUncalibrated(PyObject *self, PyObject *args, PyObject *kw)
{
  CvMat* points1;
  PyObject *pyobj_points1 = NULL;
  CvMat* points2;
  PyObject *pyobj_points2 = NULL;
  CvMat* F;
  PyObject *pyobj_F = NULL;
  CvSize imageSize;
  PyObject *pyobj_imageSize = NULL;
  CvMat* H1;
  PyObject *pyobj_H1 = NULL;
  CvMat* H2;
  PyObject *pyobj_H2 = NULL;
  double threshold = 5;

  const char *keywords[] = { "points1", "points2", "F", "imageSize", "H1", "H2", "threshold", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOOOO|d", (char**)keywords, &pyobj_points1, &pyobj_points2, &pyobj_F, &pyobj_imageSize, &pyobj_H1, &pyobj_H2, &threshold))
    return NULL;
  if (!convert_to_CvMat(pyobj_points1, &points1, "points1")) return NULL;
  if (!convert_to_CvMat(pyobj_points2, &points2, "points2")) return NULL;
  if (!convert_to_CvMat(pyobj_F, &F, "F")) return NULL;
  if (!convert_to_CvSize(pyobj_imageSize, &imageSize, "imageSize")) return NULL;
  if (!convert_to_CvMat(pyobj_H1, &H1, "H1")) return NULL;
  if (!convert_to_CvMat(pyobj_H2, &H2, "H2")) return NULL;
#ifdef CVPY_VALIDATE_StereoRectifyUncalibrated
CVPY_VALIDATE_StereoRectifyUncalibrated();
#endif
  ERRWRAP(cvStereoRectifyUncalibrated(points1, points2, F, imageSize, H1, H2, threshold));
  Py_RETURN_NONE;
}

static PyObject *pycvSub(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src1;
  PyObject *pyobj_src1 = NULL;
  CvArr* src2;
  PyObject *pyobj_src2 = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;

  const char *keywords[] = { "src1", "src2", "dst", "mask", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|O", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &pyobj_mask))
    return NULL;
  if (!convert_to_CvArr(pyobj_src1, &src1, "src1")) return NULL;
  if (!convert_to_CvArr(pyobj_src2, &src2, "src2")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_Sub
CVPY_VALIDATE_Sub();
#endif
  ERRWRAP(cvSub(src1, src2, dst, mask));
  Py_RETURN_NONE;
}

static PyObject *pycvSubRS(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvScalar value;
  PyObject *pyobj_value = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;

  const char *keywords[] = { "src", "value", "dst", "mask", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|O", (char**)keywords, &pyobj_src, &pyobj_value, &pyobj_dst, &pyobj_mask))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvScalar(pyobj_value, &value, "value")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_SubRS
CVPY_VALIDATE_SubRS();
#endif
  ERRWRAP(cvSubRS(src, value, dst, mask));
  Py_RETURN_NONE;
}

static PyObject *pycvSubS(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvScalar value;
  PyObject *pyobj_value = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;

  const char *keywords[] = { "src", "value", "dst", "mask", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|O", (char**)keywords, &pyobj_src, &pyobj_value, &pyobj_dst, &pyobj_mask))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvScalar(pyobj_value, &value, "value")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_SubS
CVPY_VALIDATE_SubS();
#endif
  ERRWRAP(cvSubS(src, value, dst, mask));
  Py_RETURN_NONE;
}

static PyObject *pycvSubdiv2DEdgeDst(PyObject *self, PyObject *args)
{
  CvSubdiv2DEdge edge;
  PyObject *pyobj_edge = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_edge))
    return NULL;
  if (!convert_to_CvSubdiv2DEdge(pyobj_edge, &edge, "edge")) return NULL;
#ifdef CVPY_VALIDATE_Subdiv2DEdgeDst
CVPY_VALIDATE_Subdiv2DEdgeDst();
#endif
  CvSubdiv2DPoint* r;
  ERRWRAP(r = cvSubdiv2DEdgeDst(edge));
  return FROM_CvSubdiv2DPointPTR(r);
}

static PyObject *pycvSubdiv2DEdgeOrg(PyObject *self, PyObject *args)
{
  CvSubdiv2DEdge edge;
  PyObject *pyobj_edge = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_edge))
    return NULL;
  if (!convert_to_CvSubdiv2DEdge(pyobj_edge, &edge, "edge")) return NULL;
#ifdef CVPY_VALIDATE_Subdiv2DEdgeOrg
CVPY_VALIDATE_Subdiv2DEdgeOrg();
#endif
  CvSubdiv2DPoint* r;
  ERRWRAP(r = cvSubdiv2DEdgeOrg(edge));
  return FROM_CvSubdiv2DPointPTR(r);
}

static PyObject *pycvSubdiv2DGetEdge(PyObject *self, PyObject *args)
{
  CvSubdiv2DEdge edge;
  PyObject *pyobj_edge = NULL;
  CvNextEdgeType type;
  PyObject *pyobj_type = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_edge, &pyobj_type))
    return NULL;
  if (!convert_to_CvSubdiv2DEdge(pyobj_edge, &edge, "edge")) return NULL;
  if (!convert_to_CvNextEdgeType(pyobj_type, &type, "type")) return NULL;
#ifdef CVPY_VALIDATE_Subdiv2DGetEdge
CVPY_VALIDATE_Subdiv2DGetEdge();
#endif
  CvSubdiv2DEdge r;
  ERRWRAP(r = cvSubdiv2DGetEdge(edge, type));
  return FROM_CvSubdiv2DEdge(r);
}

static PyObject *pycvSubdiv2DLocate(PyObject *self, PyObject *args)
;

static PyObject *pycvSubdiv2DNextEdge(PyObject *self, PyObject *args)
{
  CvSubdiv2DEdge edge;
  PyObject *pyobj_edge = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_edge))
    return NULL;
  if (!convert_to_CvSubdiv2DEdge(pyobj_edge, &edge, "edge")) return NULL;
#ifdef CVPY_VALIDATE_Subdiv2DNextEdge
CVPY_VALIDATE_Subdiv2DNextEdge();
#endif
  CvSubdiv2DEdge r;
  ERRWRAP(r = cvSubdiv2DNextEdge(edge));
  return FROM_CvSubdiv2DEdge(r);
}

static PyObject *pycvSubdiv2DRotateEdge(PyObject *self, PyObject *args)
{
  CvSubdiv2DEdge edge;
  PyObject *pyobj_edge = NULL;
  int rotate;

  if (!PyArg_ParseTuple(args, "Oi", &pyobj_edge, &rotate))
    return NULL;
  if (!convert_to_CvSubdiv2DEdge(pyobj_edge, &edge, "edge")) return NULL;
#ifdef CVPY_VALIDATE_Subdiv2DRotateEdge
CVPY_VALIDATE_Subdiv2DRotateEdge();
#endif
  CvSubdiv2DEdge r;
  ERRWRAP(r = cvSubdiv2DRotateEdge(edge, rotate));
  return FROM_CvSubdiv2DEdge(r);
}

static PyObject *pycvSubdivDelaunay2DInsert(PyObject *self, PyObject *args)
{
  CvSubdiv2D* subdiv;
  PyObject *pyobj_subdiv = NULL;
  CvPoint2D32f pt;
  PyObject *pyobj_pt = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_subdiv, &pyobj_pt))
    return NULL;
  if (!convert_to_CvSubdiv2DPTR(pyobj_subdiv, &subdiv, "subdiv")) return NULL;
  if (!convert_to_CvPoint2D32f(pyobj_pt, &pt, "pt")) return NULL;
#ifdef CVPY_VALIDATE_SubdivDelaunay2DInsert
CVPY_VALIDATE_SubdivDelaunay2DInsert();
#endif
  CvSubdiv2DPoint* r;
  ERRWRAP(r = cvSubdivDelaunay2DInsert(subdiv, pt));
  return FROM_CvSubdiv2DPointPTR(r);
}

static PyObject *pycvSum(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_arr))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
#ifdef CVPY_VALIDATE_Sum
CVPY_VALIDATE_Sum();
#endif
  CvScalar r;
  ERRWRAP(r = cvSum(arr));
  return FROM_CvScalar(r);
}

static PyObject *pycvThreshHist(PyObject *self, PyObject *args)
{
  CvHistogram* hist;
  PyObject *pyobj_hist = NULL;
  double threshold;

  if (!PyArg_ParseTuple(args, "Od", &pyobj_hist, &threshold))
    return NULL;
  if (!convert_to_CvHistogram(pyobj_hist, &hist, "hist")) return NULL;
#ifdef CVPY_VALIDATE_ThreshHist
CVPY_VALIDATE_ThreshHist();
#endif
  ERRWRAP(cvThreshHist(hist, threshold));
  Py_RETURN_NONE;
}

static PyObject *pycvThreshold(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  double threshold;
  double maxValue;
  int thresholdType;

  if (!PyArg_ParseTuple(args, "OOddi", &pyobj_src, &pyobj_dst, &threshold, &maxValue, &thresholdType))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_Threshold
CVPY_VALIDATE_Threshold();
#endif
  ERRWRAP(cvThreshold(src, dst, threshold, maxValue, thresholdType));
  Py_RETURN_NONE;
}

static PyObject *pycvTrace(PyObject *self, PyObject *args)
{
  CvArr* mat;
  PyObject *pyobj_mat = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_mat))
    return NULL;
  if (!convert_to_CvArr(pyobj_mat, &mat, "mat")) return NULL;
#ifdef CVPY_VALIDATE_Trace
CVPY_VALIDATE_Trace();
#endif
  CvScalar r;
  ERRWRAP(r = cvTrace(mat));
  return FROM_CvScalar(r);
}

static PyObject *pycvTransform(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvMat* transmat;
  PyObject *pyobj_transmat = NULL;
  CvMat* shiftvec = NULL;
  PyObject *pyobj_shiftvec = NULL;

  const char *keywords[] = { "src", "dst", "transmat", "shiftvec", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|O", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_transmat, &pyobj_shiftvec))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if (!convert_to_CvMat(pyobj_transmat, &transmat, "transmat")) return NULL;
  if ((pyobj_shiftvec != NULL) && !convert_to_CvMat(pyobj_shiftvec, &shiftvec, "shiftvec")) return NULL;
#ifdef CVPY_VALIDATE_Transform
CVPY_VALIDATE_Transform();
#endif
  ERRWRAP(cvTransform(src, dst, transmat, shiftvec));
  Py_RETURN_NONE;
}

static PyObject *pycvTranspose(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_src, &pyobj_dst))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
#ifdef CVPY_VALIDATE_Transpose
CVPY_VALIDATE_Transpose();
#endif
  ERRWRAP(cvTranspose(src, dst));
  Py_RETURN_NONE;
}

static PyObject *pycvUndistort2(PyObject *self, PyObject *args)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvMat* cameraMatrix;
  PyObject *pyobj_cameraMatrix = NULL;
  CvMat* distCoeffs;
  PyObject *pyobj_distCoeffs = NULL;

  if (!PyArg_ParseTuple(args, "OOOO", &pyobj_src, &pyobj_dst, &pyobj_cameraMatrix, &pyobj_distCoeffs))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if (!convert_to_CvMat(pyobj_cameraMatrix, &cameraMatrix, "cameraMatrix")) return NULL;
  if (!convert_to_CvMat(pyobj_distCoeffs, &distCoeffs, "distCoeffs")) return NULL;
#ifdef CVPY_VALIDATE_Undistort2
CVPY_VALIDATE_Undistort2();
#endif
  ERRWRAP(cvUndistort2(src, dst, cameraMatrix, distCoeffs));
  Py_RETURN_NONE;
}

static PyObject *pycvUndistortPoints(PyObject *self, PyObject *args, PyObject *kw)
{
  CvMat* src;
  PyObject *pyobj_src = NULL;
  CvMat* dst;
  PyObject *pyobj_dst = NULL;
  CvMat* cameraMatrix;
  PyObject *pyobj_cameraMatrix = NULL;
  CvMat* distCoeffs;
  PyObject *pyobj_distCoeffs = NULL;
  CvMat* R = NULL;
  PyObject *pyobj_R = NULL;
  CvMat* P = NULL;
  PyObject *pyobj_P = NULL;

  const char *keywords[] = { "src", "dst", "cameraMatrix", "distCoeffs", "R", "P", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOO|OO", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_cameraMatrix, &pyobj_distCoeffs, &pyobj_R, &pyobj_P))
    return NULL;
  if (!convert_to_CvMat(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvMat(pyobj_dst, &dst, "dst")) return NULL;
  if (!convert_to_CvMat(pyobj_cameraMatrix, &cameraMatrix, "cameraMatrix")) return NULL;
  if (!convert_to_CvMat(pyobj_distCoeffs, &distCoeffs, "distCoeffs")) return NULL;
  if ((pyobj_R != NULL) && !convert_to_CvMat(pyobj_R, &R, "R")) return NULL;
  if ((pyobj_P != NULL) && !convert_to_CvMat(pyobj_P, &P, "P")) return NULL;
#ifdef CVPY_VALIDATE_UndistortPoints
CVPY_VALIDATE_UndistortPoints();
#endif
  ERRWRAP(cvUndistortPoints(src, dst, cameraMatrix, distCoeffs, R, P));
  Py_RETURN_NONE;
}

static PyObject *pycvUpdateMotionHistory(PyObject *self, PyObject *args)
{
  CvArr* silhouette;
  PyObject *pyobj_silhouette = NULL;
  CvArr* mhi;
  PyObject *pyobj_mhi = NULL;
  double timestamp;
  double duration;

  if (!PyArg_ParseTuple(args, "OOdd", &pyobj_silhouette, &pyobj_mhi, &timestamp, &duration))
    return NULL;
  if (!convert_to_CvArr(pyobj_silhouette, &silhouette, "silhouette")) return NULL;
  if (!convert_to_CvArr(pyobj_mhi, &mhi, "mhi")) return NULL;
#ifdef CVPY_VALIDATE_UpdateMotionHistory
CVPY_VALIDATE_UpdateMotionHistory();
#endif
  ERRWRAP(cvUpdateMotionHistory(silhouette, mhi, timestamp, duration));
  Py_RETURN_NONE;
}

static PyObject *pycvWaitKey(PyObject *self, PyObject *args, PyObject *kw)
;

static PyObject *pycvWarpAffine(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvMat* mapMatrix;
  PyObject *pyobj_mapMatrix = NULL;
  int flags = CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS;
  CvScalar fillval = cvScalarAll(0);
  PyObject *pyobj_fillval = NULL;

  const char *keywords[] = { "src", "dst", "mapMatrix", "flags", "fillval", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|iO", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_mapMatrix, &flags, &pyobj_fillval))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if (!convert_to_CvMat(pyobj_mapMatrix, &mapMatrix, "mapMatrix")) return NULL;
  if ((pyobj_fillval != NULL) && !convert_to_CvScalar(pyobj_fillval, &fillval, "fillval")) return NULL;
#ifdef CVPY_VALIDATE_WarpAffine
CVPY_VALIDATE_WarpAffine();
#endif
  ERRWRAP(cvWarpAffine(src, dst, mapMatrix, flags, fillval));
  Py_RETURN_NONE;
}

static PyObject *pycvWarpPerspective(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvMat* mapMatrix;
  PyObject *pyobj_mapMatrix = NULL;
  int flags = CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS;
  CvScalar fillval = cvScalarAll(0);
  PyObject *pyobj_fillval = NULL;

  const char *keywords[] = { "src", "dst", "mapMatrix", "flags", "fillval", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|iO", (char**)keywords, &pyobj_src, &pyobj_dst, &pyobj_mapMatrix, &flags, &pyobj_fillval))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if (!convert_to_CvMat(pyobj_mapMatrix, &mapMatrix, "mapMatrix")) return NULL;
  if ((pyobj_fillval != NULL) && !convert_to_CvScalar(pyobj_fillval, &fillval, "fillval")) return NULL;
#ifdef CVPY_VALIDATE_WarpPerspective
CVPY_VALIDATE_WarpPerspective();
#endif
  ERRWRAP(cvWarpPerspective(src, dst, mapMatrix, flags, fillval));
  Py_RETURN_NONE;
}

static PyObject *pycvWatershed(PyObject *self, PyObject *args)
{
  CvArr* image;
  PyObject *pyobj_image = NULL;
  CvArr* markers;
  PyObject *pyobj_markers = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_image, &pyobj_markers))
    return NULL;
  if (!convert_to_CvArr(pyobj_image, &image, "image")) return NULL;
  if (!convert_to_CvArr(pyobj_markers, &markers, "markers")) return NULL;
#ifdef CVPY_VALIDATE_Watershed
CVPY_VALIDATE_Watershed();
#endif
  ERRWRAP(cvWatershed(image, markers));
  Py_RETURN_NONE;
}

static PyObject *pycvWriteFrame(PyObject *self, PyObject *args)
{
  CvVideoWriter* writer;
  PyObject *pyobj_writer = NULL;
  IplImage* image;
  PyObject *pyobj_image = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyobj_writer, &pyobj_image))
    return NULL;
  if (!convert_to_CvVideoWriterPTR(pyobj_writer, &writer, "writer")) return NULL;
  if (!convert_to_IplImage(pyobj_image, &image, "image")) return NULL;
#ifdef CVPY_VALIDATE_WriteFrame
CVPY_VALIDATE_WriteFrame();
#endif
  int r;
  ERRWRAP(r = cvWriteFrame(writer, image));
  return FROM_int(r);
}

static PyObject *pycvXor(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src1;
  PyObject *pyobj_src1 = NULL;
  CvArr* src2;
  PyObject *pyobj_src2 = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;

  const char *keywords[] = { "src1", "src2", "dst", "mask", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|O", (char**)keywords, &pyobj_src1, &pyobj_src2, &pyobj_dst, &pyobj_mask))
    return NULL;
  if (!convert_to_CvArr(pyobj_src1, &src1, "src1")) return NULL;
  if (!convert_to_CvArr(pyobj_src2, &src2, "src2")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_Xor
CVPY_VALIDATE_Xor();
#endif
  ERRWRAP(cvXor(src1, src2, dst, mask));
  Py_RETURN_NONE;
}

static PyObject *pycvXorS(PyObject *self, PyObject *args, PyObject *kw)
{
  CvArr* src;
  PyObject *pyobj_src = NULL;
  CvScalar value;
  PyObject *pyobj_value = NULL;
  CvArr* dst;
  PyObject *pyobj_dst = NULL;
  CvArr* mask = NULL;
  PyObject *pyobj_mask = NULL;

  const char *keywords[] = { "src", "value", "dst", "mask", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|O", (char**)keywords, &pyobj_src, &pyobj_value, &pyobj_dst, &pyobj_mask))
    return NULL;
  if (!convert_to_CvArr(pyobj_src, &src, "src")) return NULL;
  if (!convert_to_CvScalar(pyobj_value, &value, "value")) return NULL;
  if (!convert_to_CvArr(pyobj_dst, &dst, "dst")) return NULL;
  if ((pyobj_mask != NULL) && !convert_to_CvArr(pyobj_mask, &mask, "mask")) return NULL;
#ifdef CVPY_VALIDATE_XorS
CVPY_VALIDATE_XorS();
#endif
  ERRWRAP(cvXorS(src, value, dst, mask));
  Py_RETURN_NONE;
}

static PyObject *pycvZero(PyObject *self, PyObject *args)
{
  CvArr* arr;
  PyObject *pyobj_arr = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj_arr))
    return NULL;
  if (!convert_to_CvArr(pyobj_arr, &arr, "arr")) return NULL;
#ifdef CVPY_VALIDATE_Zero
CVPY_VALIDATE_Zero();
#endif
  ERRWRAP(cvZero(arr));
  Py_RETURN_NONE;
}

static PyObject *pycvmGet(PyObject *self, PyObject *args)
{
  CvMat* mat;
  PyObject *pyobj_mat = NULL;
  int row;
  int col;

  if (!PyArg_ParseTuple(args, "Oii", &pyobj_mat, &row, &col))
    return NULL;
  if (!convert_to_CvMat(pyobj_mat, &mat, "mat")) return NULL;
#ifdef CVPY_VALIDATE_mGet
CVPY_VALIDATE_mGet();
#endif
  double r;
  ERRWRAP(r = cvmGet(mat, row, col));
  return FROM_double(r);
}

static PyObject *pycvmSet(PyObject *self, PyObject *args)
{
  CvMat* mat;
  PyObject *pyobj_mat = NULL;
  int row;
  int col;
  double value;

  if (!PyArg_ParseTuple(args, "Oiid", &pyobj_mat, &row, &col, &value))
    return NULL;
  if (!convert_to_CvMat(pyobj_mat, &mat, "mat")) return NULL;
#ifdef CVPY_VALIDATE_mSet
CVPY_VALIDATE_mSet();
#endif
  ERRWRAP(cvmSet(mat, row, col, value));
  Py_RETURN_NONE;
}
