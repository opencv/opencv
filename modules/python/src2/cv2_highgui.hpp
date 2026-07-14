#ifndef CV2_HIGHGUI_HPP
#define CV2_HIGHGUI_HPP

#include "cv2.hpp"
#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_HIGHGUI
PyObject *pycvSetMouseCallback(PyObject*, PyObject *args, PyObject *kw);
// workaround for #20408, use nullptr, set value later
PyObject *pycvCreateTrackbar(PyObject*, PyObject *args);
PyObject *pycvCreateButton(PyObject*, PyObject *args, PyObject *kw);
#endif

#endif // CV2_HIGHGUI_HPP
