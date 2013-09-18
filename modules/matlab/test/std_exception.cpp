/*
 * file:   exception.cpp
 * author: Hilton Bristow
 * date:   Wed, 19 Jun 2013 11:15:15
 *
 * See LICENCE for full modification and redistribution details.
 * Copyright 2013 The OpenCV Foundation
 */
#include <exception>
#include "mex.h"

/*
 * exception
 * Gateway routine
 *   nlhs - number of return arguments
 *   plhs - pointers to return arguments
 *   nrhs - number of input arguments
 *   prhs - pointers to input arguments
 */
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {

  // call the opencv function
  // [out =] namespace.fun(src1, ..., srcn, dst1, ..., dstn, opt1, ..., optn);
  try {
    throw std::exception();
  } catch(std::exception& e) {
    mexErrMsgTxt(e.what());
  } catch(...) {
    mexErrMsgTxt("Incorrect exception caught!");
  }
}
