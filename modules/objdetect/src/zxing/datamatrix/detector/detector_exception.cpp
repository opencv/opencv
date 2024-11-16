/*
 * DetectorException.cpp
 *
 *  Created on: Aug 26, 2011
 *      Author: luiz
 */

#include "detector_exception.hpp"

namespace zxing {
namespace datamatrix {

DetectorException::DetectorException(const char *msg) :
Exception(msg) {
}

DetectorException::~DetectorException() throw() {
}

}  // namespace datamatrix
}  // namespace zxing
