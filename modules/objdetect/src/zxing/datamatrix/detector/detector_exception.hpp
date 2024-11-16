/*
 * DetectorException.hpp
 *
 *  Created on: Aug 26, 2011
 *      Author: luiz
 */

#ifndef DETECTOREXCEPTION_H_
#define DETECTOREXCEPTION_H_

#include "../../exception.hpp"

namespace zxing {
namespace datamatrix {

class DetectorException : public Exception {
public:
    DetectorException(const char *msg);
    virtual ~DetectorException() throw();
};
}  // namespace datamatrix
}  // namespace zxing
#endif /* DETECTOREXCEPTION_H_ */
