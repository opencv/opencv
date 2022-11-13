//
// Created by yuval on 6/10/20.
//

#ifndef PNP_USING_EIGEN_LIBRARY_MATRIXUTILS_H
#define PNP_USING_EIGEN_LIBRARY_MATRIXUTILS_H

#include "Definitions.h"

namespace NPnP {
template <int Size> void symmetrize(RowMatrix<Size, Size> &matrix) {
  matrix = 0.5 * (matrix + matrix.transpose().eval());
}
} // namespace NPnP
#endif // PNP_USING_EIGEN_LIBRARY_MATRIXUTILS_H
