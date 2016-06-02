/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */

#include "precomp.hpp"
#include <iostream>
#include <map>
#include <set>

namespace cv
{

// Removes duplicate elements in a given vector.
template<typename _Tp>
inline std::vector<_Tp> remove_dups(const std::vector<_Tp>& src) {
    typedef typename std::set<_Tp>::const_iterator constSetIterator;
    typedef typename std::vector<_Tp>::const_iterator constVecIterator;
    std::set<_Tp> set_elems;
    for (constVecIterator it = src.begin(); it != src.end(); ++it)
        set_elems.insert(*it);
    std::vector<_Tp> elems;
    for (constSetIterator it = set_elems.begin(); it != set_elems.end(); ++it)
        elems.push_back(*it);
    return elems;
}

static Mat argsort(InputArray _src, bool ascending=true)
{
    Mat src = _src.getMat();
    if (src.rows != 1 && src.cols != 1) {
        String error_message = "Wrong shape of input matrix! Expected a matrix with one row or column.";
        CV_Error(Error::StsBadArg, error_message);
    }
    int flags = SORT_EVERY_ROW | (ascending ? SORT_ASCENDING : SORT_DESCENDING);
    Mat sorted_indices;
    sortIdx(src.reshape(1,1),sorted_indices,flags);
    return sorted_indices;
}

static Mat asRowMatrix(InputArrayOfArrays src, int rtype, double alpha=1, double beta=0) {
    // make sure the input data is a vector of matrices or vector of vector
    if(src.kind() != _InputArray::STD_VECTOR_MAT && src.kind() != _InputArray::STD_VECTOR_VECTOR) {
        String error_message = "The data is expected as InputArray::STD_VECTOR_MAT (a std::vector<Mat>) or _InputArray::STD_VECTOR_VECTOR (a std::vector< std::vector<...> >).";
        CV_Error(Error::StsBadArg, error_message);
    }
    // number of samples
    size_t n = src.total();
    // return empty matrix if no matrices given
    if(n == 0)
        return Mat();
    // dimensionality of (reshaped) samples
    size_t d = src.getMat(0).total();
    // create data matrix
    Mat data((int)n, (int)d, rtype);
    // now copy data
    for(int i = 0; i < (int)n; i++) {
        // make sure data can be reshaped, throw exception if not!
        if(src.getMat(i).total() != d) {
            String error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, (int)d, (int)src.getMat(i).total());
            CV_Error(Error::StsBadArg, error_message);
        }
        // get a hold of the current row
        Mat xi = data.row(i);
        // make reshape happy by cloning for non-continuous matrices
        if(src.getMat(i).isContinuous()) {
            src.getMat(i).reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        } else {
            src.getMat(i).clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        }
    }
    return data;
}

static void sortMatrixColumnsByIndices(InputArray _src, InputArray _indices, OutputArray _dst) {
    if(_indices.getMat().type() != CV_32SC1) {
        CV_Error(Error::StsUnsupportedFormat, "cv::sortColumnsByIndices only works on integer indices!");
    }
    Mat src = _src.getMat();
    std::vector<int> indices = _indices.getMat();
    _dst.create(src.rows, src.cols, src.type());
    Mat dst = _dst.getMat();
    for(size_t idx = 0; idx < indices.size(); idx++) {
        Mat originalCol = src.col(indices[idx]);
        Mat sortedCol = dst.col((int)idx);
        originalCol.copyTo(sortedCol);
    }
}

static Mat sortMatrixColumnsByIndices(InputArray src, InputArray indices) {
    Mat dst;
    sortMatrixColumnsByIndices(src, indices, dst);
    return dst;
}


template<typename _Tp> static bool
isSymmetric_(InputArray src) {
    Mat _src = src.getMat();
    if(_src.cols != _src.rows)
        return false;
    for (int i = 0; i < _src.rows; i++) {
        for (int j = 0; j < _src.cols; j++) {
            _Tp a = _src.at<_Tp> (i, j);
            _Tp b = _src.at<_Tp> (j, i);
            if (a != b) {
                return false;
            }
        }
    }
    return true;
}

template<typename _Tp> static bool
isSymmetric_(InputArray src, double eps) {
    Mat _src = src.getMat();
    if(_src.cols != _src.rows)
        return false;
    for (int i = 0; i < _src.rows; i++) {
        for (int j = 0; j < _src.cols; j++) {
            _Tp a = _src.at<_Tp> (i, j);
            _Tp b = _src.at<_Tp> (j, i);
            if (std::abs(a - b) > eps) {
                return false;
            }
        }
    }
    return true;
}

static bool isSymmetric(InputArray src, double eps=1e-16)
{
    Mat m = src.getMat();
    switch (m.type()) {
        case CV_8SC1: return isSymmetric_<char>(m); break;
        case CV_8UC1:
            return isSymmetric_<unsigned char>(m); break;
        case CV_16SC1:
            return isSymmetric_<short>(m); break;
        case CV_16UC1:
            return isSymmetric_<unsigned short>(m); break;
        case CV_32SC1:
            return isSymmetric_<int>(m); break;
        case CV_32FC1:
            return isSymmetric_<float>(m, eps); break;
        case CV_64FC1:
            return isSymmetric_<double>(m, eps); break;
        default:
            break;
    }
    return false;
}


//------------------------------------------------------------------------------
// cv::subspaceProject
//------------------------------------------------------------------------------
Mat LDA::subspaceProject(InputArray _W, InputArray _mean, InputArray _src) {
    // get data matrices
    Mat W = _W.getMat();
    Mat mean = _mean.getMat();
    Mat src = _src.getMat();
    // get number of samples and dimension
    int n = src.rows;
    int d = src.cols;
    // make sure the data has the correct shape
    if(W.rows != d) {
        String error_message = format("Wrong shapes for given matrices. Was size(src) = (%d,%d), size(W) = (%d,%d).", src.rows, src.cols, W.rows, W.cols);
        CV_Error(Error::StsBadArg, error_message);
    }
    // make sure mean is correct if not empty
    if(!mean.empty() && (mean.total() != (size_t) d)) {
        String error_message = format("Wrong mean shape for the given data matrix. Expected %d, but was %d.", d, mean.total());
        CV_Error(Error::StsBadArg, error_message);
    }
    // create temporary matrices
    Mat X, Y;
    // make sure you operate on correct type
    src.convertTo(X, W.type());
    // safe to do, because of above assertion
    if(!mean.empty()) {
        for(int i=0; i<n; i++) {
            Mat r_i = X.row(i);
            subtract(r_i, mean.reshape(1,1), r_i);
        }
    }
    // finally calculate projection as Y = (X-mean)*W
    gemm(X, W, 1.0, Mat(), 0.0, Y);
    return Y;
}

//------------------------------------------------------------------------------
// cv::subspaceReconstruct
//------------------------------------------------------------------------------
Mat LDA::subspaceReconstruct(InputArray _W, InputArray _mean, InputArray _src)
{
    // get data matrices
    Mat W = _W.getMat();
    Mat mean = _mean.getMat();
    Mat src = _src.getMat();
    // get number of samples and dimension
    int n = src.rows;
    int d = src.cols;
    // make sure the data has the correct shape
    if(W.cols != d) {
        String error_message = format("Wrong shapes for given matrices. Was size(src) = (%d,%d), size(W) = (%d,%d).", src.rows, src.cols, W.rows, W.cols);
        CV_Error(Error::StsBadArg, error_message);
    }
    // make sure mean is correct if not empty
    if(!mean.empty() && (mean.total() != (size_t) W.rows)) {
        String error_message = format("Wrong mean shape for the given eigenvector matrix. Expected %d, but was %d.", W.cols, mean.total());
        CV_Error(Error::StsBadArg, error_message);
    }
    // initialize temporary matrices
    Mat X, Y;
    // copy data & make sure we are using the correct type
    src.convertTo(Y, W.type());
    // calculate the reconstruction
    gemm(Y, W, 1.0, Mat(), 0.0, X, GEMM_2_T);
    // safe to do because of above assertion
    if(!mean.empty()) {
        for(int i=0; i<n; i++) {
            Mat r_i = X.row(i);
            add(r_i, mean.reshape(1,1), r_i);
        }
    }
    return X;
}


class EigenvalueDecomposition {
private:

    // Holds the data dimension.
    int n;

    // Stores real/imag part of a complex division.
    double cdivr, cdivi;

    // Pointer to internal memory.
    double *d, *e, *ort;
    double **V, **H;

    // Holds the computed eigenvalues.
    Mat _eigenvalues;

    // Holds the computed eigenvectors.
    Mat _eigenvectors;

    // Allocates memory.
    template<typename _Tp>
    _Tp *alloc_1d(int m) {
        return new _Tp[m];
    }

    // Allocates memory.
    template<typename _Tp>
    _Tp *alloc_1d(int m, _Tp val) {
        _Tp *arr = alloc_1d<_Tp> (m);
        for (int i = 0; i < m; i++)
            arr[i] = val;
        return arr;
    }

    // Allocates memory.
    template<typename _Tp>
    _Tp **alloc_2d(int m, int _n) {
        _Tp **arr = new _Tp*[m];
        for (int i = 0; i < m; i++)
            arr[i] = new _Tp[_n];
        return arr;
    }

    // Allocates memory.
    template<typename _Tp>
    _Tp **alloc_2d(int m, int _n, _Tp val) {
        _Tp **arr = alloc_2d<_Tp> (m, _n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < _n; j++) {
                arr[i][j] = val;
            }
        }
        return arr;
    }

    void cdiv(double xr, double xi, double yr, double yi) {
        double r, dv;
        if (std::abs(yr) > std::abs(yi)) {
            r = yi / yr;
            dv = yr + r * yi;
            cdivr = (xr + r * xi) / dv;
            cdivi = (xi - r * xr) / dv;
        } else {
            r = yr / yi;
            dv = yi + r * yr;
            cdivr = (r * xr + xi) / dv;
            cdivi = (r * xi - xr) / dv;
        }
    }

    // Nonsymmetric reduction from Hessenberg to real Schur form.

    void hqr2() {

        //  This is derived from the Algol procedure hqr2,
        //  by Martin and Wilkinson, Handbook for Auto. Comp.,
        //  Vol.ii-Linear Algebra, and the corresponding
        //  Fortran subroutine in EISPACK.

        // Initialize
        int nn = this->n;
        int n1 = nn - 1;
        int low = 0;
        int high = nn - 1;
        double eps = std::pow(2.0, -52.0);
        double exshift = 0.0;
        double p = 0, q = 0, r = 0, s = 0, z = 0, t, w, x, y;

        // Store roots isolated by balanc and compute matrix norm

        double norm = 0.0;
        for (int i = 0; i < nn; i++) {
            if (i < low || i > high) {
                d[i] = H[i][i];
                e[i] = 0.0;
            }
            for (int j = std::max(i - 1, 0); j < nn; j++) {
                norm = norm + std::abs(H[i][j]);
            }
        }

        // Outer loop over eigenvalue index
        int iter = 0;
        while (n1 >= low) {

            // Look for single small sub-diagonal element
            int l = n1;
            while (l > low) {
                s = std::abs(H[l - 1][l - 1]) + std::abs(H[l][l]);
                if (s == 0.0) {
                    s = norm;
                }
                if (std::abs(H[l][l - 1]) < eps * s) {
                    break;
                }
                l--;
            }

            // Check for convergence
            // One root found

            if (l == n1) {
                H[n1][n1] = H[n1][n1] + exshift;
                d[n1] = H[n1][n1];
                e[n1] = 0.0;
                n1--;
                iter = 0;

                // Two roots found

            } else if (l == n1 - 1) {
                w = H[n1][n1 - 1] * H[n1 - 1][n1];
                p = (H[n1 - 1][n1 - 1] - H[n1][n1]) / 2.0;
                q = p * p + w;
                z = std::sqrt(std::abs(q));
                H[n1][n1] = H[n1][n1] + exshift;
                H[n1 - 1][n1 - 1] = H[n1 - 1][n1 - 1] + exshift;
                x = H[n1][n1];

                // Real pair

                if (q >= 0) {
                    if (p >= 0) {
                        z = p + z;
                    } else {
                        z = p - z;
                    }
                    d[n1 - 1] = x + z;
                    d[n1] = d[n1 - 1];
                    if (z != 0.0) {
                        d[n1] = x - w / z;
                    }
                    e[n1 - 1] = 0.0;
                    e[n1] = 0.0;
                    x = H[n1][n1 - 1];
                    s = std::abs(x) + std::abs(z);
                    p = x / s;
                    q = z / s;
                    r = std::sqrt(p * p + q * q);
                    p = p / r;
                    q = q / r;

                    // Row modification

                    for (int j = n1 - 1; j < nn; j++) {
                        z = H[n1 - 1][j];
                        H[n1 - 1][j] = q * z + p * H[n1][j];
                        H[n1][j] = q * H[n1][j] - p * z;
                    }

                    // Column modification

                    for (int i = 0; i <= n1; i++) {
                        z = H[i][n1 - 1];
                        H[i][n1 - 1] = q * z + p * H[i][n1];
                        H[i][n1] = q * H[i][n1] - p * z;
                    }

                    // Accumulate transformations

                    for (int i = low; i <= high; i++) {
                        z = V[i][n1 - 1];
                        V[i][n1 - 1] = q * z + p * V[i][n1];
                        V[i][n1] = q * V[i][n1] - p * z;
                    }

                    // Complex pair

                } else {
                    d[n1 - 1] = x + p;
                    d[n1] = x + p;
                    e[n1 - 1] = z;
                    e[n1] = -z;
                }
                n1 = n1 - 2;
                iter = 0;

                // No convergence yet

            } else {

                // Form shift

                x = H[n1][n1];
                y = 0.0;
                w = 0.0;
                if (l < n1) {
                    y = H[n1 - 1][n1 - 1];
                    w = H[n1][n1 - 1] * H[n1 - 1][n1];
                }

                // Wilkinson's original ad hoc shift

                if (iter == 10) {
                    exshift += x;
                    for (int i = low; i <= n1; i++) {
                        H[i][i] -= x;
                    }
                    s = std::abs(H[n1][n1 - 1]) + std::abs(H[n1 - 1][n1 - 2]);
                    x = y = 0.75 * s;
                    w = -0.4375 * s * s;
                }

                // MATLAB's new ad hoc shift

                if (iter == 30) {
                    s = (y - x) / 2.0;
                    s = s * s + w;
                    if (s > 0) {
                        s = std::sqrt(s);
                        if (y < x) {
                            s = -s;
                        }
                        s = x - w / ((y - x) / 2.0 + s);
                        for (int i = low; i <= n1; i++) {
                            H[i][i] -= s;
                        }
                        exshift += s;
                        x = y = w = 0.964;
                    }
                }

                iter = iter + 1; // (Could check iteration count here.)

                // Look for two consecutive small sub-diagonal elements
                int m = n1 - 2;
                while (m >= l) {
                    z = H[m][m];
                    r = x - z;
                    s = y - z;
                    p = (r * s - w) / H[m + 1][m] + H[m][m + 1];
                    q = H[m + 1][m + 1] - z - r - s;
                    r = H[m + 2][m + 1];
                    s = std::abs(p) + std::abs(q) + std::abs(r);
                    p = p / s;
                    q = q / s;
                    r = r / s;
                    if (m == l) {
                        break;
                    }
                    if (std::abs(H[m][m - 1]) * (std::abs(q) + std::abs(r)) < eps * (std::abs(p)
                                                                                     * (std::abs(H[m - 1][m - 1]) + std::abs(z) + std::abs(
                                                                                                                                           H[m + 1][m + 1])))) {
                        break;
                    }
                    m--;
                }

                for (int i = m + 2; i <= n1; i++) {
                    H[i][i - 2] = 0.0;
                    if (i > m + 2) {
                        H[i][i - 3] = 0.0;
                    }
                }

                // Double QR step involving rows l:n and columns m:n

                for (int k = m; k <= n1 - 1; k++) {
                    bool notlast = (k != n1 - 1);
                    if (k != m) {
                        p = H[k][k - 1];
                        q = H[k + 1][k - 1];
                        r = (notlast ? H[k + 2][k - 1] : 0.0);
                        x = std::abs(p) + std::abs(q) + std::abs(r);
                        if (x != 0.0) {
                            p = p / x;
                            q = q / x;
                            r = r / x;
                        }
                    }
                    if (x == 0.0) {
                        break;
                    }
                    s = std::sqrt(p * p + q * q + r * r);
                    if (p < 0) {
                        s = -s;
                    }
                    if (s != 0) {
                        if (k != m) {
                            H[k][k - 1] = -s * x;
                        } else if (l != m) {
                            H[k][k - 1] = -H[k][k - 1];
                        }
                        p = p + s;
                        x = p / s;
                        y = q / s;
                        z = r / s;
                        q = q / p;
                        r = r / p;

                        // Row modification

                        for (int j = k; j < nn; j++) {
                            p = H[k][j] + q * H[k + 1][j];
                            if (notlast) {
                                p = p + r * H[k + 2][j];
                                H[k + 2][j] = H[k + 2][j] - p * z;
                            }
                            H[k][j] = H[k][j] - p * x;
                            H[k + 1][j] = H[k + 1][j] - p * y;
                        }

                        // Column modification

                        for (int i = 0; i <= std::min(n1, k + 3); i++) {
                            p = x * H[i][k] + y * H[i][k + 1];
                            if (notlast) {
                                p = p + z * H[i][k + 2];
                                H[i][k + 2] = H[i][k + 2] - p * r;
                            }
                            H[i][k] = H[i][k] - p;
                            H[i][k + 1] = H[i][k + 1] - p * q;
                        }

                        // Accumulate transformations

                        for (int i = low; i <= high; i++) {
                            p = x * V[i][k] + y * V[i][k + 1];
                            if (notlast) {
                                p = p + z * V[i][k + 2];
                                V[i][k + 2] = V[i][k + 2] - p * r;
                            }
                            V[i][k] = V[i][k] - p;
                            V[i][k + 1] = V[i][k + 1] - p * q;
                        }
                    } // (s != 0)
                } // k loop
            } // check convergence
        } // while (n1 >= low)

        // Backsubstitute to find vectors of upper triangular form

        if (norm == 0.0) {
            return;
        }

        for (n1 = nn - 1; n1 >= 0; n1--) {
            p = d[n1];
            q = e[n1];

            // Real vector

            if (q == 0) {
                int l = n1;
                H[n1][n1] = 1.0;
                for (int i = n1 - 1; i >= 0; i--) {
                    w = H[i][i] - p;
                    r = 0.0;
                    for (int j = l; j <= n1; j++) {
                        r = r + H[i][j] * H[j][n1];
                    }
                    if (e[i] < 0.0) {
                        z = w;
                        s = r;
                    } else {
                        l = i;
                        if (e[i] == 0.0) {
                            if (w != 0.0) {
                                H[i][n1] = -r / w;
                            } else {
                                H[i][n1] = -r / (eps * norm);
                            }

                            // Solve real equations

                        } else {
                            x = H[i][i + 1];
                            y = H[i + 1][i];
                            q = (d[i] - p) * (d[i] - p) + e[i] * e[i];
                            t = (x * s - z * r) / q;
                            H[i][n1] = t;
                            if (std::abs(x) > std::abs(z)) {
                                H[i + 1][n1] = (-r - w * t) / x;
                            } else {
                                H[i + 1][n1] = (-s - y * t) / z;
                            }
                        }

                        // Overflow control

                        t = std::abs(H[i][n1]);
                        if ((eps * t) * t > 1) {
                            for (int j = i; j <= n1; j++) {
                                H[j][n1] = H[j][n1] / t;
                            }
                        }
                    }
                }
                // Complex vector
            } else if (q < 0) {
                int l = n1 - 1;

                // Last vector component imaginary so matrix is triangular

                if (std::abs(H[n1][n1 - 1]) > std::abs(H[n1 - 1][n1])) {
                    H[n1 - 1][n1 - 1] = q / H[n1][n1 - 1];
                    H[n1 - 1][n1] = -(H[n1][n1] - p) / H[n1][n1 - 1];
                } else {
                    cdiv(0.0, -H[n1 - 1][n1], H[n1 - 1][n1 - 1] - p, q);
                    H[n1 - 1][n1 - 1] = cdivr;
                    H[n1 - 1][n1] = cdivi;
                }
                H[n1][n1 - 1] = 0.0;
                H[n1][n1] = 1.0;
                for (int i = n1 - 2; i >= 0; i--) {
                    double ra, sa, vr, vi;
                    ra = 0.0;
                    sa = 0.0;
                    for (int j = l; j <= n1; j++) {
                        ra = ra + H[i][j] * H[j][n1 - 1];
                        sa = sa + H[i][j] * H[j][n1];
                    }
                    w = H[i][i] - p;

                    if (e[i] < 0.0) {
                        z = w;
                        r = ra;
                        s = sa;
                    } else {
                        l = i;
                        if (e[i] == 0) {
                            cdiv(-ra, -sa, w, q);
                            H[i][n1 - 1] = cdivr;
                            H[i][n1] = cdivi;
                        } else {

                            // Solve complex equations

                            x = H[i][i + 1];
                            y = H[i + 1][i];
                            vr = (d[i] - p) * (d[i] - p) + e[i] * e[i] - q * q;
                            vi = (d[i] - p) * 2.0 * q;
                            if (vr == 0.0 && vi == 0.0) {
                                vr = eps * norm * (std::abs(w) + std::abs(q) + std::abs(x)
                                                   + std::abs(y) + std::abs(z));
                            }
                            cdiv(x * r - z * ra + q * sa,
                                 x * s - z * sa - q * ra, vr, vi);
                            H[i][n1 - 1] = cdivr;
                            H[i][n1] = cdivi;
                            if (std::abs(x) > (std::abs(z) + std::abs(q))) {
                                H[i + 1][n1 - 1] = (-ra - w * H[i][n1 - 1] + q
                                                   * H[i][n1]) / x;
                                H[i + 1][n1] = (-sa - w * H[i][n1] - q * H[i][n1
                                                                            - 1]) / x;
                            } else {
                                cdiv(-r - y * H[i][n1 - 1], -s - y * H[i][n1], z,
                                     q);
                                H[i + 1][n1 - 1] = cdivr;
                                H[i + 1][n1] = cdivi;
                            }
                        }

                        // Overflow control

                        t = std::max(std::abs(H[i][n1 - 1]), std::abs(H[i][n1]));
                        if ((eps * t) * t > 1) {
                            for (int j = i; j <= n1; j++) {
                                H[j][n1 - 1] = H[j][n1 - 1] / t;
                                H[j][n1] = H[j][n1] / t;
                            }
                        }
                    }
                }
            }
        }

        // Vectors of isolated roots

        for (int i = 0; i < nn; i++) {
            if (i < low || i > high) {
                for (int j = i; j < nn; j++) {
                    V[i][j] = H[i][j];
                }
            }
        }

        // Back transformation to get eigenvectors of original matrix

        for (int j = nn - 1; j >= low; j--) {
            for (int i = low; i <= high; i++) {
                z = 0.0;
                for (int k = low; k <= std::min(j, high); k++) {
                    z = z + V[i][k] * H[k][j];
                }
                V[i][j] = z;
            }
        }
    }

    // Nonsymmetric reduction to Hessenberg form.
    void orthes() {
        //  This is derived from the Algol procedures orthes and ortran,
        //  by Martin and Wilkinson, Handbook for Auto. Comp.,
        //  Vol.ii-Linear Algebra, and the corresponding
        //  Fortran subroutines in EISPACK.
        int low = 0;
        int high = n - 1;

        for (int m = low + 1; m <= high - 1; m++) {

            // Scale column.

            double scale = 0.0;
            for (int i = m; i <= high; i++) {
                scale = scale + std::abs(H[i][m - 1]);
            }
            if (scale != 0.0) {

                // Compute Householder transformation.

                double h = 0.0;
                for (int i = high; i >= m; i--) {
                    ort[i] = H[i][m - 1] / scale;
                    h += ort[i] * ort[i];
                }
                double g = std::sqrt(h);
                if (ort[m] > 0) {
                    g = -g;
                }
                h = h - ort[m] * g;
                ort[m] = ort[m] - g;

                // Apply Householder similarity transformation
                // H = (I-u*u'/h)*H*(I-u*u')/h)

                for (int j = m; j < n; j++) {
                    double f = 0.0;
                    for (int i = high; i >= m; i--) {
                        f += ort[i] * H[i][j];
                    }
                    f = f / h;
                    for (int i = m; i <= high; i++) {
                        H[i][j] -= f * ort[i];
                    }
                }

                for (int i = 0; i <= high; i++) {
                    double f = 0.0;
                    for (int j = high; j >= m; j--) {
                        f += ort[j] * H[i][j];
                    }
                    f = f / h;
                    for (int j = m; j <= high; j++) {
                        H[i][j] -= f * ort[j];
                    }
                }
                ort[m] = scale * ort[m];
                H[m][m - 1] = scale * g;
            }
        }

        // Accumulate transformations (Algol's ortran).

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                V[i][j] = (i == j ? 1.0 : 0.0);
            }
        }

        for (int m = high - 1; m >= low + 1; m--) {
            if (H[m][m - 1] != 0.0) {
                for (int i = m + 1; i <= high; i++) {
                    ort[i] = H[i][m - 1];
                }
                for (int j = m; j <= high; j++) {
                    double g = 0.0;
                    for (int i = m; i <= high; i++) {
                        g += ort[i] * V[i][j];
                    }
                    // Double division avoids possible underflow
                    g = (g / ort[m]) / H[m][m - 1];
                    for (int i = m; i <= high; i++) {
                        V[i][j] += g * ort[i];
                    }
                }
            }
        }
    }

    // Releases all internal working memory.
    void release() {
        // releases the working data
        delete[] d;
        delete[] e;
        delete[] ort;
        for (int i = 0; i < n; i++) {
            delete[] H[i];
            delete[] V[i];
        }
        delete[] H;
        delete[] V;
    }

    // Computes the Eigenvalue Decomposition for a matrix given in H.
    void compute() {
        // Allocate memory for the working data.
        V = alloc_2d<double> (n, n, 0.0);
        d = alloc_1d<double> (n);
        e = alloc_1d<double> (n);
        ort = alloc_1d<double> (n);
        // Reduce to Hessenberg form.
        orthes();
        // Reduce Hessenberg to real Schur form.
        hqr2();
        // Copy eigenvalues to OpenCV Matrix.
        _eigenvalues.create(1, n, CV_64FC1);
        for (int i = 0; i < n; i++) {
            _eigenvalues.at<double> (0, i) = d[i];
        }
        // Copy eigenvectors to OpenCV Matrix.
        _eigenvectors.create(n, n, CV_64FC1);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                _eigenvectors.at<double> (i, j) = V[i][j];
        // Deallocate the memory by releasing all internal working data.
        release();
    }

public:
    EigenvalueDecomposition()
    : n(0) { }

    // Initializes & computes the Eigenvalue Decomposition for a general matrix
    // given in src. This function is a port of the EigenvalueSolver in JAMA,
    // which has been released to public domain by The MathWorks and the
    // National Institute of Standards and Technology (NIST).
    EigenvalueDecomposition(InputArray src) {
        compute(src);
    }

    // This function computes the Eigenvalue Decomposition for a general matrix
    // given in src. This function is a port of the EigenvalueSolver in JAMA,
    // which has been released to public domain by The MathWorks and the
    // National Institute of Standards and Technology (NIST).
    void compute(InputArray src)
    {
        if(isSymmetric(src)) {
            // Fall back to OpenCV for a symmetric matrix!
            cv::eigen(src, _eigenvalues, _eigenvectors);
        } else {
            Mat tmp;
            // Convert the given input matrix to double. Is there any way to
            // prevent allocating the temporary memory? Only used for copying
            // into working memory and deallocated after.
            src.getMat().convertTo(tmp, CV_64FC1);
            // Get dimension of the matrix.
            this->n = tmp.cols;
            // Allocate the matrix data to work on.
            this->H = alloc_2d<double> (n, n);
            // Now safely copy the data.
            for (int i = 0; i < tmp.rows; i++) {
                for (int j = 0; j < tmp.cols; j++) {
                    this->H[i][j] = tmp.at<double>(i, j);
                }
            }
            // Deallocates the temporary matrix before computing.
            tmp.release();
            // Performs the eigenvalue decomposition of H.
            compute();
        }
    }

    ~EigenvalueDecomposition() {}

    // Returns the eigenvalues of the Eigenvalue Decomposition.
    Mat eigenvalues() {    return _eigenvalues; }
    // Returns the eigenvectors of the Eigenvalue Decomposition.
    Mat eigenvectors() { return _eigenvectors; }
};


//------------------------------------------------------------------------------
// Linear Discriminant Analysis implementation
//------------------------------------------------------------------------------

LDA::LDA(int num_components) : _dataAsRow(true), _num_components(num_components) { }

LDA::LDA(InputArrayOfArrays src, InputArray labels, int num_components) : _dataAsRow(true),  _num_components(num_components)
{
    this->compute(src, labels); //! compute eigenvectors and eigenvalues
}

LDA::~LDA() {}

void LDA::save(const String& filename) const
{
    FileStorage fs(filename, FileStorage::WRITE);
    if (!fs.isOpened()) {
        CV_Error(Error::StsError, "File can't be opened for writing!");
    }
    this->save(fs);
    fs.release();
}

// Deserializes this object from a given filename.
void LDA::load(const String& filename) {
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
       CV_Error(Error::StsError, "File can't be opened for reading!");
    this->load(fs);
    fs.release();
}

// Serializes this object to a given FileStorage.
void LDA::save(FileStorage& fs) const {
    // write matrices
    fs << "num_components" << _num_components;
    fs << "eigenvalues" << _eigenvalues;
    fs << "eigenvectors" << _eigenvectors;
}

// Deserializes this object from a given FileStorage.
void LDA::load(const FileStorage& fs) {
    //read matrices
    fs["num_components"] >> _num_components;
    fs["eigenvalues"] >> _eigenvalues;
    fs["eigenvectors"] >> _eigenvectors;
}

void LDA::lda(InputArrayOfArrays _src, InputArray _lbls) {
    // get data
    Mat src = _src.getMat();
    std::vector<int> labels;
    // safely copy the labels
    {
        Mat tmp = _lbls.getMat();
        for(unsigned int i = 0; i < tmp.total(); i++) {
            labels.push_back(tmp.at<int>(i));
        }
    }
    // turn into row sampled matrix
    Mat data;
    // ensure working matrix is double precision
    src.convertTo(data, CV_64FC1);
    // maps the labels, so they're ascending: [0,1,...,C]
    std::vector<int> mapped_labels(labels.size());
    std::vector<int> num2label = remove_dups(labels);
    std::map<int, int> label2num;
    for (int i = 0; i < (int)num2label.size(); i++)
        label2num[num2label[i]] = i;
    for (size_t i = 0; i < labels.size(); i++)
        mapped_labels[i] = label2num[labels[i]];
    // get sample size, dimension
    int N = data.rows;
    int D = data.cols;
    // number of unique labels
    int C = (int)num2label.size();
    // we can't do a LDA on one class, what do you
    // want to separate from each other then?
    if(C == 1) {
        String error_message = "At least two classes are needed to perform a LDA. Reason: Only one class was given!";
        CV_Error(Error::StsBadArg, error_message);
    }
    // throw error if less labels, than samples
    if (labels.size() != static_cast<size_t>(N)) {
        String error_message = format("The number of samples must equal the number of labels. Given %d labels, %d samples. ", labels.size(), N);
        CV_Error(Error::StsBadArg, error_message);
    }
    // warn if within-classes scatter matrix becomes singular
    if (N < D) {
        std::cout << "Warning: Less observations than feature dimension given!"
                  << "Computation will probably fail."
                  << std::endl;
    }
    // clip number of components to be a valid number
    if ((_num_components <= 0) || (_num_components > (C - 1))) {
        _num_components = (C - 1);
    }
    // holds the mean over all classes
    Mat meanTotal = Mat::zeros(1, D, data.type());
    // holds the mean for each class
    std::vector<Mat> meanClass(C);
    std::vector<int> numClass(C);
    // initialize
    for (int i = 0; i < C; i++) {
        numClass[i] = 0;
        meanClass[i] = Mat::zeros(1, D, data.type()); //! Dx1 image vector
    }
    // calculate sums
    for (int i = 0; i < N; i++) {
        Mat instance = data.row(i);
        int classIdx = mapped_labels[i];
        add(meanTotal, instance, meanTotal);
        add(meanClass[classIdx], instance, meanClass[classIdx]);
        numClass[classIdx]++;
    }
    // calculate total mean
    meanTotal.convertTo(meanTotal, meanTotal.type(), 1.0 / static_cast<double> (N));
    // calculate class means
    for (int i = 0; i < C; i++) {
        meanClass[i].convertTo(meanClass[i], meanClass[i].type(), 1.0 / static_cast<double> (numClass[i]));
    }
    // subtract class means
    for (int i = 0; i < N; i++) {
        int classIdx = mapped_labels[i];
        Mat instance = data.row(i);
        subtract(instance, meanClass[classIdx], instance);
    }
    // calculate within-classes scatter
    Mat Sw = Mat::zeros(D, D, data.type());
    mulTransposed(data, Sw, true);
    // calculate between-classes scatter
    Mat Sb = Mat::zeros(D, D, data.type());
    for (int i = 0; i < C; i++) {
        Mat tmp;
        subtract(meanClass[i], meanTotal, tmp);
        mulTransposed(tmp, tmp, true);
        add(Sb, tmp, Sb);
    }
    // invert Sw
    Mat Swi = Sw.inv();
    // M = inv(Sw)*Sb
    Mat M;
    gemm(Swi, Sb, 1.0, Mat(), 0.0, M);
    EigenvalueDecomposition es(M);
    _eigenvalues = es.eigenvalues();
    _eigenvectors = es.eigenvectors();
    // reshape eigenvalues, so they are stored by column
    _eigenvalues = _eigenvalues.reshape(1, 1);
    // get sorted indices descending by their eigenvalue
    std::vector<int> sorted_indices = argsort(_eigenvalues, false);
    // now sort eigenvalues and eigenvectors accordingly
    _eigenvalues = sortMatrixColumnsByIndices(_eigenvalues, sorted_indices);
    _eigenvectors = sortMatrixColumnsByIndices(_eigenvectors, sorted_indices);
    // and now take only the num_components and we're out!
    _eigenvalues = Mat(_eigenvalues, Range::all(), Range(0, _num_components));
    _eigenvectors = Mat(_eigenvectors, Range::all(), Range(0, _num_components));
}

void LDA::compute(InputArrayOfArrays _src, InputArray _lbls) {
    switch(_src.kind()) {
    case _InputArray::STD_VECTOR_MAT:
        lda(asRowMatrix(_src, CV_64FC1), _lbls);
        break;
    case _InputArray::MAT:
        lda(_src.getMat(), _lbls);
        break;
    default:
        String error_message= format("InputArray Datatype %d is not supported.", _src.kind());
        CV_Error(Error::StsBadArg, error_message);
        break;
    }
}

// Projects one or more row aligned samples into the LDA subspace.
Mat LDA::project(InputArray src) {
   return subspaceProject(_eigenvectors, Mat(), src);
}

// Reconstructs projections from the LDA subspace from one or more row aligned samples.
Mat LDA::reconstruct(InputArray src) {
   return subspaceReconstruct(_eigenvectors, Mat(), src);
}

}
