/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#ifndef __OPENCV_FASTFILTERFLOW_H__
#define __OPENCV_FASTFILTERFLOW_H__

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "parallel_for.h"
#include "opencv2/gpu/gpu.hpp"
#include "Eigen/core"
#include "Eigen/sparse"

using namespace std;
using namespace tbb;
using namespace Eigen;

namespace cv
{
inline static void neigborPixels(int i, int j, int rows, int cols, int neighWidth, int neigborHeight, vector<vector<int> > &matPixels)
{
int iSt = i - neighWidth >= 0 ? i - neighWidth : 0;
int iEnd = i + neighWidth < rows ? i + neighWidth : rows - 1;
int jSt = j - neigborHeight >= 0 ? j - neigborHeight : 0;
int jEnd = j + neigborHeight < cols ? j + neigborHeight : cols - 1;
int elms = (iEnd - iSt + 1) * (jEnd - jSt + 1);
int cnt = 0;
vector<int> ngbPixels(elms, 0);
for (int ir = iSt; ir <= iEnd; ++ir)
{
for (int jc = jSt; jc <= jEnd; ++jc)
{
ngbPixels[cnt++] = ir * cols + jc;
}
}
matPixels.push_back(ngbPixels);
}

inline static void imgMatToArray(Mat &img, VectorXd &imgAry)
{
int rows = img.rows;
int cols = img.cols;
for (int i = 0; i != rows; ++i)
{
for (int j = 0; j != cols; ++j)
{
imgAry[i * cols + j] = img.at<double>(i, j);
}
}
}

inline static void neighborMatrix(Mat &img, vector<vector<int> > &neighMat, int nWidth, int nHeight)
{
int rows = img.rows;
int cols = img.cols;
int nPixels = rows * cols;
for (int i = 0; i < rows; ++i)
{
for (int j = 0; j < cols; ++j)
{
neigborPixels(i, j, rows, cols, nWidth, nHeight, neighMat);
}
}
}

inline static double linSearch(VectorXd &img0, VectorXd &img1, double &gamma, SparseMatrix<double> &transMat, int iPixel, int minGradInd, int &nPixels, bool &flag)
{
double s1 = 0;
double s2 = 0;
for (SparseMatrix<double>::InnerIterator it(transMat, iPixel); it; ++it)
{
int irow = it.row();
if (irow == minGradInd)
{
s1 += (it.value() + gamma * (1 - it.value())) * img0(irow);
flag = true;
}
else
{
s1 += (it.value() + gamma * (-it.value())) * img0(irow);
}
s2 += it.value() * img0(irow);
}
if (!flag)
{
s1 += gamma * img0(minGradInd);
}
s1 -= img1(iPixel);
s2 -= img1(iPixel);
s1 *= s1;
s2 *= s2;
return s1 - s2;
}

inline static double multiplyDenseSparse(VectorXd &img, SparseMatrix<double> &trans, int cols)
{
double mSum = 0;
for (SparseMatrix<double>::InnerIterator it(trans, cols); it; ++it)
{
mSum += it.value() * img[it.row()];
}
return mSum;
}

inline static void addDenseSparse(VectorXd &gradAry, SparseMatrix<double> &dmat, int cols, double lambda_1)
{
for (SparseMatrix<double>::InnerIterator it(dmat, cols); it; ++it)
{
gradAry[it.row()] += lambda_1 * it.value();
}
}

inline static void homogMultiply(Vector2d &affHom, MatrixXd &affineMat, Vector3d &pHomg, int pix)
{
affHom[0] = pHomg[0] * affineMat(pix, 0) + pHomg[1] * affineMat(pix, 1) + pHomg[2] * affineMat(pix, 2);
affHom[1] = pHomg[0] * affineMat(pix, 3) + pHomg[1] * affineMat(pix, 4) + pHomg[2] * affineMat(pix, 5);
}

inline static bool binarySearch(int irow, vector<vector<int> > &neighMat, int kPixel)
{
int low = 0;
int high = neighMat[kPixel].size() - 1;
while (low <= high)
{
int mid = (low + high) / 2;
if (neighMat[kPixel][mid] == irow)
{
return true;
}
else if (neighMat[kPixel][mid] < irow)
{
low = mid + 1;
}
else
{
high = mid - 1;
}
}
return false;
}

inline static bool indexSearch(int irow, vector<vector<int>> &neighMat, int kPixel, int &ind)
{
int low = 0;
int high = neighMat[kPixel].size() - 1;
ind = -1;
while (low <= high)
{
int mid = (low + high) / 2;
if (neighMat[kPixel][mid] == irow)
{
ind = mid;
return true;
}
else if (neighMat[kPixel][mid] < irow)
{
low = mid + 1;
}
else
{
high = mid - 1;
}
}
return false;
}
inline static double filterPixelObj(VectorXd &img0, VectorXd &img1, SparseMatrix<double> &transMat, int iPixel)
{
double s1 = 0;
for (SparseMatrix<double>::InnerIterator it(transMat, iPixel); it; ++it)
{
int irow = it.row();
s1 += it.value() * img0(irow);
}
s1 -= img1(iPixel);
s1 *= s1;
return s1;
}

inline static int bSearch(vector<int> &ind, int target)
{
int low = 0;
int high = ind.size() - 1;
while (low <= high)
{
int mid = low + (high - low) / 2;
if (ind[mid] == target)
{
return mid;
}
else if (ind[mid] < target)
{
low = mid + 1;
}
else
{
high = mid - 1;
}
}
return -1;
}
inline static double offilterPixelObj(VectorXd &img0, VectorXd &img1, SparseMatrix<double> &transMat, int iPixel, SparseMatrix<double> &DcomMat, double lambda_1, int reserveCap)
{
double s1 = 0;
vector<int> nonzeroInd;
vector<int> nonZeroVal;
nonzeroInd.reserve(reserveCap);
nonZeroVal.reserve(reserveCap);
for (SparseMatrix<double>::InnerIterator it(transMat, iPixel); it; ++it)
{
int irow = it.row();
s1 += it.value() * img0(irow);
nonzeroInd.push_back(irow);
nonZeroVal.push_back(it.value());
}
s1 -= img1(iPixel);
s1 *= s1;
for (SparseMatrix<double>::InnerIterator it(DcomMat, iPixel); it; ++it)
{
int irow = it.row();
int markPos = bSearch(nonzeroInd, irow);
if (markPos != -1)
{
s1 += lambda_1 * it.value() * nonZeroVal[markPos];
}
}
return s1;
}

inline static double iterfilterPixelObj(VectorXd &img0, VectorXd &img1, SparseMatrix<double> &transMat, int iPixel, double &gamma, int minGradInd, bool &flag)
{
double s1 = 0;
flag = false;
for (SparseMatrix<double>::InnerIterator it(transMat, iPixel); it; ++it)
{
int irow = it.row();
if (irow == minGradInd)
{
flag = true;
s1 += (it.value() + gamma * (1 - it.value())) * img0(irow);
}
else
{
s1 += (it.value() + gamma * (-it.value())) * img0(irow);
}
}
if (!flag)
{
s1 += gamma * img0(minGradInd);
}
s1 -= img1(iPixel);
s1 *= s1;
return s1;
}

inline static double totalFilterPixelObj(VectorXd &img0, VectorXd &img1, SparseMatrix<double> &transMat, int iPixel)
{
double s1 = 0;
for (SparseMatrix<double>::InnerIterator it(transMat, iPixel); it; ++it)
{
int irow = it.row();
s1 += it.value() * img0(irow);
}
s1 -= img1(iPixel);
s1 *= s1;
return s1;
}

inline static double ofTotalfilterPixelObj(VectorXd &img0, VectorXd &img1, SparseMatrix<double> &transMat, int iPixel, int reserveCap, double lambda_1, SparseMatrix<double> &DcompMat)
{
double s1 = 0;
double dtsum = 0;
vector<int> nonzeroInd;
vector<int> nonZeroVal;
nonzeroInd.reserve(reserveCap);
nonZeroVal.reserve(reserveCap);
for (SparseMatrix<double>::InnerIterator it(DcompMat, iPixel); it; ++it)
{
int irow = it.row();
nonzeroInd.push_back(irow);
nonZeroVal.push_back(it.value());
}

for (SparseMatrix<double>::InnerIterator it(transMat, iPixel); it; ++it)
{
int irow = it.row();
s1 += it.value() * img0(irow);
int ind = -1;
int posMark = bSearch(nonzeroInd, irow);
if (posMark != -1)
{
dtsum += it.value() * nonZeroVal[posMark];
}
}

s1 -= img1(iPixel);
s1 *= s1;
dtsum *= lambda_1;
s1 += dtsum;
return s1;
}

inline static double reIterfilterPixelObj(VectorXd &img0, VectorXd &img1, SparseMatrix<double> &transMat, int iPixel, double &gamma, vector<vector<int> > &neighMat, VectorXi &visited, VectorXd &transPixel)
{
double s1 = 0;
for (SparseMatrix<double>::InnerIterator it(transMat, iPixel); it; ++it)
{
int irow = it.row();
int ind = -1;
if (indexSearch(irow, neighMat, iPixel, ind))
{
visited[ind] = 1;
s1 += ((1 - gamma) * it.value() + gamma * transPixel(ind)) * img0(irow);
}
else
{
s1 += (1 - gamma) * it.value() * img0(irow);
}
}
for (int i = 0; i != neighMat[iPixel].size(); ++i)
{
if (!visited[i])
{
s1 += gamma * transPixel(i) * img0(neighMat[iPixel][i]);
}
}
s1 -= img1(iPixel);
s1 *= s1;
return s1;
}

inline static double ofIterfilterPixelObj(VectorXd &img0, VectorXd &img1, SparseMatrix<double> &transMat, int iPixel, double &gamma, vector<vector<int> > &neighMat, VectorXi &visited, VectorXd &transPixel,
int reserveCap, double lambda_1, SparseMatrix<double> &DcompMat)
{
double s1 = 0;
double dtsum = 0;
vector<int> nonzeroInd;
vector<int> nonZeroVal;
nonzeroInd.reserve(reserveCap);
nonZeroVal.reserve(reserveCap);
for (SparseMatrix<double>::InnerIterator it(DcompMat, iPixel); it; ++it)
{
int irow = it.row();
nonzeroInd.push_back(irow);
nonZeroVal.push_back(it.value());
}

for (SparseMatrix<double>::InnerIterator it(transMat, iPixel); it; ++it)
{
int irow = it.row();
int ind = -1;
int posMark = bSearch(nonzeroInd, irow);
if (indexSearch(irow, neighMat, iPixel, ind))
{
visited[ind] = 1;
s1 += ((1 - gamma) * it.value() + gamma * transPixel(ind)) * img0(irow);
if (posMark != -1)
{
dtsum += ((1 - gamma) * it.value() + gamma * transPixel(ind)) * nonZeroVal[posMark];
}
}
else
{
s1 += (1 - gamma) * it.value() * img0(irow);
if (posMark != -1)
{
dtsum += (1 - gamma) * it.value() * nonZeroVal[posMark];
}
}
}
for (int i = 0; i != neighMat[iPixel].size(); ++i)
{
if (!visited[i])
{
s1 += gamma * transPixel(i) * img0(neighMat[iPixel][i]);
int posMark = bSearch(nonzeroInd, neighMat[iPixel][i]);
if (posMark != -1)
{
dtsum += gamma * transPixel(i) * nonZeroVal[posMark];
}
}
}
s1 -= img1(iPixel);
s1 *= s1;
dtsum *= lambda_1;
s1 += dtsum;
return s1;
}

inline static void normalizeImg(Mat &img, float minP, float maxP)
{
float minVal = INT_MAX;
float maxVal = INT_MIN;
int rows = img.rows;
int cols = img.cols;
for (int i = 0; i < rows; ++i)
{
for (int j = 0; j < cols; ++j)
{
if (minVal > img.at<float>(i, j))
{
minVal = img.at<float>(i, j);
}
if (maxVal < img.at<float>(i, j))
{
maxVal = img.at<float>(i, j);
}
}
}
for (int i = 0; i < rows; ++i)
{
for (int j = 0; j < cols; ++j)
{
img.at<float>(i, j) = (img.at<float>(i, j) - minVal) / (maxVal - minVal) * (maxP - minP) + minP;
}
}
}

inline static void scaleImg(Mat &img1, Mat &img2, double minP, double maxP)
{
double minVal = INT_MAX;
double maxVal = INT_MIN;
int rows = img1.rows;
int cols = img1.cols;
for (int i = 0; i < rows; ++i)
{
for (int j = 0; j < cols; ++j)
{
if (minVal > img1.at<double>(i, j))
{
minVal = img1.at<double>(i, j);
}
if (minVal > img2.at<double>(i, j))
{
minVal = img2.at<double>(i, j);
}
if (maxVal < img1.at<double>(i, j))
{
maxVal = img1.at<double>(i, j);
}
if (maxVal < img2.at<double>(i, j))
{
maxVal = img2.at<double>(i, j);
}
}
}
for (int i = 0; i < rows; ++i)
{
for (int j = 0; j < cols; ++j)
{
img1.at<double>(i, j) = (img1.at<double>(i, j) - minVal) / (maxVal - minVal) * (maxP - minP) + minP;
img2.at<double>(i, j) = (img2.at<double>(i, j) - minVal) / (maxVal - minVal) * (maxP - minP) + minP;
}
}
}

inline static double getTransElement(SparseMatrix<double> &transMat, int iPixel, int ipx, vector<vector<int> > &neighMat)
{
double elem = 0;
for (SparseMatrix<double>::InnerIterator it(transMat, iPixel); it; ++it)
{
int irow = it.row();
if (irow == neighMat[iPixel][ipx])
{
return elem;
}
}
return elem;
}

inline static double denomTransMat(SparseMatrix<double> &transMat, int iPixel, vector<vector<int> > &neighMat, VectorXd &transPixel, VectorXd &gradAry)
{
double denom = 0.0;
for (int ipx = 0; ipx != neighMat[iPixel].size(); ++ipx)
{
denom += gradAry[neighMat[iPixel][ipx]] * (-getTransElement(transMat, iPixel, ipx, neighMat) + transPixel(ipx));
}
return denom;
}
}
