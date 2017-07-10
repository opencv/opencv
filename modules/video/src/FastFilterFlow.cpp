/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
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

#include <glpk.h>
#include "FastFilterFlow.hpp"

//
// 2D fast filter flow algorithm from the following paper:
// Sathya N. Ravi, Yunyang Xiong, Lopamudra Mukherjee, Vikas Singh
// Filter Flow made Practical: Massively Parallel and Lock-Free
// 2017 Conference on Computer Vision and Pattern Recognition

namespace cv
{
static Mat calcFastFilterFlow(Mat &from, Mat &to, MatrixXd &centroidMat) {

const double epison = 1e-6;
const double appZero = 1e-10;

int rows = from.rows;
int cols = from.cols;
int nPixels = rows * cols;

int nImgs = 2;
VectorXd scalImgAry[2];
scalImgAry[0].resize(rows * cols);
scalImgAry[1].resize(rows * cols);
double sumPixel = 0;
imgMatToArray(from, scalImgAry[i]);
imgMatToArray(to, scalImgAry[i]);

SparseMatrix <double> transMat(nPixels, nPixels);
int neigbWidth = 5;
int neigbHeight = 5;
int reserveCap = 4 * (neigbHeight + 2) * (neigbWidth + 2);
transMat.reserve(VectorXd::Constant(nPixels, reserveCap));
SparseMatrix<double> dcompMat(nPixels, nPixels);

vector<vector<int> > neighMat;

int affineX = 2;
int affineY = 3;
int affineEntry = affineX * affineY;
MatrixXd AffineMat = MatrixXd::Zero(nPixels, affineEntry);
MatrixXd tmpAffineMat = MatrixXd::Zero(nPixels, affineEntry);
dcompMat.reserve(VectorXd::Constant(nPixels, reserveCap));
neighborMatrix(from, neighMat, neigbWidth, neigbHeight);

double alpha = 0.5;
int iterMax = 2;
int iterMaxN = iterMax;
int outIterMax = 1;
const int GammaCon = 7000;
vector<vector<double> > funVal(iterMax, vector<double>(nPixels, 0));
vector<vector<double> > dualGap(iterMax, vector<double>(nPixels, 0));
VectorXd gradAry(nPixels);
Vector3d pHomog(0, 0, 1);
Vector2d affHomog(0, 0);
Vector2d tmpAffine(0, 0);
Vector2d gradAffine(0, 0);
MatrixXd gradApix[4];
gradApix[0].resize(affineX, affineY);
gradApix[1].resize(affineX, affineY);
gradApix[2].resize(affineX, affineY);
gradApix[3].resize(affineX, affineY);
VectorXd tGranApix2(affineX * affineY);
VectorXd tGranApix3(affineX * affineY);
VectorXd affineNeighSum(affineX * affineY);

int apixNorm = 10;
double ptrFlowX = 0;
double ptrFlowY = 0;
int pfx = 0;
int pfy = 0;
for (int i = 0; i != nPixels; ++i)
{
transMat.insert(i, i) = 1;
}

double lambda_1 = 1;
double lambda_3 = 0.005;
double lambda_2 = 1;
int ia[8000];
int ja[8000];
double ar[8000];
char rName[3];
char cName[4000][10];
rName[0] = 'c';
rName[2] = '\0';
rName[1] = '\0';
memset(cName, '\0', sizeof(cName));
for (int i = 0; i != reserveCap; ++i)
{
itoa(i + 1, cName[i], 10);
}

vector<vector<double> > funcVal(outIterMax, vector<double>(iterMaxN, 0));
vector<vector<double> > fixAFuncVal(outIterMax, vector<double>(iterMax, 0));
vector<vector<double> > fixTFuncVal(outIterMax, vector<double>(iterMax, 0));
vector<vector<double> > finalFuncVal(outIterMax, vector<double>(iterMax, 0));
vector<double> tuningSeq(nPixels, 0);
vector<double> pixelVal(nPixels, 0);

//Preallocate
VectorXd transPix(reserveCap);
MatrixXi eqnLhsMat = MatrixXi::Zero(3, affineEntry + reserveCap);

//Obtain initial filter flow
for (int kPixel = 0; kPixel != nPixels; ++kPixel)
{
int pr = kPixel / cols;
int pc = kPixel % cols;

//Solve sub problems using a linear programming solver
glp_prob *mip = glp_create_prob();
glp_term_out(GLP_OFF);
glp_set_prob_name(mip, "InitialT");
glp_set_obj_dir(mip, GLP_MIN);
glp_add_rows(mip, 3);

for (int ipx = 0; ipx != neighMat[kPixel].size(); ++ipx)
{
int ir = neighMat[kPixel][ipx] / cols;
int ic = neighMat[kPixel][ipx] % cols;
eqnLhsMat(0, ipx) = 1;
eqnLhsMat(1, ipx) = ir - pr;
eqnLhsMat(2, ipx) = ic - pc;
}

rName[0] = 'c';
rName[1] = '1';
glp_set_row_name(mip, 1, rName);
glp_set_row_bnds(mip, 1, GLP_FX, 1.0, 1.0);
rName[1] = '2';
glp_set_row_name(mip, 2, rName);
glp_set_row_bnds(mip, 2, GLP_FX, centroidMat(kPixel, 0), centroidMat(kPixel, 0));
rName[2] = '3';
glp_set_row_name(mip, 3, rName);
glp_set_row_bnds(mip, 3, GLP_FX, centroidMat(kPixel, 1), centroidMat(kPixel, 1));

glp_add_cols(mip, neighMat[kPixel].size());
for (int ipx = 0; ipx != neighMat[kPixel].size(); ++ipx)
{
glp_set_col_name(mip, ipx + 1, cName[ipx]);
glp_set_col_bnds(mip, ipx + 1, GLP_DB, 0.0, 1.0);
glp_set_obj_coef(mip, ipx + 1, 0);
}
int cntIter = 0;
for (int ipx = 0; ipx != 3; ++ipx)
{
for (int jpx = 0; jpx != neighMat[kPixel].size(); ++jpx)
{
if (eqnLhsMat(ipx, jpx) != 0)
{
++cntIter;
ia[cntIter] = ipx + 1;
ja[cntIter] = jpx + 1;
ar[cntIter] = eqnLhsMat(ipx, jpx);
}
}
}
glp_load_matrix(mip, cntIter, ia, ja, ar);
glp_simplex(mip, NULL);

for (int ipx = 0; ipx != neighMat[kPixel].size(); ++ipx)
{
transPix(ipx) = glp_get_col_prim(mip, ipx + 1);
}
glp_delete_prob(mip);

for (int iPixel = 0; iPixel != neighMat[kPixel].size(); ++iPixel)
{
if (transPix(iPixel) != 0)
{
transMat.coeffRef(neighMat[kPixel][iPixel], kPixel) = transPix(iPixel);
}
}
}

// Obtain Affine matrix
MatrixXi eqnLhsMata = MatrixXi::Zero(2, affineEntry);
for (int kPixel = 0; kPixel != nPixels; ++kPixel)
{
pixelVal[kPixel] = 0;
int pr = kPixel / cols;
int pc = kPixel % cols;
//Solve sub problems using a linear programming solver
glp_prob *mip = glp_create_prob();
glp_term_out(GLP_OFF);
glp_set_prob_name(mip, "InitialA");
glp_set_obj_dir(mip, GLP_MIN);
glp_add_rows(mip, 2);

eqnLhsMata(0, 0) = -pr;
eqnLhsMata(0, 1) = -pc;
eqnLhsMata(0, 2) = -1;
eqnLhsMata(0, 3) = 0;
eqnLhsMata(0, 4) = 0;
eqnLhsMata(0, 5) = 0;
eqnLhsMata(1, 0) = 0;
eqnLhsMata(1, 1) = 0;
eqnLhsMata(1, 2) = 0;
eqnLhsMata(1, 3) = -pr;
eqnLhsMata(1, 4) = -pc;
eqnLhsMata(1, 5) = -1;

rName[0] = 'c';
for (int ipx = 0; ipx != 2; ++ipx)
{
rName[1] = ipx + 1 + '0';
glp_set_row_name(mip, ipx + 1, rName);
glp_set_row_bnds(mip, ipx + 1, GLP_FX, 0.0, 0);
}

rName[0] = 'x';
glp_add_cols(mip, affineEntry);
for (int ipx = 0; ipx != affineEntry; ++ipx)
{
rName[1] = ipx + 1 + '0';
glp_set_col_name(mip, ipx + 1, rName);
glp_set_col_bnds(mip, ipx + 1, GLP_DB, -25.0, 25.0);
glp_set_obj_coef(mip, ipx + 1, 0);
}

int cntIter = 0;
for (int ipx = 0; ipx != 2; ++ipx)
{
for (int jpx = 0; jpx != affineEntry; ++jpx)
{
if (eqnLhsMata(ipx, jpx) != 0)
{
++cntIter;
ia[cntIter] = ipx + 1;
ja[cntIter] = jpx + 1;
ar[cntIter] = eqnLhsMata(ipx, jpx);
}
}
}
glp_load_matrix(mip, cntIter, ia, ja, ar);
glp_simplex(mip, NULL);
for (int ipx = 0; ipx != affineEntry; ++ipx)
{
tGranApix2(ipx) = glp_get_col_prim(mip, ipx + 1);
}
glp_delete_prob(mip);
AffineMat.row(kPixel) = tGranApix2.transpose();
}

VectorXi visited(reserveCap);
for (int i = 0; i != outIterMax; ++i)
{
int st1 = clock();
for (int j = 0; j != iterMaxN; ++j)
{
#pragma  omp parallel for
for (int kPixel = 0; kPixel != nPixels; ++kPixel)
{
pixelVal[kPixel] = 0;
int pr = kPixel / cols;
int pc = kPixel % cols;
pHomog[0] = pr;
pHomog[1] = pc;
double iTrans = multiplyDenseSparse(scalImgAry[0], transMat, kPixel);
gradAry = 2 * (iTrans - scalImgAry[1][kPixel]) * scalImgAry[0];
addDenseSparse(gradAry, dcompMat, kPixel, lambda_1);
for (int k = 0; k != tGranApix2.size(); ++k)
{
tGranApix2(k) = 0;
affineNeighSum(k) = 0;
}
for (int k = 0; k != neighMat[kPixel].size(); ++k)
{
tGranApix2 += AffineMat.row(neighMat[kPixel][k]).transpose();
affineNeighSum += AffineMat.row(neighMat[kPixel][k]).transpose();
}
double tGran = AffineMat.row(kPixel) * tGranApix2;
tGranApix2 = 2 * (AffineMat.row(kPixel).transpose() * neighMat[kPixel].size() - tGranApix2) * lambda_3;
for (int ki = 0; ki != affineX; ++ki)
{
for (int kj = 0; kj != affineY; ++kj)
{
gradApix[2](ki, kj) = tGranApix2(ki * affineY + kj);
}
}

//Solve sub problems using a linear programming solver
glp_prob *mip = glp_create_prob();
glp_term_out(GLP_OFF);
glp_set_prob_name(mip, "short");
glp_set_obj_dir(mip, GLP_MIN);
glp_add_rows(mip, 3);
eqnLhsMat(0, 0) = -pr;
eqnLhsMat(0, 1) = -pc;
eqnLhsMat(0, 2) = -1;
eqnLhsMat(0, 3) = 0;
eqnLhsMat(0, 4) = 0;
eqnLhsMat(0, 5) = 0;
eqnLhsMat(1, 0) = 0;
eqnLhsMat(1, 1) = 0;
eqnLhsMat(1, 2) = 0;
eqnLhsMat(1, 3) = -pr;
eqnLhsMat(1, 4) = -pc;
eqnLhsMat(1, 5) = -1;
for (int ipx = 0; ipx != neighMat[kPixel].size(); ++ipx)
{
int ir = neighMat[kPixel][ipx] / cols;
int ic = neighMat[kPixel][ipx] % cols;
eqnLhsMat(0, ipx + affineEntry) = ir - pr;
eqnLhsMat(1, ipx + affineEntry) = ic - pc;
eqnLhsMat(2, ipx + affineEntry) = 1;
}
rName[0] = 'c';
for (int ipx = 0; ipx != 2; ++ipx)
{
rName[1] = ipx + 1 + '0';
glp_set_row_name(mip, ipx + 1, rName);
glp_set_row_bnds(mip, ipx + 1, GLP_FX, 0.0, 0);
}
rName[1] = '3';
glp_set_row_name(mip, 3, rName);
glp_set_row_bnds(mip, 3, GLP_FX, 1.0, 1);

rName[0] = 'x';
glp_add_cols(mip, affineEntry + neighMat[kPixel].size());
for (int ipx = 0; ipx != affineEntry; ++ipx)
{
rName[1] = ipx + 1 + '0';
glp_set_col_name(mip, ipx + 1, rName);
glp_set_col_bnds(mip, ipx + 1, GLP_DB, -25.0, 25.0);
glp_set_obj_coef(mip, ipx + 1, tGranApix2(ipx));
}
for (int ipx = 0; ipx != neighMat[kPixel].size(); ++ipx)
{
glp_set_col_name(mip, ipx + 1 + affineEntry, cName[ipx + affineEntry + 1]);
glp_set_col_bnds(mip, ipx + 1 + affineEntry, GLP_DB, 0.0, 1.0);
glp_set_obj_coef(mip, ipx + 1 + affineEntry, gradAry[neighMat[kPixel][ipx]]);
}
int cntIter = 0;
for (int ipx = 0; ipx != 3; ++ipx)
{
for (int jpx = 0; jpx != affineEntry + neighMat[kPixel].size(); ++jpx)
{
if (eqnLhsMat(ipx, jpx) != 0)
{
++cntIter;
ia[cntIter] = ipx + 1;
ja[cntIter] = jpx + 1;
ar[cntIter] = eqnLhsMat(ipx, jpx);
}
}
}
glp_load_matrix(mip, cntIter, ia, ja, ar);
glp_simplex(mip, NULL);

for (int ipx = 0; ipx != affineEntry; ++ipx)
{
tGranApix2(ipx) = glp_get_col_prim(mip, ipx + 1);
}

for (int ipx = 0; ipx != neighMat[kPixel].size(); ++ipx)
{
transPix(ipx) = glp_get_col_prim(mip, ipx + affineEntry + 1);
}
glp_delete_prob(mip);

//Search optimial gamma
visited.setZero(neighMat[kPixel].size());
double gamma = 1;
gamma = 2.0 / (GammaCon + (2 + (j + 1)));
double term1 = 0;
double term2 = 0;
double term3 = 0;
for (int k = 0; k != neighMat[kPixel].size(); ++k)
{
term3 += AffineMat.row(neighMat[kPixel][k]) * AffineMat.row(neighMat[kPixel][k]).transpose();
}

term1 = neighMat[kPixel].size() * AffineMat.row(kPixel).squaredNorm();
term2 = -2 * AffineMat.row(kPixel) * affineNeighSum;
double curVal = ofTotalfilterPixelObj(scalImgAry[0], scalImgAry[1], transMat, kPixel, reserveCap, lambda_1, dcompMat);
curVal += lambda_3 * (term1 + term2 + term3);
pixelVal[kPixel] = curVal;

visited.setZero(neighMat[kPixel].size());
for (SparseMatrix<double>::InnerIterator it(transMat, kPixel); it; ++it)
{
int irow = it.row();
int ind = -1;
if (indexSearch(irow, neighMat, kPixel, ind))
{
visited[ind] = 1;
transMat.coeffRef(irow, kPixel) = ((1 - gamma) * it.value() + gamma * transPix(ind));
}
else
{
transMat.coeffRef(irow, kPixel) = (1 - gamma) * it.value();
}
}
for (int iPixel = 0; iPixel != neighMat[kPixel].size(); ++iPixel)
{
if (!visited[iPixel])
{
transMat.coeffRef(neighMat[kPixel][iPixel], kPixel) = gamma * transPix(iPixel);
}
}
AffineMat.row(kPixel) = (1 - gamma) * AffineMat.row(kPixel) + gamma * tGranApix2.transpose();
}
double fval = 0;
for (int kpixel = 0; kpixel != nPixels; ++kpixel)
{
fval += pixelVal[kpixel];
}
funcVal[i][j] = fval;
}

int r = 0;
int c = 0;
//#pragma  omp parallel for
for (int kPixel = 0; kPixel != nPixels; ++kPixel)
{
double centroidx = 0;
double centroidy = 0;
int pr = kPixel / cols;
int pc = kPixel % cols;
for (SparseMatrix<double>::InnerIterator it(transMat, kPixel); it; ++it)
{
int irow = it.row();
int ir = irow / cols;
int ic = irow % cols;
centroidx += (ir - pr) * it.value();
centroidy += (ic - pc) * it.value();
}
for (int id = 0; id != neighMat[kPixel].size(); ++id)
{
r = neighMat[kPixel][id] / cols;
c = neighMat[kPixel][id] % cols;
dcompMat.coeffRef(neighMat[kPixel][id], kPixel) = ((r - pr - centroidx) * (r - pr - centroidx) + (c - pc - centroidy) * (c - pc - centroidy));
}
}
}

cv::Mat flowX = Mat::zeros(rows, cols, CV_32FC1);
cv::Mat flowY = Mat::zeros(rows, cols, CV_32FC1);
cv::Mat flowMag = Mat::zeros(rows, cols, CV_32FC1);
cv::Mat flowAng = Mat::zeros(rows, cols, CV_32FC1);

for (int i = 0; i != nPixels; ++i)
{
int pRow = i / cols;
int pCol = i % cols;
flowX.at<float>(pRow, pCol) = centroidMat(i, 0);
flowY.at<float>(pRow, pCol) = centroidMat(i, 1);
}
cartToPolar(flowX, flowY, flowMag, flowAng, true);
double magMax = 0;
minMaxLoc(flowMag, 0, &magMax);
flowMag.convertTo(flowMag, -1, 1.0 / magMax);

Mat hsvMat[3];
Mat hsvM;
vector<Mat> channels;
channels.push_back(flowAng);
channels.push_back(Mat::ones(flowAng.size(), CV_32F));
channels.push_back(flowMag);
merge(channels, hsvM);
Mat bgr;
cvtColor(hsvM, bgr, COLOR_HSV2BGR);
return bgr;
}
}
