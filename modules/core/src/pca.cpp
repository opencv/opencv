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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#include "precomp.hpp"

/****************************************************************************************\
*                                          PCA                                           *
\****************************************************************************************/

namespace cv
{

namespace
{

template<typename T>
static void copyAndShiftByMeanTyped(const Mat& src, Mat& dst, const Mat& mean, bool meanIsRow, double sign)
{
    const int rows = src.rows;
    const int cols = src.cols;
    if (meanIsRow)
    {
        const T* mptr = mean.ptr<T>(0);
        const T s = static_cast<T>(sign);
        parallel_for_(Range(0, rows), [&](const Range& range) {
            for (int r = range.start; r < range.end; ++r)
            {
                const T* srow = src.ptr<T>(r);
                T* drow = dst.ptr<T>(r);
                for (int c = 0; c < cols; ++c)
                    drow[c] = srow[c] + s * mptr[c];
            }
        }, src.total() >= 65536 ? rows : 1);
    }
    else
    {
        parallel_for_(Range(0, rows), [&](const Range& range) {
            for (int r = range.start; r < range.end; ++r)
            {
                const T mval = static_cast<T>(sign) * mean.at<T>(r, 0);
                const T* srow = src.ptr<T>(r);
                T* drow = dst.ptr<T>(r);
                for (int c = 0; c < cols; ++c)
                    drow[c] = srow[c] + mval;
            }
        }, src.total() >= 65536 ? rows : 1);
    }
}

static void shiftByMean(const Mat& src, Mat& dst, const Mat& mean, bool meanIsRow, double sign)
{
    const int depth = dst.depth();
    if (depth == CV_32F)
        copyAndShiftByMeanTyped<float>(src, dst, mean, meanIsRow, sign);
    else if (depth == CV_64F)
        copyAndShiftByMeanTyped<double>(src, dst, mean, meanIsRow, sign);
    else
        CV_Error(Error::StsUnsupportedFormat, "Unsupported PCA depth");
}

static void convertAndCenter(const Mat& src, Mat& tmp, const Mat& mean, bool meanIsRow, int ctype)
{
    if (src.type() == ctype)
    {
        tmp.create(src.size(), ctype);
        shiftByMean(src, tmp, mean, meanIsRow, -1.0);
    }
    else
    {
        src.convertTo(tmp, ctype);
        shiftByMean(tmp, tmp, mean, meanIsRow, -1.0);
    }
}

static void computeScrambledEigenvectors(const Mat& data, const Mat& mean, int flags,
                                         int out_count, int ctype, Mat& eigenvectors)
{
    const bool asCol = (flags & CV_PCA_DATA_AS_COL) != 0;
    const int len = asCol ? data.rows : data.cols;
    const int in_count = asCol ? data.cols : data.rows;
    const size_t elemSize = CV_ELEM_SIZE(ctype);
    const size_t kTargetStripBytes = 4u << 20;
    const int blockLen = std::max(1, std::min(len,
        (int)(kTargetStripBytes / (std::max(1, in_count) * elemSize))));

    Mat evects1(out_count, len, ctype);
    Mat eigenSlice = eigenvectors.rowRange(0, out_count);
    Mat tmp_data;

    for (int pos = 0; pos < len; pos += blockLen)
    {
        const int curLen = std::min(blockLen, len - pos);
        Mat sub = asCol ? data.rowRange(pos, pos + curLen) : data.colRange(pos, pos + curLen);
        Mat mSlice = asCol ? mean.rowRange(pos, pos + curLen) : mean.colRange(pos, pos + curLen);
        convertAndCenter(sub, tmp_data, mSlice, !asCol, ctype);

        Mat outStrip = evects1.colRange(pos, pos + curLen);
        gemm(eigenSlice, tmp_data, 1, Mat(), 0, outStrip, asCol ? CV_GEMM_B_T : 0);
    }

    eigenvectors = evects1;
    for (int i = 0; i < out_count; i++)
    {
        Mat vec = eigenvectors.row(i);
        normalize(vec, vec);
    }
}

} // namespace

PCA::PCA() {}

PCA::PCA(InputArray data, InputArray _mean, int flags, int maxComponents)
{
    operator()(data, _mean, flags, maxComponents);
}

PCA::PCA(InputArray data, InputArray _mean, int flags, double retainedVariance)
{
    operator()(data, _mean, flags, retainedVariance);
}

PCA& PCA::operator()(InputArray _data, InputArray __mean, int flags, int maxComponents)
{
    Mat data = _data.getMat(), _mean = __mean.getMat();
    int covar_flags = CV_COVAR_SCALE;
    int len, in_count;
    Size mean_sz;

    CV_Assert( data.channels() == 1 );
    if( flags & CV_PCA_DATA_AS_COL )
    {
        len = data.rows;
        in_count = data.cols;
        covar_flags |= CV_COVAR_COLS;
        mean_sz = Size(1, len);
    }
    else
    {
        len = data.cols;
        in_count = data.rows;
        covar_flags |= CV_COVAR_ROWS;
        mean_sz = Size(len, 1);
    }

    int count = std::min(len, in_count), out_count = count;
    if( maxComponents > 0 )
        out_count = std::min(count, maxComponents);

    // "scrambled" way to compute PCA (when cols(A)>rows(A)):
    // B = A'A; B*x=b*x; C = AA'; C*y=c*y -> AA'*y=c*y -> A'A*(A'*y)=c*(A'*y) -> c = b, x=A'*y
    if( len <= in_count )
        covar_flags |= CV_COVAR_NORMAL;

    int ctype = std::max(CV_32F, data.depth());
    mean.create( mean_sz, ctype );

    Mat covar( count, count, ctype );

    if( !_mean.empty() )
    {
        CV_Assert( _mean.size() == mean_sz );
        _mean.convertTo(mean, ctype);
        covar_flags |= CV_COVAR_USE_AVG;
    }

    calcCovarMatrix( data, covar, mean, covar_flags, ctype );
    eigen( covar, eigenvalues, eigenvectors );

    if( !(covar_flags & CV_COVAR_NORMAL) )
    {
        // CV_PCA_DATA_AS_ROW: cols(A)>rows(A). x=A'*y -> x'=y'*A
        // CV_PCA_DATA_AS_COL: rows(A)>cols(A). x=A''*y -> x'=y'*A'
        computeScrambledEigenvectors(data, mean, flags, out_count, ctype, eigenvectors);
    }

    if( eigenvalues.rows > out_count )
        eigenvalues = eigenvalues.rowRange(0,out_count).clone();
    if( eigenvectors.rows > out_count )
        eigenvectors = eigenvectors.rowRange(0,out_count).clone();
    return *this;
}

void PCA::write(FileStorage& fs ) const
{
    CV_Assert( fs.isOpened() );

    fs << "name" << "PCA";
    fs << "vectors" << eigenvectors;
    fs << "values" << eigenvalues;
    fs << "mean" << mean;
}

void PCA::read(const FileNode& fn)
{
    CV_Assert( !fn.empty() );
    CV_Assert( (String)fn["name"] == "PCA" );

    cv::read(fn["vectors"], eigenvectors);
    cv::read(fn["values"], eigenvalues);
    cv::read(fn["mean"], mean);
}

template <typename T>
int computeCumulativeEnergy(const Mat& eigenvalues, double retainedVariance)
{
    CV_DbgAssert( eigenvalues.type() == DataType<T>::type );

    Mat g(eigenvalues.size(), DataType<T>::type);

    for(int ig = 0; ig < g.rows; ig++)
    {
        g.at<T>(ig, 0) = 0;
        for(int im = 0; im <= ig; im++)
        {
            g.at<T>(ig,0) += eigenvalues.at<T>(im,0);
        }
    }

    int L;

    for(L = 0; L < eigenvalues.rows; L++)
    {
        double energy = g.at<T>(L, 0) / g.at<T>(g.rows - 1, 0);
        if(energy > retainedVariance)
            break;
    }

    L = std::min(eigenvalues.rows, std::max(2, L));

    return L;
}

PCA& PCA::operator()(InputArray _data, InputArray __mean, int flags, double retainedVariance)
{
    Mat data = _data.getMat(), _mean = __mean.getMat();
    int covar_flags = CV_COVAR_SCALE;
    int len, in_count;
    Size mean_sz;

    CV_Assert( data.channels() == 1 );
    if( flags & CV_PCA_DATA_AS_COL )
    {
        len = data.rows;
        in_count = data.cols;
        covar_flags |= CV_COVAR_COLS;
        mean_sz = Size(1, len);
    }
    else
    {
        len = data.cols;
        in_count = data.rows;
        covar_flags |= CV_COVAR_ROWS;
        mean_sz = Size(len, 1);
    }

    CV_Assert( retainedVariance > 0 && retainedVariance <= 1 );

    int count = std::min(len, in_count);

    // "scrambled" way to compute PCA (when cols(A)>rows(A)):
    // B = A'A; B*x=b*x; C = AA'; C*y=c*y -> AA'*y=c*y -> A'A*(A'*y)=c*(A'*y) -> c = b, x=A'*y
    if( len <= in_count )
        covar_flags |= CV_COVAR_NORMAL;

    int ctype = std::max(CV_32F, data.depth());
    mean.create( mean_sz, ctype );

    Mat covar( count, count, ctype );

    if( !_mean.empty() )
    {
        CV_Assert( _mean.size() == mean_sz );
        _mean.convertTo(mean, ctype);
        covar_flags |= CV_COVAR_USE_AVG;
    }

    calcCovarMatrix( data, covar, mean, covar_flags, ctype );
    eigen( covar, eigenvalues, eigenvectors );

    // compute the cumulative energy content for each eigenvector
    int L;
    if (ctype == CV_32F)
        L = computeCumulativeEnergy<float>(eigenvalues, retainedVariance);
    else
        L = computeCumulativeEnergy<double>(eigenvalues, retainedVariance);

    if( !(covar_flags & CV_COVAR_NORMAL) )
    {
        // CV_PCA_DATA_AS_ROW: cols(A)>rows(A). x=A'*y -> x'=y'*A
        // CV_PCA_DATA_AS_COL: rows(A)>cols(A). x=A''*y -> x'=y'*A'
        computeScrambledEigenvectors(data, mean, flags, L, ctype, eigenvectors);
    }

    // use clone() to physically copy the data and thus deallocate the original matrices
    if( eigenvalues.rows > L )
        eigenvalues = eigenvalues.rowRange(0,L).clone();
    if( eigenvectors.rows > L )
        eigenvectors = eigenvectors.rowRange(0,L).clone();

    return *this;
}

void PCA::project(InputArray _data, OutputArray result) const
{
    Mat data = _data.getMat();
    CV_Assert( !mean.empty() && !eigenvectors.empty() &&
        ((mean.rows == 1 && mean.cols == data.cols) || (mean.cols == 1 && mean.rows == data.rows)));
    const bool meanIsRow = (mean.rows == 1);
    const int ctype = mean.type();
    const int D = meanIsRow ? data.cols : data.rows;
    const int N = meanIsRow ? data.rows : data.cols;
    const int k = eigenvectors.rows;
    const int resRows = meanIsRow ? N : k;
    const int resCols = meanIsRow ? k : N;

    Mat dst;
    const bool aliased = result.isMat() && result.getMat().datastart == data.datastart;
    if (aliased)
        dst.create(resRows, resCols, ctype);
    else
    {
        result.create(resRows, resCols, ctype);
        dst = result.getMat();
    }

    const size_t elemSize = CV_ELEM_SIZE(ctype);
    const size_t kTargetStripBytes = 4u << 20;
    const int blockFeat = std::max(1, std::min(D, (int)(kTargetStripBytes / elemSize)));
    const int blockSamples = std::max(1, std::min(N,
        (int)(kTargetStripBytes / ((size_t)blockFeat * elemSize))));

    Mat tmp_data;
    for (int s0 = 0; s0 < N; s0 += blockSamples)
    {
        const int s1 = std::min(N, s0 + blockSamples);
        Mat outSlice = meanIsRow ? dst.rowRange(s0, s1) : dst.colRange(s0, s1);
        for (int f0 = 0; f0 < D; f0 += blockFeat)
        {
            const int f1 = std::min(D, f0 + blockFeat);
            Mat sub = meanIsRow ? data(Range(s0, s1), Range(f0, f1))
                                : data(Range(f0, f1), Range(s0, s1));
            Mat mSlice = meanIsRow ? mean.colRange(f0, f1) : mean.rowRange(f0, f1);
            convertAndCenter(sub, tmp_data, mSlice, meanIsRow, ctype);

            Mat eSlice = eigenvectors.colRange(f0, f1);
            gemm(meanIsRow ? tmp_data : eSlice,
                 meanIsRow ? eSlice : tmp_data,
                 1, (f0 == 0) ? Mat() : outSlice, (f0 == 0) ? 0.0 : 1.0,
                 outSlice, meanIsRow ? GEMM_2_T : 0);
        }
    }

    if (aliased)
        dst.copyTo(result);
}

Mat PCA::project(InputArray data) const
{
    Mat result;
    project(data, result);
    return result;
}

void PCA::backProject(InputArray _data, OutputArray result) const
{
    Mat data = _data.getMat();
    CV_Assert( !mean.empty() && !eigenvectors.empty() &&
        ((mean.rows == 1 && eigenvectors.rows == data.cols) ||
         (mean.cols == 1 && eigenvectors.rows == data.rows)));

    Mat tmp_data;
    data.convertTo(tmp_data, mean.type());
    if( mean.rows == 1 )
        gemm( tmp_data, eigenvectors, 1, Mat(), 0, result, 0 );
    else
        gemm( eigenvectors, tmp_data, 1, Mat(), 0, result, GEMM_1_T );

    Mat dst = result.getMat();
    shiftByMean(dst, dst, mean, mean.rows == 1, 1.0);
}

Mat PCA::backProject(InputArray data) const
{
    Mat result;
    backProject(data, result);
    return result;
}

}

void cv::PCACompute(InputArray data, InputOutputArray mean,
                    OutputArray eigenvectors, int maxComponents)
{
    CV_INSTRUMENT_REGION();

    PCA pca;
    pca(data, mean, 0, maxComponents);
    pca.mean.copyTo(mean);
    pca.eigenvectors.copyTo(eigenvectors);
}

void cv::PCACompute(InputArray data, InputOutputArray mean,
                    OutputArray eigenvectors, OutputArray eigenvalues,
                    int maxComponents)
{
    CV_INSTRUMENT_REGION();

    PCA pca;
    pca(data, mean, 0, maxComponents);
    pca.mean.copyTo(mean);
    pca.eigenvectors.copyTo(eigenvectors);
    pca.eigenvalues.copyTo(eigenvalues);
}

void cv::PCACompute(InputArray data, InputOutputArray mean,
                    OutputArray eigenvectors, double retainedVariance)
{
    CV_INSTRUMENT_REGION();

    PCA pca;
    pca(data, mean, 0, retainedVariance);
    pca.mean.copyTo(mean);
    pca.eigenvectors.copyTo(eigenvectors);
}

void cv::PCACompute(InputArray data, InputOutputArray mean,
                    OutputArray eigenvectors, OutputArray eigenvalues,
                    double retainedVariance)
{
    CV_INSTRUMENT_REGION();

    PCA pca;
    pca(data, mean, 0, retainedVariance);
    pca.mean.copyTo(mean);
    pca.eigenvectors.copyTo(eigenvectors);
    pca.eigenvalues.copyTo(eigenvalues);
}

void cv::PCAProject(InputArray data, InputArray mean,
                    InputArray eigenvectors, OutputArray result)
{
    CV_INSTRUMENT_REGION();

    PCA pca;
    pca.mean = mean.getMat();
    pca.eigenvectors = eigenvectors.getMat();
    pca.project(data, result);
}

void cv::PCABackProject(InputArray data, InputArray mean,
                    InputArray eigenvectors, OutputArray result)
{
    CV_INSTRUMENT_REGION();

    PCA pca;
    pca.mean = mean.getMat();
    pca.eigenvectors = eigenvectors.getMat();
    pca.backProject(data, result);
}
