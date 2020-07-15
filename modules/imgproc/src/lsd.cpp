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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistributions in binary form must reproduce the above copyright notice,
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
#include <vector>

#if defined(_MSC_VER)
#   pragma warning(disable:4702)  // unreachable code
#endif

namespace cv {

class LineSegmentDetectorImpl CV_FINAL : public LineSegmentDetector
{
public:

/**
 * Create a LineSegmentDetectorImpl object. Specifying scale, number of subdivisions for the image, should the lines be refined and other constants as follows:
 *
 * @param _refine       How should the lines found be refined?
 *                      LSD_REFINE_NONE - No refinement applied.
 *                      LSD_REFINE_STD  - Standard refinement is applied. E.g. breaking arches into smaller line approximations.
 *                      LSD_REFINE_ADV  - Advanced refinement. Number of false alarms is calculated,
 *                                    lines are refined through increase of precision, decrement in size, etc.
 * @param _scale        The scale of the image that will be used to find the lines. Range (0..1].
 * @param _sigma_scale  Sigma for Gaussian filter is computed as sigma = _sigma_scale/_scale.
 * @param _quant        Bound to the quantization error on the gradient norm.
 * @param _ang_th       Gradient angle tolerance in degrees.
 * @param _log_eps      Detection threshold: -log10(NFA) > _log_eps
 * @param _density_th   Minimal density of aligned region points in rectangle.
 * @param _n_bins       Number of bins in pseudo-ordering of gradient modulus.
 */
    LineSegmentDetectorImpl(int _refine = LSD_REFINE_STD, double _scale = 0.8,
        double _sigma_scale = 0.6, double _quant = 2.0, double _ang_th = 22.5,
        double _log_eps = 0, double _density_th = 0.7, int _n_bins = 1024);

/**
 * Detect lines in the input image.
 *
 * @param _image    A grayscale(CV_8UC1) input image.
 *                  If only a roi needs to be selected, use
 *                  lsd_ptr->detect(image(roi), ..., lines);
 *                  lines += Scalar(roi.x, roi.y, roi.x, roi.y);
 * @param _lines    Return: A vector of Vec4i or Vec4f elements specifying the beginning and ending point of a line.
 *                          Where Vec4i/Vec4f is (x1, y1, x2, y2), point 1 is the start, point 2 - end.
 *                          Returned lines are strictly oriented depending on the gradient.
 * @param width     Return: Vector of widths of the regions, where the lines are found. E.g. Width of line.
 * @param prec      Return: Vector of precisions with which the lines are found.
 * @param nfa       Return: Vector containing number of false alarms in the line region, with precision of 10%.
 *                          The bigger the value, logarithmically better the detection.
 *                              * -1 corresponds to 10 mean false alarms
 *                              * 0 corresponds to 1 mean false alarm
 *                              * 1 corresponds to 0.1 mean false alarms
 *                          This vector will be calculated _only_ when the objects type is REFINE_ADV
 */
    void detect(InputArray _image, OutputArray _lines,
                OutputArray width = noArray(), OutputArray prec = noArray(),
                OutputArray nfa = noArray()) CV_OVERRIDE;

/**
 * Draw lines on the given canvas.
 *
 * @param image     The image, where lines will be drawn.
 *                  Should have the size of the image, where the lines were found
 * @param lines     The lines that need to be drawn
 */
    void drawSegments(InputOutputArray _image, InputArray lines) CV_OVERRIDE;

/**
 * Draw both vectors on the image canvas. Uses blue for lines 1 and red for lines 2.
 *
 * @param size      The size of the image, where lines1 and lines2 were found.
 * @param lines1    The first lines that need to be drawn. Color - Blue.
 * @param lines2    The second lines that need to be drawn. Color - Red.
 * @param image     An optional image, where lines will be drawn.
 *                  Should have the size of the image, where the lines were found
 * @return          The number of mismatching pixels between lines1 and lines2.
 */
    int compareSegments(const Size& size, InputArray lines1, InputArray lines2, InputOutputArray _image = noArray()) CV_OVERRIDE;

private:
    LineSegmentDetectorImpl& operator= (const LineSegmentDetectorImpl&); // to quiet MSVC
};

/////////////////////////////////////////////////////////////////////////////////////////

CV_EXPORTS Ptr<LineSegmentDetector> createLineSegmentDetector(
        int _refine, double _scale, double _sigma_scale, double _quant, double _ang_th,
        double _log_eps, double _density_th, int _n_bins)
{
    return makePtr<LineSegmentDetectorImpl>(
            _refine, _scale, _sigma_scale, _quant, _ang_th,
            _log_eps, _density_th, _n_bins);
}

/////////////////////////////////////////////////////////////////////////////////////////

LineSegmentDetectorImpl::LineSegmentDetectorImpl(int _refine, double _scale, double _sigma_scale, double _quant,
        double _ang_th, double _log_eps, double _density_th, int _n_bins)
{
    CV_Assert(_scale > 0 && _sigma_scale > 0 && _quant >= 0 &&
              _ang_th > 0 && _ang_th < 180 && _density_th >= 0 && _density_th < 1 &&
              _n_bins > 0);
    CV_UNUSED(_refine); CV_UNUSED(_log_eps);
    CV_Error(Error::StsNotImplemented, "Implementation has been removed due original code license issues");
}

void LineSegmentDetectorImpl::detect(InputArray _image, OutputArray _lines,
                OutputArray _width, OutputArray _prec, OutputArray _nfa)
{
    CV_INSTRUMENT_REGION();

    CV_UNUSED(_image); CV_UNUSED(_lines);
    CV_UNUSED(_width); CV_UNUSED(_prec); CV_UNUSED(_nfa);
    CV_Error(Error::StsNotImplemented, "Implementation has been removed due original code license issues");
}

void LineSegmentDetectorImpl::drawSegments(InputOutputArray _image, InputArray lines)
{
    CV_INSTRUMENT_REGION();

    CV_Assert(!_image.empty() && (_image.channels() == 1 || _image.channels() == 3));

    if (_image.channels() == 1)
    {
        cvtColor(_image, _image, COLOR_GRAY2BGR);
    }

    Mat _lines = lines.getMat();
    const int N = _lines.checkVector(4);

    CV_Assert(_lines.depth() == CV_32F || _lines.depth() == CV_32S);

    // Draw segments
    if (_lines.depth() == CV_32F)
    {
        for (int i = 0; i < N; ++i)
        {
            const Vec4f& v = _lines.at<Vec4f>(i);
            const Point2f b(v[0], v[1]);
            const Point2f e(v[2], v[3]);
            line(_image, b, e, Scalar(0, 0, 255), 1);
        }
    }
    else
    {
        for (int i = 0; i < N; ++i)
        {
            const Vec4i& v = _lines.at<Vec4i>(i);
            const Point2i b(v[0], v[1]);
            const Point2i e(v[2], v[3]);
            line(_image, b, e, Scalar(0, 0, 255), 1);
        }
    }
}


int LineSegmentDetectorImpl::compareSegments(const Size& size, InputArray lines1, InputArray lines2, InputOutputArray _image)
{
    CV_INSTRUMENT_REGION();

    Size sz = size;
    if (_image.needed() && _image.size() != size) sz = _image.size();
    CV_Assert(!sz.empty());

    Mat_<uchar> I1 = Mat_<uchar>::zeros(sz);
    Mat_<uchar> I2 = Mat_<uchar>::zeros(sz);

    Mat _lines1 = lines1.getMat();
    Mat _lines2 = lines2.getMat();
    const int N1 = _lines1.checkVector(4);
    const int N2 = _lines2.checkVector(4);

    CV_Assert(_lines1.depth() == CV_32F || _lines1.depth() == CV_32S);
    CV_Assert(_lines2.depth() == CV_32F || _lines2.depth() == CV_32S);

    if (_lines1.depth() == CV_32S)
        _lines1.convertTo(_lines1, CV_32F);
    if (_lines2.depth() == CV_32S)
        _lines2.convertTo(_lines2, CV_32F);

    // Draw segments
    for(int i = 0; i < N1; ++i)
    {
        const Point2f b(_lines1.at<Vec4f>(i)[0], _lines1.at<Vec4f>(i)[1]);
        const Point2f e(_lines1.at<Vec4f>(i)[2], _lines1.at<Vec4f>(i)[3]);
        line(I1, b, e, Scalar::all(255), 1);
    }
    for(int i = 0; i < N2; ++i)
    {
        const Point2f b(_lines2.at<Vec4f>(i)[0], _lines2.at<Vec4f>(i)[1]);
        const Point2f e(_lines2.at<Vec4f>(i)[2], _lines2.at<Vec4f>(i)[3]);
        line(I2, b, e, Scalar::all(255), 1);
    }

    // Count the pixels that don't agree
    Mat Ixor;
    bitwise_xor(I1, I2, Ixor);
    int N = countNonZero(Ixor);

    if (_image.needed())
    {
        CV_Assert(_image.channels() == 3);
        Mat img = _image.getMatRef();
        CV_Assert(img.isContinuous() && I1.isContinuous() && I2.isContinuous());

        for (unsigned int i = 0; i < I1.total(); ++i)
        {
            uchar i1 = I1.ptr()[i];
            uchar i2 = I2.ptr()[i];
            if (i1 || i2)
            {
                unsigned int base_idx = i * 3;
                if (i1) img.ptr()[base_idx] = 255;
                else img.ptr()[base_idx] = 0;
                img.ptr()[base_idx + 1] = 0;
                if (i2) img.ptr()[base_idx + 2] = 255;
                else img.ptr()[base_idx + 2] = 0;
            }
        }
    }

    return N;
}

} // namespace cv
