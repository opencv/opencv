// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include <opencv2/imgproc.hpp>


namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

Image2BlobParams::Image2BlobParams():scalefactor(Scalar::all(1.0)), size(Size()), mean(Scalar()), swapRB(false), ddepth(CV_32F),
                           datalayout(DNN_LAYOUT_NCHW), paddingmode(DNN_PMODE_NULL)
{}

Image2BlobParams::Image2BlobParams(const Scalar& scalefactor_, const Size& size_, const Scalar& mean_, bool swapRB_,
                         int ddepth_, DataLayout datalayout_, ImagePaddingMode mode_):
        scalefactor(scalefactor_), size(size_), mean(mean_), swapRB(swapRB_), ddepth(ddepth_),
        datalayout(datalayout_), paddingmode(mode_)
{}

Mat blobFromImage(InputArray image, const double scalefactor, const Size& size,
        const Scalar& mean, bool swapRB, bool crop, int ddepth)
{
    CV_TRACE_FUNCTION();
    Mat blob;
    blobFromImage(image, blob, scalefactor, size, mean, swapRB, crop, ddepth);
    return blob;
}

void blobFromImage(InputArray image, OutputArray blob, double scalefactor,
        const Size& size, const Scalar& mean, bool swapRB, bool crop, int ddepth)
{
    CV_TRACE_FUNCTION();
    std::vector<Mat> images(1, image.getMat());
    blobFromImages(images, blob, scalefactor, size, mean, swapRB, crop, ddepth);
}

Mat blobFromImages(InputArrayOfArrays images, double scalefactor, Size size,
        const Scalar& mean, bool swapRB, bool crop, int ddepth)
{
    CV_TRACE_FUNCTION();
    Mat blob;
    blobFromImages(images, blob, scalefactor, size, mean, swapRB, crop, ddepth);
    return blob;
}

void blobFromImages(InputArrayOfArrays images_, OutputArray blob_, double scalefactor,
        Size size, const Scalar& mean_, bool swapRB, bool crop, int ddepth)
{
    CV_TRACE_FUNCTION();
    Image2BlobParams param(Scalar::all(scalefactor), size, mean_, swapRB, ddepth);
    if (crop)
        param.paddingmode = DNN_PMODE_CROP_CENTER;
    blobFromImagesWithParams(images_, blob_, param);
}

Mat blobFromImageWithParams(InputArray image, const Image2BlobParams& param)
{
    CV_TRACE_FUNCTION();
    Mat blob;
    blobFromImageWithParams(image, blob, param);
    return blob;
}

void blobFromImageWithParams(InputArray image, OutputArray blob, const Image2BlobParams& param)
{
    CV_TRACE_FUNCTION();
    std::vector<Mat> images(1, image.getMat());
    blobFromImagesWithParams(images, blob, param);
}

Mat blobFromImagesWithParams(InputArrayOfArrays images, const Image2BlobParams& param)
{
    CV_TRACE_FUNCTION();
    Mat blob;
    blobFromImagesWithParams(images, blob, param);
    return blob;
}

void blobFromImagesWithParams(InputArrayOfArrays images_, OutputArray blob_, const Image2BlobParams& param)
{
    CV_TRACE_FUNCTION();
    CV_CheckType(param.ddepth, param.ddepth == CV_32F || param.ddepth == CV_8U,
                 "Blob depth should be CV_32F or CV_8U");

    Size size = param.size;
    std::vector<Mat> images;
    images_.getMatVector(images);
    CV_Assert(!images.empty());

    int nch = images[0].channels();
    Scalar scalefactor = param.scalefactor;

    if (param.ddepth == CV_8U)
    {
        CV_Assert(scalefactor == Scalar::all(1.0) && "Scaling is not supported for CV_8U blob depth");
        CV_Assert(param.mean == Scalar() && "Mean subtraction is not supported for CV_8U blob depth");
    }

    for (size_t i = 0; i < images.size(); i++)
    {
        Size imgSize = images[i].size();
        if (size == Size())
            size = imgSize;
        if (size != imgSize)
        {
            if (param.paddingmode == DNN_PMODE_CROP_CENTER)
            {
                float resizeFactor = std::max(size.width / (float)imgSize.width,
                                              size.height / (float)imgSize.height);
                resize(images[i], images[i], Size(), resizeFactor, resizeFactor, INTER_LINEAR);
                Rect crop(Point(0.5 * (images[i].cols - size.width),
                                0.5 * (images[i].rows - size.height)),
                          size);
                images[i] = images[i](crop);
            }
            else
            {
                if (param.paddingmode == DNN_PMODE_LETTERBOX)
                {
                    float resizeFactor = std::min(size.width / (float)imgSize.width,
                                                  size.height / (float)imgSize.height);
                    int rh = int(imgSize.height * resizeFactor);
                    int rw = int(imgSize.width * resizeFactor);
                    resize(images[i], images[i], Size(rw, rh), INTER_LINEAR);

                    int top = (size.height - rh)/2;
                    int bottom = size.height - top - rh;
                    int left = (size.width - rw)/2;
                    int right = size.width - left - rw;
                    copyMakeBorder(images[i], images[i], top, bottom, left, right, BORDER_CONSTANT);
                }
                else
                    resize(images[i], images[i], size, 0, 0, INTER_LINEAR);
            }
        }

        Scalar mean = param.mean;
        if (param.swapRB)
        {
            std::swap(mean[0], mean[2]);
            std::swap(scalefactor[0], scalefactor[2]);
        }

        if (images[i].depth() == CV_8U && param.ddepth == CV_32F)
            images[i].convertTo(images[i], CV_32F);

        images[i] -= mean;
        multiply(images[i], scalefactor, images[i]);
    }

    size_t nimages = images.size();
    Mat image0 = images[0];
    CV_Assert(image0.dims == 2);

    if (param.datalayout == DNN_LAYOUT_NCHW)
    {
        if (nch == 3 || nch == 4)
        {
            int sz[] = { (int)nimages, nch, image0.rows, image0.cols };
            blob_.create(4, sz, param.ddepth);
            Mat blob = blob_.getMat();
            Mat ch[4];

            for (size_t i = 0; i < nimages; i++)
            {
                const Mat& image = images[i];
                CV_Assert(image.depth() == blob_.depth());
                nch = image.channels();
                CV_Assert(image.dims == 2 && (nch == 3 || nch == 4));
                CV_Assert(image.size() == image0.size());

                for (int j = 0; j < nch; j++)
                    ch[j] = Mat(image.rows, image.cols, param.ddepth, blob.ptr((int)i, j));
                if (param.swapRB)
                    std::swap(ch[0], ch[2]);
                split(image, ch);
            }
        }
        else
        {
            CV_Assert(nch == 1);
            int sz[] = { (int)nimages, 1, image0.rows, image0.cols };
            blob_.create(4, sz, param.ddepth);
            Mat blob = blob_.getMat();

            for (size_t i = 0; i < nimages; i++)
            {
                const Mat& image = images[i];
                CV_Assert(image.depth() == blob_.depth());
                nch = image.channels();
                CV_Assert(image.dims == 2 && (nch == 1));
                CV_Assert(image.size() == image0.size());

                image.copyTo(Mat(image.rows, image.cols, param.ddepth, blob.ptr((int)i, 0)));
            }
        }
    }
    else if (param.datalayout == DNN_LAYOUT_NHWC)
    {
        int sz[] = { (int)nimages, image0.rows, image0.cols, nch};
        blob_.create(4, sz, param.ddepth);
        Mat blob = blob_.getMat();
        int subMatType = CV_MAKETYPE(param.ddepth, nch);
        for (size_t i = 0; i < nimages; i++)
        {
            const Mat& image = images[i];
            CV_Assert(image.depth() == blob_.depth());
            CV_Assert(image.channels() == image0.channels());
            CV_Assert(image.size() == image0.size());
            if (param.swapRB)
            {
                Mat tmpRB;
                cvtColor(image, tmpRB, COLOR_BGR2RGB);
                tmpRB.copyTo(Mat(tmpRB.rows, tmpRB.cols, subMatType, blob.ptr((int)i, 0)));
            }
            else
                image.copyTo(Mat(image.rows, image.cols, subMatType, blob.ptr((int)i, 0)));
        }
    }
    else
        CV_Error(Error::StsUnsupportedFormat, "Unsupported data layout in blobFromImagesWithParams function.");
}

void imagesFromBlob(const cv::Mat& blob_, OutputArrayOfArrays images_)
{
    CV_TRACE_FUNCTION();

    // A blob is a 4 dimensional matrix in floating point precision
    // blob_[0] = batchSize = nbOfImages
    // blob_[1] = nbOfChannels
    // blob_[2] = height
    // blob_[3] = width
    CV_Assert(blob_.depth() == CV_32F);
    CV_Assert(blob_.dims == 4);

    images_.create(cv::Size(1, blob_.size[0]), blob_.depth());

    std::vector<Mat> vectorOfChannels(blob_.size[1]);
    for (int n = 0; n < blob_.size[0]; ++n)
    {
        for (int c = 0; c < blob_.size[1]; ++c)
        {
            vectorOfChannels[c] = getPlane(blob_, n, c);
        }
        cv::merge(vectorOfChannels, images_.getMatRef(n));
    }
}


CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
