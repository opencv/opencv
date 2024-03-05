// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>


namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

Image2BlobParams::Image2BlobParams():scalefactor(Scalar::all(1.0)), size(Size()), mean(Scalar()), swapRB(false), ddepth(CV_32F),
                           datalayout(DNN_LAYOUT_NCHW), paddingmode(DNN_PMODE_NULL)
{}

Image2BlobParams::Image2BlobParams(const Scalar& scalefactor_, const Size& size_, const Scalar& mean_, bool swapRB_,
    int ddepth_, DataLayout datalayout_, ImagePaddingMode mode_, Scalar borderValue_):
    scalefactor(scalefactor_), size(size_), mean(mean_), swapRB(swapRB_), ddepth(ddepth_),
    datalayout(datalayout_), paddingmode(mode_), borderValue(borderValue_)
{}

void getVector(InputArrayOfArrays images_, std::vector<Mat>& images) {
    images_.getMatVector(images);
}

void getVector(InputArrayOfArrays images_, std::vector<UMat>& images) {
    images_.getUMatVector(images);
}

void getMat(UMat& blob, InputArray blob_, AccessFlag flag) {
    if(blob_.kind() == _InputArray::UMAT)
        blob = blob_.getUMat();
    else if(blob_.kind() == _InputArray::MAT) {
        blob = blob_.getUMat();
    }
}

void getMat(Mat& blob, InputArray blob_, AccessFlag flag) {
    if(blob_.kind() == _InputArray::UMAT)
        blob = blob_.getMat();
    else if(blob_.kind() == _InputArray::MAT) {
        blob = blob_.getMat();
    }
}

void getChannelFromBlob(Mat& m, InputArray blob, int i, int j, int rows, int cols, int type) {
    m = Mat(rows, cols, type, blob.getMat().ptr(i, j));
}

void getChannelFromBlob(UMat& m, InputArray blob, int i, int j, int rows, int cols, int type) {
    UMat ublob = blob.getUMat();
    int offset = (i * ublob.step.p[0] + j * ublob.step.p[1]) / ublob.elemSize();
    int length = 1;
    for(int i = 0; i < ublob.dims; ++i) {
        length *= ublob.size[i];
    }

    const int newShape[1] { length };
    UMat reshaped = ublob.reshape(1, 1, newShape);
    UMat roi = reshaped(Rect(0, offset, 1, rows * cols));
    m = roi.reshape(CV_MAT_CN(type), rows);
}

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
    if (image.kind() == _InputArray::UMAT) {
        std::vector<UMat> images(1, image.getUMat());
        blobFromImages(images, blob, scalefactor, size, mean, swapRB, crop, ddepth);
    } else {
        std::vector<Mat> images(1, image.getMat());
        blobFromImages(images, blob, scalefactor, size, mean, swapRB, crop, ddepth);
    }
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
    if (images_.kind() != _InputArray::STD_VECTOR_UMAT  && images_.kind() != _InputArray::STD_VECTOR_MAT && images_.kind() != _InputArray::STD_ARRAY_MAT &&
        images_.kind() != _InputArray::STD_VECTOR_VECTOR) {
        String error_message = "The data is expected as vectors of vectors, vectors of Mats or vectors of UMats.";
        CV_Error(Error::StsBadArg, error_message);
    }
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

Mat blobFromImagesWithParams(InputArrayOfArrays images, const Image2BlobParams& param)
{
    CV_TRACE_FUNCTION();
    Mat blob;
    blobFromImagesWithParams(images, blob, param);
    return blob;
}

template<class Tmat>
void blobFromImagesWithParamsImpl(InputArrayOfArrays images_, Tmat& blob_, const Image2BlobParams& param)
{
    CV_TRACE_FUNCTION();
    if(!std::is_same<Tmat, UMat>::value && !std::is_same<Tmat, Mat>::value) {
        String error_message = "The template parameter is expected to be either a cv::Mat or a cv::UMat";
        CV_Error(Error::StsBadArg, error_message);
    }

    CV_CheckType(param.ddepth, param.ddepth == CV_32F || param.ddepth == CV_8U,
                 "Blob depth should be CV_32F or CV_8U");
    Size size = param.size;

    std::vector<Tmat> images;
    getVector(images_, images);

    CV_Assert(!images.empty());

    if (param.ddepth == CV_8U)
    {
        CV_Assert(param.scalefactor == Scalar::all(1.0) && "Scaling is not supported for CV_8U blob depth");
        CV_Assert(param.mean == Scalar() && "Mean subtraction is not supported for CV_8U blob depth");
    }

    int nch = images[0].channels();
    Scalar scalefactor = param.scalefactor;
    Scalar mean = param.mean;

    if (param.swapRB)
    {
        if (nch > 2)
        {
            std::swap(mean[0], mean[2]);
            std::swap(scalefactor[0], scalefactor[2]);
        }
        else
        {
            CV_LOG_WARNING(NULL, "Red/blue color swapping requires at least three image channels.");
        }
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
            else if (param.paddingmode == DNN_PMODE_LETTERBOX)
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
                copyMakeBorder(images[i], images[i], top, bottom, left, right, BORDER_CONSTANT, param.borderValue);
            }
            else
            {
                resize(images[i], images[i], size, 0, 0, INTER_LINEAR);
            }
        }

        if (images[i].depth() == CV_8U && param.ddepth == CV_32F)
            images[i].convertTo(images[i], CV_32F);

        subtract(images[i], mean, images[i]);
        multiply(images[i], scalefactor, images[i]);
    }

    size_t nimages = images.size();
    Tmat image0 = images[0];
    CV_Assert(image0.dims == 2);

    if (param.datalayout == DNN_LAYOUT_NCHW)
    {
        if (nch == 3 || nch == 4)
        {
            int sz[] = { (int)nimages, nch, image0.rows, image0.cols };
            blob_.create(4, sz, param.ddepth);
            std::vector<Tmat> ch(4);

            for (size_t i = 0; i < nimages; i++)
            {
                const Tmat& image = images[i];
                CV_Assert(image.depth() == blob_.depth());
                nch = image.channels();
                CV_Assert(image.dims == 2 && (nch == 3 || nch == 4));
                CV_Assert(image.size() == image0.size());

                for (int j = 0; j < nch; j++) {
                    getChannelFromBlob(ch[j], blob_, i, j ,image.rows, image.cols, param.ddepth);
                }
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
            Mat blob;
            getMat(blob, blob_, ACCESS_RW);

            for (size_t i = 0; i < nimages; i++)
            {
                const Tmat& image = images[i];
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
        Mat blob;
        getMat(blob, blob_, ACCESS_RW);
        int subMatType = CV_MAKETYPE(param.ddepth, nch);
        for (size_t i = 0; i < nimages; i++)
        {
            const Tmat& image = images[i];
            CV_Assert(image.depth() == blob_.depth());
            CV_Assert(image.channels() == image0.channels());
            CV_Assert(image.size() == image0.size());
            if (nch > 2 && param.swapRB)
            {
                Mat tmpRB;
                cvtColor(image, tmpRB, COLOR_BGR2RGB);
                tmpRB.copyTo(Mat(tmpRB.rows, tmpRB.cols, subMatType, blob.ptr((int)i, 0)));
            }
            else
            {
                image.copyTo(Mat(image.rows, image.cols, subMatType, blob.ptr((int)i, 0)));
            }
        }
    }
    else
    {
        CV_Error(Error::StsUnsupportedFormat, "Unsupported data layout in blobFromImagesWithParams function.");
    }
    CV_Assert(blob_.total());
}

void blobFromImagesWithParams(InputArrayOfArrays images, OutputArray blob, const Image2BlobParams& param) {
    CV_TRACE_FUNCTION();

    if (images.kind() == _InputArray::STD_VECTOR_UMAT) {
        if(blob.kind() == _InputArray::UMAT) {
            UMat& u = blob.getUMatRef();
            blobFromImagesWithParamsImpl<cv::UMat>(images, u, param);
            return;
        } else if(blob.kind() == _InputArray::MAT) {
            UMat u = blob.getMatRef().getUMat(ACCESS_WRITE);
            blobFromImagesWithParamsImpl<cv::UMat>(images, u, param);
            u.copyTo(blob);
            return;
        }
    } else if (images.kind() == _InputArray::STD_VECTOR_MAT) {
        if(blob.kind() == _InputArray::UMAT) {
            Mat m = blob.getUMatRef().getMat(ACCESS_WRITE);
            blobFromImagesWithParamsImpl<cv::Mat>(images, m, param);
            m.copyTo(blob);
            return;
        } else if(blob.kind() == _InputArray::MAT) {
            Mat& m = blob.getMatRef();
            blobFromImagesWithParamsImpl<cv::Mat>(images, m, param);
            return;
        }
    }

    CV_Error(Error::StsBadArg, "Images are expected to be a vector of either a Mat or UMat and Blob is expected to be either a Mat or UMat");
}

void blobFromImageWithParams(InputArray image, OutputArray blob, const Image2BlobParams& param)
{
    CV_TRACE_FUNCTION();

    if (image.kind() == _InputArray::UMAT) {
        if(blob.kind() == _InputArray::UMAT) {
            UMat& u = blob.getUMatRef();
            std::vector<UMat> images(1, image.getUMat());
            blobFromImagesWithParamsImpl<cv::UMat>(images, u, param);
            return;
        } else if(blob.kind() == _InputArray::MAT) {
            UMat u = blob.getMatRef().getUMat(ACCESS_RW);
            std::vector<UMat> images(1, image.getUMat());
            blobFromImagesWithParamsImpl<cv::UMat>(images, u, param);
            u.copyTo(blob);
            return;
        }
    } else if (image.kind() == _InputArray::MAT) {
        if(blob.kind() == _InputArray::UMAT) {
            Mat m = blob.getUMatRef().getMat(ACCESS_RW);
            std::vector<Mat> images(1, image.getMat());
            blobFromImagesWithParamsImpl<cv::Mat>(images, m, param);
            m.copyTo(blob);
            return;
        } else if(blob.kind() == _InputArray::MAT) {
            Mat& m = blob.getMatRef();
            std::vector<Mat> images(1, image.getMat());
            blobFromImagesWithParamsImpl<cv::Mat>(images, m, param);
            return;
        }
    }

    CV_Error(Error::StsBadArg, "Image an Blob are expected to be either a Mat or UMat");
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

Rect Image2BlobParams::blobRectToImageRect(const Rect &r, const Size &oriImage)
{
    CV_Assert(!oriImage.empty());
    std::vector<Rect> rImg, rBlob;
    rBlob.push_back(Rect(r));
    rImg.resize(1);
    this->blobRectsToImageRects(rBlob, rImg, oriImage);
    return Rect(rImg[0]);
}

void Image2BlobParams::blobRectsToImageRects(const std::vector<Rect> &rBlob, std::vector<Rect>& rImg, const Size& imgSize)
{
    Size size = this->size;
    rImg.resize(rBlob.size());
    if (size != imgSize)
    {
        if (this->paddingmode == DNN_PMODE_CROP_CENTER)
        {
            float resizeFactor = std::max(size.width / (float)imgSize.width,
                size.height / (float)imgSize.height);
            for (int i = 0; i < rBlob.size(); i++)
            {
                rImg[i] = Rect((rBlob[i].x + 0.5 * (imgSize.width * resizeFactor - size.width)) / resizeFactor,
                               (rBlob[i].y + 0.5 * (imgSize.height * resizeFactor - size.height)) / resizeFactor,
                               rBlob[i].width / resizeFactor,
                               rBlob[i].height / resizeFactor);
            }
        }
        else if (this->paddingmode == DNN_PMODE_LETTERBOX)
        {
            float resizeFactor = std::min(size.width / (float)imgSize.width,
                size.height / (float)imgSize.height);
            int rh = int(imgSize.height * resizeFactor);
            int rw = int(imgSize.width * resizeFactor);

            int top = (size.height - rh) / 2;
            int left = (size.width - rw) / 2;
            for (int i = 0; i < rBlob.size(); i++)
            {
                rImg[i] = Rect((rBlob[i].x - left) / resizeFactor,
                               (rBlob[i].y - top) / resizeFactor,
                               rBlob[i].width / resizeFactor,
                               rBlob[i].height / resizeFactor);
            }
        }
        else if (this->paddingmode == DNN_PMODE_NULL)
        {
            for (int i = 0; i < rBlob.size(); i++)
            {
                rImg[i] = Rect(rBlob[i].x * (float)imgSize.width / size.width,
                               rBlob[i].y * (float)imgSize.height / size.height,
                               rBlob[i].width * (float)imgSize.width / size.width,
                               rBlob[i].height * (float)imgSize.height / size.height);
            }
        }
        else
            CV_Error(cv::Error::StsBadArg, "Unknown padding mode");
    }
}


CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
