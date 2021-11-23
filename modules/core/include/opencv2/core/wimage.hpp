/*M//////////////////////////////////////////////////////////////////////////////
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to
//  this license.  If you do not agree to this license, do not download,
//  install, copy or use the software.
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2008, Google, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//  * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//  * The name of Intel Corporation or contributors may not be used to endorse
//     or promote products derived from this software without specific
//     prior written permission.
//
// This software is provided by the copyright holders and contributors "as is"
// and any express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular purpose
// are disclaimed. In no event shall the Intel Corporation or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
/////////////////////////////////////////////////////////////////////////////////
//M*/

#ifndef OPENCV_CORE_WIMAGE_HPP
#define OPENCV_CORE_WIMAGE_HPP

#include "opencv2/core/core_c.h"

#ifdef __cplusplus

namespace cv {

//! @addtogroup core
//! @{

template <typename T> class WImage;
template <typename T> class WImageBuffer;
template <typename T> class WImageView;

template<typename T, int C> class WImageC;
template<typename T, int C> class WImageBufferC;
template<typename T, int C> class WImageViewC;

// Commonly used typedefs.
typedef WImage<uchar>            WImage_b;
typedef WImageView<uchar>        WImageView_b;
typedef WImageBuffer<uchar>      WImageBuffer_b;

typedef WImageC<uchar, 1>        WImage1_b;
typedef WImageViewC<uchar, 1>    WImageView1_b;
typedef WImageBufferC<uchar, 1>  WImageBuffer1_b;

typedef WImageC<uchar, 3>        WImage3_b;
typedef WImageViewC<uchar, 3>    WImageView3_b;
typedef WImageBufferC<uchar, 3>  WImageBuffer3_b;

typedef WImage<float>            WImage_f;
typedef WImageView<float>        WImageView_f;
typedef WImageBuffer<float>      WImageBuffer_f;

typedef WImageC<float, 1>        WImage1_f;
typedef WImageViewC<float, 1>    WImageView1_f;
typedef WImageBufferC<float, 1>  WImageBuffer1_f;

typedef WImageC<float, 3>        WImage3_f;
typedef WImageViewC<float, 3>    WImageView3_f;
typedef WImageBufferC<float, 3>  WImageBuffer3_f;

// There isn't a standard for signed and unsigned short so be more
// explicit in the typename for these cases.
typedef WImage<short>            WImage_16s;
typedef WImageView<short>        WImageView_16s;
typedef WImageBuffer<short>      WImageBuffer_16s;

typedef WImageC<short, 1>        WImage1_16s;
typedef WImageViewC<short, 1>    WImageView1_16s;
typedef WImageBufferC<short, 1>  WImageBuffer1_16s;

typedef WImageC<short, 3>        WImage3_16s;
typedef WImageViewC<short, 3>    WImageView3_16s;
typedef WImageBufferC<short, 3>  WImageBuffer3_16s;

typedef WImage<ushort>            WImage_16u;
typedef WImageView<ushort>        WImageView_16u;
typedef WImageBuffer<ushort>      WImageBuffer_16u;

typedef WImageC<ushort, 1>        WImage1_16u;
typedef WImageViewC<ushort, 1>    WImageView1_16u;
typedef WImageBufferC<ushort, 1>  WImageBuffer1_16u;

typedef WImageC<ushort, 3>        WImage3_16u;
typedef WImageViewC<ushort, 3>    WImageView3_16u;
typedef WImageBufferC<ushort, 3>  WImageBuffer3_16u;

/** @brief Image class which provides a thin layer around an IplImage.

The goals of the class design are:

    -# All the data has explicit ownership to avoid memory leaks
    -# No hidden allocations or copies for performance.
    -# Easy access to OpenCV methods (which will access IPP if available)
    -# Can easily treat external data as an image
    -# Easy to create images which are subsets of other images
    -# Fast pixel access which can take advantage of number of channels if known at compile time.

The WImage class is the image class which provides the data accessors. The 'W' comes from the fact
that it is also a wrapper around the popular but inconvenient IplImage class. A WImage can be
constructed either using a WImageBuffer class which allocates and frees the data, or using a
WImageView class which constructs a subimage or a view into external data. The view class does no
memory management. Each class actually has two versions, one when the number of channels is known
at compile time and one when it isn't. Using the one with the number of channels specified can
provide some compile time optimizations by using the fact that the number of channels is a
constant.

We use the convention (c,r) to refer to column c and row r with (0,0) being the upper left corner.
This is similar to standard Euclidean coordinates with the first coordinate varying in the
horizontal direction and the second coordinate varying in the vertical direction. Thus (c,r) is
usually in the domain [0, width) X [0, height)

Example usage:
@code
WImageBuffer3_b  im(5,7);  // Make a 5X7 3 channel image of type uchar
WImageView3_b  sub_im(im, 2,2, 3,3); // 3X3 submatrix
vector<float> vec(10, 3.0f);
WImageView1_f user_im(&vec[0], 2, 5);  // 2X5 image w/ supplied data

im.SetZero();  // same as cvSetZero(im.Ipl())
*im(2, 3) = 15;  // Modify the element at column 2, row 3
MySetRand(&sub_im);

// Copy the second row into the first.  This can be done with no memory
// allocation and will use SSE if IPP is available.
int w = im.Width();
im.View(0,0, w,1).CopyFrom(im.View(0,1, w,1));

// Doesn't care about source of data since using WImage
void MySetRand(WImage_b* im) { // Works with any number of channels
for (int r = 0; r < im->Height(); ++r) {
 float* row = im->Row(r);
 for (int c = 0; c < im->Width(); ++c) {
    for (int ch = 0; ch < im->Channels(); ++ch, ++row) {
      *row = uchar(rand() & 255);
    }
 }
}
}
@endcode

Functions that are not part of the basic image allocation, viewing, and access should come from
OpenCV, except some useful functions that are not part of OpenCV can be found in wimage_util.h
*/
template<typename T>
class WImage
{
public:
    typedef T BaseType;

    // WImage is an abstract class with no other virtual methods so make the
    // destructor virtual.
    virtual ~WImage() = 0;

    // Accessors
    IplImage* Ipl() {return image_; }
    const IplImage* Ipl() const {return image_; }
    T* ImageData() { return reinterpret_cast<T*>(image_->imageData); }
    const T* ImageData() const {
        return reinterpret_cast<const T*>(image_->imageData);
    }

    int Width() const {return image_->width; }
    int Height() const {return image_->height; }

    // WidthStep is the number of bytes to go to the pixel with the next y coord
    int WidthStep() const {return image_->widthStep; }

    int Channels() const {return image_->nChannels; }
    int ChannelSize() const {return sizeof(T); }  // number of bytes per channel

    // Number of bytes per pixel
    int PixelSize() const {return Channels() * ChannelSize(); }

    // Return depth type (e.g. IPL_DEPTH_8U, IPL_DEPTH_32F) which is the number
    // of bits per channel and with the signed bit set.
    // This is known at compile time using specializations.
    int Depth() const;

    inline const T* Row(int r) const {
        return reinterpret_cast<T*>(image_->imageData + r*image_->widthStep);
    }

    inline T* Row(int r) {
        return reinterpret_cast<T*>(image_->imageData + r*image_->widthStep);
    }

    // Pixel accessors which returns a pointer to the start of the channel
    inline T* operator() (int c, int r)  {
        return reinterpret_cast<T*>(image_->imageData + r*image_->widthStep) +
            c*Channels();
    }

    inline const T* operator() (int c, int r) const  {
        return reinterpret_cast<T*>(image_->imageData + r*image_->widthStep) +
            c*Channels();
    }

    // Copy the contents from another image which is just a convenience to cvCopy
    void CopyFrom(const WImage<T>& src) { cvCopy(src.Ipl(), image_); }

    // Set contents to zero which is just a convenient to cvSetZero
    void SetZero() { cvSetZero(image_); }

    // Construct a view into a region of this image
    WImageView<T> View(int c, int r, int width, int height);

protected:
    // Disallow copy and assignment
    WImage(const WImage&);
    void operator=(const WImage&);

    explicit WImage(IplImage* img) : image_(img) {
        CV_Assert(!img || img->depth == Depth());
    }

    void SetIpl(IplImage* image) {
        CV_Assert(!image || image->depth == Depth());
        image_ = image;
    }

    IplImage* image_;
};


/** Image class when both the pixel type and number of channels
are known at compile time.  This wrapper will speed up some of the operations
like accessing individual pixels using the () operator.
*/
template<typename T, int C>
class WImageC : public WImage<T>
{
public:
    typedef typename WImage<T>::BaseType BaseType;
    enum { kChannels = C };

    explicit WImageC(IplImage* img) : WImage<T>(img) {
        CV_Assert(!img || img->nChannels == Channels());
    }

    // Construct a view into a region of this image
    WImageViewC<T, C> View(int c, int r, int width, int height);

    // Copy the contents from another image which is just a convenience to cvCopy
    void CopyFrom(const WImageC<T, C>& src) {
        cvCopy(src.Ipl(), WImage<T>::image_);
    }

    // WImageC is an abstract class with no other virtual methods so make the
    // destructor virtual.
    virtual ~WImageC() = 0;

    int Channels() const {return C; }

protected:
    // Disallow copy and assignment
    WImageC(const WImageC&);
    void operator=(const WImageC&);

    void SetIpl(IplImage* image) {
        CV_Assert(!image || image->depth == WImage<T>::Depth());
        WImage<T>::SetIpl(image);
    }
};

/** Image class which owns the data, so it can be allocated and is always
freed.  It cannot be copied but can be explicitly cloned.
*/
template<typename T>
class WImageBuffer : public WImage<T>
{
public:
    typedef typename WImage<T>::BaseType BaseType;

    // Default constructor which creates an object that can be
    WImageBuffer() : WImage<T>(0) {}

    WImageBuffer(int width, int height, int nchannels) : WImage<T>(0) {
        Allocate(width, height, nchannels);
    }

    // Constructor which takes ownership of a given IplImage so releases
    // the image on destruction.
    explicit WImageBuffer(IplImage* img) : WImage<T>(img) {}

    // Allocate an image.  Does nothing if current size is the same as
    // the new size.
    void Allocate(int width, int height, int nchannels);

    // Set the data to point to an image, releasing the old data
    void SetIpl(IplImage* img) {
        ReleaseImage();
        WImage<T>::SetIpl(img);
    }

    // Clone an image which reallocates the image if of a different dimension.
    void CloneFrom(const WImage<T>& src) {
        Allocate(src.Width(), src.Height(), src.Channels());
        CopyFrom(src);
    }

    ~WImageBuffer() {
        ReleaseImage();
    }

    // Release the image if it isn't null.
    void ReleaseImage() {
        if (WImage<T>::image_) {
            IplImage* image = WImage<T>::image_;
            cvReleaseImage(&image);
            WImage<T>::SetIpl(0);
        }
    }

    bool IsNull() const {return WImage<T>::image_ == NULL; }

private:
    // Disallow copy and assignment
    WImageBuffer(const WImageBuffer&);
    void operator=(const WImageBuffer&);
};

/** Like a WImageBuffer class but when the number of channels is known at compile time.
*/
template<typename T, int C>
class WImageBufferC : public WImageC<T, C>
{
public:
    typedef typename WImage<T>::BaseType BaseType;
    enum { kChannels = C };

    // Default constructor which creates an object that can be
    WImageBufferC() : WImageC<T, C>(0) {}

    WImageBufferC(int width, int height) : WImageC<T, C>(0) {
        Allocate(width, height);
    }

    // Constructor which takes ownership of a given IplImage so releases
    // the image on destruction.
    explicit WImageBufferC(IplImage* img) : WImageC<T, C>(img) {}

    // Allocate an image.  Does nothing if current size is the same as
    // the new size.
    void Allocate(int width, int height);

    // Set the data to point to an image, releasing the old data
    void SetIpl(IplImage* img) {
        ReleaseImage();
        WImageC<T, C>::SetIpl(img);
    }

    // Clone an image which reallocates the image if of a different dimension.
    void CloneFrom(const WImageC<T, C>& src) {
        Allocate(src.Width(), src.Height());
        CopyFrom(src);
    }

    ~WImageBufferC() {
        ReleaseImage();
    }

    // Release the image if it isn't null.
    void ReleaseImage() {
        if (WImage<T>::image_) {
            IplImage* image = WImage<T>::image_;
            cvReleaseImage(&image);
            WImageC<T, C>::SetIpl(0);
        }
    }

    bool IsNull() const {return WImage<T>::image_ == NULL; }

private:
    // Disallow copy and assignment
    WImageBufferC(const WImageBufferC&);
    void operator=(const WImageBufferC&);
};

/** View into an image class which allows treating a subimage as an image or treating external data
as an image
*/
template<typename T> class WImageView : public WImage<T>
{
public:
    typedef typename WImage<T>::BaseType BaseType;

    // Construct a subimage.  No checks are done that the subimage lies
    // completely inside the original image.
    WImageView(WImage<T>* img, int c, int r, int width, int height);

    // Refer to external data.
    // If not given width_step assumed to be same as width.
    WImageView(T* data, int width, int height, int channels, int width_step = -1);

    // Refer to external data.  This does NOT take ownership
    // of the supplied IplImage.
    WImageView(IplImage* img) : WImage<T>(img) {}

    // Copy constructor
    WImageView(const WImage<T>& img) : WImage<T>(0) {
        header_ = *(img.Ipl());
        WImage<T>::SetIpl(&header_);
    }

    WImageView& operator=(const WImage<T>& img) {
        header_ = *(img.Ipl());
        WImage<T>::SetIpl(&header_);
        return *this;
    }

protected:
    IplImage header_;
};


template<typename T, int C>
class WImageViewC : public WImageC<T, C>
{
public:
    typedef typename WImage<T>::BaseType BaseType;
    enum { kChannels = C };

    // Default constructor needed for vectors of views.
    WImageViewC();

    virtual ~WImageViewC() {}

    // Construct a subimage.  No checks are done that the subimage lies
    // completely inside the original image.
    WImageViewC(WImageC<T, C>* img,
        int c, int r, int width, int height);

    // Refer to external data
    WImageViewC(T* data, int width, int height, int width_step = -1);

    // Refer to external data.  This does NOT take ownership
    // of the supplied IplImage.
    WImageViewC(IplImage* img) : WImageC<T, C>(img) {}

    // Copy constructor which does a shallow copy to allow multiple views
    // of same data.  gcc-4.1.1 gets confused if both versions of
    // the constructor and assignment operator are not provided.
    WImageViewC(const WImageC<T, C>& img) : WImageC<T, C>(0) {
        header_ = *(img.Ipl());
        WImageC<T, C>::SetIpl(&header_);
    }
    WImageViewC(const WImageViewC<T, C>& img) : WImageC<T, C>(0) {
        header_ = *(img.Ipl());
        WImageC<T, C>::SetIpl(&header_);
    }

    WImageViewC& operator=(const WImageC<T, C>& img) {
        header_ = *(img.Ipl());
        WImageC<T, C>::SetIpl(&header_);
        return *this;
    }
    WImageViewC& operator=(const WImageViewC<T, C>& img) {
        header_ = *(img.Ipl());
        WImageC<T, C>::SetIpl(&header_);
        return *this;
    }

protected:
    IplImage header_;
};


// Specializations for depth
template<>
inline int WImage<uchar>::Depth() const {return IPL_DEPTH_8U; }
template<>
inline int WImage<signed char>::Depth() const {return IPL_DEPTH_8S; }
template<>
inline int WImage<short>::Depth() const {return IPL_DEPTH_16S; }
template<>
inline int WImage<ushort>::Depth() const {return IPL_DEPTH_16U; }
template<>
inline int WImage<int>::Depth() const {return IPL_DEPTH_32S; }
template<>
inline int WImage<float>::Depth() const {return IPL_DEPTH_32F; }
template<>
inline int WImage<double>::Depth() const {return IPL_DEPTH_64F; }

template<typename T> inline WImage<T>::~WImage() {}
template<typename T, int C> inline WImageC<T, C>::~WImageC() {}

template<typename T>
inline void WImageBuffer<T>::Allocate(int width, int height, int nchannels)
{
    if (IsNull() || WImage<T>::Width() != width ||
        WImage<T>::Height() != height || WImage<T>::Channels() != nchannels) {
        ReleaseImage();
        WImage<T>::image_ = cvCreateImage(cvSize(width, height),
            WImage<T>::Depth(), nchannels);
    }
}

template<typename T, int C>
inline void WImageBufferC<T, C>::Allocate(int width, int height)
{
    if (IsNull() || WImage<T>::Width() != width || WImage<T>::Height() != height) {
        ReleaseImage();
        WImageC<T, C>::SetIpl(cvCreateImage(cvSize(width, height),WImage<T>::Depth(), C));
    }
}

template<typename T>
WImageView<T>::WImageView(WImage<T>* img, int c, int r, int width, int height)
        : WImage<T>(0)
{
    header_ = *(img->Ipl());
    header_.imageData = reinterpret_cast<char*>((*img)(c, r));
    header_.width = width;
    header_.height = height;
    WImage<T>::SetIpl(&header_);
}

template<typename T>
WImageView<T>::WImageView(T* data, int width, int height, int nchannels, int width_step)
          : WImage<T>(0)
{
    cvInitImageHeader(&header_, cvSize(width, height), WImage<T>::Depth(), nchannels);
    header_.imageData = reinterpret_cast<char*>(data);
    if (width_step > 0) {
        header_.widthStep = width_step;
    }
    WImage<T>::SetIpl(&header_);
}

template<typename T, int C>
WImageViewC<T, C>::WImageViewC(WImageC<T, C>* img, int c, int r, int width, int height)
        : WImageC<T, C>(0)
{
    header_ = *(img->Ipl());
    header_.imageData = reinterpret_cast<char*>((*img)(c, r));
    header_.width = width;
    header_.height = height;
    WImageC<T, C>::SetIpl(&header_);
}

template<typename T, int C>
WImageViewC<T, C>::WImageViewC() : WImageC<T, C>(0) {
    cvInitImageHeader(&header_, cvSize(0, 0), WImage<T>::Depth(), C);
    header_.imageData = reinterpret_cast<char*>(0);
    WImageC<T, C>::SetIpl(&header_);
}

template<typename T, int C>
WImageViewC<T, C>::WImageViewC(T* data, int width, int height, int width_step)
    : WImageC<T, C>(0)
{
    cvInitImageHeader(&header_, cvSize(width, height), WImage<T>::Depth(), C);
    header_.imageData = reinterpret_cast<char*>(data);
    if (width_step > 0) {
        header_.widthStep = width_step;
    }
    WImageC<T, C>::SetIpl(&header_);
}

// Construct a view into a region of an image
template<typename T>
WImageView<T> WImage<T>::View(int c, int r, int width, int height) {
    return WImageView<T>(this, c, r, width, height);
}

template<typename T, int C>
WImageViewC<T, C> WImageC<T, C>::View(int c, int r, int width, int height) {
    return WImageViewC<T, C>(this, c, r, width, height);
}

//! @} core

}  // end of namespace

#endif // __cplusplus

#endif
