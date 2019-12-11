/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

//
//  Loading and saving images.
//

#include "precomp.hpp"
#include "grfmts.hpp"
#include "utils.hpp"
#include "exif.hpp"
#undef min
#undef max
#include <iostream>
#include <fstream>
#include <opencv2/core/utils/configuration.private.hpp>


/****************************************************************************************\
*                                      Image Codecs                                      *
\****************************************************************************************/

namespace cv {

static const size_t CV_IO_MAX_IMAGE_PARAMS = cv::utils::getConfigurationParameterSizeT("OPENCV_IO_MAX_IMAGE_PARAMS", 50);
static const size_t CV_IO_MAX_IMAGE_WIDTH = utils::getConfigurationParameterSizeT("OPENCV_IO_MAX_IMAGE_WIDTH", 1 << 20);
static const size_t CV_IO_MAX_IMAGE_HEIGHT = utils::getConfigurationParameterSizeT("OPENCV_IO_MAX_IMAGE_HEIGHT", 1 << 20);
static const size_t CV_IO_MAX_IMAGE_PIXELS = utils::getConfigurationParameterSizeT("OPENCV_IO_MAX_IMAGE_PIXELS", 1 << 30);

static Size validateInputImageSize(const Size& size)
{
    CV_Assert(size.width > 0);
    CV_Assert(static_cast<size_t>(size.width) <= CV_IO_MAX_IMAGE_WIDTH);
    CV_Assert(size.height > 0);
    CV_Assert(static_cast<size_t>(size.height) <= CV_IO_MAX_IMAGE_HEIGHT);
    uint64 pixels = (uint64)size.width * (uint64)size.height;
    CV_Assert(pixels <= CV_IO_MAX_IMAGE_PIXELS);
    return size;
}


namespace {

class ByteStreamBuffer: public std::streambuf
{
public:
    ByteStreamBuffer(char* base, size_t length)
    {
        setg(base, base, base + length);
    }

protected:
    virtual pos_type seekoff( off_type offset,
                              std::ios_base::seekdir dir,
                              std::ios_base::openmode ) CV_OVERRIDE
    {
        char* whence = eback();
        if (dir == std::ios_base::cur)
        {
            whence = gptr();
        }
        else if (dir == std::ios_base::end)
        {
            whence = egptr();
        }
        char* to = whence + offset;

        // check limits
        if (to >= eback() && to <= egptr())
        {
            setg(eback(), to, egptr());
            return gptr() - eback();
        }

        return -1;
    }
};

}

/**
 * @struct ImageCodecInitializer
 *
 * Container which stores the registered codecs to be used by OpenCV
*/
struct ImageCodecInitializer
{
    /**
     * Default Constructor for the ImageCodeInitializer
    */
    ImageCodecInitializer()
    {
        /// BMP Support
        decoders.push_back( makePtr<BmpDecoder>() );
        encoders.push_back( makePtr<BmpEncoder>() );

    #ifdef HAVE_IMGCODEC_HDR
        decoders.push_back( makePtr<HdrDecoder>() );
        encoders.push_back( makePtr<HdrEncoder>() );
    #endif
    #ifdef HAVE_JPEG
        decoders.push_back( makePtr<JpegDecoder>() );
        encoders.push_back( makePtr<JpegEncoder>() );
    #endif
    #ifdef HAVE_WEBP
        decoders.push_back( makePtr<WebPDecoder>() );
        encoders.push_back( makePtr<WebPEncoder>() );
    #endif
    #ifdef HAVE_IMGCODEC_SUNRASTER
        decoders.push_back( makePtr<SunRasterDecoder>() );
        encoders.push_back( makePtr<SunRasterEncoder>() );
    #endif
    #ifdef HAVE_IMGCODEC_PXM
        decoders.push_back( makePtr<PxMDecoder>() );
        encoders.push_back( makePtr<PxMEncoder>(PXM_TYPE_AUTO) );
        encoders.push_back( makePtr<PxMEncoder>(PXM_TYPE_PBM) );
        encoders.push_back( makePtr<PxMEncoder>(PXM_TYPE_PGM) );
        encoders.push_back( makePtr<PxMEncoder>(PXM_TYPE_PPM) );
        decoders.push_back( makePtr<PAMDecoder>() );
        encoders.push_back( makePtr<PAMEncoder>() );
    #endif
    #ifdef HAVE_IMGCODEC_PFM
        decoders.push_back( makePtr<PFMDecoder>() );
        encoders.push_back( makePtr<PFMEncoder>() );
    #endif
    #ifdef HAVE_TIFF
        decoders.push_back( makePtr<TiffDecoder>() );
        encoders.push_back( makePtr<TiffEncoder>() );
    #endif
    #ifdef HAVE_PNG
        decoders.push_back( makePtr<PngDecoder>() );
        encoders.push_back( makePtr<PngEncoder>() );
    #endif
    #ifdef HAVE_GDCM
        decoders.push_back( makePtr<DICOMDecoder>() );
    #endif
    #ifdef HAVE_JASPER
        decoders.push_back( makePtr<Jpeg2KDecoder>() );
        encoders.push_back( makePtr<Jpeg2KEncoder>() );
    #endif
    #ifdef HAVE_OPENEXR
        decoders.push_back( makePtr<ExrDecoder>() );
        encoders.push_back( makePtr<ExrEncoder>() );
    #endif

    #ifdef HAVE_GDAL
        /// Attach the GDAL Decoder
        decoders.push_back( makePtr<GdalDecoder>() );
    #endif/*HAVE_GDAL*/
    }

    std::vector<ImageDecoder> decoders;
    std::vector<ImageEncoder> encoders;
};

static ImageCodecInitializer codecs;

/**
 * Find the decoders
 *
 * @param[in] filename File to search
 *
 * @return Image decoder to parse image file.
*/
static ImageDecoder findDecoder( const String& filename ) {

    size_t i, maxlen = 0;

    /// iterate through list of registered codecs
    for( i = 0; i < codecs.decoders.size(); i++ )
    {
        size_t len = codecs.decoders[i]->signatureLength();
        maxlen = std::max(maxlen, len);
    }

    /// Open the file
    FILE* f= fopen( filename.c_str(), "rb" );

    /// in the event of a failure, return an empty image decoder
    if( !f )
        return ImageDecoder();

    // read the file signature
    String signature(maxlen, ' ');
    maxlen = fread( (void*)signature.c_str(), 1, maxlen, f );
    fclose(f);
    signature = signature.substr(0, maxlen);

    /// compare signature against all decoders
    for( i = 0; i < codecs.decoders.size(); i++ )
    {
        if( codecs.decoders[i]->checkSignature(signature) )
            return codecs.decoders[i]->newDecoder();
    }

    /// If no decoder was found, return base type
    return ImageDecoder();
}

static ImageDecoder findDecoder( const Mat& buf )
{
    size_t i, maxlen = 0;

    if( buf.rows*buf.cols < 1 || !buf.isContinuous() )
        return ImageDecoder();

    for( i = 0; i < codecs.decoders.size(); i++ )
    {
        size_t len = codecs.decoders[i]->signatureLength();
        maxlen = std::max(maxlen, len);
    }

    String signature(maxlen, ' ');
    size_t bufSize = buf.rows*buf.cols*buf.elemSize();
    maxlen = std::min(maxlen, bufSize);
    memcpy( (void*)signature.c_str(), buf.data, maxlen );

    for( i = 0; i < codecs.decoders.size(); i++ )
    {
        if( codecs.decoders[i]->checkSignature(signature) )
            return codecs.decoders[i]->newDecoder();
    }

    return ImageDecoder();
}

static ImageEncoder findEncoder( const String& _ext )
{
    if( _ext.size() <= 1 )
        return ImageEncoder();

    const char* ext = strrchr( _ext.c_str(), '.' );
    if( !ext )
        return ImageEncoder();
    int len = 0;
    for( ext++; len < 128 && isalnum(ext[len]); len++ )
        ;

    for( size_t i = 0; i < codecs.encoders.size(); i++ )
    {
        String description = codecs.encoders[i]->getDescription();
        const char* descr = strchr( description.c_str(), '(' );

        while( descr )
        {
            descr = strchr( descr + 1, '.' );
            if( !descr )
                break;
            int j = 0;
            for( descr++; j < len && isalnum(descr[j]) ; j++ )
            {
                int c1 = tolower(ext[j]);
                int c2 = tolower(descr[j]);
                if( c1 != c2 )
                    break;
            }
            if( j == len && !isalnum(descr[j]))
                return codecs.encoders[i]->newEncoder();
            descr += j;
        }
    }

    return ImageEncoder();
}


static void ExifTransform(int orientation, Mat& img)
{
    switch( orientation )
    {
        case    IMAGE_ORIENTATION_TL: //0th row == visual top, 0th column == visual left-hand side
            //do nothing, the image already has proper orientation
            break;
        case    IMAGE_ORIENTATION_TR: //0th row == visual top, 0th column == visual right-hand side
            flip(img, img, 1); //flip horizontally
            break;
        case    IMAGE_ORIENTATION_BR: //0th row == visual bottom, 0th column == visual right-hand side
            flip(img, img, -1);//flip both horizontally and vertically
            break;
        case    IMAGE_ORIENTATION_BL: //0th row == visual bottom, 0th column == visual left-hand side
            flip(img, img, 0); //flip vertically
            break;
        case    IMAGE_ORIENTATION_LT: //0th row == visual left-hand side, 0th column == visual top
            transpose(img, img);
            break;
        case    IMAGE_ORIENTATION_RT: //0th row == visual right-hand side, 0th column == visual top
            transpose(img, img);
            flip(img, img, 1); //flip horizontally
            break;
        case    IMAGE_ORIENTATION_RB: //0th row == visual right-hand side, 0th column == visual bottom
            transpose(img, img);
            flip(img, img, -1); //flip both horizontally and vertically
            break;
        case    IMAGE_ORIENTATION_LB: //0th row == visual left-hand side, 0th column == visual bottom
            transpose(img, img);
            flip(img, img, 0); //flip vertically
            break;
        default:
            //by default the image read has normal (JPEG_ORIENTATION_TL) orientation
            break;
    }
}

static void ApplyExifOrientation(const String& filename, Mat& img)
{
    int orientation = IMAGE_ORIENTATION_TL;

    if (filename.size() > 0)
    {
        std::ifstream stream( filename.c_str(), std::ios_base::in | std::ios_base::binary );
        ExifReader reader( stream );
        if( reader.parse() )
        {
            ExifEntry_t entry = reader.getTag( ORIENTATION );
            if (entry.tag != INVALID_TAG)
            {
                orientation = entry.field_u16; //orientation is unsigned short, so check field_u16
            }
        }
        stream.close();
    }

    ExifTransform(orientation, img);
}

static void ApplyExifOrientation(const Mat& buf, Mat& img)
{
    int orientation = IMAGE_ORIENTATION_TL;

    if( buf.isContinuous() )
    {
        ByteStreamBuffer bsb( reinterpret_cast<char*>(buf.data), buf.total() * buf.elemSize() );
        std::istream stream( &bsb );
        ExifReader reader( stream );
        if( reader.parse() )
        {
            ExifEntry_t entry = reader.getTag( ORIENTATION );
            if (entry.tag != INVALID_TAG)
            {
                orientation = entry.field_u16; //orientation is unsigned short, so check field_u16
            }
        }
    }

    ExifTransform(orientation, img);
}

/**
 * Read an image into memory and return the information
 *
 * @param[in] filename File to load
 * @param[in] flags Flags
 * @param[in] hdrtype { LOAD_CVMAT=0,
 *                      LOAD_IMAGE=1,
 *                      LOAD_MAT=2
 *                    }
 * @param[in] mat Reference to C++ Mat object (If LOAD_MAT)
 *
*/
static bool
imread_( const String& filename, int flags, Mat& mat )
{
    /// Search for the relevant decoder to handle the imagery
    ImageDecoder decoder;

#ifdef HAVE_GDAL
    if(flags != IMREAD_UNCHANGED && (flags & IMREAD_LOAD_GDAL) == IMREAD_LOAD_GDAL ){
        decoder = GdalDecoder().newDecoder();
    }else{
#endif
        decoder = findDecoder( filename );
#ifdef HAVE_GDAL
    }
#endif

    /// if no decoder was found, return nothing.
    if( !decoder ){
        return 0;
    }

    int scale_denom = 1;
    if( flags > IMREAD_LOAD_GDAL )
    {
    if( flags & IMREAD_REDUCED_GRAYSCALE_2 )
        scale_denom = 2;
    else if( flags & IMREAD_REDUCED_GRAYSCALE_4 )
        scale_denom = 4;
    else if( flags & IMREAD_REDUCED_GRAYSCALE_8 )
        scale_denom = 8;
    }

    /// set the scale_denom in the driver
    decoder->setScale( scale_denom );

    /// set the filename in the driver
    decoder->setSource( filename );

    try
    {
        // read the header to make sure it succeeds
        if( !decoder->readHeader() )
            return 0;
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "imread_('" << filename << "'): can't read header: " << e.what() << std::endl << std::flush;
        return 0;
    }
    catch (...)
    {
        std::cerr << "imread_('" << filename << "'): can't read header: unknown exception" << std::endl << std::flush;
        return 0;
    }


    // established the required input image size
    Size size = validateInputImageSize(Size(decoder->width(), decoder->height()));

    // grab the decoded type
    int type = decoder->type();
    if( (flags & IMREAD_LOAD_GDAL) != IMREAD_LOAD_GDAL && flags != IMREAD_UNCHANGED )
    {
        if( (flags & IMREAD_ANYDEPTH) == 0 )
            type = CV_MAKETYPE(CV_8U, CV_MAT_CN(type));

        if( (flags & IMREAD_COLOR) != 0 ||
           ((flags & IMREAD_ANYCOLOR) != 0 && CV_MAT_CN(type) > 1) )
            type = CV_MAKETYPE(CV_MAT_DEPTH(type), 3);
        else
            type = CV_MAKETYPE(CV_MAT_DEPTH(type), 1);
    }

    mat.create( size.height, size.width, type );

    // read the image data
    bool success = false;
    try
    {
        if (decoder->readData(mat))
            success = true;
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "imread_('" << filename << "'): can't read data: " << e.what() << std::endl << std::flush;
    }
    catch (...)
    {
        std::cerr << "imread_('" << filename << "'): can't read data: unknown exception" << std::endl << std::flush;
    }
    if (!success)
    {
        mat.release();
        return false;
    }

    if( decoder->setScale( scale_denom ) > 1 ) // if decoder is JpegDecoder then decoder->setScale always returns 1
    {
        resize( mat, mat, Size( size.width / scale_denom, size.height / scale_denom ), 0, 0, INTER_LINEAR_EXACT);
    }

    return true;
}


/**
* Read an image into memory and return the information
*
* @param[in] filename File to load
* @param[in] flags Flags
* @param[in] mats Reference to C++ vector<Mat> object to hold the images
*
*/
static bool
imreadmulti_(const String& filename, int flags, std::vector<Mat>& mats)
{
    /// Search for the relevant decoder to handle the imagery
    ImageDecoder decoder;

#ifdef HAVE_GDAL
    if (flags != IMREAD_UNCHANGED && (flags & IMREAD_LOAD_GDAL) == IMREAD_LOAD_GDAL){
        decoder = GdalDecoder().newDecoder();
    }
    else{
#endif
        decoder = findDecoder(filename);
#ifdef HAVE_GDAL
    }
#endif

    /// if no decoder was found, return nothing.
    if (!decoder){
        return 0;
    }

    /// set the filename in the driver
    decoder->setSource(filename);

    // read the header to make sure it succeeds
    try
    {
        // read the header to make sure it succeeds
        if( !decoder->readHeader() )
            return 0;
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "imreadmulti_('" << filename << "'): can't read header: " << e.what() << std::endl << std::flush;
        return 0;
    }
    catch (...)
    {
        std::cerr << "imreadmulti_('" << filename << "'): can't read header: unknown exception" << std::endl << std::flush;
        return 0;
    }

    for (;;)
    {
        // grab the decoded type
        int type = decoder->type();
        if( (flags & IMREAD_LOAD_GDAL) != IMREAD_LOAD_GDAL && flags != IMREAD_UNCHANGED )
        {
            if ((flags & IMREAD_ANYDEPTH) == 0)
                type = CV_MAKETYPE(CV_8U, CV_MAT_CN(type));

            if ((flags & CV_LOAD_IMAGE_COLOR) != 0 ||
                ((flags & IMREAD_ANYCOLOR) != 0 && CV_MAT_CN(type) > 1))
                type = CV_MAKETYPE(CV_MAT_DEPTH(type), 3);
            else
                type = CV_MAKETYPE(CV_MAT_DEPTH(type), 1);
        }

        // established the required input image size
        Size size = validateInputImageSize(Size(decoder->width(), decoder->height()));

        // read the image data
        Mat mat(size.height, size.width, type);
        bool success = false;
        try
        {
            if (decoder->readData(mat))
                success = true;
        }
        catch (const cv::Exception& e)
        {
            std::cerr << "imreadmulti_('" << filename << "'): can't read data: " << e.what() << std::endl << std::flush;
        }
        catch (...)
        {
            std::cerr << "imreadmulti_('" << filename << "'): can't read data: unknown exception" << std::endl << std::flush;
        }
        if (!success)
            break;

        // optionally rotate the data if EXIF' orientation flag says so
        if( (flags & IMREAD_IGNORE_ORIENTATION) == 0 && flags != IMREAD_UNCHANGED )
        {
            ApplyExifOrientation(filename, mat);
        }

        mats.push_back(mat);
        if (!decoder->nextPage())
        {
            break;
        }
    }

    return !mats.empty();
}

/**
 * Read an image
 *
 *  This function merely calls the actual implementation above and returns itself.
 *
 * @param[in] filename File to load
 * @param[in] flags Flags you wish to set.
*/
Mat imread( const String& filename, int flags )
{
    CV_TRACE_FUNCTION();

    /// create the basic container
    Mat img;

    /// load the data
    imread_( filename, flags, img );

    /// optionally rotate the data if EXIF' orientation flag says so
    if( !img.empty() && (flags & IMREAD_IGNORE_ORIENTATION) == 0 && flags != IMREAD_UNCHANGED )
    {
        ApplyExifOrientation(filename, img);
    }

    /// return a reference to the data
    return img;
}

/**
* Read a multi-page image
*
*  This function merely calls the actual implementation above and returns itself.
*
* @param[in] filename File to load
* @param[in] mats Reference to C++ vector<Mat> object to hold the images
* @param[in] flags Flags you wish to set.
*
*/
bool imreadmulti(const String& filename, std::vector<Mat>& mats, int flags)
{
    CV_TRACE_FUNCTION();

    return imreadmulti_(filename, flags, mats);
}

static bool imwrite_( const String& filename, const std::vector<Mat>& img_vec,
                      const std::vector<int>& params, bool flipv )
{
    bool isMultiImg = img_vec.size() > 1;
    std::vector<Mat> write_vec;

    ImageEncoder encoder = findEncoder( filename );
    if( !encoder )
        CV_Error( Error::StsError, "could not find a writer for the specified extension" );

    for (size_t page = 0; page < img_vec.size(); page++)
    {
        Mat image = img_vec[page];
        CV_Assert(!image.empty());

        CV_Assert( image.channels() == 1 || image.channels() == 3 || image.channels() == 4 );

        Mat temp;
        if( !encoder->isFormatSupported(image.depth()) )
        {
            CV_Assert( encoder->isFormatSupported(CV_8U) );
            image.convertTo( temp, CV_8U );
            image = temp;
        }

        if( flipv )
        {
            flip(image, temp, 0);
            image = temp;
        }

        write_vec.push_back(image);
    }

    encoder->setDestination( filename );
    CV_Assert(params.size() <= CV_IO_MAX_IMAGE_PARAMS*2);
    bool code = false;
    try
    {
        if (!isMultiImg)
            code = encoder->write( write_vec[0], params );
        else
            code = encoder->writemulti( write_vec, params ); //to be implemented
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "imwrite_('" << filename << "'): can't write data: " << e.what() << std::endl << std::flush;
    }
    catch (...)
    {
        std::cerr << "imwrite_('" << filename << "'): can't write data: unknown exception" << std::endl << std::flush;
    }

    //    CV_Assert( code );
    return code;
}

bool imwrite( const String& filename, InputArray _img,
              const std::vector<int>& params )
{
    CV_TRACE_FUNCTION();

    CV_Assert(!_img.empty());

    std::vector<Mat> img_vec;
    if (_img.isMatVector() || _img.isUMatVector())
        _img.getMatVector(img_vec);
    else
        img_vec.push_back(_img.getMat());

    CV_Assert(!img_vec.empty());
    return imwrite_(filename, img_vec, params, false);
}

static bool
imdecode_( const Mat& buf, int flags, Mat& mat )
{
    CV_Assert(!buf.empty());
    CV_Assert(buf.isContinuous());
    CV_Assert(buf.checkVector(1, CV_8U) > 0);
    Mat buf_row = buf.reshape(1, 1);  // decoders expects single row, avoid issues with vector columns

    String filename;

    ImageDecoder decoder = findDecoder(buf_row);
    if( !decoder )
        return 0;

    if( !decoder->setSource(buf_row) )
    {
        filename = tempfile();
        FILE* f = fopen( filename.c_str(), "wb" );
        if( !f )
            return 0;
        size_t bufSize = buf_row.total()*buf.elemSize();
        if (fwrite(buf_row.ptr(), 1, bufSize, f) != bufSize)
        {
            fclose( f );
            CV_Error( Error::StsError, "failed to write image data to temporary file" );
        }
        if( fclose(f) != 0 )
        {
            CV_Error( Error::StsError, "failed to write image data to temporary file" );
        }
        decoder->setSource(filename);
    }

    bool success = false;
    try
    {
        if (decoder->readHeader())
            success = true;
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "imdecode_('" << filename << "'): can't read header: " << e.what() << std::endl << std::flush;
    }
    catch (...)
    {
        std::cerr << "imdecode_('" << filename << "'): can't read header: unknown exception" << std::endl << std::flush;
    }
    if (!success)
    {
        decoder.release();
        if (!filename.empty())
        {
            if (0 != remove(filename.c_str()))
            {
                std::cerr << "unable to remove temporary file:" << filename << std::endl << std::flush;
            }
        }
        return 0;
    }

    // established the required input image size
    Size size = validateInputImageSize(Size(decoder->width(), decoder->height()));

    int type = decoder->type();
    if( (flags & IMREAD_LOAD_GDAL) != IMREAD_LOAD_GDAL && flags != IMREAD_UNCHANGED )
    {
        if( (flags & IMREAD_ANYDEPTH) == 0 )
            type = CV_MAKETYPE(CV_8U, CV_MAT_CN(type));

        if( (flags & IMREAD_COLOR) != 0 ||
           ((flags & IMREAD_ANYCOLOR) != 0 && CV_MAT_CN(type) > 1) )
            type = CV_MAKETYPE(CV_MAT_DEPTH(type), 3);
        else
            type = CV_MAKETYPE(CV_MAT_DEPTH(type), 1);
    }

    mat.create( size.height, size.width, type );

    success = false;
    try
    {
        if (decoder->readData(mat))
            success = true;
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "imdecode_('" << filename << "'): can't read data: " << e.what() << std::endl << std::flush;
    }
    catch (...)
    {
        std::cerr << "imdecode_('" << filename << "'): can't read data: unknown exception" << std::endl << std::flush;
    }
    decoder.release();
    if (!filename.empty())
    {
        if (0 != remove(filename.c_str()))
        {
            std::cerr << "unable to remove temporary file:" << filename << std::endl << std::flush;
        }
    }

    if (!success)
    {
        mat.release();
        return false;
    }

    return true;
}


Mat imdecode( InputArray _buf, int flags )
{
    CV_TRACE_FUNCTION();

    Mat buf = _buf.getMat(), img;
    imdecode_( buf, flags, img );

    /// optionally rotate the data if EXIF' orientation flag says so
    if( !img.empty() && (flags & IMREAD_IGNORE_ORIENTATION) == 0 && flags != IMREAD_UNCHANGED )
    {
        ApplyExifOrientation(buf, img);
    }

    return img;
}

Mat imdecode( InputArray _buf, int flags, Mat* dst )
{
    CV_TRACE_FUNCTION();

    Mat buf = _buf.getMat(), img;
    dst = dst ? dst : &img;
    imdecode_( buf, flags, *dst );

    /// optionally rotate the data if EXIF' orientation flag says so
    if( !dst->empty() && (flags & IMREAD_IGNORE_ORIENTATION) == 0 && flags != IMREAD_UNCHANGED )
    {
        ApplyExifOrientation(buf, *dst);
    }

    return *dst;
}

bool imencode( const String& ext, InputArray _image,
               std::vector<uchar>& buf, const std::vector<int>& params )
{
    CV_TRACE_FUNCTION();

    Mat image = _image.getMat();
    CV_Assert(!image.empty());

    int channels = image.channels();
    CV_Assert( channels == 1 || channels == 3 || channels == 4 );

    ImageEncoder encoder = findEncoder( ext );
    if( !encoder )
        CV_Error( Error::StsError, "could not find encoder for the specified extension" );

    if( !encoder->isFormatSupported(image.depth()) )
    {
        CV_Assert( encoder->isFormatSupported(CV_8U) );
        Mat temp;
        image.convertTo(temp, CV_8U);
        image = temp;
    }

    bool code;
    if( encoder->setDestination(buf) )
    {
        code = encoder->write(image, params);
        encoder->throwOnEror();
        CV_Assert( code );
    }
    else
    {
        String filename = tempfile();
        code = encoder->setDestination(filename);
        CV_Assert( code );

        code = encoder->write(image, params);
        encoder->throwOnEror();
        CV_Assert( code );

        FILE* f = fopen( filename.c_str(), "rb" );
        CV_Assert(f != 0);
        fseek( f, 0, SEEK_END );
        long pos = ftell(f);
        buf.resize((size_t)pos);
        fseek( f, 0, SEEK_SET );
        buf.resize(fread( &buf[0], 1, buf.size(), f ));
        fclose(f);
        remove(filename.c_str());
    }
    return code;
}

bool haveImageReader( const String& filename )
{
    ImageDecoder decoder = cv::findDecoder(filename);
    return !decoder.empty();
}

bool haveImageWriter( const String& filename )
{
    cv::ImageEncoder encoder = cv::findEncoder(filename);
    return !encoder.empty();
}

}

/* End of file. */
