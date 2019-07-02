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
//  Loading and saving IPL images.
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
#include <opencv2/multiload.hpp>

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


static void exifTransform(int orientation, Mat &img)
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

template <class T> static void setProperty(std::map<String, String> &p, ExifReader &reader, const ExifTagName tag, const String &tag_str, T ExifEntry_t::* field) {
    ExifEntry_t entry = reader.getTag(tag);
    if (entry.tag != INVALID_TAG) {
        p[tag_str] = p[BaseImageDecoder::toString((int)tag)] = BaseImageDecoder::toString(entry.*field);
    }
}
#define setProperty(p, reader, tag, entry) setProperty(p, reader, tag, #tag, entry)

static void applyExif(std::istream &stream, Mat &img, bool rotate, std::map<String, String> *properties)
{
    ExifReader reader(stream);
    if (reader.parse()) {
        if (rotate) {
            ExifEntry_t entry = reader.getTag(ORIENTATION);
            if (entry.tag != INVALID_TAG) {
                exifTransform(entry.field_u16, img); //orientation is unsigned short, so check field_u16
            }
        }

        if (properties) {
            // TIFFTAG-like names
#define DOCUMENTNAME DOCUMENT_NAME
#define IMAGEDESCRIPTION IMAGE_DESCRIPTION
#define DATETIME DATE_TIME
            std::map<String, String> &p = *properties;
            setProperty(p, reader, DOCUMENTNAME, &ExifEntry_t::field_str);
            setProperty(p, reader, IMAGEDESCRIPTION, &ExifEntry_t::field_str);
            setProperty(p, reader, MAKE, &ExifEntry_t::field_str);
            setProperty(p, reader, MODEL, &ExifEntry_t::field_str);
            setProperty(p, reader, ORIENTATION, &ExifEntry_t::field_u16);
            setProperty(p, reader, SOFTWARE, &ExifEntry_t::field_str);
            setProperty(p, reader, DATETIME, &ExifEntry_t::field_str);
            setProperty(p, reader, COPYRIGHT, &ExifEntry_t::field_str);
            std::map<String, String>::const_iterator doc_name = p.find(BaseImageDecoder::toString((int)DOCUMENTNAME));
            if(doc_name != p.end()) {
                p[BaseImageDecoder::document_name] = doc_name->second;
            }
        }
    }
}

static void applyExif(const String &filename, Mat &img, bool rotate, std::map<String, String> *properties)
{
    if (filename.size() > 0) {
        std::ifstream stream(filename.c_str(), std::ios_base::in | std::ios_base::binary);
        applyExif(stream, img, rotate, properties);
    }
}

static void applyExif(const Mat &buf, Mat &img, bool rotate, std::map<String, String> *properties)
{
    if (buf.isContinuous()) {
        ByteStreamBuffer bsb(reinterpret_cast<char *>(buf.data), buf.total() * buf.elemSize());
        std::istream stream(&bsb);
        applyExif(stream, img, rotate, properties);
    }
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
    MultiLoad load(flags);
    return load.read(filename) ? *load : Mat();
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
    MultiLoad load(flags);
    if(!load.read(filename)) return false;
    for(cv::Mat mat: load) {
        mats.push_back(mat);
    }
    return !mats.empty();
}

static bool imwrite_( const String& filename, const std::vector<Mat>& img_vec,
                      const std::vector<int>& iparams, const std::map<int, String>& sparams,
                      bool flipv )
{
    bool isMultiImg = img_vec.size() > 1;
    std::vector<Mat> write_vec;

    ImageEncoder encoder = findEncoder( filename );
    if( !encoder )
        CV_Error( Error::StsError, "could not find a writer for the specified extension" );

    for (size_t page = 0; page < img_vec.size(); page++)
    {
        Mat image = img_vec[page];
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
    CV_Assert(iparams.size() <= CV_IO_MAX_IMAGE_PARAMS*2);
    bool code = false;
    try
    {
        if (!isMultiImg)
            code = encoder->write( write_vec[0], iparams, sparams );
        else
            code = encoder->writemulti( write_vec, iparams, sparams ); //to be implemented
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

template<class T> static std::vector<T> &map2vec(std::vector<T> &dst, const std::map<T, T> &src)
{
    dst.clear();
    dst.reserve(src.size() * 2);
    for(typename std::map<T, T>::const_iterator it = src.begin(); it != src.end(); ++it) {
        dst.push_back(it->first);
        dst.push_back(it->second);
    }
    return dst;
}

static const std::map<int, String> empty_int_String_map;

bool imwrite( const String& filename, InputArray _img, const std::vector<int>& params )
{
    return imwrite(filename, _img, params, empty_int_String_map);
}

bool imwrite( const String& filename, InputArray _img,
                           const std::map<int, int>& iparams, const std::map<int, String> &sparams)
{
    std::vector<int> tmp;
    return imwrite(filename, _img, map2vec(tmp, iparams), sparams);
}

bool imwrite( const String& filename, InputArray _img,
              const std::vector<int>& iparams, const std::map<int, String>& sparams)
{
    CV_TRACE_FUNCTION();
    std::vector<Mat> img_vec;
    if (_img.isMatVector() || _img.isUMatVector())
        _img.getMatVector(img_vec);
    else
        img_vec.push_back(_img.getMat());

    CV_Assert(!img_vec.empty());
    return imwrite_(filename, img_vec, iparams, sparams, false);
}

Mat imdecode( InputArray _buf, int flags )
{
    CV_TRACE_FUNCTION();
    return imdecode(_buf, flags, 0);
}

Mat imdecode( InputArray _buf, int flags, Mat* dst )
{
    CV_TRACE_FUNCTION();
    MultiLoad load;
    return load.decode(_buf, flags) ? load.current(flags, 0, dst) : Mat();
}

bool imencode( const String& ext, InputArray _image,
               std::vector<uchar>& buf, const std::vector<int>& params )
{
    return imencode(ext, _image, buf, params, empty_int_String_map);
}

bool imencode( const String& ext, InputArray _image,
               std::vector<uchar>& buf,
               const std::map<int, int>& iparams, const std::map<int, String> &sparams)
{
    std::vector<int> tmp;
    return imencode(ext, _image, buf, map2vec(tmp, iparams), sparams);
}

bool imencode( const String& ext, InputArray _image,
               std::vector<uchar>& buf,
               const std::vector<int>& iparams, const std::map<int, String>& sparams)
{
    CV_TRACE_FUNCTION();

    Mat image = _image.getMat();

    int channels = image.channels();
    CV_Assert( channels == 1 || channels == 3 || channels == 4 );

    ImageEncoder encoder = findEncoder( ext );
    if( !encoder )
        CV_Error( Error::StsError, "could not find encoder for the specified extension" );

    if(!encoder->supportAppend())
        buf.clear();

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
        code = encoder->write(image, iparams, sparams);
        encoder->throwOnEror();
        CV_Assert( code );
    }
    else
    {
        String filename = tempfile();

        if(!buf.empty()) {
            FILE *f = fopen(filename.c_str(), "wb");
            CV_Assert(f != 0);
            fwrite(buf.data(), 1, buf.size(), f);
            fclose(f);
        }

        code = encoder->setDestination(filename);
        CV_Assert( code );

        code = encoder->write(image, iparams, sparams);
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

MultiLoad::const_iterator MultiLoad::begin() const {
    MultiLoad copy(*this);
    if(!m_file.empty()) copy.read(m_file);
    else copy.decode(m_buf);
    return copy;
}

const MultiLoad::const_iterator& MultiLoad::end() const {
    static MultiLoad end;
    return end;
}

bool MultiLoad::operator==(const MultiLoad &other) const {
    if(this == &other) return true;
    if(!m_decoder && !other.m_decoder) return true;
    if(!m_decoder) return !other.valid();
    if(!other.m_decoder) return !valid();
    return m_buf.data == other.m_buf.data && m_file == other.m_file;
}

MultiLoad::MultiLoad(int default_flags) : m_default_flags(default_flags), m_has_current(false)
{
    CV_TRACE_FUNCTION();
}

MultiLoad::~MultiLoad()
{
    CV_TRACE_FUNCTION();
    clear();
}

bool MultiLoad::read(const String &filename, int flags)
{
    CV_TRACE_FUNCTION();
    return load(&filename, 0, flags);
}

bool MultiLoad::decode(InputArray buf, int flags)
{
    CV_TRACE_FUNCTION();
    return load(0, &buf, flags);
}

bool MultiLoad::valid() const
{
    CV_TRACE_FUNCTION();
    return m_has_current;
}

void MultiLoad::clear()
{
    CV_TRACE_FUNCTION();
    if (m_decoder) {
        m_decoder.release();
    }

    if (m_tempfile) {
        m_tempfile.release();
    }

    m_file.clear();
    m_buf.release();
    m_has_current = false;
}

bool MultiLoad::load(const String *filename, const _InputArray *buf, int flags) {
    CV_TRACE_FUNCTION();
    clear();

    if (buf) {
        m_buf = buf->getMat();
        m_decoder = findDecoder(m_buf);
        if (!m_decoder) return false;

        if (!m_decoder->setSource(m_buf)) {
            m_tempfile = Ptr<TempFile>(new TempFile(tempfile()));
            FILE *f = fopen(m_tempfile->file().c_str(), "wb");
            if (!f) return false;

            size_t bufSize = m_buf.cols * m_buf.rows * m_buf.elemSize();
            if (fwrite(m_buf.ptr(), 1, bufSize, f) != bufSize) {
                fclose(f);
                CV_Error(Error::StsError, "failed to write image data to temporary file " + m_tempfile->file());
            }
            if (fclose(f) != 0) {
                CV_Error(Error::StsError, "failed to write image data to temporary file " + m_tempfile->file());
            }
            filename = &m_tempfile->file();
        }
    }

    if (filename) {
        m_file = *filename;
#ifdef HAVE_GDAL
        if(flags != IMREAD_UNCHANGED && (flags & IMREAD_LOAD_GDAL) == IMREAD_LOAD_GDAL ){
            m_decoder = GdalDecoder().newDecoder();
        }else{
            m_decoder = findDecoder(m_file);
        }
#else
        (void) flags; // suppress "unused" warning
        m_decoder = findDecoder(m_file);
#endif

        if (!m_decoder) return false;
        m_decoder->setSource(m_file);
    }

    m_has_current = true;
    return true;
}

std::size_t MultiLoad::size() const {
    CV_TRACE_FUNCTION();
    return m_decoder ? m_decoder->pageNum() : 0;
}

static Mat released(Mat mat) {
    mat.release();
    return mat;
}

static Mat released(Mat *mat) {
    return mat ? released(*mat) : Mat();
}

Mat MultiLoad::at(int idx, int flags, std::map<String, String> *properties, Mat *dst) const {
    CV_TRACE_FUNCTION();
    if (!m_decoder || !m_decoder->gotoPage(idx)) return released(dst);
    return current(flags, properties, dst);
}

Mat MultiLoad::current(int flags, std::map<String, String> *properties, Mat *dst) const {
    CV_TRACE_FUNCTION();
    if (!m_decoder || !m_has_current) return released(dst);

    Mat dft;
    Mat &mat = dst ? *dst : dft;

    int scale_denom = 1;
    if (flags > IMREAD_LOAD_GDAL) {
        if (flags & IMREAD_REDUCED_GRAYSCALE_2)
            scale_denom = 2;
        else if (flags & IMREAD_REDUCED_GRAYSCALE_4)
            scale_denom = 4;
        else if (flags & IMREAD_REDUCED_GRAYSCALE_8)
            scale_denom = 8;
    }

    /// set the scale_denom in the driver
    m_decoder->setScale(scale_denom);

    try {
        // read the header to make sure it succeeds
        if (!m_decoder->readHeader(properties)) return released(mat);
    } catch (const cv::Exception &e) {
        std::cerr << m_file << ": can't read header: " << e.what() << std::endl << std::flush;
        return released(mat);
    } catch (...) {
        std::cerr << m_file << ": can't read header: unknown exception" << std::endl << std::flush;
        return released(mat);
    }

    // established the required input image size
    Size size = validateInputImageSize(Size(m_decoder->width(), m_decoder->height()));

    // grab the decoded type
    int type = m_decoder->type();
    if ((flags & IMREAD_LOAD_GDAL) != IMREAD_LOAD_GDAL && flags != IMREAD_UNCHANGED) {
        if ((flags & CV_LOAD_IMAGE_ANYDEPTH) == 0)
            type = CV_MAKETYPE(CV_8U, CV_MAT_CN(type));

        if ((flags & CV_LOAD_IMAGE_COLOR) != 0 ||
            ((flags & CV_LOAD_IMAGE_ANYCOLOR) != 0 && CV_MAT_CN(type) > 1))
            type = CV_MAKETYPE(CV_MAT_DEPTH(type), 3);
        else
            type = CV_MAKETYPE(CV_MAT_DEPTH(type), 1);
    }

    mat.create(size.height, size.width, type);

    // read the image data
    bool success = false;
    try {
        if (m_decoder->readData(mat, properties)) success = true;
    } catch (const cv::Exception &e) {
        std::cerr << m_file << ": can't read data: " << e.what() << std::endl << std::flush;
    } catch (...) {
        std::cerr << m_file << ": can't read data: unknown exception" << std::endl << std::flush;
    }

    if (!success) return released(mat);

    // if decoder is JpegDecoder then decoder->setScale always returns 1
    if (m_decoder->setScale(scale_denom) > 1) {
        resize(mat, mat, Size(size.width / scale_denom, size.height / scale_denom), 0, 0, INTER_LINEAR_EXACT);
    }

    // optionally rotate the data if EXIF' orientation flag says so
    bool rotate = (flags & IMREAD_IGNORE_ORIENTATION) == 0 && flags != IMREAD_UNCHANGED;
    if (!m_file.empty()) applyExif(m_file, mat, rotate, properties);
    else applyExif(m_buf, mat, rotate, properties);

    return mat;
}

bool MultiLoad::next() {
    return m_has_current = m_decoder && m_decoder->nextPage();
}

MultiLoad::TempFile::TempFile(const String &file): m_file(file) {
}

MultiLoad::TempFile::~TempFile(){
    if (!m_file.empty()) {
        if (0 != remove(m_file.c_str())) {
            std::cerr << "unable to remove temporary file:" << m_file << std::endl << std::flush;
        }
        m_file.clear();
    }
}

const String& MultiLoad::TempFile::file() const {
    return m_file;
}
}

/* End of file. */
