#include "grfmts.hpp"

namespace cv
{

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

        decoders.push_back( makePtr<HdrDecoder>() );
        encoders.push_back( makePtr<HdrEncoder>() );
    #ifdef HAVE_JPEG
        decoders.push_back( makePtr<JpegDecoder>() );
        encoders.push_back( makePtr<JpegEncoder>() );
    #endif
    #ifdef HAVE_WEBP
        decoders.push_back( makePtr<WebPDecoder>() );
        encoders.push_back( makePtr<WebPEncoder>() );
    #endif
        decoders.push_back( makePtr<SunRasterDecoder>() );
        encoders.push_back( makePtr<SunRasterEncoder>() );
        decoders.push_back( makePtr<PxMDecoder>() );
        encoders.push_back( makePtr<PxMEncoder>() );
    #ifdef HAVE_TIFF
        decoders.push_back( makePtr<TiffDecoder>() );
    #endif
        encoders.push_back( makePtr<TiffEncoder>() );
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
        decoders.push_back( makePtr<PAMDecoder>() );
        encoders.push_back( makePtr<PAMEncoder>() );
    }

    std::vector<Ptr<ImageDecoder::Impl> > decoders;
    std::vector<Ptr<ImageEncoder::Impl> > encoders;
};

static ImageCodecInitializer codecs;

ImageDecoder::ImageDecoder()
{
    p = Ptr<ImageDecoder::Impl>();
}

ImageDecoder::ImageDecoder(const ImageDecoder& d)
{
    p = d.p;
}

ImageDecoder::ImageDecoder(const String& filename, Ptr<ImageDecoder::Impl> i)
{
    p = i;
    if( !p->setSource(filename) )
    {
        p = Ptr<ImageDecoder::Impl>();
    }
}

ImageDecoder::ImageDecoder(const Mat& buf, Ptr<ImageDecoder::Impl> i)
{
    p = i;
    if( !p->setSource(buf) )
    {
        p = Ptr<ImageDecoder::Impl>();
    }
}

ImageDecoder::ImageDecoder( const String& filename )
{
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
    {
        p = Ptr<ImageDecoder::Impl>();
        return;
    }

    // read the file signature
    String signature(maxlen, ' ');
    maxlen = fread( (void*)signature.c_str(), 1, maxlen, f );
    fclose(f);
    signature = signature.substr(0, maxlen);

    /// compare signature against all decoders
    for( i = 0; i < codecs.decoders.size(); i++ )
    {
        if( codecs.decoders[i]->checkSignature(signature) )
        {
            p = codecs.decoders[i]->newDecoder();
            if( !p->setSource(filename) )
            {
                p = Ptr<ImageDecoder::Impl>();
            }
            return;
        }
    }

    /// If no decoder was found, return base type
    p = Ptr<ImageDecoder::Impl>();
}

ImageDecoder::ImageDecoder( const Mat& buf )
{
    size_t i, maxlen = 0;

    if( buf.rows*buf.cols < 1 || !buf.isContinuous() )
    {
        p = Ptr<ImageDecoder::Impl>();
        return;
    }

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
        {
            p = codecs.decoders[i]->newDecoder();
            if( !p->setSource(buf) )
            {
                p = Ptr<ImageDecoder::Impl>();
            }
            return;
        }
    }

    p = Ptr<ImageDecoder::Impl>();
}

ImageDecoder::~ImageDecoder()
{
}

ImageDecoder& ImageDecoder::operator = (const ImageDecoder& d)
{
    p = d.p;
    return *this;
}

bool ImageDecoder::empty() const { return p.empty(); }

ImageDecoder::operator bool() const { return !p.empty(); }

bool ImageDecoder::operator !() const { return p.empty(); }

bool ImageDecoder::readHeader() { return p ? p->readHeader() : false; }

bool ImageDecoder::readData( Mat& img ) { return p ? p->readData(img) : false; }

int ImageDecoder::width() const { return p ? p->width() : 0; }

int ImageDecoder::height() const { return p ? p->height() : 0; }

int ImageDecoder::type() const { return p ? p->type() : 0; }

int ImageDecoder::orientation() const { return p ? p->orientation() : IMAGE_ORIENTATION_TL; }

int ImageDecoder::setScale( const int& scale_denom ) { return p ? p->setScale(scale_denom) : 0; }

bool ImageDecoder::nextPage() { return p ? p->nextPage() : false; }

String ImageDecoder::getDescription() const { return p ? p->getDescription() : ""; }

ImageEncoder::ImageEncoder()
{
    p = Ptr<ImageEncoder::Impl>();
}

ImageEncoder::ImageEncoder(const ImageEncoder& e)
{
    p = e.p;
}

ImageEncoder::ImageEncoder(const String& filename, Ptr<ImageEncoder::Impl> i)
{
    p = i;
    if( !p->setDestination(filename) )
    {
        p = Ptr<ImageEncoder::Impl>();
    }
}

ImageEncoder::ImageEncoder(Mat& buf, Ptr<ImageEncoder::Impl> i)
{
    p = i;
    if( !p->setDestination(buf) )
    {
        p = Ptr<ImageEncoder::Impl>();
    }
}

static Ptr<ImageEncoder::Impl> findEncoder( const String& _ext )
{
    if( _ext.size() <= 1 )
        return Ptr<ImageEncoder::Impl>();

    const char* ext = strrchr( _ext.c_str(), '.' );
    if( !ext )
    {
        return Ptr<ImageEncoder::Impl>();
    }
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

    return Ptr<ImageEncoder::Impl>();
}

ImageEncoder::ImageEncoder( const String& _ext, const String& filename )
{
    p = findEncoder(_ext);
    if( p )
    {
        if( !p->setDestination(filename) )
        {
            p = Ptr<ImageEncoder::Impl>();
        }
    }
}

ImageEncoder::ImageEncoder( const String& _ext, Mat& buf )
{
    p = findEncoder(_ext);
    if( p )
    {
        if( !p->setDestination(buf) )
        {
            p = Ptr<ImageEncoder::Impl>();
        }
    }
}

ImageEncoder::~ImageEncoder()
{
}

ImageEncoder& ImageEncoder::operator = (const ImageEncoder& e)
{
    p = e.p;
    return *this;
}

bool ImageEncoder::empty() const { return p.empty(); }

ImageEncoder::operator bool() const { return !p.empty(); }

bool ImageEncoder::operator !() const { return p.empty(); }

bool ImageEncoder::isFormatSupported( int depth ) const { return p ? p->isFormatSupported(depth) : false; }

bool ImageEncoder::write( const Mat& img, InputArray params ) { return p ? p->write(img, params) : false; }

String ImageEncoder::getDescription() const { return p ? p->getDescription() : ""; }

void ImageEncoder::throwOnEror() const { if( p ) p->throwOnEror(); }

}
