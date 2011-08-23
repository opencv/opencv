/*
 *  grfmt_imageio.cpp
 *
 *
 *  Created by Morgan Conbere on 5/17/07.
 *
 */

#include "precomp.hpp"

#ifdef HAVE_IMAGEIO

#include "grfmt_imageio.hpp"

namespace cv
{

/////////////////////// ImageIODecoder ///////////////////

ImageIODecoder::ImageIODecoder()
{
    imageRef = NULL;
}

ImageIODecoder::~ImageIODecoder()
{
    close();
}


void  ImageIODecoder::close()
{
    CGImageRelease( imageRef );
    imageRef = NULL;
}


size_t ImageIODecoder::signatureLength() const
{
    return 12;
}

bool ImageIODecoder::checkSignature( const string& signature ) const
{
    // TODO: implement real signature check
    return true;
}
    
ImageDecoder ImageIODecoder::newDecoder() const
{
    return new ImageIODecoder;
}

bool ImageIODecoder::readHeader()
{
    CFURLRef         imageURLRef;
    CGImageSourceRef sourceRef;
    // diciu, if ReadHeader is called twice in a row make sure to release the previously allocated imageRef
    if (imageRef != NULL)
        CGImageRelease(imageRef);
    imageRef = NULL;

    imageURLRef = CFURLCreateFromFileSystemRepresentation( NULL,
        (const UInt8*)m_filename.c_str(), m_filename.size(), false );

    sourceRef = CGImageSourceCreateWithURL( imageURLRef, NULL );
    CFRelease( imageURLRef );
    if ( !sourceRef )
        return false;

    imageRef = CGImageSourceCreateImageAtIndex( sourceRef, 0, NULL );
    CFRelease( sourceRef );
    if( !imageRef )
        return false;

    m_width = CGImageGetWidth( imageRef );
    m_height = CGImageGetHeight( imageRef );

    CGColorSpaceRef colorSpace = CGImageGetColorSpace( imageRef );
    if( !colorSpace )
        return false;

    m_type = CGColorSpaceGetNumberOfComponents( colorSpace ) > 1 ? CV_8UC3 : CV_8UC1;

    return true;
}


bool  ImageIODecoder::readData( Mat& img )
{
    uchar* data = img.data;
    int step = img.step;
    bool color = img.channels() > 1;
    int bpp; // Bytes per pixel
    int bit_depth = 8;

    // Get Height, Width, and color information
    if( !readHeader() )
        return false;

    CGContextRef     context = NULL; // The bitmap context
    CGColorSpaceRef  colorSpace = NULL;
    uchar*           bitmap = NULL;
    CGImageAlphaInfo alphaInfo;

    // CoreGraphics will take care of converting to grayscale and back as long as the
    // appropriate colorspace is set
    if( color == CV_LOAD_IMAGE_GRAYSCALE )
    {
        colorSpace = CGColorSpaceCreateDeviceGray();
        bpp = 1;
        alphaInfo = kCGImageAlphaNone;
    }
    else if( color == CV_LOAD_IMAGE_COLOR )
    {
#if TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR
        colorSpace = CGColorSpaceCreateDeviceRGB();
#else
        colorSpace = CGColorSpaceCreateWithName( kCGColorSpaceGenericRGBLinear );
#endif
        bpp = 4; /* CG only has 8 and 32 bit color spaces, so we waste a byte */
        alphaInfo = kCGImageAlphaNoneSkipLast;
    }
    if( !colorSpace )
        return false;

    bitmap = (uchar*)malloc( bpp * m_height * m_width );
    if( !bitmap )
    {
        CGColorSpaceRelease( colorSpace );
        return false;
    }

    context = CGBitmapContextCreate( (void *)bitmap,
                                     m_width,        /* width */
                                     m_height,       /* height */
                                     bit_depth,    /* bit depth */
                                     bpp * m_width,  /* bytes per row */
                                     colorSpace,     /* color space */
                                     alphaInfo);

    CGColorSpaceRelease( colorSpace );
    if( !context )
    {
        free( bitmap );
        return false;
    }

    // Copy the image data into the bitmap region
    CGRect rect = {{0,0},{m_width,m_height}};
    CGContextDrawImage( context, rect, imageRef );

    uchar* bitdata = (uchar*)CGBitmapContextGetData( context );
    if( !bitdata )
    {
        free( bitmap);
        CGContextRelease( context );
        return false;
    }

    // Move the bitmap (in RGB) into data (in BGR)
    int bitmapIndex = 0;

    if( color == CV_LOAD_IMAGE_COLOR )
	{
		uchar * base = data;

		for (int y = 0; y < m_height; y++)
		{
			uchar * line = base + y * step;

		    for (int x = 0; x < m_width; x++)
		    {
				// Blue channel
				line[0] = bitdata[bitmapIndex + 2];
				// Green channel
				line[1] = bitdata[bitmapIndex + 1];
				// Red channel
				line[2] = bitdata[bitmapIndex + 0];

				line        += 3;
				bitmapIndex += bpp;
			}
		}
    }
    else if( color == CV_LOAD_IMAGE_GRAYSCALE )
    {
		for (int y = 0; y < m_height; y++)
			memcpy (data + y * step, bitmap + y * m_width, m_width);
    }

    free( bitmap );
    CGContextRelease( context );
    return true;
}


/////////////////////// ImageIOEncoder ///////////////////

ImageIOEncoder::ImageIOEncoder()
{
    m_description = "Apple ImageIO (*.bmp;*.dib;*.exr;*.jpeg;*.jpg;*.jpe;*.jp2;*.pdf;*.png;*.tiff;*.tif)";
}


ImageIOEncoder::~ImageIOEncoder()
{
}


ImageEncoder ImageIOEncoder::newEncoder() const
{
    return new ImageIOEncoder;
}
    
static
CFStringRef  FilenameToUTI( const char* filename )
{
    const char* ext = filename;
    char* ext_buf;
    int i;
    CFStringRef imageUTI = NULL;

    for(;;)
    {
        const char* temp = strchr( ext + 1, '.' );
        if( !temp ) break;
        ext = temp;
    }

    if(!ext)
        return NULL;

    ext_buf = (char*)malloc(strlen(ext)+1);
    for(i = 0; ext[i] != '\0'; i++)
        ext_buf[i] = (char)tolower(ext[i]);
    ext_buf[i] = '\0';
    ext = ext_buf;

    if( !strcmp(ext, ".bmp") || !strcmp(ext, ".dib") )
        imageUTI = CFSTR( "com.microsoft.bmp" );
    else if( !strcmp(ext, ".exr") )
        imageUTI = CFSTR( "com.ilm.openexr-image" );
    else if( !strcmp(ext, ".jpeg") || !strcmp(ext, ".jpg") || !strcmp(ext, ".jpe") )
        imageUTI = CFSTR( "public.jpeg" );
    else if( !strcmp(ext, ".jp2") )
        imageUTI = CFSTR( "public.jpeg-2000" );
    else if( !strcmp(ext, ".pdf") )
        imageUTI = CFSTR( "com.adobe.pdf" );
    else if( !strcmp(ext, ".png") )
        imageUTI = CFSTR( "public.png" );
    else if( !strcmp(ext, ".tiff") || !strcmp(ext, ".tif") )
        imageUTI = CFSTR( "public.tiff" );

    free(ext_buf);

    return imageUTI;
}


bool  ImageIOEncoder::write( const Mat& img, const vector<int>& params )
{
    int width = img.cols, height = img.rows;
    int _channels = img.channels();
    const uchar* data = img.data;
    int step = img.step;
    
    // Determine the appropriate UTI based on the filename extension
    CFStringRef imageUTI = FilenameToUTI( m_filename.c_str() );

    // Determine the Bytes Per Pixel
    int bpp = (_channels == 1) ? 1 : 4;

    // Write the data into a bitmap context
    CGContextRef context;
    CGColorSpaceRef colorSpace;
    uchar* bitmapData = NULL;

    if( bpp == 1 ) {
#if TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR
        colorSpace = CGColorSpaceCreateDeviceGray();
#else
        colorSpace = CGColorSpaceCreateWithName( kCGColorSpaceGenericGray );
#endif
    }
    else if( bpp == 4 ) {
#if TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR
        colorSpace = CGColorSpaceCreateDeviceRGB();
#else
        colorSpace = CGColorSpaceCreateWithName( kCGColorSpaceGenericRGBLinear );
#endif
    }
    if( !colorSpace )
        return false;

    bitmapData = (uchar*)malloc( bpp * height * width );
    if( !bitmapData )
    {
        CGColorSpaceRelease( colorSpace );
        return false;
    }

    context = CGBitmapContextCreate( bitmapData,
                                     width,
                                     height,
                                     8,
                                     bpp * width,
                                     colorSpace,
                                     (bpp == 1) ? kCGImageAlphaNone :
                                     kCGImageAlphaNoneSkipLast );
    CGColorSpaceRelease( colorSpace );
    if( !context )
    {
        free( bitmapData );
        return false;
    }

    // Copy pixel information from data into bitmapData
    if (bpp == 4)
    {
        int           bitmapIndex = 0;
		const uchar * base        = data;

		for (int y = 0; y < height; y++)
		{
			const uchar * line = base + y * step;

		    for (int x = 0; x < width; x++)
		    {
				// Blue channel
                bitmapData[bitmapIndex + 2] = line[0];
				// Green channel
				bitmapData[bitmapIndex + 1] = line[1];
				// Red channel
				bitmapData[bitmapIndex + 0] = line[2];

				line        += 3;
				bitmapIndex += bpp;
			}
		}
    }
    else if (bpp == 1)
    {
		for (int y = 0; y < height; y++)
			memcpy (bitmapData + y * width, data + y * step, width);
    }

    // Turn the bitmap context into an imageRef
    CGImageRef imageRef = CGBitmapContextCreateImage( context );
    CGContextRelease( context );
    if( !imageRef )
    {
        free( bitmapData );
        return false;
    }

    // Write the imageRef to a file based on the UTI
    CFURLRef imageURLRef = CFURLCreateFromFileSystemRepresentation( NULL,
        (const UInt8*)m_filename.c_str(), m_filename.size(), false );
    if( !imageURLRef )
    {
        CGImageRelease( imageRef );
        free( bitmapData );
        return false;
    }

    CGImageDestinationRef destRef = CGImageDestinationCreateWithURL( imageURLRef,
                                                                     imageUTI,
                                                                     1,
                                                                     NULL);
    CFRelease( imageURLRef );
    if( !destRef )
    {
        CGImageRelease( imageRef );
        free( bitmapData );
        fprintf(stderr, "!destRef\n");
        return false;
    }

    CGImageDestinationAddImage(destRef, imageRef, NULL);
    if( !CGImageDestinationFinalize(destRef) )
    {
        fprintf(stderr, "Finalize failed\n");
        return false;
    }

    CFRelease( destRef );
    CGImageRelease( imageRef );
    free( bitmapData );

    return true;
}

}

#endif /* HAVE_IMAGEIO */
