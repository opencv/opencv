/*
 *  grfmt_imageio.h
 *  
 *
 *  Created by Morgan Conbere on 5/17/07.
 *
 */

#ifndef _GRFMT_IMAGEIO_H_
#define _GRFMT_IMAGEIO_H_

#ifdef HAVE_IMAGEIO

#include "grfmt_base.hpp"
#include <TargetConditionals.h>

#if TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR

#include <MobileCoreServices/MobileCoreServices.h> 
#include <ImageIO/ImageIO.h>

#else

#include <ApplicationServices/ApplicationServices.h>

#endif

namespace cv
{

class ImageIODecoder : public BaseImageDecoder
{
public:
    
    ImageIODecoder();
    ~ImageIODecoder();
    
    bool  readData( Mat& img );
    bool  readHeader();
    void  close();
    
    size_t signatureLength() const;
    bool checkSignature( const string& signature ) const;

    ImageDecoder newDecoder() const;

protected:
    
    CGImageRef imageRef;
};

class ImageIOEncoder : public BaseImageEncoder
{
public:
    ImageIOEncoder();
    ~ImageIOEncoder();

    bool  write( const Mat& img, const vector<int>& params );

    ImageEncoder newEncoder() const;
};

}

#endif/*HAVE_IMAGEIO*/

#endif/*_GRFMT_IMAGEIO_H_*/
