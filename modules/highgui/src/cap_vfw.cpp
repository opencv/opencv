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

#include "precomp.hpp"

#include <vfw.h>

#if _MSC_VER >= 1200
#pragma warning( disable: 4711 )
#endif

#ifdef __GNUC__
#define WM_CAP_FIRSTA              (WM_USER)
#define capSendMessage(hwnd,m,w,l) (IsWindow(hwnd)?SendMessage(hwnd,m,w,l):0)
#endif

#if defined _M_X64
#pragma optimize("",off)
#pragma warning(disable: 4748)
#endif

/********************* Capturing video from AVI via VFW ************************/

static BITMAPINFOHEADER icvBitmapHeader( int width, int height, int bpp, int compression = BI_RGB )
{
    BITMAPINFOHEADER bmih;
    memset( &bmih, 0, sizeof(bmih));
    bmih.biSize = sizeof(bmih);
    bmih.biWidth = width;
    bmih.biHeight = height;
    bmih.biBitCount = (WORD)bpp;
    bmih.biCompression = compression;
    bmih.biPlanes = 1;

    return bmih;
}


static void icvInitCapture_VFW()
{
    static int isInitialized = 0;
    if( !isInitialized )
    {
        AVIFileInit();
        isInitialized = 1;
    }
}


class CvCaptureAVI_VFW : public CvCapture
{
public:
    CvCaptureAVI_VFW()
    {
      CoInitialize(NULL);
      init();
    }

    virtual ~CvCaptureAVI_VFW()
    {
        close();
        CoUninitialize();
    }

    virtual bool open( const char* filename );
    virtual void close();

    virtual double getProperty(int);
    virtual bool setProperty(int, double);
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int);
	virtual int getCaptureDomain() { return CV_CAP_VFW; } // Return the type of the capture object: CV_CAP_VFW, etc...

protected:
    void init();

    PAVIFILE            avifile;
    PAVISTREAM          avistream;
    PGETFRAME           getframe;
    AVISTREAMINFO       aviinfo;
    BITMAPINFOHEADER  * bmih;
    CvSlice             film_range;
    double              fps;
    int                 pos;
    IplImage*           frame;
    CvSize              size;
};


void CvCaptureAVI_VFW::init()
{
    avifile = 0;
    avistream = 0;
    getframe = 0;
    memset( &aviinfo, 0, sizeof(aviinfo) );
    bmih = 0;
    film_range = cvSlice(0,0);
    fps = 0;
    pos = 0;
    frame = 0;
    size = cvSize(0,0);
}


void CvCaptureAVI_VFW::close()
{
    if( getframe )
        AVIStreamGetFrameClose( getframe );

    if( avistream )
        AVIStreamRelease( avistream );

    if( avifile )
        AVIFileRelease( avifile );

    if (frame)
        cvReleaseImage( &frame );

    init();
}


bool CvCaptureAVI_VFW::open( const char* filename )
{
    close();
    icvInitCapture_VFW();

    if( !filename )
        return false;

    HRESULT hr = AVIFileOpen( &avifile, filename, OF_READ, NULL );
    if( SUCCEEDED(hr))
    {
        hr = AVIFileGetStream( avifile, &avistream, streamtypeVIDEO, 0 );
        if( SUCCEEDED(hr))
        {
            hr = AVIStreamInfo( avistream, &aviinfo, sizeof(aviinfo));
            if( SUCCEEDED(hr))
            {
                size.width = aviinfo.rcFrame.right - aviinfo.rcFrame.left;
                size.height = aviinfo.rcFrame.bottom - aviinfo.rcFrame.top;
                BITMAPINFOHEADER bmih = icvBitmapHeader( size.width, size.height, 24 );

                film_range.start_index = (int)aviinfo.dwStart;
                film_range.end_index = film_range.start_index + (int)aviinfo.dwLength;
                fps = (double)aviinfo.dwRate/aviinfo.dwScale;
                pos = film_range.start_index;
                getframe = AVIStreamGetFrameOpen( avistream, &bmih );
                if( getframe != 0 )
                    return true;
            }
        }
    }

    close();
    return false;
}

bool CvCaptureAVI_VFW::grabFrame()
{
    if( avistream )
        bmih = (BITMAPINFOHEADER*)AVIStreamGetFrame( getframe, pos++ );
    return bmih != 0;
}

IplImage* CvCaptureAVI_VFW::retrieveFrame(int)
{
    if( avistream && bmih )
    {
        IplImage src;
        cvInitImageHeader( &src, cvSize( bmih->biWidth, bmih->biHeight ),
                           IPL_DEPTH_8U, 3, IPL_ORIGIN_BL, 4 );
        cvSetData( &src, (char*)(bmih + 1), src.widthStep );
        if( !frame || frame->width != src.width || frame->height != src.height )
        {
            cvReleaseImage( &frame );
            frame = cvCreateImage( cvGetSize(&src), 8, 3 );
        }

        cvFlip( &src, frame, 0 );
        return frame;
    }

    return 0;
}

double CvCaptureAVI_VFW::getProperty( int property_id )
{
    switch( property_id )
    {
    case CV_CAP_PROP_POS_MSEC:
        return cvRound(pos*1000./fps);
    case CV_CAP_PROP_POS_FRAMES:
        return pos;
    case CV_CAP_PROP_POS_AVI_RATIO:
        return (pos - film_range.start_index)/
               (film_range.end_index - film_range.start_index + 1e-10);
    case CV_CAP_PROP_FRAME_WIDTH:
        return size.width;
    case CV_CAP_PROP_FRAME_HEIGHT:
        return size.height;
    case CV_CAP_PROP_FPS:
        return fps;
    case CV_CAP_PROP_FOURCC:
        return aviinfo.fccHandler;
    case CV_CAP_PROP_FRAME_COUNT:
        return film_range.end_index - film_range.start_index;
    }
    return 0;
}

bool CvCaptureAVI_VFW::setProperty( int property_id, double value )
{
    switch( property_id )
    {
    case CV_CAP_PROP_POS_MSEC:
    case CV_CAP_PROP_POS_FRAMES:
    case CV_CAP_PROP_POS_AVI_RATIO:
        {
            switch( property_id )
            {
            case CV_CAP_PROP_POS_MSEC:
                pos = cvRound(value*fps*0.001);
                break;
            case CV_CAP_PROP_POS_AVI_RATIO:
                pos = cvRound(value*(film_range.end_index -
                                     film_range.start_index) +
                              film_range.start_index);
                break;
            default:
                pos = cvRound(value);
            }
            if( pos < film_range.start_index )
                pos = film_range.start_index;
            if( pos > film_range.end_index )
                pos = film_range.end_index;
        }
        break;
    default:
        return false;
    }

    return true;
}

CvCapture* cvCreateFileCapture_VFW (const char* filename)
{
    CvCaptureAVI_VFW* capture = new CvCaptureAVI_VFW;
    if( capture->open(filename) )
        return capture;
    delete capture;
    return 0;
}


/********************* Capturing video from camera via VFW *********************/

class CvCaptureCAM_VFW : public CvCapture
{
public:
    CvCaptureCAM_VFW() { init(); }
    virtual ~CvCaptureCAM_VFW() { close(); }

    virtual bool open( int index );
    virtual void close();
    virtual double getProperty(int);
    virtual bool setProperty(int, double) { return false; }
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int);
	virtual int getCaptureDomain() { return CV_CAP_VFW; } // Return the type of the capture object: CV_CAP_VFW, etc...

protected:
    void init();
    void closeHIC();
    static LRESULT PASCAL frameCallback( HWND hWnd, VIDEOHDR* hdr );

    CAPDRIVERCAPS caps;
    HWND   capWnd;
    VIDEOHDR* hdr;
    DWORD  fourcc;
    HIC    hic;
    IplImage* frame;
};


void CvCaptureCAM_VFW::init()
{
    memset( &caps, 0, sizeof(caps) );
    capWnd = 0;
    hdr = 0;
    fourcc = 0;
    hic = 0;
    frame = 0;
}

void CvCaptureCAM_VFW::closeHIC()
{
    if( hic )
    {
        ICDecompressEnd( hic );
        ICClose( hic );
        hic = 0;
    }
}


LRESULT PASCAL CvCaptureCAM_VFW::frameCallback( HWND hWnd, VIDEOHDR* hdr )
{
    CvCaptureCAM_VFW* capture = 0;

    if (!hWnd) return FALSE;

    capture = (CvCaptureCAM_VFW*)capGetUserData(hWnd);
    capture->hdr = hdr;

    return (LRESULT)TRUE;
}


// Initialize camera input
bool CvCaptureCAM_VFW::open( int wIndex )
{
    char szDeviceName[80];
    char szDeviceVersion[80];
    HWND hWndC = 0;

    close();

    if( (unsigned)wIndex >= 10 )
        wIndex = 0;

    for( ; wIndex < 10; wIndex++ )
    {
        if( capGetDriverDescription( wIndex, szDeviceName,
            sizeof (szDeviceName), szDeviceVersion,
            sizeof (szDeviceVersion)))
        {
            hWndC = capCreateCaptureWindow ( "My Own Capture Window",
                WS_POPUP | WS_CHILD, 0, 0, 320, 240, 0, 0);
            if( capDriverConnect (hWndC, wIndex))
                break;
            DestroyWindow( hWndC );
            hWndC = 0;
        }
    }

    if( hWndC )
    {
        capWnd = hWndC;
        hdr = 0;
        hic = 0;
        fourcc = (DWORD)-1;

        memset( &caps, 0, sizeof(caps));
        capDriverGetCaps( hWndC, &caps, sizeof(&caps));
        ::MoveWindow( hWndC, 0, 0, 320, 240, TRUE );
        capSetUserData( hWndC, (size_t)this );
        capSetCallbackOnFrame( hWndC, frameCallback );
        CAPTUREPARMS p;
        capCaptureGetSetup(hWndC,&p,sizeof(CAPTUREPARMS));
        p.dwRequestMicroSecPerFrame = 66667/2;
        capCaptureSetSetup(hWndC,&p,sizeof(CAPTUREPARMS));
        //capPreview( hWndC, 1 );
        capPreviewScale(hWndC,FALSE);
        capPreviewRate(hWndC,1);
    }
    return capWnd != 0;
}


void CvCaptureCAM_VFW::close()
{
    if( capWnd )
    {
        capSetCallbackOnFrame( capWnd, NULL );
        capDriverDisconnect( capWnd );
        DestroyWindow( capWnd );
        closeHIC();
    }
    cvReleaseImage( &frame );
    init();
}


bool CvCaptureCAM_VFW::grabFrame()
{
    if( capWnd )
    {
        SendMessage( capWnd, WM_CAP_GRAB_FRAME_NOSTOP, 0, 0 );
        return true;
    }
    return false;
}


IplImage* CvCaptureCAM_VFW::retrieveFrame(int)
{
    BITMAPINFO vfmt;
    memset( &vfmt, 0, sizeof(vfmt));
    BITMAPINFOHEADER& vfmt0 = vfmt.bmiHeader;
    int sz, prevWidth, prevHeight;

    if( !capWnd )
        return 0;

    sz = capGetVideoFormat( capWnd, &vfmt, sizeof(vfmt));
    prevWidth = frame ? frame->width : 0;
    prevHeight = frame ? frame->height : 0;

    if( !hdr || hdr->lpData == 0 || sz == 0 )
        return 0;

    if( !frame || frame->width != vfmt0.biWidth || frame->height != vfmt0.biHeight )
    {
        cvReleaseImage( &frame );
        frame = cvCreateImage( cvSize( vfmt0.biWidth, vfmt0.biHeight ), 8, 3 );
    }

    if( vfmt.bmiHeader.biCompression != BI_RGB ||
        vfmt.bmiHeader.biBitCount != 24 )
    {
        BITMAPINFOHEADER vfmt1 = icvBitmapHeader( vfmt0.biWidth, vfmt0.biHeight, 24 );

        if( hic == 0 || fourcc != vfmt0.biCompression ||
            prevWidth != vfmt0.biWidth || prevHeight != vfmt0.biHeight )
        {
            closeHIC();
            hic = ICOpen( MAKEFOURCC('V','I','D','C'),
                          vfmt0.biCompression, ICMODE_DECOMPRESS );
            if( hic )
            {
                if( ICDecompressBegin( hic, &vfmt0, &vfmt1 ) != ICERR_OK )
                {
                    closeHIC();
                    return 0;
                }
            }
        }

        if( !hic || ICDecompress( hic, 0, &vfmt0, hdr->lpData,
            &vfmt1, frame->imageData ) != ICERR_OK )
        {
            closeHIC();
            return 0;
        }

        cvFlip( frame, frame, 0 );
    }
    else
    {
        IplImage src;
        cvInitImageHeader( &src, cvSize(vfmt0.biWidth, vfmt0.biHeight),
            IPL_DEPTH_8U, 3, IPL_ORIGIN_BL, 4 );
        cvSetData( &src, hdr->lpData, src.widthStep );
        cvFlip( &src, frame, 0 );
    }

    return frame;
}


double CvCaptureCAM_VFW::getProperty( int property_id )
{
    switch( property_id )
    {
    case CV_CAP_PROP_FRAME_WIDTH:
        return frame ? frame->width : 0;
    case CV_CAP_PROP_FRAME_HEIGHT:
        return frame ? frame->height : 0;
    case CV_CAP_PROP_FOURCC:
        return fourcc;
    }
    return 0;
}


CvCapture* cvCreateCameraCapture_VFW( int index )
{
    CvCaptureCAM_VFW* capture = new CvCaptureCAM_VFW;

    if( capture->open( index ))
        return capture;

    delete capture;
    return 0;
}


/*************************** writing AVIs ******************************/

class CvVideoWriter_VFW : public CvVideoWriter
{
public:
    CvVideoWriter_VFW() { init(); }
    virtual ~CvVideoWriter_VFW() { close(); }

    virtual bool open( const char* filename, int fourcc,
                       double fps, CvSize frameSize, bool isColor );
    virtual void close();
    virtual bool writeFrame( const IplImage* );

protected:
    void init();
    bool createStreams( CvSize frameSize, bool isColor );

    PAVIFILE      avifile;
    PAVISTREAM    compressed;
    PAVISTREAM    uncompressed;
    double        fps;
    IplImage*     tempFrame;
    long          pos;
    int           fourcc;
};


void CvVideoWriter_VFW::init()
{
    avifile = 0;
    compressed = uncompressed = 0;
    fps = 0;
    tempFrame = 0;
    pos = 0;
    fourcc = 0;
}

void CvVideoWriter_VFW::close()
{
    if( uncompressed )
        AVIStreamRelease( uncompressed );
    if( compressed )
        AVIStreamRelease( compressed );
    if( avifile )
        AVIFileRelease( avifile );
    cvReleaseImage( &tempFrame );
    init();
}


// philipg.  Made this code capable of writing 8bpp gray scale bitmaps
struct BITMAPINFO_8Bit
{
    BITMAPINFOHEADER bmiHeader;
    RGBQUAD          bmiColors[256];
};


bool CvVideoWriter_VFW::open( const char* filename, int _fourcc, double _fps, CvSize frameSize, bool isColor )
{
    close();

    icvInitCapture_VFW();
    if( AVIFileOpen( &avifile, filename, OF_CREATE | OF_WRITE, 0 ) == AVIERR_OK )
    {
        fourcc = _fourcc;
        fps = _fps;
        if( frameSize.width > 0 && frameSize.height > 0 &&
            !createStreams( frameSize, isColor ) )
        {
            close();
            return false;
        }
    }
    return true;
}


bool CvVideoWriter_VFW::createStreams( CvSize frameSize, bool isColor )
{
    if( !avifile )
        return false;
    AVISTREAMINFO aviinfo;

    BITMAPINFO_8Bit bmih;
    bmih.bmiHeader = icvBitmapHeader( frameSize.width, frameSize.height, isColor ? 24 : 8 );
    for( int i = 0; i < 256; i++ )
    {
        bmih.bmiColors[i].rgbBlue = (BYTE)i;
        bmih.bmiColors[i].rgbGreen = (BYTE)i;
        bmih.bmiColors[i].rgbRed = (BYTE)i;
        bmih.bmiColors[i].rgbReserved = 0;
    }

    memset( &aviinfo, 0, sizeof(aviinfo));
    aviinfo.fccType = streamtypeVIDEO;
    aviinfo.fccHandler = 0;
    // use highest possible accuracy for dwRate/dwScale
    aviinfo.dwScale = (DWORD)((double)0x7FFFFFFF / fps);
    aviinfo.dwRate = cvRound(fps * aviinfo.dwScale);
    aviinfo.rcFrame.top = aviinfo.rcFrame.left = 0;
    aviinfo.rcFrame.right = frameSize.width;
    aviinfo.rcFrame.bottom = frameSize.height;

    if( AVIFileCreateStream( avifile, &uncompressed, &aviinfo ) == AVIERR_OK )
    {
        AVICOMPRESSOPTIONS copts, *pcopts = &copts;
        copts.fccType = streamtypeVIDEO;
        copts.fccHandler = fourcc != -1 ? fourcc : 0;
        copts.dwKeyFrameEvery = 1;
        copts.dwQuality = 10000;
        copts.dwBytesPerSecond = 0;
        copts.dwFlags = AVICOMPRESSF_VALID;
        copts.lpFormat = &bmih;
        copts.cbFormat = (isColor ? sizeof(BITMAPINFOHEADER) : sizeof(bmih));
        copts.lpParms = 0;
        copts.cbParms = 0;
        copts.dwInterleaveEvery = 0;

        if( fourcc != -1 || AVISaveOptions( 0, 0, 1, &uncompressed, &pcopts ) == TRUE )
        {
            if( AVIMakeCompressedStream( &compressed, uncompressed, pcopts, 0 ) == AVIERR_OK &&
                AVIStreamSetFormat( compressed, 0, &bmih, sizeof(bmih)) == AVIERR_OK )
            {
                fps = fps;
                fourcc = (int)copts.fccHandler;
                frameSize = frameSize;
                tempFrame = cvCreateImage( frameSize, 8, (isColor ? 3 : 1) );
                return true;
            }
        }
    }
    return false;
}


bool CvVideoWriter_VFW::writeFrame( const IplImage* image )
{
    bool result = false;
    CV_FUNCNAME( "CvVideoWriter_VFW::writeFrame" );

    __BEGIN__;

    if( !image )
        EXIT;

    if( !compressed && !createStreams( cvGetSize(image), image->nChannels > 1 ))
        EXIT;

    if( image->width != tempFrame->width || image->height != tempFrame->height )
        CV_ERROR( CV_StsUnmatchedSizes,
            "image size is different from the currently set frame size" );

    if( image->nChannels != tempFrame->nChannels ||
        image->depth != tempFrame->depth ||
        image->origin == 0 ||
        image->widthStep != cvAlign(image->width*image->nChannels*((image->depth & 255)/8), 4))
    {
        cvConvertImage( image, tempFrame, image->origin == 0 ? CV_CVTIMG_FLIP : 0 );
        image = (const IplImage*)tempFrame;
    }

    result = AVIStreamWrite( compressed, pos++, 1, image->imageData,
                             image->imageSize, AVIIF_KEYFRAME, 0, 0 ) == AVIERR_OK;

    __END__;

    return result;
}

CvVideoWriter* cvCreateVideoWriter_VFW( const char* filename, int fourcc,
                                        double fps, CvSize frameSize, int isColor )
{
    CvVideoWriter_VFW* writer = new CvVideoWriter_VFW;
    if( writer->open( filename, fourcc, fps, frameSize, isColor != 0 ))
        return writer;
    delete writer;
    return 0;
}
