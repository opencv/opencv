#include "precomp.hpp"

#include "xiApi.h"
#include "xiExt.h"
#include "m3Api.h"

/**********************************************************************************/

class CvCaptureCAM_XIMEA : public CvCapture
{
public:
    CvCaptureCAM_XIMEA() { init(); }
    virtual ~CvCaptureCAM_XIMEA() { close(); }

    virtual bool open( int index );
    virtual void close();
    virtual double getProperty(int);
    virtual bool setProperty(int, double);
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int);
    virtual int getCaptureDomain() { return CV_CAP_XIAPI; } // Return the type of the capture object: CV_CAP_VFW, etc...

private:
    void init();
    void errMsg(const char* msg, int errNum);
    void resetCvImage();
    int  getBpp();
    IplImage* frame;

    HANDLE    hmv;
    DWORD     numDevices;
    int       timeout;
    XI_IMG    image;
};

/**********************************************************************************/

CvCapture* cvCreateCameraCapture_XIMEA( int index )
{
    CvCaptureCAM_XIMEA* capture = new CvCaptureCAM_XIMEA;

    if( capture->open( index ))
        return capture;

    delete capture;
    return 0;
}

/**********************************************************************************/
// Enumerate connected devices
void CvCaptureCAM_XIMEA::init()
{
    xiGetNumberDevices( &numDevices);
    hmv = NULL;
    frame = NULL;
    timeout = 0;
    memset(&image, 0, sizeof(XI_IMG));
}


/**********************************************************************************/
// Initialize camera input
bool CvCaptureCAM_XIMEA::open( int wIndex )
{
#define HandleXiResult(res) if (res!=XI_OK)  goto error;

    int mvret = XI_OK;

    if(numDevices == 0)
        return false;

    if((mvret = xiOpenDevice( wIndex, &hmv)) != XI_OK)
    {
        errMsg("Open XI_DEVICE failed", mvret);
        return false;
    }

    // always use auto exposure/gain
    mvret = xiSetParamInt( hmv, XI_PRM_AEAG, 1);
    HandleXiResult(mvret);

    int width = 0;
    mvret = xiGetParamInt( hmv, XI_PRM_WIDTH, &width);
    HandleXiResult(mvret);

    int height = 0;
    mvret = xiGetParamInt( hmv, XI_PRM_HEIGHT, &height);
    HandleXiResult(mvret);

    int isColor = 0;
    mvret = xiGetParamInt(hmv, XI_PRM_IMAGE_IS_COLOR, &isColor);
    HandleXiResult(mvret);

    if(isColor)	// for color cameras
    {
        // default image format RGB24
        mvret = xiSetParamInt( hmv, XI_PRM_IMAGE_DATA_FORMAT, XI_RGB24);
        HandleXiResult(mvret);

        // always use auto white ballance for color cameras
        mvret = xiSetParamInt( hmv, XI_PRM_AUTO_WB, 1);
        HandleXiResult(mvret);

        // allocate frame buffer for RGB24 image
        frame = cvCreateImage(cvSize( width, height), IPL_DEPTH_8U, 3);
    }
    else // for mono cameras
    {
        // default image format MONO8
        mvret = xiSetParamInt( hmv, XI_PRM_IMAGE_DATA_FORMAT, XI_MONO8);
        HandleXiResult(mvret);

        // allocate frame buffer for MONO8 image
        frame = cvCreateImage(cvSize( width, height), IPL_DEPTH_8U, 1);
    }

    //default capture timeout 10s
    timeout = 10000;

    mvret = xiStartAcquisition(hmv);
    if(mvret != XI_OK)
    {
        errMsg("StartAcquisition XI_DEVICE failed", mvret);
        goto error;
    }
    return true;

error:
    errMsg("Open XI_DEVICE failed", mvret);
    xiCloseDevice(hmv);
    hmv = NULL;
    return false;
}

/**********************************************************************************/

void CvCaptureCAM_XIMEA::close()
{
    if(frame)
        cvReleaseImage(&frame);

    if(hmv)
    {
        xiStopAcquisition(hmv);
        xiCloseDevice(hmv);
    }
    hmv = NULL;
}

/**********************************************************************************/

bool CvCaptureCAM_XIMEA::grabFrame()
{
    memset(&image, 0, sizeof(XI_IMG));
    image.size = sizeof(XI_IMG);
    int mvret = xiGetImage( hmv, timeout, &image);

    if(mvret == MM40_ACQUISITION_STOPED)
    {
        xiStartAcquisition(hmv);
        mvret = xiGetImage(hmv, timeout, &image);
    }

    if(mvret != XI_OK)
    {
        errMsg("Error during GetImage", mvret);
        return false;
    }

    return true;
}

/**********************************************************************************/

IplImage* CvCaptureCAM_XIMEA::retrieveFrame(int)
{
    // update cvImage after format has changed
    resetCvImage();

    // copy pixel data
    switch( image.frm)
    {
    case XI_MONO8       :
    case XI_RAW8        : memcpy( frame->imageData, image.bp, image.width*image.height); break;
    case XI_MONO16      :
    case XI_RAW16       : memcpy( frame->imageData, image.bp, image.width*image.height*sizeof(WORD)); break;
    case XI_RGB24       :
    case XI_RGB_PLANAR  : memcpy( frame->imageData, image.bp, image.width*image.height*3); break;
    case XI_RGB32       : memcpy( frame->imageData, image.bp, image.width*image.height*4); break;
    default: break;
    }
    return frame;
}

/**********************************************************************************/

void CvCaptureCAM_XIMEA::resetCvImage()
{
    int width = 0, height = 0, format = 0;
    xiGetParamInt( hmv, XI_PRM_WIDTH, &width);
    xiGetParamInt( hmv, XI_PRM_HEIGHT, &height);
    xiGetParamInt( hmv, XI_PRM_IMAGE_DATA_FORMAT, &format);

    if( (int)image.width != width || (int)image.height != height || image.frm != (XI_IMG_FORMAT)format)
    {
        if(frame) cvReleaseImage(&frame);
        frame = NULL;

        switch( image.frm)
        {
        case XI_MONO8       :
        case XI_RAW8        : frame = cvCreateImage(cvSize( image.width, image.height), IPL_DEPTH_8U, 1); break;
        case XI_MONO16      :
        case XI_RAW16       : frame = cvCreateImage(cvSize( image.width, image.height), IPL_DEPTH_16U, 1); break;
        case XI_RGB24       :
        case XI_RGB_PLANAR  : frame = cvCreateImage(cvSize( image.width, image.height), IPL_DEPTH_8U, 3); break;
        case XI_RGB32       : frame = cvCreateImage(cvSize( image.width, image.height), IPL_DEPTH_8U, 4); break;
        default :
            return;
        }
    }
    cvZero(frame);
}
/**********************************************************************************/

double CvCaptureCAM_XIMEA::getProperty( int property_id )
{
    if(hmv == NULL)
        return 0;

    int ival = 0;
    float fval = 0;

    switch( property_id )
    {
    // OCV parameters
    case CV_CAP_PROP_POS_FRAMES   : return (double) image.nframe;
    case CV_CAP_PROP_FRAME_WIDTH  : xiGetParamInt( hmv, XI_PRM_WIDTH, &ival); return ival;
    case CV_CAP_PROP_FRAME_HEIGHT : xiGetParamInt( hmv, XI_PRM_HEIGHT, &ival); return ival;
    case CV_CAP_PROP_FPS          : xiGetParamFloat( hmv, XI_PRM_FRAMERATE, &fval); return fval;
    case CV_CAP_PROP_GAIN         : xiGetParamFloat( hmv, XI_PRM_GAIN, &fval); return fval;
    case CV_CAP_PROP_EXPOSURE     : xiGetParamInt( hmv, XI_PRM_EXPOSURE, &ival); return ival;

    // XIMEA camera properties
    case CV_CAP_PROP_XI_DOWNSAMPLING  : xiGetParamInt( hmv, XI_PRM_DOWNSAMPLING, &ival); return ival;
    case CV_CAP_PROP_XI_DATA_FORMAT   : xiGetParamInt( hmv, XI_PRM_IMAGE_DATA_FORMAT, &ival); return ival;
    case CV_CAP_PROP_XI_OFFSET_X      : xiGetParamInt( hmv, XI_PRM_OFFSET_X, &ival); return ival;
    case CV_CAP_PROP_XI_OFFSET_Y      : xiGetParamInt( hmv, XI_PRM_OFFSET_Y, &ival); return ival;
    case CV_CAP_PROP_XI_TRG_SOURCE    : xiGetParamInt( hmv, XI_PRM_TRG_SOURCE, &ival); return ival;
    case CV_CAP_PROP_XI_GPI_SELECTOR  : xiGetParamInt( hmv, XI_PRM_GPI_SELECTOR, &ival); return ival;
    case CV_CAP_PROP_XI_GPI_MODE      : xiGetParamInt( hmv, XI_PRM_GPI_MODE, &ival); return ival;
    case CV_CAP_PROP_XI_GPI_LEVEL     : xiGetParamInt( hmv, XI_PRM_GPI_LEVEL, &ival); return ival;
    case CV_CAP_PROP_XI_GPO_SELECTOR  : xiGetParamInt( hmv, XI_PRM_GPO_SELECTOR, &ival); return ival;
    case CV_CAP_PROP_XI_GPO_MODE      : xiGetParamInt( hmv, XI_PRM_GPO_MODE, &ival); return ival;
    case CV_CAP_PROP_XI_LED_SELECTOR  : xiGetParamInt( hmv, XI_PRM_LED_SELECTOR, &ival); return ival;
    case CV_CAP_PROP_XI_LED_MODE      : xiGetParamInt( hmv, XI_PRM_LED_MODE, &ival); return ival;
    case CV_CAP_PROP_XI_AUTO_WB       : xiGetParamInt( hmv, XI_PRM_AUTO_WB, &ival); return ival;
    case CV_CAP_PROP_XI_AEAG          : xiGetParamInt( hmv, XI_PRM_AEAG, &ival); return ival;
    case CV_CAP_PROP_XI_EXP_PRIORITY  : xiGetParamFloat( hmv, XI_PRM_EXP_PRIORITY, &fval); return fval;
    case CV_CAP_PROP_XI_AE_MAX_LIMIT  : xiGetParamInt( hmv, XI_PRM_EXP_PRIORITY, &ival); return ival;
    case CV_CAP_PROP_XI_AG_MAX_LIMIT  : xiGetParamFloat( hmv, XI_PRM_AG_MAX_LIMIT, &fval); return fval;
    case CV_CAP_PROP_XI_AEAG_LEVEL    : xiGetParamInt( hmv, XI_PRM_AEAG_LEVEL, &ival); return ival;
    case CV_CAP_PROP_XI_TIMEOUT       : return timeout;

    }
    return 0;
}

/**********************************************************************************/

bool CvCaptureCAM_XIMEA::setProperty( int property_id, double value )
{
    int ival = (int) value;
    float fval = (float) value;

    int mvret = XI_OK;

    switch(property_id)
    {
    // OCV parameters
    case CV_CAP_PROP_FRAME_WIDTH  : mvret = xiSetParamInt( hmv, XI_PRM_WIDTH, ival); break;
    case CV_CAP_PROP_FRAME_HEIGHT : mvret = xiSetParamInt( hmv, XI_PRM_HEIGHT, ival); break;
    case CV_CAP_PROP_FPS          : mvret = xiSetParamFloat( hmv, XI_PRM_FRAMERATE, fval); break;
    case CV_CAP_PROP_GAIN         : mvret = xiSetParamFloat( hmv, XI_PRM_GAIN, fval); break;
    case CV_CAP_PROP_EXPOSURE     : mvret = xiSetParamInt( hmv, XI_PRM_EXPOSURE, ival); break;
    // XIMEA camera properties
    case CV_CAP_PROP_XI_DOWNSAMPLING  : mvret = xiSetParamInt( hmv, XI_PRM_DOWNSAMPLING, ival); break;
    case CV_CAP_PROP_XI_DATA_FORMAT   : mvret = xiSetParamInt( hmv, XI_PRM_IMAGE_DATA_FORMAT, ival); break;
    case CV_CAP_PROP_XI_OFFSET_X      : mvret = xiSetParamInt( hmv, XI_PRM_OFFSET_X, ival); break;
    case CV_CAP_PROP_XI_OFFSET_Y      : mvret = xiSetParamInt( hmv, XI_PRM_OFFSET_Y, ival); break;
    case CV_CAP_PROP_XI_TRG_SOURCE    : mvret = xiSetParamInt( hmv, XI_PRM_TRG_SOURCE, ival); break;
    case CV_CAP_PROP_XI_GPI_SELECTOR  : mvret = xiSetParamInt( hmv, XI_PRM_GPI_SELECTOR, ival); break;
    case CV_CAP_PROP_XI_TRG_SOFTWARE  : mvret = xiSetParamInt( hmv, XI_PRM_TRG_SOURCE, 1); break;
    case CV_CAP_PROP_XI_GPI_MODE      : mvret = xiSetParamInt( hmv, XI_PRM_GPI_MODE, ival); break;
    case CV_CAP_PROP_XI_GPI_LEVEL     : mvret = xiSetParamInt( hmv, XI_PRM_GPI_LEVEL, ival); break;
    case CV_CAP_PROP_XI_GPO_SELECTOR  : mvret = xiSetParamInt( hmv, XI_PRM_GPO_SELECTOR, ival); break;
    case CV_CAP_PROP_XI_GPO_MODE      : mvret = xiSetParamInt( hmv, XI_PRM_GPO_MODE, ival); break;
    case CV_CAP_PROP_XI_LED_SELECTOR  : mvret = xiSetParamInt( hmv, XI_PRM_LED_SELECTOR, ival); break;
    case CV_CAP_PROP_XI_LED_MODE      : mvret = xiSetParamInt( hmv, XI_PRM_LED_MODE, ival); break;
    case CV_CAP_PROP_XI_AUTO_WB       : mvret = xiSetParamInt( hmv, XI_PRM_AUTO_WB, ival); break;
    case CV_CAP_PROP_XI_MANUAL_WB     : mvret = xiSetParamInt( hmv, XI_PRM_LED_MODE, ival); break;
    case CV_CAP_PROP_XI_AEAG          : mvret = xiSetParamInt( hmv, XI_PRM_AEAG, ival); break;
    case CV_CAP_PROP_XI_EXP_PRIORITY  : mvret = xiSetParamFloat( hmv, XI_PRM_EXP_PRIORITY, fval); break;
    case CV_CAP_PROP_XI_AE_MAX_LIMIT  : mvret = xiSetParamInt( hmv, XI_PRM_EXP_PRIORITY, ival); break;
    case CV_CAP_PROP_XI_AG_MAX_LIMIT  : mvret = xiSetParamFloat( hmv, XI_PRM_AG_MAX_LIMIT, fval); break;
    case CV_CAP_PROP_XI_AEAG_LEVEL    : mvret = xiSetParamInt( hmv, XI_PRM_AEAG_LEVEL, ival); break;
    case CV_CAP_PROP_XI_TIMEOUT       : timeout = ival; break;
    }

    if(mvret != XI_OK)
    {
        errMsg("Set parameter error", mvret);
        return false;
    }
    else
        return true;

}

/**********************************************************************************/

void CvCaptureCAM_XIMEA::errMsg(const char* msg, int errNum)
{
#if defined WIN32 || defined _WIN32
    char buf[512]="";
    sprintf( buf, "%s : %d\n", msg, errNum);
    OutputDebugString(buf);
#else
    fprintf(stderr, "%s : %d\n", msg, errNum);
#endif
}

/**********************************************************************************/

int  CvCaptureCAM_XIMEA::getBpp()
{
    switch( image.frm)
    {
    case XI_MONO8       :
    case XI_RAW8        : return 1;
    case XI_MONO16      :
    case XI_RAW16       : return 2;
    case XI_RGB24       :
    case XI_RGB_PLANAR  : return 3;
    case XI_RGB32       : return 4;
    default :
        return 0;
    }
}

/**********************************************************************************/
