
#include "precomp.hpp"

#ifdef _WIN32
#include <xiApi.h>
#else
#include <m3api/xiApi.h>
#endif

using namespace std;

/**********************************************************************************/

class CvCaptureCAM_XIMEA : public CvCapture
{
public:
    CvCaptureCAM_XIMEA() { init(); }
    virtual ~CvCaptureCAM_XIMEA() { close(); }

    virtual bool open( int index );
    virtual void close();
    virtual double getProperty(int) const;
    virtual bool setProperty(int, double);
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int);
    virtual int getCaptureDomain() { return CV_CAP_XIAPI; } // Return the type of the capture object: CV_CAP_VFW, etc...

private:
    void init();
    void errMsg(const char* msg, int errNum) const;
    void resetCvImage();
    int  ocvParamtoXimeaParam(int value) const;
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
#if defined _WIN32
    xiGetNumberDevices( &numDevices);
#else
    // try second re-enumeration if first one fails
    if (xiGetNumberDevices( &numDevices) != XI_OK)
    {
        xiGetNumberDevices( &numDevices);
    }
#endif
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
#if defined _WIN32
        errMsg("Open XI_DEVICE failed", mvret);
        return false;
#else
        // try opening second time if first fails
        if((mvret = xiOpenDevice( wIndex, &hmv))  != XI_OK)
        {
            errMsg("Open XI_DEVICE failed", mvret);
            return false;
        }
#endif
    }

    int width   = 0;
    int height  = 0;
    int isColor = 0;

    // always use auto exposure/gain
    mvret = xiSetParamInt( hmv, XI_PRM_AEAG, 1);
    HandleXiResult(mvret);

    mvret = xiGetParamInt( hmv, XI_PRM_WIDTH, &width);
    HandleXiResult(mvret);

    mvret = xiGetParamInt( hmv, XI_PRM_HEIGHT, &height);
    HandleXiResult(mvret);

    mvret = xiGetParamInt(hmv, XI_PRM_IMAGE_IS_COLOR, &isColor);
    HandleXiResult(mvret);

    if(isColor) // for color cameras
    {
        // default image format RGB24
        mvret = xiSetParamInt( hmv, XI_PRM_IMAGE_DATA_FORMAT, XI_RGB24);
        HandleXiResult(mvret);

        // always use auto white balance for color cameras
        mvret = xiSetParamInt( hmv, XI_PRM_AUTO_WB, 1);
        HandleXiResult(mvret);

        // allocate frame buffer for RGB24 image
        frame = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
    }
    else // for mono cameras
    {
        // default image format MONO8
        mvret = xiSetParamInt( hmv, XI_PRM_IMAGE_DATA_FORMAT, XI_MONO8);
        HandleXiResult(mvret);

        // allocate frame buffer for MONO8 image
        frame = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
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

    if(mvret == XI_ACQUISITION_STOPED)
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
   bool do_reset = false;

    // first check basic image resolution
    if((int)image.width != frame->width || (int)image.height != frame->height)
        do_reset = true;

    // afterwards check image format
    switch( image.frm)
    {
    case XI_MONO8       :
    case XI_RAW8         :
        {
            if(frame->depth != IPL_DEPTH_8U || frame->nChannels != 1)
                do_reset = true;
        }
        break;
    case XI_MONO16      :
    case XI_RAW16        :
        {
            if(frame->depth != IPL_DEPTH_16U || frame->nChannels != 1)
                do_reset = true;
        }
        break;
    case XI_RGB24       :
    case XI_RGB_PLANAR  :
        {
            if(frame->depth != IPL_DEPTH_8U || frame->nChannels != 3)
                do_reset = true;
        }
        break;
    case XI_RGB32       :
        {
            if(frame->depth != IPL_DEPTH_8U || frame->nChannels != 4)
                do_reset = true;
        }
        break;
    default:
        errMsg("CvCaptureCAM_XIMEA::resetCvImage ERROR: Unknown format.", XI_NOT_SUPPORTED_DATA_FORMAT);
        return;
    }

    if(do_reset)
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
            errMsg("CvCaptureCAM_XIMEA::resetCvImage ERROR: Unknown format.", XI_NOT_SUPPORTED_DATA_FORMAT);
            return;
        }
    }
    cvZero(frame);
}

/**********************************************************************************/

int CvCaptureCAM_XIMEA::ocvParamtoXimeaParam(int property_id) const
{
    XI_RETURN stat = XI_OK;
    switch (property_id)
    {
        // OCV parameters
    case CV_CAP_PROP_POS_FRAMES:
        // Number of successfully transferred frames on transport layer.
        stat = xiSetParamInt(hmv, XI_PRM_COUNTER_SELECTOR, XI_CNT_SEL_TRANSPORT_TRANSFERRED_FRAMES);
        if (stat) errMsg("xiSetParamInt(XI_PRM_COUNTER_SELECTOR)", stat);
        return CV_CAP_PROP_XI_COUNTER_VALUE;
    case CV_CAP_PROP_FRAME_WIDTH: return CV_CAP_PROP_XI_WIDTH;
    case CV_CAP_PROP_FRAME_HEIGHT: return CV_CAP_PROP_XI_HEIGHT;
    case CV_CAP_PROP_FPS: return CV_CAP_PROP_XI_FRAMERATE;
    case CV_CAP_PROP_GAIN: return CV_CAP_PROP_XI_GAIN;
    case CV_CAP_PROP_EXPOSURE: return CV_CAP_PROP_XI_EXPOSURE;
    case CV_CAP_PROP_XI_DATA_FORMAT: return CV_CAP_PROP_XI_IMAGE_DATA_FORMAT;
    default:
        return property_id;
    }
}

/**********************************************************************************/

bool CvCaptureCAM_XIMEA::setProperty( int property_id, double value )
{
    bool setProp_result = true;
    bool doAcqReset = false;
    string ximea_param = "";
    int ival = (int) value;
    float fval = (float) value;
    XI_PRM_TYPE value_type = xiTypeInteger;
    XI_RETURN stat = XI_OK;

    if(hmv == NULL)
    {
        errMsg("CvCaptureCAM_XIMEA::setProperty", XI_INVALID_HANDLE);
        return false;
    }

    // convert OCV property id to XIMEA id if necessary
    property_id = ocvParamtoXimeaParam(property_id);

    // decode OpenCV parameter to xiAPI parameter
    switch( property_id )
    {
    case CV_CAP_PROP_XI_TIMEOUT:
        timeout = (int) value;
        return true;
    case CV_CAP_PROP_XI_EXPOSURE:
        ximea_param = "exposure";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_EXPOSURE_BURST_COUNT:
        ximea_param = "exposure_burst_count";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_GAIN_SELECTOR:
        ximea_param = "gain_selector";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_GAIN:
        ximea_param = "gain";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_DOWNSAMPLING:
        ximea_param = "downsampling";
        value_type = xiTypeEnum;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_DOWNSAMPLING_TYPE:
        ximea_param = "downsampling_type";
        value_type = xiTypeEnum;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_BINNING_SELECTOR:
        ximea_param = "binning_selector";
        value_type = xiTypeEnum;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_BINNING_VERTICAL:
        ximea_param = "binning_vertical";
        value_type = xiTypeInteger;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_BINNING_HORIZONTAL:
        ximea_param = "binning_horizontal";
        value_type = xiTypeInteger;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_BINNING_PATTERN:
        ximea_param = "binning_pattern";
        value_type = xiTypeEnum;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_DECIMATION_SELECTOR:
        ximea_param = "decimation_selector";
        value_type = xiTypeEnum;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_DECIMATION_VERTICAL:
        ximea_param = "decimation_vertical";
        value_type = xiTypeInteger;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_DECIMATION_HORIZONTAL:
        ximea_param = "decimation_horizontal";
        value_type = xiTypeInteger;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_DECIMATION_PATTERN:
        ximea_param = "decimation_pattern";
        value_type = xiTypeEnum;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_TEST_PATTERN_GENERATOR_SELECTOR:
        ximea_param = "test_pattern_generator_selector";
        value_type = xiTypeEnum;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_TEST_PATTERN:
        ximea_param = "test_pattern";
        value_type = xiTypeEnum;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_IMAGE_DATA_FORMAT:
        ximea_param = "imgdataformat";
        value_type = xiTypeEnum;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_SHUTTER_TYPE:
        ximea_param = "shutter_type";
        value_type = xiTypeEnum;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_SENSOR_TAPS:
        ximea_param = "sensor_taps";
        value_type = xiTypeEnum;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_AEAG:
        ximea_param = "aeag";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_AEAG_ROI_OFFSET_X:
        ximea_param = "aeag_roi_offset_x";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_AEAG_ROI_OFFSET_Y:
        ximea_param = "aeag_roi_offset_y";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_AEAG_ROI_WIDTH:
        ximea_param = "aeag_roi_width";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_AEAG_ROI_HEIGHT:
        ximea_param = "aeag_roi_height";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_BPC:
        ximea_param = "bpc";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_AUTO_WB:
        ximea_param = "auto_wb";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_MANUAL_WB:
        ximea_param = "manual_wb";
        value_type = xiTypeCommand;
        break;
    case CV_CAP_PROP_XI_WB_KR:
        ximea_param = "wb_kr";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_WB_KG:
        ximea_param = "wb_kg";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_WB_KB:
        ximea_param = "wb_kb";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_WIDTH:
        ximea_param = "width";
        value_type = xiTypeInteger;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_HEIGHT:
        ximea_param = "height";
        value_type = xiTypeInteger;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_OFFSET_X:
        ximea_param = "offsetX";
        value_type = xiTypeInteger;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_OFFSET_Y:
        ximea_param = "offsetY";
        value_type = xiTypeInteger;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_REGION_SELECTOR :
        ximea_param = "region_selector";
        value_type = xiTypeInteger;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_REGION_MODE :
        ximea_param = "region_mode";
        value_type = xiTypeInteger;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_EXP_PRIORITY:
        ximea_param = "exp_priority";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_AG_MAX_LIMIT:
        ximea_param = "ag_max_limit";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_AE_MAX_LIMIT:
        ximea_param = "ae_max_limit";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_AEAG_LEVEL:
        ximea_param = "aeag_level";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_LIMIT_BANDWIDTH:
        ximea_param = "limit_bandwidth";
        value_type = xiTypeInteger;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_SENSOR_DATA_BIT_DEPTH:
        ximea_param = "sensor_bit_depth";
        value_type = xiTypeEnum;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_OUTPUT_DATA_BIT_DEPTH:
        ximea_param = "output_bit_depth";
        value_type = xiTypeEnum;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_IMAGE_DATA_BIT_DEPTH:
        ximea_param = "image_data_bit_depth";
        value_type = xiTypeEnum;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_OUTPUT_DATA_PACKING:
        ximea_param = "output_bit_packing";
        value_type = xiTypeBoolean;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_OUTPUT_DATA_PACKING_TYPE:
        ximea_param = "output_bit_packing_type";
        value_type = xiTypeEnum;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_IS_COOLED:
        ximea_param = "iscooled";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_COOLING:
        ximea_param = "cooling";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_TARGET_TEMP:
        ximea_param = "target_temp";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CHIP_TEMP:
        ximea_param = "chip_temp";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_HOUS_TEMP:
        ximea_param = "hous_temp";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_HOUS_BACK_SIDE_TEMP:
        ximea_param = "hous_back_side_temp";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_SENSOR_BOARD_TEMP:
        ximea_param = "sensor_board_temp";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CMS:
        ximea_param = "cms";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_APPLY_CMS:
        ximea_param = "apply_cms";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_IMAGE_IS_COLOR:
        ximea_param = "iscolor";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_COLOR_FILTER_ARRAY:
        ximea_param = "cfa";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_GAMMAY:
        ximea_param = "gammaY";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_GAMMAC:
        ximea_param = "gammaC";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_SHARPNESS:
        ximea_param = "sharpness";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_00:
        ximea_param = "ccMTX00";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_01:
        ximea_param = "ccMTX01";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_02:
        ximea_param = "ccMTX02";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_03:
        ximea_param = "ccMTX03";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_10:
        ximea_param = "ccMTX10";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_11:
        ximea_param = "ccMTX11";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_12:
        ximea_param = "ccMTX12";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_13:
        ximea_param = "ccMTX13";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_20:
        ximea_param = "ccMTX20";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_21:
        ximea_param = "ccMTX21";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_22:
        ximea_param = "ccMTX22";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_23:
        ximea_param = "ccMTX23";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_30:
        ximea_param = "ccMTX30";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_31:
        ximea_param = "ccMTX31";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_32:
        ximea_param = "ccMTX32";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_33:
        ximea_param = "ccMTX33";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_DEFAULT_CC_MATRIX:
        ximea_param = "defccMTX";
        value_type = xiTypeCommand;
        break;
    case CV_CAP_PROP_XI_TRG_SOURCE:
        ximea_param = "trigger_source";
        value_type = xiTypeEnum;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_TRG_SOFTWARE:
        ximea_param = "trigger_software";
        value_type = xiTypeCommand;
        break;
    case CV_CAP_PROP_XI_TRG_SELECTOR:
        ximea_param = "trigger_selector";
        value_type = xiTypeEnum;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_ACQ_FRAME_BURST_COUNT:
        ximea_param = "acq_frame_burst_count";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_GPI_SELECTOR:
        ximea_param = "gpi_selector";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_GPI_MODE:
        ximea_param = "gpi_mode";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_GPI_LEVEL:
        ximea_param = "gpi_level";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_GPO_SELECTOR:
        ximea_param = "gpo_selector";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_GPO_MODE:
        ximea_param = "gpo_mode";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_LED_SELECTOR:
        ximea_param = "led_selector";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_LED_MODE:
        ximea_param = "led_mode";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_DEBOUNCE_EN:
        ximea_param = "dbnc_en";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_DEBOUNCE_T0:
        ximea_param = "dbnc_t0";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_DEBOUNCE_T1:
        ximea_param = "dbnc_t1";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_DEBOUNCE_POL:
        ximea_param = "dbnc_pol";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_LENS_MODE:
        ximea_param = "lens_mode";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_LENS_APERTURE_VALUE:
        ximea_param = "lens_aperture_value";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_LENS_FOCUS_MOVEMENT_VALUE:
        ximea_param = "lens_focus_movement_value";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_LENS_FOCUS_MOVE:
        ximea_param = "lens_focus_move";
        value_type = xiTypeCommand;
        break;
    case CV_CAP_PROP_XI_LENS_FOCUS_DISTANCE:
        ximea_param = "lens_focus_distance";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_LENS_FOCAL_LENGTH:
        ximea_param = "lens_focal_length";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_LENS_FEATURE_SELECTOR:
        ximea_param = "lens_feature_selector";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_LENS_FEATURE:
        ximea_param = "lens_feature";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_DEVICE_MODEL_ID:
        ximea_param = "device_model_id";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_DEVICE_SN:
        ximea_param = "device_sn";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_IMAGE_DATA_FORMAT_RGB32_ALPHA:
        ximea_param = "imgdataformatrgb32alpha";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_IMAGE_PAYLOAD_SIZE:
        ximea_param = "imgpayloadsize";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_TRANSPORT_PIXEL_FORMAT:
        ximea_param = "transport_pixel_format";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_SENSOR_CLOCK_FREQ_HZ:
        ximea_param = "sensor_clock_freq_hz";
        value_type = xiTypeFloat;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_SENSOR_CLOCK_FREQ_INDEX:
        ximea_param = "sensor_clock_freq_index";
        value_type = xiTypeInteger;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_SENSOR_OUTPUT_CHANNEL_COUNT:
        ximea_param = "sensor_output_channel_count";
        value_type = xiTypeEnum;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_FRAMERATE:
        ximea_param = "framerate";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_COUNTER_SELECTOR:
        ximea_param = "counter_selector";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_COUNTER_VALUE:
        ximea_param = "counter_value";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_ACQ_TIMING_MODE:
        ximea_param = "acq_timing_mode";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_AVAILABLE_BANDWIDTH:
        ximea_param = "available_bandwidth";
        value_type = xiTypeInteger;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_BUFFER_POLICY:
        ximea_param = "buffer_policy";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_LUT_EN:
        ximea_param = "LUTEnable";
        value_type = xiTypeBoolean;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_LUT_INDEX:
        ximea_param = "LUTIndex";
        value_type = xiTypeInteger;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_LUT_VALUE:
        ximea_param = "LUTValue";
        value_type = xiTypeInteger;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_TRG_DELAY:
        ximea_param = "trigger_delay";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_TS_RST_MODE:
        ximea_param = "ts_rst_mode";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_TS_RST_SOURCE:
        ximea_param = "ts_rst_source";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_IS_DEVICE_EXIST:
        ximea_param = "isexist";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_ACQ_BUFFER_SIZE:
        ximea_param = "acq_buffer_size";
        value_type = xiTypeInteger;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_ACQ_BUFFER_SIZE_UNIT:
        ximea_param = "acq_buffer_size_unit";
        value_type = xiTypeInteger;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_ACQ_TRANSPORT_BUFFER_SIZE:
        ximea_param = "acq_transport_buffer_size";
        value_type = xiTypeInteger;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_BUFFERS_QUEUE_SIZE:
        ximea_param = "buffers_queue_size";
        value_type = xiTypeInteger;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_ACQ_TRANSPORT_BUFFER_COMMIT:
        ximea_param = "acq_transport_buffer_commit";
        value_type = xiTypeInteger;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_RECENT_FRAME:
        ximea_param = "recent_frame";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_DEVICE_RESET:
        ximea_param = "device_reset";
        value_type = xiTypeCommand;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_COLUMN_FPN_CORRECTION:
        ximea_param = "column_fpn_correction";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_ROW_FPN_CORRECTION:
        ximea_param = "row_fpn_correction";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_SENSOR_MODE:
        ximea_param = "sensor_mode";
        value_type = xiTypeEnum;
        doAcqReset = true;
        break;
    case CV_CAP_PROP_XI_HDR:
        ximea_param = "hdr";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_HDR_KNEEPOINT_COUNT:
        ximea_param = "hdr_kneepoint_count";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_HDR_T1:
        ximea_param = "hdr_t1";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_HDR_T2:
        ximea_param = "hdr_t2";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_KNEEPOINT1:
        ximea_param = "hdr_kneepoint1";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_KNEEPOINT2:
        ximea_param = "hdr_kneepoint2";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_IMAGE_BLACK_LEVEL:
        ximea_param = "image_black_level";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_HW_REVISION:
        ximea_param = "hw_revision";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_DEBUG_LEVEL:
        ximea_param = "debug_level";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_AUTO_BANDWIDTH_CALCULATION:
        ximea_param = "auto_bandwidth_calculation";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_FFS_FILE_ID:
        ximea_param = "ffs_file_id";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_FFS_FILE_SIZE:
        ximea_param = "ffs_file_size";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_FREE_FFS_SIZE:
        ximea_param = "free_ffs_size";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_USED_FFS_SIZE:
        ximea_param = "used_ffs_size";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_FFS_ACCESS_KEY:
        ximea_param = "ffs_access_key";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_SENSOR_FEATURE_SELECTOR:
        ximea_param = "sensor_feature_selector";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_SENSOR_FEATURE_VALUE:
        ximea_param = "sensor_feature_value";
        value_type = xiTypeInteger;
        break;
    default:
        // report invalid parameter as it is not of numerical type
        errMsg("CvCaptureCAM_XIMEA::setProperty", XI_UNKNOWN_PARAM);
        return false;
    }

    if(doAcqReset)
    {
        stat = xiStopAcquisition(hmv);
        errMsg("CvCaptureCAM_XIMEA::setProperty, xiStopAcquisition", stat);
        if(stat != XI_OK)
            setProp_result = false;
    }

    switch(value_type)
    {
    case xiTypeInteger :               // integer parameter type
    case xiTypeEnum :                // enumerator parameter type
    case xiTypeBoolean :             // boolean parameter type
    case xiTypeCommand :          // command parameter type
        stat = xiSetParamInt(hmv, ximea_param.c_str(), ival);
        break;
    case xiTypeFloat :                  // float parameter type
        stat = xiSetParamFloat(hmv, ximea_param.c_str(), fval);
        break;
    default:
        errMsg("CvCaptureCAM_XIMEA::setProperty", XI_WRONG_PARAM_TYPE);
        setProp_result = false;
    }

    if(stat != XI_OK)
    {
        // report error on parameter setting
        errMsg("CvCaptureCAM_XIMEA::setProperty, xiSetParam", stat);
        setProp_result = false;
    }

    if(doAcqReset)
    {
        stat = xiStartAcquisition(hmv);
        errMsg("xiStartAcquisition::setProperty, xiStartAcquisition", stat);
        if(stat != XI_OK)
            setProp_result = false;
    }
    return setProp_result;
}

/**********************************************************************************/

double CvCaptureCAM_XIMEA::getProperty( int property_id ) const
{
    XI_RETURN stat = XI_OK;
    double getPropVal = 0;
    int ival = 0;
    float fval = 0;
    string ximea_param = "";
    XI_PRM_TYPE value_type = xiTypeInteger;

    if(hmv == NULL)
    {
        errMsg("CvCaptureCAM_XIMEA::getProperty", XI_INVALID_HANDLE);
        return 0;
    }

    // convert OCV property id to XIMEA id if necessary
    property_id = ocvParamtoXimeaParam(property_id);

    // decode OpenCV parameter to xiAPI parameter
    switch( property_id )
    {
    case CV_CAP_PROP_XI_TIMEOUT:
        return (double) timeout;
    case CV_CAP_PROP_XI_EXPOSURE:
        ximea_param = "exposure";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_EXPOSURE_BURST_COUNT:
        ximea_param = "exposure_burst_count";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_GAIN_SELECTOR:
        ximea_param = "gain_selector";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_GAIN:
        ximea_param = "gain";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_DOWNSAMPLING:
        ximea_param = "downsampling";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_DOWNSAMPLING_TYPE:
        ximea_param = "downsampling_type";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_BINNING_SELECTOR:
        ximea_param = "binning_selector";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_BINNING_VERTICAL:
        ximea_param = "binning_vertical";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_BINNING_HORIZONTAL:
        ximea_param = "binning_horizontal";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_BINNING_PATTERN:
        ximea_param = "binning_pattern";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_DECIMATION_SELECTOR:
        ximea_param = "decimation_selector";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_DECIMATION_VERTICAL:
        ximea_param = "decimation_vertical";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_DECIMATION_HORIZONTAL:
        ximea_param = "decimation_horizontal";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_DECIMATION_PATTERN:
        ximea_param = "decimation_pattern";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_TEST_PATTERN_GENERATOR_SELECTOR:
        ximea_param = "test_pattern_generator_selector";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_TEST_PATTERN:
        ximea_param = "test_pattern";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_IMAGE_DATA_FORMAT:
        ximea_param = "imgdataformat";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_SHUTTER_TYPE:
        ximea_param = "shutter_type";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_SENSOR_TAPS:
        ximea_param = "sensor_taps";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_AEAG:
        ximea_param = "aeag";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_AEAG_ROI_OFFSET_X:
        ximea_param = "aeag_roi_offset_x";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_AEAG_ROI_OFFSET_Y:
        ximea_param = "aeag_roi_offset_y";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_AEAG_ROI_WIDTH:
        ximea_param = "aeag_roi_width";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_AEAG_ROI_HEIGHT:
        ximea_param = "aeag_roi_height";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_BPC:
        ximea_param = "bpc";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_AUTO_WB:
        ximea_param = "auto_wb";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_MANUAL_WB:
        ximea_param = "manual_wb";
        value_type = xiTypeCommand;
        break;
    case CV_CAP_PROP_XI_WB_KR:
        ximea_param = "wb_kr";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_WB_KG:
        ximea_param = "wb_kg";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_WB_KB:
        ximea_param = "wb_kb";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_WIDTH:
        ximea_param = "width";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_HEIGHT:
        ximea_param = "height";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_OFFSET_X:
        ximea_param = "offsetX";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_OFFSET_Y:
        ximea_param = "offsetY";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_REGION_SELECTOR :
        ximea_param = "region_selector";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_REGION_MODE :
        ximea_param = "region_mode";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_EXP_PRIORITY:
        ximea_param = "exp_priority";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_AG_MAX_LIMIT:
        ximea_param = "ag_max_limit";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_AE_MAX_LIMIT:
        ximea_param = "ae_max_limit";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_AEAG_LEVEL:
        ximea_param = "aeag_level";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_LIMIT_BANDWIDTH:
        ximea_param = "limit_bandwidth";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_SENSOR_DATA_BIT_DEPTH:
        ximea_param = "sensor_bit_depth";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_OUTPUT_DATA_BIT_DEPTH:
        ximea_param = "output_bit_depth";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_IMAGE_DATA_BIT_DEPTH:
        ximea_param = "image_data_bit_depth";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_OUTPUT_DATA_PACKING:
        ximea_param = "output_bit_packing";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_OUTPUT_DATA_PACKING_TYPE:
        ximea_param = "output_bit_packing_type";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_IS_COOLED:
        ximea_param = "iscooled";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_COOLING:
        ximea_param = "cooling";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_TARGET_TEMP:
        ximea_param = "target_temp";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CHIP_TEMP:
        ximea_param = "chip_temp";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_HOUS_TEMP:
        ximea_param = "hous_temp";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_HOUS_BACK_SIDE_TEMP:
        ximea_param = "hous_back_side_temp";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_SENSOR_BOARD_TEMP:
        ximea_param = "sensor_board_temp";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CMS:
        ximea_param = "cms";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_APPLY_CMS:
        ximea_param = "apply_cms";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_IMAGE_IS_COLOR:
        ximea_param = "iscolor";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_COLOR_FILTER_ARRAY:
        ximea_param = "cfa";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_GAMMAY:
        ximea_param = "gammaY";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_GAMMAC:
        ximea_param = "gammaC";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_SHARPNESS:
        ximea_param = "sharpness";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_00:
        ximea_param = "ccMTX00";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_01:
        ximea_param = "ccMTX01";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_02:
        ximea_param = "ccMTX02";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_03:
        ximea_param = "ccMTX03";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_10:
        ximea_param = "ccMTX10";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_11:
        ximea_param = "ccMTX11";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_12:
        ximea_param = "ccMTX12";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_13:
        ximea_param = "ccMTX13";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_20:
        ximea_param = "ccMTX20";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_21:
        ximea_param = "ccMTX21";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_22:
        ximea_param = "ccMTX22";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_23:
        ximea_param = "ccMTX23";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_30:
        ximea_param = "ccMTX30";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_31:
        ximea_param = "ccMTX31";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_32:
        ximea_param = "ccMTX32";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_CC_MATRIX_33:
        ximea_param = "ccMTX33";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_DEFAULT_CC_MATRIX:
        ximea_param = "defccMTX";
        value_type = xiTypeCommand;
        break;
    case CV_CAP_PROP_XI_TRG_SOURCE:
        ximea_param = "trigger_source";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_TRG_SOFTWARE:
        ximea_param = "trigger_software";
        value_type = xiTypeCommand;
        break;
    case CV_CAP_PROP_XI_TRG_SELECTOR:
        ximea_param = "trigger_selector";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_ACQ_FRAME_BURST_COUNT:
        ximea_param = "acq_frame_burst_count";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_GPI_SELECTOR:
        ximea_param = "gpi_selector";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_GPI_MODE:
        ximea_param = "gpi_mode";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_GPI_LEVEL:
        ximea_param = "gpi_level";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_GPO_SELECTOR:
        ximea_param = "gpo_selector";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_GPO_MODE:
        ximea_param = "gpo_mode";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_LED_SELECTOR:
        ximea_param = "led_selector";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_LED_MODE:
        ximea_param = "led_mode";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_DEBOUNCE_EN:
        ximea_param = "dbnc_en";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_DEBOUNCE_T0:
        ximea_param = "dbnc_t0";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_DEBOUNCE_T1:
        ximea_param = "dbnc_t1";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_DEBOUNCE_POL:
        ximea_param = "dbnc_pol";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_LENS_MODE:
        ximea_param = "lens_mode";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_LENS_APERTURE_VALUE:
        ximea_param = "lens_aperture_value";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_LENS_FOCUS_MOVEMENT_VALUE:
        ximea_param = "lens_focus_movement_value";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_LENS_FOCUS_MOVE:
        ximea_param = "lens_focus_move";
        value_type = xiTypeCommand;
        break;
    case CV_CAP_PROP_XI_LENS_FOCUS_DISTANCE:
        ximea_param = "lens_focus_distance";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_LENS_FOCAL_LENGTH:
        ximea_param = "lens_focal_length";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_LENS_FEATURE_SELECTOR:
        ximea_param = "lens_feature_selector";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_LENS_FEATURE:
        ximea_param = "lens_feature";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_DEVICE_MODEL_ID:
        ximea_param = "device_model_id";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_DEVICE_SN:
        ximea_param = "device_sn";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_IMAGE_DATA_FORMAT_RGB32_ALPHA:
        ximea_param = "imgdataformatrgb32alpha";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_IMAGE_PAYLOAD_SIZE:
        ximea_param = "imgpayloadsize";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_TRANSPORT_PIXEL_FORMAT:
        ximea_param = "transport_pixel_format";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_SENSOR_CLOCK_FREQ_HZ:
        ximea_param = "sensor_clock_freq_hz";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_SENSOR_CLOCK_FREQ_INDEX:
        ximea_param = "sensor_clock_freq_index";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_SENSOR_OUTPUT_CHANNEL_COUNT:
        ximea_param = "sensor_output_channel_count";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_FRAMERATE:
        ximea_param = "framerate";
        value_type = xiTypeFloat;
        break;
    case CV_CAP_PROP_XI_COUNTER_SELECTOR:
        ximea_param = "counter_selector";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_COUNTER_VALUE:
        ximea_param = "counter_value";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_ACQ_TIMING_MODE:
        ximea_param = "acq_timing_mode";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_AVAILABLE_BANDWIDTH:
        ximea_param = "available_bandwidth";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_BUFFER_POLICY:
        ximea_param = "buffer_policy";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_LUT_EN:
        ximea_param = "LUTEnable";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_LUT_INDEX:
        ximea_param = "LUTIndex";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_LUT_VALUE:
        ximea_param = "LUTValue";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_TRG_DELAY:
        ximea_param = "trigger_delay";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_TS_RST_MODE:
        ximea_param = "ts_rst_mode";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_TS_RST_SOURCE:
        ximea_param = "ts_rst_source";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_IS_DEVICE_EXIST:
        ximea_param = "isexist";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_ACQ_BUFFER_SIZE:
        ximea_param = "acq_buffer_size";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_ACQ_BUFFER_SIZE_UNIT:
        ximea_param = "acq_buffer_size_unit";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_ACQ_TRANSPORT_BUFFER_SIZE:
        ximea_param = "acq_transport_buffer_size";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_BUFFERS_QUEUE_SIZE:
        ximea_param = "buffers_queue_size";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_ACQ_TRANSPORT_BUFFER_COMMIT:
        ximea_param = "acq_transport_buffer_commit";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_RECENT_FRAME:
        ximea_param = "recent_frame";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_DEVICE_RESET:
        ximea_param = "device_reset";
        value_type = xiTypeCommand;
        break;
    case CV_CAP_PROP_XI_COLUMN_FPN_CORRECTION:
        ximea_param = "column_fpn_correction";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_ROW_FPN_CORRECTION:
        ximea_param = "row_fpn_correction";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_SENSOR_MODE:
        ximea_param = "sensor_mode";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_HDR:
        ximea_param = "hdr";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_HDR_KNEEPOINT_COUNT:
        ximea_param = "hdr_kneepoint_count";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_HDR_T1:
        ximea_param = "hdr_t1";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_HDR_T2:
        ximea_param = "hdr_t2";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_KNEEPOINT1:
        ximea_param = "hdr_kneepoint1";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_KNEEPOINT2:
        ximea_param = "hdr_kneepoint2";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_IMAGE_BLACK_LEVEL:
        ximea_param = "image_black_level";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_HW_REVISION:
        ximea_param = "hw_revision";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_DEBUG_LEVEL:
        ximea_param = "debug_level";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_AUTO_BANDWIDTH_CALCULATION:
        ximea_param = "auto_bandwidth_calculation";
        value_type = xiTypeBoolean;
        break;
    case CV_CAP_PROP_XI_FFS_FILE_ID:
        ximea_param = "ffs_file_id";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_FFS_FILE_SIZE:
        ximea_param = "ffs_file_size";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_FREE_FFS_SIZE:
        ximea_param = "free_ffs_size";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_USED_FFS_SIZE:
        ximea_param = "used_ffs_size";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_FFS_ACCESS_KEY:
        ximea_param = "ffs_access_key";
        value_type = xiTypeInteger;
        break;
    case CV_CAP_PROP_XI_SENSOR_FEATURE_SELECTOR:
        ximea_param = "sensor_feature_selector";
        value_type = xiTypeEnum;
        break;
    case CV_CAP_PROP_XI_SENSOR_FEATURE_VALUE:
        ximea_param = "sensor_feature_value";
        value_type = xiTypeInteger;
        break;
    default:
        // report invalid parameter as it is not of string type
        errMsg("CvCaptureCAM_XIMEA::getProperty", XI_UNKNOWN_PARAM);
        return 0;
    }

    switch(value_type)
    {
    case xiTypeInteger :               // integer parameter type
    case xiTypeEnum :                // enumerator parameter type
    case xiTypeBoolean :             // boolean parameter type
    case xiTypeCommand :          // command parameter type
        stat = xiGetParamInt(hmv, ximea_param.c_str(), &ival);
        if(stat == XI_OK) getPropVal = ival;
        else errMsg("CvCaptureCAM_XIMEA::getProperty, xiGetParamInt", stat);
        break;
    case xiTypeFloat :                  // float parameter type
        stat = xiGetParamFloat(hmv, ximea_param.c_str(), &fval);
        if(stat == XI_OK) getPropVal = fval;
        else errMsg("CvCaptureCAM_XIMEA::getProperty, xiGetParamFloat", stat);
        break;
    default:
        // unknown value type selected
        errMsg("CvCaptureCAM_XIMEA::getProperty", XI_WRONG_PARAM_TYPE);
    }
    return getPropVal;
}

/**********************************************************************************/

void CvCaptureCAM_XIMEA::errMsg(const char* msg, int errNum) const
{
    // with XI_OK there is nothing to report
    if(errNum == XI_OK) return;
    string error_message = "";
    switch(errNum)
    {

    case XI_OK : error_message = "Function call succeeded"; break;
    case XI_INVALID_HANDLE : error_message = "Invalid handle"; break;
    case XI_READREG : error_message = "Register read error"; break;
    case XI_WRITEREG : error_message = "Register write error"; break;
    case XI_FREE_RESOURCES : error_message = "Freeing resiurces error"; break;
    case XI_FREE_CHANNEL : error_message = "Freeing channel error"; break;
    case XI_FREE_BANDWIDTH : error_message = "Freeing bandwith error"; break;
    case XI_READBLK : error_message = "Read block error"; break;
    case XI_WRITEBLK : error_message = "Write block error"; break;
    case XI_NO_IMAGE : error_message = "No image"; break;
    case XI_TIMEOUT : error_message = "Timeout"; break;
    case XI_INVALID_ARG : error_message = "Invalid arguments supplied"; break;
    case XI_NOT_SUPPORTED : error_message = "Not supported"; break;
    case XI_ISOCH_ATTACH_BUFFERS : error_message = "Attach buffers error"; break;
    case XI_GET_OVERLAPPED_RESULT : error_message = "Overlapped result"; break;
    case XI_MEMORY_ALLOCATION : error_message = "Memory allocation error"; break;
    case XI_DLLCONTEXTISNULL : error_message = "DLL context is NULL"; break;
    case XI_DLLCONTEXTISNONZERO : error_message = "DLL context is non zero"; break;
    case XI_DLLCONTEXTEXIST : error_message = "DLL context exists"; break;
    case XI_TOOMANYDEVICES : error_message = "Too many devices connected"; break;
    case XI_ERRORCAMCONTEXT : error_message = "Camera context error"; break;
    case XI_UNKNOWN_HARDWARE : error_message = "Unknown hardware"; break;
    case XI_INVALID_TM_FILE : error_message = "Invalid TM file"; break;
    case XI_INVALID_TM_TAG : error_message = "Invalid TM tag"; break;
    case XI_INCOMPLETE_TM : error_message = "Incomplete TM"; break;
    case XI_BUS_RESET_FAILED : error_message = "Bus reset error"; break;
    case XI_NOT_IMPLEMENTED : error_message = "Not implemented"; break;
    case XI_SHADING_TOOBRIGHT : error_message = "Shading too bright"; break;
    case XI_SHADING_TOODARK : error_message = "Shading too dark"; break;
    case XI_TOO_LOW_GAIN : error_message = "Gain is too low"; break;
    case XI_INVALID_BPL : error_message = "Invalid bad pixel list"; break;
    case XI_BPL_REALLOC : error_message = "Bad pixel list realloc error"; break;
    case XI_INVALID_PIXEL_LIST : error_message = "Invalid pixel list"; break;
    case XI_INVALID_FFS : error_message = "Invalid Flash File System"; break;
    case XI_INVALID_PROFILE : error_message = "Invalid profile"; break;
    case XI_INVALID_CALIBRATION : error_message = "Invalid calibration"; break;
    case XI_INVALID_BUFFER : error_message = "Invalid buffer"; break;
    case XI_INVALID_DATA : error_message = "Invalid data"; break;
    case XI_TGBUSY : error_message = "Timing generator is busy"; break;
    case XI_IO_WRONG : error_message = "Wrong operation open/write/read/close"; break;
    case XI_ACQUISITION_ALREADY_UP : error_message = "Acquisition already started"; break;
    case XI_OLD_DRIVER_VERSION : error_message = "Old version of device driver installed to the system."; break;
    case XI_GET_LAST_ERROR : error_message = "To get error code please call GetLastError function."; break;
    case XI_CANT_PROCESS : error_message = "Data cant be processed"; break;
    case XI_ACQUISITION_STOPED : error_message = "Acquisition has been stopped. It should be started before GetImage."; break;
    case XI_ACQUISITION_STOPED_WERR : error_message = "Acquisition has been stoped with error."; break;
    case XI_INVALID_INPUT_ICC_PROFILE : error_message = "Input ICC profile missed or corrupted"; break;
    case XI_INVALID_OUTPUT_ICC_PROFILE : error_message = "Output ICC profile missed or corrupted"; break;
    case XI_DEVICE_NOT_READY : error_message = "Device not ready to operate"; break;
    case XI_SHADING_TOOCONTRAST : error_message = "Shading too contrast"; break;
    case XI_ALREADY_INITIALIZED : error_message = "Module already initialized"; break;
    case XI_NOT_ENOUGH_PRIVILEGES : error_message = "Application doesnt enough privileges(one or more app"; break;
    case XI_NOT_COMPATIBLE_DRIVER : error_message = "Installed driver not compatible with current software"; break;
    case XI_TM_INVALID_RESOURCE : error_message = "TM file was not loaded successfully from resources"; break;
    case XI_DEVICE_HAS_BEEN_RESETED : error_message = "Device has been reseted, abnormal initial state"; break;
    case XI_NO_DEVICES_FOUND : error_message = "No Devices Found"; break;
    case XI_RESOURCE_OR_FUNCTION_LOCKED : error_message = "Resource(device) or function locked by mutex"; break;
    case XI_BUFFER_SIZE_TOO_SMALL : error_message = "Buffer provided by user is too small"; break;
    case XI_COULDNT_INIT_PROCESSOR : error_message = "Couldnt initialize processor."; break;
    case XI_NOT_INITIALIZED : error_message = "The object/module/procedure/process being referred to has not been started."; break;
    case XI_RESOURCE_NOT_FOUND : error_message = "Resource not found(could be processor, file, item..)."; break;
    case XI_UNKNOWN_PARAM : error_message = "Unknown parameter"; break;
    case XI_WRONG_PARAM_VALUE : error_message = "Wrong parameter value"; break;
    case XI_WRONG_PARAM_TYPE : error_message = "Wrong parameter type"; break;
    case XI_WRONG_PARAM_SIZE : error_message = "Wrong parameter size"; break;
    case XI_BUFFER_TOO_SMALL : error_message = "Input buffer too small"; break;
    case XI_NOT_SUPPORTED_PARAM : error_message = "Parameter info not supported"; break;
    case XI_NOT_SUPPORTED_PARAM_INFO : error_message = "Parameter info not supported"; break;
    case XI_NOT_SUPPORTED_DATA_FORMAT : error_message = "Data format not supported"; break;
    case XI_READ_ONLY_PARAM : error_message = "Read only parameter"; break;
    case XI_BANDWIDTH_NOT_SUPPORTED : error_message = "This camera does not support currently available bandwidth"; break;
    case XI_INVALID_FFS_FILE_NAME : error_message = "FFS file selector is invalid or NULL"; break;
    case XI_FFS_FILE_NOT_FOUND : error_message = "FFS file not found"; break;
    case XI_PROC_OTHER_ERROR : error_message = "Processing error - other"; break;
    case XI_PROC_PROCESSING_ERROR : error_message = "Error while image processing."; break;
    case XI_PROC_INPUT_FORMAT_UNSUPPORTED : error_message = "Input format is not supported for processing."; break;
    case XI_PROC_OUTPUT_FORMAT_UNSUPPORTED : error_message = "Output format is not supported for processing."; break;
    default:
        error_message = "Unknown error value";
    }

    #if defined _WIN32
    char buf[512]="";
    sprintf( buf, "%s : %d, %s\n", msg, errNum, error_message.c_str());
    OutputDebugString(buf);
    #else
    fprintf(stderr, "%s : %d, %s\n", msg, errNum, error_message.c_str());
    #endif
}

/**********************************************************************************/