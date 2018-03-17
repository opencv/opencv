/*
 * Copyright (c) 2015, Piotr Dobrowolski dobrypd[at]gmail[dot]com
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
 * THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "precomp.hpp"

#ifdef HAVE_GPHOTO2

#include <gphoto2/gphoto2.h>

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <exception>
#include <map>
#include <ostream>
#include <string>

namespace cv
{

namespace gphoto2 {

/**
 * \brief Map gPhoto2 return code into this exception.
 */
class GPhoto2Exception: public std::exception
{
private:
    int result;
    const char * method;
public:
    /**
     * @param methodStr libgphoto2 method name
     * @param gPhoto2Result libgphoto2 method result, should be less than GP_OK
     */
    GPhoto2Exception(const char * methodStr, int gPhoto2Result)
    {
        result = gPhoto2Result;
        method = methodStr;
    }
    virtual const char * what() const throw ()
    {
        return gp_result_as_string(result);
    }
    friend std::ostream & operator<<(std::ostream & ostream,
            GPhoto2Exception & e)
    {
        return ostream << e.method << ": " << e.what();
    }
};

/**
 * \brief Capture using your camera device via digital camera library - gPhoto2.
 *
 *  For library description and list of supported cameras, go to
 *  @url http://gphoto.sourceforge.net/
 *
 * Because gPhoto2 configuration is based on a widgets
 * and OpenCV CvCapture property settings are double typed
 * some assumptions and tricks has to be made.
 * 1. Device properties can be changed by IDs, use @method setProperty(int, double)
 *      and @method getProperty(int) with __additive inversed__
 *      camera setting ID as propertyId. (If you want to get camera setting
 *      with ID == x, you want to call #getProperty(-x)).
 * 2. Digital camera settings IDs are device dependent.
 * 3. You can list them by getting property CAP_PROP_GPHOTO2_WIDGET_ENUMERATE.
 * 3.1. As return you will get pointer to char array (with listed properties)
 *      instead of double. This list is in CSV type.
 * 4. There are several types of widgets (camera settings).
 * 4.1. For "menu" and "radio", you can get/set choice number.
 * 4.2. For "toggle" you can get/set int type.
 * 4.3. For "range" you can get/set float.
 * 4.4. For any other pointer will be fetched/set.
 * 5. You can fetch camera messages by using CAP_PROP_GPHOTO2_COLLECT_MSGS
 *      and CAP_PROP_GPHOTO2_FLUSH_MSGS (will return pointer to char array).
 * 6. Camera settings are fetched from device as lazy as possible.
 *      It creates problem with situation when change of one setting
 *      affects another setting. You can use CV_CAP_PROP_GPHOTO2_RELOAD_ON_CHANGE
 *      or CV_CAP_PROP_GPHOTO2_RELOAD_CONFIG to be sure that property you are
 *      planning to get will be actual.
 *
 * Capture can work in 2 main modes: preview and final.
 * Where preview is an output from digital camera "liveview".
 * Change modes with CAP_PROP_GPHOTO2_PREVIEW property.
 *
 * Moreover some generic properties are mapped to widgets, or implemented:
 *  * CV_CAP_PROP_SPEED,
 *  * CV_CAP_PROP_APERATURE,
 *  * CV_CAP_PROP_EXPOSUREPROGRAM,
 *  * CV_CAP_PROP_VIEWFINDER,
 *  * CV_CAP_PROP_POS_MSEC,
 *  * CV_CAP_PROP_POS_FRAMES,
 *  * CV_CAP_PROP_FRAME_WIDTH,
 *  * CV_CAP_PROP_FRAME_HEIGHT,
 *  * CV_CAP_PROP_FPS,
 *  * CV_CAP_PROP_FRAME_COUNT
 *  * CV_CAP_PROP_FORMAT,
 *  * CV_CAP_PROP_EXPOSURE,
 *  * CV_CAP_PROP_TRIGGER_DELAY,
 *  * CV_CAP_PROP_ZOOM,
 *  * CV_CAP_PROP_FOCUS,
 *  * CV_CAP_PROP_ISO_SPEED.
 */
class DigitalCameraCapture: public IVideoCapture
{
public:
    static const char * separator;
    static const char * lineDelimiter;

    DigitalCameraCapture();
    DigitalCameraCapture(int index);
    DigitalCameraCapture(const String &deviceName);
    virtual ~DigitalCameraCapture();

    virtual bool isOpened() const;
    virtual double getProperty(int) const;
    virtual bool setProperty(int, double);
    virtual bool grabFrame();
    virtual bool retrieveFrame(int, OutputArray);
    virtual int getCaptureDomain()
    {
        return CV_CAP_GPHOTO2;
    } // Return the type of the capture object: CV_CAP_VFW, etc...

    bool open(int index);
    void close();
    bool deviceExist(int index) const;
    int findDevice(const char * deviceName) const;

protected:
    // Known widget names
    static const char * PROP_EXPOSURE_COMPENSACTION;
    static const char * PROP_SELF_TIMER_DELAY;
    static const char * PROP_MANUALFOCUS;
    static const char * PROP_AUTOFOCUS;
    static const char * PROP_ISO;
    static const char * PROP_SPEED;
    static const char * PROP_APERTURE_NIKON;
    static const char * PROP_APERTURE_CANON;
    static const char * PROP_EXPOSURE_PROGRAM;
    static const char * PROP_VIEWFINDER;

    // Instance
    GPContext * context = NULL;
    int numDevices;
    void initContext();

    // Selected device
    bool opened;
    Camera * camera = NULL;
    Mat frame;

    // Properties
    CameraWidget * rootWidget = NULL;
    CameraWidget * getGenericProperty(int propertyId, double & output) const;
    CameraWidget * setGenericProperty(int propertyId, double value,
            bool & output) const;

    // Widgets
    void reloadConfig();
    CameraWidget * getWidget(int widgetId) const;
    CameraWidget * findWidgetByName(const char * name) const;

    // Loading
    void readFrameFromFile(CameraFile * file, OutputArray outputFrame);

    // Context feedback
    friend void ctxErrorFunc(GPContext *, const char *, void *);
    friend void ctxStatusFunc(GPContext *, const char *, void *);
    friend void ctxMessageFunc(GPContext *, const char *, void *);

    // Messages / debug
    enum MsgType
    {
        ERROR = (int) 'E',
        WARNING = (int) 'W',
        STATUS = (int) 'S',
        OTHER = (int) 'O'
    };
    template<typename OsstreamPrintable>
    void message(MsgType msgType, const char * msg,
            OsstreamPrintable & arg) const;

private:
    // Instance
    CameraAbilitiesList * abilitiesList = NULL;
    GPPortInfoList * capablePorts = NULL;
    CameraList * allDevices = NULL;

    // Selected device
    CameraAbilities cameraAbilities;
    std::deque<CameraFile *> grabbedFrames;

    // Properties
    bool preview; // CV_CAP_PROP_GPHOTO2_PREVIEW
    std::string widgetInfo; // CV_CAP_PROP_GPHOTO2_WIDGET_ENUMERATE
    std::map<int, CameraWidget *> widgets;
    bool reloadOnChange; // CV_CAP_PROP_GPHOTO2_RELOAD_ON_CHANGE
    time_t firstCapturedFrameTime;
    unsigned long int capturedFrames;

    DigitalCameraCapture(const DigitalCameraCapture&); // Disable copying
    DigitalCameraCapture& operator=(DigitalCameraCapture const&); // Disable assigning

    // Widgets
    int noOfWidgets;
    int widgetDescription(std::ostream &os, CameraWidget * widget) const;
    int collectWidgets(std::ostream &os, CameraWidget * widget);

    // Messages / debug
    mutable std::ostringstream msgsBuffer; // CV_CAP_PROP_GPHOTO2_FLUSH_MSGS
    mutable std::string lastFlush; // CV_CAP_PROP_GPHOTO2_FLUSH_MSGS
    bool collectMsgs; // CV_CAP_PROP_GPHOTO2_COLLECT_MSGS
};

/**
 * \brief Check if gPhoto2 function ends successfully. If not, throw an exception.
 */
#define CR(GPHOTO2_FUN) do {\
    int r_0629c47b758;\
    if ((r_0629c47b758 = (GPHOTO2_FUN)) < GP_OK) {\
        throw GPhoto2Exception(#GPHOTO2_FUN, r_0629c47b758);\
    };\
} while(0)

/**
 * \brief gPhoto2 context error feedback function.
 * @param thatGPhotoCap is required to be pointer to DigitalCameraCapture object.
 */
void ctxErrorFunc(GPContext *, const char * str, void * thatGPhotoCap)
{
    const DigitalCameraCapture * self =
            (const DigitalCameraCapture *) thatGPhotoCap;
    self->message(self->ERROR, "context feedback", str);
}

/**
 * \brief gPhoto2 context status feedback function.
 * @param thatGPhotoCap is required to be pointer to DigitalCameraCapture object.
 */
void ctxStatusFunc(GPContext *, const char * str, void * thatGPhotoCap)
{
    const DigitalCameraCapture * self =
            (const DigitalCameraCapture *) thatGPhotoCap;
    self->message(self->STATUS, "context feedback", str);
}

/**
 * \brief gPhoto2 context message feedback function.
 * @param thatGPhotoCap is required to be pointer to DigitalCameraCapture object.
 */
void ctxMessageFunc(GPContext *, const char * str, void * thatGPhotoCap)
{
    const DigitalCameraCapture * self =
            (const DigitalCameraCapture *) thatGPhotoCap;
    self->message(self->OTHER, "context feedback", str);
}

/**
 * \brief Separator used while creating CSV.
 */
const char * DigitalCameraCapture::separator = ",";
/**
 * \brief Line delimiter used while creating any readable output.
 */
const char * DigitalCameraCapture::lineDelimiter = "\n";
/**
 * \bief Some known widget names.
 *
 * Those are actually substrings of widget name.
 * ie. for VIEWFINDER, Nikon uses "viewfinder", while Canon can use "eosviewfinder".
 */
const char * DigitalCameraCapture::PROP_EXPOSURE_COMPENSACTION =
        "exposurecompensation";
const char * DigitalCameraCapture::PROP_SELF_TIMER_DELAY = "selftimerdelay";
const char * DigitalCameraCapture::PROP_MANUALFOCUS = "manualfocusdrive";
const char * DigitalCameraCapture::PROP_AUTOFOCUS = "autofocusdrive";
const char * DigitalCameraCapture::PROP_ISO = "iso";
const char * DigitalCameraCapture::PROP_SPEED = "shutterspeed";
const char * DigitalCameraCapture::PROP_APERTURE_NIKON = "f-number";
const char * DigitalCameraCapture::PROP_APERTURE_CANON = "aperture";
const char * DigitalCameraCapture::PROP_EXPOSURE_PROGRAM = "expprogram";
const char * DigitalCameraCapture::PROP_VIEWFINDER = "viewfinder";

/**
 * Initialize gPhoto2 context, search for all available devices.
 */
void DigitalCameraCapture::initContext()
{
    capturedFrames = noOfWidgets = numDevices = 0;
    opened = preview = reloadOnChange = false;
    firstCapturedFrameTime = 0;

    context = gp_context_new();

    gp_context_set_error_func(context, ctxErrorFunc, (void*) this);
    gp_context_set_status_func(context, ctxStatusFunc, (void*) this);
    gp_context_set_message_func(context, ctxMessageFunc, (void*) this);

    try
    {
        // Load abilities
        CR(gp_abilities_list_new(&abilitiesList));
        CR(gp_abilities_list_load(abilitiesList, context));

        // Load ports
        CR(gp_port_info_list_new(&capablePorts));
        CR(gp_port_info_list_load(capablePorts));

        // Auto-detect devices
        CR(gp_list_new(&allDevices));
        CR(gp_camera_autodetect(allDevices, context));
        CR(numDevices = gp_list_count(allDevices));
    }
    catch (GPhoto2Exception & e)
    {
        numDevices = 0;
    }
}

/**
 * Search for all devices while constructing.
 */
DigitalCameraCapture::DigitalCameraCapture()
{
    initContext();
}

/**
 * @see open(int)
 */
DigitalCameraCapture::DigitalCameraCapture(int index)
{
    initContext();
    if (deviceExist(index))
        open(index);
}

/**
 * @see findDevice(const char*)
 * @see open(int)
 */
DigitalCameraCapture::DigitalCameraCapture(const String & deviceName)
{
    initContext();
    int index = findDevice(deviceName.c_str());
    if (deviceExist(index))
        open(index);
}

/**
 * Always close connection to the device.
 */
DigitalCameraCapture::~DigitalCameraCapture()
{
    close();
    try
    {
        CR(gp_abilities_list_free(abilitiesList));
        abilitiesList = NULL;
        CR(gp_port_info_list_free(capablePorts));
        capablePorts = NULL;
        CR(gp_list_unref(allDevices));
        allDevices = NULL;
        gp_context_unref(context);
        context = NULL;
    }
    catch (GPhoto2Exception & e)
    {
        message(ERROR, "destruction error", e);
    }
}

/**
 * Connects to selected device.
 */
bool DigitalCameraCapture::open(int index)
{
    const char * model = 0, *path = 0;
    int m, p;
    GPPortInfo portInfo;

    if (isOpened()) {
        close();
    }

    try
    {
        CR(gp_camera_new(&camera));
        CR(gp_list_get_name(allDevices, index, &model));
        CR(gp_list_get_value(allDevices, index, &path));

        // Set model abilities.
        CR(m = gp_abilities_list_lookup_model(abilitiesList, model));
        CR(gp_abilities_list_get_abilities(abilitiesList, m, &cameraAbilities));
        CR(gp_camera_set_abilities(camera, cameraAbilities));

        // Set port
        CR(p = gp_port_info_list_lookup_path(capablePorts, path));
        CR(gp_port_info_list_get_info(capablePorts, p, &portInfo));
        CR(gp_camera_set_port_info(camera, portInfo));

        // Initialize connection to the camera.
        CR(gp_camera_init(camera, context));

        message(STATUS, "connected camera", model);
        message(STATUS, "connected using", path);

        // State initialization
        firstCapturedFrameTime = 0;
        capturedFrames = 0;
        preview = false;
        reloadOnChange = false;
        collectMsgs = false;

        reloadConfig();

        opened = true;
        return true;
    }
    catch (GPhoto2Exception & e)
    {
        message(WARNING, "opening device failed", e);
        return false;
    }
}

/**
 *
 */
bool DigitalCameraCapture::isOpened() const
{
    return opened;
}

/**
 * Close connection to the camera. Remove all unread frames/files.
 */
void DigitalCameraCapture::close()
{
    try
    {
        if (!frame.empty())
        {
            frame.release();
        }
        if (camera)
        {
            CR(gp_camera_exit(camera, context));
            CR(gp_camera_unref(camera));
            camera = NULL;
        }
        opened = false;
        if (int frames = grabbedFrames.size() > 0)
        {
            while (frames--)
            {
                CameraFile * file = grabbedFrames.front();
                grabbedFrames.pop_front();
                CR(gp_file_unref(file));
            }
        }
        if (rootWidget)
        {
            widgetInfo.clear();
            CR(gp_widget_unref(rootWidget));
            rootWidget = NULL;
        }
    }
    catch (GPhoto2Exception & e)
    {
        message(ERROR, "cannot close device properly", e);
    }
}

/**
 * @param output will be changed if possible, return 0 if changed,
 * @return widget, or NULL if output value was found (saved in argument),
 */
CameraWidget * DigitalCameraCapture::getGenericProperty(int propertyId,
        double & output) const
{
    switch (propertyId)
    {
        case CV_CAP_PROP_POS_MSEC:
        {
            // Only seconds level precision, FUTURE: cross-platform milliseconds
            output = (time(0) - firstCapturedFrameTime) * 1e2;
            return NULL;
        }
        case CV_CAP_PROP_POS_FRAMES:
        {
            output = capturedFrames;
            return NULL;
        }
        case CV_CAP_PROP_FRAME_WIDTH:
        {
            if (!frame.empty())
            {
                output = frame.cols;
            }
            return NULL;
        }
        case CV_CAP_PROP_FRAME_HEIGHT:
        {
            if (!frame.empty())
            {
                output = frame.rows;
            }
            return NULL;
        }
        case CV_CAP_PROP_FORMAT:
        {
            if (!frame.empty())
            {
                output = frame.type();
            }
            return NULL;
        }
        case CV_CAP_PROP_FPS: // returns average fps from the begin
        {
            double wholeProcessTime = 0;
            getGenericProperty(CV_CAP_PROP_POS_MSEC, wholeProcessTime);
            wholeProcessTime /= 1e2;
            output = capturedFrames / wholeProcessTime;
            return NULL;
        }
        case CV_CAP_PROP_FRAME_COUNT:
        {
            output = capturedFrames;
            return NULL;
        }
        case CV_CAP_PROP_EXPOSURE:
            return findWidgetByName(PROP_EXPOSURE_COMPENSACTION);
        case CV_CAP_PROP_TRIGGER_DELAY:
            return findWidgetByName(PROP_SELF_TIMER_DELAY);
        case CV_CAP_PROP_ZOOM:
            return findWidgetByName(PROP_MANUALFOCUS);
        case CV_CAP_PROP_FOCUS:
            return findWidgetByName(PROP_AUTOFOCUS);
        case CV_CAP_PROP_ISO_SPEED:
            return findWidgetByName(PROP_ISO);
        case CV_CAP_PROP_SPEED:
            return findWidgetByName(PROP_SPEED);
        case CV_CAP_PROP_APERTURE:
        {
            CameraWidget * widget = findWidgetByName(PROP_APERTURE_NIKON);
            return (widget == 0) ? findWidgetByName(PROP_APERTURE_CANON) : widget;
        }
        case CV_CAP_PROP_EXPOSUREPROGRAM:
            return findWidgetByName(PROP_EXPOSURE_PROGRAM);
        case CV_CAP_PROP_VIEWFINDER:
            return findWidgetByName(PROP_VIEWFINDER);
    }
    return NULL;
}

/**
 * Get property.
 * @see DigitalCameraCapture for more information about returned double type.
 */
double DigitalCameraCapture::getProperty(int propertyId) const
{
    CameraWidget * widget = NULL;
    double output = 0;
    if (propertyId < 0)
    {
        widget = getWidget(-propertyId);
    }
    else
    {
        switch (propertyId)
        {
            // gphoto2 cap featured
            case CV_CAP_PROP_GPHOTO2_PREVIEW:
                return preview;
            case CV_CAP_PROP_GPHOTO2_WIDGET_ENUMERATE:
                if (rootWidget == NULL)
                    return 0;
                return (intptr_t) widgetInfo.c_str();
            case CV_CAP_PROP_GPHOTO2_RELOAD_CONFIG:
                return 0; // Trigger, only by set
            case CV_CAP_PROP_GPHOTO2_RELOAD_ON_CHANGE:
                return reloadOnChange;
            case CV_CAP_PROP_GPHOTO2_COLLECT_MSGS:
                return collectMsgs;
            case CV_CAP_PROP_GPHOTO2_FLUSH_MSGS:
                lastFlush = msgsBuffer.str();
                msgsBuffer.str("");
                msgsBuffer.clear();
                return (intptr_t) lastFlush.c_str();
            default:
                widget = getGenericProperty(propertyId, output);
                /* no break */
        }
    }
    if (widget == NULL)
        return output;
    try
    {
        CameraWidgetType type;
        CR(gp_widget_get_type(widget, &type));
        switch (type)
        {
            case GP_WIDGET_MENU:
            case GP_WIDGET_RADIO:
            {
                int cnt = 0, i;
                const char * current;
                CR(gp_widget_get_value(widget, &current));
                CR(cnt = gp_widget_count_choices(widget));
                for (i = 0; i < cnt; i++)
                {
                    const char *choice;
                    CR(gp_widget_get_choice(widget, i, &choice));
                    if (std::strcmp(choice, current) == 0)
                    {
                        return i;
                    }
                }
                return -1;
            }
            case GP_WIDGET_TOGGLE:
            {
                int value;
                CR(gp_widget_get_value(widget, &value));
                return value;
            }
            case GP_WIDGET_RANGE:
            {
                float value;
                CR(gp_widget_get_value(widget, &value));
                return value;
            }
            default:
            {
                char* value;
                CR(gp_widget_get_value(widget, &value));
                return (intptr_t) value;
            }
        }
    }
    catch (GPhoto2Exception & e)
    {
        char buf[128] = "";
        sprintf(buf, "cannot get property: %d", propertyId);
        message(WARNING, (const char *) buf, e);
        return 0;
    }
}

/**
 * @param output will be changed if possible, return 0 if changed,
 * @return widget, or 0 if output value was found (saved in argument),
 */
CameraWidget * DigitalCameraCapture::setGenericProperty(int propertyId,
        double /*FUTURE: value*/, bool & output) const
{
    switch (propertyId)
    {
        case CV_CAP_PROP_POS_MSEC:
        case CV_CAP_PROP_POS_FRAMES:
        case CV_CAP_PROP_FRAME_WIDTH:
        case CV_CAP_PROP_FRAME_HEIGHT:
        case CV_CAP_PROP_FPS:
        case CV_CAP_PROP_FRAME_COUNT:
        case CV_CAP_PROP_FORMAT:
            output = false;
            return NULL;
        case CV_CAP_PROP_EXPOSURE:
            return findWidgetByName(PROP_EXPOSURE_COMPENSACTION);
        case CV_CAP_PROP_TRIGGER_DELAY:
            return findWidgetByName(PROP_SELF_TIMER_DELAY);
        case CV_CAP_PROP_ZOOM:
            return findWidgetByName(PROP_MANUALFOCUS);
        case CV_CAP_PROP_FOCUS:
            return findWidgetByName(PROP_AUTOFOCUS);
        case CV_CAP_PROP_ISO_SPEED:
            return findWidgetByName(PROP_ISO);
        case CV_CAP_PROP_SPEED:
            return findWidgetByName(PROP_SPEED);
        case CV_CAP_PROP_APERTURE:
        {
            CameraWidget * widget = findWidgetByName(PROP_APERTURE_NIKON);
            return (widget == NULL) ? findWidgetByName(PROP_APERTURE_CANON) : widget;
        }
        case CV_CAP_PROP_EXPOSUREPROGRAM:
            return findWidgetByName(PROP_EXPOSURE_PROGRAM);
        case CV_CAP_PROP_VIEWFINDER:
            return findWidgetByName(PROP_VIEWFINDER);
    }
    return NULL;
}

/**
 * Set property.
 * @see DigitalCameraCapture for more information about value, double typed, argument.
 */
bool DigitalCameraCapture::setProperty(int propertyId, double value)
{
    CameraWidget * widget = NULL;
    bool output = false;
    if (propertyId < 0)
    {
        widget = getWidget(-propertyId);
    }
    else
    {
        switch (propertyId)
        {
            // gphoto2 cap featured
            case CV_CAP_PROP_GPHOTO2_PREVIEW:
                preview = value != 0;
                return true;
            case CV_CAP_PROP_GPHOTO2_WIDGET_ENUMERATE:
                return false;
            case CV_CAP_PROP_GPHOTO2_RELOAD_CONFIG:
                reloadConfig();
                return true;
            case CV_CAP_PROP_GPHOTO2_RELOAD_ON_CHANGE:
                reloadOnChange = value != 0;
                return true;
            case CV_CAP_PROP_GPHOTO2_COLLECT_MSGS:
                collectMsgs = value != 0;
                return true;
            case CV_CAP_PROP_GPHOTO2_FLUSH_MSGS:
                return false;
            default:
                widget = setGenericProperty(propertyId, value, output);
                /* no break */
        }
    }
    if (widget == NULL)
        return output;
    try
    {
        CameraWidgetType type;
        CR(gp_widget_get_type(widget, &type));
        switch (type)
        {
            case GP_WIDGET_RADIO:
            case GP_WIDGET_MENU:
            {
                int i = static_cast<int>(value);
                char *choice;
                CR(gp_widget_get_choice(widget, i, (const char**)&choice));
                CR(gp_widget_set_value(widget, choice));
                break;
            }
            case GP_WIDGET_TOGGLE:
            {
                int i = static_cast<int>(value);
                CR(gp_widget_set_value(widget, &i));
                break;
            }
            case GP_WIDGET_RANGE:
            {
                float v = static_cast<float>(value);
                CR(gp_widget_set_value(widget, &v));
                break;
            }
            default:
            {
                CR(gp_widget_set_value(widget, (void* )(intptr_t )&value));
                break;
            }
        }
        if (!reloadOnChange)
        {
            // force widget change
            CR(gp_widget_set_changed(widget, 1));
        }

        // Use the same locale setting as while getting rootWidget.
        char * localeTmp = setlocale(LC_ALL, "C");
        CR(gp_camera_set_config(camera, rootWidget, context));
        setlocale(LC_ALL, localeTmp);

        if (reloadOnChange)
        {
            reloadConfig();
        } else {
            CR(gp_widget_set_changed(widget, 0));
        }
    }
    catch (GPhoto2Exception & e)
    {
        char buf[128] = "";
        sprintf(buf, "cannot set property: %d to %f", propertyId, value);
        message(WARNING, (const char *) buf, e);
        return false;
    }
    return true;
}

/**
 * Capture image, and store file in @field grabbedFrames.
 * Do not read a file. File will be deleted from camera automatically.
 */
bool DigitalCameraCapture::grabFrame()
{
    CameraFilePath filePath;
    CameraFile * file = NULL;
    try
    {
        CR(gp_file_new(&file));

        if (preview)
        {
            CR(gp_camera_capture_preview(camera, file, context));
        }
        else
        {
            // Capture an image
            CR(gp_camera_capture(camera, GP_CAPTURE_IMAGE, &filePath, context));
            CR(gp_camera_file_get(camera, filePath.folder, filePath.name, GP_FILE_TYPE_NORMAL,
                    file, context));
            CR(gp_camera_file_delete(camera, filePath.folder, filePath.name, context));
        }
        // State update
        if (firstCapturedFrameTime == 0)
        {
            firstCapturedFrameTime = time(0);
        }
        capturedFrames++;
        grabbedFrames.push_back(file);
    }
    catch (GPhoto2Exception & e)
    {
        if (file)
            gp_file_unref(file);
        message(WARNING, "cannot grab new frame", e);
        return false;
    }
    return true;
}

/**
 * Read stored file with image.
 */
bool DigitalCameraCapture::retrieveFrame(int, OutputArray outputFrame)
{
    if (grabbedFrames.size() > 0)
    {
        CameraFile * file = grabbedFrames.front();
        grabbedFrames.pop_front();
        try
        {
            readFrameFromFile(file, outputFrame);
            CR(gp_file_unref(file));
        }
        catch (GPhoto2Exception & e)
        {
            message(WARNING, "cannot read file grabbed from device", e);
            return false;
        }
    }
    else
    {
        return false;
    }
    return true;
}

/**
 * @return true if device exists
 */
bool DigitalCameraCapture::deviceExist(int index) const
{
    return (numDevices > 0) && (index < numDevices);
}

/**
 * @return device index if exists, otherwise -1
 */
int DigitalCameraCapture::findDevice(const char * deviceName) const
{
    const char * model = 0;
    try
    {
        if (deviceName != 0)
        {
            for (int i = 0; i < numDevices; ++i)
            {
                CR(gp_list_get_name(allDevices, i, &model));
                if (model != 0 && strstr(model, deviceName))
                {
                    return i;
                }
            }
        }
    }
    catch (GPhoto2Exception & e)
    {
        ; // pass
    }
    return -1;
}

/**
 * Load device settings.
 */
void DigitalCameraCapture::reloadConfig()
{
    std::ostringstream widgetInfoListStream;

    if (rootWidget != NULL)
    {
        widgetInfo.clear();
        CR(gp_widget_unref(rootWidget));
        rootWidget = NULL;
        widgets.clear();
    }
    // Make sure, that all configs (getting setting) will use the same locale setting.
    char * localeTmp = setlocale(LC_ALL, "C");
    CR(gp_camera_get_config(camera, &rootWidget, context));
    setlocale(LC_ALL, localeTmp);
    widgetInfoListStream << "id,label,name,info,readonly,type,value,"
            << lineDelimiter;
    noOfWidgets = collectWidgets(widgetInfoListStream, rootWidget) + 1;
    widgetInfo = widgetInfoListStream.str();
}

/**
 * Get widget which was fetched in time of last call to @reloadConfig().
 */
CameraWidget * DigitalCameraCapture::getWidget(int widgetId) const
{
    CameraWidget * widget;
    std::map<int, CameraWidget *>::const_iterator it = widgets.find(widgetId);
    if (it == widgets.end())
        return 0;
    widget = it->second;
    return widget;
}

/**
 * Search for widget with name which has @param subName substring.
 */
CameraWidget * DigitalCameraCapture::findWidgetByName(
        const char * subName) const
{
    if (subName != NULL)
    {
        try
        {
            const char * name;
            typedef std::map<int, CameraWidget *>::const_iterator it_t;
            it_t it = widgets.begin(), end = widgets.end();
            while (it != end)
            {
                CR(gp_widget_get_name(it->second, &name));
                if (strstr(name, subName))
                    break;
                ++it;
            }
            return (it != end) ? it->second : NULL;
        }
        catch (GPhoto2Exception & e)
        {
            message(WARNING, "error while searching for widget", e);
        }
    }
    return 0;
}

/**
 * Image file reader.
 *
 * @FUTURE: RAW format reader.
 */
void DigitalCameraCapture::readFrameFromFile(CameraFile * file, OutputArray outputFrame)

{
    // FUTURE: OpenCV cannot read RAW files right now.
    const char * data;
    unsigned long int size;
    CR(gp_file_get_data_and_size(file, &data, &size));
    if (size > 0)
    {
        Mat buf = Mat(1, size, CV_8UC1, (void *) data);
        if(!buf.empty())
        {
            frame = imdecode(buf, CV_LOAD_IMAGE_UNCHANGED);
        }
        frame.copyTo(outputFrame);
    }
}

/**
 * Print widget description in @param os.
 * @return real widget ID (if config was reloaded couple of times
 *         then IDs won't be the same)
 */
int DigitalCameraCapture::widgetDescription(std::ostream &os,
        CameraWidget * widget) const
{
    const char * label, *name, *info;
    int id, readonly;
    CameraWidgetType type;

    CR(gp_widget_get_id(widget, &id));
    CR(gp_widget_get_label(widget, &label));
    CR(gp_widget_get_name(widget, &name));
    CR(gp_widget_get_info(widget, &info));
    CR(gp_widget_get_type(widget, &type));
    CR(gp_widget_get_readonly(widget, &readonly));

    if ((type == GP_WIDGET_WINDOW) || (type == GP_WIDGET_SECTION)
            || (type == GP_WIDGET_BUTTON))
    {
        readonly = 1;
    }
    os << (id - noOfWidgets) << separator << label << separator << name
            << separator << info << separator << readonly << separator;

    switch (type)
    {
        case GP_WIDGET_WINDOW:
        {
            os << "window" << separator /* no value */<< separator;
            break;
        }
        case GP_WIDGET_SECTION:
        {
            os << "section" << separator /* no value */<< separator;
            break;
        }
        case GP_WIDGET_TEXT:
        {
            os << "text" << separator;
            char *txt;
            CR(gp_widget_get_value(widget, &txt));
            os << txt << separator;
            break;
        }
        case GP_WIDGET_RANGE:
        {
            os << "range" << separator;
            float f, t, b, s;
            CR(gp_widget_get_range(widget, &b, &t, &s));
            CR(gp_widget_get_value(widget, &f));
            os << "(" << b << ":" << t << ":" << s << "):" << f << separator;
            break;
        }
        case GP_WIDGET_TOGGLE:
        {
            os << "toggle" << separator;
            int t;
            CR(gp_widget_get_value(widget, &t));
            os << t << separator;
            break;
        }
        case GP_WIDGET_RADIO:
        case GP_WIDGET_MENU:
        {
            if (type == GP_WIDGET_RADIO)
            {
                os << "radio" << separator;
            }
            else
            {
                os << "menu" << separator;
            }
            int cnt = 0, i;
            char *current;
            CR(gp_widget_get_value(widget, &current));
            CR(cnt = gp_widget_count_choices(widget));
            os << "(";
            for (i = 0; i < cnt; i++)
            {
                const char *choice;
                CR(gp_widget_get_choice(widget, i, &choice));
                os << i << ":" << choice;
                if (i + 1 < cnt)
                {
                    os << ";";
                }
            }
            os << "):" << current << separator;
            break;
        }
        case GP_WIDGET_BUTTON:
        {
            os << "button" << separator /* no value */<< separator;
            break;
        }
        case GP_WIDGET_DATE:
        {
            os << "date" << separator;
            int t;
            time_t xtime;
            struct tm *xtm;
            char timebuf[200];
            CR(gp_widget_get_value(widget, &t));
            xtime = t;
            xtm = localtime(&xtime);
            strftime(timebuf, sizeof(timebuf), "%c", xtm);
            os << t << ":" << timebuf << separator;
            break;
        }
    }
    return id;
}

/**
 * Write all widget descriptions to @param os.
 * @return maximum of widget ID
 */
int DigitalCameraCapture::collectWidgets(std::ostream & os,
        CameraWidget * widget)
{
    int id = widgetDescription(os, widget);
    os << lineDelimiter;

    widgets[id - noOfWidgets] = widget;

    CameraWidget * child;
    CameraWidgetType type;
    CR(gp_widget_get_type(widget, &type));
    if ((type == GP_WIDGET_WINDOW) || (type == GP_WIDGET_SECTION))
    {
        for (int x = 0; x < gp_widget_count_children(widget); x++)
        {
            CR(gp_widget_get_child(widget, x, &child));
            id = std::max(id, collectWidgets(os, child));
        }
    }
    return id;
}

/**
 * Write message to @field msgsBuffer if user want to store them
 * (@field collectMsgs).
 * Print debug information on screen.
 */
template<typename OsstreamPrintable>
void DigitalCameraCapture::message(MsgType msgType, const char * msg,
        OsstreamPrintable & arg) const
{
#if defined(NDEBUG)
    if (collectMsgs)
    {
#endif
    std::ostringstream msgCreator;
    std::string out;
    char type = (char) msgType;
    msgCreator << "[gPhoto2][" << type << "]: " << msg << ": " << arg
            << lineDelimiter;
    out = msgCreator.str();
#if !defined(NDEBUG)
    if (collectMsgs)
    {
#endif
        msgsBuffer << out;
    }
#if !defined(NDEBUG)
#if defined(_WIN32)
    ::OutputDebugString(out.c_str());
#else
    fputs(out.c_str(), stderr);
#endif
#endif
}

} // namespace gphoto2

/**
 * \brief IVideoCapture creator form device index.
 */
Ptr<IVideoCapture> createGPhoto2Capture(int index)
{
    Ptr<IVideoCapture> capture = makePtr<gphoto2::DigitalCameraCapture>(index);

    if (capture->isOpened())
        return capture;

    return Ptr<gphoto2::DigitalCameraCapture>();
}

/**
 * IVideoCapture creator, from device name.
 *
 * @param deviceName is a substring in digital camera model name.
 */
Ptr<IVideoCapture> createGPhoto2Capture(const String & deviceName)
{
    Ptr<IVideoCapture> capture = makePtr<gphoto2::DigitalCameraCapture>(deviceName);

    if (capture->isOpened())
        return capture;

    return Ptr<gphoto2::DigitalCameraCapture>();
}

} // namespace cv

#endif
