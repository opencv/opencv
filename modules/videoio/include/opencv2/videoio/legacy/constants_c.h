// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_VIDEOIO_LEGACY_CONSTANTS_H
#define OPENCV_VIDEOIO_LEGACY_CONSTANTS_H

enum
{
    CV_CAP_ANY      =0,     // autodetect

    CV_CAP_MIL      =100,   // MIL proprietary drivers

    CV_CAP_VFW      =200,   // platform native
    CV_CAP_V4L      =200,
    CV_CAP_V4L2     =200,

    CV_CAP_FIREWARE =300,   // IEEE 1394 drivers
    CV_CAP_FIREWIRE =300,
    CV_CAP_IEEE1394 =300,
    CV_CAP_DC1394   =300,
    CV_CAP_CMU1394  =300,

    CV_CAP_STEREO   =400,   // TYZX proprietary drivers
    CV_CAP_TYZX     =400,
    CV_TYZX_LEFT    =400,
    CV_TYZX_RIGHT   =401,
    CV_TYZX_COLOR   =402,
    CV_TYZX_Z       =403,

    CV_CAP_QT       =500,   // QuickTime

    CV_CAP_UNICAP   =600,   // Unicap drivers

    CV_CAP_DSHOW    =700,   // DirectShow (via videoInput)
    CV_CAP_MSMF     =1400,  // Microsoft Media Foundation (via videoInput)

    CV_CAP_PVAPI    =800,   // PvAPI, Prosilica GigE SDK

    CV_CAP_OPENNI   =900,   // OpenNI (for Kinect)
    CV_CAP_OPENNI_ASUS =910,   // OpenNI (for Asus Xtion)

    CV_CAP_ANDROID  =1000,  // Android - not used
    CV_CAP_ANDROID_BACK =CV_CAP_ANDROID+99, // Android back camera - not used
    CV_CAP_ANDROID_FRONT =CV_CAP_ANDROID+98, // Android front camera - not used

    CV_CAP_XIAPI    =1100,   // XIMEA Camera API

    CV_CAP_AVFOUNDATION = 1200,  // AVFoundation framework for iOS (OS X Lion will have the same API)

    CV_CAP_GIGANETIX = 1300,  // Smartek Giganetix GigEVisionSDK

    CV_CAP_INTELPERC = 1500, // Intel Perceptual Computing

    CV_CAP_OPENNI2 = 1600,   // OpenNI2 (for Kinect)
    CV_CAP_GPHOTO2 = 1700,
    CV_CAP_GSTREAMER = 1800, // GStreamer
    CV_CAP_FFMPEG = 1900,    // FFMPEG
    CV_CAP_IMAGES = 2000,    // OpenCV Image Sequence (e.g. img_%02d.jpg)

    CV_CAP_ARAVIS = 2100     // Aravis GigE SDK
};

enum
{
    // modes of the controlling registers (can be: auto, manual, auto single push, absolute Latter allowed with any other mode)
    // every feature can have only one mode turned on at a time
    CV_CAP_PROP_DC1394_OFF         = -4,  //turn the feature off (not controlled manually nor automatically)
    CV_CAP_PROP_DC1394_MODE_MANUAL = -3, //set automatically when a value of the feature is set by the user
    CV_CAP_PROP_DC1394_MODE_AUTO = -2,
    CV_CAP_PROP_DC1394_MODE_ONE_PUSH_AUTO = -1,
    CV_CAP_PROP_POS_MSEC       =0,
    CV_CAP_PROP_POS_FRAMES     =1,
    CV_CAP_PROP_POS_AVI_RATIO  =2,
    CV_CAP_PROP_FRAME_WIDTH    =3,
    CV_CAP_PROP_FRAME_HEIGHT   =4,
    CV_CAP_PROP_FPS            =5,
    CV_CAP_PROP_FOURCC         =6,
    CV_CAP_PROP_FRAME_COUNT    =7,
    CV_CAP_PROP_FORMAT         =8,
    CV_CAP_PROP_MODE           =9,
    CV_CAP_PROP_BRIGHTNESS    =10,
    CV_CAP_PROP_CONTRAST      =11,
    CV_CAP_PROP_SATURATION    =12,
    CV_CAP_PROP_HUE           =13,
    CV_CAP_PROP_GAIN          =14,
    CV_CAP_PROP_EXPOSURE      =15,
    CV_CAP_PROP_CONVERT_RGB   =16,
    CV_CAP_PROP_WHITE_BALANCE_BLUE_U =17,
    CV_CAP_PROP_RECTIFICATION =18,
    CV_CAP_PROP_MONOCHROME    =19,
    CV_CAP_PROP_SHARPNESS     =20,
    CV_CAP_PROP_AUTO_EXPOSURE =21, // exposure control done by camera,
                                   // user can adjust reference level
                                   // using this feature
    CV_CAP_PROP_GAMMA         =22,
    CV_CAP_PROP_TEMPERATURE   =23,
    CV_CAP_PROP_TRIGGER       =24,
    CV_CAP_PROP_TRIGGER_DELAY =25,
    CV_CAP_PROP_WHITE_BALANCE_RED_V =26,
    CV_CAP_PROP_ZOOM          =27,
    CV_CAP_PROP_FOCUS         =28,
    CV_CAP_PROP_GUID          =29,
    CV_CAP_PROP_ISO_SPEED     =30,
    CV_CAP_PROP_MAX_DC1394    =31,
    CV_CAP_PROP_BACKLIGHT     =32,
    CV_CAP_PROP_PAN           =33,
    CV_CAP_PROP_TILT          =34,
    CV_CAP_PROP_ROLL          =35,
    CV_CAP_PROP_IRIS          =36,
    CV_CAP_PROP_SETTINGS      =37,
    CV_CAP_PROP_BUFFERSIZE    =38,
    CV_CAP_PROP_AUTOFOCUS     =39,
    CV_CAP_PROP_SAR_NUM       =40,
    CV_CAP_PROP_SAR_DEN       =41,

    CV_CAP_PROP_AUTOGRAB      =1024, // property for videoio class CvCapture_Android only
    CV_CAP_PROP_SUPPORTED_PREVIEW_SIZES_STRING=1025, // readonly, tricky property, returns cpnst char* indeed
    CV_CAP_PROP_PREVIEW_FORMAT=1026, // readonly, tricky property, returns cpnst char* indeed

    // OpenNI map generators
    CV_CAP_OPENNI_DEPTH_GENERATOR = 1 << 31,
    CV_CAP_OPENNI_IMAGE_GENERATOR = 1 << 30,
    CV_CAP_OPENNI_IR_GENERATOR    = 1 << 29,
    CV_CAP_OPENNI_GENERATORS_MASK = CV_CAP_OPENNI_DEPTH_GENERATOR + CV_CAP_OPENNI_IMAGE_GENERATOR + CV_CAP_OPENNI_IR_GENERATOR,

    // Properties of cameras available through OpenNI interfaces
    CV_CAP_PROP_OPENNI_OUTPUT_MODE     = 100,
    CV_CAP_PROP_OPENNI_FRAME_MAX_DEPTH = 101, // in mm
    CV_CAP_PROP_OPENNI_BASELINE        = 102, // in mm
    CV_CAP_PROP_OPENNI_FOCAL_LENGTH    = 103, // in pixels
    CV_CAP_PROP_OPENNI_REGISTRATION    = 104, // flag
    CV_CAP_PROP_OPENNI_REGISTRATION_ON = CV_CAP_PROP_OPENNI_REGISTRATION, // flag that synchronizes the remapping depth map to image map
                                                                          // by changing depth generator's view point (if the flag is "on") or
                                                                          // sets this view point to its normal one (if the flag is "off").
    CV_CAP_PROP_OPENNI_APPROX_FRAME_SYNC = 105,
    CV_CAP_PROP_OPENNI_MAX_BUFFER_SIZE   = 106,
    CV_CAP_PROP_OPENNI_CIRCLE_BUFFER     = 107,
    CV_CAP_PROP_OPENNI_MAX_TIME_DURATION = 108,

    CV_CAP_PROP_OPENNI_GENERATOR_PRESENT = 109,
    CV_CAP_PROP_OPENNI2_SYNC = 110,
    CV_CAP_PROP_OPENNI2_MIRROR = 111,

    CV_CAP_OPENNI_IMAGE_GENERATOR_PRESENT         = CV_CAP_OPENNI_IMAGE_GENERATOR + CV_CAP_PROP_OPENNI_GENERATOR_PRESENT,
    CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE     = CV_CAP_OPENNI_IMAGE_GENERATOR + CV_CAP_PROP_OPENNI_OUTPUT_MODE,
    CV_CAP_OPENNI_DEPTH_GENERATOR_PRESENT         = CV_CAP_OPENNI_DEPTH_GENERATOR + CV_CAP_PROP_OPENNI_GENERATOR_PRESENT,
    CV_CAP_OPENNI_DEPTH_GENERATOR_BASELINE        = CV_CAP_OPENNI_DEPTH_GENERATOR + CV_CAP_PROP_OPENNI_BASELINE,
    CV_CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH    = CV_CAP_OPENNI_DEPTH_GENERATOR + CV_CAP_PROP_OPENNI_FOCAL_LENGTH,
    CV_CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION    = CV_CAP_OPENNI_DEPTH_GENERATOR + CV_CAP_PROP_OPENNI_REGISTRATION,
    CV_CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION_ON = CV_CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION,
    CV_CAP_OPENNI_IR_GENERATOR_PRESENT            = CV_CAP_OPENNI_IR_GENERATOR + CV_CAP_PROP_OPENNI_GENERATOR_PRESENT,

    // Properties of cameras available through GStreamer interface
    CV_CAP_GSTREAMER_QUEUE_LENGTH           = 200, // default is 1

    // PVAPI
    CV_CAP_PROP_PVAPI_MULTICASTIP           = 300, // ip for anable multicast master mode. 0 for disable multicast
    CV_CAP_PROP_PVAPI_FRAMESTARTTRIGGERMODE = 301, // FrameStartTriggerMode: Determines how a frame is initiated
    CV_CAP_PROP_PVAPI_DECIMATIONHORIZONTAL  = 302, // Horizontal sub-sampling of the image
    CV_CAP_PROP_PVAPI_DECIMATIONVERTICAL    = 303, // Vertical sub-sampling of the image
    CV_CAP_PROP_PVAPI_BINNINGX              = 304, // Horizontal binning factor
    CV_CAP_PROP_PVAPI_BINNINGY              = 305, // Vertical binning factor
    CV_CAP_PROP_PVAPI_PIXELFORMAT           = 306, // Pixel format

    // Properties of cameras available through XIMEA SDK interface
    CV_CAP_PROP_XI_DOWNSAMPLING                                 = 400, // Change image resolution by binning or skipping.
    CV_CAP_PROP_XI_DATA_FORMAT                                  = 401, // Output data format.
    CV_CAP_PROP_XI_OFFSET_X                                     = 402, // Horizontal offset from the origin to the area of interest (in pixels).
    CV_CAP_PROP_XI_OFFSET_Y                                     = 403, // Vertical offset from the origin to the area of interest (in pixels).
    CV_CAP_PROP_XI_TRG_SOURCE                                   = 404, // Defines source of trigger.
    CV_CAP_PROP_XI_TRG_SOFTWARE                                 = 405, // Generates an internal trigger. PRM_TRG_SOURCE must be set to TRG_SOFTWARE.
    CV_CAP_PROP_XI_GPI_SELECTOR                                 = 406, // Selects general purpose input
    CV_CAP_PROP_XI_GPI_MODE                                     = 407, // Set general purpose input mode
    CV_CAP_PROP_XI_GPI_LEVEL                                    = 408, // Get general purpose level
    CV_CAP_PROP_XI_GPO_SELECTOR                                 = 409, // Selects general purpose output
    CV_CAP_PROP_XI_GPO_MODE                                     = 410, // Set general purpose output mode
    CV_CAP_PROP_XI_LED_SELECTOR                                 = 411, // Selects camera signalling LED
    CV_CAP_PROP_XI_LED_MODE                                     = 412, // Define camera signalling LED functionality
    CV_CAP_PROP_XI_MANUAL_WB                                    = 413, // Calculates White Balance(must be called during acquisition)
    CV_CAP_PROP_XI_AUTO_WB                                      = 414, // Automatic white balance
    CV_CAP_PROP_XI_AEAG                                         = 415, // Automatic exposure/gain
    CV_CAP_PROP_XI_EXP_PRIORITY                                 = 416, // Exposure priority (0.5 - exposure 50%, gain 50%).
    CV_CAP_PROP_XI_AE_MAX_LIMIT                                 = 417, // Maximum limit of exposure in AEAG procedure
    CV_CAP_PROP_XI_AG_MAX_LIMIT                                 = 418,  // Maximum limit of gain in AEAG procedure
    CV_CAP_PROP_XI_AEAG_LEVEL                                   = 419, // Average intensity of output signal AEAG should achieve(in %)
    CV_CAP_PROP_XI_TIMEOUT                                      = 420, // Image capture timeout in milliseconds
    CV_CAP_PROP_XI_EXPOSURE                                     = 421, // Exposure time in microseconds
    CV_CAP_PROP_XI_EXPOSURE_BURST_COUNT                         = 422, // Sets the number of times of exposure in one frame.
    CV_CAP_PROP_XI_GAIN_SELECTOR                                = 423, // Gain selector for parameter Gain allows to select different type of gains.
    CV_CAP_PROP_XI_GAIN                                         = 424, // Gain in dB
    CV_CAP_PROP_XI_DOWNSAMPLING_TYPE                            = 426, // Change image downsampling type.
    CV_CAP_PROP_XI_BINNING_SELECTOR                             = 427, // Binning engine selector.
    CV_CAP_PROP_XI_BINNING_VERTICAL                             = 428, // Vertical Binning - number of vertical photo-sensitive cells to combine together.
    CV_CAP_PROP_XI_BINNING_HORIZONTAL                           = 429, // Horizontal Binning - number of horizontal photo-sensitive cells to combine together.
    CV_CAP_PROP_XI_BINNING_PATTERN                              = 430, // Binning pattern type.
    CV_CAP_PROP_XI_DECIMATION_SELECTOR                          = 431, // Decimation engine selector.
    CV_CAP_PROP_XI_DECIMATION_VERTICAL                          = 432, // Vertical Decimation - vertical sub-sampling of the image - reduces the vertical resolution of the image by the specified vertical decimation factor.
    CV_CAP_PROP_XI_DECIMATION_HORIZONTAL                        = 433, // Horizontal Decimation - horizontal sub-sampling of the image - reduces the horizontal resolution of the image by the specified vertical decimation factor.
    CV_CAP_PROP_XI_DECIMATION_PATTERN                           = 434, // Decimation pattern type.
    CV_CAP_PROP_XI_TEST_PATTERN_GENERATOR_SELECTOR              = 587, // Selects which test pattern generator is controlled by the TestPattern feature.
    CV_CAP_PROP_XI_TEST_PATTERN                                 = 588, // Selects which test pattern type is generated by the selected generator.
    CV_CAP_PROP_XI_IMAGE_DATA_FORMAT                            = 435, // Output data format.
    CV_CAP_PROP_XI_SHUTTER_TYPE                                 = 436, // Change sensor shutter type(CMOS sensor).
    CV_CAP_PROP_XI_SENSOR_TAPS                                  = 437, // Number of taps
    CV_CAP_PROP_XI_AEAG_ROI_OFFSET_X                            = 439, // Automatic exposure/gain ROI offset X
    CV_CAP_PROP_XI_AEAG_ROI_OFFSET_Y                            = 440, // Automatic exposure/gain ROI offset Y
    CV_CAP_PROP_XI_AEAG_ROI_WIDTH                               = 441, // Automatic exposure/gain ROI Width
    CV_CAP_PROP_XI_AEAG_ROI_HEIGHT                              = 442, // Automatic exposure/gain ROI Height
    CV_CAP_PROP_XI_BPC                                          = 445, // Correction of bad pixels
    CV_CAP_PROP_XI_WB_KR                                        = 448, // White balance red coefficient
    CV_CAP_PROP_XI_WB_KG                                        = 449, // White balance green coefficient
    CV_CAP_PROP_XI_WB_KB                                        = 450, // White balance blue coefficient
    CV_CAP_PROP_XI_WIDTH                                        = 451, // Width of the Image provided by the device (in pixels).
    CV_CAP_PROP_XI_HEIGHT                                       = 452, // Height of the Image provided by the device (in pixels).
    CV_CAP_PROP_XI_REGION_SELECTOR                              = 589, // Selects Region in Multiple ROI which parameters are set by width, height, ... ,region mode
    CV_CAP_PROP_XI_REGION_MODE                                  = 595, // Activates/deactivates Region selected by Region Selector
    CV_CAP_PROP_XI_LIMIT_BANDWIDTH                              = 459, // Set/get bandwidth(datarate)(in Megabits)
    CV_CAP_PROP_XI_SENSOR_DATA_BIT_DEPTH                        = 460, // Sensor output data bit depth.
    CV_CAP_PROP_XI_OUTPUT_DATA_BIT_DEPTH                        = 461, // Device output data bit depth.
    CV_CAP_PROP_XI_IMAGE_DATA_BIT_DEPTH                         = 462, // bitdepth of data returned by function xiGetImage
    CV_CAP_PROP_XI_OUTPUT_DATA_PACKING                          = 463, // Device output data packing (or grouping) enabled. Packing could be enabled if output_data_bit_depth > 8 and packing capability is available.
    CV_CAP_PROP_XI_OUTPUT_DATA_PACKING_TYPE                     = 464, // Data packing type. Some cameras supports only specific packing type.
    CV_CAP_PROP_XI_IS_COOLED                                    = 465, // Returns 1 for cameras that support cooling.
    CV_CAP_PROP_XI_COOLING                                      = 466, // Start camera cooling.
    CV_CAP_PROP_XI_TARGET_TEMP                                  = 467, // Set sensor target temperature for cooling.
    CV_CAP_PROP_XI_CHIP_TEMP                                    = 468, // Camera sensor temperature
    CV_CAP_PROP_XI_HOUS_TEMP                                    = 469, // Camera housing temperature
    CV_CAP_PROP_XI_HOUS_BACK_SIDE_TEMP                          = 590, // Camera housing back side temperature
    CV_CAP_PROP_XI_SENSOR_BOARD_TEMP                            = 596, // Camera sensor board temperature
    CV_CAP_PROP_XI_CMS                                          = 470, // Mode of color management system.
    CV_CAP_PROP_XI_APPLY_CMS                                    = 471, // Enable applying of CMS profiles to xiGetImage (see XI_PRM_INPUT_CMS_PROFILE, XI_PRM_OUTPUT_CMS_PROFILE).
    CV_CAP_PROP_XI_IMAGE_IS_COLOR                               = 474, // Returns 1 for color cameras.
    CV_CAP_PROP_XI_COLOR_FILTER_ARRAY                           = 475, // Returns color filter array type of RAW data.
    CV_CAP_PROP_XI_GAMMAY                                       = 476, // Luminosity gamma
    CV_CAP_PROP_XI_GAMMAC                                       = 477, // Chromaticity gamma
    CV_CAP_PROP_XI_SHARPNESS                                    = 478, // Sharpness Strength
    CV_CAP_PROP_XI_CC_MATRIX_00                                 = 479, // Color Correction Matrix element [0][0]
    CV_CAP_PROP_XI_CC_MATRIX_01                                 = 480, // Color Correction Matrix element [0][1]
    CV_CAP_PROP_XI_CC_MATRIX_02                                 = 481, // Color Correction Matrix element [0][2]
    CV_CAP_PROP_XI_CC_MATRIX_03                                 = 482, // Color Correction Matrix element [0][3]
    CV_CAP_PROP_XI_CC_MATRIX_10                                 = 483, // Color Correction Matrix element [1][0]
    CV_CAP_PROP_XI_CC_MATRIX_11                                 = 484, // Color Correction Matrix element [1][1]
    CV_CAP_PROP_XI_CC_MATRIX_12                                 = 485, // Color Correction Matrix element [1][2]
    CV_CAP_PROP_XI_CC_MATRIX_13                                 = 486, // Color Correction Matrix element [1][3]
    CV_CAP_PROP_XI_CC_MATRIX_20                                 = 487, // Color Correction Matrix element [2][0]
    CV_CAP_PROP_XI_CC_MATRIX_21                                 = 488, // Color Correction Matrix element [2][1]
    CV_CAP_PROP_XI_CC_MATRIX_22                                 = 489, // Color Correction Matrix element [2][2]
    CV_CAP_PROP_XI_CC_MATRIX_23                                 = 490, // Color Correction Matrix element [2][3]
    CV_CAP_PROP_XI_CC_MATRIX_30                                 = 491, // Color Correction Matrix element [3][0]
    CV_CAP_PROP_XI_CC_MATRIX_31                                 = 492, // Color Correction Matrix element [3][1]
    CV_CAP_PROP_XI_CC_MATRIX_32                                 = 493, // Color Correction Matrix element [3][2]
    CV_CAP_PROP_XI_CC_MATRIX_33                                 = 494, // Color Correction Matrix element [3][3]
    CV_CAP_PROP_XI_DEFAULT_CC_MATRIX                            = 495, // Set default Color Correction Matrix
    CV_CAP_PROP_XI_TRG_SELECTOR                                 = 498, // Selects the type of trigger.
    CV_CAP_PROP_XI_ACQ_FRAME_BURST_COUNT                        = 499, // Sets number of frames acquired by burst. This burst is used only if trigger is set to FrameBurstStart
    CV_CAP_PROP_XI_DEBOUNCE_EN                                  = 507, // Enable/Disable debounce to selected GPI
    CV_CAP_PROP_XI_DEBOUNCE_T0                                  = 508, // Debounce time (x * 10us)
    CV_CAP_PROP_XI_DEBOUNCE_T1                                  = 509, // Debounce time (x * 10us)
    CV_CAP_PROP_XI_DEBOUNCE_POL                                 = 510, // Debounce polarity (pol = 1 t0 - falling edge, t1 - rising edge)
    CV_CAP_PROP_XI_LENS_MODE                                    = 511, // Status of lens control interface. This shall be set to XI_ON before any Lens operations.
    CV_CAP_PROP_XI_LENS_APERTURE_VALUE                          = 512, // Current lens aperture value in stops. Examples: 2.8, 4, 5.6, 8, 11
    CV_CAP_PROP_XI_LENS_FOCUS_MOVEMENT_VALUE                    = 513, // Lens current focus movement value to be used by XI_PRM_LENS_FOCUS_MOVE in motor steps.
    CV_CAP_PROP_XI_LENS_FOCUS_MOVE                              = 514, // Moves lens focus motor by steps set in XI_PRM_LENS_FOCUS_MOVEMENT_VALUE.
    CV_CAP_PROP_XI_LENS_FOCUS_DISTANCE                          = 515, // Lens focus distance in cm.
    CV_CAP_PROP_XI_LENS_FOCAL_LENGTH                            = 516, // Lens focal distance in mm.
    CV_CAP_PROP_XI_LENS_FEATURE_SELECTOR                        = 517, // Selects the current feature which is accessible by XI_PRM_LENS_FEATURE.
    CV_CAP_PROP_XI_LENS_FEATURE                                 = 518, // Allows access to lens feature value currently selected by XI_PRM_LENS_FEATURE_SELECTOR.
    CV_CAP_PROP_XI_DEVICE_MODEL_ID                              = 521, // Return device model id
    CV_CAP_PROP_XI_DEVICE_SN                                    = 522, // Return device serial number
    CV_CAP_PROP_XI_IMAGE_DATA_FORMAT_RGB32_ALPHA                = 529, // The alpha channel of RGB32 output image format.
    CV_CAP_PROP_XI_IMAGE_PAYLOAD_SIZE                           = 530, // Buffer size in bytes sufficient for output image returned by xiGetImage
    CV_CAP_PROP_XI_TRANSPORT_PIXEL_FORMAT                       = 531, // Current format of pixels on transport layer.
    CV_CAP_PROP_XI_SENSOR_CLOCK_FREQ_HZ                         = 532, // Sensor clock frequency in Hz.
    CV_CAP_PROP_XI_SENSOR_CLOCK_FREQ_INDEX                      = 533, // Sensor clock frequency index. Sensor with selected frequencies have possibility to set the frequency only by this index.
    CV_CAP_PROP_XI_SENSOR_OUTPUT_CHANNEL_COUNT                  = 534, // Number of output channels from sensor used for data transfer.
    CV_CAP_PROP_XI_FRAMERATE                                    = 535, // Define framerate in Hz
    CV_CAP_PROP_XI_COUNTER_SELECTOR                             = 536, // Select counter
    CV_CAP_PROP_XI_COUNTER_VALUE                                = 537, // Counter status
    CV_CAP_PROP_XI_ACQ_TIMING_MODE                              = 538, // Type of sensor frames timing.
    CV_CAP_PROP_XI_AVAILABLE_BANDWIDTH                          = 539, // Calculate and return available interface bandwidth(int Megabits)
    CV_CAP_PROP_XI_BUFFER_POLICY                                = 540, // Data move policy
    CV_CAP_PROP_XI_LUT_EN                                       = 541, // Activates LUT.
    CV_CAP_PROP_XI_LUT_INDEX                                    = 542, // Control the index (offset) of the coefficient to access in the LUT.
    CV_CAP_PROP_XI_LUT_VALUE                                    = 543, // Value at entry LUTIndex of the LUT
    CV_CAP_PROP_XI_TRG_DELAY                                    = 544, // Specifies the delay in microseconds (us) to apply after the trigger reception before activating it.
    CV_CAP_PROP_XI_TS_RST_MODE                                  = 545, // Defines how time stamp reset engine will be armed
    CV_CAP_PROP_XI_TS_RST_SOURCE                                = 546, // Defines which source will be used for timestamp reset. Writing this parameter will trigger settings of engine (arming)
    CV_CAP_PROP_XI_IS_DEVICE_EXIST                              = 547, // Returns 1 if camera connected and works properly.
    CV_CAP_PROP_XI_ACQ_BUFFER_SIZE                              = 548, // Acquisition buffer size in buffer_size_unit. Default bytes.
    CV_CAP_PROP_XI_ACQ_BUFFER_SIZE_UNIT                         = 549, // Acquisition buffer size unit in bytes. Default 1. E.g. Value 1024 means that buffer_size is in KiBytes
    CV_CAP_PROP_XI_ACQ_TRANSPORT_BUFFER_SIZE                    = 550, // Acquisition transport buffer size in bytes
    CV_CAP_PROP_XI_BUFFERS_QUEUE_SIZE                           = 551, // Queue of field/frame buffers
    CV_CAP_PROP_XI_ACQ_TRANSPORT_BUFFER_COMMIT                  = 552, // Number of buffers to commit to low level
    CV_CAP_PROP_XI_RECENT_FRAME                                 = 553, // GetImage returns most recent frame
    CV_CAP_PROP_XI_DEVICE_RESET                                 = 554, // Resets the camera to default state.
    CV_CAP_PROP_XI_COLUMN_FPN_CORRECTION                        = 555, // Correction of column FPN
    CV_CAP_PROP_XI_ROW_FPN_CORRECTION                           = 591, // Correction of row FPN
    CV_CAP_PROP_XI_SENSOR_MODE                                  = 558, // Current sensor mode. Allows to select sensor mode by one integer. Setting of this parameter affects: image dimensions and downsampling.
    CV_CAP_PROP_XI_HDR                                          = 559, // Enable High Dynamic Range feature.
    CV_CAP_PROP_XI_HDR_KNEEPOINT_COUNT                          = 560, // The number of kneepoints in the PWLR.
    CV_CAP_PROP_XI_HDR_T1                                       = 561, // position of first kneepoint(in % of XI_PRM_EXPOSURE)
    CV_CAP_PROP_XI_HDR_T2                                       = 562, // position of second kneepoint (in % of XI_PRM_EXPOSURE)
    CV_CAP_PROP_XI_KNEEPOINT1                                   = 563, // value of first kneepoint (% of sensor saturation)
    CV_CAP_PROP_XI_KNEEPOINT2                                   = 564, // value of second kneepoint (% of sensor saturation)
    CV_CAP_PROP_XI_IMAGE_BLACK_LEVEL                            = 565, // Last image black level counts. Can be used for Offline processing to recall it.
    CV_CAP_PROP_XI_HW_REVISION                                  = 571, // Returns hardware revision number.
    CV_CAP_PROP_XI_DEBUG_LEVEL                                  = 572, // Set debug level
    CV_CAP_PROP_XI_AUTO_BANDWIDTH_CALCULATION                   = 573, // Automatic bandwidth calculation,
    CV_CAP_PROP_XI_FFS_FILE_ID                                  = 594, // File number.
    CV_CAP_PROP_XI_FFS_FILE_SIZE                                = 580, // Size of file.
    CV_CAP_PROP_XI_FREE_FFS_SIZE                                = 581, // Size of free camera FFS.
    CV_CAP_PROP_XI_USED_FFS_SIZE                                = 582, // Size of used camera FFS.
    CV_CAP_PROP_XI_FFS_ACCESS_KEY                               = 583, // Setting of key enables file operations on some cameras.
    CV_CAP_PROP_XI_SENSOR_FEATURE_SELECTOR                      = 585, // Selects the current feature which is accessible by XI_PRM_SENSOR_FEATURE_VALUE.
    CV_CAP_PROP_XI_SENSOR_FEATURE_VALUE                         = 586, // Allows access to sensor feature value currently selected by XI_PRM_SENSOR_FEATURE_SELECTOR.


    // Properties for Android cameras
    CV_CAP_PROP_ANDROID_FLASH_MODE = 8001,
    CV_CAP_PROP_ANDROID_FOCUS_MODE = 8002,
    CV_CAP_PROP_ANDROID_WHITE_BALANCE = 8003,
    CV_CAP_PROP_ANDROID_ANTIBANDING = 8004,
    CV_CAP_PROP_ANDROID_FOCAL_LENGTH = 8005,
    CV_CAP_PROP_ANDROID_FOCUS_DISTANCE_NEAR = 8006,
    CV_CAP_PROP_ANDROID_FOCUS_DISTANCE_OPTIMAL = 8007,
    CV_CAP_PROP_ANDROID_FOCUS_DISTANCE_FAR = 8008,
    CV_CAP_PROP_ANDROID_EXPOSE_LOCK = 8009,
    CV_CAP_PROP_ANDROID_WHITEBALANCE_LOCK = 8010,

    // Properties of cameras available through AVFOUNDATION interface
    CV_CAP_PROP_IOS_DEVICE_FOCUS = 9001,
    CV_CAP_PROP_IOS_DEVICE_EXPOSURE = 9002,
    CV_CAP_PROP_IOS_DEVICE_FLASH = 9003,
    CV_CAP_PROP_IOS_DEVICE_WHITEBALANCE = 9004,
    CV_CAP_PROP_IOS_DEVICE_TORCH = 9005,

    // Properties of cameras available through Smartek Giganetix Ethernet Vision interface
    /* --- Vladimir Litvinenko (litvinenko.vladimir@gmail.com) --- */
    CV_CAP_PROP_GIGA_FRAME_OFFSET_X = 10001,
    CV_CAP_PROP_GIGA_FRAME_OFFSET_Y = 10002,
    CV_CAP_PROP_GIGA_FRAME_WIDTH_MAX = 10003,
    CV_CAP_PROP_GIGA_FRAME_HEIGH_MAX = 10004,
    CV_CAP_PROP_GIGA_FRAME_SENS_WIDTH = 10005,
    CV_CAP_PROP_GIGA_FRAME_SENS_HEIGH = 10006,

    CV_CAP_PROP_INTELPERC_PROFILE_COUNT               = 11001,
    CV_CAP_PROP_INTELPERC_PROFILE_IDX                 = 11002,
    CV_CAP_PROP_INTELPERC_DEPTH_LOW_CONFIDENCE_VALUE  = 11003,
    CV_CAP_PROP_INTELPERC_DEPTH_SATURATION_VALUE      = 11004,
    CV_CAP_PROP_INTELPERC_DEPTH_CONFIDENCE_THRESHOLD  = 11005,
    CV_CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_HORZ     = 11006,
    CV_CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_VERT     = 11007,

    // Intel PerC streams
    CV_CAP_INTELPERC_DEPTH_GENERATOR = 1 << 29,
    CV_CAP_INTELPERC_IMAGE_GENERATOR = 1 << 28,
    CV_CAP_INTELPERC_GENERATORS_MASK = CV_CAP_INTELPERC_DEPTH_GENERATOR + CV_CAP_INTELPERC_IMAGE_GENERATOR
};

enum
{
    // Data given from depth generator.
    CV_CAP_OPENNI_DEPTH_MAP                 = 0, // Depth values in mm (CV_16UC1)
    CV_CAP_OPENNI_POINT_CLOUD_MAP           = 1, // XYZ in meters (CV_32FC3)
    CV_CAP_OPENNI_DISPARITY_MAP             = 2, // Disparity in pixels (CV_8UC1)
    CV_CAP_OPENNI_DISPARITY_MAP_32F         = 3, // Disparity in pixels (CV_32FC1)
    CV_CAP_OPENNI_VALID_DEPTH_MASK          = 4, // CV_8UC1

    // Data given from RGB image generator.
    CV_CAP_OPENNI_BGR_IMAGE                 = 5,
    CV_CAP_OPENNI_GRAY_IMAGE                = 6,

    // Data given from IR image generator.
    CV_CAP_OPENNI_IR_IMAGE                  = 7
};

// Supported output modes of OpenNI image generator
enum
{
    CV_CAP_OPENNI_VGA_30HZ     = 0,
    CV_CAP_OPENNI_SXGA_15HZ    = 1,
    CV_CAP_OPENNI_SXGA_30HZ    = 2,
    CV_CAP_OPENNI_QVGA_30HZ    = 3,
    CV_CAP_OPENNI_QVGA_60HZ    = 4
};

enum
{
    CV_CAP_INTELPERC_DEPTH_MAP              = 0, // Each pixel is a 16-bit integer. The value indicates the distance from an object to the camera's XY plane or the Cartesian depth.
    CV_CAP_INTELPERC_UVDEPTH_MAP            = 1, // Each pixel contains two 32-bit floating point values in the range of 0-1, representing the mapping of depth coordinates to the color coordinates.
    CV_CAP_INTELPERC_IR_MAP                 = 2, // Each pixel is a 16-bit integer. The value indicates the intensity of the reflected laser beam.
    CV_CAP_INTELPERC_IMAGE                  = 3
};

// gPhoto2 properties, if propertyId is less than 0 then work on widget with that __additive inversed__ camera setting ID
// Get IDs by using CAP_PROP_GPHOTO2_WIDGET_ENUMERATE.
// @see CvCaptureCAM_GPHOTO2 for more info
enum
{
    CV_CAP_PROP_GPHOTO2_PREVIEW           = 17001, // Capture only preview from liveview mode.
    CV_CAP_PROP_GPHOTO2_WIDGET_ENUMERATE  = 17002, // Readonly, returns (const char *).
    CV_CAP_PROP_GPHOTO2_RELOAD_CONFIG     = 17003, // Trigger, only by set. Reload camera settings.
    CV_CAP_PROP_GPHOTO2_RELOAD_ON_CHANGE  = 17004, // Reload all settings on set.
    CV_CAP_PROP_GPHOTO2_COLLECT_MSGS      = 17005, // Collect messages with details.
    CV_CAP_PROP_GPHOTO2_FLUSH_MSGS        = 17006, // Readonly, returns (const char *).
    CV_CAP_PROP_SPEED                     = 17007, // Exposure speed. Can be readonly, depends on camera program.
    CV_CAP_PROP_APERTURE                  = 17008, // Aperture. Can be readonly, depends on camera program.
    CV_CAP_PROP_EXPOSUREPROGRAM           = 17009, // Camera exposure program.
    CV_CAP_PROP_VIEWFINDER                = 17010  // Enter liveview mode.
};

//! Macro to construct the fourcc code of the codec. Same as CV_FOURCC()
#define CV_FOURCC_MACRO(c1, c2, c3, c4) (((c1) & 255) + (((c2) & 255) << 8) + (((c3) & 255) << 16) + (((c4) & 255) << 24))

/** @brief Constructs the fourcc code of the codec function

Simply call it with 4 chars fourcc code like `CV_FOURCC('I', 'Y', 'U', 'V')`

List of codes can be obtained at [Video Codecs by FOURCC](https://fourcc.org/codecs.php) page.
FFMPEG backend with MP4 container natively uses other values as fourcc code:
see [ObjectType](http://mp4ra.org/#/codecs).
*/
CV_INLINE int CV_FOURCC(char c1, char c2, char c3, char c4)
{
    return CV_FOURCC_MACRO(c1, c2, c3, c4);
}

//! (Windows only) Open Codec Selection Dialog
#define CV_FOURCC_PROMPT -1
//! (Linux only) Use default codec for specified filename
#define CV_FOURCC_DEFAULT CV_FOURCC('I', 'Y', 'U', 'V')

#endif // OPENCV_VIDEOIO_LEGACY_CONSTANTS_H
