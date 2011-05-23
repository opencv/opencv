#include <boost/python.hpp>

#include <opencv2/highgui/highgui.hpp>

namespace bp = boost::python;
namespace opencv_wrappers
{
  void wrap_highgui_defines()
  {
    bp::object opencv = bp::scope();
    opencv.attr("CV_FONT_LIGHT") = int(CV_FONT_LIGHT);
    opencv.attr("CV_FONT_NORMAL") = int(CV_FONT_NORMAL);
    opencv.attr("CV_FONT_DEMIBOLD") = int(CV_FONT_DEMIBOLD);
    opencv.attr("CV_FONT_BOLD") = int(CV_FONT_BOLD);
    opencv.attr("CV_FONT_BLACK") = int(CV_FONT_BLACK);
    opencv.attr("CV_STYLE_NORMAL") = int(CV_STYLE_NORMAL);
    opencv.attr("CV_STYLE_ITALIC") = int(CV_STYLE_ITALIC);
    opencv.attr("CV_STYLE_OBLIQUE") = int(CV_STYLE_OBLIQUE);

    //These 3 flags are used by cvSet/GetWindowProperty
    opencv.attr("CV_WND_PROP_FULLSCREEN") = int(CV_WND_PROP_FULLSCREEN);
    opencv.attr("CV_WND_PROP_AUTOSIZE") = int(CV_WND_PROP_AUTOSIZE);
    opencv.attr("CV_WND_PROP_ASPECTRATIO") = int(CV_WND_PROP_ASPECTRATIO);
    //
    //These 2 flags are used by cvNamedWindow and cvSet/GetWindowProperty
    opencv.attr("CV_WINDOW_NORMAL") = int(CV_WINDOW_NORMAL);
    opencv.attr("CV_WINDOW_AUTOSIZE") = int(CV_WINDOW_AUTOSIZE);
    //
    //Those flags are only for Qt
    opencv.attr("CV_GUI_EXPANDED") = int(CV_GUI_EXPANDED);
    opencv.attr("CV_GUI_NORMAL") = int(CV_GUI_NORMAL);
    //
    //These 3 flags are used by cvNamedWindow and cvSet/GetWindowProperty
    opencv.attr("CV_WINDOW_FULLSCREEN") = int(CV_WINDOW_FULLSCREEN);
    opencv.attr("CV_WINDOW_FREERATIO") = int(CV_WINDOW_FREERATIO);
    opencv.attr("CV_WINDOW_KEEPRATIO") = int(CV_WINDOW_KEEPRATIO);

    opencv.attr("CV_EVENT_MOUSEMOVE") = int(CV_EVENT_MOUSEMOVE);
    opencv.attr("CV_EVENT_LBUTTONDOWN") = int(CV_EVENT_LBUTTONDOWN);
    opencv.attr("CV_EVENT_RBUTTONDOWN") = int(CV_EVENT_RBUTTONDOWN);
    opencv.attr("CV_EVENT_MBUTTONDOWN") = int(CV_EVENT_MBUTTONDOWN);
    opencv.attr("CV_EVENT_LBUTTONUP") = int(CV_EVENT_LBUTTONUP);
    opencv.attr("CV_EVENT_RBUTTONUP") = int(CV_EVENT_RBUTTONUP);
    opencv.attr("CV_EVENT_MBUTTONUP") = int(CV_EVENT_MBUTTONUP);
    opencv.attr("CV_EVENT_LBUTTONDBLCLK") = int(CV_EVENT_LBUTTONDBLCLK);
    opencv.attr("CV_EVENT_RBUTTONDBLCLK") = int(CV_EVENT_RBUTTONDBLCLK);
    opencv.attr("CV_EVENT_MBUTTONDBLCLK") = int(CV_EVENT_MBUTTONDBLCLK);
    opencv.attr("CV_EVENT_FLAG_LBUTTON") = int(CV_EVENT_FLAG_LBUTTON);
    opencv.attr("CV_EVENT_FLAG_RBUTTON") = int(CV_EVENT_FLAG_RBUTTON);
    opencv.attr("CV_EVENT_FLAG_MBUTTON") = int(CV_EVENT_FLAG_MBUTTON);
    opencv.attr("CV_EVENT_FLAG_CTRLKEY") = int(CV_EVENT_FLAG_CTRLKEY);
    opencv.attr("CV_EVENT_FLAG_SHIFTKEY") = int(CV_EVENT_FLAG_SHIFTKEY);
    opencv.attr("CV_EVENT_FLAG_ALTKEY") = int(CV_EVENT_FLAG_ALTKEY);

    opencv.attr("CV_LOAD_IMAGE_UNCHANGED") = int(CV_LOAD_IMAGE_UNCHANGED);

    opencv.attr("CV_LOAD_IMAGE_GRAYSCALE") = int(CV_LOAD_IMAGE_GRAYSCALE);
    opencv.attr("CV_LOAD_IMAGE_COLOR") = int(CV_LOAD_IMAGE_COLOR);
    opencv.attr("CV_LOAD_IMAGE_ANYDEPTH") = int(CV_LOAD_IMAGE_ANYDEPTH);
    opencv.attr("CV_LOAD_IMAGE_ANYCOLOR") = int(CV_LOAD_IMAGE_ANYCOLOR);

    opencv.attr("CV_IMWRITE_JPEG_QUALITY") = int(CV_IMWRITE_JPEG_QUALITY);
    opencv.attr("CV_IMWRITE_PNG_COMPRESSION") = int(CV_IMWRITE_PNG_COMPRESSION);
    opencv.attr("CV_IMWRITE_PXM_BINARY") = int(CV_IMWRITE_PXM_BINARY);

    opencv.attr("CV_CVTIMG_FLIP") = int(CV_CVTIMG_FLIP);
    opencv.attr("CV_CVTIMG_SWAP_RB") = int(CV_CVTIMG_SWAP_RB);

    opencv.attr("CV_CAP_ANY") = int(CV_CAP_ANY);

    opencv.attr("CV_CAP_MIL") = int(CV_CAP_MIL);

    opencv.attr("CV_CAP_VFW") = int(CV_CAP_VFW);
    opencv.attr("CV_CAP_V4L") = int(CV_CAP_V4L);
    opencv.attr("CV_CAP_V4L2") = int(CV_CAP_V4L2);

    opencv.attr("CV_CAP_FIREWARE") = int(CV_CAP_FIREWARE);
    opencv.attr("CV_CAP_FIREWIRE") = int(CV_CAP_FIREWIRE);
    opencv.attr("CV_CAP_IEEE1394") = int(CV_CAP_IEEE1394);
    opencv.attr("CV_CAP_DC1394") = int(CV_CAP_DC1394);
    opencv.attr("CV_CAP_CMU1394") = int(CV_CAP_CMU1394);

    opencv.attr("CV_CAP_STEREO") = int(CV_CAP_STEREO);
    opencv.attr("CV_CAP_TYZX") = int(CV_CAP_TYZX);
    opencv.attr("CV_TYZX_LEFT") = int(CV_TYZX_LEFT);
    opencv.attr("CV_TYZX_RIGHT") = int(CV_TYZX_RIGHT);
    opencv.attr("CV_TYZX_COLOR") = int(CV_TYZX_COLOR);
    opencv.attr("CV_TYZX_Z") = int(CV_TYZX_Z);

    opencv.attr("CV_CAP_QT") = int(CV_CAP_QT);

    opencv.attr("CV_CAP_UNICAP") = int(CV_CAP_UNICAP);

    opencv.attr("CV_CAP_DSHOW") = int(CV_CAP_DSHOW);

    opencv.attr("CV_CAP_PVAPI") = int(CV_CAP_PVAPI);

    opencv.attr("CV_CAP_OPENNI") = int(CV_CAP_OPENNI);

    opencv.attr("CV_CAP_ANDROID") = int(CV_CAP_ANDROID);

    opencv.attr("CV_CAP_PROP_DC1394_OFF") = int(CV_CAP_PROP_DC1394_OFF);
    opencv.attr("CV_CAP_PROP_DC1394_MODE_MANUAL") = int(CV_CAP_PROP_DC1394_MODE_MANUAL);
    opencv.attr("CV_CAP_PROP_DC1394_MODE_AUTO") = int(CV_CAP_PROP_DC1394_MODE_AUTO);
    opencv.attr("CV_CAP_PROP_DC1394_MODE_ONE_PUSH_AUTO") = int(CV_CAP_PROP_DC1394_MODE_ONE_PUSH_AUTO);
    opencv.attr("CV_CAP_PROP_POS_MSEC") = int(CV_CAP_PROP_POS_MSEC);
    opencv.attr("CV_CAP_PROP_POS_FRAMES") = int(CV_CAP_PROP_POS_FRAMES);
    opencv.attr("CV_CAP_PROP_POS_AVI_RATIO") = int(CV_CAP_PROP_POS_AVI_RATIO);
    opencv.attr("CV_CAP_PROP_FRAME_WIDTH") = int(CV_CAP_PROP_FRAME_WIDTH);
    opencv.attr("CV_CAP_PROP_FRAME_HEIGHT") = int(CV_CAP_PROP_FRAME_HEIGHT);
    opencv.attr("CV_CAP_PROP_FPS") = int(CV_CAP_PROP_FPS);
    opencv.attr("CV_CAP_PROP_FOURCC") = int(CV_CAP_PROP_FOURCC);
    opencv.attr("CV_CAP_PROP_FRAME_COUNT") = int(CV_CAP_PROP_FRAME_COUNT);
    opencv.attr("CV_CAP_PROP_FORMAT") = int(CV_CAP_PROP_FORMAT);
    opencv.attr("CV_CAP_PROP_MODE") = int(CV_CAP_PROP_MODE);
    opencv.attr("CV_CAP_PROP_BRIGHTNESS") = int(CV_CAP_PROP_BRIGHTNESS);
    opencv.attr("CV_CAP_PROP_CONTRAST") = int(CV_CAP_PROP_CONTRAST);
    opencv.attr("CV_CAP_PROP_SATURATION") = int(CV_CAP_PROP_SATURATION);
    opencv.attr("CV_CAP_PROP_HUE") = int(CV_CAP_PROP_HUE);
    opencv.attr("CV_CAP_PROP_GAIN") = int(CV_CAP_PROP_GAIN);
    opencv.attr("CV_CAP_PROP_EXPOSURE") = int(CV_CAP_PROP_EXPOSURE);
    opencv.attr("CV_CAP_PROP_CONVERT_RGB") = int(CV_CAP_PROP_CONVERT_RGB);
    opencv.attr("CV_CAP_PROP_WHITE_BALANCE_BLUE_U") = int(CV_CAP_PROP_WHITE_BALANCE_BLUE_U);
    opencv.attr("CV_CAP_PROP_RECTIFICATION") = int(CV_CAP_PROP_RECTIFICATION);
    opencv.attr("CV_CAP_PROP_MONOCROME") = int(CV_CAP_PROP_MONOCROME);
    opencv.attr("CV_CAP_PROP_SHARPNESS") = int(CV_CAP_PROP_SHARPNESS);
    opencv.attr("CV_CAP_PROP_AUTO_EXPOSURE") = int(CV_CAP_PROP_AUTO_EXPOSURE);
    // user can adjust refernce level
    // using this feature
    opencv.attr("CV_CAP_PROP_GAMMA") = int(CV_CAP_PROP_GAMMA);
    opencv.attr("CV_CAP_PROP_TEMPERATURE") = int(CV_CAP_PROP_TEMPERATURE);
    opencv.attr("CV_CAP_PROP_TRIGGER") = int(CV_CAP_PROP_TRIGGER);
    opencv.attr("CV_CAP_PROP_TRIGGER_DELAY") = int(CV_CAP_PROP_TRIGGER_DELAY);
    opencv.attr("CV_CAP_PROP_WHITE_BALANCE_RED_V") = int(CV_CAP_PROP_WHITE_BALANCE_RED_V);
    opencv.attr("CV_CAP_PROP_MAX_DC1394") = int(CV_CAP_PROP_MAX_DC1394);
    // OpenNI map generators
    opencv.attr("CV_CAP_OPENNI_DEPTH_GENERATOR") = int(CV_CAP_OPENNI_DEPTH_GENERATOR);
    opencv.attr("CV_CAP_OPENNI_IMAGE_GENERATOR") = int(CV_CAP_OPENNI_IMAGE_GENERATOR);
    opencv.attr("CV_CAP_OPENNI_GENERATORS_MASK") = int(CV_CAP_OPENNI_GENERATORS_MASK);

    // Properties of cameras avalible through OpenNI interfaces
    opencv.attr("CV_CAP_PROP_OPENNI_OUTPUT_MODE") = int(CV_CAP_PROP_OPENNI_OUTPUT_MODE);
    opencv.attr("CV_CAP_PROP_OPENNI_FRAME_MAX_DEPTH") = int(CV_CAP_PROP_OPENNI_FRAME_MAX_DEPTH);
    opencv.attr("CV_CAP_PROP_OPENNI_BASELINE") = int(CV_CAP_PROP_OPENNI_BASELINE);
    opencv.attr("CV_CAP_PROP_OPENNI_FOCAL_LENGTH") = int(CV_CAP_PROP_OPENNI_FOCAL_LENGTH);
    opencv.attr("CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE") = int(CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE);
    opencv.attr("CV_CAP_OPENNI_DEPTH_GENERATOR_BASELINE") = int(CV_CAP_OPENNI_DEPTH_GENERATOR_BASELINE);
    opencv.attr("CV_CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH") = int(CV_CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH);
    opencv.attr("CV_CAP_OPENNI_DEPTH_MAP") = int(CV_CAP_OPENNI_DEPTH_MAP);
    opencv.attr("CV_CAP_OPENNI_POINT_CLOUD_MAP") = int(CV_CAP_OPENNI_POINT_CLOUD_MAP);
    opencv.attr("CV_CAP_OPENNI_DISPARITY_MAP") = int(CV_CAP_OPENNI_DISPARITY_MAP);
    opencv.attr("CV_CAP_OPENNI_DISPARITY_MAP_32F") = int(CV_CAP_OPENNI_DISPARITY_MAP_32F);
    opencv.attr("CV_CAP_OPENNI_VALID_DEPTH_MASK") = int(CV_CAP_OPENNI_VALID_DEPTH_MASK);

    opencv.attr("CV_CAP_OPENNI_BGR_IMAGE") = int(CV_CAP_OPENNI_BGR_IMAGE);
    opencv.attr("CV_CAP_OPENNI_GRAY_IMAGE") = int(CV_CAP_OPENNI_GRAY_IMAGE);

    opencv.attr("CV_CAP_OPENNI_VGA_30HZ") = int(CV_CAP_OPENNI_VGA_30HZ);
    opencv.attr("CV_CAP_OPENNI_SXGA_15HZ") = int(CV_CAP_OPENNI_SXGA_15HZ);

    opencv.attr("CV_CAP_ANDROID_COLOR_FRAME") = int(CV_CAP_ANDROID_COLOR_FRAME);
    opencv.attr("CV_CAP_ANDROID_GREY_FRAME") = int(CV_CAP_ANDROID_GREY_FRAME);
    //opencv.attr("CV_CAP_ANDROID_YUV_FRAME") = int(CV_CAP_ANDROID_YUV_FRAME);
  }
}
