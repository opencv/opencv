// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "window_framebuffer.hpp"

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.defines.hpp>
#ifdef NDEBUG
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_DEBUG + 1
#else
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_VERBOSE + 1
#endif
#include <opencv2/core/utils/logger.hpp>

#include <unistd.h>
#include <stdio.h>
#include <termios.h>
#include <fcntl.h>
#include <stdlib.h>
#include <linux/fb.h>
#include <linux/input.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include "opencv2/imgproc.hpp"

#ifdef HAVE_FRAMEBUFFER_XVFB
#include <X11/XWDFile.h>
#include <X11/X.h>

#define C32INT(ptr) ((((unsigned char*)ptr)[0] << 24) | (((unsigned char*)ptr)[1] << 16) | \
  (((unsigned char*)ptr)[2] << 8) | (((unsigned char*)ptr)[3] << 0))
#endif


namespace cv {
namespace highgui_backend {

std::shared_ptr<UIBackend> createUIBackendFramebuffer()
{
    return std::make_shared<FramebufferBackend>();
}

static std::string& getFBMode()
{
    static std::string fbModeOpenCV =
    cv::utils::getConfigurationParameterString("OPENCV_HIGHGUI_FB_MODE", "FB");
    return fbModeOpenCV;
}

static std::string& getFBFileName()
{
    static std::string fbFileNameFB =
    cv::utils::getConfigurationParameterString("FRAMEBUFFER", "/dev/fb0");
    static std::string fbFileNameOpenCV =
    cv::utils::getConfigurationParameterString("OPENCV_HIGHGUI_FB_DEVICE", "");

    if (!fbFileNameOpenCV.empty()) return fbFileNameOpenCV;
    return fbFileNameFB;
}

FramebufferWindow::FramebufferWindow(FramebufferBackend &_backend, int _flags):
    backend(_backend), flags(_flags)
{
    CV_LOG_DEBUG(NULL, "UI: FramebufferWindow::FramebufferWindow()");
    FB_ID = "FramebufferWindow";
    windowRect = Rect(0,0, backend.getFBWidth(), backend.getFBHeight());
}

FramebufferWindow::~FramebufferWindow()
{
    CV_LOG_DEBUG(NULL, "UI: FramebufferWindow::~FramebufferWindow()");
}

void FramebufferWindow::imshow(InputArray image)
{
    CV_LOG_DEBUG(NULL, "UI: FramebufferWindow::imshow(InputArray image)");
    currentImg = image.getMat().clone();

    CV_LOG_INFO(NULL, "UI: InputArray image: "
    << cv::typeToString(image.type()) << " size " << image.size());

    if (currentImg.empty())
    {
        CV_LOG_WARNING(NULL, "UI: image is empty");
        return;
    }
    CV_CheckEQ(currentImg.dims, 2, "UI: dims != 2");

    Mat img = image.getMat();
    switch (img.channels())
    {
        case 1:
            {
                Mat tmp;
                switch(img.type())
                {
                    case CV_8U:
                        tmp = img;
                        break;
                    case CV_8S:
                        cv::convertScaleAbs(img, tmp, 1, 127);
                        break;
                    case CV_16S:
                        cv::convertScaleAbs(img, tmp, 1/255., 127);
                        break;
                    case CV_16U:
                        cv::convertScaleAbs(img, tmp, 1/255.);
                        break;
                    case CV_32F:
                    case CV_64F: // assuming image has values in range [0, 1)
                        img.convertTo(tmp, CV_8U, 255., 0.);
                        break;
                }
                Mat rgb(img.rows, img.cols, CV_8UC3);
                cvtColor(tmp, rgb, COLOR_GRAY2RGB);
                img = rgb;
            }
            break;
        case 3:
        case 4:
            {
                Mat tmp(img.rows, img.cols, CV_8UC3);
                convertToShow(img, tmp, true);
                img = tmp;
            }
            break;
        default:
            CV_Error(cv::Error::StsBadArg, "Bad image: wrong number of channels");
    }
    {
        Mat bgra(img.rows, img.cols, CV_8UC4);
        cvtColor(img, bgra, COLOR_RGB2BGRA, bgra.channels());
        img = bgra;
    }

    int newWidth = windowRect.width;
    int newHeight = windowRect.height;
    int cntChannel = img.channels();
    cv::Size imgSize = currentImg.size();

    if (flags & WINDOW_AUTOSIZE)
    {
        windowRect.width = imgSize.width;
        windowRect.height = imgSize.height;
        newWidth = windowRect.width;
        newHeight = windowRect.height;
    }

    if (flags & WINDOW_FREERATIO)
    {
        newWidth = windowRect.width;
        newHeight = windowRect.height;
    }
    else //WINDOW_KEEPRATIO
    {
        double aspect_ratio = ((double)img.cols) / img.rows;
        newWidth = windowRect.width;
        newHeight = (int)(windowRect.width / aspect_ratio);

        if (newHeight > windowRect.height)
        {
            newWidth = (int)(windowRect.height * aspect_ratio);
            newHeight = windowRect.height;
        }
    }

    if ((newWidth != img.cols) && (newHeight != img.rows))
    {
        Mat imResize;
        cv::resize(img, imResize, cv::Size(newWidth, newHeight), INTER_LINEAR);
        img = imResize;
    }

    CV_LOG_INFO(NULL, "UI: Formated image: "
    << cv::typeToString(img.type()) << " size " << img.size());

    if (backend.getMode() == FB_MODE_EMU)
    {
        CV_LOG_WARNING(NULL, "UI: FramebufferWindow::imshow is used in EMU mode");
        return;
    }

    if (backend.getFBPointer() == MAP_FAILED)
    {
        CV_LOG_ERROR(NULL, "UI: Framebuffer is not mapped");
        return;
    }

    int xOffset = backend.getFBXOffset();
    int yOffset = backend.getFBYOffset();
    int fbHeight = backend.getFBHeight();
    int fbWidth = backend.getFBWidth();
    int lineLength = backend.getFBLineLength();

    int img_start_x;
    int img_start_y;
    int img_end_x;
    int img_end_y;
    int fb_start_x;
    int fb_start_y;

    if (windowRect.y - yOffset < 0)
    {
        img_start_y = - (windowRect.y - yOffset);
    }
    else
    {
        img_start_y = 0;
    }
    if (windowRect.x - xOffset < 0)
    {
        img_start_x = - (windowRect.x - xOffset);
    }
    else
    {
        img_start_x = 0;
    }

    if (windowRect.y + yOffset + img.rows > fbHeight)
    {
        img_end_y = fbHeight - windowRect.y - yOffset;
    }
    else
    {
        img_end_y = img.rows;
    }
    if (windowRect.x + xOffset + img.cols > fbWidth)
    {
        img_end_x = fbWidth - windowRect.x - xOffset;
    }
    else
    {
        img_end_x = img.cols;
    }

    if (windowRect.y + yOffset >= 0)
    {
        fb_start_y = windowRect.y + yOffset;
    }
    else
    {
        fb_start_y = 0;
    }
    if (windowRect.x + xOffset >= 0)
    {
        fb_start_x = windowRect.x + xOffset;
    }
    else
    {
        fb_start_x = 0;
    }

    for (int y = img_start_y; y < img_end_y; y++)
    {
        std::memcpy(backend.getFBPointer() +
        (fb_start_y + y - img_start_y) * lineLength + fb_start_x * cntChannel,
        img.ptr<unsigned char>(y) + img_start_x * cntChannel,
        (img_end_x - img_start_x) * cntChannel);
    }
}

double FramebufferWindow::getProperty(int /*prop*/) const
{
    CV_LOG_WARNING(NULL, "UI: getProperty (not supported)");
    return 0.0;
}

bool FramebufferWindow::setProperty(int /*prop*/, double /*value*/)
{
    CV_LOG_WARNING(NULL, "UI: setProperty (not supported)");
    return false;
}

void FramebufferWindow::resize(int width, int height)
{
    CV_LOG_DEBUG(NULL, "UI: FramebufferWindow::resize(int width "
    << width <<", height " << height << ")");

    CV_Assert(width > 0);
    CV_Assert(height > 0);

    if (!(flags & WINDOW_AUTOSIZE))
    {
        windowRect.width = width;
        windowRect.height = height;

        if (!currentImg.empty())
        {
            imshow(currentImg);
        }
    }
}

void FramebufferWindow::move(int x, int y)
{
    CV_LOG_DEBUG(NULL, "UI: FramebufferWindow::move(int x " << x << ", y " << y <<")");

    windowRect.x = x;
    windowRect.y = y;

    if (!currentImg.empty())
    {
        imshow(currentImg);
    }
}

Rect FramebufferWindow::getImageRect() const
{
    CV_LOG_DEBUG(NULL, "UI: FramebufferWindow::getImageRect()");
    return windowRect;
}

void FramebufferWindow::setTitle(const std::string& /*title*/)
{
    CV_LOG_WARNING(NULL, "UI: setTitle (not supported)");
}

void FramebufferWindow::setMouseCallback(MouseCallback /*onMouse*/, void* /*userdata*/)
{
    CV_LOG_WARNING(NULL, "UI: setMouseCallback (not supported)");
}

std::shared_ptr<UITrackbar> FramebufferWindow::createTrackbar(
        const std::string& /*name*/,
        int /*count*/,
        TrackbarCallback /*onChange*/,
        void* /*userdata*/)
{
    CV_LOG_WARNING(NULL, "UI: createTrackbar (not supported)");
    return nullptr;
}

std::shared_ptr<UITrackbar> FramebufferWindow::findTrackbar(const std::string& /*name*/)
{
    CV_LOG_WARNING(NULL, "UI: findTrackbar (not supported)");
    return nullptr;
}

const std::string& FramebufferWindow::getID() const
{
    CV_LOG_DEBUG(NULL, "UI: FramebufferWindow::getID()");
    return FB_ID;
}

bool FramebufferWindow::isActive() const
{
    CV_LOG_DEBUG(NULL, "UI: FramebufferWindow::isActive()");
    return true;
}

void FramebufferWindow::destroy()
{
    CV_LOG_DEBUG(NULL, "UI: FramebufferWindow::destroy()");
}

int FramebufferBackend::fbOpenAndGetInfo()
{
    std::string fbFileName = getFBFileName();
    CV_LOG_INFO(NULL, "UI: FramebufferWindow::The following is used as a framebuffer file: \n"
    << fbFileName);

    int fb_fd = open(fbFileName.c_str(), O_RDWR);
    if (fb_fd == -1)
    {
        CV_LOG_ERROR(NULL, "UI: can't open framebuffer");
        return -1;
    }

    if (ioctl(fb_fd, FBIOGET_FSCREENINFO, &fixInfo))
    {
        CV_LOG_ERROR(NULL, "UI: can't read fix info for framebuffer");
        return -1;
    }

    if (ioctl(fb_fd, FBIOGET_VSCREENINFO, &varInfo))
    {
        CV_LOG_ERROR(NULL, "UI: can't read var info for framebuffer");
        return -1;
    }

    CV_LOG_INFO(NULL, "UI: framebuffer info: \n"
    << "   red offset  " << varInfo.red.offset << " length " << varInfo.red.length << "\n"
    << " green offset  " << varInfo.green.offset << " length " << varInfo.green.length << "\n"
    << "  blue offset  " << varInfo.blue.offset << " length " << varInfo.blue.length << "\n"
    << "transp offset  " << varInfo.transp.offset << " length " <<varInfo.transp.length << "\n"
    << "bits_per_pixel " << varInfo.bits_per_pixel);

    if ((varInfo.red.offset != 16) && (varInfo.red.length != 8) &&
        (varInfo.green.offset != 8) && (varInfo.green.length != 8) &&
        (varInfo.blue.offset != 0) && (varInfo.blue.length != 8) &&
        (varInfo.bits_per_pixel != 32) )
    {
        close(fb_fd);
        CV_LOG_ERROR(NULL, "UI: Framebuffer format is not supported "
        << "(use BGRA format with bits_per_pixel = 32)");
        return -1;
    }

    fbWidth = varInfo.xres;
    fbHeight = varInfo.yres;
    fbXOffset = varInfo.xoffset;
    fbYOffset = varInfo.yoffset;
    fbBitsPerPixel = varInfo.bits_per_pixel;
    fbLineLength = fixInfo.line_length;

    fbScreenSize = max(varInfo.xres, varInfo.xres_virtual) *
    max(varInfo.yres, varInfo.yres_virtual) *
    fbBitsPerPixel / 8;

    fbPointer = (unsigned char*)
    mmap(0, fbScreenSize, PROT_READ | PROT_WRITE, MAP_SHARED, fb_fd, 0);

    if (fbPointer == MAP_FAILED)
    {
        CV_LOG_ERROR(NULL, "UI: can't mmap framebuffer");
        return -1;
    }

    return fb_fd;
}

int FramebufferBackend::XvfbOpenAndGetInfo()
{
    int fb_fd = -1;

#ifdef HAVE_FRAMEBUFFER_XVFB
    std::string fbFileName = getFBFileName();
    CV_LOG_INFO(NULL, "UI: FramebufferWindow::The following is used as a framebuffer file: \n"
    << fbFileName);

    fb_fd = open(fbFileName.c_str(), O_RDWR);
    if (fb_fd == -1)
    {
        CV_LOG_ERROR(NULL, "UI: can't open framebuffer");
        return -1;
    }

    XWDFileHeader *xwd_header;

    xwd_header = (XWDFileHeader*)
    mmap(NULL, sizeof(XWDFileHeader), PROT_READ, MAP_SHARED, fb_fd, 0);

    if (xwd_header == MAP_FAILED)
    {
        CV_LOG_ERROR(NULL, "UI: can't mmap xwd header");
        return -1;
    }

    if (C32INT(&(xwd_header->pixmap_format)) != ZPixmap)
    {
        CV_LOG_ERROR(NULL, "Unsupported pixmap format: " << xwd_header->pixmap_format);
        return -1;
    }

    if (xwd_header->xoffset != 0)
    {
        CV_LOG_ERROR(NULL, "UI: Unsupported xoffset value: " << xwd_header->xoffset );
        return -1;
    }

    unsigned int r = C32INT(&(xwd_header->red_mask));
    unsigned int g = C32INT(&(xwd_header->green_mask));
    unsigned int b = C32INT(&(xwd_header->blue_mask));

    fbWidth = C32INT(&(xwd_header->pixmap_width));
    fbHeight = C32INT(&(xwd_header->pixmap_height));
    fbXOffset = 0;
    fbYOffset = 0;
    fbLineLength = C32INT(&(xwd_header->bytes_per_line));
    fbBitsPerPixel = C32INT(&(xwd_header->bits_per_pixel));

    CV_LOG_INFO(NULL, "UI: XVFB info: \n"
    << "   red_mask " << r << "\n"
    << " green_mask " << g << "\n"
    << "  blue_mask " << b << "\n"
    << "bits_per_pixel " << fbBitsPerPixel);

    if ((r != 16711680 ) && (g != 65280 ) && (b != 255 ) &&
        (fbBitsPerPixel != 32))
    {
        CV_LOG_ERROR(NULL, "UI: Framebuffer format is not supported "
        << "(use BGRA format with bits_per_pixel = 32)");
        return -1;
    }

    xvfb_len_header = C32INT(&(xwd_header->header_size));
    xvfb_len_colors = sizeof(XWDColor) * C32INT(&(xwd_header->ncolors));
    xvfb_len_pixmap = C32INT(&(xwd_header->bytes_per_line)) *
    C32INT(&(xwd_header->pixmap_height));

    munmap(xwd_header, sizeof(XWDFileHeader));

    fbScreenSize = xvfb_len_header + xvfb_len_colors + xvfb_len_pixmap;
    xwd_header = (XWDFileHeader*)
    mmap(NULL, fbScreenSize, PROT_READ | PROT_WRITE, MAP_SHARED, fb_fd, 0);

    fbPointer = (unsigned char*)xwd_header;
    fbPointer_dist = xvfb_len_header + xvfb_len_colors;

#else
    CV_LOG_WARNING(NULL, "UI: To use virtual framebuffer, "
    << "compile OpenCV with the WITH_FRAMEBUFFER_XVFB=ON");
#endif

    return fb_fd;
}

fb_var_screeninfo &FramebufferBackend::getVarInfo()
{
    return varInfo;
}

fb_fix_screeninfo &FramebufferBackend::getFixInfo()
{
    return fixInfo;
}

int FramebufferBackend::getFramebuffrerID()
{
    return fbID;
}

int FramebufferBackend::getFBWidth()
{
    return fbWidth;
}

int FramebufferBackend::getFBHeight()
{
    return fbHeight;
}

int FramebufferBackend::getFBXOffset()
{
    return fbXOffset;
}

int FramebufferBackend::getFBYOffset()
{
    return fbYOffset;
}

int FramebufferBackend::getFBBitsPerPixel()
{
    return fbBitsPerPixel;
}

int FramebufferBackend::getFBLineLength()
{
    return fbLineLength;
}

unsigned char* FramebufferBackend::getFBPointer()
{
    return fbPointer + fbPointer_dist;
}

Mat& FramebufferBackend::getBackgroundBuff()
{
    return backgroundBuff;
}

OpenCVFBMode FramebufferBackend::getMode()
{
    return mode;
}

FramebufferBackend::FramebufferBackend():mode(FB_MODE_FB), fbPointer_dist(0)
{
    CV_LOG_DEBUG(NULL, "UI: FramebufferWindow::FramebufferBackend()");

    std::string fbModeStr = getFBMode();

    if (fbModeStr == "EMU")
    {
        mode = FB_MODE_EMU;
        CV_LOG_WARNING(NULL, "UI: FramebufferWindow is trying to use EMU mode");
    }
    if (fbModeStr == "FB")
    {
        mode = FB_MODE_FB;
        CV_LOG_WARNING(NULL, "UI: FramebufferWindow is trying to use FB mode");
    }
    if (fbModeStr == "XVFB")
    {
        mode = FB_MODE_XVFB;
        CV_LOG_WARNING(NULL, "UI: FramebufferWindow is trying to use XVFB mode");
    }

    fbID = -1;
    if (mode == FB_MODE_FB)
    {
        fbID = fbOpenAndGetInfo();
    }
    if (mode == FB_MODE_XVFB)
    {
        fbID = XvfbOpenAndGetInfo();
    }

    CV_LOG_INFO(NULL, "UI: FramebufferWindow::fbID " << fbID);

    if (fbID == -1)
    {
        mode = FB_MODE_EMU;
        fbWidth = 640;
        fbHeight = 480;
        fbXOffset = 0;
        fbYOffset = 0;
        fbBitsPerPixel = 0;
        fbLineLength = 0;

        CV_LOG_WARNING(NULL, "UI: FramebufferWindow is used in EMU mode");
        return;
    }

    CV_LOG_INFO(NULL, "UI: Framebuffer's width, height, bits per pix: "
    << fbWidth << " " << fbHeight << " " << fbBitsPerPixel);

    CV_LOG_INFO(NULL, "UI: Framebuffer's offsets (x, y), line length: "
    << fbXOffset << " " << fbYOffset << " " << fbLineLength);

    backgroundBuff = Mat(fbHeight, fbWidth, CV_8UC4);
    int cntChannel = 4;
    for (int y = fbYOffset; y < backgroundBuff.rows + fbYOffset; y++)
    {
        std::memcpy(backgroundBuff.ptr<unsigned char>(y - fbYOffset),
        getFBPointer() + y * fbLineLength + fbXOffset * cntChannel,
        backgroundBuff.cols * cntChannel);
    }
}

FramebufferBackend::~FramebufferBackend()
{
    CV_LOG_DEBUG(NULL, "UI: FramebufferBackend::~FramebufferBackend()");
    if(fbID == -1) return;

    if (fbPointer != MAP_FAILED)
    {
        int cntChannel = 4;
        for (int y = fbYOffset; y < backgroundBuff.rows + fbYOffset; y++)
        {
            std::memcpy(getFBPointer() + y * fbLineLength + fbXOffset * cntChannel,
            backgroundBuff.ptr<cv::Vec4b>(y - fbYOffset),
            backgroundBuff.cols * cntChannel);
        }

        munmap(fbPointer, fbScreenSize);
    }
    close(fbID);
}

void FramebufferBackend::destroyAllWindows() {
    CV_LOG_DEBUG(NULL, "UI: FramebufferBackend::destroyAllWindows()");
}

// namedWindow
std::shared_ptr<UIWindow> FramebufferBackend::createWindow(
    const std::string& winname,
    int flags)
{
    CV_LOG_DEBUG(NULL, "UI: FramebufferBackend::createWindow("
    << winname << ", " << flags << ")");
    return std::make_shared<FramebufferWindow>(*this, flags);
}

void FramebufferBackend::initTermios(int echo, int wait)
{
    tcgetattr(0, &old);
    current = old;
    current.c_lflag &= ~ICANON;
    current.c_lflag &= ~ISIG;
    current.c_cc[VMIN] = wait;
    if (echo)
    {
        current.c_lflag |= ECHO;
    }
    else
    {
        current.c_lflag &= ~ECHO;
    }
    tcsetattr(0, TCSANOW, &current);
}

void FramebufferBackend::resetTermios(void)
{
    tcsetattr(0, TCSANOW, &old);
}

int FramebufferBackend::getch_(int echo, int wait)
{
    int ch;
    initTermios(echo, wait);
    ch = getchar();
    if (ch < 0)
    {
        rewind(stdin);
    }
    resetTermios();
    return ch;
}

bool FramebufferBackend::kbhit()
{
    int byteswaiting = 0;
    initTermios(0, 1);
    if (ioctl(0, FIONREAD, &byteswaiting) < 0)
    {
        CV_LOG_ERROR(NULL, "UI: Framebuffer ERR byteswaiting" );
    }
    resetTermios();

    return byteswaiting > 0;
}

int FramebufferBackend::waitKeyEx(int delay)
{
    CV_LOG_DEBUG(NULL, "UI: FramebufferBackend::waitKeyEx(int delay = " << delay << ")");

    int code = -1;

    if (delay <= 0)
    {
        int ch = getch_(0, 1);
        CV_LOG_INFO(NULL, "UI: FramebufferBackend::getch_() take value = " << (int)ch);
        code = ch;

        while ((ch = getch_(0, 0)) >= 0)
        {
            CV_LOG_INFO(NULL, "UI: FramebufferBackend::getch_() take value = "
            << (int)ch << " (additional code on <stdin>)");
            code = ch;
        }
    }
    else
    {
        bool f_kbhit = false;
        while (!(f_kbhit = kbhit()) && (delay > 0))
        {
            delay -= 1;
            usleep(1000);
        }
        if (f_kbhit)
        {
            CV_LOG_INFO(NULL, "UI: FramebufferBackend kbhit is True ");

            int ch = getch_(0, 1);
            CV_LOG_INFO(NULL, "UI: FramebufferBackend::getch_() take value = " << (int)ch);
            code = ch;

            while ((ch = getch_(0, 0)) >= 0)
            {
                CV_LOG_INFO(NULL, "UI: FramebufferBackend::getch_() take value = "
                << (int)ch << " (additional code on <stdin>)");
                code = ch;
            }
        }
    }

    CV_LOG_INFO(NULL, "UI: FramebufferBackend::waitKeyEx() result code = " << code);
    return code;
}

int FramebufferBackend::pollKey()
{
    CV_LOG_DEBUG(NULL, "UI: FramebufferBackend::pollKey()");
    int code = -1;
    bool f_kbhit = false;
    f_kbhit = kbhit();

    if (f_kbhit)
    {
        CV_LOG_INFO(NULL, "UI: FramebufferBackend kbhit is True ");

        int ch = getch_(0, 1);
        CV_LOG_INFO(NULL, "UI: FramebufferBackend::getch_() take value = " << (int)ch);
        code = ch;

        while ((ch = getch_(0, 0)) >= 0)
        {
            CV_LOG_INFO(NULL, "UI: FramebufferBackend::getch_() take value = "
            << (int)ch << " (additional code on <stdin>)");
            code = ch;
        }
    }

    return code;
}

const std::string FramebufferBackend::getName() const
{
    return "FB";
}

}} // cv::highgui_backend::
