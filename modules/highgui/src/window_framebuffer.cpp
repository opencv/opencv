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


namespace cv { namespace highgui_backend {
  
  std::shared_ptr<UIBackend> createUIBackendFramebuffer()
  {
    return std::make_shared<FramebufferBackend>();
  }

  FramebufferWindow::FramebufferWindow(FramebufferBackend &_backend): backend(_backend)
  {
    CV_LOG_INFO(NULL, "UI: FramebufferWindow::FramebufferWindow()");
    FB_ID = "FramebufferWindow";
    windowRect = Rect(0,0, backend.getFBWidth(), backend.getFBHeight());
  }
  
  FramebufferWindow::~FramebufferWindow()
  {
    CV_LOG_INFO(NULL, "UI: FramebufferWindow::~FramebufferWindow()");
  }

  void FramebufferWindow::imshow(InputArray image)
  {
    CV_LOG_INFO(NULL, "UI: FramebufferWindow::imshow(InputArray image)");
    CV_LOG_INFO(NULL, "UI: InputArray image: "
      << cv::typeToString(image.type()) << " size " << image.size());

    if (backend.getFBPointer() == MAP_FAILED) {
      CV_LOG_ERROR(NULL, "UI: Framebuffer is not mapped");
      return;
    }

    if(backend.getFBBitsPerPixel() != 32) {
      CV_LOG_ERROR(NULL, "UI: Framebuffer with bits per pixel = " 
        << backend.getFBBitsPerPixel() << " is not supported" );
      return;
    }
    
    Mat img;
    cvtColor(image, img, COLOR_RGB2RGBA);
    int new_width = windowRect.width;
    int new_height = windowRect.height;
    int cnt_channel = img.channels();
    
    cv::resize(img, img, cv::Size(new_width, new_height), INTER_LINEAR);
    
    CV_LOG_INFO(NULL, "UI: Formated image: "
      << cv::typeToString(img.type()) << " size " << img.size());
        
    // SHOW IMAGE
    int xOffset = backend.getFBXOffset();
    int yOffset = backend.getFBYOffset();
    int lineLength = backend.getFBLineLength();
    
    int showRows = min((windowRect.y + img.rows), backend.getFBHeight()) - windowRect.y;
    int showCols = min((windowRect.x + img.cols), backend.getFBWidth())  - windowRect.x;
    
    for (int y = yOffset; y < showRows + yOffset; y++)
    {
        std::memcpy(backend.getFBPointer() + (y + windowRect.y) * lineLength + 
                    xOffset + windowRect.x, 
                    img.ptr<cv::Vec4b>(y - yOffset), 
                    showCols * cnt_channel);
    }
  }

  double FramebufferWindow::getProperty(int prop) const
  {
    CV_LOG_INFO(NULL, "UI: FramebufferWindow::getProperty(int prop: " << prop << ")");
    CV_LOG_WARNING(NULL, "UI: getProperty (not supported)");

    return 0.0;
  }
  bool FramebufferWindow::setProperty(int prop, double value) 
  {
    CV_LOG_INFO(NULL, "UI: FramebufferWindow::setProperty(int prop " 
      << prop << ", value " << value << ")");
    CV_LOG_WARNING(NULL, "UI: setProperty (not supported)");

    return false;
  }

  void FramebufferWindow::resize(int width, int height)
  {
    CV_LOG_INFO(NULL, "UI: FramebufferWindow::resize(int width " 
      << width <<", height " << height << ")");
    windowRect.width = width;
    windowRect.height = height;
  }
  void FramebufferWindow::move(int x, int y) 
  {
    CV_LOG_INFO(NULL, "UI: FramebufferWindow::move(int x " << x << ", y " << y <<")");
    windowRect.x = x;
    windowRect.y = y;
  }

  Rect FramebufferWindow::getImageRect() const 
  {
    CV_LOG_INFO(NULL, "UI: FramebufferWindow::getImageRect()");
    return windowRect;
  }

  void FramebufferWindow::setTitle(const std::string& title) 
  {
    CV_LOG_INFO(NULL, "UI: FramebufferWindow::setTitle(" << title << ")");
    CV_LOG_WARNING(NULL, "UI: setTitle (not supported)");
  }

  void FramebufferWindow::setMouseCallback(MouseCallback onMouse, void* userdata )
  {
    CV_LOG_INFO(NULL, "UI: FramebufferWindow::setMouseCallback(...)");
    CV_LOG_WARNING(NULL, "UI: setMouseCallback (not supported)");
  }

  std::shared_ptr<UITrackbar> FramebufferWindow::createTrackbar(
      const std::string& name,
      int count,
      TrackbarCallback onChange,
      void* userdata)
  {
    CV_LOG_INFO(NULL, "UI: FramebufferWindow::createTrackbar(...)");
    CV_LOG_WARNING(NULL, "UI: createTrackbar (not supported)");
    return nullptr;
  }

  std::shared_ptr<UITrackbar> FramebufferWindow::findTrackbar(const std::string& name)
  {
    CV_LOG_INFO(NULL, "UI: FramebufferWindow::findTrackbar(...)");
    CV_LOG_WARNING(NULL, "UI: findTrackbar (not supported)");
    return nullptr;
  }
  
  const std::string& FramebufferWindow::getID() const  
  { 
    CV_LOG_INFO(NULL, "UI: FramebufferWindow::getID()");
    return FB_ID;
  }

  bool FramebufferWindow::isActive() const 
  {
    CV_LOG_INFO(NULL, "UI: FramebufferWindow::isActive()");
    return true;
  }

  void FramebufferWindow::destroy() 
  {
    CV_LOG_INFO(NULL, "UI: FramebufferWindow::destroy()");
  }

// !!##FramebufferBackend

//  int FramebufferBackend::OpenInputEvent()
//  {
//    int fd;
//    fd = open("/dev/input/event1", O_RDONLY);
//    if (fd == -1) {
//        std::cerr << "ERROR_OPENING_INPUT\n";
//        return -1;
//    }
//    return fd;
//  }

  static
  std::string& getFBFileName()
  {
    static std::string fbFileNameFB = 
      cv::utils::getConfigurationParameterString("FRAMEBUFFER", "");
    static std::string fbFileNameOpenCV = 
      cv::utils::getConfigurationParameterString("OPENCV_HIGHGUI_FRAMEBUFFER", "");
    static std::string fbFileNameDef = "/dev/fb0";
    
    if(!fbFileNameOpenCV.empty()) return fbFileNameOpenCV;
    if(!fbFileNameFB.empty()) return fbFileNameFB;
    return fbFileNameDef;
  }


  int FramebufferBackend::fbOpenAndGetInfo()
  {
    std::string fbFileName = getFBFileName();
    CV_LOG_INFO(NULL, "UI: FramebufferWindow::The following is used as a framebuffer file: \n" << fbFileName);
    
    int fb_fd = open(fbFileName.c_str(), O_RDWR);
    if (fb_fd == -1)
    {
      CV_LOG_ERROR(NULL, "UI: can't open framebuffer");
      return -1;
    }

    // Get fixed screen information
    if (ioctl(fb_fd, FBIOGET_FSCREENINFO, &fixInfo)) {
      CV_LOG_ERROR(NULL, "UI: can't read fix info for framebuffer");
     return -1;
    }

    // Get variable screen information
    if (ioctl(fb_fd, FBIOGET_VSCREENINFO, &varInfo)) {
      CV_LOG_ERROR(NULL, "UI: can't read var info for framebuffer");
      return -1;
    }

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
    return fbPointer;
  }
  Mat& FramebufferBackend::getBackgroundBuff()
  {
    return backgroundBuff;
  }

  FramebufferBackend::FramebufferBackend()
  {
    CV_LOG_INFO(NULL, "UI: FramebufferWindow::FramebufferBackend()");
    fbID = fbOpenAndGetInfo();
    CV_LOG_INFO(NULL, "UI: FramebufferWindow::fbID " << fbID);
    
    if(fbID == -1){
      fbWidth = 0;
      fbHeight = 0;
      fbXOffset = 0;
      fbYOffset = 0;
      fbBitsPerPixel = 0;
      fbLineLength = 0;
      return;
    }
    
    fbWidth        = varInfo.xres;
    fbHeight       = varInfo.yres;
    fbXOffset      = varInfo.yoffset;
    fbYOffset      = varInfo.xoffset;
    fbBitsPerPixel = varInfo.bits_per_pixel;
    fbLineLength   = fixInfo.line_length;
    
    CV_LOG_INFO(NULL, "UI: Framebuffer's width, height, bits per pix: " 
      << fbWidth << " " << fbHeight << " " << fbBitsPerPixel);

    CV_LOG_INFO(NULL, "UI: Framebuffer's offsets (x, y), line length: " 
      << fbXOffset << " " << fbYOffset << " " << fbLineLength);
    
    // MAP FB TO MEMORY
    fbScreenSize = max((__u32)fbWidth , varInfo.xres_virtual) * 
                   max((__u32)fbHeight, varInfo.yres_virtual) * 
                   fbBitsPerPixel / 8;
                 
    fbPointer = (unsigned char*)
      mmap(0, fbScreenSize, PROT_READ | PROT_WRITE, MAP_SHARED, 
        fbID, 0);
        
    if (fbPointer == MAP_FAILED) {
      CV_LOG_ERROR(NULL, "UI: can't mmap framebuffer");
      return;
    }

    if(fbBitsPerPixel != 32) {
      CV_LOG_WARNING(NULL, "UI: Framebuffer with bits per pixel = " 
        << fbBitsPerPixel << " is not supported" );
      return;
    }

    backgroundBuff = Mat(fbHeight, fbWidth, CV_8UC4);
    int cnt_channel = 4;
    for (int y = fbYOffset; y < backgroundBuff.rows + fbYOffset; y++)
    {
      std::memcpy(backgroundBuff.ptr<cv::Vec4b>(y - fbYOffset), 
                  fbPointer + y * fbLineLength + fbXOffset, 
                  backgroundBuff.cols * cnt_channel);
    }
  }
  
  FramebufferBackend::~FramebufferBackend()
  {
    CV_LOG_INFO(NULL, "UI: FramebufferBackend::~FramebufferBackend()");
    if(fbID == -1) return;
    
    // RESTORE BACKGROUNG
    int cnt_channel = 4;
    for (int y = fbYOffset; y < backgroundBuff.rows + fbYOffset; y++)
    {
      std::memcpy(fbPointer + y * fbLineLength + fbXOffset, 
                  backgroundBuff.ptr<cv::Vec4b>(y - fbYOffset), 
                  backgroundBuff.cols * cnt_channel);
    }

    if (fbPointer != MAP_FAILED) {
      munmap(fbPointer, fbScreenSize);
    }
    close(fbID);

  }

  void FramebufferBackend::destroyAllWindows() {
    CV_LOG_INFO(NULL, "UI: FramebufferBackend::destroyAllWindows()");
  }

  // namedWindow
  std::shared_ptr<UIWindow> FramebufferBackend::createWindow(
      const std::string& winname,
      int flags)
  {
    CV_LOG_INFO(NULL, "UI: FramebufferBackend::createWindow(" 
      << winname << ", " << flags << ")");
    return std::make_shared<FramebufferWindow>(*this);
  }

  void FramebufferBackend::initTermios(int echo, int wait) 
  {
    tcgetattr(0, &old);               // grab old terminal i/o settings
    current = old;                    // make new settings same as old settings
    current.c_lflag &= ~ICANON;       // disable buffered i/o 
    current.c_lflag &= ~ISIG;
    current.c_cc[VMIN]=wait;
    if (echo) {
        current.c_lflag |= ECHO;      // set echo mode
    } else {
        current.c_lflag &= ~ECHO;     // set no echo mode
    }
    tcsetattr(0, TCSANOW, &current);  // use these new terminal i/o settings now
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
    if(ch < 0) rewind(stdin);
    resetTermios();
    return ch;
  }
  bool FramebufferBackend::kbhit()
  {
    int byteswaiting=0;
    initTermios(0, 1);
    if ( ioctl(0, FIONREAD, &byteswaiting) < 0)
    {
      CV_LOG_ERROR(NULL, "UI: Framebuffer ERR byteswaiting" );
    }
    resetTermios();
    
    return byteswaiting > 0;
  }

  int FramebufferBackend::waitKeyEx(int delay) 
  {
    CV_LOG_INFO(NULL, "UI: FramebufferBackend::waitKeyEx(int delay = " << delay << ")");

    int code = -1;

    if(delay == 0)
    {
      int ch = getch_(0, 1);
      CV_LOG_INFO(NULL, "UI: FramebufferBackend::getch_() take value = " << (int)ch);
      code = ch;
      
      while((ch = getch_(0, 0))>=0)
      {
        CV_LOG_INFO(NULL, "UI: FramebufferBackend::getch_() take value = " 
          << (int)ch << " (additional code on <stdin>)");
        code = ch;
      }
    } else {
      if(delay > 0)
      {
        bool f_kbhit = false;
        while(!(f_kbhit = kbhit()) && (delay > 0))
        {
          delay -= 10;
          usleep(10000);
        }          
        if(f_kbhit)
        {
          CV_LOG_INFO(NULL, "UI: FramebufferBackend kbhit is True ");

          int ch = getch_(0, 1);
          CV_LOG_INFO(NULL, "UI: FramebufferBackend::getch_() take value = " << (int)ch);
          code = ch;
          
          while((ch = getch_(0, 0))>=0)
          {
            CV_LOG_INFO(NULL, "UI: FramebufferBackend::getch_() take value = " 
              << (int)ch << " (additional code on <stdin>)");
            code = ch;
          }
        }

      }
    }
    
    CV_LOG_INFO(NULL, "UI: FramebufferBackend::waitKeyEx() result code = " << code);
    return code; 
  }
  
  int FramebufferBackend::pollKey()
  {
    CV_LOG_INFO(NULL, "UI: FramebufferBackend::pollKey()");
    return 0;
  }


}
}
