#ifndef OPENCV_HIGHGUI_WINDOWS_FRAMEBUFFER_HPP
#define OPENCV_HIGHGUI_WINDOWS_FRAMEBUFFER_HPP

#include "precomp.hpp"
#include "backend.hpp"

#include <linux/fb.h>
#include <linux/input.h>

#include <termios.h>

namespace cv { namespace highgui_backend {

class FramebufferBackend;
class CV_EXPORTS FramebufferWindow : public UIWindow
{
  FramebufferBackend &backend;
  std::string FB_ID;
  Rect windowRect;

public:
  FramebufferWindow(FramebufferBackend &backend);
  virtual ~FramebufferWindow();

  virtual void imshow(InputArray image)override;

  virtual double getProperty(int prop) const override;
  virtual bool setProperty(int prop, double value)override;

  virtual void resize(int width, int height)override;
  virtual void move(int x, int y)override;

  virtual Rect getImageRect() const override;

  virtual void setTitle(const std::string& title)override;

  virtual void setMouseCallback(MouseCallback onMouse, void* userdata /*= 0*/) override ;

  virtual std::shared_ptr<UITrackbar> createTrackbar(
      const std::string& name,
      int count,
      TrackbarCallback onChange /*= 0*/,
      void* userdata /*= 0*/
  )override;

  virtual std::shared_ptr<UITrackbar> findTrackbar(const std::string& name)override;
  
  virtual const std::string& getID() const override;

  virtual bool isActive() const override;

  virtual void destroy() override;
};  // FramebufferWindow

class CV_EXPORTS FramebufferBackend: public UIBackend
{
  int OpenInputEvent();
  int eventKey;

  struct termios old, current;

  void initTermios(int echo, int wait);
  void resetTermios(void);
  int getch_(int echo, int wait);
  bool kbhit();
  

  fb_var_screeninfo varInfo;
  fb_fix_screeninfo fixInfo;
  int fbWidth;
  int fbHeight;
  int fbXOffset;
  int fbYOffset;
  int fbBitsPerPixel;
  int fbLineLength;
  long int fbScreenSize;
  unsigned char* fbPointer;
  Mat backgroundBuff;

  
  int fbOpenAndGetInfo();
  int fbID;
  
public:

  fb_var_screeninfo &getVarInfo();
  fb_fix_screeninfo &getFixInfo();
  int getFramebuffrerID();
  int getFBWidth();
  int getFBHeight();
  int getFBXOffset();
  int getFBYOffset();
  int getFBBitsPerPixel();
  int getFBLineLength();
  unsigned char* getFBPointer();
  Mat& getBackgroundBuff();

  FramebufferBackend();

  virtual ~FramebufferBackend();

  virtual void destroyAllWindows()override;

  // namedWindow
  virtual std::shared_ptr<UIWindow> createWindow(
      const std::string& winname,
      int flags
  );

  virtual int waitKeyEx(int delay /*= 0*/)override;
  virtual int pollKey() override; 
};

}

}


#endif
