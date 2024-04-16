#ifndef OPENCV_HIGHGUI_WINDOWS_FRAMEBUFFER_HPP
#define OPENCV_HIGHGUI_WINDOWS_FRAMEBUFFER_HPP

#include "precomp.hpp"
#include "backend.hpp"

#include <linux/fb.h>
#include <linux/input.h>


namespace cv { namespace highgui_backend {

class CV_EXPORTS FramebufferWindow : public UIWindow
{
  fb_var_screeninfo var_info;
  fb_fix_screeninfo fix_info;

  std::string FB_ID;
  
  int fb_open_and_get_info();
  int framebuffrer_id;
  
  int fb_w;
  int fb_h;
  int y_offset;
  int x_offset;
  int bpp;
  int line_length;
  long int screensize;
  unsigned char* fbPointer;
  
  Mat backgroundBuff;
  
public:
  FramebufferWindow();
  virtual ~FramebufferWindow(){}

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
public:
  virtual ~FramebufferBackend(){}

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
