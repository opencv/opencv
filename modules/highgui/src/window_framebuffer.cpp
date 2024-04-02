#include "window_framebuffer.hpp"

#include "opencv2/core/utils/logger.hpp"

namespace cv { namespace highgui_backend {
  
  std::shared_ptr<UIBackend> createUIBackendFramebuffer()
  {
    return std::make_shared<FramebufferBackend>();
  }


  void FramebufferWindow::imshow(InputArray image){
    std::cout  << "FramebufferWindow::imshow(InputArray image)" << std::endl;
  }

  double FramebufferWindow::getProperty(int prop) const{
    std::cout  << "FramebufferWindow::getProperty(int prop:" << prop <<")"<< std::endl; 
    return 0.0;
  }
  bool FramebufferWindow::setProperty(int prop, double value) {
    std::cout  << "FramebufferWindow::setProperty(int prop "<< prop <<", double value "<<value<<")" << std::endl; 
    return false;
  }

  void FramebufferWindow::resize(int width, int height){
    std::cout  << "FramebufferWindow::resize(int width "<< width <<", int height "<< height <<")" << std::endl;
  }
  void FramebufferWindow::move(int x, int y) {
    std::cout  << "FramebufferWindow::move(int x "<< x <<", int y "<< y <<")" << std::endl;
  }

  Rect FramebufferWindow::getImageRect() const {
    std::cout  << "FramebufferWindow::getImageRect()" << std::endl; 
    return Rect(10,10,100,100);
  }

  void FramebufferWindow::setTitle(const std::string& title) {
    std::cout  << "FramebufferWindow::setTitle(... "<< title <<")" << std::endl;
  }

  void FramebufferWindow::setMouseCallback(MouseCallback onMouse, void* userdata ){
    std::cout  << "FramebufferWindow::setMouseCallback(...)" << std::endl;
  }

  std::shared_ptr<UITrackbar> FramebufferWindow::createTrackbar(
      const std::string& name,
      int count,
      TrackbarCallback onChange,
      void* userdata
  ){
    return nullptr;
  }

  std::shared_ptr<UITrackbar> FramebufferWindow::findTrackbar(const std::string& name){
    return nullptr;
  }
  
  const std::string& FramebufferWindow::getID() const  { 
    std::cout  << "getID())" << std::endl; return FB_ID;
  }

  bool FramebufferWindow::isActive() const {
    std::cout  << "isActive()" << std::endl; 
    return true;
  }

  void FramebufferWindow::destroy() {
    std::cout  << "destroy()" << std::endl;
  }

  void FramebufferBackend::destroyAllWindows() {
    std::cout  << "destroyAllWindows()" << std::endl;
  }

  // namedWindow
  std::shared_ptr<UIWindow> FramebufferBackend::createWindow(
      const std::string& winname,
      int flags
  ){
    std::cout  << "FramebufferBackend::createWindow("<< winname <<", "<<flags<<")" << std::endl;
    return std::make_shared<FramebufferWindow>();
  }

  int FramebufferBackend::waitKeyEx(int delay) {
    std::cout  << "FramebufferBackend::waitKeyEx(int delay "<< delay <<")" << std::endl; 
    return 0; 
  }
  int FramebufferBackend::pollKey()  {
    std::cout  << "FramebufferBackend::pollKey()" << std::endl; 
    return 0;
  }


}
}
