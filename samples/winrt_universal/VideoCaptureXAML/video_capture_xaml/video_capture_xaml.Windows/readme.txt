notes for OpenCV WinRT implementation:

cvMain() in main.cpp
  implements the image processing and OpenCV app control
  it is running on a background thread, started by XAML
  see file main.cpp
  in the Application project

class VideoCapture_WinRT:
  implements the IVideoCapture interface from OpenCV
  video is initialized and frames are grabbed on the UI thread
  see files cap_winrt.hpp/cpp

class HighguiBridge, a singleton
  implements the OpenCV Highgui functions for XAML (limited at this time),
  and also bridges to the UI thread functions for XAML and video operations.
  see files cap_winrt_highgui.hpp/cpp

class Video, a singleton
  encapsulates the Media Foundation interface needed for video initialization and grabbing.
  called through Highgui and XAML, only on the UI thread
  see files cap_winrt_video.hpp/cpp

threading:
  requests from the OpenCV bg thread to the Video/XAML UI thread
  are made through HighguiBridge::requestForUIthreadAsync(), which uses
  the "progress reporter" method provided by the WinRT class
  IAsyncActionWithProgress.  Also the bg thread is started by create_async().
  see file MainPage.xaml.cpp
  in the Application project
