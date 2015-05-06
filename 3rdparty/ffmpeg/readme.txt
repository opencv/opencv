* On Linux and other Unix flavors OpenCV uses default or user-built ffmpeg/libav libraries.
  If user builds ffmpeg/libav from source and wants OpenCV to stay BSD library, not GPL/LGPL,
  he/she should use --enabled-shared configure flag and make sure that no GPL components are
  enabled (some notable examples are x264 (H264 encoder) and libac3 (Dolby AC3 audio codec)).
  See https://www.ffmpeg.org/legal.html for details.
  
  If you want to play very safe and do not want to use FFMPEG at all, regardless of whether it's installed on
  your system or not, configure and build OpenCV using CMake with WITH_FFMPEG=OFF flag. OpenCV will then use
  AVFoundation (OSX), GStreamer (Linux) or other available backends supported by opencv_videoio module.
  
  There is also our self-contained motion jpeg codec, which you can use without any worries.
  It handles CV_FOURCC('M', 'J', 'P', 'G') streams within an AVI container (".avi").
  
* On Windows OpenCV uses pre-built ffmpeg binaries, built with proper flags (without GPL components) and
  wrapped with simple, stable OpenCV-compatible API.
  The binaries are opencv_ffmpeg.dll (version for 32-bit Windows) and
  opencv_ffmpeg_64.dll (version for 64-bit Windows).
  
  See build_win32.txt for the build instructions, if you want to rebuild opencv_ffmpeg*.dll from scratch.

  The pre-built opencv_ffmpeg*.dll is:
  * LGPL library, not BSD libraries.
  * Loaded at runtime by opencv_videoio module.
    If it succeeds, ffmpeg can be used to decode/encode videos;
    otherwise, other API is used.

  If LGPL/GPL software can not be supplied with your OpenCV-based product, simply exclude
  opencv_ffmpeg*.dll from your distribution; OpenCV will stay fully functional except for the ability to
  decode/encode videos using FFMPEG (though, it may still be able to do that using other API,
  such as Video for Windows, Windows Media Foundation or our self-contained motion jpeg codec).
  
  See license.txt for the FFMPEG copyright notice and the licensing terms.
