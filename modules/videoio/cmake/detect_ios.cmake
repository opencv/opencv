if(APPLE AND IOS)
  set(HAVE_CAP_IOS TRUE)
  set(libs
    "-framework Accelerate"
    "-framework AVFoundation"
    "-framework CoreGraphics"
    "-framework CoreImage"
    "-framework CoreMedia"
    "-framework CoreVideo"
    "-framework QuartzCore"
    "-framework UIKit")
  ocv_add_external_target(cap_ios "" "${libs}" "HAVE_CAP_IOS")
endif()
