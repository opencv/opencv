if(APPLE)
  set(HAVE_AVFOUNDATION TRUE)
  if(IOS)
    set(libs "-framework AVFoundation" "-framework QuartzCore")
  else()
    set(libs
      "-framework Cocoa"
      "-framework Accelerate"
      "-framework AVFoundation"
      "-framework CoreGraphics"
      "-framework CoreMedia"
      "-framework CoreVideo"
      "-framework QuartzCore")
  endif()
  ocv_add_external_target(avfoundation "" "${libs}" "HAVE_AVFOUNDATION")
endif()

set(HAVE_AVFOUNDATION ${HAVE_AVFOUNDATION} PARENT_SCOPE)
