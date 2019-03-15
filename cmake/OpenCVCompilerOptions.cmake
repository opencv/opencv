if("${CMAKE_CXX_COMPILER};${CMAKE_C_COMPILER};${CMAKE_CXX_COMPILER_LAUNCHER}" MATCHES "ccache")
  set(CMAKE_COMPILER_IS_CCACHE 1)  # TODO: FIXIT Avoid setting of CMAKE_ variables
  set(OPENCV_COMPILER_IS_CCACHE 1)
endif()
function(access_CMAKE_COMPILER_IS_CCACHE)
  if(NOT OPENCV_SUPPRESS_DEPRECATIONS)
    message(WARNING "DEPRECATED: CMAKE_COMPILER_IS_CCACHE is replaced to OPENCV_COMPILER_IS_CCACHE.")
  endif()
endfunction()
variable_watch(CMAKE_COMPILER_IS_CCACHE access_CMAKE_COMPILER_IS_CCACHE)
if(ENABLE_CCACHE AND NOT OPENCV_COMPILER_IS_CCACHE AND NOT CMAKE_GENERATOR MATCHES "Xcode")
  # This works fine with Unix Makefiles and Ninja generators
  find_host_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    message(STATUS "Looking for ccache - found (${CCACHE_PROGRAM})")
    get_property(__OLD_RULE_LAUNCH_COMPILE GLOBAL PROPERTY RULE_LAUNCH_COMPILE)
    if(__OLD_RULE_LAUNCH_COMPILE)
      message(STATUS "Can't replace CMake compiler launcher")
    else()
      set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
      # NOTE: Actually this check doesn't work as expected.
      # "RULE_LAUNCH_COMPILE" is ignored by CMake during try_compile() step.
      # ocv_check_compiler_flag(CXX "" IS_CCACHE_WORKS)
      set(IS_CCACHE_WORKS 1)
      if(IS_CCACHE_WORKS)
        set(OPENCV_COMPILER_IS_CCACHE 1)
      else()
        message(STATUS "Unable to compile program with enabled ccache, reverting...")
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${__OLD_RULE_LAUNCH_COMPILE}")
      endif()
    endif()
  else()
    message(STATUS "Looking for ccache - not found")
  endif()
endif()

if((CV_CLANG AND NOT CMAKE_GENERATOR MATCHES "Xcode")  # PCH has no support for Clang
    OR OPENCV_COMPILER_IS_CCACHE
)
  set(ENABLE_PRECOMPILED_HEADERS OFF CACHE BOOL "" FORCE)
endif()

macro(add_extra_compiler_option option)
  ocv_check_flag_support(CXX "${option}" _varname "${OPENCV_EXTRA_CXX_FLAGS} ${ARGN}")
  if(${_varname})
    set(OPENCV_EXTRA_CXX_FLAGS "${OPENCV_EXTRA_CXX_FLAGS} ${option}")
  endif()

  ocv_check_flag_support(C "${option}" _varname "${OPENCV_EXTRA_C_FLAGS} ${ARGN}")
  if(${_varname})
    set(OPENCV_EXTRA_C_FLAGS "${OPENCV_EXTRA_C_FLAGS} ${option}")
  endif()
endmacro()

macro(add_extra_compiler_option_force option)
  set(OPENCV_EXTRA_CXX_FLAGS "${OPENCV_EXTRA_CXX_FLAGS} ${option}")
  set(OPENCV_EXTRA_C_FLAGS "${OPENCV_EXTRA_C_FLAGS} ${option}")
endmacro()


# Gets environment variable and puts its value to the corresponding preprocessor definition
# Useful for WINRT that has no access to environment variables
macro(add_env_definitions option)
  set(value $ENV{${option}})
  if("${value}" STREQUAL "")
    message(WARNING "${option} environment variable is empty. Please set it to appropriate location to get correct results")
  else()
    string(REPLACE "\\" "\\\\" value ${value})
  endif()
  add_definitions("-D${option}=\"${value}\"")
endmacro()

if(NOT MSVC)
  # OpenCV fails some tests when 'char' is 'unsigned' by default
  add_extra_compiler_option(-fsigned-char)
endif()

if(CV_ICC AND NOT ENABLE_FAST_MATH)
  if(MSVC)
    add_extra_compiler_option("/fp:precise")
  else()
    add_extra_compiler_option("-fp-model precise")
  endif()
endif()

if(CV_GCC OR CV_CLANG)
  # High level of warnings.
  add_extra_compiler_option(-W)
  if (NOT MSVC)
    # clang-cl interprets -Wall as MSVC would: -Weverything, which is more than
    # we want.
    add_extra_compiler_option(-Wall)
  endif()
  add_extra_compiler_option(-Werror=return-type)
  add_extra_compiler_option(-Werror=non-virtual-dtor)
  add_extra_compiler_option(-Werror=address)
  add_extra_compiler_option(-Werror=sequence-point)
  add_extra_compiler_option(-Wformat)
  add_extra_compiler_option(-Werror=format-security -Wformat)
  add_extra_compiler_option(-Wmissing-declarations)
  add_extra_compiler_option(-Wmissing-prototypes)
  add_extra_compiler_option(-Wstrict-prototypes)
  add_extra_compiler_option(-Wundef)
  add_extra_compiler_option(-Winit-self)
  add_extra_compiler_option(-Wpointer-arith)
  if(NOT (CV_GCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.0"))
    add_extra_compiler_option(-Wshadow)  # old GCC emits warnings for variables + methods combination
  endif()
  add_extra_compiler_option(-Wsign-promo)
  add_extra_compiler_option(-Wuninitialized)
  add_extra_compiler_option(-Winit-self)
  if(HAVE_CXX11)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT ENABLE_PRECOMPILED_HEADERS)
      add_extra_compiler_option(-Wsuggest-override)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      add_extra_compiler_option(-Winconsistent-missing-override)
    endif()
  endif()

  if(ENABLE_NOISY_WARNINGS)
    add_extra_compiler_option(-Wcast-align)
    add_extra_compiler_option(-Wstrict-aliasing=2)
  else()
    add_extra_compiler_option(-Wno-delete-non-virtual-dtor)
    add_extra_compiler_option(-Wno-unnamed-type-template-args)
    add_extra_compiler_option(-Wno-comment)
    if(NOT OPENCV_SKIP_IMPLICIT_FALLTHROUGH
        AND NOT " ${CMAKE_CXX_FLAGS} ${OPENCV_EXTRA_FLAGS} ${OPENCV_EXTRA_CXX_FLAGS}" MATCHES "implicit-fallthrough"
        AND (CV_GCC AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0.0)
    )
      add_extra_compiler_option(-Wimplicit-fallthrough=3)
    endif()
    if(CV_GCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
      add_extra_compiler_option(-Wno-strict-overflow) # Issue appears when compiling surf.cpp from opencv_contrib/modules/xfeatures2d
    endif()
    if(CV_GCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
      add_extra_compiler_option(-Wno-missing-field-initializers)  # GCC 4.x emits warnings about {}, fixed in GCC 5+
    endif()
  endif()
  add_extra_compiler_option(-fdiagnostics-show-option)

  # The -Wno-long-long is required in 64bit systems when including system headers.
  if(X86_64)
    add_extra_compiler_option(-Wno-long-long)
  endif()

  # We need pthread's
  if(UNIX AND NOT ANDROID AND NOT (APPLE AND CV_CLANG)) # TODO
    add_extra_compiler_option(-pthread)
  endif()

  if(CV_CLANG)
    add_extra_compiler_option(-Qunused-arguments)
  endif()

  if(OPENCV_WARNINGS_ARE_ERRORS)
    add_extra_compiler_option(-Werror)
  endif()

  if(APPLE)
    add_extra_compiler_option(-Wno-semicolon-before-method-body)
  endif()

  # Other optimizations
  if(ENABLE_OMIT_FRAME_POINTER)
    add_extra_compiler_option(-fomit-frame-pointer)
  elseif(DEFINED ENABLE_OMIT_FRAME_POINTER)
    add_extra_compiler_option(-fno-omit-frame-pointer)
  endif()
  if(ENABLE_FAST_MATH)
    add_extra_compiler_option(-ffast-math)
  endif()

  # Profiling?
  if(ENABLE_PROFILING)
    add_extra_compiler_option("-pg -g")
    # turn off incompatible options
    foreach(flags CMAKE_CXX_FLAGS CMAKE_C_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_C_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG CMAKE_C_FLAGS_DEBUG
                  OPENCV_EXTRA_FLAGS_RELEASE OPENCV_EXTRA_FLAGS_DEBUG OPENCV_EXTRA_C_FLAGS OPENCV_EXTRA_CXX_FLAGS)
      string(REPLACE "-fomit-frame-pointer" "" ${flags} "${${flags}}")
      string(REPLACE "-ffunction-sections" "" ${flags} "${${flags}}")
      string(REPLACE "-fdata-sections" "" ${flags} "${${flags}}")
    endforeach()
  else()
    if(MSVC)
      # TODO: Clang/C2 is not supported
    elseif(((IOS OR ANDROID) AND NOT BUILD_SHARED_LIBS) AND NOT OPENCV_FORCE_FUNCTIONS_SECTIONS)
      # don't create separate sections for functions/data, reduce package size
    else()
      # Remove unreferenced functions: function level linking
      add_extra_compiler_option(-ffunction-sections)
      add_extra_compiler_option(-fdata-sections)
      if(NOT OPENCV_SKIP_GC_SECTIONS)
        if(APPLE)
          set(OPENCV_EXTRA_EXE_LINKER_FLAGS "${OPENCV_EXTRA_EXE_LINKER_FLAGS} -Wl,-dead_strip")
          set(OPENCV_EXTRA_SHARED_LINKER_FLAGS "${OPENCV_EXTRA_SHARED_LINKER_FLAGS} -Wl,-dead_strip")
        else()
          set(OPENCV_EXTRA_EXE_LINKER_FLAGS "${OPENCV_EXTRA_EXE_LINKER_FLAGS} -Wl,--gc-sections")
          set(OPENCV_EXTRA_SHARED_LINKER_FLAGS "${OPENCV_EXTRA_SHARED_LINKER_FLAGS} -Wl,--gc-sections")
        endif()
      endif()
    endif()
  endif()

  if(ENABLE_COVERAGE)
    set(OPENCV_EXTRA_C_FLAGS "${OPENCV_EXTRA_C_FLAGS} --coverage")
    set(OPENCV_EXTRA_CXX_FLAGS "${OPENCV_EXTRA_CXX_FLAGS} --coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
  endif()

  if(ENABLE_INSTRUMENTATION)
    if(NOT HAVE_CXX11)
      message(WARNING "ENABLE_INSTRUMENTATION requires C++11 support")
    endif()
    set(WITH_VTK OFF) # There are issues with VTK 6.0
  endif()

  if(ENABLE_LTO)
    add_extra_compiler_option(-flto)
  endif()
  if(ENABLE_THIN_LTO)
    add_extra_compiler_option(-flto=thin)
  endif()

  set(OPENCV_EXTRA_FLAGS_RELEASE "${OPENCV_EXTRA_FLAGS_RELEASE} -DNDEBUG")
  if(NOT " ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} ${OPENCV_EXTRA_FLAGS_DEBUG} " MATCHES "-O")
    set(OPENCV_EXTRA_FLAGS_DEBUG "${OPENCV_EXTRA_FLAGS_DEBUG} -O0")
  endif()
  set(OPENCV_EXTRA_FLAGS_DEBUG "${OPENCV_EXTRA_FLAGS_DEBUG} -DDEBUG -D_DEBUG")

  if(BUILD_WITH_DEBUG_INFO)
    # https://gcc.gnu.org/onlinedocs/gcc/Debugging-Options.html
    # '-g' is equal to '-g2', '-g1' produces minimal information, enough for making backtraces
    ocv_update(OPENCV_DEBUG_OPTION "-g1")
    if(NOT " ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} ${OPENCV_EXTRA_CXX_FLAGS} ${OPENCV_EXTRA_FLAGS_RELEASE}" MATCHES " -g")
      set(OPENCV_EXTRA_FLAGS_RELEASE "${OPENCV_EXTRA_FLAGS_RELEASE} ${OPENCV_DEBUG_OPTION}")
    endif()
  endif()
endif()

if(MSVC)
  #TODO Code refactoring is required to resolve security warnings
  #if(NOT ENABLE_BUILD_HARDENING)
    set(OPENCV_EXTRA_FLAGS "${OPENCV_EXTRA_FLAGS} /D _CRT_SECURE_NO_DEPRECATE /D _CRT_NONSTDC_NO_DEPRECATE /D _SCL_SECURE_NO_WARNINGS")
  #endif()

  if(BUILD_WITH_DEBUG_INFO)
    set(OPENCV_EXTRA_FLAGS_RELEASE "${OPENCV_EXTRA_FLAGS_RELEASE} /Zi")
    set(OPENCV_EXTRA_EXE_LINKER_FLAGS_RELEASE "${OPENCV_EXTRA_EXE_LINKER_FLAGS_RELEASE} /debug")
    set(OPENCV_EXTRA_SHARED_LINKER_FLAGS_RELEASE "${OPENCV_EXTRA_SHARED_LINKER_FLAGS_RELEASE} /debug")
  endif()

  # Remove unreferenced functions: function level linking
  set(OPENCV_EXTRA_FLAGS "${OPENCV_EXTRA_FLAGS} /Gy")
  if(NOT MSVC_VERSION LESS 1400)
    set(OPENCV_EXTRA_FLAGS "${OPENCV_EXTRA_FLAGS} /bigobj")
  endif()

  if(OPENCV_WARNINGS_ARE_ERRORS)
    set(OPENCV_EXTRA_FLAGS "${OPENCV_EXTRA_FLAGS} /WX")
  endif()

  if(ENABLE_LTO)
    set(OPENCV_EXTRA_FLAGS_RELEASE "${OPENCV_EXTRA_FLAGS_RELEASE} /GL")
    set(OPENCV_EXTRA_EXE_LINKER_FLAGS_RELEASE "${OPENCV_EXTRA_EXE_LINKER_FLAGS_RELEASE} /LTCG")
  endif()

  if(NOT MSVC_VERSION LESS 1800 AND NOT CMAKE_GENERATOR MATCHES "Visual Studio")
    set(OPENCV_EXTRA_C_FLAGS "${OPENCV_EXTRA_C_FLAGS} /FS")
    set(OPENCV_EXTRA_CXX_FLAGS "${OPENCV_EXTRA_CXX_FLAGS} /FS")
  endif()
endif()

# Adding additional using directory for WindowsPhone 8.0 to get Windows.winmd properly
if(WINRT_PHONE AND WINRT_8_0)
  set(OPENCV_EXTRA_CXX_FLAGS "${OPENCV_EXTRA_CXX_FLAGS} /AI\$(WindowsSDK_MetadataPath)")
endif()

include(cmake/OpenCVCompilerOptimizations.cmake)
if(COMMAND ocv_compiler_optimization_options)
  ocv_compiler_optimization_options()
endif()
if(COMMAND ocv_compiler_optimization_options_finalize)
  ocv_compiler_optimization_options_finalize()
endif()

# set default visibility to hidden
if((CV_GCC OR CV_CLANG)
    AND NOT MSVC
    AND NOT OPENCV_SKIP_VISIBILITY_HIDDEN
    AND NOT " ${CMAKE_CXX_FLAGS} ${OPENCV_EXTRA_FLAGS} ${OPENCV_EXTRA_CXX_FLAGS}" MATCHES " -fvisibility")
  add_extra_compiler_option(-fvisibility=hidden)
  add_extra_compiler_option(-fvisibility-inlines-hidden)
endif()

# workaround gcc bug for aligned ld/st
# https://github.com/opencv/opencv/issues/13211
if((PPC64LE AND NOT CMAKE_CROSSCOMPILING) OR OPENCV_FORCE_COMPILER_CHECK_VSX_ALIGNED)
  ocv_check_runtime_flag("${CPU_BASELINE_FLAGS}" "OPENCV_CHECK_VSX_ALIGNED" "${OpenCV_SOURCE_DIR}/cmake/checks/runtime/cpu_vsx_aligned.cpp")
  if(NOT OPENCV_CHECK_VSX_ALIGNED)
    add_extra_compiler_option_force(-DCV_COMPILER_VSX_BROKEN_ALIGNED)
  endif()
endif()

# combine all "extra" options
if(NOT OPENCV_SKIP_EXTRA_COMPILER_FLAGS)
  set(CMAKE_C_FLAGS           "${CMAKE_C_FLAGS} ${OPENCV_EXTRA_FLAGS} ${OPENCV_EXTRA_C_FLAGS}")
  set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} ${OPENCV_EXTRA_FLAGS} ${OPENCV_EXTRA_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPENCV_EXTRA_FLAGS_RELEASE}")
  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} ${OPENCV_EXTRA_FLAGS_RELEASE}")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${OPENCV_EXTRA_FLAGS_DEBUG}")
  set(CMAKE_C_FLAGS_DEBUG     "${CMAKE_C_FLAGS_DEBUG} ${OPENCV_EXTRA_FLAGS_DEBUG}")
  set(CMAKE_EXE_LINKER_FLAGS         "${CMAKE_EXE_LINKER_FLAGS} ${OPENCV_EXTRA_EXE_LINKER_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${OPENCV_EXTRA_EXE_LINKER_FLAGS_RELEASE}")
  set(CMAKE_EXE_LINKER_FLAGS_DEBUG   "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${OPENCV_EXTRA_EXE_LINKER_FLAGS_DEBUG}")
  set(CMAKE_SHARED_LINKER_FLAGS         "${CMAKE_SHARED_LINKER_FLAGS} ${OPENCV_EXTRA_SHARED_LINKER_FLAGS}")
  set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} ${OPENCV_EXTRA_SHARED_LINKER_FLAGS_RELEASE}")
  set(CMAKE_SHARED_LINKER_FLAGS_DEBUG   "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} ${OPENCV_EXTRA_SHARED_LINKER_FLAGS_DEBUG}")
endif()

if(MSVC)
  if(NOT ENABLE_NOISY_WARNINGS)
    if(MSVC_VERSION EQUAL 1400)
      ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4510 /wd4610 /wd4312 /wd4201 /wd4244 /wd4328 /wd4267)
    endif()
  endif()

  foreach(flags CMAKE_C_FLAGS CMAKE_C_FLAGS_RELEASE CMAKE_C_FLAGS_RELEASE CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
    string(REPLACE "/Zm1000" "" ${flags} "${${flags}}")
  endforeach()

  # Enable 'extern "C"' and asynchronous (division by zero, access violation) exceptions
  if(NOT OPENCV_SKIP_MSVC_EXCEPTIONS_FLAG)
    foreach(flags CMAKE_C_FLAGS CMAKE_C_FLAGS_RELEASE CMAKE_C_FLAGS_RELEASE CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
      string(REGEX REPLACE " /EH[^ ]* " " " ${flags} " ${${flags}}")
    endforeach()
    if(NOT " ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS_DEBUG}" MATCHES " /EH")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHa")
    endif()
  endif()

  if(NOT ENABLE_NOISY_WARNINGS)
    ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127) # conditional expression is constant
    ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4251) # class 'std::XXX' needs to have dll-interface to be used by clients of YYY
    ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4324) # 'struct_name' : structure was padded due to __declspec(align())
    ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4275) # non dll-interface class 'std::exception' used as base for dll-interface class 'cv::Exception'
    ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4512) # Assignment operator could not be generated
    ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4589) # Constructor of abstract class 'cv::ORB' ignores initializer for virtual base class 'cv::Algorithm'
  endif()

  if(CV_ICC AND NOT ENABLE_NOISY_WARNINGS)
    foreach(flags CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG CMAKE_C_FLAGS CMAKE_C_FLAGS_RELEASE CMAKE_C_FLAGS_DEBUG)
      string(REGEX REPLACE "( |^)/W[0-9]+( |$)" "\\1\\2" ${flags} "${${flags}}")
    endforeach()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qwd673") # PCH warning
  endif()
endif()

if(APPLE AND NOT CMAKE_CROSSCOMPILING AND NOT DEFINED ENV{LDFLAGS} AND EXISTS "/usr/local/lib")
  link_directories("/usr/local/lib")
endif()

if(ENABLE_BUILD_HARDENING)
  include(${CMAKE_CURRENT_LIST_DIR}/OpenCVCompilerDefenses.cmake)
endif()

if(MSVC)
  include(cmake/OpenCVCRTLinkage.cmake)
  add_definitions(-D_VARIADIC_MAX=10)
endif()

# Enable compiler options for OpenCV modules/apps/samples only (ignore 3rdparty)
macro(ocv_add_modules_compiler_options)
  if(MSVC AND NOT OPENCV_SKIP_MSVC_W4_OPTION)
    foreach(flags CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
      string(REPLACE "/W3" "/W4" ${flags} "${${flags}}")
    endforeach()
  endif()
endmacro()
