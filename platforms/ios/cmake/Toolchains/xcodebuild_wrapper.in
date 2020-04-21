#!/bin/sh

# Force 'Debug' configuration
# Details: https://github.com/opencv/opencv/issues/13856
if [[ "$@" =~ "-project CMAKE_TRY_COMPILE.xcodeproj" && -z "${OPENCV_SKIP_XCODEBUILD_FORCE_TRYCOMPILE_DEBUG}" ]]; then
  ARGS=()
  for ((i=1; i<=$#; i++))
  do
    arg=${!i}
    ARGS+=("$arg")
    if [[ "$arg" == "-configuration" ]]; then
      ARGS+=("Debug")
      i=$(($i+1))
    fi
  done
  set -- "${ARGS[@]}"
fi

@CMAKE_MAKE_PROGRAM@ @XCODEBUILD_EXTRA_ARGS@ $*
