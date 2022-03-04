#!/bin/sh
# https://crascit.com/2016/04/09/using-ccache-with-cmake/

# Xcode generator doesn't include the compiler as the
# first argument, Ninja and Makefiles do. Handle both cases.
if [[ "$1" = "${CMAKE_C_COMPILER}" ]] ; then
    shift
fi

export CCACHE_CPP2=true
exec "${CCACHE_PROGRAM}" "${CMAKE_C_COMPILER}" "$@"
