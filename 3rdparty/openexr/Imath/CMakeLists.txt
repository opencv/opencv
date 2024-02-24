# SPDX-License-Identifier: BSD-3-Clause
# Copyright Contributors to the OpenEXR Project.

imath_define_library(Imath
  PRIV_EXPORT IMATH_EXPORTS
  CURDIR ${CMAKE_CURRENT_SOURCE_DIR}
  SOURCES
    half.cpp
    ImathColorAlgo.cpp
    ImathFun.cpp
    ImathMatrixAlgo.cpp
    ImathRandom.cpp
    toFloat.h
  HEADERS
    half.h
    halfFunction.h
    halfLimits.h
    ImathBox.h
    ImathBoxAlgo.h
    ImathColor.h
    ImathColorAlgo.h
    ImathEuler.h
    ImathExport.h
    ImathForward.h
    ImathFrame.h
    ImathFrustum.h
    ImathFrustumTest.h
    ImathFun.h
    ImathGL.h
    ImathGLU.h
    ImathInt64.h
    ImathInterval.h
    ImathLine.h
    ImathLineAlgo.h
    ImathMath.h
    ImathMatrix.h
    ImathMatrixAlgo.h
    ImathNamespace.h
    ImathPlane.h
    ImathPlatform.h
    ImathQuat.h
    ImathRandom.h
    ImathRoots.h
    ImathShear.h
    ImathSphere.h
    ImathTypeTraits.h
    ImathVec.h
    ImathVecAlgo.h
  )
