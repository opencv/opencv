{-# LANGUAGE ForeignFunctionInterface #-}
#include <bindings.dsl.h>
#include <point.hpp>

module OpenCVRaw.Point where
#strict_import

import OpenCVRaw.Types

#opaque_t Point3i
#opaque_t Point3f
#opaque_t Point3d

#ccall cv_create_Point , CInt -> CInt -> IO (Ptr <Point>)
#ccall cv_Point_getX   , Ptr <Point> -> IO CInt
#ccall cv_Point_getY   , Ptr <Point> -> IO CInt
#ccall cv_Point_dot    , Ptr <Point> -> Ptr <Point> -> IO CInt

#ccall cv_create_Point2f , CFloat -> CFloat -> IO (Ptr <Point2f>)
#ccall cv_Point2f_getX   , Ptr <Point2f> -> IO CFloat
#ccall cv_Point2f_getY   , Ptr <Point2f> -> IO CFloat
#ccall cv_Point2f_dot    , Ptr <Point2f> -> Ptr <Point2f> -> IO CFloat

#ccall cv_create_Point2d , CDouble -> CDouble -> IO (Ptr <Point2d>)
#ccall cv_Point2d_getX   , Ptr <Point2d> -> IO CDouble
#ccall cv_Point2d_getY   , Ptr <Point2d> -> IO CDouble
#ccall cv_Point2d_dot    , Ptr <Point2d> -> Ptr <Point2d> -> IO CDouble

#ccall cv_create_Point3i , CInt -> CInt -> CInt -> IO (Ptr <Point3i>)
#ccall cv_Point3i_getX   , Ptr <Point3i> -> IO CInt
#ccall cv_Point3i_getY   , Ptr <Point3i> -> IO CInt
#ccall cv_Point3i_getZ   , Ptr <Point3i> -> IO CInt
#ccall cv_Point3i_dot    , Ptr <Point3i> -> Ptr <Point3i> -> IO CInt
#ccall cv_Point3i_cross  , Ptr <Point3i> -> Ptr <Point3i> -> IO (Ptr <Point3i>)

#ccall cv_create_Point3f , CFloat -> CFloat -> CFloat -> IO (Ptr <Point3f>)
#ccall cv_Point3f_getX   , Ptr <Point3f> -> IO CFloat
#ccall cv_Point3f_getY   , Ptr <Point3f> -> IO CFloat
#ccall cv_Point3f_getZ   , Ptr <Point3f> -> IO CFloat
#ccall cv_Point3f_dot    , Ptr <Point3f> -> Ptr <Point3f> -> IO CFloat
#ccall cv_Point3f_cross  , Ptr <Point3f> -> Ptr <Point3f> -> IO (Ptr <Point3f>)

#ccall cv_create_Point3d , CDouble -> CDouble -> CDouble -> IO (Ptr <Point3d>)
#ccall cv_Point3d_getX   , Ptr <Point3d> -> IO CDouble
#ccall cv_Point3d_getY   , Ptr <Point3d> -> IO CDouble
#ccall cv_Point3d_getZ   , Ptr <Point3d> -> IO CDouble
#ccall cv_Point3d_dot    , Ptr <Point3d> -> Ptr <Point3d> -> IO CDouble
#ccall cv_Point3d_cross  , Ptr <Point3d> -> Ptr <Point3d> -> IO (Ptr <Point3d>)
