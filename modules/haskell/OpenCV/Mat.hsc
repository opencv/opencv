{-# LANGUAGE ForeignFunctionInterface #-}
#include <bindings.dsl.h>
#include <mat.hpp>
module OpenCVRaw.Mat where
#strict_import

import OpenCVRaw.Types
import Foreign

#num CV_64FC1
#num CV_32FC1
#num CV_32SC1
#num CV_16SC1
#num CV_8SC1
#num CV_64FC3
#num CV_32FC3
#num CV_32SC3
#num CV_16SC3
#num CV_8SC3

#opaque_t MatExpr

#ccall cv_create_Mat        , IO (Ptr <Mat>)
#ccall cv_create_Mat_typed  , CInt -> CInt -> CInt -> IO (Ptr <Mat>)
#ccall cv_Mat_getRow        , Ptr <Mat> -> CInt -> IO (Ptr <Mat>)
#ccall cv_Mat_getCol        , Ptr <Mat> -> CInt -> IO (Ptr <Mat>)
#ccall cv_Mat_getRowRange   , Ptr <Mat> -> CInt -> CInt -> IO (Ptr <Mat>)
#ccall cv_Mat_getColRange   , Ptr <Mat> -> CInt -> CInt -> IO (Ptr <Mat>)
#ccall cv_Mat_elemSize      , Ptr <Mat> -> IO CSize
#ccall cv_Mat_elemSize1     , Ptr <Mat> -> IO CSize
#ccall cv_Mat_type          , Ptr <Mat> -> IO CInt
#ccall cv_Mat_depth         , Ptr <Mat> -> IO CInt
#ccall cv_Mat_total         , Ptr <Mat> -> IO CSize
#ccall cv_Mat_isContinuous  , Ptr <Mat> -> IO CInt
#ccall cv_Mat_channels      , Ptr <Mat> -> IO CInt
#ccall cv_Mat_rows          , Ptr <Mat> -> IO CInt
#ccall cv_Mat_cols          , Ptr <Mat> -> IO CInt
#ccall cv_Mat_empty         , Ptr <Mat> -> IO CInt
#ccall cv_Mat_size          , Ptr <Mat> -> IO (Ptr <Size>)
#ccall cv_Mat_step1         , Ptr <Mat> -> IO CSize
#ccall cv_Mat_ptr           , Ptr <Mat> -> IO (Ptr CUChar)
#ccall cv_Mat_ptr_index     , Ptr <Mat> -> CInt -> IO (Ptr CUChar)

#ccall cv_Mat_assign        , Ptr <Mat> -> Ptr <Mat> -> IO (Ptr <Mat>) 
#ccall cv_Mat_assignVal     , Ptr <Mat> -> Ptr <Scalar> -> IO (Ptr <Mat>) 
#ccall cv_Mat_clone         , Ptr <Mat> -> IO (Ptr <Mat>)
#ccall cv_Mat_copyTo        , Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_Mat_copyTo_masked , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_Mat_assignTo      , Ptr <Mat> -> Ptr <Mat> -> IO ()
#ccall cv_Mat_assignTo_t    , Ptr <Mat> -> Ptr <Mat> -> CInt -> IO ()
#ccall cv_Mat_setTo         , Ptr <Mat> -> Ptr <Scalar> -> IO (Ptr <Mat>)
#ccall cv_Mat_setTo_masked  , Ptr <Mat> -> Ptr <Scalar> -> Ptr <Mat> -> IO (Ptr <Mat>)

#ccall cv_Mat_reshape       , Ptr <Mat> -> CInt -> IO (Ptr <Mat>)
#ccall cv_Mat_reshape_rows  , Ptr <Mat> -> CInt -> CInt -> IO (Ptr <Mat>)

#ccall cv_Mat_diag          , Ptr <Mat> -> IO (Ptr <Mat>)
#ccall cv_Mat_diag_d        , Ptr <Mat> -> CInt -> IO (Ptr <Mat>)
#ccall cv_create_diagMat    , Ptr <Mat> -> IO (Ptr <Mat>)
#ccall cv_create_identity   , CInt -> CInt -> CInt -> IO (Ptr <Mat>)
#ccall cv_create_ones       , CInt -> CInt -> CInt -> IO (Ptr <Mat>)
#ccall cv_create_zeros      , CInt -> CInt -> CInt -> IO (Ptr <Mat>)

#ccall promote              , Ptr <Mat> -> Ptr <MatExpr>
#ccall force                , Ptr <MatExpr> -> IO (Ptr <Mat>)
#ccall cv_Mat_transpose_mat , Ptr <MatExpr> -> Ptr <MatExpr>
#ccall cv_Mat_inv_mat       , Ptr <MatExpr> -> CInt -> Ptr <MatExpr>
#ccall cv_Mat_add           , Ptr <MatExpr> -> Ptr <MatExpr> -> Ptr <MatExpr>
#ccall cv_Mat_mult          , Ptr <MatExpr> -> Ptr <MatExpr> -> Ptr <MatExpr>
#ccall cv_Mat_scale         , Ptr <MatExpr> -> CDouble -> Ptr <MatExpr>

#ccall cv_Mat_dot           , Ptr <Mat> -> Ptr <Mat> -> CDouble
#ccall cv_Mat_cross         , Ptr <Mat> -> Ptr <Mat> -> Ptr <Mat>

#ccall cv_Mat_locateROI     , Ptr <Mat> -> Ptr <Size> -> Ptr <Point> -> IO ()
#ccall cv_Mat_adjustROI     , Ptr <Mat> -> CInt -> CInt -> CInt -> CInt -> IO (Ptr <Mat>)
