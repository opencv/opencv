// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, Intel Corporation, all rights reserved.

#ifndef __IPP_HAL_RESIZE_IPP_HPP__
#define __IPP_HAL_RESIZE_IPP_HPP__

#ifdef HAVE_IPP_IW

#include <opencv2/core.hpp>
#include "precomp_ipp.hpp"

#include "iw++/iw.hpp"

#include <atomic>
#include <type_traits>

// One body for both IPP resize backends (IwiResize/IwiWarpAffine); if constexpr picks the per-tile border arg so codegen matches the original two classes.
template <typename IwiOp>
class ipp_resizeParallelT: public cv::ParallelLoopBody
{
public:
    ipp_resizeParallelT(::ipp::IwiImage &src, ::ipp::IwiImage &dst, std::atomic_bool &ok):
        m_src(src), m_dst(dst), m_ok(ok) {}

    void Init(IppiInterpolationType inter)
    {
        iwiOp.InitAlloc(m_src.m_size, m_dst.m_size, m_src.m_dataType, m_src.m_channels, inter, ::ipp::IwiResizeParams(0, 0, 0.75, 4), ippBorderRepl);

        m_ok = true;
    }

    void Init(IppiInterpolationType inter, double scaleX, double scaleY)
    {
        double shift = (inter == ippNearest) ? -1e-10 : -0.5;
        double coeffs[2][3] = {
            {scaleX, 0,      shift+0.5*scaleX},
            {0,      scaleY, shift+0.5*scaleY}
        };

        iwiOp.InitAlloc(m_src.m_size, m_dst.m_size, m_src.m_dataType, m_src.m_channels, coeffs, iwTransForward, inter, ::ipp::IwiWarpAffineParams(0, 0, 0.75), ippBorderRepl);

        m_ok = true;
    }

    virtual void operator() (const cv::Range& range) const CV_OVERRIDE
    {
        if (m_ok && !runTile(::ipp::IwiRoi(0, range.start, m_dst.m_size.width, range.end - range.start)))
            m_ok = false;
    }
    // Compute one tile (a rectangle of the destination); returns false on an IPP error
    bool runTile(const ::ipp::IwiTile& tile) const
    {
        try
        {
            if constexpr (std::is_same_v<IwiOp, ::ipp::IwiResize>)
                CV_INSTRUMENT_FUN_IPP(iwiOp, m_src, m_dst, ippBorderRepl, tile);
            else
                CV_INSTRUMENT_FUN_IPP(iwiOp, m_src, m_dst, tile);
        }
        catch(const ::ipp::IwException &)
        {
            return false;
        }
        return true;
    }
private:
    ::ipp::IwiImage &m_src;
    ::ipp::IwiImage &m_dst;

    mutable IwiOp iwiOp;

    std::atomic_bool &m_ok;
    ipp_resizeParallelT& operator= (const ipp_resizeParallelT&);
};

typedef ipp_resizeParallelT< ::ipp::IwiResize>     ipp_resizeParallel;
typedef ipp_resizeParallelT< ::ipp::IwiWarpAffine> ipp_resizeAffineParallel;

// The checks and the choice of the IPP operation of ipp_hal_resize(): returns CV_HAL_ERROR_OK if IPP supports the resize,
// `inter` is the IPP interpolation and `affine` selects ipp_resizeAffineParallel (sub-pixel accurate scale) over ipp_resizeParallel
int ipp_resize_prepare(int src_type, int src_width, int src_height, int dst_width, int dst_height,
                       double inv_scale_x, double inv_scale_y, int interpolation, IppiInterpolationType& inter, bool& affine);

#endif // HAVE_IPP_IW

#endif // __IPP_HAL_RESIZE_IPP_HPP__
