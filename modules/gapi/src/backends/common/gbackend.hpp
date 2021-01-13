// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#ifndef OPENCV_GAPI_GBACKEND_HPP
#define OPENCV_GAPI_GBACKEND_HPP

#include <string>
#include <memory>

#include <ade/node.hpp>

#include "opencv2/gapi/garg.hpp"

#include "opencv2/gapi/util/optional.hpp"

#include "compiler/gmodel.hpp"

namespace cv {
namespace gimpl {

    inline cv::Mat asMat(RMat::View& v) {
#if !defined(GAPI_STANDALONE)
        return v.dims().empty() ? cv::Mat(v.rows(), v.cols(), v.type(), v.ptr(), v.step())
                                : cv::Mat(v.dims(), v.type(), v.ptr(), v.steps().data());
#else
        // FIXME: add a check that steps are default
        return v.dims().empty() ? cv::Mat(v.rows(), v.cols(), v.type(), v.ptr(), v.step())
                                : cv::Mat(v.dims(), v.type(), v.ptr());

#endif
    }
    inline RMat::View asView(const Mat& m, RMat::View::DestroyCallback&& cb = nullptr) {
#if !defined(GAPI_STANDALONE)
        RMat::View::stepsT steps(m.dims);
        for (int i = 0; i < m.dims; i++) {
            steps[i] = m.step[i];
        }
        return RMat::View(cv::descr_of(m), m.data, steps, std::move(cb));
#else
        return RMat::View(cv::descr_of(m), m.data, m.step, std::move(cb));
#endif
    }

    class RMatAdapter : public RMat::Adapter {
        cv::Mat m_mat;
    public:
        const void* data() const { return m_mat.data; }
        RMatAdapter(cv::Mat m) : m_mat(m) {}
        virtual RMat::View access(RMat::Access) override { return asView(m_mat); }
        virtual cv::GMatDesc desc() const override { return cv::descr_of(m_mat); }
    };

    // Forward declarations
    struct Data;
    struct RcDesc;

    struct GAPI_EXPORTS RMatMediaAdapterBGR final: public cv::RMat::Adapter
    {
        explicit RMatMediaAdapterBGR(const cv::MediaFrame& frame) : m_frame(frame) { };

        virtual cv::RMat::View access(cv::RMat::Access a) override
        {
            auto view = m_frame.access(a == cv::RMat::Access::W ? cv::MediaFrame::Access::W
                                                                : cv::MediaFrame::Access::R);
            auto ptr = reinterpret_cast<uchar*>(view.ptr[0]);
            auto stride = view.stride[0];

            std::shared_ptr<cv::MediaFrame::View> view_ptr =
                std::make_shared<cv::MediaFrame::View>(std::move(view));
            auto callback = [view_ptr]() mutable { view_ptr.reset(); };

            return cv::RMat::View(desc(), ptr, stride, callback);
        }

        virtual cv::GMatDesc desc() const override
        {
            const auto& desc = m_frame.desc();
            GAPI_Assert(desc.fmt == cv::MediaFormat::BGR);
            return cv::GMatDesc{CV_8U, 3, desc.size};
        }

        cv::MediaFrame m_frame;
    };


namespace magazine {
    template<typename... Ts> struct Class
    {
        template<typename T> using MapT = std::unordered_map<int, T>;
        using MapM = std::unordered_map<int, GRunArg::Meta>;

        template<typename T>       MapT<T>& slot()
        {
            return std::get<ade::util::type_list_index<T, Ts...>::value>(slots);
        }
        template<typename T> const MapT<T>& slot() const
        {
            return std::get<ade::util::type_list_index<T, Ts...>::value>(slots);
        }
        template<typename T> MapM& meta()
        {
            return metas[ade::util::type_list_index<T, Ts...>::value];
        }
        template<typename T> const MapM& meta() const
        {
            return metas[ade::util::type_list_index<T, Ts...>::value];
        }
    private:
        std::tuple<MapT<Ts>...> slots;
        std::array<MapM, sizeof...(Ts)> metas;
    };

} // namespace magazine

using Mag = magazine::Class< cv::Mat
                           , cv::Scalar
                           , cv::detail::VectorRef
                           , cv::detail::OpaqueRef
                           , cv::RMat
                           , cv::RMat::View
                           , cv::MediaFrame
#if !defined(GAPI_STANDALONE)
                           , cv::UMat
#endif
                           >;

namespace magazine
{
    enum class HandleRMat { BIND, SKIP };
    // Extracts a memory object from GRunArg, stores it in appropriate slot in a magazine
    // Note:
    // Only RMats are expected here as a memory object for GMat shape.
    // If handleRMat is BIND, RMat will be accessed, and RMat::View and wrapping cv::Mat
    // will be placed into the magazine.
    // If handleRMat is SKIP, this function skips'RMat handling assuming that backend will do it on its own.
    // FIXME?
    // handleRMat parameter might be redundant if all device specific backends implement own bind routines
    // without utilizing magazine at all
    void GAPI_EXPORTS bindInArg (Mag& mag, const RcDesc &rc, const GRunArg  &arg, HandleRMat handleRMat = HandleRMat::BIND);

    // Extracts a memory object reference fro GRunArgP, stores it in appropriate slot in a magazine
    // Note on RMat handling from bindInArg above is also applied here
    void GAPI_EXPORTS bindOutArg(Mag& mag, const RcDesc &rc, const GRunArgP &arg, HandleRMat handleRMat = HandleRMat::BIND);

    void         resetInternalData(Mag& mag, const Data &d);
    cv::GRunArg  getArg    (const Mag& mag, const RcDesc &ref);
    cv::GRunArgP getObjPtr (      Mag& mag, const RcDesc &rc, bool is_umat = false);
    void         writeBack (const Mag& mag, const RcDesc &rc, GRunArgP &g_arg);

    // A mandatory clean-up procedure to force proper lifetime of wrappers (cv::Mat, cv::RMat::View)
    // over not-owned data
    // FIXME? Add an RAII wrapper for that?
    // Or put objects which need to be cleaned-up into a separate stack allocated magazine?
    void         unbind(Mag &mag, const RcDesc &rc);
} // namespace magazine

namespace detail
{
template<typename... Ts> struct magazine
{
    template<typename T> using MapT = std::unordered_map<int, T>;
    template<typename T>       MapT<T>& slot()
    {
        return std::get<util::type_list_index<T, Ts...>::value>(slots);
    }
    template<typename T> const MapT<T>& slot() const
    {
        return std::get<util::type_list_index<T, Ts...>::value>(slots);
    }
private:
    std::tuple<MapT<Ts>...> slots;
};
} // namespace detail

struct GRuntimeArgs
{
    GRunArgs   inObjs;
    GRunArgsP outObjs;
};

template<typename T>
inline cv::util::optional<T> getCompileArg(const cv::GCompileArgs &args)
{
    return cv::gapi::getCompileArg<T>(args);
}

void GAPI_EXPORTS createMat(const cv::GMatDesc& desc, cv::Mat& mat);

}} // cv::gimpl

#endif // OPENCV_GAPI_GBACKEND_HPP
