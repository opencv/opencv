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
        if (v.dims().empty()) {
            return cv::Mat(v.rows(), v.cols(), v.type(), v.ptr(), v.step());
        } else {
            cv::Mat m(v.dims(), v.type(), v.ptr(), v.steps().data());
            if (v.dims().size() == 1) {
                // FIXME: cv::Mat() constructor will set m.dims to 2;
                // To obtain 1D Mat, we have to set m.dims back to 1 manually
                m.dims = 1;
            }
            return m;
        }
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
        return m.dims.empty()
            ? RMat::View(cv::descr_of(m), m.data, m.step, std::move(cb))
            // Own Mat doesn't support n-dimensional steps so default ones are used in this case
            : RMat::View(cv::descr_of(m), m.data, RMat::View::stepsT{}, std::move(cb));
#endif
    }

    class RMatOnMat : public RMat::IAdapter {
        cv::Mat m_mat;
    public:
        const void* data() const { return m_mat.data; }
        RMatOnMat(cv::Mat m) : m_mat(m) {}
        virtual RMat::View access(RMat::Access) override { return asView(m_mat); }
        virtual cv::GMatDesc desc() const override { return cv::descr_of(m_mat); }
    };

    // Forward declarations
    struct Data;
    struct RcDesc;

    struct GAPI_EXPORTS RMatMediaFrameAdapter final: public cv::RMat::IAdapter
    {
        using MapDescF = std::function<cv::GMatDesc(const GFrameDesc&)>;
        using MapDataF = std::function<cv::Mat(const GFrameDesc&, const cv::MediaFrame::View&)>;

        RMatMediaFrameAdapter(const cv::MediaFrame& frame,
                              const MapDescF& frameDescToMatDesc,
                              const MapDataF& frameViewToMat) :
            m_frame(frame),
            m_frameDesc(frame.desc()),
            m_frameDescToMatDesc(frameDescToMatDesc),
            m_frameViewToMat(frameViewToMat)
        { }

        virtual cv::RMat::View access(cv::RMat::Access a) override
        {
            auto rmatToFrameAccess = [](cv::RMat::Access rmatAccess) {
                switch(rmatAccess) {
                    case cv::RMat::Access::R:
                        return cv::MediaFrame::Access::R;
                    case cv::RMat::Access::W:
                        return cv::MediaFrame::Access::W;
                    default:
                        cv::util::throw_error(std::logic_error("cv::RMat::Access::R or "
                            "cv::RMat::Access::W can only be mapped to cv::MediaFrame::Access!"));
                }
            };

            auto fv = m_frame.access(rmatToFrameAccess(a));

            auto fvHolder = std::make_shared<cv::MediaFrame::View>(std::move(fv));
            auto callback = [fvHolder]() mutable { fvHolder.reset(); };

            return asView(m_frameViewToMat(m_frame.desc(), *fvHolder), callback);
        }

        virtual cv::GMatDesc desc() const override
        {
            return m_frameDescToMatDesc(m_frameDesc);
        }

        cv::MediaFrame m_frame;
        cv::GFrameDesc m_frameDesc;
        MapDescF m_frameDescToMatDesc;
        MapDataF m_frameViewToMat;
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

inline void convertInt64ToInt32(const int64_t* src, int* dst, size_t size)
{
    std::transform(src, src + size, dst,
                   [](int64_t el) { return static_cast<int>(el); });
}

}} // cv::gimpl

#endif // OPENCV_GAPI_GBACKEND_HPP
