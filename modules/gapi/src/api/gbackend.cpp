// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"
#include <memory> // unique_ptr

#include "opencv2/gapi/gkernel.hpp"
#include "opencv2/gapi/own/convert.hpp"

#include "api/gbackend_priv.hpp"
#include "backends/common/gbackend.hpp"
#include "compiler/gobjref.hpp"
#include "compiler/gislandmodel.hpp"

// GBackend private implementation /////////////////////////////////////////////
void cv::gapi::GBackend::Priv::unpackKernel(ade::Graph             & /*graph  */ ,
                                            const ade::NodeHandle  & /*op_node*/ ,
                                            const GKernelImpl      & /*impl   */ )
{
    // Default implementation is still there as Priv
    // is instantiated by some tests.
    // Priv is even instantiated as a mock object in a number of tests
    // as a backend and this method is called for mock objects (doing nothing).
    // FIXME: add a warning message here
    // FIXME: Do something with this! Ideally this function should be "=0";
}

std::unique_ptr<cv::gimpl::GIslandExecutable>
cv::gapi::GBackend::Priv::compile(const ade::Graph&,
                                  const GCompileArgs&,
                                  const std::vector<ade::NodeHandle> &) const
{
    // ...and this method is here for the same reason!
    GAPI_Assert(false);
    return {};
}

void cv::gapi::GBackend::Priv::addBackendPasses(ade::ExecutionEngineSetupContext &)
{
    // Do nothing by default, plugins may override this to
    // add custom (backend-specific) graph transformations
}

// GBackend public implementation //////////////////////////////////////////////
cv::gapi::GBackend::GBackend()
{
}

cv::gapi::GBackend::GBackend(std::shared_ptr<cv::gapi::GBackend::Priv> &&p)
    : m_priv(std::move(p))
{
}

cv::gapi::GBackend::Priv& cv::gapi::GBackend::priv()
{
    return *m_priv;
}

const cv::gapi::GBackend::Priv& cv::gapi::GBackend::priv() const
{
    return *m_priv;
}

std::size_t cv::gapi::GBackend::hash() const
{
    return std::hash<const cv::gapi::GBackend::Priv*>{}(m_priv.get());
}

bool cv::gapi::GBackend::operator== (const cv::gapi::GBackend &rhs) const
{
    return m_priv == rhs.m_priv;
}

// Abstract Host-side data manipulation ////////////////////////////////////////
// Reused between CPU backend and more generic GExecutor
namespace cv {
namespace gimpl {
namespace magazine {

// FIXME implement the below functions with visit()?

void bindInArg(Mag& mag, const RcDesc &rc, const GRunArg &arg, bool is_umat)
{
    switch (rc.shape)
    {
    case GShape::GMAT:
    {
        switch (arg.index())
        {
        case GRunArg::index_of<cv::gapi::own::Mat>() :
            if (is_umat)
            {
                auto& mag_umat = mag.template slot<cv::UMat>()[rc.id];
                mag_umat = to_ocv(util::get<cv::gapi::own::Mat>(arg)).getUMat(ACCESS_READ);
            }
            else
            {
                auto& mag_mat = mag.template slot<cv::gapi::own::Mat>()[rc.id];
                mag_mat = util::get<cv::gapi::own::Mat>(arg);
            }
            break;
#if !defined(GAPI_STANDALONE)
        case GRunArg::index_of<cv::Mat>() :
            if (is_umat)
            {
                auto& mag_umat = mag.template slot<cv::UMat>()[rc.id];
                mag_umat = (util::get<cv::UMat>(arg));
            }
            else
            {
                auto& mag_mat = mag.template slot<cv::gapi::own::Mat>()[rc.id];
                mag_mat = to_own(util::get<cv::Mat>(arg));
            }
            break;
#endif //  !defined(GAPI_STANDALONE)
        default: util::throw_error(std::logic_error("content type of the runtime argument does not match to resource description ?"));
        }
        break;
    }


    case GShape::GSCALAR:
    {
        auto& mag_scalar = mag.template slot<cv::gapi::own::Scalar>()[rc.id];
        switch (arg.index())
        {
            case GRunArg::index_of<cv::gapi::own::Scalar>() : mag_scalar = util::get<cv::gapi::own::Scalar>(arg); break;
#if !defined(GAPI_STANDALONE)
            case GRunArg::index_of<cv::Scalar>()            : mag_scalar = to_own(util::get<cv::Scalar>(arg));    break;
#endif //  !defined(GAPI_STANDALONE)
            default: util::throw_error(std::logic_error("content type of the runtime argument does not match to resource description ?"));
        }
        break;
    }

    case GShape::GARRAY:
        mag.template slot<cv::detail::VectorRef>()[rc.id] = util::get<cv::detail::VectorRef>(arg);
        break;

    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
    }
}

void bindOutArg(Mag& mag, const RcDesc &rc, const GRunArgP &arg, bool is_umat)
{
    switch (rc.shape)
    {
    case GShape::GMAT:
    {
        switch (arg.index())
        {
        case GRunArgP::index_of<cv::gapi::own::Mat*>() :
            if (is_umat)
            {
                auto& mag_umat = mag.template slot<cv::UMat>()[rc.id];
                mag_umat = to_ocv(*(util::get<cv::gapi::own::Mat*>(arg))).getUMat(ACCESS_RW);
            }
            else
            {
                auto& mag_mat = mag.template slot<cv::gapi::own::Mat>()[rc.id];
                mag_mat = *util::get<cv::gapi::own::Mat*>(arg);
            }
            break;
#if !defined(GAPI_STANDALONE)
        case GRunArgP::index_of<cv::Mat*>() :
            if (is_umat)
            {
                auto& mag_umat = mag.template slot<cv::UMat>()[rc.id];
                mag_umat = (*util::get<cv::UMat*>(arg));
            }
            else
            {
                auto& mag_mat = mag.template slot<cv::gapi::own::Mat>()[rc.id];
                mag_mat = to_own(*util::get<cv::Mat*>(arg));
            }
            break;
#endif //  !defined(GAPI_STANDALONE)
        default: util::throw_error(std::logic_error("content type of the runtime argument does not match to resource description ?"));
        }
        break;
    }

    case GShape::GSCALAR:
    {
        auto& mag_scalar = mag.template slot<cv::gapi::own::Scalar>()[rc.id];
        switch (arg.index())
        {
            case GRunArgP::index_of<cv::gapi::own::Scalar*>() : mag_scalar = *util::get<cv::gapi::own::Scalar*>(arg); break;
#if !defined(GAPI_STANDALONE)
            case GRunArgP::index_of<cv::Scalar*>()            : mag_scalar = to_own(*util::get<cv::Scalar*>(arg)); break;
#endif //  !defined(GAPI_STANDALONE)
            default: util::throw_error(std::logic_error("content type of the runtime argument does not match to resource description ?"));
        }
        break;
    }
    case GShape::GARRAY:
        mag.template slot<cv::detail::VectorRef>()[rc.id] = util::get<cv::detail::VectorRef>(arg);
        break;

    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}

void resetInternalData(Mag& mag, const Data &d)
{
    if (d.storage != Data::Storage::INTERNAL)
        return;

    switch (d.shape)
    {
    case GShape::GARRAY:
        util::get<cv::detail::ConstructVec>(d.ctor)
            (mag.template slot<cv::detail::VectorRef>()[d.rc]);
        break;

    case GShape::GSCALAR:
        mag.template slot<cv::gapi::own::Scalar>()[d.rc] = cv::gapi::own::Scalar();
        break;

    case GShape::GMAT:
        // Do nothign here - FIXME unify with initInternalData?
        break;

    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}

cv::GRunArg getArg(const Mag& mag, const RcDesc &ref)
{
    // Wrap associated CPU object (either host or an internal one)
    switch (ref.shape)
    {
    case GShape::GMAT:    return GRunArg(mag.template slot<cv::gapi::own::Mat>().at(ref.id));
    case GShape::GSCALAR: return GRunArg(mag.template slot<cv::gapi::own::Scalar>().at(ref.id));
    // Note: .at() is intentional for GArray as object MUST be already there
    //   (and constructed by either bindIn/Out or resetInternal)
    case GShape::GARRAY:  return GRunArg(mag.template slot<cv::detail::VectorRef>().at(ref.id));
    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}

cv::GRunArgP getObjPtr(Mag& mag, const RcDesc &rc, bool is_umat)
{
    switch (rc.shape)
    {
    case GShape::GMAT:
        if (is_umat)
            return GRunArgP(&mag.template slot<cv::UMat>()[rc.id]);
        else
            return GRunArgP(&mag.template slot<cv::gapi::own::Mat>()[rc.id]);
    case GShape::GSCALAR: return GRunArgP(&mag.template slot<cv::gapi::own::Scalar>()[rc.id]);
    // Note: .at() is intentional for GArray as object MUST be already there
    //   (and constructer by either bindIn/Out or resetInternal)
    case GShape::GARRAY:
        // FIXME(DM): For some absolutely unknown to me reason, move
        // semantics is involved here without const_cast to const (and
        // value from map is moved into return value GRunArgP, leaving
        // map with broken value I've spent few late Friday hours
        // debugging this!!!1
        return GRunArgP(const_cast<const Mag&>(mag)
                        .template slot<cv::detail::VectorRef>().at(rc.id));
    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}

void writeBack(const Mag& mag, const RcDesc &rc, GRunArgP &g_arg, bool is_umat)
{
    switch (rc.shape)
    {
    case GShape::GARRAY:
        // Do nothing - should we really do anything here?
        break;

    case GShape::GMAT:
    {
        //simply check that memory was not reallocated, i.e.
        //both instances of Mat pointing to the same memory
        uchar* out_arg_data = nullptr;
        switch (g_arg.index())
        {
            case GRunArgP::index_of<cv::gapi::own::Mat*>() : out_arg_data = util::get<cv::gapi::own::Mat*>(g_arg)->data; break;
#if !defined(GAPI_STANDALONE)
            case GRunArgP::index_of<cv::Mat*>()            : out_arg_data = util::get<cv::Mat*>(g_arg)->data; break;
            case GRunArgP::index_of<cv::UMat*>()           : out_arg_data = (util::get<cv::UMat*>(g_arg))->getMat(ACCESS_RW).data; break;
#endif //  !defined(GAPI_STANDALONE)
            default: util::throw_error(std::logic_error("content type of the runtime argument does not match to resource description ?"));
        }
        if (is_umat)
        {
            auto& in_mag = mag.template slot<cv::UMat>().at(rc.id);
            GAPI_Assert((out_arg_data == (in_mag.getMat(ACCESS_RW).data)) && " data for output parameters was reallocated ?");
        }
        else
        {
            auto& in_mag = mag.template slot<cv::gapi::own::Mat>().at(rc.id);
            GAPI_Assert((out_arg_data == in_mag.data) && " data for output parameters was reallocated ?");
        }
        break;
    }

    case GShape::GSCALAR:
    {
        switch (g_arg.index())
        {
            case GRunArgP::index_of<cv::gapi::own::Scalar*>() : *util::get<cv::gapi::own::Scalar*>(g_arg) = mag.template slot<cv::gapi::own::Scalar>().at(rc.id); break;
#if !defined(GAPI_STANDALONE)
            case GRunArgP::index_of<cv::Scalar*>()            : *util::get<cv::Scalar*>(g_arg) = cv::gapi::own::to_ocv(mag.template slot<cv::gapi::own::Scalar>().at(rc.id)); break;
#endif //  !defined(GAPI_STANDALONE)
            default: util::throw_error(std::logic_error("content type of the runtime argument does not match to resource description ?"));
        }
        break;
    }

    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}

} // namespace magazine
} // namespace gimpl
} // namespace cv
