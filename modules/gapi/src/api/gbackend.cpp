// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#include "precomp.hpp"
#include <memory> // unique_ptr
#include <functional> // multiplies

#include <opencv2/gapi/gkernel.hpp>

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

std::unique_ptr<cv::gimpl::GIslandExecutable>
cv::gapi::GBackend::Priv::compile(const ade::Graph& graph,
                                  const GCompileArgs& args,
                                  const std::vector<ade::NodeHandle>& nodes,
                                  const std::vector<cv::gimpl::Data>&,
                                  const std::vector<cv::gimpl::Data>&) const
{
    return compile(graph, args, nodes);
}

void cv::gapi::GBackend::Priv::addBackendPasses(ade::ExecutionEngineSetupContext &)
{
    // Do nothing by default, plugins may override this to
    // add custom (backend-specific) graph transformations
}

void cv::gapi::GBackend::Priv::addMetaSensitiveBackendPasses(ade::ExecutionEngineSetupContext &)
{
    // Do nothing by default, plugins may override this to
    // add custom (backend-specific) graph transformations
    // which are sensitive to metadata
}

cv::gapi::GKernelPackage cv::gapi::GBackend::Priv::auxiliaryKernels() const
{
    return {};
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

namespace {
// Utility function, used in both bindInArg and bindOutArg,
// implements default RMat bind behaviour (if backend doesn't handle RMats in specific way)
void bindRMat(Mag& mag, const RcDesc& rc, const cv::gapi::own::RMat& rmat)
{
    mag.template slot<cv::gapi::own::RMat>()[rc.id] = rmat;
    auto& mat = mag.template slot<cv::Mat>()[rc.id];
    mat = rmat.access();
}
} // anonymous namespace

// FIXME implement the below functions with visit()?

void bindInArg(Mag& mag, const RcDesc &rc, const GRunArg &arg, bool handleRMat)
{
    switch (rc.shape)
    {
    case GShape::GMAT:
    {
        // Skip default RMat binding if handleRMat flag is not set
        // (Assume that backend can work with some device-specific RMats
        // and will handle them in some specific way)
        if (!handleRMat) return;
        GAPI_Assert(arg.index() == GRunArg::index_of<cv::gapi::own::RMat>());
        auto& rmat = util::get<cv::gapi::own::RMat>(arg);
        bindRMat(mag, rc, rmat);
        break;
    }

    case GShape::GSCALAR:
    {
        auto& mag_scalar = mag.template slot<cv::Scalar>()[rc.id];
        switch (arg.index())
        {
        case GRunArg::index_of<cv::Scalar>() : mag_scalar = util::get<cv::Scalar>(arg);    break;
        default: util::throw_error(std::logic_error("content type of the runtime argument does not match to resource description ?"));
        }
        break;
    }

    case GShape::GARRAY:
        mag.template slot<cv::detail::VectorRef>()[rc.id] = util::get<cv::detail::VectorRef>(arg);
        break;

    case GShape::GOPAQUE:
        mag.template slot<cv::detail::OpaqueRef>()[rc.id] = util::get<cv::detail::OpaqueRef>(arg);
        break;

    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
    }
}

void bindOutArg(Mag& mag, const RcDesc &rc, const GRunArgP &arg, bool handleRMat)
{
    switch (rc.shape)
    {
    case GShape::GMAT:
    {
        // Skip default RMat binding if handleRMat flag is not set
        // (Assume that backend can work with some device-specific RMats
        // and will handle them in some specific way)
        if (!handleRMat) return;
        GAPI_Assert(arg.index() == GRunArgP::index_of<cv::gapi::own::RMat*>());
        auto& rmat = *util::get<cv::gapi::own::RMat*>(arg);
        bindRMat(mag, rc, rmat);
        break;
    }

    case GShape::GSCALAR:
    {
        auto& mag_scalar = mag.template slot<cv::Scalar>()[rc.id];
        switch (arg.index())
        {
        case GRunArgP::index_of<cv::Scalar*>() : mag_scalar = *util::get<cv::Scalar*>(arg); break;
        default: util::throw_error(std::logic_error("content type of the runtime argument does not match to resource description ?"));
        }
        break;
    }
    case GShape::GARRAY:
        mag.template slot<cv::detail::VectorRef>()[rc.id] = util::get<cv::detail::VectorRef>(arg);
        break;

    case GShape::GOPAQUE:
        mag.template slot<cv::detail::OpaqueRef>()[rc.id] = util::get<cv::detail::OpaqueRef>(arg);
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

    case GShape::GOPAQUE:
        util::get<cv::detail::ConstructOpaque>(d.ctor)
            (mag.template slot<cv::detail::OpaqueRef>()[d.rc]);
        break;

    case GShape::GSCALAR:
        mag.template slot<cv::Scalar>()[d.rc] = cv::Scalar();
        break;

    case GShape::GMAT:
        // Do nothing here - FIXME unify with initInternalData?
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
    case GShape::GMAT:    return GRunArg(mag.template slot<cv::gapi::own::RMat>().at(ref.id));
    case GShape::GSCALAR: return GRunArg(mag.template slot<cv::Scalar>().at(ref.id));
    // Note: .at() is intentional for GArray and GOpaque as objects MUST be already there
    //   (and constructed by either bindIn/Out or resetInternal)
    case GShape::GARRAY:  return GRunArg(mag.template slot<cv::detail::VectorRef>().at(ref.id));
    case GShape::GOPAQUE: return GRunArg(mag.template slot<cv::detail::OpaqueRef>().at(ref.id));
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
        {
#if !defined(GAPI_STANDALONE)
            return GRunArgP(&mag.template slot<cv::UMat>()[rc.id]);
#else
            util::throw_error(std::logic_error("UMat is not supported in standalone build"));
#endif //  !defined(GAPI_STANDALONE)
        }
        else
            return GRunArgP(&mag.template slot<cv::Mat>()[rc.id]);
    case GShape::GSCALAR: return GRunArgP(&mag.template slot<cv::Scalar>()[rc.id]);
    // Note: .at() is intentional for GArray and GOpaque as objects MUST be already there
    //   (and constructor by either bindIn/Out or resetInternal)
    case GShape::GARRAY:
        // FIXME(DM): For some absolutely unknown to me reason, move
        // semantics is involved here without const_cast to const (and
        // value from map is moved into return value GRunArgP, leaving
        // map with broken value I've spent few late Friday hours
        // debugging this!!!1
        return GRunArgP(const_cast<const Mag&>(mag)
                        .template slot<cv::detail::VectorRef>().at(rc.id));
    case GShape::GOPAQUE:
        // FIXME(DM): For some absolutely unknown to me reason, move
        // semantics is involved here without const_cast to const (and
        // value from map is moved into return value GRunArgP, leaving
        // map with broken value I've spent few late Friday hours
        // debugging this!!!1
        return GRunArgP(const_cast<const Mag&>(mag)
                        .template slot<cv::detail::OpaqueRef>().at(rc.id));
    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}

void writeBack(const Mag& mag, const RcDesc &rc, GRunArgP &g_arg, bool checkGMat)
{
    switch (rc.shape)
    {
    case GShape::GARRAY:
        // Do nothing - should we really do anything here?
        break;
        case GShape::GOPAQUE:
        // Do nothing - should we really do anything here?
        break;

    case GShape::GMAT:
    {
        GAPI_Assert(g_arg.index() == GRunArgP::index_of<cv::gapi::own::RMat*>());
        if (!checkGMat) return;
        uchar* out_arg_data = util::get<cv::gapi::own::RMat*>(g_arg)->access().data;
        auto& mag_mat = mag.slot<cv::gapi::own::RMat>().at(rc.id);
        GAPI_Assert((out_arg_data == mag_mat.access().data) && " data for output parameters was reallocated ?");
        break;
    }

    case GShape::GSCALAR:
    {
        switch (g_arg.index())
        {
        case GRunArgP::index_of<cv::Scalar*>() : *util::get<cv::Scalar*>(g_arg) = mag.template slot<cv::Scalar>().at(rc.id); break;
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

void createMat(const cv::GMatDesc &desc, cv::Mat& mat)
{
    // FIXME: Refactor (probably start supporting N-Dimensional blobs natively
    if (desc.dims.empty())
    {
        const auto type = desc.planar ? desc.depth : CV_MAKETYPE(desc.depth, desc.chan);
        const auto size = desc.planar ? cv::Size{desc.size.width, desc.size.height*desc.chan}
                                      : desc.size;
        mat.create(size, type);
    }
    else
    {
        GAPI_Assert(!desc.planar);
        mat.create(desc.dims, desc.depth);
    }
}

} // namespace gimpl
} // namespace cv
