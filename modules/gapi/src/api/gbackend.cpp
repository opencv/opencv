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

cv::GKernelPackage cv::gapi::GBackend::Priv::auxiliaryKernels() const
{
    return {};
}

bool cv::gapi::GBackend::Priv::controlsMerge() const
{
    return false;
}

bool cv::gapi::GBackend::Priv::allowsMerge(const cv::gimpl::GIslandModel::Graph &,
                                           const ade::NodeHandle &,
                                           const ade::NodeHandle &,
                                           const ade::NodeHandle &) const
{
    GAPI_Assert(controlsMerge());
    return true;
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
// implements default RMat bind behaviour (if backend doesn't handle RMats in specific way):
// view + wrapped cv::Mat are placed into the magazine
void bindRMat(Mag& mag, const RcDesc& rc, const cv::RMat& rmat, RMat::Access a)
{
    auto& matv = mag.template slot<RMat::View>()[rc.id];
    matv = rmat.access(a);
    mag.template slot<cv::Mat>()[rc.id] = asMat(matv);
}
} // anonymous namespace

// FIXME implement the below functions with visit()?
void bindInArg(Mag& mag, const RcDesc &rc, const GRunArg &arg, HandleRMat handleRMat)
{
    switch (rc.shape)
    {
    case GShape::GMAT:
    {
        // In case of handleRMat == SKIP
        // We assume that backend can work with some device-specific RMats
        // and will handle them in some specific way, so just return
        if (handleRMat == HandleRMat::SKIP) return;
        GAPI_Assert(arg.index() == GRunArg::index_of<cv::RMat>());
        bindRMat(mag, rc, util::get<cv::RMat>(arg), RMat::Access::R);

        // FIXME: Here meta may^WWILL be copied multiple times!
        // Replace it is reference-counted object?
        mag.meta<cv::RMat>()[rc.id] = arg.meta;
        mag.meta<cv::Mat>()[rc.id] = arg.meta;
#if !defined(GAPI_STANDALONE)
        mag.meta<cv::UMat>()[rc.id] = arg.meta;
#endif
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
        mag.meta<cv::Scalar>()[rc.id] = arg.meta;
        break;
    }

    case GShape::GARRAY:
        mag.slot<cv::detail::VectorRef>()[rc.id] = util::get<cv::detail::VectorRef>(arg);
        mag.meta<cv::detail::VectorRef>()[rc.id] = arg.meta;
        break;

    case GShape::GOPAQUE:
        mag.slot<cv::detail::OpaqueRef>()[rc.id] = util::get<cv::detail::OpaqueRef>(arg);
        mag.meta<cv::detail::OpaqueRef>()[rc.id] = arg.meta;
        break;

    case GShape::GFRAME:
        mag.slot<cv::MediaFrame>()[rc.id] = util::get<cv::MediaFrame>(arg);
        mag.meta<cv::MediaFrame>()[rc.id] = arg.meta;
        break;

    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
    }
}

void bindOutArg(Mag& mag, const RcDesc &rc, const GRunArgP &arg, HandleRMat handleRMat)
{
    switch (rc.shape)
    {
    case GShape::GMAT:
    {
        // In case of handleRMat == SKIP
        // We assume that backend can work with some device-specific RMats
        // and will handle them in some specific way, so just return
        if (handleRMat == HandleRMat::SKIP) return;
        GAPI_Assert(arg.index() == GRunArgP::index_of<cv::RMat*>());
        bindRMat(mag, rc, *util::get<cv::RMat*>(arg), RMat::Access::W);
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
    case GShape::GFRAME:
        mag.template slot<cv::MediaFrame>()[rc.id] = *util::get<cv::MediaFrame*>(arg);
        break;
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
    case GShape::GFRAME:
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
    case GShape::GMAT:
        return GRunArg(mag.slot<cv::RMat>().at(ref.id),
                       mag.meta<cv::RMat>().at(ref.id));
    case GShape::GSCALAR:
        return GRunArg(mag.slot<cv::Scalar>().at(ref.id),
                       mag.meta<cv::Scalar>().at(ref.id));
    // Note: .at() is intentional for GArray and GOpaque as objects MUST be already there
    //   (and constructed by either bindIn/Out or resetInternal)
    case GShape::GARRAY:
        return GRunArg(mag.slot<cv::detail::VectorRef>().at(ref.id),
                       mag.meta<cv::detail::VectorRef>().at(ref.id));
    case GShape::GOPAQUE:
        return GRunArg(mag.slot<cv::detail::OpaqueRef>().at(ref.id),
                       mag.meta<cv::detail::OpaqueRef>().at(ref.id));
    case GShape::GFRAME:
        return GRunArg(mag.slot<cv::MediaFrame>().at(ref.id),
                       mag.meta<cv::MediaFrame>().at(ref.id));
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
    case GShape::GFRAME:
        return GRunArgP(&mag.template slot<cv::MediaFrame>()[rc.id]);

    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}

void writeBack(const Mag& mag, const RcDesc &rc, GRunArgP &g_arg)
{
    switch (rc.shape)
    {
    case GShape::GARRAY:
    case GShape::GMAT:
    case GShape::GOPAQUE:
        // Do nothing - should we really do anything here?
        break;

    case GShape::GSCALAR:
    {
        switch (g_arg.index())
        {
        case GRunArgP::index_of<cv::Scalar*>() : *util::get<cv::Scalar*>(g_arg) = mag.template slot<cv::Scalar>().at(rc.id); break;
        default: util::throw_error(std::logic_error("content type of the runtime argument does not match to resource description ?"));
        }
        break;
    }

    case GShape::GFRAME:
    {
        *util::get<cv::MediaFrame*>(g_arg) = mag.template slot<cv::MediaFrame>().at(rc.id);
        break;
    }

    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}

void unbind(Mag& mag, const RcDesc &rc)
{
    switch (rc.shape)
    {
    case GShape::GARRAY:
    case GShape::GOPAQUE:
    case GShape::GSCALAR:
        // TODO: Do nothing - should we really do anything here?
        break;

    case GShape::GMAT:
        // Clean-up everything - a cv::Mat, cv::RMat::View, a cv::UMat, and cv::RMat
        // if applicable
        mag.slot<cv::Mat>().erase(rc.id);
#if !defined(GAPI_STANDALONE)
        mag.slot<cv::UMat>().erase(rc.id);
#endif
        mag.slot<cv::RMat::View>().erase(rc.id);
        mag.slot<cv::RMat>().erase(rc.id);
        break;

    case GShape::GFRAME:
        // MediaFrame can also be associated with external memory,
        // so requires a special handling here.
        mag.slot<cv::MediaFrame>().erase(rc.id);
        break;

    default:
        GAPI_Assert(false);
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
