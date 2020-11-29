// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#include "precomp.hpp"

#include <ade/graph.hpp>
#include <ade/util/zip_range.hpp>   // util::indexed

#include <opencv2/gapi/gproto.hpp> // can_describe
#include <opencv2/gapi/gcompiled.hpp>

#include "compiler/gstreaming_priv.hpp"
#include "backends/common/gbackend.hpp"

// GStreamingCompiled private implementation ///////////////////////////////////
void cv::GStreamingCompiled::Priv::setup(const GMetaArgs &_metaArgs,
                                         const GMetaArgs &_outMetas,
                                         std::unique_ptr<cv::gimpl::GStreamingExecutor> &&_pE)
{
    m_metas    = _metaArgs;
    m_outMetas = _outMetas;
    m_exec     = std::move(_pE);
}

void cv::GStreamingCompiled::Priv::setup(std::unique_ptr<cv::gimpl::GStreamingExecutor> &&_pE)
{
    m_exec = std::move(_pE);
}

bool cv::GStreamingCompiled::Priv::isEmpty() const
{
    return !m_exec;
}

const cv::GMetaArgs& cv::GStreamingCompiled::Priv::metas() const
{
    return m_metas;
}

const cv::GMetaArgs& cv::GStreamingCompiled::Priv::outMetas() const
{
    return m_outMetas;
}

// FIXME: What is the reason in having Priv here if Priv actually dispatches
// everything to the underlying executable?? May be this executable may become
// the G*Compiled's priv?
void cv::GStreamingCompiled::Priv::setSource(cv::GRunArgs &&args)
{
    if (!m_metas.empty() && !can_describe(m_metas, args))
    {
        util::throw_error(std::logic_error("This object was compiled "
                                           "for different metadata!"));
    }
    GAPI_Assert(m_exec != nullptr);
    m_exec->setSource(std::move(args));
}

void cv::GStreamingCompiled::Priv::start()
{
    m_exec->start();
}

bool cv::GStreamingCompiled::Priv::pull(cv::GRunArgsP &&outs)
{
    return m_exec->pull(std::move(outs));
}

bool cv::GStreamingCompiled::Priv::pull(cv::GOptRunArgsP &&outs)
{
    return m_exec->pull(std::move(outs));
}

bool cv::GStreamingCompiled::Priv::try_pull(cv::GRunArgsP &&outs)
{
    return m_exec->try_pull(std::move(outs));
}

void cv::GStreamingCompiled::Priv::stop()
{
    m_exec->stop();
}

bool cv::GStreamingCompiled::Priv::running() const
{
    return m_exec->running();
}

// GStreamingCompiled public implementation ////////////////////////////////////
cv::GStreamingCompiled::GStreamingCompiled()
    : m_priv(new Priv())
{
}

void cv::GStreamingCompiled::setSource(GRunArgs &&ins)
{
    // FIXME: verify these input parameters according to the graph input meta
    m_priv->setSource(std::move(ins));
}

void cv::GStreamingCompiled::setSource(const cv::gapi::wip::IStreamSource::Ptr &s)
{
    setSource(cv::gin(s));
}

void cv::GStreamingCompiled::start()
{
    m_priv->start();
}

bool cv::GStreamingCompiled::pull(cv::GRunArgsP &&outs)
{
    return m_priv->pull(std::move(outs));
}

std::tuple<bool, cv::GRunArgs> cv::GStreamingCompiled::pull()
{
    // FIXME: Why it is not @ priv??
    GRunArgs run_args;
    GRunArgsP outs;
    const auto& out_info = m_priv->outInfo();
    run_args.reserve(out_info.size());
    outs.reserve(out_info.size());

    for (auto&& info : out_info)
    {
        switch (info.shape)
        {
            case cv::GShape::GMAT:
            {
                run_args.emplace_back(cv::Mat{});
                outs.emplace_back(&cv::util::get<cv::Mat>(run_args.back()));
                break;
            }
            case cv::GShape::GSCALAR:
            {
                run_args.emplace_back(cv::Scalar{});
                outs.emplace_back(&cv::util::get<cv::Scalar>(run_args.back()));
                break;
            }
            case cv::GShape::GARRAY:
            {
                switch (info.kind)
                {
                    case cv::detail::OpaqueKind::CV_POINT2F:
                        run_args.emplace_back(cv::detail::VectorRef{std::vector<cv::Point2f>{}});
                        outs.emplace_back(cv::util::get<cv::detail::VectorRef>(run_args.back()));
                        break;
                    default:
                        util::throw_error(std::logic_error("Unsupported kind for GArray"));
                }
                break;
            }
            default:
                util::throw_error(std::logic_error("Only cv::GMat and cv::GScalar are supported for python output"));
        }
    }

    bool is_over = m_priv->pull(std::move(outs));
    return std::make_tuple(is_over, run_args);
}

bool cv::GStreamingCompiled::pull(cv::GOptRunArgsP &&outs)
{
    return m_priv->pull(std::move(outs));
}

bool cv::GStreamingCompiled::try_pull(cv::GRunArgsP &&outs)
{
    return m_priv->try_pull(std::move(outs));
}

void cv::GStreamingCompiled::stop()
{
    m_priv->stop();
}

bool cv::GStreamingCompiled::running() const
{
    return m_priv->running();
}

cv::GStreamingCompiled::operator bool() const
{
    return !m_priv->isEmpty();
}

const cv::GMetaArgs& cv::GStreamingCompiled::metas() const
{
    return m_priv->metas();
}

const cv::GMetaArgs& cv::GStreamingCompiled::outMetas() const
{
    return m_priv->outMetas();
}

cv::GStreamingCompiled::Priv& cv::GStreamingCompiled::priv()
{
    return *m_priv;
}
