// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022 Intel Corporation

#include "precomp.hpp"

#include <opencv2/gapi/opencv_includes.hpp>

#include "executor/gabstractstreamingexecutor.hpp"

cv::gimpl::GAbstractStreamingExecutor::GAbstractStreamingExecutor(std::unique_ptr<ade::Graph> &&g_model,
                                                                  const GCompileArgs &comp_args)
    : m_orig_graph(std::move(g_model))
    , m_island_graph(GModel::Graph(*m_orig_graph).metadata()
                     .get<IslandModel>().model)
    , m_comp_args(comp_args)
    , m_gim(*m_island_graph)
    , m_desync(GModel::Graph(*m_orig_graph).metadata()
               .contains<Desynchronized>())
{
}
