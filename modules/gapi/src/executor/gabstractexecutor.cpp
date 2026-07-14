// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022 Intel Corporation

#include "precomp.hpp"

#include <opencv2/gapi/opencv_includes.hpp>

#include "executor/gabstractexecutor.hpp"

cv::gimpl::GAbstractExecutor::GAbstractExecutor(std::unique_ptr<ade::Graph> &&g_model)
    : m_orig_graph(std::move(g_model))
    , m_island_graph(GModel::Graph(*m_orig_graph).metadata()
                     .get<IslandModel>().model)
    , m_gm(*m_orig_graph)
    , m_gim(*m_island_graph)
{
}

const cv::gimpl::GModel::Graph& cv::gimpl::GAbstractExecutor::model() const
{
    return m_gm;
}
