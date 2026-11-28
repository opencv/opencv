// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_SRC_GRAPH_PARTITION_HPP
#define OPENCV_DNN_SRC_GRAPH_PARTITION_HPP

#include <opencv2/dnn.hpp>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

//! A run of consecutive ops, in program order, placed on one backend.
struct GraphSegment
{
    int backend;
    int first, last;  //!< op indices, inclusive
    int nops;         //!< ops counted for the run; host shape ops are not counted
};

//! A value produced on one backend and read on another; consumer is -1 for a graph output.
struct ArgTransition
{
    Arg arg;
    int from, to;
    int consumer;
};

/** @brief Moves accelerator runs shorter than @p minSize ops back to CPU.
 *
 * Only applies when the graph has both CPU and accelerator ops. Ops flagged in @p hostOp run on the
 * host without breaking a run; empty ops are skipped. @p placement holds a backend id per op.
 */
CV_EXPORTS void demoteShortRuns(const std::vector<Ptr<LayerInfo> >& prog, const std::vector<uchar>& hostOp,
                                std::vector<int>& placement, size_t minSize);

/** @brief Splits the program into backend segments and lists the values that cross between them.
 *
 * Graph inputs and values from an enclosing graph are on the host, constants are uploaded once and
 * are not transitions, and a Shape op in @p hostOp reads only the tensor header.
 */
CV_EXPORTS void buildSegments(const std::vector<Ptr<LayerInfo> >& prog, const std::vector<uchar>& hostOp,
                              const std::vector<int>& placement, const std::vector<Arg>& graphOutputs,
                              const std::vector<uchar>& constArg,
                              std::vector<GraphSegment>& segments, std::vector<ArgTransition>& transitions);

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn

#endif  // OPENCV_DNN_SRC_GRAPH_PARTITION_HPP
