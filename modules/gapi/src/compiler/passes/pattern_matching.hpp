// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#ifndef OPENCV_GAPI_PATTERN_MATCHING_HPP
#define OPENCV_GAPI_PATTERN_MATCHING_HPP

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <list>

#include "compiler/gmodel.hpp"

namespace cv {
namespace gimpl {

    struct SubgraphMatch {
        using M =  std::unordered_map< ade::NodeHandle              // Pattern graph node
                                     , ade::NodeHandle              // Test graph node
                                     , ade::HandleHasher<ade::Node>
                                     >;
        using S =  std::unordered_set< ade::NodeHandle
                                     , ade::HandleHasher<ade::Node>
                                     >;
        M inputDataNodes;
        M startOpNodes;
        M finishOpNodes;
        M outputDataNodes;

        std::vector<ade::NodeHandle> inputTestDataNodes;
        std::vector<ade::NodeHandle> outputTestDataNodes;

        std::list<ade::NodeHandle> internalLayers;

        // FIXME: switch to operator bool() instead
        bool ok() const {
            return    !inputDataNodes.empty() && !startOpNodes.empty()
                   && !finishOpNodes.empty() && !outputDataNodes.empty()
                   && !inputTestDataNodes.empty() && !outputTestDataNodes.empty();
        }

       S nodes() const {
           S allNodes {};

           allNodes.insert(inputTestDataNodes.begin(), inputTestDataNodes.end());

           for (const auto& startOpMatch : startOpNodes) {
               allNodes.insert(startOpMatch.second);
           }

           for (const auto& finishOpMatch : finishOpNodes) {
               allNodes.insert(finishOpMatch.second);
           }

           allNodes.insert(outputTestDataNodes.begin(), outputTestDataNodes.end());

           allNodes.insert(internalLayers.begin(), internalLayers.end());

           return allNodes;
       }

       S startOps() {
            S sOps;
            for (const auto& opMatch : startOpNodes) {
               sOps.insert(opMatch.second);
            }
            return sOps;
       }

       S finishOps() {
            S fOps;
            for (const auto& opMatch : finishOpNodes) {
               fOps.insert(opMatch.second);
            }
            return fOps;
       }

       std::vector<ade::NodeHandle> protoIns() {
           return inputTestDataNodes;
       }


       std::vector<ade::NodeHandle> protoOuts() {
           return outputTestDataNodes;
       }
    };

    GAPI_EXPORTS SubgraphMatch findMatches(const cv::gimpl::GModel::Graph& patternGraph,
                                           const cv::gimpl::GModel::Graph& compGraph);

    GAPI_EXPORTS void performSubstitution(cv::gimpl::GModel::Graph& graph,
                                          const cv::gimpl::Protocol& patternP,
                                          const cv::gimpl::Protocol& substituteP,
                                          const cv::gimpl::SubgraphMatch& patternToGraphMatch);

} //namespace gimpl
} //namespace cv
#endif // OPENCV_GAPI_PATTERN_MATCHING_HPP
