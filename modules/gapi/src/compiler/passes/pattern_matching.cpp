#include <opencv2/gapi/pattern_matching.hpp>

std::list<ade::NodeHandle> cv::gapi::findMatches(cv::gimpl::GModel::Graph patternGraph, cv::gimpl::GModel::Graph compGraph) {
    using PatternGraph = cv::gimpl::GModel::Graph;
    using CompGraph = cv::gimpl::GModel::Graph;

    using Metadata = typename PatternGraph::MetadataT;

    using Matchings = std::vector<std::pair<ade::NodeHandle, ade::NodeHandle>>;
    using MatchingsVector = std::vector<Matchings>;
    using VisitedMatchings = std::unordered_map<ade::Node*, ade::Node*>;

    ade::NodeHandle firstPatternNode = patternGraph.metadata().get<ade::passes::TopologicalSortData>().nodes().front();
    // SO NOW ALGORITHM SUPPORTS ONLY GRAPHS WITH ONE INPUT NODE
    // To support graphs with multiple input nodes:
    // Find first data nodes in pattern graph via checking of empty set of input edges.
    // Get one of the found first data node and found match for it in computational graph via MatchFirstPatternNode
    // Add in MatchFirstPatternNode registration in visited nodes.

    MatchingsVector matchings;

    // For visited matchings to work correctly it is required that output nodes from some node found in both graphs were in the same order!
    // And that outNodes array contains these nodes right in the order that they are presented in graph (left-to-right nodes-exact).
    // Else graphs were not assumed to be equal.

    // I can keep only this array fo tracking and no need in matchings arrays?
    // May be I can use this array and index for the whole algorithm
    VisitedMatchings matchedVisitedNodes;

    class MatchFirstPatternNode {
    public:
        MatchFirstPatternNode(PatternGraph& patternGraph, CompGraph& compGraph, ade::NodeHandle firstPatternNode,
                             MatchingsVector& matchings, VisitedMatchings& matchedVisitedNodes):
            m_patternGraph(patternGraph),
            m_compGraph(compGraph),
            m_firstPatternNode(firstPatternNode),
            m_firstPatternNodeMetadata(m_patternGraph.metadata(m_firstPatternNode)),
            m_matchings(matchings),
            m_matchedVisitedNodes(matchedVisitedNodes) {

        }

        bool operator()(const ade::NodeHandle& node)
        {
            VisitedMatchings matchedVisitedNodes;

            auto compNodeMetadata = m_compGraph.metadata(node);

            if (compNodeMetadata.get<cv::gimpl::NodeType>().t != cv::gimpl::NodeType::DATA) {
                return false;
            }

            if (compNodeMetadata.get<cv::gimpl::Data>().shape != m_firstPatternNodeMetadata.get<cv::gimpl::Data>().shape) {
                return false;
            }

            auto& patternOutputNodes = m_firstPatternNode->outNodes();
            auto& compOutputNodes = node->outNodes();

            if (patternOutputNodes.size() != compOutputNodes.size()) {
                return false;
            }

            // Shall be handled more carefully - it shall contains nodes only for chosen first data node from array of matched first data nodes.
            // TODO: fix wrong usage
            matchedVisitedNodes[m_firstPatternNode.get()] = node.get();

            std::vector<std::pair<ade::NodeHandle, ade::NodeHandle>> opNodesMatchings;
            for (auto patternIt = patternOutputNodes.begin(); patternIt != patternOutputNodes.end(); ++patternIt) {
                auto matchedIt = std::find_if(compOutputNodes.begin(), compOutputNodes.end(), [this, &patternIt](const ade::NodeHandle& compNode) -> bool {
                    auto patternNodeMetadata = this->m_patternGraph.metadata(*patternIt);
                    auto compNodeMetadata = this->m_compGraph.metadata(compNode);

                    if (compNodeMetadata.get<cv::gimpl::NodeType>().t != cv::gimpl::NodeType::OP) {
                        return false;
                    }

                    if (patternNodeMetadata.get<cv::gimpl::Op>().k.name != compNodeMetadata.get<cv::gimpl::Op>().k.name) {
                        return false;
                    }

                    //ADD BACKEND_SPECIFIC LOGIC

                    //if (!patternNodeMetadata.get<cv::gimpl::ActiveBackends>().backends.empty()) {
                    //    //may be it shall be done for whole graph only
                    //    std::unordered_set<cv::gapi::GBackend> patternNodeBackends = patternNodeMetadata.get<cv::gimpl::ActiveBackends>().backends;
                    //    std::unordered_set<cv::gapi::GBackend> compNodeBackends = compNodeMetadata.get<cv::gimpl::ActiveBackends>().backends

                    //    return false;
                    //}

                    return true;
                });

                if (matchedIt == compOutputNodes.end()) {
                    return false;
                }

                matchedVisitedNodes[(*patternIt).get()] = (*matchedIt).get();
                opNodesMatchings.push_back({ *patternIt, *matchedIt });
            }

            m_matchings.push_back(opNodesMatchings);
            m_matchedVisitedNodes = matchedVisitedNodes;

            return true;
       }

    private:
        PatternGraph & m_patternGraph;
        CompGraph& m_compGraph;

        ade::NodeHandle m_firstPatternNode;
        Metadata m_firstPatternNodeMetadata;
        std::vector<std::vector<std::pair<ade::NodeHandle, ade::NodeHandle>>>& m_matchings;
        VisitedMatchings& m_matchedVisitedNodes;
    };

    auto& compNodes = compGraph.metadata().get<ade::passes::TopologicalSortData>().nodes();
    auto& possibleFirstDataNodes = ade::util::filter(compNodes, MatchFirstPatternNode(patternGraph, compGraph, firstPatternNode, matchings, matchedVisitedNodes));

    // To rethink
    // To support graphs with multiple input nodes:
    // Lets name array of first data nodes in patternGraph: firstPatternDataNodes and chosen one from this as firstPatternDataNode
    // for (auto possibleFirstDataNode : possibleFirstDataNodes) {
    //    auto nextFirstPatternDataNode = firstPatternDataNodes[1];
    //  The sense is that that I need to match other data nodes: may be I shall do filter once more.
    //  For example make filter on output op nodes from first data node to find op node which have the same input data nodes as in firstPatternDataNodes.
    // The update matchings on first data nodes, and got for each data nodes to check if they are what we need.
    // updates op nodes matching. If it fails, got to next possible first pattern data node and do the same.


    auto dataNodesComparator = [](ade::NodeHandle first, Metadata firstMetadata, ade::NodeHandle second, Metadata secondMetadata) {
        if (secondMetadata.get<cv::gimpl::NodeType>().t != cv::gimpl::NodeType::DATA) {
            std::logic_error("NodeType of passed node as second argument shall be NodeType::DATA!");
        }

        if (firstMetadata.get<cv::gimpl::Data>().shape != secondMetadata.get<cv::gimpl::Data>().shape) {
            return false;
        }

        auto& firstOutputNodes = first->outNodes();
        auto& secondOutputNodes = second->outNodes();

        if (firstOutputNodes.size() != secondOutputNodes.size()) {
            return false;
        }

        return true;
    };

    auto opNodesComparator = [&matchedVisitedNodes](ade::NodeHandle first, Metadata firstMetadata, ade::NodeHandle second, Metadata secondMetadata, bool& isAlreadyVisited) {
        if (secondMetadata.get<cv::gimpl::NodeType>().t != cv::gimpl::NodeType::OP) {
            std::logic_error("NodeType of passed node as second argument shall be NodeType::OP!");
        }

        // Assuming that if kernels names are the same then output DATA nodes counts from kernels are the same.
        // Assuming that if kernels names are the same then input DATA nodes counts to kernels are the same.
        if (firstMetadata.get<cv::gimpl::Op>().k.name != secondMetadata.get<cv::gimpl::Op>().k.name) {
            return false;
        }

        auto foundit = matchedVisitedNodes.find(first.get());
        if (foundit != matchedVisitedNodes.end()) {
            if (second.get() != foundit->second) {
                return false;
            }

            isAlreadyVisited = true;
        }

        return true;
    };

    std::list<ade::NodeHandle> subgraph;

    int i = 0;
    for (auto firstDataNode : possibleFirstDataNodes) {
        // Fill the matchedVisitedNodes with new data node and nodes from matchings[i] here.
        // Remove array fill in the code with first node search.

        subgraph.push_back(firstDataNode);

        //BFS
        bool nonStop = true;
        bool isSearchFailed = false;

        // TODO: Switch to pointers
        Matchings currentLevelMatchings(matchings[i]);
        Matchings nextLevelMatchings{ };

        while (nonStop) {
            for (auto matchIt = currentLevelMatchings.begin(); matchIt != currentLevelMatchings.end() && !isSearchFailed; ++matchIt) {

                subgraph.push_back(matchIt->second);

                auto& patternOutputNodes = matchIt->first->outNodes();
                auto& compOutputNodes = matchIt->second->outNodes();

                // 1. Firstly, for every outputs find a matching 
                if (patternOutputNodes.size() != compOutputNodes.size()) {
                    nonStop = false;
                    isSearchFailed = true;
                    subgraph.clear();
                    // Clear the matchedVisitedNodes
                    break;
                }

                if (patternOutputNodes.size() == 0 && compOutputNodes.size() == 0) {
                    continue;
                }

                for (auto patternIt = patternOutputNodes.begin(); patternIt != patternOutputNodes.end(); ++patternIt) {
                    bool isAlreadyVisited = false;

                    auto matchedIt = std::find_if(compOutputNodes.begin(), compOutputNodes.end(),
                                     [&patternIt, &patternGraph, &compGraph, &dataNodesComparator, &opNodesComparator, &isAlreadyVisited] (const ade::NodeHandle& compNode) -> bool {
                                         auto patternNodeMetadata = patternGraph.metadata(*patternIt);
                                         auto compNodeMetadata = compGraph.metadata(compNode);

                                         if (patternNodeMetadata.get<cv::gimpl::NodeType>().t == cv::gimpl::NodeType::DATA) {
                                             return dataNodesComparator(*patternIt, patternNodeMetadata, compNode, compNodeMetadata);
                                         }
                                         else {
                                             return opNodesComparator(*patternIt, patternNodeMetadata, compNode, compNodeMetadata, isAlreadyVisited);
                                         }
                                     });

                    if (matchedIt == compOutputNodes.end()) {
                        nonStop = false;
                        isSearchFailed = true;
                        subgraph.clear();
                        // Clear the matchedVisitedNodes
                        break;
                    }

                    matchedVisitedNodes[(*patternIt).get()] = (*matchedIt).get();

                    //We shall not put in the matchings already visited nodes.
                    if (!isAlreadyVisited) {
                        nextLevelMatchings.push_back({ *patternIt, *matchedIt });
                    }
                }
            }

            // 2. Secondly, update the matching array
            if (!isSearchFailed) {
                if (nextLevelMatchings.size() == 0) {
                    // Subgraph is found
                        return subgraph;
                }

                currentLevelMatchings = nextLevelMatchings;
                nextLevelMatchings.clear();
            }
        }

        ++i;
    }

    return std::list<ade::NodeHandle>();
}
