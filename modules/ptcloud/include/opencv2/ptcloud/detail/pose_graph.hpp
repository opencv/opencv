// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef POINTCLOUD_POSE_GRAPTH_HPP
#define POINTCLOUD_POSE_GRAPTH_HPP

#include "opencv2/core/affine.hpp"
#include "opencv2/core/quaternion.hpp"
#include "opencv2/geometry/3d.hpp"
#include "opencv2/geometry/detail/optimizer.hpp"

namespace cv
{
namespace detail
{

// ATTENTION! This class is used internally in Large KinFu.
// It has been pushed to publicly available headers for tests only.
// Source compatibility of this API is not guaranteed in the future.

// This class provides tools to solve so-called pose graph problem often arisen in SLAM problems
// The pose graph format, cost function and optimization techniques
// repeat the ones used in Ceres 3D Pose Graph Optimization:
// http://ceres-solver.org/nnls_tutorial.html#other-examples, pose_graph_3d.cc bullet
class CV_EXPORTS_W PoseGraph
{
public:
    static Ptr<PoseGraph> create();
    virtual ~PoseGraph();

    // Node may have any id >= 0
    virtual void addNode(size_t _nodeId, const Affine3d& _pose, bool fixed) = 0;
    virtual bool isNodeExist(size_t nodeId) const = 0;
    virtual bool setNodeFixed(size_t nodeId, bool fixed) = 0;
    virtual bool isNodeFixed(size_t nodeId) const = 0;
    virtual Affine3d getNodePose(size_t nodeId) const = 0;
    virtual std::vector<size_t> getNodesIds() const = 0;
    virtual size_t getNumNodes() const = 0;

    // Edges have consequent indices starting from 0
    virtual void addEdge(size_t _sourceNodeId, size_t _targetNodeId, const Affine3f& _transformation,
                         const Matx66f& _information = Matx66f::eye()) = 0;
    virtual size_t getEdgeStart(size_t i) const = 0;
    virtual size_t getEdgeEnd(size_t i) const = 0;
    virtual Affine3d getEdgePose(size_t i) const = 0;
    virtual Matx66f getEdgeInfo(size_t i) const = 0;
    virtual size_t getNumEdges() const = 0;

    // checks if graph is connected and each edge connects exactly 2 nodes
    virtual bool isValid() const = 0;

    // Calculates an initial pose estimate using the Minimum Spanning Tree (Prim's MST) algorithm.
    // The result serves as a starting point for further optimization.
    // Edge weights are calculated as:
    //     weight = translationNorm + lambda * rotationAngle
    // The default lambda value (0.485) was empirically chosen based on its impact on optimizer performance,
    // but can/should be tuned for different datasets.
    virtual void initializePosesWithMST(double lambda = 0.485) = 0;

    // creates an optimizer with user-defined settings and returns a pointer on it
    virtual Ptr<cv::detail::LevMarqBase> createOptimizer(const LevMarq::Settings& settings) = 0;
    // creates an optimizer with default settings and returns a pointer on it
    virtual Ptr<cv::detail::LevMarqBase> createOptimizer() = 0;

    // Creates an optimizer (with default settings) if it wasn't created before and runs it
    // Returns number of iterations elapsed or -1 if failed to optimize
    virtual LevMarq::Report optimize() = 0;

    // calculate cost function based on current nodes parameters
    virtual double calcEnergy() const = 0;
};

}  // namespace detail
}  // namespace cv

#endif // include guard
