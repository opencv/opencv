#ifndef OPENCV_RGBD_GRAPH_NODE_H
#define OPENCV_RGBD_GRAPH_NODE_H

#include <map>
#include <unordered_map>

#include "opencv2/core/affine.hpp"
#if defined(HAVE_EIGEN)
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "opencv2/core/eigen.hpp"
#endif

#if defined(CERES_FOUND)
#include <ceres/ceres.h>
#endif

namespace cv
{
namespace kinfu
{
/*! \class GraphNode
 *  \brief Defines a node/variable that is optimizable in a posegraph
 *
 *  Detailed description
 */
#if defined(HAVE_EIGEN)
struct Pose3d
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector3d t;
    Eigen::Quaterniond r;

    Pose3d()
    {
        t.setZero();
        r.setIdentity();
    };
    Pose3d(const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation)
        : t(translation), r(Eigen::Quaterniond(rotation))
    {
        normalizeRotation();
    }

    Pose3d(const Matx33d& rotation, const Vec3d& translation)
    {
        Eigen::Matrix3d R;
        cv2eigen(rotation, R);
        cv2eigen(translation, t);
        r = Eigen::Quaterniond(R);
        normalizeRotation();
    }

    explicit Pose3d(const Matx44f& pose)
    {
        Matx33d rotation(pose.val[0], pose.val[1], pose.val[2], pose.val[4], pose.val[5],
                         pose.val[6], pose.val[8], pose.val[9], pose.val[10]);
        Vec3d translation(pose.val[3], pose.val[7], pose.val[11]);
        Pose3d(rotation, translation);
    }

    // NOTE: Eigen overloads quaternion multiplication appropriately
    inline Pose3d operator*(const Pose3d& otherPose) const
    {
        Pose3d out(*this);
        out.t += r * otherPose.t;
        out.r *= otherPose.r;
        out.normalizeRotation();
        return out;
    }

    inline Pose3d& operator*=(const Pose3d& otherPose)
    {
        t += otherPose.t;
        r *= otherPose.r;
        normalizeRotation();
        return *this;
    }

    inline Pose3d inverse() const
    {
        Pose3d out;
        out.r = r.conjugate();
        out.t = out.r * (t * -1.0);
        return out;
    }

    inline void normalizeRotation()
    {
        if (r.w() < 0)
            r.coeffs() *= -1.0;
        r.normalize();
    }
};
#endif

struct PoseGraphNode
{
   public:
    explicit PoseGraphNode(int _nodeId, const Affine3f& _pose)
        : nodeId(_nodeId), isFixed(false), pose(_pose)
    {
#if defined(HAVE_EIGEN)
        se3Pose = Pose3d(_pose.rotation(), _pose.translation());
#endif
    }
    virtual ~PoseGraphNode() = default;

    int getId() const { return nodeId; }
    inline Affine3f getPose() const
    {
        return pose;
    }
    void setPose(const Affine3f& _pose)
    {
        pose = _pose;
#if defined(HAVE_EIGEN)
        se3Pose = Pose3d(pose.rotation(), pose.translation());
#endif
    }
#if defined(HAVE_EIGEN)
    void setPose(const Pose3d& _pose)
    {
        se3Pose = _pose;
        const Eigen::Matrix3d& rotation    = se3Pose.r.toRotationMatrix();
        const Eigen::Vector3d& translation = se3Pose.t;
        Matx33d rot;
        Vec3d trans;
        eigen2cv(rotation, rot);
        eigen2cv(translation, trans);
        Affine3d poseMatrix(rot, trans);
        pose = poseMatrix;
    }
#endif
    void setFixed(bool val = true) { isFixed = val; }
    bool isPoseFixed() const { return isFixed; }

   public:
    int nodeId;
    bool isFixed;
    Affine3f pose;
#if defined(HAVE_EIGEN)
    Pose3d se3Pose;
#endif
};

/*! \class PoseGraphEdge
 *  \brief Defines the constraints between two PoseGraphNodes
 *
 *  Detailed description
 */
struct PoseGraphEdge
{
   public:
    PoseGraphEdge(int _sourceNodeId, int _targetNodeId, const Affine3f& _transformation,
                  const Matx66f& _information = Matx66f::eye())
        : sourceNodeId(_sourceNodeId),
          targetNodeId(_targetNodeId),
          transformation(_transformation),
          information(_information)
    {
    }
    virtual ~PoseGraphEdge() = default;

    int getSourceNodeId() const { return sourceNodeId; }
    int getTargetNodeId() const { return targetNodeId; }

    bool operator==(const PoseGraphEdge& edge)
    {
        if ((edge.getSourceNodeId() == sourceNodeId && edge.getTargetNodeId() == targetNodeId) ||
            (edge.getSourceNodeId() == targetNodeId && edge.getTargetNodeId() == sourceNodeId))
            return true;
        return false;
    }

   public:
    int sourceNodeId;
    int targetNodeId;
    Affine3f transformation;
    Matx66f information;
};

//! @brief Reference: A tutorial on SE(3) transformation parameterizations and on-manifold
//! optimization Jose Luis Blanco Compactly represents the jacobian of the SE3 generator
// clang-format off
/* static const std::array<Matx44f, 6> generatorJacobian = { */
/*     // alpha */
/*     Matx44f(0, 0,  0, 0, */
/*             0, 0, -1, 0, */
/*             0, 1,  0, 0, */
/*             0, 0,  0, 0), */
/*     // beta */
/*     Matx44f( 0, 0, 1, 0, */
/*              0, 0, 0, 0, */
/*             -1, 0, 0, 0, */
/*              0, 0, 0, 0), */
/*     // gamma */
/*     Matx44f(0, -1, 0, 0, */
/*             1,  0, 0, 0, */
/*             0,  0, 0, 0, */
/*             0,  0, 0, 0), */
/*     // x */
/*     Matx44f(0, 0, 0, 1, */
/*             0, 0, 0, 0, */
/*             0, 0, 0, 0, */
/*             0, 0, 0, 0), */
/*     // y */
/*     Matx44f(0, 0, 0, 0, */
/*             0, 0, 0, 1, */
/*             0, 0, 0, 0, */
/*             0, 0, 0, 0), */
/*     // z */
/*     Matx44f(0, 0, 0, 0, */
/*             0, 0, 0, 0, */
/*             0, 0, 0, 1, */
/*             0, 0, 0, 0) */
/* }; */
// clang-format on

class PoseGraph
{
   public:
    typedef std::vector<PoseGraphNode> NodeVector;
    typedef std::vector<PoseGraphEdge> EdgeVector;

    explicit PoseGraph(){};
    virtual ~PoseGraph() = default;

    //! PoseGraph can be copied/cloned
    PoseGraph(const PoseGraph& _poseGraph) = default;
    PoseGraph& operator=(const PoseGraph& _poseGraph) = default;

    void addNode(const PoseGraphNode& node) { nodes.push_back(node); }
    void addEdge(const PoseGraphEdge& edge) { edges.push_back(edge); }

    bool nodeExists(int nodeId) const
    {
        return std::find_if(nodes.begin(), nodes.end(), [nodeId](const PoseGraphNode& currNode) {
                   return currNode.getId() == nodeId;
               }) != nodes.end();
    }

    bool isValid() const;

    int getNumNodes() const { return int(nodes.size()); }
    int getNumEdges() const { return int(edges.size()); }

   public:
    NodeVector nodes;
    EdgeVector edges;
};

namespace Optimizer
{
void optimize(PoseGraph& poseGraph);

#if defined(CERES_FOUND)
void createOptimizationProblem(PoseGraph& poseGraph, ceres::Problem& problem);

//! Error Functor required for Ceres to obtain an auto differentiable cost function
class Pose3dErrorFunctor
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Pose3dErrorFunctor(const Pose3d& _poseMeasurement, const Matx66d& _sqrtInformation)
        : poseMeasurement(_poseMeasurement)
    {
        cv2eigen(_sqrtInformation, sqrtInfo);
    }
    Pose3dErrorFunctor(const Pose3d& _poseMeasurement,
                       const Eigen::Matrix<double, 6, 6>& _sqrtInformation)
        : poseMeasurement(_poseMeasurement), sqrtInfo(_sqrtInformation)
    {
    }

    template<typename T>
    bool operator()(const T* const _pSourceTrans, const T* const _pSourceQuat,
                    const T* const _pTargetTrans, const T* const _pTargetQuat, T* _pResidual) const
    {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> sourceTrans(_pSourceTrans);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> targetTrans(_pTargetTrans);
        Eigen::Map<const Eigen::Quaternion<T>> sourceQuat(_pSourceQuat);
        Eigen::Map<const Eigen::Quaternion<T>> targetQuat(_pTargetQuat);
        Eigen::Map<Eigen::Matrix<T, 6, 1>> residual(_pResidual);

        Eigen::Quaternion<T> targetQuatInv = targetQuat.conjugate();

        Eigen::Quaternion<T> relativeQuat    = targetQuatInv * sourceQuat;
        Eigen::Matrix<T, 3, 1> relativeTrans = targetQuatInv * (targetTrans - sourceTrans);

        //! Definition should actually be relativeQuat * poseMeasurement.r.conjugate()
        Eigen::Quaternion<T> deltaRot =
            poseMeasurement.r.template cast<T>() * relativeQuat.conjugate();

        residual.template block<3, 1>(0, 0) = relativeTrans - poseMeasurement.t.template cast<T>();
        residual.template block<3, 1>(3, 0) = T(2.0) * deltaRot.vec();

        residual.applyOnTheLeft(sqrtInfo.template cast<T>());

        return true;
    }

    static ceres::CostFunction* create(const Pose3d& _poseMeasurement,
                                       const Matx66f& _sqrtInformation)
    {
        return new ceres::AutoDiffCostFunction<Pose3dErrorFunctor, 6, 3, 4, 3, 4>(
            new Pose3dErrorFunctor(_poseMeasurement, _sqrtInformation));
    }

   private:
    const Pose3d poseMeasurement;
    Eigen::Matrix<double, 6, 6> sqrtInfo;
};
#endif

}  // namespace Optimizer

}  // namespace kinfu
}  // namespace cv
#endif /* ifndef OPENCV_RGBD_GRAPH_NODE_H */
