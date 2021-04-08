// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

#include "sparse_block_matrix.hpp"

// matrix form of conjugation
static const cv::Matx44d M_Conj{ 1,  0,  0,  0,
                                 0, -1,  0,  0,
                                 0,  0, -1,  0,
                                 0,  0,  0, -1 };

// matrix form of quaternion multiplication from left side
static inline cv::Matx44d m_left(cv::Quatd q)
{
    // M_left(a)* V(b) =
    //    = (I_4 * a0 + [ 0 | -av    [    0 | 0_1x3
    //                   av | 0_3] +  0_3x1 | skew(av)]) * V(b)

    double w = q.w, x = q.x, y = q.y, z = q.z;
    return { w, -x, -y, -z,
             x,  w, -z,  y,
             y,  z,  w, -x,
             z, -y,  x,  w };
}

// matrix form of quaternion multiplication from right side
static inline cv::Matx44d m_right(cv::Quatd q)
{
    // M_right(b)* V(a) =
    //    = (I_4 * b0 + [ 0 | -bv    [    0 | 0_1x3
    //                   bv | 0_3] +  0_3x1 | skew(-bv)]) * V(a)

    double w = q.w, x = q.x, y = q.y, z = q.z;
    return { w, -x, -y, -z,
             x,  w,  z, -y,
             y, -z,  w,  x,
             z,  y, -x,  w };
}

// precaution against "unused function" warning when there's no Eigen
#if defined(HAVE_EIGEN)
// jacobian of quaternionic (exp(x)*q) : R_3 -> H near x == 0
static inline cv::Matx43d expQuatJacobian(cv::Quatd q)
{
    double w = q.w, x = q.x, y = q.y, z = q.z;
    return cv::Matx43d(-x, -y, -z,
                        w,  z, -y,
                       -z,  w,  x,
                        y, -x,  w);
}
#endif

// concatenate matrices vertically
template<typename _Tp, int m, int n, int k> static inline
cv::Matx<_Tp, m + k, n> concatVert(const cv::Matx<_Tp, m, n>& a, const cv::Matx<_Tp, k, n>& b)
{
    cv::Matx<_Tp, m + k, n> res;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            res(i, j) = a(i, j);
        }
    }
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < n; j++)
        {
            res(m + i, j) = b(i, j);
        }
    }
    return res;
}

// concatenate matrices horizontally
template<typename _Tp, int m, int n, int k> static inline
cv::Matx<_Tp, m, n + k> concatHor(const cv::Matx<_Tp, m, n>& a, const cv::Matx<_Tp, m, k>& b)
{
    cv::Matx<_Tp, m, n + k> res;

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            res(i, j) = a(i, j);
        }
    }
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            res(i, n + j) = b(i, j);
        }
    }
    return res;
}

namespace cv
{
namespace kinfu
{
namespace detail
{

class PoseGraphImpl : public detail::PoseGraph
{
    struct Pose3d
    {
        Vec3d t;
        Quatd q;

        Pose3d() : t(), q(1, 0, 0, 0) { }

        Pose3d(const Matx33d& rotation, const Vec3d& translation)
            : t(translation), q(Quatd::createFromRotMat(rotation).normalize())
        { }

        explicit Pose3d(const Matx44d& pose) :
            Pose3d(pose.get_minor<3, 3>(0, 0), Vec3d(pose(0, 3), pose(1, 3), pose(2, 3)))
        { }

        inline Pose3d operator*(const Pose3d& otherPose) const
        {
            Pose3d out(*this);
            out.t += q.toRotMat3x3(QUAT_ASSUME_UNIT) * otherPose.t;
            out.q = out.q * otherPose.q;
            return out;
        }

        Affine3d getAffine() const
        {
            return Affine3d(q.toRotMat3x3(QUAT_ASSUME_UNIT), t);
        }

        inline Pose3d inverse() const
        {
            Pose3d out;
            out.q = q.conjugate();
            out.t = -(out.q.toRotMat3x3(QUAT_ASSUME_UNIT) * t);
            return out;
        }

        inline void normalizeRotation()
        {
            q = q.normalize();
        }
    };

    /*! \class GraphNode
     *  \brief Defines a node/variable that is optimizable in a posegraph
     *
     *  Detailed description
     */
    struct Node
    {
    public:
        explicit Node(size_t _nodeId, const Affine3d& _pose)
            : id(_nodeId), isFixed(false), pose(_pose.rotation(), _pose.translation())
        { }

        Affine3d getPose() const
        {
            return pose.getAffine();
        }
        void setPose(const Affine3d& _pose)
        {
            pose = Pose3d(_pose.rotation(), _pose.translation());
        }

    public:
        size_t id;
        bool isFixed;
        Pose3d pose;
    };

    /*! \class PoseGraphEdge
     *  \brief Defines the constraints between two PoseGraphNodes
     *
     *  Detailed description
     */
    struct Edge
    {
    public:
        explicit Edge(size_t _sourceNodeId, size_t _targetNodeId, const Affine3f& _transformation,
                      const Matx66f& _information = Matx66f::eye());

        bool operator==(const Edge& edge)
        {
            if ((edge.sourceNodeId == sourceNodeId && edge.targetNodeId == targetNodeId) ||
                (edge.sourceNodeId == targetNodeId && edge.targetNodeId == sourceNodeId))
                return true;
            return false;
        }

    public:
        size_t sourceNodeId;
        size_t targetNodeId;
        Pose3d pose;
        Matx66f sqrtInfo;
    };


public:
    PoseGraphImpl() : nodes(), edges()
    { }
    virtual ~PoseGraphImpl() CV_OVERRIDE
    { }

    // Node may have any id >= 0
    virtual void addNode(size_t _nodeId, const Affine3d& _pose, bool fixed) CV_OVERRIDE;
    virtual bool isNodeExist(size_t nodeId) const CV_OVERRIDE
    {
        return (nodes.find(nodeId) != nodes.end());
    }

    virtual bool setNodeFixed(size_t nodeId, bool fixed) CV_OVERRIDE
    {
        auto it = nodes.find(nodeId);
        if (it != nodes.end())
        {
            it->second.isFixed = fixed;
            return true;
        }
        else
            return false;
    }

    virtual bool isNodeFixed(size_t nodeId) const CV_OVERRIDE
    {
        auto it = nodes.find(nodeId);
        if (it != nodes.end())
            return it->second.isFixed;
        else
            return false;
    }

    virtual Affine3d getNodePose(size_t nodeId) const CV_OVERRIDE
    {
        auto it = nodes.find(nodeId);
        if (it != nodes.end())
            return it->second.getPose();
        else
            return Affine3d();
    }

    virtual std::vector<size_t> getNodesIds() const CV_OVERRIDE
    {
        std::vector<size_t> ids;
        for (const auto& it : nodes)
        {
            ids.push_back(it.first);
        }
        return ids;
    }

    virtual size_t getNumNodes() const CV_OVERRIDE
    {
        return nodes.size();
    }

    // Edges have consequent indices starting from 0
    virtual void addEdge(size_t _sourceNodeId, size_t _targetNodeId, const Affine3f& _transformation,
                         const Matx66f& _information = Matx66f::eye()) CV_OVERRIDE
    {
        Edge e(_sourceNodeId, _targetNodeId, _transformation, _information);
        edges.push_back(e);
    }

    virtual size_t getEdgeStart(size_t i) const CV_OVERRIDE
    {
        return edges[i].sourceNodeId;
    }

    virtual size_t getEdgeEnd(size_t i) const CV_OVERRIDE
    {
        return edges[i].targetNodeId;
    }

    virtual size_t getNumEdges() const CV_OVERRIDE
    {
        return edges.size();
    }

    // checks if graph is connected and each edge connects exactly 2 nodes
    virtual bool isValid() const CV_OVERRIDE;

    // calculate cost function based on current nodes parameters
    virtual double calcEnergy() const CV_OVERRIDE;

    // calculate cost function based on provided nodes parameters
    double calcEnergyNodes(const std::map<size_t, Node>& newNodes) const;

    // Termination criteria are max number of iterations and min relative energy change to current energy
    // Returns number of iterations elapsed or -1 if max number of iterations was reached or failed to optimize
    virtual int optimize(const cv::TermCriteria& tc = cv::TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-6)) CV_OVERRIDE;

    std::map<size_t, Node> nodes;
    std::vector<Edge>   edges;
};


void PoseGraphImpl::addNode(size_t _nodeId, const Affine3d& _pose, bool fixed)
{
    Node node(_nodeId, _pose);
    node.isFixed = fixed;

    size_t id = node.id;
    const auto& it = nodes.find(id);
    if (it != nodes.end())
    {
        std::cout << "duplicated node, id=" << id << std::endl;
        nodes.insert(it, { id, node });
    }
    else
    {
        nodes.insert({ id, node });
    }
}


// Cholesky decomposition of symmetrical 6x6 matrix
static inline cv::Matx66d llt6(Matx66d m)
{
    Matx66d L;
    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < (i + 1); j++)
        {
            double sum = 0;
            for (int k = 0; k < j; k++)
                sum += L(i, k) * L(j, k);

            if (i == j)
                L(i, i) = sqrt(m(i, i) - sum);
            else
                L(i, j) = (1.0 / L(j, j) * (m(i, j) - sum));
        }
    }
    return L;
}

PoseGraphImpl::Edge::Edge(size_t _sourceNodeId, size_t _targetNodeId, const Affine3f& _transformation,
                          const Matx66f& _information) :
                          sourceNodeId(_sourceNodeId),
                          targetNodeId(_targetNodeId),
                          pose(_transformation.rotation(), _transformation.translation()),
                          sqrtInfo(llt6(_information))
{ }


bool PoseGraphImpl::isValid() const
{
    size_t numNodes = getNumNodes();
    size_t numEdges = getNumEdges();

    if (!numNodes || !numEdges)
        return false;

    std::unordered_set<size_t> nodesVisited;
    std::vector<size_t> nodesToVisit;

    nodesToVisit.push_back(nodes.begin()->first);

    bool isGraphConnected = false;
    while (!nodesToVisit.empty())
    {
        size_t currNodeId = nodesToVisit.back();
        nodesToVisit.pop_back();
        nodesVisited.insert(currNodeId);
        // Since each node does not maintain its neighbor list
        for (size_t i = 0; i < numEdges; i++)
        {
            const Edge& potentialEdge = edges.at(i);
            size_t nextNodeId = (size_t)(-1);

            if (potentialEdge.sourceNodeId == currNodeId)
            {
                nextNodeId = potentialEdge.targetNodeId;
            }
            else if (potentialEdge.targetNodeId == currNodeId)
            {
                nextNodeId = potentialEdge.sourceNodeId;
            }
            if (nextNodeId != (size_t)(-1))
            {
                if (nodesVisited.count(nextNodeId) == 0)
                {
                    nodesToVisit.push_back(nextNodeId);
                }
            }
        }
    }

    isGraphConnected = (nodesVisited.size() == numNodes);

    CV_LOG_INFO(NULL, "nodesVisited: " << nodesVisited.size() << " IsGraphConnected: " << isGraphConnected);

    bool invalidEdgeNode = false;
    for (size_t i = 0; i < numEdges; i++)
    {
        const Edge& edge = edges.at(i);
        // edges have spurious source/target nodes
        if ((nodesVisited.count(edge.sourceNodeId) != 1) ||
            (nodesVisited.count(edge.targetNodeId) != 1))
        {
            invalidEdgeNode = true;
            break;
        }
    }
    return isGraphConnected && !invalidEdgeNode;
}


//////////////////////////
// Optimization itself //
////////////////////////

static inline double poseError(Quatd sourceQuat, Vec3d sourceTrans, Quatd targetQuat, Vec3d targetTrans,
                               Quatd rotMeasured, Vec3d transMeasured, Matx66d sqrtInfoMatrix, bool needJacobians,
                               Matx<double, 6, 4>& sqj, Matx<double, 6, 3>& stj,
                               Matx<double, 6, 4>& tqj, Matx<double, 6, 3>& ttj,
                               Vec6d& res)
{
    // err_r = 2*Im(conj(rel_r) * measure_r) = 2*Im(conj(target_r) * source_r * measure_r)
    // err_t = conj(source_r) * (target_t - source_t) * source_r - measure_t

    Quatd sourceQuatInv = sourceQuat.conjugate();
    Vec3d deltaTrans = targetTrans - sourceTrans;

    Quatd relativeQuat = sourceQuatInv * targetQuat;
    Vec3d relativeTrans = sourceQuatInv.toRotMat3x3(cv::QUAT_ASSUME_UNIT) * deltaTrans;

    //! Definition should actually be relativeQuat * rotMeasured.conjugate()
    Quatd deltaRot = relativeQuat.conjugate() * rotMeasured;

    Vec3d terr = relativeTrans - transMeasured;
    Vec3d rerr = 2.0 * Vec3d(deltaRot.x, deltaRot.y, deltaRot.z);
    Vec6d rterr(terr[0], terr[1], terr[2], rerr[0], rerr[1], rerr[2]);

    res = sqrtInfoMatrix * rterr;

    if (needJacobians)
    {
        // d(err_r) = 2*Im(d(conj(target_r) * source_r * measure_r)) = < measure_r is constant > =
        // 2*Im((conj(d(target_r)) * source_r + conj(target_r) * d(source_r)) * measure_r)
        // d(target_r) == 0:
        //  # d(err_r) = 2*Im(conj(target_r) * d(source_r) * measure_r)
        //  # V(d(err_r)) = 2 * M_Im * M_right(measure_r) * M_left(conj(target_r)) * V(d(source_r))
        //  # d(err_r) / d(source_r) = 2 * M_Im * M_right(measure_r) * M_left(conj(target_r))
        Matx34d drdsq = 2.0 * (m_right(rotMeasured) * m_left(targetQuat.conjugate())).get_minor<3, 4>(1, 0);

        // d(source_r) == 0:
        //  # d(err_r) = 2*Im(conj(d(target_r)) * source_r * measure_r)
        //  # V(d(err_r)) = 2 * M_Im * M_right(source_r * measure_r) * M_Conj * V(d(target_r))
        //  # d(err_r) / d(target_r) = 2 * M_Im * M_right(source_r * measure_r) * M_Conj
        Matx34d drdtq = 2.0 * (m_right(sourceQuat * rotMeasured) * M_Conj).get_minor<3, 4>(1, 0);

        // d(err_t) = d(conj(source_r) * (target_t - source_t) * source_r) =
        // conj(source_r) * (d(target_t) - d(source_t)) * source_r +
        // conj(d(source_r)) * (target_t - source_t) * source_r +
        // conj(source_r) * (target_t - source_t) * d(source_r) =
        // <conj(a*b) == conj(b)*conj(a), conj(target_t - source_t) = - (target_t - source_t), 2 * Im(x) = (x - conj(x))>
        // conj(source_r) * (d(target_t) - d(source_t)) * source_r +
        // 2 * Im(conj(source_r) * (target_t - source_t) * d(source_r))
        // d(*_t) == 0:
        //  # d(err_t) = 2 * Im(conj(source_r) * (target_t - source_t) * d(source_r))
        //  # V(d(err_t)) = 2 * M_Im * M_left(conj(source_r) * (target_t - source_t)) * V(d(source_r))
        //  # d(err_t) / d(source_r) = 2 * M_Im * M_left(conj(source_r) * (target_t - source_t))
        Matx34d dtdsq = 2 * m_left(sourceQuatInv * Quatd(0, deltaTrans[0], deltaTrans[1], deltaTrans[2])).get_minor<3, 4>(1, 0);
        // deltaTrans is rotated by sourceQuatInv, so the jacobian is rot matrix of sourceQuatInv by +1 or -1
        Matx33d dtdtt = sourceQuatInv.toRotMat3x3(QUAT_ASSUME_UNIT);
        Matx33d dtdst = -dtdtt;

        Matx33d z;
        sqj = concatVert(dtdsq, drdsq);
        tqj = concatVert(Matx34d(), drdtq);
        stj = concatVert(dtdst, z);
        ttj = concatVert(dtdtt, z);

        stj = sqrtInfoMatrix * stj;
        ttj = sqrtInfoMatrix * ttj;
        sqj = sqrtInfoMatrix * sqj;
        tqj = sqrtInfoMatrix * tqj;
    }

    return res.ddot(res);
}


double PoseGraphImpl::calcEnergy() const
{
    return calcEnergyNodes(nodes);
}


// estimate current energy
double PoseGraphImpl::calcEnergyNodes(const std::map<size_t, Node>& newNodes) const
{
    double totalErr = 0;
    for (const auto& e : edges)
    {
        Pose3d srcP = newNodes.at(e.sourceNodeId).pose;
        Pose3d tgtP = newNodes.at(e.targetNodeId).pose;

        Vec6d res;
        Matx<double, 6, 3> stj, ttj;
        Matx<double, 6, 4> sqj, tqj;
        double err = poseError(srcP.q, srcP.t, tgtP.q, tgtP.t, e.pose.q, e.pose.t, e.sqrtInfo,
                               /* needJacobians = */ false, sqj, stj, tqj, ttj, res);

        totalErr += err;
    }
    return totalErr * 0.5;
};


#if defined(HAVE_EIGEN)

// from Ceres, equation energy change:
// eq. energy = 1/2 * (residuals + J * step)^2 =
// 1/2 * ( residuals^2 + 2 * residuals^T * J * step + (J*step)^T * J * step)
// eq. energy change = 1/2 * residuals^2 - eq. energy =
// residuals^T * J * step + 1/2 * (J*step)^T * J * step =
// (residuals^T * J + 1/2 * step^T * J^T * J) * step =
// step^T * ((residuals^T * J)^T + 1/2 * (step^T * J^T * J)^T) =
// 1/2 * step^T * (2 * J^T * residuals + J^T * J * step) =
// 1/2 * step^T * (2 * J^T * residuals + (J^T * J + LMDiag - LMDiag) * step) =
// 1/2 * step^T * (2 * J^T * residuals + (J^T * J + LMDiag) * step - LMDiag * step) =
// 1/2 * step^T * (J^T * residuals - LMDiag * step) =
// 1/2 * x^T * (jtb - lmDiag^T * x)
static inline double calcJacCostChange(const std::vector<double>& jtb,
                                       const std::vector<double>& x,
                                       const std::vector<double>& lmDiag)
{
    double jdiag = 0.0;
    for (size_t i = 0; i < x.size(); i++)
    {
        jdiag += x[i] * (jtb[i] - lmDiag[i] * x[i]);
    }
    double costChange = jdiag * 0.5;
    return costChange;
};


// J := J * d_inv, d_inv = make_diag(di)
// J^T*J := (J * d_inv)^T * J * d_inv = diag(di)* (J^T * J)* diag(di) = eltwise_mul(J^T*J, di*di^T)
// J^T*b := (J * d_inv)^T * b = d_inv^T * J^T*b = eltwise_mul(J^T*b, di)
static inline void doJacobiScaling(BlockSparseMat<double, 6, 6>& jtj, std::vector<double>& jtb, const std::vector<double>& di)
{
    // scaling J^T*J
    for (auto& ijv : jtj.ijValue)
    {
        Point2i bpt = ijv.first;
        Matx66d& m = ijv.second;
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                Point2i pt(bpt.x * 6 + i, bpt.y * 6 + j);
                m(i, j) *= di[pt.x] * di[pt.y];
            }
        }
    }

    // scaling J^T*b
    for (size_t i = 0; i < di.size(); i++)
    {
        jtb[i] *= di[i];
    }
}


int PoseGraphImpl::optimize(const cv::TermCriteria& tc)
{
    if (!isValid())
    {
        CV_LOG_INFO(NULL, "Invalid PoseGraph that is either not connected or has invalid nodes");
        return -1;
    }

    size_t numNodes = getNumNodes();
    size_t numEdges = getNumEdges();

    // Allocate indices for nodes
    std::vector<size_t> placesIds;
    std::map<size_t, size_t> idToPlace;
    for (const auto& ni : nodes)
    {
        if (!ni.second.isFixed)
        {
            idToPlace[ni.first] = placesIds.size();
            placesIds.push_back(ni.first);
        }
    }

    size_t nVarNodes = placesIds.size();
    if (!nVarNodes)
    {
        CV_LOG_INFO(NULL, "PoseGraph contains no non-constant nodes, skipping optimization");
        return -1;
    }

    if (numEdges == 0)
    {
        CV_LOG_INFO(NULL, "PoseGraph has no edges, no optimization to be done");
        return -1;
    }

    CV_LOG_INFO(NULL, "Optimizing PoseGraph with " << numNodes << " nodes and " << numEdges << " edges");

    size_t nVars = nVarNodes * 6;
    BlockSparseMat<double, 6, 6> jtj(nVarNodes);
    std::vector<double> jtb(nVars);

    double energy = calcEnergyNodes(nodes);
    double oldEnergy = energy;

    CV_LOG_INFO(NULL, "#s" << " energy: " << energy);

    // options
    // stop conditions
    bool checkIterations = (tc.type & TermCriteria::COUNT);
    bool checkEps = (tc.type & TermCriteria::EPS);
    const unsigned int maxIterations = tc.maxCount;
    const double minGradientTolerance = 1e-6;
    const double stepNorm2Tolerance = 1e-6;
    const double relEnergyDeltaTolerance = tc.epsilon;
    // normalize jacobian columns for better conditioning
    // slows down sparse solver, but maybe this'd be useful for some other solver
    const bool jacobiScaling = false;
    const double minDiag = 1e-6;
    const double maxDiag = 1e32;

    const double initialLambdaLevMarq = 0.0001;
    const double initialLmUpFactor = 2.0;
    const double initialLmDownFactor = 3.0;

    // finish reasons
    bool tooLong          = false; // => not found
    bool smallGradient    = false; // => found
    bool smallStep        = false; // => found
    bool smallEnergyDelta = false; // => found

    // column scale inverted, for jacobian scaling
    std::vector<double> di(nVars);

    double lmUpFactor = initialLmUpFactor;
    double lambdaLevMarq = initialLambdaLevMarq;

    unsigned int iter = 0;
    bool done = false;
    while (!done)
    {
        jtj.clear();
        std::fill(jtb.begin(), jtb.end(), 0.0);

        // caching nodes jacobians
        std::vector<cv::Matx<double, 7, 6>> cachedJac;
        for (auto id : placesIds)
        {
            Pose3d p = nodes.at(id).pose;
            Matx43d qj = expQuatJacobian(p.q);
            // x node layout is (rot_x, rot_y, rot_z, trans_x, trans_y, trans_z)
            // pose layout is (q_w, q_x, q_y, q_z, trans_x, trans_y, trans_z)
            Matx<double, 7, 6> j = concatVert(concatHor(qj, Matx43d()),
                                              concatHor(Matx33d(), Matx33d::eye()));
            cachedJac.push_back(j);
        }

        // fill jtj and jtb
        for (const auto& e : edges)
        {
            size_t srcId = e.sourceNodeId, dstId = e.targetNodeId;
            const Node& srcNode = nodes.at(srcId);
            const Node& dstNode = nodes.at(dstId);

            Pose3d srcP = srcNode.pose;
            Pose3d tgtP = dstNode.pose;
            bool srcFixed = srcNode.isFixed;
            bool dstFixed = dstNode.isFixed;

            Vec6d res;
            Matx<double, 6, 3> stj, ttj;
            Matx<double, 6, 4> sqj, tqj;
            poseError(srcP.q, srcP.t, tgtP.q, tgtP.t, e.pose.q, e.pose.t, e.sqrtInfo,
                      /* needJacobians = */ true, sqj, stj, tqj, ttj, res);

            size_t srcPlace = (size_t)(-1), dstPlace = (size_t)(-1);
            Matx66d sj, tj;
            if (!srcFixed)
            {
                srcPlace = idToPlace.at(srcId);
                sj = concatHor(sqj, stj) * cachedJac[srcPlace];

                jtj.refBlock(srcPlace, srcPlace) += sj.t() * sj;

                Vec6f jtbSrc = sj.t() * res;
                for (int i = 0; i < 6; i++)
                {
                    jtb[6 * srcPlace + i] += -jtbSrc[i];
                }
            }

            if (!dstFixed)
            {
                dstPlace = idToPlace.at(dstId);
                tj = concatHor(tqj, ttj) * cachedJac[dstPlace];

                jtj.refBlock(dstPlace, dstPlace) += tj.t() * tj;

                Vec6f jtbDst = tj.t() * res;
                for (int i = 0; i < 6; i++)
                {
                    jtb[6 * dstPlace + i] += -jtbDst[i];
                }
            }

            if (!(srcFixed || dstFixed))
            {
                Matx66d sjttj = sj.t() * tj;
                jtj.refBlock(srcPlace, dstPlace) += sjttj;
                jtj.refBlock(dstPlace, srcPlace) += sjttj.t();
            }
        }

        CV_LOG_INFO(NULL, "#LM#s" << " energy: " << energy);

        // do the jacobian conditioning improvement used in Ceres
        if (jacobiScaling)
        {
            // L2-normalize each jacobian column
            // vec d = {d_j = sum(J_ij^2) for each column j of J} = get_diag{ J^T * J }
            // di = { 1/(1+sqrt(d_j)) }, extra +1 to avoid div by zero
            if (iter == 0)
            {
                for (size_t i = 0; i < nVars; i++)
                {
                    double ds = sqrt(jtj.valElem(i, i)) + 1.0;
                    di[i] = 1.0 / ds;
                }
            }

            doJacobiScaling(jtj, jtb, di);
        }

        double gradientMax = 0.0;
        // gradient max
        for (size_t i = 0; i < nVars; i++)
        {
            gradientMax = std::max(gradientMax, abs(jtb[i]));
        }

        // Save original diagonal of jtj matrix for LevMarq
        std::vector<double> diag(nVars);
        for (size_t i = 0; i < nVars; i++)
        {
            diag[i] = jtj.valElem(i, i);
        }

        // Solve using LevMarq and get delta transform
        bool enoughLm = false;

        decltype(nodes) tempNodes = nodes;

        while (!enoughLm && !done)
        {
            // form LevMarq matrix
            std::vector<double> lmDiag(nVars);
            for (size_t i = 0; i < nVars; i++)
            {
                double v = diag[i];
                double ld = std::min(max(v * lambdaLevMarq, minDiag), maxDiag);
                lmDiag[i] = ld;
                jtj.refElem(i, i) = v + ld;
            }

            CV_LOG_INFO(NULL, "sparse solve...");

            // use double or convert everything to float
            std::vector<double> x;
            bool solved = jtj.sparseSolve(jtb, x, false);

            CV_LOG_INFO(NULL, (solved ? "OK" : "FAIL"));

            double costChange = 0.0;
            double jacCostChange = 0.0;
            double stepQuality = 0.0;
            double xNorm2 = 0.0;
            if (solved)
            {
                jacCostChange = calcJacCostChange(jtb, x, lmDiag);

                // x squared norm
                for (size_t i = 0; i < nVars; i++)
                {
                    xNorm2 += x[i] * x[i];
                }

                // undo jacobi scaling
                if (jacobiScaling)
                {
                    for (size_t i = 0; i < nVars; i++)
                    {
                        x[i] *= di[i];
                    }
                }

                tempNodes = nodes;

                // Update temp nodes using x
                for (size_t i = 0; i < nVarNodes; i++)
                {
                    Vec6d dx(&x[i * 6]);
                    Vec3d deltaRot(dx[0], dx[1], dx[2]), deltaTrans(dx[3], dx[4], dx[5]);
                    Pose3d& p = tempNodes.at(placesIds[i]).pose;

                    p.q = Quatd(0, deltaRot[0], deltaRot[1], deltaRot[2]).exp() * p.q;
                    p.t += deltaTrans;
                }

                // calc energy with temp nodes
                energy = calcEnergyNodes(tempNodes);

                costChange = oldEnergy - energy;

                stepQuality = costChange / jacCostChange;

                CV_LOG_INFO(NULL, "#LM#" << iter
                               << " energy: " << energy
                               << " deltaEnergy: " << costChange
                               << " deltaEqEnergy: " << jacCostChange
                               << " max(J^T*b): " << gradientMax
                               << " norm2(x): " << xNorm2
                               << " deltaEnergy/energy: " << costChange / energy);
            }

            if (!solved || costChange < 0)
            {
                // failed to optimize, increase lambda and repeat

                lambdaLevMarq *= lmUpFactor;
                lmUpFactor *= 2.0;

                CV_LOG_INFO(NULL, "LM goes up, lambda: " << lambdaLevMarq << ", old energy: " << oldEnergy);
            }
            else
            {
                // optimized successfully, decrease lambda and set variables for next iteration
                enoughLm = true;

                lambdaLevMarq *= std::max(1.0 / initialLmDownFactor, 1.0 - pow(2.0 * stepQuality - 1.0, 3));
                lmUpFactor = initialLmUpFactor;

                smallGradient = (gradientMax < minGradientTolerance);
                smallStep = (xNorm2 < stepNorm2Tolerance);
                smallEnergyDelta = (costChange / energy < relEnergyDeltaTolerance);

                nodes = tempNodes;

                CV_LOG_INFO(NULL, "#" << iter << " energy: " << energy);

                oldEnergy = energy;

                CV_LOG_INFO(NULL, "LM goes down, lambda: " << lambdaLevMarq << " step quality: " << stepQuality);
            }

            iter++;

            tooLong = (iter >= maxIterations);

            done = ( (checkIterations && tooLong) || smallGradient || smallStep || (checkEps && smallEnergyDelta) );
        }
    }

    // if maxIterations is given as a stop criteria, let's consider tooLong as a successful finish
    bool found = (smallGradient || smallStep || smallEnergyDelta || (checkIterations && tooLong));

    CV_LOG_INFO(NULL, "Finished: " << (found ? "" : "not") << "found");
    if (smallGradient)
        CV_LOG_INFO(NULL, "Finish reason: gradient max val dropped below threshold");
    if (smallStep)
        CV_LOG_INFO(NULL, "Finish reason: step size dropped below threshold");
    if (smallEnergyDelta)
        CV_LOG_INFO(NULL, "Finish reason: relative energy change between iterations dropped below threshold");
    if (tooLong)
        CV_LOG_INFO(NULL, "Finish reason: max number of iterations reached");

    return (found ? iter : -1);
}

#else
int PoseGraphImpl::optimize(const cv::TermCriteria& /*tc*/)
{
    CV_Error(Error::StsNotImplemented, "Eigen library required for sparse matrix solve during pose graph optimization, dense solver is not implemented");
}
#endif


Ptr<detail::PoseGraph> detail::PoseGraph::create()
{
    return makePtr<PoseGraphImpl>();
}

PoseGraph::~PoseGraph() { }

}  // namespace detail
}  // namespace kinfu
}  // namespace cv
