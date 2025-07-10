// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "../precomp.hpp"
#include "opencv2/3d/detail/optimizer.hpp"
#include <opencv2/3d/mst.hpp>
#include "sparse_block_matrix.hpp"

namespace cv
{
namespace detail
{

#if defined(HAVE_EIGEN)

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

// jacobian of quaternionic (exp(x)*q) : R_3 -> H near x == 0
static inline cv::Matx43d expQuatJacobian(cv::Quatd q)
{
    double w = q.w, x = q.x, y = q.y, z = q.z;
    return cv::Matx43d(-x, -y, -z,
                        w,  z, -y,
                       -z,  w,  x,
                        y, -x,  w);
}

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

class PoseGraphImpl;
class PoseGraphLevMarqBackend;

class PoseGraphLevMarq : public LevMarqBase
{
public:
    PoseGraphLevMarq(PoseGraphImpl* pg, const LevMarq::Settings& settings_ = LevMarq::Settings()) :
        LevMarqBase(makePtr<PoseGraphLevMarqBackend>(pg), settings_)
    { }
};


class PoseGraphImpl : public detail::PoseGraph
{
public:
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

        // jacobian of exponential (exp(x)* q) : R_6->SE(3) near x == 0
        inline Matx<double, 7, 6> expJacobian()
        {
            Matx43d qj = expQuatJacobian(q);
            // x node layout is (rot_x, rot_y, rot_z, trans_x, trans_y, trans_z)
            // pose layout is (q_w, q_x, q_y, q_z, trans_x, trans_y, trans_z)
            return concatVert(concatHor(qj, Matx43d()),
                              concatHor(Matx33d(), Matx33d::eye()));
        }

        inline Pose3d oplus(const Vec6d dx)
        {
            Vec3d deltaRot(dx[0], dx[1], dx[2]), deltaTrans(dx[3], dx[4], dx[5]);
            Pose3d p;
            p.q = Quatd(0, deltaRot[0], deltaRot[1], deltaRot[2]).exp() * this->q;
            p.t = this->t + deltaTrans;
            return p;
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

    PoseGraphImpl() : nodes(), edges(), lm()
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

    virtual Affine3d getEdgePose(size_t i) const CV_OVERRIDE
    {
        return edges[i].pose.getAffine();
    }
    virtual Matx66f getEdgeInfo(size_t i) const CV_OVERRIDE
    {
        Matx66f s = edges[i].sqrtInfo;
        return s * s;
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

    void initializePosesWithMST() CV_OVERRIDE;

    // creates an optimizer
    virtual Ptr<LevMarqBase> createOptimizer(const LevMarq::Settings& settings) CV_OVERRIDE
    {
        lm = makePtr<PoseGraphLevMarq>(this, settings);

        return lm;
    }

    // creates an optimizer
    virtual Ptr<LevMarqBase> createOptimizer() CV_OVERRIDE
    {
        lm = makePtr<PoseGraphLevMarq>(this, LevMarq::Settings()
                                             .setMaxIterations(100)
                                             .setCheckRelEnergyChange(true)
                                             .setRelEnergyDeltaTolerance(1e-6)
                                             .setGeodesic(true));

        return lm;
    }

    // Returns number of iterations elapsed or -1 if max number of iterations was reached or failed to optimize
    virtual LevMarq::Report optimize() CV_OVERRIDE;

    std::map<size_t, Node> nodes;
    std::vector<Edge> edges;

    Ptr<PoseGraphLevMarq> lm;

private:
    double calculateWeight(const PoseGraphImpl::Edge& e) const;
    void applyMST(const std::vector<cv::MSTEdge>& resultingEdges, const PoseGraphImpl::Node& rootNode);
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

double PoseGraphImpl::calculateWeight(const PoseGraphImpl::Edge& e) const
{
    double translationNorm = cv::norm(e.pose.t);

    cv::Matx33d R = e.pose.q.toRotMat3x3(cv::QUAT_ASSUME_UNIT);
    cv::Vec3d rvec;
    cv::Rodrigues(R, rvec);
    double rotationAngle = cv::norm(rvec);

    // empirically determined
    double lambda = 0.485;
    double weight = translationNorm + lambda * rotationAngle;

    return weight;
}

void PoseGraphImpl::applyMST(const std::vector<cv::MSTEdge>& resultingEdges, const PoseGraphImpl::Node& rootNode)
{
    std::unordered_map<size_t, std::vector<std::pair<size_t, PoseGraphImpl::Pose3d>>> adj;
    for (const auto& e: resultingEdges)
    {
        auto it = std::find_if(edges.begin(), edges.end(), [&](const PoseGraphImpl::Edge& edge)
        {
            return (edge.sourceNodeId == static_cast<size_t>(e.source) && edge.targetNodeId == static_cast<size_t>(e.target)) ||
                   (edge.sourceNodeId == static_cast<size_t>(e.target) && edge.targetNodeId == static_cast<size_t>(e.source));
        });
        if (it != edges.end())
        {
            size_t src = it->sourceNodeId;
            size_t tgt = it->targetNodeId;
            const PoseGraphImpl::Pose3d& relPose = it->pose;

            adj[src].emplace_back(tgt, relPose);
            adj[tgt].emplace_back(src, relPose.inverse());
        }
    }

    std::unordered_map<size_t, PoseGraphImpl::Pose3d> newPoses;
    std::stack<size_t> toVisit;
    std::unordered_set<size_t> visited;

    newPoses[rootNode.id] = rootNode.pose;
    toVisit.push(rootNode.id);

    while (!toVisit.empty())
    {
        size_t current = toVisit.top();
        toVisit.pop();
        visited.insert(current);

        const auto& currentPose = newPoses[current];

        auto it = adj.find(current);
        if (it == adj.end())
            continue;

        for (const auto& [neighbor, relativePose] : it->second)
        {
            if (visited.count(neighbor))
                continue;
            newPoses[neighbor] = currentPose * relativePose;
            toVisit.push(neighbor);
        }
    }

    // Apply the new poses
    for (const auto& [nodeId, pose] : newPoses)
    {
        if (!nodes.at(nodeId).isFixed)
            nodes.at(nodeId).setPose(pose.getAffine());
    }
}

void PoseGraphImpl::initializePosesWithMST()
{
    size_t numNodes = getNumNodes();

    std::vector<MSTEdge> MSTedges;
    for (const auto& e: edges)
    {
        double weight = calculateWeight(e);
        MSTedges.push_back({static_cast<int>(e.sourceNodeId), static_cast<int>(e.targetNodeId), weight});
    }

    size_t rootId = 0;
    PoseGraphImpl::Node rootNode = nodes.begin()->second;

    for (const auto& n: nodes)
    {
        if (isNodeFixed(n.second.id))
        {
            rootNode = n.second;
            rootId = n.second.id;
            break;
        }
    }

    std::vector<MSTEdge> resultingEdges;
    if (!cv::buildMST(numNodes, MSTedges, resultingEdges, MST_PRIM, static_cast<int>(rootId)))
    {
        CV_LOG_INFO(NULL, "Failed to build MST: graph may be disconnected.");
        return;
    }

    applyMST(resultingEdges, rootNode);
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
}


// J := J * d_inv, d_inv = make_diag(di)
// J^T*J := (J * d_inv)^T * J * d_inv = diag(di) * (J^T * J) * diag(di) = eltwise_mul(J^T*J, di*di^T)
// J^T*b := (J * d_inv)^T * b = d_inv^T * J^T*b = eltwise_mul(J^T*b, di)
static void doJacobiScalingSparse(BlockSparseMat<double, 6, 6>& jtj, Mat_<double>& jtb, const Mat_<double>& di)
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
                m(i, j) *= di(pt.x) * di(pt.y);
            }
        }
    }

    // scaling J^T*b
    jtb = jtb.mul(di);
}

//TODO: robustness
class PoseGraphLevMarqBackend : public detail::LevMarqBackend
{
public:
    PoseGraphLevMarqBackend(PoseGraphImpl* pg_) :
        LevMarqBackend(),
        pg(pg_),
        jtj(0),
        jtb(),
        tempNodes(),
        useGeo(),
        geoNodes(),
        jtrvv(),
        jtCached(),
        decomposition(),
        numNodes(),
        numEdges(),
        placesIds(),
        idToPlace(),
        nVarNodes()
    {
        if (!pg->isValid())
        {
            CV_Error(cv::Error::Code::StsBadArg, "Invalid PoseGraph that is either not connected or has invalid nodes");
        }

        this->numNodes = pg->getNumNodes();
        this->numEdges = pg->getNumEdges();

        // Allocate indices for nodes
        for (const auto& ni : pg->nodes)
        {
            if (!ni.second.isFixed)
            {
                this->idToPlace[ni.first] = this->placesIds.size();
                this->placesIds.push_back(ni.first);
            }
        }

        this->nVarNodes = this->placesIds.size();
        if (!this->nVarNodes)
        {
            CV_Error(cv::Error::Code::StsBadArg, "PoseGraph contains no non-constant nodes, skipping optimization");
        }

        if (!this->numEdges)
        {
            CV_Error(cv::Error::Code::StsBadArg, "PoseGraph has no edges, no optimization to be done");
        }

        CV_LOG_INFO(NULL, "Optimizing PoseGraph with " << this->numNodes << " nodes and " << this->numEdges << " edges");

        this->nVars = this->nVarNodes * 6;
    }


    virtual bool calcFunc(double& energy, bool calcEnergy = true, bool calcJacobian = false) CV_OVERRIDE
    {
        std::map<size_t, PoseGraphImpl::Node>& nodes = tempNodes;

        std::vector<cv::Matx<double, 7, 6>> cachedJac;
        if (calcJacobian)
        {
            jtj.clear();
            std::fill(jtb.begin(), jtb.end(), 0.0);

            // caching nodes jacobians
            for (auto id : placesIds)
            {
                cachedJac.push_back(nodes.at(id).pose.expJacobian());
            }

            if (useGeo)
                jtCached.clear();
        }

        double totalErr = 0.0;
        for (const auto& e : pg->edges)
        {
            size_t srcId = e.sourceNodeId, dstId = e.targetNodeId;
            const PoseGraphImpl::Node& srcNode = nodes.at(srcId);
            const PoseGraphImpl::Node& dstNode = nodes.at(dstId);

            const PoseGraphImpl::Pose3d& srcP = srcNode.pose;
            const PoseGraphImpl::Pose3d& tgtP = dstNode.pose;
            bool srcFixed = srcNode.isFixed;
            bool dstFixed = dstNode.isFixed;

            Vec6d res;
            Matx<double, 6, 3> stj, ttj;
            Matx<double, 6, 4> sqj, tqj;

            double err = poseError(srcP.q, srcP.t, tgtP.q, tgtP.t, e.pose.q, e.pose.t, e.sqrtInfo,
             /* needJacobians = */ calcJacobian, sqj, stj, tqj, ttj, res);
            totalErr += err;

            if (calcJacobian)
            {
                size_t srcPlace = (size_t)(-1), dstPlace = (size_t)(-1);
                Matx66d sj, tj;
                if (!srcFixed)
                {
                    srcPlace = idToPlace.at(srcId);
                    sj = concatHor(sqj, stj) * cachedJac[srcPlace];

                    jtj.refBlock(srcPlace, srcPlace) += sj.t() * sj;

                    Vec6d jtbSrc = sj.t() * res;
                    for (int i = 0; i < 6; i++)
                    {
                        jtb(6 * (int)srcPlace + i) += jtbSrc[i];
                    }
                }

                if (!dstFixed)
                {
                    dstPlace = idToPlace.at(dstId);
                    tj = concatHor(tqj, ttj) * cachedJac[dstPlace];

                    jtj.refBlock(dstPlace, dstPlace) += tj.t() * tj;

                    Vec6d jtbDst = tj.t() * res;
                    for (int i = 0; i < 6; i++)
                    {
                        jtb(6 * (int)dstPlace + i) += jtbDst[i];
                    }
                }

                if (!(srcFixed || dstFixed))
                {
                    Matx66d sjttj = sj.t() * tj;
                    jtj.refBlock(srcPlace, dstPlace) += sjttj;
                    jtj.refBlock(dstPlace, srcPlace) += sjttj.t();
                }

                if (useGeo)
                {
                    jtCached.push_back({ sj, tj });
                }
            }
        }

        if (calcEnergy)
        {
            energy = totalErr * 0.5;
        }

        return true;
    }

    virtual bool enableGeo() CV_OVERRIDE
    {
        useGeo = true;
        return true;
    }

    // adds d to current variables and writes result to probe vars or geo vars
    virtual void currentOplusX(const Mat_<double>& d, bool geo = false) CV_OVERRIDE
    {
        if (geo && !useGeo)
        {
            CV_Error(cv::Error::StsBadArg, "Geodesic acceleration is disabled");
        }

        std::map<size_t, PoseGraphImpl::Node>& nodes = geo ? geoNodes : tempNodes;

        nodes = pg->nodes;

        for (size_t i = 0; i < nVarNodes; i++)
        {
            Vec6d dx(d[0] + (i * 6));
            PoseGraphImpl::Pose3d& p = nodes.at(placesIds[i]).pose;

            p = p.oplus(dx);
        }
    }

    virtual void prepareVars() CV_OVERRIDE
    {
        jtj = BlockSparseMat<double, 6, 6>(nVarNodes);
        jtb = Mat_<double>((int)nVars, 1);
        tempNodes = pg->nodes;
        if (useGeo)
            geoNodes = pg->nodes;
    }

    // decomposes LevMarq matrix before solution
    virtual bool decompose() CV_OVERRIDE
    {
        return jtj.decompose(decomposition, false);
    }

    // solves LevMarq equation (J^T*J + lmdiag) * x = -right for current iteration using existing decomposition
    // right can be equal to J^T*b for LevMarq equation or J^T*rvv for geodesic acceleration equation
    virtual bool solveDecomposed(const Mat_<double>& right, Mat_<double>& x) CV_OVERRIDE
    {
        return jtj.solveDecomposed(decomposition, -right, x);
    }

    // calculates J^T*f(geo)
    virtual bool calcJtbv(Mat_<double>& jtbv) CV_OVERRIDE
    {
        jtbv.setZero();

        int ei = 0;
        for (const auto& e : pg->edges)
        {
            size_t srcId = e.sourceNodeId, dstId = e.targetNodeId;
            const PoseGraphImpl::Node& srcNode = geoNodes.at(srcId);
            const PoseGraphImpl::Node& dstNode = geoNodes.at(dstId);

            const PoseGraphImpl::Pose3d& srcP = srcNode.pose;
            const PoseGraphImpl::Pose3d& tgtP = dstNode.pose;
            bool srcFixed = srcNode.isFixed;
            bool dstFixed = dstNode.isFixed;

            Vec6d res;
            // dummy vars
            Matx<double, 6, 3> stj, ttj;
            Matx<double, 6, 4> sqj, tqj;

            poseError(srcP.q, srcP.t, tgtP.q, tgtP.t, e.pose.q, e.pose.t, e.sqrtInfo,
            /* needJacobians = */ false, sqj, stj, tqj, ttj, res);

            size_t srcPlace = (size_t)(-1), dstPlace = (size_t)(-1);
            Matx66d sj = jtCached[ei].first, tj = jtCached[ei].second;

            if (!srcFixed)
            {
                srcPlace = idToPlace.at(srcId);

                Vec6d jtbSrc = sj.t() * res;
                for (int i = 0; i < 6; i++)
                {
                    jtbv(6 * (int)srcPlace + i) += jtbSrc[i];
                }
            }

            if (!dstFixed)
            {
                dstPlace = idToPlace.at(dstId);

                Vec6d jtbDst = tj.t() * res;
                for (int i = 0; i < 6; i++)
                {
                    jtbv(6 * (int)dstPlace + i) += jtbDst[i];
                }
            }

            ei++;
        }

        return true;
    }

    virtual const Mat_<double> getDiag() CV_OVERRIDE
    {
        return jtj.diagonal();
    }

    virtual const Mat_<double> getJtb() CV_OVERRIDE
    {
        return jtb;
    }

    virtual void setDiag(const Mat_<double>& d) CV_OVERRIDE
    {
        for (size_t i = 0; i < nVars; i++)
        {
            jtj.refElem(i, i) = d((int)i);
        }
    }

    virtual void doJacobiScaling(const Mat_<double>& di) CV_OVERRIDE
    {
        doJacobiScalingSparse(jtj, jtb, di);
    }


    virtual void acceptProbe() CV_OVERRIDE
    {
        pg->nodes = tempNodes;
    }

    PoseGraphImpl* pg;

    // J^T*J matrix
    BlockSparseMat<double, 6, 6> jtj;
    // J^T*b vector
    Mat_<double> jtb;

    // Probe variable for different lambda tryout
    std::map<size_t, PoseGraphImpl::Node> tempNodes;

    // For geodesic acceleration
    bool useGeo;
    std::map<size_t, PoseGraphImpl::Node> geoNodes;
    Mat_<double> jtrvv;
    std::vector<std::pair<Matx66d, Matx66d>> jtCached;

    // Used for keeping intermediate matrix decomposition for further linear solve operations
    BlockSparseMat<double, 6, 6>::Decomposition decomposition;

    // The rest members are generated from pg
    size_t nVars;
    size_t numNodes;
    size_t numEdges;

    // Structures to convert node id to place in variables vector and back
    std::vector<size_t> placesIds;
    std::map<size_t, size_t> idToPlace;

    size_t nVarNodes;
};


LevMarq::Report PoseGraphImpl::optimize()
{
    if (!lm)
        createOptimizer();
    return lm->optimize();
}

Ptr<detail::PoseGraph> detail::PoseGraph::create()
{
    return makePtr<PoseGraphImpl>();
}

#else

Ptr<detail::PoseGraph> detail::PoseGraph::create()
{
    CV_Error(Error::StsNotImplemented, "Eigen library required for sparse matrix solve during pose graph optimization, dense solver is not implemented");
}

#endif

PoseGraph::~PoseGraph() { }

}  // namespace detail
}  // namespace cv
