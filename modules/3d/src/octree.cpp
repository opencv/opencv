// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "octree.hpp"
#include "opencv2/3d.hpp"

namespace cv{

void getPointRecurse(std::vector<Point3f> &restorePointCloud, std::vector<Point3f> &restoreColor, size_t x_key, size_t y_key,
                     size_t z_key, Ptr<OctreeNode> &_node, double resolution, Point3f ori, bool hasColor);

// Locate the OctreeNode corresponding to the input point from the given OctreeNode.
static Ptr<OctreeNode> index(const Point3f& point, Ptr<OctreeNode>& node,OctreeKey& key,size_t depthMask);

static bool insertPointRecurse( Ptr<OctreeNode>& node, const Point3f& point, const Point3f &color, size_t maxDepth
        ,const OctreeKey &key, size_t depthMask);
bool deletePointRecurse( Ptr<OctreeNode>& node);

// For Nearest neighbor search.
template<typename T> struct PQueueElem; // Priority queue
static void radiusNNSearchRecurse(const Ptr<OctreeNode>& node, const Point3f& query, float squareRadius,
                                  std::vector<PQueueElem<Point3f> >& candidatePoint);
static void KNNSearchRecurse(const Ptr<OctreeNode>& node, const Point3f& query, const int K,
                             float& smallestDist, std::vector<PQueueElem<Point3f> >& candidatePoint);

OctreeNode::OctreeNode() :
    children(),
    depth(0),
    size(0),
    origin(0,0,0),
    pointNum(0),
    neigh(),
    parentIndex(-1)
{ }

OctreeNode::OctreeNode(int _depth, double _size, const Point3f &_origin, const Point3f &_color,
                       int _parentIndex) :
    children(),
    depth(_depth),
    size(_size),
    origin(_origin),
    color(_color),
    pointNum(0),
    neigh(),
    parentIndex(_parentIndex)
{ }

bool OctreeNode::empty() const
{
    if(this->isLeaf)
    {
        if(this->pointList.empty())
            return true;
        else
            return false;
    }
    else
    {
        for(size_t i = 0; i < 8; i++)
        {
            if(!this->children[i].empty())
            {
                return false;
            }
        }
        return true;
    }
}


bool OctreeNode::isPointInBound(const Point3f& _point) const
{
    Point3f eps;
    eps.x = std::max(std::abs(_point.x), std::abs(this->origin.x));
    eps.y = std::max(std::abs(_point.y), std::abs(this->origin.y));
    eps.z = std::max(std::abs(_point.z), std::abs(this->origin.z));
    eps *= std::numeric_limits<float>::epsilon();
    Point3f ptEps = _point + eps;
    Point3f upPt = this->origin + eps + Point3f {(float)this->size, (float)this->size, (float)this->size};

    return (ptEps.x >= this->origin.x) && (ptEps.y >= this->origin.y) && (ptEps.z >= this->origin.z) &&
           (_point.x <= upPt.x) && (_point.y <= upPt.y) && (_point.z <= upPt.z);
}

struct Octree::Impl
{
public:
    Impl():maxDepth(0), size(0), origin(0,0,0), resolution(0)
    {}

    ~Impl()
    {}

    // The pointer to Octree root node.
    Ptr <OctreeNode> rootNode = nullptr;
    //! Max depth of the Octree. And depth must be greater than zero
    size_t maxDepth;
    //! The size of the cube of the .
    double size;
    //! The origin coordinate of root node.
    Point3f origin;
    //! The size of the leaf node.
    double resolution;
    //! Whether the point cloud has a color attribute.
    bool hasColor{};
};

Octree::Octree() : p(new Impl)
{
    p->maxDepth = 0;
    p->size = 0;
    p->origin = Point3f(0,0,0);
}

Octree::Octree(size_t _maxDepth, double _size, const Point3f& _origin ) : p(new Impl)
{
    p->maxDepth = _maxDepth;
    p->size = _size;
    p->origin = _origin;
}

Octree::Octree(const std::vector<Point3f>& _pointCloud, double resolution) : p(new Impl)
{
    std::vector<Point3f> v;
    this->create(_pointCloud,v, resolution);
}

Octree::Octree(size_t _maxDepth) : p(new Impl)
{
    p->maxDepth = _maxDepth;
    p->size = 0;
    p->origin = Point3f(0,0,0);
}

Octree::~Octree(){}

bool Octree::insertPoint(const Point3f& point){
    return insertPoint(point,Point3f(0,0,0));
}

bool Octree::insertPoint(const Point3f& point,const Point3f &color)
{
    double resolution=p->resolution;
    size_t depthMask=(size_t)1 << (p->maxDepth - 1);
    if(p->rootNode.empty())
    {
        p->rootNode = new OctreeNode( 0, p->size, p->origin,  color, -1);
    }
    bool pointInBoundFlag = p->rootNode->isPointInBound(point);
    if(p->rootNode->depth==0 && !pointInBoundFlag)
    {
        return false;
    }
    OctreeKey key((size_t)floor((point.x - this->p->origin.x) / resolution),
                  (size_t)floor((point.y - this->p->origin.y) / resolution),
                  (size_t)floor((point.z - this->p->origin.z) / resolution));

    bool result = insertPointRecurse(p->rootNode, point, color, p->maxDepth, key, depthMask);
    return result;
}


bool Octree::create(const std::vector<Point3f> &pointCloud, const std::vector<Point3f> &colorAttribute, double resolution)
{

    if (resolution > 0) {
        p->resolution = resolution;
    }
    else{
        CV_Error(Error::StsBadArg, "The resolution must be greater than 0!");
    }

    if (pointCloud.empty())
        return false;

    Point3f maxBound(pointCloud[0]);
    Point3f minBound(pointCloud[0]);

    // Find center coordinate of PointCloud data.
    for (auto idx: pointCloud) {
        maxBound.x = max(idx.x, maxBound.x);
        maxBound.y = max(idx.y, maxBound.y);
        maxBound.z = max(idx.z, maxBound.z);

        minBound.x = min(idx.x, minBound.x);
        minBound.y = min(idx.y, minBound.y);
        minBound.z = min(idx.z, minBound.z);
    }

    double maxSize = max(max(maxBound.x - minBound.x, maxBound.y - minBound.y), maxBound.z - minBound.z);
    //To use bit operation, the length of the root cube should be power of 2.
    maxSize=double(1<<int(ceil(log2(maxSize))));
    p->maxDepth = (size_t)ceil(log2(maxSize / resolution));
    this->p->size = (1<<p->maxDepth)*resolution;
    this->p->origin = Point3f(float(floor(minBound.x / resolution) * resolution),
                              float(floor(minBound.y / resolution) * resolution),
                              float(floor(minBound.z / resolution) * resolution));

    p->hasColor = !colorAttribute.empty();

    // Insert every point in PointCloud data.
    for (size_t idx = 0; idx < pointCloud.size(); idx++) {
        Point3f insertColor = p->hasColor ? colorAttribute[idx] : Point3f(0.0f, 0.0f, 0.0f);
        if (!insertPoint(pointCloud[idx], insertColor)) {
            CV_Error(Error::StsBadArg, "The point is out of boundary!");
        }
    }

    return true;
}

bool Octree::create(const std::vector<Point3f> &pointCloud, double resolution) {
    std::vector<Point3f> v;
    return this->create(pointCloud, v, resolution);
}

void Octree::setMaxDepth(size_t _maxDepth)
{
    if(_maxDepth )
        this->p->maxDepth = _maxDepth;
}

void Octree::setSize(double _size)
{
    this->p->size = _size;
};

void Octree::setOrigin(const Point3f& _origin)
{
    this->p->origin = _origin;
}

void Octree::clear()
{
    if(!p->rootNode.empty())
    {
        p->rootNode.release();
    }

    p->size = 0;
    p->maxDepth = 0;
    p->origin = Point3f (0,0,0); // origin coordinate
}

bool Octree::empty() const
{
    return p->rootNode.empty();
}

Ptr<OctreeNode> index(const Point3f& point, Ptr<OctreeNode>& _node,OctreeKey &key,size_t depthMask)
{
    OctreeNode &node = *_node;

    if(node.empty())
    {
        return Ptr<OctreeNode>();
    }

    if(node.isLeaf)
    {
        for(size_t i = 0; i < node.pointList.size(); i++ )
        {
            if((point.x == node.pointList[i].x) &&
               (point.y == node.pointList[i].y) &&
               (point.z == node.pointList[i].z)
                    )
            {
                return _node;
            }
        }
        return Ptr<OctreeNode>();
    }


    size_t childIndex = key.findChildIdxByMask(depthMask);
    if(!node.children[childIndex].empty())
    {
        return index(point, node.children[childIndex],key,depthMask>>1);
    }
    return Ptr<OctreeNode>();
}

bool Octree::isPointInBound(const Point3f& _point) const
{
    return p->rootNode->isPointInBound(_point);
}

bool Octree::deletePoint(const Point3f& point)
{
    OctreeKey key=OctreeKey((size_t)floor((point.x - this->p->origin.x) / p->resolution),
                            (size_t)floor((point.y - this->p->origin.y) / p->resolution),
                            (size_t)floor((point.z - this->p->origin.z) / p->resolution));
    size_t depthMask=(size_t)1 << (p->maxDepth - 1);
    Ptr<OctreeNode> node = index(point, p->rootNode,key,depthMask);

    if(!node.empty())
    {
        size_t i = 0;
        while (!node->pointList.empty() && i < node->pointList.size() )
        {
            if((point.x == node->pointList[i].x) &&
               (point.y == node->pointList[i].y) &&
               (point.z == node->pointList[i].z)
                    )
            {
                node->pointList.erase(node->pointList.begin() + i);
            } else{
                i++;
            }
        }

        // If it is the last point cloud in the OctreeNode, recursively delete the node.
        return deletePointRecurse(node);
    }
    else
    {
        return false;
    }
}

bool deletePointRecurse(Ptr<OctreeNode>& _node)
{
    OctreeNode& node = *_node;

    if(_node.empty())
        return false;

    if(node.isLeaf)
    {
        if( !node.pointList.empty())
        {
            Ptr<OctreeNode> parent = node.parent;
            parent->children[node.parentIndex] = nullptr;
            _node.release();

            return deletePointRecurse(parent);
        }
        else
        {
            return true;
        }
    }
    else
    {
        bool deleteFlag = true;

        // Only all children was deleted, can we delete the tree node.
        for(size_t i = 0; i< 8; i++)
        {
            if(!node.children[i].empty())
            {
                deleteFlag = false;
                break;
            }
        }

        if(deleteFlag)
        {
            Ptr<OctreeNode> parent = node.parent;
            _node.release();
            return deletePointRecurse(parent);
        }
        else
        {
            return true;
        }
    }
}

bool insertPointRecurse( Ptr<OctreeNode>& _node,  const Point3f& point,const Point3f &color, size_t maxDepth,const OctreeKey &key,
                         size_t depthMask)
{
    OctreeNode &node = *_node;
    //add point to the leaf node.
    if (node.depth == (int)maxDepth) {
        node.isLeaf = true;
        node.color = color;
        node.pointNum++;
        node.pointList.push_back(point);
        return true;
    }

    double childSize = node.size * 0.5;
    //calculate the index and the origin of child.
    size_t childIndex = key.findChildIdxByMask(depthMask);
    size_t xIndex = childIndex&1?1:0;
    size_t yIndex = childIndex&2?1:0;
    size_t zIndex = childIndex&4?1:0;
    Point3f childOrigin = node.origin + Point3f(xIndex * float(childSize), yIndex * float(childSize), zIndex * float(childSize));

    if (node.children[childIndex].empty()) {
        node.children[childIndex] = new OctreeNode(node.depth + 1, childSize, childOrigin, Point3f(0, 0, 0),
                                                   int(childIndex));
        node.children[childIndex]->parent = _node;
    }

    bool result = insertPointRecurse(node.children[childIndex], point, color, maxDepth, key, depthMask >> 1);
    node.pointNum += result;
    return result;
}


void Octree::getPointCloudByOctree(std::vector<Point3f> &restorePointCloud, std::vector<Point3f> &restoreColor) {
    Ptr<OctreeNode> root = p->rootNode;
    double resolution = p->resolution;
    getPointRecurse(restorePointCloud, restoreColor, 0, 0, 0, root, resolution, p->origin, p->hasColor);
}

void getPointRecurse(std::vector<Point3f> &restorePointCloud, std::vector<Point3f> &restoreColor, size_t x_key,
                     size_t y_key,size_t z_key, Ptr<OctreeNode> &_node, double resolution, Point3f ori,
                     bool hasColor) {
    OctreeNode node = *_node;
    if (node.isLeaf) {
        restorePointCloud.emplace_back(
                (float) (resolution * x_key) + (float) (resolution * 0.5) + ori.x,
                (float) (resolution * y_key) + (float) (resolution * 0.5) + ori.y,
                (float) (resolution * z_key) + (float) (resolution * 0.5) + ori.z);
        if (hasColor) {
            restoreColor.emplace_back(node.color);
        }
        return;
    }
    unsigned char x_mask = 1;
    unsigned char y_mask = 2;
    unsigned char z_mask = 4;
    for (unsigned char i = 0; i < 8; i++) {
        size_t x_copy = x_key;
        size_t y_copy = y_key;
        size_t z_copy = z_key;
        if (!node.children[i].empty()) {
            size_t x_offSet = !!(x_mask & i);
            size_t y_offSet = !!(y_mask & i);
            size_t z_offSet = !!(z_mask & i);
            x_copy = (x_copy << 1) | x_offSet;
            y_copy = (y_copy << 1) | y_offSet;
            z_copy = (z_copy << 1) | z_offSet;
            getPointRecurse(restorePointCloud, restoreColor, x_copy, y_copy, z_copy, node.children[i], resolution,
                            ori, hasColor);
        }
    }
};



static float SquaredDistance(const Point3f& query, const Point3f& origin)
{
    Point3f diff = query - origin;
    return diff.dot(diff);
}

bool OctreeNode::overlap(const Point3f& query, float squareRadius) const
{
    float halfSize = float(this->size * 0.5);
    Point3f center = this->origin + Point3f( halfSize, halfSize, halfSize );

    float dist = SquaredDistance(center, query);
    float temp = float(this->size) * float(this->size) * 3.0f;

    return ( dist + dist * std::numeric_limits<float>::epsilon() ) <= float(temp * 0.25f + squareRadius + sqrt(temp * squareRadius)) ;
}

void radiusNNSearchRecurse(const Ptr<OctreeNode>& node, const Point3f& query, float squareRadius,
                           std::vector<PQueueElem<Point3f> >& candidatePoint)
{
    float dist;
    Ptr<OctreeNode> child;

    // iterate eight children.
    for(size_t i = 0; i< 8; i++)
    {
        if( !node->children[i].empty() && node->children[i]->overlap(query, squareRadius))
        {
            if(!node->children[i]->isLeaf)
            {
                // Reach the branch node.
                radiusNNSearchRecurse(node->children[i], query, squareRadius, candidatePoint);
            }
            else
            {
                // Reach the leaf node.
                child = node->children[i];

                for(size_t j = 0; j < child->pointList.size(); j++)
                {
                    dist = SquaredDistance(child->pointList[j], query);
                    if(dist + dist * std::numeric_limits<float>::epsilon() <= squareRadius )
                    {
                        candidatePoint.emplace_back(dist, child->pointList[j]);
                    }
                }
            }
        }
    }
}

int Octree::radiusNNSearch(const Point3f& query, float radius,
                           std::vector<Point3f>& pointSet, std::vector<float>& squareDistSet) const
{
    if(p->rootNode.empty())
        return 0;
    float squareRadius = radius * radius;

    PQueueElem<Point3f> elem;
    std::vector<PQueueElem<Point3f> > candidatePoint;

    radiusNNSearchRecurse(p->rootNode, query, squareRadius, candidatePoint);

    for(size_t i = 0; i < candidatePoint.size(); i++)
    {
        pointSet.push_back(candidatePoint[i].t);
        squareDistSet.push_back(candidatePoint[i].dist);
    }
    return int(pointSet.size());
}

void KNNSearchRecurse(const Ptr<OctreeNode>& node, const Point3f& query, const int K,
                      float& smallestDist, std::vector<PQueueElem<Point3f> >& candidatePoint)
{
    std::vector<PQueueElem<int> > priorityQue;
    Ptr<OctreeNode> child;
    float dist = 0;
    Point3f center; // the OctreeNode Center

    // Add the non-empty OctreeNode to priorityQue.
    for(size_t i = 0; i < 8; i++)
    {
        if(!node->children[i].empty())
        {
            float halfSize = float(node->children[i]->size * 0.5);

            center = node->children[i]->origin + Point3f(halfSize, halfSize, halfSize);

            dist = SquaredDistance(query, center);
            priorityQue.emplace_back(dist, int(i));
        }
    }

    std::sort(priorityQue.rbegin(), priorityQue.rend());
    child = node->children[priorityQue.back().t];

    while (!priorityQue.empty() && child->overlap(query, smallestDist))
    {
        if (!child->isLeaf)
        {
            KNNSearchRecurse(child, query, K, smallestDist, candidatePoint);
        } else {
            for (size_t i = 0; i < child->pointList.size(); i++) {
                dist = SquaredDistance(child->pointList[i], query);

                if ( dist + dist * std::numeric_limits<float>::epsilon() <= smallestDist ) {
                    candidatePoint.emplace_back(dist, child->pointList[i]);
                }
            }

            std::sort(candidatePoint.begin(), candidatePoint.end());

            if (int(candidatePoint.size()) > K) {
                candidatePoint.resize(K);
            }

            if (int(candidatePoint.size()) == K) {
                smallestDist = candidatePoint.back().dist;
            }
        }

        priorityQue.pop_back();

        // To next child.
        if(!priorityQue.empty())
            child = node->children[priorityQue.back().t];
    }
}

void Octree::KNNSearch(const Point3f& query, const int K, std::vector<Point3f>& pointSet, std::vector<float>& squareDistSet) const
{
    if(p->rootNode.empty())
        return;

    PQueueElem<Ptr<Point3f> > elem;
    std::vector<PQueueElem<Point3f> > candidatePoint;
    float smallestDist = std::numeric_limits<float>::max();

    KNNSearchRecurse(p->rootNode, query, K, smallestDist, candidatePoint);

    for(size_t i = 0; i < candidatePoint.size(); i++)
    {
        pointSet.push_back(candidatePoint[i].t);
        squareDistSet.push_back(candidatePoint[i].dist);
    }
}

}