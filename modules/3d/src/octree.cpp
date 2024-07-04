// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "octree.hpp"
#include "opencv2/3d.hpp"

namespace cv{

OctreeNode::OctreeNode() :
    children(),
    depth(0),
    size(0),
    origin(0,0,0),
    neigh(),
    parentIndex(-1)
{ }

OctreeNode::OctreeNode(int _depth, double _size, const Point3f &_origin, int _parentIndex) :
    children(),
    depth(_depth),
    size(_size),
    origin(_origin),
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

    return (ptEps.x >= this->origin.x) &&
           (ptEps.y >= this->origin.y) &&
           (ptEps.z >= this->origin.z) &&
           (_point.x <= upPt.x) &&
           (_point.y <= upPt.y) &&
           (_point.z <= upPt.z);
}

struct Octree::Impl
{
public:
    Impl() : Impl(0, 0, {0, 0, 0}, 0, false) { }

    Impl(int _maxDepth, double _size, const Point3f& _origin, double _resolution,
         bool _hasColor) :
        maxDepth(_maxDepth),
        size(_size),
        origin(_origin),
        resolution(_resolution),
        hasColor(_hasColor)
    { }

    ~Impl() { }

    void fill(bool useResolution, InputArray pointCloud, InputArray colorAttribute);
    bool insertPoint(const Point3f& point, const Point3f &color);

    // The pointer to Octree root node
    Ptr <OctreeNode> rootNode = nullptr;
    //! Max depth of the Octree
    int maxDepth;
    //! The size of the cube
    double size;
    //! The origin coordinate of root node
    Point3f origin;
    //! The size of the leaf node
    double resolution;
    //! Whether the point cloud has a color attribute
    bool hasColor;
};

Octree::Octree() :
    p(makePtr<Impl>())
{ }

Ptr<Octree> Octree::createWithDepth(int maxDepth, double size, const Point3f& origin, bool withColors)
{
    CV_Assert(maxDepth > 0);
    CV_Assert(size > 0);

    Ptr<Octree> octree = makePtr<Octree>();
    octree->p = makePtr<Impl>(maxDepth, size, origin, /*resolution*/ 0, withColors);
    return octree;
}

Ptr<Octree> Octree::createWithDepth(int maxDepth, InputArray pointCloud, InputArray colors)
{
    CV_Assert(maxDepth > 0);

    Ptr<Octree> octree = makePtr<Octree>();
    octree->p->maxDepth = maxDepth;
    octree->p->fill(/* useResolution */ false, pointCloud, colors);
    return octree;
}

Ptr<Octree> Octree::createWithResolution(double resolution, double size, const Point3f& origin, bool withColors)
{
    CV_Assert(resolution > 0);
    CV_Assert(size > 0);

    Ptr<Octree> octree = makePtr<Octree>();
    octree->p = makePtr<Impl>(/*maxDepth*/ 0, size, origin, resolution, withColors);
    return octree;
}

Ptr<Octree> Octree::createWithResolution(double resolution, InputArray pointCloud, InputArray colors)
{
    CV_Assert(resolution > 0);

    Ptr<Octree> octree = makePtr<Octree>();
    octree->p->resolution = resolution;
    octree->p->fill(/* useResolution */ true, pointCloud, colors);
    return octree;
}

Octree::~Octree() { }

bool Octree::insertPoint(const Point3f& point, const Point3f &color)
{
    return p->insertPoint(point, color);
}

bool Octree::Impl::insertPoint(const Point3f& point, const Point3f &color)
{
    size_t depthMask = (size_t)(1ULL << (this->maxDepth - 1));

    if(this->rootNode.empty())
    {
        this->rootNode = new OctreeNode( 0, this->size, this->origin, -1);
    }

    bool pointInBoundFlag = this->rootNode->isPointInBound(point);
    if(this->rootNode->depth == 0 && !pointInBoundFlag)
    {
        return false;
    }

    OctreeKey key((size_t)floor((point.x - this->origin.x) / this->resolution),
                  (size_t)floor((point.y - this->origin.y) / this->resolution),
                  (size_t)floor((point.z - this->origin.z) / this->resolution));

    Ptr<OctreeNode> node = this->rootNode;
    while (node->depth != maxDepth)
    {
        double childSize = node->size * 0.5;

        // calculate the index and the origin of child.
        size_t childIndex = key.findChildIdxByMask(depthMask);
        size_t xIndex = (childIndex & 1) ? 1 : 0;
        size_t yIndex = (childIndex & 2) ? 1 : 0;
        size_t zIndex = (childIndex & 4) ? 1 : 0;

        Point3f childOrigin = node->origin + Point3f(float(xIndex), float(yIndex), float(zIndex)) * float(childSize);

        Ptr<OctreeNode> &childPtr = node->children[childIndex];
        if (!childPtr)
        {
            childPtr = new OctreeNode(node->depth + 1, childSize, childOrigin, int(childIndex));
            childPtr->parent = node;
        }

        node = childPtr;
        depthMask = depthMask >> 1;
    }

    node->isLeaf = true;
    node->pointList.push_back(point);
    node->colorList.push_back(color);

    return true;
}


static Vec6f getBoundingBox(const Mat& points)
{
    const float mval = std::numeric_limits<float>::max();
    Vec6f bb(mval, mval, mval, -mval, -mval, -mval);

    for (int i = 0; i < (int)points.total(); i++)
    {
        Point3f pt = points.at<Point3f>(i);
        bb[0] = min(bb[0], pt.x);
        bb[1] = min(bb[1], pt.y);
        bb[2] = min(bb[2], pt.z);
        bb[3] = max(bb[3], pt.x);
        bb[4] = max(bb[4], pt.y);
        bb[5] = max(bb[5], pt.z);
    }

    return bb;
}

void Octree::Impl::fill(bool useResolution, InputArray _points, InputArray _colors)
{
    CV_CheckFalse(_points.empty(), "No points provided");

    Mat points, colors;
    int nPoints = 0, nColors = 0;

    int pointType = _points.type();
    CV_Assert(pointType == CV_32FC1 || pointType == CV_32FC3);
    points = _points.getMat();
    // transform 3xN matrix to Nx3, except 3x3
    if ((_points.channels() == 1) && (_points.rows() == 3) && (_points.cols() != 3))
    {
        points = points.t();
    }
    // This transposition is performed on 1xN matrix so it's almost free in terms of performance
    points = points.reshape(3, 1).t();
    nPoints = (int)points.total();

    if (!_colors.empty())
    {
        int colorType = _colors.type();
        CV_Assert(colorType == CV_32FC1 || colorType == CV_32FC3);
        colors = _colors.getMat();
        // transform 3xN matrix to Nx3, except 3x3
        if ((_colors.channels() == 1) && (_colors.rows() == 3) && (_colors.cols() != 3))
        {
            colors = colors.t();
        }
        colors = colors.reshape(3, 1).t();
        nColors = (int)colors.total();

        CV_Assert(nColors == nPoints);
        this->hasColor = true;
    }

    Vec6f bbox = getBoundingBox(points);
    Point3f minBound(bbox[0], bbox[1], bbox[2]);
    Point3f maxBound(bbox[3], bbox[4], bbox[5]);

    double maxSize = max(max(maxBound.x - minBound.x, maxBound.y - minBound.y), maxBound.z - minBound.z);

    // Extend maxSize to the closest power of 2 that exceeds it for bit operations
    maxSize = double(1 << int(ceil(log2(maxSize))));

    // to calculate maxDepth from resolution or vice versa
    if (useResolution)
    {
        this->maxDepth = (int)ceil(log2(maxSize / this->resolution));
    }
    else
    {
        this->resolution = (maxSize / (1 << (this->maxDepth + 1)));
    }

    this->size = (1 << this->maxDepth) * this->resolution;
    this->origin = Point3f(float(floor(minBound.x / this->resolution) * this->resolution),
                           float(floor(minBound.y / this->resolution) * this->resolution),
                           float(floor(minBound.z / this->resolution) * this->resolution));

    // Insert every point in PointCloud data.
    for (int idx = 0; idx < nPoints; idx++)
    {
        Point3f pt = points.at<Point3f>(idx);
        Point3f insertColor = this->hasColor ? colors.at<Point3f>(idx) : Point3f { };
        if (!this->insertPoint(pt, insertColor))
        {
            CV_Error(Error::StsBadArg, "The point is out of boundary!");
        }
    }
}


void Octree::clear()
{
    p = makePtr<Impl>();
}

bool Octree::empty() const
{
    return p->rootNode.empty();
}


bool Octree::isPointInBound(const Point3f& _point) const
{
    return p->rootNode->isPointInBound(_point);
}

bool Octree::deletePoint(const Point3f& point)
{
    OctreeKey key = OctreeKey((size_t)floor((point.x - this->p->origin.x) / p->resolution),
                              (size_t)floor((point.y - this->p->origin.y) / p->resolution),
                              (size_t)floor((point.z - this->p->origin.z) / p->resolution));
    size_t depthMask = (size_t)1 << (p->maxDepth - 1);

    Ptr<OctreeNode> node = p->rootNode;
    while(node)
    {
        if (node->empty())
        {
            node = nullptr;
        }
        else if (node->isLeaf)
        {
            const float eps = 1e-9f;
            bool found = std::any_of(node->pointList.begin(), node->pointList.end(),
                [point, eps](const Point3f& pt) -> bool
                {
                    return abs(point.x - pt.x) < eps &&
                           abs(point.y - pt.y) < eps &&
                           abs(point.z - pt.z) < eps;
                });
            if (!found)
                node = nullptr;
            break;
        }
        else
        {
            node = node->children[key.findChildIdxByMask(depthMask)];
            depthMask = depthMask >> 1;
        }
    }

    if(!node)
        return false;

    const float eps = 1e-9f;

    // we've found a leaf node and delete all verts equal to given one
    size_t ctr = 0;
    while (!node->pointList.empty() && ctr < node->pointList.size())
    {
        if (abs(point.x - node->pointList[ctr].x) < eps &&
            abs(point.y - node->pointList[ctr].y) < eps &&
            abs(point.z - node->pointList[ctr].z) < eps)
        {
            node->pointList.erase(node->pointList.begin() + ctr);
        }
        else
        {
            ctr++;
        }
    }

    if (node->pointList.empty())
    {
        // empty node and its empty parents should be removed
        OctreeNode *parentPtr = node->parent;
        int parentdIdx = node->parentIndex;

        while (parentPtr)
        {
            parentPtr->children[parentdIdx].release();

            // check if all children were deleted
            bool deleteFlag = true;
            for (size_t i = 0; i < 8; i++)
            {
                if (!parentPtr->children[i].empty())
                {
                    deleteFlag = false;
                    break;
                }
            }

            if (deleteFlag)
            {
                // we're at empty node, going up
                parentdIdx = parentPtr->parentIndex;
                parentPtr = parentPtr->parent;
            }
            else
            {
                // reached first non-empty node, stopping
                parentPtr = nullptr;
            }
        }
    }
    return true;
}


void Octree::getPointCloudByOctree(OutputArray restorePointCloud, OutputArray restoreColor)
{
    Ptr<OctreeNode> root = p->rootNode;
    double resolution = p->resolution;
    std::vector<Point3f> outPts, outColors;

    typedef std::tuple<Ptr<OctreeNode>, size_t, size_t, size_t> stack_element;
    std::stack<stack_element> toCheck;
    toCheck.push(stack_element(root, 0, 0, 0));
    while (!toCheck.empty())
    {
        auto top = toCheck.top();
        toCheck.pop();
        Ptr<OctreeNode> node = std::get<0>(top);
        size_t x_key = std::get<1>(top);
        size_t y_key = std::get<2>(top);
        size_t z_key = std::get<3>(top);

        if (node->isLeaf)
        {
            outPts.emplace_back(
                    (float) (resolution * x_key) + (float) (resolution * 0.5) + p->origin.x,
                    (float) (resolution * y_key) + (float) (resolution * 0.5) + p->origin.y,
                    (float) (resolution * z_key) + (float) (resolution * 0.5) + p->origin.z);
            if (p->hasColor)
            {
                Point3f avgColor { };
                for (const auto& c : node->colorList)
                {
                    avgColor += c;
                }
                avgColor *= (1.f/(float)node->colorList.size());
                outColors.emplace_back(avgColor);
            }
        }
        else
        {
            unsigned char x_mask = 1;
            unsigned char y_mask = 2;
            unsigned char z_mask = 4;
            for (unsigned char i = 0; i < 8; i++)
            {
                size_t x_copy = x_key;
                size_t y_copy = y_key;
                size_t z_copy = z_key;
                if (!node->children[i].empty())
                {
                    size_t x_offSet = !!(x_mask & i);
                    size_t y_offSet = !!(y_mask & i);
                    size_t z_offSet = !!(z_mask & i);
                    x_copy = (x_copy << 1) | x_offSet;
                    y_copy = (y_copy << 1) | y_offSet;
                    z_copy = (z_copy << 1) | z_offSet;
                    toCheck.push(stack_element(node->children[i], x_copy, y_copy, z_copy));
                }
            }
        }
    }

    if (restorePointCloud.needed())
    {
        Mat(outPts).copyTo(restorePointCloud);
    }
    if (restoreColor.needed())
    {
        Mat(outColors).copyTo(restoreColor);
    }
}


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


int Octree::radiusNNSearch(const Point3f& query, float radius, OutputArray pointSet, OutputArray squareDistSet) const
{
    return this->radiusNNSearch(query, radius, pointSet, noArray(), squareDistSet);
}

int Octree::radiusNNSearch(const Point3f& query, float radius, OutputArray points, OutputArray colors, OutputArray squareDists) const
{
    std::vector<Point3f> outPoints, outColors;
    std::vector<float> outSqDists;

    if (!p->rootNode.empty())
    {
        float squareRadius = radius * radius;

        std::vector<std::tuple<float, Point3f, Point3f>> candidatePoints;

        std::stack<Ptr<OctreeNode>> toCheck;
        toCheck.push(p->rootNode);

        while (!toCheck.empty())
        {
            Ptr<OctreeNode> node = toCheck.top();
            toCheck.pop();
            for(size_t i = 0; i < 8; i++)
            {
                Ptr<OctreeNode> child = node->children[i];
                if( child && child->overlap(query, squareRadius))
                {
                    if(child->isLeaf)
                    {
                        for(size_t j = 0; j < child->pointList.size(); j++)
                        {
                            Point3f pt = child->pointList[j];
                            Point3f col;
                            if (!child->colorList.empty())
                            {
                                col = child->colorList[j];
                            }
                            float dist = SquaredDistance(pt, query);
                            if(dist + dist * std::numeric_limits<float>::epsilon() <= squareRadius)
                            {
                                candidatePoints.emplace_back(dist, pt, col);
                            }
                        }
                    }
                    else
                    {
                        toCheck.push(child);
                    }
                }
            }
        }

        for (size_t i = 0; i < candidatePoints.size(); i++)
        {
            auto cp = candidatePoints[i];
            outSqDists.push_back(std::get<0>(cp));
            outPoints.push_back(std::get<1>(cp));
            outColors.push_back(std::get<2>(cp));
        }
    }

    if (points.needed())
    {
        Mat(outPoints).copyTo(points);
    }
    if (colors.needed())
    {
        CV_Assert(this->p->hasColor);
        Mat(outColors).copyTo(colors);
    }
    if (squareDists.needed())
    {
        Mat(outSqDists).copyTo(squareDists);
    }

    return int(outPoints.size());
}


void OctreeNode::KNNSearchRecurse(const Point3f& query, const int K,
                                  float& smallestDist, std::vector<std::tuple<float, Point3f, Point3f>>& candidatePoint) const
{
    std::vector<std::pair<float, int>> priorityQue;

    // Add the non-empty OctreeNode to priorityQue
    for(size_t i = 0; i < 8; i++)
    {
        Ptr<OctreeNode> child = this->children[i];
        if(child)
        {
            float halfSize = float(child->size * 0.5);

            Point3f center = child->origin + Point3f(halfSize, halfSize, halfSize);

            float dist = SquaredDistance(query, center);
            priorityQue.emplace_back(dist, int(i));
        }
    }

    std::sort(priorityQue.rbegin(), priorityQue.rend(),
        [](const std::pair<float, int>& a, const std::pair<float, int>& b) -> bool
        {
            return std::get<0>(a) < std::get<0>(b);
        });
    Ptr<OctreeNode> child = this->children[std::get<1>(priorityQue.back())];

    while (!priorityQue.empty() && child->overlap(query, smallestDist))
    {
        if (!child->isLeaf)
        {
            child->KNNSearchRecurse(query, K, smallestDist, candidatePoint);
        }
        else
        {
            for (size_t i = 0; i < child->pointList.size(); i++)
            {
                float dist = SquaredDistance(child->pointList[i], query);

                if ( dist + dist * std::numeric_limits<float>::epsilon() <= smallestDist )
                {
                    Point3f pt = child->pointList[i];
                    Point3f col { };
                    if (child->colorList.empty())
                    {
                        col = child->colorList[i];
                    }
                    candidatePoint.emplace_back(dist, pt, col);
                }
            }

            std::sort(candidatePoint.begin(), candidatePoint.end(),
                [](const std::tuple<float, Point3f, Point3f>& a, const std::tuple<float, Point3f, Point3f>& b) -> bool
                {
                    return std::get<0>(a) < std::get<0>(b);
                }
            );

            if (int(candidatePoint.size()) > K)
            {
                candidatePoint.resize(K);
            }

            if (int(candidatePoint.size()) == K)
            {
                smallestDist = std::get<0>(candidatePoint.back());
            }
        }

        priorityQue.pop_back();

        // To next child
        if(!priorityQue.empty())
        {
            child = this->children[std::get<1>(priorityQue.back())];
        }
    }
}


void Octree::KNNSearch(const Point3f &query, const int K, OutputArray pointSet, OutputArray squareDistSet) const
{
    this->KNNSearch(query, K, pointSet, noArray(), squareDistSet);
}

void Octree::KNNSearch(const Point3f &query, const int K, OutputArray points, OutputArray colors, OutputArray squareDists) const
{
    std::vector<Point3f> outPoints, outColors;
    std::vector<float> outSqDists;

    if (!p->rootNode.empty())
    {
        std::vector<std::tuple<float, Point3f, Point3f>> candidatePoints;
        float smallestDist = std::numeric_limits<float>::max();

        p->rootNode->KNNSearchRecurse(query, K, smallestDist, candidatePoints);

        for(size_t i = 0; i < candidatePoints.size(); i++)
        {
            auto cp = candidatePoints[i];
            outSqDists.push_back(std::get<0>(cp));
            outPoints.push_back(std::get<1>(cp));
            outColors.push_back(std::get<2>(cp));
        }
    }

    if (points.needed())
    {
        Mat(outPoints).copyTo(points);
    }
    if (colors.needed())
    {
        CV_Assert(this->p->hasColor);
        Mat(outColors).copyTo(colors);
    }
    if (squareDists.needed())
    {
        Mat(outSqDists).copyTo(squareDists);
    }
}
}