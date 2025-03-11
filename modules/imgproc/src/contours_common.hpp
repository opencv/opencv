// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_CONTOURS_COMMON_HPP
#define OPENCV_CONTOURS_COMMON_HPP

#include "precomp.hpp"

#include "contours_blockstorage.hpp"

namespace cv {

static const schar MAX_SIZE = 16;

static const cv::Point chainCodeDeltas[8] =
    {{1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1}};

static inline int getDelta(schar s, size_t step)
{
    CV_DbgAssert(s >= 0 && s < 16);
    const cv::Point res = chainCodeDeltas[s % 8];
    return res.x + res.y * (int)step;
}

inline schar clamp_direction(schar dir)
{
    return std::min(dir, (schar)15);
}

template <typename T>
class TreeNode
{
private:
    int self_;

public:
    // tree hierarchy (parent - children)
    int parent;
    int first_child;
    // 1st linked list - bidirectional - sibling children
    int prev;
    int next;
    // 2nd linked list - unidirectional - not related to 1st list
    int ctable_next;
    T body;

public:
    TreeNode(int self, T&& body_) :
        self_(self), parent(-1), first_child(-1), prev(-1), next(-1), ctable_next(-1), body(std::move(body_))
    {
        CV_Assert(self >= 0);
    }
    TreeNode(const TreeNode&) = delete;
    TreeNode(TreeNode&&) noexcept = default;
    TreeNode& operator=(const TreeNode&) = delete;
    TreeNode& operator=(TreeNode&&) noexcept = default;
    int self() const
    {
        return self_;
    }
};

template <typename T>
class Tree
{
public:
    Tree() {}
    Tree(const Tree&) = delete;
    Tree(Tree&&) = delete;
    Tree& operator=(const Tree&) = delete;
    Tree& operator=(Tree&&) = delete;
    ~Tree() = default;
private:
    std::vector<TreeNode<T>> nodes;

public:
    TreeNode<T>& newElem(T && body_)
    {
        const size_t idx = nodes.size();
        CV_DbgAssert(idx < (size_t)std::numeric_limits<int>::max());
        nodes.emplace_back(std::move(TreeNode<T>((int)idx, std::move(body_))));
        return nodes[idx];
    }
    TreeNode<T>& elem(int idx)
    {
        CV_DbgAssert(idx >= 0 && (size_t)idx < nodes.size());
        return nodes[(size_t)idx];
    }
    const TreeNode<T>& elem(int idx) const
    {
        CV_DbgAssert(idx >= 0 && (size_t)idx < nodes.size());
        return nodes[(size_t)idx];
    }
    int lastSibling(int e) const
    {
        if (e != -1)
        {
            while (true)
            {
                const TreeNode<T>& cur_elem = elem(e);
                if (cur_elem.next == -1)
                    break;
                e = cur_elem.next;
            }
        }
        return e;
    }
    void addSiblingAfter(int prev, int idx)
    {
        TreeNode<T>& prev_item = nodes[prev];
        TreeNode<T>& child = nodes[idx];
        child.parent = prev_item.parent;
        if (prev_item.next != -1)
        {
            ((TreeNode<T>&)nodes[prev_item.next]).prev = idx;
            child.next = prev_item.next;
        }
        child.prev = prev;
        prev_item.next = idx;
    }
    void addChild(int parent_idx, int child_idx)
    {
        TreeNode<T>& parent = nodes[parent_idx];
        TreeNode<T>& child = nodes[child_idx];
        if (parent.first_child != -1)
        {
            TreeNode<T>& fchild_ = nodes[parent.first_child];
            fchild_.prev = child_idx;
            child.next = parent.first_child;
        }
        parent.first_child = child_idx;
        child.parent = parent_idx;
        child.prev = -1;
    }
    bool isEmpty() const
    {
        return nodes.size() == 0;
    }
    size_t size() const
    {
        return nodes.size();
    }
};

template <typename T>
class TreeIterator
{
public:
    TreeIterator(Tree<T>& tree_) : tree(tree_)
    {
        CV_Assert(!tree.isEmpty());
        levels.push(0);
    }
    bool isDone() const
    {
        return levels.empty();
    }
    const TreeNode<T>& getNext_s()
    {
        int idx = levels.top();
        levels.pop();
        const TreeNode<T>& res = tree.elem(idx);
        int cur = tree.lastSibling(res.first_child);
        while (cur != -1)
        {
            levels.push(cur);
            cur = tree.elem(cur).prev;
        }
        return res;
    }

private:
    Tree<T>& tree;
    std::stack<int> levels;
};

//==============================================================================

template <typename T, size_t BLOCK_SIZE_ELEM, size_t STATIC_CAPACITY_BYTES>
class ContourDataStorage
{
public:
    typedef T data_storage_t;
    typedef BlockStorage<data_storage_t, BLOCK_SIZE_ELEM, STATIC_CAPACITY_BYTES> storage_t;
public:
    ContourDataStorage(void) = delete;
    ContourDataStorage(storage_t* _storage):storage(_storage) {}
    ContourDataStorage(const ContourDataStorage&) = delete;
    ContourDataStorage(ContourDataStorage&&) noexcept = default;
    ~ContourDataStorage() = default;
    ContourDataStorage& operator=(const ContourDataStorage&) = delete;
    ContourDataStorage& operator=(ContourDataStorage&&) noexcept = default;
public:
    typename storage_t::RangeIterator getRangeIterator(void) const {return storage->getRangeIterator(first, last);}
public:
    bool empty(void) const {return first == last;}
    size_t size(void) const {return last - first;}
public:
    void clear(void) {first = last;}
    bool resize(size_t newSize)
    {
        bool ok = (newSize <= size());
        if (ok)
            last = first+newSize;
        return ok;
    }
    void push_back(const data_storage_t& value)
    {
        if (empty())
        {
            first = storage->size();
        }
        storage->push_back(value);
        last = storage->size();
    }
    const data_storage_t& at(size_t index) const {return storage->at(first+index);}
    data_storage_t& at(size_t index) {return storage->at(first+index);}
    const data_storage_t& operator[](size_t index) const {return at(index);}
    data_storage_t& operator[](size_t index) {return at(index);}
private:
    storage_t* storage = nullptr;
    size_t first = 0;
    size_t last = 0;
};

typedef ContourDataStorage<cv::Point, 1024, 0> ContourPointsStorage;
typedef ContourDataStorage<schar, 1024, 0> ContourCodesStorage;

class Contour
{
public:
    ContourPointsStorage pts;
    cv::Rect brect;
    cv::Point origin;
    ContourCodesStorage codes;
    bool isHole = false;
    bool isChain = false;

    explicit Contour(ContourPointsStorage::storage_t* pointStorage_,
                     ContourCodesStorage::storage_t* codesStorage_)
                    :pts(pointStorage_),codes(codesStorage_) {}
    Contour(const Contour&) = delete;
    Contour(Contour&& other) noexcept = default;
    Contour& operator=(const Contour&) = delete;
    Contour& operator=(Contour&& other) noexcept = default;
    ~Contour() = default;
    void updateBoundingRect() {}
    bool isEmpty() const
    {
        return pts.size() == 0 && codes.size() == 0;
    }
    size_t size() const
    {
        return isChain ? codes.size() : pts.size();
    }
    void addPoint(const Point& pt)
    {
        pts.push_back(pt);
    }
    void copyTo(void* data) const
    {
        // NOTE: Mat::copyTo doesn't work because it creates new Mat object
        //       instead of reusing existing vector data
        if (isChain)
        {
            /*memcpy(data, codes.data(), codes.size() * sizeof(typename decltype(codes)::value_type));*/
            schar* dst = reinterpret_cast<schar*>(data);
            for(auto rangeIterator = codes.getRangeIterator() ; !rangeIterator.done() ; ++rangeIterator)
            {
                const auto range = *rangeIterator;
                memcpy(dst, range.first, range.second*sizeof(schar));
                dst += range.second;
            }
        }
        else
        {
            /*for (size_t i = 0, count = pts.size() ; i < count ; ++i)
                ((Point*)data)[i] = pts.at(i);
                */
            cv::Point* dst = reinterpret_cast<cv::Point*>(data);
            for(auto rangeIterator = pts.getRangeIterator() ; !rangeIterator.done() ; ++rangeIterator)
            {
                const auto range = *rangeIterator;
                memcpy(dst, range.first, range.second*sizeof(cv::Point));
                dst += range.second;
            }
        }
    }
};

typedef TreeNode<Contour> CNode;
typedef Tree<Contour> CTree;
typedef TreeIterator<Contour> CIterator;


void contourTreeToResults(CTree& tree,
                          int res_type,
                          cv::OutputArrayOfArrays& _contours,
                          cv::OutputArray& _hierarchy);


void approximateChainTC89(const ContourCodesStorage& chain, const Point& origin, const int method,
                          ContourPointsStorage& output);

}  // namespace cv

#endif  // OPENCV_CONTOURS_COMMON_HPP
