// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_CONTOURS_COMMON_HPP
#define OPENCV_CONTOURS_COMMON_HPP

#include "precomp.hpp"
#include <stack>

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
    TreeNode(int self) :
        self_(self), parent(-1), first_child(-1), prev(-1), next(-1), ctable_next(-1)
    {
        CV_Assert(self >= 0);
    }
    int self() const
    {
        return self_;
    }
};

template <typename T>
class Tree
{
private:
    std::vector<TreeNode<T>> nodes;

public:
    TreeNode<T>& newElem()
    {
        const size_t idx = nodes.size();
        CV_DbgAssert(idx < (size_t)std::numeric_limits<int>::max());
        nodes.push_back(TreeNode<T>((int)idx));
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
            nodes[prev_item.next].prev = idx;
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
    std::stack<int> levels;
    Tree<T>& tree;
};

//==============================================================================

class Contour
{
public:
    cv::Rect brect;
    cv::Point origin;
    std::vector<cv::Point> pts;
    std::vector<schar> codes;
    bool isHole;
    bool isChain;

    Contour() : isHole(false), isChain(false) {}
    void updateBoundingRect() {}
    bool isEmpty() const
    {
        return pts.size() == 0 && codes.size() == 0;
    }
    size_t size() const
    {
        return isChain ? codes.size() : pts.size();
    }
    void copyTo(void* data) const
    {
        // NOTE: Mat::copyTo doesn't work because it creates new Mat object
        //       instead of reusing existing vector data
        if (isChain)
        {
            memcpy(data, &codes[0], codes.size() * sizeof(codes[0]));
        }
        else
        {
            memcpy(data, &pts[0], pts.size() * sizeof(pts[0]));
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


std::vector<Point>
    approximateChainTC89(std::vector<schar> chain, const Point& origin, const int method);

}  // namespace cv

#endif  // OPENCV_CONTOURS_COMMON_HPP
