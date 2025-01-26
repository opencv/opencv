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


template <typename T, size_t CAPACITY = 4096>
class ArenaStackBuffer
{
    public:
        class Item
        {
            friend class ArenaStackBuffer;
            public:
                Item(ArenaStackBuffer* owner = nullptr, T* baseAddress = nullptr)
                    :_owner(owner),_baseAddress(baseAddress) {}
                Item(const Item&) = delete;
                Item(Item&& other) noexcept
                    :_owner(nullptr),_baseAddress(nullptr) {*this = std::move(other);}
                ~Item() {if (_owner != nullptr) _owner->releaseItem(*this);}
                Item& operator=(const Item&) = delete;
                Item& operator=(Item&& other) noexcept {
                    if (&other != this) {
                        std::swap(this->_owner, other._owner);
                        std::swap(this->_baseAddress, other._baseAddress);
                    }
                    return *this;
                }
            public:
                operator T& () noexcept {return get();}
                operator const T& () const noexcept {return get();}
                T& get() {return *_baseAddress;}
                const T& get() const {return *_baseAddress;}
            private:
                ArenaStackBuffer* _owner;
                T* _baseAddress;
        };
    public:
          ArenaStackBuffer(void):bufferCapacityInElements(CAPACITY/sizeof(T)), nextAllocOffsetInElements(0) {}
          ArenaStackBuffer(const ArenaStackBuffer&) = delete;
          ArenaStackBuffer(ArenaStackBuffer&&) noexcept = delete;
          ~ArenaStackBuffer() = default;
          ArenaStackBuffer& operator=(const ArenaStackBuffer&) = delete;
          ArenaStackBuffer& operator=(ArenaStackBuffer&&) noexcept = delete;
    public:
        template<typename... Ts>
        Item newItem(Ts&&... params) {
            if (nextAllocOffsetInElements < bufferCapacityInElements)
                return Item(this, new(_buffer+sizeof(T)*(nextAllocOffsetInElements++)) T(std::forward<Ts>(params)...));
            else
                return Item(this, new T(std::forward<Ts>(params)...));
        }
        void releaseItem(Item& item) {
            if (!item._owner){
            }
            else if (item._baseAddress != nullptr)
            {
                T* beginBufferAddress = reinterpret_cast<T*>(_buffer);
                T* endBufferAddress = beginBufferAddress+nextAllocOffsetInElements;
                const bool isInBuffer = (beginBufferAddress <= item._baseAddress) && (item._baseAddress < endBufferAddress);
                if (!isInBuffer)
                    delete item._baseAddress;
                else
                {
                    item._baseAddress->~T();
                    if (item._baseAddress+1 == endBufferAddress)//last address can be reused
                        --nextAllocOffsetInElements;
                }
                item._owner = nullptr;
            }
        }
        void releaseItems(std::vector<Item>& items) {
          size_t count = items.size();
          while(count--)
            releaseItem(items[count]);
        }
        T& get(const Item& item) {return item.get();}
        const T& get(const Item& item) const {return item.get();}
    private:
        unsigned char _buffer[CAPACITY];
        size_t bufferCapacityInElements;
        size_t nextAllocOffsetInElements;
};

template <typename T, size_t CAPACITY = 4096>
class ArenaDynamicBuffer
{
    public:
        class Item
        {
            friend class ArenaDynamicBuffer;
            public:
                  Item(ArenaDynamicBuffer* owner = nullptr, size_t index = 0):_owner(owner),_index(index) {}
                  Item(const Item&) = delete;
                  Item(Item&& other) noexcept
                      :_owner(nullptr),_index(0) {*this = std::move(other);}
                  ~Item() {if (_owner != nullptr) _owner->releaseItem(*this);}
                  Item& operator=(const Item&) = delete;
                  Item& operator=(Item&& other) noexcept {
                      if (&other != this) {
                          std::swap(_owner, other._owner);
                          std::swap(_index, other._index);
                      }
                      return *this;
                  }
            public:
                  operator T& () noexcept {return get();}
                  operator const T& () const noexcept {return get();}
                  T& get() {return _owner->get(*this);}
                  const T& get() const {return _owner->get(*this);}
            private:
                ArenaDynamicBuffer* _owner;
                size_t _index;
        };
    public:
        ArenaDynamicBuffer(void) {_buffer.reserve(CAPACITY/sizeof(T));_freeIndices.reserve(_buffer.capacity());}
        ArenaDynamicBuffer(const ArenaDynamicBuffer&) = delete;
        ArenaDynamicBuffer(ArenaDynamicBuffer&&) noexcept = delete;
        ~ArenaDynamicBuffer() = default;
        ArenaDynamicBuffer& operator=(const ArenaDynamicBuffer&) = delete;
        ArenaDynamicBuffer& operator=(ArenaDynamicBuffer&&) noexcept = delete;
    public:
        template<typename... Ts>
        Item newItem(Ts&&... params) {
            if (!_freeIndices.empty())
            {
                const size_t index = _freeIndices.back();
                _freeIndices.pop_back();
                _buffer[index] = std::move(T(std::forward<Ts>(params)...));
                return Item(this, index);
            }
            else
            {
                const size_t index = _buffer.size();
                _buffer.emplace_back(std::move(T(std::forward<Ts>(params)...)));
                return Item(this, index);
            }
        }
        void releaseItem(Item& item) {
            if (item._owner != nullptr)
            {
                if (item._index == _buffer.size())
                    _buffer.pop_back();
                else
                    _freeIndices.push_back(item._index);
                item._owner = nullptr;
            }
        }
        void releaseItems(std::vector<Item>& items) {
            size_t count = items.size();
            while(count--)
                releaseItem(items[count]);
        }
        T& get(const Item& item) {return _buffer[item._index];}
        const T& get(const Item& item) const {return _buffer[item._index];}
    private:
        std::vector<T> _buffer;
        std::vector<size_t> _freeIndices;
};

template <typename T, size_t CAPACITY = 4096>
class ArenaDynamicBufferIndexed
{
    public:
        class Item
        {
            friend class ArenaDynamicBufferIndexed;
            public:
                typedef unsigned int index_storage_t;
            public:
                  explicit Item(index_storage_t index = 0):_index(index) {}
                  Item(const Item&) = delete;
                  Item(Item&& other) noexcept
                      :_index(0) {*this = std::move(other);}
                  ~Item() {}
                  Item& operator=(const Item&) = delete;
                  Item& operator=(Item&& other) noexcept {
                      if (&other != this) {
                          std::swap(_index, other._index);
                      }
                      return *this;
                  }
            public:
                index_storage_t get(void) const {return _index;}
            private:
                index_storage_t _index;
        };
    public:
        ArenaDynamicBufferIndexed(void) {_buffer.reserve(CAPACITY/sizeof(T));}
        ArenaDynamicBufferIndexed(const ArenaDynamicBufferIndexed&) = delete;
        ArenaDynamicBufferIndexed(ArenaDynamicBufferIndexed&&) noexcept = delete;
        ~ArenaDynamicBufferIndexed() = default;
        ArenaDynamicBufferIndexed& operator=(const ArenaDynamicBufferIndexed&) = delete;
        ArenaDynamicBufferIndexed& operator=(ArenaDynamicBufferIndexed&&) noexcept = delete;
    public:
        class Range {
            public:
                class const_iterator {
                    public:
                        typedef typename ArenaDynamicBufferIndexed::Item::index_storage_t index_storage_t;
                    public:
                        const_iterator(index_storage_t _value):value(_value) {}
                        const_iterator& operator++() {++value; return *this;}
                        const_iterator operator++(int) {return result(value++);}
                        const index_storage_t& operator*() const {return value;}
                        index_storage_t& operator*() {return value;}
                        bool operator==(const const_iterator& other) const {return this->value == other.value;}
                        bool operator!=(const const_iterator& other) const {return this->value != other.value;}
                        bool operator<(const const_iterator& other) const {return this->value < other.value;}
                    private:
                        index_storage_t value;
                };
                typedef const_iterator iterator;
            public:
                Range(void):_size(0) {}
            public:
                size_t size(void) const {return _size;}
                void resize(size_t value) {_size = value;}
                void reserve(size_t) {}
                void emplace_back(Item&& value) {
                    if (!_size)
                        first = std::move(value);
                    else
                        CV_Assert(value.get() == first.get()+_size);
                    ++_size;
                }
                const_iterator cbegin(void) const noexcept {return const_iterator(first.get());}
                const_iterator cend(void) const noexcept {return const_iterator(static_cast<typename Item::index_storage_t>(first.get()+_size));}
                iterator begin(void) const noexcept {return iterator(first.get());}
                iterator end(void) const noexcept {return iterator(static_cast<typename Item::index_storage_t>(first.get()+_size));}
                Item operator[](size_t index) const noexcept {return Item(static_cast<unsigned int>(first.get()+index));}
                Item operator[](size_t index) noexcept {return Item(static_cast<unsigned int>(first.get()+index));}
            public:
                Item first;
                size_t _size;
        };
    public:
        template<typename... Ts>
        Item newItem(Ts&&... params) {
            const size_t index = _buffer.size();
            _buffer.emplace_back(std::move(T(std::forward<Ts>(params)...)));
            return Item(static_cast<typename Item::index_storage_t>(index));
        }
        void releaseItem(Item item) {
            if (item._index == _buffer.size())
                _buffer.pop_back();
        }
        void releaseItems(Range& items) {
            if (items.first.get()+items._size == _buffer.size())
                _buffer.resize(_buffer.size()-items._size);
        }
        T& get(const Item& item) {return get(item._index);}
        const T& get(const Item& item) const {return get(item._index);}
        T& get(typename Item::index_storage_t index) {return _buffer[index];}
        const T& get(typename Item::index_storage_t index) const {return _buffer[index];}
    private:
        std::vector<T> _buffer;

};

template <typename T>
class ArenaDummy
{
    public:
        typedef T Item;
    public:
        ArenaDummy(void) {}
        ArenaDummy(const ArenaDummy&) = delete;
        ArenaDummy(ArenaDummy&&) noexcept = delete;
        ~ArenaDummy() = default;
        ArenaDummy& operator=(const ArenaDummy&) = delete;
        ArenaDummy& operator=(ArenaDummy&&) noexcept = delete;
    public:
        template<typename... Ts>
        Item newItem(Ts&&... params) {return T(std::forward<Ts>(params)...);}
        void releaseItem(Item&) {}
        void releaseItems(std::vector<Item>&) {}
        T& get(const Item& item) {return item;}
        const T& get(const Item& item) const {return item;}
};

//typedef ArenaDummy<Point> ContourArena;
//typedef ArenaStackBuffer<Point> ContourArena;
//typedef ArenaDynamicBuffer<Point> ContourArena;
typedef ArenaDynamicBufferIndexed<Point> ContourArena;
typedef ContourArena::Item ContourPoint;

//typedef std::vector<ContourPoint> ContourPointsStorage;
typedef ArenaDynamicBufferIndexed<Point>::Range ContourPointsStorage;

template<typename T, size_t BLOCK_LENGTH = 256>
class vectorWithArena
{
    public:
        typedef struct {T data[BLOCK_LENGTH];} block_t;
        typedef ArenaStackBuffer<block_t> arena_t;
    public:
        vectorWithArena(arena_t* arena):_arena(arena),_capacity(0),_size(0) {}
        vectorWithArena(const vectorWithArena&) = delete;
        vectorWithArena(vectorWithArena&& other) noexcept :_arena(nullptr),_capacity(0),_size(0) {*this = other;}
        ~vectorWithArena() {
            if (_arena != nullptr)
                _arena->releaseItems(_blocks);
        }
    public:
        vectorWithArena& operator=(const vectorWithArena&) = delete;
        vectorWithArena& operator=(vectorWithArena&& other) noexcept {
            if (&other != this) {
              std::swap(this->_arena, other._arena);
              std::swap(this->_blocks, other._blocks);
              std::swap(this->_capacity, other._capacity);
              std::swap(this->_size, other._size);
            }
        }
    public:
        bool empty(void) const {return !_size;}
        size_t capacity(void) const {return _capacity;}
        size_t size(void) const {return _size;}
        T& at(size_t index) {return _blocks[index/BLOCK_LENGTH].get().data[index%BLOCK_LENGTH];}
        const T& at(size_t index) const {return _blocks[index/BLOCK_LENGTH].get().data[index%BLOCK_LENGTH];}
        T& back(void) {return at(_size-1);}
        const T& back(void) const {return at(_size-1);}
    public:
        void push_back(const T& value) {
             if (_size == _capacity)
             {
                 _blocks.emplace_back(std::move(_arena->newItem()));
                 _capacity += BLOCK_LENGTH;
             }
             at(_size++) = value;
        }
        void emplace_back(T&& value) {
            if (_size == _capacity)
            {
                _blocks.emplace_back(std::move(_arena->newItem()));
                _capacity += BLOCK_LENGTH;
            }
            at(_size++) = std::move(value);
        }
        void pop_back(void) {
            if (!((--_size) % BLOCK_LENGTH))
            {
                _blocks.pop_back();
                _capacity -= BLOCK_LENGTH;
            }
        }
    private:
        arena_t* _arena;
        std::vector<typename arena_t::Item> _blocks;
        size_t _capacity;
        size_t _size;
};

template<typename T>
class vectorRanges
{
    public:
        vectorRanges(void) = default;
        vectorRanges(const vectorRanges&) = default;
        vectorRanges(vectorRanges&& other) noexcept = default;
        ~vectorRanges() = default;
    public:
        vectorRanges& operator=(const vectorRanges&) = default;
        vectorRanges& operator=(vectorRanges&& other) noexcept = default;
    public:
        bool empty(void) const {return !_size;}
        size_t size(void) const {return _size;}
        T at(size_t index) const {
            for(const auto& range : _ranges) {
                if (index < range.second)
                    return static_cast<T>(range.first+index);
                else
                    index -= range.second;
          }
          return _ranges[0].first;//should not occur
        }
        T back(void) const {return at(_size-1);}
    public:
        void push_back(const T& value) {
            if (_ranges.empty() || (value != back()+1))
                _ranges.push_back(std::make_pair(value, 1));
            else
                ++_ranges.back().second;
            ++_size;
        }
        void pop_back(void) {
            if (_ranges.back().second == 1)
                _ranges.pop_back();
            else
                --_ranges.back().second;
            --_size;
        }
    private:
        std::vector<std::pair<T, size_t> > _ranges;
        size_t _size = 0;
};

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
    TreeNode(int self, ContourArena* arena) :
        self_(self), parent(-1), first_child(-1), prev(-1), next(-1), ctable_next(-1), body(arena)
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
    Tree(ContourArena* contoursArena):_contoursArena(contoursArena) {}
    Tree(const Tree&) = delete;
    Tree(Tree&&) = delete;
    Tree& operator=(const Tree&) = delete;
    Tree& operator=(Tree&&) = delete;
    ~Tree() = default;
private:
    ContourArena* _contoursArena;
    ArenaStackBuffer<TreeNode<T> > _treeNodesArena;
public:
    typename vectorWithArena<int>::arena_t _treeIteratorArena;
private:
    typedef typename ArenaStackBuffer<TreeNode<T> >::Item node_t;
    std::vector<node_t> nodes;

public:
    TreeNode<T>& newElem()
    {
        const size_t idx = nodes.size();
        CV_DbgAssert(idx < (size_t)std::numeric_limits<int>::max());
        nodes.emplace_back(std::move(_treeNodesArena.newItem((int)idx, _contoursArena)));
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
    TreeIterator(Tree<T>& tree_) : tree(tree_)//,levels(&tree._treeIteratorArena)
    {
        CV_Assert(!tree.isEmpty());
        levels.push_back(0);
    }
    bool isDone() const
    {
        return levels.empty();
    }
    const TreeNode<T>& getNext_s()
    {
        int idx = levels.back();
        levels.pop_back();
        const TreeNode<T>& res = tree.elem(idx);
        int cur = tree.lastSibling(res.first_child);
        while (cur != -1)
        {
            levels.push_back(cur);
            cur = tree.elem(cur).prev;
        }
        return res;
    }

private:
    Tree<T>& tree;
    vectorRanges<int> levels;
};

//==============================================================================

class Contour
{
public:
    ContourArena* _arena;
    cv::Rect brect;
    cv::Point origin;
    ContourPointsStorage pts;
    std::vector<schar> codes;
    bool isHole;
    bool isChain;

    explicit Contour(ContourArena* arena) : _arena(arena), isHole(false), isChain(false) {
    }
    Contour(const Contour&) = delete;
    Contour(Contour&& other) noexcept : _arena(nullptr) {*this = std::move(other);}
    Contour& operator=(const Contour&) = delete;
    Contour& operator=(Contour&& other) noexcept {
        if (&other != this) {
            std::swap(this->_arena, other._arena);
            std::swap(this->brect, other.brect);
            std::swap(this->origin, other.origin);
            std::swap(this->pts, other.pts);
            std::swap(this->codes, other.codes);
            std::swap(this->isHole, other.isHole);
            std::swap(this->isChain, other.isChain);
        }
        return *this;
    }
    ~Contour() {
        if (_arena != nullptr)
            _arena->releaseItems(pts);
    }
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
          unsigned char* dst = reinterpret_cast<unsigned char*>(data);
          memcpy(dst, &_arena->get(*pts.begin()), pts.size()*sizeof(Point));
          /*for(auto& it : pts)
          {
            const Point& point = _arena->get(it);
            memcpy(dst, &point, sizeof(point));
            dst += sizeof(point);
          }*/
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


void approximateChainTC89(ContourArena& arena, std::vector<schar> chain, const Point& origin, const int method,
                          ContourPointsStorage& output);

}  // namespace cv

#endif  // OPENCV_CONTOURS_COMMON_HPP
