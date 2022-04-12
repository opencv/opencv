// itlib-small-vector v1.03
//
// std::vector-like class with a static buffer for initial capacity
//
// SPDX-License-Identifier: MIT
// MIT License:
// Copyright(c) 2016-2018 Chobolabs Inc.
// Copyright(c) 2020-2021 Borislav Stanimirov
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files(the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and / or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions :
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//
//                  VERSION HISTORY
//
//  1.03 (2021-10-05) Use allocator member instead of inheriting from allocator
//                    Allow compare with small_vector of different static_size
//                    Don't rely on operator!= from T. Use operator== instead
//  1.02 (2021-09-15) Bugfix! Fixed bad deallocation when reverting to
//                    static size on resize()
//  1.01 (2021-08-05) Bugfix! Fixed return value of erase
//  1.00 (2020-10-14) Rebranded release from chobo-small-vector
//
//
//                  DOCUMENTATION
//
// Simply include this file wherever you need.
// It defines the class itlib::small_vector, which is a drop-in replacement of
// std::vector, but with an initial capacity as a template argument.
// It gives you the benefits of using std::vector, at the cost of having a statically
// allocated buffer for the initial capacity, which gives you cache-local data
// when the vector is small (smaller than the initial capacity).
//
// When the size exceeds the capacity, the vector allocates memory via the provided
// allocator, falling back to classic std::vector behavior.
//
// The second size_t template argument, RevertToStaticSize, is used when a
// small_vector which has already switched to dynamically allocated size reduces
// its size to a number smaller than that. In this case the vector's buffer
// switches back to the staticallly allocated one
//
// A default value for the initial static capacity is provided so a replacement
// in an existing code is possible with minimal changes to it.
//
// Example:
//
// itlib::small_vector<int, 4, 5> myvec; // a small_vector of size 0, initial capacity 4, and revert size 4 (smaller than 5)
// myvec.resize(2); // vector is {0,0} in static buffer
// myvec[1] = 11; // vector is {0,11} in static buffer
// myvec.push_back(7); // vector is {0,11,7}  in static buffer
// myvec.insert(myvec.begin() + 1, 3); // vector is {0,3,11,7} in static buffer
// myvec.push_back(5); // vector is {0,3,11,7,5} in dynamically allocated memory buffer
// myvec.erase(myvec.begin());  // vector is {3,11,7,5} back in static buffer
// myvec.resize(5); // vector is {3,11,7,5,0} back in dynamically allocated memory
//
//
// Reference:
//
// itlib::small_vector is fully compatible with std::vector with
// the following exceptions:
// * when reducing the size with erase or resize the new size may fall below
//   RevertToStaticSize (if it is not 0). In such a case the vector will
//   revert to using its static buffer, invalidating all iterators (contrary
//   to the standard)
// * a method is added `revert_to_static()` which reverts to the static buffer
//   if possible, but doesn't free the dynamically allocated one
//
// Other notes:
//
// * the default value for RevertToStaticSize is zero. This means that once a dynamic
//   buffer is allocated the data will never be put into the static one, even if the
//   size allows it. Even if clear() is called. The only way to do so is to call
//   shrink_to_fit() or revert_to_static()
// * shrink_to_fit will free and reallocate if size != capacity and the data
//   doesn't fit into the static buffer. It also will revert to the static buffer
//   whenever possible regardless of the RevertToStaticSize value
//
//
//                  Configuration
//
// The library has two configuration options. They can be set as #define-s
// before including the header file, but it is recommended to change the code
// of the library itself with the values you want, especially if you include
// the library in many compilation units (as opposed to, say, a precompiled
// header or a central header).
//
//                  Config out of range error handling
//
// An out of range error is a runtime error which is triggered when a method is
// called with an iterator that doesn't belong to the vector's current range.
// For example: vec.erase(vec.end() + 1);
//
// This is set by defining ITLIB_SMALL_VECTOR_ERROR_HANDLING to one of the
// following values:
// * ITLIB_SMALL_VECTOR_ERROR_HANDLING_NONE - no error handling. Crashes WILL
//      ensue if the error is triggered.
// * ITLIB_SMALL_VECTOR_ERROR_HANDLING_THROW - std::out_of_range is thrown.
// * ITLIB_SMALL_VECTOR_ERROR_HANDLING_ASSERT - asserions are triggered.
// * ITLIB_SMALL_VECTOR_ERROR_HANDLING_ASSERT_AND_THROW - combines assert and
//      throw to catch errors more easily in debug mode
//
// To set this setting by editing the file change the line:
// ```
// #   define ITLIB_SMALL_VECTOR_ERROR_HANDLING ITLIB_SMALL_VECTOR_ERROR_HANDLING_THROW
// ```
// to the default setting of your choice
//
//                  Config bounds checks:
//
// By default bounds checks are made in debug mode (via an asser) when accessing
// elements (with `at` or `[]`). Iterators are not checked (yet...)
//
// To disable them, you can define ITLIB_SMALL_VECTOR_NO_DEBUG_BOUNDS_CHECK
// before including the header.
//
//
//                  TESTS
//
// You can find unit tests for small_vector in its official repo:
// https://github.com/iboB/itlib/blob/master/test/
//
#pragma once

#include <type_traits>
#include <cstddef>
#include <memory>

#define ITLIB_SMALL_VECTOR_ERROR_HANDLING_NONE  0
#define ITLIB_SMALL_VECTOR_ERROR_HANDLING_THROW 1
#define ITLIB_SMALL_VECTOR_ERROR_HANDLING_ASSERT 2
#define ITLIB_SMALL_VECTOR_ERROR_HANDLING_ASSERT_AND_THROW 3

#if !defined(ITLIB_SMALL_VECTOR_ERROR_HANDLING)
#   define ITLIB_SMALL_VECTOR_ERROR_HANDLING ITLIB_SMALL_VECTOR_ERROR_HANDLING_THROW
#endif


#if ITLIB_SMALL_VECTOR_ERROR_HANDLING == ITLIB_SMALL_VECTOR_ERROR_HANDLING_NONE
#   define I_ITLIB_SMALL_VECTOR_OUT_OF_RANGE_IF(cond)
#elif ITLIB_SMALL_VECTOR_ERROR_HANDLING == ITLIB_SMALL_VECTOR_ERROR_HANDLING_THROW
#   include <stdexcept>
#   define I_ITLIB_SMALL_VECTOR_OUT_OF_RANGE_IF(cond) if (cond) throw std::out_of_range("itlib::small_vector out of range")
#elif ITLIB_SMALL_VECTOR_ERROR_HANDLING == ITLIB_SMALL_VECTOR_ERROR_HANDLING_ASSERT
#   include <cassert>
#   define I_ITLIB_SMALL_VECTOR_OUT_OF_RANGE_IF(cond, rescue_return) assert(!(cond) && "itlib::small_vector out of range")
#elif ITLIB_SMALL_VECTOR_ERROR_HANDLING == ITLIB_SMALL_VECTOR_ERROR_HANDLING_ASSERT_AND_THROW
#   include <stdexcept>
#   include <cassert>
#   define I_ITLIB_SMALL_VECTOR_OUT_OF_RANGE_IF(cond, rescue_return) \
    do { if (cond) { assert(false && "itlib::small_vector out of range"); throw std::out_of_range("itlib::small_vector out of range"); } } while(false)
#else
#error "Unknown ITLIB_SMALL_VECTOR_ERRROR_HANDLING"
#endif


#if defined(ITLIB_SMALL_VECTOR_NO_DEBUG_BOUNDS_CHECK)
#   define I_ITLIB_SMALL_VECTOR_BOUNDS_CHECK(i)
#else
#   include <cassert>
#   define I_ITLIB_SMALL_VECTOR_BOUNDS_CHECK(i) assert((i) < this->size())
#endif

namespace itlib
{

template<typename T, size_t StaticCapacity = 16, size_t RevertToStaticSize = 0, class Alloc = std::allocator<T>>
struct small_vector
{
    static_assert(RevertToStaticSize <= StaticCapacity + 1, "itlib::small_vector: the revert-to-static size shouldn't exceed the static capacity by more than one");

    using atraits = std::allocator_traits<Alloc>;
public:
    using allocator_type = Alloc;
    using value_type = typename atraits::value_type;
    using size_type = typename atraits::size_type;
    using difference_type = typename atraits::difference_type;
    using reference = T&;
    using const_reference = const T&;
    using pointer = typename atraits::pointer;
    using const_pointer = typename atraits::const_pointer;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    static constexpr size_t static_capacity = StaticCapacity;
    static constexpr intptr_t revert_to_static_size = RevertToStaticSize;

    small_vector()
        : small_vector(Alloc())
    {}

    small_vector(const Alloc& alloc)
        : m_alloc(alloc)
        , m_capacity(StaticCapacity)
        , m_dynamic_capacity(0)
        , m_dynamic_data(nullptr)
    {
        m_begin = m_end = static_begin_ptr();
    }

    explicit small_vector(size_t count, const Alloc& alloc = Alloc())
        : small_vector(alloc)
    {
        resize(count);
    }

    explicit small_vector(size_t count, const T& value, const Alloc& alloc = Alloc())
        : small_vector(alloc)
    {
        assign_impl(count, value);
    }

    template <class InputIterator, typename = decltype(*std::declval<InputIterator>())>
    small_vector(InputIterator first, InputIterator last, const Alloc& alloc = Alloc())
        : small_vector(alloc)
    {
        assign_impl(first, last);
    }

    small_vector(std::initializer_list<T> l, const Alloc& alloc = Alloc())
        : small_vector(alloc)
    {
        assign_impl(l);
    }

    small_vector(const small_vector& v)
        : small_vector(v, atraits::select_on_container_copy_construction(v.get_allocator()))
    {}

    small_vector(const small_vector& v, const Alloc& alloc)
        : m_alloc(alloc)
        , m_dynamic_capacity(0)
        , m_dynamic_data(nullptr)
    {
        if (v.size() > StaticCapacity)
        {
            m_dynamic_capacity = v.size();
            m_begin = m_end = m_dynamic_data = atraits::allocate(get_alloc(), m_dynamic_capacity);
            m_capacity = v.size();
        }
        else
        {
            m_begin = m_end = static_begin_ptr();
            m_capacity = StaticCapacity;
        }

        for (auto p = v.m_begin; p != v.m_end; ++p)
        {
            atraits::construct(get_alloc(), m_end, *p);
            ++m_end;
        }
    }

    small_vector(small_vector&& v)
        : m_alloc(std::move(v.get_alloc()))
        , m_capacity(v.m_capacity)
        , m_dynamic_capacity(v.m_dynamic_capacity)
        , m_dynamic_data(v.m_dynamic_data)
    {
        if (v.m_begin == v.static_begin_ptr())
        {
            m_begin = m_end = static_begin_ptr();
            for (auto p = v.m_begin; p != v.m_end; ++p)
            {
                atraits::construct(get_alloc(), m_end, std::move(*p));
                ++m_end;
            }

            v.clear();
        }
        else
        {
            m_begin = v.m_begin;
            m_end = v.m_end;
        }

        v.m_dynamic_capacity = 0;
        v.m_dynamic_data = nullptr;
        v.m_begin = v.m_end = v.static_begin_ptr();
        v.m_capacity = StaticCapacity;
    }

    ~small_vector()
    {
        clear();

        if (m_dynamic_data)
        {
            atraits::deallocate(get_alloc(), m_dynamic_data, m_dynamic_capacity);
        }
    }

    small_vector& operator=(const small_vector& v)
    {
        if (this == &v)
        {
            // prevent self usurp
            return *this;
        }

        clear();

        m_begin = m_end = choose_data(v.size());

        for (auto p = v.m_begin; p != v.m_end; ++p)
        {
            atraits::construct(get_alloc(), m_end, *p);
            ++m_end;
        }

        update_capacity();

        return *this;
    }

    small_vector& operator=(small_vector&& v)
    {
        clear();

        get_alloc() = std::move(v.get_alloc());
        m_capacity = v.m_capacity;
        m_dynamic_capacity = v.m_dynamic_capacity;
        m_dynamic_data = v.m_dynamic_data;

        if (v.m_begin == v.static_begin_ptr())
        {
            m_begin = m_end = static_begin_ptr();
            for (auto p = v.m_begin; p != v.m_end; ++p)
            {
                atraits::construct(get_alloc(), m_end, std::move(*p));
                ++m_end;
            }

            v.clear();
        }
        else
        {
            m_begin = v.m_begin;
            m_end = v.m_end;
        }

        v.m_dynamic_capacity = 0;
        v.m_dynamic_data = nullptr;
        v.m_begin = v.m_end = v.static_begin_ptr();
        v.m_capacity = StaticCapacity;

        return *this;
    }

    void assign(size_type count, const T& value)
    {
        clear();
        assign_impl(count, value);
    }

    template <class InputIterator, typename = decltype(*std::declval<InputIterator>())>
    void assign(InputIterator first, InputIterator last)
    {
        clear();
        assign_impl(first, last);
    }

    void assign(std::initializer_list<T> ilist)
    {
        clear();
        assign_impl(ilist);
    }

    allocator_type get_allocator() const
    {
        return get_alloc();
    }

    const_reference at(size_type i) const
    {
        I_ITLIB_SMALL_VECTOR_BOUNDS_CHECK(i);
        return *(m_begin + i);
    }

    reference at(size_type i)
    {
        I_ITLIB_SMALL_VECTOR_BOUNDS_CHECK(i);
        return *(m_begin + i);
    }

    const_reference operator[](size_type i) const
    {
        return at(i);
    }

    reference operator[](size_type i)
    {
        return at(i);
    }

    const_reference front() const
    {
        return at(0);
    }

    reference front()
    {
        return at(0);
    }

    const_reference back() const
    {
        return *(m_end - 1);
    }

    reference back()
    {
        return *(m_end - 1);
    }

    const_pointer data() const noexcept
    {
        return m_begin;
    }

    pointer data() noexcept
    {
        return m_begin;
    }

    // iterators
    iterator begin() noexcept
    {
        return m_begin;
    }

    const_iterator begin() const noexcept
    {
        return m_begin;
    }

    const_iterator cbegin() const noexcept
    {
        return m_begin;
    }

    iterator end() noexcept
    {
        return m_end;
    }

    const_iterator end() const noexcept
    {
        return m_end;
    }

    const_iterator cend() const noexcept
    {
        return m_end;
    }

    reverse_iterator rbegin() noexcept
    {
        return reverse_iterator(end());
    }

    const_reverse_iterator rbegin() const noexcept
    {
        return const_reverse_iterator(end());
    }

    const_reverse_iterator crbegin() const noexcept
    {
        return const_reverse_iterator(end());
    }

    reverse_iterator rend() noexcept
    {
        return reverse_iterator(begin());
    }

    const_reverse_iterator rend() const noexcept
    {
        return const_reverse_iterator(begin());
    }

    const_reverse_iterator crend() const noexcept
    {
        return const_reverse_iterator(begin());
    }

    // capacity
    bool empty() const noexcept
    {
        return m_begin == m_end;
    }

    size_t size() const noexcept
    {
        return m_end - m_begin;
    }

    size_t max_size() const noexcept
    {
        return atraits::max_size();
    }

    void reserve(size_type new_cap)
    {
        if (new_cap <= m_capacity) return;

        auto new_buf = choose_data(new_cap);

        assert(new_buf != m_begin); // should've been handled by new_cap <= m_capacity
        assert(new_buf != static_begin_ptr()); // we should never reserve into static memory

        const auto s = size();
        if(s < RevertToStaticSize)
        {
            // we've allocated enough memory for the dynamic buffer but don't move there until we have to
            return;
        }

        // now we need to transfer the existing elements into the new buffer
        for (size_type i = 0; i < s; ++i)
        {
            atraits::construct(get_alloc(), new_buf + i, std::move(*(m_begin + i)));
        }

        // free old elements
        for (size_type i = 0; i < s; ++i)
        {
            atraits::destroy(get_alloc(), m_begin + i);
        }

        if (m_begin != static_begin_ptr())
        {
            // we've moved from dyn to dyn memory, so deallocate the old one
            atraits::deallocate(get_alloc(), m_begin, m_capacity);
        }

        m_begin = new_buf;
        m_end = new_buf + s;
        m_capacity = m_dynamic_capacity;
    }

    size_t capacity() const noexcept
    {
        return m_capacity;
    }

    void shrink_to_fit()
    {
        const auto s = size();

        if (s == m_capacity) return;
        if (m_begin == static_begin_ptr()) return;

        auto old_end = m_end;

        if (s < StaticCapacity)
        {
            // revert to static capacity
            m_begin = m_end = static_begin_ptr();
            m_capacity = StaticCapacity;
        }
        else
        {
            // alloc new smaller buffer
            m_begin = m_end = atraits::allocate(get_alloc(), s);
            m_capacity = s;
        }

        for (auto p = m_dynamic_data; p != old_end; ++p)
        {
            atraits::construct(get_alloc(), m_end, std::move(*p));
            ++m_end;
            atraits::destroy(get_alloc(), p);
        }

        atraits::deallocate(get_alloc(), m_dynamic_data, m_dynamic_capacity);
        m_dynamic_data = nullptr;
        m_dynamic_capacity = 0;
    }

    void revert_to_static()
    {
        const auto s = size();
        if (m_begin == static_begin_ptr()) return; //we're already there
        if (s > StaticCapacity) return; // nothing we can do

        // revert to static capacity
        auto old_end = m_end;
        m_begin = m_end = static_begin_ptr();
        m_capacity = StaticCapacity;
        for (auto p = m_dynamic_data; p != old_end; ++p)
        {
            atraits::construct(get_alloc(), m_end, std::move(*p));
            ++m_end;
            atraits::destroy(get_alloc(), p);
        }
    }

    // modifiers
    void clear() noexcept
    {
        for (auto p = m_begin; p != m_end; ++p)
        {
            atraits::destroy(get_alloc(), p);
        }

        if (RevertToStaticSize > 0)
        {
            m_begin = m_end = static_begin_ptr();
            m_capacity = StaticCapacity;
        }
        else
        {
            m_end = m_begin;
        }
    }

    iterator insert(const_iterator position, const value_type& val)
    {
        auto pos = grow_at(position, 1);
        atraits::construct(get_alloc(), pos, val);
        return pos;
    }

    iterator insert(const_iterator position, value_type&& val)
    {
        auto pos = grow_at(position, 1);
        atraits::construct(get_alloc(), pos, std::move(val));
        return pos;
    }

    iterator insert(const_iterator position, size_type count, const value_type& val)
    {
        auto pos = grow_at(position, count);
        for (size_type i = 0; i < count; ++i)
        {
            atraits::construct(get_alloc(), pos + i, val);
        }
        return pos;
    }

    template <typename InputIterator, typename = decltype(*std::declval<InputIterator>())>
    iterator insert(const_iterator position, InputIterator first, InputIterator last)
    {
        auto pos = grow_at(position, last - first);
        size_type i = 0;
        auto np = pos;
        for (auto p = first; p != last; ++p, ++np)
        {
            atraits::construct(get_alloc(), np, *p);
        }
        return pos;
    }

    iterator insert(const_iterator position, std::initializer_list<T> ilist)
    {
        auto pos = grow_at(position, ilist.size());
        size_type i = 0;
        for (auto& elem : ilist)
        {
            atraits::construct(get_alloc(), pos + i, elem);
            ++i;
        }
        return pos;
    }

    template<typename... Args>
    iterator emplace(const_iterator position, Args&&... args)
    {
        auto pos = grow_at(position, 1);
        atraits::construct(get_alloc(), pos, std::forward<Args>(args)...);
        return pos;
    }

    iterator erase(const_iterator position)
    {
        return shrink_at(position, 1);
    }

    iterator erase(const_iterator first, const_iterator last)
    {
        I_ITLIB_SMALL_VECTOR_OUT_OF_RANGE_IF(first > last);
        return shrink_at(first, last - first);
    }

    void push_back(const_reference val)
    {
        auto pos = grow_at(m_end, 1);
        atraits::construct(get_alloc(), pos, val);
    }

    void push_back(T&& val)
    {
        auto pos = grow_at(m_end, 1);
        atraits::construct(get_alloc(), pos, std::move(val));
    }

    template<typename... Args>
    reference emplace_back(Args&&... args)
    {
        auto pos = grow_at(m_end, 1);
        atraits::construct(get_alloc(), pos, std::forward<Args>(args)...);
        return *pos;
    }

    void pop_back()
    {
        shrink_at(m_end - 1, 1);
    }

    void resize(size_type n, const value_type& v)
    {
        auto new_buf = choose_data(n);

        if (new_buf == m_begin)
        {
            // no special transfers needed

            auto new_end = m_begin + n;

            while (m_end > new_end)
            {
                atraits::destroy(get_alloc(), --m_end);
            }

            while (new_end > m_end)
            {
                atraits::construct(get_alloc(), m_end++, v);
            }
        }
        else
        {
            // we need to transfer the elements into the new buffer

            const auto s = size();
            const auto num_transfer = n < s ? n : s;

            for (size_type i = 0; i < num_transfer; ++i)
            {
                atraits::construct(get_alloc(), new_buf + i, std::move(*(m_begin + i)));
            }

            // free obsoletes
            for (size_type i = 0; i < s; ++i)
            {
                atraits::destroy(get_alloc(), m_begin + i);
            }

            // construct new elements
            for (size_type i = num_transfer; i < n; ++i)
            {
                atraits::construct(get_alloc(), new_buf + i, v);
            }

            if (new_buf == static_begin_ptr())
            {
                m_capacity = StaticCapacity;
            }
            else
            {
                if (m_begin != static_begin_ptr())
                {
                    // we've moved from dyn to dyn memory, so deallocate the old one
                    atraits::deallocate(get_alloc(), m_begin, m_capacity);
                }
                m_capacity = m_dynamic_capacity;
            }

            m_begin = new_buf;
            m_end = new_buf + n;
        }
    }

    void resize(size_type n)
    {
        auto new_buf = choose_data(n);

        if (new_buf == m_begin)
        {
            // no special transfers needed

            auto new_end = m_begin + n;

            while (m_end > new_end)
            {
                atraits::destroy(get_alloc(), --m_end);
            }

            while (new_end > m_end)
            {
                atraits::construct(get_alloc(), m_end++);
            }
        }
        else
        {
            // we need to transfer the elements into the new buffer

            const auto s = size();
            const auto num_transfer = n < s ? n : s;

            for (size_type i = 0; i < num_transfer; ++i)
            {
                atraits::construct(get_alloc(), new_buf + i, std::move(*(m_begin + i)));
            }

            // free obsoletes
            for (size_type i = 0; i < s; ++i)
            {
                atraits::destroy(get_alloc(), m_begin + i);
            }

            // construct new elements
            for (size_type i = num_transfer; i < n; ++i)
            {
                atraits::construct(get_alloc(), new_buf + i);
            }

            if (new_buf == static_begin_ptr())
            {
                m_capacity = StaticCapacity;
            }
            else
            {
                if (m_begin != static_begin_ptr())
                {
                    // we've moved from dyn to dyn memory, so deallocate the old one
                    atraits::deallocate(get_alloc(), m_begin, m_capacity);
                }
                m_capacity = m_dynamic_capacity;
            }

            m_begin = new_buf;
            m_end = new_buf + n;
        }
    }

private:
    T* static_begin_ptr()
    {
        return reinterpret_cast<pointer>(m_static_data + 0);
    }

    // increase the size by splicing the elements in such a way that
    // a hole of uninitialized elements is left at position, with size num
    // returns the (potentially new) address of the hole
    T* grow_at(const T* cp, size_t num)
    {
        auto position = const_cast<T*>(cp);

        I_ITLIB_SMALL_VECTOR_OUT_OF_RANGE_IF(position < m_begin || position > m_end);

        const auto s = size();
        auto new_buf = choose_data(s + num);

        if (new_buf == m_begin)
        {
            // no special transfers needed

            m_end = m_begin + s + num;

            for (auto p = m_end - num - 1; p >= position; --p)
            {
                atraits::construct(get_alloc(), p + num, std::move(*p));
                atraits::destroy(get_alloc(), p);
            }

            return position;
        }
        else
        {
            // we need to transfer the elements into the new buffer

            position = new_buf + (position - m_begin);

            auto p = m_begin;
            auto np = new_buf;

            for (; np != position; ++p, ++np)
            {
                atraits::construct(get_alloc(), np, std::move(*p));
            }

            np += num;
            for (; p != m_end; ++p, ++np)
            {
                atraits::construct(get_alloc(), np, std::move(*p));
            }

            // destroy old
            for (p = m_begin; p != m_end; ++p)
            {
                atraits::destroy(get_alloc(), p);
            }

            if (m_begin != static_begin_ptr())
            {
                // we've moved from dyn to dyn memory, so deallocate the old one
                atraits::deallocate(get_alloc(), m_begin, m_capacity);
            }

            m_capacity = m_dynamic_capacity;

            m_begin = new_buf;
            m_end = new_buf + s + num;

            return position;
        }
    }

    T* shrink_at(const T* cp, size_t num)
    {
        auto position = const_cast<T*>(cp);

        I_ITLIB_SMALL_VECTOR_OUT_OF_RANGE_IF(position < m_begin || position > m_end || position + num > m_end);

        const auto s = size();
        if (s - num == 0)
        {
            clear();
            return m_end;
        }

        auto new_buf = choose_data(s - num);

        if (new_buf == m_begin)
        {
            // no special transfers needed

            for (auto p = position, np = position + num; np != m_end; ++p, ++np)
            {
                atraits::destroy(get_alloc(), p);
                atraits::construct(get_alloc(), p, std::move(*np));
            }

            for (auto p = m_end - num; p != m_end; ++p)
            {
                atraits::destroy(get_alloc(), p);
            }

            m_end -= num;
        }
        else
        {
            // we need to transfer the elements into the new buffer

            assert(new_buf == static_begin_ptr()); // since we're shrinking that's the only way to have a new buffer

            m_capacity = StaticCapacity;

            auto p = m_begin, np = new_buf;
            for (; p != position; ++p, ++np)
            {
                atraits::construct(get_alloc(), np, std::move(*p));
                atraits::destroy(get_alloc(), p);
            }

            for (; p != position + num; ++p)
            {
                atraits::destroy(get_alloc(), p);
            }

            for (; np != new_buf + s - num; ++p, ++np)
            {
                atraits::construct(get_alloc(), np, std::move(*p));
                atraits::destroy(get_alloc(), p);
            }

            position = new_buf + (position - m_begin);
            m_begin = new_buf;
            m_end = np;
        }

        return position;
    }

    void assign_impl(size_type count, const T& value)
    {
        assert(m_begin);
        assert(m_begin == m_end);

        m_begin = m_end = choose_data(count);
        for (size_type i = 0; i < count; ++i)
        {
            atraits::construct(get_alloc(), m_end, value);
            ++m_end;
        }

        update_capacity();
    }

    template <class InputIterator>
    void assign_impl(InputIterator first, InputIterator last)
    {
        assert(m_begin);
        assert(m_begin == m_end);

        m_begin = m_end = choose_data(last - first);
        for (auto p = first; p != last; ++p)
        {
            atraits::construct(get_alloc(), m_end, *p);
            ++m_end;
        }

        update_capacity();
    }

    void assign_impl(std::initializer_list<T> ilist)
    {
        assert(m_begin);
        assert(m_begin == m_end);

        m_begin = m_end = choose_data(ilist.size());
        for (auto& elem : ilist)
        {
            atraits::construct(get_alloc(), m_end, elem);
            ++m_end;
        }

        update_capacity();
    }

    void update_capacity()
    {
        if (m_begin == static_begin_ptr())
        {
            m_capacity = StaticCapacity;
        }
        else
        {
            m_capacity = m_dynamic_capacity;
        }
    }

    T* choose_data(size_t desired_capacity)
    {
        if (m_begin == m_dynamic_data)
        {
            // we're at the dyn buffer, so see if it needs resize or revert to static

            if (desired_capacity > m_dynamic_capacity)
            {
                while (m_dynamic_capacity < desired_capacity)
                {
                    // grow by roughly 1.5
                    m_dynamic_capacity *= 3;
                    ++m_dynamic_capacity;
                    m_dynamic_capacity /= 2;
                }

                m_dynamic_data = atraits::allocate(get_alloc(), m_dynamic_capacity);
                return m_dynamic_data;
            }
            else if (desired_capacity < RevertToStaticSize)
            {
                // we're reverting to the static buffer
                return static_begin_ptr();
            }
            else
            {
                // if the capacity and we don't revert to static, just do nothing
                return m_dynamic_data;
            }
        }
        else
        {
            assert(m_begin == static_begin_ptr()); // corrupt begin ptr?

            if (desired_capacity > StaticCapacity)
            {
                // we must move to dyn memory

                // see if we have enough
                if (desired_capacity > m_dynamic_capacity)
                {
                    // we need to allocate more
                    // we don't have anything to destroy, so we can also deallocate the buffer
                    if (m_dynamic_data)
                    {
                        atraits::deallocate(get_alloc(), m_dynamic_data, m_dynamic_capacity);
                    }

                    m_dynamic_capacity = desired_capacity;
                    m_dynamic_data = atraits::allocate(get_alloc(), m_dynamic_capacity);
                }

                return m_dynamic_data;
            }
            else
            {
                // we have enough capacity as it is
                return static_begin_ptr();
            }
        }
    }

    allocator_type& get_alloc() { return m_alloc; }
    const allocator_type& get_alloc() const { return m_alloc; }

    allocator_type m_alloc;

    pointer m_begin;
    pointer m_end;

    size_t m_capacity;
    typename std::aligned_storage<sizeof(T), std::alignment_of<T>::value>::type m_static_data[StaticCapacity];

    size_t m_dynamic_capacity;
    pointer m_dynamic_data;
};

template<typename T,
    size_t StaticCapacityA, size_t RevertToStaticSizeA, class AllocA,
    size_t StaticCapacityB, size_t RevertToStaticSizeB, class AllocB
>
bool operator==(const small_vector<T, StaticCapacityA, RevertToStaticSizeA, AllocA>& a,
    const small_vector<T, StaticCapacityB, RevertToStaticSizeB, AllocB>& b)
{
    if (a.size() != b.size())
    {
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i)
    {
        if (!(a[i] == b[i]))
            return false;
    }

    return true;
}

template<typename T,
    size_t StaticCapacityA, size_t RevertToStaticSizeA, class AllocA,
    size_t StaticCapacityB, size_t RevertToStaticSizeB, class AllocB
>
bool operator!=(const small_vector<T, StaticCapacityA, RevertToStaticSizeA, AllocA>& a,
    const small_vector<T, StaticCapacityB, RevertToStaticSizeB, AllocB>& b)

{
    return !operator==(a, b);
}

}
