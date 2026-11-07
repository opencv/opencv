#pragma once

#include <cstddef>

#include "Eigen/Core"

namespace g2o
{
    // 16 byte aligned allocation functions
    template<typename Type>
    Type* allocate_aligned(size_t n)
    {
        return (Type*)Eigen::internal::aligned_malloc(n * sizeof(Type));
    }

    template<typename Type>
    Type* reallocate_aligned(Type* ptr, size_t newSize, size_t oldSize)
    {
        return (Type*)Eigen::internal::aligned_realloc(ptr, newSize * sizeof(Type), oldSize * sizeof(Type));
    }

    template<typename Type>
    void free_aligned(Type* block)
    {
        Eigen::internal::aligned_free(block);
    }

    template<typename Type>
    struct dynamic_aligned_buffer
    {
        dynamic_aligned_buffer(size_t size)
            : m_size{ 0 }, m_ptr{ nullptr }
        {
            allocate(size);
        }

        ~dynamic_aligned_buffer()
        {
            free();
        }

        Type* request(size_t n)
        {
            if (n <= m_size)
                return m_ptr;

            m_ptr = reallocate_aligned<Type>(m_ptr, n, m_size);
            m_size = m_ptr ? n : 0;

            return m_ptr;
        }

    private:
        void allocate(size_t size)
        {
            m_ptr = allocate_aligned<Type>(size);
            if (m_ptr != nullptr)
                m_size = size;
        }

        void free()
        {
            if (m_ptr != nullptr)
            {
                free_aligned<Type>(m_ptr);
                m_size = 0;
                m_ptr = nullptr;
            }
        }

        std::size_t m_size;
        Type* m_ptr;
    };

    template<typename T>
    struct aligned_deleter
    {
        void operator()(T* block)
        {
            free_aligned(block);
        }
    };
}