// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_UTIL_OPTIONAL_HPP
#define OPENCV_GAPI_UTIL_OPTIONAL_HPP

#include <opencv2/gapi/util/variant.hpp>

// A poor man's `optional` implementation, incompletely modeled against C++17 spec.
namespace cv
{
namespace util
{
    class bad_optional_access: public std::exception
    {
    public:
        virtual const char *what() const noexcept override
        {
            return "Bad optional access";
        }
    };

    // TODO: nullopt_t

    // Interface ///////////////////////////////////////////////////////////////
    template<typename T> class optional
    {
    public:
        // Constructors
        // NB.: there were issues with Clang 3.8 when =default() was used
        // instead {}
        optional() {};
        optional(const optional&) = default;
        explicit optional(T&&) noexcept;
        explicit optional(const T&) noexcept;
        optional(optional&&) noexcept;
        // TODO: optional(nullopt_t) noexcept;
        // TODO: optional(const optional<U> &)
        // TODO: optional(optional<U> &&)
        // TODO: optional(Args&&...)
        // TODO: optional(initializer_list<U>)
        // TODO: optional(U&& value);

        // Assignment
        optional& operator=(const optional&) = default;
        optional& operator=(optional&&);

        // Observers
        T* operator-> ();
        const T* operator-> () const;
        T& operator* ();
        const T& operator* () const;
        // TODO: && versions

        operator bool() const noexcept;
        bool has_value() const noexcept;

        T& value();
        const T& value() const;
        // TODO: && versions

        template<class U>
        T value_or(U &&default_value) const;

        void swap(optional &other) noexcept;
        void reset() noexcept;
        // TODO: emplace

        // TODO: operator==, !=, <, <=, >, >=

    private:
        struct nothing {};
        util::variant<nothing, T> m_holder;
    };

    template<class T>
    optional<typename std::decay<T>::type> make_optional(T&& value);

    // TODO: Args... and initializer_list versions

    // Implementation //////////////////////////////////////////////////////////
    template<class T> optional<T>::optional(T &&v) noexcept
        : m_holder(std::move(v))
    {
    }

    template<class T> optional<T>::optional(const T &v) noexcept
        : m_holder(v)
    {
    }

    template<class T> optional<T>::optional(optional&& rhs) noexcept
        : m_holder(std::move(rhs.m_holder))
    {
        rhs.reset();
    }

    template<class T> optional<T>& optional<T>::operator=(optional&& rhs)
    {
        m_holder = std::move(rhs.m_holder);
        rhs.reset();
        return *this;
    }

    template<class T> T* optional<T>::operator-> ()
    {
        return & *(*this);
    }

    template<class T> const T* optional<T>::operator-> () const
    {
        return & *(*this);
    }

    template<class T> T& optional<T>::operator* ()
    {
        return this->value();
    }

    template<class T> const T& optional<T>::operator* () const
    {
        return this->value();
    }

    template<class T> optional<T>::operator bool() const noexcept
    {
        return this->has_value();
    }

    template<class T> bool optional<T>::has_value() const noexcept
    {
        return util::holds_alternative<T>(m_holder);
    }

    template<class T> T& optional<T>::value()
    {
        if (!this->has_value())
            throw_error(bad_optional_access());
        return util::get<T>(m_holder);
    }

    template<class T> const T& optional<T>::value() const
    {
        if (!this->has_value())
            throw_error(bad_optional_access());
        return util::get<T>(m_holder);
    }

    template<class T>
    template<class U> T optional<T>::value_or(U &&default_value) const
    {
        return (this->has_value() ? this->value() : T(default_value));
    }

    template<class T> void optional<T>::swap(optional<T> &other) noexcept
    {
        m_holder.swap(other.m_holder);
    }

    template<class T> void optional<T>::reset() noexcept
    {
        if (this->has_value())
            m_holder = nothing{};
    }

    template<class T>
    optional<typename std::decay<T>::type> make_optional(T&& value)
    {
        return optional<typename std::decay<T>::type>(std::forward<T>(value));
    }
} // namespace util
} // namespace cv

#endif // OPENCV_GAPI_UTIL_OPTIONAL_HPP
