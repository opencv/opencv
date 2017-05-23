#ifndef CVVISUAL_OBSERVER_PTR_HPP
#define CVVISUAL_OBSERVER_PTR_HPP

// required for utilities
#include <initializer_list>
#include <memory>
#include <utility>
#include <type_traits>

// included for convinience of others:
#include <cstddef>   //size_t
#include <cstdint>   // [u]intXX_t
#include <algorithm> // since some people like to forget that one

namespace cvv
{
namespace util
{
/**
 * ObserverPtr-class to signal that a type is not owned.
 *
 * Note that const ObserverPtr<Foo> does not mean that the pointed to Foo is
 *const. If that is what
 * you want use ObserverPtr<const Foo>.
 *
 * Unlike util::Reference ObserverPtr may be null, even though this is not
 *really recommended. If it
 * points to null the only things that you may do are:
 *
 * 1) reassign another value
 * 2) compare it to another ObserverPtr
 * 3) request whether it is null via isNull()
 *
 * Everything else will result in a std::logical_error being thrown.
 */
template <typename T> class ObserverPtr
{
      public:
	// Since null is often a bad idea, don't create it by default:
	ObserverPtr() = delete;

	// these are all just the defaults but it is nice to see them explicitly
	ObserverPtr(const ObserverPtr &) = default;
	ObserverPtr(ObserverPtr &&) = default;
	ObserverPtr &operator=(const ObserverPtr &) = default;
	ObserverPtr &operator=(ObserverPtr &&) = default;

	// Constructing only works from references
	ObserverPtr(T &pointee) : ptr{ &pointee }
	{
	}
	ObserverPtr &operator=(T &pointee)
	{
		ptr = &pointee;
		return *this;
	}

	// there is no point in having a ObserverPtr to a temporary object:
	ObserverPtr(T &&) = delete;

	ObserverPtr(std::nullptr_t) : ptr{ nullptr }
	{
	}
	ObserverPtr &operator=(std::nullptr_t)
	{
		ptr = nullptr;
		return *this;
	}

	/**
	 * @brief Creates a Ref from a ObserverPtr to a type that inherits T.
	 *
	 * Trying to pass in any other type results in a compiler-error.
	 */
	template <typename Derived>
	ObserverPtr(const ObserverPtr<Derived> other)
	    : ptr{ other.getPtr() }
	{
		static_assert(std::is_base_of<T, Derived>::value,
		              "ObserverPtr<T> can only be constructed from "
		              "ObserverPtr<U> if "
		              "T is either equal to U or a base-class of U");
	}

	/**
	 * @brief Get a reference to the referenced object.
	 */
	T &operator*() const
	{
		enforceNotNull();
		return *ptr;
	}

	T *operator->() const
	{
		enforceNotNull();
		return ptr;
	}

	/**
	 * @brief Get a reference to the referenced object.
	 */
	T &get() const
	{
		enforceNotNull();
		return *ptr;
	}

	/**
	 * @brief Get a pointer to the referenced object.
	 */
	T *getPtr() const
	{
		enforceNotNull();
		return ptr;
	}

	/**
	 * @brief Tries to create a ObserverPtr to a type that inherits T.
	 *
	 * If the target-type does not inherit T, a compiler-error is created.
	 * @throws std::bad_cast if the pointee is not of the requested type.
	 */
	template <typename TargetType> ObserverPtr<TargetType> castTo() const
	{
		static_assert(std::is_base_of<T, TargetType>::value,
		              "ObserverPtr<Base>::castTo<>() can only cast to "
		              "ObserverPtr<Derived>, "
		              "where Derived inherits from Base");
		enforceNotNull();
		return makeRef(dynamic_cast<TargetType &>(*ptr));
	}

	/**
	 * Requests whether this points to nullptr;
	 */
	bool isNull() const
	{
		return !ptr;
	}

	/**
	 * @brief True if this is no nullptr.
	 */
	operator bool() const
	{
		return ptr;
	}

	/**
	 * @brief Compare to references for identity of the referenced object.
	 *
	 * @note identity != equality: two references to two different ints that
	 *both happen to have
	 * the value 1, will still compare unequal.
	 */
	bool friend operator==(const ObserverPtr &l, const ObserverPtr &r)
	{
		return l.ptr == r.ptr;
	}

	/**
	 * Dito.
	 */
	bool friend operator!=(const ObserverPtr &l, const ObserverPtr &r)
	{
		return l.ptr != r.ptr;
	}

      private:
	T *ptr;

	void enforceNotNull() const
	{
		if (!ptr)
		{
			throw std::logic_error{ "attempt to access nullptr via "
			                        "an ObserverPtr" };
		}
	}
};
}
} // namespaces

#endif
