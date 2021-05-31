// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020-2021 Intel Corporation

#ifndef OPENCV_GAPI_S11N_HPP
#define OPENCV_GAPI_S11N_HPP

#include <vector>
#include <map>
#include <unordered_map>
#include <opencv2/gapi/s11n/base.hpp>
#include <opencv2/gapi/gcomputation.hpp>
#include <opencv2/gapi/rmat.hpp>

namespace cv {
namespace gapi {

/**
* \addtogroup gapi_serialization
* @{
*/

namespace detail {
    /// @private -- Exclude this function from OpenCV documentation
    GAPI_EXPORTS cv::GComputation getGraph(const std::vector<char> &p);

    /// @private -- Exclude this function from OpenCV documentation
    GAPI_EXPORTS cv::GMetaArgs getMetaArgs(const std::vector<char> &p);

    /// @private -- Exclude this function from OpenCV documentation
    GAPI_EXPORTS cv::GRunArgs getRunArgs(const std::vector<char> &p);

    /// @private -- Exclude this function from OpenCV documentation
    GAPI_EXPORTS std::vector<std::string> getVectorOfStrings(const std::vector<char> &p);

    /// @private -- Exclude this function from OpenCV documentation
    template<typename... Types>
    cv::GCompileArgs getCompileArgs(const std::vector<char> &p);

    /// @private -- Exclude this function from OpenCV documentation
    template<typename RMatAdapterType>
    cv::GRunArgs getRunArgsWithRMats(const std::vector<char> &p);
} // namespace detail

/** @brief This function allows to serialize GComputation.
 *
 * Check different overloads for more examples.
 * @param c GComputation to serialize.
 * @return serialized vector of bytes.
 */
GAPI_EXPORTS std::vector<char> serialize(const cv::GComputation &c);

/** @brief Generic function for deserializing different types.
 *
 * Check different overloads for more examples.
 * @param p vector of bytes to deserialize specified object from.
 * @return deserialized object with specified type.
 */
template<typename T> static inline
T deserialize(const std::vector<char> &p);

//! @overload
GAPI_EXPORTS std::vector<char> serialize(const cv::GCompileArgs& ca);

//! @overload
GAPI_EXPORTS std::vector<char> serialize(const cv::GMetaArgs& ma);

//! @overload
GAPI_EXPORTS std::vector<char> serialize(const cv::GRunArgs& ra);

//! @overload
GAPI_EXPORTS std::vector<char> serialize(const std::vector<std::string>& vs);

//! @overload
template<> inline
cv::GComputation deserialize(const std::vector<char> &p) {
    return detail::getGraph(p);
}

//! @overload
template<> inline
cv::GMetaArgs deserialize(const std::vector<char> &p) {
    return detail::getMetaArgs(p);
}

//! @overload
template<> inline
cv::GRunArgs deserialize(const std::vector<char> &p) {
    return detail::getRunArgs(p);
}

//! @overload
template<> inline
std::vector<std::string> deserialize(const std::vector<char> &p) {
    return detail::getVectorOfStrings(p);
}

/** @brief This function is used to deserialize GCompileArgs which
 * types were specified in the template.
 *
 * @note To be used properly all GCompileArgs types must be de serializable.
 * @param p vector of bytes to deserialize GCompileArgs object from.
 * @return GCompileArgs object.
 * @see GCompileArgs S11N
 */
template<typename T, typename... Types> inline
typename std::enable_if<std::is_same<T, GCompileArgs>::value, GCompileArgs>::
type deserialize(const std::vector<char> &p) {
    return detail::getCompileArgs<Types...>(p);
}

/** @brief This function is used to deserialize GRunArgs including RMat objects.
 *
 * RMat adapter type is specified in the template.
 * @note To be used properly specified adapter type must overload its serialize() and
 * desialize() methods.
 * @param p vector of bytes to deserialize GRunArgs object from.
 * @return GRunArgs including RMat objects.
 * @see RMat
 */
template<typename T, typename RMatAdapterType> inline
typename std::enable_if<std::is_same<T, GRunArgs>::value, GRunArgs>::
type deserialize(const std::vector<char> &p) {
    return detail::getRunArgsWithRMats<RMatAdapterType>(p);
}
} // namespace gapi
} // namespace cv

namespace cv {
namespace gapi {
namespace s11n {

/** @brief This structure is used for serialization routines.
 *
 * It's main purpose is to provide multiple overloads for operator<<()
 * with basic C++ in addition to OpenCV/G-API types and store serialized bytes.
 *
 * This sctructure can be inherited and further extended with additional types.
 *
 * For example, it is utilized in S11N as input parameter in serialize() method.
 * @see S11N
 */
struct GAPI_EXPORTS IOStream {
    virtual ~IOStream() = default;
    // Define the native support for basic C++ types at the API level:
    virtual IOStream& operator<< (bool) = 0;
    virtual IOStream& operator<< (char) = 0;
    virtual IOStream& operator<< (unsigned char) = 0;
    virtual IOStream& operator<< (short) = 0;
    virtual IOStream& operator<< (unsigned short) = 0;
    virtual IOStream& operator<< (int) = 0;
    virtual IOStream& operator<< (uint32_t) = 0;
    virtual IOStream& operator<< (uint64_t) = 0;
    virtual IOStream& operator<< (float) = 0;
    virtual IOStream& operator<< (double) = 0;
    virtual IOStream& operator<< (const std::string&) = 0;
};

/** @brief This structure is used for deserialization routines.
 *
 * It's main purpose is to provide multiple overloads for operator>>()
 * with basic C++ in addition to OpenCV/G-API types and store serialized bytes.
 *
 * This sctructure can be inherited and further extended with additional types.
 *
 * For example, it is utilized in S11N as input parameter in deserialize() method.
 * @see S11N
 */
struct GAPI_EXPORTS IIStream {
    virtual ~IIStream() = default;
    virtual IIStream& operator>> (bool &) = 0;
    virtual IIStream& operator>> (std::vector<bool>::reference) = 0;
    virtual IIStream& operator>> (char &) = 0;
    virtual IIStream& operator>> (unsigned char &) = 0;
    virtual IIStream& operator>> (short &) = 0;
    virtual IIStream& operator>> (unsigned short &) = 0;
    virtual IIStream& operator>> (int &) = 0;
    virtual IIStream& operator>> (float &) = 0;
    virtual IIStream& operator>> (double &) = 0;
    virtual IIStream& operator >> (uint32_t &) = 0;
    virtual IIStream& operator >> (uint64_t &) = 0;
    virtual IIStream& operator>> (std::string &) = 0;
};

namespace detail {
/// @private -- Exclude this function from OpenCV documentation
GAPI_EXPORTS std::unique_ptr<IIStream> getInStream(const std::vector<char> &p);
} // namespace detail

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// S11N operators
// Note: operators for basic types are defined in IIStream/IOStream

// OpenCV types ////////////////////////////////////////////////////////////////

/** @brief This operator is used to serialize cv::Point into IOStream object.
 *
 * For instance, it can be used when overloading serialization routines for
 * RMat's adapter or types inside GCompileArgs.
 *
 * Check overloads for more examples.
 * @param os IOStream object to serialize and store cv::Point.
 * @param pt cv::Point to serialize.
 * @return IOStream object with serialized cv::Point inside.
 * @see serialize IOStream
 */
GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::Point &pt);
/** @brief This operator is used to deserialize cv::Point from IIStream object.
 *
 * For instance, it can be used when overloading deserialization routines for
 * RMat's adapter or types inside GCompileArgs.
 *
 * Check overloads for more examples.
 * @param is IIStream object to deserialize cv::Point from.
 * @param pt reference to cv::Point to deserialize into.
 * @return IIStream object with deserialized and extracted cv::Point from.
 * @see deserialize IIStream
 */
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::Point &pt);

//! @overload
GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::Point2f &pt);
//! @overload
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::Point2f &pt);

//! @overload
GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::Size &sz);
//! @overload
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::Size &sz);

//! @overload
GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::Rect &rc);
//! @overload
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::Rect &rc);

//! @overload
GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::Scalar &s);
//! @overload
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::Scalar &s);

//! @overload
GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::Mat &m);
//! @overload
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::Mat &m);

// FIXME: for GRunArgs serailization
#if !defined(GAPI_STANDALONE)
//! @overload
GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::UMat & um);
//! @overload
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::UMat & um);
#endif // !defined(GAPI_STANDALONE)

/** @brief This operator is used to serialize cv::RMat into IOStream object.
 *
 * It actually serializes the adapter, so the adapter type need to implement
 * its serialize() method properly.
 * @param os IOStream object to serialize and store cv::RMat.
 * @param r cv::RMat to serialize.
 * @return IOStream object with serialized cv::RMat inside.
 * @see serialize RMat IOStream
 */
GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::RMat &r);
/** @brief This operator is used to deserialize cv::RMat from IIStream object.
 *
 * It actually deserializes the adapter, so the adapter type need to implement
 * its deserialize() method properly.
 * @param is IIStream object to deserialize cv::RMat from.
 * @param r reference to cv::RMat to deserialize into.
 * @return IIStream object with deserialized and extracted cv::RMat from.
 * @see deserialize RMat IIStream
 */
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::RMat &r);

//! @overload
GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::gapi::wip::IStreamSource::Ptr &issptr);
//! @overload
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::gapi::wip::IStreamSource::Ptr &issptr);

//! @overload
GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::detail::VectorRef &vr);
//! @overload
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::detail::VectorRef &vr);

//! @overload
GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::detail::OpaqueRef &opr);
//! @overload
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::detail::OpaqueRef &opr);

/** @brief This operator is used to serialize cv::MediaFrame into IOStream object.
 *
 * It actually serializes the adapter, so the adapter type need to implement
 * its serialization mechanism.
 * @note Currently serialization of cv::MediaFrame is not supported.
 * @param os IOStream object to serialize and store cv::MediaFrame.
 * @param mf cv::MediaFrame to serialize.
 * @return IOStream object with serialized cv::MediaFrame inside.
 * @see serialize MediaFrame IOStream
 */
GAPI_EXPORTS IOStream& operator<< (IOStream& os, const cv::MediaFrame &mf);
/** @brief This operator is used to deserialize cv::MediaFrame from IIStream object.
 *
 * It actually deserializes the adapter, so the adapter type need to implement
 * its deserialization mechanism.
 * @note Currently deserialization of cv::MediaFrame is not supported.
 * @param is IIStream object to deserialize cv::MediaFrame from.
 * @param mf reference to cv::MediaFrame to deserialize into.
 * @return IIStream object with deserialized and extracted cv::MediaFrame from.
 * @see deserialize MediaFrame IIStream
 */
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       cv::MediaFrame &mf);

// Generic STL types ////////////////////////////////////////////////////////////////
//! @overload
template<typename K, typename V>
IOStream& operator<< (IOStream& os, const std::map<K, V> &m) {
    const uint32_t sz = static_cast<uint32_t>(m.size());
    os << sz;
    for (const auto& it : m) os << it.first << it.second;
    return os;
}
//! @overload
template<typename K, typename V>
IIStream& operator>> (IIStream& is, std::map<K, V> &m) {
    m.clear();
    uint32_t sz = 0u;
    is >> sz;
    for (std::size_t i = 0; i < sz; ++i) {
        K k{};
        V v{};
        is >> k >> v;
        m[k] = v;
    }
    return is;
}

//! @overload
template<typename K, typename V>
IOStream& operator<< (IOStream& os, const std::unordered_map<K, V> &m) {
    const uint32_t sz = static_cast<uint32_t>(m.size());
    os << sz;
    for (auto &&it : m) os << it.first << it.second;
    return os;
}
//! @overload
template<typename K, typename V>
IIStream& operator>> (IIStream& is, std::unordered_map<K, V> &m) {
    m.clear();
    uint32_t sz = 0u;
    is >> sz;
    for (std::size_t i = 0; i < sz; ++i) {
        K k{};
        V v{};
        is >> k >> v;
        m[k] = v;
    }
    return is;
}

//! @overload
template<typename T>
IOStream& operator<< (IOStream& os, const std::vector<T> &ts) {
    const uint32_t sz = static_cast<uint32_t>(ts.size());
    os << sz;
    for (auto &&v : ts) os << v;
    return os;
}
//! @overload
template<typename T>
IIStream& operator>> (IIStream& is, std::vector<T> &ts) {
    uint32_t sz = 0u;
    is >> sz;
    if (sz == 0u) {
        ts.clear();
    }
    else {
        ts.resize(sz);
        for (std::size_t i = 0; i < sz; ++i) is >> ts[i];
    }
    return is;
}

// Generic: variant serialization
namespace detail {
/// @private -- Exclude this function from OpenCV documentation
template<typename V>
IOStream& put_v(IOStream&, const V&, std::size_t) {
    GAPI_Assert(false && "variant>>: requested index is invalid");
};

/// @private -- Exclude this function from OpenCV documentation
template<typename V, typename X, typename... Xs>
IOStream& put_v(IOStream& os, const V& v, std::size_t x) {
    return (x == 0u)
        ? os << cv::util::get<X>(v)
        : put_v<V, Xs...>(os, v, x-1);
}

/// @private -- Exclude this function from OpenCV documentation
template<typename V>
IIStream& get_v(IIStream&, V&, std::size_t, std::size_t) {
    GAPI_Assert(false && "variant<<: requested index is invalid");
}

/// @private -- Exclude this function from OpenCV documentation
template<typename V, typename X, typename... Xs>
IIStream& get_v(IIStream& is, V& v, std::size_t i, std::size_t gi) {
    if (i == gi) {
        X x{};
        is >> x;
        v = V{std::move(x)};
        return is;
    } else return get_v<V, Xs...>(is, v, i+1, gi);
}
} // namespace detail

//! @overload
template<typename... Ts>
IOStream& operator<< (IOStream& os, const cv::util::variant<Ts...> &v) {
    os << static_cast<uint32_t>(v.index());
    return detail::put_v<cv::util::variant<Ts...>, Ts...>(os, v, v.index());
}
//! @overload
template<typename... Ts>
IIStream& operator>> (IIStream& is, cv::util::variant<Ts...> &v) {
    int idx = -1;
    is >> idx;
    GAPI_Assert(idx >= 0 && idx < (int)sizeof...(Ts));
    return detail::get_v<cv::util::variant<Ts...>, Ts...>(is, v, 0u, idx);
}

// FIXME: consider a better solution
/// @private -- Exclude this function from OpenCV documentation
template<typename... Ts>
void getRunArgByIdx (IIStream& is, cv::util::variant<Ts...> &v, uint32_t idx) {
    is = detail::get_v<cv::util::variant<Ts...>, Ts...>(is, v, 0u, idx);
}
} // namespace s11n

namespace detail
{
/// @private -- Exclude this struct from OpenCV documentation
template<typename T> struct try_deserialize_comparg;

/// @private -- Exclude this struct from OpenCV documentation
template<> struct try_deserialize_comparg<std::tuple<>> {
static cv::util::optional<GCompileArg> exec(const std::string&, cv::gapi::s11n::IIStream&) {
        return { };
    }
};

/// @private -- Exclude this struct from OpenCV documentation
template<typename T, typename... Types>
struct try_deserialize_comparg<std::tuple<T, Types...>> {
static cv::util::optional<GCompileArg> exec(const std::string& tag, cv::gapi::s11n::IIStream& is) {
    if (tag == cv::detail::CompileArgTag<T>::tag()) {
        static_assert(cv::gapi::s11n::detail::has_S11N_spec<T>::value,
            "cv::gapi::deserialize<GCompileArgs, Types...> expects Types to have S11N "
            "specializations with deserialization callbacks!");
        return cv::util::optional<GCompileArg>(
            GCompileArg { cv::gapi::s11n::detail::S11N<T>::deserialize(is) });
    }
    return try_deserialize_comparg<std::tuple<Types...>>::exec(tag, is);
}
};

/// @private -- Exclude this struct from OpenCV documentation
template<typename T> struct deserialize_runarg;

/// @private -- Exclude this struct from OpenCV documentation
template<typename RMatAdapterType>
struct deserialize_runarg {
static GRunArg exec(cv::gapi::s11n::IIStream& is, uint32_t idx) {
    if (idx == GRunArg::index_of<RMat>()) {
        auto ptr = std::make_shared<RMatAdapterType>();
        ptr->deserialize(is);
        return GRunArg { RMat(std::move(ptr)) };
    } else { // non-RMat arg - use default deserialization
        GRunArg arg;
        getRunArgByIdx(is, arg, idx);
        return arg;
    }
}
};

/// @private -- Exclude this function from OpenCV documentation
template<typename... Types>
inline cv::util::optional<GCompileArg> tryDeserializeCompArg(const std::string& tag,
                                                             const std::vector<char>& sArg) {
    std::unique_ptr<cv::gapi::s11n::IIStream> pArgIs = cv::gapi::s11n::detail::getInStream(sArg);
    return try_deserialize_comparg<std::tuple<Types...>>::exec(tag, *pArgIs);
}

/// @private -- Exclude this function from OpenCV documentation
template<typename... Types>
cv::GCompileArgs getCompileArgs(const std::vector<char> &sArgs) {
    cv::GCompileArgs args;

    std::unique_ptr<cv::gapi::s11n::IIStream> pIs = cv::gapi::s11n::detail::getInStream(sArgs);
    cv::gapi::s11n::IIStream& is = *pIs;

    uint32_t sz = 0;
    is >> sz;
    for (uint32_t i = 0; i < sz; ++i) {
        std::string tag;
        is >> tag;

        std::vector<char> sArg;
        is >> sArg;

        cv::util::optional<GCompileArg> dArg =
            cv::gapi::detail::tryDeserializeCompArg<Types...>(tag, sArg);

        if (dArg.has_value())
        {
            args.push_back(dArg.value());
        }
    }

    return args;
}

/// @private -- Exclude this function from OpenCV documentation
template<typename RMatAdapterType>
cv::GRunArgs getRunArgsWithRMats(const std::vector<char> &p) {
    std::unique_ptr<cv::gapi::s11n::IIStream> pIs = cv::gapi::s11n::detail::getInStream(p);
    cv::gapi::s11n::IIStream& is = *pIs;
    cv::GRunArgs args;

    uint32_t sz = 0;
    is >> sz;
    for (uint32_t i = 0; i < sz; ++i) {
        uint32_t idx = 0;
        is >> idx;
        args.push_back(cv::gapi::detail::deserialize_runarg<RMatAdapterType>::exec(is, idx));
    }

    return args;
}
} // namespace detail
/** @} */

} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_S11N_HPP
