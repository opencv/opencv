// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019-2020 Intel Corporation


#ifndef OPENCV_GAPI_GOPAQUE_HPP
#define OPENCV_GAPI_GOPAQUE_HPP

#include <functional>
#include <ostream>
#include <memory>

#include <opencv2/gapi/own/exports.hpp>
#include <opencv2/gapi/opencv_includes.hpp>

#include <opencv2/gapi/util/variant.hpp>
#include <opencv2/gapi/util/throw.hpp>
#include <opencv2/gapi/util/type_traits.hpp>
#include <opencv2/gapi/own/assert.hpp>

namespace cv
{
// Forward declaration; GNode and GOrigin are an internal
// (user-inaccessible) classes.
class GNode;
struct GOrigin;
template<typename T> class GOpaque;

/**
 * \addtogroup gapi_meta_args
 * @{
 */
struct GOpaqueDesc
{
    // FIXME: Body
    // FIXME: Also implement proper operator== then
    bool operator== (const GOpaqueDesc&) const { return true; }
};
template<typename U> GOpaqueDesc descr_of(const U &) { return {};}
static inline GOpaqueDesc empty_gopaque_desc() {return {}; }
/** @} */

std::ostream& operator<<(std::ostream& os, const cv::GOpaqueDesc &desc);

namespace detail
{
    // ConstructOpaque is a callback which stores information about T and is used by
    // G-API runtime to construct an object in host memory (T remains opaque for G-API).
    // ConstructOpaque is carried into G-API internals by GOpaqueU.
    // Currently it is suitable for Host (CPU) plugins only, real offload may require
    // more information for manual memory allocation on-device.
    class OpaqueRef;
    using ConstructOpaque = std::function<void(OpaqueRef&)>;

    // FIXME: garray.hpp already contains hint classes (for actual T type verification),
    // need to think where it can be moved (currently opaque uses it from garray)

    // This class strips type information from GOpaque<T> and makes it usable
    // in the G-API graph compiler (expression unrolling, graph generation, etc).
    // Part of GProtoArg.
    class GAPI_EXPORTS GOpaqueU
    {
    public:
        GOpaqueU(const GNode &n, std::size_t out); // Operation result constructor

        template <typename T>
        bool holds() const;                       // Check if was created from GOpaque<T>

        GOrigin& priv();                          // Internal use only
        const GOrigin& priv() const;              // Internal use only

    protected:
        GOpaqueU();                                // Default constructor
        template<class> friend class cv::GOpaque;  // (available for GOpaque<T> only)

        void setConstructFcn(ConstructOpaque &&cv);  // Store T-aware constructor

        template <typename T>
        void specifyType();                       // Store type of initial GOpaque<T>

        template <typename T>
        void storeKind();

        void setKind(cv::detail::OpaqueKind);

        std::shared_ptr<GOrigin> m_priv;
        std::shared_ptr<TypeHintBase> m_hint;
    };

    template <typename T>
    bool GOpaqueU::holds() const{
        GAPI_Assert(m_hint != nullptr);
        using U = util::decay_t<T>;
        return dynamic_cast<TypeHint<U>*>(m_hint.get()) != nullptr;
    };

    template <typename T>
    void GOpaqueU::specifyType(){
        m_hint.reset(new TypeHint<util::decay_t<T>>);
    };

    template <typename T>
    void GOpaqueU::storeKind(){
        // FIXME: Add assert here on cv::Mat and cv::Scalar?
        setKind(cv::detail::GOpaqueTraits<T>::kind);
    };

    // This class represents a typed object reference.
    // Depending on origins, this reference may be either "just a" reference to
    // an object created externally, OR actually own the underlying object
    // (be value holder).
    class BasicOpaqueRef
    {
    public:
        cv::GOpaqueDesc m_desc;
        virtual ~BasicOpaqueRef() {}

        virtual void mov(BasicOpaqueRef &ref) = 0;
        virtual const void* ptr() const = 0;
    };

    template<typename T> class OpaqueRefT final: public BasicOpaqueRef
    {
        using empty_t  = util::monostate;
        using ro_ext_t = const T *;
        using rw_ext_t =       T *;
        using rw_own_t =       T  ;
        util::variant<empty_t, ro_ext_t, rw_ext_t, rw_own_t> m_ref;

        inline bool isEmpty() const { return util::holds_alternative<empty_t>(m_ref);  }
        inline bool isROExt() const { return util::holds_alternative<ro_ext_t>(m_ref); }
        inline bool isRWExt() const { return util::holds_alternative<rw_ext_t>(m_ref); }
        inline bool isRWOwn() const { return util::holds_alternative<rw_own_t>(m_ref); }

        void init(const T* obj = nullptr)
        {
            if (obj) m_desc = cv::descr_of(*obj);
        }

    public:
        OpaqueRefT() { init(); }
        virtual ~OpaqueRefT() {}

        explicit OpaqueRefT(const T&  obj) : m_ref(&obj)           { init(&obj); }
        explicit OpaqueRefT(      T&  obj) : m_ref(&obj)           { init(&obj); }
        explicit OpaqueRefT(      T&& obj) : m_ref(std::move(obj)) { init(&obj); }

        // Reset a OpaqueRefT. Called only for objects instantiated
        // internally in G-API (e.g. temporary GOpaque<T>'s within a
        // computation).  Reset here means both initialization
        // (creating an object) and reset (discarding its existing
        // content before the next execution). Must never be called
        // for external OpaqueRefTs.
        void reset()
        {
            if (isEmpty())
            {
                T empty_obj{};
                m_desc = cv::descr_of(empty_obj);
                m_ref  = std::move(empty_obj);
                GAPI_Assert(isRWOwn());
            }
            else if (isRWOwn())
            {
                util::get<rw_own_t>(m_ref) = {};
            }
            else GAPI_Assert(false); // shouldn't be called in *EXT modes
        }

        // Obtain a WRITE reference to underlying object
        // Used by CPU kernel API wrappers when a kernel execution frame
        // is created
        T& wref()
        {
            GAPI_Assert(isRWExt() || isRWOwn());
            if (isRWExt()) return *util::get<rw_ext_t>(m_ref);
            if (isRWOwn()) return  util::get<rw_own_t>(m_ref);
            util::throw_error(std::logic_error("Impossible happened"));
        }

        // Obtain a READ reference to underlying object
        // Used by CPU kernel API wrappers when a kernel execution frame
        // is created
        const T& rref() const
        {
            // ANY object can be accessed for reading, even if it declared for
            // output. Example -- a GComputation from [in] to [out1,out2]
            // where [out2] is a result of operation applied to [out1]:
            //
            //            GComputation boundary
            //            . . . . . . .
            //            .           .
            //     [in] ----> foo() ----> [out1]
            //            .           .    :
            //            .           . . .:. . .
            //            .                V    .
            //            .              bar() ---> [out2]
            //            . . . . . . . . . . . .
            //
            if (isROExt()) return *util::get<ro_ext_t>(m_ref);
            if (isRWExt()) return *util::get<rw_ext_t>(m_ref);
            if (isRWOwn()) return  util::get<rw_own_t>(m_ref);
            util::throw_error(std::logic_error("Impossible happened"));
        }

        virtual void mov(BasicOpaqueRef &v) override {
            OpaqueRefT<T> *tv = dynamic_cast<OpaqueRefT<T>*>(&v);
            GAPI_Assert(tv != nullptr);
            wref() = std::move(tv->wref());
        }

        virtual const void* ptr() const override { return &rref(); }
    };

    // This class strips type information from OpaqueRefT<> and makes it usable
    // in the G-API executables (carrying run-time data/information to kernels).
    // Part of GRunArg.
    // Its methods are typed proxies to OpaqueRefT<T>.
    // OpaqueRef maintains "reference" semantics so two copies of OpaqueRef refer
    // to the same underlying object.
    class OpaqueRef
    {
        std::shared_ptr<BasicOpaqueRef> m_ref;
        cv::detail::OpaqueKind m_kind;

        template<typename T> inline void check() const
        {
            GAPI_DbgAssert(dynamic_cast<OpaqueRefT<T>*>(m_ref.get()) != nullptr);
        }

    public:
        OpaqueRef() = default;

        template<
            typename T,
            typename = util::are_different_t<OpaqueRef, T>
        >
        // FIXME: probably won't work with const object
        explicit OpaqueRef(T&& obj) :
            m_ref(new OpaqueRefT<util::decay_t<T>>(std::forward<T>(obj))),
            m_kind(GOpaqueTraits<util::decay_t<T>>::kind) {}

        cv::detail::OpaqueKind getKind() const
        {
            return m_kind;
        }

        template<typename T> void reset()
        {
            if (!m_ref) m_ref.reset(new OpaqueRefT<T>());
            check<T>();
            storeKind<T>();
            static_cast<OpaqueRefT<T>&>(*m_ref).reset();
        }

        template <typename T>
        void storeKind()
        {
            m_kind = cv::detail::GOpaqueTraits<T>::kind;
        }

        template<typename T> T& wref()
        {
            check<T>();
            return static_cast<OpaqueRefT<T>&>(*m_ref).wref();
        }

        template<typename T> const T& rref() const
        {
            check<T>();
            return static_cast<OpaqueRefT<T>&>(*m_ref).rref();
        }

        void mov(OpaqueRef &v)
        {
            m_ref->mov(*v.m_ref);
        }

        cv::GOpaqueDesc descr_of() const
        {
            return m_ref->m_desc;
        }

        // May be used to uniquely identify this object internally
        const void *ptr() const { return m_ref->ptr(); }
    };
} // namespace detail

/** \addtogroup gapi_data_objects
 * @{
 */

template<typename T> class GOpaque
{
public:
    GOpaque() { putDetails(); }              // Empty constructor
    explicit GOpaque(detail::GOpaqueU &&ref) // GOpaqueU-based constructor
        : m_ref(ref) { putDetails(); }       // (used by GCall, not for users)

    detail::GOpaqueU strip() const { return m_ref; }

private:
    // Host type (or Flat type) - the type this GOpaque is actually
    // specified to.
    using HT = typename detail::flatten_g<util::decay_t<T>>::type;

    static void CTor(detail::OpaqueRef& ref) {
        ref.reset<HT>();
        ref.storeKind<HT>();
    }
    void putDetails() {
        m_ref.setConstructFcn(&CTor);
        m_ref.specifyType<HT>(); // FIXME: to unify those 2 to avoid excessive dynamic_cast
        m_ref.storeKind<HT>();   //
    }

    detail::GOpaqueU m_ref;
};

/** @} */

} // namespace cv

#endif // OPENCV_GAPI_GOPAQUE_HPP
