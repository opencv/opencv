// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GSTREAMING_COMPILED_HPP
#define OPENCV_GAPI_GSTREAMING_COMPILED_HPP

#include <memory>
#include <vector>

#include <opencv2/gapi/opencv_includes.hpp>
#include <opencv2/gapi/own/assert.hpp>
#include <opencv2/gapi/util/optional.hpp>
#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/streaming/source.hpp>

namespace cv {

template<class T> using optional = cv::util::optional<T>;

namespace detail {
template<typename T> struct wref_spec {
    using type = T;
};
template<typename T> struct wref_spec<std::vector<T> > {
    using type = T;
};

template<typename RefHolder>
struct OptRef {
    struct OptHolder {
        virtual void mov(RefHolder &h) = 0;
        virtual void reset() = 0;
        virtual ~OptHolder() = default;
        using Ptr = std::shared_ptr<OptHolder>;
    };
    template<class T> struct Holder final: OptHolder {
        std::reference_wrapper<cv::optional<T> > m_opt_ref;

        explicit Holder(cv::optional<T>& opt) : m_opt_ref(std::ref(opt)) {
        }
        virtual void mov(RefHolder &h) override {
            using U = typename wref_spec<T>::type;
            m_opt_ref.get() = cv::util::make_optional(std::move(h.template wref<U>()));
        }
        virtual void reset() override {
            m_opt_ref.get().reset();
        }
    };
    template<class T>
    explicit OptRef(cv::optional<T>& t) : m_opt{new Holder<T>(t)} {}
    void mov(RefHolder &h) { m_opt->mov(h); }
    void reset()           { m_opt->reset();}
private:
    typename OptHolder::Ptr m_opt;
};
using OptionalVectorRef = OptRef<cv::detail::VectorRef>;
using OptionalOpaqueRef = OptRef<cv::detail::OpaqueRef>;
} // namespace detail

// TODO: Keep it in sync with GRunArgP (derive the type automatically?)
using GOptRunArgP = util::variant<
    optional<cv::Mat>*,
    optional<cv::RMat>*,
    optional<cv::Scalar>*,
    cv::detail::OptionalVectorRef,
    cv::detail::OptionalOpaqueRef
>;
using GOptRunArgsP = std::vector<GOptRunArgP>;

namespace detail {

template<typename T> inline GOptRunArgP wrap_opt_arg(optional<T>& arg) {
    // By default, T goes to an OpaqueRef. All other types are specialized
    return GOptRunArgP{OptionalOpaqueRef(arg)};
}

template<typename T> inline GOptRunArgP wrap_opt_arg(optional<std::vector<T> >& arg) {
    return GOptRunArgP{OptionalVectorRef(arg)};
}

template<> inline GOptRunArgP wrap_opt_arg(optional<cv::Mat> &m) {
    return GOptRunArgP{&m};
}

template<> inline GOptRunArgP wrap_opt_arg(optional<cv::Scalar> &s) {
    return GOptRunArgP{&s};
}

} // namespace detail

// Now cv::gout() may produce an empty vector (see "dynamic graphs"), so
// there may be a conflict between these two. State here that Opt version
// _must_ have at least one input for this overload
template<typename T, typename... Ts>
inline GOptRunArgsP gout(optional<T>&arg, optional<Ts>&... args)
{
    return GOptRunArgsP{ detail::wrap_opt_arg(arg), detail::wrap_opt_arg(args)... };
}

/**
 * \addtogroup gapi_main_classes
 * @{
 */
/**
 * @brief Represents a computation (graph) compiled for streaming.
 *
 * This class represents a product of graph compilation (calling
 * cv::GComputation::compileStreaming()). Objects of this class
 * actually do stream processing, and the whole pipeline execution
 * complexity is incapsulated into objects of this class. Execution
 * model has two levels: at the very top, the execution of a
 * heterogeneous graph is aggressively pipelined; at the very bottom
 * the execution of every internal block is determined by its
 * associated backend. Backends are selected based on kernel packages
 * passed via compilation arguments ( see @ref gapi_compile_args,
 * GNetworkPackage, GKernelPackage for details).
 *
 * GStreamingCompiled objects have a "player" semantics -- there are
 * methods like start() and stop(). GStreamingCompiled has a full
 * control over a videostream and so is stateful. You need to specify the
 * input stream data using setSource() and then call start() to
 * actually start processing. After that, use pull() or try_pull() to
 * obtain next processed data frame from the graph in a blocking or
 * non-blocking way, respectively.
 *
 * Currently a single GStreamingCompiled can process only one video
 * streat at time. Produce multiple GStreamingCompiled objects to run the
 * same graph on multiple video streams.
 *
 * @sa GCompiled
 */
class GAPI_EXPORTS_W_SIMPLE GStreamingCompiled
{
public:
    class GAPI_EXPORTS Priv;
    GAPI_WRAP GStreamingCompiled();

    // FIXME: More overloads?
    /**
     * @brief Specify the input data to GStreamingCompiled for
     * processing, a generic version.
     *
     * Use gin() to create an input parameter vector.
     *
     * Input vectors must have the same number of elements as defined
     * in the cv::GComputation protocol (at the moment of its
     * construction). Shapes of elements also must conform to protocol
     * (e.g. cv::Mat needs to be passed where cv::GMat has been
     * declared as input, and so on). Run-time exception is generated
     * on type mismatch.
     *
     * In contrast with regular GCompiled, user can also pass an
     * object of type GVideoCapture for a GMat parameter of the parent
     * GComputation.  The compiled pipeline will start fetching data
     * from that GVideoCapture and feeding it into the
     * pipeline. Pipeline stops when a GVideoCapture marks end of the
     * stream (or when stop() is called).
     *
     * Passing a regular Mat for a GMat parameter makes it "infinite"
     * source -- pipeline may run forever feeding with this Mat until
     * stopped explicitly.
     *
     * Currently only a single GVideoCapture is supported as input. If
     * the parent GComputation is declared with multiple input GMat's,
     * one of those can be specified as GVideoCapture but all others
     * must be regular Mat objects.
     *
     * Throws if pipeline is already running. Use stop() and then
     * setSource() to run the graph on a new video stream.
     *
     * @note This method is not thread-safe (with respect to the user
     * side) at the moment. Protect the access if
     * start()/stop()/setSource() may be called on the same object in
     * multiple threads in your application.
     *
     * @param ins vector of inputs to process.
     * @sa gin
     */
    GAPI_WRAP void setSource(GRunArgs &&ins);

    /**
     * @brief Specify an input video stream for a single-input
     * computation pipeline.
     *
     * Throws if pipeline is already running. Use stop() and then
     * setSource() to run the graph on a new video stream.
     *
     * @overload
     * @param s a shared pointer to IStreamSource representing the
     * input video stream.
     */
    GAPI_WRAP void setSource(const gapi::wip::IStreamSource::Ptr& s);

    /**
     * @brief Constructs and specifies an input video stream for a
     * single-input computation pipeline with the given parameters.
     *
     * Throws if pipeline is already running. Use stop() and then
     * setSource() to run the graph on a new video stream.
     *
     * @overload
     * @param args arguments used to contruct and initialize a stream
     * source.
     */
    template<typename T, typename... Args>
    void setSource(Args&&... args) {
        setSource(cv::gapi::wip::make_src<T>(std::forward<Args>(args)...));
    }

    /**
     * @brief Start the pipeline execution.
     *
     * Use pull()/try_pull() to obtain data. Throws an exception if
     * a video source was not specified.
     *
     * setSource() must be called first, even if the pipeline has been
     * working already and then stopped (explicitly via stop() or due
     * stream completion)
     *
     * @note This method is not thread-safe (with respect to the user
     * side) at the moment. Protect the access if
     * start()/stop()/setSource() may be called on the same object in
     * multiple threads in your application.
     */
    GAPI_WRAP void start();

    /**
     * @brief Get the next processed frame from the pipeline.
     *
     * Use gout() to create an output parameter vector.
     *
     * Output vectors must have the same number of elements as defined
     * in the cv::GComputation protocol (at the moment of its
     * construction). Shapes of elements also must conform to protocol
     * (e.g. cv::Mat needs to be passed where cv::GMat has been
     * declared as output, and so on). Run-time exception is generated
     * on type mismatch.
     *
     * This method writes new data into objects passed via output
     * vector.  If there is no data ready yet, this method blocks. Use
     * try_pull() if you need a non-blocking version.
     *
     * @param outs vector of output parameters to obtain.
     * @return true if next result has been obtained,
     *    false marks end of the stream.
     */
    bool pull(cv::GRunArgsP &&outs);

    // NB: Used from python
    GAPI_WRAP std::tuple<bool, cv::GRunArgs> pull();

    /**
     * @brief Get some next available data from the pipeline.
     *
     * This method takes a vector of cv::optional object. An object is
     * assigned to some value if this value is available (ready) at
     * the time of the call, and resets the object to empty() if it is
     * not.
     *
     * This is a blocking method which guarantees that some data has
     * been written to the output vector on return.
     *
     * Using this method only makes sense if the graph has
     * desynchronized parts (see cv::gapi::desync). If there is no
     * desynchronized parts in the graph, the behavior of this
     * method is identical to the regular pull() (all data objects are
     * produced synchronously in the output vector).
     *
     * Use gout() to create an output parameter vector.
     *
     * Output vectors must have the same number of elements as defined
     * in the cv::GComputation protocol (at the moment of its
     * construction). Shapes of elements also must conform to protocol
     * (e.g. cv::optional<cv::Mat> needs to be passed where cv::GMat
     * has been declared as output, and so on). Run-time exception is
     * generated on type mismatch.
     *
     * This method writes new data into objects passed via output
     * vector.  If there is no data ready yet, this method blocks. Use
     * try_pull() if you need a non-blocking version.
     *
     * @param outs vector of output parameters to obtain.
     * @return true if next result has been obtained,
     *    false marks end of the stream.
     *
     * @sa cv::gapi::desync
     */
    bool pull(cv::GOptRunArgsP &&outs);

    /**
     * @brief Try to get the next processed frame from the pipeline.
     *
     * Use gout() to create an output parameter vector.
     *
     * This method writes new data into objects passed via output
     * vector.  If there is no data ready yet, the output vector
     * remains unchanged and false is returned.
     *
     * @return true if data has been obtained, and false if it was
     *    not. Note: false here doesn't mark the end of the stream.
     */
    bool try_pull(cv::GRunArgsP &&outs);

    /**
     * @brief Stop (abort) processing the pipeline.
     *
     * Note - it is not pause but a complete stop. Calling start()
     * will cause G-API to start processing the stream from the early beginning.
     *
     * Throws if the pipeline is not running.
     */
    GAPI_WRAP void stop();

    /**
     * @brief Test if the pipeline is running.
     *
     * @note This method is not thread-safe (with respect to the user
     * side) at the moment. Protect the access if
     * start()/stop()/setSource() may be called on the same object in
     * multiple threads in your application.
     *
     * @return true if the current stream is not over yet.
     */
    GAPI_WRAP bool running() const;

    /// @private
    Priv& priv();

    /**
     * @brief Check if compiled object is valid (non-empty)
     *
     * @return true if the object is runnable (valid), false otherwise
     */
    explicit operator bool () const;

    /**
     * @brief Vector of metadata this graph was compiled for.
     *
     * @return Unless _reshape_ is not supported, return value is the
     * same vector which was passed to cv::GComputation::compile() to
     * produce this compiled object. Otherwise, it is the latest
     * metadata vector passed to reshape() (if that call was
     * successful).
     */
    const GMetaArgs& metas() const; // Meta passed to compile()

    /**
     * @brief Vector of metadata descriptions of graph outputs
     *
     * @return vector with formats/resolutions of graph's output
     * objects, auto-inferred from input metadata vector by
     * operations which form this computation.
     *
     * @note GCompiled objects produced from the same
     * cv::GComputiation graph with different input metas may return
     * different values in this vector.
     */
    const GMetaArgs& outMetas() const;

protected:
    /// @private
    std::shared_ptr<Priv> m_priv;
};
/** @} */

}

#endif // OPENCV_GAPI_GSTREAMING_COMPILED_HPP
