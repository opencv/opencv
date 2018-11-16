// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GCOMPUTATION_HPP
#define OPENCV_GAPI_GCOMPUTATION_HPP

#include <functional>

#include "opencv2/gapi/util/util.hpp"
#include "opencv2/gapi/gcommon.hpp"
#include "opencv2/gapi/gproto.hpp"
#include "opencv2/gapi/garg.hpp"
#include "opencv2/gapi/gcompiled.hpp"

namespace cv {

namespace detail
{
    // FIXME: move to algorithm, cover with separate tests
    // FIXME: replace with O(1) version (both memory and compilation time)
    template<typename...>
    struct last_type;

    template<typename T>
    struct last_type<T> { using type = T;};

    template<typename T, typename... Ts>
    struct last_type<T, Ts...> { using type = typename last_type<Ts...>::type; };

    template<typename... Ts>
    using last_type_t = typename last_type<Ts...>::type;
}

/**
 * \addtogroup gapi_main_classes
 * @{
 */
/**
 * @brief GComputation class represents a captured computation
 * graph. GComputation objects form boundaries for expression code
 * user writes with G-API, allowing to compile and execute it.
 *
 * G-API computations are defined with input/output data
 * objects. G-API will track automatically which operations connect
 * specified outputs to the inputs, forming up a call graph to be
 * executed. The below example expresses calculation of Sobel operator
 * for edge detection (\f$G = \sqrt{G_x^2 + G_y^2}\f$):
 *
 * @snippet modules/gapi/samples/api_ref_snippets.cpp graph_def
 *
 * Full pipeline can be now captured with this object declaration:
 *
 * @snippet modules/gapi/samples/api_ref_snippets.cpp graph_cap_full
 *
 * Input/output data objects on which a call graph should be
 * reconstructed are passed using special wrappers cv::GIn and
 * cv::GOut. G-API will track automatically which operations form a
 * path from inputs to outputs and build the execution graph appropriately.
 *
 * Note that cv::GComputation doesn't take ownership on data objects
 * it is defined. Moreover, multiple GComputation objects may be
 * defined on the same expressions, e.g. a smaller pipeline which
 * expects that image gradients are already pre-calculated may be
 * defined like this:
 *
 * @snippet modules/gapi/samples/api_ref_snippets.cpp graph_cap_sub
 *
 * The resulting graph would expect two inputs and produce one
 * output. In this case, it doesn't matter if gx/gy data objects are
 * results of cv::gapi::Sobel operators -- G-API will stop unrolling
 * expressions and building the underlying graph one reaching this
 * data objects.
 *
 * The way how GComputation is defined is important as its definition
 * specifies graph _protocol_ -- the way how the graph should be
 * used. Protocol is defined by number of inputs, number of outputs,
 * and shapes of inputs and outputs.
 *
 * In the above example, sobelEdge expects one Mat on input and
 * produces one Mat; while sobelEdgeSub expects two Mats on input and
 * produces one Mat. GComputation's protocol defines how other
 * computaion methods should be used -- cv::GComputation::compile() and
 * cv::GComputation::apply(). For example, if a graph is defined on
 * two GMat inputs, two cv::Mat objects have to be passed to apply()
 * for execution. GComputation checks protocol correctness in runtime
 * so passing a different number of objects in apply() or passing
 * cv::Scalar instead of cv::Mat there would compile well as a C++
 * source but raise an exception in run-time. G-API also comes with a
 * typed wrapper cv::GComputationT<> which introduces this type-checking in
 * compile-time.
 *
 * cv::GComputation itself is a thin object which just captures what
 * the graph is. The compiled graph (which actually process data) is
 * represented by class GCompiled. Use compile() method to generate a
 * compiled graph with given compile options. cv::GComputation can
 * also be used to process data with implicit graph compilation
 * on-the-fly, see apply() for details.
 *
 * GComputation is a reference-counted object -- once defined, all its
 * copies will refer to the same instance.
 *
 * @sa GCompiled
 */
class GAPI_EXPORTS GComputation
{
public:
    class Priv;
    typedef std::function<GComputation()> Generator;

    // Various constructors enable different ways to define a computation: /////
    // 1. Generic constructors
    /**
     * @brief Define a computation using a generator function.
     *
     * Graph can be defined in-place directly at the moment of its
     * construction with a lambda:
     *
     * @snippet modules/gapi/samples/api_ref_snippets.cpp graph_gen
     *
     * This may be useful since all temporary objects (cv::GMats) and
     * namespaces can be localized to scope of lambda, without
     * contaminating the parent scope with probably unecessary objects
     * and information.
     *
     * @param gen generator function which returns a cv::GComputation,
     * see Generator.
     */
    GComputation(const Generator& gen);                // Generator
                                                       // overload

    /**
     * @brief Generic GComputation constructor.
     *
     * Constructs a new graph with a given protocol, specified as a
     * flow of operations connecting input/output objects. Throws if
     * the passed boundaries are invalid, e.g. if there's no
     * functional dependency (path) between given outputs and inputs.
     *
     * @param ins Input data vector.
     * @param outs Output data vector.
     *
     * @note Don't construct GProtoInputArgs/GProtoOutputArgs objects
     * directly, use cv::GIn()/cv::GOut() wrapper functions instead.
     *
     * @sa @ref gapi_data_objects
     */
    GComputation(GProtoInputArgs &&ins,
                 GProtoOutputArgs &&outs);             // Arg-to-arg overload

    // 2. Syntax sugar and compatibility overloads
    /**
     * @brief Defines an unary (one input -- one output) computation
     *
     * @overload
     * @param in input GMat of the defined unary computation
     * @param out output GMat of the defined unary computation
     */
    GComputation(GMat in, GMat out);                   // Unary overload

    /**
     * @brief Defines an unary (one input -- one output) computation
     *
     * @overload
     * @param in input GMat of the defined unary computation
     * @param out output GScalar of the defined unary computation
     */
    GComputation(GMat in, GScalar out);                // Unary overload (scalar)

    /**
     * @brief Defines a binary (two inputs -- one output) computation
     *
     * @overload
     * @param in1 first input GMat of the defined binary computation
     * @param in2 second input GMat of the defined binary computation
     * @param out output GMat of the defined binary computation
     */
    GComputation(GMat in1, GMat in2, GMat out);        // Binary overload

    /**
     * @brief Defines a binary (two inputs -- one output) computation
     *
     * @overload
     * @param in1 first input GMat of the defined binary computation
     * @param in2 second input GMat of the defined binary computation
     * @param out output GScalar of the defined binary computation
     */
    GComputation(GMat in1, GMat in2, GScalar out);     // Binary
                                                       // overload
                                                       // (scalar)

    /**
     * @brief Defines a computation with arbitrary input/output number.
     *
     * @overload
     * @param ins vector of inputs GMats for this computation
     * @param outs vector of outputs GMats for this computation
     *
     * Use this overload for cases when number of computation
     * inputs/outputs is not known in compile-time -- e.g. when graph
     * is programmatically generated to build an image pyramid with
     * the given number of levels, etc.
     */
    GComputation(const std::vector<GMat> &ins,         // Compatibility overload
                 const std::vector<GMat> &outs);

    // Various versions of apply(): ////////////////////////////////////////////
    // 1. Generic apply()
    /**
     * @brief Compile graph on-the-fly and immediately execute it on
     * the inputs data vectors.
     *
     * Number of input/output data objects must match GComputation's
     * protocol, also types of host data objects (cv::Mat, cv::Scalar)
     * must match the shapes of data objects from protocol (cv::GMat,
     * cv::GScalar). If there's a mismatch, a run-time exception will
     * be generated.
     *
     * Internally, a cv::GCompiled object is created for the given
     * input format configuration, which then is executed on the input
     * data immediately. cv::GComputation caches compiled objects
     * produced within apply() -- if this method would be called next
     * time with the same input parameters (image formats, image
     * resolution, etc), the underlying compiled graph will be reused
     * without recompilation. If new metadata doesn't match the cached
     * one, the underlying compiled graph is regenerated.
     *
     * @note compile() always triggers a compilation process and
     * produces a new GCompiled object regardless if a similar one has
     * been cached via apply() or not.
     *
     * @param ins vector of input data to process. Don't create
     * GRunArgs object manually, use cv::gin() wrapper instead.
     * @param outs vector of output data to fill results in. cv::Mat
     * objects may be empty in this vector, G-API will automatically
     * initialize it with the required format & dimensions. Don't
     * create GRunArgsP object manually, use cv::gout() wrapper instead.
     * @param args a list of compilation arguments to pass to the
     * underlying compilation process. Don't create GCompileArgs
     * object manually, use cv::compile_args() wrapper instead.
     *
     * @sa @ref gapi_data_objects, @ref gapi_compile_args
     */
    void apply(GRunArgs &&ins, GRunArgsP &&outs, GCompileArgs &&args = {});       // Arg-to-arg overload

    /// @private -- Exclude this function from OpenCV documentation
    void apply(const std::vector<cv::gapi::own::Mat>& ins,                        // Compatibility overload
               const std::vector<cv::gapi::own::Mat>& outs,
               GCompileArgs &&args = {});

    // 2. Syntax sugar and compatibility overloads
#if !defined(GAPI_STANDALONE)
    /**
     * @brief Execute an unary computation (with compilation on the fly)
     *
     * @overload
     * @param in input cv::Mat for unary computation
     * @param out output cv::Mat for unary computation
     * @param args compilation arguments for underlying compilation
     * process.
     */
    void apply(cv::Mat in, cv::Mat &out, GCompileArgs &&args = {});               // Unary overload

    /**
     * @brief Execute an unary computation (with compilation on the fly)
     *
     * @overload
     * @param in input cv::Mat for unary computation
     * @param out output cv::Scalar for unary computation
     * @param args compilation arguments for underlying compilation
     * process.
     */
    void apply(cv::Mat in, cv::Scalar &out, GCompileArgs &&args = {});            // Unary overload (scalar)

    /**
     * @brief Execute a binary computation (with compilation on the fly)
     *
     * @overload
     * @param in1 first input cv::Mat for binary computation
     * @param in2 second input cv::Mat for binary computation
     * @param out output cv::Mat for binary computation
     * @param args compilation arguments for underlying compilation
     * process.
     */
    void apply(cv::Mat in1, cv::Mat in2, cv::Mat &out, GCompileArgs &&args = {}); // Binary overload

    /**
     * @brief Execute an binary computation (with compilation on the fly)
     *
     * @overload
     * @param in1 first input cv::Mat for binary computation
     * @param in2 second input cv::Mat for binary computation
     * @param out output cv::Scalar for binary computation
     * @param args compilation arguments for underlying compilation
     * process.
     */
    void apply(cv::Mat in1, cv::Mat in2, cv::Scalar &out, GCompileArgs &&args = {}); // Binary overload (scalar)

    /**
     * @brief Execute a computation with arbitrary number of
     * inputs/outputs (with compilation on-the-fly).
     *
     * @overload
     * @param ins vector of input cv::Mat objects to process by the
     * computation.
     * @param outs vector of output cv::Mat objects to produce by the
     * computation.
     * @param args compilation arguments for underlying compilation
     * process.
     *
     * Numbers of elements in ins/outs vectos must match numbers of
     * inputs/outputs which were used to define this GComputation.
     */
    void apply(const std::vector<cv::Mat>& ins,         // Compatibility overload
               const std::vector<cv::Mat>& outs,
               GCompileArgs &&args = {});
#endif // !defined(GAPI_STANDALONE)
    // Various versions of compile(): //////////////////////////////////////////
    // 1. Generic compile() - requires metas to be passed as vector
    /**
     * @brief Compile the computation for specific input format(s).
     *
     * This method triggers compilation process and produces a new
     * GCompiled object which then can process data of the given
     * format. Passing data with different format to the compiled
     * computation will generate a run-time exception.
     *
     * @param in_metas vector of input metadata configuration. Grab
     * metadata from real data objects (like cv::Mat or cv::Scalar)
     * using cv::descr_of(), or create it on your own.
     * @param args compilation arguments for this compilation
     * process. Compilation arguments directly affect what kind of
     * executable object would be produced, e.g. which kernels (and
     * thus, devices) would be used to execute computation.
     *
     * @return GCompiled, an executable computation compiled
     * specifically for the given input parameters.
     *
     * @sa @ref gapi_compile_args
     */
    GCompiled compile(GMetaArgs &&in_metas, GCompileArgs &&args = {});

    // 2. Syntax sugar - variadic list of metas, no extra compile args
    // FIXME: SFINAE looks ugly in the generated documentation
    /**
     * @overload
     *
     * Takes a variadic parameter pack with metadata
     * descriptors for which a compiled object needs to be produced.
     *
     * @return GCompiled, an executable computation compiled
     * specifically for the given input parameters.
     */
    template<typename... Ts>
    auto compile(const Ts&... metas) ->
        typename std::enable_if<detail::are_meta_descrs<Ts...>::value, GCompiled>::type
    {
        return compile(GMetaArgs{GMetaArg(metas)...}, GCompileArgs());
    }

    // 3. Syntax sugar - variadic list of metas, extra compile args
    // (seems optional parameters don't work well when there's an variadic template
    // comes first)
    //
    // Ideally it should look like:
    //
    //     template<typename... Ts>
    //     GCompiled compile(const Ts&... metas, GCompileArgs &&args)
    //
    // But not all compilers can hande this (and seems they shouldn't be able to).
    // FIXME: SFINAE looks ugly in the generated documentation
    /**
     * @overload
     *
     * Takes a  variadic parameter pack with metadata
     * descriptors for which a compiled object needs to be produced,
     * followed by GCompileArgs object representing compilation
     * arguments for this process.
     *
     * @return GCompiled, an executable computation compiled
     * specifically for the given input parameters.
     */
    template<typename... Ts>
    auto compile(const Ts&... meta_and_compile_args) ->
        typename std::enable_if<detail::are_meta_descrs_but_last<Ts...>::value
                                && std::is_same<GCompileArgs, detail::last_type_t<Ts...> >::value,
                                GCompiled>::type
    {
        //FIXME: wrapping meta_and_compile_args into a tuple to unwrap them inside a helper function is the overkill
        return compile(std::make_tuple(meta_and_compile_args...),
                       typename detail::MkSeq<sizeof...(Ts)-1>::type());
    }

    // Internal use only
    /// @private
    Priv& priv();
    /// @private
    const Priv& priv() const;

protected:

    // 4. Helper method for (3)
    /// @private
    template<typename... Ts, int... IIs>
    GCompiled compile(const std::tuple<Ts...> &meta_and_compile_args, detail::Seq<IIs...>)
    {
        GMetaArgs meta_args = {GMetaArg(std::get<IIs>(meta_and_compile_args))...};
        GCompileArgs comp_args = std::get<sizeof...(Ts)-1>(meta_and_compile_args);
        return compile(std::move(meta_args), std::move(comp_args));
    }
    /// @private
    std::shared_ptr<Priv> m_priv;
};
/** @} */

namespace gapi
{
    // FIXME: all these standalone functions need to be added to some
    // common documentation section
    /**
     * @brief Define an tagged island (subgraph) within a computation.
     *
     * Declare an Island tagged with `name` and defined from `ins` to `outs`
     * (exclusively, as ins/outs are data objects, and regioning is done on
     * operations level).
     * Throws if any operation between `ins` and `outs` are already assigned
     * to another island.
     *
     * Islands allow to partition graph into subgraphs, fine-tuning
     * the way it is scheduled by the underlying executor.
     *
     * @param name name of the Island to create
     * @param ins vector of input data objects where the subgraph
     * begins
     * @param outs vector of output data objects where the subgraph
     * ends.
     *
     * The way how an island is defined is similar to how
     * cv::GComputation is defined on input/output data objects.
     * Same rules apply here as well -- if there's no functional
     * dependency between inputs and outputs or there's not enough
     * input data objects were specified to properly calculate all
     * outputs, an exception is thrown.
     *
     * Use cv::GIn() / cv::GOut() to specify input/output vectors.
     */
    void GAPI_EXPORTS island(const std::string &name,
                             GProtoInputArgs  &&ins,
                             GProtoOutputArgs &&outs);
} // namespace gapi

} // namespace cv
#endif // OPENCV_GAPI_GCOMPUTATION_HPP
