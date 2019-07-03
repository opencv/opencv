// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GSTREAMING_COMPILED_HPP
#define OPENCV_GAPI_GSTREAMING_COMPILED_HPP

#include <vector>

#include <opencv2/gapi/opencv_includes.hpp>
#include <opencv2/gapi/own/assert.hpp>
#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/streaming/cap.hpp>

namespace cv {

class GAPI_EXPORTS GStreamingCompiled
{
public:
    class GAPI_EXPORTS Priv;
    GStreamingCompiled();

    // FIXME: More overloads?
    void setSource(GRunArgs &&ins);
    void setSource(const gapi::GVideoCapture &c);

    void start();

    /**
     * @return true if next result has been obtained,
     *    false marks end of the stream.
     */
    bool pull(cv::GRunArgsP &&outs);
    void stop();

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
