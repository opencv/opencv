// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CSL_WORKSPACE_HPP
#define OPENCV_DNN_CSL_WORKSPACE_HPP

#include <cstddef>
#include <memory>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl {

    /** @brief maintains a single block of reusable device memory
     *
     * Each Workspace object is intended to be used by a single entity at a time but by
     * different entities at different times. It maintains a single reusable block of memory which
     * is sufficient for the largest consumer.
     */
    class Workspace {
    public:
        Workspace();

        /** @brief reserve \p bytes of memory */
        void require(std::size_t bytes);

        /** @brief number of bytes reserved by the largest consumer */
        std::size_t size() const noexcept;

    private:
        friend class WorkspaceAccessor;

        class Impl;
        std::shared_ptr<Impl> impl;
    };

}}}} /* cv::dnn::cuda4dnn::csl */

#endif /* OPENCV_DNN_CSL_WORKSPACE_HPP */
