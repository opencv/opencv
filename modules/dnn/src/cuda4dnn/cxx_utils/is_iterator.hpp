// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_CXX_UTILS_IS_ITERATOR_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_CXX_UTILS_IS_ITERATOR_HPP

#include <iterator>
#include <type_traits>

namespace cv { namespace dnn { namespace cuda4dnn { namespace cxx_utils {

    namespace detail {
        template <class T, class Tag, class = void>
        struct is_iterator_helper : std::false_type {};

        template <class T, class Tag>
        struct is_iterator_helper<T, Tag,
                typename std::enable_if<std::is_base_of<Tag, typename std::iterator_traits<T>::iterator_category>::value, void>::type
            > : std::true_type {};
    }

    template <class T>
    using is_iterator = typename detail::is_iterator_helper<T, std::input_iterator_tag>;

    template <class T>
    using is_forward_iterator = typename detail::is_iterator_helper<T, std::forward_iterator_tag>;

}}}} /* namespace cv::dnn::cuda4dnn::csl::cxx_utils */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_CXX_UTILS_IS_ITERATOR_HPP */
