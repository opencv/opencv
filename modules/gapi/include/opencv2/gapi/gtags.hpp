#ifndef OPENCV_GAPI_GTAGS_HPP
#define OPENCV_GAPI_GTAGS_HPP

#include <type_traits>

namespace cv
{
namespace gapi
{
template<typename ...Tags>
struct TagHolder : Tags...
{
};

namespace tag
{
    struct Meta {};
    struct GraphRejected {};
}

#define GAPI_OBJECT(TAG_1)        using tags_t = cv::gapi::TagHolder<cv::gapi::tag:: ## TAG_1 >;
#define GAPI_OBJECT_2(TAG_1, TAG_2) using tags_t = cv::gapi::TagHolder<cv::gapi::tag:: ## TAG_1, cv::gapi::tag:: ## TAG_2 >;

template<typename... Ts>
struct make_void { typedef void type;};

template<typename... Ts>
using void_t = typename make_void<Ts...>::type;

template<typename, typename = void>
struct is_tagged_type : std::false_type {};

template<typename TaggedTypeCandidate>
struct is_tagged_type<TaggedTypeCandidate,
                      void_t<typename TaggedTypeCandidate::tags_t>> :
    std::true_type
{};

template <typename Type, typename Tag>
struct is_contain_tag_impl : std::is_base_of<Tag, typename Type::tags_t> {
};

template<typename Type, typename Tag>
struct has_tag : std::conditional<is_tagged_type<Type>::value,
                                         is_contain_tag_impl<Type, Tag>,
                                         std::false_type>::type
{};

}
}
#endif // OPENCV_GAPI_GTAGS_HPP
