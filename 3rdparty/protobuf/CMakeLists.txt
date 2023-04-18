project(libprotobuf)
include(CheckIncludeFiles)

if(HAVE_PTHREAD)
  add_definitions(-DHAVE_PTHREAD=1)
endif()

if(MSVC)
  add_definitions( -D_CRT_SECURE_NO_WARNINGS=1 )
  ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4244 /wd4267 /wd4018 /wd4355 /wd4800 /wd4251 /wd4996 /wd4146
                                       /wd4305 /wd4127 /wd4100 /wd4512 /wd4125 /wd4389 /wd4510 /wd4610
                                       /wd4702 /wd4456 /wd4457 /wd4065 /wd4310 /wd4661 /wd4506
                                       /wd4701 /wd4703     # potentially uninitialized local/pointer variable 'value' used
                                       /wd4505             # unreferenced local function has been removed
  )
  if(MSVC_VERSION LESS 1920)  # MSVS 2015/2017
    ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4309)          # 'static_cast': truncation of constant value
  endif()
else()
  #NOTE: -Wno-invalid-offsetof was used as solution for invalid offset warning on protobuf #3450
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-deprecated -Wmissing-prototypes -Wmissing-declarations -Wshadow
                                       -Wunused-parameter -Wunused-local-typedefs -Wsign-compare -Wsign-promo
                                       -Wundef -Wtautological-undefined-compare -Wignored-qualifiers -Wextra
                                       -Wunused-function -Wunused-const-variable -Wshorten-64-to-32 -Wno-invalid-offsetof
                                       -Wenum-compare-switch
                                       -Wsuggest-override -Winconsistent-missing-override
                                       -Wimplicit-fallthrough
                                       -Warray-bounds  # GCC 9+
  )
endif()
if(CV_ICC)
  ocv_warnings_disable(CMAKE_CXX_FLAGS
      -wd265 -wd858 -wd873 -wd2196
  )
endif()
if(CV_GCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0)
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wclass-memaccess)
endif()

# Easier to support different versions of protobufs
function(append_if_exist OUTPUT_LIST)
    set(${OUTPUT_LIST})
    foreach(fil ${ARGN})
        if(EXISTS ${fil})
            list(APPEND ${OUTPUT_LIST} "${fil}")
        else()
            message(WARNING "file missing: ${fil}")
        endif()
    endforeach()
    set(${OUTPUT_LIST} ${${OUTPUT_LIST}} PARENT_SCOPE)
endfunction()

set(PROTOBUF_ROOT "${CMAKE_CURRENT_LIST_DIR}")


append_if_exist(Protobuf_SRCS
# libprotobuf-lite
  ${PROTOBUF_ROOT}/src/google/protobuf/any_lite.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/arena.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/arenastring.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/extension_set.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/generated_enum_util.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/generated_message_table_driven_lite.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/generated_message_tctable_lite.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/generated_message_util.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/implicit_weak_message.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/inlined_string_field.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/io/coded_stream.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/io/io_win32.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/io/strtod.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/io/zero_copy_stream.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/io/zero_copy_stream_impl.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/io/zero_copy_stream_impl_lite.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/map.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/message_lite.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/parse_context.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/repeated_field.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/repeated_ptr_field.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/bytestream.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/common.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/int128.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/status.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/statusor.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/stringpiece.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/stringprintf.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/structurally_valid.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/strutil.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/time.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/wire_format_lite.cc

# libprotobuf
  ${PROTOBUF_ROOT}/src/google/protobuf/any.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/any.pb.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/api.pb.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/compiler/importer.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/compiler/parser.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/descriptor.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/descriptor.pb.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/descriptor_database.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/duration.pb.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/dynamic_message.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/empty.pb.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/extension_set_heavy.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/field_mask.pb.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/generated_message_bases.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/generated_message_reflection.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/generated_message_table_driven.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/generated_message_tctable_full.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/io/gzip_stream.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/io/printer.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/io/tokenizer.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/map_field.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/message.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/reflection_ops.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/service.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/source_context.pb.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/struct.pb.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/substitute.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/text_format.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/timestamp.pb.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/type.pb.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/unknown_field_set.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/util/delimited_message_util.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/util/field_comparator.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/util/field_mask_util.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/datapiece.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/default_value_objectwriter.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/error_listener.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/field_mask_utility.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/json_escaping.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/json_objectwriter.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/json_stream_parser.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/object_writer.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/proto_writer.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/protostream_objectsource.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/protostream_objectwriter.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/type_info.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/utility.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/util/json_util.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/util/message_differencer.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/util/time_util.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/util/type_resolver_util.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/wire_format.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/wrappers.pb.cc
)


include_directories(BEFORE "${PROTOBUF_ROOT}/src")  # ensure using of own headers: https://github.com/opencv/opencv/issues/13328

add_library(libprotobuf STATIC ${OPENCV_3RDPARTY_EXCLUDE_FROM_ALL} ${Protobuf_SRCS})
target_include_directories(libprotobuf SYSTEM PUBLIC $<BUILD_INTERFACE:${PROTOBUF_ROOT}/src>)
set_target_properties(libprotobuf
    PROPERTIES
    FOLDER "3rdparty"
    OUTPUT_NAME libprotobuf
    DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
    COMPILE_PDB_NAME libprotobuf
    COMPILE_PDB_NAME_DEBUG "libprotobuf${OPENCV_DEBUG_POSTFIX}"
    ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
    )

if(ANDROID)
  # https://github.com/opencv/opencv/issues/17282
  target_link_libraries(libprotobuf INTERFACE "-landroid" "-llog")
endif()

get_protobuf_version(Protobuf_VERSION "${PROTOBUF_ROOT}/src")
set(Protobuf_VERSION ${Protobuf_VERSION} CACHE INTERNAL "" FORCE)

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(libprotobuf EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev OPTIONAL)
endif()

ocv_install_3rdparty_licenses(protobuf LICENSE README.md)
