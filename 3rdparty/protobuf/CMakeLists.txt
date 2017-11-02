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
  )
else()
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-deprecated -Wmissing-prototypes -Wmissing-declarations -Wshadow
                                       -Wunused-parameter -Wunused-local-typedefs -Wsign-compare -Wsign-promo
                                       -Wundef -Wtautological-undefined-compare -Wignored-qualifiers -Wextra
                                       -Wunused-function -Wunused-const-variable -Wshorten-64-to-32
  )
endif()
if(CV_ICC)
  ocv_warnings_disable(CMAKE_CXX_FLAGS
      -wd265 -wd858 -wd873 -wd2196
  )
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

if(MSVC)
  set(ATOMICOPS_INTERNALS ${PROTOBUF_ROOT}/src/google/protobuf/stubs/atomicops_internals_x86_msvc.cc)
else()
  set(ATOMICOPS_INTERNALS ${PROTOBUF_ROOT}/src/google/protobuf/stubs/atomicops_internals_x86_gcc.cc)
endif()


append_if_exist(Protobuf_SRCS
# libprotobuf-lite
  ${PROTOBUF_ROOT}/src/google/protobuf/arena.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/arenastring.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/extension_set.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/generated_message_util.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/io/coded_stream.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/io/zero_copy_stream.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/io/zero_copy_stream_impl_lite.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/message_lite.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/repeated_field.cc
  ${ATOMICOPS_INTERNALS}
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/bytestream.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/common.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/int128.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/once.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/status.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/statusor.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/stringpiece.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/stringprintf.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/structurally_valid.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/strutil.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/time.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/wire_format_lite.cc
# libprotobuf
  ${PROTOBUF_ROOT}/src/google/protobuf/any.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/any.pb.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/api.pb.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/compiler/importer.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/compiler/parser.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/descriptor.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/descriptor.pb.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/descriptor_database.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/duration.pb.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/dynamic_message.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/empty.pb.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/extension_set_heavy.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/field_mask.pb.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/generated_message_reflection.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/io/gzip_stream.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/io/printer.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/io/strtod.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/io/tokenizer.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/io/zero_copy_stream_impl.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/map_field.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/message.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/reflection_ops.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/service.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/source_context.pb.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/struct.pb.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/mathlimits.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/stubs/substitute.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/text_format.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/timestamp.pb.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/type.pb.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/unknown_field_set.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/util/field_comparator.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/util/field_mask_util.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/datapiece.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/default_value_objectwriter.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/error_listener.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/field_mask_utility.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/json_escaping.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/json_objectwriter.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/json_stream_parser.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/object_writer.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/proto_writer.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/protostream_objectsource.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/protostream_objectwriter.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/type_info.cc
#  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/type_info_test_helper.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/util/internal/utility.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/util/json_util.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/util/message_differencer.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/util/time_util.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/util/type_resolver_util.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/wire_format.cc
  ${PROTOBUF_ROOT}/src/google/protobuf/wrappers.pb.cc
)

if(CMAKE_VERSION VERSION_LESS 2.8.9 AND UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

add_library(libprotobuf STATIC ${Protobuf_SRCS})
ocv_include_directories(${PROTOBUF_ROOT}/src)

set_target_properties(libprotobuf
    PROPERTIES
    FOLDER "3rdparty"
    POSITION_INDEPENDENT_CODE 1    # CMake 2.8.9+
    OUTPUT_NAME libprotobuf
    DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
    ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
    )

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(libprotobuf EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()
