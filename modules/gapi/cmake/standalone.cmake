if (NOT TARGET ade )
  find_package(ade 0.1.0 REQUIRED)
endif()

set(FLUID_TARGET fluid)
set(FLUID_ROOT "${CMAKE_CURRENT_LIST_DIR}/../")

file(GLOB FLUID_includes "${FLUID_ROOT}/include/opencv2/*.hpp"
                         "${FLUID_ROOT}/include/opencv2/gapi/g*.hpp"
                         "${FLUID_ROOT}/include/opencv2/gapi/util/*.hpp"
                         "${FLUID_ROOT}/include/opencv2/gapi/own/*.hpp"
                         "${FLUID_ROOT}/include/opencv2/gapi/fluid/*.hpp")
file(GLOB FLUID_sources  "${FLUID_ROOT}/src/api/g*.cpp"
                         "${FLUID_ROOT}/src/compiler/*.cpp"
                         "${FLUID_ROOT}/src/compiler/passes/*.cpp"
                         "${FLUID_ROOT}/src/executor/*.cpp"
                         "${FLUID_ROOT}/src/backends/fluid/*.cpp"
                         "${FLUID_ROOT}/src/backends/common/*.cpp")

add_library(${FLUID_TARGET} STATIC ${FLUID_includes} ${FLUID_sources})

target_include_directories(${FLUID_TARGET}
  PUBLIC          $<BUILD_INTERFACE:${FLUID_ROOT}/include>
  PRIVATE         ${FLUID_ROOT}/src)

target_compile_definitions(${FLUID_TARGET} PUBLIC -DGAPI_STANDALONE
# This preprocessor definition resolves symbol clash when
# standalone fluid meets gapi ocv module in one application
                                           PUBLIC cv=fluidcv)

set_target_properties(${FLUID_TARGET} PROPERTIES POSITION_INDEPENDENT_CODE True)
set_property(TARGET ${FLUID_TARGET} PROPERTY CXX_STANDARD 11)

target_link_libraries(${FLUID_TARGET} PRIVATE ade)
