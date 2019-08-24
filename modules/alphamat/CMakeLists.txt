cmake_minimum_required(VERSION 2.6)

set(the_description "Information Flow alphamatting")

include_directories(./include)

option (WITH_TESTS "build tests" ON)

if (WITH_TESTS)
  add_subdirectory(test)
endif()
