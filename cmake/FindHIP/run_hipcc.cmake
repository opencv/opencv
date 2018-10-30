###############################################################################
# Runs commands using HIPCC
###############################################################################

###############################################################################
# This file runs the hipcc commands to produce the desired output file
# along with the dependency file needed by CMake to compute dependencies.
#
# Input variables:
#
# verbose:BOOL=<>               OFF: Be as quiet as possible (default)
#                               ON : Describe each step
# build_configuration:STRING=<> Build configuration. Defaults to Debug.
# generated_file:STRING=<>      File to generate. Mandatory argument.

if(NOT build_configuration)
    set(build_configuration Debug)
endif()
if(NOT generated_file)
    message(FATAL_ERROR "You must specify generated_file on the command line")
endif()

# Set these up as variables to make reading the generated file easier
set(HIP_HIPCC_EXECUTABLE "@HIP_HIPCC_EXECUTABLE@") # path
set(HIP_HIPCONFIG_EXECUTABLE "@HIP_HIPCONFIG_EXECUTABLE@") #path
set(HIP_HOST_COMPILER "@HIP_HOST_COMPILER@") # path
set(CMAKE_COMMAND "@CMAKE_COMMAND@") # path
set(HIP_run_make2cmake "@HIP_run_make2cmake@") # path
set(HCC_HOME "@HCC_HOME@") #path

@HIP_HOST_FLAGS@
@_HIP_HIPCC_FLAGS@
@_HIP_HCC_FLAGS@
@_HIP_NVCC_FLAGS@
set(HIP_HIPCC_INCLUDE_ARGS "@HIP_HIPCC_INCLUDE_ARGS@") # list (needs to be in quotes to handle spaces properly)

set(cmake_dependency_file "@cmake_dependency_file@") # path
set(source_file "@source_file@") # path
set(host_flag "@host_flag@") # bool

# Determine compiler and compiler flags
execute_process(COMMAND ${HIP_HIPCONFIG_EXECUTABLE} --platform OUTPUT_VARIABLE HIP_PLATFORM OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT host_flag)
    set(__CC ${HIP_HIPCC_EXECUTABLE})
    if(HIP_PLATFORM STREQUAL "hcc")
        if(NOT "x${HCC_HOME}" STREQUAL "x")
            set(ENV{HCC_HOME} ${HCC_HOME})
        endif()
        set(__CC_FLAGS ${HIP_HIPCC_FLAGS} ${HIP_HCC_FLAGS} ${HIP_HIPCC_FLAGS_${build_configuration}} ${HIP_HCC_FLAGS_${build_configuration}})
    else()
        set(__CC_FLAGS ${HIP_HIPCC_FLAGS} ${HIP_NVCC_FLAGS} ${HIP_HIPCC_FLAGS_${build_configuration}} ${HIP_NVCC_FLAGS_${build_configuration}})
    endif()
else()
    set(__CC ${HIP_HOST_COMPILER})
    set(__CC_FLAGS ${CMAKE_HOST_FLAGS} ${CMAKE_HOST_FLAGS_${build_configuration}})
endif()
set(__CC_INCLUDES ${HIP_HIPCC_INCLUDE_ARGS})

# hip_execute_process - Executes a command with optional command echo and status message.
#   status     - Status message to print if verbose is true
#   command    - COMMAND argument from the usual execute_process argument structure
#   ARGN       - Remaining arguments are the command with arguments
#   HIP_result - Return value from running the command
macro(hip_execute_process status command)
    set(_command ${command})
    if(NOT "x${_command}" STREQUAL "xCOMMAND")
        message(FATAL_ERROR "Malformed call to hip_execute_process.  Missing COMMAND as second argument. (command = ${command})")
    endif()
    if(verbose)
        execute_process(COMMAND "${CMAKE_COMMAND}" -E echo -- ${status})
        # Build command string to print
        set(hip_execute_process_string)
        foreach(arg ${ARGN})
            # Escape quotes if any
            string(REPLACE "\"" "\\\"" arg ${arg})
            # Surround args with spaces with quotes
            if(arg MATCHES " ")
                list(APPEND hip_execute_process_string "\"${arg}\"")
            else()
                list(APPEND hip_execute_process_string ${arg})
            endif()
        endforeach()
        # Echo the command
        execute_process(COMMAND ${CMAKE_COMMAND} -E echo ${hip_execute_process_string})
    endif()
    # Run the command
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE HIP_result)
endmacro()

# Delete the target file
hip_execute_process(
    "Removing ${generated_file}"
    COMMAND "${CMAKE_COMMAND}" -E remove "${generated_file}"
    )

# Generate the dependency file
hip_execute_process(
    "Generating dependency file: ${cmake_dependency_file}.pre"
    COMMAND "${__CC}"
    -M
    "${source_file}"
    -o "${cmake_dependency_file}.pre"
    ${__CC_FLAGS}
    ${__CC_INCLUDES}
    )

if(HIP_result)
    message(FATAL_ERROR "Error generating ${generated_file}")
endif()

# Generate the cmake readable dependency file to a temp file
hip_execute_process(
    "Generating temporary cmake readable file: ${cmake_dependency_file}.tmp"
    COMMAND "${CMAKE_COMMAND}"
    -D "input_file:FILEPATH=${cmake_dependency_file}.pre"
    -D "output_file:FILEPATH=${cmake_dependency_file}.tmp"
    -D "verbose=${verbose}"
    -P "${HIP_run_make2cmake}"
    )

if(HIP_result)
    message(FATAL_ERROR "Error generating ${generated_file}")
endif()

# Copy the file if it is different
hip_execute_process(
    "Copy if different ${cmake_dependency_file}.tmp to ${cmake_dependency_file}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${cmake_dependency_file}.tmp" "${cmake_dependency_file}"
    )

if(HIP_result)
    message(FATAL_ERROR "Error generating ${generated_file}")
endif()

# Delete the temporary file
hip_execute_process(
    "Removing ${cmake_dependency_file}.tmp and ${cmake_dependency_file}.pre"
    COMMAND "${CMAKE_COMMAND}" -E remove "${cmake_dependency_file}.tmp" "${cmake_dependency_file}.pre"
    )

if(HIP_result)
    message(FATAL_ERROR "Error generating ${generated_file}")
endif()

# Generate the output file
hip_execute_process(
    "Generating ${generated_file}"
    COMMAND "${__CC}"
    -c
    "${source_file}"
    -o "${generated_file}"
    ${__CC_FLAGS}
    ${__CC_INCLUDES}
    )

if(HIP_result)
    # Make sure that we delete the output file
    hip_execute_process(
        "Removing ${generated_file}"
        COMMAND "${CMAKE_COMMAND}" -E remove "${generated_file}"
        )
    message(FATAL_ERROR "Error generating file ${generated_file}")
else()
    if(verbose)
        message("Generated ${generated_file} successfully.")
    endif()
endif()
# vim: ts=4:sw=4:expandtab:smartindent
