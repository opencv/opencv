# Search packages for host system instead of packages for target system
# in case of cross compilation thess macro should be defined by toolchain file
if(NOT COMMAND find_host_package)
    macro(find_host_package)
        find_package(${ARGN})
    endmacro()
endif()
if(NOT COMMAND find_host_program)
    macro(find_host_program)
        find_program(${ARGN})
    endmacro()
endif()


# Provides an option that the user can optionally select.
# Can accept condition to control when option is available for user.
# Usage:
#   option(<option_variable> "help string describing option" <initial value> [IF <condition>])
macro(OCV_OPTION variable description value)
    SET(__condition ${ARGN})
    if("${__condition}" STREQUAL "")
        SET(__condition 1)
    endif()
    list(REMOVE_ITEM __condition "IF" "if")
    if(${__condition})
        OPTION(${variable} "${description}" ${value})
    else()
        UNSET(${variable} CACHE)
    endif()
    UNSET(__condition)
endmacro()


# Macros that checks if module have been installed.
# After it adds module to build and define
# constants passed as second arg
macro(CHECK_MODULE module_name define)
    set(${define} 0)
    if(PKG_CONFIG_FOUND)
        set(ALIAS               ALIASOF_${module_name})
        set(ALIAS_FOUND                 ${ALIAS}_FOUND)
        set(ALIAS_INCLUDE_DIRS   ${ALIAS}_INCLUDE_DIRS)
        set(ALIAS_LIBRARY_DIRS   ${ALIAS}_LIBRARY_DIRS)
        set(ALIAS_LIBRARIES         ${ALIAS}_LIBRARIES)

        PKG_CHECK_MODULES(${ALIAS} ${module_name})

        if (${ALIAS_FOUND})
            set(${define} 1)
            foreach(P "${ALIAS_INCLUDE_DIRS}")
                if (${P})
                    list(APPEND HIGHGUI_INCLUDE_DIRS ${${P}})
                endif()
            endforeach()

            foreach(P "${ALIAS_LIBRARY_DIRS}")
                if (${P})
                    list(APPEND HIGHGUI_LIBRARY_DIRS ${${P}})
                endif()
            endforeach()

            list(APPEND HIGHGUI_LIBRARIES ${${ALIAS_LIBRARIES}})
        endif()
    endif()
endmacro()

# Status report macro.
# Automatically align right column and selects text based on condition.
# Usage:
#   status(<text>)
#   status(<heading> <value1> [<value2> ...])
#   status(<heading> <condition> THEN <text for TRUE> ELSE <text for FALSE> )
macro(status text)
    SET(status_cond)
    SET(status_then)
    SET(status_else)

    SET(status_current_name "cond")
    foreach(arg ${ARGN})
       if(arg STREQUAL "THEN")
           SET(status_current_name "then")
       elseif(arg STREQUAL "ELSE")
           SET(status_current_name "else")
       else()
           LIST(APPEND status_${status_current_name} ${arg})
       endif()
    endforeach()

    if(DEFINED status_cond)
        SET(status_placeholder_length 32)
        string(RANDOM LENGTH ${status_placeholder_length} ALPHABET " " status_placeholder)
        string(LENGTH "${text}" status_text_length)
        if (status_text_length LESS status_placeholder_length)
            string(SUBSTRING "${text}${status_placeholder}" 0 ${status_placeholder_length} status_text)
        elseif (DEFINED status_then OR DEFINED status_else)
            message(STATUS "${text}")
            SET(status_text "${status_placeholder}")
        else()
            SET(status_text "${text}")
        endif()

        if (DEFINED status_then OR DEFINED status_else)
            if(${status_cond})
                string(REPLACE ";" " " status_then "${status_then}")
                message(STATUS "${status_text}" "${status_then}")
            else()
                string(REPLACE ";" " " status_else "${status_else}")
                message(STATUS "${status_text}" "${status_else}")
            endif()
        else()
            string(REPLACE ";" " " status_cond "${status_cond}")
            message(STATUS "${status_text}" "${status_cond}")
        endif()
     else()
         message(STATUS "${text}")
     endif()
endmacro()

