# --- Wayland ---
macro(ocv_wayland_generate protocol_file output_file)
    add_custom_command(OUTPUT ${output_file}.h
            COMMAND ${WAYLAND_SCANNER_EXECUTABLE} client-header < ${protocol_file} > ${output_file}.h
            DEPENDS ${protocol_file})
    add_custom_command(OUTPUT ${output_file}.c
            COMMAND ${WAYLAND_SCANNER_EXECUTABLE} private-code < ${protocol_file} > ${output_file}.c
            DEPENDS ${protocol_file})
    list(APPEND WAYLAND_PROTOCOL_SOURCES ${output_file}.h ${output_file}.c)
endmacro()

ocv_clear_vars(HAVE_WAYLAND_CLIENT HAVE_WAYLAND_CURSOR HAVE_XKBCOMMON HAVE_WAYLAND_PROTOCOLS HAVE_WAYLAND_EGL)
if(WITH_WAYLAND)
    ocv_check_modules(WAYLAND_CLIENT wayland-client)
    if(WAYLAND_CLIENT_FOUND)
        set(HAVE_WAYLAND_CLIENT ON)
    endif()
    ocv_check_modules(WAYLAND_CURSOR wayland-cursor)
    if(WAYLAND_CURSOR_FOUND)
        set(HAVE_WAYLAND_CURSOR ON)
    endif()
    ocv_check_modules(XKBCOMMON xkbcommon)
    if(XKBCOMMON_FOUND)
        set(HAVE_XKBCOMMON ON)
    endif()
    ocv_check_modules(WAYLAND_PROTOCOLS wayland-protocols>=1.13)
    if(HAVE_WAYLAND_PROTOCOLS)
        pkg_get_variable(WAYLAND_PROTOCOLS_BASE wayland-protocols pkgdatadir)
        find_host_program(WAYLAND_SCANNER_EXECUTABLE NAMES wayland-scanner REQUIRED)
    endif()

    if(HAVE_WAYLAND_CLIENT AND HAVE_WAYLAND_CURSOR AND HAVE_XKBCOMMON AND HAVE_WAYLAND_PROTOCOLS)
        set(HAVE_WAYLAND TRUE)
    endif()

    # WAYLAND_EGL is option
    ocv_check_modules(WAYLAND_EGL wayland-egl)
    if(WAYLAND_EGL_FOUND)
        set(HAVE_WAYLAND_EGL ON)
    endif()
endif()
