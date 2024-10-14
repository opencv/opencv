# detect-install-dirs.cmake -- Detect install directory parameters
# Copyright (C) 2021 Hans Kristian Rosbach
# Licensed under the Zlib license, see LICENSE.md for details

# Determine installation directory for executables
if (DEFINED BIN_INSTALL_DIR)
    set(BIN_INSTALL_DIR "${BIN_INSTALL_DIR}" CACHE PATH "Installation directory for executables (Deprecated)" FORCE)
    set(CMAKE_INSTALL_BINDIR "${BIN_INSTALL_DIR}")
elseif (DEFINED INSTALL_BIN_DIR)
    set(CMAKE_INSTALL_BINDIR "${INSTALL_BIN_DIR}")
endif()

# Determine installation directory for libraries
if (DEFINED LIB_INSTALL_DIR)
    set(LIB_INSTALL_DIR "${LIB_INSTALL_DIR}" CACHE PATH "Installation directory for libraries (Deprecated)" FORCE)
    set(CMAKE_INSTALL_LIBDIR "${LIB_INSTALL_DIR}")
elseif (DEFINED INSTALL_LIB_DIR)
    set(CMAKE_INSTALL_LIBDIR "${INSTALL_LIB_DIR}")
endif()

# Determine installation directory for include files
if (DEFINED INC_INSTALL_DIR)
    set(INC_INSTALL_DIR "${INC_INSTALL_DIR}" CACHE PATH "Installation directory for headers (Deprecated)" FORCE)
    set(CMAKE_INSTALL_INCLUDEDIR "${INC_INSTALL_DIR}")
elseif (DEFINED INSTALL_INC_DIR)
    set(CMAKE_INSTALL_INCLUDEDIR "${INSTALL_INC_DIR}")
endif()

# Define GNU standard installation directories
include(GNUInstallDirs)

# Determine installation directory for pkgconfig files
if (DEFINED PKGCONFIG_INSTALL_DIR)
    set(PKGCONFIG_INSTALL_DIR "${PKGCONFIG_INSTALL_DIR}" CACHE PATH "Installation directory for pkgconfig (.pc) files" FORCE)
elseif (DEFINED INSTALL_PKGCONFIG_DIR)
    set(PKGCONFIG_INSTALL_DIR "${INSTALL_PKGCONFIG_DIR}" CACHE PATH "Installation directory for pkgconfig (.pc) files" FORCE)
elseif (DEFINED CMAKE_INSTALL_PKGCONFIGDIR)
    set(PKGCONFIG_INSTALL_DIR "${CMAKE_INSTALL_PKGCONFIGDIR}" CACHE PATH "Installation directory for pkgconfig (.pc) files" FORCE)
elseif (DEFINED CMAKE_INSTALL_FULL_PKGCONFIGDIR)
    set(PKGCONFIG_INSTALL_DIR "${CMAKE_INSTALL_FULL_PKGCONFIGDIR}" CACHE PATH "Installation directory for pkgconfig (.pc) files" FORCE)
else()
    set(PKGCONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/pkgconfig" CACHE PATH "Installation directory for pkgconfig (.pc) files")
endif()
