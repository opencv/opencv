/*******************************************************************************
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef VAS_COMMON_EXCEPTION_HPP
#define VAS_COMMON_EXCEPTION_HPP

#include <vas/common.hpp>

#include <exception>
#include <stdexcept>

#define ETHROW(condition, exception_class, message, ...)                                                               \
    {                                                                                                                  \
        if (!(condition)) {                                                                                            \
            throw std::exception_class(message);                                                                       \
        }                                                                                                              \
    }

#define TRACE(fmt, ...)

#endif // VAS_COMMON_EXCEPTION_HPP
