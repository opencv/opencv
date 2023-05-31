#COPYRIGHT
#
#All contributions by the University of California:
#Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
#All rights reserved.
#
#All other contributions:
#Copyright (c) 2014, 2015, the respective contributors
#All rights reserved.
#
#Caffe uses a shared copyright model: each contributor holds copyright over
#their contributions to Caffe. The project versioning records all such
#contribution and copyright details. If a contributor wants to further mark
#their specific copyright on a particular contribution, they should indicate
#their copyright solely in the commit message of the change when it is
#committed.
#
#LICENSE
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
#ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#CONTRIBUTION AGREEMENT
#
#By contributing to the BVLC/caffe repository through pull-request, comment,
#or otherwise, the contributor releases their content to the
#license and copyright terms herein.


# Find the Atlas (and Lapack) libraries
#
# The following variables are optionally searched for defaults
#  Atlas_ROOT_DIR:            Base directory where all Atlas components are found
#
# The following are set after configuration is done:
#  Atlas_FOUND
#  Atlas_INCLUDE_DIRS
#  Atlas_LIBRARIES
#  Atlas_LIBRARYRARY_DIRS

set(Atlas_INCLUDE_SEARCH_PATHS
  /usr/include/atlas
  /usr/include/atlas-base
  $ENV{Atlas_ROOT_DIR}
  $ENV{Atlas_ROOT_DIR}/include
)

set(Atlas_LIB_SEARCH_PATHS
  /usr/lib/atlas
  /usr/lib/atlas-base
  $ENV{Atlas_ROOT_DIR}
  $ENV{Atlas_ROOT_DIR}/lib
)

find_path(Atlas_CBLAS_INCLUDE_DIR   NAMES cblas.h   PATHS ${Atlas_INCLUDE_SEARCH_PATHS})
find_path(Atlas_CLAPACK_INCLUDE_DIR NAMES lapacke.h PATHS ${Atlas_INCLUDE_SEARCH_PATHS})

find_library(Atlas_CBLAS_LIBRARY NAMES  ptcblas_r ptcblas cblas_r cblas PATHS ${Atlas_LIB_SEARCH_PATHS})
find_library(Atlas_BLAS_LIBRARY NAMES   atlas_r   atlas                 PATHS ${Atlas_LIB_SEARCH_PATHS})
find_library(Atlas_LAPACK_LIBRARY NAMES lapack alapack_r alapack lapack_atlas  PATHS ${Atlas_LIB_SEARCH_PATHS})

set(LOOKED_FOR
  Atlas_CBLAS_INCLUDE_DIR
  Atlas_CLAPACK_INCLUDE_DIR

  Atlas_CBLAS_LIBRARY
  Atlas_BLAS_LIBRARY
  Atlas_LAPACK_LIBRARY
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Atlas DEFAULT_MSG ${LOOKED_FOR})

if(ATLAS_FOUND)
  set(Atlas_INCLUDE_DIR ${Atlas_CBLAS_INCLUDE_DIR} ${Atlas_CLAPACK_INCLUDE_DIR})
  set(Atlas_LIBRARIES ${Atlas_LAPACK_LIBRARY} ${Atlas_CBLAS_LIBRARY} ${Atlas_BLAS_LIBRARY})
  mark_as_advanced(${LOOKED_FOR})

  message(STATUS "Found Atlas (include: ${Atlas_CBLAS_INCLUDE_DIR}, library: ${Atlas_BLAS_LIBRARY})")
endif(ATLAS_FOUND)