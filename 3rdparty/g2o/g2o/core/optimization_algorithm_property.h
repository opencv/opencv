// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, W. Burgard
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef G2O_OPTIMIZATION_ALGORITHM_PROPERTY_H
#define G2O_OPTIMIZATION_ALGORITHM_PROPERTY_H

#include <string>

#include "g2o_core_api.h"

namespace g2o {

/**
 * \brief describe the properties of a solver
 */
struct G2O_CORE_API OptimizationAlgorithmProperty
{
  std::string name;           ///< name of the solver, e.g., var
  std::string desc;           ///< short description of the solver
  std::string type;           ///< type of solver, e.g., "CSparse Cholesky", "PCG"
  bool requiresMarginalize;   ///< whether the solver requires marginalization of landmarks
  int poseDim;                ///< dimension of the pose vertices (-1 if variable)
  int landmarkDim;            ///< dimension of the landmar vertices (-1 if variable)
  OptimizationAlgorithmProperty() :
    name(), desc(), type(), requiresMarginalize(false), poseDim(-1), landmarkDim(-1)
  {
  }
  OptimizationAlgorithmProperty(const std::string& name_, const std::string& desc_, const std::string& type_, bool requiresMarginalize_, int poseDim_, int landmarkDim_) :
    name(name_), desc(desc_), type(type_), requiresMarginalize(requiresMarginalize_), poseDim(poseDim_), landmarkDim(landmarkDim_)
  {
  }
};

} // end namespace

#endif
