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

#include "batch_stats.h"
#include <cstring>

namespace g2o {
  using namespace std;

  G2OBatchStatistics* G2OBatchStatistics::_globalStats=0;

  #ifndef PTHING
  #define PTHING(s) \
    #s << "= " << (st.s) << "\t "
  #endif

  G2OBatchStatistics::G2OBatchStatistics(){
    // zero all.
    memset (this, 0, sizeof(G2OBatchStatistics));

    // set the iteration to -1 to show that it isn't valid
    iteration = -1;
  }

  std::ostream& operator << (std::ostream& os , const G2OBatchStatistics& st)
  {
    os << PTHING(iteration);

    os << PTHING( numVertices ); // how many vertices are involved
    os << PTHING( numEdges ); // hoe many edges
    os << PTHING(  chi2 );  // total chi2
    
    /** timings **/
    // nonlinear part
    os << PTHING(  timeResiduals );  
    os << PTHING(  timeLinearize );   // jacobians
    os << PTHING(  timeQuadraticForm ); // construct the quadratic form in the graph
    
    // block_solver (constructs Ax=b, plus maybe schur);
    os << PTHING(  timeSchurComplement ); // compute schur complement (0 if not done);
    
    // linear solver (computes Ax=b); );
    os << PTHING(  timeSymbolicDecomposition ); // symbolic decomposition (0 if not done);
    os << PTHING(  timeNumericDecomposition ); // numeric decomposition  (0 if not done);
    os << PTHING(  timeLinearSolution );             // total time for solving Ax=b
    os << PTHING(  iterationsLinearSolver );  // iterations of PCG
    os << PTHING(  timeUpdate ); // oplus
    os << PTHING(  timeIteration ); // total time );

    os << PTHING( levenbergIterations );
    os << PTHING( timeLinearSolver);

    os << PTHING(hessianDimension);
    os << PTHING(hessianPoseDimension);
    os << PTHING(hessianLandmarkDimension);
    os << PTHING(choleskyNNZ);
    os << PTHING(timeMarginals);

    return os;
  };

  void G2OBatchStatistics::setGlobalStats(G2OBatchStatistics* b)
  {
    _globalStats = b;
  }

} // end namespace
