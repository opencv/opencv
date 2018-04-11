/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2018, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

namespace cv
{
    class LBFGSSolverImpl : public LBFGSSolver
    {
    public:
        Ptr<Function> getFunction() const;
        void setFunction(const Ptr<Function>& f);
        TermCriteria getTermCriteria() const;
        LBFGSSolverImpl();
        void setTermCriteria(const TermCriteria& termcrit);
        double minimize(InputOutputArray x);

    protected:
        Ptr<MinProblemSolver::Function> _Function;
        TermCriteria _termcrit;
    };

    LBFGSSolverImpl::LBFGSSolverImpl() {
        _Function = Ptr<Function>();
    }

    Ptr<MinProblemSolver::Function> LBFGSSolverImpl::getFunction() const {
        return _Function;
    }

    void LBFGSSolverImpl::setFunction(const Ptr<Function>& f) {
        _Function = f;
    }

    TermCriteria LBFGSSolverImpl::getTermCriteria() const {
        return _termcrit;
    }

    void LBFGSSolverImpl::setTermCriteria(const TermCriteria& termcrit) {
        CV_Assert(
            (termcrit.type == (TermCriteria::MAX_ITER + TermCriteria::EPS) && termcrit.epsilon > 0 && termcrit.maxCount > 0) ||
            ((termcrit.type == TermCriteria::MAX_ITER) && termcrit.maxCount > 0)
        );
        _termcrit = termcrit;
    }

    double LBFGSSolverImpl::minimize(InputOutputArray) {
        CV_Assert(_Function.empty() == false);
        // TODO
        return 0;
    }

    Ptr<LBFGSSolver> LBFGSSolver::create(const Ptr<MinProblemSolver::Function>& f, TermCriteria termcrit) {
        Ptr<LBFGSSolver> bfgs = makePtr<LBFGSSolverImpl>();
        bfgs->setFunction(f);
        bfgs->setTermCriteria(termcrit);
        return bfgs;
    }
};
