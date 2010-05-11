/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#ifndef SIMPLEX_DOWNHILL_H
#define SIMPLEX_DOWNHILL_H

namespace flann
{

/**
    Adds val to array vals (and point to array points) and keeping the arrays sorted by vals.
*/
template <typename T>
void addValue(int pos, float val, float* vals, T* point, T* points, int n)
{
    vals[pos] = val;
    for (int i=0;i<n;++i) {
        points[pos*n+i] = point[i];
    }

    // bubble down
    int j=pos;
    while (j>0 && vals[j]<vals[j-1]) {
        swap(vals[j],vals[j-1]);
        for (int i=0;i<n;++i) {
            swap(points[j*n+i],points[(j-1)*n+i]);
        }
        --j;
    }
}


/**
    Simplex downhill optimization function.
    Preconditions: points is a 2D mattrix of size (n+1) x n
                    func is the cost function taking n an array of n params and returning float
                    vals is the cost function in the n+1 simplex points, if NULL it will be computed

    Postcondition: returns optimum value and points[0..n] are the optimum parameters
*/
template <typename T, typename F>
float optimizeSimplexDownhill(T* points, int n, F func, float* vals = NULL )
{
    const int MAX_ITERATIONS = 10;

    assert(n>0);

    T* p_o = new T[n];
    T* p_r = new T[n];
    T* p_e = new T[n];

    int alpha = 1;

    int iterations = 0;

    bool ownVals = false;
    if (vals == NULL) {
        ownVals = true;
        vals = new float[n+1];
        for (int i=0;i<n+1;++i) {
            float val = func(points+i*n);
            addValue(i, val, vals, points+i*n, points, n);
        }
    }
    int nn = n*n;

    while (true) {

        if (iterations++ > MAX_ITERATIONS) break;

        // compute average of simplex points (except the highest point)
        for (int j=0;j<n;++j) {
            p_o[j] = 0;
            for (int i=0;i<n;++i) {
                p_o[i] += points[j*n+i];
            }
        }
        for (int i=0;i<n;++i) {
            p_o[i] /= n;
        }

        bool converged = true;
        for (int i=0;i<n;++i) {
            if (p_o[i] != points[nn+i]) {
                converged = false;
            }
        }
        if (converged) break;

        // trying a reflection
        for (int i=0;i<n;++i) {
            p_r[i] = p_o[i] + alpha*(p_o[i]-points[nn+i]);
        }
        float val_r = func(p_r);

        if (val_r>=vals[0] && val_r<vals[n]) {
            // reflection between second highest and lowest
            // add it to the simplex
            logger.info("Choosing reflection\n");
            addValue(n, val_r,vals, p_r, points, n);
            continue;
        }

        if (val_r<vals[0]) {
            // value is smaller than smalest in simplex

            // expand some more to see if it drops further
            for (int i=0;i<n;++i) {
                p_e[i] = 2*p_r[i]-p_o[i];
            }
            float val_e = func(p_e);

            if (val_e<val_r) {
                logger.info("Choosing reflection and expansion\n");
                addValue(n, val_e,vals,p_e,points,n);
            }
            else {
                logger.info("Choosing reflection\n");
                addValue(n, val_r,vals,p_r,points,n);
            }
            continue;
        }
        if (val_r>=vals[n]) {
            for (int i=0;i<n;++i) {
                p_e[i] = (p_o[i]+points[nn+i])/2;
            }
            float val_e = func(p_e);

            if (val_e<vals[n]) {
                logger.info("Choosing contraction\n");
                addValue(n,val_e,vals,p_e,points,n);
                continue;
            }
        }
        {
          logger.info("Full contraction\n");
            for (int j=1;j<=n;++j) {
                for (int i=0;i<n;++i) {
                    points[j*n+i] = (points[j*n+i]+points[i])/2;
                }
                float val = func(points+j*n);
                addValue(j,val,vals,points+j*n,points,n);
            }
        }
    }

    float bestVal = vals[0];

    delete[] p_r;
    delete[] p_o;
    delete[] p_e;
    if (ownVals) delete[] vals;

    return bestVal;
}

}

#endif //SIMPLEX_DOWNHILL_H
