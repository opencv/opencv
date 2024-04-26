// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

/*
    Partially based on Yossi Rubner code:
    =========================================================================
    emd.c

    Last update: 3/14/98

    An implementation of the Earth Movers Distance.
    Based of the solution for the Transportation problem as described in
    "Introduction to Mathematical Programming" by F. S. Hillier and
    G. J. Lieberman, McGraw-Hill, 1990.

    Copyright (C) 1998 Yossi Rubner
    Computer Science Department, Stanford University
    E-Mail: rubner@cs.stanford.edu   URL: http://vision.stanford.edu/~rubner
    ==========================================================================
*/

#include "precomp.hpp"

using namespace cv;

namespace {


//==============================================================================
// Distance functions

typedef float (*DistFunc)(const float* a, const float* b, int dims);

static float distL1(const float* x, const float* y, int dims)
{
    double s = 0;
    for (int i = 0; i < dims; i++)
    {
        const double t = x[i] - y[i];
        s += fabs(t);
    }
    return (float)s;
}

static float distL2(const float* x, const float* y, int dims)
{
    double s = 0;
    for (int i = 0; i < dims; i++)
    {
        const double t = x[i] - y[i];
        s += t * t;
    }
    return sqrt((float)s);
}

static float distC(const float* x, const float* y, int dims)
{
    double s = 0;
    for (int i = 0; i < dims; i++)
    {
        const double t = fabs(x[i] - y[i]);
        if (s < t)
            s = t;
    }
    return (float)s;
}


//==============================================================================
// Data structures

/* Node1D is used for lists, representing 1D sparse array */
struct Node1D
{
    float val;
    Node1D* next;
};

/* Node2D is used for lists, representing 2D sparse matrix */
struct Node2D
{
    float val;
    int i, j;
    Node2D* next[2]; /* next row & next column */
};


//==============================================================================
// Main class

struct EMDSolver
{
    static constexpr int MAX_ITERATIONS = 500;
    static constexpr float CV_EMD_INF = 1e20f;
    static constexpr float CV_EMD_EPS = 1e-5f;

    int ssize, dsize;

    float* cost_buf;
    Node2D* data_x;
    Node2D* end_x;
    Node2D* enter_x;
    char* is_x;

    Node2D** rows_x;
    Node2D** cols_x;

    Node1D* u;
    Node1D* v;

    int* idx1;
    int* idx2;

    /* find_loop buffers */
    Node2D** loop;
    char* is_used;

    /* russel buffers */
    float* s;
    float* d;
    float* delta;

    float weight, max_cost;

    utils::BufferArea area, area2;

public:
    float getWeight() const
    {
        return weight;
    }

    float& getCost(int i, int j)
    {
        return *(this->cost_buf + i * dsize + j);
    }
    const float& getCost(int i, int j) const
    {
        return *(this->cost_buf + i * dsize + j);
    }
    char& getIsX(int i, int j)
    {
        return *(this->is_x + i * dsize + j);
    }
    const char& getIsX(int i, int j) const
    {
        return *(this->is_x + i * dsize + j);
    }

    EMDSolver() :
        ssize(0), dsize(0), cost_buf(0), data_x(0), end_x(0), enter_x(0), is_x(0), rows_x(0),
        cols_x(0), u(0), v(0), idx1(0), idx2(0), loop(0), is_used(0), s(0), d(0), delta(0),
        weight(0), max_cost(0)
    {
    }

public:
    bool init(const Mat& sign1,
              const Mat& sign2,
              int dims,
              DistFunc dfunc,
              const Mat& cost,
              float* lowerBound);
    bool checkLowerBound(const Mat& sign1,
                         const Mat& sign2,
                         int dims,
                         DistFunc dfunc,
                         float& lowerBound);
    bool calcSums(const Mat& sign1, const Mat& sign2);
    float calcCost(const Mat& sign1, const Mat& sign2, int dims, DistFunc dfunc, const Mat& cost);
    void solve();
    double calcFlow(Mat* flow_) const;
    int findBasicVars() const;
    float checkOptimal() const;
    void callRussel();
    bool checkNewSolution();
    int findLoop() const;
    void addBasicVar(int min_i,
                     int min_j,
                     Node1D* prev_u_min_i,
                     Node1D* prev_v_min_j,
                     Node1D* u_head);
};


//==============================================================================
// Implementations

bool EMDSolver::init(const Mat& sign1,
                     const Mat& sign2,
                     int dims,
                     DistFunc dfunc,
                     const Mat& cost,
                     float* lowerBound)
{
    const int size1 = sign1.size().height;
    const int size2 = sign2.size().height;

    area.allocate(this->idx1, size1 + 1);
    area.allocate(this->idx2, size2 + 1);
    area.allocate(this->s, size1 + 1);
    area.allocate(this->d, size2 + 1);
    area.commit();
    area.zeroFill();

    const bool areSumsEqual = calcSums(sign1, sign2);
    if (areSumsEqual && lowerBound)
    {
        if (checkLowerBound(sign1, sign2, dims, dfunc, *lowerBound))
            return false;
    }

    area2.allocate(this->u, ssize, 64);
    area2.allocate(this->v, dsize, 64);
    area2.allocate(this->is_used, ssize + dsize);
    area2.allocate(this->delta, ssize * dsize);
    area2.allocate(this->cost_buf, ssize * dsize);
    area2.allocate(this->is_x, ssize * dsize);
    area2.allocate(this->data_x, ssize + dsize, 64);
    area2.allocate(this->rows_x, ssize, 64);
    area2.allocate(this->cols_x, dsize, 64);
    area2.allocate(this->loop, ssize + dsize + 1, 64);
    area2.commit();
    area2.zeroFill();

    this->end_x = this->data_x;
    this->max_cost = calcCost(sign1, sign2, dims, dfunc, cost);
    callRussel();
    this->enter_x = (this->end_x)++;
    return true;
}


bool EMDSolver::checkLowerBound(const Mat& sign1,
                                const Mat& sign2,
                                int dims,
                                DistFunc dfunc,
                                float& lowerBound)
{
    AutoBuffer<float> buf;
    buf.allocate(dims * 2);
    memset(buf.data(), 0, dims * 2 * sizeof(float));

    float* xs = buf.data();
    float* xd = buf.data() + dims;

    for (int j = 0; j < sign1.rows; ++j)
    {
        const float weight_ = sign1.at<float>(j, 0);
        for (int i = 0; i < dims; i++)
            xs[i] += sign1.at<float>(j, i + 1) * weight_;
    }

    for (int j = 0; j < sign2.rows; ++j)
    {
        const float weight_ = sign2.at<float>(j, 0);
        for (int i = 0; i < dims; i++)
            xd[i] += sign2.at<float>(j, i + 1) * weight_;
    }

    const float lb = dfunc(xs, xd, dims) / this->weight;
    const bool result = (lowerBound <= lb);
    lowerBound = lb;
    return result;
}


// return true if total sums of signatures are equal, false otherwise
bool EMDSolver::calcSums(const Mat& sign1, const Mat& sign2)
{
    bool result = true;
    /* sum up the supply and demand */
    int ssize_ = 0, dsize_ = 0;
    float s_sum = 0, d_sum = 0, diff;
    for (int i = 0; i < sign1.size().height; i++)
    {
        const float weight_ = sign1.at<float>(i, 0);

        if (weight_ > 0)
        {
            s_sum += weight_;
            this->s[ssize_] = weight_;
            this->idx1[ssize_++] = i;
        }
        else if (weight_ < 0)
            CV_Error(cv::Error::StsBadArg, "sign1 must not contain negative weights");
    }

    for (int i = 0; i < sign2.size().height; i++)
    {
        const float weight_ = sign2.at<float>(i, 0);

        if (weight_ > 0)
        {
            d_sum += weight_;
            this->d[dsize_] = weight_;
            this->idx2[dsize_++] = i;
        }
        else if (weight_ < 0)
            CV_Error(cv::Error::StsBadArg, "sign2 must not contain negative weights");
    }

    if (ssize_ == 0)
        CV_Error(cv::Error::StsBadArg, "sign1 must contain at least one non-zero value");
    if (dsize_ == 0)
        CV_Error(cv::Error::StsBadArg, "sign2 must contain at least one non-zero value");

    /* if supply different than the demand, add a zero-cost dummy cluster */
    diff = s_sum - d_sum;
    if (fabs(diff) >= CV_EMD_EPS * s_sum)
    {
        result = false;
        if (diff < 0)
        {
            this->s[ssize_] = -diff;
            this->idx1[ssize_++] = -1;
        }
        else
        {
            this->d[dsize_] = diff;
            this->idx2[dsize_++] = -1;
        }
    }

    this->ssize = ssize_;
    this->dsize = dsize_;
    this->weight = s_sum > d_sum ? s_sum : d_sum;
    return result;
}


// returns maximum cost over all possible s->d combinations
float EMDSolver::calcCost(const Mat& sign1,
                          const Mat& sign2,
                          int dims,
                          DistFunc dfunc,
                          const Mat& cost)
{
    if (!dfunc)
    {
        CV_Assert(!cost.empty());
    }
    float result = 0;

    /* compute the distance matrix */
    for (int i = 0; i < ssize; i++)
    {
        const int ci = this->idx1[i];
        if (ci >= 0)
        {
            for (int j = 0; j < dsize; j++)
            {
                const int cj = this->idx2[j];
                if (cj < 0)
                    getCost(i, j) = 0;
                else
                {
                    float val;
                    if (dfunc)
                    {
                        val = dfunc(sign1.ptr<float>(ci, 1), sign2.ptr<float>(cj, 1), dims);
                    }
                    else
                    {
                        val = cost.at<float>(ci, cj);
                    }
                    getCost(i, j) = val;
                    if (result < val)
                        result = val;
                }
            }
        }
        else
        {
            for (int j = 0; j < dsize; j++)
                getCost(i, j) = 0;
        }
    }
    return result;
}


// runs solver iterations
void EMDSolver::solve()
{
    if (ssize > 1 && dsize > 1)
    {
        const float eps = CV_EMD_EPS * max_cost;
        for (int itr = 1; itr < MAX_ITERATIONS; itr++)
        {
            /* find basic variables */
            if (findBasicVars() < 0)
                break;

            /* check for optimality */
            const float min_delta = checkOptimal();

            if (min_delta == CV_EMD_INF)
                CV_Error(cv::Error::StsNoConv, "");

            /* if no negative deltamin, we found the optimal solution */
            if (min_delta >= -eps)
                break;

            /* improve solution */
            if (!checkNewSolution())
                CV_Error(cv::Error::StsNoConv, "");
        }
    }
}

double EMDSolver::calcFlow(Mat* flow_) const
{
    double result = 0.;
    Node2D* xp = 0;
    for (xp = data_x; xp < end_x; xp++)
    {
        float val = xp->val;
        const int i = xp->i;
        const int j = xp->j;

        if (xp == enter_x)
            continue;

        const int ci = idx1[i];
        const int cj = idx2[j];

        if (ci >= 0 && cj >= 0)
        {
            result += (double)val * getCost(i, j);
            if (flow_)
            {
                flow_->at<float>(ci, cj) = val;
            }
        }
    }
    return result;
}


int EMDSolver::findBasicVars() const
{
    int i, j;
    int u_cfound, v_cfound;
    Node1D u0_head, u1_head, *cur_u, *prev_u;
    Node1D v0_head, v1_head, *cur_v, *prev_v;
    bool found;

    CV_Assert(u != 0 && v != 0);

    /* initialize the rows list (u) and the columns list (v) */
    u0_head.next = u;
    for (i = 0; i < ssize; i++)
    {
        u[i].next = u + i + 1;
    }
    u[ssize - 1].next = 0;
    u1_head.next = 0;

    v0_head.next = ssize > 1 ? v + 1 : 0;
    for (i = 1; i < dsize; i++)
    {
        v[i].next = v + i + 1;
    }
    v[dsize - 1].next = 0;
    v1_head.next = 0;

    /* there are ssize+dsize variables but only ssize+dsize-1 independent equations,
       so set v[0]=0 */
    v[0].val = 0;
    v1_head.next = v;
    v1_head.next->next = 0;

    /* loop until all variables are found */
    u_cfound = v_cfound = 0;
    while (u_cfound < ssize || v_cfound < dsize)
    {
        found = false;
        if (v_cfound < dsize)
        {
            /* loop over all marked columns */
            prev_v = &v1_head;
            cur_v = v1_head.next;
            found = found || (cur_v != 0);
            for (; cur_v != 0; cur_v = cur_v->next)
            {
                float cur_v_val = cur_v->val;

                j = (int)(cur_v - v);
                /* find the variables in column j */
                prev_u = &u0_head;
                for (cur_u = u0_head.next; cur_u != 0;)
                {
                    i = (int)(cur_u - u);
                    if (getIsX(i, j))
                    {
                        /* compute u[i] */
                        cur_u->val = getCost(i, j) - cur_v_val;
                        /* ...and add it to the marked list */
                        prev_u->next = cur_u->next;
                        cur_u->next = u1_head.next;
                        u1_head.next = cur_u;
                        cur_u = prev_u->next;
                    }
                    else
                    {
                        prev_u = cur_u;
                        cur_u = cur_u->next;
                    }
                }
                prev_v->next = cur_v->next;
                v_cfound++;
            }
        }

        if (u_cfound < ssize)
        {
            /* loop over all marked rows */
            prev_u = &u1_head;
            cur_u = u1_head.next;
            found = found || (cur_u != 0);
            for (; cur_u != 0; cur_u = cur_u->next)
            {
                float cur_u_val = cur_u->val;
                i = (int)(cur_u - u);
                /* find the variables in rows i */
                prev_v = &v0_head;
                for (cur_v = v0_head.next; cur_v != 0;)
                {
                    j = (int)(cur_v - v);
                    if (getIsX(i, j))
                    {
                        /* compute v[j] */
                        cur_v->val = getCost(i, j) - cur_u_val;
                        /* ...and add it to the marked list */
                        prev_v->next = cur_v->next;
                        cur_v->next = v1_head.next;
                        v1_head.next = cur_v;
                        cur_v = prev_v->next;
                    }
                    else
                    {
                        prev_v = cur_v;
                        cur_v = cur_v->next;
                    }
                }
                prev_u->next = cur_u->next;
                u_cfound++;
            }
        }

        if (!found)
            return -1;
    }

    return 0;
}

float EMDSolver::checkOptimal() const
{
    int i, j, min_i = 0, min_j = 0;

    float min_delta = CV_EMD_INF;
    /* find the minimal cij-ui-vj over all i,j */
    for (i = 0; i < ssize; i++)
    {
        float u_val = u[i].val;
        for (j = 0; j < dsize; j++)
        {
            if (!getIsX(i, j))
            {
                const float delta_ = getCost(i, j) - u_val - v[j].val;
                if (min_delta > delta_)
                {
                    min_delta = delta_;
                    min_i = i;
                    min_j = j;
                }
            }
        }
    }

    enter_x->i = min_i;
    enter_x->j = min_j;

    return min_delta;
}

bool EMDSolver::checkNewSolution()
{
    int i, j;
    float min_val = CV_EMD_INF;
    int steps;
    Node2D head {0, 0, 0, {0, 0}}, *cur_x, *next_x, *leave_x = 0;
    Node2D* enter_x_ = this->enter_x;
    Node2D** loop_ = this->loop;

    /* enter the new basic variable */
    i = enter_x_->i;
    j = enter_x_->j;
    getIsX(i, j) = 1;
    enter_x_->next[0] = this->rows_x[i];
    enter_x->next[1] = this->cols_x[j];
    enter_x_->val = 0;
    this->rows_x[i] = enter_x_;
    this->cols_x[j] = enter_x_;

    /* find a chain reaction */
    steps = findLoop();

    if (steps == 0)
        return false;

    /* find the largest value in the loop */
    for (i = 1; i < steps; i += 2)
    {
        float temp = loop_[i]->val;

        if (min_val > temp)
        {
            leave_x = loop_[i];
            min_val = temp;
        }
    }

    /* update the loop */
    for (i = 0; i < steps; i += 2)
    {
        float temp0 = loop_[i]->val + min_val;
        float temp1 = loop_[i + 1]->val - min_val;

        loop_[i]->val = temp0;
        loop_[i + 1]->val = temp1;
    }

    /* remove the leaving basic variable */
    CV_Assert(leave_x != NULL);
    i = leave_x->i;
    j = leave_x->j;
    getIsX(i, j) = 0;

    head.next[0] = this->rows_x[i];
    cur_x = &head;
    while ((next_x = cur_x->next[0]) != leave_x)
    {
        cur_x = next_x;
        CV_Assert(cur_x);
    }
    cur_x->next[0] = next_x->next[0];
    this->rows_x[i] = head.next[0];

    head.next[1] = this->cols_x[j];
    cur_x = &head;
    while ((next_x = cur_x->next[1]) != leave_x)
    {
        cur_x = next_x;
        CV_Assert(cur_x);
    }
    cur_x->next[1] = next_x->next[1];
    this->cols_x[j] = head.next[1];

    /* set enter_x to be the new empty slot */
    this->enter_x = leave_x;

    return true;
}

int EMDSolver::findLoop() const
{
    int i;

    memset(is_used, 0, this->ssize + this->dsize);

    Node2D* new_x = loop[0] = enter_x;
    is_used[enter_x - data_x] = 1;
    int steps = 1;

    do
    {
        if ((steps & 1) == 1)
        {
            /* find an unused x in the row */
            new_x = this->rows_x[new_x->i];
            while (new_x != 0 && is_used[new_x - data_x])
                new_x = new_x->next[0];
        }
        else
        {
            /* find an unused x in the column, or the entering x */
            new_x = this->cols_x[new_x->j];
            while (new_x != 0 && is_used[new_x - data_x] && new_x != enter_x)
                new_x = new_x->next[1];
            if (new_x == enter_x)
                break;
        }

        if (new_x != 0) /* found the next x */
        {
            /* add x to the loop */
            loop[steps++] = new_x;
            is_used[new_x - data_x] = 1;
        }
        else /* didn't find the next x */
        {
            /* backtrack */
            do
            {
                i = steps & 1;
                new_x = loop[steps - 1];
                do
                {
                    new_x = new_x->next[i];
                }
                while (new_x != 0 && is_used[new_x - data_x]);

                if (new_x == 0)
                {
                    is_used[loop[--steps] - data_x] = 0;
                }
            }
            while (new_x == 0 && steps > 0);

            is_used[loop[steps - 1] - data_x] = 0;
            loop[steps - 1] = new_x;
            is_used[new_x - data_x] = 1;
        }
    }
    while (steps > 0);

    return steps;
}

void EMDSolver::callRussel()
{
    int i, j, min_i = -1, min_j = -1;
    float min_delta, diff;
    Node1D u_head, *cur_u, *prev_u;
    Node1D v_head, *cur_v, *prev_v;
    Node1D *prev_u_min_i = 0, *prev_v_min_j = 0, *remember;
    float eps = CV_EMD_EPS * this->max_cost;

    /* initialize the rows list (ur), and the columns list (vr) */
    u_head.next = u;
    for (i = 0; i < ssize; i++)
    {
        u[i].next = u + i + 1;
    }
    u[ssize - 1].next = 0;

    v_head.next = v;
    for (i = 0; i < dsize; i++)
    {
        v[i].val = -CV_EMD_INF;
        v[i].next = v + i + 1;
    }
    v[dsize - 1].next = 0;

    /* find the maximum row and column values (ur[i] and vr[j]) */
    for (i = 0; i < ssize; i++)
    {
        float u_val = -CV_EMD_INF;
        for (j = 0; j < dsize; j++)
        {
            float temp = getCost(i, j);

            if (u_val < temp)
                u_val = temp;
            if (v[j].val < temp)
                v[j].val = temp;
        }
        u[i].val = u_val;
    }

    /* compute the delta matrix */
    for (i = 0; i < ssize; i++)
    {
        float u_val = u[i].val;
        float* delta_row = delta + i * dsize;
        for (j = 0; j < dsize; j++)
        {
            delta_row[j] = getCost(i, j) - u_val - v[j].val;
        }
    }

    /* find the basic variables */
    do
    {
        /* find the smallest delta[i][j] */
        min_i = -1;
        min_delta = CV_EMD_INF;
        prev_u = &u_head;
        for (cur_u = u_head.next; cur_u != 0; cur_u = cur_u->next)
        {
            i = (int)(cur_u - u);
            float* delta_row = delta + i * dsize;

            prev_v = &v_head;
            for (cur_v = v_head.next; cur_v != 0; cur_v = cur_v->next)
            {
                j = (int)(cur_v - v);
                if (min_delta > delta_row[j])
                {
                    min_delta = delta_row[j];
                    min_i = i;
                    min_j = j;
                    prev_u_min_i = prev_u;
                    prev_v_min_j = prev_v;
                }
                prev_v = cur_v;
            }
            prev_u = cur_u;
        }

        if (min_i < 0)
            break;

        /* add x[min_i][min_j] to the basis, and adjust supplies and cost */
        remember = prev_u_min_i->next;
        addBasicVar(min_i, min_j, prev_u_min_i, prev_v_min_j, &u_head);

        /* update the necessary delta[][] */
        if (remember == prev_u_min_i->next) /* line min_i was deleted */
        {
            for (cur_v = v_head.next; cur_v != 0; cur_v = cur_v->next)
            {
                j = (int)(cur_v - v);
                if (cur_v->val == getCost(min_i, j)) /* column j needs updating */
                {
                    float max_val = -CV_EMD_INF;

                    /* find the new maximum value in the column */
                    for (cur_u = u_head.next; cur_u != 0; cur_u = cur_u->next)
                    {
                        float temp = getCost((int)(cur_u - u), j);

                        if (max_val < temp)
                            max_val = temp;
                    }

                    /* if needed, adjust the relevant delta[*][j] */
                    diff = max_val - cur_v->val;
                    cur_v->val = max_val;
                    if (fabs(diff) < eps)
                    {
                        for (cur_u = u_head.next; cur_u != 0; cur_u = cur_u->next)
                            *(delta + (cur_u - u) * dsize + j) += diff;
                    }
                }
            }
        }
        else /* column min_j was deleted */
        {
            for (cur_u = u_head.next; cur_u != 0; cur_u = cur_u->next)
            {
                i = (int)(cur_u - u);
                if (cur_u->val == getCost(i, min_j)) /* row i needs updating */
                {
                    float max_val = -CV_EMD_INF;

                    /* find the new maximum value in the row */
                    for (cur_v = v_head.next; cur_v != 0; cur_v = cur_v->next)
                    {
                        float temp = getCost(i, (int)(cur_v - v));

                        if (max_val < temp)
                            max_val = temp;
                    }

                    /* if needed, adjust the relevant delta[i][*] */
                    diff = max_val - cur_u->val;
                    cur_u->val = max_val;

                    if (fabs(diff) < eps)
                    {
                        for (cur_v = v_head.next; cur_v != 0; cur_v = cur_v->next)
                            *(delta + i * dsize + (cur_v - v)) += diff;
                    }
                }
            }
        }
    }
    while (u_head.next != 0 || v_head.next != 0);
}

void EMDSolver::addBasicVar(int min_i,
                            int min_j,
                            Node1D* prev_u_min_i,
                            Node1D* prev_v_min_j,
                            Node1D* u_head)
{
    float temp;

    if (this->s[min_i] < this->d[min_j] + this->weight * CV_EMD_EPS)
    { /* supply exhausted */
        temp = this->s[min_i];
        this->s[min_i] = 0;
        this->d[min_j] -= temp;
    }
    else /* demand exhausted */
    {
        temp = this->d[min_j];
        this->d[min_j] = 0;
        this->s[min_i] -= temp;
    }

    /* x(min_i,min_j) is a basic variable */
    getIsX(min_i, min_j) = 1;

    end_x->val = temp;
    end_x->i = min_i;
    end_x->j = min_j;
    end_x->next[0] = this->rows_x[min_i];
    end_x->next[1] = this->cols_x[min_j];
    this->rows_x[min_i] = end_x;
    this->cols_x[min_j] = end_x;
    this->end_x = end_x + 1;

    /* delete supply row only if the empty, and if not last row */
    if (this->s[min_i] == 0 && u_head->next->next != 0)
        prev_u_min_i->next = prev_u_min_i->next->next; /* remove row from list */
    else
        prev_v_min_j->next = prev_v_min_j->next->next; /* remove column from list */
}

}  // namespace


//==============================================================================
// External interface

float cv::EMD(InputArray _sign1,
              InputArray _sign2,
              int distType,
              InputArray _cost,
              float* lowerBound,
              OutputArray _flow)
{
    CV_INSTRUMENT_REGION();

    Mat sign1 = _sign1.getMat();
    Mat sign2 = _sign2.getMat();
    Mat cost = _cost.getMat();

    CV_CheckEQ(sign1.cols, sign2.cols, "Signatures must have equal number of columns");
    CV_CheckEQ(sign1.type(), CV_32FC1, "The sign1 must be 32FC1");
    CV_CheckEQ(sign2.type(), CV_32FC1, "The sign2 must be 32FC1");

    const int dims = sign1.cols - 1;
    const int size1 = sign1.rows;
    const int size2 = sign2.rows;

    Mat flow;
    if (_flow.needed())
    {
        _flow.create(sign1.rows, sign2.rows, CV_32F);
        flow = _flow.getMat();
        flow = Scalar::all(0);
        CV_CheckEQ(flow.type(), CV_32FC1, "Flow matrix must have type 32FC1");
        CV_CheckTrue(flow.rows == size1 && flow.cols == size2,
                     "Flow matrix size does not match signatures");
    }

    DistFunc dfunc = 0;
    if (distType == DIST_USER)
    {
        if (!cost.empty())
        {
            CV_CheckEQ(cost.type(), CV_32FC1, "Cost matrix must have type 32FC1");
            CV_CheckTrue(cost.rows == size1 && cost.cols == size2,
                         "Cost matrix size does not match signatures");
            CV_CheckTrue(lowerBound == NULL,
                         "Lower boundary can not be calculated if the cost matrix is used");
        }
        else
        {
            CV_CheckTrue(dfunc == NULL, "Dist function must be set if cost matrix is empty");
        }
    }
    else
    {
        CV_CheckNE(dims, 0, "Number of dimensions can be 0 only if a user-defined metric is used");
        switch (distType)
        {
            case cv::DIST_L1: dfunc = distL1; break;
            case cv::DIST_L2: dfunc = distL2; break;
            case cv::DIST_C: dfunc = distC; break;
            default: CV_Error(cv::Error::StsBadFlag, "Bad or unsupported metric type");
        }
    }

    EMDSolver state;
    const bool result = state.init(sign1, sign2, dims, dfunc, cost, lowerBound);
    if (!result && lowerBound)
    {
        return *lowerBound;
    }
    state.solve();
    return (float)(state.calcFlow(_flow.needed() ? &flow : 0) / state.getWeight());
}

float cv::wrapperEMD(InputArray _sign1,
                     InputArray _sign2,
                     int distType,
                     InputArray _cost,
                     Ptr<float> lowerBound,
                     OutputArray _flow)
{
    return EMD(_sign1, _sign2, distType, _cost, lowerBound.get(), _flow);
}
