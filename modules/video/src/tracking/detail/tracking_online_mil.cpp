// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "tracking_online_mil.hpp"

namespace cv {
namespace detail {
inline namespace tracking {

#define sign(s) ((s > 0) ? 1 : ((s < 0) ? -1 : 0))

template <class T>
class SortableElementRev
{
public:
    T _val;
    int _ind;
    SortableElementRev()
        : _val(), _ind(0)
    {
    }
    SortableElementRev(T val, int ind)
    {
        _val = val;
        _ind = ind;
    }
    bool operator<(SortableElementRev<T>& b)
    {
        return (_val < b._val);
    }
};

static bool CompareSortableElementRev(const SortableElementRev<float>& i, const SortableElementRev<float>& j)
{
    return i._val < j._val;
}

template <class T>
void sort_order_des(std::vector<T>& v, std::vector<int>& order)
{
    uint n = (uint)v.size();
    std::vector<SortableElementRev<T>> v2;
    v2.resize(n);
    order.clear();
    order.resize(n);
    for (uint i = 0; i < n; i++)
    {
        v2[i]._ind = i;
        v2[i]._val = v[i];
    }
    //std::sort( v2.begin(), v2.end() );
    std::sort(v2.begin(), v2.end(), CompareSortableElementRev);
    for (uint i = 0; i < n; i++)
    {
        order[i] = v2[i]._ind;
        v[i] = v2[i]._val;
    }
}

//implementations for strong classifier

ClfMilBoost::Params::Params()
{
    _numSel = 50;
    _numFeat = 250;
    _lRate = 0.85f;
}

ClfMilBoost::ClfMilBoost()
    : _numsamples(0)
    , _counter(0)
{
    _myParams = ClfMilBoost::Params();
    _numsamples = 0;
}

ClfMilBoost::~ClfMilBoost()
{
    _selectors.clear();
    for (size_t i = 0; i < _weakclf.size(); i++)
        delete _weakclf.at(i);
}

void ClfMilBoost::init(const ClfMilBoost::Params& parameters)
{
    _myParams = parameters;
    _numsamples = 0;

    //_ftrs = Ftr::generate( _myParams->_ftrParams, _myParams->_numFeat );
    // if( params->_storeFtrHistory )
    //  Ftr::toViz( _ftrs, "haarftrs" );
    _weakclf.resize(_myParams._numFeat);
    for (int k = 0; k < _myParams._numFeat; k++)
    {
        _weakclf[k] = new ClfOnlineStump(k);
        _weakclf[k]->_lRate = _myParams._lRate;
    }
    _counter = 0;
}

void ClfMilBoost::update(const Mat& posx, const Mat& negx)
{
    int numneg = negx.rows;
    int numpos = posx.rows;

    // compute ftrs
    //if( !posx.ftrsComputed() )
    //  Ftr::compute( posx, _ftrs );
    //if( !negx.ftrsComputed() )
    //  Ftr::compute( negx, _ftrs );

    // initialize H
    static std::vector<float> Hpos, Hneg;
    Hpos.clear();
    Hneg.clear();
    Hpos.resize(posx.rows, 0.0f), Hneg.resize(negx.rows, 0.0f);

    _selectors.clear();
    std::vector<float> posw(posx.rows), negw(negx.rows);
    std::vector<std::vector<float>> pospred(_weakclf.size()), negpred(_weakclf.size());

    // train all weak classifiers without weights
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int m = 0; m < _myParams._numFeat; m++)
    {
        _weakclf[m]->update(posx, negx);
        pospred[m] = _weakclf[m]->classifySetF(posx);
        negpred[m] = _weakclf[m]->classifySetF(negx);
    }

    // pick the best features
    for (int s = 0; s < _myParams._numSel; s++)
    {

        // compute errors/likl for all weak clfs
        std::vector<float> poslikl(_weakclf.size(), 1.0f), neglikl(_weakclf.size()), likl(_weakclf.size());
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int w = 0; w < (int)_weakclf.size(); w++)
        {
            float lll = 1.0f;
            for (int j = 0; j < numpos; j++)
                lll *= (1 - sigmoid(Hpos[j] + pospred[w][j]));
            poslikl[w] = (float)-log(1 - lll + 1e-5);

            lll = 0.0f;
            for (int j = 0; j < numneg; j++)
                lll += (float)-log(1e-5f + 1 - sigmoid(Hneg[j] + negpred[w][j]));
            neglikl[w] = lll;

            likl[w] = poslikl[w] / numpos + neglikl[w] / numneg;
        }

        // pick best weak clf
        std::vector<int> order;
        sort_order_des(likl, order);

        // find best weakclf that isn't already included
        for (uint k = 0; k < order.size(); k++)
            if (std::count(_selectors.begin(), _selectors.end(), order[k]) == 0)
            {
                _selectors.push_back(order[k]);
                break;
            }

            // update H = H + h_m
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int k = 0; k < posx.rows; k++)
            Hpos[k] += pospred[_selectors[s]][k];
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int k = 0; k < negx.rows; k++)
            Hneg[k] += negpred[_selectors[s]][k];
    }

    //if( _myParams->_storeFtrHistory )
    //for ( uint j = 0; j < _selectors.size(); j++ )
    // _ftrHist( _selectors[j], _counter ) = 1.0f / ( j + 1 );

    _counter++;
    /* */
    return;
}

std::vector<float> ClfMilBoost::classify(const Mat& x, bool logR)
{
    int numsamples = x.rows;
    std::vector<float> res(numsamples);
    std::vector<float> tr;

    for (uint w = 0; w < _selectors.size(); w++)
    {
        tr = _weakclf[_selectors[w]]->classifySetF(x);
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int j = 0; j < numsamples; j++)
        {
            res[j] += tr[j];
        }
    }

    // return probabilities or log odds ratio
    if (!logR)
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int j = 0; j < (int)res.size(); j++)
        {
            res[j] = sigmoid(res[j]);
        }
    }

    return res;
}

//implementations for weak classifier

ClfOnlineStump::ClfOnlineStump()
    : _mu0(0), _mu1(0), _sig0(0), _sig1(0)
    , _q(0)
    , _s(0)
    , _log_n1(0), _log_n0(0)
    , _e1(0), _e0(0)
    , _lRate(0)
{
    _trained = false;
    _ind = -1;
    init();
}

ClfOnlineStump::ClfOnlineStump(int ind)
    : _mu0(0), _mu1(0), _sig0(0), _sig1(0)
    , _q(0)
    , _s(0)
    , _log_n1(0), _log_n0(0)
    , _e1(0), _e0(0)
    , _lRate(0)
{
    _trained = false;
    _ind = ind;
    init();
}
void ClfOnlineStump::init()
{
    _mu0 = 0;
    _mu1 = 0;
    _sig0 = 1;
    _sig1 = 1;
    _lRate = 0.85f;
    _trained = false;
}

void ClfOnlineStump::update(const Mat& posx, const Mat& negx, const Mat_<float>& /*posw*/, const Mat_<float>& /*negw*/)
{
    //std::cout << " ClfOnlineStump::update" << _ind << std::endl;
    float posmu = 0.0, negmu = 0.0;
    if (posx.cols > 0)
        posmu = float(mean(posx.col(_ind))[0]);
    if (negx.cols > 0)
        negmu = float(mean(negx.col(_ind))[0]);

    if (_trained)
    {
        if (posx.cols > 0)
        {
            _mu1 = (_lRate * _mu1 + (1 - _lRate) * posmu);
            cv::Mat diff = posx.col(_ind) - _mu1;
            _sig1 = _lRate * _sig1 + (1 - _lRate) * float(mean(diff.mul(diff))[0]);
        }
        if (negx.cols > 0)
        {
            _mu0 = (_lRate * _mu0 + (1 - _lRate) * negmu);
            cv::Mat diff = negx.col(_ind) - _mu0;
            _sig0 = _lRate * _sig0 + (1 - _lRate) * float(mean(diff.mul(diff))[0]);
        }

        _q = (_mu1 - _mu0) / 2;
        _s = sign(_mu1 - _mu0);
        _log_n0 = std::log(float(1.0f / pow(_sig0, 0.5f)));
        _log_n1 = std::log(float(1.0f / pow(_sig1, 0.5f)));
        //_e1 = -1.0f/(2.0f*_sig1+1e-99f);
        //_e0 = -1.0f/(2.0f*_sig0+1e-99f);
        _e1 = -1.0f / (2.0f * _sig1 + std::numeric_limits<float>::min());
        _e0 = -1.0f / (2.0f * _sig0 + std::numeric_limits<float>::min());
    }
    else
    {
        _trained = true;
        if (posx.cols > 0)
        {
            _mu1 = posmu;
            cv::Scalar scal_mean, scal_std_dev;
            cv::meanStdDev(posx.col(_ind), scal_mean, scal_std_dev);
            _sig1 = float(scal_std_dev[0]) * float(scal_std_dev[0]) + 1e-9f;
        }

        if (negx.cols > 0)
        {
            _mu0 = negmu;
            cv::Scalar scal_mean, scal_std_dev;
            cv::meanStdDev(negx.col(_ind), scal_mean, scal_std_dev);
            _sig0 = float(scal_std_dev[0]) * float(scal_std_dev[0]) + 1e-9f;
        }

        _q = (_mu1 - _mu0) / 2;
        _s = sign(_mu1 - _mu0);
        _log_n0 = std::log(float(1.0f / pow(_sig0, 0.5f)));
        _log_n1 = std::log(float(1.0f / pow(_sig1, 0.5f)));
        //_e1 = -1.0f/(2.0f*_sig1+1e-99f);
        //_e0 = -1.0f/(2.0f*_sig0+1e-99f);
        _e1 = -1.0f / (2.0f * _sig1 + std::numeric_limits<float>::min());
        _e0 = -1.0f / (2.0f * _sig0 + std::numeric_limits<float>::min());
    }
}

bool ClfOnlineStump::classify(const Mat& x, int i)
{
    float xx = x.at<float>(i, _ind);
    double log_p0 = (xx - _mu0) * (xx - _mu0) * _e0 + _log_n0;
    double log_p1 = (xx - _mu1) * (xx - _mu1) * _e1 + _log_n1;
    return log_p1 > log_p0;
}

float ClfOnlineStump::classifyF(const Mat& x, int i)
{
    float xx = x.at<float>(i, _ind);
    double log_p0 = (xx - _mu0) * (xx - _mu0) * _e0 + _log_n0;
    double log_p1 = (xx - _mu1) * (xx - _mu1) * _e1 + _log_n1;
    return float(log_p1 - log_p0);
}

inline std::vector<float> ClfOnlineStump::classifySetF(const Mat& x)
{
    std::vector<float> res(x.rows);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int k = 0; k < (int)res.size(); k++)
    {
        res[k] = classifyF(x, k);
    }
    return res;
}

}}}  // namespace cv::detail::tracking
