// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#ifndef OPENCV_HAL_RVV_EXP_HPP_INCLUDED
#define OPENCV_HAL_RVV_EXP_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_exp32f
#define cv_hal_exp32f cv::cv_hal_rvv::exp32f
#undef cv_hal_exp64f
#define cv_hal_exp64f cv::cv_hal_rvv::exp64f

namespace detail {

static constexpr size_t exp_scale = 6;
static constexpr size_t exp_tab_size = 1 << exp_scale;
static constexpr size_t exp_mask = exp_tab_size - 1;

static constexpr double exp_prescale = 1.4426950408889634073599246810019 * (1 << exp_scale);
static constexpr double exp_postscale = 1. / (1 << exp_scale);
// log10(DBL_MAX) < 3000
static constexpr double exp_max_val = (3000. * (1 << exp_scale)) / exp_prescale;
static constexpr double exp_min_val = -exp_max_val;

static constexpr double exp32f_a0 = .9670371139572337719125840413672004409288e-2;
static constexpr double exp32f_a1 = .5550339366753125211915322047004666939128e-1 / exp32f_a0;
static constexpr double exp32f_a2 = .2402265109513301490103372422686535526573 / exp32f_a0;
static constexpr double exp32f_a3 = .6931471805521448196800669615864773144641 / exp32f_a0;
static constexpr double exp32f_a4 = 1.000000000000002438532970795181890933776 / exp32f_a0;

static constexpr double exp64f_a0 = .13369713757180123244806654839424e-2 / exp32f_a0;
static constexpr double exp64f_a1 = .96180973140732918010002372686186e-2 / exp32f_a0;
static constexpr double exp64f_a2 = .55504108793649567998466049042729e-1 / exp32f_a0;
static constexpr double exp64f_a3 = .24022650695886477918181338054308 / exp32f_a0;
static constexpr double exp64f_a4 = .69314718055994546743029643825322 / exp32f_a0;
static constexpr double exp64f_a5 = .99999999999999999998285227504999 / exp32f_a0;

#define EXP_TAB_VALUE                                  \
    {                                                  \
        1.0 * exp32f_a0,                               \
        1.0108892860517004600204097905619 * exp32f_a0, \
        1.0218971486541166782344801347833 * exp32f_a0, \
        1.0330248790212284225001082839705 * exp32f_a0, \
        1.0442737824274138403219664787399 * exp32f_a0, \
        1.0556451783605571588083413251529 * exp32f_a0, \
        1.0671404006768236181695211209928 * exp32f_a0, \
        1.0787607977571197937406800374385 * exp32f_a0, \
        1.0905077326652576592070106557607 * exp32f_a0, \
        1.1023825833078409435564142094256 * exp32f_a0, \
        1.1143867425958925363088129569196 * exp32f_a0, \
        1.126521618608241899794798643787 * exp32f_a0,  \
        1.1387886347566916537038302838415 * exp32f_a0, \
        1.151189229952982705817759635202 * exp32f_a0,  \
        1.1637248587775775138135735990922 * exp32f_a0, \
        1.1763969916502812762846457284838 * exp32f_a0, \
        1.1892071150027210667174999705605 * exp32f_a0, \
        1.2021567314527031420963969574978 * exp32f_a0, \
        1.2152473599804688781165202513388 * exp32f_a0, \
        1.2284805361068700056940089577928 * exp32f_a0, \
        1.2418578120734840485936774687266 * exp32f_a0, \
        1.2553807570246910895793906574423 * exp32f_a0, \
        1.2690509571917332225544190810323 * exp32f_a0, \
        1.2828700160787782807266697810215 * exp32f_a0, \
        1.2968395546510096659337541177925 * exp32f_a0, \
        1.3109612115247643419229917863308 * exp32f_a0, \
        1.3252366431597412946295370954987 * exp32f_a0, \
        1.3396675240533030053600306697244 * exp32f_a0, \
        1.3542555469368927282980147401407 * exp32f_a0, \
        1.3690024229745906119296011329822 * exp32f_a0, \
        1.3839098819638319548726595272652 * exp32f_a0, \
        1.3989796725383111402095281367152 * exp32f_a0, \
        1.4142135623730950488016887242097 * exp32f_a0, \
        1.4296133383919700112350657782751 * exp32f_a0, \
        1.4451808069770466200370062414717 * exp32f_a0, \
        1.4609177941806469886513028903106 * exp32f_a0, \
        1.476826145939499311386907480374 * exp32f_a0,  \
        1.4929077282912648492006435314867 * exp32f_a0, \
        1.5091644275934227397660195510332 * exp32f_a0, \
        1.5255981507445383068512536895169 * exp32f_a0, \
        1.5422108254079408236122918620907 * exp32f_a0, \
        1.5590044002378369670337280894749 * exp32f_a0, \
        1.5759808451078864864552701601819 * exp32f_a0, \
        1.5931421513422668979372486431191 * exp32f_a0, \
        1.6104903319492543081795206673574 * exp32f_a0, \
        1.628027421857347766848218522014 * exp32f_a0,  \
        1.6457554781539648445187567247258 * exp32f_a0, \
        1.6636765803267364350463364569764 * exp32f_a0, \
        1.6817928305074290860622509524664 * exp32f_a0, \
        1.7001063537185234695013625734975 * exp32f_a0, \
        1.7186192981224779156293443764563 * exp32f_a0, \
        1.7373338352737062489942020818722 * exp32f_a0, \
        1.7562521603732994831121606193753 * exp32f_a0, \
        1.7753764925265212525505592001993 * exp32f_a0, \
        1.7947090750031071864277032421278 * exp32f_a0, \
        1.8142521755003987562498346003623 * exp32f_a0, \
        1.8340080864093424634870831895883 * exp32f_a0, \
        1.8539791250833855683924530703377 * exp32f_a0, \
        1.8741676341102999013299989499544 * exp32f_a0, \
        1.8945759815869656413402186534269 * exp32f_a0, \
        1.9152065613971472938726112702958 * exp32f_a0, \
        1.9360617934922944505980559045667 * exp32f_a0, \
        1.9571441241754002690183222516269 * exp32f_a0, \
        1.9784560263879509682582499181312 * exp32f_a0, \
    }

static constexpr float exp_tab_32f[exp_tab_size] = EXP_TAB_VALUE;
static constexpr double exp_tab_64f[exp_tab_size] = EXP_TAB_VALUE;

#undef EXP_TAB_VALUE

}  // namespace detail

inline int exp32f(const float* src, float* dst, int _len)
{
    size_t vl = __riscv_vsetvlmax_e32m4();
    auto exp_a2 = __riscv_vfmv_v_f_f32m4(detail::exp32f_a2, vl);
    auto exp_a3 = __riscv_vfmv_v_f_f32m4(detail::exp32f_a3, vl);
    auto exp_a4 = __riscv_vfmv_v_f_f32m4(detail::exp32f_a4, vl);
    for (size_t len = _len; len > 0; len -= vl, src += vl, dst += vl)
    {
        vl = __riscv_vsetvl_e32m4(len);
        auto x0 = __riscv_vle32_v_f32m4(src, vl);

        x0 = __riscv_vfmax(x0, detail::exp_min_val, vl);
        x0 = __riscv_vfmin(x0, detail::exp_max_val, vl);
        x0 = __riscv_vfmul(x0, detail::exp_prescale, vl);

        auto xi = __riscv_vfcvt_rtz_x_f_v_i32m4(x0, vl);
        x0 = __riscv_vfsub(x0, __riscv_vfcvt_f_x_v_f32m4(xi, vl), vl);
        x0 = __riscv_vfmul(x0, detail::exp_postscale, vl);

        auto t = __riscv_vsra(xi, detail::exp_scale, vl);
        t = __riscv_vadd(t, 127, vl);
        t = __riscv_vmax(t, 0, vl);
        t = __riscv_vmin(t, 255, vl);
        auto buf = __riscv_vreinterpret_f32m4(__riscv_vsll(t, 23, vl));

        auto _xi = __riscv_vreinterpret_u32m4(xi);
        _xi = __riscv_vsll(__riscv_vand(_xi, detail::exp_mask, vl), 2, vl);
        auto tab_v = __riscv_vluxei32(detail::exp_tab_32f, _xi, vl);

        auto res = __riscv_vfmul(buf, tab_v, vl);
        auto xn = __riscv_vfadd(x0, detail::exp32f_a1, vl);
        xn = __riscv_vfmadd(xn, x0, exp_a2, vl);
        xn = __riscv_vfmadd(xn, x0, exp_a3, vl);
        xn = __riscv_vfmadd(xn, x0, exp_a4, vl);

        res = __riscv_vfmul(res, xn, vl);
        __riscv_vse32(dst, res, vl);
    }

    return CV_HAL_ERROR_OK;
}

inline int exp64f(const double* src, double* dst, int _len)
{
    size_t vl = __riscv_vsetvlmax_e64m4();
    // all vector registers are used up, so not load more constants
    auto exp_a2 = __riscv_vfmv_v_f_f64m4(detail::exp64f_a2, vl);
    auto exp_a3 = __riscv_vfmv_v_f_f64m4(detail::exp64f_a3, vl);
    auto exp_a4 = __riscv_vfmv_v_f_f64m4(detail::exp64f_a4, vl);
    auto exp_a5 = __riscv_vfmv_v_f_f64m4(detail::exp64f_a5, vl);
    for (size_t len = _len; len > 0; len -= vl, src += vl, dst += vl)
    {
        vl = __riscv_vsetvl_e64m4(len);
        auto x0 = __riscv_vle64_v_f64m4(src, vl);

        x0 = __riscv_vfmax(x0, detail::exp_min_val, vl);
        x0 = __riscv_vfmin(x0, detail::exp_max_val, vl);
        x0 = __riscv_vfmul(x0, detail::exp_prescale, vl);

        auto xi = __riscv_vfcvt_rtz_x_f_v_i64m4(x0, vl);
        x0 = __riscv_vfsub(x0, __riscv_vfcvt_f_x_v_f64m4(xi, vl), vl);
        x0 = __riscv_vfmul(x0, detail::exp_postscale, vl);

        auto t = __riscv_vsra(xi, detail::exp_scale, vl);
        t = __riscv_vadd(t, 1023, vl);
        t = __riscv_vmax(t, 0, vl);
        t = __riscv_vmin(t, 2047, vl);
        auto buf = __riscv_vreinterpret_f64m4(__riscv_vsll(t, 52, vl));

        auto _xi = __riscv_vreinterpret_u64m4(xi);
        _xi = __riscv_vsll(__riscv_vand(_xi, detail::exp_mask, vl), 3, vl);
        auto tab_v = __riscv_vluxei64(detail::exp_tab_64f, _xi, vl);

        auto res = __riscv_vfmul(buf, tab_v, vl);
        auto xn = __riscv_vfadd(__riscv_vfmul(x0, detail::exp64f_a0, vl), detail::exp64f_a1, vl);
        xn = __riscv_vfmadd(xn, x0, exp_a2, vl);
        xn = __riscv_vfmadd(xn, x0, exp_a3, vl);
        xn = __riscv_vfmadd(xn, x0, exp_a4, vl);
        xn = __riscv_vfmadd(xn, x0, exp_a5, vl);

        res = __riscv_vfmul(res, xn, vl);
        __riscv_vse64(dst, res, vl);
    }

    return CV_HAL_ERROR_OK;
}

}}  // namespace cv::cv_hal_rvv

#endif //OPENCV_HAL_RVV_EXP_HPP_INCLUDED
