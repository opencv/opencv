// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// This file is not standalone.
// It is included with these active namespaces:
//namespace opencv_test { namespace hal { namespace intrinXXX {
//CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void test_hal_intrin_uint8();
void test_hal_intrin_int8();
void test_hal_intrin_uint16();
void test_hal_intrin_int16();
void test_hal_intrin_uint32();
void test_hal_intrin_int32();
void test_hal_intrin_uint64();
void test_hal_intrin_int64();
void test_hal_intrin_float32();
void test_hal_intrin_float64();

void test_hal_intrin_float16();

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

//==================================================================================================

template <typename R> struct Data
{
    typedef typename VTraits<R>::lane_type LaneType;
    typedef typename V_TypeTraits<LaneType>::int_type int_type;

    Data()
    {
        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
            d[i] = (LaneType)(i + 1);
    }
    Data(LaneType val)
    {
        fill(val);
    }
    Data(const R & r)
    {
        *this = r;
    }
    operator R () const
    {
        CV_Assert(VTraits<R>::vlanes() <= VTraits<R>::max_nlanes);
        return vx_load(d);
    }
    Data<R> & operator=(const R & r)
    {
        v_store(d, r);
        return *this;
    }
    template <typename T> Data<R> & operator*=(T m)
    {
        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
            d[i] *= (LaneType)m;
        return *this;
    }
    template <typename T> Data<R> & operator+=(T m)
    {
        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
            d[i] += (LaneType)m;
        return *this;
    }
    void fill(LaneType val, int s, int c = VTraits<R>::vlanes())
    {
        for (int i = s; i < c; ++i)
            d[i] = val;
    }
    void fill(LaneType val)
    {
        fill(val, 0);
    }
    void reverse()
    {
        for (int i = 0; i < VTraits<R>::vlanes() / 2; ++i)
            std::swap(d[i], d[VTraits<R>::vlanes() - i - 1]);
    }
    const LaneType & operator[](int i) const
    {
#if 0   // TODO: strange bug - AVX2 tests are failed with this
        CV_CheckGE(i, 0, ""); CV_CheckLT(i, (int)VTraits<R>::vlanes(), "");
#else
        CV_Assert(i >= 0 && i < VTraits<R>::max_nlanes);
#endif
        return d[i];
    }
    LaneType & operator[](int i)
    {
        CV_CheckGE(i, 0, ""); CV_CheckLT(i, (int)VTraits<R>::max_nlanes, "");
        return d[i];
    }
    int_type as_int(int i) const
    {
        CV_CheckGE(i, 0, ""); CV_CheckLT(i, (int)VTraits<R>::max_nlanes, "");
        union
        {
            LaneType l;
            int_type i;
        } v;
        v.l = d[i];
        return v.i;
    }
    const LaneType * mid() const
    {
        return d + VTraits<R>::vlanes() / 2;
    }
    LaneType * mid()
    {
        return d + VTraits<R>::vlanes() / 2;
    }
    LaneType sum(int s, int c)
    {
        LaneType res = 0;
        for (int i = s; i < s + c; ++i)
            res += d[i];
        return res;
    }
    LaneType sum()
    {
        return sum(0, VTraits<R>::vlanes());
    }
    bool operator==(const Data<R> & other) const
    {
        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
            if (d[i] != other.d[i])
                return false;
        return true;
    }
    void clear()
    {
        fill(0);
    }
    bool isZero() const
    {
        return isValue(0);
    }
    bool isValue(uchar val) const
    {
        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
            if (d[i] != val)
                return false;
        return true;
    }
    LaneType d[VTraits<R>::max_nlanes];
};

template<typename R> struct AlignedData
{
    Data<R> CV_DECL_ALIGNED(sizeof(typename VTraits<R>::lane_type)*VTraits<R>::max_nlanes) a; // aligned
    char dummy;
    Data<R> u; // unaligned
};

template <typename R> std::ostream & operator<<(std::ostream & out, const Data<R> & d)
{
    out << "{ ";
    for (int i = 0; i < VTraits<R>::vlanes(); ++i)
    {
        // out << std::hex << +V_TypeTraits<typename VTraits<R>::lane_type>::reinterpret_int(d.d[i]);
        out << +d.d[i];
        if (i + 1 < VTraits<R>::vlanes())
            out << ", ";
    }
    out << " }";
    return out;
}

template<typename T> static inline void EXPECT_COMPARE_EQ_(const T a, const T b)
{
    EXPECT_EQ(a, b);
}
template<> inline void EXPECT_COMPARE_EQ_<float>(const float a, const float b)
{
    EXPECT_FLOAT_EQ( a, b );
}

template<> inline void EXPECT_COMPARE_EQ_<double>(const double a, const double b)
{
    EXPECT_DOUBLE_EQ( a, b );
}

// pack functions do not do saturation when converting from 64-bit types
template<typename T, typename W>
inline T pack_saturate_cast(W a) { return saturate_cast<T>(a); }
template<>
inline int pack_saturate_cast<int, int64>(int64 a) { return static_cast<int>(a); }
template<>
inline unsigned pack_saturate_cast<unsigned, uint64>(uint64 a) { return static_cast<unsigned>(a); }

template<typename R> struct TheTest
{
    typedef typename VTraits<R>::lane_type LaneType;

    template <typename T1, typename T2>
    static inline void EXPECT_COMPARE_EQ(const T1 a, const T2 b)
    {
        EXPECT_COMPARE_EQ_<LaneType>((LaneType)a, (LaneType)b);
    }

    TheTest & test_loadstore()
    {
        AlignedData<R> data;
        AlignedData<R> out;

        // check if addresses are aligned and unaligned respectively
        EXPECT_EQ((size_t)0, (size_t)&data.a.d % (sizeof(typename VTraits<R>::lane_type) * VTraits<R>::vlanes()));
        EXPECT_NE((size_t)0, (size_t)&data.u.d % (sizeof(typename VTraits<R>::lane_type) * VTraits<R>::vlanes()));
        EXPECT_EQ((size_t)0, (size_t)&out.a.d % (sizeof(typename VTraits<R>::lane_type) * VTraits<R>::vlanes()));
        EXPECT_NE((size_t)0, (size_t)&out.u.d % (sizeof(typename VTraits<R>::lane_type) * VTraits<R>::vlanes()));

        // check some initialization methods
        R r1 = data.a;
        R r2 = vx_load(data.u.d);
        R r3 = vx_load_aligned(data.a.d);
        R r4(r2);
        EXPECT_EQ(data.a[0], v_get0(r1));
        EXPECT_EQ(data.u[0], v_get0(r2));
        EXPECT_EQ(data.a[0], v_get0(r3));
        EXPECT_EQ(data.u[0], v_get0(r4));

        R r_low = vx_load_low((LaneType*)data.u.d);
        EXPECT_EQ(data.u[0], v_get0(r_low));
        v_store(out.u.d, r_low);
        for (int i = 0; i < VTraits<R>::vlanes()/2; ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ((LaneType)data.u[i], (LaneType)out.u[i]);
        }

        R r_low_align8byte = vx_load_low((LaneType*)((char*)data.u.d + (sizeof(typename VTraits<R>::lane_type) * VTraits<R>::vlanes() / 2)));
        EXPECT_EQ(data.u[VTraits<R>::vlanes()/2], v_get0(r_low_align8byte));
        v_store(out.u.d, r_low_align8byte);
        for (int i = 0; i < VTraits<R>::vlanes()/2; ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ((LaneType)data.u[i + VTraits<R>::vlanes()/2], (LaneType)out.u[i]);
        }

        // check some store methods
        out.u.clear();
        out.a.clear();
        v_store(out.u.d, r1);
        v_store_aligned(out.a.d, r2);
        EXPECT_EQ(data.a, out.a);
        EXPECT_EQ(data.u, out.u);

        // check more store methods
        Data<R> d, res(0);
        R r5 = d;
        v_store_high(res.mid(), r5);
        v_store_low(res.d, r5);
        EXPECT_EQ(d, res);

        // check halves load correctness
        res.clear();
        R r6 = vx_load_halves(d.d, d.mid());
        v_store(res.d, r6);
        EXPECT_EQ(d, res);

        // zero, all
        Data<R> resZ, resV;
        resZ.fill((LaneType)0);
        resV.fill((LaneType)8);
        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ((LaneType)0, resZ[i]);
            EXPECT_EQ((LaneType)8, resV[i]);
        }

        // reinterpret_as
        v_uint8 vu8 = v_reinterpret_as_u8(r1); out.a.clear(); v_store((uchar*)out.a.d, vu8); EXPECT_EQ(data.a, out.a);
        v_int8 vs8 = v_reinterpret_as_s8(r1); out.a.clear(); v_store((schar*)out.a.d, vs8); EXPECT_EQ(data.a, out.a);
        v_uint16 vu16 = v_reinterpret_as_u16(r1); out.a.clear(); v_store((ushort*)out.a.d, vu16); EXPECT_EQ(data.a, out.a);
        v_int16 vs16 = v_reinterpret_as_s16(r1); out.a.clear(); v_store((short*)out.a.d, vs16); EXPECT_EQ(data.a, out.a);
        v_uint32 vu32 = v_reinterpret_as_u32(r1); out.a.clear(); v_store((unsigned*)out.a.d, vu32); EXPECT_EQ(data.a, out.a);
        v_int32 vs32 = v_reinterpret_as_s32(r1); out.a.clear(); v_store((int*)out.a.d, vs32); EXPECT_EQ(data.a, out.a);
        v_uint64 vu64 = v_reinterpret_as_u64(r1); out.a.clear(); v_store((uint64*)out.a.d, vu64); EXPECT_EQ(data.a, out.a);
        v_int64 vs64 = v_reinterpret_as_s64(r1); out.a.clear(); v_store((int64*)out.a.d, vs64); EXPECT_EQ(data.a, out.a);
        v_float32 vf32 = v_reinterpret_as_f32(r1); out.a.clear(); v_store((float*)out.a.d, vf32); EXPECT_EQ(data.a, out.a);
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
        v_float64 vf64 = v_reinterpret_as_f64(r1); out.a.clear(); v_store((double*)out.a.d, vf64); EXPECT_EQ(data.a, out.a);
#endif

#if CV_SIMD_WIDTH == 16
        R setall_res1 = v_setall((LaneType)5);
        R setall_res2 = v_setall<LaneType>(6);
#elif CV_SIMD_WIDTH == 32
        R setall_res1 = v256_setall((LaneType)5);
        R setall_res2 = v256_setall<LaneType>(6);
#elif CV_SIMD_WIDTH == 64
        R setall_res1 = v512_setall((LaneType)5);
        R setall_res2 = v512_setall<LaneType>(6);
#elif CV_SIMD_SCALABLE
        R setall_res1 = v_setall((LaneType)5);
        R setall_res2 = v_setall<LaneType>(6);
#else
#error "Configuration error"
#endif
        R setall_res3 = v_setall_<R>((LaneType)7);
        R setall_resz = v_setzero_<R>();
#if CV_SIMD_WIDTH > 0
        Data<R> setall_res1_; v_store(setall_res1_.d, setall_res1);
        Data<R> setall_res2_; v_store(setall_res2_.d, setall_res2);
        Data<R> setall_res3_; v_store(setall_res3_.d, setall_res3);
        Data<R> setall_resz_; v_store(setall_resz_.d, setall_resz);
        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ((LaneType)5, setall_res1_[i]);
            EXPECT_EQ((LaneType)6, setall_res2_[i]);
            EXPECT_EQ((LaneType)7, setall_res3_[i]);
            EXPECT_EQ((LaneType)0, setall_resz_[i]);
        }
#endif

        R vx_setall_res1 = vx_setall((LaneType)11);
        R vx_setall_res2 = vx_setall<LaneType>(12);
        Data<R> vx_setall_res1_; v_store(vx_setall_res1_.d, vx_setall_res1);
        Data<R> vx_setall_res2_; v_store(vx_setall_res2_.d, vx_setall_res2);
        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ((LaneType)11, vx_setall_res1_[i]);
            EXPECT_EQ((LaneType)12, vx_setall_res2_[i]);
        }

#if CV_SIMD_WIDTH == 16
        {
            uint64 a = CV_BIG_INT(0x7fffffffffffffff);
            uint64 b = (uint64)CV_BIG_INT(0xcfffffffffffffff);
            v_uint64x2 uint64_vec(a, b);
            EXPECT_EQ(a, v_get0(uint64_vec));
            EXPECT_EQ(b, v_extract_n<1>(uint64_vec));
        }
        {
            int64 a = CV_BIG_INT(0x7fffffffffffffff);
            int64 b = CV_BIG_INT(-1);
            v_int64x2 int64_vec(a, b);
            EXPECT_EQ(a, v_get0(int64_vec));
            EXPECT_EQ(b, v_extract_n<1>(int64_vec));
        }
#endif

        return *this;
    }

    TheTest & test_interleave()
    {
        Data<R> data1, data2, data3, data4;
        data2 += 20;
        data3 += 40;
        data4 += 60;


        R a = data1, b = data2, c = data3;
        R d = data1, e = data2, f = data3, g = data4;

        LaneType buf3[VTraits<R>::max_nlanes * 3];
        LaneType buf4[VTraits<R>::max_nlanes * 4];

        v_store_interleave(buf3, a, b, c);
        v_store_interleave(buf4, d, e, f, g);

        Data<R> z(0);
        a = b = c = d = e = f = g = z;

        v_load_deinterleave(buf3, a, b, c);
        v_load_deinterleave(buf4, d, e, f, g);

        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ(data1, Data<R>(a));
            EXPECT_EQ(data2, Data<R>(b));
            EXPECT_EQ(data3, Data<R>(c));

            EXPECT_EQ(data1, Data<R>(d));
            EXPECT_EQ(data2, Data<R>(e));
            EXPECT_EQ(data3, Data<R>(f));
            EXPECT_EQ(data4, Data<R>(g));
        }

        return *this;
    }
    TheTest & test_interleave_pq()
    {
        Data<R> dataA;
        R a = dataA;
        Data<R> resP = v_interleave_pairs(a);
        Data<R> resQ = v_interleave_quads(a);
        for (int i = 0; i < VTraits<R>::vlanes()/4; ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ(resP[4*i],     dataA[4*i  ]);
            EXPECT_EQ(resP[4*i + 1], dataA[4*i+2]);
            EXPECT_EQ(resP[4*i + 2], dataA[4*i+1]);
            EXPECT_EQ(resP[4*i + 3], dataA[4*i+3]);
        }
        for (int i = 0; i < VTraits<R>::vlanes()/8; ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ(resQ[8*i],     dataA[8*i  ]);
            EXPECT_EQ(resQ[8*i + 1], dataA[8*i+4]);
            EXPECT_EQ(resQ[8*i + 2], dataA[8*i+1]);
            EXPECT_EQ(resQ[8*i + 3], dataA[8*i+5]);
            EXPECT_EQ(resQ[8*i + 4], dataA[8*i+2]);
            EXPECT_EQ(resQ[8*i + 5], dataA[8*i+6]);
            EXPECT_EQ(resQ[8*i + 6], dataA[8*i+3]);
            EXPECT_EQ(resQ[8*i + 7], dataA[8*i+7]);
        }
        return *this;
    }

    // float32x4 only
    TheTest & test_interleave_2channel()
    {
        Data<R> data1, data2;
        data2 += 20;

        R a = data1, b = data2;

        LaneType buf2[VTraits<R>::max_nlanes * 2];

        v_store_interleave(buf2, a, b);

        Data<R> z(0);
        a = b = z;

        v_load_deinterleave(buf2, a, b);

        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ(data1, Data<R>(a));
            EXPECT_EQ(data2, Data<R>(b));
        }

        return *this;
    }

    // v_expand and v_load_expand
    TheTest & test_expand()
    {
        typedef typename V_RegTraits<R>::w_reg Rx2;
        Data<R> dataA;
        R a = dataA;

        Data<Rx2> resB = vx_load_expand(dataA.d);

        Rx2 c, d, e, f;
        v_expand(a, c, d);

        e = v_expand_low(a);
        f = v_expand_high(a);

        Data<Rx2> resC = c, resD = d, resE = e, resF = f;
        const int n = VTraits<Rx2>::vlanes();
        for (int i = 0; i < n; ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ(dataA[i], resB[i]);
            EXPECT_EQ(dataA[i], resC[i]);
            EXPECT_EQ(dataA[i + n], resD[i]);
            EXPECT_EQ(dataA[i], resE[i]);
            EXPECT_EQ(dataA[i + n], resF[i]);
        }

        return *this;
    }

    TheTest & test_expand_q()
    {
        typedef typename V_RegTraits<R>::q_reg Rx4;
        Data<R> data;
        Data<Rx4> out = vx_load_expand_q(data.d);
        const int n = VTraits<Rx4>::vlanes();
        for (int i = 0; i < n; ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ(data[i], out[i]);
        }

        return *this;
    }

    TheTest & test_addsub()
    {
        Data<R> dataA, dataB, dataC;
        dataB.reverse();
        dataA[1] = static_cast<LaneType>(std::numeric_limits<LaneType>::max());
        R a = dataA, b = dataB, c = dataC;

        Data<R> resD = v_add(a, b), resE = v_add(a, b, c), resF = v_sub(a, b);
        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ(saturate_cast<LaneType>(dataA[i] + dataB[i]), resD[i]);
            EXPECT_EQ(saturate_cast<LaneType>(dataA[i] + dataB[i] + dataC[i]), resE[i]);
            EXPECT_EQ(saturate_cast<LaneType>(dataA[i] - dataB[i]), resF[i]);
        }

        return *this;
    }

    TheTest & test_arithm_wrap()
    {
        Data<R> dataA, dataB;
        dataB.reverse();
        R a = dataA, b = dataB;

        Data<R> resC = v_add_wrap(a, b),
                resD = v_sub_wrap(a, b),
                resE = v_mul_wrap(a, b);
        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ((LaneType)(dataA[i] + dataB[i]), resC[i]);
            EXPECT_EQ((LaneType)(dataA[i] - dataB[i]), resD[i]);
            EXPECT_EQ((LaneType)(dataA[i] * dataB[i]), resE[i]);
        }
        return *this;
    }

    TheTest & test_mul()
    {
        Data<R> dataA, dataB, dataC;
        dataA[1] = static_cast<LaneType>(std::numeric_limits<LaneType>::max());
        dataB.reverse();
        R a = dataA, b = dataB, c = dataC;

        Data<R> resD = v_mul(a, b);
        Data<R> resE = v_mul(a, b, c);
        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ(saturate_cast<LaneType>(dataA[i] * dataB[i]), resD[i]);
            EXPECT_EQ(saturate_cast<LaneType>(dataA[i] * dataB[i] * dataC[i]), resE[i]);
        }

        return *this;
    }

    TheTest & test_div()
    {
        Data<R> dataA, dataB;
        dataB.reverse();
        R a = dataA, b = dataB;

        Data<R> resC = v_div(a, b);
        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ(dataA[i] / dataB[i], resC[i]);
        }

        return *this;
    }

    TheTest & test_mul_expand()
    {
        typedef typename V_RegTraits<R>::w_reg Rx2;
        Data<R> dataA, dataB(2);
        R a = dataA, b = dataB;
        Rx2 c, d;

        v_mul_expand(a, b, c, d);

        Data<Rx2> resC = c, resD = d;
        const int n = VTraits<R>::vlanes() / 2;
        for (int i = 0; i < n; ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ((typename VTraits<Rx2>::lane_type)dataA[i] * dataB[i], resC[i]);
            EXPECT_EQ((typename VTraits<Rx2>::lane_type)dataA[i + n] * dataB[i + n], resD[i]);
        }

        return *this;
    }

    TheTest & test_mul_hi()
    {
        // typedef typename V_RegTraits<R>::w_reg Rx2;
        Data<R> dataA, dataB(32767);
        R a = dataA, b = dataB;

        R c = v_mul_hi(a, b);

        Data<R> resC = c;
        const int n = VTraits<R>::vlanes() / 2;
        for (int i = 0; i < n; ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ((typename VTraits<R>::lane_type)((dataA[i] * dataB[i]) >> 16), resC[i]);
        }

        return *this;
    }

    TheTest & test_abs()
    {
        typedef typename V_RegTraits<R>::u_reg Ru;
        typedef typename VTraits<Ru>::lane_type u_type;
        typedef typename VTraits<R>::lane_type R_type;
        Data<R> dataA, dataB(10);
        R a = dataA, b = dataB;
        a = v_sub(a, b);

        Data<Ru> resC = v_abs(a);

        for (int i = 0; i < VTraits<Ru>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            R_type ssub = dataA[i] - dataB[i] < std::numeric_limits<R_type>::lowest() ? std::numeric_limits<R_type>::lowest() : dataA[i] - dataB[i];
            EXPECT_EQ((u_type)std::abs(ssub), resC[i]);
        }

        return *this;
    }

    template <int s>
    TheTest & test_shift()
    {
        SCOPED_TRACE(s);
        Data<R> dataA;
        dataA[0] = static_cast<LaneType>(std::numeric_limits<LaneType>::max());
        R a = dataA;

        Data<R> resB = v_shl<s>(a), resC = v_shl<s>(a), resD = v_shr<s>(a), resE = v_shr<s>(a);

        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ(static_cast<LaneType>(dataA[i] << s), resB[i]);
            EXPECT_EQ(static_cast<LaneType>(dataA[i] << s), resC[i]);
            EXPECT_EQ(static_cast<LaneType>(dataA[i] >> s), resD[i]);
            EXPECT_EQ(static_cast<LaneType>(dataA[i] >> s), resE[i]);
        }
        return *this;
    }

    TheTest & test_cmp()
    {
        Data<R> dataA, dataB;
        dataB.reverse();
        dataB += 1;
        R a = dataA, b = dataB;

        Data<R> resC = (v_eq(a, b));
        Data<R> resD = (v_ne(a, b));
        Data<R> resE = (v_gt(a, b));
        Data<R> resF = (v_ge(a, b));
        Data<R> resG = (v_lt(a, b));
        Data<R> resH = (v_le(a, b));

        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ(dataA[i] == dataB[i], resC[i] != 0);
            EXPECT_EQ(dataA[i] != dataB[i], resD[i] != 0);
            EXPECT_EQ(dataA[i] >  dataB[i], resE[i] != 0);
            EXPECT_EQ(dataA[i] >= dataB[i], resF[i] != 0);
            EXPECT_EQ(dataA[i] <  dataB[i], resG[i] != 0);
            EXPECT_EQ(dataA[i] <= dataB[i], resH[i] != 0);
        }
        return *this;
    }

    TheTest & test_dotprod()
    {
        typedef typename V_RegTraits<R>::w_reg Rx2;
        typedef typename VTraits<Rx2>::lane_type w_type;

        Data<R> dataA, dataB;
        dataA += std::numeric_limits<LaneType>::max() - VTraits<R>::vlanes();
        dataB += std::numeric_limits<LaneType>::min() + VTraits<R>::vlanes();
        R a = dataA, b = dataB;

        Data<Rx2> dataC;
        dataC += std::numeric_limits<w_type>::is_signed ?
                    std::numeric_limits<w_type>::min() :
                    std::numeric_limits<w_type>::max() - VTraits<R>::vlanes() * (dataB[0] + 1);
        Rx2 c = dataC;

        Data<Rx2> resD = v_dotprod(a, b),
                  resE = v_dotprod(a, b, c);

        const int n = VTraits<R>::vlanes() / 2;
        w_type sumAB = 0, sumABC = 0, tmp_sum;
        for (int i = 0; i < n; ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));

            tmp_sum = (w_type)dataA[i*2] * (w_type)dataB[i*2] +
                      (w_type)dataA[i*2 + 1] * (w_type)dataB[i*2 + 1];
            sumAB  += tmp_sum;
            EXPECT_EQ(tmp_sum, resD[i]);

            tmp_sum = tmp_sum + dataC[i];
            sumABC += tmp_sum;
            EXPECT_EQ(tmp_sum, resE[i]);
        }

        w_type resF = v_reduce_sum(v_dotprod_fast(a, b)),
               resG = v_reduce_sum(v_dotprod_fast(a, b, c));
        EXPECT_EQ(sumAB,  resF);
        EXPECT_EQ(sumABC, resG);
        return *this;
    }

    TheTest & test_dotprod_expand()
    {
        typedef typename V_RegTraits<R>::q_reg Rx4;
        typedef typename VTraits<Rx4>::lane_type l4_type;

        Data<R> dataA, dataB;
        dataA += std::numeric_limits<LaneType>::max() - VTraits<R>::vlanes();
        dataB += std::numeric_limits<LaneType>::min() + VTraits<R>::vlanes();
        R a = dataA, b = dataB;

        Data<Rx4> dataC;
        Rx4 c = dataC;

        Data<Rx4> resD = v_dotprod_expand(a, b),
                  resE = v_dotprod_expand(a, b, c);

        l4_type sumAB = 0, sumABC = 0, tmp_sum;
        for (int i = 0; i < VTraits<Rx4>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            tmp_sum  = (l4_type)dataA[i*4]     * (l4_type)dataB[i*4]     +
                       (l4_type)dataA[i*4 + 1] * (l4_type)dataB[i*4 + 1] +
                       (l4_type)dataA[i*4 + 2] * (l4_type)dataB[i*4 + 2] +
                       (l4_type)dataA[i*4 + 3] * (l4_type)dataB[i*4 + 3];
            sumAB  += tmp_sum;
            EXPECT_EQ(tmp_sum, resD[i]);

            tmp_sum = tmp_sum + dataC[i];
            sumABC += tmp_sum;
            EXPECT_EQ(tmp_sum, resE[i]);
        }

        l4_type resF = v_reduce_sum(v_dotprod_expand_fast(a, b)),
                resG = v_reduce_sum(v_dotprod_expand_fast(a, b, c));
        EXPECT_EQ(sumAB,  resF);
        EXPECT_EQ(sumABC, resG);

        return *this;
    }

    TheTest & test_dotprod_expand_f64()
    {
    #if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
        Data<R> dataA, dataB;
        dataA += std::numeric_limits<LaneType>::max() - VTraits<R>::vlanes();
        dataB += std::numeric_limits<LaneType>::min();
        R a = dataA, b = dataB;

        Data<v_float64> dataC;
        v_float64 c = dataC;

        Data<v_float64> resA = v_dotprod_expand(a, a),
                        resB = v_dotprod_expand(b, b),
                        resC = v_dotprod_expand(a, b, c);

        const int n = VTraits<R>::vlanes() / 2;
        for (int i = 0; i < n; ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_COMPARE_EQ((double)dataA[i*2]     * (double)dataA[i*2] +
                              (double)dataA[i*2 + 1] * (double)dataA[i*2  + 1], resA[i]);
            EXPECT_COMPARE_EQ((double)dataB[i*2]     * (double)dataB[i*2] +
                              (double)dataB[i*2 + 1] * (double)dataB[i*2  + 1], resB[i]);
            EXPECT_COMPARE_EQ((double)dataA[i*2]     * (double)dataB[i*2] +
                              (double)dataA[i*2 + 1] * (double)dataB[i*2  + 1] + dataC[i], resC[i]);
        }
    #endif
        return *this;
    }

    TheTest & test_logic()
    {
        Data<R> dataA, dataB(2);
        R a = dataA, b = dataB;

        Data<R> resC = v_and(a, b), resD = v_or(a, b), resE = v_xor(a, b), resF = v_not(a);
        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ(dataA[i] & dataB[i], resC[i]);
            EXPECT_EQ(dataA[i] | dataB[i], resD[i]);
            EXPECT_EQ(dataA[i] ^ dataB[i], resE[i]);
            EXPECT_EQ((LaneType)~dataA[i], resF[i]);
        }

        return *this;
    }

    TheTest & test_sqrt_abs()
    {
        Data<R> dataA, dataD;
        dataD *= -1.0;
        R a = dataA, d = dataD;

        Data<R> resB = v_sqrt(a), resC = v_invsqrt(a), resE = v_abs(d);
        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_COMPARE_EQ((float)std::sqrt(dataA[i]), (float)resB[i]);
            EXPECT_COMPARE_EQ((float)(1/std::sqrt(dataA[i])), (float)resC[i]);
            EXPECT_COMPARE_EQ((float)abs(dataA[i]), (float)resE[i]);
        }

        return *this;
    }

    TheTest & test_min_max()
    {
        Data<R> dataA, dataB;
        dataB.reverse();
        R a = dataA, b = dataB;

        Data<R> resC = v_min(a, b), resD = v_max(a, b);
        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ(std::min(dataA[i], dataB[i]), resC[i]);
            EXPECT_EQ(std::max(dataA[i], dataB[i]), resD[i]);
        }

        return *this;
    }

    TheTest & test_popcount()
    {
        typedef typename V_RegTraits<R>::u_reg Ru;
        static unsigned popcountTable[] = {
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, //0x00-0x0f
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, //0x10-0x1f
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, //0x20-0x2f
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, //0x30-0x3f
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, //0x40-0x4f
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, //0x50-0x5f
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, //0x60-0x6f
            3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, //0x70-0x7f
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, //0x80-0x8f
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, //0x90-0x9f
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, //0xa0-0xaf
            3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, //0xb0-0xbf
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, //0xc0-0xcf
            3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, //0xd0-0xdf
            3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, //0xe0-0xef
            4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8, //0xf0-0xff
            0
        };
        Data<R> dataA;
        R a = dataA;

        Data<Ru> resB = v_popcount(a);
        for (int i = 0; i < VTraits<Ru>::vlanes(); ++i)
            EXPECT_EQ(popcountTable[i + 1], resB[i]);

        return *this;
    }

    TheTest & test_absdiff()
    {
        typedef typename V_RegTraits<R>::u_reg Ru;
        typedef typename VTraits<Ru>::lane_type u_type;
        Data<R> dataA(std::numeric_limits<LaneType>::max()),
                dataB(std::numeric_limits<LaneType>::min());
        dataA[0] = (LaneType)-1;
        dataB[0] = 1;
        dataA[1] = 2;
        dataB[1] = (LaneType)-2;
        R a = dataA, b = dataB;
        Data<Ru> resC = v_absdiff(a, b);
        const u_type mask = std::numeric_limits<LaneType>::is_signed ? (u_type)(1 << (sizeof(u_type)*8 - 1)) : 0;
        for (int i = 0; i < VTraits<Ru>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            u_type uA = dataA[i] ^ mask;
            u_type uB = dataB[i] ^ mask;
            EXPECT_EQ(uA > uB ? uA - uB : uB - uA, resC[i]);
        }
        return *this;
    }

    TheTest & test_float_absdiff()
    {
        Data<R> dataA(std::numeric_limits<LaneType>::max()),
                dataB(std::numeric_limits<LaneType>::min());
        dataA[0] = -1;
        dataB[0] = 1;
        dataA[1] = 2;
        dataB[1] = -2;
        R a = dataA, b = dataB;
        Data<R> resC = v_absdiff(a, b);
        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ(dataA[i] > dataB[i] ? dataA[i] - dataB[i] : dataB[i] - dataA[i], resC[i]);
        }
        return *this;
    }

    TheTest & test_absdiffs()
    {
        Data<R> dataA(std::numeric_limits<LaneType>::max()),
                dataB(std::numeric_limits<LaneType>::min());
        dataA[0] = (LaneType)-1;
        dataB[0] = 1;
        dataA[1] = 2;
        dataB[1] = (LaneType)-2;
        R a = dataA, b = dataB;
        Data<R> resC = v_absdiffs(a, b);
        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            EXPECT_EQ(saturate_cast<LaneType>(std::abs(dataA[i] - dataB[i])), resC[i]);
        }
        return *this;
    }

    TheTest & test_reduce()
    {
        Data<R> dataA;
        LaneType min = (LaneType)VTraits<R>::vlanes(), max = 0;
        int sum = 0;
        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            min = std::min<LaneType>(min, dataA[i]);
            max = std::max<LaneType>(max, dataA[i]);
            sum += (int)(dataA[i]);   // To prevent a constant overflow with int8
        }
        R a = dataA;
        EXPECT_EQ((LaneType)min, (LaneType)v_reduce_min(a));
        EXPECT_EQ((LaneType)max, (LaneType)v_reduce_max(a));
        EXPECT_EQ((int)(sum), (int)v_reduce_sum(a));
        dataA[0] += (LaneType)VTraits<R>::vlanes();
        R an = dataA;
        min = (LaneType)VTraits<R>::vlanes();
        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            min = std::min<LaneType>(min, dataA[i]);
        }
        EXPECT_EQ((LaneType)min, (LaneType)v_reduce_min(an));
        return *this;
    }

    TheTest & test_reduce_sad()
    {
        Data<R> dataA, dataB((LaneType)VTraits<R>::vlanes() /2);
        R a = dataA;
        R b = dataB;
        uint sum = 0;
        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            sum += std::abs(int(dataA[i] - dataB[i]));
        }
        EXPECT_EQ(sum, v_reduce_sad(a, b));
        return *this;
    }

    TheTest & test_mask()
    {
        typedef typename V_RegTraits<R>::int_reg int_reg;
        typedef typename V_RegTraits<int_reg>::u_reg uint_reg;
        typedef typename VTraits<int_reg>::lane_type int_type;
        typedef typename VTraits<uint_reg>::lane_type uint_type;

        Data<R> dataA, dataB(0), dataC, dataD(1), dataE(2);
        dataA[0] = (LaneType)std::numeric_limits<int_type>::max();
        dataA[1] *= (LaneType)-1;
        union
        {
            LaneType l;
            uint_type ui;
        }
        all1s;
        all1s.ui = (uint_type)-1;
        LaneType mask_one = all1s.l;
        dataB[VTraits<R>::vlanes() - 1] = mask_one;
        R l = dataB;
        dataB[1] = mask_one;
        dataB[VTraits<R>::vlanes() / 2] = mask_one;
        for (int i = 0; i < VTraits<R>::vlanes(); i++)
        {
            auto c_signed = dataC.as_int(i);
            dataC[i] = c_signed == 0 ? -1 : -std::abs(c_signed);
        }
        R a = dataA, b = dataB, c = dataC, d = dataD, e = dataE;
        dataC[VTraits<R>::vlanes() - 1] = 0;
        R nl = dataC;

        EXPECT_EQ(2, v_signmask(a));
#if (CV_SIMD_WIDTH <= 32) && (!CV_SIMD_SCALABLE)
        EXPECT_EQ(2 | (1 << (VTraits<R>::vlanes() / 2)) | (1 << (VTraits<R>::vlanes() - 1)), v_signmask(b));
#endif

        EXPECT_EQ(false, v_check_all(a));
        EXPECT_EQ(false, v_check_all(b));
        EXPECT_EQ(true, v_check_all(c));
        EXPECT_EQ(false, v_check_all(nl));

        EXPECT_EQ(true, v_check_any(a));
        EXPECT_EQ(true, v_check_any(b));
        EXPECT_EQ(true, v_check_any(c));
        EXPECT_EQ(true, v_check_any(l));
        R f = v_select(b, d, e);
        Data<R> resF = f;
        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            int_type m2 = dataB.as_int(i);
            EXPECT_EQ((dataD.as_int(i) & m2) | (dataE.as_int(i) & ~m2), resF.as_int(i));
        }

        return *this;
    }

    template <int s>
    TheTest & test_pack()
    {
        SCOPED_TRACE(s);
        typedef typename V_RegTraits<R>::w_reg Rx2;
        typedef typename VTraits<Rx2>::lane_type w_type;
        Data<Rx2> dataA, dataB;
        dataA += std::numeric_limits<LaneType>::is_signed ? -10 : 10;
        dataB *= 10;
        dataB[0] = static_cast<w_type>(std::numeric_limits<LaneType>::max()) + 17; // to check saturation
        Rx2 a = dataA, b = dataB;

        Data<R> resC = v_pack(a, b);
        Data<R> resD = v_rshr_pack<s>(a, b);

        Data<R> resE(0);
        v_pack_store(resE.d, b);

        Data<R> resF(0);
        v_rshr_pack_store<s>(resF.d, b);

        const int n = VTraits<Rx2>::vlanes();
        const w_type add = (w_type)1 << (s - 1);
        for (int i = 0; i < n; ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ(pack_saturate_cast<LaneType>(dataA[i]), resC[i]);
            EXPECT_EQ(pack_saturate_cast<LaneType>(dataB[i]), resC[i + n]);
            EXPECT_EQ(pack_saturate_cast<LaneType>((dataA[i] + add) >> s), resD[i]);
            EXPECT_EQ(pack_saturate_cast<LaneType>((dataB[i] + add) >> s), resD[i + n]);
            EXPECT_EQ(pack_saturate_cast<LaneType>(dataB[i]), resE[i]);
            EXPECT_EQ((LaneType)0, resE[i + n]);
            EXPECT_EQ(pack_saturate_cast<LaneType>((dataB[i] + add) >> s), resF[i]);
            EXPECT_EQ((LaneType)0, resF[i + n]);
        }
        return *this;
    }

    TheTest & test_pack_triplets()
    {
        Data<R> dataA;
        R a = dataA;
        Data<R> res = v_pack_triplets(a);

        for (int i = 0; i < VTraits<R>::vlanes()/4; ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ(dataA[4*i],   res[3*i]);
            EXPECT_EQ(dataA[4*i+1], res[3*i+1]);
            EXPECT_EQ(dataA[4*i+2], res[3*i+2]);
        }
        return *this;
    }

    template <int s>
    TheTest & test_pack_u()
    {
        SCOPED_TRACE(s);
        //typedef typename V_RegTraits<LaneType>::w_type LaneType_w;
        typedef typename V_RegTraits<R>::w_reg R2;
        typedef typename V_RegTraits<R2>::int_reg Ri2;
        typedef typename VTraits<Ri2>::lane_type w_type;

        Data<Ri2> dataA, dataB;
        dataA += -10;
        dataB *= 10;
        dataB[0] = static_cast<w_type>(std::numeric_limits<LaneType>::max()) + 17; // to check saturation
        Ri2 a = dataA, b = dataB;

        Data<R> resC = v_pack_u(a, b);
        Data<R> resD = v_rshr_pack_u<s>(a, b);

        Data<R> resE(0);
        v_pack_u_store(resE.d, b);

        Data<R> resF(0);
        v_rshr_pack_u_store<s>(resF.d, b);

        const int n = VTraits<Ri2>::vlanes();
        const w_type add = (w_type)1 << (s - 1);
        for (int i = 0; i < n; ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ(pack_saturate_cast<LaneType>(dataA[i]), resC[i]);
            EXPECT_EQ(pack_saturate_cast<LaneType>(dataB[i]), resC[i + n]);
            EXPECT_EQ(pack_saturate_cast<LaneType>((dataA[i] + add) >> s), resD[i]);
            EXPECT_EQ(pack_saturate_cast<LaneType>((dataB[i] + add) >> s), resD[i + n]);
            EXPECT_EQ(pack_saturate_cast<LaneType>(dataB[i]), resE[i]);
            EXPECT_EQ((LaneType)0, resE[i + n]);
            EXPECT_EQ(pack_saturate_cast<LaneType>((dataB[i] + add) >> s), resF[i]);
            EXPECT_EQ((LaneType)0, resF[i + n]);
        }
        return *this;
    }

    // v_uint8 only
    TheTest & test_pack_b()
    {
        // 16-bit
        Data<R> dataA, dataB;
        dataB.fill(0, VTraits<R>::vlanes() / 2);

        R a = dataA, b = dataB;
        Data<R> maskA = v_eq(a, b), maskB = v_ne(a, b);

        a = maskA; b = maskB;
        Data<R> res  = v_pack_b(v_reinterpret_as_u16(a), v_reinterpret_as_u16(b));
        for (int i = 0; i < VTraits<v_uint16>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ(maskA[i * 2], res[i]);
            EXPECT_EQ(maskB[i * 2], res[i + VTraits<v_uint16>::vlanes()]);
        }

        // 32-bit
        Data<R> dataC, dataD;
        dataD.fill(0, VTraits<R>::vlanes() / 2);

        R c = dataC, d = dataD;
        Data<R> maskC = v_eq(c, d), maskD = v_ne(c, d);

        c = maskC; d = maskD;
        res = v_pack_b
        (
            v_reinterpret_as_u32(a), v_reinterpret_as_u32(b),
            v_reinterpret_as_u32(c), v_reinterpret_as_u32(d)
        );

        for (int i = 0; i < VTraits<v_uint32>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ(maskA[i * 4], res[i]);
            EXPECT_EQ(maskB[i * 4], res[i + VTraits<v_uint32>::vlanes()]);
            EXPECT_EQ(maskC[i * 4], res[i + VTraits<v_uint32>::vlanes() * 2]);
            EXPECT_EQ(maskD[i * 4], res[i + VTraits<v_uint32>::vlanes() * 3]);
        }

        // 64-bit
        Data<R> dataE, dataF, dataG(0), dataH(0xFF);
        dataF.fill(0, VTraits<R>::vlanes() / 2);

        R e = dataE, f = dataF, g = dataG, h = dataH;
        Data<R> maskE = v_eq(e, f), maskF = v_ne(e, f);

        e = maskE; f = maskF;
        res = v_pack_b
        (
            v_reinterpret_as_u64(a), v_reinterpret_as_u64(b),
            v_reinterpret_as_u64(c), v_reinterpret_as_u64(d),
            v_reinterpret_as_u64(e), v_reinterpret_as_u64(f),
            v_reinterpret_as_u64(g), v_reinterpret_as_u64(h)
        );

        for (int i = 0; i < VTraits<v_uint64>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ(maskA[i * 8], res[i]);
            EXPECT_EQ(maskB[i * 8], res[i + VTraits<v_uint64>::vlanes()]);
            EXPECT_EQ(maskC[i * 8], res[i + VTraits<v_uint64>::vlanes() * 2]);
            EXPECT_EQ(maskD[i * 8], res[i + VTraits<v_uint64>::vlanes() * 3]);

            EXPECT_EQ(maskE[i * 8], res[i + VTraits<v_uint64>::vlanes() * 4]);
            EXPECT_EQ(maskF[i * 8], res[i + VTraits<v_uint64>::vlanes() * 5]);
            EXPECT_EQ(dataG[i * 8], res[i + VTraits<v_uint64>::vlanes() * 6]);
            EXPECT_EQ(dataH[i * 8], res[i + VTraits<v_uint64>::vlanes() * 7]);
        }

        return *this;
    }

    TheTest & test_unpack()
    {
        Data<R> dataA, dataB;
        dataB *= 10;
        R a = dataA, b = dataB;

        R c, d, e, f, lo, hi;
        v_zip(a, b, c, d);
        v_recombine(a, b, e, f);
        lo = v_combine_low(a, b);
        hi = v_combine_high(a, b);

        Data<R> resC = c, resD = d, resE = e, resF = f, resLo = lo, resHi = hi;

        const int n = VTraits<R>::vlanes()/2;
        for (int i = 0; i < n; ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ(dataA[i], resC[i*2]);
            EXPECT_EQ(dataB[i], resC[i*2+1]);
            EXPECT_EQ(dataA[i+n], resD[i*2]);
            EXPECT_EQ(dataB[i+n], resD[i*2+1]);

            EXPECT_EQ(dataA[i], resE[i]);
            EXPECT_EQ(dataB[i], resE[i+n]);
            EXPECT_EQ(dataA[i+n], resF[i]);
            EXPECT_EQ(dataB[i+n], resF[i+n]);

            EXPECT_EQ(dataA[i], resLo[i]);
            EXPECT_EQ(dataB[i], resLo[i+n]);
            EXPECT_EQ(dataA[i+n], resHi[i]);
            EXPECT_EQ(dataB[i+n], resHi[i+n]);
        }

        return *this;
    }

    TheTest & test_reverse()
    {
        Data<R> dataA;
        R a = dataA;

        Data<R> resB = v_reverse(a);

        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ(dataA[VTraits<R>::vlanes() - i - 1], resB[i]);
        }

        return *this;
    }

    template<int s>
    TheTest & test_extract()
    {
        SCOPED_TRACE(s);
        Data<R> dataA, dataB;
        dataB *= 10;
        R a = dataA, b = dataB;

        Data<R> resC = v_extract<s>(a, b);

        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            if (i + s >= VTraits<R>::vlanes())
                EXPECT_EQ(dataB[i - VTraits<R>::vlanes() + s], resC[i]);
            else
                EXPECT_EQ(dataA[i + s], resC[i]);
        }

        return *this;
    }

    template<int s>
    TheTest & test_rotate()
    {
        SCOPED_TRACE(s);
        Data<R> dataA, dataB;
        dataB *= 10;
        R a = dataA, b = dataB;

        Data<R> resC = v_rotate_right<s>(a);
        Data<R> resD = v_rotate_right<s>(a, b);

        Data<R> resE = v_rotate_left<s>(a);
        Data<R> resF = v_rotate_left<s>(a, b);

        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            if (i + s >= VTraits<R>::vlanes())
            {
                EXPECT_EQ((LaneType)0, resC[i]);
                EXPECT_EQ(dataB[i - VTraits<R>::vlanes() + s], resD[i]);

                EXPECT_EQ((LaneType)0, resE[i - VTraits<R>::vlanes() + s]);
                EXPECT_EQ(dataB[i], resF[i - VTraits<R>::vlanes() + s]);
            }
            else
            {
                EXPECT_EQ(dataA[i + s], resC[i]);
                EXPECT_EQ(dataA[i + s], resD[i]);

                EXPECT_EQ(dataA[i], resE[i + s]);
                EXPECT_EQ(dataA[i], resF[i + s]);
            }
        }
        return *this;
    }

    template<int s>
    TheTest & test_extract_n()
    {
        SCOPED_TRACE(s);
        Data<R> dataA;
        LaneType test_value = (LaneType)(s + 50);
        dataA[s] = test_value;
        R a = dataA;

        LaneType res = v_extract_n<s>(a);
        EXPECT_EQ(test_value, res);

        return *this;
    }

    TheTest & test_extract_highest()
    {
        Data<R> dataA;
        LaneType test_value = (LaneType)(VTraits<R>::vlanes()-1 + 50);
        dataA[VTraits<R>::vlanes()-1] = test_value;
        R a = dataA;

        LaneType res = v_extract_highest(a);
        EXPECT_EQ(test_value, res);

        return *this;
    }

    template<int s>
    TheTest & test_broadcast_element()
    {
        SCOPED_TRACE(s);
        Data<R> dataA;
        LaneType test_value = (LaneType)(s + 50);
        dataA[s] = test_value;
        R a = dataA;

        Data<R> res = v_broadcast_element<s>(a);

        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            SCOPED_TRACE(i);
            EXPECT_EQ(test_value, res[i]);
        }
        return *this;
    }

    TheTest & test_broadcast_highest()
    {
        Data<R> dataA;
        LaneType test_value = (LaneType)(VTraits<R>::vlanes()-1 + 50);
        dataA[VTraits<R>::vlanes()-1] = test_value;
        R a = dataA;

        Data<R> res = v_broadcast_highest(a);

        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            SCOPED_TRACE(i);
            EXPECT_EQ(test_value, res[i]);
        }
        return *this;
    }

    TheTest & test_float_math()
    {
        typedef typename V_RegTraits<R>::round_reg Ri;
        Data<R> data1, data1_border, data2, data3;
        // See https://github.com/opencv/opencv/issues/24213
        data1_border *= 0.5;
        data1 *= 1.1;
        data2 += 10;
        R a1 = data1, a1_border = data1_border, a2 = data2, a3 = data3;

        Data<Ri> resB = v_round(a1),
                 resB_border = v_round(a1_border),
                 resC = v_trunc(a1),
                 resD = v_floor(a1),
                 resE = v_ceil(a1);

        Data<R> resF = v_magnitude(a1, a2),
                resG = v_sqr_magnitude(a1, a2),
                resH = v_muladd(a1, a2, a3);

        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ(cvRound(data1[i]), resB[i]);
            EXPECT_EQ(cvRound(data1_border[i]), resB_border[i]);
            EXPECT_EQ((typename VTraits<Ri>::lane_type)data1[i], resC[i]);
            EXPECT_EQ(cvFloor(data1[i]), resD[i]);
            EXPECT_EQ(cvCeil(data1[i]), resE[i]);

            EXPECT_COMPARE_EQ(std::sqrt(data1[i]*data1[i] + data2[i]*data2[i]), resF[i]);
            EXPECT_COMPARE_EQ(data1[i]*data1[i] + data2[i]*data2[i], resG[i]);
            EXPECT_COMPARE_EQ(data1[i]*data2[i] + data3[i], resH[i]);
        }

        return *this;
    }
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
    TheTest & test_round_pair_f64()
    {
        typedef typename V_RegTraits<R>::round_reg Ri;
        Data<R> data1, data1_border, data2;
        // See https://github.com/opencv/opencv/issues/24213
        // https://github.com/opencv/opencv/issues/24163
        // https://github.com/opencv/opencv/pull/24271
        data1_border *= 0.5;
        data1 *= 1.1;
        data2 += 10;
        R a1 = data1, a1_border = data1_border, a2 = data2;

        Data<Ri> resA = v_round(a1, a1),
                 resB = v_round(a1_border, a1_border),
                 resC = v_round(a2, a2);

        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            EXPECT_EQ(cvRound(data1[i]), resA[i]);
            EXPECT_EQ(cvRound(data1_border[i]), resB[i]);
            EXPECT_EQ(cvRound(data2[i]), resC[i]);
        }

        return *this;
    }
#endif

    TheTest & test_float_cvt32()
    {
        typedef v_float32 Rt;
        Data<R> dataA;
        dataA *= 1.1;
        R a = dataA;
        Rt b = v_cvt_f32(a);
        Data<Rt> resB = b;
        int n = std::min<int>(VTraits<Rt>::vlanes(), VTraits<R>::vlanes());
        for (int i = 0; i < n; ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ((typename VTraits<Rt>::lane_type)dataA[i], resB[i]);
        }
        return *this;
    }

    TheTest & test_float_cvt64()
    {
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
        typedef v_float64 Rt;
        Data<R> dataA;
        dataA *= 1.1;
        R a = dataA;
        Rt b = v_cvt_f64(a);
        Rt c = v_cvt_f64_high(a);
        Data<Rt> resB = b;
        Data<Rt> resC = c;
        int n = std::min<int>(VTraits<Rt>::vlanes(), VTraits<R>::vlanes());
        for (int i = 0; i < n; ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ((typename VTraits<Rt>::lane_type)dataA[i], resB[i]);
        }
        for (int i = 0; i < n; ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ((typename VTraits<Rt>::lane_type)dataA[i+n], resC[i]);
        }
#endif
        return *this;
    }

    TheTest & test_cvt64_double()
    {
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
        Data<R> dataA(std::numeric_limits<LaneType>::max()),
                dataB(std::numeric_limits<LaneType>::min());
        dataB += VTraits<R>::vlanes();

        R a = dataA, b = dataB;
        v_float64 c = v_cvt_f64(a), d = v_cvt_f64(b);

        Data<v_float64> resC = c;
        Data<v_float64> resD = d;

        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_EQ((double)dataA[i], resC[i]);
            EXPECT_EQ((double)dataB[i], resD[i]);
        }
#endif
        return *this;
    }

    TheTest & test_matmul()
    {
        Data<R> dataV, dataA, dataB, dataC, dataD;
        dataB.reverse();
        dataC += 2;
        dataD *= 0.3;
        R v = dataV, a = dataA, b = dataB, c = dataC, d = dataD;

        Data<R> res = v_matmul(v, a, b, c, d);
        // for (int i = 0; i < VTraits<R>::vlanes(); i += 4)
        // {
            int i = 0;
            for (int j = i; j < i + 4; ++j)
            {
                SCOPED_TRACE(cv::format("i=%d j=%d", i, j));
                LaneType val = dataV[i]     * dataA[j]
                             + dataV[i + 1] * dataB[j]
                             + dataV[i + 2] * dataC[j]
                             + dataV[i + 3] * dataD[j];
                EXPECT_COMPARE_EQ(val, res[j]);
            }
        // }

        Data<R> resAdd = v_matmuladd(v, a, b, c, d);
        // for (int i = 0; i < VTraits<R>::vlanes(); i += 4)
        // {
            i = 0;
            for (int j = i; j < i + 4; ++j)
            {
                SCOPED_TRACE(cv::format("i=%d j=%d", i, j));
                LaneType val = dataV[i]     * dataA[j]
                             + dataV[i + 1] * dataB[j]
                             + dataV[i + 2] * dataC[j]
                             + dataD[j];
                EXPECT_COMPARE_EQ(val, resAdd[j]);
            }
        // }
        return *this;
    }

    TheTest & test_transpose()
    {
        Data<R> dataA, dataB, dataC, dataD;
        dataB *= 5;
        dataC *= 10;
        dataD *= 15;
        R a = dataA, b = dataB, c = dataC, d = dataD;
        R e, f, g, h;
        v_transpose4x4(a, b, c, d,
                       e, f, g, h);

        // Data<R> res[4] = {e, f, g, h}; // Generates incorrect data in certain RVV case.
        Data<R> res0 = e, res1 = f, res2 = g, res3 = h;
        EXPECT_EQ(dataA[0], res0[0]);
        EXPECT_EQ(dataB[0], res0[1]);
        EXPECT_EQ(dataC[0], res0[2]);
        EXPECT_EQ(dataD[0], res0[3]);

        EXPECT_EQ(dataA[1], res1[0]);
        EXPECT_EQ(dataB[1], res1[1]);
        EXPECT_EQ(dataC[1], res1[2]);
        EXPECT_EQ(dataD[1], res1[3]);

        EXPECT_EQ(dataA[2], res2[0]);
        EXPECT_EQ(dataB[2], res2[1]);
        EXPECT_EQ(dataC[2], res2[2]);
        EXPECT_EQ(dataD[2], res2[3]);

        EXPECT_EQ(dataA[3], res3[0]);
        EXPECT_EQ(dataB[3], res3[1]);
        EXPECT_EQ(dataC[3], res3[2]);
        EXPECT_EQ(dataD[3], res3[3]);
        return *this;
    }

    TheTest & test_reduce_sum4()
    {
        Data<R> dataA, dataB, dataC, dataD;
        dataB *= 0.01f;
        dataC *= 0.001f;
        dataD *= 0.002f;

        R a = dataA, b = dataB, c = dataC, d = dataD;
        Data<R> res = v_reduce_sum4(a, b, c, d);

        for (int i = 0; i < VTraits<R>::vlanes(); i += 4)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            EXPECT_COMPARE_EQ(dataA.sum(i, 4), res[i]);
            EXPECT_COMPARE_EQ(dataB.sum(i, 4), res[i + 1]);
            EXPECT_COMPARE_EQ(dataC.sum(i, 4), res[i + 2]);
            EXPECT_COMPARE_EQ(dataD.sum(i, 4), res[i + 3]);
        }
        return *this;
    }

    TheTest & test_loadstore_fp16_f32()
    {
        printf("test_loadstore_fp16_f32 ...\n");
        AlignedData<v_uint16> data; data.a.clear();
        data.a.d[0] = 0x3c00; // 1.0
        data.a.d[VTraits<R>::vlanes() - 1] = (unsigned short)0xc000; // -2.0
        AlignedData<v_float32> data_f32; data_f32.a.clear();
        AlignedData<v_uint16> out;

        R r1 = vx_load_expand((const cv::hfloat*)data.a.d);
        R r2(r1);
        EXPECT_EQ(1.0f, v_get0(r1));
        v_store(data_f32.a.d, r2);
        EXPECT_EQ(-2.0f, data_f32.a.d[VTraits<R>::vlanes() - 1]);

        out.a.clear();
        v_pack_store((cv::hfloat*)out.a.d, r2);
        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            EXPECT_EQ(data.a[i], out.a[i]) << "i=" << i;
        }

        return *this;
    }

#if 0
    TheTest & test_loadstore_fp16()
    {
        printf("test_loadstore_fp16 ...\n");
        AlignedData<R> data;
        AlignedData<R> out;

        // check if addresses are aligned and unaligned respectively
        EXPECT_EQ((size_t)0, (size_t)&data.a.d % VTraits<R>::max_nlanes);
        EXPECT_NE((size_t)0, (size_t)&data.u.d % VTraits<R>::max_nlanes);
        EXPECT_EQ((size_t)0, (size_t)&out.a.d % VTraits<R>::max_nlanes);
        EXPECT_NE((size_t)0, (size_t)&out.u.d % VTraits<R>::max_nlanes);

        // check some initialization methods
        R r1 = data.u;
        R r2 = vx_load_expand((const hfloat*)data.a.d);
        R r3(r2);
        EXPECT_EQ(data.u[0], v_get0(r1));
        EXPECT_EQ(data.a[0], v_get0(r2));
        EXPECT_EQ(data.a[0], v_get0(r3));

        // check some store methods
        out.a.clear();
        v_store(out.a.d, r1);
        EXPECT_EQ(data.a, out.a);

        return *this;
    }
    TheTest & test_float_cvt_fp16()
    {
        printf("test_float_cvt_fp16 ...\n");
        AlignedData<v_float32> data;

        // check conversion
        v_float32 r1 = vx_load(data.a.d);
        v_float16 r2 = v_cvt_f16(r1, vx_setzero_f32());
        v_float32 r3 = v_cvt_f32(r2);
        EXPECT_EQ(0x3c00, v_get0(r2));
        EXPECT_EQ(v_get0(r3), v_get0(r1));

        return *this;
    }
#endif

    void do_check_cmp64(const Data<R>& dataA, const Data<R>& dataB)
    {
        R a = dataA;
        R b = dataB;

        Data<R> dataEQ = v_eq(a, b);
        Data<R> dataNE = v_ne(a, b);

        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            SCOPED_TRACE(cv::format("i=%d", i));
            if (cvtest::debugLevel > 0) cout << "i=" << i << " ( " << dataA[i] << " vs " << dataB[i] << " ): eq=" << dataEQ[i] << " ne=" << dataNE[i] << endl;
            EXPECT_NE((LaneType)dataEQ[i], (LaneType)dataNE[i]);
            if (dataA[i] == dataB[i])
                EXPECT_EQ((LaneType)-1, (LaneType)dataEQ[i]);
            else
                EXPECT_EQ((LaneType)0, (LaneType)dataEQ[i]);
            if (dataA[i] != dataB[i])
                EXPECT_EQ((LaneType)-1, (LaneType)dataNE[i]);
            else
                EXPECT_EQ((LaneType)0, (LaneType)dataNE[i]);
        }
    }

    TheTest & test_cmp64()
    {
        Data<R> dataA;
        Data<R> dataB;

        for (int i = 0; i < VTraits<R>::vlanes(); ++i)
        {
            dataA[i] = dataB[i];
        }
        dataA[0]++;

        do_check_cmp64(dataA, dataB);
        do_check_cmp64(dataB, dataA);

        dataA[0] = dataB[0];
        dataA[1] += (((LaneType)1) << 32);
        do_check_cmp64(dataA, dataB);
        do_check_cmp64(dataB, dataA);

        dataA[0] = (LaneType)-1;
        dataB[0] = (LaneType)-1;
        dataA[1] = (LaneType)-1;
        dataB[1] = (LaneType)2;

        do_check_cmp64(dataA, dataB);
        do_check_cmp64(dataB, dataA);

        return *this;
    }

    void __test_exp(LaneType dataMax, LaneType diff_thr, LaneType enlarge_factor, LaneType flt_min) {
        int n = VTraits<R>::vlanes();

        // Test overflow and underflow values with step
        const LaneType step = (LaneType) 0.01;
        for (LaneType i = dataMax + 1; i <= dataMax + 11;) {
            Data<R> dataUpperBound, dataLowerBound, resOverflow, resUnderflow;
            for (int j = 0; j < n; ++j) {
                dataUpperBound[j] = i;
                dataLowerBound[j] = -i;
                i += step;
            }
            R upperBound = dataUpperBound, lowerBound = dataLowerBound;
            resOverflow = v_exp(upperBound);
            resUnderflow = v_exp(lowerBound);
            for (int j = 0; j < n; ++j) {
                SCOPED_TRACE(cv::format("Overflow/Underflow test value: %f", i));
                EXPECT_TRUE(resOverflow[j] > 0 && std::isinf(resOverflow[j]));
                EXPECT_GE(resUnderflow[j], 0);
                EXPECT_LT(resUnderflow[j], flt_min);
            }
        }

        // Test random values combined with special values
        std::vector<LaneType> specialValues = {0, 1, INFINITY, -INFINITY, NAN, dataMax};
        const int testRandNum = 10000;
        const double specialValueProbability = 0.1; // 10% chance to insert a special value
        cv::RNG_MT19937 rng;

        for (int i = 0; i < testRandNum; i++) {
            Data<R> dataRand, resRand;
            for (int j = 0; j < n; ++j) {
                if (rng.uniform(0.f, 1.f) <= specialValueProbability) {
                    // Insert a special value
                    int specialValueIndex = rng.uniform(0, (int) specialValues.size());
                    dataRand[j] = specialValues[specialValueIndex];
                } else {
                    // Generate random data in [-dataMax*1.1, dataMax*1.1]
                    dataRand[j] = (LaneType) rng.uniform(-dataMax * 1.1, dataMax * 1.1);
                }
            }
            // Compare with std::exp
            R x = dataRand;
            resRand = v_exp(x);
            for (int j = 0; j < n; ++j) {
                SCOPED_TRACE(cv::format("Random test value: %f", dataRand[j]));
                LaneType std_exp = std::exp(dataRand[j]);
                if (dataRand[j] == 0) {
                    // input 0 -> output 1
                    EXPECT_EQ(resRand[j], 1);
                } else if (dataRand[j] == 1) {
                    // input 1 -> output e
                    EXPECT_NEAR((LaneType) M_E, resRand[j], 1e-15);
                } else if (dataRand[j] > 0 && std::isinf(dataRand[j])) {
                    // input INF -> output INF
                    EXPECT_TRUE(resRand[j] > 0 && std::isinf(resRand[j]));
                } else if (dataRand[j] < 0 && std::isinf(dataRand[j])) {
                    // input -INF -> output 0
                    EXPECT_EQ(resRand[j], 0);
                } else if (std::isnan(dataRand[j])) {
                    // input NaN -> output NaN
                    EXPECT_TRUE(std::isnan(resRand[j]));
                } else if (dataRand[j] == dataMax) {
                    // input dataMax -> output less than INFINITY
                    EXPECT_LT(resRand[j], (LaneType) INFINITY);
                } else if (std::isinf(resRand[j])) {
                    // output INF -> input close to edge
                    EXPECT_GT(dataRand[j], dataMax);
                } else {
                    EXPECT_GE(resRand[j], 0);
                    EXPECT_LT(std::abs(resRand[j] - std_exp), diff_thr * (std_exp + flt_min * enlarge_factor));
                }
            }
        }
    }

    TheTest &test_exp_fp16() {
#if CV_SIMD_FP16
        float16_t flt16_min;
        uint16_t flt16_min_hex = 0x0400;
        std::memcpy(&flt16_min, &flt16_min_hex, sizeof(float16_t));
        __test_exp((float16_t) 10, (float16_t) 1e-2, (float16_t) 1e2, flt16_min);
#endif
        return *this;
    }

    TheTest &test_exp_fp32() {
        __test_exp(88.0f, 1e-6f, 1e6f, FLT_MIN);
        return *this;
    }

    TheTest &test_exp_fp64() {
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
        __test_exp(709.0, 1e-15, 1e15, DBL_MIN);
#endif
        return *this;
    }

    void __test_log(LaneType expBound, LaneType diff_thr, LaneType flt_min) {
        int n = VTraits<R>::vlanes();
        // Test special values
        std::vector<LaneType> specialValues = {0, 1, (LaneType) M_E, INFINITY, -INFINITY, NAN};
        const int testRandNum = 10000;
        const double specialValueProbability = 0.1; // 10% chance to insert a special value
        cv::RNG_MT19937 rng;

        for (int i = 0; i < testRandNum; i++) {
            Data<R> dataRand, resRand;
            for (int j = 0; j < n; ++j) {
                if (rng.uniform(0.f, 1.f) <= specialValueProbability) {
                    // Insert a special value
                    int specialValueIndex = rng.uniform(0, (int) specialValues.size());
                    dataRand[j] = specialValues[specialValueIndex];
                } else {
                    // Generate uniform random data in [-expBound, expBound]
                    dataRand[j] = (LaneType) std::exp(rng.uniform(-expBound, expBound));
                }
            }

            // Compare with std::log
            R x = dataRand;
            resRand = v_log(x);
            for (int j = 0; j < n; ++j) {
                SCOPED_TRACE(cv::format("Random test value: %f", dataRand[j]));
                LaneType std_log = std::log(dataRand[j]);
                if (dataRand[j] == 0) {
                    // input 0 -> output -INF
                    EXPECT_TRUE(std::isinf(resRand[j]) && resRand[j] < 0);
                } else if (dataRand[j] < 0 || std::isnan(dataRand[j])) {
                    // input less than 0 -> output NAN
                    // input NaN -> output NaN
                    EXPECT_TRUE(std::isnan(resRand[j]));
                } else if (dataRand[j] == 1) {
                    // input 1 -> output 0
                    EXPECT_EQ((LaneType) 0, resRand[j]);
                } else if (std::isinf(dataRand[j]) && dataRand[j] > 0) {
                    // input INF -> output INF
                    EXPECT_TRUE(std::isinf(resRand[j]) && resRand[j] > 0);
                } else {
                    EXPECT_LT(std::abs(resRand[j] - std_log), diff_thr * (std::abs(std_log) + flt_min * 100));
                }
            }
        }
    }

    TheTest &test_log_fp16() {
#if CV_SIMD_FP16
        float16_t flt16_min;
        uint16_t flt16_min_hex = 0x0400;
        std::memcpy(&flt16_min, &flt16_min_hex, sizeof(float16_t));
        __test_log((float16_t) 9, (float16_t) 1e-3, flt16_min);
#endif
        return *this;
    }

    TheTest &test_log_fp32() {
        __test_log(25.f, 1e-6f, FLT_MIN);
        return *this;
    }

    TheTest &test_log_fp64() {
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
        __test_log(200., 1e-15, DBL_MIN);
#endif
        return *this;
    }

    TheTest &test_erf_fp32() {
        int n = VTraits<R>::vlanes();

        constexpr int num_loops = 10000;
        const std::vector<LaneType> singular_inputs{INFINITY, -INFINITY, NAN};
        constexpr double insert_singular_input_probability = 0.1;
        cv::RNG_MT19937 rng;

        for (int i = 0; i < num_loops; i++) {
            Data<R> inputs;
            for (int j = 0; j < n; j++) {
                if (rng.uniform(0.f, 1.f) <= insert_singular_input_probability) {
                    int singular_input_index = rng.uniform(0, int(singular_inputs.size()));
                    inputs[j] = singular_inputs[singular_input_index];
                } else {
                    // std::exp(float) overflows at about 88.0f.
                    // In v_erf, exp is called on input*input. So test range is [-sqrt(88.0f), sqrt(88.0f)]
                    inputs[j] = (LaneType) rng.uniform(-9.4f, 9.4f);
                }
            }

            Data<R> outputs = v_erf(R(inputs));
            for (int j = 0; j < n; j++) {
                SCOPED_TRACE(cv::format("Random test value: %f", inputs[j]));
                if (std::isinf(inputs[j])) {
                    if (inputs[j] < 0) {
                        EXPECT_EQ(-1, outputs[j]);
                    } else {
                        EXPECT_EQ(1, outputs[j]);
                    }
                } else if (std::isnan(inputs[j])) {
                    EXPECT_TRUE(std::isnan(outputs[j]));
                } else {
                    LaneType ref_output = std::erf(inputs[j]);
                    EXPECT_LT(std::abs(outputs[j] - ref_output), 9e-3f * (std::abs(ref_output) + FLT_MIN * 1e4f));
                }
            }
        }

        return *this;
    }

    void __test_sincos(LaneType diff_thr, LaneType flt_min) {
        int n = VTraits<R>::vlanes();
        // Test each value for a period, from -PI to PI
        const LaneType step = (LaneType) 0.01;
        for (LaneType i = 0; i <= (LaneType)M_PI;) {
            Data<R> dataPosPI, dataNegPI;
            for (int j = 0; j < n; ++j) {
                dataPosPI[j] = i;
                dataNegPI[j] = -i;
                i += step;
            }
            R posPI = dataPosPI, negPI = dataNegPI, sinPos, cosPos, sinNeg, cosNeg;
            v_sincos(posPI, sinPos, cosPos);
            v_sincos(negPI, sinNeg, cosNeg);
            Data<R> resSinPos = sinPos, resCosPos = cosPos, resSinNeg = sinNeg, resCosNeg = cosNeg;
            for (int j = 0; j < n; ++j) {
                LaneType std_sin_pos = (LaneType) std::sin(dataPosPI[j]);
                LaneType std_cos_pos = (LaneType) std::cos(dataPosPI[j]);
                LaneType std_sin_neg = (LaneType) std::sin(dataNegPI[j]);
                LaneType std_cos_neg = (LaneType) std::cos(dataNegPI[j]);
                SCOPED_TRACE(cv::format("Period test value: %lf and %lf", (double) dataPosPI[j], (double) dataNegPI[j]));
                EXPECT_LT(std::abs(resSinPos[j] - std_sin_pos), diff_thr * (std::abs(std_sin_pos) + flt_min * 100));
                EXPECT_LT(std::abs(resCosPos[j] - std_cos_pos), diff_thr * (std::abs(std_cos_pos) + flt_min * 100));
                EXPECT_LT(std::abs(resSinNeg[j] - std_sin_neg), diff_thr * (std::abs(std_sin_neg) + flt_min * 100));
                EXPECT_LT(std::abs(resCosNeg[j] - std_cos_neg), diff_thr * (std::abs(std_cos_neg) + flt_min * 100));
            }
        }

        // Test special values
        std::vector<LaneType> specialValues = {(LaneType) 0, (LaneType) M_PI, (LaneType) (M_PI / 2), (LaneType) INFINITY, (LaneType) -INFINITY, (LaneType) NAN};
        const int testRandNum = 10000;
        const double specialValueProbability = 0.1; // 10% chance to insert a special value
        cv::RNG_MT19937 rng;

        for (int i = 0; i < testRandNum; i++) {
            Data<R> dataRand;
            for (int j = 0; j < n; ++j) {
                if (rng.uniform(0.f, 1.f) <= specialValueProbability) {
                    // Insert a special value
                    int specialValueIndex = rng.uniform(0, (int) specialValues.size());
                    dataRand[j] = specialValues[specialValueIndex];
                } else {
                    // Generate uniform random data in [-1000, 1000]
                    dataRand[j] = (LaneType) rng.uniform(-1000, 1000);
                }
            }

            // Compare with std::sin and std::cos
            R x = dataRand, s, c;
            v_sincos(x, s, c);
            Data<R> resSin = s, resCos = c;
            for (int j = 0; j < n; ++j) {
                SCOPED_TRACE(cv::format("Random test value: %lf", (double) dataRand[j]));
                LaneType std_sin = (LaneType) std::sin(dataRand[j]);
                LaneType std_cos = (LaneType) std::cos(dataRand[j]);
                // input NaN, +INF, -INF -> output NaN
                if (std::isnan(dataRand[j]) || std::isinf(dataRand[j])) {
                    EXPECT_TRUE(std::isnan(resSin[j]));
                    EXPECT_TRUE(std::isnan(resCos[j]));
                } else if(dataRand[j] == 0) {
                    // sin(0) -> 0, cos(0) -> 1
                    EXPECT_EQ(resSin[j], 0);
                    EXPECT_EQ(resCos[j], 1);
                } else {
                    EXPECT_LT(std::abs(resSin[j] - std_sin), diff_thr * (std::abs(std_sin) + flt_min * 100));
                    EXPECT_LT(std::abs(resCos[j] - std_cos), diff_thr * (std::abs(std_cos) + flt_min * 100));
                }
            }
        }
    }

    TheTest &test_sincos_fp16() {
#if CV_SIMD_FP16
        hfloat flt16_min;
        uint16_t flt16_min_hex = 0x0400;
        std::memcpy(&flt16_min, &flt16_min_hex, sizeof(hfloat));
        __test_sincos((hfloat) 1e-3, flt16_min);
#endif
        return *this;
    }

    TheTest &test_sincos_fp32() {
        __test_sincos(1e-6f, FLT_MIN);
        return *this;
    }

    TheTest &test_sincos_fp64() {
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
        __test_sincos(1e-11, DBL_MIN);
#endif
        return *this;
    }
};

#define DUMP_ENTRY(type) printf("SIMD%d: %s\n", 8*VTraits<v_uint8>::vlanes(), CV__TRACE_FUNCTION);
//=============  8-bit integer =====================================================================

void test_hal_intrin_uint8()
{
    DUMP_ENTRY(v_uint8);
    TheTest<v_uint8>()
        .test_loadstore()
        .test_interleave()
        .test_interleave_pq()
        .test_expand()
        .test_expand_q()
        .test_addsub()
        .test_arithm_wrap()
        .test_mul()
        .test_mul_expand()
        .test_cmp()
        .test_logic()
        .test_dotprod_expand()
        .test_min_max()
        .test_absdiff()
        .test_reduce()
        .test_reduce_sad()
        .test_mask()
        .test_popcount()
        .test_pack<1>().test_pack<2>().test_pack<3>().test_pack<8>()
        .test_pack_u<1>().test_pack_u<2>().test_pack_u<3>().test_pack_u<8>()
        .test_pack_b()
        .test_unpack()
        .test_reverse()
        .test_extract<0>().test_extract<1>().test_extract<8>().test_extract<15>()
        .test_rotate<0>().test_rotate<1>().test_rotate<8>().test_rotate<15>()
        .test_extract_n<0>().test_extract_n<1>()
        .test_extract_highest()
        .test_pack_triplets()
        //.test_broadcast_element<0>().test_broadcast_element<1>()
#if CV_SIMD_WIDTH == 32
        .test_pack<9>().test_pack<10>().test_pack<13>().test_pack<15>()
        .test_pack_u<9>().test_pack_u<10>().test_pack_u<13>().test_pack_u<15>()
        .test_extract<16>().test_extract<17>().test_extract<23>().test_extract<31>()
        .test_rotate<16>().test_rotate<17>().test_rotate<23>().test_rotate<31>()
#endif
        ;
}

void test_hal_intrin_int8()
{
    DUMP_ENTRY(v_int8);
    TheTest<v_int8>()
        .test_loadstore()
        .test_interleave()
        .test_interleave_pq()
        .test_expand()
        .test_expand_q()
        .test_addsub()
        .test_arithm_wrap()
        .test_mul()
        .test_mul_expand()
        .test_cmp()
        .test_logic()
        .test_dotprod_expand()
        .test_min_max()
        .test_absdiff()
        .test_absdiffs()
        .test_abs()
        .test_reduce()
        .test_reduce_sad()
        .test_mask()
        .test_popcount()
        .test_pack<1>().test_pack<2>().test_pack<3>().test_pack<8>()
        .test_unpack()
        .test_reverse()
        .test_extract<0>().test_extract<1>().test_extract<8>().test_extract<15>()
        .test_rotate<0>().test_rotate<1>().test_rotate<8>().test_rotate<15>()
        .test_extract_n<0>().test_extract_n<1>()
        .test_extract_highest()
        .test_pack_triplets()
        //.test_broadcast_element<0>().test_broadcast_element<1>()
        ;
}

//============= 16-bit integer =====================================================================

void test_hal_intrin_uint16()
{
    DUMP_ENTRY(v_uint16);
    TheTest<v_uint16>()
        .test_loadstore()
        .test_interleave()
        .test_interleave_pq()
        .test_expand()
        .test_addsub()
        .test_arithm_wrap()
        .test_mul()
        .test_mul_expand()
        .test_mul_hi()
        .test_cmp()
        .test_shift<1>()
        .test_shift<8>()
        .test_dotprod_expand()
        .test_logic()
        .test_min_max()
        .test_absdiff()
        .test_reduce()
        .test_reduce_sad()
        .test_mask()
        .test_popcount()
        .test_pack<1>().test_pack<2>().test_pack<7>().test_pack<16>()
        .test_pack_u<1>().test_pack_u<2>().test_pack_u<7>().test_pack_u<16>()
        .test_unpack()
        .test_reverse()
        .test_extract<0>().test_extract<1>().test_extract<4>().test_extract<7>()
        .test_rotate<0>().test_rotate<1>().test_rotate<4>().test_rotate<7>()
        .test_extract_n<0>().test_extract_n<1>()
        .test_extract_highest()
        .test_pack_triplets()
        //.test_broadcast_element<0>().test_broadcast_element<1>()
        ;
}

void test_hal_intrin_int16()
{
    DUMP_ENTRY(v_int16);
    TheTest<v_int16>()
        .test_loadstore()
        .test_interleave()
        .test_interleave_pq()
        .test_expand()
        .test_addsub()
        .test_arithm_wrap()
        .test_mul()
        .test_mul_expand()
        .test_mul_hi()
        .test_cmp()
        .test_shift<1>()
        .test_shift<8>()
        .test_dotprod()
        .test_dotprod_expand()
        .test_logic()
        .test_min_max()
        .test_absdiff()
        .test_absdiffs()
        .test_abs()
        .test_reduce()
        .test_reduce_sad()
        .test_mask()
        .test_popcount()
        .test_pack<1>().test_pack<2>().test_pack<7>().test_pack<16>()
        .test_unpack()
        .test_reverse()
        .test_extract<0>().test_extract<1>().test_extract<4>().test_extract<7>()
        .test_rotate<0>().test_rotate<1>().test_rotate<4>().test_rotate<7>()
        .test_extract_n<0>().test_extract_n<1>()
        .test_extract_highest()
        .test_pack_triplets()
        //.test_broadcast_element<0>().test_broadcast_element<1>()
        ;
}

//============= 32-bit integer =====================================================================

void test_hal_intrin_uint32()
{
    DUMP_ENTRY(v_uint32);
    TheTest<v_uint32>()
        .test_loadstore()
        .test_interleave()
        // .test_interleave_pq() //not implemented in AVX
        .test_expand()
        .test_addsub()
        .test_mul()
        .test_mul_expand()
        .test_cmp()
        .test_shift<1>()
        .test_shift<8>()
        .test_logic()
        .test_min_max()
        .test_absdiff()
        .test_reduce()
        .test_reduce_sad()
        .test_mask()
        .test_popcount()
        .test_pack<1>().test_pack<2>().test_pack<15>().test_pack<32>()
        .test_unpack()
        .test_reverse()
        .test_extract<0>().test_extract<1>().test_extract<2>().test_extract<3>()
        .test_rotate<0>().test_rotate<1>().test_rotate<2>().test_rotate<3>()
        .test_extract_n<0>().test_extract_n<1>()
        .test_broadcast_element<0>().test_broadcast_element<1>()
        .test_extract_highest()
        .test_broadcast_highest()
        .test_transpose()
        .test_pack_triplets()
        ;
}

void test_hal_intrin_int32()
{
    DUMP_ENTRY(v_int32);
    TheTest<v_int32>()
        .test_loadstore()
        .test_interleave()
        // .test_interleave_pq() //not implemented in AVX
        .test_expand()
        .test_addsub()
        .test_mul()
        .test_abs()
        .test_cmp()
        .test_popcount()
        .test_shift<1>().test_shift<8>()
        .test_dotprod()
        .test_dotprod_expand_f64()
        .test_logic()
        .test_min_max()
        .test_absdiff()
        .test_reduce()
        .test_reduce_sad()
        .test_mask()
        .test_pack<1>().test_pack<2>().test_pack<15>().test_pack<32>()
        .test_unpack()
        .test_reverse()
        .test_extract<0>().test_extract<1>().test_extract<2>().test_extract<3>()
        .test_rotate<0>().test_rotate<1>().test_rotate<2>().test_rotate<3>()
        .test_extract_n<0>().test_extract_n<1>()
        .test_broadcast_element<0>().test_broadcast_element<1>()
        .test_float_cvt32()
        .test_float_cvt64()
        .test_transpose()
        .test_extract_highest()
        .test_broadcast_highest()
        .test_pack_triplets()
        ;
}

//============= 64-bit integer =====================================================================

void test_hal_intrin_uint64()
{
    DUMP_ENTRY(v_uint64);
    TheTest<v_uint64>()
        .test_loadstore()
        .test_addsub()
        .test_cmp64()
        //.test_cmp() - not declared as supported
        .test_shift<1>().test_shift<8>()
        .test_logic()
        .test_reverse()
        .test_extract<0>().test_extract<1>()
        .test_rotate<0>().test_rotate<1>()
        .test_extract_n<0>().test_extract_n<1>()
        .test_extract_highest()
        .test_popcount()
        //.test_broadcast_element<0>().test_broadcast_element<1>()
        ;
}

void test_hal_intrin_int64()
{
    DUMP_ENTRY(v_int64);
    TheTest<v_int64>()
        .test_loadstore()
        .test_addsub()
        .test_cmp64()
        //.test_cmp() - not declared as supported
        .test_shift<1>().test_shift<8>()
        .test_logic()
        .test_reverse()
        .test_extract<0>().test_extract<1>()
        .test_rotate<0>().test_rotate<1>()
        .test_extract_n<0>().test_extract_n<1>()
        .test_extract_highest()
        //.test_broadcast_element<0>().test_broadcast_element<1>()
        .test_cvt64_double()
        .test_popcount()
        ;
}

//============= Floating point =====================================================================
void test_hal_intrin_float32()
{
    DUMP_ENTRY(v_float32);
    TheTest<v_float32>()
        .test_loadstore()
        .test_interleave()
        .test_interleave_2channel()
        // .test_interleave_pq() //not implemented in AVX
        .test_addsub()
        .test_mul()
        .test_div()
        .test_abs()
        .test_cmp()
        .test_sqrt_abs()
        .test_min_max()
        .test_float_absdiff()
        .test_reduce()
        .test_reduce_sad()
        .test_mask()
        .test_unpack()
        .test_float_math()
        .test_float_cvt64()
        .test_matmul()
        .test_transpose()
        .test_reduce_sum4()
        .test_reverse()
        .test_extract<0>().test_extract<1>().test_extract<2>().test_extract<3>()
        .test_rotate<0>().test_rotate<1>().test_rotate<2>().test_rotate<3>()
        .test_extract_n<0>().test_extract_n<1>()
        .test_broadcast_element<0>().test_broadcast_element<1>()
        .test_extract_highest()
        .test_broadcast_highest()
        .test_pack_triplets()
        .test_exp_fp32()
        .test_log_fp32()
        .test_sincos_fp32()
        .test_erf_fp32()
#if CV_SIMD_WIDTH == 32
        .test_extract<4>().test_extract<5>().test_extract<6>().test_extract<7>()
        .test_rotate<4>().test_rotate<5>().test_rotate<6>().test_rotate<7>()
#endif
        ;
}

void test_hal_intrin_float64()
{
    DUMP_ENTRY(v_float64);
#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
    TheTest<v_float64>()
        .test_loadstore()
        .test_addsub()
        .test_mul()
        .test_div()
        .test_abs()
        .test_cmp()
        .test_sqrt_abs()
        .test_min_max()
        .test_float_absdiff()
        .test_mask()
        .test_unpack()
        .test_float_math()
        .test_round_pair_f64()
        .test_float_cvt32()
        .test_reverse()
        .test_extract<0>().test_extract<1>()
        .test_rotate<0>().test_rotate<1>()
        .test_extract_n<0>().test_extract_n<1>()
        .test_extract_highest()
        .test_exp_fp64()
        .test_log_fp64()
        .test_sincos_fp64()
        //.test_broadcast_element<0>().test_broadcast_element<1>()
#if CV_SIMD_WIDTH == 32
        .test_extract<2>().test_extract<3>()
        .test_rotate<2>().test_rotate<3>()
#endif
        ;
#else
    std::cout << "SKIP: CV_SIMD_64F is not available" << std::endl;
#endif
}

void test_hal_intrin_float16()
{
    DUMP_ENTRY(v_float16);
#if CV_FP16
    TheTest<v_float32>()
        .test_loadstore_fp16_f32()
#if CV_SIMD_FP16
        .test_loadstore_fp16()
        .test_float_cvt_fp16()
        .test_exp_fp16()
        .test_log_fp16()
        .test_sincos_fp16()
#endif
        ;
#else
    std::cout << "SKIP: CV_FP16 is not available" << std::endl;
#endif
}


/*#if defined(CV_CPU_DISPATCH_MODE_FP16) && CV_CPU_DISPATCH_MODE == FP16
void test_hal_intrin_float16()
{
    TheTest<v_float16>()
        .test_loadstore_fp16()
        .test_float_cvt_fp16()
        ;
}
#endif*/

#endif //CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

//CV_CPU_OPTIMIZATION_NAMESPACE_END
//}}} // namespace
