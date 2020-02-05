// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

static SparseMat cvTsGetRandomSparseMat(int dims, const int* sz, int type,
                                        int nzcount, double a, double b, RNG& rng)
{
    SparseMat m(dims, sz, type);
    int i, j;
    CV_Assert(CV_MAT_CN(type) == 1);
    for( i = 0; i < nzcount; i++ )
    {
        int idx[CV_MAX_DIM];
        for( j = 0; j < dims; j++ )
            idx[j] = cvtest::randInt(rng) % sz[j];
        double val = cvtest::randReal(rng)*(b - a) + a;
        uchar* ptr = m.ptr(idx, true, 0);
        if( type == CV_8U )
            *(uchar*)ptr = saturate_cast<uchar>(val);
        else if( type == CV_8S )
            *(schar*)ptr = saturate_cast<schar>(val);
        else if( type == CV_16U )
            *(ushort*)ptr = saturate_cast<ushort>(val);
        else if( type == CV_16S )
            *(short*)ptr = saturate_cast<short>(val);
        else if( type == CV_32S )
            *(int*)ptr = saturate_cast<int>(val);
        else if( type == CV_32F )
            *(float*)ptr = saturate_cast<float>(val);
        else
            *(double*)ptr = saturate_cast<double>(val);
    }

    return m;
}

static bool cvTsCheckSparse(const cv::SparseMat& m1, const cv::SparseMat& m2, double eps)
{
    cv::SparseMatConstIterator it1, it1_end = m1.end();
    int depth = m1.depth();

    if( m1.nzcount() != m2.nzcount() ||
       m1.dims() != m2.dims() || m1.type() != m2.type() )
        return false;

    for( it1 = m1.begin(); it1 != it1_end; ++it1 )
    {
        const cv::SparseMat::Node* node1 = it1.node();
        const uchar* v2 = m2.find<uchar>(node1->idx, (size_t*)&node1->hashval);
        if( !v2 )
            return false;
        if( depth == CV_8U || depth == CV_8S )
        {
            if( m1.value<uchar>(node1) != *v2 )
                return false;
        }
        else if( depth == CV_16U || depth == CV_16S )
        {
            if( m1.value<ushort>(node1) != *(ushort*)v2 )
                return false;
        }
        else if( depth == CV_32S )
        {
            if( m1.value<int>(node1) != *(int*)v2 )
                return false;
        }
        else if( depth == CV_32F )
        {
            if( fabs(m1.value<float>(node1) - *(float*)v2) > eps*(fabs(*(float*)v2) + 1) )
                return false;
        }
        else if( fabs(m1.value<double>(node1) - *(double*)v2) > eps*(fabs(*(double*)v2) + 1) )
            return false;
    }

    return true;
}


class Core_IOTest : public cvtest::BaseTest
{
public:
    Core_IOTest() { }
protected:
    void run(int)
    {
        double ranges[][2] = {{0, 256}, {-128, 128}, {0, 65536}, {-32768, 32768},
            {-1000000, 1000000}, {-10, 10}, {-10, 10}};
        RNG& rng = ts->get_rng();
        RNG rng0;
        int progress = 0;
        MemStorage storage(cvCreateMemStorage(0));
        const char * suffixs[3] = {".yml", ".xml", ".json" };
        test_case_count = 6;

        for( int idx = 0; idx < test_case_count; idx++ )
        {
            ts->update_context( this, idx, false );
            progress = update_progress( progress, idx, test_case_count, 0 );

            cvClearMemStorage(storage);

            bool mem = (idx % test_case_count) >= (test_case_count >> 1);
            string filename = tempfile(suffixs[idx % (test_case_count >> 1)]);

            FileStorage fs(filename, FileStorage::WRITE + (mem ? FileStorage::MEMORY : 0));

            int test_int = (int)cvtest::randInt(rng);
            double test_real = (cvtest::randInt(rng)%2?1:-1)*exp(cvtest::randReal(rng)*18-9);
            string test_string = "vw wv23424rt\"&amp;&lt;&gt;&amp;&apos;@#$@$%$%&%IJUKYILFD@#$@%$&*&() ";

            int depth = cvtest::randInt(rng) % (CV_64F+1);
            int cn = cvtest::randInt(rng) % 4 + 1;
            Mat test_mat(cvtest::randInt(rng)%30+1, cvtest::randInt(rng)%30+1, CV_MAKETYPE(depth, cn));

            rng0.fill(test_mat, CV_RAND_UNI, Scalar::all(ranges[depth][0]), Scalar::all(ranges[depth][1]));
            if( depth >= CV_32F )
            {
                exp(test_mat, test_mat);
                Mat test_mat_scale(test_mat.size(), test_mat.type());
                rng0.fill(test_mat_scale, CV_RAND_UNI, Scalar::all(-1), Scalar::all(1));
                cv::multiply(test_mat, test_mat_scale, test_mat);
            }

            depth = cvtest::randInt(rng) % (CV_64F+1);
            cn = cvtest::randInt(rng) % 4 + 1;
            int sz[] = {
                static_cast<int>(cvtest::randInt(rng)%10+1),
                static_cast<int>(cvtest::randInt(rng)%10+1),
                static_cast<int>(cvtest::randInt(rng)%10+1),
            };
            MatND test_mat_nd(3, sz, CV_MAKETYPE(depth, cn));

            rng0.fill(test_mat_nd, CV_RAND_UNI, Scalar::all(ranges[depth][0]), Scalar::all(ranges[depth][1]));
            if( depth >= CV_32F )
            {
                exp(test_mat_nd, test_mat_nd);
                MatND test_mat_scale(test_mat_nd.dims, test_mat_nd.size, test_mat_nd.type());
                rng0.fill(test_mat_scale, CV_RAND_UNI, Scalar::all(-1), Scalar::all(1));
                cv::multiply(test_mat_nd, test_mat_scale, test_mat_nd);
            }

            int ssz[] = {
                static_cast<int>(cvtest::randInt(rng)%10+1),
                static_cast<int>(cvtest::randInt(rng)%10+1),
                static_cast<int>(cvtest::randInt(rng)%10+1),
                static_cast<int>(cvtest::randInt(rng)%10+1),
            };
            SparseMat test_sparse_mat = cvTsGetRandomSparseMat(4, ssz, cvtest::randInt(rng)%(CV_64F+1),
                                                               cvtest::randInt(rng) % 10000, 0, 100, rng);

            fs << "test_int" << test_int << "test_real" << test_real << "test_string" << test_string;
            fs << "test_mat" << test_mat;
            fs << "test_mat_nd" << test_mat_nd;
            fs << "test_sparse_mat" << test_sparse_mat;

            fs << "test_list" << "[" << 0.0000000000001 << 2 << CV_PI << -3435345 << "2-502 2-029 3egegeg" <<
            "{:" << "month" << 12 << "day" << 31 << "year" << 1969 << "}" << "]";
            fs << "test_map" << "{" << "x" << 1 << "y" << 2 << "width" << 100 << "height" << 200 << "lbp" << "[:";

            const uchar arr[] = {0, 1, 1, 0, 1, 1, 0, 1};
            fs.writeRaw("u", arr, sizeof(arr));

            fs << "]" << "}";
            fs.writeComment("test comment", false);

            string content = fs.releaseAndGetString();

            if(!fs.open(mem ? content : filename, FileStorage::READ + (mem ? FileStorage::MEMORY : 0)))
            {
                ts->printf( cvtest::TS::LOG, "filename %s can not be read\n", !mem ? filename.c_str() : content.c_str());
                ts->set_failed_test_info( cvtest::TS::FAIL_MISSING_TEST_DATA );
                return;
            }

            int real_int = (int)fs["test_int"];
            double real_real = (double)fs["test_real"];
            String real_string = (String)fs["test_string"];

            if( real_int != test_int ||
               fabs(real_real - test_real) > DBL_EPSILON*(fabs(test_real)+1) ||
               real_string != test_string )
            {
                ts->printf( cvtest::TS::LOG, "the read scalars are not correct\n" );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                return;
            }

            Mat m;
            fs["test_mat"] >> m;
            double max_diff = 0;
            Mat stub1 = m.reshape(1, 0);
            Mat test_stub1 = test_mat.reshape(1, 0);
            vector<int> pt;

            if( m.empty() || m.rows != test_mat.rows || m.cols != test_mat.cols ||
               cvtest::cmpEps( stub1, test_stub1, &max_diff, 0, &pt, true) < 0 )
            {
                ts->printf( cvtest::TS::LOG, "the read matrix is not correct at (%d, %d)\n",
                            pt[0], pt[1] );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                return;
            }
            m.release();

            Mat m_nd;
            fs["test_mat_nd"] >> m_nd;

            if( m_nd.empty() )
            {
                ts->printf( cvtest::TS::LOG, "the read nd-matrix is not correct\n" );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                return;
            }

            stub1 = m_nd.reshape(1, 0);
            test_stub1 = test_mat_nd.reshape(1, 0);

            if( stub1.type() != test_stub1.type() ||
                stub1.size != test_stub1.size ||
                cvtest::cmpEps( stub1, test_stub1, &max_diff, 0, &pt, true) < 0 )
            {
                ts->printf( cvtest::TS::LOG, "readObj method: the read nd matrix is not correct at (%d,%d)\n",
                           pt[0], pt[1] );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                return;
            }
            m_nd.release();

            SparseMat m_s;
            fs["test_sparse_mat"] >> m_s;

            if( m_s.nzcount() == 0 || !cvTsCheckSparse(m_s, test_sparse_mat, 0))
            {
                ts->printf( cvtest::TS::LOG, "the read sparse matrix is not correct\n" );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                return;
            }

            FileNode tl = fs["test_list"];
            if( tl.type() != FileNode::SEQ || tl.size() != 6 ||
               fabs((double)tl[0] - 0.0000000000001) >= DBL_EPSILON ||
               (int)tl[1] != 2 ||
               fabs((double)tl[2] - CV_PI) >= DBL_EPSILON ||
               (int)tl[3] != -3435345 ||
               (String)tl[4] != "2-502 2-029 3egegeg" ||
               tl[5].type() != FileNode::MAP || tl[5].size() != 3 ||
               (int)tl[5]["month"] != 12 ||
               (int)tl[5]["day"] != 31 ||
               (int)tl[5]["year"] != 1969 )
            {
                ts->printf( cvtest::TS::LOG, "the test list is incorrect\n" );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                return;
            }

            FileNode tm = fs["test_map"];
            FileNode tm_lbp = tm["lbp"];

            int real_x = (int)tm["x"];
            int real_y = (int)tm["y"];
            int real_width = (int)tm["width"];
            int real_height = (int)tm["height"];

            int real_lbp_val = 0;
            FileNodeIterator it;
            it = tm_lbp.begin();
            real_lbp_val |= (int)*it << 0;
            ++it;
            real_lbp_val |= (int)*it << 1;
            it++;
            real_lbp_val |= (int)*it << 2;
            it += 1;
            real_lbp_val |= (int)*it << 3;
            FileNodeIterator it2(it);
            it2++;
            real_lbp_val |= (int)*it2 << 4;
            ++it2;
            real_lbp_val |= (int)*it2 << 5;
            it2 += 1;
            real_lbp_val |= (int)*it2 << 6;
            it2++;
            real_lbp_val |= (int)*it2 << 7;
            ++it2;
            CV_Assert( it2 == tm_lbp.end() );

            if( tm.type() != FileNode::MAP || tm.size() != 5 ||
               real_x != 1 ||
               real_y != 2 ||
               real_width != 100 ||
               real_height != 200 ||
               tm_lbp.type() != FileNode::SEQ ||
               tm_lbp.size() != 8 ||
               real_lbp_val != 0xb6 )
            {
                ts->printf( cvtest::TS::LOG, "the test map is incorrect\n" );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                return;
            }

            fs.release();
            if( !mem )
                remove(filename.c_str());
        }
    }
};

TEST(Core_InputOutput, write_read_consistency) { Core_IOTest test; test.safe_run(); }


struct UserDefinedType
{
    int a;
    float b;
};

static inline bool operator==(const UserDefinedType &x,
                              const UserDefinedType &y) {
    return (x.a == y.a) && (x.b == y.b);
}

static inline void write(FileStorage &fs,
                         const String&,
                         const UserDefinedType &value)
{
    fs << "{:" << "a" << value.a << "b" << value.b << "}";
}

static inline void read(const FileNode& node,
                        UserDefinedType& value,
                        const UserDefinedType& default_value
                          = UserDefinedType()) {
    if(node.empty())
    {
        value = default_value;
    }
    else
    {
        node["a"] >> value.a;
        node["b"] >> value.b;
    }
}

class CV_MiscIOTest : public cvtest::BaseTest
{
public:
    CV_MiscIOTest() {}
    ~CV_MiscIOTest() {}
protected:
    void run(int)
    {
        const char * suffix[] = {
            ".yml",
            ".xml",
            ".json"
        };
        int ncases = (int)(sizeof(suffix)/sizeof(suffix[0]));

        for ( int i = 0; i < ncases; i++ )
        {
            try
            {
                string fname = cv::tempfile(suffix[i]);
                vector<int> mi, mi2, mi3, mi4;
                vector<Mat> mv, mv2, mv3, mv4;
                vector<UserDefinedType> vudt, vudt2, vudt3, vudt4;
                Mat m(10, 9, CV_32F);
                Mat empty;
                UserDefinedType udt = { 8, 3.3f };
                randu(m, 0, 1);
                mi3.push_back(5);
                mv3.push_back(m);
                vudt3.push_back(udt);
                Point_<float> p1(1.1f, 2.2f), op1;
                Point3i p2(3, 4, 5), op2;
                Size s1(6, 7), os1;
                Complex<int> c1(9, 10), oc1;
                Rect r1(11, 12, 13, 14), or1;
                Vec<int, 5> v1(15, 16, 17, 18, 19), ov1;
                Scalar sc1(20.0, 21.1, 22.2, 23.3), osc1;
                Range g1(7, 8), og1;

                FileStorage fs(fname, FileStorage::WRITE);
                fs << "mi" << mi;
                fs << "mv" << mv;
                fs << "mi3" << mi3;
                fs << "mv3" << mv3;
                fs << "vudt" << vudt;
                fs << "vudt3" << vudt3;
                fs << "empty" << empty;
                fs << "p1" << p1;
                fs << "p2" << p2;
                fs << "s1" << s1;
                fs << "c1" << c1;
                fs << "r1" << r1;
                fs << "v1" << v1;
                fs << "sc1" << sc1;
                fs << "g1" << g1;
                fs.release();

                fs.open(fname, FileStorage::READ);
                fs["mi"] >> mi2;
                fs["mv"] >> mv2;
                fs["mi3"] >> mi4;
                fs["mv3"] >> mv4;
                fs["vudt"] >> vudt2;
                fs["vudt3"] >> vudt4;
                fs["empty"] >> empty;
                fs["p1"] >> op1;
                fs["p2"] >> op2;
                fs["s1"] >> os1;
                fs["c1"] >> oc1;
                fs["r1"] >> or1;
                fs["v1"] >> ov1;
                fs["sc1"] >> osc1;
                fs["g1"] >> og1;
                CV_Assert( mi2.empty() );
                CV_Assert( mv2.empty() );
                CV_Assert( cvtest::norm(Mat(mi3), Mat(mi4), CV_C) == 0 );
                CV_Assert( mv4.size() == 1 );
                double n = cvtest::norm(mv3[0], mv4[0], CV_C);
                CV_Assert( vudt2.empty() );
                CV_Assert( vudt3 == vudt4 );
                CV_Assert( n == 0 );
                CV_Assert( op1 == p1 );
                CV_Assert( op2 == p2 );
                CV_Assert( os1 == s1 );
                CV_Assert( oc1 == c1 );
                CV_Assert( or1 == r1 );
                CV_Assert( ov1 == v1 );
                CV_Assert( osc1 == sc1 );
                CV_Assert( og1 == g1 );
            }
            catch(...)
            {
                ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
            }
        }
    }
};

TEST(Core_InputOutput, misc) { CV_MiscIOTest test; test.safe_run(); }

#if 0 // 4+ GB of data, 40+ GB of estimated result size, it is very slow
BIGDATA_TEST(Core_InputOutput, huge)
{
    RNG& rng = theRNG();
    int N = 1000, M = 1200000;
    std::cout << "Allocating..." << std::endl;
    Mat mat(M, N, CV_32F);
    std::cout << "Initializing..." << std::endl;
    rng.fill(mat, RNG::UNIFORM, 0, 1);
    std::cout << "Writing..." << std::endl;
    {
        FileStorage fs(cv::tempfile(".xml"), FileStorage::WRITE);
        fs << "mat" << mat;
        fs.release();
    }
}
#endif

TEST(Core_globbing, accuracy)
{
    std::string patternLena    = cvtest::TS::ptr()->get_data_path() + "lena*.*";
    std::string patternLenaPng = cvtest::TS::ptr()->get_data_path() + "lena.png";

    std::vector<String> lenas, pngLenas;
    cv::glob(patternLena, lenas, true);
    cv::glob(patternLenaPng, pngLenas, true);

    ASSERT_GT(lenas.size(), pngLenas.size());

    for (size_t i = 0; i < pngLenas.size(); ++i)
    {
        ASSERT_NE(std::find(lenas.begin(), lenas.end(), pngLenas[i]), lenas.end());
    }
}

TEST(Core_InputOutput, FileStorage)
{
    std::string file = cv::tempfile(".xml");
    cv::FileStorage f(file, cv::FileStorage::WRITE);

    char arr[66];
    sprintf(arr, "sprintf is hell %d", 666);
    EXPECT_NO_THROW(f << arr);
}

TEST(Core_InputOutput, FileStorageKey)
{
    cv::FileStorage f("dummy.yml", cv::FileStorage::WRITE | cv::FileStorage::MEMORY);

    EXPECT_NO_THROW(f << "key1" << "value1");
    EXPECT_NO_THROW(f << "_key2" << "value2");
    EXPECT_NO_THROW(f << "key_3" << "value3");
    const std::string expected = "%YAML:1.0\n---\nkey1: value1\n_key2: value2\nkey_3: value3\n";
    ASSERT_STREQ(f.releaseAndGetString().c_str(), expected.c_str());
}

TEST(Core_InputOutput, FileStorageSpaces)
{
    cv::FileStorage f("dummy.yml", cv::FileStorage::WRITE | cv::FileStorage::MEMORY);
    const int valueCount = 5;
    std::string values[5] = { "", " ", " ", "  a", " some string" };
    for (size_t i = 0; i < valueCount; i++) {
        EXPECT_NO_THROW(f << cv::format("key%zu", i) << values[i]);
    }
    cv::FileStorage f2(f.releaseAndGetString(), cv::FileStorage::READ | cv::FileStorage::MEMORY);
    std::string valuesRead[valueCount];
    for (size_t i = 0; i < valueCount; i++) {
        EXPECT_NO_THROW(f2[cv::format("key%zu", i)] >> valuesRead[i]);
        ASSERT_STREQ(values[i].c_str(), valuesRead[i].c_str());
    }
    std::string fileName = cv::tempfile(".xml");
    cv::FileStorage g1(fileName, cv::FileStorage::WRITE);
    for (size_t i = 0; i < 2; i++) {
        EXPECT_NO_THROW(g1 << cv::format("key%zu", i) << values[i]);
    }
    g1.release();
    cv::FileStorage g2(fileName, cv::FileStorage::APPEND);
    for (size_t i = 2; i < valueCount; i++) {
        EXPECT_NO_THROW(g2 << cv::format("key%zu", i) << values[i]);
    }
    g2.release();
    cv::FileStorage g3(fileName, cv::FileStorage::READ);
    std::string valuesReadAppend[valueCount];
    for (size_t i = 0; i < valueCount; i++) {
        EXPECT_NO_THROW(g3[cv::format("key%zu", i)] >> valuesReadAppend[i]);
        ASSERT_STREQ(values[i].c_str(), valuesReadAppend[i].c_str());
    }
    g3.release();
}

struct data_t
{
    typedef uchar  u;
    typedef char   b;
    typedef ushort w;
    typedef short  s;
    typedef int    i;
    typedef float  f;
    typedef double d;

    /*0x00*/ u u1   ;u u2   ;                i i1                           ;
    /*0x08*/ i i2                           ;i i3                           ;
    /*0x10*/ d d1                                                           ;
    /*0x18*/ d d2                                                           ;
    /*0x20*/ i i4                           ;i required_alignment_field_for_linux32;
    /*
     * OpenCV persistence.cpp stuff expects: sizeof(data_t) = alignSize(36, sizeof(largest type = double)) = 40
     * Some compilers on some archs returns sizeof(data_t) = 36 due struct packaging UB
     */

    static inline const char * signature() {
        if (sizeof(data_t) != 40)
        {
            printf("sizeof(data_t)=%d, u1=%p u2=%p i1=%p i2=%p i3=%p d1=%p d2=%p i4=%p\n", (int)sizeof(data_t),
                    &(((data_t*)0)->u1),
                    &(((data_t*)0)->u2),
                    &(((data_t*)0)->i1),
                    &(((data_t*)0)->i2),
                    &(((data_t*)0)->i3),
                    &(((data_t*)0)->d1),
                    &(((data_t*)0)->d2),
                    &(((data_t*)0)->i4)
            );
        }
        CV_Assert(sizeof(data_t) == 40);
        CV_Assert((size_t)&(((data_t*)0)->u1) == 0x0);
        CV_Assert((size_t)&(((data_t*)0)->u2) == 0x1);
        CV_Assert((size_t)&(((data_t*)0)->i1) == 0x4);
        CV_Assert((size_t)&(((data_t*)0)->i2) == 0x8);
        CV_Assert((size_t)&(((data_t*)0)->i3) == 0xc);
        CV_Assert((size_t)&(((data_t*)0)->d1) == 0x10);
        CV_Assert((size_t)&(((data_t*)0)->d2) == 0x18);
        CV_Assert((size_t)&(((data_t*)0)->i4) == 0x20);
        return "2u3i2di";
    }
};

static void test_filestorage_basic(int write_flags, const char* suffix_name, bool testReadWrite, bool useMemory = false)
{
    const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    CV_Assert(test_info);
    std::string name = (std::string(test_info->test_case_name()) + "--" + test_info->name() + suffix_name);
    if (!testReadWrite)
        name = string(cvtest::TS::ptr()->get_data_path()) + "io/" + name;

    {
        const size_t rawdata_N = 40;
        std::vector<data_t> rawdata;

        cv::Mat _em_out, _em_in;
        cv::Mat _2d_out, _2d_in;
        cv::Mat _nd_out, _nd_in;
        cv::Mat _rd_out(8, 16, CV_64FC1), _rd_in;

        {   /* init */

            /* a normal mat */
            _2d_out = cv::Mat(10, 20, CV_8UC3, cvScalar(1U, 2U, 127U));
            for (int i = 0; i < _2d_out.rows; ++i)
                for (int j = 0; j < _2d_out.cols; ++j)
                    _2d_out.at<cv::Vec3b>(i, j)[1] = (i + j) % 256;

            /* a 4d mat */
            const int Size[] = {4, 4, 4, 4};
            cv::Mat _4d(4, Size, CV_64FC4, cvScalar(0.888, 0.111, 0.666, 0.444));
            const cv::Range ranges[] = {
                cv::Range(0, 2),
                cv::Range(0, 2),
                cv::Range(1, 2),
                cv::Range(0, 2) };
            _nd_out = _4d(ranges);

            /* a random mat */
            cv::randu(_rd_out, cv::Scalar(0.0), cv::Scalar(1.0));

            /* raw data */
            for (int i = 0; i < (int)rawdata_N; i++) {
                data_t tmp;
                tmp.u1 = 1;
                tmp.u2 = 2;
                tmp.i1 = 1;
                tmp.i2 = 2;
                tmp.i3 = 3;
                tmp.d1 = 0.1;
                tmp.d2 = 0.2;
                tmp.i4 = i;
                rawdata.push_back(tmp);
            }
        }
#ifdef GENERATE_TEST_DATA
#else
        if (testReadWrite || useMemory)
#endif
        {
            cv::FileStorage fs(name, write_flags + (useMemory ? cv::FileStorage::MEMORY : 0));
            fs << "normal_2d_mat" << _2d_out;
            fs << "normal_nd_mat" << _nd_out;
            fs << "empty_2d_mat"  << _em_out;
            fs << "random_mat"    << _rd_out;

            fs << "rawdata" << "[:";
            for (int i = 0; i < (int)rawdata_N/10; i++)
                fs.writeRaw(data_t::signature(), (const uchar*)&rawdata[i * 10], sizeof(data_t) * 10);
            fs << "]";

            size_t sz = 0;
            if (useMemory)
            {
                name = fs.releaseAndGetString();
                sz = name.size();
            }
            else
            {
                fs.release();
                std::ifstream f(name.c_str(), std::ios::in|std::ios::binary);
                f.seekg(0, std::fstream::end);
                sz = (size_t)f.tellg();
                f.close();
            }
            std::cout << "Storage size: " << sz << std::endl;
            EXPECT_LE(sz, (size_t)6000);
        }
        {   /* read */
            cv::FileStorage fs(name, cv::FileStorage::READ + (useMemory ? cv::FileStorage::MEMORY : 0));

            /* mat */
            fs["empty_2d_mat"]  >> _em_in;
            fs["normal_2d_mat"] >> _2d_in;
            fs["normal_nd_mat"] >> _nd_in;
            fs["random_mat"]    >> _rd_in;

            /* raw data */
            std::vector<data_t>(rawdata_N).swap(rawdata);
            fs["rawdata"].readRaw(data_t::signature(), (uchar*)&rawdata[0], rawdata.size() * sizeof(data_t));

            fs.release();
        }

        int errors = 0;
        for (int i = 0; i < (int)rawdata_N; i++)
        {
            EXPECT_EQ((int)rawdata[i].u1, 1);
            EXPECT_EQ((int)rawdata[i].u2, 2);
            EXPECT_EQ((int)rawdata[i].i1, 1);
            EXPECT_EQ((int)rawdata[i].i2, 2);
            EXPECT_EQ((int)rawdata[i].i3, 3);
            EXPECT_EQ(rawdata[i].d1, 0.1);
            EXPECT_EQ(rawdata[i].d2, 0.2);
            EXPECT_EQ((int)rawdata[i].i4, i);
            if (::testing::Test::HasNonfatalFailure())
            {
                printf("i = %d\n", i);
                errors++;
            }
            if (errors >= 3)
                break;
        }

        EXPECT_EQ(_em_in.rows   , _em_out.rows);
        EXPECT_EQ(_em_in.cols   , _em_out.cols);
        EXPECT_EQ(_em_in.depth(), _em_out.depth());
        EXPECT_TRUE(_em_in.empty());

        ASSERT_EQ(_2d_in.rows   , _2d_out.rows);
        ASSERT_EQ(_2d_in.cols   , _2d_out.cols);
        ASSERT_EQ(_2d_in.dims   , _2d_out.dims);
        ASSERT_EQ(_2d_in.depth(), _2d_out.depth());

        errors = 0;
        for(int i = 0; i < _2d_out.rows; ++i)
        {
            for (int j = 0; j < _2d_out.cols; ++j)
            {
                EXPECT_EQ(_2d_in.at<cv::Vec3b>(i, j), _2d_out.at<cv::Vec3b>(i, j));
                if (::testing::Test::HasNonfatalFailure())
                {
                    printf("i = %d, j = %d\n", i, j);
                    errors++;
                }
                if (errors >= 3)
                {
                    i = _2d_out.rows;
                    break;
                }
            }
        }

        ASSERT_EQ(_nd_in.rows   , _nd_out.rows);
        ASSERT_EQ(_nd_in.cols   , _nd_out.cols);
        ASSERT_EQ(_nd_in.dims   , _nd_out.dims);
        ASSERT_EQ(_nd_in.depth(), _nd_out.depth());
        EXPECT_EQ(0, cv::norm(_nd_in, _nd_out, NORM_INF));

        ASSERT_EQ(_rd_in.rows   , _rd_out.rows);
        ASSERT_EQ(_rd_in.cols   , _rd_out.cols);
        ASSERT_EQ(_rd_in.dims   , _rd_out.dims);
        ASSERT_EQ(_rd_in.depth(), _rd_out.depth());
        EXPECT_EQ(0, cv::norm(_rd_in, _rd_out, NORM_INF));
    }
}

TEST(Core_InputOutput, filestorage_base64_basic_read_XML)
{
    test_filestorage_basic(cv::FileStorage::WRITE_BASE64, ".xml", false);
}
TEST(Core_InputOutput, filestorage_base64_basic_read_YAML)
{
    test_filestorage_basic(cv::FileStorage::WRITE_BASE64, ".yml", false);
}
TEST(Core_InputOutput, DISABLED_filestorage_base64_basic_read_JSON)
{
    test_filestorage_basic(cv::FileStorage::WRITE_BASE64, ".json", false);
}
TEST(Core_InputOutput, DISABLED_filestorage_base64_basic_rw_XML)
{
    test_filestorage_basic(cv::FileStorage::WRITE_BASE64, ".xml", true);
}
TEST(Core_InputOutput, DISABLED_filestorage_base64_basic_rw_YAML)
{
    test_filestorage_basic(cv::FileStorage::WRITE_BASE64, ".yml", true);
}
TEST(Core_InputOutput, DISABLED_filestorage_base64_basic_rw_JSON)
{
    test_filestorage_basic(cv::FileStorage::WRITE_BASE64, ".json", true);
}
TEST(Core_InputOutput, DISABLED_filestorage_base64_basic_memory_XML)
{
    test_filestorage_basic(cv::FileStorage::WRITE_BASE64, ".xml", true, true);
}
TEST(Core_InputOutput, DISABLED_filestorage_base64_basic_memory_YAML)
{
    test_filestorage_basic(cv::FileStorage::WRITE_BASE64, ".yml", true, true);
}
TEST(Core_InputOutput, DISABLED_filestorage_base64_basic_memory_JSON)
{
    test_filestorage_basic(cv::FileStorage::WRITE_BASE64, ".json", true, true);
}


TEST(Core_InputOutput, filestorage_base64_valid_call)
{
    const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string basename = (test_info == 0)
        ? "filestorage_base64_valid_call"
        : (std::string(test_info->test_case_name()) + "--" + test_info->name());

    char const * filenames[] = {
        "core_io_base64_other_test.yml",
        "core_io_base64_other_test.xml",
        "core_io_base64_other_test.json",
        "core_io_base64_other_test.yml?base64",
        "core_io_base64_other_test.xml?base64",
        "core_io_base64_other_test.json?base64",
        0
    };
    char const * real_name[] = {
        "core_io_base64_other_test.yml",
        "core_io_base64_other_test.xml",
        "core_io_base64_other_test.json",
        "core_io_base64_other_test.yml",
        "core_io_base64_other_test.xml",
        "core_io_base64_other_test.json",
        0
    };

    std::vector<int> rawdata(10, static_cast<int>(0x00010203));
    cv::String str_out = "test_string";

    for (int n = 0; n < 6; n++)
    {
        char const* suffix_name = filenames[n];
        SCOPED_TRACE(suffix_name);
        std::string name = basename + '_' + suffix_name;
        std::string file_name = basename + '_' + real_name[n];

        EXPECT_NO_THROW(
        {
            cv::FileStorage fs(name, cv::FileStorage::WRITE_BASE64);

            fs << "manydata" << "[";
            fs << "[:";
            for (int i = 0; i < 10; i++)
                fs.writeRaw( "i", rawdata.data(), rawdata.size()*sizeof(rawdata[0]));
            fs << "]";
            fs << str_out;
            fs << "]";

            fs.release();
        });

        {
            cv::FileStorage fs(file_name, cv::FileStorage::READ);
            std::vector<int> data_in(rawdata.size());
            fs["manydata"][0].readRaw("i", (uchar *)data_in.data(), data_in.size() * sizeof(data_in[0]));
            EXPECT_TRUE(fs["manydata"][0].isSeq());
            EXPECT_TRUE(std::equal(rawdata.begin(), rawdata.end(), data_in.begin()));
            cv::String str_in;
            fs["manydata"][1] >> str_in;
            EXPECT_TRUE(fs["manydata"][1].isString());
            EXPECT_EQ(str_in, str_out);
            fs.release();
        }

        EXPECT_NO_THROW(
        {
            cv::FileStorage fs(name, cv::FileStorage::WRITE);

            fs << "manydata" << "[";
            fs << str_out;
            fs << "[";
            for (int i = 0; i < 10; i++)
                fs.writeRaw("i", rawdata.data(), rawdata.size()*sizeof(rawdata[0]));
            fs << "]";
            fs << "]";

            fs.release();
        });

        {
            cv::FileStorage fs(file_name, cv::FileStorage::READ);
            cv::String str_in;
            fs["manydata"][0] >> str_in;
            EXPECT_TRUE(fs["manydata"][0].isString());
            EXPECT_EQ(str_in, str_out);
            std::vector<int> data_in(rawdata.size());
            fs["manydata"][1].readRaw("i", (uchar *)data_in.data(), data_in.size() * sizeof(data_in[0]));
            EXPECT_TRUE(fs["manydata"][1].isSeq());
            EXPECT_TRUE(std::equal(rawdata.begin(), rawdata.end(), data_in.begin()));
            fs.release();
        }

        EXPECT_EQ(0, remove(file_name.c_str()));
    }
}

TEST(Core_InputOutput, filestorage_base64_invalid_call)
{
    const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string basename = (test_info == 0)
        ? "filestorage_base64_invalid_call"
        : (std::string(test_info->test_case_name()) + "--" + test_info->name());

    char const * filenames[] = {
        "core_io_base64_other_test.yml",
        "core_io_base64_other_test.xml",
        "core_io_base64_other_test.json",
        0
    };

    for (char const ** ptr = filenames; *ptr; ptr++)
    {
        char const * suffix_name = *ptr;
        std::string name = basename + '_' + suffix_name;

        EXPECT_NO_THROW({
            cv::FileStorage fs(name, cv::FileStorage::WRITE);
            fs << "rawdata" << "[";
            fs << "[:";
        });

        EXPECT_NO_THROW({
            cv::FileStorage fs(name, cv::FileStorage::WRITE);
            fs << "rawdata" << "[";
            fs << "[:";
            fs.writeRaw("u", name.c_str(), 1);
        });

        remove(name.c_str());
    }
}

TEST(Core_InputOutput, filestorage_yml_vec2i)
{
    const std::string file_name = "vec2i.yml";
    cv::Vec2i vec(2, 1), ovec;

    /* write */
    {
        cv::FileStorage fs(file_name, cv::FileStorage::WRITE);
        fs << "prms0" << "{" << "vec0" << vec << "}";
        fs.release();
    }

    /* read */
    {
        cv::FileStorage fs(file_name, cv::FileStorage::READ);
        fs["prms0"]["vec0"] >> ovec;
        fs.release();
    }

    EXPECT_EQ(vec(0), ovec(0));
    EXPECT_EQ(vec(1), ovec(1));

    remove(file_name.c_str());
}

TEST(Core_InputOutput, filestorage_json_comment)
{
    String mem_str =
        "{ /* comment */\n"
        "  \"key\": \"value\"\n"
        "  /************\n"
        "   * multiline comment\n"
        "   ************/\n"
        "  // 233\n"
        "  // \n"
        "}\n"
        ;

    String str;

    EXPECT_NO_THROW(
    {
        cv::FileStorage fs(mem_str, cv::FileStorage::READ | cv::FileStorage::MEMORY);
        fs["key"] >> str;
        fs.release();
    });

    EXPECT_EQ(str, String("value"));
}

TEST(Core_InputOutput, filestorage_utf8_bom)
{
    EXPECT_NO_THROW(
    {
        String content ="\xEF\xBB\xBF<?xml version=\"1.0\"?>\n<opencv_storage>\n</opencv_storage>\n";
        cv::FileStorage fs(content, cv::FileStorage::READ | cv::FileStorage::MEMORY);
        fs.release();
    });
    EXPECT_NO_THROW(
    {
        String content ="\xEF\xBB\xBF%YAML:1.0\n";
        cv::FileStorage fs(content, cv::FileStorage::READ | cv::FileStorage::MEMORY);
        fs.release();
    });
    EXPECT_NO_THROW(
    {
        String content ="\xEF\xBB\xBF{\n}\n";
        cv::FileStorage fs(content, cv::FileStorage::READ | cv::FileStorage::MEMORY);
        fs.release();
    });
}

TEST(Core_InputOutput, filestorage_vec_vec_io)
{
    std::vector<std::vector<Mat> > outputMats(3);
    for(size_t i = 0; i < outputMats.size(); i++)
    {
        outputMats[i].resize(i+1);
        for(size_t j = 0; j < outputMats[i].size(); j++)
        {
            outputMats[i][j] = Mat::eye((int)i + 1, (int)i + 1, CV_8U);
        }
    }

    String fileName = "vec_vec_io_test.";

    std::vector<String> formats;
    formats.push_back("xml");
    formats.push_back("yml");
    formats.push_back("json");

    for(size_t i = 0; i < formats.size(); i++)
    {
        FileStorage writer(fileName + formats[i], FileStorage::WRITE);
        writer << "vecVecMat" << outputMats;
        writer.release();

        FileStorage reader(fileName + formats[i], FileStorage::READ);
        std::vector<std::vector<Mat> > testMats;
        reader["vecVecMat"] >> testMats;

        ASSERT_EQ(testMats.size(), testMats.size());

        for(size_t j = 0; j < testMats.size(); j++)
        {
            ASSERT_EQ(testMats[j].size(), outputMats[j].size());

            for(size_t k = 0; k < testMats[j].size(); k++)
            {
                ASSERT_TRUE(cvtest::norm(outputMats[j][k] - testMats[j][k], NORM_INF) == 0);
            }
        }

        reader.release();
        remove((fileName + formats[i]).c_str());
    }
}

TEST(Core_InputOutput, filestorage_yaml_advanvced_type_heading)
{
    String content = "%YAML:1.0\n cameraMatrix: !<tag:yaml.org,2002:opencv-matrix>\n"
            "   rows: 1\n"
            "   cols: 1\n"
            "   dt: d\n"
            "   data: [ 1. ]";

    cv::FileStorage fs(content, cv::FileStorage::READ | cv::FileStorage::MEMORY);

    cv::Mat inputMatrix;
    cv::Mat actualMatrix = cv::Mat::eye(1, 1, CV_64F);
    fs["cameraMatrix"] >> inputMatrix;

    ASSERT_EQ(cv::norm(inputMatrix, actualMatrix, NORM_INF), 0.);
}

TEST(Core_InputOutput, filestorage_matx_io)
{
    Matx33d matxTest(1.234, 2, 3, 4, 5, 6, 7, 8, 9.876);

    FileStorage writer("", FileStorage::WRITE | FileStorage::MEMORY);
    writer << "matxTest" << matxTest;
    String content = writer.releaseAndGetString();

    FileStorage reader(content, FileStorage::READ | FileStorage::MEMORY);
    Matx33d matxTestRead;
    reader["matxTest"] >> matxTestRead;
    ASSERT_TRUE( cv::norm(matxTest, matxTestRead, NORM_INF) == 0 );

    reader.release();
}

TEST(Core_InputOutput, filestorage_matx_io_size_mismatch)
{
    Matx32d matxTestWrongSize(1, 2, 3, 4, 5, 6);

    FileStorage writer("", FileStorage::WRITE | FileStorage::MEMORY);
    writer << "matxTestWrongSize" << matxTestWrongSize;
    String content = writer.releaseAndGetString();

    FileStorage reader(content, FileStorage::READ | FileStorage::MEMORY);
    Matx33d matxTestRead;
    try
    {
        reader["matxTestWrongSize"] >> matxTestRead;
        FAIL() << "wrong size matrix read but no exception thrown";
    }
    catch (const std::exception&)
    {
    }

    reader.release();
}

TEST(Core_InputOutput, filestorage_matx_io_with_mat)
{
    Mat normalMat = Mat::eye(3, 3, CV_64F);

    FileStorage writer("", FileStorage::WRITE | FileStorage::MEMORY);
    writer << "normalMat" << normalMat;
    String content = writer.releaseAndGetString();

    FileStorage reader(content, FileStorage::READ | FileStorage::MEMORY);
    Matx33d matxTestRead;
    reader["normalMat"] >> matxTestRead;
    ASSERT_TRUE( cv::norm(Mat::eye(3, 3, CV_64F), matxTestRead, NORM_INF) == 0 );

    reader.release();
}

TEST(Core_InputOutput, filestorage_keypoints_vec_vec_io)
{
    vector<vector<KeyPoint> > kptsVec;
    vector<KeyPoint> kpts;
    kpts.push_back(KeyPoint(0, 0, 1.1f));
    kpts.push_back(KeyPoint(1, 1, 1.1f));
    kptsVec.push_back(kpts);
    kpts.clear();
    kpts.push_back(KeyPoint(0, 0, 1.1f, 10.1f, 34.5f, 10, 11));
    kptsVec.push_back(kpts);

    FileStorage writer("", FileStorage::WRITE + FileStorage::MEMORY + FileStorage::FORMAT_XML);
    writer << "keypoints" << kptsVec;
    String content = writer.releaseAndGetString();

    FileStorage reader(content, FileStorage::READ + FileStorage::MEMORY);
    vector<vector<KeyPoint> > readKptsVec;
    reader["keypoints"] >> readKptsVec;

    ASSERT_EQ(kptsVec.size(), readKptsVec.size());

    for(size_t i = 0; i < kptsVec.size(); i++)
    {
        ASSERT_EQ(kptsVec[i].size(), readKptsVec[i].size());
        for(size_t j = 0; j < kptsVec[i].size(); j++)
        {
            ASSERT_FLOAT_EQ(kptsVec[i][j].pt.x, readKptsVec[i][j].pt.x);
            ASSERT_FLOAT_EQ(kptsVec[i][j].pt.y, readKptsVec[i][j].pt.y);
            ASSERT_FLOAT_EQ(kptsVec[i][j].angle, readKptsVec[i][j].angle);
            ASSERT_FLOAT_EQ(kptsVec[i][j].size, readKptsVec[i][j].size);
            ASSERT_FLOAT_EQ(kptsVec[i][j].response, readKptsVec[i][j].response);
            ASSERT_EQ(kptsVec[i][j].octave, readKptsVec[i][j].octave);
            ASSERT_EQ(kptsVec[i][j].class_id, readKptsVec[i][j].class_id);
        }
    }
}

TEST(Core_InputOutput, FileStorage_DMatch)
{
    cv::FileStorage fs("dmatch.yml", cv::FileStorage::WRITE | cv::FileStorage::MEMORY);

    cv::DMatch d(1, 2, 3, -1.5f);

    EXPECT_NO_THROW(fs << "d" << d);
    cv::String fs_result = fs.releaseAndGetString();
#if defined _MSC_VER && _MSC_VER <= 1800 /* MSVC 2013 and older */
    EXPECT_STREQ(fs_result.c_str(), "%YAML:1.0\n---\nd: [ 1, 2, 3, -1.5000000000000000e+000 ]\n");
#else
    EXPECT_STREQ(fs_result.c_str(), "%YAML:1.0\n---\nd: [ 1, 2, 3, -1.5000000000000000e+00 ]\n");
#endif

    cv::FileStorage fs_read(fs_result, cv::FileStorage::READ | cv::FileStorage::MEMORY);

    cv::DMatch d_read;
    ASSERT_NO_THROW(fs_read["d"] >> d_read);

    EXPECT_EQ(d.queryIdx, d_read.queryIdx);
    EXPECT_EQ(d.trainIdx, d_read.trainIdx);
    EXPECT_EQ(d.imgIdx, d_read.imgIdx);
    EXPECT_EQ(d.distance, d_read.distance);
}

TEST(Core_InputOutput, FileStorage_DMatch_vector)
{
    cv::FileStorage fs("dmatch.yml", cv::FileStorage::WRITE | cv::FileStorage::MEMORY);

    cv::DMatch d1(1, 2, 3, -1.5f);
    cv::DMatch d2(2, 3, 4, 1.5f);
    cv::DMatch d3(3, 2, 1, 0.5f);
    std::vector<cv::DMatch> dv;
    dv.push_back(d1);
    dv.push_back(d2);
    dv.push_back(d3);

    EXPECT_NO_THROW(fs << "dv" << dv);
    cv::String fs_result = fs.releaseAndGetString();
#if defined _MSC_VER && _MSC_VER <= 1800 /* MSVC 2013 and older */
    EXPECT_STREQ(fs_result.c_str(),
"%YAML:1.0\n"
"---\n"
"dv:\n"
"   - [ 1, 2, 3, -1.5000000000000000e+000 ]\n"
"   - [ 2, 3, 4, 1.5000000000000000e+000 ]\n"
"   - [ 3, 2, 1, 5.0000000000000000e-001 ]\n"
);
#else
    EXPECT_STREQ(fs_result.c_str(),
"%YAML:1.0\n"
"---\n"
"dv:\n"
"   - [ 1, 2, 3, -1.5000000000000000e+00 ]\n"
"   - [ 2, 3, 4, 1.5000000000000000e+00 ]\n"
"   - [ 3, 2, 1, 5.0000000000000000e-01 ]\n"
);
#endif

    cv::FileStorage fs_read(fs_result, cv::FileStorage::READ | cv::FileStorage::MEMORY);

    std::vector<cv::DMatch> dv_read;
    ASSERT_NO_THROW(fs_read["dv"] >> dv_read);

    ASSERT_EQ(dv.size(), dv_read.size());
    for (size_t i = 0; i < dv.size(); i++)
    {
        EXPECT_EQ(dv[i].queryIdx, dv_read[i].queryIdx);
        EXPECT_EQ(dv[i].trainIdx, dv_read[i].trainIdx);
        EXPECT_EQ(dv[i].imgIdx, dv_read[i].imgIdx);
        EXPECT_EQ(dv[i].distance, dv_read[i].distance);
    }
}

TEST(Core_InputOutput, FileStorage_DMatch_vector_vector)
{
    cv::FileStorage fs("dmatch.yml", cv::FileStorage::WRITE | cv::FileStorage::MEMORY);

    cv::DMatch d1(1, 2, 3, -1.5f);
    cv::DMatch d2(2, 3, 4, 1.5f);
    cv::DMatch d3(3, 2, 1, 0.5f);
    std::vector<cv::DMatch> dv1;
    dv1.push_back(d1);
    dv1.push_back(d2);
    dv1.push_back(d3);

    std::vector<cv::DMatch> dv2;
    dv2.push_back(d3);
    dv2.push_back(d1);

    std::vector< std::vector<cv::DMatch> > dvv;
    dvv.push_back(dv1);
    dvv.push_back(dv2);

    EXPECT_NO_THROW(fs << "dvv" << dvv);
    cv::String fs_result = fs.releaseAndGetString();
#ifndef OPENCV_TRAITS_ENABLE_DEPRECATED
#if defined _MSC_VER && _MSC_VER <= 1800 /* MSVC 2013 and older */
    EXPECT_STREQ(fs_result.c_str(),
"%YAML:1.0\n"
"---\n"
"dvv:\n"
"   -\n"
"      - [ 1, 2, 3, -1.5000000000000000e+000 ]\n"
"      - [ 2, 3, 4, 1.5000000000000000e+000 ]\n"
"      - [ 3, 2, 1, 5.0000000000000000e-001 ]\n"
"   -\n"
"      - [ 3, 2, 1, 5.0000000000000000e-001 ]\n"
"      - [ 1, 2, 3, -1.5000000000000000e+000 ]\n"
);
#else
    EXPECT_STREQ(fs_result.c_str(),
"%YAML:1.0\n"
"---\n"
"dvv:\n"
"   -\n"
"      - [ 1, 2, 3, -1.5000000000000000e+00 ]\n"
"      - [ 2, 3, 4, 1.5000000000000000e+00 ]\n"
"      - [ 3, 2, 1, 5.0000000000000000e-01 ]\n"
"   -\n"
"      - [ 3, 2, 1, 5.0000000000000000e-01 ]\n"
"      - [ 1, 2, 3, -1.5000000000000000e+00 ]\n"
);
#endif
#endif // OPENCV_TRAITS_ENABLE_DEPRECATED

    cv::FileStorage fs_read(fs_result, cv::FileStorage::READ | cv::FileStorage::MEMORY);

    std::vector< std::vector<cv::DMatch> > dvv_read;
    ASSERT_NO_THROW(fs_read["dvv"] >> dvv_read);

    ASSERT_EQ(dvv.size(), dvv_read.size());
    for (size_t j = 0; j < dvv.size(); j++)
    {
        const std::vector<cv::DMatch>& dv = dvv[j];
        const std::vector<cv::DMatch>& dv_read = dvv_read[j];
        ASSERT_EQ(dvv.size(), dvv_read.size());
        for (size_t i = 0; i < dv.size(); i++)
        {
            EXPECT_EQ(dv[i].queryIdx, dv_read[i].queryIdx);
            EXPECT_EQ(dv[i].trainIdx, dv_read[i].trainIdx);
            EXPECT_EQ(dv[i].imgIdx, dv_read[i].imgIdx);
            EXPECT_EQ(dv[i].distance, dv_read[i].distance);
        }
    }
}


TEST(Core_InputOutput, FileStorage_KeyPoint)
{
    cv::FileStorage fs("keypoint.xml", cv::FileStorage::WRITE | cv::FileStorage::MEMORY);

    cv::KeyPoint k(Point2f(1, 2), 16, 0, 100, 1, -1);

    EXPECT_NO_THROW(fs << "k" << k);
    cv::String fs_result = fs.releaseAndGetString();
    EXPECT_STREQ(fs_result.c_str(),
"<?xml version=\"1.0\"?>\n"
"<opencv_storage>\n"
"<k>\n"
"  1. 2. 16. 0. 100. 1 -1</k>\n"
"</opencv_storage>\n"
);

    cv::FileStorage fs_read(fs_result, cv::FileStorage::READ | cv::FileStorage::MEMORY);

    cv::KeyPoint k_read;
    ASSERT_NO_THROW(fs_read["k"] >> k_read);

    EXPECT_EQ(k.pt, k_read.pt);
    EXPECT_EQ(k.size, k_read.size);
    EXPECT_EQ(k.angle, k_read.angle);
    EXPECT_EQ(k.response, k_read.response);
    EXPECT_EQ(k.octave, k_read.octave);
    EXPECT_EQ(k.class_id, k_read.class_id);
}

TEST(Core_InputOutput, FileStorage_KeyPoint_vector)
{
    cv::FileStorage fs("keypoint.xml", cv::FileStorage::WRITE | cv::FileStorage::MEMORY);

    cv::KeyPoint k1(Point2f(1, 2), 16, 0, 100, 1, -1);
    cv::KeyPoint k2(Point2f(2, 3), 16, 45, 100, 1, -1);
    cv::KeyPoint k3(Point2f(1, 2), 16, 90, 100, 1, -1);
    std::vector<cv::KeyPoint> kv;
    kv.push_back(k1);
    kv.push_back(k2);
    kv.push_back(k3);

    EXPECT_NO_THROW(fs << "kv" << kv);
    cv::String fs_result = fs.releaseAndGetString();
    EXPECT_STREQ(fs_result.c_str(),
"<?xml version=\"1.0\"?>\n"
"<opencv_storage>\n"
"<kv>\n"
"  <_>\n"
"    1. 2. 16. 0. 100. 1 -1</_>\n"
"  <_>\n"
"    2. 3. 16. 45. 100. 1 -1</_>\n"
"  <_>\n"
"    1. 2. 16. 90. 100. 1 -1</_></kv>\n"
"</opencv_storage>\n"
);

    cv::FileStorage fs_read(fs_result, cv::FileStorage::READ | cv::FileStorage::MEMORY);

    std::vector<cv::KeyPoint> kv_read;
    ASSERT_NO_THROW(fs_read["kv"] >> kv_read);

    ASSERT_EQ(kv.size(), kv_read.size());
    for (size_t i = 0; i < kv.size(); i++)
    {
        EXPECT_EQ(kv[i].pt, kv_read[i].pt);
        EXPECT_EQ(kv[i].size, kv_read[i].size);
        EXPECT_EQ(kv[i].angle, kv_read[i].angle);
        EXPECT_EQ(kv[i].response, kv_read[i].response);
        EXPECT_EQ(kv[i].octave, kv_read[i].octave);
        EXPECT_EQ(kv[i].class_id, kv_read[i].class_id);
    }
}

TEST(Core_InputOutput, FileStorage_KeyPoint_vector_vector)
{
    cv::FileStorage fs("keypoint.xml", cv::FileStorage::WRITE | cv::FileStorage::MEMORY);

    cv::KeyPoint k1(Point2f(1, 2), 16, 0, 100, 1, -1);
    cv::KeyPoint k2(Point2f(2, 3), 16, 45, 100, 1, -1);
    cv::KeyPoint k3(Point2f(1, 2), 16, 90, 100, 1, -1);
    std::vector<cv::KeyPoint> kv1;
    kv1.push_back(k1);
    kv1.push_back(k2);
    kv1.push_back(k3);

    std::vector<cv::KeyPoint> kv2;
    kv2.push_back(k3);
    kv2.push_back(k1);

    std::vector< std::vector<cv::KeyPoint> > kvv;
    kvv.push_back(kv1);
    kvv.push_back(kv2);

    EXPECT_NO_THROW(fs << "kvv" << kvv);
    cv::String fs_result = fs.releaseAndGetString();
#ifndef OPENCV_TRAITS_ENABLE_DEPRECATED
    EXPECT_STREQ(fs_result.c_str(),
"<?xml version=\"1.0\"?>\n"
"<opencv_storage>\n"
"<kvv>\n"
"  <_>\n"
"    <_>\n"
"      1. 2. 16. 0. 100. 1 -1</_>\n"
"    <_>\n"
"      2. 3. 16. 45. 100. 1 -1</_>\n"
"    <_>\n"
"      1. 2. 16. 90. 100. 1 -1</_></_>\n"
"  <_>\n"
"    <_>\n"
"      1. 2. 16. 90. 100. 1 -1</_>\n"
"    <_>\n"
"      1. 2. 16. 0. 100. 1 -1</_></_></kvv>\n"
"</opencv_storage>\n"
);
#endif //OPENCV_TRAITS_ENABLE_DEPRECATED

    cv::FileStorage fs_read(fs_result, cv::FileStorage::READ | cv::FileStorage::MEMORY);

    std::vector< std::vector<cv::KeyPoint> > kvv_read;
    ASSERT_NO_THROW(fs_read["kvv"] >> kvv_read);

    ASSERT_EQ(kvv.size(), kvv_read.size());
    for (size_t j = 0; j < kvv.size(); j++)
    {
        const std::vector<cv::KeyPoint>& kv = kvv[j];
        const std::vector<cv::KeyPoint>& kv_read = kvv_read[j];
        ASSERT_EQ(kvv.size(), kvv_read.size());
        for (size_t i = 0; i < kv.size(); i++)
        {
            EXPECT_EQ(kv[i].pt, kv_read[i].pt);
            EXPECT_EQ(kv[i].size, kv_read[i].size);
            EXPECT_EQ(kv[i].angle, kv_read[i].angle);
            EXPECT_EQ(kv[i].response, kv_read[i].response);
            EXPECT_EQ(kv[i].octave, kv_read[i].octave);
            EXPECT_EQ(kv[i].class_id, kv_read[i].class_id);
        }
    }
}


#ifdef CV__LEGACY_PERSISTENCE
TEST(Core_InputOutput, FileStorage_LEGACY_DMatch_vector)
{
    cv::DMatch d1(1, 2, 3, -1.5f);
    cv::DMatch d2(2, 3, 4, 1.5f);
    cv::DMatch d3(3, 2, 1, 0.5f);
    std::vector<cv::DMatch> dv;
    dv.push_back(d1);
    dv.push_back(d2);
    dv.push_back(d3);

    String fs_result =
"<?xml version=\"1.0\"?>\n"
"<opencv_storage>\n"
"<dv>\n"
"  1 2 3 -1.5000000000000000e+00 2 3 4 1.5000000000000000e+00 3 2 1\n"
"  5.0000000000000000e-01</dv>\n"
"</opencv_storage>\n"
    ;

    cv::FileStorage fs_read(fs_result, cv::FileStorage::READ | cv::FileStorage::MEMORY);

    std::vector<cv::DMatch> dv_read;
    ASSERT_NO_THROW(fs_read["dv"] >> dv_read);

    ASSERT_EQ(dv.size(), dv_read.size());
    for (size_t i = 0; i < dv.size(); i++)
    {
        EXPECT_EQ(dv[i].queryIdx, dv_read[i].queryIdx);
        EXPECT_EQ(dv[i].trainIdx, dv_read[i].trainIdx);
        EXPECT_EQ(dv[i].imgIdx, dv_read[i].imgIdx);
        EXPECT_EQ(dv[i].distance, dv_read[i].distance);
    }
}


TEST(Core_InputOutput, FileStorage_LEGACY_KeyPoint_vector)
{
    cv::KeyPoint k1(Point2f(1, 2), 16, 0, 100, 1, -1);
    cv::KeyPoint k2(Point2f(2, 3), 16, 45, 100, 1, -1);
    cv::KeyPoint k3(Point2f(1, 2), 16, 90, 100, 1, -1);
    std::vector<cv::KeyPoint> kv;
    kv.push_back(k1);
    kv.push_back(k2);
    kv.push_back(k3);

    cv::String fs_result =
"<?xml version=\"1.0\"?>\n"
"<opencv_storage>\n"
"<kv>\n"
"    1. 2. 16. 0. 100. 1 -1\n"
"    2. 3. 16. 45. 100. 1 -1\n"
"    1. 2. 16. 90. 100. 1 -1</kv>\n"
"</opencv_storage>\n"
    ;

    cv::FileStorage fs_read(fs_result, cv::FileStorage::READ | cv::FileStorage::MEMORY);

    std::vector<cv::KeyPoint> kv_read;
    ASSERT_NO_THROW(fs_read["kv"] >> kv_read);

    ASSERT_EQ(kv.size(), kv_read.size());
    for (size_t i = 0; i < kv.size(); i++)
    {
        EXPECT_EQ(kv[i].pt, kv_read[i].pt);
        EXPECT_EQ(kv[i].size, kv_read[i].size);
        EXPECT_EQ(kv[i].angle, kv_read[i].angle);
        EXPECT_EQ(kv[i].response, kv_read[i].response);
        EXPECT_EQ(kv[i].octave, kv_read[i].octave);
        EXPECT_EQ(kv[i].class_id, kv_read[i].class_id);
    }
}
#endif

TEST(Core_InputOutput, FileStorage_format_xml)
{
    FileStorage fs;
    fs.open("opencv_storage.xml", FileStorage::WRITE | FileStorage::MEMORY);
    EXPECT_EQ(FileStorage::FORMAT_XML, fs.getFormat());
}

TEST(Core_InputOutput, FileStorage_format_xml_gz)
{
    FileStorage fs;
    fs.open("opencv_storage.xml.gz", FileStorage::WRITE | FileStorage::MEMORY);
    EXPECT_EQ(FileStorage::FORMAT_XML, fs.getFormat());
}

TEST(Core_InputOutput, FileStorage_format_json)
{
    FileStorage fs;
    fs.open("opencv_storage.json", FileStorage::WRITE | FileStorage::MEMORY);
    EXPECT_EQ(FileStorage::FORMAT_JSON, fs.getFormat());
}

TEST(Core_InputOutput, FileStorage_format_json_gz)
{
    FileStorage fs;
    fs.open("opencv_storage.json.gz", FileStorage::WRITE | FileStorage::MEMORY);
    EXPECT_EQ(FileStorage::FORMAT_JSON, fs.getFormat());
}

TEST(Core_InputOutput, FileStorage_format_yaml)
{
    FileStorage fs;
    fs.open("opencv_storage.yaml", FileStorage::WRITE | FileStorage::MEMORY);
    EXPECT_EQ(FileStorage::FORMAT_YAML, fs.getFormat());
}

TEST(Core_InputOutput, FileStorage_format_yaml_gz)
{
    FileStorage fs;
    fs.open("opencv_storage.yaml.gz", FileStorage::WRITE | FileStorage::MEMORY);
    EXPECT_EQ(FileStorage::FORMAT_YAML, fs.getFormat());
}

TEST(Core_InputOutput, FileStorage_format_yml)
{
    FileStorage fs;
    fs.open("opencv_storage.yml", FileStorage::WRITE | FileStorage::MEMORY);
    EXPECT_EQ(FileStorage::FORMAT_YAML, fs.getFormat());
}

TEST(Core_InputOutput, FileStorage_format_yml_gz)
{
    FileStorage fs;
    fs.open("opencv_storage.yml.gz", FileStorage::WRITE | FileStorage::MEMORY);
    EXPECT_EQ(FileStorage::FORMAT_YAML, fs.getFormat());
}

TEST(Core_InputOutput, FileStorage_json_named_nodes)
{
    std::string test =
        "{ "
            "\"int_value\": -324,"
            "\"map_value\": {"
                "\"str_value\": \"mystring\""
            "},"
            "\"array\": [0.2, 0.1]"
        "}";
    FileStorage fs(test, FileStorage::READ | FileStorage::MEMORY);

    ASSERT_TRUE(fs["int_value"].isNamed());
    ASSERT_TRUE(fs["map_value"].isNamed());
    ASSERT_TRUE(fs["map_value"]["str_value"].isNamed());
    ASSERT_TRUE(fs["array"].isNamed());
    ASSERT_FALSE(fs["array"][0].isNamed());
    ASSERT_FALSE(fs["array"][1].isNamed());

    ASSERT_EQ(fs["int_value"].name(), "int_value");
    ASSERT_EQ(fs["map_value"].name(), "map_value");
    ASSERT_EQ(fs["map_value"]["str_value"].name(), "str_value");
    ASSERT_EQ(fs["array"].name(), "array");
    fs.release();
}

TEST(Core_InputOutput, FileStorage_json_bool)
{
    std::string test =
        "{ "
            "\"str_true\": \"true\","
            "\"map_value\": {"
                "\"int_value\": -33333,\n"
                "\"bool_true\": true,"
                "\"str_false\": \"false\","
            "},"
            "\"bool_false\": false, \n"
            "\"array\": [0.1, 0.2]"
        "}";
    FileStorage fs(test, FileStorage::READ | FileStorage::MEMORY);

    ASSERT_TRUE(fs["str_true"].isString());
    ASSERT_TRUE(fs["map_value"]["bool_true"].isInt());
    ASSERT_TRUE(fs["map_value"]["str_false"].isString());
    ASSERT_TRUE(fs["bool_false"].isInt());

    ASSERT_EQ((std::string)fs["str_true"], "true");
    ASSERT_EQ((int)fs["map_value"]["bool_true"], 1);
    ASSERT_EQ((std::string)fs["map_value"]["str_false"], "false");
    ASSERT_EQ((int)fs["bool_false"], 0);

    std::vector<String> keys = fs["map_value"].keys();
    ASSERT_EQ((int)keys.size(), 3);
    ASSERT_EQ(keys[0], "int_value");
    ASSERT_EQ(keys[1], "bool_true");
    ASSERT_EQ(keys[2], "str_false");
    fs.release();
}

TEST(Core_InputOutput, FileStorage_free_file_after_exception)
{
    const std::string fileName = "FileStorage_free_file_after_exception_test.yml";
    const std::string content = "%YAML:1.0\n cameraMatrix;:: !<tag:yaml.org,2002:opencv-matrix>\n";

    std::fstream testFile;
    testFile.open(fileName.c_str(), std::fstream::out);
    if(!testFile.is_open()) FAIL();
    testFile << content;
    testFile.close();

    try
    {
        FileStorage fs(fileName, FileStorage::READ + FileStorage::FORMAT_YAML);
        FAIL();
    }
    catch (const std::exception&)
    {
    }
    ASSERT_EQ(0, std::remove(fileName.c_str()));
}

TEST(Core_InputOutput, FileStorage_YAML_parse_multiple_documents)
{
    const std::string filename = "FileStorage_YAML_parse_multiple_documents.yml";
    FileStorage fs;

    fs.open(filename, FileStorage::WRITE);
    fs << "a" << 42;
    fs.release();

    fs.open(filename, FileStorage::APPEND);
    fs << "b" << 1988;
    fs.release();

    fs.open(filename, FileStorage::READ);

    EXPECT_EQ(42, (int)fs["a"]);
    EXPECT_EQ(1988, (int)fs["b"]);

    EXPECT_EQ(42, (int)fs.root(0)["a"]);
    EXPECT_TRUE(fs.root(0)["b"].empty());

    EXPECT_TRUE(fs.root(1)["a"].empty());
    EXPECT_EQ(1988, (int)fs.root(1)["b"]);

    fs.release();

    ASSERT_EQ(0, std::remove(filename.c_str()));
}

}} // namespace
