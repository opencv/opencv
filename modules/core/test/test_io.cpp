#include "test_precomp.hpp"

using namespace cv;
using namespace std;

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

static bool cvTsCheckSparse(const CvSparseMat* m1, const CvSparseMat* m2, double eps)
{
    CvSparseMatIterator it1;
    CvSparseNode* node1;
    int depth = CV_MAT_DEPTH(m1->type);
    
    if( m1->heap->active_count != m2->heap->active_count ||
       m1->dims != m2->dims || CV_MAT_TYPE(m1->type) != CV_MAT_TYPE(m2->type) )
        return false;
    
    for( node1 = cvInitSparseMatIterator( m1, &it1 );
        node1 != 0; node1 = cvGetNextSparseNode( &it1 ))
    {
        uchar* v1 = (uchar*)CV_NODE_VAL(m1,node1);
        uchar* v2 = cvPtrND( m2, CV_NODE_IDX(m1,node1), 0, 0, &node1->hashval );
        if( !v2 )
            return false;
        if( depth == CV_8U || depth == CV_8S )
        {
            if( *v1 != *v2 )
                return false;
        }
        else if( depth == CV_16U || depth == CV_16S )
        {
            if( *(ushort*)v1 != *(ushort*)v2 )
                return false;
        }
        else if( depth == CV_32S )
        {
            if( *(int*)v1 != *(int*)v2 )
                return false;
        }
        else if( depth == CV_32F )
        {
            if( fabs(*(float*)v1 - *(float*)v2) > eps*(fabs(*(float*)v2) + 1) )
                return false;
        }
        else if( fabs(*(double*)v1 - *(double*)v2) > eps*(fabs(*(double*)v2) + 1) )
            return false;
    }
    
    return true;
}


class Core_IOTest : public cvtest::BaseTest
{
public:
    Core_IOTest() {};
protected:
    void run(int)
    {
        double ranges[][2] = {{0, 256}, {-128, 128}, {0, 65536}, {-32768, 32768},
            {-1000000, 1000000}, {-10, 10}, {-10, 10}};
        RNG& rng = ts->get_rng();
        RNG rng0;
        test_case_count = 4;
        int progress = 0;
        MemStorage storage(cvCreateMemStorage(0));
        
        for( int idx = 0; idx < test_case_count; idx++ )
        {
            ts->update_context( this, idx, false );
            progress = update_progress( progress, idx, test_case_count, 0 );
            
            cvClearMemStorage(storage);
            
            bool mem = (idx % 4) >= 2;
            string filename = tempfile(idx % 2 ? ".yml" : ".xml");
            
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
                multiply(test_mat, test_mat_scale, test_mat);
            }
            
            CvSeq* seq = cvCreateSeq(test_mat.type(), (int)sizeof(CvSeq),
                                     (int)test_mat.elemSize(), storage);
            cvSeqPushMulti(seq, test_mat.data, test_mat.cols*test_mat.rows); 
            
            CvGraph* graph = cvCreateGraph( CV_ORIENTED_GRAPH,
                                           sizeof(CvGraph), sizeof(CvGraphVtx),
                                           sizeof(CvGraphEdge), storage );
            int edges[][2] = {{0,1},{1,2},{2,0},{0,3},{3,4},{4,1}};
            int i, vcount = 5, ecount = 6;
            for( i = 0; i < vcount; i++ )
                cvGraphAddVtx(graph);
            for( i = 0; i < ecount; i++ )
            {
                CvGraphEdge* edge;
                cvGraphAddEdge(graph, edges[i][0], edges[i][1], 0, &edge);
                edge->weight = (float)(i+1);
            }
            
            depth = cvtest::randInt(rng) % (CV_64F+1);
            cn = cvtest::randInt(rng) % 4 + 1;
            int sz[] = {cvtest::randInt(rng)%10+1, cvtest::randInt(rng)%10+1, cvtest::randInt(rng)%10+1};
            MatND test_mat_nd(3, sz, CV_MAKETYPE(depth, cn));
            
            rng0.fill(test_mat_nd, CV_RAND_UNI, Scalar::all(ranges[depth][0]), Scalar::all(ranges[depth][1]));
            if( depth >= CV_32F )
            {
                exp(test_mat_nd, test_mat_nd);
                MatND test_mat_scale(test_mat_nd.dims, test_mat_nd.size, test_mat_nd.type());
                rng0.fill(test_mat_scale, CV_RAND_UNI, Scalar::all(-1), Scalar::all(1));
                multiply(test_mat_nd, test_mat_scale, test_mat_nd);
            }
            
            int ssz[] = {cvtest::randInt(rng)%10+1, cvtest::randInt(rng)%10+1,
                cvtest::randInt(rng)%10+1,cvtest::randInt(rng)%10+1};
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
            fs.writeRaw("u", arr, (int)(sizeof(arr)/sizeof(arr[0])));
            
            fs << "]" << "}";
            cvWriteComment(*fs, "test comment", 0);
            
            fs.writeObj("test_seq", seq);
            fs.writeObj("test_graph",graph);
            CvGraph* graph2 = (CvGraph*)cvClone(graph);
            
            string content;
            fs.release(content);
            
            if(!fs.open(mem ? content : filename, FileStorage::READ + (mem ? FileStorage::MEMORY : 0)))
            {
                ts->printf( cvtest::TS::LOG, "filename %s can not be read\n", !mem ? filename.c_str() : content.c_str());
                ts->set_failed_test_info( cvtest::TS::FAIL_MISSING_TEST_DATA );
                return;
            }
            
            int real_int = (int)fs["test_int"];
            double real_real = (double)fs["test_real"];
            string real_string = (string)fs["test_string"];
            
            if( real_int != test_int ||
               fabs(real_real - test_real) > DBL_EPSILON*(fabs(test_real)+1) ||
               real_string != test_string )
            {
                ts->printf( cvtest::TS::LOG, "the read scalars are not correct\n" );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                return;
            }
            
            CvMat* m = (CvMat*)fs["test_mat"].readObj();
            CvMat _test_mat = test_mat;
            double max_diff = 0;
            CvMat stub1, _test_stub1;
            cvReshape(m, &stub1, 1, 0);
            cvReshape(&_test_mat, &_test_stub1, 1, 0);
            vector<int> pt;
            
            if( !m || !CV_IS_MAT(m) || m->rows != test_mat.rows || m->cols != test_mat.cols ||
               cvtest::cmpEps( Mat(&stub1), Mat(&_test_stub1), &max_diff, 0, &pt, true) < 0 )
            {
                ts->printf( cvtest::TS::LOG, "the read matrix is not correct: (%.20g vs %.20g) at (%d,%d)\n",
                            cvGetReal2D(&stub1, pt[0], pt[1]), cvGetReal2D(&_test_stub1, pt[0], pt[1]),
                            pt[0], pt[1] );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                return;
            }
            if( m && CV_IS_MAT(m))
                cvReleaseMat(&m);
            
            CvMatND* m_nd = (CvMatND*)fs["test_mat_nd"].readObj();
            CvMatND _test_mat_nd = test_mat_nd;
            
            if( !m_nd || !CV_IS_MATND(m_nd) )
            {
                ts->printf( cvtest::TS::LOG, "the read nd-matrix is not correct\n" );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                return;
            }
            
            CvMat stub, _test_stub;
            cvGetMat(m_nd, &stub, 0, 1);
            cvGetMat(&_test_mat_nd, &_test_stub, 0, 1);
            cvReshape(&stub, &stub1, 1, 0);
            cvReshape(&_test_stub, &_test_stub1, 1, 0);
            
            if( !CV_ARE_TYPES_EQ(&stub, &_test_stub) ||
               !CV_ARE_SIZES_EQ(&stub, &_test_stub) ||
               //cvNorm(&stub, &_test_stub, CV_L2) != 0 ) 
               cvtest::cmpEps( Mat(&stub1), Mat(&_test_stub1), &max_diff, 0, &pt, true) < 0 )
            {
                ts->printf( cvtest::TS::LOG, "readObj method: the read nd matrix is not correct: (%.20g vs %.20g) vs at (%d,%d)\n",
                           cvGetReal2D(&stub1, pt[0], pt[1]), cvGetReal2D(&_test_stub1, pt[0], pt[1]),
                           pt[0], pt[1] );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                return;
            }
            
            MatND mat_nd2;
            fs["test_mat_nd"] >> mat_nd2;
            CvMatND m_nd2 = mat_nd2;
            cvGetMat(&m_nd2, &stub, 0, 1);
            cvReshape(&stub, &stub1, 1, 0);
            
            if( !CV_ARE_TYPES_EQ(&stub, &_test_stub) ||
               !CV_ARE_SIZES_EQ(&stub, &_test_stub) ||
               //cvNorm(&stub, &_test_stub, CV_L2) != 0 ) 
               cvtest::cmpEps( Mat(&stub1), Mat(&_test_stub1), &max_diff, 0, &pt, true) < 0 )
            {
                ts->printf( cvtest::TS::LOG, "C++ method: the read nd matrix is not correct: (%.20g vs %.20g) vs at (%d,%d)\n",
                           cvGetReal2D(&stub1, pt[0], pt[1]), cvGetReal2D(&_test_stub1, pt[1], pt[0]),
                           pt[0], pt[1] );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                return;
            }
            
            cvRelease((void**)&m_nd);
            
            Ptr<CvSparseMat> m_s = (CvSparseMat*)fs["test_sparse_mat"].readObj();
            Ptr<CvSparseMat> _test_sparse_ = (CvSparseMat*)test_sparse_mat;
            Ptr<CvSparseMat> _test_sparse = (CvSparseMat*)cvClone(_test_sparse_);
            SparseMat m_s2;
            fs["test_sparse_mat"] >> m_s2;
            Ptr<CvSparseMat> _m_s2 = (CvSparseMat*)m_s2;
            
            if( !m_s || !CV_IS_SPARSE_MAT(m_s) ||
               !cvTsCheckSparse(m_s, _test_sparse,0) ||
               !cvTsCheckSparse(_m_s2, _test_sparse,0))
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
               (string)tl[4] != "2-502 2-029 3egegeg" ||
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
            it2 += 4;
            real_lbp_val |= (int)*it2 << 7;
            --it2;
            real_lbp_val |= (int)*it2 << 6;
            it2--;
            real_lbp_val |= (int)*it2 << 5;
            it2 -= 1;
            real_lbp_val |= (int)*it2 << 4;
            it2 += -1;
            CV_Assert( it == it2 );
            
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
            
            CvGraph* graph3 = (CvGraph*)fs["test_graph"].readObj();
            if(graph2->active_count != vcount || graph3->active_count != vcount ||
               graph2->edges->active_count != ecount || graph3->edges->active_count != ecount)
            {
                ts->printf( cvtest::TS::LOG, "the cloned or read graph have wrong number of vertices or edges\n" );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                return;
            }
            
            for( i = 0; i < ecount; i++ )
            {
                CvGraphEdge* edge2 = cvFindGraphEdge(graph2, edges[i][0], edges[i][1]);
                CvGraphEdge* edge3 = cvFindGraphEdge(graph3, edges[i][0], edges[i][1]);
                if( !edge2 || edge2->weight != (float)(i+1) ||
                   !edge3 || edge3->weight != (float)(i+1) )
                {
                    ts->printf( cvtest::TS::LOG, "the cloned or read graph do not have the edge (%d, %d)\n", edges[i][0], edges[i][1] );
                    ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                    return;
                }
            }
            
            fs.release();
            if( !mem )
                remove(filename.c_str());
        }
    }
};

TEST(Core_InputOutput, write_read_consistency) { Core_IOTest test; test.safe_run(); }


class CV_MiscIOTest : public cvtest::BaseTest
{
public:
    CV_MiscIOTest() {}
    ~CV_MiscIOTest() {}   
protected:
    void run(int)
    {
        try
        {
            FileStorage fs("test.xml", FileStorage::WRITE);
            vector<int> mi, mi2, mi3, mi4;
            vector<Mat> mv, mv2, mv3, mv4;
            Mat m(10, 9, CV_32F);
            Mat empty;
            randu(m, 0, 1);
            mi3.push_back(5);
            mv3.push_back(m);
            fs << "mi" << mi;
            fs << "mv" << mv;
            fs << "mi3" << mi3;
            fs << "mv3" << mv3;
            fs << "empty" << empty;
            fs.release();
            fs.open("test.xml", FileStorage::READ);
            fs["mi"] >> mi2;
            fs["mv"] >> mv2;
            fs["mi3"] >> mi4;
            fs["mv3"] >> mv4;
            fs["empty"] >> empty;
            CV_Assert( mi2.empty() );
            CV_Assert( mv2.empty() );
            CV_Assert( norm(mi3, mi4, CV_C) == 0 );
            CV_Assert( mv4.size() == 1 );
            double n = norm(mv3[0], mv4[0], CV_C);
            CV_Assert( n == 0 );
        }
        catch(...)
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        }
    }
};

TEST(Core_InputOutput, misc) { CV_MiscIOTest test; test.safe_run(); }

/*class CV_BigMatrixIOTest : public cvtest::BaseTest
{
public:
    CV_BigMatrixIOTest() {}
    ~CV_BigMatrixIOTest() {}   
protected:
    void run(int)
    {
        try
        {
            RNG& rng = theRNG();
            int N = 1000, M = 1200000;
            Mat mat(M, N, CV_32F);
            rng.fill(mat, RNG::UNIFORM, 0, 1);
            FileStorage fs("test.xml", FileStorage::WRITE);
            fs << "mat" << mat;
            fs.release();
        }
        catch(...)
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        }
    }
};

TEST(Core_InputOutput, huge) { CV_BigMatrixIOTest test; test.safe_run(); }
*/

