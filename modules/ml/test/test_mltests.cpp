// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

struct DatasetDesc
{
    string name;
    int resp_idx;
    int train_count;
    int cat_num;
    string type_desc;
public:
    Ptr<TrainData> load()
    {
        string filename = findDataFile(name + ".data");
        Ptr<TrainData> data = TrainData::loadFromCSV(filename, 0, resp_idx, resp_idx + 1, type_desc);
        data->setTrainTestSplit(train_count);
        data->shuffleTrainTest();
        return data;
    }
};

// see testdata/ml/protocol.txt (?)
DatasetDesc datasets[] = {
    { "mushroom", 0, 4000, 16, "cat" },
    { "adult", 14, 22561, 16, "ord[0,2,4,10-12],cat[1,3,5-9,13,14]" },
    { "vehicle", 18, 761, 4, "ord[0-17],cat[18]" },
    { "abalone", 8, 3133, 16, "ord[1-8],cat[0]" },
    { "ringnorm", 20, 300, 2, "ord[0-19],cat[20]" },
    { "spambase", 57, 3221, 3, "ord[0-56],cat[57]" },
    { "waveform", 21, 300, 3, "ord[0-20],cat[21]" },
    { "elevators", 18, 5000, 0, "ord" },
    { "letter", 16, 10000, 26, "ord[0-15],cat[16]" },
    { "twonorm", 20, 300, 3, "ord[0-19],cat[20]" },
    { "poletelecomm", 48, 2500, 0, "ord" },
};

static DatasetDesc & getDataset(const string & name)
{
    const int sz = sizeof(datasets)/sizeof(datasets[0]);
    for (int i = 0; i < sz; ++i)
    {
        DatasetDesc & desc = datasets[i];
        if (desc.name == name)
            return desc;
    }
    CV_Error(Error::StsInternal, "");
}

//==================================================================================================

// interfaces and templates

template <typename T> string modelName() { return "Unknown"; };
template <typename T> Ptr<T> tuneModel(const DatasetDesc &, Ptr<T> m) { return m; }

struct IModelFactory
{
    virtual Ptr<StatModel> createNew(const DatasetDesc &dataset) const = 0;
    virtual Ptr<StatModel> loadFromFile(const string &filename) const = 0;
    virtual string name() const = 0;
    virtual ~IModelFactory() {}
};

template <typename T>
struct ModelFactory : public IModelFactory
{
    Ptr<StatModel> createNew(const DatasetDesc &dataset) const CV_OVERRIDE
    {
        return tuneModel<T>(dataset, T::create());
    }
    Ptr<StatModel> loadFromFile(const string & filename) const CV_OVERRIDE
    {
        return T::load(filename);
    }
    string name() const CV_OVERRIDE { return modelName<T>(); }
};

// implementation

template <> string modelName<NormalBayesClassifier>() { return "NormalBayesClassifier"; }
template <> string modelName<DTrees>() { return "DTrees"; }
template <> string modelName<KNearest>() { return "KNearest"; }
template <> string modelName<RTrees>() { return "RTrees"; }
template <> string modelName<SVMSGD>() { return "SVMSGD"; }

template<> Ptr<DTrees> tuneModel<DTrees>(const DatasetDesc &dataset, Ptr<DTrees> m)
{
    m->setMaxDepth(10);
    m->setMinSampleCount(2);
    m->setRegressionAccuracy(0);
    m->setUseSurrogates(false);
    m->setCVFolds(0);
    m->setUse1SERule(false);
    m->setTruncatePrunedTree(false);
    m->setPriors(Mat());
    m->setMaxCategories(dataset.cat_num);
    return m;
}

template<> Ptr<RTrees> tuneModel<RTrees>(const DatasetDesc &dataset, Ptr<RTrees> m)
{
    m->setMaxDepth(20);
    m->setMinSampleCount(2);
    m->setRegressionAccuracy(0);
    m->setUseSurrogates(false);
    m->setPriors(Mat());
    m->setCalculateVarImportance(true);
    m->setActiveVarCount(0);
    m->setTermCriteria(TermCriteria(TermCriteria::COUNT, 100, 0.0));
    m->setMaxCategories(dataset.cat_num);
    return m;
}

template<> Ptr<SVMSGD> tuneModel<SVMSGD>(const DatasetDesc &, Ptr<SVMSGD> m)
{
    m->setSvmsgdType(SVMSGD::ASGD);
    m->setMarginType(SVMSGD::SOFT_MARGIN);
    m->setMarginRegularization(0.00001f);
    m->setInitialStepSize(0.1f);
    m->setStepDecreasingPower(0.75);
    m->setTermCriteria(TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10000, 0.00001));
    return m;
}

template <>
struct ModelFactory<Boost> : public IModelFactory
{
    ModelFactory(int boostType_) : boostType(boostType_) {}
    Ptr<StatModel> createNew(const DatasetDesc &) const CV_OVERRIDE
    {
        Ptr<Boost> m = Boost::create();
        m->setBoostType(boostType);
        m->setWeakCount(20);
        m->setWeightTrimRate(0.95);
        m->setMaxDepth(4);
        m->setUseSurrogates(false);
        m->setPriors(Mat());
        return m;
    }
    Ptr<StatModel> loadFromFile(const string &filename) const { return Boost::load(filename); }
    string name() const CV_OVERRIDE { return "Boost"; }
    int boostType;
};

template <>
struct ModelFactory<SVM> : public IModelFactory
{
    ModelFactory(int svmType_, int kernelType_, double gamma_, double c_, double nu_)
        : svmType(svmType_), kernelType(kernelType_), gamma(gamma_), c(c_), nu(nu_) {}
    Ptr<StatModel> createNew(const DatasetDesc &) const CV_OVERRIDE
    {
        Ptr<SVM> m = SVM::create();
        m->setType(svmType);
        m->setKernel(kernelType);
        m->setDegree(0);
        m->setGamma(gamma);
        m->setCoef0(0);
        m->setC(c);
        m->setNu(nu);
        m->setP(0);
        return m;
    }
    Ptr<StatModel> loadFromFile(const string &filename) const { return SVM::load(filename); }
    string name() const CV_OVERRIDE { return "SVM"; }
    int svmType;
    int kernelType;
    double gamma;
    double c;
    double nu;
};

//==================================================================================================

struct ML_Params_t
{
    Ptr<IModelFactory> factory;
    string dataset;
    float mean;
    float sigma;
};

void PrintTo(const ML_Params_t & param, std::ostream *os)
{
    *os << param.factory->name() << "_" << param.dataset;
}

ML_Params_t ML_Params_List[] = {
    { makePtr< ModelFactory<DTrees> >(), "mushroom", 0.027401f, 0.036236f },
    { makePtr< ModelFactory<DTrees> >(), "adult", 14.279000f, 0.354323f },
    { makePtr< ModelFactory<DTrees> >(), "vehicle", 29.761162f, 4.823927f },
    { makePtr< ModelFactory<DTrees> >(), "abalone", 7.297540f, 0.510058f },
    { makePtr< ModelFactory<Boost> >(Boost::REAL), "adult", 13.894001f, 0.337763f },
    { makePtr< ModelFactory<Boost> >(Boost::DISCRETE), "mushroom", 0.007274f, 0.029400f },
    { makePtr< ModelFactory<Boost> >(Boost::LOGIT), "ringnorm", 9.993943f, 0.860256f },
    { makePtr< ModelFactory<Boost> >(Boost::GENTLE), "spambase", 5.404347f, 0.581716f },
    { makePtr< ModelFactory<RTrees> >(), "waveform", 17.100641f, 0.630052f },
    { makePtr< ModelFactory<RTrees> >(), "mushroom", 0.006547f, 0.028248f },
    { makePtr< ModelFactory<RTrees> >(), "adult", 13.5129f, 0.266065f },
    { makePtr< ModelFactory<RTrees> >(), "abalone", 4.745199f, 0.282112f },
    { makePtr< ModelFactory<RTrees> >(), "vehicle", 24.964712f, 4.469287f },
    { makePtr< ModelFactory<RTrees> >(), "letter", 5.334999f, 0.261142f },
    { makePtr< ModelFactory<RTrees> >(), "ringnorm", 6.248733f, 0.904713f },
    { makePtr< ModelFactory<RTrees> >(), "twonorm", 4.506479f, 0.449739f },
    { makePtr< ModelFactory<RTrees> >(), "spambase", 5.243477f, 0.54232f },
};

typedef testing::TestWithParam<ML_Params_t> ML_Params;

TEST_P(ML_Params, accuracy)
{
    const ML_Params_t & param = GetParam();
    DatasetDesc &dataset = getDataset(param.dataset);
    Ptr<TrainData> data = dataset.load();
    ASSERT_TRUE(data);
    ASSERT_TRUE(data->getNSamples() > 0);

    Ptr<StatModel> m = param.factory->createNew(dataset);
    ASSERT_TRUE(m);
    ASSERT_TRUE(m->train(data, 0));

    float err = m->calcError(data, true, noArray());
    EXPECT_NEAR(err, param.mean, 4 * param.sigma);
}

INSTANTIATE_TEST_CASE_P(/**/, ML_Params, testing::ValuesIn(ML_Params_List));


//==================================================================================================

struct ML_SL_Params_t
{
    Ptr<IModelFactory> factory;
    string dataset;
};

void PrintTo(const ML_SL_Params_t & param, std::ostream *os)
{
    *os << param.factory->name() << "_" << param.dataset;
}

ML_SL_Params_t ML_SL_Params_List[] = {
    { makePtr< ModelFactory<NormalBayesClassifier> >(), "waveform" },
    { makePtr< ModelFactory<KNearest> >(), "waveform" },
    { makePtr< ModelFactory<KNearest> >(), "abalone" },
    { makePtr< ModelFactory<SVM> >(SVM::C_SVC, SVM::LINEAR, 1, 0.5, 0), "waveform" },
    { makePtr< ModelFactory<SVM> >(SVM::NU_SVR, SVM::RBF, 0.00225, 62.5, 0.03), "poletelecomm" },
    { makePtr< ModelFactory<DTrees> >(), "mushroom" },
    { makePtr< ModelFactory<DTrees> >(), "abalone" },
    { makePtr< ModelFactory<Boost> >(Boost::REAL), "adult" },
    { makePtr< ModelFactory<RTrees> >(), "waveform" },
    { makePtr< ModelFactory<RTrees> >(), "abalone" },
    { makePtr< ModelFactory<SVMSGD> >(), "waveform" },
};

typedef testing::TestWithParam<ML_SL_Params_t> ML_SL_Params;

TEST_P(ML_SL_Params, save_load)
{
    const ML_SL_Params_t & param = GetParam();

    DatasetDesc &dataset = getDataset(param.dataset);
    Ptr<TrainData> data = dataset.load();
    ASSERT_TRUE(data);
    ASSERT_TRUE(data->getNSamples() > 0);

    Mat responses1, responses2;
    string file1 = tempfile(".json.gz");
    string file2 = tempfile(".json.gz");
    {
        Ptr<StatModel> m = param.factory->createNew(dataset);
        ASSERT_TRUE(m);
        ASSERT_TRUE(m->train(data, 0));
        m->calcError(data, true, responses1);
        m->save(file1 + "?base64");
    }
    {
        Ptr<StatModel> m = param.factory->loadFromFile(file1);
        ASSERT_TRUE(m);
        m->calcError(data, true, responses2);
        m->save(file2 + "?base64");
    }
    EXPECT_MAT_NEAR(responses1, responses2, 0.0);
    {
        ifstream f1(file1.c_str(), std::ios_base::binary);
        ifstream f2(file2.c_str(), std::ios_base::binary);
        ASSERT_TRUE(f1.is_open() && f2.is_open());
        const size_t BUFSZ = 10000;
        vector<char> buf1(BUFSZ, 0);
        vector<char> buf2(BUFSZ, 0);
        while (true)
        {
            f1.read(&buf1[0], BUFSZ);
            f2.read(&buf2[0], BUFSZ);
            EXPECT_EQ(f1.gcount(), f2.gcount());
            EXPECT_EQ(f1.eof(), f2.eof());
            if (!f1.good() || !f2.good() || f1.gcount() != f2.gcount())
                break;
            ASSERT_EQ(buf1, buf2);
        }
    }
    remove(file1.c_str());
    remove(file2.c_str());
}

INSTANTIATE_TEST_CASE_P(/**/, ML_SL_Params, testing::ValuesIn(ML_SL_Params_List));

//==================================================================================================

TEST(TrainDataGet, layout_ROW_SAMPLE)  // Details: #12236
{
    cv::Mat test = cv::Mat::ones(150, 30, CV_32FC1) * 2;
    test.col(3) += Scalar::all(3);
    cv::Mat labels = cv::Mat::ones(150, 3, CV_32SC1) * 5;
    labels.col(1) += 1;
    cv::Ptr<cv::ml::TrainData> train_data = cv::ml::TrainData::create(test, cv::ml::ROW_SAMPLE, labels);
    train_data->setTrainTestSplitRatio(0.9);

    Mat tidx = train_data->getTestSampleIdx();
    EXPECT_EQ((size_t)15, tidx.total());

    Mat tresp = train_data->getTestResponses();
    EXPECT_EQ(15, tresp.rows);
    EXPECT_EQ(labels.cols, tresp.cols);
    EXPECT_EQ(5, tresp.at<int>(0, 0)) << tresp;
    EXPECT_EQ(6, tresp.at<int>(0, 1)) << tresp;
    EXPECT_EQ(6, tresp.at<int>(14, 1)) << tresp;
    EXPECT_EQ(5, tresp.at<int>(14, 2)) << tresp;

    Mat tsamples = train_data->getTestSamples();
    EXPECT_EQ(15, tsamples.rows);
    EXPECT_EQ(test.cols, tsamples.cols);
    EXPECT_EQ(2, tsamples.at<float>(0, 0)) << tsamples;
    EXPECT_EQ(5, tsamples.at<float>(0, 3)) << tsamples;
    EXPECT_EQ(2, tsamples.at<float>(14, test.cols - 1)) << tsamples;
    EXPECT_EQ(5, tsamples.at<float>(14, 3)) << tsamples;
}

TEST(TrainDataGet, layout_COL_SAMPLE)  // Details: #12236
{
    cv::Mat test = cv::Mat::ones(30, 150, CV_32FC1) * 3;
    test.row(3) += Scalar::all(3);
    cv::Mat labels = cv::Mat::ones(3, 150, CV_32SC1) * 5;
    labels.row(1) += 1;
    cv::Ptr<cv::ml::TrainData> train_data = cv::ml::TrainData::create(test, cv::ml::COL_SAMPLE, labels);
    train_data->setTrainTestSplitRatio(0.9);

    Mat tidx = train_data->getTestSampleIdx();
    EXPECT_EQ((size_t)15, tidx.total());

    Mat tresp = train_data->getTestResponses();  // always row-based, transposed
    EXPECT_EQ(15, tresp.rows);
    EXPECT_EQ(labels.rows, tresp.cols);
    EXPECT_EQ(5, tresp.at<int>(0, 0)) << tresp;
    EXPECT_EQ(6, tresp.at<int>(0, 1)) << tresp;
    EXPECT_EQ(6, tresp.at<int>(14, 1)) << tresp;
    EXPECT_EQ(5, tresp.at<int>(14, 2)) << tresp;


    Mat tsamples = train_data->getTestSamples();
    EXPECT_EQ(15, tsamples.cols);
    EXPECT_EQ(test.rows, tsamples.rows);
    EXPECT_EQ(3, tsamples.at<float>(0, 0)) << tsamples;
    EXPECT_EQ(6, tsamples.at<float>(3, 0)) << tsamples;
    EXPECT_EQ(6, tsamples.at<float>(3, 14)) << tsamples;
    EXPECT_EQ(3, tsamples.at<float>(test.rows - 1, 14)) << tsamples;
}

}} // namespace
