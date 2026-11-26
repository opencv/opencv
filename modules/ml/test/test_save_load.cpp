// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {


void randomFillCategories(const string & filename, Mat & input)
{
    Mat catMap;
    Mat catCount;
    Mat varIdx;
    std::vector<uchar> varTypes;

    FileStorage fs(filename, FileStorage::READ);
    FileNode root = fs.getFirstTopLevelNode();
    root["cat_map"] >> catMap;
    root["cat_count"] >> catCount;
    root["var_idx"] >> varIdx;
    root["var_type"] >> varTypes;

    int offset = 0;
    int countOffset = 0;
    uint var = 0, varCount = (uint)varTypes.size();
    if (!varIdx.empty())
        varCount = std::min(varCount, (uint)varIdx.total());
    for (; var < varCount; ++var)
    {
        if (varTypes[var] == ml::VAR_CATEGORICAL)
        {
            int size = catCount.at<int>(0, countOffset);
            for (int row = 0; row < input.rows; ++row)
            {
                int randomChosenIndex = offset + ((uint)cv::theRNG()) % size;
                int value = catMap.at<int>(0, randomChosenIndex);
                // Legacy var_type and cat_count use active-variable order;
                // prediction without COMPRESSED_INPUT takes original columns.
                const int inputVar = varIdx.empty() ? (int)var : varIdx.at<int>((int)var);
                input.at<float>(row, inputVar) = (float)value;
            }
            offset += size;
            ++countOffset;
        }
    }
}

//==================================================================================================

typedef tuple<string, string> ML_Legacy_Param;
typedef testing::TestWithParam< ML_Legacy_Param > ML_Legacy_Params;

TEST_P(ML_Legacy_Params, legacy_load)
{
    const string modelName = get<0>(GetParam());
    const string dataName = get<1>(GetParam());
    const string filename = findDataFile("legacy/" + modelName + "_" + dataName + ".xml");
    const bool isTree = modelName == CV_BOOST || modelName == CV_DTREE || modelName == CV_RTREES;

    Ptr<StatModel> model;
    if (modelName == CV_BOOST)
        model = Algorithm::load<Boost>(filename);
    else if (modelName == CV_ANN)
        model = Algorithm::load<ANN_MLP>(filename);
    else if (modelName == CV_DTREE)
        model = Algorithm::load<DTrees>(filename);
    else if (modelName == CV_NBAYES)
        model = Algorithm::load<NormalBayesClassifier>(filename);
    else if (modelName == CV_SVM)
        model = Algorithm::load<SVM>(filename);
    else if (modelName == CV_RTREES)
        model = Algorithm::load<RTrees>(filename);
    else if (modelName == CV_SVMSGD)
        model = Algorithm::load<SVMSGD>(filename);
    ASSERT_TRUE(model);

    Mat input = Mat(isTree ? 10 : 1, model->getVarCount(), CV_32F);
    cv::theRNG().fill(input, RNG::UNIFORM, 0, 40);

    if (isTree)
        randomFillCategories(filename, input);

    Mat output;
    EXPECT_NO_THROW(model->predict(input, output, StatModel::RAW_OUTPUT | (isTree ? DTrees::PREDICT_SUM : 0)));
    // just check if no internal assertions or errors thrown
}

ML_Legacy_Param param_list[] = {
    ML_Legacy_Param(CV_ANN, "waveform"),
    ML_Legacy_Param(CV_BOOST, "adult"),
    ML_Legacy_Param(CV_BOOST, "1"),
    ML_Legacy_Param(CV_BOOST, "2"),
    ML_Legacy_Param(CV_BOOST, "3"),
    ML_Legacy_Param(CV_DTREE, "abalone"),
    ML_Legacy_Param(CV_DTREE, "mushroom"),
    ML_Legacy_Param(CV_NBAYES, "waveform"),
    ML_Legacy_Param(CV_SVM, "poletelecomm"),
    ML_Legacy_Param(CV_SVM, "waveform"),
    ML_Legacy_Param(CV_RTREES, "waveform"),
    ML_Legacy_Param(CV_SVMSGD, "waveform"),
};

INSTANTIATE_TEST_CASE_P(/**/, ML_Legacy_Params, testing::ValuesIn(param_list));

TEST(ML_DTrees, load_bad_categorical_split)
{
    // Train a tree with a categorical input so the model carries a
    // categorical split serialized as an "in"/"not_in" value list.
    const int n = 40;
    Mat samples(n, 1, CV_32F);
    Mat responses(n, 1, CV_32S);
    for (int i = 0; i < n; i++)
    {
        int cat = i % 4;
        samples.at<float>(i, 0) = (float)cat;
        responses.at<int>(i, 0) = (cat == 1 || cat == 2) ? 1 : 0;
    }
    Mat varType(2, 1, CV_8U);
    varType.at<uchar>(0) = ml::VAR_CATEGORICAL;
    varType.at<uchar>(1) = ml::VAR_CATEGORICAL;

    Ptr<ml::TrainData> td = ml::TrainData::create(samples, ml::ROW_SAMPLE, responses,
                                                  noArray(), noArray(), noArray(), varType);
    Ptr<ml::DTrees> dt = ml::DTrees::create();
    dt->setMaxDepth(4);
    dt->setCVFolds(0);
    dt->setMaxCategories(4);
    dt->setMinSampleCount(1);
    dt->train(td);

    const string filename = cv::tempfile(".yml");
    dt->save(filename);

    string model;
    {
        std::ifstream in(filename.c_str());
        std::stringstream ss;
        ss << in.rdbuf();
        model = ss.str();
    }

    // Category values in the split list index a per-split subset bitmask.
    // Injecting a value larger than the number of categories used to write
    // past that bitmask; loading such a model must be rejected, not crash.
    size_t pos = model.find("in:");
    ASSERT_NE(pos, string::npos);
    size_t br = model.find('[', pos);
    ASSERT_NE(br, string::npos);
    model.insert(br + 1, "1000000,");
    {
        std::ofstream out(filename.c_str());
        out << model;
    }

    Ptr<ml::DTrees> bad;
    EXPECT_THROW(bad = ml::DTrees::load(filename), Exception);
    remove(filename.c_str());
}

static std::string dtreesToString(const Ptr<ml::DTrees>& model)
{
    FileStorage fs(".yml", FileStorage::WRITE + FileStorage::MEMORY);
    fs << model->getDefaultName() << "{";
    model->write(fs);
    fs << "}";
    return fs.releaseAndGetString();
}

static std::string replacedOnce(const std::string& s, const std::string& from, const std::string& to)
{
    size_t pos = s.find(from);
    EXPECT_NE(pos, std::string::npos) << from;
    EXPECT_EQ(s.find(from, pos + 1), std::string::npos) << from;
    if (pos == std::string::npos)
        return s;
    return s.substr(0, pos) + to + s.substr(pos + from.size());
}

// Two-class tree over three ordered inputs; with useVarIdx only inputs 0 and 2 are used.
static Ptr<ml::DTrees> trainOrderedDTrees(bool useVarIdx)
{
    const int n = 40;
    Mat samples(n, 3, CV_32F), responses(n, 1, CV_32S);
    for (int i = 0; i < n; i++)
    {
        samples.at<float>(i, 0) = (float)(i % 10);
        samples.at<float>(i, 1) = 3.f;
        samples.at<float>(i, 2) = (float)((i * 3) % 7);
        responses.at<int>(i) = (i % 10) >= 5;
    }
    Mat varType = (Mat_<uchar>(4, 1) << ml::VAR_ORDERED, ml::VAR_ORDERED, ml::VAR_ORDERED, ml::VAR_CATEGORICAL);
    Mat varIdx;
    if (useVarIdx)
        varIdx = (Mat_<int>(1, 2) << 0, 2);
    Ptr<ml::TrainData> td = ml::TrainData::create(samples, ml::ROW_SAMPLE, responses,
                                                  varIdx, noArray(), noArray(), varType);
    Ptr<ml::DTrees> dt = ml::DTrees::create();
    dt->setMaxDepth(4);
    dt->setCVFolds(0);
    dt->setMinSampleCount(1);
    dt->train(td);
    return dt;
}

// Each test trains a genuine model (the valid control) and confirms it loads and
// predicts, then alters one serialized field and confirms the invalid/malformed
// model is rejected with a cv::Exception.
TEST(ML_DTrees, load_bad_class_index)
{
    const std::string model = dtreesToString(trainOrderedDTrees(false));
    const Mat sample = (Mat_<float>(1, 3) << 9.f, 3.f, 1.f);
    Ptr<ml::DTrees> good = Algorithm::loadFromString<ml::DTrees>(model);
    ASSERT_FALSE(good.empty());
    EXPECT_EQ(1.f, good->predict(sample));

    // norm_class_idx selects the vote counter and the class label in predict()
    size_t pos = model.rfind("norm_class_idx: 1");
    ASSERT_NE(pos, std::string::npos);
    std::string bad = model;
    bad.replace(pos, sizeof("norm_class_idx: 1") - 1, "norm_class_idx: 264");
    EXPECT_THROW(Algorithm::loadFromString<ml::DTrees>(bad), Exception);
}

TEST(ML_DTrees, predict_split_on_unused_variable)
{
    const std::string model = dtreesToString(trainOrderedDTrees(true));
    const Mat sample = (Mat_<float>(1, 3) << 9.f, 3.f, 1.f);
    Ptr<ml::DTrees> good = Algorithm::loadFromString<ml::DTrees>(model);
    ASSERT_FALSE(good.empty());
    EXPECT_EQ(1.f, good->predict(sample));

    // variable 1 is not in var_idx, so it has no position in the sample
    Ptr<ml::DTrees> bad = Algorithm::loadFromString<ml::DTrees>(replacedOnce(model, "var:0,", "var:1,"));
    ASSERT_FALSE(bad.empty());
    EXPECT_THROW(bad->predict(sample), Exception);
}

TEST(ML_DTrees, predict_valid_variable_subset)
{
    Mat samples = (Mat_<float>(4, 3) << 7.f, 10.f, 0.f, 7.f, 10.f, 0.f,
                                       7.f, 10.f, 1.f, 7.f, 10.f, 1.f);
    Mat responses = (Mat_<int>(4, 1) << 0, 0, 1, 1);
    Mat varIdx = (Mat_<int>(1, 2) << 0, 2);
    Ptr<ml::TrainData> data = ml::TrainData::create(samples, ml::ROW_SAMPLE,
        responses, varIdx);
    Ptr<ml::DTrees> model = ml::DTrees::create();
    model->setMaxDepth(1);
    model->setCVFolds(0);
    model->setMinSampleCount(1);
    ASSERT_TRUE(model->train(data));
    ASSERT_FALSE(model->getSplits().empty());
    ASSERT_EQ(2, model->getSplits()[0].varIdx);
    Mat compressed(1, 2, CV_32F);
    for (int i = 0; i < samples.rows; ++i)
    {
        data->getSample(varIdx, i, compressed.ptr<float>());
        EXPECT_EQ((float)responses.at<int>(i), model->predict(samples.row(i)));
        EXPECT_EQ((float)responses.at<int>(i),
            model->predict(compressed, noArray(), ml::StatModel::COMPRESSED_INPUT));
    }
    // A short row is rejected before looking up the split's sample value,
    // in both the global and compressed coordinate systems.
    EXPECT_THROW(model->predict(samples.row(0).colRange(0, 2)), Exception);
    EXPECT_THROW(model->predict(compressed.colRange(0, 1), noArray(),
        ml::StatModel::COMPRESSED_INPUT), Exception);
}

TEST(ML_Boost, train_predict_valid_variable_subset)
{
    const int boostTypes[] = { ml::Boost::DISCRETE, ml::Boost::REAL,
                              ml::Boost::LOGIT, ml::Boost::GENTLE };
    for (int categorical = 0; categorical < 2; ++categorical)
    {
        Mat samples(40, 5, CV_32F, Scalar(-10)), responses(40, 1, CV_32S);
        for (int i = 0; i < samples.rows; ++i)
        {
            samples.at<float>(i, 1) = 0.f;
            samples.at<float>(i, 4) = (float)(i % 2);
            responses.at<int>(i) = i % 2;
        }
        Mat varIdx = (Mat_<int>(1, 2) << 1, 4);
        Mat varType(6, 1, CV_8U, Scalar(ml::VAR_ORDERED));
        varType.at<uchar>(4) = categorical ? ml::VAR_CATEGORICAL : ml::VAR_ORDERED;
        varType.at<uchar>(5) = ml::VAR_CATEGORICAL;
        Ptr<ml::TrainData> data = ml::TrainData::create(samples, ml::ROW_SAMPLE,
            responses, varIdx, noArray(), noArray(), varType);
        ASSERT_EQ(2, data->getNVars());
        ASSERT_EQ(5, data->getNAllVars());

        for (size_t b = 0; b < sizeof(boostTypes)/sizeof(boostTypes[0]); ++b)
        {
            SCOPED_TRACE(testing::Message() << "boost=" << boostTypes[b]
                         << " categorical=" << categorical);
            Ptr<ml::Boost> model = ml::Boost::create();
            model->setBoostType(boostTypes[b]);
            model->setWeakCount(3);
            model->setMaxDepth(1);
            model->setMinSampleCount(1);
            model->setWeightTrimRate(0.);
            // updateWeightsAndTrim predicts on a two-element compressed sample,
            // while the informative split has global variable index four.
            ASSERT_TRUE(model->train(data));
            ASSERT_FALSE(model->getSplits().empty());
            EXPECT_EQ(4, model->getSplits()[0].varIdx);

            Ptr<ml::Boost> loaded = Algorithm::loadFromString<ml::Boost>(dtreesToString(model));
            ASSERT_FALSE(loaded.empty());
            for (int i = 0; i < samples.rows; ++i)
            {
                EXPECT_EQ((float)responses.at<int>(i), model->predict(samples.row(i)));
                EXPECT_EQ((float)responses.at<int>(i), loaded->predict(samples.row(i)));
            }
            // Public Boost prediction still requires all original input columns.
            EXPECT_THROW(model->predict(samples.row(0).colRange(0, 2)), Exception);
            EXPECT_THROW(loaded->predict(samples.row(0).colRange(0, 2)), Exception);
        }
    }
}

// One categorical input taking the values 0..3, so cat_map holds 4 values and the
// single split reads "not_in:[ 1, 2 ]".
static std::string categoricalDTreesModel()
{
    const int n = 40;
    Mat samples(n, 1, CV_32F), responses(n, 1, CV_32S);
    for (int i = 0; i < n; i++)
    {
        int cat = i % 4;
        samples.at<float>(i, 0) = (float)cat;
        responses.at<int>(i, 0) = (cat == 1 || cat == 2) ? 1 : 0;
    }
    Mat varType = (Mat_<uchar>(2, 1) << ml::VAR_CATEGORICAL, ml::VAR_CATEGORICAL);
    Ptr<ml::TrainData> td = ml::TrainData::create(samples, ml::ROW_SAMPLE, responses,
                                                  noArray(), noArray(), noArray(), varType);
    Ptr<ml::DTrees> dt = ml::DTrees::create();
    dt->setMaxDepth(4);
    dt->setCVFolds(0);
    dt->setMaxCategories(4);
    dt->setMinSampleCount(1);
    dt->train(td);
    return dtreesToString(dt);
}

TEST(ML_DTrees, load_bad_categorical_offsets)
{
    const std::string model = categoricalDTreesModel();
    const Mat sample = (Mat_<float>(1, 1) << 1.f);
    Ptr<ml::DTrees> good = Algorithm::loadFromString<ml::DTrees>(model);
    ASSERT_FALSE(good.empty());
    EXPECT_EQ(1.f, good->predict(sample));

    // cat_ofs ranges index cat_map, which holds 4 category values here
    EXPECT_THROW(Algorithm::loadFromString<ml::DTrees>(replacedOnce(model, "cat_ofs: [ 0, 4 ]", "cat_ofs: [ 0, 9 ]")), Exception);
    EXPECT_THROW(Algorithm::loadFromString<ml::DTrees>(replacedOnce(model, "cat_ofs: [ 0, 4 ]", "cat_ofs: [ 3, 1 ]")), Exception);
    EXPECT_THROW(Algorithm::loadFromString<ml::DTrees>(replacedOnce(model, "cat_ofs: [ 0, 4 ]", "cat_ofs: [ -1, 4 ]")), Exception);
}

TEST(ML_DTrees, load_empty_categorical_range)
{
    const std::string model = categoricalDTreesModel();
    const Mat sample = (Mat_<float>(1, 1) << 1.f);

    // a split without a category list is valid on a non-empty range
    const std::string noList = replacedOnce(model, ", not_in:[ 1, 2 ]", "");
    Ptr<ml::DTrees> good = Algorithm::loadFromString<ml::DTrees>(noList);
    ASSERT_FALSE(good.empty());
    EXPECT_NO_THROW(good->predict(sample));

    // a categorical variable cannot have an empty range, at the end of cat_map or elsewhere
    EXPECT_THROW(Algorithm::loadFromString<ml::DTrees>(replacedOnce(model, "cat_ofs: [ 0, 4 ]", "cat_ofs: [ 4, 4 ]")), Exception);
    EXPECT_THROW(Algorithm::loadFromString<ml::DTrees>(replacedOnce(noList, "cat_ofs: [ 0, 4 ]", "cat_ofs: [ 4, 4 ]")), Exception);
    EXPECT_THROW(Algorithm::loadFromString<ml::DTrees>(replacedOnce(noList, "cat_ofs: [ 0, 4 ]", "cat_ofs: [ 0, 0 ]")), Exception);
}

TEST(ML_DTrees, load_legacy_bad_var_idx_and_cat_count)
{
    // Pre-3.0 layout (no "format" field): var_type lists the used variables only,
    // var_idx maps them to the var_all inputs, cat_count gives each categorical
    // variable's number of entries in cat_map.
    const std::string header =
        "%YAML:1.0\n---\nmy_tree:\n"
        "   is_classifier: 0\n   var_all: 2\n   var_count: 2\n   var_type: [ 1 ]\n   cat_map: [ 0, 1 ]\n";
    const std::string nodes = "   nodes:\n      - { depth:0, value:1.5 }\n";
    const Mat sample = (Mat_<float>(1, 2) << 0.f, 0.f);

    Ptr<ml::DTrees> good = Algorithm::loadFromString<ml::DTrees>(
        header + "   var_idx: [ 1 ]\n   cat_count: [ 2 ]\n" + nodes);
    ASSERT_FALSE(good.empty());
    EXPECT_EQ(1.5f, good->predict(sample));

    EXPECT_THROW(Algorithm::loadFromString<ml::DTrees>(header + "   var_idx: [ 3 ]\n   cat_count: [ 2 ]\n" + nodes), Exception);
    EXPECT_THROW(Algorithm::loadFromString<ml::DTrees>(header + "   var_idx: [ 1 ]\n" + nodes), Exception);
}

// Minimal regression tree in the legacy layout with one ordered input and the given "nodes" entries.
static std::string tinyTreeModel(const std::string& nodeEntries)
{
    return std::string("%YAML:1.0\n---\nmy_tree:\n"
        "   is_classifier: 0\n   var_all: 1\n   var_count: 1\n   var_type: [ 0 ]\n   nodes:\n") + nodeEntries;
}

static const char* const kTinySplit = "      - { depth:0, value:1., splits:[ { var:0, quality:1., le:5. } ] }\n";
static const char* const kTinyLeftLeaf = "      - { depth:1, value:2. }\n";
static const char* const kTinyRightLeaf = "      - { depth:1, value:3. }\n";

// predictTrees() follows node.left or node.right of every split node, and readTree() derives both
// from the order of the "nodes" list: a split node whose children are missing keeps -1 and
// predict() would index nodes[-1].
TEST(ML_DTrees, load_incomplete_tree)
{
    const Mat low = (Mat_<float>(1, 1) << 1.f), high = (Mat_<float>(1, 1) << 9.f);
    const std::string split = kTinySplit, left = kTinyLeftLeaf, right = kTinyRightLeaf;

    // valid control: a split with both children, and a single leaf
    Ptr<ml::DTrees> good = Algorithm::loadFromString<ml::DTrees>(tinyTreeModel(split + left + right));
    ASSERT_FALSE(good.empty());
    EXPECT_EQ(2.f, good->predict(low));
    EXPECT_EQ(3.f, good->predict(high));
    Ptr<ml::DTrees> leaf = Algorithm::loadFromString<ml::DTrees>(tinyTreeModel(right));
    ASSERT_FALSE(leaf.empty());
    EXPECT_EQ(3.f, leaf->predict(low));

    // a split node with no children, a split node with only a left child (the right one is -1)
    EXPECT_THROW(Algorithm::loadFromString<ml::DTrees>(tinyTreeModel(split)), Exception);
    EXPECT_THROW(Algorithm::loadFromString<ml::DTrees>(tinyTreeModel(split + left)), Exception);
    // a nested split that is cut short: an inner split with only a left leaf under a root without a right child,
    // and a root with two children whose right child is a split without children
    EXPECT_THROW(Algorithm::loadFromString<ml::DTrees>(tinyTreeModel(split + split + left)), Exception);
    EXPECT_THROW(Algorithm::loadFromString<ml::DTrees>(tinyTreeModel(split + left + split)), Exception);
    // nothing after a complete tree: the extra node would silently replace the root
    EXPECT_THROW(Algorithm::loadFromString<ml::DTrees>(tinyTreeModel(split + left + right + right)), Exception);
    EXPECT_THROW(Algorithm::loadFromString<ml::DTrees>(tinyTreeModel(right + right)), Exception);
    // an empty node list has no root
    EXPECT_THROW(Algorithm::loadFromString<ml::DTrees>(tinyTreeModel("      []\n")), Exception);
}

TEST(ML_DTrees, load_truncated_trained_tree)
{
    const std::string model = dtreesToString(trainOrderedDTrees(false));
    const Mat sample = (Mat_<float>(1, 3) << 9.f, 3.f, 1.f);
    Ptr<ml::DTrees> good = Algorithm::loadFromString<ml::DTrees>(model);
    ASSERT_FALSE(good.empty());
    EXPECT_EQ(1.f, good->predict(sample));

    // the "nodes" list holds three entries, each opened by a line "      -": the split, then its two leaves
    const std::string mark = "\n      -\n";
    std::vector<size_t> starts;
    for (size_t pos = model.find(mark); pos != std::string::npos; pos = model.find(mark, pos + 1))
        starts.push_back(pos + 1);
    ASSERT_EQ((size_t)3, starts.size());
    const std::string head = model.substr(0, starts[0]);
    std::string entry[3];
    for (size_t i = 0; i < 3; i++)
    {
        entry[i] = model.substr(starts[i], (i + 1 < 3 ? starts[i + 1] : model.size()) - starts[i]);
        if (entry[i].empty() || entry[i].back() != '\n')
            entry[i] += "\n";
    }

    // the same entries put back together are the valid control
    Ptr<ml::DTrees> again = Algorithm::loadFromString<ml::DTrees>(head + entry[0] + entry[1] + entry[2]);
    ASSERT_FALSE(again.empty());
    EXPECT_EQ(1.f, again->predict(sample));

    // the right leaf cut off (the split keeps -1 as its right child), then both leaves cut off
    EXPECT_THROW(Algorithm::loadFromString<ml::DTrees>(head + entry[0] + entry[1]), Exception);
    EXPECT_THROW(Algorithm::loadFromString<ml::DTrees>(head + entry[0]), Exception);
    // a node after the end of the complete tree
    EXPECT_THROW(Algorithm::loadFromString<ml::DTrees>(head + entry[0] + entry[1] + entry[2] + entry[2]), Exception);
}

// Three ordered inputs; only input 2 decides the class, so the single split is on variable 2.
static std::string missingSubstDTreesModel()
{
    const int n = 40;
    Mat samples(n, 3, CV_32F), responses(n, 1, CV_32S);
    for (int i = 0; i < n; i++)
    {
        samples.at<float>(i, 0) = 3.f;
        samples.at<float>(i, 1) = 3.f;
        samples.at<float>(i, 2) = (float)(i % 10);
        responses.at<int>(i) = (i % 10) >= 5;
    }
    Mat varType = (Mat_<uchar>(4, 1) << ml::VAR_ORDERED, ml::VAR_ORDERED, ml::VAR_ORDERED, ml::VAR_CATEGORICAL);
    Ptr<ml::TrainData> td = ml::TrainData::create(samples, ml::ROW_SAMPLE, responses,
                                                  noArray(), noArray(), noArray(), varType);
    Ptr<ml::DTrees> dt = ml::DTrees::create();
    dt->setMaxDepth(4);
    dt->setCVFolds(0);
    dt->setMinSampleCount(1);
    dt->train(td);
    return dtreesToString(dt);
}

static std::string withMissingSubst(const std::string& model, const std::string& values)
{
    const size_t pos = model.find("missing_subst:");
    EXPECT_NE(pos, std::string::npos);
    if (pos == std::string::npos)
        return model;
    const size_t end = model.find(']', pos);
    EXPECT_NE(end, std::string::npos);
    if (end == std::string::npos)
        return model;
    return model.substr(0, pos) + "missing_subst: [ " + values + " ]" + model.substr(end + 1);
}

TEST(ML_DTrees, load_short_missing_subst)
{
    const std::string model = missingSubstDTreesModel();
    ASSERT_NE(model.find("var:2,"), std::string::npos);
    const Mat sample = (Mat_<float>(1, 3) << 3.f, 3.f, 9.f);
    const Mat missing = (Mat_<float>(1, 3) << 3.f, 3.f, ml::TrainData::missingValue());

    // predict() substitutes a missing input with its missing_subst entry
    Ptr<ml::DTrees> good = Algorithm::loadFromString<ml::DTrees>(withMissingSubst(model, "0., 0., 9."));
    ASSERT_FALSE(good.empty());
    EXPECT_EQ(1.f, good->predict(sample));
    EXPECT_EQ(1.f, good->predict(missing));
    good = Algorithm::loadFromString<ml::DTrees>(withMissingSubst(model, "0., 0., 0."));
    ASSERT_FALSE(good.empty());
    EXPECT_EQ(0.f, good->predict(missing));

    // an empty list means no substitution; the split's default direction is used instead
    Ptr<ml::DTrees> noSubst = Algorithm::loadFromString<ml::DTrees>(withMissingSubst(model, ""));
    ASSERT_FALSE(noSubst.empty());
    EXPECT_EQ(1.f, noSubst->predict(sample));
    EXPECT_NO_THROW(noSubst->predict(missing));

    // the split is on variable 2, so a list with fewer than 3 entries is read past its end
    EXPECT_THROW(Algorithm::loadFromString<ml::DTrees>(withMissingSubst(model, "0., 0.")), Exception);
    EXPECT_THROW(Algorithm::loadFromString<ml::DTrees>(withMissingSubst(model, "9.")), Exception);
}

TEST(ML_DTrees, read_legacy_model_into_trained_instance)
{
    // trained on one input, so this instance holds a short missing_subst list (the input and the response)
    Mat samples = (Mat_<float>(4, 1) << 0.f, 1.f, 2.f, 3.f);
    Mat responses = (Mat_<int>(4, 1) << 0, 0, 1, 1);
    Ptr<ml::DTrees> dt = ml::DTrees::create();
    dt->setMaxDepth(2);
    dt->setCVFolds(0);
    dt->setMinSampleCount(1);
    dt->train(ml::TrainData::create(samples, ml::ROW_SAMPLE, responses));

    // the legacy format stores no missing_subst, so the trained values must not survive read()
    const string filename = findDataFile("legacy/dtree_abalone.xml");
    FileStorage fs(filename, FileStorage::READ);
    dt->read(fs.getFirstTopLevelNode());
    Ptr<ml::DTrees> fresh = Algorithm::load<ml::DTrees>(filename);
    ASSERT_FALSE(fresh.empty());
    ASSERT_EQ(fresh->getVarCount(), dt->getVarCount());

    Mat input(10, fresh->getVarCount(), CV_32F);
    cv::theRNG().fill(input, RNG::UNIFORM, 0, 40);
    randomFillCategories(filename, input);
    input.row(0).setTo(ml::TrainData::missingValue());
    Mat expected, actual;
    fresh->predict(input, expected);
    dt->predict(input, actual);
    EXPECT_EQ(0, cvtest::norm(expected, actual, NORM_INF));
}

/*TEST(ML_SVM, throw_exception_when_save_untrained_model)
{
    Ptr<cv::ml::SVM> svm;
    string filename = tempfile("svm.xml");
    ASSERT_THROW(svm.save(filename.c_str()), Exception);
    remove(filename.c_str());
}*/

}} // namespace
