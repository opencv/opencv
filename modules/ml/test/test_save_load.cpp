// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {


void randomFillCategories(const string & filename, Mat & input)
{
    Mat catMap;
    Mat catCount;
    std::vector<uchar> varTypes;

    FileStorage fs(filename, FileStorage::READ);
    FileNode root = fs.getFirstTopLevelNode();
    root["cat_map"] >> catMap;
    root["cat_count"] >> catCount;
    root["var_type"] >> varTypes;

    int offset = 0;
    int countOffset = 0;
    uint var = 0, varCount = (uint)varTypes.size();
    for (; var < varCount; ++var)
    {
        if (varTypes[var] == ml::VAR_CATEGORICAL)
        {
            int size = catCount.at<int>(0, countOffset);
            for (int row = 0; row < input.rows; ++row)
            {
                int randomChosenIndex = offset + ((uint)cv::theRNG()) % size;
                int value = catMap.at<int>(0, randomChosenIndex);
                input.at<float>(row, var) = (float)value;
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

/*TEST(ML_SVM, throw_exception_when_save_untrained_model)
{
    Ptr<cv::ml::SVM> svm;
    string filename = tempfile("svm.xml");
    ASSERT_THROW(svm.save(filename.c_str()), Exception);
    remove(filename.c_str());
}*/

}} // namespace
