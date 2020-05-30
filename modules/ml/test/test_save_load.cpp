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

/*TEST(ML_SVM, throw_exception_when_save_untrained_model)
{
    Ptr<cv::ml::SVM> svm;
    string filename = tempfile("svm.xml");
    ASSERT_THROW(svm.save(filename.c_str()), Exception);
    remove(filename.c_str());
}*/

}} // namespace
