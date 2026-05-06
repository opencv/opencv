#pragma once
#include "opencv2/objdetect/aruco_detector.hpp"
#include "opencv2/objdetect/aruco_board.hpp"
namespace  cv::aruco{

class CharucoBoard2{
public:
    cv::Size bSize={-1,-1};
    cv::aruco::Dictionary  dictionary;
    std::vector<int> ids;
    float markerLength = 0.f, markerSeparation = 0.f;
    CharucoBoard2();
    /**
     * @brief CharucoBoard2
     * @param bSize
     * @param markerLength
     * @param markerSeparation ignored
     * @param dictionary
     * @param ids
     */
    CharucoBoard2(cv::Size bSize, float markerLength, float markerSeparation,const cv::aruco::Dictionary &dictionary, cv::InputArray ids = cv::noArray());
    void generateImage(int markerSizePix, cv::Mat& outImage) const;
    void generateImage(cv::Size outSize, cv::Mat& outImage, int marginSize=0, int borderBits=1) const;
    //returns the row,col of a given marker id
    std::pair<int,int> getIdPos(int id)const;
    //returns the id at the row,col indicated
    int getId(int row,int col)const;

    /** @brief Given a board configuration and a set of detected markers, returns the corresponding
     * image points and object points, can be used in solvePnP()
     *
     * @param detectedCorners List of detected marker corners of the board.
     * For cv::Board and cv::GridBoard the method expects std::vector<std::vector<Point2f>> or std::vector<Mat> with Aruco marker corners.
     * For cv::CharucoBoard the method expects std::vector<Point2f> or Mat with ChAruco corners (chess board corners matched with Aruco markers).
     *
     * @param detectedIds List of identifiers for each marker or charuco corner.
     * For any Board class the method expects std::vector<int> or Mat.
     *
     * @param objPoints Vector of marker points in the board coordinate space.
     * For any Board class the method expects std::vector<cv::Point3f> objectPoints or cv::Mat
     *
     * @param imgPoints Vector of marker points in the image coordinate space.
     * For any Board class the method expects std::vector<cv::Point2f> objectPoints or cv::Mat
     *
     * @sa solvePnP
     */
    void matchImagePoints(cv::InputArrayOfArrays detectedCorners, cv::InputArray detectedIds,
                                  cv::OutputArray objPoints, cv::OutputArray imgPoints) const;

};

class     CharucoDetector2{
    CharucoBoard2 board;
 public:
    CharucoDetector2(const CharucoBoard2& board);
     CharucoDetector2();
    void setBoard(const CharucoBoard2& board);
    //void detectBoard(InputArray image, OutputArray imgPoints,  OutputArray objPoints,OutputArray markerIds) const;

    void detectBoard(cv::InputArray image, cv::OutputArray charucoCorners, cv::OutputArray charucoIds,
                     cv::InputOutputArrayOfArrays markerCorners=cv::noArray(), cv::InputOutputArray markerIds=cv::noArray());
    void detectDiamonds(cv::InputArray image, cv::OutputArrayOfArrays _diamondCorners, cv::OutputArray _diamondIds,
                        cv::InputOutputArrayOfArrays inMarkerCorners, cv::InputOutputArray inMarkerIds) ;


};

}
