/*
  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.

  By downloading, copying, installing or using the software you agree to this license.
  If you do not agree to this license, do not download, install,
  copy or use the software.


                          BSD 3-Clause License

 Copyright (C) 2014, Olexa Bilaniuk, Hamid Bazargani & Robert Laganiere, all rights reserved.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

   * Redistribution's of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

   * Redistribution's in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.

   * The name of the copyright holders may not be used to endorse or promote products
     derived from this software without specific prior written permission.

 This software is provided by the copyright holders and contributors "as is" and
 any express or implied warranties, including, but not limited to, the implied
 warranties of merchantability and fitness for a particular purpose are disclaimed.
 In no event shall the Intel Corporation or contributors be liable for any direct,
 indirect, incidental, special, exemplary, or consequential damages
 (including, but not limited to, procurement of substitute goods or services;
 loss of use, data, or profits; or business interruption) however caused
 and on any theory of liability, whether in contract, strict liability,
 or tort (including negligence or otherwise) arising in any way out of
 the use of this software, even if advised of the possibility of such damage.
*/

/**
 * Bilaniuk, Olexa, Hamid Bazargani, and Robert Laganiere. "Fast Target
 * Recognition on Mobile Devices: Revisiting Gaussian Elimination for the
 * Estimation of Planar Homographies." In Computer Vision and Pattern
 * Recognition Workshops (CVPRW), 2014 IEEE Conference on, pp. 119-125.
 * IEEE, 2014.
 */

#ifndef CHESSBOARD_HPP_
#define CHESSBOARD_HPP_

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include <vector>
#include <set>
#include <map>

namespace cv {
namespace details{
/**
 * \brief Fast point sysmetric cross detector based on a localized radon transformation
 */
class CV_EXPORTS_W FastX : public cv::Feature2D
{
    public:
        struct CV_EXPORTS_W Parameters
        {
            CV_PROP_RW float strength;       //!< minimal strength of a valid junction in dB
            CV_PROP_RW float resolution;     //!< angle resolution in radians
            CV_PROP_RW int branches;         //!< the number of branches
            CV_PROP_RW int min_scale;        //!< scale level [0..8]
            CV_PROP_RW int max_scale;        //!< scale level [0..8]
            CV_PROP_RW bool filter;          //!< post filter feature map to improve impulse response
            CV_PROP_RW bool super_resolution; //!< up-sample

            CV_WRAP Parameters()
            {
                strength = 40;
                resolution = float(M_PI*0.25);
                branches = 2;
                min_scale = 2;
                max_scale = 5;
                super_resolution = 1;
                filter = true;
            }
        };

    public:
        CV_WRAP FastX(const Parameters &config = Parameters());
        virtual ~FastX(){};

        CV_WRAP void reconfigure(const Parameters &para);

        //declaration to be wrapped by rbind
        CV_WRAP void detect(cv::InputArray image,CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask=cv::Mat())override
        {cv::Feature2D::detect(image.getMat(),keypoints,mask.getMat());}

        CV_WRAP virtual void detectAndCompute(cv::InputArray image,
                                              cv::InputArray mask,
                                              std::vector<cv::KeyPoint>& keypoints,
                                              cv::OutputArray descriptors,
                                              bool useProvidedKeyPoints = false)override;

        CV_WRAP void detectImpl(const cv::Mat& image,
                               CV_OUT std::vector<cv::KeyPoint>& keypoints,
                               CV_OUT std::vector<cv::Mat> &feature_maps,
                               const cv::Mat& mask=cv::Mat())const;

        CV_WRAP void detectImpl(const cv::Mat& image,
                                CV_OUT std::vector<cv::Mat> &rotated_images,
                                CV_OUT std::vector<cv::Mat> &feature_maps,
                                const cv::Mat& mask=cv::Mat())const;

        CV_WRAP void findKeyPoints(const std::vector<cv::Mat> &feature_map,
                                   CV_OUT std::vector<cv::KeyPoint>& keypoints,
                                   const cv::Mat& mask = cv::Mat())const;

        CV_WRAP std::vector<std::vector<float> > calcAngles(const std::vector<cv::Mat> &rotated_images,
                                                            std::vector<cv::KeyPoint> &keypoints)const;
        // define pure virtual methods
        virtual int descriptorSize()const override{return 0;};
        virtual int descriptorType()const override{return 0;};
        virtual void operator()( cv::InputArray image, cv::InputArray mask, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors, bool useProvidedKeypoints=false )const
        {
            descriptors.clear();
            detectImpl(image.getMat(),keypoints,mask);
            if(!useProvidedKeypoints)        // suppress compiler warning
                return;
            return;
        }

    protected:
        virtual void computeImpl( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)const
        {
            descriptors = cv::Mat();
            detectImpl(image,keypoints);
        }

    private:
        void detectImpl(const cv::Mat& _src, std::vector<cv::KeyPoint>& keypoints, const cv::Mat& mask)const;
        virtual void detectImpl(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask=cv::noArray())const;

        void rotate(float angle,const cv::Mat &img,cv::Size size,CV_OUT cv::Mat &out)const;
        void calcFeatureMap(const cv::Mat &images,cv::Mat& out)const;

    private:
        Parameters parameters;
};

/**
 * \brief Ellipse class
 */
class CV_EXPORTS_W Ellipse
{
    public:
        CV_WRAP Ellipse();
        CV_WRAP Ellipse(const cv::Point2f &center, const cv::Size2f &axes, float angle);
        CV_WRAP Ellipse(const Ellipse &other);


        CV_WRAP void draw(cv::InputOutputArray img,const cv::Scalar &color = cv::Scalar::all(120))const;
        CV_WRAP bool contains(const cv::Point2f &pt)const;
        CV_WRAP cv::Point2f getCenter()const;
        CV_WRAP const cv::Size2f &getAxes()const;

    private:
        cv::Point2f center;
        cv::Size2f axes;
        float angle,cosf,sinf;
};

/**
 * \brief Chessboard corner detector
 *
 * The detectors tries to find all chessboard corners of an imaged
 * chessboard and returns them as an ordered vector of KeyPoints.
 * Thereby, the left top corner has index 0 and the bottom right
 * corner n*m-1.
 */
class CV_EXPORTS_W Chessboard: public cv::Feature2D
{
    public:
        static const int DUMMY_FIELD_SIZE = 100;  // in pixel

        /**
         * \brief Configuration of a chessboard corner detector
         *
         */
        struct CV_EXPORTS_W Parameters
        {
            CV_PROP_RW cv::Size chessboard_size; //!< size of the chessboard
            CV_PROP_RW int min_scale;            //!< scale level [0..8]
            CV_PROP_RW int max_scale;            //!< scale level [0..8]
            CV_PROP_RW int max_points;           //!< maximal number of points regarded
            CV_PROP_RW int max_tests;            //!< maximal number of tested hypothesis
            CV_PROP_RW bool super_resolution;    //!< use super-repsolution for chessboard detection
            CV_PROP_RW bool larger;              //!< indicates if larger boards should be returned

            CV_WRAP Parameters()
            {
                chessboard_size = cv::Size(9,6);
                min_scale = 2;
                max_scale = 4;
                super_resolution = true;
                max_points = 400;
                max_tests = 100;
                larger = false;
            }

            CV_WRAP Parameters(int scale,int _max_points):
                min_scale(scale),
                max_scale(scale),
                max_points(_max_points)
            {
                chessboard_size = cv::Size(9,6);
            }
        };


        /**
         * \brief Gets the 3D objects points for the chessboard assuming the
         * left top corner is located at the origin.
         *
         * \param[in] pattern_size Number of rows and cols of the pattern
         * \param[in] cell_size Size of one cell
         *
         * \returns Returns the object points as CV_32FC3
         */
        CV_WRAP static cv::Mat getObjectPoints(const cv::Size &pattern_size,float cell_size);

        /**
         * \brief Class for searching and storing chessboard corners.
         *
         * The search is based on a feature map having strong pixel
         * values at positions where a chessboard corner is located.
         *
         * The board must be rectangular but supports empty cells
         *
         */
        class CV_EXPORTS_W Board
        {
            public:
                /**
                 * \brief Estimates the position of the next point on a line using cross ratio constrain
                 *
                 * cross ratio:
                 * d12/d34 = d13/d24
                 *
                 * point order on the line:
                 * pt1 --> pt2 --> pt3 --> pt4
                 *
                 * \param[in] pt1 First point coordinate
                 * \param[in] pt2 Second point coordinate
                 * \param[in] pt3 Third point coordinate
                 * \param[out] pt4 Forth point coordinate
                 *
                 */
                CV_WRAP static bool estimatePoint(const cv::Point2f &p0,const cv::Point2f &p1,const cv::Point2f &p2,cv::Point2f &p3);

                // using 1D homography
                CV_WRAP static bool estimatePoint(const cv::Point2f &p0,const cv::Point2f &p1,const cv::Point2f &p2,const cv::Point2f &p3, cv::Point2f &p4);

                /**
                 * \brief Checks if all points of a row or column have a valid cross ratio constraint
                 *
                 * cross ratio:
                 * d12/d34 = d13/d24
                 *
                 * point order on the row/column:
                 * pt1 --> pt2 --> pt3 --> pt4
                 *
                 * \param[in] points THe points of the row/column
                 *
                 */
                CV_WRAP static bool checkRowColumn(const std::vector<cv::Point2f> &points);

                /**
                 * \brief Estimates the search area for the next point on the line using cross ratio
                 *
                 * point order on the line:
                 * (p0) --> p1 --> p2 --> p3 --> search area
                 *
                 * \param[in] p1 First point coordinate
                 * \param[in] p2 Second point coordinate
                 * \param[in] p3 Third point coordinate
                 * \param[in] p Percentage of d34 used for the search area width and height [0..1]
                 * \param[out] ellipse The search area
                 * \param[in] p0 optional point to improve accuracy
                 *
                 * \return Returns false if no search area can be calculated
                 *
                 */
                CV_WRAP static bool estimateSearchArea(const cv::Point2f &p1,const cv::Point2f &p2,const cv::Point2f &p3,float p,
                                                       Ellipse &ellipse,const cv::Point2f *p0 =NULL);

                /**
                 * \brief Estimates the search area for a specific point based on the given homography
                 *
                 * \param[in] H homography descriping the transformation from ideal board to real one
                 * \param[in] row Row of the point
                 * \param[in] col Col of the point
                 * \param[in] p Percentage [0..1]
                 *
                 * \return Returns false if no search area can be calculated
                 *
                 */
                CV_WRAP static Ellipse estimateSearchArea(cv::Mat H,int row, int col,float p,int field_size = DUMMY_FIELD_SIZE);

                /**
                 * \brief Searches for the maximum in a given search area
                 *
                 * \param[in] map feature map
                 * \param[in] ellipse search area
                 * \param[in] min_val Minimum value of the maximum to be accepted as maximum
                 *
                 * \return Returns a negative value if all points are outside the ellipse
                 *
                 */
                CV_WRAP static float findMaxPoint(cv::flann::Index &index,const cv::Mat &data,const Ellipse &ellipse,float white_angle,float black_angle,cv::Point2f &pt);

                /**
                 * \brief Searches for the next point using cross ratio constrain
                 *
                 * \param[in] index flann index
                 * \param[in] data extended flann data
                 * \param[in] pt1
                 * \param[in] pt2
                 * \param[in] pt3
                 * \param[in] white_angle
                 * \param[in] black_angle
                 * \param[in] min_response
                 * \param[out] point The resulting point
                 *
                 * \return Returns false if no point could be found
                 *
                 */
                CV_WRAP static bool findNextPoint(cv::flann::Index &index,const cv::Mat &data,
                                                  const cv::Point2f &pt1,const cv::Point2f &pt2, const cv::Point2f &pt3,
                                                  float white_angle,float black_angle,float min_response,cv::Point2f &point);

                /**
                 * \brief Creates a new Board object
                 *
                 */
                CV_WRAP Board(float white_angle=0,float black_angle=0);
                CV_WRAP Board(const cv::Size &size, const std::vector<cv::Point2f> &points,float white_angle=0,float black_angle=0);
                CV_WRAP Board(const Chessboard::Board &other);
                virtual ~Board();

                Board& operator=(const Chessboard::Board &other);

                /**
                 * \brief Draws the corners into the given image
                 *
                 * \param[in] m The image
                 * \param[out] m The resulting image
                 * \param[in] H optional homography to calculate search area
                 *
                 */
                CV_WRAP void draw(cv::InputArray m,cv::OutputArray out,cv::InputArray H=cv::Mat())const;

                /**
                 * \brief Estimates the pose of the chessboard
                 *
                 */
                CV_WRAP bool estimatePose(const cv::Size2f &real_size,cv::InputArray _K,cv::OutputArray rvec,cv::OutputArray tvec)const;

                /**
                 * \brief Clears all internal data of the object
                 *
                 */
                CV_WRAP void clear();

                /**
                 * \brief Returns the angle of the black diagnonale
                 *
                 */
                CV_WRAP float getBlackAngle()const;

                /**
                 * \brief Returns the angle of the black diagnonale
                 *
                 */
                CV_WRAP float getWhiteAngle()const;

                /**
                 * \brief Initializes a 3x3 grid from 9 corner coordinates
                 *
                 * All points must be ordered:
                 * p0 p1 p2
                 * p3 p4 p5
                 * p6 p7 p8
                 *
                 * \param[in] points vector of points
                 *
                 * \return Returns false if the grid could not be initialized
                 */
                CV_WRAP bool init(const std::vector<cv::Point2f> points);

                /**
                 * \brief Returns true if the board is empty
                 *
                 */
                CV_WRAP bool isEmpty() const;

                /**
                 * \brief Returns all board corners as ordered vector
                 *
                 * The left top corner has index 0 and the bottom right
                 * corner rows*cols-1. All corners which only belong to
                 * empty cells are returned as NaN.
                 */ 
                CV_WRAP std::vector<cv::Point2f> getCorners(bool ball=true) const;

                /**
                 * \brief Returns all board corners as ordered vector of KeyPoints
                 *
                 * The left top corner has index 0 and the bottom right
                 * corner rows*cols-1.
                 *
                 * \param[in] ball if set to false only non empty points are returned
                 *
                 */
                CV_WRAP std::vector<cv::KeyPoint> getKeyPoints(bool ball=true) const;

                /**
                 * \brief Returns the centers of the chessboard cells
                 *
                 * The left top corner has index 0 and the bottom right
                 * corner (rows-1)*(cols-1)-1.
                 *
                 */
                CV_WRAP std::vector<cv::Point2f> getCellCenters() const;

                /**
                 * \brief Estimates the homography between an ideal board
                 * and reality based on the already recovered points
                 *
                 * \param[in] rect selecting a subset of the already recovered points
                 * \param[in] field_size The field size of the ideal board
                 *
                 */
                CV_WRAP cv::Mat estimateHomography(cv::Rect rect,int field_size = DUMMY_FIELD_SIZE)const;

                /**
                 * \brief Estimates the homography between an ideal board
                 * and reality based on the already recovered points
                 *
                 * \param[in] field_size The field size of the ideal board
                 *
                 */
                CV_WRAP cv::Mat estimateHomography(int field_size = DUMMY_FIELD_SIZE)const;

                /**
                 * \brief Returns the size of the board
                 *
                 */
                CV_WRAP cv::Size getSize() const;

                /**
                 * \brief Returns the number of cols
                 *
                 */
                CV_WRAP size_t colCount() const;

                /**
                 * \brief Returns the number of rows
                 *
                 */
                CV_WRAP size_t rowCount() const;

                /**
                 * \brief Returns the inner contour of the board inlcuding only valid corners
                 *
                 * \info the contour might be non squared if not all points of the board are defined
                 *
                 */
                CV_WRAP std::vector<cv::Point2f> getContour()const;


                /**
                 * \brief Grows the board in all direction until no more corners are found in the feature map
                 *
                 * \param[in] data CV_32FC1 data of the flann index
                 * \param[in] flann_index flann index
                 *
                 * \returns the number of grows
                 */
                CV_WRAP int grow(const cv::Mat &data,cv::flann::Index &flann_index);

                /**
                 * \brief Validates all corners using guided search based on the given homography
                 *
                 * \param[in] data CV_32FC1 data of the flann index
                 * \param[in] flann_index flann index
                 * \param[in] h Homography describing the transformation from ideal board to the real one
                 * \param[in] min_response Min response
                 *
                 * \returns the number of valid corners
                 */
                CV_WRAP int validateCorners(const cv::Mat &data,cv::flann::Index &flann_index,const cv::Mat &h,float min_response=0);

                /**
                 * \brief check that no corner is used more than once
                 *
                 * \returns Returns false if a corner is used more than once
                 */
                 CV_WRAP bool checkUnique()const;

                 /**
                  * \brief Returns false if the angles of the contour are smaller than 35°
                  *
                  */
                 bool validateContour()const;

                /**
                 * \brief Grows the board to the left by adding one column.
                 *
                 * \param[in] map CV_32FC1 feature map
                 *
                 * \returns Returns false if the feature map has no maxima at the requested positions
                 */
                CV_WRAP bool growLeft(const cv::Mat &map,cv::flann::Index &flann_index);
                CV_WRAP void growLeft();

                /**
                 * \brief Grows the board to the top by adding one row.
                 *
                 * \param[in] map CV_32FC1 feature map
                 *
                 * \returns Returns false if the feature map has no maxima at the requested positions
                 */
                CV_WRAP bool growTop(const cv::Mat &map,cv::flann::Index &flann_index);
                CV_WRAP void growTop();

                /**
                 * \brief Grows the board to the right by adding one column.
                 *
                 * \param[in] map CV_32FC1 feature map
                 *
                 * \returns Returns false if the feature map has no maxima at the requested positions
                 */
                CV_WRAP bool growRight(const cv::Mat &map,cv::flann::Index &flann_index);
                CV_WRAP void growRight();

                /**
                 * \brief Grows the board to the bottom by adding one row.
                 *
                 * \param[in] map CV_32FC1 feature map
                 *
                 * \returns Returns false if the feature map has no maxima at the requested positions
                 */
                CV_WRAP bool growBottom(const cv::Mat &map,cv::flann::Index &flann_index);
                CV_WRAP void growBottom();

                /**
                 * \brief Adds one column on the left side
                 *
                 * \param[in] points The corner coordinates
                 *
                 */
                CV_WRAP void addColumnLeft(const std::vector<cv::Point2f> &points);

                /**
                 * \brief Adds one column at the top
                 *
                 * \param[in] points The corner coordinates
                 *
                 */
                CV_WRAP void addRowTop(const std::vector<cv::Point2f> &points);

                /**
                 * \brief Adds one column on the right side
                 *
                 * \param[in] points The corner coordinates
                 *
                 */
                CV_WRAP void addColumnRight(const std::vector<cv::Point2f> &points);

                /**
                 * \brief Adds one row at the bottom
                 *
                 * \param[in] points The corner coordinates
                 *
                 */
                CV_WRAP void addRowBottom(const std::vector<cv::Point2f> &points);

                /**
                 * \brief Rotates the board 90° degrees to the left
                 */
                CV_WRAP void rotateLeft();

                /**
                 * \brief Rotates the board 90° degrees to the right
                 */
                CV_WRAP void rotateRight();

                /**
                 * \brief Flips the board along its local x(width) coordinate direction
                 */
                CV_WRAP void flipVertical();

                /**
                 * \brief Flips the board along its local y(height) coordinate direction
                 */
                CV_WRAP void flipHorizontal();

                /**
                 * \brief Flips the board so that its top left corner is closest to the coordinate 0/0.
                 */
                CV_WRAP void normalizeTopLeft();
                /**
                 * \brief Flips and rotates the board so that the anlge of
                 * either the black or white diagonale is bigger than the x
                 * and y axis of the board and from a right handed
                 * coordinate system
                 */
                CV_WRAP void normalizeOrientation(bool bblack=true);

                /**
                 * \brief Exchanges the stored board with the board stored in other
                 */
                CV_WRAP void swap(Chessboard::Board &other);

                bool operator==(const Chessboard::Board& other) const {return rows*cols == other.rows*other.cols;};
                bool operator< (const Chessboard::Board& other) const {return rows*cols < other.rows*other.cols;};
                bool operator> (const Chessboard::Board& other) const {return rows*cols > other.rows*other.cols;};
                bool operator>= (const cv::Size& size)const { return rows*cols >= size.width*size.height; };

                /**
                 * \brief Returns a specific corner
                 *
                 * \info raises runtime_error if row col does not exists
                 */
                CV_WRAP cv::Point2f& getCorner(int row,int col);

                /**
                 * \brief Returns true if the cell is empty meaning at least one corner is NaN
                 */
                CV_WRAP bool isCellEmpty(int row,int col);

                /**
                 * \brief Returns the mapping from all corners idx to only valid corners idx
                 */
                std::map<int,int> getMapping()const;

                /**
                 * \brief Estimates rotation of the board around the camera axis
                 */
                 CV_WRAP double estimateRotZ()const;

            private:
                // stores one cell
                // in general a cell is initialized by the Board so that:
                // * all corners are always pointing to a valid cv::Point2f
                // * depending on the position left,top,right and bottom might be set to NaN
                // * A cell is empty if at least one corner is NaN
                struct Cell
                {
                    cv::Point2f *top_left,*top_right,*bottom_right,*bottom_left; // corners
                    Cell *left,*top,*right,*bottom;         // neighbouring cells
                    Cell();
                    bool empty()const;                      // indicates if the cell is empty (one of its corners has NaN)
                    int getRow()const;
                    int getCol()const;
                };

                // corners
                enum CornerIndex
                {
                    TOP_LEFT,
                    TOP_RIGHT,
                    BOTTOM_RIGHT,
                    BOTTOM_LEFT
                };

                Cell* getCell(int row,int column); // returns a specific cell
                const Cell* getCell(int row,int column)const; // returns a specific cell
                void drawEllipses(const std::vector<Ellipse> &ellipses);

                // Iterator for iterating over board corners
                class PointIter
                {
                    public:
                        PointIter(Cell *cell,CornerIndex corner_index);
                        PointIter(const PointIter &other);
                        void operator=(const PointIter &other);
                        bool valid() const;                   // returns if the pointer is pointing to a cell

                        bool left(bool check_empty=false);    // moves one corner to the left or returns false
                        bool right(bool check_empty=false);   // moves one corner to the right or returns false
                        bool bottom(bool check_empty=false);  // moves one corner to the bottom or returns false
                        bool top(bool check_empty=false);     // moves one corner to the top or returns false
                        bool checkCorner()const;              // returns ture if the current corner belongs to at least one
                                                              // none empty cell
                        bool isNaN()const;                    // returns true if the currnet corner is NaN

                        const cv::Point2f* operator*() const;  // current corner coordinate
                        cv::Point2f* operator*();              // current corner coordinate
                        const cv::Point2f* operator->() const; // current corner coordinate
                        cv::Point2f* operator->();             // current corner coordinate

                        Cell *getCell();                 // current cell
                    private:
                        CornerIndex corner_index;
                        Cell *cell;
                };

                std::vector<Cell*> cells;          // storage for all board cells
                std::vector<cv::Point2f*> corners; // storage for all corners
                Cell *top_left;                    // pointer to the top left corner of the board in its local coordinate system
                int rows;                          // number of row cells
                int cols;                          // number of col cells
                float white_angle,black_angle;
        };
    public:

        /**
         * \brief Creates a chessboard corner detectors
         *
         * \param[in] config Configuration used to detect chessboard corners
         *
         */
        CV_WRAP Chessboard(const Parameters &config = Parameters());
        virtual ~Chessboard();
        CV_WRAP void reconfigure(const Parameters &config = Parameters());
        CV_WRAP Parameters getPara()const;

        /*
         * \brief Detects chessboard corners in the given image.
         *
         * The detectors tries to find all chessboard corners of an imaged
         * chessboard and returns them as an ordered vector of KeyPoints.
         * Thereby, the left top corner has index 0 and the bottom right
         * corner n*m-1.
         *
         * \param[in] image The image
         * \param[out] keypoints The detected corners as a vector of ordered KeyPoints
         * \param[in] mask Currently not supported
         *
         */
        CV_WRAP void detect(cv::InputArray image,CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask=cv::Mat())override
        {cv::Feature2D::detect(image.getMat(),keypoints,mask.getMat());}

        CV_WRAP virtual void detectAndCompute(cv::InputArray image,cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints,cv::OutputArray descriptors,
                                              bool useProvidedKeyPoints = false)override;

        /*
         * \brief Detects chessboard corners in the given image.
         *
         * The detectors tries to find all chessboard corners of an imaged
         * chessboard and returns them as an ordered vector of KeyPoints.
         * Thereby, the left top corner has index 0 and the bottom right
         * corner n*m-1.
         *
         * \param[in] image The image
         * \param[out] keypoints The detected corners as a vector of ordered KeyPoints
         * \param[out] feature_maps The feature map generated by LRJT and used to find the corners
         * \param[in] mask Currently not supported
         *
         */
        CV_WRAP void detectImpl(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,std::vector<cv::Mat> &feature_maps,const cv::Mat& mask)const;
        CV_WRAP Chessboard::Board detectImpl(const cv::Mat& image,std::vector<cv::Mat> &feature_maps,const cv::Mat& mask)const;

        // define pure virtual methods
        virtual int descriptorSize()const override{return 0;};
        virtual int descriptorType()const override{return 0;};
        virtual void operator()( cv::InputArray image, cv::InputArray mask, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors, bool useProvidedKeypoints=false )const
        {
            descriptors.clear();
            detectImpl(image.getMat(),keypoints,mask);
            if(!useProvidedKeypoints)        // suppress compiler warning
                return;
            return;
        }

    protected:
        virtual void computeImpl( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)const
        {
            descriptors = cv::Mat();
            detectImpl(image,keypoints);
        }

        // indicates why a board could not be initialized for a certain keypoint
        enum BState
        {
            MISSING_POINTS = 0,       // at least 5 points are needed
            MISSING_PAIRS = 1,        // at least two pairs are needed
            WRONG_PAIR_ANGLE = 2,     // angle between pairs is too small
            WRONG_CONFIGURATION = 3,  // point configuration is wrong and does not belong to a board
            FOUND_BOARD = 4           // board was found
        };

        void findKeyPoints(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,std::vector<cv::Mat> &feature_maps,
                           std::vector<std::vector<float> > &angles ,const cv::Mat& mask)const;
        cv::Mat buildData(const std::vector<cv::KeyPoint>& keypoints)const;
        std::vector<cv::KeyPoint> getInitialPoints(cv::flann::Index &flann_index,const cv::Mat &data,const cv::KeyPoint &center,float white_angle,float black_angle, float min_response = 0)const;
        BState generateBoards(cv::flann::Index &flann_index,const cv::Mat &data, const cv::KeyPoint &center,
                             float white_angle,float black_angle,float min_response,const cv::Mat &img,
                             std::vector<Chessboard::Board> &boards)const;

    private:
        void detectImpl(const cv::Mat&,std::vector<cv::KeyPoint>&, const cv::Mat& mast =cv::Mat())const;
        virtual void detectImpl(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask=cv::noArray())const;

    private:
        Parameters parameters; // storing the configuration of the detector
};
}}  // end namespace details and cv

#endif
