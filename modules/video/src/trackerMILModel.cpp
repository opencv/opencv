#include "trackerMILModel.hpp"
/**
 * TrackerMILModel
 */

namespace cv
{

TrackerMILModel::TrackerMILModel( const Rect& boundingBox )
{
	currentSample.clear();
	mode = MODE_POSITIVE;
	width = boundingBox.width;
	height = boundingBox.height;

	Ptr<TrackerMILTargetState> initState = new TrackerMILTargetState( Point2f( boundingBox.x, boundingBox.y ),
		boundingBox.width, boundingBox.height, true, Mat() );
	trajectory.push_back( initState );
}

void TrackerMILModel::responseToConfidenceMap( const std::vector<Mat>& responses, ConfidenceMap& confidenceMap )
{
	if( currentSample.empty() )
	{
		CV_Error(-1, "The samples in Model estimation are empty");
		return;
	}


	for( size_t i = 0; i <  responses.size(); i++ )
	{
		//for each column (one sample) there are #num_feature
		//get informations from currentSample
		for( int j = 0; j < responses.at(i).cols; j++)
		{

			Size currentSize;
			Point currentOfs;
			currentSample.at(j).locateROI( currentSize, currentOfs );
			bool foreground;
			if( mode == MODE_POSITIVE || mode == MODE_ESTIMATON )
			{
				foreground = true;
			}
			else if( mode == MODE_NEGATIVE )
			{
				foreground = false;
			}

			//get the column of the HAAR responses
			Mat singleResponse = responses.at(i).col(j);

			//create the state
			Ptr<TrackerMILTargetState> currentState = new TrackerMILTargetState( currentOfs,
					width,
					height,
					foreground,
					singleResponse);

			confidenceMap.push_back( std::make_pair(currentState, 0) );

		}


	}
}

void TrackerMILModel:: modelEstimationImpl( const std::vector<Mat>& responses )
{
	responseToConfidenceMap( responses, currentConfidenceMap );

}

void TrackerMILModel::modelUpdateImpl()
{

}

void TrackerMILModel::setMode( int trainingMode, const std::vector<Mat>& samples )
{
	currentSample.clear();
	currentSample = samples;

	mode = trainingMode;
}

}
