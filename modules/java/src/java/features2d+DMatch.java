package org.opencv.features2d;

//C++: class DMatch
//javadoc: DMatch
public class DMatch {
	
	//javadoc: DMatch::queryIdx
	public int queryIdx;
	//javadoc: DMatch::trainIdx
	public int trainIdx;
	//javadoc: DMatch::imgIdx
	public int imgIdx;
	//javadoc: DMatch::distance
	public float distance;
    
    
    //javadoc: DMatch::DMatch()
    public DMatch() {
    	this(-1, -1, Float.MAX_VALUE);
	}
	
	
    public DMatch( int _queryIdx, int _trainIdx, float _distance ) {
    	queryIdx = _queryIdx;
    	trainIdx = _trainIdx;
    	imgIdx = -1;
    	distance = _distance; 
    }
    
    
    public DMatch( int _queryIdx, int _trainIdx, int _imgIdx, float _distance ) {
    	queryIdx = _queryIdx;
    	trainIdx = _trainIdx;
    	imgIdx = _imgIdx;
    	distance = _distance; 
    }

    // less is better
    boolean lessThan(DMatch it) {
        return distance < it.distance;
    }


	@Override
	public String toString() {
		return "DMatch [queryIdx=" + queryIdx + ", trainIdx=" + trainIdx
				+ ", imgIdx=" + imgIdx + ", distance=" + distance + "]";
	}

}
