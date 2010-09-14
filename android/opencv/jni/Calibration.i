/*
 * include the headers required by the generated cpp code
 */
%{
#include "Calibration.h"
#include "image_pool.h"
using namespace cv;
%}


class Calibration {
public:

	Size patternsize;
	
	Calibration();
	virtual ~Calibration();

	bool detectAndDrawChessboard(int idx, image_pool* pool);
	
	void resetChess();
	
	int getNumberDetectedChessboards();
	
	void calibrate(const char* filename);
	
	void drawText(int idx, image_pool* pool, const char* text);
};
