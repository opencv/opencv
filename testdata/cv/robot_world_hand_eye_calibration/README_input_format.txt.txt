This README file is to accompany code for robot-world, hand-eye calibration, produced by Amy Tabb and Khalil M. Ahmad Yousef as a companion to their paper:
	Solving the Robot-World Hand-Eye(s) Calibration Problem with Iterative Methods, which is currently under review.



If you use this code in project that results in a publication, please cite the paper above. 

This README covers the input format required for each dataset directory, and how to interpret that format.


Jan 12, 2017

Comments/Bugs/Problems: amy.tabb@ars.usda.gov



Required Directories:

images

internal_images



Each of these directories, should contain directories by camera with images.  For instance, for one camera, the "images" directory should contain "camera0".  For two cameras, it should contain "camera0"" and "camera1." Use the provided datasets as templates for how to set up your own datasets.



The "images" directory represents the images for each stop of the robot.  "internal_images" contains extra images that are used for camera calibration only.  The directory structure of "camera0", "camera1", etc. within "internal_images" should be used, if "internal_images" is present.



Required files:


calibration_object.txt : This file gives the specification of the calibration pattern.  The units can be changed, and the particular strings used 'chess_mm_height', 'chess_mm_width' are not used by the program.  However, the ORDER of the parameters matters.  So make sure the use of units is consistent, and do not switch the height and width ordering within the file.

	Example from dataset 1:

	
		chess_mm_height 28.5

		chess_mm_width 28.5

		chess_height 6
	
	chess_width 8



robot_cali.txt : This file gives the number of robot positions, as well as the rotation and translation parameters for the robot.  In the notation of our paper, these matrices are B_i.  Note that our definition is the inverse of some other works. 

 

Optional Included file:
ExperimentDetails.txt : This file gives some parameters of the experiments for the provided datasets and is not required to run the code.

