Initialization
--------------
Download the source images into the /original directory from http://www.openjpeg.org/OPJ_Validate_OriginalImages.7z

Usage
-----
Usage: OPJ_Validate batch_text_file bin_directory
Example: OPJ_Validate OPJ_Param_File_v0_1.txt v1.1.a
where OPJ_Param_File_v0_1.txt is a file containing a list of compression and decompression parameters
and v1.1.a is a directory inside the directory OPJ_Binaries containing the openjpeg executables (j2k_to_image.exe and image_to_j2k.exe)

Example with batch file: You consider revision 490 (/rev490) as stable, and would like to compare it a new version, revision 493 (rev493).

Batch mode
----------
1) Calculate the reference by running the "OPJ_Validate_Create_Ref rev490" file (.sh or .bat depending on your os)
2) Compare the candidate revision with ther reference by running the "OPJ_Validate_Candidate_vs_Ref rev493" file
3) The results of the comparison are given at the end of the processing. They are also available in the bin directory OPJ_Binaries/rev493/report.txt

Manual mode
-----------
1) Put the j2k_to_image.exe and image_to_j2k.exe binaries of both revisions in the OPJ_Binaries directory (OPJ_Binaries/rev490 and OPJ_Binaries/rev493)  
2) Start by initializing the validation with revision 490. 
	a) Modify OPJ_Validate_init.bat and set the last line to "OPJ_Validate.exe OPJ_Param_File_v0_1.txt rev490"
	b) Execute OPJ_Validate_init.bat
3) Compare the reference files generated in the previous step with files generated with revision 493
	a) Modify OPJ_Validate_run.bat and set the last line to "OPJ_Validate.exe OPJ_Param_File_v0_1.txt rev493"
	b) Execute OPJ_Validate_run.bat
4) Read the results in the binaries directory of revision 493 (OPJ_Binaries/rev493/report.txt)
	Search for the word "ERROR:" in that file. 
	If this word is not present in the report, this means that both codecs of rev490 and rev493 gave the same results.
	Otherwise, it means that for certain encoding/decoding parameters, the codecs behave differently.

	Example of error
		Task 17
		   MD5 file: temp/A_4_2K_24_185_CBR_WB_000.tif.md5
		   Command line: "OPJ_Binaries/rev473/j2k_to_image.exe -i original/A_4_2K_24_185_CBR_WB_000.j2k -o temp/A_4_2K_24_185_CBR_WB_000.tif "
		ERROR: temp/tempmd5.txt and temp/A_4_2K_24_185_CBR_WB_000.tif.md5 are different.
		The codec seems to behave differently.

	This means that the rev490 and rev493 created two different versions of file A_4_2K_24_185_CBR_WB_000.tif with the command line given above.
	An error might have been caused by switching to this new revision.

	Warning: Do not take the last line of the report.txt file into account ( Cool. All files passed the tests !) as it is a bug. Search for the word "ERROR:" to detect potential errors.
5) If no error is detected, you can commit the changes to the OpenJPEG repository

