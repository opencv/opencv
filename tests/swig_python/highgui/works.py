# needed for access() and remove()
import os

# check for required featurest listet in 'filelist' and removes the old .works file of 'testname'
def check_files( filelist, testname ):
	# delete old .works file of the calling test, if it exists
	filename = "./"+testname+".works"

	if os.access(filename,os.F_OK):
		os.remove(filename)

	# now check for existint .works files
	if len(filelist) > 0:
		for i in range(0,len(filelist)):
			tmpname = "./"+filelist[i]+".works"
			if not os.access(tmpname,os.F_OK):
				print "(INFO) Skipping '"+testname+"' due to SKIP/FAIL of '"+filelist[i]+"'"
				return False

	# either the filelist is empty (no requirements) or all requirements match
	return True

	
# create the .works file for test 'testname'
def set_file( testname ):
	# create .works file of calling test
	works_file = file("./"+testname+".works", 'w',1)
	works_file.close
	return
