if(WIN32)
	set(WSH_SCRIPT "CScript.exe")
	OCV_OPTION(WSH_USE4PYTHON "Use Windows Script Host instead of Python" ON  IF (NOT PYTHON_EXECUTABLE) )
endif(WIN32)