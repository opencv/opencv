
#include "opencv2/contrib/contrib.hpp"

#ifdef WIN32
	#include <windows.h>
	#include <tchar.h>
#else
	#include <dirent.h>
#endif

namespace cv
{
	std::vector<std::string> Directory::GetListFiles( const string& directoryName, bool addPath )
	{
		std::vector<std::string> list;
		list.clear();
		std::string path = directoryName + "/*";
		#ifdef WIN32
			WIN32_FIND_DATA FindFileData;
			HANDLE hFind;

			hFind = FindFirstFile((LPCSTR)path.c_str(), &FindFileData);
			if (hFind == INVALID_HANDLE_VALUE) 
			{
				return list;
			}
			else
			{
				do
				{
					if (FindFileData.dwFileAttributes == FILE_ATTRIBUTE_NORMAL  || 
						FindFileData.dwFileAttributes == FILE_ATTRIBUTE_ARCHIVE || 
						FindFileData.dwFileAttributes == FILE_ATTRIBUTE_HIDDEN  || 
						FindFileData.dwFileAttributes == FILE_ATTRIBUTE_SYSTEM  || 
						FindFileData.dwFileAttributes == FILE_ATTRIBUTE_READONLY)
					{
						if (addPath)
							list.push_back(directoryName + "/" + FindFileData.cFileName);
						else
							list.push_back(FindFileData.cFileName);
					}
				} 
				while(FindNextFile(hFind, &FindFileData));
				FindClose(hFind);  
			}
		#else
			DIR *dp;
			struct dirent *dirp;
			if((dp  = opendir(directoryName.c_str())) == NULL) 
			{				
				return list;
			}

			while ((dirp = readdir(dp)) != NULL) 
			{
				if (dirp->d_type == DT_REG)
					list.push_back(static_cast<string>(dirp->d_name));
			}
			closedir(dp);
		#endif

		return list;
	}

	std::vector<std::string> Directory::GetListFolders( const string& directoryName, bool addPath )
	{
		std::vector<std::string> list;
		std::string path = directoryName + "/*";
		list.clear();
		#ifdef WIN32
			WIN32_FIND_DATA FindFileData;
			HANDLE hFind;

			hFind = FindFirstFile((LPCSTR)path.c_str(), &FindFileData);
			if (hFind == INVALID_HANDLE_VALUE) 
			{
				return list;
			}
			else
			{
				do
				{
					if (FindFileData.dwFileAttributes == FILE_ATTRIBUTE_DIRECTORY && 
						strcmp(FindFileData.cFileName, ".") != 0 && 
						strcmp(FindFileData.cFileName, "..") != 0)
					{
						if (addPath)
							list.push_back(directoryName + "/" + FindFileData.cFileName);
						else
							list.push_back(FindFileData.cFileName);
					}
				} 
				while(FindNextFile(hFind, &FindFileData));
				FindClose(hFind);  
			}

		#else
			DIR *dp;
			struct dirent *dirp;
			if((dp  = opendir(path.c_str())) == NULL) 
			{				
				return list;
			}

			while ((dirp = readdir(dp)) != NULL) 
			{
				if (dirp->d_type == DT_DIR)
					list.push_back(static_cast<string>(dirp->d_name));
			}
			closedir(dp);
		#endif

		return list;
	}

	std::vector<std::string> Directory::GetListFilesR ( const string& directoryName, bool addPath )
	{
		std::vector<std::string> list = Directory::GetListFiles(directoryName, addPath);

		std::vector<std::string> dirs = Directory::GetListFolders(directoryName, addPath);

		std::vector<std::string>::const_iterator it;
		for (it = dirs.begin(); it != dirs.end(); ++it)
		{
			std::vector<std::string> cl = Directory::GetListFiles(*it, addPath);
			list.insert(list.end(), cl.begin(), cl.end());
		}

		return list;
	}

}