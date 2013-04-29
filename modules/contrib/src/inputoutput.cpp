
#include "opencv2/contrib.hpp"

#ifdef WIN32
    #include <windows.h>
    #include <tchar.h>
#else
    #include <dirent.h>
#endif

namespace cv
{
    std::vector<String> Directory::GetListFiles(  const String& path, const String & exten, bool addPath )
    {
        std::vector<String> list;
        list.clear();
        String path_f = path + "/" + exten;
        #ifdef WIN32
            WIN32_FIND_DATA FindFileData;
            HANDLE hFind;

            hFind = FindFirstFile((LPCSTR)path_f.c_str(), &FindFileData);
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
                            list.push_back(path + "/" + FindFileData.cFileName);
                        else
                            list.push_back(FindFileData.cFileName);
                    }
                }
                while(FindNextFile(hFind, &FindFileData));
                FindClose(hFind);
            }
        #else
            (void)addPath;
            DIR *dp;
            struct dirent *dirp;
            if((dp = opendir(path.c_str())) == NULL)
            {
                return list;
            }

            while ((dirp = readdir(dp)) != NULL)
            {
                if (dirp->d_type == DT_REG)
                {
                    if (exten.compare("*") == 0)
                        list.push_back(static_cast<String>(dirp->d_name));
                    else
                        if (String(dirp->d_name).find(exten) != String::npos)
                            list.push_back(static_cast<String>(dirp->d_name));
                }
            }
            closedir(dp);
        #endif

        return list;
    }

    std::vector<String> Directory::GetListFolders( const String& path, const String & exten, bool addPath )
    {
        std::vector<String> list;
        String path_f = path + "/" + exten;
        list.clear();
        #ifdef WIN32
            WIN32_FIND_DATA FindFileData;
            HANDLE hFind;

            hFind = FindFirstFile((LPCSTR)path_f.c_str(), &FindFileData);
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
                            list.push_back(path + "/" + FindFileData.cFileName);
                        else
                            list.push_back(FindFileData.cFileName);
                    }
                }
                while(FindNextFile(hFind, &FindFileData));
                FindClose(hFind);
            }

        #else
            (void)addPath;
            DIR *dp;
            struct dirent *dirp;
            if((dp = opendir(path_f.c_str())) == NULL)
            {
                return list;
            }

            while ((dirp = readdir(dp)) != NULL)
            {
                if (dirp->d_type == DT_DIR &&
                    strcmp(dirp->d_name, ".") != 0 &&
                    strcmp(dirp->d_name, "..") != 0 )
                {
                    if (exten.compare("*") == 0)
                        list.push_back(static_cast<String>(dirp->d_name));
                    else
                        if (String(dirp->d_name).find(exten) != String::npos)
                            list.push_back(static_cast<String>(dirp->d_name));
                }
            }
            closedir(dp);
        #endif

        return list;
    }

    std::vector<String> Directory::GetListFilesR ( const String& path, const String & exten, bool addPath )
    {
        std::vector<String> list = Directory::GetListFiles(path, exten, addPath);

        std::vector<String> dirs = Directory::GetListFolders(path, exten, addPath);

        std::vector<String>::const_iterator it;
        for (it = dirs.begin(); it != dirs.end(); ++it)
        {
            std::vector<String> cl = Directory::GetListFiles(*it, exten, addPath);
            list.insert(list.end(), cl.begin(), cl.end());
        }

        return list;
    }

}
