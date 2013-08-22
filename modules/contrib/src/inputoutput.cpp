
#include "opencv2/contrib/contrib.hpp"
#include <cvconfig.h>

#if defined(WIN32) || defined(_WIN32)
    #include <windows.h>
    #include <tchar.h>
#else
    #include <dirent.h>
#endif

namespace cv
{
    std::vector<std::string> Directory::GetListFiles( const std::string& path, const std::string & exten, bool addPath )
    {
        std::vector<std::string> list;
        list.clear();
        std::string path_f = path + "/" + exten;
        #ifdef WIN32
        #ifdef HAVE_WINRT
            WIN32_FIND_DATAW FindFileData;
        #else
            WIN32_FIND_DATAA FindFileData;
        #endif
        HANDLE hFind;

        #ifdef HAVE_WINRT
            wchar_t wpath[MAX_PATH];
            size_t copied = mbstowcs(wpath, path_f.c_str(), MAX_PATH);
            CV_Assert((copied != MAX_PATH) && (copied != (size_t)-1));
            hFind = FindFirstFileExW(wpath, FindExInfoStandard, &FindFileData, FindExSearchNameMatch, NULL, 0);
        #else
            hFind = FindFirstFileA((LPCSTR)path_f.c_str(), &FindFileData);
        #endif
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
                        char* fname;
                    #ifdef HAVE_WINRT
                        char fname_tmp[MAX_PATH] = {0};
                        size_t copied = wcstombs(fname_tmp, FindFileData.cFileName, MAX_PATH);
                        CV_Assert((copied != MAX_PATH) && (copied != (size_t)-1));
                        fname = fname_tmp;
                    #else
                        fname = FindFileData.cFileName;
                    #endif
                        if (addPath)
                            list.push_back(path + "/" + std::string(fname));
                        else
                            list.push_back(std::string(fname));
                    }
                }
            #ifdef HAVE_WINRT
                while(FindNextFileW(hFind, &FindFileData));
            #else
                while(FindNextFileA(hFind, &FindFileData));
            #endif
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
                        list.push_back(static_cast<std::string>(dirp->d_name));
                    else
                        if (std::string(dirp->d_name).find(exten) != std::string::npos)
                            list.push_back(static_cast<std::string>(dirp->d_name));
                }
            }
            closedir(dp);
        #endif

        return list;
    }

    std::vector<std::string> Directory::GetListFolders( const std::string& path, const std::string & exten, bool addPath )
    {
        std::vector<std::string> list;
        std::string path_f = path + "/" + exten;
        list.clear();
        #ifdef WIN32
        #ifdef HAVE_WINRT
            WIN32_FIND_DATAW FindFileData;
        #else
            WIN32_FIND_DATAA FindFileData;
        #endif
            HANDLE hFind;

        #ifdef HAVE_WINRT
            wchar_t wpath [MAX_PATH];
            size_t copied = mbstowcs(wpath, path_f.c_str(), path_f.size());
            CV_Assert((copied != MAX_PATH) && (copied != (size_t)-1));

            hFind = FindFirstFileExW(wpath, FindExInfoStandard, &FindFileData, FindExSearchNameMatch, NULL, 0);
        #else
            hFind = FindFirstFileA((LPCSTR)path_f.c_str(), &FindFileData);
        #endif
            if (hFind == INVALID_HANDLE_VALUE)
            {
                return list;
            }
            else
            {
                do
                {
#ifdef HAVE_WINRT
                    if (FindFileData.dwFileAttributes == FILE_ATTRIBUTE_DIRECTORY &&
                        wcscmp(FindFileData.cFileName, L".") != 0 &&
                        wcscmp(FindFileData.cFileName, L"..") != 0)
#else
                    if (FindFileData.dwFileAttributes == FILE_ATTRIBUTE_DIRECTORY &&
                        strcmp(FindFileData.cFileName, ".") != 0 &&
                        strcmp(FindFileData.cFileName, "..") != 0)
#endif
                    {
                        char* fname;
                    #ifdef HAVE_WINRT
                        char fname_tmp[MAX_PATH];
                        size_t copied = wcstombs(fname_tmp, FindFileData.cFileName, MAX_PATH);
                        CV_Assert((copied != MAX_PATH) && (copied != (size_t)-1));
                        fname = fname_tmp;
                    #else
                        fname = FindFileData.cFileName;
                    #endif

                        if (addPath)
                            list.push_back(path + "/" + std::string(fname));
                        else
                            list.push_back(std::string(fname));
                    }
                }
            #ifdef HAVE_WINRT
                while(FindNextFileW(hFind, &FindFileData));
            #else
                while(FindNextFileA(hFind, &FindFileData));
            #endif
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
                        list.push_back(static_cast<std::string>(dirp->d_name));
                    else
                        if (std::string(dirp->d_name).find(exten) != std::string::npos)
                            list.push_back(static_cast<std::string>(dirp->d_name));
                }
            }
            closedir(dp);
        #endif

        return list;
    }

    std::vector<std::string> Directory::GetListFilesR ( const std::string& path, const std::string & exten, bool addPath )
    {
        std::vector<std::string> list = Directory::GetListFiles(path, exten, addPath);

        std::vector<std::string> dirs = Directory::GetListFolders(path, exten, addPath);

        std::vector<std::string>::const_iterator it;
        for (it = dirs.begin(); it != dirs.end(); ++it)
        {
            std::vector<std::string> cl = Directory::GetListFiles(*it, exten, addPath);
            list.insert(list.end(), cl.begin(), cl.end());
        }

        return list;
    }

}
