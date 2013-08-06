#include "opencv2/contrib.hpp"
#include "cvconfig.h"

#if defined(WIN32) || defined(_WIN32)
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
        #ifdef HAVE_WINRT
            WIN32_FIND_DATAW FindFileData;
        #else
            WIN32_FIND_DATAA FindFileData;
        #endif
        HANDLE hFind;

        #ifdef HAVE_WINRT
            size_t size = mbstowcs(NULL, path_f.c_str(), path_f.size());
            Ptr<wchar_t> wpath = new wchar_t[size+1];
            wpath[size] = 0;
            mbstowcs(wpath, path_f.c_str(), path_f.size());
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
                        cv::Ptr<char> fname;
                    #ifdef HAVE_WINRT
                        size_t asize = wcstombs(NULL, FindFileData.cFileName, 0);
                        fname = new char[asize+1];
                        fname[asize] = 0;
                        wcstombs(fname, FindFileData.cFileName, asize);
                    #else
                        fname = FindFileData.cFileName;
                    #endif
                        if (addPath)
                            list.push_back(path + "/" + String(fname));
                        else
                            list.push_back(String(fname));
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
        #ifdef HAVE_WINRT
            WIN32_FIND_DATAW FindFileData;
        #else
            WIN32_FIND_DATAA FindFileData;
        #endif
            HANDLE hFind;

        #ifdef HAVE_WINRT
            size_t size = mbstowcs(NULL, path_f.c_str(), path_f.size());
            Ptr<wchar_t> wpath = new wchar_t[size+1];
            wpath[size] = 0;
            mbstowcs(wpath, path_f.c_str(), path_f.size());
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
                        cv::Ptr<char> fname;
                    #ifdef HAVE_WINRT
                        size_t asize = wcstombs(NULL, FindFileData.cFileName, 0);
                        fname = new char[asize+1];
                        fname[asize] = 0;
                        wcstombs(fname, FindFileData.cFileName, asize);
                    #else
                        fname = FindFileData.cFileName;
                    #endif

                        if (addPath)
                            list.push_back(path + "/" + String(fname));
                        else
                            list.push_back(String(fname));
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
