/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2008-2013, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and / or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

#if defined WIN32 || defined _WIN32 || defined WINCE
# include <windows.h>
const char dir_separators[] = "/\\";
const char native_separator = '\\';

namespace
{
    struct dirent
    {
        const char* d_name;
    };

    struct DIR
    {
#ifdef WINRT
        WIN32_FIND_DATAW data;
#else
        WIN32_FIND_DATA data;
#endif
        HANDLE handle;
        dirent ent;
#ifdef WINRT
        DIR() { }
        ~DIR()
        {
            if (ent.d_name)
                delete[] ent.d_name;
        }
#endif
    };

    DIR* opendir(const char* path)
    {
        DIR* dir = new DIR;
        dir->ent.d_name = 0;
#ifdef WINRT
        cv::String full_path = cv::String(path) + "\\*";
        wchar_t wfull_path[MAX_PATH];
        size_t copied = mbstowcs(wfull_path, full_path.c_str(), MAX_PATH);
        CV_Assert((copied != MAX_PATH) && (copied != (size_t)-1));
        dir->handle = ::FindFirstFileExW(wfull_path, FindExInfoStandard,
                        &dir->data, FindExSearchNameMatch, NULL, 0);
#else
        dir->handle = ::FindFirstFileExA((cv::String(path) + "\\*").c_str(),
            FindExInfoStandard, &dir->data, FindExSearchNameMatch, NULL, 0);
#endif
        if(dir->handle == INVALID_HANDLE_VALUE)
        {
            /*closedir will do all cleanup*/
            delete dir;
            return 0;
        }
        return dir;
    }

    dirent* readdir(DIR* dir)
    {
#ifdef WINRT
        if (dir->ent.d_name != 0)
        {
            if (::FindNextFileW(dir->handle, &dir->data) != TRUE)
                return 0;
        }
        size_t asize = wcstombs(NULL, dir->data.cFileName, 0);
        CV_Assert((asize != 0) && (asize != (size_t)-1));
        char* aname = new char[asize+1];
        aname[asize] = 0;
        wcstombs(aname, dir->data.cFileName, asize);
        dir->ent.d_name = aname;
#else
        if (dir->ent.d_name != 0)
        {
            if (::FindNextFileA(dir->handle, &dir->data) != TRUE)
                return 0;
        }
        dir->ent.d_name = dir->data.cFileName;
#endif
        return &dir->ent;
    }

    void closedir(DIR* dir)
    {
        ::FindClose(dir->handle);
        delete dir;
    }


}
#else
# include <dirent.h>
# include <sys/stat.h>
const char dir_separators[] = "/";
const char native_separator = '/';
#endif

static bool isDir(const cv::String& path, DIR* dir)
{
#if defined WIN32 || defined _WIN32 || defined WINCE
    DWORD attributes;
    BOOL status = TRUE;
    if (dir)
        attributes = dir->data.dwFileAttributes;
    else
    {
        WIN32_FILE_ATTRIBUTE_DATA all_attrs;
#ifdef WINRT
        wchar_t wpath[MAX_PATH];
        size_t copied = mbstowcs(wpath, path.c_str(), MAX_PATH);
        CV_Assert((copied != MAX_PATH) && (copied != (size_t)-1));
        status = ::GetFileAttributesExW(wpath, GetFileExInfoStandard, &all_attrs);
#else
        status = ::GetFileAttributesExA(path.c_str(), GetFileExInfoStandard, &all_attrs);
#endif
        attributes = all_attrs.dwFileAttributes;
    }

    return status && ((attributes & FILE_ATTRIBUTE_DIRECTORY) != 0);
#else
    (void)dir;
    struct stat stat_buf;
    if (0 != stat( path.c_str(), &stat_buf))
        return false;
    int is_dir = S_ISDIR( stat_buf.st_mode);
    return is_dir != 0;
#endif
}

static bool wildcmp(const char *string, const char *wild)
{
    // Based on wildcmp written by Jack Handy - <A href="mailto:jakkhandy@hotmail.com">jakkhandy@hotmail.com</A>
    const char *cp = 0, *mp = 0;

    while ((*string) && (*wild != '*'))
    {
        if ((*wild != *string) && (*wild != '?'))
        {
            return false;
        }

        wild++;
        string++;
    }

    while (*string)
    {
        if (*wild == '*')
        {
            if (!*++wild)
            {
                return true;
            }

            mp = wild;
            cp = string + 1;
        }
        else if ((*wild == *string) || (*wild == '?'))
        {
            wild++;
            string++;
        }
        else
        {
            wild = mp;
            string = cp++;
        }
    }

    while (*wild == '*')
    {
        wild++;
    }

    return *wild == 0;
}

static void glob_rec(const cv::String& directory, const cv::String& wildchart, std::vector<cv::String>& result, bool recursive)
{
    DIR *dir;
    struct dirent *ent;

    if ((dir = opendir (directory.c_str())) != 0)
    {
        /* find all the files and directories within directory */
        try
        {
            while ((ent = readdir (dir)) != 0)
            {
                const char* name = ent->d_name;
                if((name[0] == 0) || (name[0] == '.' && name[1] == 0) || (name[0] == '.' && name[1] == '.' && name[2] == 0))
                    continue;

                cv::String path = directory + native_separator + name;

                if (isDir(path, dir))
                {
                    if (recursive)
                        glob_rec(path, wildchart, result, recursive);
                }
                else
                {
                    if (wildchart.empty() || wildcmp(name, wildchart.c_str()))
                        result.push_back(path);
                }
            }
        }
        catch (...)
        {
            closedir(dir);
            throw;
        }
        closedir(dir);
    }
    else CV_Error(CV_StsObjectNotFound, cv::format("could not open directory: %s", directory.c_str()));
}

void cv::glob(String pattern, std::vector<String>& result, bool recursive)
{
    result.clear();
    String path, wildchart;

    if (isDir(pattern, 0))
    {
        if(strchr(dir_separators, pattern[pattern.size() - 1]) != 0)
        {
            path = pattern.substr(0, pattern.size() - 1);
        }
        else
        {
            path = pattern;
        }
    }
    else
    {
        size_t pos = pattern.find_last_of(dir_separators);
        if (pos == String::npos)
        {
            wildchart = pattern;
            path = ".";
        }
        else
        {
            path = pattern.substr(0, pos);
            wildchart = pattern.substr(pos + 1);
        }
    }

    glob_rec(path, wildchart, result, recursive);
    std::sort(result.begin(), result.end());
}
