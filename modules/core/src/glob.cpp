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
        WIN32_FIND_DATA data;
        HANDLE handle;
        dirent ent;
    };

    DIR* opendir(const char* path)
    {
        DIR* dir = new DIR;
        dir->ent.d_name = 0;
        dir->handle = ::FindFirstFileA((std::string(path) + "\\*").c_str(), &dir->data);
        if(dir->handle == INVALID_HANDLE_VALUE)
        {
            /*closedir will do all cleanup*/
            return 0;
        }
        return dir;
    }

    dirent* readdir(DIR* dir)
    {
        if (dir->ent.d_name != 0)
        {
            if (::FindNextFile(dir->handle, &dir->data) != TRUE)
                return 0;
        }
        dir->ent.d_name = dir->data.cFileName;
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

static bool isDir(const std::string& path, DIR* dir)
{
#if defined WIN32 || defined _WIN32 || defined WINCE
    DWORD attributes;
    if (dir)
        attributes = dir->data.dwFileAttributes;
    else
        attributes = ::GetFileAttributes(path.c_str());

    return (attributes != INVALID_FILE_ATTRIBUTES) && ((attributes & FILE_ATTRIBUTE_DIRECTORY) != 0);
#else
    struct stat stat_buf;
    stat( path.c_str(), &stat_buf);
    int is_dir = S_ISDIR( stat_buf.st_mode);
    (void)dir;

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

static void glob_rec(const std::string& directory, const std::string& wildchart, std::vector<std::string>& result, bool recursive)
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

                std::string path = directory + native_separator + name;

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

void cv::glob(std::string pattern, std::vector<std::string>& result, bool recursive)
{
    result.clear();
    std::string path, wildchart;

    if (isDir(pattern, 0))
    {
        printf("WE ARE HERE: %s\n", pattern.c_str());
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
        if (pos == std::string::npos)
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