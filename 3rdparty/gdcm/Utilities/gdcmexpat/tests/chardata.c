/* Copyright (c) 1998-2003 Thai Open Source Software Center Ltd
   See the file COPYING for copying permission.

   chardata.c
*/

#ifdef HAVE_EXPAT_CONFIG_H
#include <expat_config.h>
#endif
#ifdef HAVE_CHECK_H
#include <check.h>
#else
#include "minicheck.h"
#endif

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "chardata.h"


static int
xmlstrlen(const XML_Char *s)
{
    int len = 0;
    assert(s != NULL);
    while (s[len] != 0)
        ++len;
    return len;
}


void
CharData_Init(CharData *storage)
{
    assert(storage != NULL);
    storage->count = -1;
}

void
CharData_AppendString(CharData *storage, const char *s)
{
    int maxchars = sizeof(storage->data) / sizeof(storage->data[0]);
    int len;

    assert(s != NULL);
    len = strlen(s);
    if (storage->count < 0)
        storage->count = 0;
    if ((len + storage->count) > maxchars) {
        len = (maxchars - storage->count);
    }
    if (len + storage->count < sizeof(storage->data)) {
        memcpy(storage->data + storage->count, s, len);
        storage->count += len;
    }
}

void
CharData_AppendXMLChars(CharData *storage, const XML_Char *s, int len)
{
    int maxchars;

    assert(storage != NULL);
    assert(s != NULL);
    maxchars = sizeof(storage->data) / sizeof(storage->data[0]);
    if (storage->count < 0)
        storage->count = 0;
    if (len < 0)
        len = xmlstrlen(s);
    if ((len + storage->count) > maxchars) {
        len = (maxchars - storage->count);
    }
    if (len + storage->count < sizeof(storage->data)) {
        memcpy(storage->data + storage->count, s,
               len * sizeof(storage->data[0]));
        storage->count += len;
    }
}

int
CharData_CheckString(CharData *storage, const char *expected)
{
    char buffer[1280];
    int len;
    int count;

    assert(storage != NULL);
    assert(expected != NULL);
    count = (storage->count < 0) ? 0 : storage->count;
    len = strlen(expected);
    if (len != count) {
        if (sizeof(XML_Char) == 1)
            sprintf(buffer, "wrong number of data characters:"
                    " got %d, expected %d:\n%s", count, len, storage->data);
        else
            sprintf(buffer,
                    "wrong number of data characters: got %d, expected %d",
                    count, len);
        fail(buffer);
        return 0;
    }
    if (memcmp(expected, storage->data, len) != 0) {
        fail("got bad data bytes");
        return 0;
    }
    return 1;
}

int
CharData_CheckXMLChars(CharData *storage, const XML_Char *expected)
{
    char buffer[1024];
    int len = xmlstrlen(expected);
    int count;

    assert(storage != NULL);
    count = (storage->count < 0) ? 0 : storage->count;
    if (len != count) {
        sprintf(buffer, "wrong number of data characters: got %d, expected %d",
                count, len);
        fail(buffer);
        return 0;
    }
    if (memcmp(expected, storage->data, len * sizeof(storage->data[0])) != 0) {
        fail("got bad data bytes");
        return 0;
    }
    return 1;
}
