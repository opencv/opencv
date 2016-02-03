/* Copyright (c) 1998, 1999, 2000 Thai Open Source Software Center Ltd
   See the file COPYING for copying permission.

   runtest.c : run the Expat test suite
*/

#ifdef HAVE_EXPAT_CONFIG_H
#include <expat_config.h>
#endif

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "expat.h"
#include "chardata.h"
#include "minicheck.h"

#ifdef AMIGA_SHARED_LIB
#include <proto/expat.h>
#endif

#ifdef XML_LARGE_SIZE
#define XML_FMT_INT_MOD "ll"
#else
#define XML_FMT_INT_MOD "l"
#endif

static XML_Parser parser;


static void
basic_setup(void)
{
    parser = XML_ParserCreate(NULL);
    if (parser == NULL)
        fail("Parser not created.");
}

static void
basic_teardown(void)
{
    if (parser != NULL)
        XML_ParserFree(parser);
}

/* Generate a failure using the parser state to create an error message;
   this should be used when the parser reports an error we weren't
   expecting.
*/
static void
_xml_failure(XML_Parser parser, const char *file, int line)
{
    char buffer[1024];
    sprintf(buffer,
            "\n    %s (line %" XML_FMT_INT_MOD "u, offset %"\
                XML_FMT_INT_MOD "u)\n    reported from %s, line %d",
            XML_ErrorString(XML_GetErrorCode(parser)),
            XML_GetCurrentLineNumber(parser),
            XML_GetCurrentColumnNumber(parser),
            file, line);
    _fail_unless(0, file, line, buffer);
}

#define xml_failure(parser) _xml_failure((parser), __FILE__, __LINE__)

static void
_expect_failure(char *text, enum XML_Error errorCode, char *errorMessage,
                char *file, int lineno)
{
    if (XML_Parse(parser, text, strlen(text), XML_TRUE) == XML_STATUS_OK)
        /* Hackish use of _fail_unless() macro, but let's us report
           the right filename and line number. */
        _fail_unless(0, file, lineno, errorMessage);
    if (XML_GetErrorCode(parser) != errorCode)
        _xml_failure(parser, file, lineno);
}

#define expect_failure(text, errorCode, errorMessage) \
        _expect_failure((text), (errorCode), (errorMessage), \
                        __FILE__, __LINE__)

/* Dummy handlers for when we need to set a handler to tickle a bug,
   but it doesn't need to do anything.
*/

static void XMLCALL
dummy_start_doctype_handler(void           *userData,
                            const XML_Char *doctypeName,
                            const XML_Char *sysid,
                            const XML_Char *pubid,
                            int            has_internal_subset)
{}

static void XMLCALL
dummy_end_doctype_handler(void *userData)
{}

static void XMLCALL
dummy_entity_decl_handler(void           *userData,
                          const XML_Char *entityName,
                          int            is_parameter_entity,
                          const XML_Char *value,
                          int            value_length,
                          const XML_Char *base,
                          const XML_Char *systemId,
                          const XML_Char *publicId,
                          const XML_Char *notationName)
{}

static void XMLCALL
dummy_notation_decl_handler(void *userData,
                            const XML_Char *notationName,
                            const XML_Char *base,
                            const XML_Char *systemId,
                            const XML_Char *publicId)
{}

static void XMLCALL
dummy_element_decl_handler(void *userData,
                           const XML_Char *name,
                           XML_Content *model)
{}

static void XMLCALL
dummy_attlist_decl_handler(void           *userData,
                           const XML_Char *elname,
                           const XML_Char *attname,
                           const XML_Char *att_type,
                           const XML_Char *dflt,
                           int            isrequired)
{}

static void XMLCALL
dummy_comment_handler(void *userData, const XML_Char *data)
{}

static void XMLCALL
dummy_pi_handler(void *userData, const XML_Char *target, const XML_Char *data)
{}

static void XMLCALL
dummy_start_element(void *userData,
                    const XML_Char *name, const XML_Char **atts)
{}


/*
 * Character & encoding tests.
 */

START_TEST(test_nul_byte)
{
    char text[] = "<doc>\0</doc>";

    /* test that a NUL byte (in US-ASCII data) is an error */
    if (XML_Parse(parser, text, sizeof(text) - 1, XML_TRUE) == XML_STATUS_OK)
        fail("Parser did not report error on NUL-byte.");
    if (XML_GetErrorCode(parser) != XML_ERROR_INVALID_TOKEN)
        xml_failure(parser);
}
END_TEST


START_TEST(test_u0000_char)
{
    /* test that a NUL byte (in US-ASCII data) is an error */
    expect_failure("<doc>&#0;</doc>",
                   XML_ERROR_BAD_CHAR_REF,
                   "Parser did not report error on NUL-byte.");
}
END_TEST

START_TEST(test_bom_utf8)
{
    /* This test is really just making sure we don't core on a UTF-8 BOM. */
    char *text = "\357\273\277<e/>";

    if (XML_Parse(parser, text, strlen(text), XML_TRUE) == XML_STATUS_ERROR)
        xml_failure(parser);
}
END_TEST

START_TEST(test_bom_utf16_be)
{
    char text[] = "\376\377\0<\0e\0/\0>";

    if (XML_Parse(parser, text, sizeof(text)-1, XML_TRUE) == XML_STATUS_ERROR)
        xml_failure(parser);
}
END_TEST

START_TEST(test_bom_utf16_le)
{
    char text[] = "\377\376<\0e\0/\0>\0";

    if (XML_Parse(parser, text, sizeof(text)-1, XML_TRUE) == XML_STATUS_ERROR)
        xml_failure(parser);
}
END_TEST

static void XMLCALL
accumulate_characters(void *userData, const XML_Char *s, int len)
{
    CharData_AppendXMLChars((CharData *)userData, s, len);
}

static void XMLCALL
accumulate_attribute(void *userData, const XML_Char *name,
                     const XML_Char **atts)
{
    CharData *storage = (CharData *)userData;
    if (storage->count < 0 && atts != NULL && atts[0] != NULL) {
        /* "accumulate" the value of the first attribute we see */
        CharData_AppendXMLChars(storage, atts[1], -1);
    }
}


static void
_run_character_check(XML_Char *text, XML_Char *expected,
                     const char *file, int line)
{
    CharData storage;

    CharData_Init(&storage);
    XML_SetUserData(parser, &storage);
    XML_SetCharacterDataHandler(parser, accumulate_characters);
    if (XML_Parse(parser, text, strlen(text), XML_TRUE) == XML_STATUS_ERROR)
        _xml_failure(parser, file, line);
    CharData_CheckXMLChars(&storage, expected);
}

#define run_character_check(text, expected) \
        _run_character_check(text, expected, __FILE__, __LINE__)

static void
_run_attribute_check(XML_Char *text, XML_Char *expected,
                     const char *file, int line)
{
    CharData storage;

    CharData_Init(&storage);
    XML_SetUserData(parser, &storage);
    XML_SetStartElementHandler(parser, accumulate_attribute);
    if (XML_Parse(parser, text, strlen(text), XML_TRUE) == XML_STATUS_ERROR)
        _xml_failure(parser, file, line);
    CharData_CheckXMLChars(&storage, expected);
}

#define run_attribute_check(text, expected) \
        _run_attribute_check(text, expected, __FILE__, __LINE__)

/* Regression test for SF bug #491986. */
START_TEST(test_danish_latin1)
{
    char *text =
        "<?xml version='1.0' encoding='iso-8859-1'?>\n"
        "<e>J\xF8rgen \xE6\xF8\xE5\xC6\xD8\xC5</e>";
    run_character_check(text,
             "J\xC3\xB8rgen \xC3\xA6\xC3\xB8\xC3\xA5\xC3\x86\xC3\x98\xC3\x85");
}
END_TEST


/* Regression test for SF bug #514281. */
START_TEST(test_french_charref_hexidecimal)
{
    char *text =
        "<?xml version='1.0' encoding='iso-8859-1'?>\n"
        "<doc>&#xE9;&#xE8;&#xE0;&#xE7;&#xEA;&#xC8;</doc>";
    run_character_check(text,
                        "\xC3\xA9\xC3\xA8\xC3\xA0\xC3\xA7\xC3\xAA\xC3\x88");
}
END_TEST

START_TEST(test_french_charref_decimal)
{
    char *text =
        "<?xml version='1.0' encoding='iso-8859-1'?>\n"
        "<doc>&#233;&#232;&#224;&#231;&#234;&#200;</doc>";
    run_character_check(text,
                        "\xC3\xA9\xC3\xA8\xC3\xA0\xC3\xA7\xC3\xAA\xC3\x88");
}
END_TEST

START_TEST(test_french_latin1)
{
    char *text =
        "<?xml version='1.0' encoding='iso-8859-1'?>\n"
        "<doc>\xE9\xE8\xE0\xE7\xEa\xC8</doc>";
    run_character_check(text,
                        "\xC3\xA9\xC3\xA8\xC3\xA0\xC3\xA7\xC3\xAA\xC3\x88");
}
END_TEST

START_TEST(test_french_utf8)
{
    char *text =
        "<?xml version='1.0' encoding='utf-8'?>\n"
        "<doc>\xC3\xA9</doc>";
    run_character_check(text, "\xC3\xA9");
}
END_TEST

/* Regression test for SF bug #600479.
   XXX There should be a test that exercises all legal XML Unicode
   characters as PCDATA and attribute value content, and XML Name
   characters as part of element and attribute names.
*/
START_TEST(test_utf8_false_rejection)
{
    char *text = "<doc>\xEF\xBA\xBF</doc>";
    run_character_check(text, "\xEF\xBA\xBF");
}
END_TEST

/* Regression test for SF bug #477667.
   This test assures that any 8-bit character followed by a 7-bit
   character will not be mistakenly interpreted as a valid UTF-8
   sequence.
*/
START_TEST(test_illegal_utf8)
{
    char text[100];
    int i;

    for (i = 128; i <= 255; ++i) {
        sprintf(text, "<e>%ccd</e>", i);
        if (XML_Parse(parser, text, strlen(text), XML_TRUE) == XML_STATUS_OK) {
            sprintf(text,
                    "expected token error for '%c' (ordinal %d) in UTF-8 text",
                    i, i);
            fail(text);
        }
        else if (XML_GetErrorCode(parser) != XML_ERROR_INVALID_TOKEN)
            xml_failure(parser);
        /* Reset the parser since we use the same parser repeatedly. */
        XML_ParserReset(parser, NULL);
    }
}
END_TEST

START_TEST(test_utf16)
{
    /* <?xml version="1.0" encoding="UTF-16"?>
       <doc a='123'>some text</doc>
    */
    char text[] =
        "\000<\000?\000x\000m\000\154\000 \000v\000e\000r\000s\000i\000o"
        "\000n\000=\000'\0001\000.\000\060\000'\000 \000e\000n\000c\000o"
        "\000d\000i\000n\000g\000=\000'\000U\000T\000F\000-\0001\000\066"
        "\000'\000?\000>\000\n"
        "\000<\000d\000o\000c\000 \000a\000=\000'\0001\0002\0003\000'"
        "\000>\000s\000o\000m\000e\000 \000t\000e\000x\000t\000<\000/"
        "\000d\000o\000c\000>";
    if (XML_Parse(parser, text, sizeof(text)-1, XML_TRUE) == XML_STATUS_ERROR)
        xml_failure(parser);
}
END_TEST

START_TEST(test_utf16_le_epilog_newline)
{
    int first_chunk_bytes = 17;
    char text[] = 
        "\xFF\xFE"                      /* BOM */
        "<\000e\000/\000>\000"          /* document element */
        "\r\000\n\000\r\000\n\000";     /* epilog */

    if (first_chunk_bytes >= sizeof(text) - 1)
        fail("bad value of first_chunk_bytes");
    if (  XML_Parse(parser, text, first_chunk_bytes, XML_FALSE)
          == XML_STATUS_ERROR)
        xml_failure(parser);
    else {
        enum XML_Status rc;
        rc = XML_Parse(parser, text + first_chunk_bytes,
                       sizeof(text) - first_chunk_bytes - 1, XML_TRUE);
        if (rc == XML_STATUS_ERROR)
            xml_failure(parser);
    }
}
END_TEST

/* Regression test for SF bug #481609, #774028. */
START_TEST(test_latin1_umlauts)
{
    char *text =
        "<?xml version='1.0' encoding='iso-8859-1'?>\n"
        "<e a='\xE4 \xF6 \xFC &#228; &#246; &#252; &#x00E4; &#x0F6; &#xFC; >'\n"
        "  >\xE4 \xF6 \xFC &#228; &#246; &#252; &#x00E4; &#x0F6; &#xFC; ></e>";
    char *utf8 =
        "\xC3\xA4 \xC3\xB6 \xC3\xBC "
        "\xC3\xA4 \xC3\xB6 \xC3\xBC "
        "\xC3\xA4 \xC3\xB6 \xC3\xBC >";
    run_character_check(text, utf8);
    XML_ParserReset(parser, NULL);
    run_attribute_check(text, utf8);
}
END_TEST

/* Regression test #1 for SF bug #653180. */
START_TEST(test_line_number_after_parse)
{  
    char *text =
        "<tag>\n"
        "\n"
        "\n</tag>";
    XML_Size lineno;

    if (XML_Parse(parser, text, strlen(text), XML_FALSE) == XML_STATUS_ERROR)
        xml_failure(parser);
    lineno = XML_GetCurrentLineNumber(parser);
    if (lineno != 4) {
        char buffer[100];
        sprintf(buffer, 
            "expected 4 lines, saw %" XML_FMT_INT_MOD "u", lineno);
        fail(buffer);
    }
}
END_TEST

/* Regression test #2 for SF bug #653180. */
START_TEST(test_column_number_after_parse)
{
    char *text = "<tag></tag>";
    XML_Size colno;

    if (XML_Parse(parser, text, strlen(text), XML_FALSE) == XML_STATUS_ERROR)
        xml_failure(parser);
    colno = XML_GetCurrentColumnNumber(parser);
    if (colno != 11) {
        char buffer[100];
        sprintf(buffer, 
            "expected 11 columns, saw %" XML_FMT_INT_MOD "u", colno);
        fail(buffer);
    }
}
END_TEST

static void XMLCALL
start_element_event_handler2(void *userData, const XML_Char *name,
			     const XML_Char **attr)
{
    CharData *storage = (CharData *) userData;
    char buffer[100];

    sprintf(buffer,
        "<%s> at col:%" XML_FMT_INT_MOD "u line:%"\
            XML_FMT_INT_MOD "u\n", name,
	    XML_GetCurrentColumnNumber(parser),
	    XML_GetCurrentLineNumber(parser));
    CharData_AppendString(storage, buffer);
}

static void XMLCALL
end_element_event_handler2(void *userData, const XML_Char *name)
{
    CharData *storage = (CharData *) userData;
    char buffer[100];

    sprintf(buffer,
        "</%s> at col:%" XML_FMT_INT_MOD "u line:%"\
            XML_FMT_INT_MOD "u\n", name,
	    XML_GetCurrentColumnNumber(parser),
	    XML_GetCurrentLineNumber(parser));
    CharData_AppendString(storage, buffer);
}

/* Regression test #3 for SF bug #653180. */
START_TEST(test_line_and_column_numbers_inside_handlers)
{
    char *text =
        "<a>\n"        /* Unix end-of-line */
        "  <b>\r\n"    /* Windows end-of-line */
        "    <c/>\r"   /* Mac OS end-of-line */
        "  </b>\n"
        "  <d>\n"
        "    <f/>\n"
        "  </d>\n"
        "</a>";
    char *expected =
        "<a> at col:0 line:1\n"
        "<b> at col:2 line:2\n"
        "<c> at col:4 line:3\n"
        "</c> at col:8 line:3\n"
        "</b> at col:2 line:4\n"
        "<d> at col:2 line:5\n"
        "<f> at col:4 line:6\n"
        "</f> at col:8 line:6\n"
        "</d> at col:2 line:7\n"
        "</a> at col:0 line:8\n";
    CharData storage;

    CharData_Init(&storage);
    XML_SetUserData(parser, &storage);
    XML_SetStartElementHandler(parser, start_element_event_handler2);
    XML_SetEndElementHandler(parser, end_element_event_handler2);
    if (XML_Parse(parser, text, strlen(text), XML_TRUE) == XML_STATUS_ERROR)
        xml_failure(parser);

    CharData_CheckString(&storage, expected); 
}
END_TEST

/* Regression test #4 for SF bug #653180. */
START_TEST(test_line_number_after_error)
{
    char *text =
        "<a>\n"
        "  <b>\n"
        "  </a>";  /* missing </b> */
    XML_Size lineno;
    if (XML_Parse(parser, text, strlen(text), XML_FALSE) != XML_STATUS_ERROR)
        fail("Expected a parse error");

    lineno = XML_GetCurrentLineNumber(parser);
    if (lineno != 3) {
        char buffer[100];
        sprintf(buffer, "expected 3 lines, saw %" XML_FMT_INT_MOD "u", lineno);
        fail(buffer);
    }
}
END_TEST
    
/* Regression test #5 for SF bug #653180. */
START_TEST(test_column_number_after_error)
{
    char *text =
        "<a>\n"
        "  <b>\n"
        "  </a>";  /* missing </b> */
    XML_Size colno;
    if (XML_Parse(parser, text, strlen(text), XML_FALSE) != XML_STATUS_ERROR)
        fail("Expected a parse error");

    colno = XML_GetCurrentColumnNumber(parser);
    if (colno != 4) { 
        char buffer[100];
        sprintf(buffer, 
            "expected 4 columns, saw %" XML_FMT_INT_MOD "u", colno);
        fail(buffer);
    }
}
END_TEST

/* Regression test for SF bug #478332. */
START_TEST(test_really_long_lines)
{
    /* This parses an input line longer than INIT_DATA_BUF_SIZE
       characters long (defined to be 1024 in xmlparse.c).  We take a
       really cheesy approach to building the input buffer, because
       this avoids writing bugs in buffer-filling code.
    */
    char *text =
        "<e>"
        /* 64 chars */
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-+"
        /* until we have at least 1024 characters on the line: */
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-+"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-+"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-+"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-+"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-+"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-+"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-+"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-+"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-+"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-+"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-+"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-+"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-+"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-+"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-+"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-+"
        "</e>";
    if (XML_Parse(parser, text, strlen(text), XML_TRUE) == XML_STATUS_ERROR)
        xml_failure(parser);
}
END_TEST


/*
 * Element event tests.
 */

static void XMLCALL
end_element_event_handler(void *userData, const XML_Char *name)
{
    CharData *storage = (CharData *) userData;
    CharData_AppendString(storage, "/");
    CharData_AppendXMLChars(storage, name, -1);
}

START_TEST(test_end_element_events)
{
    char *text = "<a><b><c/></b><d><f/></d></a>";
    char *expected = "/c/b/f/d/a";
    CharData storage;

    CharData_Init(&storage);
    XML_SetUserData(parser, &storage);
    XML_SetEndElementHandler(parser, end_element_event_handler);
    if (XML_Parse(parser, text, strlen(text), XML_TRUE) == XML_STATUS_ERROR)
        xml_failure(parser);
    CharData_CheckString(&storage, expected);
}
END_TEST


/*
 * Attribute tests.
 */

/* Helpers used by the following test; this checks any "attr" and "refs"
   attributes to make sure whitespace has been normalized.

   Return true if whitespace has been normalized in a string, using
   the rules for attribute value normalization.  The 'is_cdata' flag
   is needed since CDATA attributes don't need to have multiple
   whitespace characters collapsed to a single space, while other
   attribute data types do.  (Section 3.3.3 of the recommendation.)
*/
static int
is_whitespace_normalized(const XML_Char *s, int is_cdata)
{
    int blanks = 0;
    int at_start = 1;
    while (*s) {
        if (*s == ' ')
            ++blanks;
        else if (*s == '\t' || *s == '\n' || *s == '\r')
            return 0;
        else {
            if (at_start) {
                at_start = 0;
                if (blanks && !is_cdata)
                    /* illegal leading blanks */
                    return 0;
            }
            else if (blanks > 1 && !is_cdata)
                return 0;
            blanks = 0;
        }
        ++s;
    }
    if (blanks && !is_cdata)
        return 0;
    return 1;
}

/* Check the attribute whitespace checker: */
static void
testhelper_is_whitespace_normalized(void)
{
    assert(is_whitespace_normalized("abc", 0));
    assert(is_whitespace_normalized("abc", 1));
    assert(is_whitespace_normalized("abc def ghi", 0));
    assert(is_whitespace_normalized("abc def ghi", 1));
    assert(!is_whitespace_normalized(" abc def ghi", 0));
    assert(is_whitespace_normalized(" abc def ghi", 1));
    assert(!is_whitespace_normalized("abc  def ghi", 0));
    assert(is_whitespace_normalized("abc  def ghi", 1));
    assert(!is_whitespace_normalized("abc def ghi ", 0));
    assert(is_whitespace_normalized("abc def ghi ", 1));
    assert(!is_whitespace_normalized(" ", 0));
    assert(is_whitespace_normalized(" ", 1));
    assert(!is_whitespace_normalized("\t", 0));
    assert(!is_whitespace_normalized("\t", 1));
    assert(!is_whitespace_normalized("\n", 0));
    assert(!is_whitespace_normalized("\n", 1));
    assert(!is_whitespace_normalized("\r", 0));
    assert(!is_whitespace_normalized("\r", 1));
    assert(!is_whitespace_normalized("abc\t def", 1));
}

static void XMLCALL
check_attr_contains_normalized_whitespace(void *userData,
                                          const XML_Char *name,
                                          const XML_Char **atts)
{
    int i;
    for (i = 0; atts[i] != NULL; i += 2) {
        const XML_Char *attrname = atts[i];
        const XML_Char *value = atts[i + 1];
        if (strcmp("attr", attrname) == 0
            || strcmp("ents", attrname) == 0
            || strcmp("refs", attrname) == 0) {
            if (!is_whitespace_normalized(value, 0)) {
                char buffer[256];
                sprintf(buffer, "attribute value not normalized: %s='%s'",
                        attrname, value);
                fail(buffer);
            }
        }
    }
}

START_TEST(test_attr_whitespace_normalization)
{
    char *text =
        "<!DOCTYPE doc [\n"
        "  <!ATTLIST doc\n"
        "            attr NMTOKENS #REQUIRED\n"
        "            ents ENTITIES #REQUIRED\n"
        "            refs IDREFS   #REQUIRED>\n"
        "]>\n"
        "<doc attr='    a  b c\t\td\te\t' refs=' id-1   \t  id-2\t\t'  \n"
        "     ents=' ent-1   \t\r\n"
        "            ent-2  ' >\n"
        "  <e id='id-1'/>\n"
        "  <e id='id-2'/>\n"
        "</doc>";

    XML_SetStartElementHandler(parser,
                               check_attr_contains_normalized_whitespace);
    if (XML_Parse(parser, text, strlen(text), XML_TRUE) == XML_STATUS_ERROR)
        xml_failure(parser);
}
END_TEST


/*
 * XML declaration tests.
 */

START_TEST(test_xmldecl_misplaced)
{
    expect_failure("\n"
                   "<?xml version='1.0'?>\n"
                   "<a/>",
                   XML_ERROR_MISPLACED_XML_PI,
                   "failed to report misplaced XML declaration");
}
END_TEST

/* Regression test for SF bug #584832. */
static int XMLCALL
UnknownEncodingHandler(void *data,const XML_Char *encoding,XML_Encoding *info)
{
    if (strcmp(encoding,"unsupported-encoding") == 0) {
        int i;
        for (i = 0; i < 256; ++i)
            info->map[i] = i;
        info->data = NULL;
        info->convert = NULL;
        info->release = NULL;
        return XML_STATUS_OK;
    }
    return XML_STATUS_ERROR;
}

START_TEST(test_unknown_encoding_internal_entity)
{
    char *text =
        "<?xml version='1.0' encoding='unsupported-encoding'?>\n"
        "<!DOCTYPE test [<!ENTITY foo 'bar'>]>\n"
        "<test a='&foo;'/>";

    XML_SetUnknownEncodingHandler(parser, UnknownEncodingHandler, NULL);
    if (XML_Parse(parser, text, strlen(text), XML_TRUE) == XML_STATUS_ERROR)
        xml_failure(parser);
}
END_TEST

/* Regression test for SF bug #620106. */
static int XMLCALL
external_entity_loader_set_encoding(XML_Parser parser,
                                    const XML_Char *context,
                                    const XML_Char *base,
                                    const XML_Char *systemId,
                                    const XML_Char *publicId)
{
    /* This text says it's an unsupported encoding, but it's really
       UTF-8, which we tell Expat using XML_SetEncoding().
    */
    char *text =
        "<?xml encoding='iso-8859-3'?>"
        "\xC3\xA9";
    XML_Parser extparser;

    extparser = XML_ExternalEntityParserCreate(parser, context, NULL);
    if (extparser == NULL)
        fail("Could not create external entity parser.");
    if (!XML_SetEncoding(extparser, "utf-8"))
        fail("XML_SetEncoding() ignored for external entity");
    if (  XML_Parse(extparser, text, strlen(text), XML_TRUE)
          == XML_STATUS_ERROR) {
        xml_failure(parser);
        return 0;
    }
    return 1;
}

START_TEST(test_ext_entity_set_encoding)
{
    char *text =
        "<!DOCTYPE doc [\n"
        "  <!ENTITY en SYSTEM 'http://xml.libexpat.org/dummy.ent'>\n"
        "]>\n"
        "<doc>&en;</doc>";

    XML_SetExternalEntityRefHandler(parser,
                                    external_entity_loader_set_encoding);
    run_character_check(text, "\xC3\xA9");
}
END_TEST

/* Test that no error is reported for unknown entities if we don't
   read an external subset.  This was fixed in Expat 1.95.5.
*/
START_TEST(test_wfc_undeclared_entity_unread_external_subset) {
    char *text =
        "<!DOCTYPE doc SYSTEM 'foo'>\n"
        "<doc>&entity;</doc>";

    if (XML_Parse(parser, text, strlen(text), XML_TRUE) == XML_STATUS_ERROR)
        xml_failure(parser);
}
END_TEST

/* Test that an error is reported for unknown entities if we don't
   have an external subset.
*/
START_TEST(test_wfc_undeclared_entity_no_external_subset) {
    expect_failure("<doc>&entity;</doc>",
                   XML_ERROR_UNDEFINED_ENTITY,
                   "Parser did not report undefined entity w/out a DTD.");
}
END_TEST

/* Test that an error is reported for unknown entities if we don't
   read an external subset, but have been declared standalone.
*/
START_TEST(test_wfc_undeclared_entity_standalone) {
    char *text =
        "<?xml version='1.0' encoding='us-ascii' standalone='yes'?>\n"
        "<!DOCTYPE doc SYSTEM 'foo'>\n"
        "<doc>&entity;</doc>";

    expect_failure(text,
                   XML_ERROR_UNDEFINED_ENTITY,
                   "Parser did not report undefined entity (standalone).");
}
END_TEST

static int XMLCALL
external_entity_loader(XML_Parser parser,
                       const XML_Char *context,
                       const XML_Char *base,
                       const XML_Char *systemId,
                       const XML_Char *publicId)
{
    char *text = (char *)XML_GetUserData(parser);
    XML_Parser extparser;

    extparser = XML_ExternalEntityParserCreate(parser, context, NULL);
    if (extparser == NULL)
        fail("Could not create external entity parser.");
    if (  XML_Parse(extparser, text, strlen(text), XML_TRUE)
          == XML_STATUS_ERROR) {
        xml_failure(parser);
        return XML_STATUS_ERROR;
    }
    return XML_STATUS_OK;
}

/* Test that an error is reported for unknown entities if we have read
   an external subset, and standalone is true.
*/
START_TEST(test_wfc_undeclared_entity_with_external_subset_standalone) {
    char *text =
        "<?xml version='1.0' encoding='us-ascii' standalone='yes'?>\n"
        "<!DOCTYPE doc SYSTEM 'foo'>\n"
        "<doc>&entity;</doc>";
    char *foo_text =
        "<!ELEMENT doc (#PCDATA)*>";

    XML_SetParamEntityParsing(parser, XML_PARAM_ENTITY_PARSING_ALWAYS);
    XML_SetUserData(parser, foo_text);
    XML_SetExternalEntityRefHandler(parser, external_entity_loader);
    expect_failure(text,
                   XML_ERROR_UNDEFINED_ENTITY,
                   "Parser did not report undefined entity (external DTD).");
}
END_TEST

/* Test that no error is reported for unknown entities if we have read
   an external subset, and standalone is false.
*/
START_TEST(test_wfc_undeclared_entity_with_external_subset) {
    char *text =
        "<?xml version='1.0' encoding='us-ascii'?>\n"
        "<!DOCTYPE doc SYSTEM 'foo'>\n"
        "<doc>&entity;</doc>";
    char *foo_text =
        "<!ELEMENT doc (#PCDATA)*>";

    XML_SetParamEntityParsing(parser, XML_PARAM_ENTITY_PARSING_ALWAYS);
    XML_SetUserData(parser, foo_text);
    XML_SetExternalEntityRefHandler(parser, external_entity_loader);
    if (XML_Parse(parser, text, strlen(text), XML_TRUE) == XML_STATUS_ERROR)
        xml_failure(parser);
}
END_TEST

START_TEST(test_wfc_no_recursive_entity_refs)
{
    char *text =
        "<!DOCTYPE doc [\n"
        "  <!ENTITY entity '&#38;entity;'>\n"
        "]>\n"
        "<doc>&entity;</doc>";

    expect_failure(text,
                   XML_ERROR_RECURSIVE_ENTITY_REF,
                   "Parser did not report recursive entity reference.");
}
END_TEST

/* Regression test for SF bug #483514. */
START_TEST(test_dtd_default_handling)
{
    char *text =
        "<!DOCTYPE doc [\n"
        "<!ENTITY e SYSTEM 'http://xml.libexpat.org/e'>\n"
        "<!NOTATION n SYSTEM 'http://xml.libexpat.org/n'>\n"
        "<!ELEMENT doc EMPTY>\n"
        "<!ATTLIST doc a CDATA #IMPLIED>\n"
        "<?pi in dtd?>\n"
        "<!--comment in dtd-->\n"
        "]><doc/>";

    XML_SetDefaultHandler(parser, accumulate_characters);
    XML_SetDoctypeDeclHandler(parser,
                              dummy_start_doctype_handler,
                              dummy_end_doctype_handler);
    XML_SetEntityDeclHandler(parser, dummy_entity_decl_handler);
    XML_SetNotationDeclHandler(parser, dummy_notation_decl_handler);
    XML_SetElementDeclHandler(parser, dummy_element_decl_handler);
    XML_SetAttlistDeclHandler(parser, dummy_attlist_decl_handler);
    XML_SetProcessingInstructionHandler(parser, dummy_pi_handler);
    XML_SetCommentHandler(parser, dummy_comment_handler);
    run_character_check(text, "\n\n\n\n\n\n\n<doc/>");
}
END_TEST

/* See related SF bug #673791.
   When namespace processing is enabled, setting the namespace URI for
   a prefix is not allowed; this test ensures that it *is* allowed
   when namespace processing is not enabled.
   (See Namespaces in XML, section 2.)
*/
START_TEST(test_empty_ns_without_namespaces)
{
    char *text =
        "<doc xmlns:prefix='http://www.example.com/'>\n"
        "  <e xmlns:prefix=''/>\n"
        "</doc>";

    if (XML_Parse(parser, text, strlen(text), XML_TRUE) == XML_STATUS_ERROR)
        xml_failure(parser);
}
END_TEST

/* Regression test for SF bug #824420.
   Checks that an xmlns:prefix attribute set in an attribute's default
   value isn't misinterpreted.
*/
START_TEST(test_ns_in_attribute_default_without_namespaces)
{
    char *text =
        "<!DOCTYPE e:element [\n"
        "  <!ATTLIST e:element\n"
        "    xmlns:e CDATA 'http://example.com/'>\n"
        "      ]>\n"
        "<e:element/>";

    if (XML_Parse(parser, text, strlen(text), XML_TRUE) == XML_STATUS_ERROR)
        xml_failure(parser);
}
END_TEST


/*
 * Namespaces tests.
 */

static void
namespace_setup(void)
{
    parser = XML_ParserCreateNS(NULL, ' ');
    if (parser == NULL)
        fail("Parser not created.");
}

static void
namespace_teardown(void)
{
    basic_teardown();
}

/* Check that an element name and attribute name match the expected values.
   The expected values are passed as an array reference of string pointers
   provided as the userData argument; the first is the expected
   element name, and the second is the expected attribute name.
*/
static void XMLCALL
triplet_start_checker(void *userData, const XML_Char *name,
                      const XML_Char **atts)
{
    char **elemstr = (char **)userData;
    char buffer[1024];
    if (strcmp(elemstr[0], name) != 0) {
        sprintf(buffer, "unexpected start string: '%s'", name);
        fail(buffer);
    }
    if (strcmp(elemstr[1], atts[0]) != 0) {
        sprintf(buffer, "unexpected attribute string: '%s'", atts[0]);
        fail(buffer);
    }
}

/* Check that the element name passed to the end-element handler matches
   the expected value.  The expected value is passed as the first element
   in an array of strings passed as the userData argument.
*/
static void XMLCALL
triplet_end_checker(void *userData, const XML_Char *name)
{
    char **elemstr = (char **)userData;
    if (strcmp(elemstr[0], name) != 0) {
        char buffer[1024];
        sprintf(buffer, "unexpected end string: '%s'", name);
        fail(buffer);
    }
}

START_TEST(test_return_ns_triplet)
{
    char *text =
        "<foo:e xmlns:foo='http://expat.sf.net/' bar:a='12'\n"
        "       xmlns:bar='http://expat.sf.net/'></foo:e>";
    char *elemstr[] = {
        "http://expat.sf.net/ e foo",
        "http://expat.sf.net/ a bar"
    };
    XML_SetReturnNSTriplet(parser, XML_TRUE);
    XML_SetUserData(parser, elemstr);
    XML_SetElementHandler(parser, triplet_start_checker, triplet_end_checker);
    if (XML_Parse(parser, text, strlen(text), XML_TRUE) == XML_STATUS_ERROR)
        xml_failure(parser);
}
END_TEST

static void XMLCALL
overwrite_start_checker(void *userData, const XML_Char *name,
                        const XML_Char **atts)
{
    CharData *storage = (CharData *) userData;
    CharData_AppendString(storage, "start ");
    CharData_AppendXMLChars(storage, name, -1);
    while (*atts != NULL) {
        CharData_AppendString(storage, "\nattribute ");
        CharData_AppendXMLChars(storage, *atts, -1);
        atts += 2;
    }
    CharData_AppendString(storage, "\n");
}

static void XMLCALL
overwrite_end_checker(void *userData, const XML_Char *name)
{
    CharData *storage = (CharData *) userData;
    CharData_AppendString(storage, "end ");
    CharData_AppendXMLChars(storage, name, -1);
    CharData_AppendString(storage, "\n");
}

static void
run_ns_tagname_overwrite_test(char *text, char *result)
{
    CharData storage;
    CharData_Init(&storage);
    XML_SetUserData(parser, &storage);
    XML_SetElementHandler(parser,
                          overwrite_start_checker, overwrite_end_checker);
    if (XML_Parse(parser, text, strlen(text), XML_TRUE) == XML_STATUS_ERROR)
        xml_failure(parser);
    CharData_CheckString(&storage, result);
}

/* Regression test for SF bug #566334. */
START_TEST(test_ns_tagname_overwrite)
{
    char *text =
        "<n:e xmlns:n='http://xml.libexpat.org/'>\n"
        "  <n:f n:attr='foo'/>\n"
        "  <n:g n:attr2='bar'/>\n"
        "</n:e>";
    char *result =
        "start http://xml.libexpat.org/ e\n"
        "start http://xml.libexpat.org/ f\n"
        "attribute http://xml.libexpat.org/ attr\n"
        "end http://xml.libexpat.org/ f\n"
        "start http://xml.libexpat.org/ g\n"
        "attribute http://xml.libexpat.org/ attr2\n"
        "end http://xml.libexpat.org/ g\n"
        "end http://xml.libexpat.org/ e\n";
    run_ns_tagname_overwrite_test(text, result);
}
END_TEST

/* Regression test for SF bug #566334. */
START_TEST(test_ns_tagname_overwrite_triplet)
{
    char *text =
        "<n:e xmlns:n='http://xml.libexpat.org/'>\n"
        "  <n:f n:attr='foo'/>\n"
        "  <n:g n:attr2='bar'/>\n"
        "</n:e>";
    char *result =
        "start http://xml.libexpat.org/ e n\n"
        "start http://xml.libexpat.org/ f n\n"
        "attribute http://xml.libexpat.org/ attr n\n"
        "end http://xml.libexpat.org/ f n\n"
        "start http://xml.libexpat.org/ g n\n"
        "attribute http://xml.libexpat.org/ attr2 n\n"
        "end http://xml.libexpat.org/ g n\n"
        "end http://xml.libexpat.org/ e n\n";
    XML_SetReturnNSTriplet(parser, XML_TRUE);
    run_ns_tagname_overwrite_test(text, result);
}
END_TEST


/* Regression test for SF bug #620343. */
static void XMLCALL
start_element_fail(void *userData,
                   const XML_Char *name, const XML_Char **atts)
{
    /* We should never get here. */
    fail("should never reach start_element_fail()");
}

static void XMLCALL
start_ns_clearing_start_element(void *userData,
                                const XML_Char *prefix,
                                const XML_Char *uri)
{
    XML_SetStartElementHandler((XML_Parser) userData, NULL);
}

START_TEST(test_start_ns_clears_start_element)
{
    /* This needs to use separate start/end tags; using the empty tag
       syntax doesn't cause the problematic path through Expat to be
       taken.
    */
    char *text = "<e xmlns='http://xml.libexpat.org/'></e>";

    XML_SetStartElementHandler(parser, start_element_fail);
    XML_SetStartNamespaceDeclHandler(parser, start_ns_clearing_start_element);
    XML_UseParserAsHandlerArg(parser);
    if (XML_Parse(parser, text, strlen(text), XML_TRUE) == XML_STATUS_ERROR)
        xml_failure(parser);
}
END_TEST

/* Regression test for SF bug #616863. */
static int XMLCALL
external_entity_handler(XML_Parser parser,
                        const XML_Char *context,
                        const XML_Char *base,
                        const XML_Char *systemId,
                        const XML_Char *publicId) 
{
    int callno = 1 + (int)XML_GetUserData(parser);
    char *text;
    XML_Parser p2;

    if (callno == 1)
        text = ("<!ELEMENT doc (e+)>\n"
                "<!ATTLIST doc xmlns CDATA #IMPLIED>\n"
                "<!ELEMENT e EMPTY>\n");
    else
        text = ("<?xml version='1.0' encoding='us-ascii'?>"
                "<e/>");

    XML_SetUserData(parser, (void *) callno);
    p2 = XML_ExternalEntityParserCreate(parser, context, NULL);
    if (XML_Parse(p2, text, strlen(text), XML_TRUE) == XML_STATUS_ERROR) {
        xml_failure(p2);
        return 0;
    }
    XML_ParserFree(p2);
    return 1;
}

START_TEST(test_default_ns_from_ext_subset_and_ext_ge)
{
    char *text =
        "<?xml version='1.0'?>\n"
        "<!DOCTYPE doc SYSTEM 'http://xml.libexpat.org/doc.dtd' [\n"
        "  <!ENTITY en SYSTEM 'http://xml.libexpat.org/entity.ent'>\n"
        "]>\n"
        "<doc xmlns='http://xml.libexpat.org/ns1'>\n"
        "&en;\n"
        "</doc>";

    XML_SetParamEntityParsing(parser, XML_PARAM_ENTITY_PARSING_ALWAYS);
    XML_SetExternalEntityRefHandler(parser, external_entity_handler);
    /* We actually need to set this handler to tickle this bug. */
    XML_SetStartElementHandler(parser, dummy_start_element);
    XML_SetUserData(parser, NULL);
    if (XML_Parse(parser, text, strlen(text), XML_TRUE) == XML_STATUS_ERROR)
        xml_failure(parser);
}
END_TEST

/* Regression test #1 for SF bug #673791. */
START_TEST(test_ns_prefix_with_empty_uri_1)
{
    char *text =
        "<doc xmlns:prefix='http://xml.libexpat.org/'>\n"
        "  <e xmlns:prefix=''/>\n"
        "</doc>";

    expect_failure(text,
                   XML_ERROR_UNDECLARING_PREFIX,
                   "Did not report re-setting namespace"
                   " URI with prefix to ''.");
}
END_TEST

/* Regression test #2 for SF bug #673791. */
START_TEST(test_ns_prefix_with_empty_uri_2)
{
    char *text =
        "<?xml version='1.0'?>\n"
        "<docelem xmlns:pre=''/>";

    expect_failure(text,
                   XML_ERROR_UNDECLARING_PREFIX,
                   "Did not report setting namespace URI with prefix to ''.");
}
END_TEST

/* Regression test #3 for SF bug #673791. */
START_TEST(test_ns_prefix_with_empty_uri_3)
{
    char *text =
        "<!DOCTYPE doc [\n"
        "  <!ELEMENT doc EMPTY>\n"
        "  <!ATTLIST doc\n"
        "    xmlns:prefix CDATA ''>\n"
        "]>\n"
        "<doc/>";

    expect_failure(text,
                   XML_ERROR_UNDECLARING_PREFIX,
                   "Didn't report attr default setting NS w/ prefix to ''.");
}
END_TEST

/* Regression test #4 for SF bug #673791. */
START_TEST(test_ns_prefix_with_empty_uri_4)
{
    char *text =
        "<!DOCTYPE doc [\n"
        "  <!ELEMENT prefix:doc EMPTY>\n"
        "  <!ATTLIST prefix:doc\n"
        "    xmlns:prefix CDATA 'http://xml.libexpat.org/'>\n"
        "]>\n"
        "<prefix:doc/>";
    /* Packaged info expected by the end element handler;
       the weird structuring lets us re-use the triplet_end_checker()
       function also used for another test. */
    char *elemstr[] = {
        "http://xml.libexpat.org/ doc prefix"
    };
    XML_SetReturnNSTriplet(parser, XML_TRUE);
    XML_SetUserData(parser, elemstr);
    XML_SetEndElementHandler(parser, triplet_end_checker);
    if (XML_Parse(parser, text, strlen(text), XML_TRUE) == XML_STATUS_ERROR)
        xml_failure(parser);
}
END_TEST

START_TEST(test_ns_default_with_empty_uri)
{
    char *text =
        "<doc xmlns='http://xml.libexpat.org/'>\n"
        "  <e xmlns=''/>\n"
        "</doc>";
    if (XML_Parse(parser, text, strlen(text), XML_TRUE) == XML_STATUS_ERROR)
        xml_failure(parser);
}
END_TEST

/* Regression test for SF bug #692964: two prefixes for one namespace. */
START_TEST(test_ns_duplicate_attrs_diff_prefixes)
{
    char *text =
        "<doc xmlns:a='http://xml.libexpat.org/a'\n"
        "     xmlns:b='http://xml.libexpat.org/a'\n"
        "     a:a='v' b:a='v' />";
    expect_failure(text,
                   XML_ERROR_DUPLICATE_ATTRIBUTE,
                   "did not report multiple attributes with same URI+name");
}
END_TEST

/* Regression test for SF bug #695401: unbound prefix. */
START_TEST(test_ns_unbound_prefix_on_attribute)
{
    char *text = "<doc a:attr=''/>";
    expect_failure(text,
                   XML_ERROR_UNBOUND_PREFIX,
                   "did not report unbound prefix on attribute");
}
END_TEST

/* Regression test for SF bug #695401: unbound prefix. */
START_TEST(test_ns_unbound_prefix_on_element)
{
    char *text = "<a:doc/>";
    expect_failure(text,
                   XML_ERROR_UNBOUND_PREFIX,
                   "did not report unbound prefix on element");
}
END_TEST

static Suite *
make_suite(void)
{
    Suite *s = suite_create("basic");
    TCase *tc_basic = tcase_create("basic tests");
    TCase *tc_namespace = tcase_create("XML namespaces");

    suite_add_tcase(s, tc_basic);
    tcase_add_checked_fixture(tc_basic, basic_setup, basic_teardown);
    tcase_add_test(tc_basic, test_nul_byte);
    tcase_add_test(tc_basic, test_u0000_char);
    tcase_add_test(tc_basic, test_bom_utf8);
    tcase_add_test(tc_basic, test_bom_utf16_be);
    tcase_add_test(tc_basic, test_bom_utf16_le);
    tcase_add_test(tc_basic, test_illegal_utf8);
    tcase_add_test(tc_basic, test_utf16);
    tcase_add_test(tc_basic, test_utf16_le_epilog_newline);
    tcase_add_test(tc_basic, test_latin1_umlauts);
    /* Regression test for SF bug #491986. */
    tcase_add_test(tc_basic, test_danish_latin1);
    /* Regression test for SF bug #514281. */
    tcase_add_test(tc_basic, test_french_charref_hexidecimal);
    tcase_add_test(tc_basic, test_french_charref_decimal);
    tcase_add_test(tc_basic, test_french_latin1);
    tcase_add_test(tc_basic, test_french_utf8);
    tcase_add_test(tc_basic, test_utf8_false_rejection);
    tcase_add_test(tc_basic, test_line_number_after_parse);
    tcase_add_test(tc_basic, test_column_number_after_parse);
    tcase_add_test(tc_basic, test_line_and_column_numbers_inside_handlers);
    tcase_add_test(tc_basic, test_line_number_after_error);
    tcase_add_test(tc_basic, test_column_number_after_error);
    tcase_add_test(tc_basic, test_really_long_lines);
    tcase_add_test(tc_basic, test_end_element_events);
    tcase_add_test(tc_basic, test_attr_whitespace_normalization);
    tcase_add_test(tc_basic, test_xmldecl_misplaced);
    tcase_add_test(tc_basic, test_unknown_encoding_internal_entity);
    tcase_add_test(tc_basic,
                   test_wfc_undeclared_entity_unread_external_subset);
    tcase_add_test(tc_basic, test_wfc_undeclared_entity_no_external_subset);
    tcase_add_test(tc_basic, test_wfc_undeclared_entity_standalone);
    tcase_add_test(tc_basic, test_wfc_undeclared_entity_with_external_subset);
    tcase_add_test(tc_basic,
                   test_wfc_undeclared_entity_with_external_subset_standalone);
    tcase_add_test(tc_basic, test_wfc_no_recursive_entity_refs);
    tcase_add_test(tc_basic, test_ext_entity_set_encoding);
    tcase_add_test(tc_basic, test_dtd_default_handling);
    tcase_add_test(tc_basic, test_empty_ns_without_namespaces);
    tcase_add_test(tc_basic, test_ns_in_attribute_default_without_namespaces);

    suite_add_tcase(s, tc_namespace);
    tcase_add_checked_fixture(tc_namespace,
                              namespace_setup, namespace_teardown);
    tcase_add_test(tc_namespace, test_return_ns_triplet);
    tcase_add_test(tc_namespace, test_ns_tagname_overwrite);
    tcase_add_test(tc_namespace, test_ns_tagname_overwrite_triplet);
    tcase_add_test(tc_namespace, test_start_ns_clears_start_element);
    tcase_add_test(tc_namespace, test_default_ns_from_ext_subset_and_ext_ge);
    tcase_add_test(tc_namespace, test_ns_prefix_with_empty_uri_1);
    tcase_add_test(tc_namespace, test_ns_prefix_with_empty_uri_2);
    tcase_add_test(tc_namespace, test_ns_prefix_with_empty_uri_3);
    tcase_add_test(tc_namespace, test_ns_prefix_with_empty_uri_4);
    tcase_add_test(tc_namespace, test_ns_default_with_empty_uri);
    tcase_add_test(tc_namespace, test_ns_duplicate_attrs_diff_prefixes);
    tcase_add_test(tc_namespace, test_ns_unbound_prefix_on_attribute);
    tcase_add_test(tc_namespace, test_ns_unbound_prefix_on_element);

    return s;
}


#ifdef AMIGA_SHARED_LIB
int
amiga_main(int argc, char *argv[])
#else
int
main(int argc, char *argv[])
#endif
{
    int i, nf;
    int forking = 0, forking_set = 0;
    int verbosity = CK_NORMAL;
    Suite *s = make_suite();
    SRunner *sr = srunner_create(s);

    /* run the tests for internal helper functions */
    testhelper_is_whitespace_normalized();

    for (i = 1; i < argc; ++i) {
        char *opt = argv[i];
        if (strcmp(opt, "-v") == 0 || strcmp(opt, "--verbose") == 0)
            verbosity = CK_VERBOSE;
        else if (strcmp(opt, "-q") == 0 || strcmp(opt, "--quiet") == 0)
            verbosity = CK_SILENT;
        else if (strcmp(opt, "-f") == 0 || strcmp(opt, "--fork") == 0) {
            forking = 1;
            forking_set = 1;
        }
        else if (strcmp(opt, "-n") == 0 || strcmp(opt, "--no-fork") == 0) {
            forking = 0;
            forking_set = 1;
        }
        else {
            fprintf(stderr, "runtests: unknown option '%s'\n", opt);
            return 2;
        }
    }
    if (forking_set)
        srunner_set_fork_status(sr, forking ? CK_FORK : CK_NOFORK);
    if (verbosity != CK_SILENT)
        printf("Expat version: %s\n", XML_ExpatVersion());
    srunner_run_all(sr, verbosity);
    nf = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (nf == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
