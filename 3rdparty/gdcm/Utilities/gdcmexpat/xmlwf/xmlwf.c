/* Copyright (c) 1998, 1999 Thai Open Source Software Center Ltd
   See the file COPYING for copying permission.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#include "expat.h"
#include "codepage.h"
#include "xmlfile.h"
#include "xmltchar.h"

#ifdef _MSC_VER
#include <crtdbg.h>
#endif

#ifdef AMIGA_SHARED_LIB
#include <proto/expat.h>
#endif

/* This ensures proper sorting. */

#define NSSEP T('\001')

static void XMLCALL
characterData(void *userData, const XML_Char *s, int len)
{
  FILE *fp = (FILE *)userData;
  for (; len > 0; --len, ++s) {
    switch (*s) {
    case T('&'):
      fputts(T("&amp;"), fp);
      break;
    case T('<'):
      fputts(T("&lt;"), fp);
      break;
    case T('>'):
      fputts(T("&gt;"), fp);
      break;
#ifdef W3C14N
    case 13:
      fputts(T("&#xD;"), fp);
      break;
#else
    case T('"'):
      fputts(T("&quot;"), fp);
      break;
    case 9:
    case 10:
    case 13:
      ftprintf(fp, T("&#%d;"), *s);
      break;
#endif
    default:
      puttc(*s, fp);
      break;
    }
  }
}

static void
attributeValue(FILE *fp, const XML_Char *s)
{
  puttc(T('='), fp);
  puttc(T('"'), fp);
  for (;;) {
    switch (*s) {
    case 0:
    case NSSEP:
      puttc(T('"'), fp);
      return;
    case T('&'):
      fputts(T("&amp;"), fp);
      break;
    case T('<'):
      fputts(T("&lt;"), fp);
      break;
    case T('"'):
      fputts(T("&quot;"), fp);
      break;
#ifdef W3C14N
    case 9:
      fputts(T("&#x9;"), fp);
      break;
    case 10:
      fputts(T("&#xA;"), fp);
      break;
    case 13:
      fputts(T("&#xD;"), fp);
      break;
#else
    case T('>'):
      fputts(T("&gt;"), fp);
      break;
    case 9:
    case 10:
    case 13:
      ftprintf(fp, T("&#%d;"), *s);
      break;
#endif
    default:
      puttc(*s, fp);
      break;
    }
    s++;
  }
}

/* Lexicographically comparing UTF-8 encoded attribute values,
is equivalent to lexicographically comparing based on the character number. */

static int
attcmp(const void *att1, const void *att2)
{
  return tcscmp(*(const XML_Char **)att1, *(const XML_Char **)att2);
}

static void XMLCALL
startElement(void *userData, const XML_Char *name, const XML_Char **atts)
{
  int nAtts;
  const XML_Char **p;
  FILE *fp = (FILE *)userData;
  puttc(T('<'), fp);
  fputts(name, fp);

  p = atts;
  while (*p)
    ++p;
  nAtts = (p - atts) >> 1;
  if (nAtts > 1)
    qsort((void *)atts, nAtts, sizeof(XML_Char *) * 2, attcmp);
  while (*atts) {
    puttc(T(' '), fp);
    fputts(*atts++, fp);
    attributeValue(fp, *atts);
    atts++;
  }
  puttc(T('>'), fp);
}

static void XMLCALL
endElement(void *userData, const XML_Char *name)
{
  FILE *fp = (FILE *)userData;
  puttc(T('<'), fp);
  puttc(T('/'), fp);
  fputts(name, fp);
  puttc(T('>'), fp);
}

static int
nsattcmp(const void *p1, const void *p2)
{
  const XML_Char *att1 = *(const XML_Char **)p1;
  const XML_Char *att2 = *(const XML_Char **)p2;
  int sep1 = (tcsrchr(att1, NSSEP) != 0);
  int sep2 = (tcsrchr(att1, NSSEP) != 0);
  if (sep1 != sep2)
    return sep1 - sep2;
  return tcscmp(att1, att2);
}

static void XMLCALL
startElementNS(void *userData, const XML_Char *name, const XML_Char **atts)
{
  int nAtts;
  int nsi;
  const XML_Char **p;
  FILE *fp = (FILE *)userData;
  const XML_Char *sep;
  puttc(T('<'), fp);

  sep = tcsrchr(name, NSSEP);
  if (sep) {
    fputts(T("n1:"), fp);
    fputts(sep + 1, fp);
    fputts(T(" xmlns:n1"), fp);
    attributeValue(fp, name);
    nsi = 2;
  }
  else {
    fputts(name, fp);
    nsi = 1;
  }

  p = atts;
  while (*p)
    ++p;
  nAtts = (p - atts) >> 1;
  if (nAtts > 1)
    qsort((void *)atts, nAtts, sizeof(XML_Char *) * 2, nsattcmp);
  while (*atts) {
    name = *atts++;
    sep = tcsrchr(name, NSSEP);
    puttc(T(' '), fp);
    if (sep) {
      ftprintf(fp, T("n%d:"), nsi);
      fputts(sep + 1, fp);
    }
    else
      fputts(name, fp);
    attributeValue(fp, *atts);
    if (sep) {
      ftprintf(fp, T(" xmlns:n%d"), nsi++);
      attributeValue(fp, name);
    }
    atts++;
  }
  puttc(T('>'), fp);
}

static void XMLCALL
endElementNS(void *userData, const XML_Char *name)
{
  FILE *fp = (FILE *)userData;
  const XML_Char *sep;
  puttc(T('<'), fp);
  puttc(T('/'), fp);
  sep = tcsrchr(name, NSSEP);
  if (sep) {
    fputts(T("n1:"), fp);
    fputts(sep + 1, fp);
  }
  else
    fputts(name, fp);
  puttc(T('>'), fp);
}

#ifndef W3C14N

static void XMLCALL
processingInstruction(void *userData, const XML_Char *target,
                      const XML_Char *data)
{
  FILE *fp = (FILE *)userData;
  puttc(T('<'), fp);
  puttc(T('?'), fp);
  fputts(target, fp);
  puttc(T(' '), fp);
  fputts(data, fp);
  puttc(T('?'), fp);
  puttc(T('>'), fp);
}

#endif /* not W3C14N */

static void XMLCALL
defaultCharacterData(void *userData, const XML_Char *s, int len)
{
  XML_DefaultCurrent((XML_Parser) userData);
}

static void XMLCALL
defaultStartElement(void *userData, const XML_Char *name,
                    const XML_Char **atts)
{
  XML_DefaultCurrent((XML_Parser) userData);
}

static void XMLCALL
defaultEndElement(void *userData, const XML_Char *name)
{
  XML_DefaultCurrent((XML_Parser) userData);
}

static void XMLCALL
defaultProcessingInstruction(void *userData, const XML_Char *target,
                             const XML_Char *data)
{
  XML_DefaultCurrent((XML_Parser) userData);
}

static void XMLCALL
nopCharacterData(void *userData, const XML_Char *s, int len)
{
}

static void XMLCALL
nopStartElement(void *userData, const XML_Char *name, const XML_Char **atts)
{
}

static void XMLCALL
nopEndElement(void *userData, const XML_Char *name)
{
}

static void XMLCALL
nopProcessingInstruction(void *userData, const XML_Char *target,
                         const XML_Char *data)
{
}

static void XMLCALL
markup(void *userData, const XML_Char *s, int len)
{
  FILE *fp = (FILE *)XML_GetUserData((XML_Parser) userData);
  for (; len > 0; --len, ++s)
    puttc(*s, fp);
}

static void
metaLocation(XML_Parser parser)
{
  const XML_Char *uri = XML_GetBase(parser);
  if (uri)
    ftprintf((FILE *)XML_GetUserData(parser), T(" uri=\"%s\""), uri);
  ftprintf((FILE *)XML_GetUserData(parser),
           T(" byte=\"%" XML_FMT_INT_MOD "d\" nbytes=\"%d\" \
			 line=\"%" XML_FMT_INT_MOD "u\" col=\"%" XML_FMT_INT_MOD "u\""),
           XML_GetCurrentByteIndex(parser),
           XML_GetCurrentByteCount(parser),
           XML_GetCurrentLineNumber(parser),
           XML_GetCurrentColumnNumber(parser));
}

static void
metaStartDocument(void *userData)
{
  fputts(T("<document>\n"), (FILE *)XML_GetUserData((XML_Parser) userData));
}

static void
metaEndDocument(void *userData)
{
  fputts(T("</document>\n"), (FILE *)XML_GetUserData((XML_Parser) userData));
}

static void XMLCALL
metaStartElement(void *userData, const XML_Char *name,
                 const XML_Char **atts)
{
  XML_Parser parser = (XML_Parser) userData;
  FILE *fp = (FILE *)XML_GetUserData(parser);
  const XML_Char **specifiedAttsEnd
    = atts + XML_GetSpecifiedAttributeCount(parser);
  const XML_Char **idAttPtr;
  int idAttIndex = XML_GetIdAttributeIndex(parser);
  if (idAttIndex < 0)
    idAttPtr = 0;
  else
    idAttPtr = atts + idAttIndex;
    
  ftprintf(fp, T("<starttag name=\"%s\""), name);
  metaLocation(parser);
  if (*atts) {
    fputts(T(">\n"), fp);
    do {
      ftprintf(fp, T("<attribute name=\"%s\" value=\""), atts[0]);
      characterData(fp, atts[1], tcslen(atts[1]));
      if (atts >= specifiedAttsEnd)
        fputts(T("\" defaulted=\"yes\"/>\n"), fp);
      else if (atts == idAttPtr)
        fputts(T("\" id=\"yes\"/>\n"), fp);
      else
        fputts(T("\"/>\n"), fp);
    } while (*(atts += 2));
    fputts(T("</starttag>\n"), fp);
  }
  else
    fputts(T("/>\n"), fp);
}

static void XMLCALL
metaEndElement(void *userData, const XML_Char *name)
{
  XML_Parser parser = (XML_Parser) userData;
  FILE *fp = (FILE *)XML_GetUserData(parser);
  ftprintf(fp, T("<endtag name=\"%s\""), name);
  metaLocation(parser);
  fputts(T("/>\n"), fp);
}

static void XMLCALL
metaProcessingInstruction(void *userData, const XML_Char *target,
                          const XML_Char *data)
{
  XML_Parser parser = (XML_Parser) userData;
  FILE *fp = (FILE *)XML_GetUserData(parser);
  ftprintf(fp, T("<pi target=\"%s\" data=\""), target);
  characterData(fp, data, tcslen(data));
  puttc(T('"'), fp);
  metaLocation(parser);
  fputts(T("/>\n"), fp);
}

static void XMLCALL
metaComment(void *userData, const XML_Char *data)
{
  XML_Parser parser = (XML_Parser) userData;
  FILE *fp = (FILE *)XML_GetUserData(parser);
  fputts(T("<comment data=\""), fp);
  characterData(fp, data, tcslen(data));
  puttc(T('"'), fp);
  metaLocation(parser);
  fputts(T("/>\n"), fp);
}

static void XMLCALL
metaStartCdataSection(void *userData)
{
  XML_Parser parser = (XML_Parser) userData;
  FILE *fp = (FILE *)XML_GetUserData(parser);
  fputts(T("<startcdata"), fp);
  metaLocation(parser);
  fputts(T("/>\n"), fp);
}

static void XMLCALL
metaEndCdataSection(void *userData)
{
  XML_Parser parser = (XML_Parser) userData;
  FILE *fp = (FILE *)XML_GetUserData(parser);
  fputts(T("<endcdata"), fp);
  metaLocation(parser);
  fputts(T("/>\n"), fp);
}

static void XMLCALL
metaCharacterData(void *userData, const XML_Char *s, int len)
{
  XML_Parser parser = (XML_Parser) userData;
  FILE *fp = (FILE *)XML_GetUserData(parser);
  fputts(T("<chars str=\""), fp);
  characterData(fp, s, len);
  puttc(T('"'), fp);
  metaLocation(parser);
  fputts(T("/>\n"), fp);
}

static void XMLCALL
metaStartDoctypeDecl(void *userData,
                     const XML_Char *doctypeName,
                     const XML_Char *sysid,
                     const XML_Char *pubid,
                     int has_internal_subset)
{
  XML_Parser parser = (XML_Parser) userData;
  FILE *fp = (FILE *)XML_GetUserData(parser);
  ftprintf(fp, T("<startdoctype name=\"%s\""), doctypeName);
  metaLocation(parser);
  fputts(T("/>\n"), fp);
}

static void XMLCALL
metaEndDoctypeDecl(void *userData)
{
  XML_Parser parser = (XML_Parser) userData;
  FILE *fp = (FILE *)XML_GetUserData(parser);
  fputts(T("<enddoctype"), fp);
  metaLocation(parser);
  fputts(T("/>\n"), fp);
}

static void XMLCALL
metaNotationDecl(void *userData,
                 const XML_Char *notationName,
                 const XML_Char *base,
                 const XML_Char *systemId,
                 const XML_Char *publicId)
{
  XML_Parser parser = (XML_Parser) userData;
  FILE *fp = (FILE *)XML_GetUserData(parser);
  ftprintf(fp, T("<notation name=\"%s\""), notationName);
  if (publicId)
    ftprintf(fp, T(" public=\"%s\""), publicId);
  if (systemId) {
    fputts(T(" system=\""), fp);
    characterData(fp, systemId, tcslen(systemId));
    puttc(T('"'), fp);
  }
  metaLocation(parser);
  fputts(T("/>\n"), fp);
}


static void XMLCALL
metaEntityDecl(void *userData,
               const XML_Char *entityName,
               int  is_param,
               const XML_Char *value,
               int  value_length,
               const XML_Char *base,
               const XML_Char *systemId,
               const XML_Char *publicId,
               const XML_Char *notationName)
{
  XML_Parser parser = (XML_Parser) userData;
  FILE *fp = (FILE *)XML_GetUserData(parser);

  if (value) {
    ftprintf(fp, T("<entity name=\"%s\""), entityName);
    metaLocation(parser);
    puttc(T('>'), fp);
    characterData(fp, value, value_length);
    fputts(T("</entity/>\n"), fp);
  }
  else if (notationName) {
    ftprintf(fp, T("<entity name=\"%s\""), entityName);
    if (publicId)
      ftprintf(fp, T(" public=\"%s\""), publicId);
    fputts(T(" system=\""), fp);
    characterData(fp, systemId, tcslen(systemId));
    puttc(T('"'), fp);
    ftprintf(fp, T(" notation=\"%s\""), notationName);
    metaLocation(parser);
    fputts(T("/>\n"), fp);
  }
  else {
    ftprintf(fp, T("<entity name=\"%s\""), entityName);
    if (publicId)
      ftprintf(fp, T(" public=\"%s\""), publicId);
    fputts(T(" system=\""), fp);
    characterData(fp, systemId, tcslen(systemId));
    puttc(T('"'), fp);
    metaLocation(parser);
    fputts(T("/>\n"), fp);
  }
}

static void XMLCALL
metaStartNamespaceDecl(void *userData,
                       const XML_Char *prefix,
                       const XML_Char *uri)
{
  XML_Parser parser = (XML_Parser) userData;
  FILE *fp = (FILE *)XML_GetUserData(parser);
  fputts(T("<startns"), fp);
  if (prefix)
    ftprintf(fp, T(" prefix=\"%s\""), prefix);
  if (uri) {
    fputts(T(" ns=\""), fp);
    characterData(fp, uri, tcslen(uri));
    fputts(T("\"/>\n"), fp);
  }
  else
    fputts(T("/>\n"), fp);
}

static void XMLCALL
metaEndNamespaceDecl(void *userData, const XML_Char *prefix)
{
  XML_Parser parser = (XML_Parser) userData;
  FILE *fp = (FILE *)XML_GetUserData(parser);
  if (!prefix)
    fputts(T("<endns/>\n"), fp);
  else
    ftprintf(fp, T("<endns prefix=\"%s\"/>\n"), prefix);
}

static int XMLCALL
unknownEncodingConvert(void *data, const char *p)
{
  return codepageConvert(*(int *)data, p);
}

static int XMLCALL
unknownEncoding(void *userData, const XML_Char *name, XML_Encoding *info)
{
  int cp;
  static const XML_Char prefixL[] = T("windows-");
  static const XML_Char prefixU[] = T("WINDOWS-");
  int i;

  for (i = 0; prefixU[i]; i++)
    if (name[i] != prefixU[i] && name[i] != prefixL[i])
      return 0;
  
  cp = 0;
  for (; name[i]; i++) {
    static const XML_Char digits[] = T("0123456789");
    const XML_Char *s = tcschr(digits, name[i]);
    if (!s)
      return 0;
    cp *= 10;
    cp += s - digits;
    if (cp >= 0x10000)
      return 0;
  }
  if (!codepageMap(cp, info->map))
    return 0;
  info->convert = unknownEncodingConvert;
  /* We could just cast the code page integer to a void *,
  and avoid the use of release. */
  info->release = free;
  info->data = malloc(sizeof(int));
  if (!info->data)
    return 0;
  *(int *)info->data = cp;
  return 1;
}

static int XMLCALL
notStandalone(void *userData)
{
  return 0;
}

static void
showVersion(XML_Char *prog)
{
  XML_Char *s = prog;
  XML_Char ch;
  const XML_Feature *features = XML_GetFeatureList();
  while ((ch = *s) != 0) {
    if (ch == '/'
#ifdef WIN32
        || ch == '\\'
#endif
        )
      prog = s + 1;
    ++s;
  }
  ftprintf(stdout, T("%s using %s\n"), prog, XML_ExpatVersion());
  if (features != NULL && features[0].feature != XML_FEATURE_END) {
    int i = 1;
    ftprintf(stdout, T("%s"), features[0].name);
    if (features[0].value)
      ftprintf(stdout, T("=%ld"), features[0].value);
    while (features[i].feature != XML_FEATURE_END) {
      ftprintf(stdout, T(", %s"), features[i].name);
      if (features[i].value)
        ftprintf(stdout, T("=%ld"), features[i].value);
      ++i;
    }
    ftprintf(stdout, T("\n"));
  }
}

static void
usage(const XML_Char *prog, int rc)
{
  ftprintf(stderr,
           T("usage: %s [-n] [-p] [-r] [-s] [-w] [-x] [-d output-dir] "
             "[-e encoding] file ...\n"), prog);
  exit(rc);
}

#ifdef AMIGA_SHARED_LIB
int
amiga_main(int argc, char *argv[])
#else
int
tmain(int argc, XML_Char **argv)
#endif
{
  int i, j;
  const XML_Char *outputDir = NULL;
  const XML_Char *encoding = NULL;
  unsigned processFlags = XML_MAP_FILE;
  int windowsCodePages = 0;
  int outputType = 0;
  int useNamespaces = 0;
  int requireStandalone = 0;
  enum XML_ParamEntityParsing paramEntityParsing = 
    XML_PARAM_ENTITY_PARSING_NEVER;
  int useStdin = 0;

#ifdef _MSC_VER
  _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF|_CRTDBG_LEAK_CHECK_DF);
#endif

  i = 1;
  j = 0;
  while (i < argc) {
    if (j == 0) {
      if (argv[i][0] != T('-'))
        break;
      if (argv[i][1] == T('-') && argv[i][2] == T('\0')) {
        i++;
        break;
      }
      j++;
    }
    switch (argv[i][j]) {
    case T('r'):
      processFlags &= ~XML_MAP_FILE;
      j++;
      break;
    case T('s'):
      requireStandalone = 1;
      j++;
      break;
    case T('n'):
      useNamespaces = 1;
      j++;
      break;
    case T('p'):
      paramEntityParsing = XML_PARAM_ENTITY_PARSING_ALWAYS;
      /* fall through */
    case T('x'):
      processFlags |= XML_EXTERNAL_ENTITIES;
      j++;
      break;
    case T('w'):
      windowsCodePages = 1;
      j++;
      break;
    case T('m'):
      outputType = 'm';
      j++;
      break;
    case T('c'):
      outputType = 'c';
      useNamespaces = 0;
      j++;
      break;
    case T('t'):
      outputType = 't';
      j++;
      break;
    case T('d'):
      if (argv[i][j + 1] == T('\0')) {
        if (++i == argc)
          usage(argv[0], 2);
        outputDir = argv[i];
      }
      else
        outputDir = argv[i] + j + 1;
      i++;
      j = 0;
      break;
    case T('e'):
      if (argv[i][j + 1] == T('\0')) {
        if (++i == argc)
          usage(argv[0], 2);
        encoding = argv[i];
      }
      else
        encoding = argv[i] + j + 1;
      i++;
      j = 0;
      break;
    case T('h'):
      usage(argv[0], 0);
      return 0;
    case T('v'):
      showVersion(argv[0]);
      return 0;
    case T('\0'):
      if (j > 1) {
        i++;
        j = 0;
        break;
      }
      /* fall through */
    default:
      usage(argv[0], 2);
    }
  }
  if (i == argc) {
    useStdin = 1;
    processFlags &= ~XML_MAP_FILE;
    i--;
  }
  for (; i < argc; i++) {
    FILE *fp = 0;
    XML_Char *outName = 0;
    int result;
    XML_Parser parser;
    if (useNamespaces)
      parser = XML_ParserCreateNS(encoding, NSSEP);
    else
      parser = XML_ParserCreate(encoding);
    if (requireStandalone)
      XML_SetNotStandaloneHandler(parser, notStandalone);
    XML_SetParamEntityParsing(parser, paramEntityParsing);
    if (outputType == 't') {
      /* This is for doing timings; this gives a more realistic estimate of
         the parsing time. */
      outputDir = 0;
      XML_SetElementHandler(parser, nopStartElement, nopEndElement);
      XML_SetCharacterDataHandler(parser, nopCharacterData);
      XML_SetProcessingInstructionHandler(parser, nopProcessingInstruction);
    }
    else if (outputDir) {
      const XML_Char *file = useStdin ? T("STDIN") : argv[i];
      if (tcsrchr(file, T('/')))
        file = tcsrchr(file, T('/')) + 1;
#ifdef WIN32
      if (tcsrchr(file, T('\\')))
        file = tcsrchr(file, T('\\')) + 1;
#endif
      outName = (XML_Char *)malloc((tcslen(outputDir) + tcslen(file) + 2)
                       * sizeof(XML_Char));
      tcscpy(outName, outputDir);
      tcscat(outName, T("/"));
      tcscat(outName, file);
      fp = tfopen(outName, T("wb"));
      if (!fp) {
        tperror(outName);
        exit(1);
      }
      setvbuf(fp, NULL, _IOFBF, 16384);
#ifdef XML_UNICODE
      puttc(0xFEFF, fp);
#endif
      XML_SetUserData(parser, fp);
      switch (outputType) {
      case 'm':
        XML_UseParserAsHandlerArg(parser);
        XML_SetElementHandler(parser, metaStartElement, metaEndElement);
        XML_SetProcessingInstructionHandler(parser, metaProcessingInstruction);
        XML_SetCommentHandler(parser, metaComment);
        XML_SetCdataSectionHandler(parser, metaStartCdataSection,
                                   metaEndCdataSection);
        XML_SetCharacterDataHandler(parser, metaCharacterData);
        XML_SetDoctypeDeclHandler(parser, metaStartDoctypeDecl,
                                  metaEndDoctypeDecl);
        XML_SetEntityDeclHandler(parser, metaEntityDecl);
        XML_SetNotationDeclHandler(parser, metaNotationDecl);
        XML_SetNamespaceDeclHandler(parser, metaStartNamespaceDecl,
                                    metaEndNamespaceDecl);
        metaStartDocument(parser);
        break;
      case 'c':
        XML_UseParserAsHandlerArg(parser);
        XML_SetDefaultHandler(parser, markup);
        XML_SetElementHandler(parser, defaultStartElement, defaultEndElement);
        XML_SetCharacterDataHandler(parser, defaultCharacterData);
        XML_SetProcessingInstructionHandler(parser,
                                            defaultProcessingInstruction);
        break;
      default:
        if (useNamespaces)
          XML_SetElementHandler(parser, startElementNS, endElementNS);
        else
          XML_SetElementHandler(parser, startElement, endElement);
        XML_SetCharacterDataHandler(parser, characterData);
#ifndef W3C14N
        XML_SetProcessingInstructionHandler(parser, processingInstruction);
#endif /* not W3C14N */
        break;
      }
    }
    if (windowsCodePages)
      XML_SetUnknownEncodingHandler(parser, unknownEncoding, 0);
    result = XML_ProcessFile(parser, useStdin ? NULL : argv[i], processFlags);
    if (outputDir) {
      if (outputType == 'm')
        metaEndDocument(parser);
      fclose(fp);
      if (!result)
        tremove(outName);
      free(outName);
    }
    XML_ParserFree(parser);
  }
  return 0;
}
