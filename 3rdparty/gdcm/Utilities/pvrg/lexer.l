/*************************************************************
Copyright (C) 1990, 1991, 1993 Andy C. Hung, all rights reserved.
PUBLIC DOMAIN LICENSE: Stanford University Portable Video Research
Group. If you use this software, you agree to the following: This
program package is purely experimental, and is licensed "as is".
Permission is granted to use, modify, and distribute this program
without charge for any purpose, provided this license/ disclaimer
notice appears in the copies.  No warranty or maintenance is given,
either expressed or implied.  In no event shall the author(s) be
liable to you or a third party for any special, incidental,
consequential, or other damages, arising out of the use or inability
to use the program for any purpose (or the loss of data), even if we
have been advised of such possibilities.  Any public reference or
advertisement of this source code should refer to it as the Portable
Video Research Group (PVRG) code, and not by any author(s) (or
Stanford University) name.
*************************************************************/
%option yylineno
%{

/*LABEL lexer.c */

/*
 * flex command was run this way:
 *
 *  $ lex -olexer.c lexer.l
 */

/* We do not care of interactive mode */
#define YY_NEVER_INTERACTIVE 1
#define YY_NO_UNPUT 1

/* Do not include unistd.h in generated source. */
#define YY_NO_UNISTD_H

/* Skip declaring this function.  It is a macro.  */
#define YY_SKIP_YYWRAP

/* Redefine the yywrap so that we don't have
   to worry about lex library */

# define yywrap() (1)

static char *ReservedWords[] = {   /* Token names */
"COMPONENT",
"SCAN",
"QUANTIZATION",
"DCSPEC",
"ACCUSTOM",
"DCCUSTOM",
"PRINTSCAN",
"PRINTFRAME",
"PRINTIMAGE",
"OPENSCAN",
"ACSPEC",
"WRITESCAN",
"WRITEFRAME",
"WRITESOI",
"WRITEQUANTIZATION",
"WRITERESYNC",
"WRITEHUFFMAN",
"FREQUENCY",
"ACSEND",
"DCSEND",
"QSEND",
"STREAMNAME",
"IMAGEHEIGHT",
"IMAGEWIDTH",
"RESYNC",
"BUFFER",
"OPENSTREAM",
"CLOSESTREAM",
"FRAMEHEIGHT",
"FRAMEWIDTH",
"CLOSESCAN",
"WRITEEOI",
"ECHO",
"WRITESPECIAL",
"WRITEDIRECT",
"LUMINANCEDEFAULT",
"CHROMINANCEDEFAULT",
"ENABLE",
"SCANDNL",
"WRITEDNL",
"AUTO",
"EMPTY",
""};

#define R_COMPONENT 1           /* Token values mapped to token names */
#define R_SCAN 2
#define R_QUANTIZATION 3
#define R_DCSPEC 4
#define R_ACCUSTOM 5
#define R_DCCUSTOM 6
#define R_PRINTSCAN 7
#define R_PRINTFRAME 8
#define R_PRINTIMAGE 9
#define R_OPENSCAN 10
#define R_ACSPEC 11
#define R_WRITESCAN 12
#define R_WRITEFRAME 13
#define R_WRITESOI 14
#define R_WRITEQUANTIZATION 15
#define R_WRITERESYNC 16
#define R_WRITEHUFFMAN 17
#define R_FREQUENCY 18
#define R_ACSEND 19
#define R_DCSEND 20
#define R_QSEND 21
#define R_STREAMNAME 22
#define R_IMAGEHEIGHT 23
#define R_IMAGEWIDTH 24
#define R_RESYNC 25
#define R_BUFFER 26
#define R_OPENSTREAM 27
#define R_CLOSESTREAM 28
#define R_FRAMEHEIGHT 29
#define R_FRAMEWIDTH 30
#define R_CLOSESCAN 31
#define R_WRITEEOI 32
#define R_ECHO 33
#define R_WRITESPECIAL 34
#define R_WRITEDIRECT 35
#define R_LUMINANCEDEFAULT 36
#define R_CHROMINANCEDEFAULT 37
#define R_ENABLE 38
#define R_SCANDNL 39
#define R_WRITEDNL 40
#define R_AUTO 41
#define R_EMPTY 42

#define R_INTEGER 1000      /* Special TYPES for tokens */
#define R_LBRACKET 1001
#define R_RBRACKET 1002
#define R_ID 1003
#define R_STRING 1004

int CommentDepth = 0;  /* depth of comment nesting */
int yyint=0;           /* Return value for integers */
int LexDebug=0;        /* Status of lex debugging */

#define PRIME 211
#define EOS '\0'

#define MakeStructure(S) (S *) malloc(sizeof(S))
#define InsertLink(link,list){\
if(!list){list=link;}else{link->next=list;list=link;}}

#define LINK struct link_def
struct id {         /* Default id structure */
  char *name;       /* Key */
  int tokentype;    /* Token type */
  int count;        /* Count of # references */
};

LINK {              /* A link for the hash buckets */
struct id *lid;     /* Current id */
LINK *next;         /* Pointer to next id */
};

/*PUBLIC*/

extern void initparser();
extern void parser();

static int hashpjw();
static LINK * MakeLink();
static struct id * enter();
static int getint();
static char * getstr();

/*PRIVATE*/

/*NOPROTO*/

%}

Delim    [ \t\n]
WhiteSpace  {Delim}+
Letter      [a-zA-Z]
Digit     [0-9]
HexDigit    ({Digit}|[a-fA-F])
OctalDigit  [0-7]
Id    {Letter}({Letter}|{Digit})*
DecInteger   {Digit}+
HexInteger   0[xX]{HexDigit}+
OctInteger   0[oO]{OctalDigit}+
HexInteger2  {HexDigit}+[Hh]
OctInteger2  {OctalDigit}+[BCObco]
CharInteger      '([^\\]|\\([\n^\n]|{OctalDigit}{1,3}))'
ScaleFactor  E[-+]?{Digit}+
Real1    ({Digit}+"."{Digit}*({ScaleFactor})?)
Real2    ({Digit}*"."{Digit}+({ScaleFactor})?)
Real3    ({Digit}+{ScaleFactor})
Real    {Real1}|{Real2}|{Real3}
Operator   (\+|=|\-|#|\*|\<|\>|\/|:=|\<\>|\&|\<=|\.|\>=|\,|\.\.|;|:|\(|\)|\[|\]|\{|\}|\^|\||~)
String    \"([^\"]|\\\")*\"

%S NORMAL COMMENT
%%

<NORMAL>{WhiteSpace}  {}

<NORMAL>{Id}  {struct id *temp; temp = enter(0,yytext,yyleng);
     if (LexDebug)
       {
         printf("%s : %s (%d)\n",
          yytext,
          ((temp->tokentype) ? "RESERVED" : "IDENTIFIER"),
          temp->count);
       }
     if (temp->tokentype)
       {
         return(temp->tokentype);
       }
     else
       {
         return(R_ID);
       }
         }
<NORMAL>{Real}          {if (LexDebug)
         {
           printf("%s : %s\n", yytext, "REAL");
         }
           }
<NORMAL>{DecInteger}  {if (LexDebug)
         {
           printf("%s : %s\n", yytext, "INTEGER");
         }
       yyint = atoi(yytext);
       return(R_INTEGER);}
<NORMAL>{HexInteger}  {if (LexDebug)
         {
           printf("%s : %s\n", yytext, "(HEX)INTEGER");
         }
       yyint = strtol(yytext+2,NULL,16);
       return(R_INTEGER);}
<NORMAL>{HexInteger2}  {if (LexDebug)
         {
           printf("%s : %s\n", yytext, "(HEX)INTEGER");
         }
       yyint = strtol(yytext,NULL,16);
       return(R_INTEGER);}
<NORMAL>{OctInteger}  {if (LexDebug)
         {
           printf("%s : %s\n", yytext, "(OCT)INTEGER");
         }
       yyint = strtol(yytext+2,NULL,8);
       return(R_INTEGER);}
<NORMAL>{OctInteger2}  {if (LexDebug)
         {
           printf("%s : %s\n", yytext, "(OCT)INTEGER");
         }
       yyint = strtol(yytext,NULL,8);
       return(R_INTEGER);}
<NORMAL>{CharInteger}  {if (LexDebug)
         {
           printf("%s : %s\n", yytext, "(CHAR)INTEGER");
         }
       if (yyleng>4)
         {
           yyint = strtol(yytext+2,NULL,8);
         }
       else
         {
           if (*(yytext+1)=='\\')
             {
         switch(*(yytext+2))
           {
           case '0':
             yyint=0;
             break;
           case 'b':
             yyint = 0x8;
             break;
           case 'i':
             yyint = 0x9;
             break;
           case 'n':
             yyint = 0xa;
             break;
           case 'v':
             yyint = 0xb;
             break;
           case 'f':
             yyint = 0xc;
             break;
           case 'r':
             yyint = 0xd;
             break;
           default:
             yyint=(*yytext+2);
             break;
           }
             }
           else
             {
         yyint = *(yytext+1);
             }
         }
       return(R_INTEGER);}
<NORMAL>\[          {if (LexDebug)
         {
           printf("%s : %s\n", yytext, "LBRACKET");
         }
       return(R_LBRACKET);}
<NORMAL>\]          {if (LexDebug)
         {
           printf("%s : %s\n", yytext, "RBRACKET");
         }
       return(R_RBRACKET);}
<NORMAL>{Operator}  {if (LexDebug)
         {
           printf("%s : %s\n", yytext, "OPERATOR");
         }
           }
<NORMAL>{String}  {if (LexDebug)
         {
           printf("%s : %s\n", yytext, "STRING");
         }
       return(R_STRING);}

<NORMAL,COMMENT>"/*"  {CommentDepth++; BEGIN COMMENT;}

<COMMENT>"*/"    {CommentDepth--;if(!CommentDepth) BEGIN NORMAL;}

<NORMAL>.          {
            /* None of the above rules applicable, so
             it's a bad symbol. */
                              printf("Bad input char '%c' on line %d\n",
                    yytext[0],
                    yylineno);
            }

<COMMENT>.|\n    {} /*Everything's AOK */

%%

/*PROTO*/
LINK *HashTable[PRIME];  /* My little hash table */

/*START*/

/*BFUNC

initparser() is used to place the Reserved Words into the hash table.
It must be called before the parser command is called.

EFUNC*/

void initparser()
{
  char i,**sptr;
  BEGIN NORMAL;

  for(i=1,sptr=ReservedWords;**sptr!='\0';i++,sptr++)
    {     /* Add Reserved Words */
      enter(i,*sptr,strlen(*sptr));        /* Put reserved words in */
    }                                      /* hash table */
}

#undef BEGIN
#undef MakeStructure
#include "globals.h"
#include "stream.h"
#include "tables.h"

extern FRAME *CFrame;
extern IMAGE *CImage;
extern SCAN *CScan;
extern int ErrorValue;

/*BFUNC

hashpjw() returns a hash value for a string input.

EFUNC*/

static int hashpjw(s)
     char *s;
{
  BEGIN("hashpjw")
  char *p;
  unsigned int g, h;
  h=0;

  for(p=s;*p!=EOS;p++)       /* Taken from Aho Sethi Ullman Compilers book. */
    {
      h = (h << 4) + *p;
      if ((g = h&0xf0000000))
  {
    h = h ^(g >> 24);
    h = h ^ g;
  }
    }
  return(h % PRIME);
}

/*BFUNC

MakeLink() is used to construct a link object. The link
is used for the hash table construct.

EFUNC*/


static LINK *MakeLink(tokentype,str,len)
     int tokentype;
     char *str;
     int len;
{
  BEGIN("MakeLink")
  LINK *temp;

  if (!(temp = MakeStructure(LINK)))            /* Make link */
    {
      WHEREAMI();
      printf("Cannot make a LINK.\n");
      exit(ERROR_MEMORY);
    }
  if (!(temp->lid = MakeStructure(struct id)))  /* Make id */
    {
      printf("Cannot make an id.\n");
      exit(ERROR_MEMORY);
    }
  temp->next = NULL;                            /* Set fields */
  if (!(temp->lid->name =(char *)calloc(len+1,sizeof(char))))
    {
      printf("Cannot make a string space for the link.\n");
      exit(ERROR_MEMORY);
    }
  strcpy(temp->lid->name,str);                 /* Copy key */
  temp->lid->tokentype = tokentype;
  temp->lid->count = 1;
  return(temp);
}

/*BFUNC

enter() is used to enter a Reserved Word or ID into the hash table.

EFUNC*/

static struct id *enter(tokentype,str,len)
     int tokentype;
     char *str;
     int len;
{
  BEGIN("enter")
  int hashnum;
  LINK *temp,*current;
  char *ptr;

  for(ptr=str;*ptr!='\0';ptr++)          /* All capitals is fine */
    {
      if ((*ptr>='a') && (*ptr<='z'))
  {
    *ptr = *ptr - ('a'-'A');
  }
    }
  hashnum = hashpjw(str);                /* Check if in hash table */
  for(temp=NULL,current=HashTable[hashnum];
      current!= NULL;
      current=current->next)
    {
      if (strcmp(str,current->lid->name) == 0)
  {
    temp=current;
    break;
  }
    }
  if (temp)                          /* Yes, found ID then return */
    {
      temp->lid->count++;
      return(temp->lid);
    }
  else                              /* Else make our own ID and return that*/
    {
      temp = MakeLink(tokentype,str,len);
      InsertLink(temp,HashTable[hashnum]);
      return(temp->lid);
    }
}

/*BFUNC

getint() takes an integer from the input.

EFUNC*/

static int getint()
{
  BEGIN("getint")
  int type;
  if ((type=yylex())!=R_INTEGER)
    {
      printf("Bad input, not integer, '%s' on line %d\n",
       yytext,
       yylineno);
      return(0);
    }
  return(yyint);
}

/*BFUNC

getstr() gets a string from the input. It copies the string to
temporary storage before it returns the pointer.

EFUNC*/

static char *getstr()
{
  BEGIN("getstr")
  char *tmp,*ptr,*bptr;
  int i,accum,flag;
  if (yylex() != R_STRING)                      /* Check if string */
    {
      printf("String expected.\n");
      if (!(tmp=(char *) malloc(sizeof(char))))
  {
    WHEREAMI();
    printf("Cannot allocate for null string.\n");
    exit(ERROR_MEMORY);
  }
      *tmp='\0';
      return(tmp);
    }
  if (!(tmp=(char *)calloc(strlen(yytext)+1,sizeof(char)))) /* Make space */
    {
      WHEREAMI();
      printf("Cannot allocate %d string space.\n",yyleng);
      exit(ERROR_MEMORY);
    }
  for(bptr=yytext+1,ptr=tmp;*bptr!='"';bptr++,ptr++)   /* Copy to string */
    {
      if (*bptr=='\\')
  {
    bptr++;
    for(flag=0,accum=0,i=0;i<3;i++)  /* Octal character lookahead */
      {
        if ((*bptr>='0')&&(*bptr<='7'))
    {
      accum = (accum<<3)+(*bptr-'0');
      bptr++;
      flag=1;
    }
        else
    {
      break;
    }
      }
    if (flag)
      {
        bptr--;
        *ptr=accum;
      }
    else                           /* Do conversions, if necessary */
      {
        switch(*(bptr))
    {
    case '0':
      *ptr = 0;
      break;
    case 'b':
      *ptr = 0x8;
      break;
    case 'i':
      *ptr = 0x9;
      break;
    case 'n':
      *ptr = 0xa;
      break;
    case 'v':
      *ptr = 0xb;
      break;
    case 'f':
      *ptr = 0xc;
      break;
    case 'r':
      *ptr = 0xd;
      break;
    default:
      *ptr=(*bptr);
    }
      }
  }
      else
  {
    *ptr = (*bptr);
  }
    }
  *ptr='\0';
  return(tmp);
}

/*BFUNC

parser() handles all of the parsing required for the Command
Interpreter.  It is basically a while statement with a very large case
statement for every input. The Command Interpreter is essentially
driven by the keywords. All unmatched values such as integers,
strings, and brackets, are ignored.

EFUNC*/

#define ARRAYBEGIN if (ntoken==R_LBRACKET)\
      {\
        arrayflag=1;\
        ntoken=yylex();\
      }\
    if (ntoken!=R_INTEGER)\
      {\
        WHEREAMI();\
        printf("Expected integer.\n");\
        break;\
      }\
    while(1)\
      {

#define ARRAYEND  if (arrayflag)\
    {\
      if ((ntoken=yylex())==R_RBRACKET) break;\
      else if (ntoken!=R_INTEGER)\
        {\
          WHEREAMI();\
          printf("Expected integer or right bracket.\n");\
          break;\
        }\
    }\
        else break;\
        }

void parser()
{
  BEGIN("parser")
  int i,dest,value,token,ntoken,arrayflag;
  int accum;
  int Start,End;
  int *ptr,*ptr2;

  while((token=yylex()))        /* The code handling is simple enough. */
    {                         /* just read the code and documentation */
      ErrorValue=0;           /* book... */
      arrayflag=0;
      switch(token)
  {
  case R_ECHO:
    printf("%s\n",getstr());
    break;
  case R_PRINTIMAGE:
    PrintImage();
    break;
  case R_PRINTFRAME:
    PrintFrame();
    break;
  case R_PRINTSCAN:
    PrintScan();
    break;
  case R_COMPONENT:
    ntoken=yylex();
    ARRAYBEGIN;
    dest = yyint;
    InBounds(dest,0,MAXIMUM_COMPONENTS-1,"Bad component reference");
    if (ErrorValue) break;
    if (yylex()!=R_LBRACKET)
      {
        WHEREAMI();
        printf("Expected left bracket.\n");
        break;
      }
    CFrame->ComponentFileName[dest] = getstr();
    value=getint();
    InBounds(value,0,MAXIMUM_HORIZONTAL_FREQUENCY,
       "Bad horizontal frequency");
    if (ErrorValue) break;
    CFrame->hf[dest]=value;
    value=getint();
    InBounds(value,0,MAXIMUM_VERTICAL_FREQUENCY,
       "Bad vertical frequency");
    if (ErrorValue) break;
    CFrame->vf[dest]=value;
    value=getint();
    InBounds(value,0,MAXIMUM_DEVICES-1,"Bad device reference");
    if (ErrorValue) break;
    CFrame->tq[dest]=value;
    CFrame->cn[CFrame->GlobalNumberComponents++]=dest;/*Know to use it*/
    if (yylex()!=R_RBRACKET)
      {
        WHEREAMI();
        printf("Expected right bracket.\n");
        break;
      }
    ARRAYEND;
    break;
  case R_SCAN:
    CScan->NumberComponents=0;
    ntoken=yylex();
    ARRAYBEGIN;
    if (CScan->NumberComponents>=MAXIMUM_SOURCES)
      {
        WHEREAMI();
        printf("Exceeded number of sources per scan.\n");
        break;
      }
    InBounds(yyint,0,MAXIMUM_COMPONENTS-1,"Bad component reference");
    if (ErrorValue) break;
    for(i=0;i<CFrame->GlobalNumberComponents;i++)  /* Check there */
            if (CFrame->cn[i]==yyint) break;
    if (i==CFrame->GlobalNumberComponents)
      {
        WHEREAMI();
        printf("Scan index not defined in frame yet.\n");
        break;
      }
    CScan->ci[CScan->NumberComponents] = yyint;
    if (yylex()!=R_LBRACKET)
      {
        WHEREAMI();
        printf("Expected left bracket.\n");
        break;
      }
    value=getint();
    InBounds(value,0,MAXIMUM_DEVICES-1,"Bad device reference");
    if (ErrorValue) break;
    CScan->td[CScan->NumberComponents]=value;
    value=getint();
    InBounds(value,0,MAXIMUM_DEVICES-1,"Bad device reference");
    if (ErrorValue) break;
    CScan->ta[CScan->NumberComponents]=value;
    CScan->NumberComponents++;
    if (yylex()!=R_RBRACKET)
      {
        WHEREAMI();
        printf("Expected right bracket.\n");
        break;
      }
    ARRAYEND;
    break;
  case R_QUANTIZATION:
    ntoken=yylex();
    ARRAYBEGIN;
    dest = yyint;
    InBounds(dest,0,MAXIMUM_DEVICES-1,
       "Bad quantization reference.");
    if (ErrorValue) break;
    ntoken=yylex();
    if (ntoken==R_LUMINANCEDEFAULT)
      {
        CImage->QuantizationMatrices[dest]=LuminanceQuantization;
        break;
      }
    else if (ntoken==R_CHROMINANCEDEFAULT)
      {
        CImage->QuantizationMatrices[dest]=ChrominanceQuantization;
        break;
      }
    else if (ntoken!=R_LBRACKET)
      {
        WHEREAMI();
        printf("Expected left bracket.\n");
        break;
      }
    CImage->NumberQuantizationMatrices =
      MAX(CImage->NumberQuantizationMatrices,(dest+1));
    if (!(ptr=(int *)calloc(64,sizeof(int))))
      {
        WHEREAMI();
        printf("Cannot allocate quantization matrix.\n");
        exit(ERROR_MEMORY);
      }
    CImage->NumberQuantizationMatrices =
      MAX(CImage->NumberQuantizationMatrices,(dest+1));
    CImage->QuantizationMatrices[dest]=ptr;
    for(i=0;i<64;i++)
      {
        ptr[i]=16;
      }
    for(i=0;i<65;i++,ptr++) /* One additional to force r-bracket */
      {
        if ((ntoken=yylex())!=R_INTEGER) break;
        InBounds(yyint,1,65535,"Integer out of bounds");
        if (ErrorValue) yyint=16;
        *ptr = yyint;
      }
    if (ntoken!=R_RBRACKET)
      {
        WHEREAMI();
        printf("Expected integer or right bracket.\n");
        break;
      }
    ARRAYEND;
    break;
  case R_ACSEND:
    ntoken=yylex();
    if (ntoken==R_EMPTY)
      {
        CScan->NumberACTablesSend = 0;
        break;
      }
    ARRAYBEGIN;
    if (CScan->NumberACTablesSend>=MAXIMUM_DEVICES)
      {
        WHEREAMI();
        printf("AC Huffman queue full.\n");
        break;
      }
    InBounds(yyint,0,MAXIMUM_DEVICES-1,"Bad device reference");
    if (ErrorValue) break;
    CScan->sa[CScan->NumberACTablesSend++] = yyint;
    ARRAYEND;
    break;
  case R_DCSEND:
    ntoken=yylex();
    if (ntoken==R_EMPTY)
      {
        CScan->NumberDCTablesSend = 0;
        break;
      }
    ARRAYBEGIN;
    if (CScan->NumberDCTablesSend>=MAXIMUM_DEVICES)
      {
        WHEREAMI();
        printf("DC Huffman queue full.\n");
        break;
      }
    InBounds(yyint,0,MAXIMUM_DEVICES-1,"Bad device reference");
    if (ErrorValue) break;
    CScan->sd[CScan->NumberDCTablesSend++] = yyint;
    ARRAYEND;
    break;
  case R_QSEND:
    ntoken=yylex();
    if (ntoken==R_EMPTY)
      {
        CScan->NumberQTablesSend = 0;
        break;
      }
    ARRAYBEGIN;
    if (CScan->NumberQTablesSend>=MAXIMUM_DEVICES)
      {
        WHEREAMI();
        printf("Quantization queue full.\n");
        break;
      }
    InBounds(yyint,0,MAXIMUM_DEVICES-1,"Bad device reference");
    if (ErrorValue) break;
    CScan->sq[CScan->NumberQTablesSend++] = yyint;
    ARRAYEND;
    break;
  case R_STREAMNAME:
    CImage->StreamFileName = getstr();
    break;
  case R_IMAGEWIDTH:
    value=getint();
    InBounds(value,0,MAXIMUM_IMAGE_WIDTH,"Bad image width");
    CFrame->GlobalWidth = value;
    break;
  case R_IMAGEHEIGHT:
    value=getint();
    InBounds(value,0,MAXIMUM_IMAGE_HEIGHT,"Bad image height");
    CFrame->GlobalHeight = value;
    break;
  case R_SCANDNL:
    ntoken=yylex();
    switch(ntoken)
      {
      case R_AUTO:
        CFrame->InsertDnl= -2;
        break;
      case R_ENABLE:
        CFrame->InsertDnl= -1;
        break;
      case R_INTEGER:
        CFrame->InsertDnl = yyint;
        break;
      default:
        WHEREAMI();
        printf("Expected integer.\n");
        break;
      }
    break;
  case R_FRAMEWIDTH:
    ntoken=yylex();
    ARRAYBEGIN;
    dest = yyint;
    InBounds(dest,0,MAXIMUM_COMPONENTS-1,"Bad component destination");
    if (ErrorValue) break;
    value=getint();
    InBounds(value,0,MAXIMUM_IMAGE_WIDTH,"Bad frame width");
    if (ErrorValue) break;
    CFrame->Width[dest] = value;
    ARRAYEND;
    break;
  case R_FRAMEHEIGHT:
    ntoken=yylex();
    ARRAYBEGIN;
    dest = yyint;
    InBounds(dest,0,MAXIMUM_COMPONENTS-1,"Bad component destination");
    if (ErrorValue) break;
    value=getint();
    InBounds(value,0,MAXIMUM_IMAGE_HEIGHT,"Bad frame height");
    if (ErrorValue) break;
    CFrame->Height[dest] = value;
    ARRAYEND;
    break;
  case R_RESYNC:
    value = getint();
    InBounds(value,0,MAXIMUM_RESYNC_INTERVAL,"Bad resync interval");
    if (ErrorValue) break;
    CFrame->ResyncInterval = value;
    break;
  case R_BUFFER:
    value = getint();
    InBounds(value,MINIMUM_BUFFERSIZE,
       MAXIMUM_BUFFERSIZE,"Bad buffersize");
    if (ErrorValue) break;
    CFrame->BufferSize = value;
    break;
  case R_OPENSCAN:
    CheckValidity();
    CheckBaseline();
    ConfirmFileSize();
    MakeIob(IOB_BLOCK,O_RDONLY,1);
    break;
  case R_CLOSESCAN:
    for(i=0;i<CScan->NumberComponents;i++)  /* Close all components */
      {
        InstallIob(i);
        CloseIob();
      }
    break;
  case R_OPENSTREAM:
    if (CImage->StreamFileName)
      {
        swopen(CImage->StreamFileName,0);  /* Index 0 open */
      }
    else
      {
        printf("StreamFileName: Null. Failed\n");
      }
    break;
  case R_CLOSESTREAM:
    swclose();
    break;
  case R_FREQUENCY:
    JpegFrequencyScan();
    break;
  case R_WRITESPECIAL:
    ntoken=yylex();
    ARRAYBEGIN;
    value = yyint;
    swbytealign();
    bputc(0xFF);  /* Marker */
    bputc(value&0xff);
    Start = swtell();
    bputw(0);
    if (yylex()!=R_LBRACKET)
      {
        WHEREAMI();
        printf("Expected left bracket.\n");
        End = swtell();
        swseek(Start);
        bputw((End-Start) >> 3);
        swseek(End);
        break;
      }
    while((ntoken=yylex())==R_INTEGER)
      {
        bputc(yyint&0xff);
      }
    if (ntoken!=R_RBRACKET)
      {
        WHEREAMI();
        printf("Expected integer or right bracket.\n");
        End = swtell();
        swseek(Start);
        bputw((End-Start) >> 3);
        swseek(End);
        break;
      }
    End = swtell();
    swseek(Start);
    bputw((End-Start) >> 3);
    swseek(End);
    ARRAYEND;
    break;
  case R_WRITEDIRECT:
    swbytealign();
    if (yylex()!=R_LBRACKET)
      {
        WHEREAMI();
        printf("Expected left bracket.\n");
        break;
      }
    while((ntoken=yylex())==R_INTEGER)
      {
        bputc(yyint&0xff);
      }
    if (ntoken!=R_RBRACKET)
      {
        WHEREAMI();
        printf("Expected integer or right bracket.\n");
        break;
      }
    break;
  case R_WRITESCAN:
    JpegEncodeScan();
    break;
  case R_WRITEFRAME:
    MakeConsistentFrameSize();  /* Do it here when everything defined */
    WriteSof();
    break;
  case R_WRITESOI:
    WriteSoi();
    break;
  case R_WRITEEOI:
    WriteEoi();
    break;
  case R_WRITEQUANTIZATION:
    WriteDqt();
    break;
  case R_WRITERESYNC:
    WriteDri();
    break;
  case R_WRITEDNL:
    WriteDnl();
    break;
  case R_WRITEHUFFMAN:
    WriteDht();
    break;
  case R_ACCUSTOM:
    ntoken=yylex();
    ARRAYBEGIN;
    dest = yyint;
    InBounds(dest,0,MAXIMUM_DEVICES-1,"Bad device reference");
    if (ErrorValue) break;
    if (yylex()!=R_LBRACKET)
      {
        WHEREAMI();
        printf("Expected left bracket.\n");
        break;
      }
    MakeXhuff();
    MakeEhuff();
    if (!(ptr=(int *)calloc(257,sizeof(int))))
      {
        WHEREAMI();
        printf("Out of custom frequency space.\n");
        exit(ERROR_MEMORY);
      }
    for(i=0;i<257;i++)
      {
        ptr[i]=0;
      }
    while((ntoken=yylex())==R_INTEGER)
      {
        InBounds(yyint,0,MAXIMUM_SOURCES-1,"Bad frequency reference");
        if(ErrorValue) yyint=0;
        AddFrequency(ptr,CScan->ACFrequency[yyint]);
      }
    if (ntoken!=R_RBRACKET)
      {
        WHEREAMI();
        printf("Expected right bracket.\n");
        break;
      }
    MakeHuffman(ptr);
    SetACHuffman(dest);
    CImage->NumberACTables =
      MAX(CImage->NumberACTables,(dest+1));
    ARRAYEND;
    break;
  case R_DCCUSTOM:
    ntoken=yylex();
    ARRAYBEGIN;
    dest = yyint;
    InBounds(dest,0,MAXIMUM_DEVICES-1,"Bad device reference");
    if (ErrorValue) break;
    if (yylex()!=R_LBRACKET)
      {
        WHEREAMI();
        printf("Expected left bracket.\n");
        break;
      }
    MakeXhuff();
    MakeEhuff();
    if (!(ptr=(int *)calloc(257,sizeof(int))))
      {
        WHEREAMI();
        printf("Out of custom frequency space.\n");
        exit(ERROR_MEMORY);
      }
    for(i=0;i<257;i++)
      {
        ptr[i]=0;
      }
    while((ntoken=yylex())==R_INTEGER)
      {
        InBounds(yyint,0,MAXIMUM_SOURCES-1,"Bad frequency reference");
        if(ErrorValue) yyint=0;
        AddFrequency(ptr,CScan->DCFrequency[yyint]);
      }
    if (ntoken!=R_RBRACKET)
      {
        WHEREAMI();
        printf("Expected right bracket.\n");
        break;
      }
    MakeHuffman(ptr);
    SetDCHuffman(dest);
    CImage->NumberDCTables =
      MAX(CImage->NumberDCTables,(dest+1));
    ARRAYEND;
    break;
  case R_ACSPEC:
    ntoken=yylex();
    ARRAYBEGIN;
    dest = yyint;
    InBounds(dest,0,MAXIMUM_DEVICES-1,"Bad device reference");
    if (ErrorValue) break;
    MakeXhuff();
    MakeEhuff();
    if ((ntoken=yylex())==R_LBRACKET)
      {
        if (!(ptr=(int *)calloc(38,sizeof(int)))) /* Get bits */
    {
      WHEREAMI();
      printf("Out of custom bits space.\n");
      exit(ERROR_MEMORY);
    }
        for(i=0;i<32;i++)
    {
      ptr[i]=0;
    }
        for(accum=0,i=0;i<17;i++) /* First index is bitlength of 1. */
    {                 /* One additional to force r-bracket. */
      ntoken=yylex();
      if (ntoken==R_INTEGER)
        {
          accum+=yyint;
          ptr[i]=yyint;
        }
      else break;
    }
        if (ntoken!=R_RBRACKET)
    {
      WHEREAMI();
      printf("Expected integer or right bracket.\n");
      break;
    }
        if (yylex()!=R_LBRACKET)  /* Get values */
    {
      WHEREAMI();
      printf("Expected left bracket.\n");
      break;
    }
        if (!(ptr2=(int *)calloc(257,sizeof(int))))
    {
      WHEREAMI();
      printf("Out of custom Huffman value space.\n");
      exit(ERROR_MEMORY);
    }
        for(i=0;i<257;i++)
    {
      ptr2[i]=0;
    }
        for(i=0;i<257;i++) /* One additinal to force r-bracket */
    {
      ntoken=yylex();
      if (ntoken==R_INTEGER)
        {
          ptr2[i]=yyint;
        }
      else break;
    }
        if (i!=accum)
    {
      WHEREAMI();
      printf("Number of bitlengths != number of values.");
    }
        if (ntoken!=R_RBRACKET)
    {
      WHEREAMI();
      printf("Expected integer or right bracket.\n");
      break;
    }
        SpecifiedHuffman(ptr,ptr2);
      }
    else if (ntoken==R_CHROMINANCEDEFAULT)
      {
        SpecifiedHuffman(ChrominanceACBits,ChrominanceACValues);
      }
    else if (ntoken==R_LUMINANCEDEFAULT)
      {
        SpecifiedHuffman(LuminanceACBits,LuminanceACValues);
      }
    else
      {
        WHEREAMI();
        printf("Expected left bracket or ACDEFAULT.\n");
        break;
      }
    SetACHuffman(dest);
    CImage->NumberACTables =
      MAX(CImage->NumberACTables,(dest+1));
    ARRAYEND;
    break;
  case R_DCSPEC:
    ntoken=yylex();
    ARRAYBEGIN;
    dest = yyint;
    InBounds(dest,0,MAXIMUM_DEVICES-1,"Bad device reference");
    if (ErrorValue) break;
    MakeXhuff();
    MakeEhuff();
    if ((ntoken=yylex())==R_LBRACKET)
      {
        if (!(ptr=(int *)calloc(38,sizeof(int)))) /* Get bits */
    {
      WHEREAMI();
      printf("Out of custom bits space.\n");
      exit(ERROR_MEMORY);
    }
        for(i=0;i<32;i++)
    {
      ptr[i]=0;
    }
        for(accum=0,i=0;i<17;i++) /* First index is bitlength of 1. */
    {                /* 0-16 to force right bracket. */
      ntoken=yylex();
      if (ntoken==R_INTEGER)
        {
          accum+=yyint;
          ptr[i]=yyint;
        }
      else break;
    }
        if (ntoken!=R_RBRACKET)
    {
      WHEREAMI();
      printf("Expected integer or right bracket.\n");
      break;
    }
        if (yylex()!=R_LBRACKET)  /* Get values */
    {
      WHEREAMI();
      printf("Expected left bracket.\n");
      break;
    }
        if (!(ptr2=(int *)calloc(257,sizeof(int))))
    {
      WHEREAMI();
      printf("Out of custom Huffman value space.\n");
      exit(ERROR_MEMORY);
    }
        for(i=0;i<257;i++)
    {
      ptr2[i]=0;
    }
        for(i=0;i<257;i++) /*One additional to force r-bracket.*/
    {
      ntoken=yylex();
      if (ntoken==R_INTEGER)
        {
          ptr2[i]=yyint;
        }
      else break;
    }
        if (i!=accum)
    {
      WHEREAMI();
      printf("Number of bitlengths != number of values.");
    }
        if (ntoken!=R_RBRACKET)
    {
      WHEREAMI();
      printf("Expected integer or right bracket.\n");
      break;
    }
        SpecifiedHuffman(ptr,ptr2);
      }
    else if (ntoken==R_CHROMINANCEDEFAULT)
      {
        SpecifiedHuffman(ChrominanceDCBits,ChrominanceDCValues);
      }
    else if (ntoken==R_LUMINANCEDEFAULT)
      {
        SpecifiedHuffman(LuminanceDCBits,LuminanceDCValues);
      }
    else
      {
        WHEREAMI();
        printf("Expected left bracket or DCDEFAULT.\n");
        break;
      }
    SetDCHuffman(dest);
    CImage->NumberDCTables =
      MAX(CImage->NumberDCTables,(dest+1));
    ARRAYEND;
    break;
  }
    }
}

/*NOPROTO*/
/*END*/
