/*
 * Copyright (c) 1987, 1993, 1994
 *	The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed by the University of
 *	California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

/* last review : october 29th, 2002 */

#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)getopt.c	8.3 (Berkeley) 4/27/95";
#endif				/* LIBC_SCCS and not lint */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int opterr = 1,			/* if error message should be printed */
 optind = 1,			/* index into parent argv vector */
 optopt,			/* character checked for validity */
 optreset;			/* reset getopt */
const char *optarg;			/* argument associated with option */

typedef struct option
{
	const char *name;
	int has_arg;
	int *flag;
	int val;
}option_t;

#define	BADCH	(int)'?'
#define	BADARG	(int)':'
#define	EMSG	""

/* As this class remembers its values from one Java call to the other, reset the values before each use */
void reset_options_reading() {
	opterr = 1;
	optind = 1;
}

/*
 * getopt --
 *	Parse argc/argv argument vector.
 */
int getopt(int nargc, char *const *nargv, const char *ostr) {
#  define __progname nargv[0]
  static const char *place = EMSG;	/* option letter processing */
  char *oli;			/* option letter list index */

  if (optreset || !*place) {	/* update scanning pointer */
    optreset = 0;
    if (optind >= nargc || *(place = nargv[optind]) != '-') {
      place = EMSG;
      return (-1);
    }
    if (place[1] && *++place == '-') {	/* found "--" */
      ++optind;
      place = EMSG;
      return (-1);
    }
  }				/* option letter okay? */
  if ((optopt = (int) *place++) == (int) ':' ||
      !(oli = strchr(ostr, optopt))) {
    /*
     * if the user didn't specify '-' as an option,
     * assume it means -1.
     */
    if (optopt == (int) '-')
      return (-1);
    if (!*place)
      ++optind;
		if (opterr && *ostr != ':') {
      fprintf(stderr,
		     "%s: illegal option -- %c\n", __progname, optopt);
			return (BADCH);
		}
  }
  if (*++oli != ':') {		/* don't need argument */
    optarg = NULL;
    if (!*place)
      ++optind;
  } else {			/* need an argument */
    if (*place)			/* no white space */
      optarg = place;
    else if (nargc <= ++optind) {	/* no arg */
      place = EMSG;
      if (*ostr == ':')
	return (BADARG);
			if (opterr) {
				fprintf(stderr,
		       "%s: option requires an argument -- %c\n",
		       __progname, optopt);
				return (BADCH);
			}
    } else			/* white space */
      optarg = nargv[optind];
    place = EMSG;
    ++optind;
  }
  return (optopt);		/* dump back option letter */
}


int getopt_long(int argc, char * const argv[], const char *optstring,
struct option *longopts, int totlen) {
	static int lastidx,lastofs;
	char *tmp;
	int i,len;
	char param = 1;

again:
	if (optind >= argc || !argv[optind] || *argv[optind]!='-')
		return -1;

	if (argv[optind][0]=='-' && argv[optind][1]==0) {
		if(optind >= (argc - 1)){ /* no more input parameters */
			param = 0;
		}
		else{ /* more input parameters */
			if(argv[optind + 1][0] == '-'){
				param = 0; /* Missing parameter after '-' */
			}
			else{
				param = 2;
			}
		}
	}

	if (param == 0) {
		++optind;
		return (BADCH);
	}

	if (argv[optind][0]=='-') {	/* long option */
		char* arg=argv[optind]+1;
		const struct option* o;
		o=longopts;
		len=sizeof(longopts[0]);

		if (param > 1){
			arg = argv[optind+1];
			optind++;
		}
		else
			arg = argv[optind]+1;

		if(strlen(arg)>1){
			for (i=0;i<totlen;i=i+len,o++) {
				if (!strcmp(o->name,arg)) {	/* match */
					if (o->has_arg == 0) {
						if ((argv[optind+1])&&(!(argv[optind+1][0]=='-'))){
							fprintf(stderr,"%s: option does not require an argument. Ignoring %s\n",arg,argv[optind+1]);
							++optind;
						}
					}else{ 
						optarg=argv[optind+1];
						if(optarg){
							if (optarg[0] == '-'){ /* Has read next input parameter: No arg for current parameter */								
								if (opterr) {
									fprintf(stderr,"%s: option requires an argument\n",arg);
									return (BADCH);
								}
							}
						}
						if (!optarg && o->has_arg==1) {	/* no argument there */
							if (opterr) {
								fprintf(stderr,"%s: option requires an argument \n",arg);
								return (BADCH);
							}
						}
						++optind;
					}
					++optind;
					if (o->flag)
						*(o->flag)=o->val;
					else
						return o->val;
					return 0;
				}
			}//(end for)String not found in the list
			fprintf(stderr,"Invalid option %s\n",arg);
			++optind;
			return (BADCH);
		}else{ /*Single character input parameter*/
			if (*optstring==':') return ':';
			if (lastidx!=optind) {
				lastidx=optind; lastofs=0;
			}
			optopt=argv[optind][lastofs+1];
			if ((tmp=strchr(optstring,optopt))) {/*Found input parameter in list*/
				if (*tmp==0) {	/* apparently, we looked for \0, i.e. end of argument */
					++optind;
					goto again;
				}
				if (tmp[1]==':') {	/* argument expected */
					if (tmp[2]==':' || argv[optind][lastofs+2]) {	/* "-foo", return "oo" as optarg */
						if (!*(optarg=argv[optind]+lastofs+2)) optarg=0;
						goto found;
					}
					optarg=argv[optind+1];
					if(optarg){
						if (optarg[0] == '-'){ /* Has read next input parameter: No arg for current parameter */
							if (opterr) {
								fprintf(stderr,"%s: option requires an argument\n",arg);
								return (BADCH);
							}
						}
					}
					if (!optarg) {	/* missing argument */
						if (opterr) {
							fprintf(stderr,"%s: option requires an argument\n",arg);
							return (BADCH);
						}
					}
					++optind;
				}else {/*Argument not expected*/
					++lastofs;
					return optopt;
				}
found:
				++optind;
				return optopt;
			}	else {	/* not found */
				fprintf(stderr,"Invalid option %s\n",arg);
				++optind;
				return (BADCH);
			}//end of not found
		
		}// end of single character
	}//end '-'
	fprintf(stderr,"Invalid option\n");
	++optind;
	return (BADCH);;
}//end function
