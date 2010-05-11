#!/bin/sh
# configure script for zlib. This script is needed only if
# you wish to build a shared library and your system supports them,
# of if you need special compiler, flags or install directory.
# Otherwise, you can just use directly "make test; make install"
#
# To create a shared library, use "configure --shared"; by default a static
# library is created. If the primitive shared library support provided here
# does not work, use ftp://prep.ai.mit.edu/pub/gnu/libtool-*.tar.gz
#
# To impose specific compiler or flags or install directory, use for example:
#    prefix=$HOME CC=cc CFLAGS="-O4" ./configure
# or for csh/tcsh users:
#    (setenv prefix $HOME; setenv CC cc; setenv CFLAGS "-O4"; ./configure)
# LDSHARED is the command to be used to create a shared library

# Incorrect settings of CC or CFLAGS may prevent creating a shared library.
# If you have problems, try without defining CC and CFLAGS before reporting
# an error.

LIBS=libz.a
LDFLAGS="-L. ${LIBS}"
VER=`sed -n -e '/VERSION "/s/.*"\(.*\)".*/\1/p' < zlib.h`
VER2=`sed -n -e '/VERSION "/s/.*"\([0-9]*\\.[0-9]*\)\\..*/\1/p' < zlib.h`
VER1=`sed -n -e '/VERSION "/s/.*"\([0-9]*\)\\..*/\1/p' < zlib.h`
AR=${AR-"ar rc"}
RANLIB=${RANLIB-"ranlib"}
prefix=${prefix-/usr/local}
exec_prefix=${exec_prefix-'${prefix}'}
libdir=${libdir-'${exec_prefix}/lib'}
includedir=${includedir-'${prefix}/include'}
mandir=${mandir-'${prefix}/share/man'}
shared_ext='.so'
shared=0
gcc=0
old_cc="$CC"
old_cflags="$CFLAGS"

while test $# -ge 1
do
case "$1" in
    -h* | --h*)
      echo 'usage:'
      echo '  configure [--shared] [--prefix=PREFIX]  [--exec_prefix=EXPREFIX]'
      echo '     [--libdir=LIBDIR] [--includedir=INCLUDEDIR]'
        exit 0;;
    -p*=* | --p*=*) prefix=`echo $1 | sed 's/[-a-z_]*=//'`; shift;;
    -e*=* | --e*=*) exec_prefix=`echo $1 | sed 's/[-a-z_]*=//'`; shift;;
    -l*=* | --libdir=*) libdir=`echo $1 | sed 's/[-a-z_]*=//'`; shift;;
    -i*=* | --includedir=*) includedir=`echo $1 | sed 's/[-a-z_]*=//'`;shift;;
    -p* | --p*) prefix="$2"; shift; shift;;
    -e* | --e*) exec_prefix="$2"; shift; shift;;
    -l* | --l*) libdir="$2"; shift; shift;;
    -i* | --i*) includedir="$2"; shift; shift;;
    -s* | --s*) shared=1; shift;;
    *) echo "unknown option: $1"; echo "$0 --help for help"; exit 1;;
    esac
done

test=ztest$$
cat > $test.c <<EOF
extern int getchar();
int hello() {return getchar();}
EOF

test -z "$CC" && echo Checking for gcc...
cc=${CC-gcc}
cflags=${CFLAGS-"-O3"}
# to force the asm version use: CFLAGS="-O3 -DASMV" ./configure
case "$cc" in
  *gcc*) gcc=1;;
esac

if test "$gcc" -eq 1 && ($cc -c $cflags $test.c) 2>/dev/null; then
  CC="$cc"
  SFLAGS=${CFLAGS-"-fPIC -O3"}
  CFLAGS="$cflags"
  case `(uname -s || echo unknown) 2>/dev/null` in
  Linux | linux | GNU | GNU/*) LDSHARED=${LDSHARED-"$cc -shared -Wl,-soname,libz.so.1"};;
  CYGWIN* | Cygwin* | cygwin* | OS/2* )
             EXE='.exe';;
  QNX*)  # This is for QNX6. I suppose that the QNX rule below is for QNX2,QNX4
         # (alain.bonnefoy@icbt.com)
                 LDSHARED=${LDSHARED-"$cc -shared -Wl,-hlibz.so.1"};;
  HP-UX*)        LDSHARED=${LDSHARED-"$cc -shared $SFLAGS"}
                 shared_ext='.sl'
                 SHAREDLIB='libz.sl';;
  Darwin*)   shared_ext='.dylib'
             SHAREDLIB=libz$shared_ext
             SHAREDLIBV=libz.$VER$shared_ext
             SHAREDLIBM=libz.$VER1$shared_ext
             LDSHARED=${LDSHARED-"$cc -dynamiclib -install_name $libdir/$SHAREDLIBV -compatibility_version $VER2 -current_version $VER"};;
  *)             LDSHARED=${LDSHARED-"$cc -shared"};;
  esac
else
  # find system name and corresponding cc options
  CC=${CC-cc}
  case `(uname -sr || echo unknown) 2>/dev/null` in
  HP-UX*)    SFLAGS=${CFLAGS-"-O +z"}
             CFLAGS=${CFLAGS-"-O"}
#            LDSHARED=${LDSHARED-"ld -b +vnocompatwarnings"}
             LDSHARED=${LDSHARED-"ld -b"}
             shared_ext='.sl'
             SHAREDLIB='libz.sl';;
  IRIX*)     SFLAGS=${CFLAGS-"-ansi -O2 -rpath ."}
             CFLAGS=${CFLAGS-"-ansi -O2"}
             LDSHARED=${LDSHARED-"cc -shared"};;
  OSF1\ V4*) SFLAGS=${CFLAGS-"-O -std1"}
             CFLAGS=${CFLAGS-"-O -std1"}
             LDSHARED=${LDSHARED-"cc -shared  -Wl,-soname,libz.so -Wl,-msym -Wl,-rpath,$(libdir) -Wl,-set_version,${VER}:1.0"};;
  OSF1*)     SFLAGS=${CFLAGS-"-O -std1"}
             CFLAGS=${CFLAGS-"-O -std1"}
             LDSHARED=${LDSHARED-"cc -shared"};;
  QNX*)      SFLAGS=${CFLAGS-"-4 -O"}
             CFLAGS=${CFLAGS-"-4 -O"}
             LDSHARED=${LDSHARED-"cc"}
             RANLIB=${RANLIB-"true"}
             AR="cc -A";;
  SCO_SV\ 3.2*) SFLAGS=${CFLAGS-"-O3 -dy -KPIC "}
             CFLAGS=${CFLAGS-"-O3"}
             LDSHARED=${LDSHARED-"cc -dy -KPIC -G"};;
  SunOS\ 5*) SFLAGS=${CFLAGS-"-fast -xcg89 -KPIC -R."}
             CFLAGS=${CFLAGS-"-fast -xcg89"}
             LDSHARED=${LDSHARED-"cc -G"};;
  SunOS\ 4*) SFLAGS=${CFLAGS-"-O2 -PIC"}
             CFLAGS=${CFLAGS-"-O2"}
             LDSHARED=${LDSHARED-"ld"};;
  UNIX_System_V\ 4.2.0)
             SFLAGS=${CFLAGS-"-KPIC -O"}
             CFLAGS=${CFLAGS-"-O"}
             LDSHARED=${LDSHARED-"cc -G"};;
  UNIX_SV\ 4.2MP)
             SFLAGS=${CFLAGS-"-Kconform_pic -O"}
             CFLAGS=${CFLAGS-"-O"}
             LDSHARED=${LDSHARED-"cc -G"};;
  OpenUNIX\ 5)
             SFLAGS=${CFLAGS-"-KPIC -O"}
             CFLAGS=${CFLAGS-"-O"}
             LDSHARED=${LDSHARED-"cc -G"};;
  AIX*)  # Courtesy of dbakker@arrayasolutions.com
             SFLAGS=${CFLAGS-"-O -qmaxmem=8192"}
             CFLAGS=${CFLAGS-"-O -qmaxmem=8192"}
             LDSHARED=${LDSHARED-"xlc -G"};;
  # send working options for other systems to support@gzip.org
  *)         SFLAGS=${CFLAGS-"-O"}
             CFLAGS=${CFLAGS-"-O"}
             LDSHARED=${LDSHARED-"cc -shared"};;
  esac
fi

SHAREDLIB=${SHAREDLIB-"libz$shared_ext"}
SHAREDLIBV=${SHAREDLIBV-"libz$shared_ext.$VER"}
SHAREDLIBM=${SHAREDLIBM-"libz$shared_ext.$VER1"}

if test $shared -eq 1; then
  echo Checking for shared library support...
  # we must test in two steps (cc then ld), required at least on SunOS 4.x
  if test "`($CC -c $SFLAGS $test.c) 2>&1`" = "" &&
     test "`($LDSHARED -o $test$shared_ext $test.o) 2>&1`" = ""; then
    CFLAGS="$SFLAGS"
    LIBS="$SHAREDLIBV"
    echo Building shared library $SHAREDLIBV with $CC.
  elif test -z "$old_cc" -a -z "$old_cflags"; then
    echo No shared library support.
    shared=0;
  else
    echo 'No shared library support; try without defining CC and CFLAGS'
    shared=0;
  fi
fi
if test $shared -eq 0; then
  LDSHARED="$CC"
  echo Building static library $LIBS version $VER with $CC.
else
  LDFLAGS="-L. ${SHAREDLIBV}"
fi

cat > $test.c <<EOF
#include <unistd.h>
int main() { return 0; }
EOF
if test "`($CC -c $CFLAGS $test.c) 2>&1`" = ""; then
  sed < zconf.in.h "/HAVE_UNISTD_H/s%0%1%" > zconf.h
  echo "Checking for unistd.h... Yes."
else
  cp -p zconf.in.h zconf.h
  echo "Checking for unistd.h... No."
fi

cat > $test.c <<EOF
#include <stdio.h>
#include <stdarg.h>
#include "zconf.h"

int main()
{
#ifndef STDC
  choke me
#endif

  return 0;
}
EOF

if test "`($CC -c $CFLAGS $test.c) 2>&1`" = ""; then
  echo "Checking whether to use vs[n]printf() or s[n]printf()... using vs[n]printf()"

  cat > $test.c <<EOF
#include <stdio.h>
#include <stdarg.h>

int mytest(char *fmt, ...)
{
  char buf[20];
  va_list ap;

  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  return 0;
}

int main()
{
  return (mytest("Hello%d\n", 1));
}
EOF

  if test "`($CC $CFLAGS -o $test $test.c) 2>&1`" = ""; then
    echo "Checking for vsnprintf() in stdio.h... Yes."

    cat >$test.c <<EOF
#include <stdio.h>
#include <stdarg.h>

int mytest(char *fmt, ...)
{
  int n;
  char buf[20];
  va_list ap;

  va_start(ap, fmt);
  n = vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  return n;
}

int main()
{
  return (mytest("Hello%d\n", 1));
}
EOF

    if test "`($CC -c $CFLAGS $test.c) 2>&1`" = ""; then
      echo "Checking for return value of vsnprintf()... Yes."
    else
      CFLAGS="$CFLAGS -DHAS_vsnprintf_void"
      echo "Checking for return value of vsnprintf()... No."
      echo "  WARNING: apparently vsnprintf() does not return a value. zlib"
      echo "  can build but will be open to possible string-format security"
      echo "  vulnerabilities."
    fi
  else
    CFLAGS="$CFLAGS -DNO_vsnprintf"
    echo "Checking for vsnprintf() in stdio.h... No."
    echo "  WARNING: vsnprintf() not found, falling back to vsprintf(). zlib"
    echo "  can build but will be open to possible buffer-overflow security"
    echo "  vulnerabilities."

    cat >$test.c <<EOF
#include <stdio.h>
#include <stdarg.h>

int mytest(char *fmt, ...)
{
  int n;
  char buf[20];
  va_list ap;

  va_start(ap, fmt);
  n = vsprintf(buf, fmt, ap);
  va_end(ap);
  return n;
}

int main()
{
  return (mytest("Hello%d\n", 1));
}
EOF

    if test "`($CC -c $CFLAGS $test.c) 2>&1`" = ""; then
      echo "Checking for return value of vsprintf()... Yes."
    else
      CFLAGS="$CFLAGS -DHAS_vsprintf_void"
      echo "Checking for return value of vsprintf()... No."
      echo "  WARNING: apparently vsprintf() does not return a value. zlib"
      echo "  can build but will be open to possible string-format security"
      echo "  vulnerabilities."
    fi
  fi
else
  echo "Checking whether to use vs[n]printf() or s[n]printf()... using s[n]printf()"

  cat >$test.c <<EOF
#include <stdio.h>

int mytest()
{
  char buf[20];

  snprintf(buf, sizeof(buf), "%s", "foo");
  return 0;
}

int main()
{
  return (mytest());
}
EOF

  if test "`($CC $CFLAGS -o $test $test.c) 2>&1`" = ""; then
    echo "Checking for snprintf() in stdio.h... Yes."

    cat >$test.c <<EOF
#include <stdio.h>

int mytest()
{
  char buf[20];

  return snprintf(buf, sizeof(buf), "%s", "foo");
}

int main()
{
  return (mytest());
}
EOF

    if test "`($CC -c $CFLAGS $test.c) 2>&1`" = ""; then
      echo "Checking for return value of snprintf()... Yes."
    else
      CFLAGS="$CFLAGS -DHAS_snprintf_void"
      echo "Checking for return value of snprintf()... No."
      echo "  WARNING: apparently snprintf() does not return a value. zlib"
      echo "  can build but will be open to possible string-format security"
      echo "  vulnerabilities."
    fi
  else
    CFLAGS="$CFLAGS -DNO_snprintf"
    echo "Checking for snprintf() in stdio.h... No."
    echo "  WARNING: snprintf() not found, falling back to sprintf(). zlib"
    echo "  can build but will be open to possible buffer-overflow security"
    echo "  vulnerabilities."

    cat >$test.c <<EOF
#include <stdio.h>

int mytest()
{
  char buf[20];

  return sprintf(buf, "%s", "foo");
}

int main()
{
  return (mytest());
}
EOF

    if test "`($CC -c $CFLAGS $test.c) 2>&1`" = ""; then
      echo "Checking for return value of sprintf()... Yes."
    else
      CFLAGS="$CFLAGS -DHAS_sprintf_void"
      echo "Checking for return value of sprintf()... No."
      echo "  WARNING: apparently sprintf() does not return a value. zlib"
      echo "  can build but will be open to possible string-format security"
      echo "  vulnerabilities."
    fi
  fi
fi

cat >$test.c <<EOF
#include <errno.h>
int main() { return 0; }
EOF
if test "`($CC -c $CFLAGS $test.c) 2>&1`" = ""; then
  echo "Checking for errno.h... Yes."
else
  echo "Checking for errno.h... No."
  CFLAGS="$CFLAGS -DNO_ERRNO_H"
fi

cat > $test.c <<EOF
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
caddr_t hello() {
  return mmap((caddr_t)0, (off_t)0, PROT_READ, MAP_SHARED, 0, (off_t)0);
}
EOF
if test "`($CC -c $CFLAGS $test.c) 2>&1`" = ""; then
  CFLAGS="$CFLAGS -DUSE_MMAP"
  echo Checking for mmap support... Yes.
else
  echo Checking for mmap support... No.
fi

CPP=${CPP-"$CC -E"}
case $CFLAGS in
  *ASMV*)
    if test "`nm $test.o | grep _hello`" = ""; then
      CPP="$CPP -DNO_UNDERLINE"
      echo Checking for underline in external names... No.
    else
      echo Checking for underline in external names... Yes.
    fi;;
esac

rm -f $test.[co] $test $test$shared_ext

# udpate Makefile
sed < Makefile.in "
/^CC *=/s#=.*#=$CC#
/^CFLAGS *=/s#=.*#=$CFLAGS#
/^CPP *=/s#=.*#=$CPP#
/^LDSHARED *=/s#=.*#=$LDSHARED#
/^LIBS *=/s#=.*#=$LIBS#
/^SHAREDLIB *=/s#=.*#=$SHAREDLIB#
/^SHAREDLIBV *=/s#=.*#=$SHAREDLIBV#
/^SHAREDLIBM *=/s#=.*#=$SHAREDLIBM#
/^AR *=/s#=.*#=$AR#
/^RANLIB *=/s#=.*#=$RANLIB#
/^EXE *=/s#=.*#=$EXE#
/^prefix *=/s#=.*#=$prefix#
/^exec_prefix *=/s#=.*#=$exec_prefix#
/^libdir *=/s#=.*#=$libdir#
/^includedir *=/s#=.*#=$includedir#
/^mandir *=/s#=.*#=$mandir#
/^LDFLAGS *=/s#=.*#=$LDFLAGS#
" > Makefile
