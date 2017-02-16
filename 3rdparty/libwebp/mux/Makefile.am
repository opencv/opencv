lib_LTLIBRARIES = libwebpmux.la

libwebpmux_la_SOURCES =
libwebpmux_la_SOURCES += anim_encode.c
libwebpmux_la_SOURCES += animi.h
libwebpmux_la_SOURCES += muxedit.c
libwebpmux_la_SOURCES += muxi.h
libwebpmux_la_SOURCES += muxinternal.c
libwebpmux_la_SOURCES += muxread.c

libwebpmuxinclude_HEADERS =
libwebpmuxinclude_HEADERS += ../webp/mux.h
libwebpmuxinclude_HEADERS += ../webp/mux_types.h
libwebpmuxinclude_HEADERS += ../webp/types.h

libwebpmux_la_LIBADD = ../libwebp.la
libwebpmux_la_LDFLAGS = -no-undefined -version-info 3:0:0 -lm
libwebpmuxincludedir = $(includedir)/webp
pkgconfig_DATA = libwebpmux.pc
