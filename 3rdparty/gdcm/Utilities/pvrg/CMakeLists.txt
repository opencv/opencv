
set(
  PVRG_SRCS
  jpeg.c
  codec.c
  huffman.c
  io.c
  chendct.c
  leedct.c
  marker.c
  stream.c
  transform.c
  lexer.c
)


# might not be portable, so simply rename the function in the code itself...
#set_source_files_properties(jpeg.c
#  PROPERTIES
#  COMPILE_FLAGS -Dmain=pvrgmain
#)
if(MSVC)
add_definitions(-DNOTRUNCATE)
else()
add_definitions(-DSYSV -DNOTRUNCATE)
endif()
#add_definitions(-DIO_DEBUG)
#add_library(pvrg ${PVRG_SRCS})

add_executable(gdcmjpeg ${PVRG_SRCS})
#target_link_libraries(pvrgjpeg pvrg)


#
# Should you wish to modify the interpreter
# modify this portion here.
#
# Make sure that you move the first comment from the start to
# within the braces %{ /* */ %} in lexer.l,
# otherwise it will give you a error (definitions too long).
#
#  Caution: Sometimes -ll is required.
#
#
#lexer.c: lexer.l
#	lex lexer.l
#	mv lex.yy.c lexer.c
#
