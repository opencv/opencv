# Makefile for the main OpenJPEG codecs: j2k_to_image and image_to_j2k

CFLAGS = -O3 -lstdc++ # -g -p -pg

all: j2k_to_image image_to_j2k

j2k_to_image: j2k_to_image.c ../libopenjpeg.a
	gcc $(CFLAGS) compat/getopt.c index.c convert.c j2k_to_image.c -o j2k_to_image -L.. -lopenjpeg -I ../libopenjpeg/ -lm -ltiff

image_to_j2k: image_to_j2k.c ../libopenjpeg.a
	gcc $(CFLAGS) compat/getopt.c index.c convert.c image_to_j2k.c -o image_to_j2k -L.. -lopenjpeg -I ../libopenjpeg/ -lm -ltiff

clean:
	rm -f j2k_to_image image_to_j2k
