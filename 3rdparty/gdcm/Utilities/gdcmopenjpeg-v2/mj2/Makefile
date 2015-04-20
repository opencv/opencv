# Makefile for the MJ2 codecs of the OpenJPEG library: frames_to_mj2, mj2_to_frames, extract_j2k_from_mj2 and wrap_j2k_in_mj2

CFLAGS = -O3 -lstdc++ # -g -p -pg

all: frames_to_mj2 mj2_to_frames extract_j2k_from_mj2 wrap_j2k_in_mj2

frames_to_mj2: frames_to_mj2.c ../libopenjpeg.a
	gcc $(CFLAGS) compat/getopt.c mj2_convert.c mj2.c frames_to_mj2.c -o frames_to_mj2 -L.. -lopenjpeg -I ../libopenjpeg/ -lm

mj2_to_frames: mj2_to_frames.c ../libopenjpeg.a
	gcc $(CFLAGS) compat/getopt.c mj2_convert.c mj2.c mj2_to_frames.c -o mj2_to_frames -L.. -lopenjpeg -I ../libopenjpeg/ -lm

extract_j2k_from_mj2: extract_j2k_from_mj2.c ../libopenjpeg.a
	gcc $(CFLAGS) mj2.c extract_j2k_from_mj2.c -o extract_j2k_from_mj2 -L.. -lopenjpeg -I ../libopenjpeg/ -lm

wrap_j2k_in_mj2: wrap_j2k_in_mj2.c ../libopenjpeg.a
	gcc $(CFLAGS) mj2.c wrap_j2k_in_mj2.c -o wrap_j2k_in_mj2 -L.. -lopenjpeg -I ../libopenjpeg/ -lm

clean:
	rm -f frames_to_mj2 mj2_to_frames extract_j2k_from_mj2 wrap_j2k_in_mj2
