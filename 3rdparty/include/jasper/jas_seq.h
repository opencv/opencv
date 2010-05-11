/*
 * Copyright (c) 1999-2000 Image Power, Inc. and the University of
 *   British Columbia.
 * Copyright (c) 2001-2002 Michael David Adams.
 * All rights reserved.
 */

/* __START_OF_JASPER_LICENSE__
 * 
 * JasPer License Version 2.0
 * 
 * Copyright (c) 2001-2006 Michael David Adams
 * Copyright (c) 1999-2000 Image Power, Inc.
 * Copyright (c) 1999-2000 The University of British Columbia
 * 
 * All rights reserved.
 * 
 * Permission is hereby granted, free of charge, to any person (the
 * "User") obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, and/or sell copies of the Software, and to permit
 * persons to whom the Software is furnished to do so, subject to the
 * following conditions:
 * 
 * 1.  The above copyright notices and this permission notice (which
 * includes the disclaimer below) shall be included in all copies or
 * substantial portions of the Software.
 * 
 * 2.  The name of a copyright holder shall not be used to endorse or
 * promote products derived from the Software without specific prior
 * written permission.
 * 
 * THIS DISCLAIMER OF WARRANTY CONSTITUTES AN ESSENTIAL PART OF THIS
 * LICENSE.  NO USE OF THE SOFTWARE IS AUTHORIZED HEREUNDER EXCEPT UNDER
 * THIS DISCLAIMER.  THE SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS
 * "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT OF THIRD PARTY RIGHTS.  IN NO
 * EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL
 * INDIRECT OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING
 * FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
 * NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
 * WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.  NO ASSURANCES ARE
 * PROVIDED BY THE COPYRIGHT HOLDERS THAT THE SOFTWARE DOES NOT INFRINGE
 * THE PATENT OR OTHER INTELLECTUAL PROPERTY RIGHTS OF ANY OTHER ENTITY.
 * EACH COPYRIGHT HOLDER DISCLAIMS ANY LIABILITY TO THE USER FOR CLAIMS
 * BROUGHT BY ANY OTHER ENTITY BASED ON INFRINGEMENT OF INTELLECTUAL
 * PROPERTY RIGHTS OR OTHERWISE.  AS A CONDITION TO EXERCISING THE RIGHTS
 * GRANTED HEREUNDER, EACH USER HEREBY ASSUMES SOLE RESPONSIBILITY TO SECURE
 * ANY OTHER INTELLECTUAL PROPERTY RIGHTS NEEDED, IF ANY.  THE SOFTWARE
 * IS NOT FAULT-TOLERANT AND IS NOT INTENDED FOR USE IN MISSION-CRITICAL
 * SYSTEMS, SUCH AS THOSE USED IN THE OPERATION OF NUCLEAR FACILITIES,
 * AIRCRAFT NAVIGATION OR COMMUNICATION SYSTEMS, AIR TRAFFIC CONTROL
 * SYSTEMS, DIRECT LIFE SUPPORT MACHINES, OR WEAPONS SYSTEMS, IN WHICH
 * THE FAILURE OF THE SOFTWARE OR SYSTEM COULD LEAD DIRECTLY TO DEATH,
 * PERSONAL INJURY, OR SEVERE PHYSICAL OR ENVIRONMENTAL DAMAGE ("HIGH
 * RISK ACTIVITIES").  THE COPYRIGHT HOLDERS SPECIFICALLY DISCLAIM ANY
 * EXPRESS OR IMPLIED WARRANTY OF FITNESS FOR HIGH RISK ACTIVITIES.
 * 
 * __END_OF_JASPER_LICENSE__
 */

/*
 * Sequence/Matrix Library
 *
 * $Id: jas_seq.h,v 1.2 2008-05-26 09:41:51 vp153 Exp $
 */

#ifndef JAS_SEQ_H
#define JAS_SEQ_H

/******************************************************************************\
* Includes.
\******************************************************************************/

#include <jasper/jas_config.h>

#include <jasper/jas_stream.h>
#include <jasper/jas_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************\
* Constants.
\******************************************************************************/

/* This matrix is a reference to another matrix. */
#define JAS_MATRIX_REF	0x0001

/******************************************************************************\
* Types.
\******************************************************************************/

/* An element in a sequence. */
typedef int_fast32_t jas_seqent_t;

/* An element in a matrix. */
typedef int_fast32_t jas_matent_t;

/* Matrix. */

typedef struct {

	/* Additional state information. */
	int flags_;

	/* The starting horizontal index. */
	int_fast32_t xstart_;

	/* The starting vertical index. */
	int_fast32_t ystart_;

	/* The ending horizontal index. */
	int_fast32_t xend_;

	/* The ending vertical index. */
	int_fast32_t yend_;

	/* The number of rows in the matrix. */
	int_fast32_t numrows_;

	/* The number of columns in the matrix. */
	int_fast32_t numcols_;

	/* Pointers to the start of each row. */
	jas_seqent_t **rows_;

	/* The allocated size of the rows array. */
	int_fast32_t maxrows_;

	/* The matrix data buffer. */
	jas_seqent_t *data_;

	/* The allocated size of the data array. */
	int_fast32_t datasize_;

} jas_matrix_t;

typedef jas_matrix_t jas_seq2d_t;
typedef jas_matrix_t jas_seq_t;

/******************************************************************************\
* Functions/macros for matrix class.
\******************************************************************************/

/* Get the number of rows. */
#define jas_matrix_numrows(matrix) \
	((matrix)->numrows_)

/* Get the number of columns. */
#define jas_matrix_numcols(matrix) \
	((matrix)->numcols_)

/* Get a matrix element. */
#define jas_matrix_get(matrix, i, j) \
	((matrix)->rows_[i][j])

/* Set a matrix element. */
#define jas_matrix_set(matrix, i, j, v) \
	((matrix)->rows_[i][j] = (v))

/* Get an element from a matrix that is known to be a row or column vector. */
#define jas_matrix_getv(matrix, i) \
	(((matrix)->numrows_ == 1) ? ((matrix)->rows_[0][i]) : \
	  ((matrix)->rows_[i][0]))

/* Set an element in a matrix that is known to be a row or column vector. */
#define jas_matrix_setv(matrix, i, v) \
	(((matrix)->numrows_ == 1) ? ((matrix)->rows_[0][i] = (v)) : \
	  ((matrix)->rows_[i][0] = (v)))

/* Get the address of an element in a matrix. */
#define	jas_matrix_getref(matrix, i, j) \
	(&(matrix)->rows_[i][j])

#define	jas_matrix_getvref(matrix, i) \
	(((matrix)->numrows_ > 1) ? jas_matrix_getref(matrix, i, 0) : jas_matrix_getref(matrix, 0, i))

#define jas_matrix_length(matrix) \
	(max((matrix)->numrows_, (matrix)->numcols_))

/* Create a matrix with the specified dimensions. */
jas_matrix_t *jas_matrix_create(int numrows, int numcols);

/* Destroy a matrix. */
void jas_matrix_destroy(jas_matrix_t *matrix);

/* Resize a matrix.  The previous contents of the matrix are lost. */
int jas_matrix_resize(jas_matrix_t *matrix, int numrows, int numcols);

int jas_matrix_output(jas_matrix_t *matrix, FILE *out);

/* Create a matrix that references part of another matrix. */
void jas_matrix_bindsub(jas_matrix_t *mat0, jas_matrix_t *mat1, int r0, int c0,
  int r1, int c1);

/* Create a matrix that is a reference to a row of another matrix. */
#define jas_matrix_bindrow(mat0, mat1, r) \
  (jas_matrix_bindsub((mat0), (mat1), (r), 0, (r), (mat1)->numcols_ - 1))

/* Create a matrix that is a reference to a column of another matrix. */
#define jas_matrix_bindcol(mat0, mat1, c) \
  (jas_matrix_bindsub((mat0), (mat1), 0, (c), (mat1)->numrows_ - 1, (c)))

/* Clip the values of matrix elements to the specified range. */
void jas_matrix_clip(jas_matrix_t *matrix, jas_seqent_t minval,
  jas_seqent_t maxval);

/* Arithmetic shift left of all elements in a matrix. */
void jas_matrix_asl(jas_matrix_t *matrix, int n);

/* Arithmetic shift right of all elements in a matrix. */
void jas_matrix_asr(jas_matrix_t *matrix, int n);

/* Almost-but-not-quite arithmetic shift right of all elements in a matrix. */
void jas_matrix_divpow2(jas_matrix_t *matrix, int n);

/* Set all elements of a matrix to the specified value. */
void jas_matrix_setall(jas_matrix_t *matrix, jas_seqent_t val);

/* The spacing between rows of a matrix. */
#define	jas_matrix_rowstep(matrix) \
	(((matrix)->numrows_ > 1) ? ((matrix)->rows_[1] - (matrix)->rows_[0]) : (0))

/* The spacing between columns of a matrix. */
#define	jas_matrix_step(matrix) \
	(((matrix)->numrows_ > 1) ? (jas_matrix_rowstep(matrix)) : (1))

/* Compare two matrices for equality. */
int jas_matrix_cmp(jas_matrix_t *mat0, jas_matrix_t *mat1);

jas_matrix_t *jas_matrix_copy(jas_matrix_t *x);

jas_matrix_t *jas_matrix_input(FILE *);

/******************************************************************************\
* Functions/macros for 2-D sequence class.
\******************************************************************************/

jas_seq2d_t *jas_seq2d_copy(jas_seq2d_t *x);

jas_matrix_t *jas_seq2d_create(int xstart, int ystart, int xend, int yend);

#define	jas_seq2d_destroy(s) \
	jas_matrix_destroy(s)

#define	jas_seq2d_xstart(s) \
	((s)->xstart_)
#define	jas_seq2d_ystart(s) \
	((s)->ystart_)
#define	jas_seq2d_xend(s) \
	((s)->xend_)
#define	jas_seq2d_yend(s) \
	((s)->yend_)
#define	jas_seq2d_getref(s, x, y) \
	(jas_matrix_getref(s, (y) - (s)->ystart_, (x) - (s)->xstart_))
#define	jas_seq2d_get(s, x, y) \
	(jas_matrix_get(s, (y) - (s)->ystart_, (x) - (s)->xstart_))
#define	jas_seq2d_rowstep(s) \
	jas_matrix_rowstep(s)
#define	jas_seq2d_width(s) \
	((s)->xend_ - (s)->xstart_)
#define	jas_seq2d_height(s) \
	((s)->yend_ - (s)->ystart_)
#define	jas_seq2d_setshift(s, x, y) \
	((s)->xstart_ = (x), (s)->ystart_ = (y), \
	  (s)->xend_ = (s)->xstart_ + (s)->numcols_, \
	  (s)->yend_ = (s)->ystart_ + (s)->numrows_)

void jas_seq2d_bindsub(jas_matrix_t *s, jas_matrix_t *s1, int xstart,
  int ystart, int xend, int yend);

/******************************************************************************\
* Functions/macros for 1-D sequence class.
\******************************************************************************/

#define	jas_seq_create(start, end) \
	(jas_seq2d_create(start, 0, end, 1))

#define	jas_seq_destroy(seq) \
	(jas_seq2d_destroy(seq))

#define jas_seq_set(seq, i, v) \
	((seq)->rows_[0][(i) - (seq)->xstart_] = (v))
#define	jas_seq_getref(seq, i) \
	(&(seq)->rows_[0][(i) - (seq)->xstart_])
#define	jas_seq_get(seq, i) \
	((seq)->rows_[0][(i) - (seq)->xstart_])
#define	jas_seq_start(seq) \
	((seq)->xstart_)
#define	jas_seq_end(seq) \
	((seq)->xend_)

#ifdef __cplusplus
}
#endif

#endif
