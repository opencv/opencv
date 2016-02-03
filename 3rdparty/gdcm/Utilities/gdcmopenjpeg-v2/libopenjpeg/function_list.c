/*
 * Copyright (c) 2008, Jerome Fimes, Communications & Systemes <jerome.fimes@c-s.fr>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS `AS IS'
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "function_list.h"
#include "opj_includes.h"
#include "opj_malloc.h"
/**
 * Default size of the validation list, if not sufficient, data will be reallocated with a double size.
 */
#define OPJ_VALIDATION_SIZE 10

/**
 * Creates a validation list.
 *
 * @return  the newly created validation list.
 */
opj_procedure_list_t *  opj_procedure_list_create()
{
  /* memory allocation */
  opj_procedure_list_t * l_validation = (opj_procedure_list_t *) opj_malloc(sizeof(opj_procedure_list_t));
  if
    (! l_validation)
  {
    return 00;
  }
  /* initialization */
  memset(l_validation,0,sizeof(opj_procedure_list_t));
  l_validation->m_nb_max_procedures = OPJ_VALIDATION_SIZE;
  l_validation->m_procedures = (void**)opj_malloc(
    OPJ_VALIDATION_SIZE * sizeof(opj_procedure));
  if
    (! l_validation->m_procedures)
  {
    opj_free(l_validation);
    return 00;
  }
  memset(l_validation->m_procedures,0,OPJ_VALIDATION_SIZE * sizeof(opj_procedure));
  return l_validation;
}



/**
 * Destroys a validation list.
 *
 * @param p_list the list to destroy.
 */
void  opj_procedure_list_destroy(opj_procedure_list_t * p_list)
{
  if
    (! p_list)
  {
    return;
  }
  /* initialization */
  if
    (p_list->m_procedures)
  {
    opj_free(p_list->m_procedures);
  }
  opj_free(p_list);
}

/**
 * Adds a new validation procedure.
 *
 * @param  p_validation_list the list of procedure to modify.
 * @param  p_procedure    the procedure to add.
 */
bool  opj_procedure_list_add_procedure (opj_procedure_list_t * p_validation_list, opj_procedure p_procedure)
{
  if
    (p_validation_list->m_nb_max_procedures == p_validation_list->m_nb_procedures)
  {
    p_validation_list->m_nb_max_procedures += OPJ_VALIDATION_SIZE;
    p_validation_list->m_procedures = (void**)opj_realloc(
    p_validation_list->m_procedures,p_validation_list->m_nb_max_procedures * sizeof(opj_procedure));
    if
      (! p_validation_list->m_procedures)
    {
      p_validation_list->m_nb_max_procedures = 0;
      p_validation_list->m_nb_procedures = 0;
      return false;
    }
  }
  p_validation_list->m_procedures[p_validation_list->m_nb_procedures] = p_procedure;
  ++p_validation_list->m_nb_procedures;
  return true;
}

/**
 * Gets the number of validation procedures.
 *
 * @param  p_validation_list the list of procedure to modify.
 *
 * @return the number of validation procedures.
 */
OPJ_UINT32 opj_procedure_list_get_nb_procedures (opj_procedure_list_t * p_validation_list)
{
  return p_validation_list->m_nb_procedures;
}

/**
 * Gets the pointer on the first validation procedure. This function is similar to the C++
 * iterator class to iterate through all the procedures inside the validation list.
 * the caller does not take ownership of the pointer.
 *
 * @param  p_validation_list the list of procedure to get the first procedure from.
 *
 * @return  a pointer to the first procedure.
 */
opj_procedure* opj_procedure_list_get_first_procedure (opj_procedure_list_t * p_validation_list)
{
  return p_validation_list->m_procedures;
}

/**
 * Clears the list of validation procedures.
 *
 * @param  p_validation_list the list of procedure to clear.
 *
 */
void  opj_procedure_list_clear (opj_procedure_list_t * p_validation_list)
{
  p_validation_list->m_nb_procedures = 0;
}
