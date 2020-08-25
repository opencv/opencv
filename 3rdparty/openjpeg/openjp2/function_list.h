/*
 * The copyright in this software is being made available under the 2-clauses
 * BSD License, included below. This software may be subject to other third
 * party and contributor rights, including patent rights, and no such rights
 * are granted under this license.
 *
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

#ifndef OPJ_FUNCTION_LIST_H
#define OPJ_FUNCTION_LIST_H

/**
 * @file function_list.h
 * @brief Implementation of a list of procedures.

 * The functions in validation.c aims to have access to a list of procedures.
*/

/** @defgroup VAL VAL - validation procedure*/
/*@{*/

/**************************************************************************************************
 ***************************************** FORWARD DECLARATION ************************************
 **************************************************************************************************/

/**
 * declare a function pointer
 */
typedef void (*opj_procedure)(void);

/**
 * A list of procedures.
*/
typedef struct opj_procedure_list {
    /**
     * The number of validation procedures.
     */
    OPJ_UINT32 m_nb_procedures;
    /**
     * The number of the array of validation procedures.
     */
    OPJ_UINT32 m_nb_max_procedures;
    /**
     * The array of procedures.
     */
    opj_procedure * m_procedures;

} opj_procedure_list_t;

/* ----------------------------------------------------------------------- */

/**
 * Creates a validation list.
 *
 * @return  the newly created validation list.
 */
opj_procedure_list_t *  opj_procedure_list_create(void);

/**
 * Destroys a validation list.
 *
 * @param p_list the list to destroy.
 */
void  opj_procedure_list_destroy(opj_procedure_list_t * p_list);

/**
 * Adds a new validation procedure.
 *
 * @param   p_validation_list the list of procedure to modify.
 * @param   p_procedure     the procedure to add.
 * @param   p_manager the user event manager.
 *
 * @return  OPJ_TRUE if the procedure could be added.
 */
OPJ_BOOL opj_procedure_list_add_procedure(opj_procedure_list_t *
        p_validation_list, opj_procedure p_procedure, opj_event_mgr_t* p_manager);

/**
 * Gets the number of validation procedures.
 *
 * @param   p_validation_list the list of procedure to modify.
 *
 * @return the number of validation procedures.
 */
OPJ_UINT32 opj_procedure_list_get_nb_procedures(opj_procedure_list_t *
        p_validation_list);

/**
 * Gets the pointer on the first validation procedure. This function is similar to the C++
 * iterator class to iterate through all the procedures inside the validation list.
 * the caller does not take ownership of the pointer.
 *
 * @param   p_validation_list the list of procedure to get the first procedure from.
 *
 * @return  a pointer to the first procedure.
 */
opj_procedure* opj_procedure_list_get_first_procedure(opj_procedure_list_t *
        p_validation_list);


/**
 * Clears the list of validation procedures.
 *
 * @param   p_validation_list the list of procedure to clear.
 *
 */
void opj_procedure_list_clear(opj_procedure_list_t * p_validation_list);
/*@}*/

#endif /* OPJ_FUNCTION_LIST_H */

