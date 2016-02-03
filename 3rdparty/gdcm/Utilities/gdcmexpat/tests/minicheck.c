/* Miniature re-implementation of the "check" library.
 *
 * This is intended to support just enough of check to run the Expat
 * tests.  This interface is based entirely on the portion of the
 * check library being used.
 */

#include <stdio.h>
#include <stdlib.h>
#include <setjmp.h>
#include <assert.h>

#include "minicheck.h"

Suite *
suite_create(char *name)
{
    Suite *suite = (Suite *) calloc(1, sizeof(Suite));
    if (suite != NULL) {
        suite->name = name;
    }
    return suite;
}

TCase *
tcase_create(char *name)
{
    TCase *tc = (TCase *) calloc(1, sizeof(TCase));
    if (tc != NULL) {
        tc->name = name;
    }
    return tc;
}

void
suite_add_tcase(Suite *suite, TCase *tc) 
{
    assert(suite != NULL);
    assert(tc != NULL);
    assert(tc->next_tcase == NULL);

    tc->next_tcase = suite->tests;
    suite->tests = tc;
}

void
tcase_add_checked_fixture(TCase *tc,
                          tcase_setup_function setup,
                          tcase_teardown_function teardown)
{
    assert(tc != NULL);
    tc->setup = setup;
    tc->teardown = teardown;
}

void
tcase_add_test(TCase *tc, tcase_test_function test)
{
    assert(tc != NULL);
    if (tc->allocated == tc->ntests) {
        int nalloc = tc->allocated + 100;
        size_t new_size = sizeof(tcase_test_function) * nalloc;
        tcase_test_function *new_tests = realloc(tc->tests, new_size);
        assert(new_tests != NULL);
        if (new_tests != tc->tests) {
            free(tc->tests);
            tc->tests = new_tests;
        }
        tc->allocated = nalloc;
    }
    tc->tests[tc->ntests] = test;
    tc->ntests++;
}

SRunner *
srunner_create(Suite *suite)
{
    SRunner *runner = calloc(1, sizeof(SRunner));
    if (runner != NULL) {
        runner->suite = suite;
    }
    return runner;
}

void
srunner_set_fork_status(SRunner *runner, int status)
{
    /* We ignore this. */
}

static jmp_buf env;

static char const *_check_current_function = NULL;
static int _check_current_lineno = -1;
static char const *_check_current_filename = NULL;

void
_check_set_test_info(char const *function, char const *filename, int lineno)
{
    _check_current_function = function;
    _check_current_lineno = lineno;
    _check_current_filename = filename;
}


static void
add_failure(SRunner *runner, int verbosity)
{
    runner->nfailures++;
    if (verbosity >= CK_VERBOSE) {
        printf("%s:%d: %s\n", _check_current_filename,
               _check_current_lineno, _check_current_function);
    }
}

void
srunner_run_all(SRunner *runner, int verbosity)
{
    Suite *suite;
    TCase *tc;
    assert(runner != NULL);
    suite = runner->suite;
    tc = suite->tests;
    while (tc != NULL) {
        int i;
        for (i = 0; i < tc->ntests; ++i) {
            runner->nchecks++;

            if (tc->setup != NULL) {
                /* setup */
                if (setjmp(env)) {
                    add_failure(runner, verbosity);
                    continue;
                }
                tc->setup();
            }
            /* test */
            if (setjmp(env)) {
                add_failure(runner, verbosity);
                continue;
            }
            (tc->tests[i])();

            /* teardown */
            if (tc->teardown != NULL) {
                if (setjmp(env)) {
                    add_failure(runner, verbosity);
                    continue;
                }
                tc->teardown();
            }
        }
        tc = tc->next_tcase;
    }
    if (verbosity) {
        int passed = runner->nchecks - runner->nfailures;
        double percentage = ((double) passed) / runner->nchecks;
        int display = (int) (percentage * 100);
        printf("%d%%: Checks: %d, Failed: %d\n",
               display, runner->nchecks, runner->nfailures);
    }
}

void
_fail_unless(int condition, const char *file, int line, char *msg)
{
    longjmp(env, 1);
}

int
srunner_ntests_failed(SRunner *runner)
{
    assert(runner != NULL);
    return runner->nfailures;
}

void
srunner_free(SRunner *runner)
{
    free(runner->suite);
    free(runner);
}
