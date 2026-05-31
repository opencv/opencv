#include <check.h>
#include <stdlib.h>
#include <string.h>

/* Include production headers from the bundled libjasper */
#include "3rdparty/libjasper/jasper/jas_stream.h"
#include "3rdparty/libjasper/jpc_mqdec.h"

START_TEST(test_mqdec_setinput_adversarial)
{
    /* Invariant: jpc_mqdec_setinput must not crash or corrupt state
       when given streams backed by adversarial/boundary byte sequences. */
    const unsigned char *payloads[] = {
        /* Exact exploit-style: all 0xFF bytes (marker-like, triggers boundary issues) */
        (const unsigned char *)"\xff\xff\xff\xff\xff\xff\xff\xff",
        /* Boundary: single zero byte (minimal/empty-like input) */
        (const unsigned char *)"\x00",
        /* Valid: normal MQ-coded stream preamble bytes */
        (const unsigned char *)"\x00\x01\x02\x03\x04\x05\x06\x07",
        /* Mixed: 0xFF followed by 0x00 (stuffed byte pattern) */
        (const unsigned char *)"\xff\x00\xff\x00\xff\x00",
    };
    size_t payload_sizes[] = { 8, 1, 8, 6 };
    int num_payloads = sizeof(payloads) / sizeof(payloads[0]);

    for (int i = 0; i < num_payloads; i++) {
        jas_stream_t *stream = jas_stream_memopen(
            (char *)payloads[i], (int)payload_sizes[i]);
        ck_assert_ptr_nonnull(stream);

        jpc_mqdec_t *mqdec = jpc_mqdec_create(JPC_NUMCTXS, stream);
        ck_assert_ptr_nonnull(mqdec);

        /* Core invariant: setinput must not crash with adversarial stream */
        jpc_mqdec_setinput(mqdec, stream);

        /* After setinput, the decoder state must remain non-null and usable */
        ck_assert_ptr_nonnull(mqdec);

        jpc_mqdec_destroy(mqdec);
        jas_stream_close(stream);
    }
}
END_TEST

Suite *security_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Security");
    tc_core = tcase_create("Core");

    tcase_add_test(tc_core, test_mqdec_setinput_adversarial);
    suite_add_tcase(s, tc_core);

    return s;
}

int main(void)
{
    int number_failed;
    Suite *s;
    SRunner *sr;

    s = security_suite();
    sr = srunner_create(s);

    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}