#include <check.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>
#include "jpeglib.h"

struct my_error_mgr {
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};

static void my_error_exit(j_common_ptr cinfo) {
    struct my_error_mgr *myerr = (struct my_error_mgr *)cinfo->err;
    longjmp(myerr->setjmp_buffer, 1);
}

START_TEST(test_jpeg_consume_input_adversarial)
{
    /* Invariant: jpeg_consume_input must not crash or cause undefined behavior
       on malformed/truncated JPEG data; it must gracefully error out. */
    static const unsigned char truncated_jpeg[] = {
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46,
        0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
        0x00, 0x01, 0x00, 0x00  /* truncated after APP0 */
    };
    static const unsigned char garbage[] = {
        0xFF, 0xD8, 0xDE, 0xAD, 0xBE, 0xEF, 0xFF, 0xFF,
        0x00, 0x00, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41
    };
    static const unsigned char not_jpeg[] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    };

    const unsigned char *payloads[] = { truncated_jpeg, garbage, not_jpeg };
    size_t sizes[] = { sizeof(truncated_jpeg), sizeof(garbage), sizeof(not_jpeg) };

    for (int i = 0; i < 3; i++) {
        struct jpeg_decompress_struct cinfo;
        struct my_error_mgr jerr;

        cinfo.err = jpeg_std_error(&jerr.pub);
        jerr.pub.error_exit = my_error_exit;

        if (setjmp(jerr.setjmp_buffer)) {
            /* Graceful error path — this is acceptable */
            jpeg_destroy_decompress(&cinfo);
            continue;
        }

        jpeg_create_decompress(&cinfo);
        jpeg_mem_src(&cinfo, payloads[i], sizes[i]);

        /* This must not segfault or corrupt memory */
        int ret = jpeg_consume_input(&cinfo);
        (void)ret;

        jpeg_destroy_decompress(&cinfo);
        /* If we reach here without crash, the invariant holds */
        ck_assert(1);
    }
}
END_TEST

Suite *security_suite(void)
{
    Suite *s = suite_create("Security");
    TCase *tc_core = tcase_create("Core");
    tcase_add_test(tc_core, test_jpeg_consume_input_adversarial);
    suite_add_tcase(s, tc_core);
    return s;
}

int main(void)
{
    int number_failed;
    Suite *s = security_suite();
    SRunner *sr = srunner_create(s);
    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}