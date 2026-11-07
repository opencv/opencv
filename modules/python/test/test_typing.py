import os
import subprocess
import sys
import textwrap

import cv2 as cv

from tests_common import NewOpenCVTests


class typing_test(NewOpenCVTests):
    def test_top_level_typing_not_shadowed_by_cv2_package(self):
        code = textwrap.dedent(
            """
            import importlib.util
            import pathlib
            import sys

            cv2_spec = importlib.util.find_spec("cv2")
            cv2_dir = pathlib.Path(cv2_spec.origin).parent
            sys.modules.pop("typing", None)
            sys.path.insert(0, str(cv2_dir))

            import typing

            typing_path = pathlib.Path(typing.__file__).resolve()
            cv2_typing_dir = (cv2_dir / "typing").resolve()
            if cv2_typing_dir in typing_path.parents:
                raise AssertionError(
                    "stdlib typing shadowed by {}".format(typing_path)
                )
            """
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        self.assertEqual(
            result.returncode, 0,
            "stdout:\n{}\nstderr:\n{}".format(result.stdout, result.stderr)
        )

    def test_cv2_typing_is_available(self):
        import cv2.typing

        self.assertTrue(hasattr(cv.typing, "MatLike"))


if __name__ == "__main__":
    NewOpenCVTests.bootstrap()
