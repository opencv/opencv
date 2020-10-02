To enable HarfBuzz bindings for Python among other languages, make sure
you have latest version of gobject-introspection available.  On Ubuntu,
you can install that this way:

```bash
sudo apt-get install libgirepository1.0-dev
```

And then run `meson setup` and make sure that `Introspection` is reported
enabled in output.

Compile and install.

Make sure you have the installation lib dir in `LD_LIBRARY_PATH`, as needed
for the linker to find the library.

Then make sure you also have `GI_TYPELIB_PATH` pointing to the resulting
`$prefix/lib/girepository-*` directory.

Make sure you have pygobject installed.  Then check that the following
import works in your Python interpreter:

```python
from gi.repository import HarfBuzz
```

If it does, you are ready to call HarfBuzz from Python!  Congratulations.
See [`src/sample.py`](src/sample.py).

The Python API will change.  Let us know on the mailing list if you are
using it, and send lots of feedback.
