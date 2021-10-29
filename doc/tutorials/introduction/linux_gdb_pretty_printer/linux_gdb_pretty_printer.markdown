Using OpenCV with gdb-powered IDEs {#tutorial_linux_gdb_pretty_printer}
=====================

@prev_tutorial{tutorial_linux_install}
@next_tutorial{tutorial_linux_gcc_cmake}

|    |    |
| -: | :- |
| Original author | Egor Smirnov |
| Compatibility | OpenCV >= 4.0 |

@tableofcontents

# Capabilities {#tutorial_linux_gdb_pretty_printer_capabilities}

This pretty-printer can show element type, `is_continuous`, `is_submatrix` flags and (possibly truncated) matrix. It is known to work in Clion, VS Code and gdb.

![Clion example](images/example.png)


# Installation {#tutorial_linux_gdb_pretty_printer_installation}

Move into `opencv/samples/gdb/`. Place `mat_pretty_printer.py` in a convinient place, rename `gdbinit` to `.gdbinit`  and move it into your home folder. Change 'source' line of `.gdbinit` to point to your `mat_pretty_printer.py` path.

In order to check version of python bundled with your gdb, use the following commands from the gdb shell:

    python
    import sys
    print(sys.version_info)
    end

If the version of python 3 installed in your system doesn't match the version in gdb, create a new virtual environment with the exact same version, install `numpy` and change the path to python3 in `.gdbinit` accordingly.


# Usage {#tutorial_linux_gdb_pretty_printer_usage}

The fields in a debugger prefixed with `view_` are pseudo-fields added for convinience, the rest are left as is.
If you feel that the number of elements in truncated view is too low, you can edit `mat_pretty_printer.py` - `np.set_printoptions` controlls everything matrix display-related.
