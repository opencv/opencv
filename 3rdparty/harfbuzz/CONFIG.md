# Configuring HarfBuzz

Most of the time you will not need any custom configuration.  The configuration
options provided by `meson` should be enough.  In particular, if you just want
HarfBuzz library plus hb-shape / hb-view utilities, make sure FreeType and Cairo
are available and found during configuration.

If you are building for distribution, you should more carefully consider whether
you need Glib, ICU, Graphite2, as well as CoreText / Uniscribe / DWrite.  Make
sure the relevant ones are enabled.

If you are building for custom environment (embedded, downloadable app, etc)
where you mostly just want to call `hb_shape()` and the binary size of the
resulting library is very important to you, the rest of this file guides you
through your options to disable features you may not need, in exchange for
binary size savings.

## Compiler Options

Make sure you build with your compiler's "optimize for size" option.  On `gcc`
this is `-Os`, and can be enabled by passing `CXXFLAGS=-Os`.  On clang there
is an even more extreme flag, `-Oz`.  Meson also provides `--buildtype=minsize`
for more convenience.

HarfBuzz heavily uses inline functions and the optimize-size flag can make the
library smaller by 20% or more.  Moreover, sometimes, based on the target CPU,
the optimize-size builds perform *faster* as well, thanks to lower code
footprint and caching effects.  So, definitely try that even if size is not
extremely tight but you have a huge application.  For example, Chrome does
that.  Note that this configuration also automatically enables certain internal
optimizations.  Search for `HB_OPTIMIZE_SIZE` for details, if you are using
other compilers, or continue reading.

Another compiler option to consider is "link-time optimization", also known as
'lto'.  To enable that, feel free to use `-Db_lto=true` of meson.
This, also, can have a huge impact on the final size, 20% or more.

Finally, if you are making a static library build or otherwise linking the
library into your app, make sure your linker removes unused functions.  This
can be tricky and differ from environment to environment, but you definitely
want to make sure this happens.  Otherwise, every unused public function will
be adding unneeded bytes to your binary.  The following pointers might come
handy:

 * https://lwn.net/Articles/741494/ (all of the four-part series)
 * https://elinux.org/images/2/2d/ELC2010-gc-sections_Denys_Vlasenko.pdf

Combining the above three build options should already shrink your library a lot.
The rest of this file shows you ways to shrink the library even further at the
expense of removing functionality (that may not be needed).  The remaining
options are all enabled by defining pre-processor macros, which can be done
via `CXXFLAGS` or `CPPFLAGS` similarly.


## Unicode-functions

Access to Unicode data can be configured at compile time as well as run-time.
By default, HarfBuzz ships with its own compact subset of properties from
Unicode Character Database that it needs.  This is a highly-optimized
implementation that depending on compile settings (optimize-size or not)
takes around ~40kb or ~60kb.  Using this implementation (default) is highly
recommended, as HarfBuzz always ships with data from latest version of Unicode.
This implementation can be disabled by defining `HB_NO_UCD`.

For example, if you are enabling ICU as a built-in option, or GLib, those
can provide Unicode data as well, so defining `HB_NO_UCD` might save you
space without reducing functionality (to the extent that the Unicode version
of those implementations is recent.)

If, however, you provide your own Unicode data to HarfBuzz at run-time by
calling `hb_buffer_set_unicode_funcs` on every buffer you create, and you do
not rely on `hb_unicode_funcs_get_default()` results, you can disable the
internal implementation by defining both `HB_NO_UCD` and `HB_NO_UNICODE_FUNCS`.
The latter is needed to guard against accidentally building a library without
any default Unicode implementations.


## Font-functions

Access to certain font functionalities can also be configured at run-time.  By
default, HarfBuzz uses an efficient internal implementation of OpenType
functionality for this.  This internal implementation is called `hb-ot-font`.
All newly-created `hb_font_t` objects by default use `hb-ot-font`.  Using this
is highly recommended, and is what fonts use by default when they are created.

Most embedded uses will probably use HarfBuzz with FreeType using `hb-ft.h`.
In that case, or if you otherwise provide those functions by calling
`hb_font_set_funcs()` on every font you create, you can disable `hb-ot-font`
without loss of functionality by defining `HB_NO_OT_FONT`.


## Shapers

Most HarfBuzz clients use it for the main shaper, called "ot".  However, it
is legitimate to want to compile HarfBuzz with only another backend, eg.
CoreText, for example for an iOS app.  For that, you want `HB_NO_OT_SHAPE`.
If you are going down that route, check if you want `HB_NO_OT`.

This is very rarely what you need.  Make sure you understand exactly what you
are doing.

Defining `HB_NO_FALLBACK_SHAPE` however is pretty harmless.  That removes the
(unused) "fallback" shaper.


## Thread-safety

By default HarfBuzz builds as a thread-safe library.  The exception is that
the `HB_TINY` predefined configuring (more below) disables thread-safety.

If you do /not/ need thread-safety in the library (eg. you always call into
HarfBuzz from the same thread), you can disable thread-safety by defining
`HB_NO_MT`.  As noted already, this is enabled by `HB_TINY`.


## Pre-defined configurations

The [`hb-config.hh`](src/hb-config.hh) internal header supports three
pre-defined configurations as well grouping of various configuration options.
The pre-defined configurations are:

  * `HB_MINI`: Disables shaping of AAT as well as legacy fonts.  Ie. it produces
    a capable OpenType shaper only.

  * `HB_LEAN`: Disables various non-shaping functionality in the library, as well
    as esoteric or rarely-used shaping features.  See the definition for details.

  * `HB_TINY`: Enables both `HB_MINI` and `HB_LEAN` configurations, as well as
    disabling thread-safety and debugging, and use even more size-optimized data
    tables.


## Tailoring configuration

Most of the time, one of the pre-defined configuration is exactly what one needs.
Sometimes, however, the pre-defined configuration cuts out features that might
be desired in the library.  Unfortunately there is no quick way to undo those
configurations from the command-line.  But one can add a header file called
`config-override.h` to undefine certain `HB_NO_*` symbols as desired.  Then
define `HAVE_CONFIG_OVERRIDE_H` to make `hb-config.hh` include your configuration
overrides at the end.


## Notes

Note that the config option `HB_NO_CFF`, which is enabled by `HB_LEAN` and
`HB_TINY` does /not/ mean that the resulting library won't work with CFF fonts.
The library can shape valid CFF fonts just fine, with or without this option.
This option disables (among other things) the code to calculate glyph exntents
for CFF fonts.
