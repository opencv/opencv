Prism.js - Lightweight, extensible syntax highlighter
=====================================================

Version: 1.29.0
Website: https://prismjs.com/
Repository: https://github.com/PrismJS/prism
License: MIT (see LICENSE file)

Files in this directory:
-------------------------
- LICENSE                - MIT License for Prism.js
- prism.js              - Core Prism.js library with JavaScript language support
- prism.css             - Default theme stylesheet for syntax highlighting
- prism-textarea.js     - Custom extension for textarea syntax highlighting
                          (Developed for OpenCV.js tutorials)

Usage:
------
These files are used to add syntax highlighting to OpenCV.js tutorial pages.
During the documentation build process (BUILD_DOCS=ON), these files are:
1. Copied to the documentation output directory
2. Minified using doc/minify_js.py
3. Included in tutorial HTML pages via Doxygen templates

The prism-textarea.js extension is specific to OpenCV's documentation needs,
as the tutorials use editable <textarea> elements rather than standard <code>
blocks. This extension creates an overlay to display syntax highlighting while
preserving textarea editability.

Updating Prism.js:
------------------
To update to a new version of Prism.js:
1. Download the latest prism.js and prism.css from https://prismjs.com/download.html
   - Select "JavaScript" language
   - Use the default theme
2. Replace prism.js and prism.css in this directory
3. Keep prism-textarea.js (OpenCV-specific, not from upstream)
4. Update this README with the new version number
5. Test the documentation build to ensure compatibility

License Compatibility:
----------------------
Prism.js is licensed under the MIT License, which is compatible with
OpenCV's Apache 2.0 License. The MIT License is more permissive and
allows use in Apache 2.0 licensed projects.

For more information, see:
- Prism.js documentation: https://prismjs.com/docs/
- MIT License: https://opensource.org/licenses/MIT
- Apache 2.0 License: https://www.apache.org/licenses/LICENSE-2.0
