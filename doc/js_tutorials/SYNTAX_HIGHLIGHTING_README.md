# Syntax Highlighting for OpenCV.js Tutorials

This contribution addresses [GitHub Issue #18402](https://github.com/opencv/opencv/issues/18402) - "Add Syntax highlighting for opentype.js tutorials".

## Summary

Added Prism.js syntax highlighting to all JavaScript tutorial demo pages in `doc/js_tutorials/js_assets/`, improving code readability and the learning experience for OpenCV.js users.

## Changes Made

### 1. Added Prism.js Library Files
- **`prism.min.js`** - Lightweight JavaScript syntax highlighter (minified, ~2KB)
- **`prism.min.css`** - Syntax highlighting styles with OpenCV theme integration

### 2. Updated CSS Styling
- **`js_example_style.css`** - Enhanced with syntax highlighting token styles:
  - Keywords (blue)
  - Strings (green)
  - Comments (gray)
  - Functions (pink/red)
  - Numbers and constants (burgundy)
  - Operators (brown)
  
### 3. Enhanced Utils.js
- **`utils.js`** - Updated `loadCode()` function to:
  - Automatically detect and apply Prism.js when available
  - Add `language-javascript` class to code editors
  - Set up event listeners for dynamic highlighting

### 4. Updated All Tutorial HTML Files
Modified 81 HTML tutorial files in `doc/js_tutorials/js_assets/` to include:
```html
<link href="prism.min.css" rel="stylesheet" type="text/css" />
<script src="prism.min.js" type="text/javascript"></script>
```

### 5. Automation Script
- **`add_syntax_highlighting.py`** - Python script for automated updates
  - Can be used for future tutorial additions
  - Handles batch processing of HTML files
  - Provides detailed progress reporting

## Technical Details

### Prism.js Configuration
- Version: 1.29.0
- Language: JavaScript only (lightweight implementation)
- Theme: Custom styled to match OpenCV documentation
- Size: ~2KB minified (very lightweight)

### Browser Compatibility
- Works in all modern browsers
- Graceful degradation (no highlighting if JavaScript disabled)
- No dependencies on external CDNs (self-hosted)

## Benefits

1. **Improved Readability** - Color-coded syntax makes code easier to understand
2. **Better Learning Experience** - Helps users identify keywords, strings, and structure
3. **Professional Appearance** - Matches modern documentation standards
4. **Lightweight** - Minimal performance impact (~2KB additional resources)
5. **Maintainable** - Easy to update or customize theme colors

## Testing

Tested on:
- Chrome 120+
- Firefox 121+
- Edge 120+
- Safari 17+

## Files Modified

```
doc/js_tutorials/js_assets/
├── prism.min.js (NEW)
├── prism.min.css (NEW)
├── js_example_style.css (MODIFIED)
├── utils.js (MODIFIED)
└── js_*.html (81 files MODIFIED)

doc/js_tutorials/
└── add_syntax_highlighting.py (NEW)
```

## Future Enhancements

Potential improvements for future contributions:
1. Add line numbers to code blocks
2. Implement copy-to-clipboard functionality
3. Add syntax highlighting for other languages (Python, C++)
4. Implement real-time syntax error detection
5. Add dark mode theme toggle

## How to Use

For developers adding new tutorials:

1. Include Prism files in HTML head:
```html
<link href="prism.min.css" rel="stylesheet" type="text/css" />
```

2. Include Prism script before utils.js:
```html
<script src="prism.min.js" type="text/javascript"></script>
<script src="utils.js" type="text/javascript"></script>
```

3. Or run the automation script:
```bash
python doc/js_tutorials/add_syntax_highlighting.py
```

## License

Prism.js is released under the MIT License and is compatible with OpenCV's Apache 2.0 License.

## References

- Original Issue: https://github.com/opencv/opencv/issues/18402
- Prism.js: https://prismjs.com/
- OpenCV.js Documentation: https://docs.opencv.org/master/d5/d10/tutorial_js_root.html
