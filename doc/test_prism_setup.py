#!/usr/bin/env python3
"""
Manual testing script for Prism.js textarea highlighting.
Run this to verify the minification and setup is working.
"""

import os
import sys
import subprocess

def test_minification():
    """Test that minification script works."""
    print("=" * 60)
    print("Testing JavaScript Minification")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    minify_script = os.path.join(script_dir, "minify_js.py")
    test_js = os.path.join(script_dir, "js_tutorials", "js_assets", "prism-textarea.js")
    output_js = os.path.join(script_dir, "test_minified.js")
    
    if not os.path.exists(test_js):
        print(f"ERROR: Source file not found: {test_js}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, minify_script, test_js, output_js],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"ERROR: Minification failed")
            print(result.stderr)
            return False
        
        print(result.stdout)
        
        if os.path.exists(output_js):
            original_size = os.path.getsize(test_js)
            minified_size = os.path.getsize(output_js)
            print(f"✓ Original: {original_size} bytes")
            print(f"✓ Minified: {minified_size} bytes")
            print(f"✓ Reduction: {100 - (minified_size * 100 / original_size):.1f}%")
            os.remove(output_js)
            return True
        else:
            print("ERROR: Output file not created")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def check_files_exist():
    """Check that all required files exist."""
    print("\n" + "=" * 60)
    print("Checking Required Files")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    files_to_check = [
        ("minify_js.py", "Minification script"),
        ("js_tutorials/js_assets/prism-textarea.js", "Textarea highlighter"),
        ("js_tutorials/js_assets/prism.min.js", "Prism.js library"),
        ("js_tutorials/js_assets/prism.min.css", "Prism.js styles"),
        ("js_tutorials/js_assets/js_example_style.css", "Tutorial styles"),
        ("js_tutorials/js_assets/test_prism_textarea.html", "Test page"),
    ]
    
    all_exist = True
    for file_path, description in files_to_check:
        full_path = os.path.join(script_dir, file_path)
        if os.path.exists(full_path):
            size = os.path.getsize(full_path)
            print(f"✓ {description}: {size} bytes")
        else:
            print(f"✗ {description}: NOT FOUND")
            all_exist = False
    
    return all_exist

def print_next_steps():
    """Print next steps for user."""
    print("\n" + "=" * 60)
    print("Next Steps for Testing")
    print("=" * 60)
    print("""
1. Open test_prism_textarea.html in your browser:
   e:\\contribution\\opencv\\opencv\\doc\\js_tutorials\\js_assets\\test_prism_textarea.html

2. Verify you see:
   - Colorful syntax highlighting in textareas
   - Blue keywords (let, const, function)
   - Green strings ('test')
   - Pink/red function names
   - Gray comments

3. Test interactivity:
   - Click in textarea (see cursor)
   - Type code (see real-time highlighting)
   - Scroll (see synchronized scrolling)

4. Test in browsers:
   - Chrome
   - Firefox
   - Edge

5. Take screenshots showing it works

6. If all tests pass, reply to the contributor with proof!
""")

def main():
    print("\n" + "#" * 60)
    print("# OpenCV Prism.js Integration Test")
    print("#" * 60 + "\n")
    
    files_ok = check_files_exist()
    minify_ok = test_minification()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    if files_ok and minify_ok:
        print("✓ All automated tests passed!")
        print_next_steps()
        return 0
    else:
        print("✗ Some tests failed")
        if not files_ok:
            print("  - Missing required files")
        if not minify_ok:
            print("  - Minification test failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
