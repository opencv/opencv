#!/usr/bin/env python3
"""
Script to add Prism.js syntax highlighting to OpenCV.js tutorial HTML files.
This addresses GitHub issue #18402 - "Add Syntax highlighting for opentype.js tutorials"
"""

import os
import re
from pathlib import Path

def add_syntax_highlighting_to_html(file_path):
    """Add Prism.js CSS and JS to an HTML file if not already present."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    modified = False
    
    # Check if Prism CSS is already included
    if 'prism.min.css' not in content:
        # Add Prism CSS link after js_example_style.css
        content = re.sub(
            r'(<link href="js_example_style\.css"[^>]*>)',
            r'\1\n<link href="prism.min.css" rel="stylesheet" type="text/css" />',
            content
        )
        modified = True
        print(f"  ✓ Added Prism CSS link")
    
    # Check if Prism JS is already included
    if 'prism.min.js' not in content:
        # Add Prism JS before utils.js
        content = re.sub(
            r'(<script src="utils\.js")',
            r'<script src="prism.min.js" type="text/javascript"></script>\n\1',
            content
        )
        modified = True
        print(f"  ✓ Added Prism JS script")
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False

def main():
    """Process all HTML tutorial files in js_assets directory."""
    
    script_dir = Path(__file__).parent
    assets_dir = script_dir / 'js_assets'
    
    if not assets_dir.exists():
        print(f"❌ Directory not found: {assets_dir}")
        return
    
    print(f"🔍 Searching for HTML files in: {assets_dir}")
    print(f"{'='*60}")
    
    html_files = list(assets_dir.glob('js_*.html'))
    
    if not html_files:
        print("❌ No HTML tutorial files found!")
        return
    
    print(f"📝 Found {len(html_files)} tutorial files\n")
    
    modified_count = 0
    skipped_count = 0
    
    for html_file in sorted(html_files):
        print(f"Processing: {html_file.name}")
        
        if add_syntax_highlighting_to_html(html_file):
            modified_count += 1
        else:
            print(f"  ⊘ Already has syntax highlighting")
            skipped_count += 1
        print()
    
    print(f"{'='*60}")
    print(f"✅ Summary:")
    print(f"   Modified: {modified_count} files")
    print(f"   Skipped:  {skipped_count} files")
    print(f"   Total:    {len(html_files)} files")
    print(f"\n🎉 Syntax highlighting setup complete!")

if __name__ == '__main__':
    main()
