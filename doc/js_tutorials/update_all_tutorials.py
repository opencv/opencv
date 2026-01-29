import os
import re

# Directory containing the tutorial HTML files
js_assets_dir = r"e:\contribution\opencv\opencv\doc\js_tutorials\js_assets"

# Pattern to find the prism.min.js script tag
pattern = r'<script src="prism\.min\.js" type="text/javascript"></script>'

# Replacement: add prism-textarea.min.js right after prism.min.js
replacement = '''<script src="prism.min.js" type="text/javascript"></script>
<script src="prism-textarea.min.js" type="text/javascript"></script>'''

updated_count = 0
already_updated = 0
files_processed = 0

# Process all HTML files
for filename in os.listdir(js_assets_dir):
    if filename.endswith('.html'):
        filepath = os.path.join(js_assets_dir, filename)
        files_processed += 1
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if already has prism-textarea.min.js
        if 'prism-textarea.min.js' in content:
            already_updated += 1
            continue
        
        # Check if has prism.min.js
        if 'prism.min.js' in content:
            # Add prism-textarea.min.js after prism.min.js
            new_content = re.sub(pattern, replacement, content)
            
            if new_content != content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                updated_count += 1
                print(f"✓ Updated: {filename}")

print(f"\n{'='*60}")
print(f"Total files processed: {files_processed}")
print(f"Files updated: {updated_count}")
print(f"Files already updated: {already_updated}")
print(f"Files without prism.js: {files_processed - updated_count - already_updated}")
print(f"{'='*60}")
