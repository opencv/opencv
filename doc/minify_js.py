#!/usr/bin/env python3
"""
Simple JavaScript minifier for OpenCV documentation build.
Removes comments and excess whitespace to reduce file size.
"""

import sys
import re

def minify_js(content):
    """Minify JavaScript by removing comments and whitespace."""
    # Remove single-line comments (but preserve URLs)
    content = re.sub(r'(?<!:)//[^\n]*', '', content)
    
    # Remove multi-line comments
    content = re.sub(r'/\*[\s\S]*?\*/', '', content)
    
    # Remove leading/trailing whitespace from lines
    lines = [line.strip() for line in content.split('\n')]
    
    # Remove empty lines
    lines = [line for line in lines if line]
    
    # Join with spaces
    result = ' '.join(lines)
    
    # Reduce multiple spaces to single space
    result = re.sub(r'\s+', ' ', result)
    
    # Remove spaces around operators and punctuation
    result = re.sub(r'\s*([{}();,=+\-*/<>!&|:])\s*', r'\1', result)
    
    return result.strip()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: minify_js.py input.js output.min.js')
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        minified = minify_js(content)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(minified)
        
        original_size = len(content)
        minified_size = len(minified)
        reduction = 100 - (minified_size * 100 / original_size)
        
        print(f'Minified {input_file} -> {output_file}')
        print(f'Size: {original_size} -> {minified_size} bytes ({reduction:.1f}% reduction)')
        
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)
