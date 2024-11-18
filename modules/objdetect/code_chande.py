import os

# 要添加的内容
header = """// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

"""

def add_header_to_file(file_path, header):
    """在文件顶部添加指定的头部内容"""
    try:
        with open(file_path, 'r+', encoding='utf-8') as file:
            content = file.read()
            # 重置文件指针到文件开头
            file.seek(0, 0)
            # 写入头部和原内容
            file.write(header + content)
        print(f"Header added to: {file_path}")
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

def process_directory(directory, extensions):
    """处理指定目录中的所有文件"""
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                add_header_to_file(file_path, header)

def main():
    directory = input("Enter the directory path to process: ").strip()
    if os.path.exists(directory):
        extensions = ['.cpp', '.hpp', '.h', '.c']  # 指定需要处理的文件类型
        process_directory(directory, extensions)
    else:
        print("The directory does not exist.")

if __name__ == "__main__":
    main()