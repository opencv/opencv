import os
import re

def update_include_paths(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".hpp") or file.endswith(".cpp"):  # 只处理C++文件
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    lines = f.readlines()

                updated_lines = []
                modified = False

                for line in lines:
                    # 匹配 #include "..."
                    match = re.match(r'#include\s+"(.+)"', line)
                    if match:
                        included_file = match.group(1)
                        included_file_path = os.path.join(base_dir, included_file)

                        # 检查目标文件是否存在
                        if os.path.exists(included_file_path):
                            # 计算相对路径
                            relative_path = os.path.relpath(included_file_path, root)
                            new_include = f'#include "{relative_path}"\n'
                            updated_lines.append(new_include)
                            modified = True
                        else:
                            updated_lines.append(line)
                    else:
                        updated_lines.append(line)

                # 如果有修改，写回文件
                if modified:
                    with open(file_path, "w") as f:
                        f.writelines(updated_lines)
                    print(f"Updated includes in: {file_path}")

# 使用方法
if __name__ == "__main__":
    base_directory = "./src"  # 修改为你的代码根目录
    update_include_paths(base_directory)