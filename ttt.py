import os

def count_lines_of_code(directory):
    total_lines = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    total_lines += sum(1 for line in f)
    return total_lines

current_directory = '.'  # 当前目录
lines = count_lines_of_code(current_directory)
print(f"当前目录中所有Python代码文件共有 {lines} 行代码。")
