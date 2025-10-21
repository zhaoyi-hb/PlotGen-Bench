import os
import subprocess
import argparse

# 创建参数解析器
parser = argparse.ArgumentParser(description="批量渲染指定文件夹及子文件夹下的所有 .py 文件")
parser.add_argument(
    "-f", "--folder", 
    type=str, 
    required=True,
    help="目标文件夹路径"
)
args = parser.parse_args()
folder_path = args.folder

# 检查文件夹是否存在
if not os.path.isdir(folder_path):
    print(f"错误：文件夹 {folder_path} 不存在")
    exit(1)

# 遍历文件夹及子文件夹
py_files = []
for root, _, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".py"):
            py_files.append(os.path.join(root, file))

print(f"找到 {len(py_files)} 个 .py 文件，开始渲染...")

# 调用 render_static3.py 渲染
success_count = 0
fail_count = 0
for py_file in py_files:
    print(f"渲染 {py_file} ...")
    try:
        result = subprocess.run(
            ["python", "render_static3.py", "--file", py_file],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        success_count += 1
    except subprocess.CalledProcessError as e:
        print(f"渲染失败: {py_file}")
        print(e.stdout)
        print(e.stderr)
        fail_count += 1

print(f"\n渲染完成，总数: {len(py_files)}, 成功: {success_count}, 失败: {fail_count}")
