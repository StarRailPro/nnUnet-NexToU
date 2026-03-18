import os
import numpy as np
import nrrd  # <--- 关键：改用 pynrrd 库

# ================= 设置路径 =================
# 指向您存放 NRRD 原始标签文件的文件夹
# 注意：这里应该是您原始数据的路径，而不是 nnUNet_raw (因为那边应该放 .nii.gz)
nrrd_labels_dir = r"D:\Pycharm Workshops\nnUNet-2.2\nnUNet_Data\nnUNet_raw\Dataset501_BrainVessels\labelsTr"
# ===========================================

def get_unique_labels_from_nrrd(file_path):
    try:
        # nrrd.read 返回两个值：data 和 header
        data, header = nrrd.read(file_path)
        return np.unique(data).astype(int)
    except Exception as e:
        return f"读取错误: {e}"

print("正在分析 NRRD 文件的标签值...")

files = [f for f in os.listdir(nrrd_labels_dir) if 'nrrd' in f.lower()]

# 区分 hh 和 guys
hh_files = [f for f in files if 'hh' in f.lower()]
guys_files = [f for f in files if 'guys' in f.lower()]

print(f"发现 HH 文件: {len(hh_files)} 个")
print(f"发现 Guys 文件: {len(guys_files)} 个")

if hh_files:
    print("\n--- HH 样本采样 (NRRD) ---")
    for f in hh_files[:3]:
        path = os.path.join(nrrd_labels_dir, f)
        print(f"{f}: {get_unique_labels_from_nrrd(path)}")

if guys_files:
    print("\n--- Guys 样本采样 (NRRD) ---")
    for f in guys_files[:3]:
        path = os.path.join(nrrd_labels_dir, f)
        print(f"{f}: {get_unique_labels_from_nrrd(path)}")