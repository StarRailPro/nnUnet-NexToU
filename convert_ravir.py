import os
import shutil
import numpy as np
import SimpleITK as sitk
import json
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

# ================= 修改这里 =================
# 您存放原始 RAVIR 数据的路径 (注意 Windows 路径要用双斜杠 \\ 或反斜杠 /)
ravir_source_folder = "D:\Pycharm Workshops\\nnUNet-2.2\\nnUNet_Data\RAVIR_Source"
# ===========================================

# nnU-Net 识别的目标路径 (根据环境变量自动获取)
nnunet_raw = os.environ.get('nnUNet_raw')
if nnunet_raw is None:
    raise RuntimeError("请先在 Terminal 中设置 nnUNet_raw 环境变量！")

task_name = "Dataset810_RAVIR"
target_base = join(nnunet_raw, task_name)
target_imagesTr = join(target_base, "imagesTr")
target_labelsTr = join(target_base, "labelsTr")

maybe_mkdir_p(target_imagesTr)
maybe_mkdir_p(target_labelsTr)


def convert_to_nifti(src_path, dest_path, is_label=False):
    # 读取 PNG
    img = sitk.ReadImage(src_path)
    img_arr = sitk.GetArrayFromImage(img)

    if is_label:
        # RAVIR 标签处理: Artery=128 -> 1, Vein=256 -> 2 (可能有微小偏差，做区间映射)
        new_label = np.zeros_like(img_arr, dtype=np.uint8)
        # 128左右是动脉，设为 1
        new_label[(img_arr >= 100) & (img_arr < 200)] = 1
        # 255/256左右是静脉，设为 2
        new_label[img_arr >= 200] = 2
        out_img = sitk.GetImageFromArray(new_label)
    else:
        out_img = img  # 图片直接存

    out_img.SetSpacing((1.0, 1.0, 1.0))  # 2D 图像通常设个虚拟 spacing
    sitk.WriteImage(out_img, dest_path)


print("正在转换训练集数据...")
train_img_dir = join(ravir_source_folder, "train", "training_images")
train_mask_dir = join(ravir_source_folder, "train", "training_masks")

# 获取所有图片文件名
files = [f for f in os.listdir(train_img_dir) if f.endswith('.png')]

for f in files:
    case_id = f.replace(".png", "")  # 比如 IR_Case_001

    # 转换图片: 名字必须变成 xxxxx_0000.nii.gz
    src_img = join(train_img_dir, f)
    dst_img = join(target_imagesTr, f"{case_id}_0000.nii.gz")
    convert_to_nifti(src_img, dst_img, is_label=False)

    # 转换标签: 名字必须变成 xxxxx.nii.gz (没有 _0000)
    src_mask = join(train_mask_dir, f)  # 假设掩码文件名和图片一样
    dst_mask = join(target_labelsTr, f"{case_id}.nii.gz")
    convert_to_nifti(src_mask, dst_mask, is_label=True)
    print(f"Processed {case_id}")

print("生成 dataset.json ...")
json_dict = {
    "channel_names": {"0": "IR_Image"},
    "labels": {"background": 0, "Artery": 1, "Vein": 2},
    "numTraining": len(files),
    "file_ending": ".nii.gz"
}
with open(join(target_base, "dataset.json"), 'w') as jf:
    json.dump(json_dict, jf, indent=4)

print("完成！数据已就绪。")