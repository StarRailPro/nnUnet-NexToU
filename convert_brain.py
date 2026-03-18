import os
import numpy as np
import nibabel as nib
import nrrd
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

# ================= ⚙️ 请修改这三个路径 =================
# 1. 存放原始图像的文件夹 (.nii.gz)
source_images_dir = r"D:\Pycharm Workshops\nnUNet-2.2\nnUNet_Data\nnUNet_raw\Dataset501_BrainVessels\imagesTr"

# 2. 存放原始标注的文件夹 (.nrrd)
source_labels_dir = r"D:\Pycharm Workshops\nnUNet-2.2\nnUNet_Data\nnUNet_raw\Dataset501_BrainVessels\Original_Labels"

# 3. nnU-Net 的目标输出路径
nnunet_raw = r"D:\Pycharm Workshops\nnUNet-2.2\nnUNet_Data\nnUNet_raw\labelsTr"
# ======================================================

# 任务设置
task_name = "Dataset501_BrainVessels"
target_base = join(nnunet_raw, task_name)
target_imagesTr = join(target_base, "imagesTr")
target_labelsTr = join(target_base, "labelsTr")

# 🏷️ 标签清洗规则 (根据之前分析，将不连续的标签映射为连续整数)
# 假设我们要保留 1-13 (合并 rare 的 14 到 13)
label_mapping = {
    0: 0,
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
    7: 7, 8: 8, 9: 9, 10: 10, 11: 11,
    12: 12, 13: 13,
    14: 13  # 将 14 归并到 13
}


def process_dataset():
    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTr)

    print(f"正在扫描标注文件夹: {source_labels_dir} ...")

    # 获取所有标注文件
    label_files = [f for f in os.listdir(source_labels_dir) if 'nrrd' in f.lower()]
    print(f"找到 {len(label_files)} 个标注文件，开始配对处理...")

    valid_case_count = 0
    final_labels_set = set()

    for lb_file in label_files:
        # === 1. 提取核心 ID ===
        # 文件名示例: IXI002-Guys-0828-MRA.nii.gz.nrrd
        # 逻辑: 找到 "-MRA" 之前的部分作为 ID
        if "-MRA" in lb_file:
            case_id = lb_file.split("-MRA")[0]  # 得到 IXI002-Guys-0828
        else:
            print(f"⚠️ 跳过无法识别 ID 的文件: {lb_file}")
            continue

        # === 2. 寻找对应的原始图像 ===
        # 图像文件名应该叫: ID + "-MRA.nii.gz"
        expected_img_name = f"{case_id}-MRA.nii.gz"
        src_img_path = join(source_images_dir, expected_img_name)
        src_lb_path = join(source_labels_dir, lb_file)

        if not os.path.exists(src_img_path):
            print(f"❌ 配对失败: 找不到图像 {expected_img_name} (对应标签 {lb_file})")
            continue

        # === 3. 处理图像 (Copy NIfTI) ===
        # 我们直接读取 NIfTI 图像，用来获取最正确的空间坐标 (Affine)
        try:
            img_obj = nib.load(src_img_path)
            # 保存到 nnU-Net 目录: ID_0000.nii.gz
            dst_img_path = join(target_imagesTr, f"{case_id}_0000.nii.gz")
            nib.save(img_obj, dst_img_path)
        except Exception as e:
            print(f"❌ 读取图像出错 {case_id}: {e}")
            continue

        # === 4. 处理标签 (NRRD -> NIfTI + 清洗) ===
        try:
            # 读取 NRRD 数据
            lb_data, _ = nrrd.read(src_lb_path)

            # 执行标签映射 (清洗)
            clean_data = lb_data.copy()
            unique_vals = np.unique(lb_data)

            # 只有当发现不在映射表里的值，或者需要修改时才遍历
            for old_val, new_val in label_mapping.items():
                if old_val in unique_vals and old_val != new_val:
                    clean_data[lb_data == old_val] = new_val

            # 记录最终用到的标签用于 json
            final_labels_set.update(np.unique(clean_data))

            # === 关键步骤：使用原图的 Affine 保存标签 ===
            # 这样保证标签和原图在空间上完美重合！
            lb_nifti = nib.Nifti1Image(clean_data.astype(np.uint8), img_obj.affine)

            dst_lb_path = join(target_labelsTr, f"{case_id}.nii.gz")
            nib.save(lb_nifti, dst_lb_path)

            print(f"✅ 成功处理: {case_id}")
            valid_case_count += 1

        except Exception as e:
            print(f"❌ 处理标签出错 {case_id}: {e}")

    print(f"\n全部完成！共转换 {valid_case_count} 对数据。")
    print(f"最终标签类别集合: {final_labels_set}")

    # === 5. 自动生成 dataset.json ===
    json_labels = {"background": 0}
    # 排序并排除0
    sorted_labels = sorted([int(x) for x in final_labels_set if x != 0])
    for label_val in sorted_labels:
        # 给每个类别起个名字，比如 Class_1, Class_2
        json_labels[f"Vessel_{label_val}"] = label_val

    generate_dataset_json(
        target_base,
        channel_names={"0": "MRA"},  # 模态名称
        labels=json_labels,
        file_ending=".nii.gz",
        num_training_cases=valid_case_count
    )
    print("dataset.json 已生成！")


if __name__ == "__main__":
    process_dataset()