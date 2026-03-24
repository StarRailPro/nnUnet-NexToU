import numpy as np
from typing import Dict, List, Tuple

from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset


class nnUNetDataLoader3D_PairedSemiSupervised(nnUNetDataLoaderBase):
    """
    配对式半监督 dataloader

    命名规则假设：
    - labeled case key:   IXI002-Guys-0828
    - unlabeled case key: IXI002-Guys-0828-MRA

    返回：
    {
        "data_l":   [B, C, D, H, W],
        "target_l": [B, Cseg, D, H, W],
        "data_u":   [B, C, D, H, W],
        "properties": [...],
        "keys_l": [...],
        "keys_u": [...]
    }
    """

    def __init__(
        self,
        data: nnUNetDataset,
        batch_size: int,
        patch_size,
        final_patch_size,
        oversample_foreground_percent: float,
        sampling_probabilities=None,
        pad_sides=None,
        probabilistic_oversampling: bool = False,
        unlabeled_suffix: str = "-MRA",
    ):
        super().__init__(
            data,
            batch_size,
            patch_size,
            final_patch_size,
            oversample_foreground_percent,
            sampling_probabilities,
            pad_sides,
            probabilistic_oversampling,
        )
        self.unlabeled_suffix = unlabeled_suffix

        # 只从“有标签 key”里采样
        self.labeled_keys = [k for k in self._data.keys() if not k.endswith(self.unlabeled_suffix)]
        self.labeled_keys.sort()

        # 简单检查：每个 labeled key 是否都有对应 unlabeled key
        missing = [k for k in self.labeled_keys if (k + self.unlabeled_suffix) not in self._data.keys()]
        if len(missing) > 0:
            raise RuntimeError(
                f"以下 labeled cases 没有找到对应的 unlabeled case (suffix={self.unlabeled_suffix}): "
                f"{missing[:10]}{' ...' if len(missing) > 10 else ''}"
            )

    def get_indices(self):
        """
        重载采样逻辑：只从 labeled_keys 中抽样
        """
        if self.sampling_probabilities is None:
            return np.random.choice(self.labeled_keys, self.batch_size, replace=True)
        else:
            # 如果你后面真要配 sampling_probabilities，需要保证它和 labeled_keys 对齐
            probs = np.array(self.sampling_probabilities, dtype=float)
            probs = probs / probs.sum()
            return np.random.choice(self.labeled_keys, self.batch_size, replace=True, p=probs)

    def _get_unlabeled_key(self, labeled_key: str) -> str:
        return labeled_key + self.unlabeled_suffix

    def _crop_and_pad(
        self,
        data: np.ndarray,
        seg: np.ndarray,
        bbox_lbs: List[int],
        bbox_ubs: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        复用原 nnUNetDataLoader3D 的 crop/pad 思路。:contentReference[oaicite:4]{index=4}
        """
        shape = data.shape[1:]
        dim = len(shape)

        valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
        valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

        data_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
        seg_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])

        data = data[data_slice]
        seg = seg[seg_slice]

        padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]

        data = np.pad(data, ((0, 0), *padding), "constant", constant_values=0)
        seg = np.pad(seg, ((0, 0), *padding), "constant", constant_values=-1)
        return data, seg

    def generate_train_batch(self):
        selected_keys = self.get_indices()

        # 预分配
        data_l_all = np.zeros(self.data_shape, dtype=np.float32)
        data_u_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_l_all = np.zeros(self.seg_shape, dtype=np.int16)

        case_properties = []
        selected_u_keys = []

        for j, key_l in enumerate(selected_keys):
            key_u = self._get_unlabeled_key(key_l)
            selected_u_keys.append(key_u)

            # oversampling foreground 仍然基于 labeled 分支
            force_fg = self.get_do_oversample(j)

            # 读取 labeled
            data_l, seg_l, properties_l = self._data.load_case(key_l)

            # 读取 unlabeled
            # 这里复用 load_case：要求 paired unlabeled 的 npz 里也有 seg 字段。
            # 如果没有 seg，后面我再给你一个 dataset 的最小补丁。
            data_u, _, properties_u = self._data.load_case(key_u)

            case_properties.append(
                {
                    "labeled": properties_l,
                    "unlabeled": properties_u,
                    "key_l": key_l,
                    "key_u": key_u,
                }
            )

            # bbox 只根据 labeled 的 properties 决定，这样 seg 监督更稳定
            shape = data_l.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties_l["class_locations"])

            # labeled: 正常 crop/pad
            data_l_patch, seg_l_patch = self._crop_and_pad(data_l, seg_l, bbox_lbs, bbox_ubs)

            # unlabeled: 用同一组 bbox
            # 给一个假的 seg 占位，保证复用相同 crop/pad 代码
            dummy_seg_u = np.zeros((1, *data_u.shape[1:]), dtype=np.int16)
            data_u_patch, _ = self._crop_and_pad(data_u, dummy_seg_u, bbox_lbs, bbox_ubs)

            data_l_all[j] = data_l_patch
            seg_l_all[j] = seg_l_patch
            data_u_all[j] = data_u_patch

        return {
            "data_l": data_l_all,
            "target_l": seg_l_all,
            "data_u": data_u_all,
            "properties": case_properties,
            "keys_l": selected_keys,
            "keys_u": selected_u_keys,
        }