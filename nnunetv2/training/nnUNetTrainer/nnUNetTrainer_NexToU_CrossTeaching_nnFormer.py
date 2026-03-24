import math
import torch
from torch import nn
from typing import Union, Tuple, List

from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_NexToU import nnUNetTrainer_NexToU

# 你需要按你自己的 nnFormer 实际路径修改这里
# 例如:
# from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnFormer import nnFormer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnformer_src.nnFormer_tumor import nnFormer 

# 下面这个 loss 你可以按你项目里已有的 loss 路径调整
# 目标：给 cross teaching 单独用一个“非 deep-supervision 包装”的简单 loss
# 如果你项目里没有现成单层 dice loss，可以先换成你自己的实现
from nnunetv2.training.loss.dice import SoftDiceLoss


class nnUNetTrainer_NexToU_CrossTeaching_nnFormer(nnUNetTrainer_NexToU):
    """
    设计目标:
    - main branch: NexToU
    - aux branch : nnFormer
    - supervised: 两支各自监督
    - cross teaching: 仅在最高分辨率输出上做
    - inference / validation: 只看 main branch
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        # ===== 可调超参数 =====
        self.lambda_ct_max = 0.1
        self.lambda_aux_sup = 0.5
        self.ct_warmup_epochs = 50

        self.use_confidence_mask = True
        self.confidence_threshold = 0.7

        # 注意：这要求你的 dataloader 确实把 batch 组织成
        # 前半 labeled，后半 unlabeled
        self.require_half_labeled_half_unlabeled = True

        # 单独给 CT 用的 loss，不走 deep supervision 包装
        # 这里用 SoftDiceLoss 举例；若你的项目已有更合适实现，可替换
        self.ct_loss = SoftDiceLoss(
            apply_nonlin=nn.Identity(),
            batch_dice=True,
            smooth=1e-5,
            do_bg=False,
            ddp=self.is_ddp,
        )

    def build_network_architecture(
        self,
        plans_manager,
        dataset_json,
        configuration_manager,
        num_input_channels,
        enable_deep_supervision: bool = True,
    ):
        network_main = super().build_network_architecture(
            plans_manager,
            dataset_json,
            configuration_manager,
            num_input_channels,
            enable_deep_supervision,
        )

        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        patch_size = configuration_manager.patch_size

        network_aux = nnFormer(
            crop_size=list(patch_size),
            embedding_dim=192,
            input_channels=num_input_channels,
            num_classes=num_classes,
            conv_op=nn.Conv3d,
            depths=[2, 2, 2, 2],
            num_heads=[6, 12, 24, 48],
            patch_size=[4, 4, 4],
            window_size=[4, 4, 8, 4],
            deep_supervision=enable_deep_supervision,
        )

        network = nn.ModuleDict({
            "main": network_main,
            "aux": network_aux,
        })
        return network

    def _get_highres_output(self, x):
        return x[0] if isinstance(x, (tuple, list)) else x

    def _compute_lambda_ct(self) -> float:
        """
        高斯 warm-up，和论文思路一致:
        lambda(t) = lambda_max * exp(-5 * (1 - t/T)^2)
        """
        if self.current_epoch <= 0:
            return 0.0

        T = max(self.ct_warmup_epochs, 1)
        t = min(self.current_epoch, T)
        ratio = 1.0 - (t / T)
        return self.lambda_ct_max * math.exp(-5.0 * ratio * ratio)

    def _make_pseudo_label_and_mask(self, logits: torch.Tensor):
        """
        logits: [B, C, D, H, W]
        return:
            pseudo_label: [B, 1, D, H, W]
            mask: [B, 1, D, H, W] or None
        """
        probs = torch.softmax(logits.detach(), dim=1)
        conf, pseudo = torch.max(probs, dim=1, keepdim=True)

        if self.use_confidence_mask:
            mask = (conf > self.confidence_threshold).float()
        else:
            mask = None

        return pseudo.long(), mask

    def _masked_dice_loss(self, logits: torch.Tensor, pseudo_label: torch.Tensor, mask: torch.Tensor = None):
        """
        先计算 Softmax 获取真实的概率分布，然后再用 mask 过滤，最后计算 Dice。
        """
        # pseudo_label: [B, 1, D, H, W] -> onehot [B, C, D, H, W]
        pseudo_onehot = torch.zeros_like(logits)
        pseudo_onehot.scatter_(1, pseudo_label, 1.0)

        probs = torch.softmax(logits, dim=1)

        if mask is not None:
            probs = probs * mask
            pseudo_onehot = pseudo_onehot * mask

        return self.ct_loss(probs, pseudo_onehot)

    def train_step(self, batch: dict) -> dict:
        data_l = batch["data_l"].to(self.device, non_blocking=True)
        data_u = batch["data_u"].to(self.device, non_blocking=True)
        target_l = batch["target_l"]

        if not isinstance(target_l, list):
            target_l = [target_l]
        target_l = [t.to(self.device, non_blocking=True) for t in target_l]

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(self.device.type, enabled=True):
            # 1) supervised
            out_main_l = self.network["main"](data_l)
            out_aux_l = self.network["aux"](data_l)

            loss_sup_main = self.loss(out_main_l, target_l)
            loss_sup_aux = self.loss(out_aux_l, target_l)

            # 2) unlabeled cross teaching
            out_main_u = self.network["main"](data_u)
            out_aux_u = self.network["aux"](data_u)

            logits_main_u = self._get_highres_output(out_main_u)
            logits_aux_u = self._get_highres_output(out_aux_u)

            pseudo_from_main, mask_main = self._make_pseudo_label_and_mask(logits_main_u)
            pseudo_from_aux, mask_aux = self._make_pseudo_label_and_mask(logits_aux_u)

            loss_ct_main = self._masked_dice_loss(logits_main_u, pseudo_from_aux, mask_aux)
            loss_ct_aux = self._masked_dice_loss(logits_aux_u, pseudo_from_main, mask_main)

            lambda_ct = self._compute_lambda_ct()

            total_loss = (
                loss_sup_main
                + self.lambda_aux_sup * loss_sup_aux
                + lambda_ct * (loss_ct_main + loss_ct_aux)
            )

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {
            "loss": total_loss.detach().cpu().numpy(),
            "loss_sup_main": loss_sup_main.detach().cpu().numpy(),
            "loss_sup_aux": loss_sup_aux.detach().cpu().numpy(),
            "loss_ct_main": loss_ct_main.detach().cpu().numpy(),
            "loss_ct_aux": loss_ct_aux.detach().cpu().numpy(),
            "lambda_ct": lambda_ct,
        }

    def validation_step(self, batch: dict) -> dict:
        """
        验证只用 main branch。
        最稳的做法是尽量复用父类 validation_step 的逻辑。
        如果你父类 validation_step 很依赖 self.network，
        可以临时切换到 main branch 调用父类。
        """
        original_network = self.network
        try:
            self.network = original_network["main"]
            return super().validation_step(batch)
        finally:
            self.network = original_network

    def predict_step(self, batch: dict) -> torch.Tensor:
        """推理时同样只使用 main branch"""
        original_network = self.network
        try:
            self.network = original_network["main"]
            return super().predict_step(batch)
        finally:
            self.network = original_network