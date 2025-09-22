import torch
import torch.nn as nn
import torch.nn.functional as F


class RNKCLoss(nn.Module):
    def __init__(self, gamma_base: float = 1.0):
        super(RNKCLoss, self).__init__()
        self.gamma_base = gamma_base
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, student_logits, teacher_logits, ignore_mask):
        student_logits_flat = student_logits.permute(0, 2, 3, 1).reshape(
            -1, student_logits.shape[1]
        )
        teacher_logits_flat = teacher_logits.permute(0, 2, 3, 1).reshape(
            -1, teacher_logits.shape[1]
        )
        mask_flat = ignore_mask.view(-1).bool()

        if not mask_flat.any():
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

        student_logits_valid = student_logits_flat[mask_flat]
        teacher_logits_valid = teacher_logits_flat[mask_flat]

        with torch.no_grad():
            _, pseudo_labels_valid = torch.max(teacher_logits_valid, dim=1)

        student_probs_valid = self.softmax(student_logits_valid)
        teacher_probs_valid = self.softmax(teacher_logits_valid)
        student_log_probs_valid = self.log_softmax(student_logits_valid)

        one_hot_labels_valid = F.one_hot(
            pseudo_labels_valid, num_classes=teacher_logits.shape[1]
        ).float()
        non_target_mask = 1.0 - one_hot_labels_valid

        teacher_non_target_probs = teacher_probs_valid * non_target_mask
        student_non_target_probs = student_probs_valid * non_target_mask

        sum_teacher_non_target = torch.sum(
            teacher_non_target_probs, dim=1, keepdim=True
        )
        norm_teacher_non_target = teacher_non_target_probs / (
            sum_teacher_non_target + 1e-8
        )

        sum_student_non_target = torch.sum(
            student_non_target_probs, dim=1, keepdim=True
        )
        norm_student_non_target = student_non_target_probs / (
            sum_student_non_target + 1e-8
        )

        log_norm_student_non_target = torch.log(norm_student_non_target + 1e-8)

        l_non_target_ce_per_pixel = -torch.sum(
            norm_teacher_non_target * log_norm_student_non_target, dim=1
        )

        entropy_per_pixel = -torch.sum(
            student_probs_valid * student_log_probs_valid, dim=1
        )

        # entropy_normalized = (entropy_per_pixel - torch.min(entropy_per_pixel)) / (
        #     torch.max(entropy_per_pixel) - torch.min(entropy_per_pixel) + 1e-8
        # )

        dynamic_gamma = self.gamma_base * entropy_per_pixel
        l_non_target = torch.mean(dynamic_gamma * l_non_target_ce_per_pixel)

        return l_non_target


class RNKCLoss3D(nn.Module):
    def __init__(self, gamma_base: float = 1.0):
        super(RNKCLoss3D, self).__init__()
        self.gamma_base = gamma_base
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, student_logits, teacher_logits, ignore_mask):
        student_logits_flat = student_logits.permute(0, 2, 3, 4, 1).reshape(
            -1, student_logits.shape[1]
        )
        teacher_logits_flat = teacher_logits.permute(0, 2, 3, 4, 1).reshape(
            -1, teacher_logits.shape[1]
        )
        mask_flat = ignore_mask.view(-1).bool()

        if not mask_flat.any():
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

        student_logits_valid = student_logits_flat[mask_flat]
        teacher_logits_valid = teacher_logits_flat[mask_flat]

        with torch.no_grad():
            _, pseudo_labels_valid = torch.max(teacher_logits_valid, dim=1)

        student_probs_valid = self.softmax(student_logits_valid)
        teacher_probs_valid = self.softmax(teacher_logits_valid)
        student_log_probs_valid = self.log_softmax(student_logits_valid)

        one_hot_labels_valid = F.one_hot(
            pseudo_labels_valid, num_classes=teacher_logits.shape[1]
        ).float()
        non_target_mask = 1.0 - one_hot_labels_valid

        teacher_non_target_probs = teacher_probs_valid * non_target_mask
        student_non_target_probs = student_probs_valid * non_target_mask

        sum_teacher_non_target = torch.sum(
            teacher_non_target_probs, dim=1, keepdim=True
        )
        norm_teacher_non_target = teacher_non_target_probs / (
            sum_teacher_non_target + 1e-8
        )

        sum_student_non_target = torch.sum(
            student_non_target_probs, dim=1, keepdim=True
        )
        norm_student_non_target = student_non_target_probs / (
            sum_student_non_target + 1e-8
        )

        log_norm_student_non_target = torch.log(norm_student_non_target + 1e-8)

        l_non_target_ce_per_pixel = -torch.sum(
            norm_teacher_non_target * log_norm_student_non_target, dim=1
        )

        entropy_per_pixel = -torch.sum(
            student_probs_valid * student_log_probs_valid, dim=1
        )

        entropy_normalized = (entropy_per_pixel - torch.min(entropy_per_pixel)) / (
            torch.max(entropy_per_pixel) - torch.min(entropy_per_pixel) + 1e-8
        )

        dynamic_gamma = self.gamma_base * entropy_normalized
        l_non_target = torch.mean(dynamic_gamma * l_non_target_ce_per_pixel)

        return l_non_target


# # --- 示例用法 ---
# batch_size = 4
# num_classes = 2
# H, W, D = 96, 96, 96
# gamma_base_value = 10.0  # 这里的基础权重可能需要调整得更大一些

# student_logits = torch.randn(batch_size, num_classes, H, W, D, requires_grad=True)
# teacher_logits = torch.randn(batch_size, num_classes, H, W, D, requires_grad=False)
# ground_truth_labels = torch.randint(0, num_classes, (batch_size, H, W, D))

# combined_criterion = RNKCLoss3D(gamma_base=gamma_base_value)

# loss = combined_criterion(student_logits, teacher_logits, ground_truth_labels)

# print(f"动态熵加权 NKD 损失: {loss.item()}")
