import torch
import torch.nn as nn
from torchvision.models import resnet34, resnet50, resnet101
from efficientnet_pytorch import EfficientNet
from torchvision import models

class OutputLayer(nn.Module):
    def __init__(self, fc, num_extra):
        super(OutputLayer, self).__init__()
        self.regular_outputs_layer = fc
        self.num_extra = num_extra
        if num_extra > 0:
            self.extra_outputs_layer = nn.Linear(fc.in_features, num_extra)

    def forward(self, x):
        regular_outputs = self.regular_outputs_layer(x)
        if self.num_extra > 0:
            extra_outputs = self.extra_outputs_layer(x)
        else:
            extra_outputs = None
        return regular_outputs, extra_outputs


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, value, key, query, mask=None):
        attention_output, attention_weights = self.attention(query, key, value, attn_mask=mask)
        output = self.norm(attention_output + query)  # Add residual connection
        return output, attention_weights



class PolyRegression(nn.Module):
    def __init__(self,
                 num_outputs,
                 backbone,
                 pretrained,
                 curriculum_steps=None,
                 extra_outputs=0,
                 share_top_y=True,
                 pred_category=False,
                 attention_heads=5):
                #  Điều chỉnh attention_heads sao cho chia hết cho num_outputs
        super(PolyRegression, self).__init__()

        # Kiểm tra và điều chỉnh attention_heads sao cho num_outputs chia hết cho attention_heads num_outputs: 35
        if num_outputs % attention_heads != 0:
            raise ValueError(f"embed_size ({num_outputs}) must be divisible by num_heads ({attention_heads})")

        # Khởi tạo mô hình backbone
        if 'efficientnet' in backbone:
            if pretrained:
                self.model = EfficientNet.from_pretrained(backbone, num_classes=num_outputs)
            else:
                self.model = EfficientNet.from_name(backbone, override_params={'num_classes': num_outputs})
            self.model._fc = OutputLayer(self.model._fc, extra_outputs)
        elif backbone == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_outputs)
            self.model.fc = OutputLayer(self.model.fc, extra_outputs)
        elif backbone == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_outputs)
            self.model.fc = OutputLayer(self.model.fc, extra_outputs)
        elif backbone == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_outputs)
            self.model.fc = OutputLayer(self.model.fc, extra_outputs)
        elif backbone == 'mobilenet_v2':
            self.model = models.mobilenet_v2(pretrained=pretrained)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_outputs)
            self.model.classifier[1] = OutputLayer(self.model.classifier[1], extra_outputs)
        elif backbone == 'mobilenet_v3_small':
            self.model = models.mobilenet_v3_small(pretrained=pretrained)
            self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_outputs)
            self.model.classifier[3] = OutputLayer(self.model.classifier[3], extra_outputs)
        elif backbone == 'mobilenet_v3_large':
            self.model = models.mobilenet_v3_large(pretrained=pretrained)
            self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_outputs)
            self.model.classifier[3] = OutputLayer(self.model.classifier[3], extra_outputs)
        elif backbone == 'squeezenet1_0':
            self.model = models.squeezenet1_0(pretrained=pretrained)
            self.model.classifier[1] = nn.Conv2d(512, num_outputs, kernel_size=(1, 1), stride=(1, 1))
            self.model.classifier[1] = OutputLayer(self.model.classifier[1], extra_outputs)
        elif backbone == 'squeezenet1_1':
            self.model = models.squeezenet1_1(pretrained=pretrained)
            self.model.classifier[1] = nn.Conv2d(512, num_outputs, kernel_size=(1, 1), stride=(1, 1))
            self.model.classifier[1] = OutputLayer(self.model.classifier[1], extra_outputs)
        elif backbone == 'shufflenet_v2_x0_5':
            self.model = models.shufflenet_v2_x0_5(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_outputs)
            self.model.fc = OutputLayer(self.model.fc, extra_outputs)
        elif backbone == 'shufflenet_v2_x1_0':
            self.model = models.shufflenet_v2_x1_0(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_outputs)
            self.model.fc = OutputLayer(self.model.fc, extra_outputs)
        elif backbone == 'shufflenet_v2_x1_5':
            self.model = models.shufflenet_v2_x1_5(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_outputs)
            self.model.fc = OutputLayer(self.model.fc, extra_outputs)
        elif backbone == 'shufflenet_v2_x2_0':
            self.model = models.shufflenet_v2_x2_0(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_outputs)
            self.model.fc = OutputLayer(self.model.fc, extra_outputs)
        else:
            raise NotImplementedError()

        # Cài đặt SelfAttention
        self.attention = SelfAttention(embed_size=num_outputs, heads=attention_heads)

        # Các tham số khác
        self.curriculum_steps = [0, 0, 0, 0] if curriculum_steps is None else curriculum_steps
        self.share_top_y = share_top_y
        self.extra_outputs = extra_outputs
        self.pred_category = pred_category
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, epoch=None, **kwargs):
        output, extra_outputs = self.model(x, **kwargs)

        # Áp dụng Self-Attention
        output, _ = self.attention(output, output, output)

        for i in range(len(self.curriculum_steps)):
            if epoch is not None and epoch < self.curriculum_steps[i]:
                output[:, -len(self.curriculum_steps) + i] = 0
        return output, extra_outputs

    def decode(self, all_outputs, labels, conf_threshold=0.5):
        outputs, extra_outputs = all_outputs
        if extra_outputs is not None:
            extra_outputs = extra_outputs.reshape(labels.shape[0], 5, -1)
            extra_outputs = extra_outputs.argmax(dim=2)
        outputs = outputs.reshape(len(outputs), -1, 7)  # score + upper + lower + 4 coeffs = 7
        outputs[:, :, 0] = self.sigmoid(outputs[:, :, 0])
        outputs[outputs[:, :, 0] < conf_threshold] = 0

        return outputs, extra_outputs

    def focal_loss(self, pred_confs, target_confs, alpha=0.25, gamma=2.0):
        """
        Focal Loss for binary classification.

        :param pred_confs: Predicted confidences (sigmoid outputs).
        :param target_confs: Target labels (0 or 1).
        :param alpha: Balancing factor for the class imbalance (default=0.25).
        :param gamma: Focusing parameter to reduce loss for easy examples (default=2.0).
        :return: Focal loss value.
        """
        # Clip predictions to prevent log(0) errors
        pred_confs = torch.clamp(pred_confs, min=1e-7, max=1 - 1e-7)

        # Compute the focal loss
        target_confs = target_confs.float()
        cross_entropy_loss = -target_confs * torch.log(pred_confs) - (1 - target_confs) * torch.log(1 - pred_confs)
        loss = alpha * ((1 - pred_confs) ** gamma) * cross_entropy_loss

        return loss.mean()  # Average loss over the batch
# Poly loss calculation (giữ nguyên phần tính poly loss)


    def chamfer_distance(self, points1, points2):
        """
        Tính Chamfer Distance giữa hai tập điểm.
        :param points1: Tensor (N, 2) - tập điểm của đường cong thực tế
        :param points2: Tensor (M, 2) - tập điểm của đường cong dự đoán
        :return: Chamfer Distance
        """
        dist_matrix = torch.cdist(points1, points2, p=2)  # Khoảng cách Euclidean
        dist1 = torch.min(dist_matrix, dim=1)[0].mean()  # Khoảng cách nhỏ nhất từ curve1 đến curve2
        dist2 = torch.min(dist_matrix, dim=0)[0].mean()  # Khoảng cách nhỏ nhất từ curve2 đến curve1
        return dist1 + dist2

    def loss(self,
             outputs,
             target,
             conf_weight=1,
             lower_weight=1,
             upper_weight=1,
             cls_weight=1,
             poly_weight=300,
             curve_weight=500,
             threshold=15 / 720.):
        pred, extra_outputs = outputs
        mse = nn.MSELoss()
        s = nn.Sigmoid()
        threshold = nn.Threshold(threshold**2, 0.)
        pred = pred.reshape(-1, target.shape[1], 1 + 2 + 4)
        target_categories, pred_confs = target[:, :, 0].reshape((-1, 1)), s(pred[:, :, 0]).reshape((-1, 1))
        target_uppers, pred_uppers = target[:, :, 2].reshape((-1, 1)), pred[:, :, 2].reshape((-1, 1))
        target_points, pred_polys = target[:, :, 3:].reshape((-1, target.shape[2] - 3)), pred[:, :, 3:].reshape(-1, 4)
        target_lowers, pred_lowers = target[:, :, 1], pred[:, :, 1]

        # Share top y adjustment
        if self.share_top_y:
            target_lowers[target_lowers < 0] = 1
            target_lowers[...] = target_lowers.min(dim=1, keepdim=True)[0]
            pred_lowers[...] = pred_lowers[:, 0].reshape(-1, 1).expand(pred.shape[0], pred.shape[1])

        target_lowers = target_lowers.reshape((-1, 1))
        pred_lowers = pred_lowers.reshape((-1, 1))

        # Compute valid indices for confidence
        target_confs = (target_categories > 0).float()
        valid_lanes_idx = target_confs == 1
        valid_lanes_idx_flat = valid_lanes_idx.reshape(-1)
        lower_loss = mse(target_lowers[valid_lanes_idx], pred_lowers[valid_lanes_idx])
        upper_loss = mse(target_uppers[valid_lanes_idx], pred_uppers[valid_lanes_idx])

        # Poly loss calculation
        target_xs = target_points[valid_lanes_idx_flat, :target_points.shape[1] // 2]
        ys = target_points[valid_lanes_idx_flat, target_points.shape[1] // 2:].t()
        valid_xs = target_xs >= 0
        pred_polys = pred_polys[valid_lanes_idx_flat]
        pred_xs = pred_polys[:, 0] * ys**3 + pred_polys[:, 1] * ys**2 + pred_polys[:, 2] * ys + pred_polys[:, 3]
        pred_xs.t_()
        weights = (torch.sum(valid_xs, dtype=torch.float32) / torch.sum(valid_xs, dim=1, dtype=torch.float32))**0.5
        pred_xs = (pred_xs.t_() * weights).t()
        target_xs = (target_xs.t_() * weights).t()
        poly_loss = mse(pred_xs[valid_xs], target_xs[valid_xs]) / valid_lanes_idx.sum()

        # Curve Distance Loss: Ensure target_xs and ys have the same size
        min_size = min(target_xs.shape[0], ys.shape[0])
        target_xs = target_xs[:min_size]
        ys = ys[:min_size]
        pred_xs = pred_xs[:min_size]

        # Further adjust dimensions to align in shape
        if target_xs.shape[1] != ys.shape[1]:
            min_size = min(target_xs.shape[1], ys.shape[1])
            target_xs = target_xs[:, :min_size]
            ys = ys[:, :min_size]
            pred_xs = pred_xs[:, :min_size]

        # Create curve points
        target_curve_points = torch.stack([target_xs, ys], dim=-1)  # Tập điểm thực tế
        pred_curve_points = torch.stack([pred_xs, ys], dim=-1)  # Tập điểm dự đoán
        curve_loss = self.chamfer_distance(target_curve_points, pred_curve_points) * curve_weight

        # Classification loss (using Focal Loss instead of BCE Loss)
        cls_loss = 0  # Initialize cls_loss to prevent UnboundLocalError
        if self.pred_category and self.extra_outputs > 0:
            ce = nn.CrossEntropyLoss()
            pred_categories = extra_outputs.reshape(target.shape[0] * target.shape[1], -1)
            target_categories = target_categories.reshape(pred_categories.shape[:-1]).long()
            pred_categories = pred_categories[target_categories > 0]
            target_categories = target_categories[target_categories > 0]
            cls_loss = ce(pred_categories, target_categories - 1)

        # Applying weights to partial losses
        poly_loss = poly_loss * poly_weight
        lower_loss = lower_loss * lower_weight
        upper_loss = upper_loss * upper_weight
        cls_loss = cls_loss * cls_weight
        conf_loss = self.focal_loss(pred_confs, target_confs) * conf_weight

        # Total loss
        loss = conf_loss + lower_loss + upper_loss + poly_loss + cls_loss + curve_loss

        return loss, {
            'conf': conf_loss,
            'lower': lower_loss,
            'upper': upper_loss,
            'poly': poly_loss,
            'cls_loss': cls_loss,
            'curve_loss': curve_loss
        }
