import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict


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


class FeatureFlipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, axis=1, kernel_size=3):
        super(FeatureFlipBlock, self).__init__()
        self.axis = axis

        # Tích chập để giảm chiều kênh từ 2C về C sau khi pooling
        self.conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

        # Pooling trung bình để giảm chiều rộng (W -> W/2)
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 2))  # Giảm chiều rộng còn một nửa

    def forward(self, x):
        # print(f"Input to FeatureFlipBlock: {x.shape}")  # Debugging input shape
        # Lật đặc trưng theo chiều axis
        flipped = torch.flip(x, [self.axis])

        # Kết hợp đặc trưng gốc và lật theo chiều kênh (2C)
        merged = torch.cat([x, flipped], dim=1)
        # print(f"After concatenation: {merged.shape}")  # Debugging shape after concatenation

        # Pooling để giảm chiều rộng
        pooled = self.avg_pool(merged)
        # print(f"After average pooling: {pooled.shape}")  # Debugging shape after pooling

        # Tích chập để giảm chiều kênh về out_channels
        output = self.conv(pooled)
        # print(f"Output of FeatureFlipBlock: {output.shape}")  # Debugging output shape

        return output


class PolyRegression(nn.Module):
    def __init__(self,
                 num_outputs,
                 backbone,
                 pretrained=True,
                 curriculum_steps=None,
                 extra_outputs=0,
                 share_top_y=True,
                 pred_category=False,
                 attention_heads=5,
                 use_flip_block=True,
                 flip_axis=1):
        super(PolyRegression, self).__init__()

        self.use_flip_block = use_flip_block
        self.flip_axis = flip_axis
        self.curriculum_steps = curriculum_steps if curriculum_steps else [0, 0, 0, 0]
        self.share_top_y = share_top_y
        self.extra_outputs = extra_outputs
        self.pred_category = pred_category
        self.sigmoid = nn.Sigmoid()

        # Initialize backbone
        self.model = self._initialize_backbone(backbone, num_outputs, pretrained, extra_outputs)

        # Adjust Feature Pyramid Network
        in_channels_list = [16, 24, 32, 96]  # Corrected channels from mobilenet_v2
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=256)

        # Output layers
        self.fpn_output = nn.Conv2d(256, num_outputs, kernel_size=1)
        self.extra_output_layer = (
            nn.Conv2d(256, extra_outputs, kernel_size=1) if extra_outputs > 0 else None
        )

        # Self-Attention
        self.attention = SelfAttention(embed_size=num_outputs, heads=attention_heads)

        # Flip block
        if self.use_flip_block:
            self.flip_block = FeatureFlipBlock(in_channels=3, out_channels=256, axis=self.flip_axis)  # Match the expected input channels of the backbone

        # Channel adapter to match the backbone input channels
        self.channel_adapter = nn.Conv2d(256, 3, kernel_size=1)

    def _initialize_backbone(self, backbone, num_outputs, pretrained, extra_outputs):
        if 'mobilenet_v2' in backbone:
            model = models.mobilenet_v2(pretrained=pretrained)
            return nn.ModuleList([
                model.features[:2],   # Stage 0 (16 channels)
                model.features[2:4],  # Stage 1 (24 channels)
                model.features[4:7],  # Stage 2 (32 channels)
                model.features[7:14], # Stage 3 (96 channels)
            ])
        else:
            raise NotImplementedError(f"Backbone {backbone} not supported")

    def _extract_backbone_features(self, x):
        """
        Extract features from backbone for each stage. Ensure features match FPN stages.
        """
        features = OrderedDict()
        for idx, layer in enumerate(self.model):
            x = layer(x)
            # print(f"Backbone stage {idx} output shape: {x.shape}")  # Debugging output shape
            features[str(idx)] = x
        return features

    def forward(self, x, epoch=None, **kwargs):
        
        if self.use_flip_block:
            x = self.flip_block(x)
            x = self.channel_adapter(x)  # Adjust channels to match backbone input
            # print(f"After channel adapter: {x.shape}")  # Debugging shape after channel adapter

        # Extract features from backbone
        features = self._extract_backbone_features(x)

        # Apply FPN
        fpn_features = self.fpn(features)

        # Use top FPN output for regression
        fpn_output = self.fpn_output(fpn_features["3"])
        fpn_output = F.adaptive_avg_pool2d(fpn_output, (1, 1)).view(fpn_output.size(0), -1)

        # Compute extra outputs if needed
        extra_outputs = None
        if self.extra_outputs > 0:
            extra_outputs = self.extra_output_layer(fpn_features["3"])
            extra_outputs = F.adaptive_avg_pool2d(extra_outputs, (1, 1)).view(extra_outputs.size(0), -1)

        # Apply Self-Attention
        fpn_output, _ = self.attention(fpn_output, fpn_output, fpn_output)

        # Curriculum learning
        for i in range(len(self.curriculum_steps)):
            if epoch is not None and epoch < self.curriculum_steps[i]:
                fpn_output[:, -len(self.curriculum_steps) + i] = 0

        return fpn_output, extra_outputs

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
    def line_iou_loss(self, pred_points, target_points, radius=0.1):
        """
        Calculate Line IoU Loss.
        
        :param pred_points: Predicted lane points (N x 2 or N x 1 for x coordinates)
        :param target_points: Ground truth lane points (N x 2 or N x 1 for x coordinates)
        :param radius: Radius for extending points into line segments.
        :return: Line IoU Loss (1 - Line IoU)
        """
        # Ensure inputs are in the same shape
        assert pred_points.shape == target_points.shape, "Shape mismatch between predictions and targets"

        # Extend points into line segments
        pred_start = pred_points - radius
        pred_end = pred_points + radius
        target_start = target_points - radius
        target_end = target_points + radius

        # Calculate intersection and union
        intersection_start = torch.max(pred_start, target_start)
        intersection_end = torch.min(pred_end, target_end)
        intersection = torch.clamp(intersection_end - intersection_start, min=0)

        union_start = torch.min(pred_start, target_start)
        union_end = torch.max(pred_end, target_end)
        union = torch.clamp(union_end - union_start, min=0)

        iou = intersection / (union + 1e-7)  # Add epsilon to prevent division by zero
        line_iou = iou.sum(dim=1) / pred_points.shape[1]  # Average IoU over all points

        # Line IoU Loss
        liou_loss = 1 - line_iou.mean()
        return liou_loss

    def loss(self,
            outputs,
            target,
            conf_weight=1,
            lower_weight=1,
            upper_weight=1,
            cls_weight=1,
            poly_weight=300,
            line_iou_weight=0.5,  # New weight for Line IoU Loss
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

        if self.share_top_y:
            target_lowers[target_lowers < 0] = 1
            target_lowers[...] = target_lowers.min(dim=1, keepdim=True)[0]
            pred_lowers[...] = pred_lowers[:, 0].reshape(-1, 1).expand(pred.shape[0], pred.shape[1])

        target_lowers = target_lowers.reshape((-1, 1))
        pred_lowers = pred_lowers.reshape((-1, 1))

        target_confs = (target_categories > 0).float()
        valid_lanes_idx = target_confs == 1
        valid_lanes_idx_flat = valid_lanes_idx.reshape(-1)
        lower_loss = mse(target_lowers[valid_lanes_idx], pred_lowers[valid_lanes_idx])
        upper_loss = mse(target_uppers[valid_lanes_idx], pred_uppers[valid_lanes_idx])

        # Classification loss (using Focal Loss instead of BCE Loss)
        if self.pred_category and self.extra_outputs > 0:
            ce = nn.CrossEntropyLoss()
            pred_categories = extra_outputs.reshape(target.shape[0] * target.shape[1], -1)
            target_categories = target_categories.reshape(pred_categories.shape[:-1]).long()
            pred_categories = pred_categories[target_categories > 0]
            target_categories = target_categories[target_categories > 0]
            cls_loss = ce(pred_categories, target_categories - 1)
        else:
            cls_loss = 0

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
        poly_loss = threshold(
            (pred_xs[valid_xs] - target_xs[valid_xs])**2).sum() / (valid_lanes_idx.sum() * valid_xs.sum())

        # Line IoU Loss calculation
        pred_points = pred_xs.T  # Reshape to (N, points)
        target_points = target_xs.T  # Reshape to (N, points)
        line_iou_loss = self.line_iou_loss(pred_points, target_points)

        # Applying weights to partial losses
        poly_loss = poly_loss * poly_weight
        lower_loss = lower_loss * lower_weight
        upper_loss = upper_loss * upper_weight
        cls_loss = cls_loss * cls_weight
        line_iou_loss = line_iou_loss * line_iou_weight

        # Focal loss for confidence prediction
        conf_loss = self.focal_loss(pred_confs, target_confs) * conf_weight

        loss = conf_loss + lower_loss + upper_loss + poly_loss + cls_loss + line_iou_loss

        return loss, {
            'conf': conf_loss,
            'lower': lower_loss,
            'upper': upper_loss,
            'poly': poly_loss,
            'cls_loss': cls_loss,
            'line_iou': line_iou_loss
        }
