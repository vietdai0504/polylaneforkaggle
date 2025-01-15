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


class FeatureFlipBlock(nn.Module):
    def __init__(self, axis=1):
        super(FeatureFlipBlock, self).__init__()
        self.axis = axis

    def forward(self, x):
        return torch.flip(x, [self.axis])

class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[1, 2, 4]):
        super(SpatialPyramidPooling, self).__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        features = []
        for pool_size in self.pool_sizes:
            kernel_size = (height // pool_size, width // pool_size)
            stride = kernel_size
            pooling = nn.AdaptiveAvgPool2d(kernel_size)
            pooled = pooling(x)
            features.append(pooled.view(batch_size, -1))
        return torch.cat(features, dim=1)

class PolyRegression(nn.Module):
    def __init__(self,
                 num_outputs,
                 backbone,
                 pretrained,
                 curriculum_steps=None,
                 extra_outputs=0,
                 share_top_y=True,
                 pred_category=False,
                 attention_heads=5,
                 vit_embed_dim=255,
                 vit_num_layers=4,
                 use_flip_block=True,
                 flip_axis=3,
                 spp_pool_sizes=[1, 2, 4]):  # Added SPP pool sizes
        super(PolyRegression, self).__init__()

        # Check if attention_heads divides embed_dim
        if vit_embed_dim % attention_heads != 0:
            raise ValueError(f"embed_dim ({vit_embed_dim}) must be divisible by num_heads ({attention_heads})")

        self.use_flip_block = use_flip_block
        self.flip_axis = flip_axis

        # Initialize the backbone
        self.model = self._initialize_backbone(backbone, num_outputs, pretrained, extra_outputs)

        # Backbone feature output size
        self.backbone_out_features = self.model.last_channel if backbone == 'mobilenet_v2' else 1280

        # SPP Layer
        self.spp = SpatialPyramidPooling(pool_sizes=spp_pool_sizes)
        
        # Dynamically compute the SPP output size
        self.spp_output_size = self._compute_spp_output_size(spp_pool_sizes, feature_map_size=(12, 20))  # Adjust as needed

        # Add a projection layer to reduce SPP output size
        self.spp_projection = nn.Linear(self.spp_output_size, vit_embed_dim)

        # Output Layer
        self.output_layer = OutputLayer(nn.Linear(vit_embed_dim, num_outputs), extra_outputs)

        # Self-Attention Layer
        self.attention = SelfAttention(embed_size=num_outputs, heads=attention_heads)

        # Feature Flip Block (if needed)
        if self.use_flip_block:
            self.feature_flip = FeatureFlipBlock(axis=self.flip_axis)

        # Other attributes
        self.curriculum_steps = curriculum_steps if curriculum_steps else [0, 0, 0, 0]
        self.share_top_y = share_top_y
        self.extra_outputs = extra_outputs
        self.pred_category = pred_category
        self.sigmoid = nn.Sigmoid()

    def _initialize_backbone(self, backbone, num_outputs, pretrained, extra_outputs):
        """
        Initialize the backbone network based on the provided backbone type.
        """
        if backbone == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=pretrained)
            self.backbone_out_features = model.last_channel  # MobileNetV2 default is 1280
            model.classifier[1] = OutputLayer(nn.Linear(model.classifier[1].in_features, num_outputs), extra_outputs)
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented")

        return model

    def _compute_spp_output_size(self, pool_sizes, feature_map_size):
        """
        Dynamically compute the output size of the SPP layer based on pooling sizes and feature map dimensions.
        """
        h, w = feature_map_size
        output_size = 0
        for pool_size in pool_sizes:
            kernel_h, kernel_w = h // pool_size, w // pool_size
            output_size += self.backbone_out_features * kernel_h * kernel_w
        return output_size

    def forward(self, x, epoch=None):
        print(f"Input shape: {x.shape}")  # Debug
        features = self.model.features(x)  # Extract features from backbone
        print(f"Shape after backbone: {features.shape}")  # Debug

        if self.use_flip_block:
            features = self.feature_flip(features)
            print(f"Shape after FeatureFlipBlock: {features.shape}")  # Debug

        # Apply SPP
        features = self.spp(features)
        print(f"Shape after SPP: {features.shape}")  # Debug

        # Apply SPP projection
        features = self.spp_projection(features)
        print(f"Shape after SPP Projection: {features.shape}")  # Debug

        # Linear Layer and Self-Attention
        output, extra_outputs = self.output_layer(features)
        output, _ = self.attention(output, output, output)

        # Curriculum Learning
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

    def loss(self,
             outputs,
             target,
             conf_weight=1,
             lower_weight=1,
             upper_weight=1,
             cls_weight=1,
             poly_weight=300,
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

        # Applying weights to partial losses
        poly_loss = poly_loss * poly_weight
        lower_loss = lower_loss * lower_weight
        upper_loss = upper_loss * upper_weight
        cls_loss = cls_loss * cls_weight

        # Focal loss for confidence prediction
        conf_loss = self.focal_loss(pred_confs, target_confs) * conf_weight

        loss = conf_loss + lower_loss + upper_loss + poly_loss + cls_loss

        return loss, {
            'conf': conf_loss,
            'lower': lower_loss,
            'upper': upper_loss,
            'poly': poly_loss,
            'cls_loss': cls_loss
        }
