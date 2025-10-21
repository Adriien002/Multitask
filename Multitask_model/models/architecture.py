from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.networks.nets.basic_unet import TwoConv, Down, UpCat
from monai.utils import ensure_tuple_rep

class MS_CAM_3D(nn.Module):
    # Adapté pour 3D et Batch Size = 1 (remplacement BN par GroupNorm ici )
    def __init__(self, channels=64, r=4, spatial_dims=3):
        super(MS_CAM_3D, self).__init__()
        inter_channels = int(channels // r)
        Conv = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        
        # Helper for Group Norm (equivalent to LayerNorm on C, H, W, D when num_groups=channels)
        Norm = lambda num_features: nn.GroupNorm(num_features=num_features, num_groups=1)
        
        # Local Attention (Local context)
        self.local_att = nn.Sequential(
            Conv(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            Norm(inter_channels),
            nn.ReLU(inplace=True),
            Conv(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            Norm(channels),
        )

        # Global Attention (Global context)
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1) if spatial_dims == 3 else nn.AdaptiveAvgPool2d(1),
            Conv(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            Norm(inter_channels),
            nn.ReLU(inplace=True),
            Conv(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            Norm(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        # L'AFF utilise cette sortie avant la Sigmoid pour le calcul de poids binaire
        return xlg # On retourne les logits pour l'agrégation N-aire, on va utiliser Softmax
        
class PatchFusionAttention_MSCAM(nn.Module):
    """
    Adapte la logique de l'AFF/MS-CAM pour la fusion N-aire de patches 3D.
    Utilise le MS-CAM pour calculer des poids d'attention multi-échelle (local + global)
    pour chaque voxel, puis une Softmax sur la dimension des patches (N).
    """
    def __init__(self, in_channels, r=4):
        super().__init__()
        # Le module MS-CAM (sans la Sigmoid) pour générer les logits de poids
        self.ms_cam_logits = MS_CAM_3D(channels=in_channels, r=r, spatial_dims=3)
        # La Sigmoid n'est pas utilisée ici; on utilisera une Softmax sur la dimension N.
        
    def forward(self, x4):
        """
        x4: [B, N, C, H', W', D'] - Tenseur des caractéristiques de patches
        """
        B, N, C, H, W, D = x4.shape
        
        # 1. Reshape et Calcul des Logits MS-CAM par Patch
        # x_flat: [B*N, C, H', W', D']
        x_flat = x4.view(B * N, C, H, W, D) 
        
        # logits_flat: [B*N, C, H', W', D'] (Scores d'attention non normalisés)
        # Ces logits intègrent le contexte local (point-wise conv) et global (avg pool) de chaque patch.
        logits_flat = self.ms_cam_logits(x_flat) 
        
        # 2. Reshape pour Softmax
        # logits: [B, N, C, H', W', D']
        logits = logits_flat.view(B, N, C, H, W, D)
        
        # 3. Normalisation Softmax sur la dimension des patches (dim=1)
        # Softmax appliquée élément-par-élément. Pour chaque voxel (c, h, w, d), 
        # la somme des N poids (sur les N patches) est égale à 1.
        # attn: [B, N, C, H', W', D'] (Les poids M_i)
        attn_weights = F.softmax(logits, dim=1)
        
        # 4. Pondération et Agrégation Finale
        # weighted: [B, N, C, H', W', D']
        weighted = x4 * attn_weights
        
        # aggregated: [B, C, H', W', D']
        aggregated = weighted.sum(dim=1) 

        return aggregated, attn_weights        

class BasicUNetWithClassification(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 6,  # pour segmentation
        num_cls_classes: int = 6,  # pour classification
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}), #True
        norm: str | tuple = ("instance", {"affine": True}), # True
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
    ):
        super().__init__()
        
        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        # Encoder
        self.conv_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        
        # Decoder
        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)
        bottleneck_channels = fea[4] 
        self.patch_fusion = PatchFusionAttention_MSCAM(in_channels=bottleneck_channels, r=4)
        # Classification head → à partir du bottleneck x4
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool3d((4, 4, 4)),
            nn.Flatten(),
            nn.Linear(fea[4] * 4 * 4 * 4, 512),
            nn.LayerNorm(512), #nn.BatchNorm1d(512)
            nn.LeakyReLU(inplace=True), #True
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256), #nn.BatchNorm1d(256)
            nn.LeakyReLU(inplace=True), #True
            nn.Dropout(0.3),
            nn.Linear(256, num_cls_classes)
        )

    def forward(self, x: torch.Tensor, task: str = "segmentation"):
        if task == "segmentation":
            # Ton code actuel pour la segmentation
            x0 = self.conv_0(x)
            x1 = self.down_1(x0)
            x2 = self.down_2(x1)
            x3 = self.down_3(x2)
            x4 = self.down_4(x3)
            
            u4 = self.upcat_4(x4, x3)
            u3 = self.upcat_3(u4, x2)
            u2 = self.upcat_2(u3, x1)
            u1 = self.upcat_1(u2, x0)
            seg_logits = self.final_conv(u1)
            
            return seg_logits, None

        elif task == "classification":
        # Gestion des multi-patches
            if x.dim() == 6:  # [B, N, C, H, W, D]
                batch_size, num_patches = x.shape[0], x.shape[1]
                
                # RESHAPE CRITIQUE: [B, N, C, H, W, D] -> [B*N, C, H, W, D]
                x_reshaped = x.view(-1, *x.shape[2:])  # [B*N, C, H, W, D]
                
                # Forward sur tous les patches
                x0 = self.conv_0(x_reshaped)
                x1 = self.down_1(x0)
                x2 = self.down_2(x1)
                x3 = self.down_3(x2)
                x4 = self.down_4(x3)  # [B*N, features, H', W', D']
                
                # Reshape back: [B*N, C, H, W, D] -> [B, N, C, H, W, D]
                x4 = x4.view(batch_size, num_patches, *x4.shape[1:])
                
                # Agrégation: [B, N, C, H, W, D] -> [B, C, H, W, D]
                #aggregated = torch.max(x4, dim=1)[0]  # Max pooling sur les patches
                #aggregated = torch.mean(x4, dim=1) # ici metrre fusion attention
                aggregated, attn_weights = self.patch_fusion(x4)
                # Classification
                #cls_logits = self.cls_head(x4)
                cls_logits = self.cls_head(aggregated)
                return None, cls_logits
          
      
        
    
    # def forward(self, x: torch.Tensor):
    #     # Encoder
    #     x0 = self.conv_0(x)
    #     x1 = self.down_1(x0)
    #     x2 = self.down_2(x1)
    #     x3 = self.down_3(x2)
    #     x4 = self.down_4(x3)

    #     # Decoder (segmentation)
    #     u4 = self.upcat_4(x4, x3)
    #     u3 = self.upcat_3(u4, x2)
    #     u2 = self.upcat_2(u3, x1)
    #     u1 = self.upcat_1(u2, x0)
    #     seg_logits = self.final_conv(u1)

    #     # Classification
    #     cls_logits = self.cls_head(x4)  # x4 est le bottleneck

    #     return seg_logits  , cls_logits




class BasicUNetEncoder(nn.Module):
    def __init__(self, spatial_dims=3, in_channels=1, features=(32, 32, 64, 128, 256, 32), act=("LeakyReLU", {"negative_slope": 0.1,"inplace": True}), norm=("instance", {"affine": True}), bias=True, dropout=0.0):
        super().__init__()
        from monai.networks.nets.basic_unet import TwoConv, Down
        from monai.utils import ensure_tuple_rep
        fea = ensure_tuple_rep(features, 6)

        self.conv_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

    def forward(self, x):
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)
        return [x0, x1, x2, x3, x4]


class SegDecoder(nn.Module):
    def __init__(self, spatial_dims=3, features=(32, 32, 64, 128, 256, 32), out_channels=6, act=("LeakyReLU", {"negative_slope": 0.1,"inplace": True}), norm=("instance", {"affine": True}), bias=True, dropout=0.0, upsample="deconv"):
        super().__init__()
        from monai.networks.nets.basic_unet import UpCat
        from monai.networks.layers.factories import Conv
        from monai.utils import ensure_tuple_rep

        fea = ensure_tuple_rep(features, 6)
        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)
        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

    def forward(self, features):
        x0, x1, x2, x3, x4 = features
        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)
        return self.final_conv(u1)


class ClsHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d((4, 4, 4)),
            nn.Flatten(),
            nn.Linear(in_channels * 4 * 4 * 4, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        
    def forward(self, x):
        return self.head(x)


