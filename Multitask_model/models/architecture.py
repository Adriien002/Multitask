from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.networks.nets.basic_unet import TwoConv, Down, UpCat
from monai.utils import ensure_tuple_rep

class PatchAttentionAggregator(nn.Module):
    """
    Module d'agrégation par attention (Multiple Instance Learning).
    Prend [B, N, C, H, W, D] et retourne [B, C, H, W, D]
    """
    def __init__(self, in_channels: int, hidden_channels: int = 256):
        super().__init__()
        
        # 1. Un "pooler" pour obtenir un vecteur par patch
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # 2. Un réseau "gating" pour calculer le score de chaque patch
        #    Nous utilisons LayerNorm ici, comme dans votre cls_head.
        self.gate_nn = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_channels, 1) # Sortie : 1 score par patch
        )
        
        # 3. Softmax pour normaliser les scores en poids (somme=1)
        self.softmax = nn.Softmax(dim=1) # Appliquer sur la dimension N

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tenseur d'entrée de forme [B, N, C, H, W, D]
        """
        # Sauvegarde des dimensions
        B, N, C, H, W, D = x.shape
        
        # 1. Reshape pour traiter les N patchs comme un batch
        # [B, N, C, H, W, D] -> [B*N, C, H, W, D]
        x_reshaped = x.view(-1, C, H, W, D)
        
        # 2. Pooler : résume chaque patch en 1 vecteur
        # [B*N, C, H, W, D] -> [B*N, C, 1, 1, 1]
        pooled = self.pool(x_reshaped)
        
        # Aplatir pour le MLP
        # [B*N, C, 1, 1, 1] -> [B*N, C]
        pooled_flat = pooled.view(B * N, C)
        
        # 3. Gating : Calcule le score (logit) pour chaque patch
        # [B*N, C] -> [B*N, 1]
        scores = self.gate_nn(pooled_flat)
        
        # 4. Reshape pour Softmax : regroupe les scores par batch original
        # [B*N, 1] -> [B, N]
        scores_by_batch = scores.view(B, N)
        
        # 5. Softmax : Calcule les poids d'attention
        # Les poids des N patchs pour chaque item du batch somment à 1
        # [B, N]
        weights = self.softmax(scores_by_batch)
        
        # 6. Agrandir les poids pour la multiplication (broadcasting)
        # [B, N] -> [B, N, 1, 1, 1, 1]
        weights_broadcast = weights.view(B, N, 1, 1, 1, 1)
        
        # 7. Appliquer les poids :
        # [B, N, C, H, W, D] * [B, N, 1, 1, 1, 1] -> [B, N, C, H, W, D]
        weighted_features = x * weights_broadcast
        
        # 8. Agréger (somme pondérée)
        # torch.sum sur la dim N -> [B, C, H, W, D]
        aggregated = torch.sum(weighted_features, dim=1)
        
        return aggregated
  
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
        self.patch_aggregator = PatchAttentionAggregator(
            in_channels=fea[4], 
            hidden_channels=256 # Taille ajustable
        )
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

    def forward(self, x: torch.Tensor, task: str = "segmentation",batch_size: int = None): # ajout de batch_size pour séparer patches et batch
    
        if task == "segmentation":
            # code actuel pour la segmentation
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
         
           
        # X est de la shape : [B*N, C, H, W, D]
        # Gestion des multi-patches

            if batch_size is None:
                raise ValueError("Attention , fournir batch_size ")
            total_items = x.shape[0] # B*N
            if total_items % batch_size != 0:
                raise ValueError(f"Taille de batch incohérente ! total_items={total_items}, batch_size={batch_size}")
            num_patches = total_items // batch_size # 18 normalement
                
               
                # Forward sur tous les patches
            x0 = self.conv_0(x)
            x1 = self.down_1(x0)
            x2 = self.down_2(x1)
            x3 = self.down_3(x2)
            x4 = self.down_4(x3)  # [B*N, features, H', W', D']
                
          
            #Nécessaire pour l'attention 
            x4 = x4.view(batch_size, num_patches, *x4.shape[1:])
                
            # Shape d'entrée (x4): torch.Size([2, 18, 256, 6, 6, 6])
            #-> B=2, N=18, C=256, H=6, W=6, D=6
                #aggregated = torch.max(x4, dim=1)[0]  # Max pooling sur les patches
                #aggregated = torch.mean(x4, dim=1) # Moyejnne sur les patches
            aggregated = self.patch_aggregator(x4)
            #print(f"ShAAAPEPEPEPEPEP ON RENTREEEKOAK OK: {aggregated.shape}, super-patch")
            # Shape agrégée : torch.Size([2, 256, 6, 6, 6]), super-patch
            cls_logits = self.cls_head(aggregated) #et shape de sortie : torch.Size([2, 6])
            return None, cls_logits
          
      
        #Possibilité d'ajouter d'autres tâches ici 
  




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


