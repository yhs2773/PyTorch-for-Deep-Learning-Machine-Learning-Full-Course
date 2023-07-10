import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels: int=3,
                 patch_size: int=16,
                 embedding_dim: int=768):
        super().__init__()
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)
        self.flatten = nn.Flatten(start_dim=2,
                                  end_dim=3)
        
    def forward(self, x):
        x = self.patcher(x)
        x = self.flatten(x)
        return x.permute(0, 2, 1)

class ViT(nn.Module):
    def __init__(self,
                 img_size: int=224,
                 in_channels: int=3,
                 patch_size: int=16,
                 embedding_dim: int=768,
                 embedding_dropout: float=0.1,
                 d_model: int=768,
                 nhead: int=12,
                 dim_feedforward: int=3072,
                 activation: str="gelu",
                 dropout: float=0.1,
                 batch_first: bool=True,
                 norm_first: bool=True,
                 num_layers: int=12,
                 num_classes: int=1000
                 ):
        super().__init__()
        self.num_patches = (img_size * img_size) // patch_size**2
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)
        self.position_embedding = nn.Parameter(data=torch.rand(1, self.num_patches + 1, embedding_dim),
                                               requires_grad=True)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=d_model,                  # Create a single transformer encoder
                                                                                                  nhead=nhead,
                                                                                                  dim_feedforward=dim_feedforward,
                                                                                                  activation=activation,
                                                                                                  dropout=dropout,
                                                                                                  batch_first=batch_first,
                                                                                                  norm_first=norm_first),
                                                        num_layers=num_layers)                                                      # stack the encoder N times
        self.classifier = nn.Sequential(nn.LayerNorm(normalized_shape=embedding_dim),
                                        nn.Linear(in_features=embedding_dim,
                                                  out_features=num_classes))
        
    def forward(self, x):
        batch_size = x.shape[0]                                         # get the batch size from x
        class_token = self.class_embedding.expand(batch_size, -1, -1)   # expand the class token across the batch size
        x = self.patch_embedding(x)                                     # create the patch embedding
        x = torch.cat((class_token, x), dim=1)                          # prepend the class token to the patch embedding
        x = self.position_embedding + x                                 # add the position embedding to the patch embedding with class token
        x = self.embedding_dropout(x)                                   # apply dropout on patch + positional embedding
        x = self.transformer_encoder(x)                                 # pass embedding through transformer encoder stack
        x = self.classifier(x[:, 0])                                    # pass 0th index of x through MLP head
        return x
