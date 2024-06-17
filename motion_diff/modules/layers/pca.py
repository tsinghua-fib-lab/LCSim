import torch
from torch import nn
from sklearn.decomposition import PCA


class PCA(nn.Module):
    def __init__(self, pca: PCA):
        super(PCA, self).__init__()
        self.pca = pca
        self.n_components = pca.n_components
        self.whiten = pca.whiten
        self.register_buffer("mean_", torch.tensor(pca.mean_, dtype=torch.float32))
        self.register_buffer(
            "components_", torch.tensor(pca.components_, dtype=torch.float32)
        )
        self.register_buffer(
            "explained_variance_",
            torch.tensor(pca.explained_variance_, dtype=torch.float32),
        )
        self.register_buffer(
            "explained_variance_ratio_",
            torch.tensor(pca.explained_variance_ratio_, dtype=torch.float32),
        )
        self.register_buffer(
            "singular_values_", torch.tensor(pca.singular_values_, dtype=torch.float32)
        )

    def forward(self, x, inverse=False):
        x = x.to(self.mean_.device, self.mean_.dtype)
        if inverse:
            assert x.shape[-1] == self.n_components
            if self.whiten:
                x = x * torch.sqrt(self.explained_variance_)
            x = torch.matmul(x, self.components_)
            if self.mean_ is not None:
                x = x + self.mean_
        else:
            assert x.shape[-1] == self.components_.shape[-1]
            if self.mean_ is not None:
                x = x - self.mean_
            x = torch.matmul(x, self.components_.T)
            if self.whiten:
                x = x / torch.sqrt(self.explained_variance_)
        return x
