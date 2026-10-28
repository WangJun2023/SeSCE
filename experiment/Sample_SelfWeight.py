import torch
import torch.nn.functional as F
from torch.nn import Linear


class AttentionMechanism:
    def __init__(self, feature_dim):
        self.query_layer = Linear(feature_dim, feature_dim, bias=False).cuda()
        self.key_layer = Linear(feature_dim, feature_dim, bias=False).cuda()
        self.value_layer = Linear(feature_dim, feature_dim, bias=False).cuda()
        self.scale = torch.sqrt(torch.tensor(feature_dim, dtype=torch.float32)).cuda()

    def compute_attention_weights(self, z_all, zs):
        """
        计算注意力权重，基于全局特征 (z_all) 和每个视图的特征 (zs)。

        参数:
        - z_all: 全局特征张量，形状为 [batch_size, feature_dim]
        - zs: 每个视图的特征列表，其中每个特征形状为 [batch_size, feature_dim]

        返回:
        - attention_weights: 注意力权重张量，形状为 [batch_size, view_count]
        """

        batch_size = z_all.size(0)
        view_count = len(zs)

        Q = self.query_layer(z_all.cuda())
        K = torch.stack([self.key_layer(z.cuda()) for z in zs], dim=1)
        V = torch.stack([self.value_layer(z.cuda()) for z in zs], dim=1)

        assert Q.shape == (
            batch_size, Q.size(-1)), f"Query shape mismatch: expected {(batch_size, Q.size(-1))}, got {Q.shape}"

        assert K.shape == (batch_size, view_count, K.size(
            -1)), f"Key shape mismatch: expected {(batch_size, view_count, K.size(-1))}, got {K.shape}"

        assert V.shape == (batch_size, view_count, V.size(
            -1)), f"Value shape mismatch: expected {(batch_size, view_count, V.size(-1))}, got {V.shape}"

        scores = torch.einsum('bf,bvf->bv', Q, K) / self.scale
        attention_weights = F.softmax(scores, dim=1)

        return attention_weights
