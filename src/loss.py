import torch
from torch import Tensor
import torch.distributed as dist
import torch.nn.functional as F


class SimpleContrastiveLoss:
    def __init__(self, temperature: float = 0.02):
        self.temperature = temperature

    def __call__(self, x: Tensor, y: Tensor, target: Tensor = None) -> Tensor:
        if target is None:
            target_per_qry = y.size(0) // x.size(0)
            target = torch.arange(
                0,
                x.size(0) * target_per_qry,
                target_per_qry,
                device=x.device,
                dtype=torch.long,
            )

        dim = x.shape[-1]
        loss = 0.0

        for d in [
            dim,
            dim // 2,
            dim // 4,
            dim // 8,
            dim // 16,
            dim // 32,
            dim // 64,
        ]:
            loss += self._loss_fn(x, y, target, dim=d)

        return loss

    def _loss_fn(
        self,
        x: Tensor,
        y: Tensor,
        target: Tensor,
        dim: int = None,
    ) -> Tensor:
        logits = torch.matmul(x[:, :dim], y[:, :dim].transpose(0, 1))
        return F.cross_entropy(logits / self.temperature, target)


class DistributedContrastiveLoss(SimpleContrastiveLoss):
    def __init__(self, scale_loss: bool = True, temperature: float = 0.02):
        assert (
            dist.is_initialized()
        ), "Distributed training has not been properly initialized."
        super().__init__()
        self.word_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss
        self.temperature = temperature

    def __call__(self, x: Tensor, y: Tensor, **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        loss = super().__call__(dist_x, dist_y, **kwargs)
        return loss * (self.word_size if self.scale_loss else 1.0)

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)
