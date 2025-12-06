# from typing import Sequence, Union

# import numpy as np
# import torch
# from albumentations import Normalize
# from torch import Tensor


# class Denormalize:

#     def __init__(self, mean: Sequence[float], std: Sequence[float]):
#         self.mean = torch.tensor(mean)
#         self.std = torch.tensor(std)

#     def __call__(self, tensor: Tensor) -> Tensor:
#         """
#         Args:
#             tensor (Tensor): Tensor image of size (B, C, H, W) to be normalized.
#         Returns:
#             Tensor: Normalized image.
#         """
#         single_image = tensor.ndim == 3
#         tensor = tensor.unsqueeze(0) if single_image else tensor
#         channels = tensor.size(1)
#         # slice to support a lower number of channels
#         means = self.mean[:channels].view(1, -1, 1, 1).to(tensor.device)
#         stds = self.std[:channels].view(1, -1, 1, 1).to(tensor.device)
#         tensor = tensor * stds + means
#         # swap from [B, C, H, W] to [B, H, W, C]
#         tensor = tensor.permute(0, 2, 3, 1)
#         tensor = tensor[0] if single_image else tensor
#         return tensor.detach().cpu().numpy()


# class ClipNormalize(Normalize):

#     def __init__(self,
#                  mean: tuple,
#                  std: tuple,
#                  clip_min: Union[float, tuple],
#                  clip_max: Union[float, tuple],
#                  max_pixel_value: float = 1.0,
#                  always_apply: bool = False,
#                  p: float = 1.0):
#         super().__init__(mean=mean, std=std, max_pixel_value=max_pixel_value, always_apply=always_apply, p=p)
#         self.clip_min = clip_min
#         self.clip_max = clip_max

#     def apply(self, image: np.ndarray, **params) -> np.ndarray:
#         result = super().apply(img=image, **params)
#         return np.clip(result, self.clip_min, self.clip_max)

#     def get_transform_init_args_names(self):
#         parent = list(super().get_transform_init_args_names())
#         return tuple(parent + ["clip_min", "clip_max"])
from typing import Sequence, Union
import numpy as np
import torch
from albumentations import Normalize
from torch import Tensor

class Denormalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor image of size (B, C, H, W) to be normalized.
        Returns:
            Tensor: Denormalized image as numpy array.
        """
        single_image = tensor.ndim == 3
        tensor = tensor.unsqueeze(0) if single_image else tensor
        channels = tensor.size(1)
        means = self.mean[:channels].view(1, -1, 1, 1).to(tensor.device)
        stds = self.std[:channels].view(1, -1, 1, 1).to(tensor.device)
        tensor = tensor * stds + means
        tensor = tensor.permute(0, 2, 3, 1)
        tensor = tensor[0] if single_image else tensor
        return tensor.detach().cpu().numpy()


class ClipNormalize(Normalize):
    def __init__(self,
                 mean: tuple,
                 std: tuple,
                 clip_min: Union[float, tuple],
                 clip_max: Union[float, tuple],
                 max_pixel_value: float = 1.0,
                 always_apply: bool = False,
                 p: float = 1.0):
        super().__init__(
            mean=mean,
            std=std,
            max_pixel_value=max_pixel_value,
            always_apply=always_apply,
            p=p,
        )
        self.clip_min = clip_min
        self.clip_max = clip_max

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        # --- 核心修改：手动实现归一化，绕过 super().apply 的版本兼容问题 ---
        
        # 1. 准备参数 (转为 float32 以避免精度问题)
        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)
        img = image.astype(np.float32)

        # 2. 如果 max_pixel_value 不是 1，先缩放
        if self.max_pixel_value != 1.0:
            img /= self.max_pixel_value

        # 3. 执行归一化公式: (x - mean) / std
        img -= mean
        img /= std

        # 4. 执行 Clip (截断)
        return np.clip(img, self.clip_min, self.clip_max)

    def get_transform_init_args_names(self):
        parent = list(super().get_transform_init_args_names())
        return tuple(parent + ["clip_min", "clip_max"])
