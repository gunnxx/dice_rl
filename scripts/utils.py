from typing import Union

import torch


class TorchStandardScaler:
    """
    Scaler for pytorch tensor.
    """

    def __init__(self) -> None:
        """
        Constructor.
        """
        self.mean = None
        self.std = None

    def fit(self, x: torch.Tensor) -> None:
        """
        Fit the normalizing constant.

        :param x: Tensor to compute the mean and std.
        """
        self.mean = x.mean(0, keepdim=False)
        self.std = x.std(0, unbiased=False, keepdim=False)
        assert torch.all(self.std != 0), "There is at least one element that has 0 std!"

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input tensor.

        :param x: Tensor to be normalized.

        Return:
            Normalized tensor.
        """
        x_scaled = x - self.mean
        x_scaled = x_scaled / self.std
        return x_scaled

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize the input tensor.

        :param x: Tensor to be unnormalized.

        Return:
            Unnormalized tensor.
        """
        x_inv_scaled = x * self.std
        x_inv_scaled = x_inv_scaled + self.mean
        return x_inv_scaled

    def to_device(self, device: Union[str, torch.device]) -> None:
        """
        Move the mean and std tensor to device.

        :param device: Device.
        """
        self.mean = self.mean.to(device)
        self.std  = self.std.to(device)