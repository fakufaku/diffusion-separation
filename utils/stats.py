import torch


class StandardScaler(torch.nn.Module):
    def __init__(self, size, dim=-1):
        super().__init__()
        self.register_buffer("_mean", torch.zeros(size))
        self.register_buffer("_var", torch.zeros(size))
        self.register_buffer("_count", torch.zeros(1, dtype=torch.int64))
        self.size = size
        self.dim = dim

    def update(self, data):
        data = torch.moveaxis(data, self.dim, -1)
        if data.shape[-1] != self.size:
            raise ValueError(
                f"The size of the Scaler if {self.size} but "
                f"the input has size {data.shape[-1]}"
            )

        data = data.reshape((-1, data.shape[-1]))

        old_mean = self._mean
        old_var = self._var

        # update count
        block_count = data.shape[0]
        block_sum = data.sum(dim=0)

        # update the mean
        self._count += block_count
        self._mean = old_mean + (block_sum - block_count * old_mean) / self._count

        # update the variance
        up = torch.sum((data - old_mean) * (data - self._mean), dim=0)
        self._var = old_var + up

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._var / (self._count - 1)

    @property
    def scale(self):
        return torch.sqrt(self.var)

    def __len__(self):
        return self._count

    def forward(self, data):
        data = torch.moveaxis(data, self.dim, -1)
        data = (data - self.mean) / self.scale
        data = torch.moveaxis(data, -1, self.dim)
        return data


if __name__ == "__main__":

    n_blocks = 100
    block_size = 4
    n_dim = 5

    scaler = StandardScaler(n_dim, dim=-1)

    x = torch.zeros((n_blocks, block_size, n_dim)).uniform_()

    for block in x:
        scaler.update(block)

    mean = x.reshape((-1, n_dim)).mean(dim=0)
    std = x.reshape((-1, n_dim)).std(dim=0)

    # check that the running computation gives the same result as usual routines
    assert abs(mean - scaler.mean).max() < 1e-5
    assert abs(std - scaler.scale).max() < 1e-5

    # show the result
    print("usual  ", mean, std)
    print("running", scaler.mean, scaler.scale)

    # make sure the scaler works
    y = scaler(x)
    mean = y.reshape((-1, n_dim)).mean(dim=0)
    std = y.reshape((-1, n_dim)).std(dim=0)
    assert abs(mean).max() < 1e-5
    assert abs(std - 1.0).max() < 1e-5
