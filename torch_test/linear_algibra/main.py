import torch


def print_basic_param():
    x = torch.tensor(3.0)
    y = torch.tensor(2.0)
    z = torch.tensor(4, dtype=torch.float32)
    x_d1 = torch.arange(4)
    print(x_d1.shape)
    print(x_d1[0], x_d1[0].shape)

    x_d2 = torch.arange(20).reshape(4, 5)
    print(x_d2)
    print(x_d2.T)

    x_d2_symmetric = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
    print(x_d2_symmetric == x_d2_symmetric.T)

    x_d2_calculate1 = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    x_d2_calculate2 = x_d2_calculate1.clone()
    print(x_d2_calculate2, x_d2_calculate1 + x_d2_calculate2)

    # sum
    print(x_d2_calculate1.sum())

    # dimension shape
    print(x_d2_calculate1.shape[0])

    # sum dimension
    x_d2_without_axis = x_d2_calculate1.sum(axis=0)
    print(x_d2_without_axis.shape, x_d2_without_axis)

    # mean dimension
    x_d2_mean_without_axis = x_d2_calculate1.mean(axis=0)
    print(x_d2_mean_without_axis.shape, x_d2_mean_without_axis)

    # sum dimension neglect
    x_d2_with_axis = x_d2_calculate1.sum(axis=0, keepdims=True)
    print(x_d2_with_axis, x_d2_with_axis.shape)

    x_d2_xums = x_d2_calculate1.cumsum(axis=0)
    print(x_d2_xums, x_d2_xums.shape)

    x_d2_sum_deux_dim = x_d2_calculate1.sum(axis=(0,1))
    print(x_d2_sum_deux_dim)


def broadcast_instance():
    return


def print_advanced_test():
    # vector times vector
    x = torch.randn(4, dtype=torch.float32)
    y = torch.randn(4, dtype=torch.float32)

    print(x, y, torch.dot(x, y))
    print(torch.dot(x, y) == torch.sum(x * y))

    # vector times matrix
    x_d2 = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    x_d1 = torch.randn(4, dtype=torch.float32)
    print(torch.mv(x_d2, x_d1))

    # matrix times matrix
    x_d2_mm1 = torch.arange(25, dtype=torch.float32).reshape(5, 5)
    x_d2_mm2 = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    print(torch.mm(x_d2_mm1, x_d2_mm2))
    print(torch.norm(x_d2_mm1))
    return


if __name__ == '__main__':
    print_basic_param()
    broadcast_instance()
    print_advanced_test()
