import torch

tensor_2d = torch.tensor([[1., 1., 1., 1., 1.],
                          [2., 2., 2., 2., 2.],
                          [3., 3., 3., 3., 3.]])

tensor_3d = torch.tensor([[[1., 1., 1., 1., 1.],
                            [2., 2., 2., 2., 2.],
                            [3., 3., 3., 3., 3.]],

                             [[4., 4., 4., 4., 4.],
                            [5., 5., 5., 5., 5.],
                            [6., 6., 6., 6., 6.]]])




print(tensor_2d)

print(tensor_3d)


tensor_2d_transposed = tensor_2d.transpose(0, 1)
print(tensor_2d_transposed)

tensor_2d_view = tensor_2d.view(5, 3)
print(tensor_2d_view)

tensor_2d_reshape = tensor_2d.reshape(5, 3)
print(tensor_2d_reshape)

concatenated_tensor = torch.cat((tensor_2d, tensor_2d), dim=0)
stacked_tensor = torch.stack((tensor_2d, tensor_2d), dim=0)
print(concatenated_tensor)
print(stacked_tensor)