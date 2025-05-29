import torch as tc
import numpy as np

class Tutorial:

    def __init__(self):
        pass


    def is_cuda_available():
        print("-" * 100) 
        print(f"--> CUDA: {tc.cuda.is_available()}")


    def run(self): 
        print("-" * 100) 
        
        tensor_empty = tc.empty(2, 3)
        tensor_zeros = tc.zeros(2, 3)
        tensor_ones = tc.ones(2, 3, dtype=tc.int)

        tensor_ = tc.tensor([2.5, 0.1])

        # print(f'-> Empty: {tensor_empty}')
        # print(f'-> Zeros: {tensor_zeros}')
        # print(f'-> Ones: {tensor_ones}')
# 
        # print(f'-> Dtype: {tensor_ones.dtype}')
        # print(f'-> Size: {tensor_ones.size()}')

          # _ inplace operation
        x = tc.rand(3, 5)
        y = tc.rand(3, 5)

        z = x + y
        z = y.add_(x)
        z = tc.add(x, y)

        z = x * y
        z = y.mul(x)
        z = tc.mul(x, y)

        print(x)
        print("-" * 100) 
        print(y)
        print("-" * 100) 
        print(z)
        print("-" * 100) 

        print(z[:, 0]) # Todas as linhas da primeira coluna
        print("-" * 100) 
        print(z[1, :]) # Todos os itens da linha 2
        print("-" * 100) 
        print(z[2, 3].item())

        print("Tensor --> Numpy")
        a = tc.ones(2, 5)
        b = a.numpy()
        print(a)
        print(b)
        
        a.add_(1)
        print(a)
        print(b)

        print("Numpy --> Tensor")
        aa = np.ones(5)
        bb = tc.from_numpy(aa)
        print(aa)
        print(bb)


Tutorial.is_cuda_available()
Tutorial().run()
print("-" * 100) 