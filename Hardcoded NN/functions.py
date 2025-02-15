import numpy as np
def sigmoid(x: ndarray) -> ndarray:
            return 1 / (1 + np.exp(-x))
def sigmoid_prime(x: ndarray) -> ndarray:
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x: ndarray) -> ndarray:
            return np.tanh(x)
def tanh_prime(x: ndarray) -> ndarray:
    return 1 - np.tanh(x) ** 2

def relu(x: ndarray) -> ndarray:
            return np.maximum(0, x)
def relu_prime(x: ndarray) -> ndarray:
    return np.where(x > 0, 1, 0)

def mse(x: ndarray, y: ndarray) -> float:
        return np.sum((x - y) ** 2) / y.shape[0]

def mse_prime(x: ndarray, y: ndarray) -> ndarray:
        return 2 * (x - y) / y.shape[0]

def one_hot_encode(labels: ndarray, num_classes: ndarray) -> ndarray:
    return np.eye(num_classes)[labels].reshape(-1, 10, 1)

def correlate(kernel: ndarray, input: ndarray, mode: str = "valid") -> ndarray:
    if input.ndim == 2:
        input = np.expand_dims(input, axis=0)
    if kernel.ndim == 2:
        kernel = np.expand_dims(kernel, axis=0)
    kernel_size = kernel.shape[1]
    inp_h = input.shape[-2]
    inp_w = input.shape[-1]
    if mode == "valid":
        output: ndarray = np.zeros((inp_h - kernel_size + 1, inp_w - kernel_size + 1))
    if mode == "full":
        output: ndarray = np.zeros((inp_h + kernel_size - 1, inp_w + kernel_size - 1))
        input = np.pad(input, ((0,0), (kernel_size - 1, kernel_size - 1), (kernel_size - 1, kernel_size - 1)))
    for r in range(output.shape[0]):
        for c in range(output.shape[1]):
            r_e, c_e = r + kernel_size, c + kernel_size
            patch: ndarray = input[:, r:r_e, c:c_e]
            output[r, c] = np.sum(patch * kernel)
    return output

def t_conv(kernel: ndarray, input: ndarray) -> ndarray:
        _, inp_h, inp_w = input.shape
        kernel_size: int = kernel.shape[-1]
        output = np.zeros((inp_h - 1 + kernel_size, inp_w - 1 + kernel_size))
        for r in range(inp_h):
            for c in range(inp_w):
                print(input[:, r, c].shape, kernel.shape)
                patch: ndarray = np.sum(input[:, r, c] * kernel, axis = 0)
                output[r * kernel_size: (r + 1) * kernel_size , c * kernel_size: (c + 1) * kernel_size] += patch
        return output