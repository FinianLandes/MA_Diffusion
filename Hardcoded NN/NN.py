import numpy as np
from numpy import ndarray
import logging, time, pickle, os
from functions import *

class Layer():
    def __init__(self) -> None:
        self.input: ndarray = None
        self.output: ndarray = None
        self.out_grad: ndarray = None
        self.inp_grad: ndarray = None
    def fwd(self, input: ndarray)  -> ndarray:
        pass
    def bwd(self, output_gradient: ndarray, learning_rate: float) -> ndarray:
        pass
    def log_fwd(self, logger: logging.Logger) -> None:
        logger.debug(f"{self.__class__.__name__} fwd pass: input shape {self.input.shape}, output shape {self.output.shape}")
    def log_bwd(self, logger: logging.Logger) -> None:
        logger.debug(f"{self.__class__.__name__} bwd pass: output grad shape {self.out_grad.shape}, input grad shape {self.inp_grad.shape}")
    def __repr__(self):
        return f"{self.__class__.__name__} Input: {self.input} Output: {self.output}"

class Dense_Layer(Layer):
    def __init__(self, inp_size: ndarray, out_size: ndarray):
        super().__init__()
        self.weights: ndarray = np.random.randn(out_size, inp_size) * np.sqrt(2. / inp_size)
        self.bias: ndarray = np.zeros((out_size, 1))
    def fwd(self, input: ndarray)  -> ndarray:
        self.input = input
        self.output = np.dot(self.weights, self.input) + self.bias
        return self.output
    def bwd(self, output_gradient: ndarray, learning_rate: float) -> ndarray:
        self.out_grad = output_gradient
        self.weights -= np.dot(self.out_grad, self.input.T) * learning_rate
        self.bias -= self.out_grad * learning_rate
        self.inp_grad = np.dot(self.weights.T, self.out_grad)
        return self.inp_grad

class Soft_Max(Layer):
    def fwd(self, input: ndarray) -> ndarray:
        super().__init__()
        self.input = input
        exp_vals: ndarray = np.exp(self.input)
        self.output = exp_vals / np.sum(exp_vals)
        return self.output
    def bwd(self, output_gradient: ndarray, learning_rate: float) -> ndarray:
        self.out_grad = output_gradient
        n: int = np.size(self.output)
        self.inp_grad = np.dot((np.identity(n) - self.output.T) * self.output, self.out_grad)
        return self.inp_grad

class Convolution(Layer):
    def __init__(self, inp_shape: tuple[int,int,int], kernel_size: int, n_filters: int) -> None:
        super().__init__()
        self.inp_depth, self.inp_height, self.inp_width = inp_shape
        self.inp_shape = inp_shape
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.output_shape: ndarray = np.array((n_filters, self.inp_height - kernel_size + 1, self.inp_width - kernel_size + 1))
        self.kernels_shape: ndarray = np.array((n_filters, self.inp_depth, kernel_size, kernel_size))
        self.kernels: ndarray = np.random.randn(*self.kernels_shape)
        self.bias: ndarray = np.random.randn(*self.output_shape)
    def fwd(self, input: ndarray) -> ndarray:
        self.input = input
        self.output = np.zeros(self.output_shape)
        for n in range(self.n_filters):
            self.output[n] = correlate(self.kernels[n], self.input) + self.bias[n]
        return self.output
    def bwd(self, output_gradient: ndarray, learning_rate: float) -> ndarray:
        self.out_grad = output_gradient
        grad_kernels: ndarray = np.zeros(self.kernels_shape)
        self.inp_grad = np.zeros(self.inp_shape)
        for n in range(self.n_filters):
            for d in range(self.inp_depth):
                grad_kernels[n,d] = correlate(self.out_grad[n], self.input[d])
                self.inp_grad += self._correlate(np.rot90(self.kernels[n, d], 2, (0, 1)), self.out_grad[n], "full")
        self.kernels -= learning_rate * grad_kernels
        self.bias -= learning_rate * self.out_grad
        return self.inp_grad

class Up_Convolution(Layer):
    def __init__(self, inp_shape: tuple[int,int,int], kernel_size: int, n_filters: int ) -> None:
        super().__init__()
        self.inp_depth, self.inp_height, self.inp_width = inp_shape
        self.inp_shape = inp_shape
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.kernels_shape: ndarray = np.array((n_filters, self.inp_depth, kernel_size, kernel_size))
        self.kernels: ndarray = np.random.randn(*self.kernels_shape)
        self.output_shape = np.array((self.n_filters, self.inp_height - 1 + self.kernel_size, self.inp_width - 1 + self.kernel_size))
        self.bias: ndarray = np.random.randn(*self.output_shape)
    def fwd(self, input: ndarray)  -> ndarray:
        self.input = input
        self.output = np.zeros(self.output_shape)
        for n in range(self.n_filters):
            self.output[n] = t_conv(self.kernels[n], self.input) + self.bias[n]
        return self.output
    def bwd(self, output_gradient: ndarray, learning_rate: float) -> ndarray:
        self.out_grad = output_gradient
        kernels_grad = np.zeros_like(self.kernels)
        bias_grad = np.zeros_like(self.bias)
        for n in range(self.n_filters):
            for d in range(self.inp_depth):
                kernels_grad[n, d] = correlate(self.out_grad[n], self.inp[d], mode="valid")
            bias_grad[n] = np.sum(self.out_grad[n])
            self.inp_grad += t_conv(np.flip(self.kernels[n], axis=(1, 2)), self.out_grad[n][np.newaxis, :, :])
        self.kernels -= learning_rate * kernels_grad
        self.bias -= learning_rate * bias_grad
        return self.inp_grad
        

class Max_Pool(Layer):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size
    def _pool(self, input: ndarray) -> ndarray:
        output: ndarray = np.zeros((input.shape[0] // self.size, input.shape[1] // self.size))
        for i in range(input.shape[0] // self.size):
            for j in range(input.shape[1] // self.size):
                patch: ndarray = input[i * self.size: i * self.size + self.size, j * self.size: j * self.size + self.size]
                output[i,j] = np.max(patch)
        return output
    def fwd(self, input: ndarray) -> ndarray:
        self.input = input
        depth, h, w = input.shape
        p_h: int = h % self.size
        p_w: int = w % self.size
        self.input = np.pad(self.input, ((0,0), (p_h, 0), (p_w,0)))
        self.output: ndarray = np.zeros((depth, (h + p_h) // self.size, (w + p_w) // self.size))
        for d in range(depth):
            self.output[d] = self._pool(self.input[d])
        return self.output
    def bwd(self, output_gradient: ndarray, learning_rate: float) -> ndarray:
        self.out_grad = output_gradient
        depth, h, w = self.input.shape
        p_h: int = (self.size - h % self.size) % self.size
        p_w: int = (self.size - w % self.size) % self.size
        self.inp_grad = np.zeros_like(self.input)
        for d in range(depth):
            for i in range(h // self.size):
                for j in range(w // self.size):
                    patch: ndarray = self.input[d, i * self.size: i * self.size + self.size, j * self.size: j * self.size + self.size]
                    max_val = np.max(patch)
                    for m in range(self.size):
                        for n in range(self.size):
                            if patch[m, n] == max_val:
                                self.inp_grad[d, i * self.size + m, j * self.size + n] = self.out_grad[d, i, j]
        self.inp_grad = self.inp_grad[:, :h - p_h, :w - p_w]
        return self.inp_grad

class Reshape(Layer):
    def __init__(self, input_shape: ndarray, output_shape: ndarray) -> None:
        super().__init__()
        self.inp_shape = input_shape
        self.out_shape = output_shape
    def fwd(self, input: ndarray)  -> ndarray:
        self.input = input
        self.output = np.reshape(self.input, self.out_shape)
        return self.output
    def bwd(self, output_gradient: ndarray, learning_rate: float) -> ndarray:
        self.out_grad = output_gradient
        self.inp_grad = np.reshape(self.out_grad, self.inp_shape)
        return self.inp_grad

class Activation(Layer):
    def __init__(self, activation: callable, activation_prime: callable) -> None:
        super().__init__()
        self.activation: callable = activation
        self.activation_prime: callable = activation_prime
    def fwd(self, input: ndarray) -> ndarray:
        self.input = input
        self.output = self.activation(self.input)
        return self.output
    def bwd(self, output_gradient: ndarray, learning_rate: float) -> ndarray:
        self.out_grad = output_gradient
        self.inp_grad = np.multiply(output_gradient, self.activation_prime(self.input))
        return self.inp_grad

class ReLu(Activation):
    def __init__(self):
        activation: callable = relu
        activation_prime: callable = relu_prime
        super().__init__(activation, activation_prime)

class Tanh(Activation):
    def __init__(self) -> None:
        activation: callable = tanh
        activation_prime: callable = tanh_prime
        super().__init__(activation, activation_prime)

class Sigmoid(Activation):
    def __init__(self) -> None:
        activation: callable = sigmoid
        activation_prime: callable = sigmoid_prime
        super().__init__(activation, activation_prime)

class Loss():
    def __init__(self, loss: callable, loss_prime: callable) -> None:
        self.loss_function = loss
        self.loss_function_prime = loss_prime
    def loss(self, y_pred: ndarray, y_true: ndarray) -> float:
        return self.loss_function(y_pred, y_true)
    def loss_prime(self, y_pred: ndarray, y_true: ndarray) -> float:
        return self.loss_function_prime(y_pred, y_true)

class MSE(Loss):
    def __init__(self) -> None:
        self.loss_function: callable = mse
        self.loss_function_prime: callable = mse_prime
        super().__init__(self.loss_function, self.loss_function_prime)

class Model():
    def __init__(self, model_name: str, layers: list[Layer], loss_function: Loss, learning_rate: float = 0.001, log_level: int = logging.INFO, save_model: bool = False) -> None:
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.save_model = save_model
        self.layers = layers
        self.model_name = model_name
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        if self.model_name and os.path.exists(self.model_name):
            m:'Model' = self._load(self.model_name)
            self.__dict__.update(m.__dict__)
        else:
            self.logger.info("Initialized new model")
    def train(self, train_data: ndarray, train_label: ndarray, epochs: int = 100, batch_size: int = 1) -> float:
        data_size: int = train_data.shape[0]
        self.logger.info(f"Training started")
        for e in range(epochs):
            error: float = 0
            start: float = time.time()
            for data, label in zip(train_data, train_label):
                output: ndarray = self._fwd_prop(data)
                ce: float = self.loss_function.loss(label, output)
                error += ce
                loss_prime: ndarray = self.loss_function.loss_prime(output, label)
                self.logger.debug(f"Loss: {ce} loss prime shape: {loss_prime.shape}")
                self._bwd_prop(loss_prime)
            period: float = round((epochs - e - 1) * (time.time() - start))
            self.logger.info(f"Epoch: {e + 1:02d} Loss: {round(error / data_size, 5):.5f} Time remaining: {period // 3600:02d}h {(period % 3600) // 60:02d}min {round(period % 60):02d}s")
        self.logger.info(f"Training done")
        if self.save_model:
            self._save(self.model_name)
        return error / data_size
    def _fwd_prop(self, input: ndarray) -> ndarray:
        output: ndarray = input
        for layer in self.layers:
            output = layer.fwd(output)
            layer.log_fwd(self.logger)
        return output
    def _bwd_prop(self, output_grad: ndarray) -> None:
        output: ndarray = output_grad
        for layer in reversed(self.layers):
            output = layer.bwd(output, self.learning_rate)
            layer.log_bwd(self.logger)
        return output
    def classification_test(self, test_data: ndarray, test_label: ndarray) -> float:
        n_correct: int = 0
        for data, label in zip(test_data, test_label):
            result = self._fwd_prop(data)
            n_correct += np.argmax(result) == np.argmax(label)
        res: float = n_correct / len(test_data)
        self.logger.info(f"Successrate: {res * 100:.2f}% ")
        return res
    def _save(self, filename) -> bool:
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        self.logger.info(f"Model {filename} saved")
    @staticmethod
    def _load(filename: str) -> 'Model':
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        model.logger.info(f"Model {filename} loaded")
        return model

