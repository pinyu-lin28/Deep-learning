import numpy as np


# This class is only for establishing the basic structure of following layer
class _Layer(object):
    def __init__(self):
        pass

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *output_grad):
        raise NotImplementedError


class ConvolutionLayer(_Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvolutionLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # W shape is (out_channels, in_channels, kernel_size, kernel_size)
        # W 的形狀為 (輸出通道數, 輸入通道數, 卷積核高度, 卷積核寬度)
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        self.b = np.zeros((out_channels, 1))
        self.input = None

    def forward(self, input):
        # input shape: (in_channels, height, width, batch_size)
        in_channels, height, width, batch_size = input.shape
        self.input = input

        out_h = int((height + 2 * self.padding - self.kernel_size) / self.stride) + 1
        out_w = int((width + 2 * self.padding - self.kernel_size) / self.stride) + 1

        if self.padding > 0:
            padded_input = np.pad(input, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                                  'constant')
        else:
            padded_input = input

        # 向量化前向傳播
        # Im2col: 將輸入圖像轉換為一個矩陣，每一列都是一個卷積核的接收域
        patches = np.lib.stride_tricks.as_strided(
            padded_input,
            shape=(in_channels, out_h, out_w, self.kernel_size, self.kernel_size, batch_size),
            strides=(
                padded_input.strides[0],
                self.stride * padded_input.strides[1],
                self.stride * padded_input.strides[2],
                padded_input.strides[1],
                padded_input.strides[2],
                padded_input.strides[3]
            )
        )

        # 將權重 (W) 重新塑形為 2D 矩陣 (out_channels, in_features)
        W_reshaped = self.W.reshape(self.out_channels, -1)

        # 將 patches 重新塑形為 (in_features, batch_size, out_h, out_w)
        patches_reshaped = patches.transpose(2, 3, 0, 1, 4, 5).reshape(
            self.in_channels * self.kernel_size * self.kernel_size, batch_size * out_h * out_w)

        # 執行矩陣乘法，並加上偏置
        output = W_reshaped.dot(patches_reshaped) + self.b
        output = output.reshape(self.out_channels, out_h, out_w, batch_size)

        return output

    def backward(self, output_grad):
        out_channels, out_h, out_w, batch_size = output_grad.shape
        in_channels, height, width, _ = self.input.shape

        # 初始化梯度
        dW = np.zeros_like(self.W)
        input_grad = np.zeros_like(self.input)

        # 向量化反向傳播
        if self.padding > 0:
            padded_input = np.pad(self.input,
                                  ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                                  'constant')
            padded_input_grad = np.zeros_like(padded_input)
        else:
            padded_input = self.input
            padded_input_grad = np.zeros_like(padded_input)

        for b in range(batch_size):
            for i in range(out_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size

                        input_slice = padded_input[:, h_start:h_end, w_start:w_end, b]
                        dW[i] += input_slice * output_grad[i, h, w, b]
                        padded_input_grad[:, h_start:h_end, w_start:w_end, b] += self.W[i] * output_grad[i, h, w, b]

        if self.padding > 0:
            input_grad = padded_input_grad[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            input_grad = padded_input_grad

        db = np.sum(output_grad, axis=(1, 2, 3), keepdims=True)
        self.dW = dW / batch_size
        self.db = db / batch_size
        return input_grad



class FullyConnected(_Layer):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.W = np.random.randn(out_features, in_features) * 0.01
        self.b = np.zeros((out_features, 1))

    def forward(self, input):
        self.input = input
        output = self.W.dot(self.input) + self.b
        return output

    def backward(self, output_grad):
        output_grad = output_grad
        dW = output_grad.dot(self.input.T)
        db = np.sum(output_grad, axis=1, keepdims=True)
        input_grad = self.W.T.dot(output_grad)

        self.dW = dW
        self.db = db
        return input_grad


class Activation1(_Layer):
    def __init__(self, activation_type):
        self.activation_type = activation_type
        self.input = None

    def ReLU(self, Z):
        self.input = Z
        return np.maximum(Z, 0)

    def ReLU_backward(self, output_grad):
        grad = output_grad.copy()
        grad[self.input <= 0] = 0
        return grad


    def forward(self, input):
        if self.activation_type == 'ReLU':
            return self.ReLU(input)
        else:
            raise NotImplementedError

    def backward(self, output_grad):
        if self.activation_type == 'ReLU':
            return self.ReLU_backward(output_grad)
        elif self.activation_type == 'softmax':
            pass
        else:
            raise NotImplementedError


class SoftmaxWithloss(_Layer):
    def __init__(self):
        pass

    def forward(self, input, target):
        self.input = input
        self.target = target

        exp_input = np.exp(input - np.max(input, axis=0, keepdims=True))
        predict = exp_input / np.sum(exp_input, axis=0, keepdims=True)

        epsilon = 1e-12
        loss = -np.sum(self.target * np.log(predict + epsilon)) / self.target.shape[1]

        self.predict = predict
        return predict, loss

    def backward(self):
        input_grad = self.predict - self.target
        input_grad = input_grad / self.target.shape[1]
        return input_grad


class Flatten(_Layer):
    def __init__(self):
        self.original_shape = None

    def forward(self, input):
        self.original_shape = input.shape
        batch_size = input.shape[3]
        output = input.reshape(-1, batch_size)
        return output

    def backward(self, output_grad):
        return output_grad.reshape(self.original_shape)


class Reshape(_Layer):
    def __init__(self, target_shape):
        self.target_shape = target_shape
        self.original_shape = None

    def forward(self, input):
        # Input shape: (batch_size, 784)
        self.original_shape = input.shape
        batch_size = input.shape[0]
        # Reshape to (in_channels, height, width, batch_size)
        output = input.T.reshape(self.target_shape + (batch_size,))
        return output

    def backward(self, output_grad):
        # output_grad shape: (channels, height, width, batch_size)
        return output_grad.reshape(self.original_shape).T