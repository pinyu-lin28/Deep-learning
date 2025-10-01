import numpy as np

## by yourself .Finish your own NN framework
## Just an example.You can alter sample code anywhere.

#This class is only for establishing the basic structure of following layer
class _Layer(object):
    def __init__(self):
        pass

    def forward(self, *input):
        r"""Define the forward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def backward(self, *output_grad):
        r"""Define the backward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError


class ConvolutionLayer(_Layer):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(ConvolutionLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 初始化卷積核 (W) 和偏置 (b)
        # 權重W的維度為(out_channels, in_channels, kernel_size, kernel_size)
        # 偏置b的維度為(out_channels, 1)
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        self.b = np.zeros((out_channels, 1))

    def forward(self, input):
        # input.shape is (batch_size, channels, height, width)
        batch_size, channels, height, width = input.shape
        self.input = input

        # Calculate output dimensions
        out_h = int((height + 2 * self.padding - self.kernel_size) / self.stride) + 1
        out_w = int((width + 2 * self.padding - self.kernel_size) / self.stride) + 1
        output = np.zeros((batch_size, self.out_channels, out_h, out_w))

        # Padding
        if self.padding > 0:
            padded_input = np.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                                  'constant')
        else:
            padded_input = input

        # Batch-wise convolution
        for b in range(batch_size):
            for i in range(self.out_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        input_slice = padded_input[b, :, h_start:h_end, w_start:w_end]
                        output[b, i, h, w] = np.sum(input_slice * self.W[i]) + self.b[i]
        return output

    def backward(self, output_grad):
        # output_grad.shape is (batch_size, out_channels, out_h, out_w)
        batch_size, out_channels, out_h, out_w = output_grad.shape
        input_grad = np.zeros_like(self.input)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        # Batch-wise backpropagation
        for b in range(batch_size):
            for i in range(out_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        input_slice = self.input[b, :, h_start:h_end, w_start:w_end]
                        dW[i] += input_slice * output_grad[b, i, h, w]
                        input_grad[b, :, h_start:h_end, w_start:w_end] += self.W[i] * output_grad[b, i, h, w]

        db = np.sum(output_grad, axis=(0, 2, 3))
        self.dW = dW / batch_size
        self.db = db / batch_size
        return input_grad


## by yourself .Finish your own NN framework
class FullyConnected(_Layer):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        # 隨機初始化權重矩陣W和偏置向量b
        # 權重W的維度為(out_features, in_features)，偏置b的維度為(out_features, 1)
        self.W = np.random.randn(out_features, in_features) * 0.01
        self.b = np.zeros((out_features, 1))

    def forward(self, input):
        self.input = input
        # 線性變換: Z = W * X + b
        output = self.W.dot(self.input) + self.b

        return output

    def backward(self, output_grad):
        # output_grad來自下一層的反向傳播
        # 計算權重W的梯度
        dW = output_grad.dot(self.input.T)
        # 計算偏置b的梯度
        db = np.sum(output_grad, axis=1, keepdims=True)
        # 計算輸入input的梯度，並傳遞給前一層
        input_grad = self.W.T.dot(output_grad)

        # 在訓練時，你需要用梯度dW和db來更新權重W和偏置b
        self.dW = dW
        self.db = db

        return input_grad

## by yourself .Finish your own NN framework
class Activation1(_Layer):
    def __init__(self, activation_type):
        self.activation_type = activation_type
        # 為了反向傳播，儲存forward的輸入
        self.input = None
        pass

    def ReLU(self, Z):
        self.input = Z
        return np.maximum(Z, 0)

    def ReLU_backward(self, output_grad):
        grad = output_grad.copy()
        grad[self.input <= 0] = 0
        return grad

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    def forward(self, input):
        if self.activation_type == 'ReLU':
            return self.ReLU(input)
        elif self.activation_type == 'softmax':
            return self.softmax(input)
        else:
            raise NotImplementedError

    def backward(self, output_grad):
        if self.activation_type == 'ReLU':
            return self.ReLU_backward(output_grad)
        elif self.activation_type == 'softmax':
            # SoftmaxWithloss 已經處理了梯度計算，所以這裡可以留空
            pass
        else:
            raise NotImplementedError

class SoftmaxWithloss(_Layer):
    def __init__(self):
        pass

    def forward(self, input, target):
        # 儲存輸入input和真實標籤target，用於反向傳播
        self.input = input
        self.target = target

        # Softmax 計算
        exp_input = np.exp(input - np.max(input, axis=0, keepdims=True))
        predict = exp_input / np.sum(exp_input, axis=0, keepdims=True)

        # 交叉熵損失計算
        # 為了避免 log(0) 的情況，可以給預測值加上一個微小的epsilon
        epsilon = 1e-12
        loss = -np.sum(self.target * np.log(predict + epsilon)) / self.target.shape[1]

        # 儲存預測結果
        self.predict = predict

        return predict, loss

    def backward(self):
        # 合併Softmax和Cross-Entropy Loss的反向傳播，
        # 梯度計算公式非常簡潔: dL/dZ = A - Y
        input_grad = self.predict - self.target
        # 因為損失是平均的，所以梯度也需要平均
        input_grad = input_grad / self.target.shape[1]

        return input_grad


class Flatten(_Layer):
    def __init__(self):
        self.original_shape = None

    def forward(self, input):
        # 儲存原始形狀以供反向傳播使用
        self.original_shape = input.shape
        # 將輸入展平為一維向量
        output = input.reshape(input.shape[0], -1)
        return output

    def backward(self, output_grad):
        # 將梯度從一維向量恢復回原始形狀
        return output_grad.reshape(self.original_shape)

class Reshape(_Layer):
    def __init__(self, target_shape):
        self.target_shape = target_shape
        self.original_shape = None

    def forward(self, input):
        self.original_shape = input.shape
        # 重塑輸入，並將 batch_size 維度保留
        batch_size = input.shape[0]
        output_shape = (batch_size,) + self.target_shape
        output = input.reshape(output_shape)
        return output

    def backward(self, output_grad):
        # 將梯度重塑回原始的 2D 形狀
        return output_grad.reshape(self.original_shape)


#############
class Reshape(_Layer):
    def __init__(self, target_shape):
        self.target_shape = target_shape
        self.original_shape = None

    def forward(self, input):
        #Input shape:(batch size, 784)
        self.original_shape = input.shape
        batch_size = input.shape[0]
        #Output shape:(channels, height, width, batch_size)
        output_shape =  self.target_shape + (batch_size, )
        output = input.reshape(output_shape)
        return output

    def backward(self, output_grad):
        return output_grad.reshape(self.original_shape)