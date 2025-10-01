from .layer import *

class Network(object):
    def __init__(self, in_channels, conv_out_channels, kernel_size, num_classes, fc_hidden_size):
        self.layers = []
        # Define the first layer: A convolution layer with ReLU activation
        self.layers.append(Reshape((in_channels, 28, 28)))
        self.layers.append(ConvolutionLayer(in_channels, conv_out_channels, kernel_size, stride=1, padding=1))
        self.layers.append(Activation1('ReLU'))

        #flatten the output
        conv_output_height = (28 + 2 * 1 - kernel_size) // 1 + 1
        conv_output_width = (28 + 2 * 1 - kernel_size) // 1 + 1
        flatten_size = conv_out_channels * conv_output_height * conv_output_width
        self.layers.append(Flatten())

        # Define the hidden layer
        self.layers.append(FullyConnected(flatten_size, fc_hidden_size))

        #Activation function
        self.layers.append(Activation1('ReLU'))

        # Define the output layer
        self.layers.append(FullyConnected(fc_hidden_size, num_classes))
        # The loss layer is handled separately as it combines softmax and cross-entropy
        self.loss_layer = SoftmaxWithloss()

        ## by yourself .Finish your own NN framework
        ## Just an example.You can alter sample code anywhere.

    def forward(self, input, target):
        ## by yourself .Finish your own NN framework
        self.input = input  # Store the initial input for the backward pass
        self.target = target.T

        # Pass the input sequentially through all layers
        current_output = input
        for layer in self.layers:
            current_output = layer.forward(current_output)
            print('finish running layer: ', layer)
            print('current output shape: ', current_output.shape)

        # Calculate the final prediction and loss using the dedicated loss layer
        pred, loss = self.loss_layer.forward(current_output, target.T)
        return pred, loss

    def backward(self):
        # Start with the gradient from the loss layer
        grad = self.loss_layer.backward()

        # Propagate the gradient back through the layers in reverse order
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            print('Finish running layer: ', layer)
            print('current grad shape: ', grad.shape)


    def update(self, lr):
        ## by yourself .Finish your own NN framework
        for layer in self.layers:
            # Check if the layer has parameters (like a FullyConnected layer)
            if isinstance(layer, FullyConnected):
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db
        
        
