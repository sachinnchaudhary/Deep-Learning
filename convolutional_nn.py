"""This is implementation of Convolutional nueral network on Mnist dataset which has three convolutional layers and three maxpooling and dense network connected to final layer of maxpooling. 
Dense network has the three layers 100 then 50 and 25 with the output of 10,1 vector which we apply softmax function for image recongnition"""

"""I built this for understanding purpose on using python and numpy only and thus it's too slow :( and not can be used for such other purpose, but it's definitely worth it for understanding the mechanism behind CNN"""

from tensorflow.keras.datasets import mnist
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') / 255

X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255


class Cnn:

     def __init__(self, input, kernel1, kernel2, kernel3):

          self.input = input
          self.kernel1 = kernel1
          self.kernel2 = kernel2
          self.kernel3 = kernel3

          self.weight = np.random.normal(0, np.sqrt(2/10),(10,100))
          self.bias = np.zeros((100,))
          self.weight1 = np.random.normal(0, np.sqrt(2/100),(100, 50))
          self.bias1 = np.zeros((50,))
          self.weight2 = np.random.normal(0, np.sqrt(2/50),(50,25))
          self.bias2 = np.zeros((25,))
          self.weight3 = np.random.normal(0, np.sqrt(2/25),(25,10))
          self.bias3 = np.zeros((10,))
     
     #layer 1 of CNN

     def feature_maps(self):   
            
            filters = np.zeros((25, 25, 10))
            for i in range(25):
                 for j in range(25):
                        for k in range(10):
                             filters[i][j][k] = np.sum(self.input[i:i+4, j:j+4] * self.kernel1[:, :, k])
            
            
            return filters
    
     def Relu(self, alpha=0.01):
           
           feature_maps = self.feature_maps()

           return np.where(feature_maps > 0, feature_maps,alpha * feature_maps)
     
     def pooling(self):
           
             feature_maps = self.Relu()

             maxpool = np.zeros((20,20,10))

             for i in range(20):
                for j in range(20):
                    for k in range(10):
                            maxpool[i][j][k] = np.max(feature_maps[i:i+6, j:j+6, k])

             return maxpool



    #layer 2 of CNN 
     
     def feature_maps1(self):   
            
            filters = np.zeros((15,15,10))
            layer_output = self.pooling()
            for i in range(15):
                 for j in range(15):
                        for k in range(10):
                             filters[i][j][k] = np.sum(layer_output[i:i+6, j:j+6, k] * self.kernel2[:, :, k])
            
            
            return filters
     
     def Relu1(self, alpha=0.01):
           
           feature_maps = self.feature_maps1()

           return np.where(feature_maps > 0, feature_maps,alpha * feature_maps)

     def pooling1(self):

                feature_maps = self.Relu1()
    
                maxpool = np.zeros((10,10,10))
    
                for i in range(10):
                    for j in range(10):
                        for k in range(10):
                                maxpool[i][j][k] = np.max(feature_maps[i:i+6, j:j+6, k])
    
                return maxpool 
                
     def feature_maps2(self):
           
           filters = np.zeros((5,5,10))
           layer_output = self.pooling1()
           
           for i in range(5):
                    for j in range(5):  
                          for k in range(10):
                               filters[i][j][k] = np.sum(layer_output[i:i+6, j:j+6, k] * self.kernel3[:, :, k])

           return filters 
     
     def Relu2(self, alpha=0.01):
           
           feature_maps = self.feature_maps2()

           return np.where(feature_maps > 0, feature_maps,alpha * feature_maps)

     def pooling2(self):
           
           layer_output = self.Relu2()

           maxpool = np.zeros((1,1,10))

           for i in range(1):
                  for j in range(1):
                        for k in range(10):
                              maxpool[i][j][k] = np.max(layer_output[i:i+5, j:j+5, k])

           return maxpool.reshape(-1,1)


     def nn(self, alpha = 0.01, learning_rate= 0.001):  
           
           vector = self.pooling2().flatten()
           self.vector = vector

           #first layer 
           self.for1 = np.dot(vector, self.weight) + self.bias
           output1 = np.where(self.for1 > 0, self.for1, alpha * self.for1)
   
           #second layer
           self.for2 = np.dot(output1, self.weight1) + self.bias1 
           output2 = np.where(self.for2 > 0 , self.for2, alpha * self.for2)

           #third layer
           self.for3 = np.dot(output2, self.weight2) + self.bias2
           output3 = np.where(self.for3 > 0, self.for3, alpha * self.for3)

           #fourth layer
           self.for4 = np.dot(output3, self.weight3) + self.bias3 
           
           logits_max = np.max(self.for4)
           exp_logits = np.exp(self.for4 - logits_max)
           softmax = exp_logits / np.sum(exp_logits)

           probabilities = np.argmax(softmax)
           confidence = np.max(softmax)

           
           return output1, output2, output3, softmax

    
           
     def relu_derivative(self,x ,alpha = 0.001): 

            return np.where(x > 0, 1.0, alpha)


     def convo_backprop(self, grad_vector, learning_rate = 0.001):
      
    
        grad_pooling2_out = grad_vector.reshape(1, 1, 10)
    
    
        layer_output = self.Relu2()  
        grad_relu2_out = np.zeros_like(layer_output)
    
    
        for k in range(10):
       
           max_idx = np.unravel_index(np.argmax(layer_output[:,:,k]), (5,5))
           grad_relu2_out[max_idx[0], max_idx[1], k] = grad_pooling2_out[0, 0, k]
    
  
        feature_maps2 = self.feature_maps2() 
        relu2_derivative = self.relu_derivative(feature_maps2)
        grad_feature2_out = grad_relu2_out * relu2_derivative
    
   
        layer1_output = self.pooling1() 
        grad_kernel3 = np.zeros_like(self.kernel3)  
        grad_pooling1_out = np.zeros_like(layer1_output) 
    
    
        for i in range(5):  
          for j in range(5):  
            for k in range(10): 
                
                input_region = layer1_output[i:i+6, j:j+6, k]
                
                # Gradient w.r.t kernel
                grad_kernel3[:,:,k] += grad_feature2_out[i,j,k] * input_region
                
                # Gradient w.r.t input (for previous layer)
                grad_pooling1_out[i:i+6, j:j+6, k] += grad_feature2_out[i,j,k] * self.kernel3[:,:,k]
    
    
        relu1_output = self.Relu1()  
        grad_relu1_out = np.zeros_like(relu1_output)
     
        for i in range(10):
           for j in range(10):
              for k in range(10):
                
                pool_region = relu1_output[i:i+6, j:j+6, k]
                max_idx = np.unravel_index(np.argmax(pool_region), (6,6))
                grad_relu1_out[i + max_idx[0], j + max_idx[1], k] = grad_pooling1_out[i,j,k]
    
    
        feature_maps1 = self.feature_maps1()  
        relu1_derivative = self.relu_derivative(feature_maps1)
        grad_feature1_out = grad_relu1_out * relu1_derivative
    
    
        pooling_output = self.pooling() 
        grad_kernel2 = np.zeros_like(self.kernel2) 
        grad_pooling_out = np.zeros_like(pooling_output)  
    
        for i in range(15):
           for j in range(15):
               for k in range(10):
               
                input_region = pooling_output[i:i+6, j:j+6, k]
                
                
                grad_kernel2[:,:,k] += grad_feature1_out[i,j,k] * input_region
                
                
                grad_pooling_out[i:i+6, j:j+6, k] += grad_feature1_out[i,j,k] * self.kernel2[:,:,k]
    
   
        relu_output = self.Relu()  
        grad_relu_out = np.zeros_like(relu_output)
    
        for i in range(20):
           for j in range(20):
               for k in range(10):
                
                pool_region = relu_output[i:i+6, j:j+6, k]
                max_idx = np.unravel_index(np.argmax(pool_region), (6,6))
                grad_relu_out[i + max_idx[0], j + max_idx[1], k] = grad_pooling_out[i,j,k]
    
   
        feature_maps = self.feature_maps()  
        relu_derivative = self.relu_derivative(feature_maps)
        grad_feature_out = grad_relu_out * relu_derivative
    
    
        grad_kernel1 = np.zeros_like(self.kernel1)  
        grad_input = np.zeros_like(self.input) 
    
        for i in range(25):
           for j in range(25):
               for k in range(10):
                
                input_region = self.input[i:i+4, j:j+4, 0]  
                
                
                grad_kernel1[:,:,k] += grad_feature_out[i,j,k] * input_region
                
                
                grad_input[i:i+4, j:j+4, 0] += grad_feature_out[i,j,k] * self.kernel1[:,:,k]
    
   
        self.kernel1 -= learning_rate * grad_kernel1
        self.kernel2 -= learning_rate * grad_kernel2
        self.kernel3 -= learning_rate * grad_kernel3
    
        return grad_kernel1, grad_kernel2, grad_kernel3
                             
     def back_prop(self, target, learning_rate = 0.001):

           
        output1, output2, output3,softmax = self.nn()
        loss = -np.log(softmax[target] + 1e-8 )
        
        hot_vector = np.zeros(10)
        hot_vector[target] = 1

        grad_output4 = softmax - hot_vector

        #dense gradients 

        self.grad_w3 = np.outer(output3, grad_output4)
        self.grad_b3 = grad_output4

        grad_output3 = np.dot(grad_output4, self.weight3.T)
        relu_grad3 = self.relu_derivative(self.for3)
        grad_output3 =   grad_output3 * relu_grad3 

        self.grad_w2 = np.outer(output2, grad_output3)
        self.grad_b2 = grad_output3


        grad_output2 = np.dot(grad_output3, self.weight2.T)


        relu_grad2 = self.relu_derivative(self.for2)
        grad_output2 = relu_grad2 * grad_output2 
        
        self.grad_w1 = np.outer(output1, grad_output2)
        self.grad_b1 = grad_output2

        grad_output1 = np.dot(self.weight1, grad_output2)
        
        relu_grad1 = self.relu_derivative(self.for1)
        grad_output1 = relu_grad1 * grad_output1

        self.grad_w = np.outer(self.vector,grad_output1)
        self.grad_b = grad_output1



        #changing paramters 

        self.weight -= learning_rate * self.grad_w
        self.bias -=  learning_rate * self.grad_b

        self.weight1 -= learning_rate * self.grad_w1
        self.bias1 -= learning_rate * self.grad_b1

        self.weight2 -= learning_rate * self.grad_w2
        self.bias2 -= learning_rate * self.grad_b2

        self.weight3 -= learning_rate * self.grad_w3
        self.bias3 -= learning_rate * self.grad_b3


        grad_vector = np.dot(grad_output1, self.weight.T)
        self.convo_backprop(grad_vector, learning_rate)

      
        return loss  
          


input = X_train[1]
target = y_train[1]




def he_init(shape):
    stddev = np.sqrt(2 / np.prod(shape[:-1]))
    return np.random.randn(*shape) * stddev

kernel1 = he_init((4, 4, 10))
kernel2 = he_init((6, 6, 10))
kernel3 = he_init((6, 6, 10))


input_sample = X_train[0]
cnn = Cnn(input_sample, kernel1, kernel2, kernel3)

# Training parameters
num_epochs = 1
learning_rate = 0.001
num_samples = 5000 # Use subset for faster training


print(f"Samples: {num_samples}")
print(f"Epochs: {num_epochs}")
print(f"Learning Rate: {learning_rate}")
print("=" * 50)

# Main training loop
for epoch in range(num_epochs):
    total_loss = 0
    correct_predictions = 0
    
    # Shuffle training data
    indices = np.random.permutation(len(X_train))[:num_samples]
    
    for sample_count, idx in enumerate(indices, 1):
        # Feed new image to CNN
        cnn.input = X_train[idx]
        target = y_train[idx]
        
        # Forward pass - CNN processes the image
        _, _, _, softmax = cnn.nn()
        prediction = np.argmax(softmax)
        
        # Track accuracy
        if prediction == target:
            correct_predictions += 1
        
        # Backward pass - CNN learns from mistakes
        loss = cnn.back_prop(target, learning_rate)
        total_loss += loss
        
        # Progress updates
        if sample_count % 1000 == 0:
            current_acc = correct_predictions / sample_count
            print(f"Sample {sample_count}/{num_samples} | Loss: {loss:.4f} | Accuracy: {current_acc:.4f}")
    
    # End of epoch summary
    final_accuracy = correct_predictions / num_samples
    avg_loss = total_loss / num_samples
    
    print(f"\nEpoch {epoch+1} Complete:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Training Accuracy: {final_accuracy:.4f}")
