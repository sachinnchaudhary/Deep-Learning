import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#this is primaryly for understanding multilayer perceptron, forward propagation, backward propagation, gradient descent
#this is the implementation of vanila neural network though it is designed to understand what is happening in the neural network.
class Nueralnetwork:

      def  __init__(self, inputs, weight):

         self.inputs = np.random.rand(inputs, 1) 
         self.weight = np.random.rand(inputs, 1)
         self.bias = np.zeros((1,1))
         self.loss_history = []

      def forward_prop(self):

          output = np.dot(self.inputs.flatten(), self.weight.flatten()) + self.bias 

          output = 1 / (1 + np.exp(-output))
     
          return output 

      def compute_loss(self):

          output = self.forward_prop()

          loss = (output - 1) * (output - 1)

          return loss

      def back_prop(self, learning_rate = 0.01):

          output = self.forward_prop()
          
          der_loss = 2 * (output - 1)

          sig_der = output *  (1 - output) 

          gradients = der_loss * sig_der * self.inputs.flatten()
          
          

          
          self.weight = self.weight.flatten() - learning_rate  * gradients
          
          self.weight = self.weight.reshape(-1,1)
          

          self.bias  = self.bias - learning_rate * der_loss


      def train(self, epochs= 10000, learning_rate = 0.01):

          for epoch in range(epochs): 

              output = self.forward_prop()

              loss = self.compute_loss() 
              self.loss_history.append(loss[0][0])

              parameters_update = self.back_prop()  

              if epoch % 10 == 0 :

                print(f"epoch: {epoch} : and loss is {loss}")


      def plot_loss(self):
        # Set seaborn style
        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 7))
        
        # Create seaborn line plot
        epochs = range(len(self.loss_history))
        sns.lineplot(x=list(epochs), y=self.loss_history, linewidth=2.5, color='#2E86AB')
        
        plt.title('Neural Network Training Loss Over Time', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=14, fontweight='bold')
        plt.ylabel('Loss', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


nn = Nueralnetwork(10,10)
nn.forward_prop()
nn.compute_loss()
nn.back_prop()
nn.train()
nn.plot_loss()
