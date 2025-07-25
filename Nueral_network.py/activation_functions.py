import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#impelementation of all important activation function to play with it.

#sigmoid function....................................
class Sigmoid:

     def __init__(self, x):

        self.x = x

     def sig(self, x):

        return 1 / (1 + np.exp(- x))


sigm = Sigmoid(x = 0)

print(sigm.sig(x = 0.9999999999))

#ReLu...................................................

class Relu:

    def __init__(self, x):

       self.x  = x

    def relu(self, x):

          if x < 0: 
           
            return 0
 
          else: 

            return x 




rl = Relu(-4)

print(rl.relu(-4))       


#softmax.......................................................

class Softmax:
 

   def __init__(self, v):

     self.v  = v

      
   def softmx(self, v):

       for vc in v:
          
          sum = 0

          sum += (np.exp(vc)) 
          
          

       
       for vct in v:
        
         print(((np.exp(vct)))  / sum)


softmax = Softmax(np.random.rand(5,1))



class Softmax: 
   
   def __init__(self, v):
            
         self.v = v


   def softmx(self, v):

       sm = []
       for vc in v:
  
          sm.append((np.exp(vc)))
   
       for vct in v: 

           print(np.exp(vct) / (np.sum(sm)))
 

array = np.random.rand(5,1)

softmax = Softmax(array)
softmax.softmx(array)
