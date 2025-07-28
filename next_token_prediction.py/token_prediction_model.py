"""This is 2.75 parameter vanilla nueral network which i trained on 55000+ tokens(i collecteed this data from various stories) i implemented this using only python and numpy without any libraries like pytorch. 
    Sole aim of training this type of nueral network is to get foundational understanding that what is happening behind this neural network and to compute it without any libraries."""



import numpy as np
import re
from collections import Counter


from gensim.test.utils import common_texts
from gensim.models import Word2Vec



def safe_tokenize(text, max_chunk_size=1000000):  # 1MB chunks
    
    all_sentences = []
    
    # Process text in chunks
    for i in range(0, len(text), max_chunk_size):
        chunk = text[i:i + max_chunk_size].lower()
        sentences = chunk.split('.')
        
        for sentence in sentences:
            tokens = re.findall(r"\b\w+\b", sentence)
            if len(tokens) > 3:
                all_sentences.append(tokens)
    
    return all_sentences

path = r"your file path"
text  = Path(path).read_text(encoding="utf‑8")
tokenized_sentences = safe_tokenize(text)

model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,    
    window=5,           
    min_count=2,        
    workers=4,
    epochs = 50 )  

    

#neural network for next token prediction.............................

class Nueralnetwork:

    def __init__(self, inputs,final_vecs):

        #initializing weights and bias
      
        self.inputs = inputs
        self.weight = np.random.normal(0, np.sqrt(2/3), (100, 2000))
        self.bias = np.zeros((1, 2000))
        self.weight2 = np.random.normal(0, np.sqrt(2/2000),(2000, 1000))
        self.bias2 = np.zeros((1, 1000))
        self.weight3 = np.random.normal(0, np.sqrt(2/1000) , (1000, 500))
        self.bias3 = np.zeros((1, 500))
        self.weight4 = np.random.normal(0, np.sqrt(2/500) , (500, 100))
        self.bias4 = np.zeros((1, 100))
        
        
        self.embedding_matrix = final_vecs
        self.loss_history = []

        self.cache = {}

    #batch normalization for making weights stable if needed.
  
    def batch_normalization(self, x, epsilon=1e-8):
        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)
        return (x - mean) / np.sqrt(var + epsilon) 
    
    def relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1.0, alpha)

    #feed forward propagation 
  
    def forward_prop(self, alpha=0.01):

        if len(self.inputs.shape) == 1:
            inputs = self.inputs.reshape(1, -1)
        else:
            inputs = self.inputs
        
        weight1 = np.dot(inputs, self.weight) + self.bias
        act_f1 = np.where(weight1 > 0, weight1, alpha * weight1)
        
        
        weight2 = np.dot(act_f1,self.weight2) + self.bias2
        act_f2 = np.where(weight2 > 0, weight2, alpha * weight2)
        
  
        weight3 = np.dot(act_f2, self.weight3) + self.bias3
        act_f3 = np.where(weight3 > 0, weight3, alpha * weight3)
        

        weight4 = np.dot(act_f3, self.weight4) + self.bias4
        act_f4 = np.where(weight4 > 0, weight4, alpha * weight4)
        

        
        
        
        logits = np.dot(act_f4, self.embedding_matrix.T)

        self.cache = {
            'inputs': self.inputs,
            'z1': weight1, 'a1': act_f1,
            'z2': weight2, 'a2': act_f2, 
            'z3': weight3, 'a3': act_f3,
            'z4': weight4, 'a4': act_f4,
            'logits': logits
        }

        
        return logits
    
    def softmax(self, logits):
        
        exp_output = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # for numerical stability
        softmax_output = exp_output / np.sum(exp_output, axis=1, keepdims=True)
        
        return softmax_output


    def predict_word(self):

        logits = self.forward_prop()
        probabilities = self.softmax(logits)
        predicted_indexes = np.argmax(probabilities, axis=1)
        max_probabilities = np.max(probabilities)
        
        return predicted_indexes,max_probabilities, probabilities
    
    def compute_loss(self,target):

        logits = self.forward_prop()
        probabilities = self.softmax(logits)

        loss = -np.log(probabilities[0,target] + 1e-8)

        return loss, probabilities

    # back propagation 
  
    def back_prop(self, target_index, learning_rate = 0.01):
        
         loss, probabilities = self.compute_loss(target_index)
           
         grad_logits = probabilities.copy()

         grad_logits[0, target_index] -= 1.0  

         grad_logts = np.dot(grad_logits, self.embedding_matrix)

         #layer 4 gradients

         grad_z4 = grad_logts * self.relu_derivative(self.cache['z4'])
         grad_w4 = np.dot(self.cache['a3'].T, grad_z4)
         grad_b4 = np.sum(grad_z4, axis=0 , keepdims=True)
         grad_a3 = np.dot(grad_z4, self.weight4.T)

         #layer 3 gradients 

         grad_z3 = grad_a3 * self.relu_derivative(self.cache['z3'])
         grad_w3 = np.dot(self.cache['a2'].T, grad_z3)
         grad_b3 = np.sum(grad_z3, axis=0, keepdims=True)
         grad_a2 = np.dot(grad_z3, self.weight3.T)

         #layer 2 gradients

         grad_z2 = grad_a2 * self.relu_derivative(self.cache['z2'])
         grad_w2 = np.dot(self.cache['a1'].T, grad_z2)
         grad_b2 = np.sum(grad_z2, axis=0, keepdims=True)
         grad_a1 = np.dot(grad_z2, self.weight2.T)

         #layer 1 gradients 

         grad_z1 = grad_a1 * self.relu_derivative(self.cache['z1'])
         grad_w1 = np.dot(self.inputs.T, grad_z1)
         grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)


         #adjusting parameters

         self.weight4 -= learning_rate * grad_w4
         self.bias4 -= learning_rate * grad_b4
         self.weight3 -= learning_rate * grad_w3
         self.bias3 -= learning_rate * grad_b3
         self.weight2 -= learning_rate * grad_w2
         self.bias2 -= learning_rate * grad_b2
         self.weight -= learning_rate * grad_w1
         self.bias -= learning_rate * grad_b1

         return loss 
    

    def train_step(self, target_index,learning_rate = 0.001,):

        loss = self.back_prop(target_index, learning_rate)
        self.loss_history.append(loss)
    
    
    #training a model on n epochs.
  
    def train(self,target_index, epochs=1000, learning_rate = 0.001):

        for epoch in range(epochs):
            self.train_step(target_index, learning_rate)
            if epoch % 100 == 0:
                current_loss = float(self.loss_history[-1])
                print(f"Epoch {epoch}, loss: {current_loss:.4f}")


        return self.loss_history
    

input = model.wv["upon"].reshape(1, -1) 
final_vecs = model.wv.vectors 
nn = Nueralnetwork(input, final_vecs)



def test_words(nn, model, input_word, target_word, word_to_index, epochs=1000, learning_rate=0.0001):
    input_vec = model.wv[input_word].reshape(1, -1)
    nn.input = input_vec

    target_idx = word_to_index[target_word]
    nn.train(target_idx, epochs=epochs, learning_rate=learning_rate)

    predicted_idx, confidence, _ = nn.predict_word()
    predicted_word = model.wv.index_to_key[predicted_idx[0]]

    print(f"Input: '{input_word}' | Target: '{target_word}' | Predicted: '{predicted_word}' | Confidence: {confidence:.4f}")

def test_multiple_words(nn, model, word_pairs, word_to_index, epochs=1000, learning_rate=0.0001):
    for input_word, target_word in word_pairs:
        test_words(nn, model, input_word, target_word, word_to_index, epochs, learning_rate)
        print("-" * 30)

# Example usage
word_to_index = {word: i for i, word in enumerate(model.wv.index_to_key)}
test_words(nn, model, "once", "upon", word_to_index)
test_multiple_words(nn, model, [("upon", "a"), ("a", "time"), ("time", "there"), ("there", "was"), ("was", "boy")], word_to_index)
