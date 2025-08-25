"""This is the implementation of Seq to Seq paper by Ilya sutskever which is considered earlier effort for translation using neural network.
   This implementation is good for getting intuition and even produce 2/3 output correct which is good for this much dataset :)
   
   Model's prediction:
   English: hello world
   French: bon matin

   English: good morning
   French: bon matin

   English: how are you
   French: comment allez vous"""



eng_sentences = [
    ["hello", "world"],
    ["good", "morning"], 
    ["how", "are", "you"],
    ["thank", "you"]
]

frch_sentences = [
    ["bonjour", "monde"],
    ["bon", "matin"],
    ["comment", "allez", "vous"], 
    ["merci"]
]

# Train Word2Vec models
en_model = Word2Vec(
    sentences=eng_sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    epochs=50
)

fr_model = Word2Vec(
    sentences=frch_sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    epochs=50
)


fr_vocab_size = len(fr_model.wv.key_to_index)
fr_model.wv.key_to_index['<START>'] = fr_vocab_size
fr_model.wv.key_to_index['<END>'] = fr_vocab_size + 1


start_embedding = np.random.randn(1, 100)
end_embedding = np.random.randn(1, 100)
fr_model.wv.vectors = np.vstack([fr_model.wv.vectors, start_embedding, end_embedding])

print(f"English vocab: {len(en_model.wv.key_to_index)} words")
print(f"French vocab: {len(fr_model.wv.key_to_index)} words")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SeqtoSeq(nn.Module):
   
    def __init__(self, en_model, fr_model, hidden_size=100, num_layers=4):
        super(SeqtoSeq, self).__init__()
        
        self.en_model = en_model
        self.fr_model = fr_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.encoder =nn.LSTM(
            input_size=100, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True,
            bias=True
        )
        
        self.decoder = nn.LSTM(
            input_size=100, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True, 
            bias=True
        )
        
        self.logits = nn.Linear(self.hidden_size, len(fr_model.wv.key_to_index), bias=True)

        
    
    def forward(self, inword, outword):
        
        encoder_output, encoder_hidden = self.encoder(inword)
        
        decoder_output, decoder_hidden = self.decoder(outword.to(torch.float32), encoder_hidden)
        
        logits = self.logits(decoder_output)
        
        return logits
    
    def translate(self, english_sentence, max_length=10):
       
        self.eval()
        with torch.no_grad():
            
            eng_tensor = self.sentence_to_tensor(english_sentence, self.en_model)
            
            encoder_output, encoder_hidden = self.encoder(eng_tensor)
            
            decoder_input = torch.tensor([[self.get_embedding('<START>', self.fr_model)]], dtype= torch.float32)
            decoder_hidden = encoder_hidden
            
            result = []
            
            for _ in range(max_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                logits = self.logits(decoder_output)
                
                predicted_id = torch.argmax(logits, dim=-1).item()
                predicted_word = self.id_to_word(predicted_id, self.fr_model)
                
                if predicted_word == '<END>':
                    break
                    
                result.append(predicted_word)
                
               
                decoder_input = torch.tensor([[self.get_embedding(predicted_word, self.fr_model)]], dtype= torch.float32)
            
            return result
        
    def sentence_to_tensor(self, sentence, model):
        
        embeddings = [self.get_embedding(word, model) for word in sentence]
        return torch.tensor(embeddings).unsqueeze(0)
    
    def get_embedding(self, word, model):
        
            return model.wv[word]
       
    
    def id_to_word(self, word_id, model):
        
        for word, idx in model.wv.key_to_index.items():
            if idx == word_id:
                return word
        return 'no word found'


def prepare_training_data(eng_sentences, frch_sentences, en_model, fr_model):
    
    inputs = []
    decoder_inputs = []
    targets = []
    
    for eng_sent, fr_sent in zip(eng_sentences, frch_sentences):
        
        eng_embs = [en_model.wv[word] for word in eng_sent]
        inputs.append(torch.tensor(eng_embs))
        
        
        dec_input_words = ['<START>'] + fr_sent
        dec_input_embs = [fr_model.wv[word] for word in dec_input_words]
        decoder_inputs.append(torch.tensor(dec_input_embs))
        
        
        target_words = fr_sent + ['<END>']
        target_ids = [fr_model.wv.key_to_index[word] for word in target_words]
        targets.append(torch.tensor(target_ids))
    
    return inputs, decoder_inputs, targets

def pad_sequences(sequences, pad_value=0):
   
    max_len = max(len(seq) for seq in sequences)
    padded = []
    
    for seq in sequences:
        if len(seq.shape) == 1:  
            padded_seq = torch.cat([seq, torch.full((max_len - len(seq),), pad_value)])
        else:  
            padding = torch.zeros((max_len - len(seq), seq.shape[1]))
            padded_seq = torch.cat([seq, padding])
        padded.append(padded_seq)
    
    return torch.stack(padded)

def train_model():
  
    inputs, decoder_inputs, targets = prepare_training_data(
        eng_sentences, frch_sentences, en_model, fr_model
    )
    
   
    padded_inputs = pad_sequences(inputs)
    padded_decoder_inputs = pad_sequences(decoder_inputs)
    padded_targets = pad_sequences(targets, pad_value=-1)  
   
    model = SeqtoSeq(en_model, fr_model)
    
   
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    
    model.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        
        logits = model(padded_inputs, padded_decoder_inputs)
        
       
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = padded_targets.view(-1)
        
       
        loss = criterion(logits_flat, targets_flat)
        
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')
    
    return model


trained_model = train_model()
test_sentences = [
    ["hello", "world"],
    ["good", "morning"],
    ["how", "are", "you"]
]

for eng_sent in test_sentences:
    translation = trained_model.translate(eng_sent)
    print(f"English: {' '.join(eng_sent)}")
    print(f"French: {' '.join(translation)}")
    print()    
    

    
input = torch.tensor(en_model.wv[["hello", "world"]])
output = torch.tensor(fr_model.wv[["<START>", "bonjour", "monde", "<END>"]])
seq = SeqtoSeq(en_model, fr_model)

print(seq.forward(input,output))

    translation = trained_model.translate(eng_sent)
    print(f"English: {' '.join(eng_sent)}")
    print(f"French: {' '.join(translation)}")
    print()
