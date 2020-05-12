# -*- coding: utf-8 -*-
import tensorflow as tf

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker
#from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time


class DatasetPreparer:
    pathToData = "Dataset/"
    val_file = "ger_validation.txt"
    train_file = "ger_train.txt"
    test_file = "ger_test.txt"
    all_vocab_file = "all_vocab.txt"
    
    def __init__(self, if_test):
        if if_test == True:
            print("   ")
            print("##### Sentence Processing Test ####")
            en_sentence = u"May I borrow this book?"
            sp_sentence = u"¿Puedo tomar prestado este libro?"
            ger_sentence = u"Gefahrguttransport auf der Straße"
            ger_sentence2 = u"Reform der europäischen Wettbewerbspolitik"
            print(self.preprocess_sentence(en_sentence))
            print(self.preprocess_sentence(sp_sentence).encode('utf-8'))
            print(self.preprocess_sentence(ger_sentence))
            print(self.preprocess_sentence(ger_sentence2))
            print("   ")
        #create train, validation and test datasets
        ger_train, en_train = self.create_dataset(self.pathToData+self.train_file, None)
        ger_val, en_val = self.create_dataset(self.pathToData+self.val_file, None)
        ger_test, en_test = self.create_dataset(self.pathToData+self.test_file, None)
        ger_all_vocab, en_all_vocab = self.create_dataset(self.pathToData+self.all_vocab_file, None) #for tokenizing only
        if if_test == True:
            print("   ")
            print("##### Printing Last Sentenses From TRAINING, VALIDATION and TEST sets ####")
            print(en_train[-1])
            print(ger_train[-1])
            print(en_val[-1])
            print(ger_val[-1])
            print(en_test[-1])
            print(ger_test[-1])
            print("   ")
            print("   ")
            print("##### Printing First Sentenses From TRAINING, VALIDATION and TEST sets ####")
            print(en_train[0])
            print(ger_train[0])
            print(en_val[0])
            print(ger_val[0])
            print(en_test[0])
            print(ger_test[0])
            print("   ")
        #tokenizing datasets
        en_train_tensor, en_train_tokenizer = self.tokenize(en_train)
        ger_train_tensor, ger_train_tokenizer = self.tokenize(ger_train)
        en_val_tensor, en_val_tokenizer = self.tokenize(en_val)
        ger_val_tensor, ger_val_tokenizer = self.tokenize(ger_val)
        en_test_tensor, en_test_tokenizer = self.tokenize(en_test)
        ger_test_tensor, ger_test_tokenizer = self.tokenize(ger_test)
        #all vocab for tokenizing
        dummy, en_train_tokenizer = self.tokenize(en_train)
        #Getting area 
        if if_test == True:
            print("   ")
            print("##### Index To Words Mapping of First Sentences ####")
            print ("English")
            self.convert(en_train_tokenizer, en_train_tensor[0])
            print ()
            print ("German")
            self.convert(ger_train_tokenizer, ger_train_tensor[0])
            print("   ")
        #creating tensor flow dataset
        BUFFER_SIZE = len(en_train_tensor)
        BATCH_SIZE = 64
        self.BATCH_SIZE = BATCH_SIZE
        self.steps_per_epoch = len(en_train_tensor)//BATCH_SIZE
        self.embedding_dim = 256
        self.units = 1024
        self.vocab_inp_size = len(en_train_tokenizer.word_index)+1
        self.vocab_tar_size = len(ger_train_tokenizer.word_index)+1
        dataset = tf.data.Dataset.from_tensor_slices((en_train_tensor, ger_train_tensor)).shuffle(BUFFER_SIZE)
        self.max_length_targ, self.max_length_inp = ger_train_tensor.shape[1], en_train_tensor.shape[1]
        self.dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
        self.ger_train_tokenizer = ger_train_tokenizer
        self.en_train_tokenizer = en_train_tokenizer
    
    def returnMaxLengths(self):
        return self.max_length_targ, self.max_length_inp

    def returnInTarTokens(self):
        return self.en_train_tokenizer, self.ger_train_tokenizer
            
    def returnData(self):
        return self.dataset
        
    def returnVariables(self):
        return self.steps_per_epoch, self.vocab_tar_size,  self.vocab_inp_size, self.embedding_dim, self.units, self.BATCH_SIZE
            
    def convert(self, lang, tensor):
        for t in tensor:
            if t!=0:
                print ("%d ----> %s" % (t, lang.index_word[t]))

    def tokenize(self, lang):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        lang_tokenizer.fit_on_texts(lang)
        tensor = lang_tokenizer.texts_to_sequences(lang)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,padding='post')
        return tensor, lang_tokenizer
    
    
    # Converts the unicode file to ascii
    def unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')
    
    def preprocess_sentence(self, w):
        w = self.unicode_to_ascii(w.lower().strip())
        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z?.!,¿]+ß", " ", w)
        w = w.strip()
        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.
        w = '<start> ' + w + ' <end>'
        return w
        
    # 1. Remove the accents
    # 2. Clean the sentences
    # 3. Return word pairs in the format: [GERMAN, ENGLISH]
    def create_dataset(self, path, num_examples):
        lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
        word_pairs = [[self.preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
        return zip(*word_pairs)

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights





def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)





if __name__ == "__main__":
    
    #choose between "train", "learn", "both"
    MODE = "both"
    
    DP = DatasetPreparer(True)
    en_token,  ger_token = DP.returnInTarTokens()
    steps_per_epoch,  vocab_tar_size,  vocab_inp_size, embedding_dim, units, BATCH_SIZE = DP.returnVariables()
    dataset = DP.returnData()
    example_input_batch, example_target_batch = next(iter(dataset))
    Enc = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    sample_hidden = Enc.initialize_hidden_state()
    sample_output, sample_hidden = Enc(example_input_batch, sample_hidden)
    attention_layer = BahdanauAttention(10)
    attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
    sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),sample_hidden, sample_output)
    print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    checkpoint_dir = './training_checkpoints/'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=Enc, decoder=decoder)
    
    max_length_targ, max_length_inp = DP.returnMaxLengths()
    
    @tf.function
    def train_step(inp, targ, enc_hidden):
        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = Enc(inp, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([ger_token.word_index['<start>']] * BATCH_SIZE, 1)
            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                loss += loss_function(targ[:, t], predictions)
                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)
            batch_loss = (loss / int(targ.shape[1]))
            variables = Enc.trainable_variables + decoder.trainable_variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            return batch_loss
            
    if MODE == "train" or MODE == "both":
        EPOCHS = 2
        for epoch in range(EPOCHS):
            start = time.time()
            enc_hidden = Enc.initialize_hidden_state()
            total_loss = 0
    
            for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = train_step(inp, targ, enc_hidden)
                total_loss += batch_loss
    
                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
    
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    
    
    
    
    def evaluate(sentence):
        attention_plot = np.zeros((max_length_targ, max_length_inp))
        sentence = DP.preprocess_sentence(sentence)
        inputs = [en_token.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
        inputs = tf.convert_to_tensor(inputs)
        result = ''
        hidden = [tf.zeros((1, units))]
        enc_out, enc_hidden = Enc(inputs, hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([ger_token.word_index['<start>']], 0)
        for t in range(max_length_targ):
            predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden,enc_out)
            # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1, ))
            attention_plot[t] = attention_weights.numpy()
            predicted_id = tf.argmax(predictions[0]).numpy()
            result += ger_token.index_word[predicted_id] + ' '
            if ger_token.index_word[predicted_id] == '<end>':
                return result, sentence, attention_plot
            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)
        return result, sentence, attention_plot



    # function for plotting the attention weights
    def plot_attention(attention, sentence, predicted_sentence):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1, 1, 1)
        ax.matshow(attention, cmap='viridis')
        fontdict = {'fontsize': 14}
        ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
        ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.show()

    def translate(sentence):
        translation, sentence, attention_plot = evaluate(sentence)
        #print('Input: %s' % (sentence))
        #print('Predicted translation: {}'.format(result))
        #attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
        #plot_attention(attention_plot, sentence.split(' '), result.split(' '))
        return sentence,  translation
        
    if MODE == "translate" or MODE == "both":
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("   ")
        print("##### Trying Some Translations From Learnig Dataset ####")
        print("Sentence: Resumption of the session, Should be translated into: Wiederaufnahme der Sitzungsperiode")
        sentence, translation = translate(u"Resumption of the session")
        print("Sentence: " + sentence + "Was translated to: " + translation)
        print("Sentence: Mr Hänsch represented you on this occasion., Should be translated into: Der Kollege Hänsch hat Sie dort vertreten.")
        sentence, translation = translate(u"Mr Hänsch represented you on this occasion.")
        print("Sentence: " + sentence + "Was translated to: " + translation)
        #print("Sentence: , Should be translated into: ")
        print("   ")
        
        #translating all sentences
        outF = open("translation.txt", "w")
        lines = io.open("Dataset/ger_test.txt").read().strip().split('\n')
        for line in lines:
            #print(line)
            ger,  en = re.split(r'\t+', line)
            print(en)
            #print(en)
            #print()
            #print(ger)
            sentence, translation = translate(en)
            outF.write(ger.encode('utf-8') + "\t" + en.encode('utf-8') + "\n")
        outF.close()
            
        

        

      
      
      
      
      
      
      
      
      
      
      
      
