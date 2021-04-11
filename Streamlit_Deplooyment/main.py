import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
CUDA_VISIBLE_DEVICES=""
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import streamlit as st
import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, LSTM

DECODER_SEQ_LEN = 41
required_chars = []
vocabulary = dict()
input_vocab_size = 97
target_vocab_size = 97

for char in string.printable:
    if 31 < ord(char) < 126:
        required_chars.append(char)

for i in range(len(required_chars)):
    vocabulary[required_chars[i]] = i+1

vocabulary['\n'] = 95
vocabulary['\t'] = 96

tokenizer_raw_ip = Tokenizer(
    char_level=True,
    lower=False,
    filters=None
)

tokenizer_target_ip = Tokenizer(
    char_level=True,
    lower=False,
    filters=None
)

# Using the printable 94 characters as the vocabulary
tokenizer_target_ip.word_index = vocabulary
tokenizer_raw_ip.word_index = vocabulary

start_index = tokenizer_target_ip.word_index['\t']
end_index = tokenizer_target_ip.word_index['\n']

target_vocab_size=len(tokenizer_target_ip.word_index.keys())
input_vocab_size=len(tokenizer_raw_ip.word_index.keys())


# Encoder class with Embedding layer and LSTM layer.
class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, input_length, enc_units):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.enc_units= enc_units
        self.lstm_output = 0
        self.lstm_state_h=0
        self.lstm_state_c=0
        
    def build(self, input_shape):
        self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.input_length, mask_zero=True, name="Embedding_Layer_Encoder")
        self.lstm_1 = LSTM(self.enc_units, recurrent_dropout=0.2, return_state=True, return_sequences=True, name="Encoder_LSTM_1")
        self.lstm_2 = LSTM(self.enc_units, recurrent_dropout=0.2, return_state=True, return_sequences=True, name="Encoder_LSTM_2")
        
    def call(self, input_sentances, training=True):
        # input_embedded = self.embedding(input_sentances)
        self.lstm_output,_,_ = self.lstm_1(input_sentances)
        self.lstm_output, self.lstm_state_h, self.lstm_state_c = self.lstm_2(self.lstm_output)

        return self.lstm_output, self.lstm_state_h,self.lstm_state_c

    def get_states(self):
        return self.lstm_state_h,self.lstm_state_c
    
# Decoder class with embedding and LSTM layer.    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, input_length, dec_units):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.input_length = input_length
        # we are using embedding_matrix and not training the embedding layer
        self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.input_length, mask_zero=True, name="Embedding_Layer_Decoder",)
        self.lstm = LSTM(self.dec_units,  dropout=0.2, return_sequences=True, return_state=True, name="Decoder_LSTM")
    
    def call(self, target_sentences, state_h, state_c):
        # target_embedded           = self.embedding(target_sentences)
        lstm_output, _,_        = self.lstm(target_sentences, initial_state=[state_h, state_c])
        return lstm_output

# Model 1 - 1 layer LSTM model for each encoder and decoder
class Model1(Model):
    def __init__(self, encoder_inputs_length,decoder_inputs_length, output_vocab_size):
        super().__init__() # https://stackoverflow.com/a/27134600/4084039
        self.encoder = Encoder(vocab_size=input_vocab_size+1, embedding_dim=30, input_length=encoder_inputs_length, enc_units=512)
        self.decoder = Decoder(vocab_size=target_vocab_size+1, embedding_dim=30, input_length=decoder_inputs_length, dec_units=512)
        self.dense   = Dense(output_vocab_size+1, activation='softmax')
        
    def call(self, data):
        input,output = data[0], data[1]
        encoder_output, encoder_h, encoder_c = self.encoder(input)
        decoder_output                       = self.decoder(output, encoder_h, encoder_c)
        output                               = self.dense(decoder_output)
        return output        


# Loading the Model 1 and 2 for Encoder and Decoders with 1 and 2 Layers of LSTM respectively
model = Model1(encoder_inputs_length=40,decoder_inputs_length=41,output_vocab_size=target_vocab_size)
model.load_weights("./model/model_2")

# Chooses between the model 1 and 2 for output
def input_to_model(input_string):
    output = predict(input_string)
    return output


# Predict Output from Model 1 with single layer of LSTM in the Encoder and Decoder Models
def predict(input_sentence):
  word_list = []
  split_sentence = input_sentence.split(" ")
  for word in split_sentence:

      encoder_seq = tokenizer_raw_ip.texts_to_sequences([word])

      encoder_seq = pad_sequences(encoder_seq, maxlen=40, dtype='int32', padding='post')

      encoder_seq = tf.keras.utils.to_categorical(encoder_seq, num_classes=len(tokenizer_raw_ip.word_index.keys())+1)

      enc_output, enc_state_h, enc_state_c = model.layers[0](encoder_seq)

      dec_input = np.zeros((1, 1, len(tokenizer_raw_ip.word_index.keys())+1))

      dec_input[0, 0, tokenizer_target_ip.word_index['\t']] = 1.

      input_state = [enc_state_h, enc_state_c]

      output_word = []

      for i in range(DECODER_SEQ_LEN):
          # cur_emb = model.layers[1].embedding(dec_input)

          predicted_out, state_h, state_c = model.layers[1].lstm(dec_input, input_state)

          dense_layer_out = model.layers[2](predicted_out)

          input_state = [state_h, state_c]
      
          output_word_index = np.argmax(dense_layer_out)

          # print(output_word_index)

          for key, value in tokenizer_target_ip.word_index.items():

            if output_word_index == value:
                output_word.append(key)

          dec_input = np.reshape(output_word_index, (1, 1))

          dec_input = np.zeros((1, 1, len(tokenizer_raw_ip.word_index.keys())+1))

          dec_input[0, 0, output_word_index] = 1.


          if output_word_index == tokenizer_target_ip.word_index['\n']:
            break

      word = "".join(output_word)
      word_list.append(word)
      # print(word_list)
  sentence = ''.join(word_list)
  sentence = sentence.replace("\n", " ")
  return sentence



def main():
    readme_text = st.markdown(get_file_content_as_string("instructions.md"))

    st.sidebar.title("What do you want to do ?")
    modes = st.sidebar.selectbox("Choose the app mode",
                                    ["Show instructions", "Run the app", "Show the source code", "See train data"])

    if modes == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif modes == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string("main.py"))
    elif modes == "See train data":
        readme_text.empty()
        data = show_data()
        st.write(data)
    elif modes == "Run the app":
        readme_text.empty()
        run()

@st.cache(suppress_st_warning=True)
def show_data():
    data_load_state = st.text("Loading data")
    data = pd.read_csv("train.csv", nrows=10)
    return data

# Obtains the output from the models by using the user input.
def run():
    st.sidebar.info('Model with 2 layer of LSTM in Encoder and 1 layer LSTM in Decoder')
    st.set_option('deprecation.showfileUploaderEncoding', False)
    add_selectbox = st.sidebar.selectbox("Which Model would like to use to predict", ("1 layer"))
    st.title("Sentence Correction")
    st.header('This app can correct short sentences')
                
    text1 = st.text_area('Enter text')
    output = ""
    if st.button("Predict"):
        output = input_to_model(text1)
        output = str(output)
        st.success(f"The Output String : {output}")
        st.balloons()


@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    with open(path, 'r') as file:
        data = file.read()
    return data


if __name__ == "__main__":
    main()




