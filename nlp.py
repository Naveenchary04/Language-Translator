import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Attention, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from googletrans import Translator  # Import Google Translator

# Initialize Google Translator
translator = Translator()

# Example data (replace with a larger dataset for real-world usage)
input_texts = ['hello', 'how are you?', 'good morning', 'good evening', 'how is it going?']
target_texts = ['startseq hi endseq', 'startseq i am fine endseq', 'startseq good morning endseq',
                'startseq good evening endseq', 'startseq i am doing well endseq']

# Tokenization
tokenizer_source = Tokenizer()
tokenizer_target = Tokenizer()
tokenizer_source.fit_on_texts(input_texts)
tokenizer_target.fit_on_texts(target_texts)

input_sequences = tokenizer_source.texts_to_sequences(input_texts)
target_sequences = tokenizer_target.texts_to_sequences(target_texts)

max_input_len = max(len(seq) for seq in input_sequences)
max_target_len = max(len(seq) for seq in target_sequences)

input_sequences = pad_sequences(input_sequences, maxlen=max_input_len, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_target_len, padding='post')

X_train = input_sequences
y_train = np.expand_dims(target_sequences, -1)

vocab_size_source = len(tokenizer_source.word_index) + 1
vocab_size_target = len(tokenizer_target.word_index) + 1

embedding_dim = 256
latent_dim = 256

# Encoder
encoder_inputs = Input(shape=(max_input_len,))
encoder_embedding = Embedding(vocab_size_source, embedding_dim)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(latent_dim, return_sequences=True, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(max_target_len,))
decoder_embedding = Embedding(vocab_size_target, embedding_dim)
decoder_embedded = decoder_embedding(decoder_inputs)

decoder_lstm, _, _ = LSTM(latent_dim, return_sequences=True, return_state=True)(decoder_embedded, 
                                                                                initial_state=encoder_states)

# Attention Mechanism
attention = Attention()
attention_output = attention([decoder_lstm, encoder_outputs])

decoder_combined_context = Concatenate(axis=-1)([attention_output, decoder_lstm])
decoder_dense = Dense(vocab_size_target, activation='softmax')
decoder_outputs = decoder_dense(decoder_combined_context)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit([X_train, target_sequences], y_train, epochs=10, batch_size=2)

# Encoder Model
encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

# Decoder Inference Model
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_input = [decoder_state_input_h, decoder_state_input_c]

decoder_inference_inputs = Input(shape=(1,))
decoder_inference_embedded = decoder_embedding(decoder_inference_inputs)

decoder_lstm_inference, state_h_inf, state_c_inf = LSTM(latent_dim, return_sequences=True, return_state=True)(
    decoder_inference_embedded, initial_state=decoder_states_input)

attention_inference = Attention()
attention_output_inf = attention_inference([decoder_lstm_inference, encoder_outputs])

decoder_combined_context_inf = Concatenate(axis=-1)([attention_output_inf, decoder_lstm_inference])
decoder_outputs_inf = decoder_dense(decoder_combined_context_inf)

decoder_model = Model(
    [decoder_inference_inputs] + decoder_states_input + [encoder_outputs],
    [decoder_outputs_inf] + [state_h_inf, state_c_inf]
)

# Translation Function with Google Translate Fallback
def translate(input_text, target_lang='fr'):
    input_seq = tokenizer_source.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_input_len, padding='post')

    encoder_output, state_h, state_c = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer_target.word_index['startseq']

    decoded_sentence = ''
    states_value = [state_h, state_c]

    for _ in range(max_target_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value + [encoder_output])

        sampled_token_index = np.argmax(output_tokens[0, -1, :])  # Fixed np.argm issue
        sampled_word = tokenizer_target.index_word.get(sampled_token_index, '')

        if sampled_word == 'endseq' or sampled_word == '':
            break

        decoded_sentence += ' ' + sampled_word

        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    decoded_sentence = decoded_sentence.strip()

    # If the model output is empty, fall back to Google Translate
    if not decoded_sentence:
        print("Using Google Translator instead...")
        translated_text = translator.translate(input_text, dest=target_lang).text
        return translated_text

    return decoded_sentence

# Example Translation with Fallback
print(translate("hello"))  # Expected: "hi"
print(translate("how are you?"))  # Expected: "i am fine"
print(translate("I love programming", target_lang='es'))  # Fallback to Google Translate in Spanish
