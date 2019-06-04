# keras module for building LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
# set seeds for reproducability
from tensorflow import set_random_seed
from numpy.random import seed
set_random_seed(23)
seed(122)

import pandas as pd
import string, os
import warnings
import generation_of_text
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

curr_dir = '/home/user/Desktop/downs/nyt-comments/'

all_headlines = []
for filename in os.listdir(curr_dir):
    if 'Articles' in filename:
        article_df = pd.read_csv(curr_dir + filename)
        all_headlines.extend(list(article_df.headline.values))
        break

all_headlines = [h for h in all_headlines if h != "Unknown"]
len(all_headlines)

def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt

corpus = [clean_text(x) for x in all_headlines]
print(corpus[:10])

tokenizer = Tokenizer()

def get_sequence_of_tokens(corpus):
    ## tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    ## convert data to sequence of tokens
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words


inp_sequences, total_words = get_sequence_of_tokens(corpus)
print(inp_sequences[:10])

predictors, label, max_sequence_len = generation_of_text.generate_padded_sequences(inp_sequences, total_words)
print(predictors, label, max_sequence_len)


# model = generation_of_text.create_model(max_sequence_len, total_words)
# model.summary()
#
# model.fit(predictors, label, epochs=100, verbose=5)
# model.save_weights('checkpoints/4-june-2019')


def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    print(seed_text)
    return seed_text.title()


# Restore the weights
model = generation_of_text.create_model(max_sequence_len, total_words)
model.load_weights('checkpoints/4-june-2019')


generate_text("this", 18, model=model, max_sequence_len=24)

