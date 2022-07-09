from underthesea import word_tokenize
from collections import Counter
import pandas as pd
import spacy

nlp = spacy.load('vi_core_news_lg')

# Search matching
def sentence2word(stopwords: list, sentence:str ='chuột đồng chạy quanh sân'):
    words = []
    wt = word_tokenize(sentence)
    for word in wt:
        if word not in stopwords:
            words.append(word)

    return words

def main(sentence='chuột đồng chạy quanh sân', similarity_threshold=0.75):
    stopwords = open('stopwords.txt', 'r', encoding="utf-8")
    stopwords = stopwords.read().splitlines()
    
    col_list = ["stt", "word", "lucky_number"]
    dreamwords = pd.read_csv("dreambook_preprocessed.csv", usecols=col_list)
    dreamwords_list = list(dreamwords["word"])
    number_list = list(dreamwords["lucky_number"])
    
    nums = []
    words = sentence2word(stopwords=stopwords, sentence=sentence)
    for word in words:
        
        # # Check search matching
        # if word in dreamwords_list:
        #     index = dreamwords_list.index(word)
        #     lucky_nums = number_list[index].split("-")
        #     for lucky_num in lucky_nums: nums.append(lucky_num)
        
        # Check synonyms by vi_spacy
        token1 = nlp(str(word))
        for dw in dreamwords_list:
            token2 = nlp(str(dw))
            similarity = token1.similarity(token2)
            if similarity > similarity_threshold:
                index = dreamwords_list.index(str(dw))
                lucky_nums = number_list[index].split("-")
                for lucky_num in lucky_nums: nums.append(lucky_num)
                         
    nums = dict(Counter(nums))
    return nums

# print(main(sentence='chuột đồng chạy quanh sân'))

def tf_tokenize():
    from tensorflow.keras.preprocessing.text import Tokenizer

    sentences = [
        'chém chuột',
        'chuột bạch',
        'chuột cống',
        'chuột đồng',
        'con chuột nhà',
        'con chuột'
    ]
    
    # # Tokenize by space
    # tokenizer = Tokenizer(num_words = 100)
    # tokenizer.fit_on_texts(sentences)
    # word_index = tokenizer.word_index
    # print(word_index)
    
    # Tokenize by Underthesea
    all_word = set()
    
    for sentence in sentences:
        [all_word.add(word) for word in word_tokenize(sentence)]
    
    print(all_word)
    
        
tf_tokenize()

# import spacy
# import time

# nlp = spacy.load('vi_core_news_lg')
# col_list = ["stt", "word", "lucky_number"]
# dreamwords = pd.read_csv("dreambook_preprocessed.csv", usecols=col_list)
# dreamwords_list = list(dreamwords["word"])
# number_list = list(dreamwords["lucky_number"])

# start = time.time()
# token1 = nlp("chuột")
# for dw in dreamwords_list:
#     token2 = nlp(str(dw))
#     similarity = token1.similarity(token2)
#     if similarity > 0.75:
#         print(">> similar > 0.75", token2)
        
# end = time.time()
# print(">> Time cost:", end-start)