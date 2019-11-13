import pandas as pd
import jieba
import re
import numpy as np
import time
import pickle
from collections import Counter

stopwords = [x for x in open('stop_words.txt', 'r', encoding='utf-8').read()]


def token(sentence): return re.findall(r'[\w|\d]+', sentence)


def cut(sentence): return " ".join(jieba.cut(sentence))


def build_cut_news():
    news = pd.read_csv('sqlResult_1558435.csv', encoding='gb18030')
    news.fillna("")
    content = news['content'].values.tolist()
    content = [''.join(token(str(x))) for x in content]
    with open('cutted_news.txt', 'w', encoding='utf-8') as f:
        for news in content:
            f.write(cut(str(news)) + '\n')


# build_cut_news()

def build_frequency():
    with open('cutted_news.txt', 'r', encoding='utf-8') as f:
        words = []
        start = time.time()
        content = f.read()
        content = content.split('\n')
        for i in content:
            word = [w for w in i.split() if w not in stopwords]
            for x in word:
                words.append(x)
        count = Counter(words)
        sum = sum(count.values())
        frequence = {word: frequency/sum for word, frequency in count.items()}
        with open('frequence.pkl', 'wb') as b:
            pickle.dump(frequence, b, pickle.HIGHEST_PROTOCOL)

# build_frequency()


with open('frequence.pkl', 'rb')as f:
    frequency = pickle.load(f)

print(len(frequency))
