import jieba
import re
import numpy as np


class ExtractiveSummary:
    def __init__(self, model, frequency, min_len=8, max_out=50):
        self.w2v_model = model
        self.frequency = frequency
        self.stop_words = [w for w in open('stop_words.txt', 'r',  encoding='utf-8').read()]
        self.min_len = min_len
        self.max_out = max_out
        self.dim = self.w2v_model.wv.vector_size
        self.max_fre = max(self.frequency.values.tolist())

    def tokenize(self, sentence): return re.findall(r'[\w|\d]+', sentence)

    def cut(self, sentence): return ' '.join(jieba.cut(sentence))

    def sentence_embedding(self, sentence, smooth_alpha=1e-4):
        alpha = smooth_alpha
        sentence = self.tokenize(sentence)
        words = jieba.cut(sentence)
        sentence_vector = np.zeros(self.dim)
        for word in words:
            if word in self.w2v_model.wv.vocab and word not in self.stop_words:
                word_vec = self.w2v_model.wv[word]
                weight = alpha / (alpha + self.frequence.get(word, self.max_fre))
                sentence_vector += weight * word_vec
        sentence_vector /= len(words)
        return sentence_vector

    def text_embedding(self):
        pass





