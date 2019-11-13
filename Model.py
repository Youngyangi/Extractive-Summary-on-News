import jieba
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ExtractiveSummary:
    def __init__(self, model, frequency, min_len=8, max_out=50):
        self.w2v_model = model
        self.frequency = frequency
        self.stop_words = [w for w in open('stop_words.txt', 'r',  encoding='utf-8').read()]
        self.min_len = min_len
        self.max_out = max_out
        self.dim = self.w2v_model.wv.vector_size
        self.max_fre = max(self.frequency.values())

    def _tokenize(self, sentence): return ''.join(re.findall(r'[\w|\d]+', sentence))

    def _cut(self, sentence): return ' '.join(jieba.cut(sentence))

    def sentence_embedding(self, sentence, smooth_alpha=1e-4):
        alpha = smooth_alpha
        sentence = self._tokenize(sentence)
        sentence = self._cut(sentence)
        sentence_vector = np.zeros(self.dim)
        words = sentence.split()
        for word in words:
            if word in self.w2v_model.wv.vocab and word not in self.stop_words:
                word_vec = self.w2v_model.wv[word]
                weight = alpha / (alpha + self.frequency.get(word, self.max_fre))
                sentence_vector += weight * word_vec
        sentence_vector /= len(words)
        return sentence_vector

    def sentence_similarity(self, sent1, sent2):
        sent1 = self.sentence_embedding(sent1)
        sent2 = self.sentence_embedding(sent2)
        cos = cosine_similarity(sent1.reshape(1, -1), sent2.reshape(1, -1))[0][0]
        return cos

    def split_sentences(self, text):
        text = text.replace('\n', '')
        text = text.replace('\r', '')
        sentences = re.split('[,.，。?!？！]', text)
        for i, sentence in enumerate(sentences):
            if len(sentence) < self.min_len and i != len(sentences)-1:
                sentences[i+1] = sentence + sentences[i+1]
        split_sentence = [x for x in sentences if len(x) >= self.min_len]
        return split_sentence

    def get_correlation_rank(self, sentences):
        text = ','.join(sentences)
        cos = []
        for sentence in sentences:
            cos.append(self.sentence_similarity(text, sentence))
        return [(sentences[i], i) for i in sorted(range(len(sentences)), key=lambda x: cos[x], reverse=True)]

    def get_summary(self, text):
        sentences = self.split_sentences(text)
        ranked = self.get_correlation_rank(sentences)
        summary = []
        index = []
        summary_len = 0
        for sentence, i in ranked:
            if summary_len > self.max_out:
                return summary
            index.append(i)
            summary.append(sentence)
            summary_len += len(sentence)
        summary = sorted(summary, key=lambda x: index[x], reverse=False)
        return ','.join(summary)







