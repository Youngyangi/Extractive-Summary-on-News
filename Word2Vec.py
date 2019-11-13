from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


w2v = Word2Vec(LineSentence("cutted_news.txt"), sg=1, workers=5)
w2v.save('w2v.bin')
