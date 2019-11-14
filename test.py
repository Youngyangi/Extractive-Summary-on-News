from Model import ExtractiveSummary
from gensim.models import Word2Vec
import pickle


w2v = Word2Vec.load('w2v.bin')
with open('frequence.pkl', 'rb') as f:
    frequency = pickle.load(f)

extract = ExtractiveSummary(w2v, frequency)

text = '据江苏当地媒体报道，常州女子林某，30岁，未婚，与母亲王某同住。今年8月，林某与母亲发生争吵遭到殴' \
           '打，报警后，王某被刑拘。在相关视频中，王某称，“正好菜架子上有个铁棍子，我就随手抄起来打了她一' \
           '下。” 报道称，林某认为自己长期遭受母亲虐打，身心受到严重伤害，要母亲答应赔偿自己8万元损失费方可谅' \
           '解。今日，上述相关视频在网络上传播，引起关注。新京报记者从常州市公安局天宁分局获悉，此前林某多次报警' \
           '称遭到王某殴打，因情节轻微，未达到立案标准，警方选择进行民事调解。此次林某鉴定为轻伤二级，警方立案调' \
           '查并对王某依法刑拘。目前，王某已被取保候审。至于母女双方调解的具体过程及赔偿金额等，警方表示不便透露。'
print(extract.get_summary(text))
print(extract.get_textrank_summary(text))