# Extractive-Summary-on-News
An extractive method for news summary.  
Sentence Vector and Text Rank.

用法:
1. 原始数据文件"sqlResult_1558545.csv"内容是新闻文本；stop_words.txt（辅助去除停用词，可以搜集网上开源的，也可以不用）
2. Builder.py 28行执行build_cut_news()读取每条新闻，切词，并按空格分开，写在cutted_news.txt中，每一行是一个新闻；执行到
46行的build_frequency()是接着上面生成的文件，统计词语出现的频次，并保存成frequnce.pkl;
3. 调用Word2Vec.py 读取cutted_news.txt并训练词向量，保存成w2v.bin;
4. 至此，自动摘要模型已经准备好了；
5. test.py中测试摘要模型效果，6-8行加载前面生成的几个准备文件，10行实例化文本摘要模型；输入任意新闻文本，测试摘要的效果；
get_summary()是计算每个句向量和文章的向量（所有句子向量加和的平均）的余弦相似度最高的；get_textrank_summary()是用text-rank原理
找到的相关性最高的句子；（text-rank具体是啥，百度下，基于google有名的page-rank）
