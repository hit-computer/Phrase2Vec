# Phrase2Vec
本模型参考了Miklov的Word2Vec，依旧采用上下文预测当前词（短语）来训练词（短语）向量。我们将词和短语训练到同一个向量空间，模型的具体细节内容参见我的博客[PHRASE2VEC——短语向量学习](http://glacier.iego.net/phrase2vec/)。

===========================================
###模型参数
模型各个参数介绍如下：
- size：词（短语）向量的维度
- min-wf：最小词频，词频低于该值的词将被忽略
- window：上下文窗口大小
- lr：学习率
- neg-sample：负采样词（短语）数目
- max-num-epochs：迭代次数
- save-epochs：每迭代多少次保存一下模型
- （前面这五个参数和Word2Vec差不多，这里还有另外两个不一样的参数）
- min-pf：短语频率，高于该频率的bi-gram将被认为是一个短语
- margin：正负例的间距

模型训练，在命令行输入：

    python Phrase2vec.py [训练语料]

若机器上有GPU，可以使用GPU进行训练，速度比CPU能快很多，输入命令改为：

    THEANO_FLAGS='mode=FAST_RUN,device=gpu,floatX=float32' python Phrase2vec.py [训练语料]

训练语料可以选用word2vec的测试语料，[text8.zip](http://mattmahoney.net/dc/text8.zip)

###实验结果
用word2vec提供的测试语料训练我们的phrase2vec模型（GPU配置：Tesla K40m，迭代一轮所需时间：大约30h），利用余弦相似度计算词（短语）之间的相似度，以下是与“hong kong”相似的词语：
>singapore, beijing, taiwan, shanghai, south africa, of hong, antrim, new zealand, calcutta, in dublin, warwickshire, south korea, germany s, auckland adrift, el salvador, new jersey, ferdinando, in Sweden, woolwich, yangon
