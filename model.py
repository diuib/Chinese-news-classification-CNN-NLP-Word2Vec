import os
import random
import thulac
import tensorflow as tf
import numpy as np
import fileinput
import pretreat

WORD2VEC_DIC = 'input/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'      # Chinese Word Vectors提供的预训练词向量

DICTIONARY = pretreat.pretreatment_date()    # 读取利用train数据包归纳出的字典
CLASSES = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']    # 所有输出

SEQUENCE_LENGTH = 50            # 句子最长含词量。为了训练速度，超过后，多余的词截断
INPUT_SIZE = SEQUENCE_LENGTH    # 神经网络输入尺寸
OUTPUT_SIZE = len(CLASSES)      # 神经网络输出尺寸
HIDDEN_DIM = 512                # 隐藏层神经元数
EMBEDDING_SIZE = 300            # 每个词的词向量维度
NUM_FILTERS = 256               # 每个核卷积后输出的维度
BATCH_SIZE = 50                 # 每批次样本数量
FILTERS = [2, 3, 4]             # 卷积核尺寸
USE_L2 = False                  # 是否启用L2正则化
L2_LAMBDA = 0

x = tf.placeholder(tf.int32, [None, INPUT_SIZE])
y_ = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])
dropout_keep_prob = tf.placeholder(tf.float32)

thu1 = thulac.thulac(seg_only=True)  # 只分词，不需要标注词性


def get_word2vec(dic,   # 之前归纳好的字典
                 pha,   # 词向量
                 file): # 已预训练好的词向量文件地址
    """采用预训练好的部分词向量。仅使用部分，是为了节省内存。
    1. 遍历已训练好的词向量文件
    2. 替换掉本例词典中存在词的词向量
    """
    print("正在加载预训练词向量……")
    if os.path.isfile(file):
        with fileinput.input(files=(WORD2VEC_DIC), openhook=fileinput.hook_encoded('UTF-8')) as f:
            for line in f:
                word_and_vec = line.split(' ')
                word = word_and_vec[0]
                vec = word_and_vec[1:301]
                if word in dic:
                    pha[DICTIONARY.index(word)] = vec
    print("预训练词向量加载完毕。")


def get_some_date(file_name='input/cnews.train.txt',
                  num=100):
    """取出数据包中的n条数据，处理成需要的格式，并返回。
    1. 使用蓄水池采样算法，取出数据包中的 num 条数据。
    2. 分词
    3. 参照字典把输入转换成序列，把输出转换成 One-hot 格式
    4. 输入和输出分别放入 x_batch 和 y_batch 中返回
    """
    x_batch = []
    y_batch = []
    getlines = []
    i = 0

    # 此种读取算法，可保存每行被取到的概率相同
    with fileinput.input(files=file_name, openhook=fileinput.hook_encoded('UTF-8')) as f:
        # fileinput 节省内存
        for line in f:
            i += 1
            if len(getlines) <= num:
                getlines.append(line)
            else:
                random_num = random.randint(0, i)
                if random_num <= num:
                    random_line = random.randint(0, num)
                    getlines[random_line] = line

    # 将数据处理成期望的格式
    for line in getlines:
        label = str(line.split('\t')[0])
        text = str(line.split('\t')[1:])

        # 将 label 转换成 One-hot 格式
        label_transl = np.zeros(len(CLASSES))
        for i in range(len(CLASSES)):
            if CLASSES[i] == label:
                label_transl[i] = 1

        # 将输入转换成序列 list
        text_separate = thu1.cut(text, True).split(' ')
        text_transl = np.zeros(SEQUENCE_LENGTH)
        for index, word in enumerate(text_separate[:SEQUENCE_LENGTH]):
            if word in DICTIONARY:
                text_transl[index] = DICTIONARY.index(word)

        x_batch.append(text_transl)
        y_batch.append(label_transl)
    return [x_batch, y_batch]


def create_cnn():
    """创建 CNN + Full NN 的神经网络

    用词向量表示词，就是把字典中的每个词，都拓展成 EMBEDDING_SIZE 维。

    可以这样通俗理解：
    将每个词表示为 EMBEDDING_SIZE 维，其实意味着这 EMBEDDING_SIZE 维中的每个维，都单独表示了词的每个属性，
    例如第0维代表动名词性，第1维代表男女词性， 第2维代表褒贬词性……
    这些各不相关的属性两两正交，便可以，以给不同的维度赋予不同的权重的形式，表示出每个词。

    假如只有三个维度，[冷暖、方圆、大小]
    那么太阳可以表示为[9999,0,9999999]
    苹果[4,0,5]
    色子[-2,6,2]
    如果又来了一个铅球，我们衡量了一下各个权重，认为铅球也应该使用[4,0,5]表示，但是这就跟苹果一样了。为什么呢，
    这代表我们的维度设置小了。此时就考虑到，本应该使用4个维度来表示词，少了个重量。

    实际上，EMBEDDING_SIZE 中的每个维并不一定是和人类逻辑能够理解的维度一一对应，但总体是这个意思。
    应该设置多少维，这是我们需要调试的超参数。针对不同的词，每个维应该填充多大的数值，是机器自动学习的。
    """

    # 1. 生成[词向量个数, 300维]的随机均匀分布
    word2vec_random = np.random.uniform(-1.0, 1.0, [len(DICTIONARY), EMBEDDING_SIZE])
    # 2. 使用预训练好的词向量替换掉随机生成的分布
    get_word2vec(DICTIONARY, word2vec_random, WORD2VEC_DIC)
    # 3. 使用此分布创建Tensor对象
    phalanx = tf.Variable(initial_value=word2vec_random, dtype=tf.float32)

    embedded_chars = tf.nn.embedding_lookup(phalanx, x)
    embedded_expanded = tf.expand_dims(embedded_chars, -1)

    pooled_out = []
    for filter_window in FILTERS:
        # conv2d
        filter = tf.Variable(tf.random_normal([filter_window, EMBEDDING_SIZE, 1, NUM_FILTERS], stddev=0.1))
        """
        第2个维度为EMBEDDING_SIZE，是为了保证卷积是针对词与词之间的，而不会将词向量卷积掉。
        例如输入为
        我 [[1,0,5,9]
        今  [7,6,0,5]
        天  [5,6,8,8]
        真  [9,0,1,5]
        帅  [6,0,7,4]
        。  [0,0,4,6]]
        EMBEDDING_SIZE等于4表示，我们是拿着一个[x,4]的框去框选这个方阵，来实现卷积的，可以保证词向量不会被分裂看待。
        """
        conv = tf.nn.conv2d(embedded_expanded,
                            filter,
                            strides=[1, 1, 1, 1],
                            padding="VALID")
        b1 = tf.Variable(tf.constant(0.1, shape=[NUM_FILTERS]))
        l = tf.nn.relu(tf.nn.bias_add(conv, b1))

        # maxpooling
        pooled = tf.nn.max_pool(
            l,
            ksize=[1, INPUT_SIZE-filter_window+1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        pooled_out.append(pooled)

    # 拼合3个pooling的结果，准备输出给全连接层
    pooled_out_sum = tf.concat( pooled_out, 3)
    cnn_out = tf.reshape(pooled_out_sum, [-1, NUM_FILTERS*len(FILTERS)])
    # dropout
    cnn_out = tf.nn.dropout(cnn_out, dropout_keep_prob)

    # 添加3个全连接层
    w1 = tf.get_variable('w1', shape=[NUM_FILTERS*len(FILTERS), HIDDEN_DIM],
                         initializer=tf.contrib.layers.xavier_initializer())
    if USE_L2:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(L2_LAMBDA)(w1))
    b1 = tf.Variable(tf.constant(0.1, shape=[HIDDEN_DIM]))
    wb1 = tf.matmul(cnn_out, w1)+b1
    l1 = tf.nn.relu(wb1)

    w2 = tf.get_variable('w2', shape=[HIDDEN_DIM, HIDDEN_DIM],
                         initializer=tf.contrib.layers.xavier_initializer())
    if USE_L2:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(L2_LAMBDA)(w2))
    b2 = tf.Variable(tf.constant(0.1, shape=[HIDDEN_DIM]))
    wb2 = tf.matmul(l1, w2) + b2
    l2 = tf.nn.relu(wb2)

    w3 = tf.get_variable('w3', shape=[HIDDEN_DIM, OUTPUT_SIZE],
                         initializer=tf.contrib.layers.xavier_initializer())
    if USE_L2:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(L2_LAMBDA)(w3))
    b3 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_SIZE]))
    y = tf.matmul(l2, w3) + b3

    # loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    tf.add_to_collection('losses', loss)
    total_loss = tf.add_n(tf.get_collection('losses'))
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(total_loss)

    # 准确率
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    acc = tf.reduce_mean(tf.cast(correct, tf.float32))

    return [loss, optimizer, acc]

