import os
import random
import thulac
import pickle
from collections import Counter
import tensorflow as tf
import numpy as np
import fileinput

TOTAL_LINE = 0
DICTIONARY = []
WORD2VECS = []
CLASSES = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']
SAVER_DIC = 'input/model.ckpt' # 保存学习到的参数
TRAIN_LOG = 'input/log.txt'
WORD2VEC_DIC = 'word2vec/百度Word/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
SEQUENCE_LENGTH = 5000  # 句子长度，超过后截断，为了训练速度

'''
def read_word2vec(file):
    if os.path.isfile(file):
        with fileinput.input(files=(WORD2VEC_DIC), openhook=fileinput.hook_encoded('UTF-8')) as f:
            if not f.isfirstline():
                for line in f:
                    word_and_vec = line.split(' ')
                    DICTIONARY.append(word_and_vec[0])
                    WORD2VECS.append(word_and_vec[1:])


read_word2vec(WORD2VEC_DIC)
print(DICTIONARY)
'''
def get_word2vec(dic, pha, file):
    if os.path.isfile(file):
        with fileinput.input(files=(WORD2VEC_DIC), openhook=fileinput.hook_encoded('UTF-8')) as f:
            for line in f:
                word_and_vec = line.split(' ')
                word = word_and_vec[0]
                vec = word_and_vec[1:301]
                if word in dic:
                    pha[DICTIONARY.index(word)] = vec

def pretreatment_date(gather_file="input/cnews.train.txt"):
    '''
    数据预处理
    '''
    global TOTAL_LINE
    global DICTIONARY
    lex = Counter()

    # 统计文本总共有多少行
    total_line_num = 0
    for total_line_num, line in enumerate(open(gather_file, 'r', encoding='UTF-8')):
        pass
    total_line_num += 1
    TOTAL_LINE = total_line_num
    print("总共", TOTAL_LINE, "行")

    # 若事先处理过且已保存，则直接读取字典；若不存在，则处理好后保存
    if os.path.isfile("input/dictionary.pickle"):
        with open("input/dictionary.pickle", "rb") as f:
            DICTIONARY = pickle.load(f)
    else:
        count = 0
        thu1 = thulac.thulac(seg_only=True)  # 只进行分词，不进行词性标注
        for line in open(gather_file, "r", encoding='UTF-8'):  # 为了省内存，没有使用readlines()
            count += 1
            if count % 500 == 0:
                print('处理进度：', count / TOTAL_LINE * 100, '%')

            words = thu1.cut(line[3:], True).split(' ')
            lex.update(words)

        '''
        # 排序，查看字典表，确定排除的虚词及样本量较低的词
        with open("checkdic.txt", "w", encoding='utf-8') as f:
            for word in lex:
                f.write(word + ',' + str(lex[word]) + '\n')
        '''

        # 样本量较高的词为虚词或无意义词（如本例中的“今天”），较低的词无法学习到知识
        for word in lex:
            if 1000 < lex[word] < 3500:
                DICTIONARY.append(word)

        # 保存入文件
        with open("input/dictionary.pickle", 'wb') as f:
            pickle.dump(DICTIONARY, f)


pretreatment_date("input/cnews.train.txt") # 创建/读取字典


def get_some_date(file_name='input/cnews.train.txt', num=100):
    # 读取文件中的随机num行,并处理
    x_batch = []
    y_batch = []
    getlines = []
    i = 0
    thu1 = thulac.thulac(seg_only=True)  # 只进行分词，不进行词性标注

    #此种读取算法，可保存每行被取到的概率相同
    with open(file_name, 'r', encoding='UTF-8') as f:
        for line in f:
            i += 1
            if len(getlines) <= num:
                getlines.append(line)
            else:
                random_num = random.randint(0, i)
                if random_num <= num:
                    random_line = random.randint(0, num)
                    getlines[random_line] = line

    for line in getlines:
        label = str(line.split('\t')[0])
        text = str(line.split('\t')[1:])

        label_transl = np.zeros(len(CLASSES))
        for i in range(len(CLASSES)):
            if CLASSES[i] == label:
                label_transl[i] = 1

        text_separate = thu1.cut(text, True).split(' ')
        text_transl = np.zeros(SEQUENCE_LENGTH)
        for index, word in enumerate(text_separate[:50]):
            if word in DICTIONARY:
                text_transl[index] = DICTIONARY.index(word)

        x_batch.append(text_transl)
        y_batch.append(label_transl)
    return x_batch, y_batch


INPUT_SIZE = SEQUENCE_LENGTH
OUTPUT_SIZE = len(CLASSES)
HIDDEN_DIM = 1000 # 隐藏层神经元数

VOCAB_SIZE = len(DICTIONARY)
BATCH_SIZE = 50 # 每批次数量
EMBEDDING_SIZE = 300 # 每个词的词向量维度
NUM_FILTERS = 256 # 每个核卷积后输出的维度
FILTERS = [2, 3, 5]

x = tf.placeholder(tf.int32, [None, INPUT_SIZE])
y_ = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])
dropout_keep_prob = tf.placeholder(tf.float32)



def create_cnn():
    global  dropout_keep_prob
    global  x
    '''
    把One-hot字典中每个元素（每个词），都拓展成EMBEDDING_SIZE维。换句话说，用词向量来表示词。
    每个词有EMBEDDING_SIZE维这件事，可以理解为EMBEDDING_SIZE中的每个维度分别是动名词性、男女词性、褒贬词性……
    这些各不相同的维度两两正交，便可以通过给不同的维度赋予不同的权重表示每个词。
    假如只有三个维度，[冷暖、方圆、大小]
    那么太阳可以表示为[9999,0,9999999]
    苹果[4,0,5]
    色子[-2,6,2]
    如果又来了一个铅球，我们发现也是得用[4,0,5]表示，跟苹果一样，那就代表我们的维度设置小了，此时就想到原本应该再增加一个叫重量的维度。
    实际EMBEDDING_SIZE中每个维并不一定和人类逻辑上能够理解的维度一一对应，但总体是这个意思。
    设置多少维，是我们要调试的超参数。每个词分别用多大的数值填充各个维，是机器自动学习的。
    '''
    word2vec_random = np.random.uniform(-1.0, 1.0, [VOCAB_SIZE, EMBEDDING_SIZE])
    get_word2vec(DICTIONARY, word2vec_random, WORD2VEC_DIC)
    phalanx = tf.Variable(initial_value=word2vec_random, dtype=tf.float32)

    embedded_chars = tf.nn.embedding_lookup(phalanx, x)
    embeded_expanded = tf.expand_dims(embedded_chars, -1)

    pooled_out = []

    for filter_window in FILTERS:
        # conv2d
        filter = tf.Variable(tf.random_normal([filter_window, EMBEDDING_SIZE, 1, NUM_FILTERS], stddev=0.1))
        '''
        第2个维度为EMBEDDING_SIZE，则可以保证卷积时是针对词与词的卷积，而不会将每个词的向量卷积掉。
        比如输入为
        我 [1,0,5,9]
        今 [7,6,0,5]
        天 [5,6,8,8]
        真 [9,0,1,5]
        帅 [6,0,7,4]
        。 [0,0,4,6]
        第二个维度等于4表示卷积时拿着一个[x,4]的框在框选这个方阵去卷积，可以保证不会将词向量分裂开去看待。
        '''
        conv = tf.nn.conv2d(embeded_expanded,
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

    cnn_out = tf.nn.dropout(cnn_out, dropout_keep_prob)


    w1 = tf.get_variable('w1', shape=[NUM_FILTERS*len(FILTERS), HIDDEN_DIM], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.constant(0.1, shape=[HIDDEN_DIM]))
    wb1 = tf.matmul(cnn_out, w1)+b1
    # 加入drop
    #wb1 = tf.nn.dropout(wb1, dropout_keep_prob)
    l1 = tf.nn.relu(wb1)


    w2 = tf.get_variable('w2', shape=[HIDDEN_DIM, HIDDEN_DIM],
                         initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.constant(0.1, shape=[HIDDEN_DIM]))
    wb2 = tf.matmul(l1, w2) + b2
    l2 = tf.nn.relu(wb2)


    w3 = tf.get_variable('w3', shape=[HIDDEN_DIM, OUTPUT_SIZE],
                        initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_SIZE]))
    y = tf.matmul(l2, w3) + b3

    return y


def train_network(file = 'input/cnews.train.txt'):
    y = create_cnn()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

    # 准确率
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    acc = tf.reduce_mean(tf.cast(correct, tf.float32))

    # 用来保存进度
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # 若进度存在，则读取学习到的参数
    if os.path.isfile(SAVER_DIC+'.index'):
        saver.restore(sess, SAVER_DIC)

    for i in range(1000):
        print('now is ',i/500*100,'%')
        x_data, y_data = get_some_date(file, BATCH_SIZE)
        feed_dict = {x: x_data, y_: y_data, dropout_keep_prob: 0.5}
        _, loss_value, acc_value = sess.run([optimizer, loss, acc], feed_dict=feed_dict)
        if i%50 == 0:
            saver.save(sess, SAVER_DIC)
        with open(TRAIN_LOG, 'a') as f:
           f.write(str(loss_value) + ',' + str(acc_value) + '\n')

def test(file = 'input/cnews.test.txt'):
    y = create_cnn()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    # 准确率
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    acc = tf.reduce_mean(tf.cast(correct, tf.float32))

    # 用来读取参数
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # 若进度存在，则读取学习到的参数
    if os.path.isfile(SAVER_DIC):
        saver.restore(sess, SAVER_DIC)
    
    for i in range(5):
        print('now is ', i)
        x_data, y_data = get_some_date(file, 200)
        feed_dict = {x: x_data, y_: y_data, dropout_keep_prob: 1}
        loss_value, acc_value = sess.run([loss, acc], feed_dict=feed_dict)
        with open(TRAIN_LOG, 'a') as f:
           f.write(str(loss_value) + ',' + str(acc_value) + '\n')

train_network('input/cnews.train.txt')
# print("IT'S TESTING--------------------------------")
# test('input/cnews.test.txt')
