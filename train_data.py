import os
import sys
import tensorflow as tf
import model

SAVER_DIC = 'output/model.ckpt'      # 保存学习到的参数
TRAIN_LOG = 'output/log_train.txt'         # 保存loss和正确率
TRAIN_DIC = 'input/cnews.train.txt' # 训练数据存放的地址
CLASSES = model.CLASSES              # 输出的所有类别

ITERATIONS = 500                # 迭代次数
BATCH_SIZE = 50                 # 每批次数量

SEQUENCE_LENGTH = 50            # 句子最长含词量。为了训练速度，超过后，多余的词截断
INPUT_SIZE = SEQUENCE_LENGTH    # 神经网络输入尺寸
OUTPUT_SIZE = len(CLASSES)      # 神经网络输出尺寸


def view_bar(text, num, total):
    """优化进度条显示"""
    rate = num / total
    rate_num = int(rate * 100)
    r = '\r' + text + '[%s%s]%d%%' % ("=" * rate_num, " " * (100 - rate_num), rate_num,)
    sys.stdout.write(r)
    sys.stdout.flush()

def train_network(file = 'input/cnews.train.txt'):
    """训练部分
    """
    y = model.create_cnn()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=model.y_, logits=y))
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

    # 准确率
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(model.y_, 1))
    acc = tf.reduce_mean(tf.cast(correct, tf.float32))

    # 用来保存进度
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # 若有上次的进度，则加载
    if os.path.isfile(SAVER_DIC+'.index'):
        saver.restore(sess, SAVER_DIC)

    for i in range(ITERATIONS):
        view_bar('训练进度：', i, ITERATIONS)
        x_data, y_data = model.get_some_date(file, BATCH_SIZE)    # 取批次数据
        feed_dict = {model.x: x_data, model.y_: y_data, model.dropout_keep_prob: 0.5}
        _, loss_value, acc_value = sess.run([optimizer, loss, acc], feed_dict=feed_dict)
        if i%10 == 0:
            saver.save(sess, SAVER_DIC)
        # 输出 loss 和正确率
        with open(TRAIN_LOG, 'a') as f:
           f.write(str(loss_value) + ',' + str(acc_value) + '\n')


if __name__ == '__main__':
    print('开始训练……')
    print('训练过程中进度即时保存。')
    print('进度保存在：                         {}'.format(SAVER_DIC))
    print('正在使用的训练数据为：               {}'.format(TRAIN_DIC))
    print('训练结果以"loss,正确率"的形式保存在：{}'.format(TRAIN_LOG))
    train_network(TRAIN_DIC)
    print('训练结束。')