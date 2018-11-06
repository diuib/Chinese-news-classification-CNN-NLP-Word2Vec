import os
import sys
import tensorflow as tf
import model

SAVER_DIC = 'output/model.ckpt'          # 保存学习到的参数
TEST_LOG = 'output/log_test.txt'         # 保存loss和正确率
TEST_DIC = 'input/cnews.test.txt'       # 测试数据存放的地址
CLASSES = model.CLASSES                  # 输出的所有类别

ITERATIONS = 100                  # 迭代次数
model.BATCH_SIZE = 50           # 每批次数量

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

def test(file='input/cnews.test.txt'):
    """测试部分
    """
    loss, optimizer, acc = model.create_cnn()

    # 用来读取参数
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # 若进度存在，则读取学习到的参数
    if os.path.isfile(SAVER_DIC + '.index'):
        saver.restore(sess, SAVER_DIC)

    for i in range(ITERATIONS):
        view_bar('测试进度：', i, ITERATIONS)
        x_test_data, y_test_data = model.get_some_date(file, model.BATCH_SIZE)
        feed_dict = {model.x: x_test_data, model.y_: y_test_data, model.dropout_keep_prob: 1}
        loss_test_value, acc_test_value = sess.run([loss, acc], feed_dict=feed_dict)
        # 输出 test loss 和正确率
        with open(TEST_LOG, 'a') as f:
            f.write(str(i) + ',' + str(loss_test_value) + ',' + str(acc_test_value) + '\n')


if __name__ == '__main__':
    print('开始测试……')
    print('正在使用的测试数据为：               {}'.format(TEST_DIC))
    print('测试结果以"loss,正确率"的形式保存在：{}'.format(TEST_LOG))
    test(TEST_DIC)
    print('测试结束。')