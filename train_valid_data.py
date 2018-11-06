import os
import sys
import tensorflow as tf
import model

SAVER_DIC = 'output/model.ckpt'      # 保存学习到的参数
TRAIN_DIC = 'input/cnews.train.txt'  # 训练集存放地址
VALID_DIC = 'input/cnews.val.txt'    # 验证集存放地址
TRAIN_LOG = 'output/log_train.txt'   # 保存loss和正确率
VALID_LOG = 'output/log_valid.txt'   # 保存loss和正确率

CLASSES = model.CLASSES              # 输出的所有类别

TOTAL_ITERATIONS = 20000             # 总迭代次数
VALID_ITERATION = 100                # 每多少轮验证一次
DELAY_NUM = 20                       # 经过多少次验证后正确率还不提升，则停止

model.USE_L2 = True                  # 启用L2正则化
model.L2_LAMBDA = 0.04               # 正则化参数


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
    acc_value = 0.0   # 验证集正确率
    iterations = 0    # 迭代次数

    loss, optimizer, acc = model.create_cnn()

    # 用来保存进度
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # 若有上次的进度，则加载
    if os.path.isfile(SAVER_DIC+'.index'):
        saver.restore(sess, SAVER_DIC)

    for i in range(TOTAL_ITERATIONS):
        view_bar('训练进度：', i, TOTAL_ITERATIONS)
        x_data, y_data = model.get_some_date(file, model.BATCH_SIZE)    # 取批次数据
        feed_dict = {model.x: x_data, model.y_: y_data, model.dropout_keep_prob: 0.5}
        _, loss_test_value, acc_test_value = sess.run([optimizer, loss, acc], feed_dict=feed_dict)
        # 输出 train的 loss 和正确率
        with open(TRAIN_LOG, 'a') as f:
            f.write(str(i) + ',' + str(loss_test_value) + ',' + str(acc_test_value) + '\n')

        # 在验证集中测试
        if i%VALID_ITERATION == 0:
            model.USE_L2 = False  # 关闭L2正则化
            x_valid_data, y_valid_data = model.get_some_date(VALID_DIC, model.BATCH_SIZE)
            feed_dict = {model.x: x_valid_data, model.y_: y_valid_data, model.dropout_keep_prob: 1}
            _, loss_valid_value, acc_valid_value = sess.run([optimizer, loss, acc], feed_dict=feed_dict)
            if acc_valid_value > acc_value:
                acc_value = acc_valid_value
                saver.save(sess, SAVER_DIC)
                iterations = 0
                # 输出 valid的 loss 和正确率
                with open(VALID_LOG, 'a') as f:
                    f.write(str(i) + ',' + str(loss_valid_value) + ',' + str(acc_valid_value) + '\n')
            else:
                iterations += 1
                # 正确率久不提升，则停止
                if iterations >= DELAY_NUM:
                    break
            model.USE_L2 = True  # 启用L2正则化


if __name__ == '__main__':
    print('开始训练……')
    print('训练过程中进度即时保存。')
    print('进度保存在：                         {}'.format(SAVER_DIC))
    print('正在使用的训练数据为：               {}'.format(TRAIN_DIC))
    print('训练结果以"loss,正确率"的形式保存在：{}'.format(TRAIN_LOG))
    train_network(TRAIN_DIC)
    print('训练结束。')