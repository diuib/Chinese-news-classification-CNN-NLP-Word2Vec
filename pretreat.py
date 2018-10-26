import os
import sys
import pickle
import thulac
import fileinput
from collections import Counter

TRAIN_DATA_DIC = "input/cnews.train.txt"        # 训练数据包存放地址，用来总结字典
DICTIONARY_DIC = "input/dictionary.pickle"      # 存放总结出的字典，以节省时间
CHICK_DICT_DIC = "output/checkdic.txt"          # 可自行将词频打印出来，以确定 WORD_FREQUENCY_LOW 和 WORD_FREQUENCY_HIGH

WORD_FREQUENCY_LOW = 1000
WORD_FREQUENCY_HIGH = 3500

thu1 = thulac.thulac(seg_only=True)  # THULAC 中文分词包


def view_bar(text, num, total):
    """优化进度条显示"""
    rate = num / total
    rate_num = int(rate * 100)
    r = '\r' + text + '[%s%s]%d%%' % ("=" * rate_num, " " * (100 - rate_num), rate_num,)
    sys.stdout.write(r)
    sys.stdout.flush()


def pretreatment_date():
    """数据预处理。
    1. 按条读取数据，并分词
    2. 将所有词放在一起排序、观察后，舍掉词频过高和过低的词
    3. 将词放入 list 中做成字典，并保存在硬盘里，以待以后使用
    """
    dic = []

    print('正在加载字典……')
    # 统计数据包总条数
    total_line_num = 0
    for total_line_num, line in enumerate(open(TRAIN_DATA_DIC, 'r', encoding='UTF-8')):
        pass
    total_line_num += 1

    if os.path.isfile(DICTIONARY_DIC):
        # 若事先处理过且已保存，则直接读取
        with open(DICTIONARY_DIC, "rb") as f:
            dic = pickle.load(f)
    else:
        print("总共", total_line_num, "条")
        # 若不存在，则处理好后保存
        count = 0
        lex = Counter()
        with fileinput.input(files=TRAIN_DATA_DIC, openhook=fileinput.hook_encoded('UTF-8')) as f:
            # 为了节省内存，使用 fileinput 读数据包
            for line in f:
                count += 1
                if count % 500 == 0:
                    # print('处理进度：', count / total_line_num * 100, '%')
                    view_bar('处理进度：', count, total_line_num)

                words = thu1.cut(line[3:], True).split(' ')
                lex.update(words)

        """排序，查看字典表，确定排除的虚词及样本量较低的词
        with open(CHICK_DICT_DIC, "w", encoding='utf-8') as f:
            for word in lex:
                f.write(word + ',' + str(lex[word]) + '\n')
        """

        # 样本量较高的词为虚词或无意义词（如本案例中的“今天”），较低的词无法学习到知识
        for word in lex:
            if WORD_FREQUENCY_LOW < lex[word] < WORD_FREQUENCY_HIGH:
                dic.append(word)

        # 保存入文件
        with open(DICTIONARY_DIC, 'wb') as f:
            pickle.dump(dic, f)

    print('字典加载完成。')
    return dic


if __name__ == '__main__':
    pretreatment_date()