# author: Shane Yu  date: April 8, 2017
import subprocess
import logging
from gensim.models import word2vec
import multiprocessing
from gensim.corpora import WikiCorpus
from gensim.models import word2vec
import multiprocessing
import jieba
from os import path


class build(object):
    """
    build class can build the word2vec model automatically from downloading the wiki raw data all the way to the training porcess,
    and the model will be created in the CURRENT directory.

    P.S. An extra directiory will be created during the process.
    """
    def __init__(self, jieba_dict_init_path, jieba_dict_customize_path, stopwordsPath, dimension):
        self.jieba_dict_init_path = jieba_dict_init_path
        self.jieba_dict_customize_path = jieba_dict_customize_path
        self.stopwordsPath = stopwordsPath
        self.dimension = dimension

    def creatBuildDir(self):
        subprocess.call(['mkdir', 'build'])

    def getWiki(self):
        subprocess.call(['wget', 'https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2', '-P', './build/'])

    def wikiToTxt(self):
        # This function takes about 25 minutes
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        wiki_corpus = WikiCorpus('./build/zhwiki-latest-pages-articles.xml.bz2', dictionary={})
        
        texts_num = 0
        with open('./build/wiki_texts.txt', 'w') as output:
            for text in wiki_corpus.get_texts(): # get_texts()一次會回傳一篇文章，其中一句話為一個item組成一個list
                output.write(' '.join(text) + '\n')
                texts_num += 1
                if texts_num % 10000 == 0:
                    logging.info("壓縮檔轉為文字檔（以空格分開句子），已處理 %d 篇文章" % texts_num)

    def opencc(self):
        subprocess.call(['opencc', '-i', './build/wiki_texts.txt', '-o', './build/wiki_zh_tw.txt'])

    def segmentation(self):
        # takes about 30 minutes
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        # jieba custom setting.
        jieba.initialize(self.jieba_dict_init_path)
        jieba.load_userdict(self.jieba_dict_customize_path)

        # load stopwords set
        stopwordset = set()
        with open(self.stopwordsPath, 'r', encoding='utf-8') as sw:
            for line in sw:
                stopwordset.add(line.strip('\n'))

        output = open('./build/wiki_seg.txt', 'w')
        
        texts_num = 0
        
        with open('./build/wiki_zh_tw.txt','r') as content :
            for line in content:
                words = jieba.cut(line, cut_all=False)
                for word in words:
                    if word not in stopwordset:
                        output.write(word +' ')
                texts_num += 1
                if texts_num % 10000 == 0:
                    logging.info("已完成前 %d 行的斷詞" % texts_num)
        output.close()

    def train(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        sentences = word2vec.Text8Corpus('./build/wiki_seg.txt')
        model = word2vec.Word2Vec(sentences, size=self.dimension, workers=multiprocessing.cpu_count())

        # Save model.
        model.wv.save_word2vec_format('./med' + str(self.dimension) + '.model.bin', binary=True)

    def main(self):
        if not path.exists('./med' + str(self.dimension) + '.model.bin'):
            if not path.exists('./build/zhwiki-latest-pages-articles.xml.bz2'):
                print('========================== 開始下載wiki壓縮檔 ==========================')
                self.getWiki()
            if not path.exists('./build/wiki_texts.txt'):
                print('========================== 開始將壓縮檔轉成文字檔 ==========================')
                self.wikiToTxt()
            if not path.exists('./build/wiki_zh_tw.txt'):
                print('========================== 開始繁轉簡 ==========================')
                self.opencc()
            if not path.exists('./build/wiki_seg.txt'):
                print('========================== 開始斷詞 ==========================')
                self.segmentation()
            print('========================== 開始訓練 ==========================')
            self.train()
        print('========================== ' + str(self.dimension) + '維model訓練完畢，model存放在當前目錄 ==========================')



if __name__ == '__main__':
    import sys
    obj = build('../../jieba_dictionary/dict.txt.big',
                '../../jieba_dictionary/NameDict_Ch_v2',
                '../../jieba_dictionary/stopwords',
                400)
    obj.main()
