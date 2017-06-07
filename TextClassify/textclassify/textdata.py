import os
import codecs
import re

import jieba
import numpy as np
from tqdm import tqdm
from tensorflow.contrib import learn
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

class TextData(object):
    def __init__(self,args):
        self.args = args
        corpus_dir = self.args.corpus_dir

        self.load_data(corpus_dir)

    def load_data(self,corpus_dir):
        self.text = []
        self.label = []
        self.label_name = {}
        self.max_doc_len = 0
        self.label_num = 0
        self.vocab_size = 0

        raw_text = []
        raw_label = []
        tag_list = os.listdir(corpus_dir)
        for tag in tqdm(tag_list,desc='load_data',leave=False):
            data_path = os.path.join(corpus_dir,tag)
            for data_file in os.listdir(data_path):
                file_name = os.path.join(data_path,data_file)
                with codecs.open(file_name,'r',encoding='utf-8') as fr_raw:
                    raw_content = fr_raw.read()
                    text_word = [word for word in jieba.cut(raw_content) if re.match(u".*[\u4e00-\u9fa5]+", word)]
                    if text_word.__len__() < self.args.max_doc_len:
                        raw_text.append(text_word)
                        raw_label.append(tag)

        labelEncode = LabelEncoder()
        num_label = labelEncode.fit_transform(raw_label)
        self.label = OneHotEncoder(sparse=False).fit_transform(np.reshape(num_label,[-1,1]))
        self.label_num = len(labelEncode.classes_)

        #self.max_doc_len = max([len(doc) for doc in raw_text])
        self.max_doc_len = self.args.max_doc_len
        vocab_processor = learn.preprocessing.VocabularyProcessor(self.max_doc_len,tokenizer_fn=tokenizer_fn)
        self.text = np.array(list(vocab_processor.fit_transform(raw_text)))
        self.vocab_size = len(vocab_processor.vocabulary_)

    def shuffle_data(self):
        np.random.seed(3)
        shuffled = np.random.permutation(np.arange(len(self.label)))
        self.text = self.text[shuffled]
        self.label = self.label[shuffled]

    def get_batches(self):
        self.shuffle_data()

        sample_size = len(self.text)
        batch_size = self.args.batch_size

        for i in range(0,sample_size,batch_size):
            yield self.text[i:min(i+batch_size,sample_size)],self.label[i:min(i+batch_size,sample_size)]

#chinese cut word has completed instead of tensorflow cut word that cannt support chinese word
def tokenizer_fn(iterator):
    for value in iterator:
        yield value
