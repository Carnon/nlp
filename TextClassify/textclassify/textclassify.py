import tensorflow as tf
import argparse
#import numpy as np
import os
#import configparser
from tqdm import tqdm

from textclassify.textdata import TextData
from textclassify.cnnmodel import CNNModel
from textclassify.rnnmodel import RNNModel

class TextClassify(object):
    def __init__(self):
        self.args = None

        self.textData = None
        self.model = None
        self.saver = None

    @staticmethod
    def parseArg(args):
        parser = argparse.ArgumentParser()

        globalArgs = parser.add_argument_group('Global option')
        globalArgs.add_argument('--root_dir',type=str,default=os.getcwd(),help='the root directory')

        dataArgs = parser.add_argument_group('Data option')
        dataArgs.add_argument('--corpus_dir',type=str,default=os.path.join(os.getcwd(),'data/corpus',),help='the corpus data dir')
        dataArgs.add_argument('--max_doc_len',type=int,default=4096,help='filter_doc_size')
        dataArgs.add_argument('--embedding_size', type=int, default=128, help='the word dimen')

        nnArgs = parser.add_argument_group('NNModel option')
        nnArgs.add_argument('--dropout', type=float, default=0.9, help='drop_out')
        nnArgs.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')

        cnnArgs = parser.add_argument_group('CNNModel option')
        cnnArgs.add_argument('--use_cnn',type=bool,default=False,help='use_cnn or not')
        cnnArgs.add_argument('--filter_sizes',type=list,default=[3,4,5],help='filter_size')
        cnnArgs.add_argument('--num_filters',type=int,default=128,help='num_filter')
        #cnnArgs.add_argument('--l2_reg_lambda',type=float,default=0.0,help='l2_reg_lambda')

        rnnArgs = parser.add_argument_group('RNNModel option')
        rnnArgs.add_argument('--use_rnn',type=bool,default=True,help='use rnn or not')
        rnnArgs.add_argument('--rnn_cell_size',type=int,default=128,help='hidden cell')

        trainArgs = parser.add_argument_group('Train option')
        trainArgs.add_argument('--batch_size',type=int,default=16,help='bath_size')
        trainArgs.add_argument('--epoch_num',type=int,default=2,help='epochNum')
        trainArgs.add_argument('--save_every',type=int,default=2,help='save model every steps')
        trainArgs.add_argument('--check_every',type=int,default=2,help='check the model accuracy every steps')
        trainArgs.add_argument('--model_name',type=str,default='',help='model-name')

        return parser.parse_args(args)

    def main(self,args=None):
        self.args = self.parseArg(self.args)
        self.textData = TextData(self.args)
        if self.args.use_cnn:
            self.model =CNNModel(self.args,self.textData)
        elif self.args.use_rnn:
            self.model = RNNModel(self.args,self.textData)
        else:
            print("error..... only support cnn or rnn ")
            return
        self.sess = tf.Session()
        print("Initialize variables...")
        self.saver = tf.train.Saver(max_to_keep=20)
        self.sess.run(tf.global_variables_initializer())
        self.main_train(self.sess)

    def main_train(self,sess):
        step =0
        for _ in range(self.args.epoch_num):
            for input_x,input_y in tqdm(self.textData.get_batches(),desc='Training',leave=False):
                if self.args.use_rnn and input_x.shape[0] != self.args.batch_size:
                    continue    #dropout rnn last batch that not equal batch_size can cause exception
                feed_dict = {
                    self.model.input_x : input_x,
                    self.model.input_y : input_y,
                    self.model.dropout : self.args.dropout
                }
                _,accuracy = sess.run([self.model.train,self.model.accuracy],feed_dict)
                step += 1
                if step % self.args.check_every == 0:
                    print("step:{}  accuracy:{}".format(step,accuracy))
                if step % self.args.save_every == 0:
                    self.save_session(sess)

    def save_session(self,sess):
        model_dir = os.path.join(self.args.root_dir, 'save', 'model')
        if self.args.model_name:
            model_dir += '-' + self.args.model_name + '/'
        else:
            model_dir += '/'
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        self.saver.save(sess, model_dir)