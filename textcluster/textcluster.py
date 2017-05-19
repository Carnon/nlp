import argparse
import codecs
import os
from collections import defaultdict
from collections import Counter

import jieba
from gensim.models import doc2vec
from sklearn.cluster import KMeans

class TextCluster(object):

	def __init__(self):
		self.arg = None
		self.DATA_DIR = 'data'
		self.RESULT_DIR = 'save'

	def parseArgs(self,args):
		parse = argparse.ArgumentParser()

		globalArgs = parse.add_argument_group("Dataset options")
		globalArgs.add_argument('--corpus',type=str,default='isr_question.txt',help='the base corpus file such as isr_question.txt')

		trainArgs = parse.add_argument_group("Train options")
		trainArgs.add_argument('--numCluster',type=int,default=3,help='the number of cluster')
		trainArgs.add_argument('--maxIter',type=int,default=30,help='the number of iter')
		return parse.parse_args(args)

	def main(self,args=None):
		print("this is a sentence cluster demo....")

		self.arg = self.parseArgs(args)
		#if self.arg.corpus == None:
		#	raise "please input the corpus fileName"

		self.readCorpus()
		self.mainTrain()
		self.save()

	def mainTrain(self):
		print('train model....')
		questions = doc2vec.TaggedLineDocument(self.seg_file)
		dv = doc2vec.Doc2Vec(questions,size=200,window=300,min_count=1,workers=4)
		self.km = KMeans(n_clusters=self.arg.numCluster,max_iter=self.arg.maxIter)
		self.km.fit([vec for vec in dv.docvecs])

		print('ok')


		#tf_vectorizer = CountVectorizer()
		#tf_matrix = tf_vectorizer.fit_transform(self.sen_seg_list)
		#tfidf_transformer = TfidfTransformer()
		#tfidf_matrix = tfidf_transformer.fit_transform(tf_matrix)
		#self.km = KMeans(n_clusters=self.arg.numCluster,max_iter=self.arg.maxIter)
		#self.km.fit(tfidf_matrix)

	def readCorpus(self):
		print('read corpus ....')
		self.corpus_list = []
		sen_seg_list = []
		dirName = os.path.join(os.getcwd(),self.DATA_DIR,self.arg.corpus)
		with codecs.open(dirName,encoding='utf-8') as fr:
			self.corpus_list=fr.readlines()
			for corpus in self.corpus_list:
				sen_seg_list.append(" ".join(jieba.cut(corpus.strip())))
		fileName = 'sen_seg.txt'
		self.seg_file = os.path.join(os.getcwd(),self.RESULT_DIR,fileName)
		with codecs.open(self.seg_file,'w+',encoding='utf-8') as fw:
			for sen in sen_seg_list:
				fw.writelines(sen+'\n')


	def save(self):
		print('save result....')
		result = defaultdict(list)
		for sen,tag in zip(self.corpus_list,self.km.labels_):
			result[tag].append(sen)
		fileName = 'cluster_{}_result_doc2vec_kmean.txt'.format(self.arg.numCluster)
		cluster_ResFileName = os.path.join(os.getcwd(),self.RESULT_DIR,fileName)
		with codecs.open(cluster_ResFileName,'w+','utf-8') as fw:
			for tag,sentences in result.items():
				for sen in sentences:
					fw.write(str(tag)+' +++$+++ '+sen)
