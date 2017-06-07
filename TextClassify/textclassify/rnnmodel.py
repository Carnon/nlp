import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell

class RNNModel(object):
    def __init__(self,args,text_data):
        self.args = args
        self.text_data = text_data

        self.input_x = None
        self.input_y = None
        self.dropout = None

        self.losses = None
        self.train = None
        self.prediction = None
        self.accuracy = None

        self.build_network()

    def build_network(self):

        embedding_size = self.args.embedding_size
        rnn_cell_size = self.args.rnn_cell_size
        batch_size = self.args.batch_size
        learning_rate = self.args.learning_rate
        max_doc_len = self.args.max_doc_len

        label_num = self.text_data.label_num
        vocab_size = self.text_data.vocab_size
        print('vocab_size: {} label_num: {} max_doc_len: {} batch_size: {} embedding_size: {} rnn_cell_size: {}'.format(vocab_size,label_num,max_doc_len,batch_size,embedding_size,rnn_cell_size))
        self.input_x = tf.placeholder(tf.int32,[None,max_doc_len],name='input_x')
        self.input_y = tf.placeholder(tf.float32,[None,label_num],name='input_y')
        self.dropout = tf.placeholder(tf.float32,name='drop_out')

        We = tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0))
        embedding_char = tf.nn.embedding_lookup(We,self.input_x)
        embedding_char_expand = tf.reshape(embedding_char,[-1,embedding_size])

        W_in = tf.Variable(tf.random_uniform([embedding_size,rnn_cell_size]))
        b_in = tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[rnn_cell_size,]))
        X_in = tf.matmul(embedding_char_expand,W_in)+b_in
        Xs = tf.reshape(X_in,[-1,max_doc_len,rnn_cell_size])

        cell = BasicLSTMCell(rnn_cell_size)
        init_state = cell.zero_state(batch_size,dtype=tf.float32)
        outputs,final_state = tf.nn.dynamic_rnn(cell,Xs,initial_state=init_state)
        #outputs:(batch,time_step,input)
        output = outputs[:,-1,:]
        tf.nn.xw_plus_b(output,)
        scores = tf.layers.dense(outputs[:,-1,:],label_num)
        self.prediction = tf.argmax(scores, 1)

        self.losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=scores))
        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.losses)
        correct_predictions = tf.equal(self.prediction,tf.argmax(self.input_y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,tf.float32))
