import tensorflow as tf

class CNNModel(object):
    def __init__(self,args,textData):
        print('CNNModel creationg...')
        self.args = args
        self.textData = textData


        self.input_x = None
        self.input_y = None
        self.dropout = None

        self.scores = None
        self.predictions = None
        self.loss = None
        self.accuracy = None
        self.train = None

        self.buildNetwork()

    def buildNetwork(self):
        num_filters = self.args.num_filters
        filter_sizes = self.args.filter_sizes
        embedding_size = self.args.embedding_size
        #l2_reg_lambda = self.args.l2_reg_lambda

        vocab_size = self.textData.vocab_size
        max_doc_len = self.textData.max_doc_len
        label_num = self.textData.label_num

        self.input_x = tf.placeholder(tf.int32,[None,max_doc_len],name='input_x')
        self.input_y = tf.placeholder(tf.float32,[None,label_num],name='input_y')
        self.dropout = tf.placeholder(tf.float32,name='dropout')

        Wx = tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0))
        embedding_chars = tf.nn.embedding_lookup(Wx,self.input_x)
        embedding_chars_expanded = tf.expand_dims(embedding_chars,-1)

        pooled_outputs = []
        for i,filter_size in enumerate(filter_sizes):
            filter_shape = [filter_size,embedding_size,1,num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name="W")
            b = tf.Variable(tf.constant(0.1,shape=[num_filters]),name='b')
            conv = tf.nn.conv2d(embedding_chars_expanded,W,strides=[1,1,1,1],padding='VALID')  #(?,54,1,128)
            h = tf.nn.relu(tf.nn.bias_add(conv,b),name='relu')
            pooled = tf.nn.max_pool(h,ksize=[1,max_doc_len-filter_size+1,1,1],strides=[1,1,1,1],padding='VALID') #(?,1,1,128)
            pooled_outputs.append(pooled)

        num_filters_toltal = num_filters*len(filter_sizes)

        h_pool = tf.concat(pooled_outputs,3)
        h_pool_flat = tf.reshape(h_pool,[-1,num_filters_toltal])
        h_drop = tf.nn.dropout(h_pool_flat,self.dropout)

        W = tf.get_variable("W",shape=[num_filters_toltal,label_num],initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1,shape=[label_num]),name='b')
        scores = tf.nn.xw_plus_b(h_drop,W,b,name='score')
        self.predictions = tf.argmax(scores,1,name='prediction')
        self.losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores,labels=self.input_y))
        self.train = tf.train.AdamOptimizer().minimize(self.losses)
        correct_predictions = tf.equal(self.predictions,tf.argmax(self.input_y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,'float'),name='accuracy')
