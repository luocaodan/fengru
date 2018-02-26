#!/usr/bin/env Python
# coding=utf-8

# In[10]:

import tensorflow as tf
import numpy as np
import os
import re
import json
import time
import random
import collections


# In[2]:

class AttPoolModel(object):
   
    def __init__(self, batch_size, encoding_size, embedding_size, max_q_len, max_a_len,    
                                    vocab_size,
                                    session,
                                    grad_norm_clip = 2., 
                                    l2_reg_coef=1e-4,                   
                                    name='Wiki_APN'):
        self._batch_size = batch_size
        self._encode_size = encoding_size        #128
        self._embedding_size = embedding_size    #84    
        self._vocab_size = vocab_size
        self._query_len = max_q_len     
        self._doc_len = max_a_len
        self._sess = session
        self._name = name
        
        self._build_placeholders()
        self._build_variables()
        with tf.variable_scope('cell1'):  
            cell = tf.nn.rnn_cell.GRUCell(encoding_size, activation = tf.nn.elu)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 1, state_is_tuple=True) 
            self.fw_cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self._keep_prob,
                                                                                                                                             output_keep_prob=self._keep_prob)
        with tf.variable_scope('cell2'):  
            cell = tf.nn.rnn_cell.GRUCell(encoding_size, activation = tf.nn.elu)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 1, state_is_tuple=True) 
            self.bw_cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self._keep_prob, output_keep_prob=self._keep_prob)
        
        cos_q_a = self._compute_score(self.question, self.q_len, self.answer, self.a_len,  self.a_lap)
        length = 10
        pred_order = tf.nn.top_k(- tf.nn.top_k(- cos_q_a, length)[1], length)[1]   # get the order
        self.pred_order = length - pred_order  # 值越小排名越靠前       

        lambdas = tf.map_fn(get_lambdas, (cos_q_a, self.label), dtype = tf.float32)  
        self._opt = tf.train.AdamOptimizer(learning_rate=self._learning_rate) 
        l2 = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.01), tf.trainable_variables())
        lambdas = tf.stop_gradient(lambdas)
        loss = cos_q_a*lambdas + l2
        
        grads_and_vars = self._opt.compute_gradients(loss, var_list=tf.trainable_variables() )#,  grad_loss= lambdas)  
        clipped_grads_and_vars = [(tf.clip_by_norm(g, grad_norm_clip), v) for g,v in grads_and_vars if g is not None] 
        
        self._global_step = tf.Variable(0, name='step', dtype = tf.int32)
        self._operation = self._opt.apply_gradients(clipped_grads_and_vars, global_step=self._global_step)

        self.log_file = cos_q_a 
        self.saver = tf.train.Saver(tf.global_variables())
        
    def _build_placeholders(self):
        self.question = tf.placeholder(tf.int32, [self._batch_size, self._query_len])
        self.q_len = tf.placeholder(tf.int32, [self._batch_size])
        self.answer    = tf.placeholder(tf.int32,   [10*self._batch_size, self._doc_len]) 
        self.a_len = tf.placeholder(tf.int32, [10*self._batch_size]) 
        self.label = tf.placeholder(tf.int32, [self._batch_size, 10])
        
        self.a_lap = tf.placeholder(tf.int32, [10*self._batch_size, self._doc_len])
        
        self._keep_prob = tf.placeholder(tf.float32)
        self._learning_rate = tf.placeholder(tf.float32)

    def _build_variables(self): 
        with tf.device("/cpu:0"):
            init = tf.random_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32) 
            emb = np.array(glove_emb() , dtype=np.float32)
            self._embeddings = tf.get_variable('emb', dtype = tf.float32, initializer = emb, trainable=False)
            self.overlap_emb = tf.get_variable("overlap", [3, 50], initializer= init )
            self._U = tf.get_variable("U",   [self._batch_size, 2*self._encode_size, 2*self._encode_size], initializer= init )
            
            self.alpha = [0]*7
            for i in range(7):
                self.alpha[i] = tf.get_variable("alpha%i"%i, dtype = tf.float32,  initializer=tf.truncated_normal([1], mean=0.2, stddev=0.05))

            self._b = tf.get_variable("bias", [self._batch_size, 2*self._encode_size, 1],  initializer=tf.constant_initializer(1.0))

    def _embed(self, sequence):
        return tf.nn.embedding_lookup(self._embeddings, sequence)

    def _compute_score(self, questions, q_lens, answers, a_lens, a_overlap):
        #  get query and answer's represation, calculate their similarity 
        with tf.variable_scope('question') as scope:        #Encode query        
            encoded_questions =  tf.nn.dropout(self._embed(questions) , self._keep_prob)
            encoded_questions = self.ques_enc(encoded_questions, q_lens, 20, self._encode_size)
            ques1 = tf.tile(tf.expand_dims(encoded_questions, 1), [1, self._doc_len, 1])

        with tf.variable_scope('answers') as scope:  # Encode hypothesis
            a = tf.split(answers, 10, 0) 
            a_lap = tf.split(a_overlap, 10, 0) 
            a_len = tf.split(a_lens, 10, 0)
            encoded_a = [0]*10
                         
            for i in range(10):
                if i>0: scope.reuse_variables()   #所有回答共用一个CNN来编码
                tmp = self._embed(a[i]) 
                lap_features = tf.nn.embedding_lookup(self.overlap_emb, a_lap[i])
                tmp = tf.concat([tmp, lap_features], 2)
                print(tmp)
                print(ques1)
                tmp = tf.concat([tmp, ques1], 2)
                
                encoded_a[i] = self._bidirectional_encode(tmp, a_len[i], self._doc_len, self._encode_size)
        
        with tf.variable_scope('correlations') as scope:  
            cos_q_a = [0]*10          
            for i in range(10):
                if i>0: scope.reuse_variables()
                tmp =[]
                for j in range(5):
                    c = self.batch_cosin(encoded_questions, encoded_a[i][j]) 
                    tmp.append(c)# *self.alpha[j])
                cos_q_a[i] =  sum(tmp)
                
            return tf.stack(cos_q_a, 1)   # tensor shape (batch_size, 10)

            
    def _bidirectional_encode(self, sequence, seq_len, l,  size):
        filter_sizes = [1,2,3,4,5]
        outs = []
        sequence = tf.expand_dims(sequence, -1)
        
        for i, filter_size in enumerate(filter_sizes): 
            with tf.variable_scope("conv1-maxpool-%s" % filter_size) as scope:
                init1 = tf.truncated_normal([filter_size, self._embedding_size +50+ 2*size, 1, 2*size], stddev=0.1)

                W1 = tf.get_variable("filter1", [filter_size, self._embedding_size +50+ 2*size, 1, 2*size],
                                                                                                 dtype=tf.float32, initializer= tf.constant_initializer(0.1) )
                b1 = tf.get_variable("conv_b1", [2*size], dtype=tf.float32, initializer= tf.constant_initializer(0.3) )
                conv1 = tf.nn.conv2d( sequence, W1, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
                h1 = tf.nn.elu(tf.nn.bias_add(conv1, b1), name="elu1")
                h1 = tf.nn.dropout(h1, self._keep_prob)
                pooled = tf.nn.max_pool(h1, ksize=[1, l - filter_size+1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool1")

                pooled = tf.reshape(pooled, [-1, 2*size])   # (batch_size, 2*size) 
                outs.append(pooled)
        
        return outs 
        
    def ques_enc(self, sequence, seq_len, l, size):
        with tf.variable_scope('encode'):
            with tf.variable_scope("_FW") as fw_scope:
                fw_cell = self.fw_cell #RHNCell(size, scope ='fw', activation = tf.nn.elu)    # 用双向GRU对序列编码
                fw_state = fw_cell.zero_state(self._batch_size, tf.float32)
                
                output_fw, output_state_fw = tf.nn.dynamic_rnn( cell=fw_cell, inputs=sequence, initial_state=fw_state, scope=fw_scope, swap_memory=True)                      
            with tf.variable_scope("_BW") as bw_scope:
                bw_cell = self.bw_cell  #RHNCell(size, scope ='bw', activation = tf.nn.elu) 
                
                bw_state = bw_cell.zero_state(self._batch_size, tf.float32)
                
                inputs_reverse = tf.reverse_sequence( input=sequence, seq_lengths=seq_len, seq_dim=1, batch_dim=0)            
                tmp, output_state_bw = tf.nn.dynamic_rnn( cell=bw_cell, inputs=inputs_reverse,
                                                         sequence_length=seq_len, initial_state = bw_state, scope=bw_scope, swap_memory=True)
                output_bw = tf.reverse_sequence( input=tmp, seq_lengths=seq_len, seq_dim=1, batch_dim=0)            
            encoded = tf.concat((output_fw, output_bw), 2)
          
            mask = tf.sequence_mask(seq_len, encoded.get_shape()[1], dtype= tf.float32)
            encoded *= tf.expand_dims(mask, -1)   #perform masking according to their lengths
            encoded =  tf.reduce_max(encoded, 1)
            return encoded

    def batch_cosin(self, encoded_queries, encoded_docs):   #encoded_queries:(batch_size, q_len, 2*size) 

        r_q, r_a = encoded_queries, encoded_docs
        q_norm = tf.nn.l2_normalize(r_q, 1)  
        a_norm = tf.nn.l2_normalize(r_a, 1)
        L2 = tf.reduce_sum((q_norm - a_norm)*(q_norm - a_norm), 1)
        GESD = 1./(1+L2)*tf.sigmoid(tf.reduce_sum(q_norm*a_norm, 1) + 1)
        cos_theta = tf.reduce_sum(q_norm*a_norm, 1)

        bilinear = tf.matmul( tf.matmul(tf.expand_dims(q_norm, 1), self._U) , tf.expand_dims(a_norm, 2) ) # qWa
        bilinear = tf.squeeze(bilinear)
        return  GESD+cos_theta + bilinear
        
    def fit(self, data_batch, lr=1e-3):
        question, q_len, answer, a_len, label, a_lap, threads = data_batch
        feed_dict ={ self.question : question,
                              self.q_len       : q_len, 
                              self.answer    : answer, 
                              self.a_len       : a_len, 
                              self.label        : label, 
                   
                              self.a_lap        : a_lap,
                              self._keep_prob   : 0.8, 
                              self._learning_rate: lr }                   
        pred_order, log_file, step, _ = self._sess.run([self.pred_order, self.log_file, self._global_step, self._operation], feed_dict=feed_dict)

        map_acc, mrr_acc = MAP_MRR(pred_order, label, threads)
        return map_acc, mrr_acc, log_file, step

    def predict(self, data_batch, lr=1e-3):
        question, q_len, answer, a_len, label, a_lap, threads = data_batch
        feed_dict ={ self.question : question,
                              self.q_len       : q_len, 
                              self.answer    : answer, 
                              self.a_len       : a_len, 
                              self.label        : label, 
                  
                              self.a_lap        : a_lap,
                              self._keep_prob   : 1., 
                              self._learning_rate: lr  }
                     
        pred_order, log_file, step = self._sess.run([self.pred_order, self.log_file, self._global_step], feed_dict=feed_dict)

        map_acc, mrr_acc = MAP_MRR(pred_order, label, threads)
        return map_acc, mrr_acc, log_file, step


# In[3]:

# if change the max len , first delete data.pkl, or it will use the orignal data
tf.flags.DEFINE_integer("vocab",20000, "numbers of sinograms")
tf.flags.DEFINE_integer("max_q", 20, "maximum length of question")
tf.flags.DEFINE_integer("max_a", 40, "maximum length of answer")
# # Hyperparameters
tf.flags.DEFINE_integer("embedding",100, "Dimension of word embedding (default: 100)")
tf.flags.DEFINE_integer("encoding", 100, "Number of neural units every layer in bidirectional GRU")
tf.flags.DEFINE_float("dropout_keep_prob", 0.6, "Dropout keep probability (default: 0.6)")
tf.flags.DEFINE_float("l2_reg_lambda", 1e-4, "L2 regularizaion lambda (default: 0.0001)")
tf.flags.DEFINE_float("learning_rate", 0.05, "AdamOptimizer learning rate (default: 0.001)")
tf.flags.DEFINE_float("learning_rate_decay", 0.8, "lr will decay after half epoch of non-decreasing loss (default: 0.8)")
tf.flags.DEFINE_integer("batch", 20, "Batch Size (default: 50)")
tf.flags.DEFINE_integer("early_stop", 5, "validating times to wait before early stop if no improvement in valid accurancy")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 4)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on validation set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 10000, "Save model after this many steps (default: 10000)")

tf.flags.DEFINE_boolean("debug", False, "Debug (load smaller dataset)")    
tf.flags.DEFINE_boolean("trace", False, "Whether to generate a debug trace of training step")
tf.flags.DEFINE_string("trace_file", "timeline.ctf.json", "Chrome tracefile name for debugging(default: timeline.ctf.json)")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


# In[5]:

def glove_emb():
    if os.path.exists('glove_emb.json'):  
        with open('glove_emb.json', 'r') as f:
            emb = json.load(f)
    else:        
        dic1 = {}
        word2id = get_dic()
        words = word2id.keys()
        emb = [0]*(len(words) )
        for i,line in enumerate(open(glove_path)):
            dic1[line.split()[0]] = map(float, line.split()[1:])
        for i,w in enumerate(words):   # 1~20000
            if w in dic1:
                emb[word2id[w]] = dic1[w]
            else:                               # Random normal initialization.    1152 words  aren't  in  glove 
                emb[word2id[w]] = [float(i) for i in np.random.normal(scale=0.1, size=100).astype(np.float32)]
        emb[0] = [0]*100# list(np.random.normal(scale=0.1, size=100).astype(np.float32)) 
        with open('glove_emb.json', 'w') as f:
            json.dump(emb, f)
    return np.array(emb)


# In[12]:

def get_lambdas(elems, length=10):  # # if pred is [0.34 , 0.21, 0.65, 0.12, 0.7] , then y's order is [2,1,3,0,4]
    pred_score, label_gold  = elems
    label_gold = tf.to_float(label_gold)
    
    order = tf.nn.top_k(-tf.nn.top_k(-pred_score, k=length)[1], k=length)[1] 
    order = tf.to_float(order)
      
    lambdas = [0]*length 
    for i in range(length):        
        for j in range(length):
                      
            P = tf.sign(label_gold[i] - label_gold[j])           
            lambda_ij = -P*tf.sigmoid(-P*( pred_score[i]- pred_score[j])) 
            delta = delta_ndcg(order, label_gold, i,j, length)
            lambdas[i] += lambda_ij#*(1+delta )
     
    return tf.convert_to_tensor(lambdas,  dtype=tf.float32)


# In[14]:

def delta_ndcg(label_raw, label_gold, i,j,length=10):
    idx = list(range(length))
    idx[i],idx[j] = idx[j],idx[i]
    label_swap = tf.gather(label_raw, idx) 
    
    delta = ndcg(label_raw, label_gold)-ndcg(label_swap, label_gold)
    return tf.abs(delta)


# In[15]:

def MAP_MRR(pred_order, gold_label, threads):    # (batch_size, 10)
    def func(pred, label, N):     # one dimension vector (10,).  N is the number of answers for a question
        if 1 in label:  
            # rank place of true and false answers
            rank = np.argsort(np.argsort(pred[:N]))+1       # this will convert [9, 3, 6, 5, 7, 1, 4, 2, 8, 0] to [4, 1, 3, 2] if N=4
            rank_1 = np.sort(rank[np.where(label==1)])   # rank place of true answers
            score_map = np.arange(1., len(rank_1)+1)/rank_1

            return np.mean(score_map), 1./rank_1[0]
    scores = filter(None, map(func, pred_order, gold_label, threads))
    maps, mrrs = zip(*scores)
    return np.mean(maps), np.mean(mrrs)


# In[17]:

def ndcg(label_raw, label_gold, length = 10):  
    label_sorted, index = tf.nn.top_k(label_gold, k=length)   
    label = tf.gather(label_raw, index)
    discount = tf.log(np.arange(length, dtype=np.float32)+2)    
    
    dcg  = tf.reduce_sum(label/discount , -1)
    idcg = tf.reduce_sum(label_sorted/discount , -1)
    return dcg/idcg


# In[18]:

tf.reset_default_graph()
with tf.Session() as sess:
    model = AttPoolModel(FLAGS.batch, FLAGS.encoding, FLAGS.embedding,
                         FLAGS.max_q, FLAGS.max_a, FLAGS.vocab, sess)


# In[19]:

dic_path = 'dict.txt'
pattern =  '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'


# In[21]:

def get_dic():
    with open(dic_path, encoding='UTF-8') as f:
        word2id = collections.OrderedDict()
        lines = f.readlines()
        for line in lines[:20000]:  
            k,v = line.split()
            word2id[k] = int(v)
        return word2id


# In[22]:

dic = get_dic()               
def map_sentence(sentence, max_len):
    if sentence == 'None':
        return list(np.random.randint(1, 20000, size=(max_len,))) , max_len
    sentence = re.sub(pattern, ' ', sentence)
    sentence = re.sub(r'[0-9]' , ' ', sentence)
    ids = []
    for w in sentence.lower().split():
        if w in dic: # and w not in stop:
            ids.append(dic[w])
        #else: ids.append(0)
    #ids = [dic[w] if (w in dic and w not in stop) for w in sentence.lower().split()]
    vector = [0]*max_len 
    if len(ids)>max_len:
        vector = ids[:max_len]
        length = max_len
    else:
        vector[:len(ids)] = ids
        if 2*len(ids) < max_len:
            vector[len(ids):2*len(ids)] = ids
        length = len(ids)
    if len(vector) !=max_len: print(vector)
    return vector, length


# In[23]:

# question is a sentence, answer is the list containing 10 sentences
def ques_ans_handle(question, answer, max_q_len = 20, max_a_len = 40):
    ques, q_len, ans, a_len, a_lap, threads = [], [], [], [], [], []
    ans_base = 0
    for i in range(20):
        vec_q, len_q = map_sentence(question[i], max_q_len)
        ques.append(vec_q)
        q_len.append(len_q)
        N = 0
        for ans_num in range(10):
            vec_a, len_a = map_sentence(answer[ans_base + ans_num], max_a_len)
            ans.append(vec_a)
            a_len.append(len_a)
            if answer[ans_num] != 'None':
                N+=1
            overlap = set(vec_q[:len_q]).intersection(vec_a[:len_a])            
#           assert(0 not in overlap)          
                        
            tmp=[2]*max_a_len
            for k in range(len_a):
                tmp[k] = 1 if vec_a[k] in overlap else 0
            a_lap.append(tmp)
        ans_base = ans_base + 10
        threads.append(N)
    one_sentence_batch = (ques, q_len, ans, a_len, a_lap, threads)
    
    return one_sentence_batch, threads[0]


# In[24]:

def load_ques_ans(ques_path, ans_path):
    with open(ques_path) as f:
        ques = f.read().splitlines()
    
    with open(ans_path) as f:
        ans = f.read().splitlines()
    
    return ques, ans


# In[25]:

def make_it_batch(question, answer):
    for i in range(19):
        question.append('None')
        for j in range(10):
            answer.append('None')
    test_sentence, thread = ques_ans_handle(question, answer)
    
    return test_sentence, thread


# In[26]:

def predict_for_show(model, sentence):
    question, q_len, answer, a_len, a_lap, threads = sentence
    feed_dict ={ 
        model.question : question,
        model.q_len : q_len, 
        model.answer : answer, 
        model.a_len : a_len, 
        model.a_lap : a_lap,
        model._keep_prob   : 1. 
    }
    pred_order = sess.run([model.pred_order], feed_dict = feed_dict)
    return pred_order


# In[27]:

def sort_answers(ques_path, ans_path):
    q, a = load_ques_ans(ques_path, ans_path)
    test_sentence, thread = make_it_batch(q, a)
    sentence_pred = predict_for_show(model, test_sentence)
    pred_array = sentence_pred[0]
    order = pred_array[0]
    print('The question: %s'%q[0])
    print('The answer order is: ', order)
    final_order = np.argsort(np.argsort(order[:thread]))
    print('The final order is (without None answers): ', final_order)
    order_index = np.argsort(final_order)
    print('A few answers:')
    index = ['One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten']
    for i in range(thread):
        print('Thumb %s: %s'%(index[i], a[order_index[i]]))

def make_answers(q, a):
    res = []
    test_sentence, thread = make_it_batch(q, a)
    sentence_pred = predict_for_show(model, test_sentence)
    pred_array = sentence_pred[0]
    order = pred_array[0]
    final_order = np.argsort(np.argsort(order[:thread]))
    order_index = np.argsort(final_order)
    for i in range(thread):
        #print('Thumb %s: %s'%(index[i], a[order_index[i]]))
        res.append(a[order_index[i]])
    return res

# In[29]:

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, tf.train.latest_checkpoint('model'))


# In[30]:
#q, a = load_ques_ans('question.txt', 'answer.txt')
#print(a)
#print(q)
#print(a)
#sort_answers('question.txt', 'answer.txt')
#a = ['African immigration to the United States refers to immigrants to the United States who are or were nationals of Africa .']
#make_answers(['HOW AFRICAN AMERICANS WERE IMMIGRATED TO THE US'], a)

# In[ ]:
