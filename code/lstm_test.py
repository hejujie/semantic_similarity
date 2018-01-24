# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 19:55:53 2018

@author: jie
"""

import tensorflow as tf
import numpy as np
import time

begin_time = time.time()

filename = "../output/embedding_all_60.csv"
data = np.loadtxt(filename, delimiter = ',')

size = int((data.shape[1] - 1) / 2)
problem_data = data[:, 0:size]
answer_data = data[:, size:-1]
label_data = data[:, -1]
print(problem_data.shape, answer_data.shape, label_data.shape)
#==============================================================================
# train and valid set divide
#==============================================================================
seed = np.random.seed(0)
index_val = np.random.choice(data.shape[0], int(data.shape[0]*0.3))
problem_val = problem_data[index_val]
answer_val = answer_data[index_val]
label_val = label_data[index_val]
#index_train = [x for x in list(range(data.shape[0]))]
index_train = []
for x in list(range(data.shape[0])):
    if x not in index_val:
        index_train.append(x)
problem_train = problem_data[index_train]
answer_train = answer_data[index_train]
label_train = label_data[index_train]


n_inputs = 20
n_hiddens = 20
n_outputs = 4

Weights = {
    'w3' : tf.get_variable("W3", shape=[n_inputs * 4, n_hiddens],
                           initializer=tf.contrib.layers.xavier_initializer()),
    'w4' : tf.get_variable("W4", shape=[n_hiddens, n_outputs],
                               initializer=tf.contrib.layers.xavier_initializer())
}

Biases = {
    'b3' : tf.Variable(tf.random_normal([n_hiddens])),
    'b4' : tf.Variable(tf.random_normal([n_outputs]))
}

n_shape = 1280
problem = tf.placeholder(tf.float32, [None, n_shape])
answer = tf.placeholder(tf.float32, [None, n_shape])
label = tf.placeholder(tf.float32, [None, 1])
keep_probs = tf.placeholder(tf.float32)


def bi_rnn(X, keep_probs, scope, n_inputs, n_hiddens, n_layers):
    
    X = tf.unstack(tf.transpose(X, perm=[2,0,1]))
    with tf.name_scope("fw" + scope), tf.variable_scope("fw" + scope):
        stacked_rnn_fw = []
        for _ in range(n_layers):
            forward_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hiddens, forget_bias=1, state_is_tuple=True)
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(forward_cell, output_keep_prob=keep_probs)
            stacked_rnn_fw.append(lstm_fw_cell)
        lstm_fw_cell_n = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)
        
    with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
        stacked_rnn_bw = []
        for _ in range(n_layers):
            backword_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hiddens, forget_bias=1, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(backword_cell, output_keep_prob=keep_probs)
            stacked_rnn_bw.append(lstm_bw_cell)
        lstm_bw_cell_n = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)
        
    with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
        outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell_n, lstm_bw_cell_n, X, dtype=tf.float32)
    
    return outputs[-1]    


def get_output(problem, answer, keep_probs):
    problem = tf.reshape(problem, shape = [-1, 20, 64])
    answer = tf.reshape(answer, shape = [-1, 20, 64])
    
    a_left_2 = bi_rnn(problem, keep_probs, "problem1", n_inputs, 20, 4)
    a_right_2 = bi_rnn(answer, keep_probs, "answer1", n_inputs, 20, 4)
    
#     a_left_2 = tf.reshape(a_left_2, [-1, 7*7*64])
#     a_right_2 = tf.reshape(a_right_2, [-1, 7*7*64])
    print(a_left_2.shape, a_right_2.shape)
    a_all_2 = tf.concat([a_left_2, a_right_2], 1)
    a_all_3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(a_all_2, Weights.get('w3')), Biases.get('b3')))
    a_all_3 = tf.nn.dropout(a_all_3, keep_probs)
    
    a_output = tf.add(tf.matmul(a_all_3, Weights.get('w4')), Biases.get('b4'))
    return a_output


learning_rate = 0.001

predict = get_output(problem=problem, answer=answer, keep_probs=keep_probs)
label_onehot = tf.one_hot(indices= tf.reshape((tf.cast(label, tf.int32)), [1, -1]), depth=4, axis=0)
label_onehot = tf.transpose(label_onehot)
label_onehot = tf.squeeze(label_onehot)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict, 
                                                              labels = label_onehot))
optimize = tf.train.AdamOptimizer(learning_rate).minimize(cost)

initial = tf.global_variables_initializer()
square = tf.square(tf.cast(tf.subtract(tf.arg_max(predict, 1), tf.arg_max(label_onehot, 1)), tf.float32))
accuracy = tf.sqrt(tf.reduce_mean(square))
# accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"))



training_epochs = 1000
batch_size = 256
display_step = 2
# with tf.Session() as sess:
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True 
config.gpu_options.per_process_gpu_memory_fraction = 0.7
#with tf.device('gpu:0'):
sess = tf.Session(config=config) 
#sess = tf.Session()
sess.run(initial)


for epoch in range(training_epochs):
    aver_cost = 0
    total_batch = int(data.shape[0] / batch_size)
    for i in range(total_batch):
        index = np.random.choice(data.shape[0], batch_size)
        problem_train = problem_data[index]
        answer_train = answer_data[index]
        label_train = label_data[index].reshape(-1, 1)
        _, c = sess.run([optimize, cost], feed_dict = {problem: problem_train, 
                                                      answer: problem_train,
                                                      label: label_train,
                                                      keep_probs: 0.7})
        aver_cost += c
    aver_cost /= total_batch
    if epoch % display_step == 0:
        print("epochs is {}, average cost is {}".format(epoch+1, aver_cost))

    if epoch % (10) == 0:
        accuracy_all = sess.run(accuracy, feed_dict = {problem: problem_val, 
                                                       answer: answer_val,
                                                       label: label_val.reshape(-1, 1), 
                                                      keep_probs:1})
        print(accuracy_all)
    
print("runtime is", (time.time() - begin_time))
saver = tf.train.Saver()
model_path = "../input/model_save/model_lstm_epoch_1000.ckpt"
save_path = saver.save(sess, model_path)
##    
#    #saver = tf.train.Saver()
#    #saver.restore(sess, "../input/model_save/model_nn_.ckpt")
#    #accuracy_all = sess.run(accuracy, feed_dict = {problem: problem_val, answer: answer_val,
#    #                                              label: label_val.reshape(-1, 1)})
#    #print(accuracy_all)
#
#
