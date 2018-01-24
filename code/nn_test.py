# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time

begin_time = time.time()

filename = "../output/embedding_mean.csv"
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

        


n_input = size
n_hidden_1 = int(size / 4)
n_hidden_2 = int(size / 8)
n_hidden_3 = 35
n_output = 4

Weights = {
    'w1_l' : tf.get_variable("W1_l",shape=[n_input, n_hidden_1],
                               initializer=tf.contrib.layers.xavier_initializer()),
    'w2_l' : tf.get_variable("W2_l", shape=[n_hidden_1, n_hidden_2],
                               initializer=tf.contrib.layers.xavier_initializer()),
    'w3_l' : tf.get_variable("W3_l", shape=[n_hidden_2, n_hidden_3],
                               initializer=tf.contrib.layers.xavier_initializer()),
    'w4' : tf.get_variable("W4", shape=[2*n_hidden_3, n_output],
                               initializer=tf.contrib.layers.xavier_initializer()),
    
    'w1_r' : tf.get_variable("W1_r",shape=[n_input, n_hidden_1],
                               initializer=tf.contrib.layers.xavier_initializer()),
    'w2_r' : tf.get_variable("W2_r", shape=[n_hidden_1, n_hidden_2],
                               initializer=tf.contrib.layers.xavier_initializer()),
    'w3_r' : tf.get_variable("W3_r", shape=[n_hidden_2, n_hidden_3],
                               initializer=tf.contrib.layers.xavier_initializer()),
}

Biases = {
    'b1_l' : tf.Variable(tf.random_normal([n_hidden_1])),
    'b2_l' : tf.Variable(tf.random_normal([n_hidden_2])),
    'b3_l' : tf.Variable(tf.random_normal([n_hidden_3])),
    'b4' : tf.Variable(tf.random_normal([n_output])),
    'b1_r' : tf.Variable(tf.random_normal([n_hidden_1])),
    'b2_r' : tf.Variable(tf.random_normal([n_hidden_2])),
    'b3_r' : tf.Variable(tf.random_normal([n_hidden_3])),
}

problem = tf.placeholder(tf.float32, [None, n_input])
answer = tf.placeholder(tf.float32, [None, n_input])
label = tf.placeholder(tf.float32, [None, 1])

def three_layer_network_left(X, keep_prob):
    z_left_1 = tf.add(tf.matmul(X, Weights.get('w1_l')), Biases.get('b1_l'))
    a_left_1 = tf.nn.relu(z_left_1)
    a_left_1 = tf.nn.dropout(a_left_1, keep_prob)
    
    z_left_2 = tf.add(tf.matmul(a_left_1, Weights.get('w2_l')), Biases.get('b2_l'))
    a_left_2 = tf.nn.relu(z_left_2)
    a_left_2 = tf.nn.dropout(a_left_2, keep_prob)

    
    z_left_3 = tf.add(tf.matmul(a_left_2, Weights.get('w3_l')), Biases.get('b3_l'))
    a_left_3 = tf.nn.relu(z_left_3)
    a_left_3 = tf.nn.dropout(a_left_3, keep_prob)
    
    return a_left_3
    
def three_layer_network_right(X, keep_prob):
    z_right_1 = tf.add(tf.matmul(X, Weights.get('w1_r')), Biases.get('b1_r'))
    a_right_1 = tf.nn.relu(z_right_1)
    a_right_1 = tf.nn.dropout(a_right_1, keep_prob)

    
    z_right_2 = tf.add(tf.matmul(a_right_1, Weights.get('w2_r')), Biases.get('b2_r'))
    a_right_2 = tf.nn.relu(z_right_2)
    a_right_2 = tf.nn.dropout(a_right_2, keep_prob)
    
    z_right_3 = tf.add(tf.matmul(a_right_2, Weights.get('w3_r')), Biases.get('b3_r'))
    a_right_3 = tf.nn.relu(z_right_3)
    a_right_3 = tf.nn.dropout(a_right_3, keep_prob)
    
    return a_right_3


def get_output(problem, answer):
    a_left_3 = three_layer_network_left(problem, 0.7)
    a_right_3 = three_layer_network_right(answer, 0.7)
    a_all_3 = tf.concat([a_left_3, a_right_3], 1)
    
    a_output = tf.add(tf.matmul(a_all_3, Weights.get('w4')), Biases.get('b4'))
    return a_output

learning_rate = 0.001

predict = get_output(problem=problem, answer=answer)
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

#==============================================================================
# train_val divide
#==============================================================================




training_epochs = 300
batch_size = 10
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
                                                      label: label_train})
        aver_cost += c
    aver_cost /= total_batch
    if epoch % display_step == 0:
        print("epochs is {}, average cost is {}".format(epoch+1, aver_cost))

    if epoch % (10) == 0:
        accuracy_all = sess.run(accuracy, feed_dict = {problem: problem_val, answer: answer_val,
                                                  label: label_val.reshape(-1, 1)})
        print(accuracy_all)
    
print("runtime is", (time.time() - begin_time))
##    saver = tf.train.Saver()
##    model_path = "../input/model_save/model_nn.ckpt"
##    save_path = saver.save(sess, model_path)
##    
#    #saver = tf.train.Saver()
#    #saver.restore(sess, "../input/model_save/model_nn_.ckpt")
#    #accuracy_all = sess.run(accuracy, feed_dict = {problem: problem_val, answer: answer_val,
#    #                                              label: label_val.reshape(-1, 1)})
#    #print(accuracy_all)
#
#
