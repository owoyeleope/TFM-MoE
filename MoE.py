import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys


species=['H', 'H2', 'O', 'O2', 'OH', 'H2O', 'N2', 'CO', 'HCO', 'CO2', 'CH3', 'CH4', 'HO2', 'H2O2', 
         'C2H2', 'C2H4', 'NC12H26', 'CH2(S)', 'C2H6', 'HCCO', 'C12H25-2', 'C9H19-1', 'C12H25O-5', 'CH2O']

inp_list = ['chi', 'PV', 'Zvar', 'Z']
ii = species.index('CO2') 

data = np.load('data5D_1000K.npy')
data_len  = data.shape[0]

inputs_unscaled = data[:,:5]
targets_unscaled = data[:,5+ii:5+ii+1]
inp_scaler = MinMaxScaler()
inp_scaler.fit(inputs_unscaled)
inp = inp_scaler.transform(inputs_unscaled)
out_scaler = MinMaxScaler(feature_range=(0, 1))
out_scaler.fit(targets_unscaled)
targets = out_scaler.transform(targets_unscaled)
out = targets

num_neurons = 5
num_neurons_g = 5
num_experts = 2
#######################################################################
tf.reset_default_graph()

x = tf.placeholder(tf.float64, [None, 5], name = 'x')
y = tf.placeholder(tf.float64, [None, 1], name ='y')
w_ph = tf.placeholder(tf.float64, [None], name ='scaler')
scaler = tf.placeholder(tf.float64, [num_experts], name ='scaler')
beta = tf.placeholder(tf.float64, shape = (), name ='beta')

W1 = []; W2 = []; b1 = []; b2 = [];
error = []; loss_list = [];
net_out = [];

for i in range(num_experts):
    W1 += [tf.Variable(tf.random_uniform([5, num_neurons], dtype = tf.float64))]
    b1 += [tf.Variable(tf.constant(0.1, shape=[num_neurons], dtype = tf.float64))]
    W2 += [tf.Variable(tf.random_uniform([num_neurons, 1], dtype = tf.float64))]
    b2 += [tf.Variable(tf.constant(0.1, shape=[1], dtype = tf.float64))]
    net_out += [tf.nn.sigmoid(tf.matmul(tf.nn.sigmoid(tf.matmul(x, W1[i]) + b1[i]), W2[i])+b2[i])]
    error += [net_out[i] - y]
    loss_list += [tf.square(error[i])]
    
error_all = tf.concat(error, axis = 1)
loss_all = tf.concat(loss_list, axis = 1)
#######################################################################
W1_g = tf.Variable(tf.random_uniform([5, num_neurons_g], maxval = 2, minval = -2, dtype = tf.float64), name = "W1_g")
b1_g = tf.Variable(tf.constant(0.0, shape=[num_neurons_g], dtype = tf.float64))
W2_g = tf.Variable(tf.random_uniform([num_neurons_g, num_experts], maxval = 2, minval = -2, dtype = tf.float64), name = "W2_g")
b2_g = tf.Variable(tf.constant(0.0, shape=[num_experts], dtype = tf.float64))

g_out = tf.matmul(tf.nn.sigmoid(tf.matmul(x, W1_g)), W2_g)
tf.identity(g_out,"g_out")
net_out_g = tf.nn.softmax(g_out, name = "gating_output")

loss = tf.reduce_mean(tf.reduce_sum(net_out_g*loss_all, axis = 1))
############################################################
error_n = [];
for i in range(num_experts):
    error_n += [tf.abs(error[i])*1]
error_all_n = tf.concat(error_n, axis = 1)

lossx = -tf.log(tf.reduce_sum(net_out_g*tf.exp(-beta*error_all_n**2), axis = 1) + 1e-15)

denum = tf.exp(-beta*error_n[0]**2) + 1e-15
for i in range(1, num_experts):
    denum += tf.exp(-beta*error_n[i]**2)

abc = (tf.exp(-beta*error_all_n**2) + 1e-15)/denum
abc2 = tf.where(tf.equal(tf.reduce_max(abc, axis=1, keepdims=True), abc), tf.ones_like(abc), tf.zeros_like(abc))
abc_ = tf.round(abc)
loss_g = tf.losses.softmax_cross_entropy(abc2, g_out, weights = 1.0)
lssx = tf.reduce_mean(lossx)
net_out_g2 = tf.where(tf.equal(tf.reduce_max(net_out_g, axis=1, keepdims=True), net_out_g), tf.ones_like(net_out_g), tf.zeros_like(net_out_g))
lossx2 = tf.reduce_sum(net_out_g2*loss_all, axis = 1)
############################################################
num_inds = []
for ii in range(num_experts):
    num_inds += [tf.where(tf.equal(tf.argmax(net_out_g, axis = 1), ii))]

#learning_rate = 0.01
decay_rate = 0.95
decay_steps = 100
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.01
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                       decay_steps, decay_rate, staircase=False)

optimizer = tf.train.AdamOptimizer(learning_rate=0.005,epsilon=1e-8).minimize(lssx, var_list = 
                                  [W1,b1,W2,b2])
optimizer2 = tf.train.AdamOptimizer(learning_rate=0.005,epsilon=1e-8).minimize(loss_g, var_list = 
                                  [W1_g, W2_g])
optimizer_r = []
lssx2 = []
for i in range(num_experts):
    lssx2 += [tf.reduce_mean(tf.gather(error[i]**2, num_inds[i]))] 
    optimizer_r +=  [tf.train.AdamOptimizer(learning_rate=0.001,epsilon=1e-8).minimize(lssx2[i], var_list = 
                                  [W1[i],b1[i],W2[i],b2[i]])]

init = tf.global_variables_initializer()
session = tf.Session()

session.run(init)

print(session.run(W1))
beta_np = 1
scaler_np = np.ones(shape = (num_experts))
w_np = np.ones(shape = (data.shape[0]))
for i in range(20000):
    fd = {x: inp, y: out, beta: beta_np, scaler: scaler_np, w_ph: w_np}
    session.run(optimizer, feed_dict = fd)
    session.run(optimizer2, feed_dict = fd)
    
    if i%100 == 0:
        fd = {x: inp, y: out, beta: beta_np, scaler: scaler_np}
        inds_np = session.run(num_inds, feed_dict = fd)

        
        abc_np = session.run(abc, feed_dict = fd)
        aabb = []
        aabb1 = []
        for ii in range(num_experts):
            aabb1 += [np.where(np.argmax(abc_np, axis = 1) == ii)[0]]
            aabb += [len(aabb1[ii])]
        abc_max = abc_np.max(axis = 0)
        print('abc:', abc_max, aabb)
        g_output = session.run(net_out_g, feed_dict = fd)
        for ii in range(num_experts):
            length = len(inds_np[ii])
            w_np[aabb1[ii]] = 1 - aabb[ii]/data.shape[0]
            print(i, length)
        if np.min(aabb) > 50000:
            w_np[:] = 1.0
        absum = 0
        ab = session.run(loss_all, feed_dict = fd)
        for ii in range(num_experts):
            absum += np.sum(ab[np.where(np.argmax(g_output, axis = 1) == ii),ii])
            
        print('gating output max:', g_output.max(axis = 0))
        if num_experts > 1:
            ab_ = ab
            wc = np.argmin(ab_/(np.sort(ab_, axis = 1)[:,1:2]), axis = 0)
            wsc_ = ab_[wc,:]
            zxc = []
            for iii in range(num_experts):
                if (np.sort(wsc_[iii])[1] - wsc_[iii,iii]) > 1e-10:
                    zxc += [(np.log(0.99) - np.log(0.01))/(np.sort(wsc_[iii])[1] - wsc_[iii,iii])]
            if len(zxc) > 0:
                beta_np = np.max(zxc)
            print("beta_np:", beta_np)
			
        sys.stdout.flush()
    if i%100 == 0:
        print("writing data out...")
        np.save('inds_np', inds_np)
        np.save('inds_np_e', aabb1)
        output_graph_def = tf.graph_util.convert_variables_to_constants(session, tf.get_default_graph().as_graph_def(), ["gating_output", "g_out", "W1_g", "W2_g"])    
        with tf.gfile.GFile("./G0_1.pb", "wb") as f:
            f.write(output_graph_def.SerializeToString())
    sys.stdout.flush()
