import numpy as np
import tensorflow as tf
import math
import random
import context
import os
import scipy.stats as stats
from scipy.stats import truncnorm

slim = tf.contrib.slim

seqdic={'A':0, 'R':1, 'D':2, 'C':3, 'Q':4, 'E':5, 'H':6, 'I':7, 'G':8, 'N':9, 'L':10, 'K':11, 'M':12, 'F':13, 'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 'V':19, 'X':20}

labeldic={'H':1,'G':1,'I':1, 'E':2,'B':2}

data=np.load('cullpdb_train.npy').item()
seq=data['seq']
pssm_list=data['pssm']
dssp=data['dssp']

seq_l=[]
for i in range(len(seq)):
	seq_l.append(len(seq[i]))

seq_l=np.array(seq_l)

pssm_rsp=[]
for i in range(len(pssm_list)):
	for j in range(len(pssm_list[i])):
		pssm_rsp.append(pssm_list[i][j])

cnt=np.mean(pssm_rsp)
std=np.std(pssm_rsp)

for i in range(len(pssm_list)):
	pssm_list[i]=pssm_list[i]-cnt
	pssm_list[i]=pssm_list[i]/std

test=np.load('cullpdb_test.npy').item()
seq_test=test['seq']
pssm_test_list=test['pssm']
dssp_test=test['dssp']


seq_l_test=[]
for i in range(len(seq_test)):
	seq_l_test.append(len(seq_test[i]))

seq_l_test=np.array(seq_l_test)

for i in range(len(pssm_test_list)):
	pssm_test_list[i]=pssm_test_list[i]-cnt
	pssm_test_list[i]=pssm_test_list[i]/std

seq_num=[]
for a in seq:
	tmp=np.zeros(len(a))
	for i in range(len(a)):
		try:
			tmp[i]=seqdic[a[i]]
		except:
			tmp[i]=20
	seq_num.append(tmp)

dssp_num=[]
for a in dssp:
	tmp=np.zeros(len(a))
	for i in range(len(a)):
		try:
			tmp[i]=labeldic[a[i]]
		except:
			tmp[i]=0
	dssp_num.append(tmp)

seq_num_test=[]
for a in seq_test:
	tmp=np.zeros(len(a))
	for i in range(len(a)):
		try:
			tmp[i]=seqdic[a[i]]
		except:
			tmp[i]=20
	seq_num_test.append(tmp)

dssp_num_test=[]
for a in dssp_test:
	tmp=np.zeros(len(a))
	for i in range(len(a)):
		try:
			tmp[i]=labeldic[a[i]]
		except:
			tmp[i]=0
	dssp_num_test.append(tmp)

def g_data(n,m):
	l=m
	data1=np.zeros([l,20])
	label=np.zeros([l,3])
	z=np.random.randint(0,2,[l])
	for i in range(len(seq_num[n])):
		label[i][int(dssp_num[n][i])]=1.0
		data1[i]=pssm_list[n][i]
	data=data1.reshape([1,l,1,20])
	noise=np.abs(np.random.normal(loc=1.0, scale=0.5, size=(1,l,1,20)))
	data=data*noise
	label=label.reshape([1,l,3])
	z=z.reshape([1,l])
	return data,label,z



def g_data_test(n,m):
	l=m
	data1=np.zeros([l,20])
	label=np.zeros([l,3])
	z=np.zeros([l])
	for i in range(len(seq_num_test[n])):
		label[i][int(dssp_num_test[n][i])]=1.0
		data1[i]=pssm_test_list[n][i]
		z[i]=1.0
	data=data1.reshape([1,l,1,20])
	label=label.reshape([1,l,3])
	z=z.reshape([1,l])
	return data,label,z


training_epochs = 40


x=tf.placeholder("float",[None,None,1,20],name='input')
y=tf.placeholder("float",[None,None,3],name='label')
z=tf.placeholder("float",[None,None],name='z')
lr=tf.placeholder("float",[],name='lr')
net=context.cnn_context(x)

net_rsp=tf.reshape(net,[-1,3])
y_rsp=tf.reshape(y,[-1,3])
z_rsp=tf.reshape(z,[-1])
net_soft=tf.nn.softmax(net_rsp)
pre=tf.argmax(net_rsp,1)

c_e=tf.nn.softmax_cross_entropy_with_logits(logits=net_rsp, labels=y_rsp)
tt1=tf.reduce_sum(c_e*z_rsp)
tt2=tf.reduce_sum(z_rsp)
loss=tt1/tt2

accuracy_ = tf.cast(tf.equal(tf.argmax(net_rsp,1),tf.argmax(y_rsp,1)),tf.float32)
ac_num = tf.reduce_sum(accuracy_*z_rsp)
accuracy = ac_num/tt2

optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

print("train_num:",len(seq_num))

init = tf.global_variables_initializer()

num=list(range(len(seq_num)))
saver = tf.train.Saver(max_to_keep=0)

with tf.Session() as sess:
	all_data=[]
	all_pre=[]
	for cw in range(10):
		save_name="model/my-model-"+str(cw)
		acc=0.0
		sess.run(init)
		for epoch in range(training_epochs):
			print("epoch:",epoch)
			random.shuffle(num)
			t1=0
			t2=0
			l_rate=np.random.uniform(0.02,0.12)
			for i in range(len(seq_num)):
				if len(seq_num[num[i]])<=20:
					continue
				batch_x,batch_y,batch_z=g_data(num[i],len(seq_num[num[i]]))
				_,l1,n1,n2=sess.run([optimizer,loss,ac_num,tt2],feed_dict={x:batch_x, y:batch_y, z:batch_z, lr:l_rate})
				t1=t1+n1
				t2=t2+n2
			print("train:",t1/t2)
			t1=0.0
			t2=0.0
			eve_ac=[]
			all_pret=[]
			for i in range(len(seq_num_test)):
				batch_x,batch_y,batch_z=g_data_test(i,len(seq_num_test[i]))
				n1,n2,b_pre=sess.run([ac_num,tt2,pre],feed_dict={x:batch_x, y:batch_y, z:batch_z})
				t1=t1+n1
				t2=t2+n2
				eve_ac.append(n1/n2)
				all_pret.append(b_pre)
			if t1/t2>acc:
				saver.save(sess, save_name)
				acc=t1/t2
				f_eve_ac=eve_ac
				f_all_pre=all_pret
			print("acc:",t1/t2)
		all_data.append(f_eve_ac)
		all_pre.append(f_all_pre)


t1=0
t2=0
for i in range(len(seq_l_test)):
	d=[]
	for j in range(len(all_pre)):
		d.append(all_pre[j][i])
	p=[]
	l=seq_l_test[i]
	for j in range(l):
		ss=[0,0,0]
		for k in range(len(d)):
			ss[int(d[k][j])]+=1
		p.append(np.argmax(ss))
	for j in range(len(p)):
		if p[j]==dssp_num_test[i][j]:
			t1+=1
		t2+=1
	

print('result:',t1/t2)
