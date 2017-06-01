import tensorflow as tf
import numpy as np


with tf.variable_scope("model1"):
	x = tf.placeholder(tf.float32, [4, 1])
	y = tf.placeholder(tf.float32, [4, 2])
	w = tf.Variable(tf.zeros([1,2]))
	b = tf.Variable(tf.zeros([2]))
	linear = tf.matmul(x, w) + b
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=linear, labels=y))
	print 'x', x.shape
	print 'w', w.shape
	print 'b', b.shape
	print 'linear', linear.shape
	print 'loss', loss.shape
	#print loss.shape
	#loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(linear), reduction_indices=[1]))


t_vars = tf.trainable_variables()
model1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'model1')
model1_gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'model1')


print[var.name for var in t_vars]
print[var.name for var in model1_vars]
print[var.name for var in model1_gvars]


x_train = np.asarray([[1],[2],[3],[4]])
y_train = np.asarray([[0,1],[0,1],[1,0],[1,0]])

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
	
#opt1 = tf.train.GradientDescentOptimizer(0.01)
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss, var_list=t_vars)
grad_loss_vars = tf.gradients(loss, model1_vars)
grad_loss_linear = tf.gradients(loss, linear)
grad_loss_x = tf.gradients(loss, x)
grad_linear_vars = tf.gradients(linear, model1_vars)
grad_linear_w = tf.gradients(linear, w)
grad_loss_vars_via_linear = tf.gradients(linear, model1_vars, grad_loss_linear)

print grad_loss_vars
print grad_loss_linear
print grad_loss_x
print grad_linear_vars

for i in range(1):
	_, loss_, w_, b_, linear_, grad_loss_vars_, grad_loss_linear_, grad_loss_x_, grad_linear_vars_, grad_linear_w_, grad_loss_vars_via_linear_= \
		sess.run([train, loss, w, b, linear, grad_loss_vars, grad_loss_linear, grad_loss_x, grad_linear_vars, grad_linear_w, grad_loss_vars_via_linear], {x:x_train, y:y_train})
	#print loss_, w_, b_, linear_
	#print linear_
	print i
	print grad_loss_vars_, grad_loss_vars_[0].shape, grad_loss_vars_[1].shape #dl/dw, dl/db #(1, 2) (2,)
	print grad_loss_linear_, grad_loss_linear_[0].shape #dl/d(wx+b) #(4, 2)
	#print grad_loss_x_
	print grad_linear_vars_, grad_linear_vars_[0].shape, grad_linear_vars_[1].shape #d(wx+b)/dw, d(wx+b)/db #(1, 2) (2,)

	# print np.dot(np.transpose(grad_loss_linear_[0]), grad_linear_vars_[1])

	# print grad_linear_w_
	print grad_loss_vars_via_linear_

	#print np.sum(np.asarray(grad_loss_linear_))
	#print np.sum(np.asarray(grad_loss_linear_))*np.asarray(grad_linear_vars_)



