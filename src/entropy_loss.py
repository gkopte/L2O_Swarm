import tensorflow as tf
import numpy as np
import pdb
import problems
import PSO
import TF_PSO_Working_OwnFit

debug_mode = True



def distance_matrix(x1, n1, x2, n2, dim): 
	# calculating the distance matrix of x1, x2; x1 shape:(n1, dim)  x2 shape(n2, dim), return shape (n1, n2)
	y1 = tf.tile(x1, [1, n2, 1])  # y1 shape: (n1*n2, dim)
	y2 = tf.tile(x2, [1, 1, n1]) # y2 shape: (n2, n1*dim)


	y2 = tf.reshape(y2, (-1, n1*n2, dim))

	#dis = tf.square(tf.norm(tf.math.subtract(y1,y2), axis=-1, ord='euclidean'))
	dis = tf.reduce_sum(tf.square(tf.math.subtract(y1,y2)), axis=-1)

	dis = tf.reshape(dis, (-1, n1,n2))

	
	return dis



def self_loss (x, fx_array, n):
	# lambda values for controling the  balance between exploitation and exploration.
	lam=0.001
	print(x)
	#pdb.set_trace()
	print (fx_array.shape)

	x, x1 = tf.split(x, [n, 0], 0)
	fx_array,f1 = tf.split(fx_array, [n,0], 0)
	problem_dim = x.shape.as_list()[-1]
	batch_size = x.shape.as_list()[1]
	print("Batch size ",batch_size)
	x = tf.transpose(x, [1,0,2])
	fx_array = tf.transpose(fx_array, [1,0])

	print (fx_array.shape, x.shape)
	
	
	def entropy(x, fx_array,  preh):



		dis = (distance_matrix(x, n, x, n, problem_dim))

		#dis =  2* (tf.square(x)-tf.matmul(x, x, transpose_b=True))

		
		l=1
		dis1 =tf.math.divide(dis, 2*l)
		KK = tf.math.exp(tf.math.negative(dis1))


		#KK = tf.reshape(KK1, (n, n))
		epsilon =2.1


		KKK = tf.linalg.inv(tf.math.add(KK, tf.eye(n)*epsilon))  #(n,n)


		tt = []
		for i in range(1):
			numofsamples = 1000
			samples = tf.random.normal((batch_size, numofsamples, problem_dim), mean=0.0, stddev=0.01)
			#samples = tf.random.normal((batch_size, numofsamples, problem_dim), mean=0.0, stddev=1)
			dis2 = distance_matrix(samples, numofsamples, x, n, problem_dim)

		

			l=1
			dis3 =tf.math.divide(dis2, 2*l)
			Kx = (tf.math.exp(tf.math.negative(dis3)))  #  (n,100000)

		
			final1 = tf.matmul(tf.matmul(Kx, KKK, transpose_a=False), tf.expand_dims(fx_array,axis=-1))
			final1 = tf.reshape(final1, (batch_size, numofsamples, ))
			#tt.append(final1)

		#final1 = tf.reshape(tf.transpose(tf.convert_to_tensor(tt), [1, 0 ,2]), (batch_size, -1))
		print (final1.shape)
		
		

		h=preh[0]
		rho_0=1
		import numpy as np

		if(preh[1]==0):
			rho = rho_0*np.exp(1./h*(n**(1/2.)))
		else:
			rho = rho_0*tf.math.exp(1./h*(n**(1/2.)))
		
		
		# for numerical concern. minus the largest value to make all values non-positve before doing exponential
		rhofx = -rho*final1
		rhofx = rhofx - tf.expand_dims(tf.reduce_max(rhofx, -1), axis=-1)


		c = tf.math.exp(rhofx)

		# for numerical concern
		px = tf.divide(c+0.0001, tf.expand_dims(tf.reduce_sum(c+0.0001, axis=-1), axis=-1))
		


		ent = tf.reduce_mean(tf.reduce_sum(tf.math.negative(tf.multiply(px, tf.math.log(px))), axis=-1))

		print (c.shape, px.shape, final1.shape, ent.shape)
	

		return ent


	sumfx = tf.reduce_mean(tf.reduce_sum(fx_array, -1))
	preh = np.log(5.**problem_dim)
	h0 = entropy(x, fx_array,  [preh,0])
	h  = entropy(x, fx_array,  [h0,1])

	# def fitness_fn(IL_sess, problem, x_val):
	# 	x_val = np.array(x_val,dtype=np.float32).reshape(1, -1)
	# 	IL_sess.run(x.assign(x_val))
	# 	#print(sess.run(x))
	# 	result = IL_sess.run(problem)
	# 	return result.reshape(-1, 1).astype(np.float32)

	def imitation_error(x, fx_array, n):
		# with tf.Session() as IL_sess:			
		# with tf.variable_scope("problem", reuse=tf.AUTO_REUSE):
		# x_val = tf.get_variable("x",shape=[n,problem_dim],dtype=np.float32,initializer=tf.random_uniform_initializer(-3, 3))
		# w_val = tf.get_variable("w",dtype=np.float32,initializer=problems.indentity_init(1, 2, 0.01/2),trainable=False)
		# y_val = tf.get_variable("y",shape=[n,problem_dim],dtype=np.float32,initializer=tf.random_normal_initializer(stddev=0.01/2),trainable=False)
		# w_val = tf.get_variable("wcos",shape=[n,problem_dim],dtype=np.float32,initializer=tf.random_normal_initializer(mean=1.0, stddev=0.01/2),trainable=False)
	
		
		# problem = problems.square_cos(batch_size=1, num_dims=problem_dim, mode='test')()

		# init_pos = [[i-1,i+1] for i in range(x.shape[1])]
		print("x shape:", x.shape)
		#print("x_val shape:", x_val.shape)

		first_instance = tf.slice(x, [0, 0, 0], [1, n,problem_dim])
		first_instance = tf.squeeze(first_instance)
		print(first_instance)
		x_pso = tf.Variable(tf.zeros(tf.shape(first_instance)), dtype=tf.float32)
		x_pso.assign(first_instance)
		

		# last_instance = tf.slice(x, [batch_size-1, 0, 0], [1, n,problem_dim])
		# last_instance = tf.squeeze(last_instance)
		# IL_sess.run(tf.global_variables_initializer())
		# last_instance_np = IL_sess.run(last_instance)

		# init_pos = IL_sess.run(first_instance).tolist()
		# print("init_pos", init_pos)
		# print("n :",n)
		# init_vel = np.zeros(shape=(n,problem_dim)).tolist()
		# # init_vel = [np.zeros(x.shape[1:]).tolist() for i in range(n)]
		# print("init_vel", init_vel)
		# best_position, swarm = PSO.pso(IL_sess, PSO.fitness_fn,problem, x_val,batch_size, n, problem_dim, -3, 3,init_pos,init_vel)
		pso_ = TF_PSO_Working_OwnFit.pso(fitness_fn=TF_PSO_Working_OwnFit.fitness_function(),pop_size=n, dim=problem_dim, n_iter=batch_size,x_init=x_pso)
		# IL_sess.run(tf.global_variables_initializer())
		pso_.train()

		x_history = tf.reshape(pso_.x_history[1::], (batch_size, n, problem_dim))
		# final_pos_pso = IL_sess.run(pso_.x)
		# print("final pos PSO: ",final_pos_pso)
		# final_pos_pso_array = np.array(final_pos_pso)
		# # tf.reduce_mean(tf.reduce_sum(fx_array, -1))

		# output = tf.reduce_mean(tf.reduce_sum(tf.constant((final_pos_pso_array - last_instance_np)**2),-1))
		# output = tf.reduce_mean(tf.reduce_sum((x_history - x)**2,0))
		if debug_mode:
			pdb.set_trace()
		output = tf.reduce_sum(tf.reduce_sum((x_history - x)**2,0))

		# num = IL_sess.run(output)
		return output
			

			#sess, fitness_fn,problem, max_iter, num_particles, num_dims, -3, 3,init_pos,init_vel

	im_error = imitation_error(x, fx_array, n)
	print("im_error shape:", im_error.shape)
	print("sumfx shape:", sumfx.shape)
	# return sumfx+lam*h+im_error
	return im_error

if __name__ == "__main__":
	# with tf.variable_scope("square_cos", reuse=tf.AUTO_REUSE):
	# 	x = tf.get_variable("x", [300,128, 2], dtype=tf.float32, initializer=tf.random_normal_initializer)
	# 	x.assign(np.zeros((300,128, 2)))
		
	# 	fx_array = tf.get_variable("y", [300, 128], dtype=tf.float32, initializer=tf.random_normal_initializer)
	# 	fx_array.assign(np.zeros((300, 128)))
	# 	loss = self_loss(x, fx_array, 300)
	# 	# loss, t1, t2 = self_loss(x, fx_array, 300)
		
		
	with tf.Session() as sess:
		with tf.variable_scope("self_loss", reuse=tf.AUTO_REUSE):
			x = tf.get_variable("x", [10,128, 2], dtype=tf.float32, initializer=tf.random_normal_initializer)
			fx_array = tf.get_variable("y", [10, 128], dtype=tf.float32, initializer=tf.random_normal_initializer)		
			# loss, t1, t2 = self_loss(x, fx_array, 300)

			sess.run(tf.global_variables_initializer())
			loss = self_loss(x, fx_array, 10)
		
			print(loss)
			print(loss.shape)
			
		
			# for i in range(20):
			# 	# sess.run(tf.global_variables_initializer())
			# 	if i == 10:
			# 		sess.run(x.assign(tf.zeros((10,128, 2))))
			# 		#print(sess.run(x))
					
			# 		sess.run(fx_array.assign(tf.zeros((10, 128))))		
			# 	print(sess.run(loss))

				# print()
				#tt1 = sess.run(t1)
				#tt2 = sess.run(t2)
				#print (np.max(tt1), np.min(tt2))
				#print (sess.run(t2))
