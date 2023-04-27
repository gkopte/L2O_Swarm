import tensorflow as tf
import numpy as np
import pdb
import problems
import PSO
import TF_PSO_Working_OwnFit

debug_mode = False




def distance_matrix(x1, n1, x2, n2, dim): 
	# calculating the distance matrix of x1, x2; x1 shape:(n1, dim)  x2 shape(n2, dim), return shape (n1, n2)
	y1 = tf.tile(x1, [1, n2, 1])  # y1 shape: (n1*n2, dim)
	y2 = tf.tile(x2, [1, 1, n1]) # y2 shape: (n2, n1*dim)


	y2 = tf.reshape(y2, (-1, n1*n2, dim))

	#dis = tf.square(tf.norm(tf.math.subtract(y1,y2), axis=-1, ord='euclidean'))
	dis = tf.reduce_sum(tf.square(tf.math.subtract(y1,y2)), axis=-1)

	dis = tf.reshape(dis, (-1, n1,n2))

	
	return dis



def self_loss (x, fx_array, n,im_loss_option):
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

	def imitation_error(x, fx_array, n, option=''):
		print('imitation_error option: ',option)
		print("x shape:", x.shape)

		num_particle = 7
		x_shape = tf.shape(x)

		with tf.Session() as sess:
			batch_size = x_shape[0].eval() 
			unroll_and_part = x_shape[1].eval()
			dim = x_shape[2].eval()
			
		unroll_length = unroll_and_part//num_particle
			
		
		im_loss  = tf.constant(0, dtype=tf.float32)
		# im_loss = tf.get_variable("im_loss",shape=[], dtype=tf.float32, initializer=tf.constant_initializer(0),trainable=False)
		for batch in range(batch_size):
			
			x_batch = tf.slice(x, [tf.stop_gradient(batch), 0, 0], [1, unroll_and_part, dim])
			x_batch = tf.squeeze(x_batch)
			x_batch  = tf.reshape(x_batch, [unroll_length, num_particle, dim])
			# gettinng initial x for this unroll
			init_x_batch = tf.slice(x_batch, [0, 0, 0], [1, num_particle, dim])
			init_x_batch = tf.squeeze(init_x_batch)

			# print(init_x_batch)

			# #creating tensor for pso x and coping initial x to it
			# x_pso = tf.Variable(tf.zeros(tf.shape(first_instance)), dtype=tf.float32)
			# x_pso.assign(first_instance)
			
			# building pso graph 
			pso_ = TF_PSO_Working_OwnFit.pso(fitness_fn=TF_PSO_Working_OwnFit.fitness_function(),pop_size=num_particle, dim=dim, n_iter=unroll_length,x_init=init_x_batch)
			pso_.train()

			# getting pso x history to calculate loss
			pso_x_history = tf.reshape(pso_.x_history[1::], (unroll_length, num_particle, dim))
			pso_x_history_detached = tf.stop_gradient(pso_x_history)
			# print(pso_x_history)

			def custom_loss(y_true, y_pred):
				z = tf.abs(y_true - y_pred)
				quadratic = tf.maximum(1.0, z)**2
				absolute = tf.minimum(1.0, z)
				return tf.reduce_mean(tf.where(z >= 1.0, quadratic, absolute))
			
			if option=='custom':
				im_loss += custom_loss(pso_x_history_detached, x_batch)
			elif option=='rmse':
				im_loss += tf.sqrt(tf.reduce_mean(tf.square(pso_x_history_detached - x_batch)))
			elif option=='huber':
				huber_loss = tf.keras.losses.Huber(delta=1.0)
				im_loss += huber_loss(pso_x_history_detached, x_batch)
			elif option=='mse': 
				im_loss += tf.reduce_mean(tf.reduce_mean((pso_x_history_detached - x_batch)**2,0))
			else: #sumed square
				im_loss += tf.reduce_sum(tf.reduce_sum((pso_x_history_detached - x_batch)**2,0))

		return im_loss/batch_size
	
	# im_loss_option = 'mse'
	
	k = 1.0 # imitation scaling factor
	if im_loss_option=='mse':
		im_loss = imitation_error(x, fx_array, n,'mse')
		print("im_loss shape:", im_loss.shape)
		print("sumfx shape:", sumfx.shape)
		return sumfx+im_loss*k
	elif im_loss_option=='square':
		im_loss = imitation_error(x, fx_array, n,'square')
		print("im_loss shape:", im_loss.shape)
		print("sumfx shape:", sumfx.shape)
		return sumfx+im_loss*k
	elif im_loss_option=='rmse':
		im_loss = imitation_error(x, fx_array, n,'rmse')
		print("im_loss shape:", im_loss.shape)
		print("sumfx shape:", sumfx.shape)
		return sumfx+im_loss*k
	elif im_loss_option=='huber':
		im_loss = imitation_error(x, fx_array, n,'huber')
		print("im_loss shape:", im_loss.shape)
		print("sumfx shape:", sumfx.shape)
		return sumfx+im_loss*k
	elif im_loss_option=='custom':
		im_loss = imitation_error(x, fx_array, n,'custom')
		print("im_loss shape:", im_loss.shape)
		print("sumfx shape:", sumfx.shape)
		return sumfx+im_loss*k
	elif im_loss_option=='only_im':
		im_loss = imitation_error(x, fx_array, n,'mse')
		print("im_loss shape:", im_loss.shape)
		print("sumfx shape:", sumfx.shape)
		return im_loss*k
	elif im_loss_option=='only_sumfx':
		print("sumfx shape:", sumfx.shape)
		return sumfx
	elif im_loss_option=='entropy_im':
		im_loss = imitation_error(x, fx_array, n,'square')
		print("im_loss shape:", im_loss.shape)
		print("sumfx shape:", sumfx.shape)
		return sumfx+lam*h+im_loss*k
	elif im_loss_option is None or im_loss_option.lower()=='none': #Sanity check
		print("Warning: No self loss")
		return 0
	else:
		print("sumfx shape:", sumfx.shape)
		return sumfx+lam*h

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
