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

		# gettinng initial x for this unroll
		first_instance = tf.slice(x, [0, 0, 0], [1, n,problem_dim])
		first_instance = tf.squeeze(first_instance)
		print(first_instance)

		#creating tensor for pso x and coping initial x to it
		x_pso = tf.Variable(tf.zeros(tf.shape(first_instance)), dtype=tf.float32)
		x_pso.assign(first_instance)

		x_last = tf.Variable(tf.zeros(tf.shape(x)), dtype=tf.float32)
		x_pso_hist_last = tf.Variable(tf.zeros(tf.shape(x)), dtype=tf.float32)
		x_vel = tf.Variable(tf.zeros(tf.shape(x)), dtype=tf.float32)
		# vel_pso_x_history = tf.Variable(tf.zeros(tf.shape(x)), dtype=tf.float32)
		
		# building pso graph 
		print('n: ', n)
		pso_ = TF_PSO_Working_OwnFit.pso(fitness_fn=TF_PSO_Working_OwnFit.fitness_function(),pop_size=n, dim=problem_dim, n_iter=batch_size,x_init=x_pso)
		pso_.train()

		# getting pso x history to calculate loss
		pso_x_history = tf.reshape(pso_.x_history[1::], (batch_size, n, problem_dim))

		# velocity calculation
		vel_pso_x_history = tf.stop_gradient(pso_x_history) - tf.stop_gradient(x_pso_hist_last)
		x_vel = tf.stop_gradient(x) - x_last

		def custom_loss(y_true, y_pred):
			z = tf.abs(y_true - y_pred)
			quadratic = tf.maximum(1.0, z)**2
			absolute = tf.minimum(1.0, z)
			return tf.reduce_mean(tf.where(z >= 1.0, quadratic, absolute))
		
		if option=='custom':
			im_loss = custom_loss(vel_pso_x_history, x_vel)
		elif option=='rmse':
			im_loss = tf.sqrt(tf.reduce_mean(tf.square(vel_pso_x_history - x_vel)))
		elif option=='huber':
			huber_loss = tf.keras.losses.Huber(delta=1.0)
			im_loss = huber_loss(vel_pso_x_history, x_vel)
		elif option=='mse': 
			im_loss = tf.reduce_mean(tf.reduce_mean((vel_pso_x_history - x_vel)**2,0))
		else: #sumed square
			im_loss = tf.reduce_sum(tf.reduce_sum((vel_pso_x_history - x_vel)**2,0))
		
		x_pso_hist_last.assign(pso_x_history)
		x_last.assign(x)
		return im_loss
	
	# im_loss_option = 'mse'
	def create_counter():
		count = 0

		def counter():
			nonlocal count  # Declare count as nonlocal to modify its value inside the nested function
			count += 1
			return count

		return counter

	def explore_exploit_factor(step, max_steps, shape_factor=10, min_factor=0, max_factor=1):
		"""
		Generates a factor that balances exploration and exploitation for an optimization algorithm.
		
		:param step: Current step in the optimization process.
		:param max_steps: Maximum number of steps in the optimization process.
		:param shape_factor: Optional parameter to balance the shape of the logarithmic function.
							Higher values make the function closer to an "L" shape.
		:param min_factor: Minimum value for the exploration-exploitation factor.
		:param max_factor: Maximum value for the exploration-exploitation factor.
		:return: A factor between [min_factor, max_factor] that decays logarithmically.
		"""
		# Normalize the step value within the range [0, 1]
		normalized_step = step / max_steps

		# Calculate the exploration-exploitation factor using logarithmic decay
		factor = 1 - np.log(1 + shape_factor * normalized_step) / np.log(1 + shape_factor)

		# Scale the factor to be in the range [min_factor, max_factor]
		scaled_factor = min_factor + (max_factor - min_factor) * factor

		return scaled_factor
	
	# uniform_distribution = tf.random.uniform((1,), minval=0, maxval=10)
	# Apply the logarithm function to transform the uniform distribution to a log uniform distribution
	# k = tf.math.log(uniform_distribution)
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
		# counter = create_counter()
		# step = counter()
		# ca = explore_exploit_factor(step, 128, shape_factor=1000, min_factor=0, max_factor=1)
		# return sumfx*(1-ca)+im_loss*ca
		return sumfx+im_loss*k
	elif im_loss_option=='rmse':
		im_loss = imitation_error(x, fx_array, n,'rmse')
		print("im_loss shape:", im_loss.shape)
		print("sumfx shape:", sumfx.shape)
		# c = 0.2
		# return sumfx*(1-c)+im_loss*c
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
