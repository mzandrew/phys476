#!/usr/bin/env python
# taken from mlp_keras.py written 2025 by keisuke yoshihara
# modified 2025-04-10 by mza
# last updated 2025-04-14 by mza

# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
# https://keras.io/api/losses/
# https://www.tensorflow.org/api_docs/python/tf/keras

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.losses import CategoricalCrossentropy
import matplotlib.pyplot as plt
import csv, sys
epsilon = -0.000001
random_seed = 123
offset = 0.1
num_classes = 3
batch_size = 10
num_epochs = 1000
index = [ 8, 11, 17, 20, 23 ] # test_loss: 0.242, test_acc: 0.923 for 1000 epochs and 300, 30, 3
#index = [ 8, 11 ] # test_loss: 0.056, test_acc: 0.985
#index = [ 8, 17 ] # test_loss: 0.011, test_acc: 1.000; 1000 epochs: test_loss: 0.001, test_acc: 1.000
#index = [ 8, 20 ] # test_loss: 0.013, test_acc: 1.000
#index = [ 8, 23 ] # test_loss: 0.016, test_acc: 1.000
#index = [ 11, 17 ] # test_loss: 0.092, test_acc: 0.985
#index = [ 11, 20 ] # test_loss: 0.097, test_acc: 0.985
#index = [ 11, 23 ] # test_loss: 0.075, test_acc: 0.970
#index = [ 17, 20 ] # test_loss: 0.109, test_acc: 0.970
#index = [ 20, 23 ] # test_loss: 0.105, test_acc: 0.970

if __name__ == '__main__':
	np.random.seed(random_seed)
	tf.random.set_seed(random_seed)
	''' 1. Dataset '''
	if 0:
		N = 300
		from sklearn import datasets
		x, t = datasets.make_moons(N, noise=0.3)
		#print(type(x)) # <class 'numpy.ndarray'>
		t = t.reshape(N, 1)
#		for i in range(N):
#			print(str(x[i]) + "; " + str(t[i])) # [-0.88346164  1.1119377 ]; [0]
	else:
		x = []; t = []
		N = 0
		with open('train.csv') as csvfile:
			dataset = csv.reader(csvfile)
			for srow in dataset:
				try:
					frow = [ float(string) for string in srow ]
					if epsilon<frow[-1]:
#						print(', '.join(srow))
						# 24 is the truth
						# 8, 11, 17, 20, 23 are great for discerning type 1 from type 2
						# 10 is good
						# 9, 12, 15, 16, 19 are okay
						# the rest are bad or marginal:
						#0:1, 0:4, 0:5 diagonal line
						#0:6, 0:7 horizontal line
						#0:2, 0:3 expanding horn scatter plot
						#0:13, 0:14, 0:18, 0:21, 0:22 is a rectangular scatter plot
						mylist = []
						for i in index:
							mylist.append(frow[i])
						x.append(mylist)
						one_hot = [ 0 for c in range(num_classes) ]
						one_hot[int(frow[-1])] = 1
						t.append(one_hot) # one_hot encoding for categorical_crossentropy
						N += 1
				except Exception as e:
					#print("EXCEPTION: " + str(e))
					pass
		print("successfully read " + str(N) + " entries")
		x = np.asarray(x)
		t = np.asarray(t)
		#for i in range(N):
		#	print(str(x[i]) + "; " + str(t[i])) # [1.6577802, 1.6518945]; [1]
	print(str(len(x)) + "; " + str(len(t)))
	x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.2)
	print(str(len(x_train)) + "; " + str(len(t_train)))
	''' 2. Model building '''
	model = Sequential()
	model.add(Dense(300, activation='sigmoid', kernel_initializer=RandomNormal(mean=0.0, stddev=1.0, seed=random_seed), bias_initializer='zeros'))
	model.add(Dense(30, activation='sigmoid', kernel_initializer=RandomNormal(mean=0.0, stddev=1.0, seed=random_seed), bias_initializer='zeros'))
	model.add(Dense(3, activation='softmax', kernel_initializer=RandomNormal(mean=0.0, stddev=1.0, seed=random_seed), bias_initializer='zeros'))
	''' 3. Model learning '''
	optimizer = optimizers.SGD(learning_rate=0.1)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
	model.fit(x_train, t_train, epochs=num_epochs, batch_size=batch_size, verbose=1)
	''' 4. Model evaluation '''
	loss, acc = model.evaluate(x_test, t_test, verbose=0)
	print('test_loss: {:.3f}, test_acc: {:.3f}'.format(loss, acc))
	''' 5. Decision boundary plotting '''
	#print(str(type(x_train[:, 0])))
	plt.figure(figsize=(8, 6))
	plt.scatter(x_train[:,0], x_train[:,1], c=[ 1*triplet[0]/14+4*triplet[1]/14+9*triplet[2]/14 for triplet in t_train ], cmap='viridis', edgecolor='k', label='Train data')
	if 0:
		# Create a grid for decision boundary
		xx, yy = np.meshgrid(
			np.linspace(x[:,0].min() - offset, x[:,0].max() + offset, 200),
			np.linspace(x[:,1].min() - offset, x[:,1].max() + offset, 200)
		)
		grid = np.c_[xx.ravel(), yy.ravel()]
		preds = model(grid).numpy().reshape(xx.shape)
		# Plot the training data points
		plt.contourf(xx, yy, preds, alpha=0.6, levels=np.linspace(0, 1, 3), cmap='viridis')
	plt.colorbar(label='Model output (probability)')
	plt.title("Decision Boundary with Training Data")
	plt.xlabel("x1")
	plt.ylabel("x2")
	plt.legend()
	plt.show()

