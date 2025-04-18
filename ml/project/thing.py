#!/usr/bin/env python
# taken from mlp_keras.py written 2025 by keisuke yoshihara
# modified 2025-04-10 by mza
# last updated 2025-04-17 by mza

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

num_epochs = 1000
#mode = "halfmoons"
mode = "realdata"
num_classes = 3
skim = 1

epsilon = 0.001
#epsilon_low = epsilon # this makes the accuracy 5% lower
#epsilon_high = 1.0 - epsilon # this makes the accuracy 15% lower (in combination with epsilon_low = 0.001)
epsilon_low = 0.0
epsilon_high = 1.0
random_seed = 123
offset = 0.1
batch_size = 10
meshsteps = 200
encoding = "onehot"
cmap = [ 'Blues', 'Reds', 'Greens' ]
num_color_gradations = 4

if mode=="halfmoons":
	index = [ 0, 1 ]
if mode=="realdata":
	#index = range(24) # adam100epochs acc=
	#index = [ 8, 11 ] # test_loss: 0.056, test_acc: 0.985; adam100epochs acc=0.836
	#index = [ 8, 17 ] # test_loss: 0.011, test_acc: 1.000; 1000 epochs: test_loss: 0.001, test_acc: 1.000
	#index = [ 8, 11, 17 ] # adam100epochs acc=0.847
	#index = [ 8, 11, 17, 20 ] # adam100epochs acc=0.842
	#index = [ 8, 11, 17, 20, 23 ] # adam100epochs acc=0.929
	index = [ 8, 23, 17, 11, 16, 20 ] # SGD100epochs acc=0.931; adam100epochs acc=0.926; adam/mse/24-12-3/1000epochs acc=0.929
	#index = [ 8, 11, 17, 23 ] # adam100epochs acc=0.929
	#index = [ 11, 17, 23 ] # adam100epochs acc=0.897
	#index = [ 8, 17, 23 ] # adam100epochs acc=0.926
	#index = [ 17, 23 ] # adam100epochs acc=0.873
	#index = [ 8, 23 ] # adam100epochs acc=0.918; adagrad100epochs acc=0.871; rmsprop100epochs acc=0.920; sgd100epochs acc=0.912; rmsprop/categorical_crossentropy/24-12-3/1000epochs acc=0.915
	#index = [ 8, 20 ] # test_loss: 0.013, test_acc: 1.000
	#index = [ 8, 23 ] # test_loss: 0.016, test_acc: 1.000
	#index = [ 11, 17 ] # test_loss: 0.092, test_acc: 0.985
	#index = [ 11, 20 ] # test_loss: 0.097, test_acc: 0.985
	#index = [ 11, 23 ] # test_loss: 0.075, test_acc: 0.970
	#index = [ 17, 20 ] # test_loss: 0.109, test_acc: 0.970
	#index = [ 20, 23 ] # test_loss: 0.105, test_acc: 0.970
print(str(index))

if __name__ == '__main__':
	np.random.seed(random_seed)
	tf.random.set_seed(random_seed)
	''' 1. Dataset '''
	if mode=="halfmoons":
		N = 300
		from sklearn import datasets
		x, ti = datasets.make_moons(N, noise=0.3)
		if encoding=="onehot":
			t = []
			for i in range(len(ti)):
				one_hot = [ 0 for c in range(num_classes) ]
				one_hot[ti[i]] = 1
				t.append(one_hot) # one_hot encoding for categorical_crossentropy
			t = np.asarray(t)
		else:
			t = ti
#			for i in range(N):
#				print(str(x[i]) + "; " + str(t[i])) # [-0.88346164  1.1119377 ]; [0]
	if mode=="realdata":
		x = []; t = []
		N = 0
		M = 0
		lines = 0
		with open('train.csv') as csvfile:
			dataset = csv.reader(csvfile)
			for srow in dataset:
				lines += 1
				try:
					frow = []
					#frow = [ float(string) for string in srow ]
					for string in srow:
						try:
							frow.append(float(string))
						except:
							frow.append(0.0)
					truth = int(frow[-1])
#					print(', '.join(srow))
					# 24 is the truth
					# 8, 11, 17, 20, 23 are great for discerning type 1 from type 2
					# 10 is good
					# 9, 12, 15, 16, 19 are okay
					# the rest are bad or marginal:
					#0:1, 0:4, 0:5 diagonal line
					#0:6, 0:7 horizontal line
					#0:2, 0:3 expanding horn scatter plot
					#0:13, 0:14, 0:18, 0:21, 0:22 is a rectangular scatter plot
					keeper = False
					if N%skim==0:
						keeper = True
					mylist = []
					for i in index:
						mylist.append(frow[i])
						if frow[i]<epsilon_low:
							keeper = False
						if epsilon_high<frow[i]:
							keeper = False
					one_hot = [ 0 for c in range(num_classes) ]
					one_hot[truth] = 1
					if keeper:
						x.append(mylist)
						t.append(one_hot) # one_hot encoding for categorical_crossentropy
						M += 1
					N += 1
				except Exception as e:
					print("EXCEPTION: " + str(e))
					#pass
		print("file contains " + str(lines) + " lines")
		print("successfully read " + str(N) + " entries")
		print("using " + str(M) + " entries")
		x = np.asarray(x)
		t = np.asarray(t)
		#for i in range(N):
		#	print(str(x[i]) + "; " + str(t[i])) # [1.6577802, 1.6518945]; [1]
	print(str(len(x)) + "; " + str(len(t)))
	x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.2)
	print(str(len(x_train)) + "; " + str(len(t_train)))
	''' 2. Model building '''
	model = Sequential()
	if mode=="halfmoons":
		model.add(Dense(num_classes, activation='softmax', kernel_initializer=RandomNormal(mean=0.0, stddev=1.0, seed=random_seed), bias_initializer='zeros'))
	if mode=="realdata":
		#model.add(Dense(300, activation='softmax', kernel_initializer=RandomNormal(mean=0.0, stddev=1.0, seed=random_seed), bias_initializer='zeros'))
		model.add(Dense(24, activation='softmax', kernel_initializer=RandomNormal(mean=0.0, stddev=1.0, seed=random_seed), bias_initializer='zeros'))
		model.add(Dense(12, activation='softmax', kernel_initializer=RandomNormal(mean=0.0, stddev=1.0, seed=random_seed), bias_initializer='zeros'))
		model.add(Dense(num_classes, activation='softmax', kernel_initializer=RandomNormal(mean=0.0, stddev=1.0, seed=random_seed), bias_initializer='zeros'))
	architecture_string = "24softmax-12softmax-3softmax"
	''' 3. Model learning '''
	optimizer = optimizers.SGD(learning_rate=0.1); optimizer_string="sgd"
	#optimizer = optimizers.Adagrad(learning_rate=0.01); optimizer_string="adagrad"
	#optimizer = optimizers.RMSprop(learning_rate=0.001, rho=0.9); optimizer_string="rmsprop"
	#optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999); optimizer_string="adam"
	#model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy']); loss_string="sparse_categorical_crossentropy" # ValueError: Argument `output` must have rank (ndim) `target.ndim - 1`. Received: target.shape=(None, 3), output.shape=(None, 3)
	#model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy']); loss_string="categorical_crossentropy" # acc=0.922
	model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy']); loss_string="mse" # meansquarederror # acc=0.920
	#model.compile(optimizer=optimizer, loss='msle', metrics=['accuracy']); loss_string="msle" # meansquaredlogarithmicerror acc=0.918
	model.fit(x_train, t_train, epochs=num_epochs, batch_size=batch_size, verbose=1)
	''' 4. Model evaluation '''
	loss, acc = model.evaluate(x_test, t_test, verbose=0)
	print('test_acc: {:.3f}, test_loss: {:.3f}'.format(acc, loss))
	#from keras_visualizer import visualizer # pip install keras-visualizer
	#visualizer(model, file_format='png')
	''' 5. Decision boundary plotting '''
	# https://stackoverflow.com/q/51219154
	xmin = []; xmax = []; x_span = []
	for i in range(len(index)):
		xmin.append(x[:,i].min() - offset)
		xmax.append(x[:,i].max() + offset)
		x_span.append(np.linspace(xmin[i], xmax[i], meshsteps))
	print("")
	epochs_string = str(num_epochs) + "epochs"
	png_basename = optimizer_string + "." + loss_string + "." + architecture_string + "." + epochs_string
	for i in range(len(index)):
		for j in range(len(index)):
			if j<=i:
				continue
			fig = plt.figure(figsize=(8, 6))
			if 1:
				xx, yy = np.meshgrid(x_span[i], x_span[j])
				ravels = [ xx.ravel(), yy.ravel() ]
				for q in range(len(index)-2):
					zz = np.copy(yy)
					ravels.append(zz.ravel())
				grid = np.vstack(ravels).T
				#print(str(grid.shape)) # (40000, 5)
				#print(str(model(grid).shape)) # (40000, 3)
				preds = []
				for c in range(num_classes):
					preds.append(model(grid)[:,c].numpy().reshape(xx.shape))
				#print(str(preds.shape)) # (40000,)
				#print(str(preds.shape)) # (200, 200)
				for c in range(num_classes):
					if c==0:
						continue
					plt.contourf(xx, yy, preds[c], alpha=0.3, cmap=cmap[c]) # , levels=np.linspace(0, 1, num_color_gradations)
				# https://scikit-learn.org/stable/modules/generated/sklearn.inspection.DecisionBoundaryDisplay.html
				#from sklearn.inspection import DecisionBoundaryDisplay
				#from sklearn.tree import DecisionTreeClassifier
				#grid = np.vstack([xx.ravel(), yy.ravel()]).T
				#tree = DecisionTreeClassifier().fit(x[:,[i,j]], [ int(1*triplet[1]+2*triplet[2]) for triplet in t ])
				#z = np.reshape(tree.predict(grid), xx.shape)
				#plt.contourf(xx, yy, z)
			plt.scatter(x_train[:,i], x_train[:,j], c=[ (1*triplet[0]+2*triplet[1]+4*triplet[2])/8. for triplet in t_train ], cmap='brg', edgecolor='k', label='Train data')
			plt.colorbar(label='Model output (probability)')
			plt.title("Decision Boundary with Training Data")
			plt.xlabel(str(index[i]))
			plt.ylabel(str(index[j]))
			plt.legend()
			#plt.show()
			png_filename = png_basename + "." + str(index[i]) + "-" + str(index[j]) + ".png"
			fig.savefig(png_filename)
			print("wrote file " + png_filename)
			plt.close()

