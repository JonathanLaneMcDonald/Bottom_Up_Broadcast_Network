
import os
import keras
import librosa
import numpy as np

from sys import argv

from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Activation, Concatenate
from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from numpy.random import random as npr

# implementation of the model from 
# Bottom-up Broadcast Neural Network For Music Genre Classification
# https://arxiv.org/abs/1901.08928
# 
# if BN is off, my model weights match the paper's weights exactly (180,458 trainable params)
# however, I think it's good practice to use BN and they specify it in the paper

USING_BATCH_NORMALIZATION = True
FILTERS = 32

def add_bbnn_first_stage(x):
	x = Convolution2D(FILTERS, (3,3), padding='same')(x)
	if USING_BATCH_NORMALIZATION:
		x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = MaxPooling2D((4,1))(x)
	return x

def add_bn_conv(x, kernel):
	if USING_BATCH_NORMALIZATION:
		x = BatchNormalization()(x)
	x = Convolution2D(FILTERS, kernel, padding='same', activation='relu')(x)
	return x

def add_inception_block(x):
	incept_top_1_conv_1x1 = add_bn_conv(x, (1,1))
	incept_top_2_conv_1x1 = add_bn_conv(x, (1,1))
	incept_top_3_conv_1x1 = add_bn_conv(x, (1,1))
	incept_top_4_pool_3x3 = MaxPooling2D((3,3),strides=(1,1),padding='same')(x)
	
	incept_bot_1_conv_1x1 = incept_top_1_conv_1x1
	incept_bot_2_conv_3x3 = add_bn_conv(incept_top_2_conv_1x1, (3,3))
	incept_bot_3_conv_5x5 = add_bn_conv(incept_top_3_conv_1x1, (5,5))
	incept_bot_4_conv_1x1 = add_bn_conv(incept_top_4_pool_3x3, (1,1))

	concat = Concatenate()([	incept_bot_1_conv_1x1,
								incept_bot_2_conv_3x3,
								incept_bot_3_conv_5x5,
								incept_bot_4_conv_1x1	])

	return concat

def add_bbnn_transition_stage(x):
	x = add_bn_conv(x, (1,1))
	x = MaxPooling2D((2,2))(x)
	return x

def add_bbnn_closing_stage(x, num_classes):
	if USING_BATCH_NORMALIZATION:
		x = BatchNormalization()(x)
	x = GlobalAveragePooling2D()(x)
	x = Dense(num_classes, activation='softmax')(x)
	return x

def build_bbnn(input_shape, num_classes):

	input = Input(shape=input_shape)
	dense_connections = [add_bbnn_first_stage(input)]

	for broadcast_module in range(3):
		x = None
		if len(dense_connections) == 1:
			x = dense_connections[-1]
		else:
			x = Concatenate()(dense_connections)
		x = add_inception_block(x)
		dense_connections.append(x)

	x = Concatenate()(dense_connections)
	x = add_bbnn_transition_stage(x)
	x = add_bbnn_closing_stage(x, num_classes)
	
	output = x
	model = Model(inputs=input, outputs=output)
	return model

def generate_dataset(audio_files, genres):
	samples = list(np.random.permutation(list(range(len(audio_files)))))

	features = np.zeros((len(samples), 645, 128, 1),dtype=np.float16)
	labels = np.zeros((len(samples),1),dtype=np.int)

	for s in range(len(samples)):
		data, sr = librosa.load(audio_files[samples[s]])
		features[s] = np.log10(np.add(features[s],librosa.feature.melspectrogram(data,hop_length=1024).transpose()[:645].reshape(645,128,1))+1)
		labels[s][0] = genres.index(audio_files[samples[s]].split('/')[-2])

	return features, labels

def split_dataset(dataset, validation_split):
	classes = dict()
	for name in dataset:
		this_class = name.split('/')[-2]
		if this_class in classes:
			classes[this_class] += [name]
		else:
			classes[this_class] = [name]
	
	training_dataset = []
	validation_dataset = []
	
	for _, items in classes.items():
		permuted = list(np.random.permutation(items))
		training_dataset += permuted[int(validation_split*len(permuted)):]
		validation_dataset += permuted[:int(validation_split*len(permuted))]

	training_dataset = np.random.permutation(training_dataset)
	validation_dataset = np.random.permutation(validation_dataset)
	class_list = sorted(set([x[0] for x in classes.items()]))
	
	return class_list, training_dataset, validation_dataset

def build_and_train_bbnn_model_from_filelist(audio_files, genre_list_fileout):
	print (len(audio_files),'audio files found')

	genres, training_dataset, validation_dataset = split_dataset(audio_files, 0.20)
	print (len(genres),'genres identified:',genres)
	print (len(training_dataset),'items in training set')
	print (len(validation_dataset),'items in validation set')

	# write the genre list to the supplied file
	open(genre_list_fileout,'w').write('\n'.join(genres))

	model = build_bbnn((645,128,1),10)
	model.summary()

	lr=0.001
	print ('learning rate:',lr)
	model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])

	best_loss = 2**20
	training_features, training_labels = generate_dataset(training_dataset, genres)
	validation_features, validation_labels = generate_dataset(validation_dataset, genres)
	for e in range(1, 1000):
		if e % 100 == 0:
			lr/=3
			print ('learning rate changed to:',lr)
			model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])

		noise = np.random.uniform(0, 0.1, size=training_features.shape)
		noisy_features = np.add(training_features, noise)
		model.fit(noisy_features, training_labels, batch_size=4, epochs=1, verbose=1, validation_data=(validation_features, validation_labels))
		loss, accuracy = model.evaluate(validation_features, validation_labels, batch_size=4, verbose=1)

		if loss < best_loss:
			best_loss = loss
			model.save('bbnn-'+str(loss)+'-'+str(accuracy)+'.h5')

	features, labels = generate_dataset(training_dataset, genres)
	predictions = model.predict(features)

	cm = np.zeros((len(genres), len(genres)))
	for p in range(len(predictions)):
		cm[labels[p][0]] = np.add(cm[labels[p][0]], predictions[p])
	for r in range(len(cm)):
		print ('\t'.join([str(int(x)) for x in cm[r]]))

class STP_Classifier:
	def __init__(self, modelPath, genreListPath):
		assert os.path.exists(modelPath)
		self.model = load_model(modelPath)
		self.model.summary()

		assert os.path.exists(genreListPath)
		self.genre = [x for x in open(genreListPath,'r').read().split('\n') if len(x)]
		print ('loaded genres:',self.genre)

	def classify(self, audioPath):
		assert os.path.exists(audioPath)
		features = np.zeros((1, 645, 128, 1),dtype=np.float16)

		data, sr = librosa.load(audioPath)
		features[0] = np.log10(np.add(features[0],librosa.feature.melspectrogram(data,hop_length=1024).transpose()[:645].reshape(645,128,1))+1)
		prediction = self.model.predict(features)[0]

		return list(zip(prediction, self.genre))

command = argv[1]
if command == 'train':
	build_and_train_bbnn_model_from_filelist([x for x in open('audiofiles.txt','r').read().split('\n') if len(x)], 'genrelist.txt')
elif command == 'classify':
	classifier = STP_Classifier('best bbnn model', 'genrelist.txt')

	audiofiles = np.random.permutation([x for x in open('audiofiles.txt','r').read().split('\n') if len(x)])

	for file in audiofiles[:10]:
		prediction = classifier.classify(file)
		print (file, prediction)

else:
	print ('Usage: bbnn\ audiotagger.py [train/classify]')
