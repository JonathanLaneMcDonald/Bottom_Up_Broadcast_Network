
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import Convolution1D, MaxPooling1D, Concatenate
from keras.layers import BatchNormalization, Activation
from keras.optimizers import Adam

import keras
import librosa
import numpy as np
from numpy.random import random as npr

def build_audio_tagger(input_shape, num_classes):

	input = Input(shape=input_shape)
	x = input
	
	for filters in [128,128,256,256,384,384,512,512]:
		x = Convolution1D(filters, kernel_size=5, padding='same')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = MaxPooling1D(3)(x)

	x = Flatten()(x)
	x = Dropout(0.5)(x)

	output = Dense(num_classes, activation='softmax')(x)
	
	model = Model(inputs=input, outputs=output)
	model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0002), metrics=['accuracy'])
	return model

def index_of(item, array):
	for x in range(len(array)):
		if item == array[x]:
			return x
	return -1

def generate_dataset(samples, input_size, audio_files, genres, noise_stdev):
	features = np.zeros((samples, input_size, 1),dtype=np.float16)
	labels = np.zeros((samples,1),dtype=np.int)
	
	for s in range(samples):		
		index = int(npr()*len(audio_files))
	
		#data, sr = librosa.load(audio_files[index])
		buff = np.frombuffer(open(audio_files[index],'rb').read(), dtype=np.float32)
		data = buff.astype(np.float16)
		startpos = max(0,int(npr()*data.shape[0])-input_size)

		signal = data[startpos:startpos+input_size]
		noise = np.random.normal(0,noise_stdev,input_size)

		features[s] = np.array(np.add(signal,noise)).reshape(input_size,1)
		labels[s][0] = index_of(audio_files[index].split('/')[1], genres)

	return features, labels

audio_files = [x for x in open('audiofiles.txt','r').read().split('\n') if len(x)]
print (len(audio_files),'audio files found')

genres = sorted(set([x.split('/')[1] for x in audio_files]))
print (len(genres),'genres identified:',genres)

input_size = 32768
num_classes = len(genres)

model = build_audio_tagger((input_size,1), num_classes)
model.summary()

samples = 10000
for e in range(1,1000):
	features, labels = generate_dataset(samples, input_size, audio_files, genres, 0.02)
	model.fit(features, labels, batch_size=8, epochs=1, verbose=1, validation_split=0.10)

	features, labels = generate_dataset(samples, input_size, audio_files, genres, 0.0)
	predictions = model.predict(features)

	cm = np.zeros((len(genres), len(genres)))
	for p in range(len(predictions)):
		cm[labels[p][0]] = np.add(cm[labels[p][0]], predictions[p])
	for r in range(len(cm)):
		print ('\t'.join([str(int(x)) for x in cm[r]]))




