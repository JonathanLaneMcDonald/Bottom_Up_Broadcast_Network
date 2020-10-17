
"""
Resample audio
"""

import librosa
import numpy as np

audio_files = [x for x in open('audiofiles.txt', 'r').read().split('\n') if len(x)]
print(len(audio_files), 'audio files found')

sample_rate = 8192
for af in audio_files:
	data, sr = librosa.load(af, sr=sample_rate)
	open(af, 'wb').write(np.array(data).tostring())
	print('wrote', af, 'at', sr)
