
import time
import numpy as np

from BottomUpBroadcastNetwork import STP_Classifier

start = time.time()
classifier = STP_Classifier('bbnn-0.41823064860422166-0.885.h5', 'genrelist.txt')

audiofiles = np.random.permutation([x for x in open('audiofiles.txt','r').read().split('\n') if len(x)])

for file in audiofiles[:100]:
    prediction = classifier.classify(file)
    print (file, list(reversed(sorted(prediction)))[0][1])
print ('test time (loading model, preprocessing, and classification):', time.time()-start, 'seconds')
