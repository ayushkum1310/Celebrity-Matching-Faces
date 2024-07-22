from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from PIL import Image
from pathlib import Path
feature_list = np.array(pickle.load(open(Path('D:/Celebrity-Matching-Faces/artifacts/embedding.pkl'),'rb')))
filenames = pickle.load(open(Path('D:\Celebrity-Matching-Faces/artifacts/feature_list.pkl'),'rb'))

model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

detector = MTCNN()
# load img -> face detection
sample_img = cv2.imread(Path('D:/Celebrity-Matching-Faces/ab.jpg'))
results = detector.detect_faces(sample_img)

x,y,width,height = results[0]['box']

face = sample_img[y:y+height,x:x+width]

#  extract its features
image = Image.fromarray(face)
image = image.resize((224,224))

face_array = np.asarray(image)

face_array = face_array.astype('float32')

expanded_img = np.expand_dims(face_array,axis=0)
preprocessed_img = preprocess_input(expanded_img)
result = model.predict(preprocessed_img).flatten()
#print(result)
#print(result.shape)
# find the cosine distance of current image with all the 8655 features
similarity = []
for i in range(len(feature_list)):
    similarity.append(cosine_similarity(result.reshape(1,-1),feature_list[i].reshape(1,-1))[0][0])

index_pos = sorted(list(enumerate(similarity)),reverse=True,key=lambda x:x[1])[0][0]

temp_img = cv2.imread(filenames[index_pos])
cv2.imshow('output',temp_img)
cv2.waitKey(0)
# recommend that image