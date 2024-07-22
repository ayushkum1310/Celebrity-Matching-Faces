import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path

def feature_extractor(img_path, model):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img)
        result = model.predict(preprocessed_img).flatten()
        return result

def Feature_image(files_path:Path,output_dir:Path):
    filenames = pickle.load(open(files_path, 'rb'))

    # Initialize the model
    model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

    

    features = []
    for file in tqdm(filenames):
        features.append(feature_extractor(file, model))

    # Save features
    
    out=output_dir/'embedding.pkl'
    
    with open(out, 'wb') as f:
        pickle.dump(features, f)
        

if __name__=='__main__':
    input_path=Path('artifacts/feature_list.pkl')
    output_path=Path('artifacts')
    Feature_image(input_path,output_path)