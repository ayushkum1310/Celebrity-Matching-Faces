import os
import pickle
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from mtcnn import MTCNN

# Initialize face detector and model
detector = MTCNN()
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Load precomputed feature list and filenames
feature_list = np.array(pickle.load(open(Path('D:/Celebrity-Matching-Faces/artifacts/embedding.pkl'), 'rb')))
filenames = pickle.load(open(Path('D:/Celebrity-Matching-Faces/artifacts/feature_list.pkl'), 'rb'))

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving image: {e}")
        return False

def extract_features(img_path, model, detector):
    try:
        img = cv2.imread(img_path)
        results = detector.detect_faces(img)
        if not results:
            raise ValueError("No face detected in the image.")
        x, y, width, height = results[0]['box']
        face = img[y:y + height, x:x + width]
        image = Image.fromarray(face)
        image = image.resize((224, 224))
        face_array = np.asarray(image).astype('float32')
        expanded_img = np.expand_dims(face_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img)
        result = model.predict(preprocessed_img).flatten()
        return result
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def recommend(feature_list, features):
    try:
        similarity = []
        for i in range(len(feature_list)):
            similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])
        index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
        return index_pos
    except Exception as e:
        st.error(f"Error recommending celebrity: {e}")
        return None

# Streamlit UI
st.title('Which Bollywood Celebrity Are You?')

uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
    if save_uploaded_image(uploaded_image):
        img_path = os.path.join('uploads', uploaded_image.name)
        display_image = Image.open(img_path)
        
        features = extract_features(img_path, model, detector)
        
        if features is not None:
            index_pos = recommend(feature_list, features)
            
            if index_pos is not None:
                predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
                
                # Display images
                col1, col2 = st.columns(2)
                with col1:
                    st.header('Your Uploaded Image')
                    st.image(display_image)
                with col2:
                    st.header("Seems like " + predicted_actor)
                    st.image(filenames[index_pos], width=300)
            else:
                st.error("Error in recommendation.")
        else:
            st.error("Error in feature extraction.")
    else:
        st.error("Error saving uploaded image.")
