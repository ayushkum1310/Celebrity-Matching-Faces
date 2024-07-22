import os
import sys
import pickle
from src.logger import logging
from src.exception import CustomException
from pathlib import Path

def feature_list_maker(input_path: Path, output_path: Path):
    try:
        main_folders = os.listdir(input_path)
        sub_dir = [os.path.join(input_path, i) for i in main_folders]

        filenames = []
        for i in sub_dir:
            actors = os.listdir(i)
            for actor in actors:
                actor_path = os.path.join(i, actor)
                for file in os.listdir(actor_path):
                    filenames.append(os.path.join(actor_path, file))
        logging.info(f"Found {len(filenames)} images")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as file:
            pickle.dump(filenames, file)

        logging.info(f"Feature list created and saved to {output_path}")
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise CustomException(e, sys)

if __name__ == '__main__':
    input_path = Path('data/processed/Bollywood_celeb_face_localized')  # Use forward slashes or raw strings
    output_path = Path('D:/Celebrity-Matching-Faces/artifacts/feature_list.pkl')
    feature_list_maker(input_path, output_path)
