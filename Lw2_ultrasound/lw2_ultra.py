
import os
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras import layers, models
from google.colab import drive

def mount_drive():
    drive.mount('/content/drive')

def load_datasets():
    train_pixel_size_path = '/content/drive/My Drive/B3_code/ultrasound_dataset/training_set_pixel_size_and_HC.csv'
    test_pixel_size_path = '/content/drive/My Drive/B3_code/ultrasound_dataset/test_set_pixel_size.csv'
    train_df = pd.read_csv(train_pixel_size_path)
    test_df = pd.read_csv(test_pixel_size_path)
    return train_df, test_df

def load_image_and_annotation(image_path, annotation_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    annotation = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
    annotation = cv2.resize(annotation, (224, 224))
    return image, annotation

def prepare_data(df, images_dir, annotations_dir):
    images = []
    labels = []
    for _, row in df.iterrows():
        image_path = os.path.join(images_dir, row['filename'])
        annotation_path = os.path.join(annotations_dir, row['filename'].replace('.png', '_Annotation.png'))
        image, _ = load_image_and_annotation(image_path, annotation_path)
        images.append(image)
        labels.append(row['head circumference (mm)'])
    return np.array(images), np.array(labels)

def prepare_test_data(df, images_dir):
    images = []
    for _, row in df.iterrows():
        image_path = os.path.join(images_dir, row['filename'])
        image, _ = load_image_and_annotation(image_path, image_path)
        images.append(image)
    return np.array(images)

def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model(X_train, y_train):
    model = build_model()
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    return model, history

def save_predictions(test_df, y_pred):
    test_df['predicted_head_circumference'] = y_pred
    test_df.to_csv('/content/drive/My Drive/B3_code/ultrasound_dataset/test_predictions.csv', index=False)

if __name__ == '__main__':
    mount_drive()
    train_df, test_df = load_datasets()
    images_dir = '/content/drive/My Drive/B3_code/ultrasound_dataset/training_set'
    annotations_dir = '/content/drive/My Drive/B3_code/ultrasound_dataset/training_set'
    X_train, y_train = prepare_data(train_df, images_dir, annotations_dir)
    X_train = X_train.reshape(X_train.shape[0], 224, 224, 1)
    model, history = train_model(X_train, y_train)
    X_test = prepare_test_data(test_df, '/content/drive/My Drive/B3_code/ultrasound_dataset/test_set')
    X_test = X_test.reshape(X_test.shape[0], 224, 224, 1)
    y_pred = model.predict(X_test)
    save_predictions(test_df, y_pred)
    print(test_df[['filename', 'predicted_head_circumference']].head())
