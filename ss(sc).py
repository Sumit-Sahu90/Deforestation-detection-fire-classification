"""
Deforestation Fire Classification System

This script provides machine learning models for detecting fires using:
1. Tabular environmental data (Random Forest)
2. Satellite imagery (CNN)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


class TabularFireClassifier:
    """Handles fire classification using environmental/sensor data"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=3,
            class_weight='balanced',
            random_state=42
        )
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = []

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and preprocess tabular data"""
        data = pd.read_csv(filepath)

        # Handle missing values
        numeric_cols = data.select_dtypes(include=np.number).columns
        data[numeric_cols] = self.imputer.fit_transform(data[numeric_cols])

        # Calculate vegetation indices if available
        if all(col in data.columns for col in ['NIR', 'RED']):
            data['NDVI'] = (data['NIR'] - data['RED']) / (data['NIR'] + data['RED'] + 1e-10)
        if all(col in data.columns for col in ['NIR', 'SWIR']):
            data['NBR'] = (data['NIR'] - data['SWIR']) / (data['NIR'] + data['SWIR'] + 1e-10)

        return data

    def prepare_features(self, data: pd.DataFrame, target_col='fire_presence'):
        """Select and prepare features for modeling"""
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        features = ['temperature', 'NDVI', 'NBR', 'humidity', 'rainfall', 'wind_speed']
        self.feature_names = [f for f in features if f in data.columns]

        X = data[self.feature_names]
        y = data[target_col]

        return X, y

    def train(self, X, y):
        """Train the Random Forest model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        self.plot_feature_importance()

    def plot_feature_importance(self):
        """Visualize feature importance"""
        importance = self.model.feature_importances_
        fi_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=fi_df)
        plt.title('Feature Importance for Fire Detection')
        plt.tight_layout()
        plt.show()

    def predict(self, new_data):
        """Make predictions on new data"""
        if not isinstance(new_data, pd.DataFrame):
            new_data = pd.DataFrame([new_data])

        new_data = new_data.reindex(columns=self.feature_names, fill_value=0)
        scaled = self.scaler.transform(new_data)

        pred = self.model.predict(scaled)
        proba = self.model.predict_proba(scaled)[:, 1]

        return pred, proba


class ImageFireClassifier:
    """Handles fire classification using satellite imagery"""

    def __init__(self, input_shape=(256, 256, 3)):
        self.input_shape = input_shape
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape):
        """Construct CNN architecture"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32):
        """Train the CNN model"""
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size
        )
        self.plot_training_history(history)

    def plot_training_history(self, history):
        """Plot training metrics"""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def predict(self, images):
        """Make predictions on new images"""
        if not isinstance(images, np.ndarray):
            images = np.array(images)

        if images.ndim == 3:
            images = np.expand_dims(images, axis=0)

        preds = self.model.predict(images)
        return (preds > 0.5).astype(int), preds


def main():
    print("Deforestation Fire Classification System")

    # Tabular data example
    print("\n=== Tabular Data Model ===")
    tabular_clf = TabularFireClassifier()

    try:
        data = tabular_clf.load_data('fire_data.csv')
        X, y = tabular_clf.prepare_features(data)
        tabular_clf.train(X, y)

        sample_data = {
            'temperature': 34.2,
            'NDVI': 0.38,
            'NBR': 0.15,
            'humidity': 0.35,
            'rainfall': 0,
            'wind_speed': 12
        }

        pred, prob = tabular_clf.predict(sample_data)
        print(f"\nSample Prediction: {'FIRE' if pred[0] else 'No fire'} (Confidence: {prob[0]:.1%})")

    except Exception as e:
        print(f"Error in tabular model: {str(e)}")

    # Image model section (example placeholder)
    print("\n=== Image Model ===")
    print("Note: This requires preprocessed image data")
    image_clf = ImageFireClassifier(input_shape=(256, 256, 3))

    # Example usage placeholder
    # X_train, y_train = load_image_data()
    # image_clf.train(X_train, y_train, epochs=15)


if __name__ == "__main__":
    main()
