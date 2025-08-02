import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
from io import StringIO

# ------------------------
# Load Fire Data
# ------------------------
data = """latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight,type
28.0993,96.9983,303,1.1,1.1,2021-01-01,0409,Terra,MODIS,44,6.03,292.6,8.6,D,0
23.7779,86.3951,314.6,1.3,1.1,2021-01-01,0720,Aqua,MODIS,66,6.03,302.5,10.9,D,2"""

df = pd.read_csv(StringIO(data))

# ------------------------
# Data Exploration
# ------------------------
print("First few rows of the dataset:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

# ------------------------
# Visualizations
# ------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
sns.histplot(df['brightness'], bins=20, kde=True)
plt.title('Brightness Distribution')

plt.subplot(2, 2, 2)
sns.scatterplot(x='brightness', y='frp', data=df, hue='satellite')
plt.title('Brightness vs FRP')

plt.subplot(2, 2, 3)
sns.boxplot(x='satellite', y='frp', data=df)
plt.title('FRP by Satellite')

plt.subplot(2, 2, 4)
sns.countplot(x='type', data=df)
plt.title('Fire Type Count')

plt.tight_layout()
plt.show()

# ------------------------
# Correlation Matrix
# ------------------------
plt.figure(figsize=(10, 8))
corr = df[['brightness', 'scan', 'track', 'confidence', 'bright_t31', 'frp']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# ------------------------
# Satellite-wise Stats
# ------------------------
sat_stats = df.groupby('satellite').agg({
    'brightness': ['mean', 'max', 'min'],
    'frp': ['mean', 'sum', 'max'],
    'type': 'count'
})
print("\nSatellite Statistics:")
print(sat_stats)

# ------------------------
# High Confidence Fires
# ------------------------
high_conf = df[df['confidence'] > 50]
print(f"\nNumber of high-confidence fires (>50): {len(high_conf)}")

# ------------------------
# Image-Based Deforestation Detection
# ------------------------
def detect_deforestation(before_path='before.jpg', after_path='after.jpg'):
    try:
        before = cv2.imread(before_path)
        after = cv2.imread(after_path)

        if before is None or after is None:
            print("❌ Could not load one or both images. Check file paths.")
            return

        before = cv2.resize(before, (512, 512))
        after = cv2.resize(after, (512, 512))

        gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray_before, gray_after)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Save result image
        cv2.imwrite('deforestation_detected.jpg', thresh)
        print("✅ Deforestation detection complete! Saved as 'deforestation_detected.jpg'.")

        # Display result
        cv2.imshow('Before Image', gray_before)
        cv2.imshow('After Image', gray_after)
        cv2.imshow('Detected Deforestation', thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"⚠️ Error during deforestation detection: {e}")

# ------------------------
# Run Detection
# ------------------------
detect_deforestation('before.jpg', 'after.jpg')
