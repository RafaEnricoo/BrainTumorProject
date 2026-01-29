import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization
from tensorflow.keras.metrics import Precision, Recall
import glob

# --- 1. KONFIGURASI ---
# Pastikan path ini benar (sesuai folder dataset di VSCode kamu)
DATASET_PATH = 'dataset/kaggle_3m' 
EVAL_DIR = 'eval_graphs' # Folder baru untuk simpan grafik
IMG_SIZE = (128, 128)
EPOCHS = 20 # Saya naikkan sedikit biar grafiknya kelihatan tren-nya
BATCH_SIZE = 16

# Buat folder evaluasi jika belum ada
if not os.path.exists(EVAL_DIR):
    os.makedirs(EVAL_DIR)

# --- 2. DEFINISI METRIK EVALUASI (CUSTOM METRICS) ---
# Ini rumus matematika untuk Dice & IoU agar bisa dihitung Live saat training

def dice_coefficient(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_coefficient(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# --- 3. PERSIAPAN DATA ---
print("1. Menyiapkan Data...")
mask_files = glob.glob(os.path.join(DATASET_PATH, '*/*_mask.tif'))
image_files = [m.replace('_mask.tif', '.tif') for m in mask_files]

def load_data(image_list, mask_list):
    images = []
    masks = []
    for img_path, mask_path in zip(image_list, mask_list):
        if not os.path.exists(img_path): continue
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, IMG_SIZE)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=-1)
        
        # Ambil hanya yang ada tumornya (biar training fokus)
        if np.max(mask) > 0: 
            images.append(img)
            masks.append(mask)
            
    return np.array(images), np.array(masks)

X, Y = load_data(image_files, mask_files)
print(f"   Total data loaded: {len(X)} images")

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

# --- 4. MEMBANGUN MODEL U-NET ---
print("\n2. Membangun Arsitektur U-Net...")
def unet_model(input_size=(128, 128, 3)):
    inputs = Input(input_size)
    
    # Encoder
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    # Bottleneck
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    
    # Decoder
    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)
    
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(c7)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)
    
    return Model(inputs=[inputs], outputs=[outputs])

model = unet_model()

# Compile dengan Metrik Lengkap
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy', dice_coefficient, iou_coefficient, Precision(), Recall()])

model.summary()

# --- 5. TRAINING ---
print("\n3. Mulai Training...")
history = model.fit(X_train, y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS, 
                    validation_data=(X_test, y_test))

# --- 6. SIMPAN MODEL ---
print("\n4. Menyimpan Model...")
target_dir = 'web_app/model'
if not os.path.exists(target_dir): os.makedirs(target_dir)
model.save(os.path.join(target_dir, 'unet_model.h5'))

# --- 7. PLOTTING & EVALUASI OTOMATIS ---
print(f"\n5. Membuat Grafik Evaluasi di folder '{EVAL_DIR}'...")

def plot_metric(history, metric_name, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history[metric_name], label='Training ' + title)
    plt.plot(history.history['val_' + metric_name], label='Validation ' + title)
    plt.title('Model ' + title)
    plt.ylabel(title)
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(EVAL_DIR, filename))
    plt.close()
    print(f"   -> Grafik {filename} tersimpan.")

# Generate semua grafik yang dibutuhkan untuk Skripsi
plot_metric(history, 'accuracy', 'Pixel Accuracy', '1_accuracy_plot.png')
plot_metric(history, 'loss', 'Loss (Binary Crossentropy)', '2_loss_plot.png')
plot_metric(history, 'dice_coefficient', 'Dice Coefficient (F1-Score)', '3_dice_plot.png')
plot_metric(history, 'iou_coefficient', 'Intersection over Union (IoU)', '4_iou_plot.png')
plot_metric(history, 'precision', 'Precision', '5_precision_plot.png')
plot_metric(history, 'recall', 'Recall (Sensitivity)', '6_recall_plot.png')

print("\n=== TRAINING SELESAI & EVALUASI TERSIMPAN ===")