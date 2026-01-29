import os
import numpy as np
import cv2
from flask import Flask, render_template, request, url_for, jsonify
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- KONFIGURASI FOLDER ---
UPLOAD_FOLDER = 'web_app/static/uploads'
RESULT_FOLDER = 'web_app/static/results'
MODEL_PATH = 'web_app/model/unet_model.h5'

for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# --- LOAD MODEL SEKALI SAJA ---
print("Sedang memuat model Segmentasi U-Net...")
model = load_model(MODEL_PATH, compile=False) 
print("Model Siap!")

# --- FUNGSI CORE AI ---
def process_segmentation(image_path, filename):
    # 1. Baca Gambar
    original_img = cv2.imread(image_path)
    if original_img is None: return None, None, None, "Error", 0, 0

    h, w = original_img.shape[:2]
    
    # Simpan PNG Asli (jika input TIF)
    clean_name = filename.rsplit('.', 1)[0]
    original_png_name = 'orig_' + clean_name + '.png'
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], original_png_name), original_img)

    # 2. Preprocessing
    img_input = cv2.resize(original_img, (128, 128))
    img_input = img_input / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    # 3. Prediksi AI
    pred_mask = model.predict(img_input, verbose=0)
    mask = (pred_mask[0] > 0.5).astype(np.uint8)
    mask_resized = cv2.resize(mask, (w, h))

    # 4. Simpan Masker Hitam Putih
    mask_visual = mask_resized * 255
    mask_png_name = 'mask_' + clean_name + '.png'
    cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], mask_png_name), mask_visual)

    # 5. Hitung Area & Status
    tumor_pixel_count = np.count_nonzero(mask_resized)
    tumor_area_cm = (np.sqrt(tumor_pixel_count) * 0.026) ** 2
    status = "Tumor Detected" if tumor_pixel_count > 0 else "No Tumor"
    
    # 6. Hitung Jumlah Lesi/Spot (Count)
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tumor_count = len(contours)

    # 7. Hitung Densitas Tumor (Relative to Brain Area)
    # Ubah gambar asli ke Grayscale untuk membedakan otak vs background
    gray_original = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # Hitung piksel yang BUKAN hitam (Pixel > 10 dianggap jaringan otak)
    brain_pixel_count = np.count_nonzero(gray_original > 10)
    
    if brain_pixel_count > 0:
        # Rumus: (Piksel Tumor / Piksel Otak) * 100
        tumor_density = (tumor_pixel_count / brain_pixel_count) * 100
    else:
        tumor_density = 0

    # Pastikan tidak lebih dari 100% (jika ada noise)
    tumor_density = min(tumor_density, 100.0)
    
    # 8. Visualisasi Overlay
    if tumor_pixel_count > 0:
        heatmap = np.zeros_like(original_img)
        heatmap[:, :, 2] = 255 # Merah
        heatmap = cv2.bitwise_and(heatmap, heatmap, mask=mask_resized)
        output_img = cv2.addWeighted(original_img, 1, heatmap, 0.5, 0)
        cv2.drawContours(output_img, contours, -1, (0, 0, 255), 2) # Garis Tepi Merah
    else:
        output_img = original_img

    result_png_name = 'res_' + clean_name + '.png'
    cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], result_png_name), output_img)

    # Return nambah variabel 'tumor_density'
    return original_png_name, mask_png_name, result_png_name, status, tumor_area_cm, tumor_count, tumor_density

# --- ROUTE HALAMAN UTAMA (GET) ---
@app.route('/')
def index():
    # Hanya me-render kerangka HTML kosong. 
    # Tidak ada logika upload di sini lagi.
    return render_template('index.html')

# --- ROUTE API KHUSUS AJAX (POST) ---
@app.route('/analyze', methods=['POST'])
def analyze():
    # Validasi Request
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Tidak ada file dikirim'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Nama file kosong'}), 400

    if file:
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Panggil Fungsi AI
            orig, mask, res, status, area, count, density = process_segmentation(file_path, filename)

            if orig is None:
                return jsonify({'success': False, 'error': 'Gagal membaca gambar atau format tidak didukung'}), 400

            # Siapkan Data Visual untuk Frontend
            status_color = "text-red-500" if status == "Tumor Detected" else "text-green-400"
            status_text_color = "text-red-400" if status == "Tumor Detected" else "text-green-400"
            status_label = "Bahaya" if status == "Tumor Detected" else "Aman"
            
            # Kirim balasan JSON (Data Mentah) ke JavaScript
            return jsonify({
                'success': True,
                'orig_url': url_for('static', filename='uploads/' + orig),
                'mask_url': url_for('static', filename='results/' + mask),
                'res_url': url_for('static', filename='results/' + res),
                'prediction': status,
                'area': f"{area:.2f}",
                'count': count,
                'density': f"{density:.2f}",
                'status_color': status_color,
                'status_label': status_label,
                'status_text_color': status_text_color
            })

        except Exception as e:
            print(f"SERVER ERROR: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)