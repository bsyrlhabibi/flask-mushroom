from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

# Path ke file model TFLite
model_path = 'Mushroom-Classification-BiT.tflite'

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Daftar nama kelas
class_names = ['Amanita muscaria', 'Amanita rubescens', 'Boletus edulis', 'Calycina citrina', 'Cerioporus squamosus', 'Flammulina velutipes', 'Fomes fomentarius', 'Ganoderma applanatum',
               'Gyromitra gigas', 'Leccinum aurantiacum', 'Paxillus involutus', 'Pleurotus ostreatus', 'Schizophyllum commune', 'Trichaptum biforme', 'Xanthoria parietina']

# Load data nutrisi dari file JSON
with open('mushroom.json', 'r') as file:
    fastfood_data = json.load(file)

# Load data nutrisi dari file JSON
with open('mushroom.json', 'r') as file:
    mushroom_data = json.load(file)

# Fungsi untuk melakukan preprocessing gambar dengan tipe FLOAT32
def preprocess_image(image_data, target_size=(256, 256)):
    # Load gambar menggunakan PIL
    image = Image.open(image_data)

    # Konversi gambar ke mode RGB (jika bukan format RGB)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize gambar ke ukuran yang diinginkan
    image = image.resize(target_size)

    # Konversi nilai piksel menjadi FLOAT32
    image_array = np.array(image, dtype=np.float32) / 255.0

    # Ekspansi dimensi untuk membuat batch
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

# Fungsi untuk melakukan prediksi berdasarkan gambar menggunakan model TFLite
def predict_image(image_data):
    # Preprocessing gambar
    processed_image = preprocess_image(image_data)

    # Salin data gambar ke tensor input model TFLite
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], processed_image)

    # Lakukan inferensi
    interpreter.invoke()

    # Dapatkan output dari tensor output model TFLite
    output_details = interpreter.get_output_details()
    output = interpreter.get_tensor(output_details[0]['index'])

    # Proses hasil output
    predicted_class = class_names[np.argmax(output)]
    confidence = np.max(output)

    # Cari informasi sesuai dengan kelas yang diprediksi
    information = {}
    for mushroom in mushroom_data['jamur']:  # Ubah disini
        if mushroom['mushroom'] == predicted_class:
            information = {
                'nama_indonesia': mushroom['nama_indonesia'],
                'status_edibility': mushroom['status_edibility'],
                'keterangan': mushroom['keterangan'],
            }
            break

    return predicted_class, confidence, information

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    try:
        predicted_class, confidence, information = predict_image(file)
        # Jika prediksi sukses, tambahkan pesan success
        message = "Success"
    except Exception as e:
        return jsonify({'error': str(e)})

    # Atur urutan respons sesuai keinginan di dalam fungsi predict
    response = {
        'message': message,
        'prediksi': predicted_class,
        'tingkat_kepercayaan': float(confidence),
        'infomasi': information
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)