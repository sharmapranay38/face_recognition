import os
import cv2
from flask import Flask, request, jsonify
from imgbeddings import imgbeddings
import psycopg2
from PIL import Image
import numpy as np

app = Flask(__name__)

# Path to the Haar Cascade for face detection
CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Database connection function
def get_db_connection():
    conn = psycopg2.connect("postgres://avnadmin:AVNS_JehIaMQy7ho7CrQGG0W@pg-396e8f1c-sharmapranay38-f5ed.i.aivencloud.com:16167/defaultdb?sslmode=require")
    return conn

# Function to detect faces and return their embeddings
def detect_faces_and_get_embeddings(image):
    haar_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    embeddings = []
    ibed = imgbeddings()

    for (x, y, w, h) in faces:
        face_crop = image[y:y+h, x:x+w]
        pil_image = Image.fromarray(face_crop)
        embedding = ibed.to_embeddings(pil_image)
        embeddings.append(embedding[0])  # Append the face embedding

    return faces, embeddings

# Function to find matching names based on embeddings
def find_matching_faces(face_embeddings):
    matches = []
    conn = get_db_connection()
    cur = conn.cursor()

    for embedding in face_embeddings:
        string_representation = "[" + ",".join(str(x) for x in embedding.tolist()) + "]"
        cur.execute("SELECT picture FROM pictures ORDER BY embedding <-> %s LIMIT 1;", (string_representation,))
        rows = cur.fetchall()

        if rows:
            matches.append(rows[0][0])  # Get the filename of the matching face

    cur.close()
    conn.close()
    return matches

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    image = Image.open(image_file)
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    faces, embeddings = detect_faces_and_get_embeddings(image_np)

    if not faces:
        return jsonify({"error": "No faces detected"}), 404

    matching_faces = find_matching_faces(embeddings)

    return jsonify({"matches": matching_faces})

if __name__ == '__main__':
    app.run(debug=True)
