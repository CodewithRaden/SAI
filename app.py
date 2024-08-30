from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import cv2
import face_recognition
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.neighbors import KDTree
import sqlite3

app = Flask(__name__)
app.secret_key = 'matherfather'


import sqlite3

def init_db():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            pin TEXT NOT NULL UNIQUE
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            nama TEXT NOT NULL,
            tanggal DATE NOT NULL,
            waktu_masuk TIME,
            waktu_keluar TIME,
            keterlambatan TEXT,
            status TEXT,
            checkin_image TEXT,
            checkout_image TEXT,
            FOREIGN KEY(user_id) REFERENCES user(id)
        )
    ''')
    
    conn.commit()
    conn.close()   

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    if request.method == 'POST':
        pin = request.form.get('pin')
        action = request.form.get('action')

        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM user WHERE pin = ?', (pin,))
        user = cursor.fetchone()

        if not user:
            flash('Invalid PIN. Please try again.', 'error')
            conn.close()
            return redirect(url_for('attendance'))

        user_id = user[0]  
        name = user[1]  
    
        today = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H:%M:%S')

        cursor.execute('SELECT * FROM attendance WHERE user_id = ? AND tanggal = ?', (user_id, today))
        attendance_record = cursor.fetchone()

        if action == 'check_in':
            if attendance_record and attendance_record[4]:
                flash(f'{name}  already checked in today.', 'error')
            else:
                cursor.execute('''
                    INSERT INTO attendance (user_id, nama, tanggal, waktu_masuk, status)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user_id, name, today, current_time, 'Hadir'))
                conn.commit()
                flash(f'Check-in successful for {name} at {current_time}.', 'success')

        elif action == 'check_out':
            if not attendance_record or not attendance_record[4]: 
                flash(f'{name} have not checked in today.', 'error')
            elif attendance_record[5]: 
                flash(f'{name} have already checked out today.', 'error')
            else:
                cursor.execute('''
                    UPDATE attendance
                    SET waktu_keluar = ?, status = ?
                    WHERE user_id = ? AND tanggal = ?
                ''', (current_time, 'Hadir', user_id, today))
                conn.commit()
                flash(f'Check-out successful for {name} at {current_time}.', 'success')

        conn.close()
        return redirect(url_for('attendance'))

    return render_template('pin_attendance.html')



@app.route('/capture', methods=['POST'])
def capture():
    if 'image' not in request.files or 'type' not in request.form:
        return jsonify({'error': 'No file uploaded or type not specified'}), 400

    file = request.files['image']
    attendance_type = request.form['type']
    image_path = 'captured_image.jpg'
    file.save(image_path)

    image = face_recognition.load_image_file(image_path)
    image = resize_image(image)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    recognized_names = []
    message = ""

    name = None 

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        match_index = find_matches(face_encoding)
        name = known_face_names[match_index] if match_index != -1 else "Unknown"
        recognized_names.append(name)
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        font = ImageFont.load_default()
        text_bbox = draw.textbbox((left, bottom), name, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle(((left, bottom), (left + text_width, bottom + text_height)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left, bottom), name, fill=(255, 255, 255, 255), font=font)

    if name and name != "Unknown": 
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()

        cursor.execute('SELECT id FROM user WHERE name = ?', (name,))
        user_id = cursor.fetchone()

        if user_id:
            user_id = user_id[0]
            now = datetime.now()
            today = now.strftime('%Y-%m-%d')
            current_time = now.strftime('%H:%M:%S')
            saved_image_path = save_image(pil_image, name, attendance_type)

            cursor.execute('''
                SELECT * FROM attendance WHERE user_id = ? AND tanggal = ?
            ''', (user_id, today))
            record = cursor.fetchone()

            if attendance_type == 'checkin':
                if record:
                    message = f'{name} sudah melakukan check-in hari ini.'
                else:
                    keterlambatan = "Ya" if now.time() > datetime.strptime("09:15:00", '%H:%M:%S').time() else "Tidak"
                    cursor.execute('''
                        INSERT INTO attendance (user_id, nama, tanggal, waktu_masuk, keterlambatan, status, checkin_image)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (user_id, name, today, current_time, keterlambatan, 'Hadir', saved_image_path))
                    message = f'{name} berhasil check-in.'

            elif attendance_type == 'checkout':
                if not record or not record[4]:
                    message = f'{name} belum melakukan check-in hari ini.'
                elif record[5]: 
                    message = f'{name} sudah melakukan check-out hari ini.'
                else:
                    cursor.execute('''
                        UPDATE attendance SET waktu_keluar = ?, status = ?, checkout_image = ? WHERE user_id = ? AND tanggal = ?
                    ''', (current_time, 'Hadir', saved_image_path, user_id, today))
                    message = f'{name} berhasil check-out.'

            conn.commit()
        conn.close()

    del draw
    annotated_image_path = os.path.join('static', 'annotated_image.jpg')
    pil_image.save(annotated_image_path)

    return jsonify({'message': message, 'names': recognized_names, 'image_path': annotated_image_path})


def resize_image(image, target_width=500):
    height, width = image.shape[:2]
    ratio = target_width / float(width)
    new_dimensions = (target_width, int(height * ratio))
    return cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

def save_image(pil_image, name, attendance_type):
    dir_path = f'static/captured/{name}/'
    os.makedirs(dir_path, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{name}_{attendance_type}_{timestamp}.jpg'
    filepath = os.path.join(dir_path, filename)
    pil_image.save(filepath)
    return filepath  


def find_matches(encoding, threshold=0.5):
    distances, indices = face_tree.query([encoding], k=1)
    return indices[0][0] if distances[0][0] < threshold else -1

def load_known_faces(known_faces_dir):
    global known_face_encodings, known_face_names, face_tree
    face_encodings = []
    face_names = []

    for person_dir in os.listdir(known_faces_dir):
        person_path = os.path.join(known_faces_dir, person_dir)
        if os.path.isdir(person_path):
            for filename in os.listdir(person_path):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    image_path = os.path.join(person_path, filename)
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        face_encodings.append(encodings[0])
                        face_names.append(person_dir)

    known_face_encodings = face_encodings
    known_face_names = face_names
    face_tree = KDTree(np.array(known_face_encodings), leaf_size=2)

@app.route('/face_attedance')
def face_index():
    return render_template('face_attendance.html')


@app.route('/view_combined')
def view_combined_attendance():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT 
            nama, 
            tanggal, 
            waktu_masuk, 
            waktu_keluar, 
            checkin_image, 
            checkout_image, 
            keterlambatan, 
            status
        FROM attendance
    ''')
    records = cursor.fetchall()
    conn.close()
    return render_template('view_combined.html', records=records)

@app.route('/register_combined', methods=['POST'])
def register_combined():
    name = request.form['name']
    pin = request.form['pin']
    files = request.files.getlist('images')

    if not name or not pin or not files:
        return jsonify({'error': 'Name, PIN, and photos are required'}), 400

    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM user WHERE pin = ?', (pin,))
    existing_user = cursor.fetchone()

    if existing_user:
        return jsonify({'error': 'PIN already in use. Please choose another PIN.'}), 400


    cursor.execute('INSERT INTO user (name, pin) VALUES (?, ?)', (name, pin))
    conn.commit()

    person_dir = os.path.join('faces', name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    for file in files:
        filename = file.filename
        file.save(os.path.join(person_dir, filename))

    load_known_faces('faces')
    conn.close()

    return jsonify({'message': f'{name} has been registered successfully.'}), 200

@app.route('/register_combined_view')
def register_combined_view():
    return render_template('register_combined.html')


if __name__ == '__main__':
    init_db()
    load_known_faces('faces')
    app.run(debug=True)