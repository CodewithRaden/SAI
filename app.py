from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import cv2
import face_recognition
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.neighbors import KDTree
import requests
import pandas as pd
import numpy as np
import psycopg2

app = Flask(__name__)
app.secret_key = "matherfather"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/attendance", methods=["GET", "POST"])
def attendance():
    if request.method == "POST":
        pin = request.form.get("pin")
        action = request.form.get("action")

        # Establish a connection to the PostgreSQL database
        conn = psycopg2.connect(
            database="verceldb",
            user="default",
            password="lfQrZe6KxYi5",
            host="ep-billowing-darkness-a1nux2qh-pooler.ap-southeast-1.aws.neon.tech",
            port="5432",
        )
        cursor = conn.cursor()

        # Debug: Print the PIN that is being looked up
        print(f"Searching for PIN: {pin}")

        # Find the Pegawai associated with the provided PIN
        cursor.execute(
            'SELECT id, "namaPegawai" FROM public."Pegawai" WHERE pin = %s', (pin,)
        )
        pegawai = cursor.fetchone()

        # Debug: Print what was fetched
        print(f"Fetched Pegawai: {pegawai}")

        if not pegawai:
            flash("Invalid PIN. Please try again.", "error")
            conn.close()
            return redirect(url_for("attendance"))

        pegawai_id = pegawai[0]
        name = pegawai[1]

        today = datetime.now().strftime("%Y-%m-%d")

        # Check if there's an attendance record for today
        cursor.execute(
            'SELECT * FROM public."Absensi" WHERE "pegawaiId" = %s AND DATE("waktuMasuk") = %s',
            (pegawai_id, today),
        )
        attendance_record = cursor.fetchone()

        # Check-in Action
        if action == "check_in":
            if attendance_record:
                flash(f"{name} already checked in today.", "error")
            else:
                conn.commit()
                flash(f"Check-in successful for {name}.", "success")
                send_attendance_data_to_server(pegawai_id, "check_in", today)

        # Check-out Action
        elif action == "check_out":
            if not attendance_record:
                flash(f"{name} has not checked in today.", "error")
            elif attendance_record[2]:
                flash(f"{name} has already checked out today.", "error")
            else:
                conn.commit()
                flash(f"Check-out successful for {name}.", "success")
                send_attendance_data_to_server(pegawai_id, "check_out", today)

        # Close the connection
        conn.close()
        return redirect(url_for("attendance"))

    return render_template("pin_attendance.html")


@app.route("/capture", methods=["POST"])
def capture():
    if "image" not in request.files or "type" not in request.form:
        return jsonify({"error": "No file uploaded or type not specified"}), 400

    file = request.files["image"]
    attendance_type = request.form["type"]
    image_path = "captured_image.jpg"
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

    for (top, right, bottom, left), face_encoding in zip(
        face_locations, face_encodings
    ):
        match_index = find_matches(face_encoding)
        name = known_face_names[match_index] if match_index != -1 else "Unknown"
        recognized_names.append(name)
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        font = ImageFont.load_default()
        text_bbox = draw.textbbox((left, bottom), name, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle(
            ((left, bottom), (left + text_width, bottom + text_height)),
            fill=(0, 0, 255),
            outline=(0, 0, 255),
        )
        draw.text((left, bottom), name, fill=(255, 255, 255, 255), font=font)

    if name and name != "Unknown":
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            database="verceldb",
            user="default",
            password="lfQrZe6KxYi5",
            host="ep-billowing-darkness-a1nux2qh-pooler.ap-southeast-1.aws.neon.tech",
            port="5432",
        )
        cursor = conn.cursor()

        # Find Pegawai by name
        cursor.execute(
            'SELECT id FROM public."Pegawai" WHERE "namaPegawai" = %s', (name,)
        )
        pegawai = cursor.fetchone()

        if pegawai:
            user_id = pegawai[0]
            now = datetime.now()
            today = now.strftime("%Y-%m-%d")
            current_time = now.strftime("%H:%M:%S")
            saved_image_path = save_image(pil_image, name, attendance_type)

            # Check if there's an attendance record for today
            cursor.execute(
                """
                SELECT * FROM public."Absensi" 
                WHERE "pegawaiId" = %s AND DATE("waktuMasuk") = %s
            """,
                (user_id, today),
            )
            record = cursor.fetchone()

            if attendance_type == "checkin":
                if record:
                    message = f"{name} already checked in today."
                else:
                    message = f"{name} successfully checked in."
                    send_attendance_data_to_server(user_id, "check_in", current_time)

            elif attendance_type == "checkout":
                if not record:
                    message = f"{name} has not checked in today."
                elif record[
                    2
                ]:  # Assuming "waktuKeluar" is the third column in the result
                    message = f"{name} already checked out today."
                else:
                    cursor.execute(
                        """
                        UPDATE public."Absensi" 
                        SET "waktuKeluar" = %s 
                        WHERE "pegawaiId" = %s AND DATE("waktuMasuk") = %s
                    """,
                        (current_time, user_id, today),
                    )
                    message = f"{name} successfully checked out."
                    send_attendance_data_to_server(user_id, "check_out", current_time)

            conn.commit()

        conn.close()

    del draw
    annotated_image_path = os.path.join("static", "annotated_image.jpg")
    pil_image.save(annotated_image_path)

    return jsonify(
        {
            "message": message,
            "names": recognized_names,
            "image_path": annotated_image_path,
        }
    )


def resize_image(image, target_width=500):
    height, width = image.shape[:2]
    ratio = target_width / float(width)
    new_dimensions = (target_width, int(height * ratio))
    return cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)


def save_image(pil_image, name, attendance_type):
    dir_path = f"static/captured/{name}/"
    os.makedirs(dir_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{attendance_type}_{timestamp}.jpg"
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
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(person_path, filename)
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        face_encodings.append(encodings[0])
                        face_names.append(person_dir)

    known_face_encodings = face_encodings
    known_face_names = face_names
    face_tree = KDTree(np.array(known_face_encodings), leaf_size=2)


@app.route("/face_attedance")
def face_index():
    return render_template("face_attendance.html")


@app.route("/register_combined", methods=["POST"])
def register_combined():
    name = request.form["name"]
    pin = request.form["pin"]
    files = request.files.getlist("images")

    if not name or not pin or not files:
        return jsonify({"error": "Name, PIN, and photos are required"}), 400

    # Connect to the PostgreSQL database
    conn = psycopg2.connect(
        database="verceldb",
        user="default",
        password="lfQrZe6KxYi5",
        host="ep-billowing-darkness-a1nux2qh-pooler.ap-southeast-1.aws.neon.tech",
        port="5432",
    )
    cursor = conn.cursor()

    # Check if the PIN is already in use
    cursor.execute('SELECT * FROM public."Pegawai" WHERE pin = %s', (pin,))
    existing_user = cursor.fetchone()

    if existing_user:
        conn.close()
        return jsonify({"error": "PIN already in use. Please choose another PIN."}), 400

    # Update the Pegawai record with the provided PIN
    cursor.execute(
        """
        UPDATE public."Pegawai" 
        SET pin = %s 
        WHERE "namaPegawai" = %s
        RETURNING id
    """,
        (pin, name),
    )
    conn.commit()

    updated_pegawai_id = cursor.fetchone()

    if not updated_pegawai_id:
        conn.close()
        return (
            jsonify(
                {
                    "error": "No Pegawai found with that name. Please check the name and try again."
                }
            ),
            400,
        )

    person_dir = os.path.join("faces", name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    for file in files:
        filename = file.filename
        file.save(os.path.join(person_dir, filename))

    load_known_faces("faces")
    conn.close()

    return (
        jsonify(
            {"message": f"{name} has been registered successfully with PIN {pin}."}
        ),
        200,
    )


@app.route("/register_combined_view")
def register_combined_view():
    return render_template("register_combined.html")


# Add New api for sai?


def send_attendance_data_to_server(user_id, action, time):
    url = f"http://sai-web-alpha.vercel.app/api/addAbsensi/{user_id}"
    headers = {"x-vercel-protection-bypass": "fx7j8AGuf6OHN4FlXqWJjcrLhEYxIArj"}

    data = {"user_id": user_id, "action": action, "time": time}

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        print(f"Successfully sent {action} data to server for user_id {user_id}.")
    else:
        print(
            f"Failed to send data to server. Status Code: {response.status_code}. Response: {response.text}"
        )


@app.route("/get_pg_names", methods=["GET"])
def get_pg_names():
    try:
        # Establish a connection to the PostgreSQL database
        conn = psycopg2.connect(
            database="verceldb",
            user="default",
            password="lfQrZe6KxYi5",
            host="ep-billowing-darkness-a1nux2qh-pooler.ap-southeast-1.aws.neon.tech",
            port="5432",
        )
        cursor = conn.cursor()

        # Execute the query to fetch names of Pegawai
        cursor.execute('SELECT "namaPegawai" FROM public."Pegawai"')
        names = cursor.fetchall()

        # Close the connection
        conn.close()

        # Return the names as a JSON array
        return jsonify([name[0] for name in names])

    except Exception as e:
        # Handle exceptions, such as database connection errors
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_known_faces("faces")
    app.run(debug=True)
