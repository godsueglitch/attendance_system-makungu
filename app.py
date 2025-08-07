import base64
import datetime
import numpy as np
import cv2
import insightface

from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:12345678@localhost/attendance_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load InsightFace model
model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0)

# ======================= MODELS =======================

class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    face_embedding = db.Column(db.LargeBinary, nullable=False)
    joined_date = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(50), nullable=False)

class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String(200), nullable=False)
    status = db.Column(db.String(50), default="pending")
    supplier_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    customer_email = db.Column(db.String(120), nullable=False)
    blockchain_hash = db.Column(db.String(256))

# ======================= INIT DB =======================
with app.app_context():
    db.create_all()

# ======================= HELPERS =======================

def readb64(base64_string):
    encoded_data = base64_string.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2)

# ======================= ROUTES =======================

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if Admin.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
            return redirect(url_for('signup'))
        hashed_password = generate_password_hash(password)
        admin = Admin(email=email, password=hashed_password)
        db.session.add(admin)
        db.session.commit()
        flash('Signup successful. Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        admin = Admin.query.filter_by(email=email).first()
        if admin and check_password_hash(admin.password, password):
            session['admin_id'] = admin.id
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid email or password.', 'error')
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    total_students = Student.query.count()
    recent_attendance = Attendance.query.order_by(Attendance.timestamp.desc()).limit(10).all()
    return render_template('dashboard.html', total_students=total_students, recent_attendance=recent_attendance)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.')
    return redirect(url_for('home'))

@app.route('/register_student', methods=['GET', 'POST'])
def register_student():
    if request.method == 'POST':
        student_id = request.form['student_id']
        name = request.form['name']
        captured_image = request.form['captured_image']

        img = readb64(captured_image)
        faces = model.get(img)
        if len(faces) != 1:
            flash('Error: Please ensure exactly one face is visible.')
            return redirect(url_for('register_student'))

        face_embedding = faces[0].embedding.astype(np.float32)

        # Check for duplicate face
        students = Student.query.all()
        for student in students:
            stored_embedding = np.frombuffer(student.face_embedding, dtype=np.float32)
            similarity = cosine_similarity(face_embedding, stored_embedding)
            if similarity > 0.7:
                attendance = Attendance(student_id=student.student_id, name=student.name)
                db.session.add(attendance)
                db.session.commit()
                flash(f"Face already exists. Attendance marked for {student.name} (ID: {student.student_id}).")
                return redirect(url_for('register_student'))

        new_student = Student(
            student_id=student_id,
            name=name,
            face_embedding=face_embedding.tobytes()
        )
        db.session.add(new_student)
        db.session.commit()
        flash('Student registered successfully.')
        return redirect(url_for('dashboard'))

    return render_template('register_student.html')

@app.route('/mark_attendance', methods=['GET', 'POST'])
def mark_attendance():
    if request.method == 'POST':
        captured_image = request.form['captured_image']
        img = readb64(captured_image)

        faces = model.get(img)
        if len(faces) != 1:
            flash('Error: Please ensure exactly one face is visible.')
            return redirect(url_for('mark_attendance'))

        face_embedding = faces[0].embedding.astype(np.float32)

        students = Student.query.all()
        for student in students:
            stored_embedding = np.frombuffer(student.face_embedding, dtype=np.float32)
            similarity = cosine_similarity(face_embedding, stored_embedding)
            if similarity > 0.7:
                attendance = Attendance(student_id=student.student_id, name=student.name)
                db.session.add(attendance)
                db.session.commit()
                flash(f"Attendance marked for {student.name} (ID: {student.student_id})")
                return redirect(url_for('mark_attendance'))

        flash("Face not recognized. Please register first.")
        return redirect(url_for('mark_attendance'))

    return render_template('mark_attendance.html')

@app.route('/view_attendance')
def view_attendance():
    students = Student.query.all()
    recent_attendance = Attendance.query.order_by(Attendance.timestamp.desc()).limit(10).all()

    # Attendance count per student
    attendance_data = db.session.query(
        Student.name,
        db.func.count(Attendance.id).label("count")
    ).join(Attendance, Student.student_id == Attendance.student_id
    ).group_by(Student.name).all()
    attendance_json = [{"name": name, "count": count} for name, count in attendance_data]

    # Heatmap data by hour/day
    heatmap_raw = db.session.query(
        db.extract('dow', Attendance.timestamp).label('day_of_week'),
        db.extract('hour', Attendance.timestamp).label('hour'),
        db.func.count().label('count')
    ).group_by('day_of_week', 'hour').all()

    heatmap_data = [
        {"x": f"{int(hour)}:00", "y": int(day), "v": count}
        for day, hour, count in heatmap_raw
    ]

    return render_template(
        'view_attendance.html',
        students=students,
        recent_attendance=recent_attendance,
        attendance_json=attendance_json,
        heatmap_json=heatmap_data
    )

# ======================= MAIN =======================
if __name__ == '__main__':
    app.run(debug=True)
