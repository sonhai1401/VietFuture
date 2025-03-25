from dlibModel import DlibModel
import numpy as np
from db.conn import MongoConnection
from flask import Flask, request, jsonify, render_template, send_from_directory # type: ignore
from flask_cors import CORS # type: ignore
import dlib # type: ignore
from gevent.pywsgi import WSGIServer # type: ignore
import cv2 # type: ignore
import json
import redis # type: ignore
import os
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Load environment variables
load_dotenv()

# Configuration
app = Flask(
    __name__,
    template_folder='template',  # Thư mục chứa HTML
    static_folder='css'  # Thư mục chứa file tĩnh (CSS, JS, Images)
)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # Set the maximum request size to 32 MB
CORS(app)

# Redis and MongoDB configuration
redis_password = os.getenv("REDIS_PASSWORD")
mongo_uri = os.getenv("MONGO_URI")

r = redis.Redis(
    host="redis-15991.c292.ap-southeast-1-1.ec2.redns.redis-cloud.com", 
    port=15991,
    username="lehuusonhaigmailcom-free-db",  # use your Redis user
    password=redis_password,  # use your Redis password
)

mongoConn = MongoConnection()

# Load dlib models
face_detector = dlib.get_frontal_face_detector()
face_encoder = dlib.face_recognition_model_v1("realtime2/models/dlib_face_recognition_resnet_model_v1.dat")
shape_predictor = dlib.shape_predictor("realtime2/models/shape_predictor_68_face_landmarks.dat")

# Initialize the Limiter for rate-limiting requests
limiter = Limiter(get_remote_address, app=app, default_limits=["200 per day", "50 per hour"])

users, index = mongoConn.get_all()

# Constraint value for face recognition distance threshold
constraint = 0.042

def normalize_vector(vector):
    return vector / np.linalg.norm(vector)

# Cache user data using Redis
def get_user_from_cache(user_id):
    cached_user = r.get(user_id)
    if cached_user:
        return json.loads(cached_user)
    else:
        user_data = mongoConn.get_user_data(user_id)
        if user_data:
            r.setex(user_id, 3600, json.dumps(user_data))  # Cache for 1 hour
            return user_data
    return None

@app.route('/js/<path:path>')
def serve_js(path):
    return send_from_directory('js', path)

@app.route('/images/<path:path>')
def serve_images(path):
    return send_from_directory('images', path)

@app.route('/assets/<path:path>')
def serve_assets(path):
    return send_from_directory('assets', path)

@app.route("/")
def index_get():
    return render_template("index.html")

@app.route('/verify', methods=['POST'])
@limiter.limit("50 per hour")  # Rate limiting
def verify():
    try:
        if 'img' not in request.files:
            return jsonify({"status": "error", "message": "No image provided"}), 400

        image_file = request.files['img']
        img_array = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Face detection
        face_locations = face_detector(img, 1)
        if not face_locations:
            return jsonify({"status": "error", "message": "No face detected"}), 400

        face_location = face_locations[0]
        shape = shape_predictor(img, face_location)
        face_chip = dlib.get_face_chip(img, shape)
        embedding = normalize_vector(np.array(face_encoder.compute_face_descriptor(face_chip), dtype=np.float32))

        # Compare face embeddings
        labels, distances = index.knn_query(embedding, k=1)
        if distances[0][0] < constraint:
            user = users[labels[0][0]]
            user_data = get_user_from_cache(user)
            if user_data:
                return jsonify({
                    "status": "success",
                    "message": "Face recognized",
                    "user": user_data.get("user"),
                    "email": user_data.get("email"),
                })
        return jsonify({"status": "error", "message": "Face not recognized"}), 400

    except Exception as e:
        return jsonify({"status": "error", "message": "An error occurred", "details": str(e)}), 500

@app.route('/add', methods=['POST'])
def add():
    try:
        user = request.form.get('user')
        email = request.form.get('email')
        accountnumber = request.form.get('accountnumber')
        
        # Check if the user already exists
        if mongoConn.collection.find_one({"user": user}):
            return jsonify({"status": "error", "message": "User already exists"}), 400

        image_file = request.files['img']  # Get image from request
        img_array = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Face detection
        face_locations = face_detector(img, 1)
        embeddings = []
        for face_location in face_locations:
            shape = shape_predictor(img, face_location)
            face_chip = dlib.get_face_chip(img, shape)
            embedding = np.array(face_encoder.compute_face_descriptor(face_chip))
            embeddings.append(embedding.tolist())

        if not embeddings:
            return jsonify({"status": "error", "message": "No face detected"}), 400

        # Save face embeddings and user data
        mongoConn.insert(user, email, accountnumber, embeddings)
        return jsonify({"status": "success", "message": "User registered successfully"})

    except Exception as e:
        return jsonify({"status": "error", "message": "An error occurred", "details": str(e)}), 500

@app.route('/store')
def store():
    return render_template('store.html')

@app.route('/product')
def product():
    return render_template('product.html')

@app.route('/fine_tune', methods=['POST']) 
def fine_tune_constraint():
    global constraint
    constraint = float(request.form['constraint'])
    return jsonify({"status": "success", "message": "Constraint updated successfully"})

@app.route("/balance", methods=["POST"])
def balance():
    user = request.form["user"]
    balance = mongoConn.getBalance(user)
    return jsonify({"status": "success", "message": balance})

@app.route("/invoice", methods=["POST"])
def invoice():
    user = request.form["user"]
    invoices = mongoConn.getAllInvoiceOfUser(user)
    return jsonify({"status": "success", "message": invoices})

@app.route("/all_invoices", methods=["GET"])
def all_invoices():
    invoices = mongoConn.getAllInvoice()
    return jsonify({"status": "success", "message": invoices})

@app.route("/login", methods=["POST"])
def login():
    user = request.form["user"]
    email = request.form["email"]

    # Get user data from MongoDB
    user_data = mongoConn.collection.find_one({"user": user, "email": email})
    
    if user_data:
        response = jsonify({"status": "success", "message": "Login successful"})
        response.set_cookie('user', user)  # Store the username in a cookie
        return response, 200
    return jsonify({"status": "error", "message": "Invalid credentials"}), 401

@app.route('/session', methods=['GET'])
def check_session():
    user = request.cookies.get('user')  # Get the user from cookie
    if user:
        user_data = mongoConn.get_user_data(user)
        if user_data:
            return jsonify({"loggedIn": True, "username": user}), 200
    return jsonify({"loggedIn": False}), 200

@app.route('/logout', methods=['POST'])
def logout():
    response = jsonify({"status": "success", "message": "Logged out successfully"})
    response.set_cookie('username', '', expires=0)  # Delete cookie
    return response

def reset():
    global users, index
    users, index = mongoConn.get_all()

if __name__ == "__main__":
    # app.run(debug=False, port=8080)
    print("Starting server...")
    http_server = WSGIServer(('', 8080), app)
    http_server.serve_forever()
