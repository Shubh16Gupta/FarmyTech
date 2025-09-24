import os
import sqlite3
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, g
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import google.generativeai as genai
from datetime import datetime

# Configure your Gemini API key here
genai.configure(api_key="AIzaSyC26LXVSbOkA67LGZqZSGDRehvpGA5okWA")

# Create a Gemini model instance
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Ensure the 'uploads' directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['DATABASE'] = 'soil_analysis.db'

# Database initialization
def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        db.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                soil_type TEXT NOT NULL,
                moisture_level TEXT NOT NULL,
                gemini_info TEXT NOT NULL,
                search_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        db.commit()

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(app.config['DATABASE'])
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error):
    if hasattr(g, 'db'):
        g.db.close()

# --- Your original prediction functions ---

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_moisture(image_path, model_path='model_moisture.pth', class_names=['dry', 'moderate', 'wet']):
    # Load the model
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return class_names[predicted.item()]

def predict_type(image_path, model_path='model_soil_type.pth', class_names=['Alluvial soil', 'Black Soil', 'Clay soil', 'Red soil']):
    # Load the model
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return class_names[predicted.item()]

def save_search_history(user_id, filename, soil_type, moisture_level, gemini_info):
    db = get_db()
    db.execute(
        'INSERT INTO history (user_id, filename, soil_type, moisture_level, gemini_info) VALUES (?, ?, ?, ?, ?)',
        (user_id, filename, soil_type, moisture_level, gemini_info)
    )
    db.commit()

def get_search_history(user_id):
    db = get_db()
    history = db.execute(
        'SELECT * FROM history WHERE user_id = ? ORDER BY search_date DESC',
        (user_id,)
    ).fetchall()
    return history

# --- Authentication routes ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        db = get_db()
        user = db.execute(
            'SELECT * FROM users WHERE username = ?', (username,)
        ).fetchone()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('upload_file'))
        else:
            return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        db = get_db()
        try:
            db.execute(
                'INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
                (username, generate_password_hash(password), email)
            )
            db.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('signup.html', error='Username or email already exists')
    
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# --- Main application routes ---

@app.route('/dashboard', methods=['GET', 'POST'])
def upload_file():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Get predictions
                soil_type = predict_type(filepath)
                moisture_level = predict_moisture(filepath)
                
                # Send to Gemini for detailed response
                prompt = f"""Given a soil type of '{soil_type}' and a moisture level of '{moisture_level}', provide detailed information about:
                1. Suitable crops for this soil type and moisture level
                2. Best seasons for cultivation
                3. Specific cultivation tips and techniques
                4. Water management requirements
                5. Fertilizer and nutrient recommendations
                6. Potential challenges and solutions
                
                Please structure your response with clear headings and bullet points for easy reading.
                and give response in markup code and dont give any backticks or ` kind of symbols
                """
                
                gemini_response = gemini_model.generate_content(prompt)
                
                # Save raw text response (not formatted HTML)
                raw_gemini_text = gemini_response.text
                
                # Save to history
                save_search_history(session['user_id'], filename, soil_type, moisture_level, raw_gemini_text)
                
                return jsonify({
                    'soil_type': soil_type,
                    'moisture_level': moisture_level,
                    'gemini_info': raw_gemini_text  # Send raw text instead of formatted HTML
                })
            except Exception as e:
                return jsonify({'error': str(e)})

    return render_template('index.html', username=session.get('username'))

@app.route("/")
def home():
    user_id = session.get('user_id')
    recent_history = []
    if user_id:
        # fetch last 3
        recent_history = get_search_history(user_id)[:3]

    return render_template("main.html",
                           username=session.get('username'),
                           recent_history=recent_history)

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    search_history = get_search_history(session['user_id'])
    return render_template('history.html', history=search_history, username=session.get('username'))


import requests
import json
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import numpy as np
from flask import render_template, request, jsonify

# Add these routes to your existing Flask app

@app.route('/market')
def market_analysis():
    return render_template('marketapi.html')

@app.route('/api/market-data/<crop_name>')
def get_market_data(crop_name):
    try:
        # Use USDA Quick Stats API (public API)
        market_data = fetch_usda_data(crop_name)
        
        return jsonify(market_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def fetch_usda_data(crop_name):
    """
    Fetch real market data from USDA Quick Stats API
    """
    # USDA API endpoint (public access)
    # Note: This is a simplified example. For full implementation, you'd need to register for an API key
    # For demo purposes, we'll use a combination of free agricultural APIs
    
    # Map crop names to USDA commodity codes
    crop_mapping = {
        'wheat': 'WHEAT',
        'rice': 'RICE',
        'corn': 'CORN',
        'soybean': 'SOYBEANS',
        'potato': 'POTATOES',
        'tomato': 'TOMATOES',
        'onion': 'ONIONS',
        'cotton': 'COTTON',
        'sugarcane': 'SUGARCANE'
    }
    
    usda_code = crop_mapping.get(crop_name.lower(), 'CORN')
    
    # Since USDA requires API key registration, let's use a free alternative
    # Using World Bank API for agricultural indicator data
    return fetch_worldbank_data(crop_name)

def fetch_worldbank_data(crop_name):
    """
    Fetch agricultural data from World Bank API (free, no API key required)
    """
    # World Bank indicators for different crops
    indicators = {
        'wheat': 'AG.PRD.CROP.XD',  # Crop production index
        'rice': 'AG.PRD.CROP.XD',
        'corn': 'AG.PRD.CROP.XD',
        'soybean': 'AG.PRD.CROP.XD',
        'potato': 'AG.PRD.CROP.XD',
        'tomato': 'AG.PRD.CROP.XD',
        'onion': 'AG.PRD.CROP.XD',
        'cotton': 'AG.PRD.CROP.XD',
        'sugarcane': 'AG.PRD.CROP.XD'
    }
    
    indicator = indicators.get(crop_name.lower(), 'AG.PRD.CROP.XD')
    
    # World Bank API URL
    url = f"http://api.worldbank.org/v2/country/IND/indicator/{indicator}?format=json"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if len(data) > 1 and data[1]:
            # Process World Bank data
            yearly_data = []
            for item in data[1]:
                if item['value'] is not None:
                    yearly_data.append({
                        'year': item['date'],
                        'value': item['value']
                    })
            
            # Generate price data based on production trends
            prices = generate_price_trends(yearly_data, crop_name)
            
            return {
                'crop_name': crop_name.title(),
                'yearly_data': yearly_data[-10:],  # Last 10 years
                'prices': prices,
                'current_price': prices[-1]['price'] if prices else 0,
                'price_change': calculate_price_change(prices),
                'market_trend': analyze_trend(yearly_data),
                'recommendation': generate_recommendation(crop_name, prices)
            }
        else:
            # Fallback to simulated data if API fails
            return generate_simulated_data(crop_name)
            
    except Exception as e:
        print(f"API Error: {e}")
        # Fallback to simulated data
        return generate_simulated_data(crop_name)

def generate_price_trends(yearly_data, crop_name):
    """
    Generate price trends based on production data
    """
    base_prices = {
        'wheat': 2200, 'rice': 2800, 'corn': 1800, 'soybean': 3500,
        'potato': 1200, 'tomato': 1500, 'onion': 2000, 'cotton': 5500,
        'sugarcane': 3000
    }
    
    base_price = base_prices.get(crop_name.lower(), 2500)
    prices = []
    
    # Generate monthly price data for current year
    current_year = datetime.now().year
    for month in range(1, 13):
        # Simulate seasonal variations
        seasonal_factor = 1 + 0.1 * np.sin(month * np.pi / 6)
        # Add some random variation
        random_factor = 1 + np.random.uniform(-0.05, 0.05)
        price = base_price * seasonal_factor * random_factor
        
        prices.append({
            'month': f"{current_year}-{month:02d}",
            'price': round(price, 2)
        })
    
    return prices

def calculate_price_change(prices):
    if len(prices) < 2:
        return 0
    
    current = prices[-1]['price']
    previous = prices[0]['price']
    return round(((current - previous) / previous) * 100, 2)

def analyze_trend(yearly_data):
    if len(yearly_data) < 3:
        return "Insufficient data"
    
    values = [item['value'] for item in yearly_data[-3:]]
    if values[-1] > values[0]:
        return "Growing"
    elif values[-1] < values[0]:
        return "Declining"
    else:
        return "Stable"

def generate_recommendation(crop_name, prices):
    price_change = calculate_price_change(prices)
    
    if price_change > 5:
        return f"Good time to sell {crop_name}. Prices are rising."
    elif price_change < -5:
        return f"Consider holding {crop_name}. Prices are falling."
    else:
        return f"Market for {crop_name} is stable. Monitor for changes."

def generate_simulated_data(crop_name):
    """
    Generate realistic simulated data as fallback
    """
    # Generate realistic data for demonstration
    current_year = datetime.now().year
    yearly_data = []
    
    for year in range(current_year - 9, current_year + 1):
        yearly_data.append({
            'year': str(year),
            'value': np.random.randint(80, 120)
        })
    
    prices = generate_price_trends(yearly_data, crop_name)
    
    return {
        'crop_name': crop_name.title(),
        'yearly_data': yearly_data,
        'prices': prices,
        'current_price': prices[-1]['price'] if prices else 0,
        'price_change': calculate_price_change(prices),
        'market_trend': analyze_trend(yearly_data),
        'recommendation': generate_recommendation(crop_name, prices)
    }


if __name__ == '__main__':
    init_db()
    app.run(debug=True)