```markdown
# Soil Analysis and Crop Recommendation System

This project provides a comprehensive web application for analyzing soil images to determine soil type and moisture level, then offering crop recommendations and market insights using AI. It integrates machine learning models for image classification, the Gemini AI model for detailed agricultural advice, and external APIs for real-time market data.

## Features

*   **User Authentication:** Secure user registration and login system.
*   **Soil Image Analysis:** Upload an image of soil to predict its:
    *   **Soil Type:** (e.g., Alluvial soil, Black Soil, Clay soil, Red soil)
    *   **Moisture Level:** (e.g., dry, moderate, wet)
*   **AI-Powered Crop Recommendations:** Based on the predicted soil type and moisture level, the Gemini AI model provides detailed information including:
    *   Suitable crops
    *   Best seasons for cultivation
    *   Specific cultivation tips and techniques
    *   Water management requirements
    *   Fertilizer and nutrient recommendations
    *   Potential challenges and solutions
*   **Search History:** Users can view their past soil analysis reports.
*   **Agricultural Market Analysis:** Get insights into crop prices, market trends, and recommendations using data fetched from external APIs (World Bank API for production data, simulated data for prices).
*   **Responsive Web Interface:** Built with Flask for a smooth user experience.

## Importance of Features

*   **Accurate Soil Analysis:** Helps farmers and gardeners make informed decisions about what to grow, optimizing yield and resource usage.
*   **Personalized Recommendations:** Gemini AI provides tailored advice, moving beyond generic information to address specific soil conditions.
*   **Data-Driven Decisions:** Market analysis empowers users with economic insights, helping them decide when to plant or sell for maximum profitability.
*   **User-Friendly History:** Easy access to past analyses allows for tracking and learning from previous experiences.
*   **Scalable Architecture:** The modular design (ML models, AI integration, API calls) allows for future expansion and improvements.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository_url>
cd soil-analysis-crop-recommendation
```

### 2. Set up a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: `venv\Scripts\activate`
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** The `requirements.txt` should contain:
```
Flask
Pillow
torch
torchvision
numpy
werkzeug
sqlite3 # Usually built-in
google-generativeai
matplotlib
requests
```

### 4. Obtain Gemini API Key

1.  Go to the [Google AI Studio](https://aistudio.google.com/app/apikey) and generate a new API key.
2.  Replace `"AIzaSyC26LXVSbOkA67LGZqZSGDRehvpGA5okWA"` in `app.py` with your actual Gemini API key:
    ```python
    genai.configure(api_key="YOUR_GEMINI_API_KEY")
    ```

### 5. Prepare Machine Learning Models

This project assumes you have pre-trained PyTorch models for soil type and moisture level prediction.

*   **Model Files:** Place your `model_moisture.pth` and `model_soil_type.pth` files in the root directory of your project.
    *   If you don't have these models, you'll need to train them separately. The provided `predict_moisture` and `predict_type` functions expect a ResNet18 architecture.

### 6. Initialize the Database

The application uses an SQLite database (`soil_analysis.db`). It will be created and initialized automatically when you run the app for the first time.

### 7. Run the Application

```bash
python app.py
```

### 8. Access the Application

Open your web browser and navigate to `http://127.0.0.1:5000/`.

## Usage

1.  **Sign Up / Log In:**
    *   Go to `/signup` to create a new account or `/login` if you already have one.
    *   You will be redirected to the main dashboard after successful login.

2.  **Analyze Soil:**
    *   On the dashboard, upload an image of your soil.
    *   The system will predict the soil type and moisture level.
    *   The Gemini AI will then provide detailed crop recommendations and cultivation advice.

3.  **View History:**
    *   Navigate to the `/history` page to see a record of all your past soil analyses.

4.  **Market Analysis:**
    *   Go to the `/market` page.
    *   Select a crop to view its simulated market data, including price trends and recommendations.

## Project Structure

```
.
├── app.py                     # Main Flask application file
├── requirements.txt           # Python dependencies
├── model_moisture.pth         # Pre-trained ML model for moisture prediction
├── model_soil_type.pth        # Pre-trained ML model for soil type prediction
├── uploads/                   # Directory to store uploaded soil images
├── templates/                 # HTML templates for the web interface
│   ├── main.html
│   ├── index.html
│   ├── login.html
│   ├── signup.html
│   ├── history.html
│   └── marketapi.html
└── static/                    # Static files (CSS, JS, images)
    ├── css/
    ├── js/
    └── img/
```

## Contributing

Feel free to fork the repository, open issues, and submit pull requests.

## License

This project is open-source and available under the MIT License.

---
```
