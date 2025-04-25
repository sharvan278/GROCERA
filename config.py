
import os

class Config:
    # Flask Configuration
    SECRET_KEY = 'coe_project'  # Change this to a secure key
    GEMINI_API_KEY='AIzaSyDfQBM1UKyyHEsOdH_DrhiIx9IBB7veAaE'
    GOOGLE_API_KEY='AIzaSyAz6tkm0PUXWeP1vFCxF8FGH-oqUPo_znU'
    
    # File Upload Configuration
    UPLOAD_FOLDER = 'uploads'
    PROCESSED_FOLDER = 'processed'
    ALLOWED_EXTENSIONS = {'csv'}
    
    # Alert Thresholds
    STOCK_ALERT_THRESHOLD = 5  # Items
    EXPIRY_ALERT_DAYS = 7      # Days