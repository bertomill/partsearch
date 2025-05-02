from flask import Flask
import os
import secrets
from datetime import timedelta
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
import tempfile
from flask_cors import CORS

# Import Firebase config
from firebase_config import FIREBASE_CONFIG

# Load environment variables
load_dotenv()

# Initialize global variables that will be set in create_app
db = None

def create_app():
    # Create the Flask application
    app = Flask(__name__, template_folder='../templates')
    
    # Enable CORS for all routes
    CORS(app)
    
    # Configure the app
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(16))
    app.permanent_session_lifetime = timedelta(days=1)
    
    # Initialize Firebase
    initialize_firebase()
    
    # Register blueprints
    from app.auth.routes import auth_bp
    from app.market_research.routes import market_research_bp
    from app.earnings.routes import earnings_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(market_research_bp)
    app.register_blueprint(earnings_bp)
    
    return app

def initialize_firebase():
    global db
    
    # Get Firebase config from environment variables, falling back to imported config
    firebase_config = {
        "apiKey": os.environ.get("FIREBASE_API_KEY", FIREBASE_CONFIG.get("apiKey")),
        "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN", FIREBASE_CONFIG.get("authDomain")),
        "projectId": os.environ.get("FIREBASE_PROJECT_ID", FIREBASE_CONFIG.get("projectId")),
        "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET", FIREBASE_CONFIG.get("storageBucket")),
        "messagingSenderId": os.environ.get("FIREBASE_MESSAGING_SENDER_ID", FIREBASE_CONFIG.get("messagingSenderId")),
        "appId": os.environ.get("FIREBASE_APP_ID", FIREBASE_CONFIG.get("appId")),
        "measurementId": os.environ.get("FIREBASE_MEASUREMENT_ID", FIREBASE_CONFIG.get("measurementId"))
    }
    
    try:
        # Initialize Firebase App if not already initialized
        if not firebase_admin._apps:
            # Check if the service account file exists
            cred_path = 'firebase-credentials.json'
            # Check for credentials in environment variable (for Vercel)
            firebase_creds_json = os.environ.get('FIREBASE_CREDENTIALS')
            
            # Print some debug info
            print(f"Firebase initialization - checking credentials:")
            print(f"  - Service account file exists: {os.path.exists(cred_path)}")
            print(f"  - Environment credentials: {'Available' if firebase_creds_json else 'Not available'}")
            print(f"  - Project ID: {firebase_config['projectId']}")
            
            if firebase_creds_json:
                # Use credentials from environment variable
                import json
                
                # Create a temporary file to store the credentials
                with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
                    temp_file.write(firebase_creds_json.encode())
                    temp_cred_path = temp_file.name
                
                # Initialize with the temporary credentials file
                cred = credentials.Certificate(temp_cred_path)
                firebase_admin.initialize_app(cred)
                
                # Remove the temporary file
                os.unlink(temp_cred_path)
                
                print("Firebase initialized with credentials from environment variable")
            elif os.path.exists(cred_path):
                # Use the service account credentials file
                try:
                    cred = credentials.Certificate(cred_path)
                    firebase_admin.initialize_app(cred)
                    print("Firebase initialized with service account credentials")
                except Exception as cert_error:
                    print(f"Error initializing with service account file: {cert_error}")
                    raise
            else:
                # For deployment: Use a certificate if provided, otherwise try application default credentials
                # For local development: Fall back to using just the project ID
                try:
                    # Try to initialize with application default credentials first
                    firebase_admin.initialize_app(options={
                        'projectId': firebase_config["projectId"],
                    })
                    print("Firebase initialized with application default credentials")
                except Exception as credential_error:
                    print(f"Falling back to unauthenticated mode: {credential_error}")
                    # Fall back to a minimal configuration (limited capabilities)
                    firebase_admin.initialize_app(options={
                        'projectId': firebase_config["projectId"],
                    }, name='minimal')
                    print("Firebase initialized in minimal mode")
        
        # Get a Firestore client
        db = firestore.client()
        print("Firebase initialized successfully")
    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        db = None 