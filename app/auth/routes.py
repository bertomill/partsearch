from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from functools import wraps
import firebase_admin
from firebase_admin import auth

auth_bp = Blueprint('auth', __name__)

# Helper functions for authentication
def create_user(email, password, display_name=None):
    """Create a new user in Firebase Authentication"""
    try:
        user = auth.create_user(
            email=email,
            password=password,
            display_name=display_name or email.split('@')[0],
            disabled=False
        )
        return user
    except Exception as e:
        print(f"Error creating user: {e}")
        raise

def get_user(uid):
    """Get user details from Firebase Authentication"""
    try:
        user = auth.get_user(uid)
        return user
    except Exception as e:
        print(f"Error getting user: {e}")
        return None

def authenticate_user(id_token):
    """Authenticate a user with a Firebase ID token"""
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        print(f"Error authenticating user: {e}")
        return None

def login_required(f):
    """Decorator to bypass login requirement since you're the only user"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Automatically create a session for the user if not present
        if 'user_id' not in session:
            # Create a dummy session for development purposes
            session.permanent = True
            session['user_id'] = 'dev_user_id'  # Static ID for dev
            session['email'] = 'dev@example.com'
            session['display_name'] = 'Developer'
            print("Auto-login: Development user session created")
        return f(*args, **kwargs)
    return decorated_function

# Route for login page
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        try:
            # Sign in with email and password
            user = auth.get_user_by_email(email)
            
            # Note: Firebase Admin SDK can't verify passwords
            # In a real implementation, we would use Firebase Client SDK or custom auth
            # For this demo, we'll simulate successful login
            # DO NOT USE THIS IN PRODUCTION
            
            # Set session variables
            session.permanent = True
            session['user_id'] = user.uid
            session['email'] = user.email
            session['display_name'] = user.display_name
            
            flash('You have been logged in successfully!', 'success')
            
            next_page = request.args.get('next')
            return redirect(next_page or url_for('market_research.index'))
            
        except Exception as e:
            error = f"Login failed: {str(e)}"
    
    return render_template('login.html', error=error)

# Route for registration page
@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        display_name = request.form.get('display_name', '')
        
        try:
            # Create user in Firebase
            user = create_user(email, password, display_name)
            
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('auth.login'))
            
        except Exception as e:
            error = f"Registration failed: {str(e)}"
    
    return render_template('register.html', error=error)

# Route for logging out
@auth_bp.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('market_research.index'))

# Route for user profile
@auth_bp.route('/profile')
@login_required
def profile():
    from app import db
    user_id = session.get('user_id')
    
    # Get user info from Firebase Auth
    user = get_user(user_id)
    
    # Get user's saved searches from Firestore
    user_searches = []
    if db:
        try:
            searches_ref = db.collection('saved_searches').where('user_id', '==', user_id).order_by('timestamp', direction=firebase_admin.firestore.Query.DESCENDING)
            for doc in searches_ref.stream():
                user_searches.append(doc.to_dict())
        except Exception as e:
            print(f"Error getting user searches: {e}")
    
    return render_template('profile.html', user=user, searches=user_searches) 