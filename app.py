# app.py
# Import the Flask library. Think of this like getting a toolkit for building web apps.
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
# Import the os module to access environment variables
import os
# Import sys for system functions
import sys
# Import the Tavily client
from tavily import TavilyClient
# Import the function to load .env files
from dotenv import load_dotenv
# Import re for pattern matching
import re
# Import datetime for date handling
from datetime import datetime, timedelta
# Import OpenAI for advanced analysis
from openai import OpenAI
# Import Google's Gemini for earnings call analysis (large context)
import google.generativeai as genai
# Import requests for API calls
import requests
# Import Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, firestore, auth
# Import for generating unique IDs
import uuid
# Import for secure password handling
import hashlib
import secrets
# Import Firebase config
from firebase_config import FIREBASE_CONFIG
# Import for authentication decorators
from functools import wraps
# Import Claude
from anthropic import Anthropic, APIError

# Load environment variables from .env file *before* accessing them
load_dotenv()

# Create an instance of the Flask class. This is our actual web application object.
# __name__ tells Flask where to look for resources like templates.
app = Flask(__name__)

# Set up a secret key for session management
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(16))
# Set session to expire after 1 day
app.permanent_session_lifetime = timedelta(days=1)

# Get API keys from environment variables
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")
SEARCH1API_KEY = os.environ.get("SEARCH1API_KEY")

# Initialize Tavily Client globally
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

# Initialize Gemini API (if key is available)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API initialized")

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

# Initialize Firebase Admin
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
            import tempfile
            
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

# Function that uses OpenAI to analyze search results
def ai_analyze_results(results, company, metric):
    # If OpenAI API key is not available, fall back to our basic synthesizer
    if not OPENAI_API_KEY:
        basic_answer = synthesize_answer(results, company, metric)
        return basic_answer, basic_answer, False
    
    try:
        # Initialize the OpenAI client
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Create a prompt with the search results
        prompt = f"I need to find accurate information about {company}'s {metric}. Here are search results:\n\n"
        
        # Add context about each source
        for i, result in enumerate(results[:3], 1):
            source_name = result.get('url', 'Unknown').split('/')[2]  # Extract domain
            content = result.get('content', 'No content')
            title = result.get('title', 'No title')
            
            prompt += f"Source {i} ({source_name}, title: '{title}'): {content}\n\n"
        
        # Add specific instructions for both analysis and conclusion
        prompt += "\nPlease provide TWO parts in your response:\n"
        prompt += "1. ANALYSIS: Analyze these sources critically, noting agreements/conflicts between them, the reliability of each source, and the overall confidence in the information. Include relevant dates and context.\n"
        prompt += "2. FINAL ANSWER: Provide a single, clear, direct statement of the most accurate answer about this metric based on the sources. Make this 1-2 sentences maximum.\n\n"
        prompt += "Format your response exactly like this:\n"
        prompt += "ANALYSIS: Your detailed analysis here...\n\n"
        prompt += "FINAL ANSWER: Your concise answer here."
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using a standard model, could be upgraded to GPT-4 for better analysis
            messages=[
                {"role": "system", "content": "You are a financial analyst who provides accurate, concise analyses of business metrics from multiple sources."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2  # Low temperature for more factual responses
        )
        
        # Get the complete response text
        full_response = response.choices[0].message.content
        
        # Split the response into analysis and final answer
        analysis = ""
        final_answer = ""
        
        # Parse the response to extract the two parts
        if "ANALYSIS:" in full_response and "FINAL ANSWER:" in full_response:
            parts = full_response.split("FINAL ANSWER:")
            if len(parts) >= 2:
                analysis_part = parts[0]
                final_answer = parts[1].strip()
                
                # Clean up the analysis part
                if "ANALYSIS:" in analysis_part:
                    analysis = analysis_part.split("ANALYSIS:")[1].strip()
                else:
                    analysis = analysis_part.strip()
        else:
            # If the format wasn't followed, just use the whole response as analysis
            analysis = full_response
            final_answer = synthesize_answer(results, company, metric)  # Use our basic synthesizer for the conclusion
        
        # Return both parts and flag that we used AI
        return analysis, final_answer, True
    
    except Exception as e:
        # If there's any error, log it and fall back to our basic synthesizer
        print(f"Error using OpenAI API: {e}")
        basic_answer = synthesize_answer(results, company, metric)
        return basic_answer, basic_answer, False

# Function to synthesize a "smart" answer from multiple search results (used as fallback)
def synthesize_answer(results, company, metric):
    if not results or len(results) == 0:
        return f"No information found for {metric} of {company}."
    
    # Extract data points, confidence scores, and dates
    data_points = []
    
    for result in results:
        content = result.get('content', '')
        score = result.get('score', 0)
        
        # Try to extract dates from the content
        date_match = re.search(r'(?:as of|in|end of|at the end of)\s(?:the year)?\s?(\w+\s\d{1,2},?\s\d{4}|\d{4})', content, re.IGNORECASE)
        date_str = date_match.group(1) if date_match else "Unknown date"
        
        # Try to extract numeric values (assuming many metrics involve numbers)
        numbers = re.findall(r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:million|billion|percent|%)?', content)
        
        data_points.append({
            'content': content,
            'score': score,
            'date': date_str,
            'numbers': numbers,
            'source': result.get('url', 'Unknown source')
        })
    
    # Sort by score (highest first)
    data_points.sort(key=lambda x: x['score'], reverse=True)
    
    # Identify the most likely correct answer
    top_result = data_points[0]
    
    # Generate a synthesized answer that combines the most credible information
    # This is where an LLM would normally do a better job
    synthesized = f"Based on the highest-rated source ({top_result['source'].split('/')[2]}), "
    
    if "employees" in metric.lower() or "workforce" in metric.lower() or "staff" in metric.lower():
        employee_match = re.search(r'(\d+(?:,\d+)*)\s*employees', top_result['content'], re.IGNORECASE)
        if employee_match:
            synthesized += f"the number of employees at {company} is {employee_match.group(1)}"
            if top_result['date'] != "Unknown date":
                synthesized += f" as of {top_result['date']}."
            else:
                synthesized += "."
        else:
            synthesized = top_result['content']
    elif "revenue" in metric.lower() or "income" in metric.lower() or "profit" in metric.lower() or "sales" in metric.lower():
        # For financial metrics, try to extract dollar amounts
        money_match = re.search(r'\$(\d+(?:,\d+)*(?:\.\d+)?)\s*(million|billion|trillion)?', top_result['content'], re.IGNORECASE)
        if money_match:
            amount = money_match.group(1)
            unit = money_match.group(2) if money_match.group(2) else ""
            synthesized += f"the {metric} for {company} is ${amount} {unit}"
            if top_result['date'] != "Unknown date":
                synthesized += f" as of {top_result['date']}."
            else:
                synthesized += "."
        else:
            synthesized = top_result['content']
    else:
        # For other metrics, just use the top result content
        synthesized = top_result['content']
    
    # Add a disclaimer about different sources
    if len(data_points) > 1 and data_points[0]['content'] != data_points[1]['content']:
        synthesized += " Note: Other sources may show different values."
    
    return synthesized

# Define a route for the homepage of our website (like www.example.com/).
# This function will handle requests for the main page.
# methods=['GET', 'POST'] means this function can handle both loading the page (GET)
# and submitting the form (POST).
@app.route('/', methods=['GET', 'POST'])
def index():
    # Initialize variables to hold the result and any potential error messages.
    result_content = None
    error_message = None
    search_results = None  # New variable to store multiple results
    search_query = None    # Store the query for display
    analysis = None        # Store the detailed AI analysis
    final_answer = None    # Store the concise final answer
    is_ai_answer = False   # Flag to indicate if the answer is from AI or our fallback

    # Check if the API key is available. If not, set an error message.
    if not TAVILY_API_KEY:
        error_message = "Tavily API key is not configured. Please set the TAVILY_API_KEY environment variable."

    # Check if the user submitted the form (a POST request) AND if the API key is available.
    if request.method == 'POST' and TAVILY_API_KEY:
        try:
            # Get the company name entered by the user from the form data.
            company = request.form.get('company')
            # Get the metric name entered by the user from the form data.
            metric = request.form.get('metric')
            
            # Construct the search query for Tavily.
            search_query = f"{company} {metric}"
            print(f"Searching Tavily for: '{search_query}'") # Log the query for debugging

            # Perform the search using Tavily, requesting advanced depth
            # and including raw content, limiting to 5 results for now.
            response = tavily_client.search(
                query=search_query,
                search_depth="advanced",
                include_raw_content=True, # Good practice with advanced depth
                max_results=5
            )

            print(f"Tavily response: {response}") # Log the full response for debugging

            # --- Result Handling (Updated for Multiple Results) ---
            if response and response.get('results'):
                # Store all results (limiting to top 3 for display)
                search_results = response['results'][:3]
                
                # Still keep one main result for backward compatibility
                # (and to potentially use as the "definitive" answer)
                first_result = response['results'][0]
                result_content = first_result.get('content')
                
                # Generate a synthesized answer using OpenAI or our fallback
                analysis, final_answer, is_ai_answer = ai_analyze_results(response['results'], company, metric)
                
                if not result_content:
                    result_content = "Tavily found a result, but it had no content snippet."
            else:
                # Handle cases where Tavily returns no results.
                result_content = f"Sorry, Tavily couldn't find results for '{search_query}'."
            # --------------------------------

        except Exception as e:
            # Catch any errors during the API call or processing.
            print(f"An error occurred: {e}") # Log the error
            error_message = f"An error occurred while searching: {e}"

    # Render the HTML page ('index.html').
    # Pass the main result, error message, ALL search results, the query, and the synthesized answer to the template.
    return render_template('index.html', 
                          result=result_content, 
                          error=error_message,
                          search_results=search_results,
                          search_query=search_query,
                          synthesized_answer=analysis,
                          final_answer=final_answer,
                          is_ai_answer=is_ai_answer)

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
@app.route('/login', methods=['GET', 'POST'])
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
            return redirect(next_page or url_for('index'))
            
        except Exception as e:
            error = f"Login failed: {str(e)}"
    
    return render_template('login.html', error=error)

# Route for registration page
@app.route('/register', methods=['GET', 'POST'])
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
            return redirect(url_for('login'))
            
        except Exception as e:
            error = f"Registration failed: {str(e)}"
    
    return render_template('register.html', error=error)

# Route for logging out
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('index'))

# Route for user profile
@app.route('/profile')
@login_required
def profile():
    user_id = session.get('user_id')
    
    # Get user info from Firebase Auth
    user = get_user(user_id)
    
    # Get user's saved searches from Firestore
    user_searches = []
    if db:
        try:
            searches_ref = db.collection('saved_searches').where('user_id', '==', user_id).order_by('timestamp', direction=firestore.Query.DESCENDING)
            for doc in searches_ref.stream():
                user_searches.append(doc.to_dict())
        except Exception as e:
            print(f"Error getting user searches: {e}")
    
    return render_template('profile.html', user=user, searches=user_searches)

# Update the saved_searches route to check for authentication
@app.route('/saved_searches', methods=['GET'])
@login_required
def view_saved_searches():
    user_id = session.get('user_id')
    
    if not db:
        return render_template('saved_searches.html', 
                              error="Firebase not initialized", 
                              searches=[])
    
    try:
        # Get saved searches for the current user
        searches_ref = db.collection('saved_searches').where('user_id', '==', user_id).order_by('timestamp', direction=firestore.Query.DESCENDING)
        searches = []
        
        # Convert Firestore documents to dictionary
        for doc in searches_ref.stream():
            search_data = doc.to_dict()
            # Format timestamp if it exists
            if 'timestamp' in search_data and search_data['timestamp']:
                try:
                    # Convert Firestore timestamp to Python datetime
                    timestamp = search_data['timestamp']
                    # Format the datetime for display
                    search_data['formatted_time'] = timestamp.strftime('%B %d, %Y - %I:%M %p')
                except Exception as e:
                    search_data['formatted_time'] = 'Unknown time'
            
            searches.append(search_data)
        
        return render_template('saved_searches.html', searches=searches, error=None)
    
    except Exception as e:
        print(f"Error retrieving saved searches: {e}")
        return render_template('saved_searches.html', 
                              error=f"Error retrieving saved searches: {e}", 
                              searches=[])

# Update save_search route to include user ID
@app.route('/save_search', methods=['POST'])
@login_required
def save_search():
    if not db:
        return jsonify({"success": False, "error": "Firebase not initialized"}), 500
    
    user_id = session.get('user_id')
    
    try:
        # Get data from request
        data = request.get_json()
        company = data.get('company')
        metric = data.get('metric')
        final_answer = data.get('final_answer')
        analysis = data.get('analysis')
        search_results = data.get('search_results', [])
        
        # Generate a unique ID for this search
        search_id = str(uuid.uuid4())
        
        # Prepare document data
        search_data = {
            'id': search_id,
            'user_id': user_id,  # Add user ID to the saved search
            'company': company,
            'metric': metric,
            'final_answer': final_answer,
            'analysis': analysis,
            'search_results': search_results,
            'timestamp': firestore.SERVER_TIMESTAMP,
            'query': f"{company} {metric}"
        }
        
        # Save to Firestore
        db.collection('saved_searches').document(search_id).set(search_data)
        
        return jsonify({"success": True, "search_id": search_id}), 200
    
    except Exception as e:
        print(f"Error saving search: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# Route to delete a saved search
@app.route('/delete_search/<search_id>', methods=['POST'])
def delete_search(search_id):
    if not db:
        return jsonify({"success": False, "error": "Firebase not initialized"}), 500
    
    try:
        # Delete the document from Firestore
        db.collection('saved_searches').document(search_id).delete()
        return jsonify({"success": True}), 200
    
    except Exception as e:
        print(f"Error deleting search: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/earnings_call')
def earnings_call():
    """Route for earnings call analysis page"""
    return render_template('earnings_call.html')

@app.route('/analyze_earnings_call', methods=['POST'])
def analyze_earnings_call():
    """API endpoint to analyze an earnings call transcript"""
    # Check if user is logged in
    user_id = session.get('user_id')
    
    # Get data from request
    data = request.json
    company_name = data.get('company_name')
    themes = data.get('themes', '')
    
    # Validate input
    if not company_name:
        return jsonify({'error': 'Company name is required'}), 400
    
    # --- Step 1: Fetch Data with Tavily ---
    fetched_data_context = ""
    if tavily_client:
        try:
            print(f"Searching Tavily for earnings call info for: {company_name}")
            tavily_query = f"{company_name} latest earnings call transcript summary"
            # Use search_depth="advanced" and include_raw_content=True for potentially richer data
            response = tavily_client.search(
                query=tavily_query,
                search_depth="advanced",
                include_raw_content=True, # Get full text if available
                max_results=3 # Get a few top results
            )

            print(f"Tavily response for earnings call: {response}") # Log for debugging

            # Extract content from results
            if response and response.get('results'):
                # Combine the content of the top results to form a context
                context_parts = []
                for result in response['results']:
                    # Prefer raw_content if available, otherwise use content snippet
                    content_to_use = result.get('raw_content') or result.get('content')
                    if content_to_use:
                        context_parts.append(f"Source: {result.get('url', 'Unknown')}\nContent:\n{content_to_use}\n---")

                fetched_data_context = "\n\n".join(context_parts)

            if not fetched_data_context:
                 print(f"Warning: Tavily search for '{tavily_query}' returned no usable content.")
                 # Decide how to handle this: either return an error or let the LLM try without context
                 # For now, we'll proceed and let the LLM handle it, but add a note.
                 fetched_data_context = "No specific earnings call data found via web search."

        except Exception as e:
            print(f"Error during Tavily search for earnings call: {e}")
            # Optionally return an error here, or proceed with a note
            fetched_data_context = f"Error fetching earnings call data: {e}"
    else:
        print("Warning: Tavily client not initialized. Cannot fetch earnings call data.")
        return jsonify({'error': 'Tavily API key not configured, cannot fetch data.'}), 500
    # --- End of Step 1 ---

    try:
        # --- Step 2: Analyze the fetched data ---
        # Common system prompt regardless of which LLM we use
        system_prompt = "You are a financial analyst specializing in earnings call analysis. Provide detailed, insightful analysis with specific metrics and context, based *only* on the provided text."

        # User prompt for both models
        user_prompt = f"Analyze the following earnings call information for {company_name}. "
        if themes:
            user_prompt += f"Focus on these specific themes: {themes}. "

        user_prompt += f"\n\n--- Earnings Call Information ---\n{fetched_data_context}\n--- End of Information ---\n\n"

        user_prompt += """Based *only* on the information provided above, provide a comprehensive analysis with the following sections:
        1. Executive Summary - Key takeaways in bullet points
        2. Financial Performance - Revenue, profit, and key metrics mentioned
        3. Strategic Initiatives - Major announcements and future plans mentioned
        4. Market Context - How the company is positioning relative to competitors, based on the text
        5. Analyst Questions - Notable questions and management responses, if mentioned
        6. Outlook - Management's guidance and future expectations, if mentioned

        If the provided information is insufficient to answer a section, state that clearly.
        Format the response in HTML with appropriate headings (e.g., <h4> for sections), paragraphs, and bullet points (<ul> and <li>).
        """
        
        # Attempt to determine the content length (rough estimate)
        approx_token_count = len(fetched_data_context.split())
        print(f"Estimated word count of fetched content: {approx_token_count}")
        
        # Try Claude for large transcripts if key is available
        if os.environ.get("ANTHROPIC_API_KEY"):
            print(f"Using Claude for {company_name} earnings call analysis (large transcript)...")
            
            try:
                # Initialize Anthropic client
                anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                
                # Call Claude to generate the analysis
                claude_response = anthropic_client.messages.create(
                    model="claude-3-5-haiku-20241022",  # Cost-effective with 200K context
                    max_tokens=4000,
                    temperature=0.4,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                
                # Extract content
                analysis = claude_response.content[0].text
                print("Claude analysis completed successfully")
            except APIError as e:
                print(f"Claude API error: {e}")
                # Fall back to OpenAI if available
                if OPENAI_API_KEY:
                    # Use your existing OpenAI code here
                    print(f"Falling back to OpenAI for {company_name} analysis...")
                    # Initialize OpenAI client
                    openai_client = OpenAI(api_key=OPENAI_API_KEY)
                    # Your existing OpenAI code...
                else:
                    return jsonify({'error': f'Error using Claude API: {str(e)}'}), 500
        else:
            # Use OpenAI for smaller transcripts
            if OPENAI_API_KEY:
                print(f"Using OpenAI for {company_name} analysis...")
                # Initialize OpenAI client
                openai_client = OpenAI(api_key=OPENAI_API_KEY)
                
                # Call OpenAI to generate the analysis
                completion = openai_client.chat.completions.create(
                    model="gpt-4", # Better analysis quality
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.5
                )
                
                # Extract the analysis from the OpenAI response
                analysis = completion.choices[0].message.content
                print("OpenAI analysis completed successfully")
            else:
                return jsonify({'error': 'OpenAI API key not configured.'}), 500
        
        # --- Step 3: Save to Firebase ---
        search_id = None
        if user_id and db:
            # Create a document in Firebase with the analysis
            search_id = str(uuid.uuid4())
            
            try:
                db.collection('earnings_calls').document(search_id).set({
                    'user_id': user_id,
                    'company_name': company_name,
                    'themes': themes,
                    'fetched_context': fetched_data_context[:10000] + "..." if len(fetched_data_context) > 10000 else fetched_data_context,  # Truncate if too large for Firestore
                    'analysis': analysis,
                    'timestamp': firestore.SERVER_TIMESTAMP,
                    'favorited': False
                })
            except Exception as e:
                print(f"Error saving to Firebase: {e}")
                # Continue even if save fails
        
        # Return the analysis
        return jsonify({
            'analysis': analysis,
            'search_id': search_id,
            'fetched_context': fetched_data_context  # Add the fetched context to the response
        })
        
    except Exception as e:
        print(f"Error analyzing earnings call: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/save_earnings_call/<search_id>', methods=['POST'])
def save_earnings_call(search_id):
    """API endpoint to save an earnings call analysis to favorites"""
    # Check if user is logged in
    user_id = session.get('user_id')
    
    if not user_id:
        return jsonify({'error': 'You must be logged in to save analyses'}), 401
        
    if not db:
        return jsonify({'error': 'Database not available'}), 500
    
    try:
        # Update the document in Firebase
        doc_ref = db.collection('earnings_calls').document(search_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            return jsonify({'error': 'Analysis not found'}), 404
            
        # Check if this analysis belongs to the logged-in user
        if doc.to_dict().get('user_id') != user_id:
            return jsonify({'error': 'You can only save your own analyses'}), 403
            
        # Update the favorited status
        doc_ref.update({
            'favorited': True
        })
        
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"Error saving earnings call analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/my_earnings_analyses')
@login_required
def my_earnings_analyses():
    """Page to view all saved earnings call analyses"""
    return render_template('my_earnings_analyses.html')

@app.route('/api/my_earnings_analyses')
@login_required
def get_my_earnings_analyses():
    """API endpoint to get user's earnings call analyses"""
    user_id = session.get('user_id')
    
    if not db:
        return jsonify({'error': 'Database not available'}), 500
    
    try:
        # Get all earnings call analyses for this user
        analyses_ref = db.collection('earnings_calls').where(
            'user_id', '==', user_id
        ).order_by('timestamp', direction=firestore.Query.DESCENDING)
        
        analyses = []
        for doc in analyses_ref.stream():
            data = doc.to_dict()
            # Convert timestamp to string if it exists
            timestamp = data.get('timestamp')
            if timestamp:
                timestamp = timestamp.timestamp() * 1000  # Convert to milliseconds for JS
            
            analyses.append({
                'id': doc.id,
                'company_name': data.get('company_name', ''),
                'themes': data.get('themes', ''),
                'timestamp': timestamp,
                'favorited': data.get('favorited', False)
            })
            
        return jsonify({'analyses': analyses})
    
    except Exception as e:
        print(f"Error retrieving analyses: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/view_earnings_analysis/<analysis_id>')
@login_required
def view_earnings_analysis(analysis_id):
    """Page to view a specific earnings call analysis with Q&A history"""
    return render_template('view_earnings_analysis.html', analysis_id=analysis_id)

@app.route('/api/earnings_analysis/<analysis_id>')
@login_required
def get_earnings_analysis(analysis_id):
    """API endpoint to get a specific earnings analysis with its conversation history"""
    user_id = session.get('user_id')
    
    if not db:
        return jsonify({'error': 'Database not available'}), 500
    
    try:
        # Get the analysis
        doc = db.collection('earnings_calls').document(analysis_id).get()
        if not doc.exists:
            return jsonify({'error': 'Analysis not found'}), 404
            
        analysis_data = doc.to_dict()
        
        # Check if user has access
        if analysis_data.get('user_id') != user_id:
            return jsonify({'error': 'Unauthorized access'}), 403
            
        # Get conversation history for this analysis
        conversations_ref = db.collection('conversation_threads').where(
            'analysis_id', '==', analysis_id
        ).order_by('timestamp')
        
        conversations = []
        for conv_doc in conversations_ref.stream():
            conv_data = conv_doc.to_dict()
            conversations.append({
                'id': conv_doc.id,
                'question': conv_data.get('question', ''),
                'answer': conv_data.get('answer', ''),
                'timestamp': conv_data.get('timestamp', '')
            })
            
        # Format timestamp for display
        timestamp = analysis_data.get('timestamp')
        formatted_time = timestamp.strftime('%B %d, %Y at %I:%M %p') if timestamp else 'Unknown date'
            
        return jsonify({
            'analysis': {
                'id': doc.id,
                'company_name': analysis_data.get('company_name', ''),
                'themes': analysis_data.get('themes', ''),
                'analysis': analysis_data.get('analysis', ''),
                'fetched_context': analysis_data.get('fetched_context', ''),
                'formatted_time': formatted_time,
                'favorited': analysis_data.get('favorited', False)
            },
            'conversations': conversations
        })
    
    except Exception as e:
        print(f"Error retrieving analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/followup_question', methods=['POST'])
@login_required
def followup_question():
    """API endpoint to handle follow-up questions on earnings call analyses"""
    user_id = session.get('user_id')
    
    data = request.json
    analysis_id = data.get('analysis_id')
    question = data.get('question')
    search_type = data.get('search_type', 'none')  # Options: 'none', 'tavily', 'perplexity', 'search1api', 'all'
    generate_answer = data.get('generate_answer', False)  # Flag to determine if we should generate answer now
    cached_search_results = data.get('cached_search_results', [])  # For second-phase calls with cached results
    
    if not analysis_id or not question:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    try:
        # Retrieve the original analysis
        if db:
            doc = db.collection('earnings_calls').document(analysis_id).get()
            if not doc.exists:
                return jsonify({'error': 'Analysis not found'}), 404
            
            analysis_data = doc.to_dict()
            
            # Check if this analysis belongs to the logged-in user
            if analysis_data.get('user_id') != user_id:
                return jsonify({'error': 'You can only ask questions about your own analyses'}), 403
            
            original_analysis = analysis_data.get('analysis', '')
            fetched_context = analysis_data.get('fetched_context', '')
            company_name = analysis_data.get('company_name', '')
            
            # Get previous conversation history
            conversations_ref = db.collection('conversation_threads').where(
                'analysis_id', '==', analysis_id
            ).order_by('timestamp')
            
            conversation_history = ""
            for conv_doc in conversations_ref.stream():
                conv_data = conv_doc.to_dict()
                conversation_history += f"Q: {conv_data.get('question', '')}\n"
                conversation_history += f"A: {conv_data.get('answer', '')}\n\n"
            
            # If we already have cached search results and want to generate answer, skip to generating answer
            if generate_answer and cached_search_results:
                search_results = cached_search_results
                additional_context = ""
                
                # Rebuild additional_context from cached results
                for result in search_results:
                    source_name = result.get('source', '')
                    url = result.get('url', '')
                    title = result.get('title', '')
                    content = result.get('full_content', result.get('snippet', ''))
                    
                    if content:
                        additional_context += f"\nSource ({source_name}): {url}\nTitle: {title}\nContent:\n{content}\n\n"
                
                # Skip to AI answer generation
                return generate_ai_answer(user_id, analysis_id, company_name, original_analysis, 
                                         fetched_context, conversation_history, additional_context, 
                                         question, search_type, search_results)
            
            # If we want to generate answer without searching (none option)
            if search_type == 'none' and generate_answer:
                search_results = []
                additional_context = ""
                return generate_ai_answer(user_id, analysis_id, company_name, original_analysis, 
                                         fetched_context, conversation_history, additional_context, 
                                         question, search_type, search_results)
            
            # Otherwise, proceed with searches based on search_type
            all_search_results = {
                'tavily': [],
                'perplexity': [],
                'search1api': [],
                'search1api_news': []
            }
            
            # Get additional context from search if requested
            additional_context = ""
            search_results = []
            
            # Tavily search for new information if requested
            if search_type in ['tavily', 'all'] and tavily_client:
                try:
                    print(f"Searching Tavily for follow-up information: {company_name} {question}")
                    tavily_query = f"{company_name} {question} earnings financial data"
                    tavily_response = tavily_client.search(
                        query=tavily_query,
                        search_depth="advanced",
                        include_raw_content=True,
                        max_results=3
                    )
                    
                    if tavily_response and tavily_response.get('results'):
                        tavily_context = []
                        for result in tavily_response['results']:
                            content = result.get('raw_content') or result.get('content')
                            if content:
                                url = result.get('url', 'Unknown')
                                title = result.get('title', 'Unknown')
                                tavily_context.append(f"Source (Tavily): {url}\nTitle: {title}\nContent:\n{content}\n")
                                search_result = {
                                    'source': 'Tavily',
                                    'url': url,
                                    'title': title,
                                    'snippet': result.get('content', '')[:200] + '...' if result.get('content') else 'No preview available',
                                    'full_content': content
                                }
                                search_results.append(search_result)
                                all_search_results['tavily'].append(search_result)
                        
                        additional_context += "\n\nAdditional Tavily Search Results:\n\n" + "\n\n".join(tavily_context)
                    
                except Exception as e:
                    print(f"Error searching Tavily: {e}")
                    additional_context += "\n\nTavily search was attempted but failed."
            
            # Perplexity search for new information if requested
            if search_type in ['perplexity', 'all'] and PERPLEXITY_API_KEY:
                try:
                    print(f"Searching Perplexity for follow-up information: {company_name} {question}")
                    perplexity_query = f"{company_name} {question} earnings financial data"
                    
                    perplexity_url = "https://api.perplexity.ai/chat/completions"
                    perplexity_headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {PERPLEXITY_API_KEY}"
                    }
                    perplexity_payload = {
                        "model": "sonar-pro",
                        "messages": [
                            {"role": "system", "content": "You are a financial analyst retrieving factual information. Return the most reliable factual information about the company's financial data. IMPORTANT: Always include your sources with URLs in a 'Sources:' section at the end of your response. List each source on a new line."},
                            {"role": "user", "content": perplexity_query}
                        ]
                    }
                    
                    perplexity_response = requests.post(perplexity_url, headers=perplexity_headers, json=perplexity_payload)
                    perplexity_data = perplexity_response.json()
                    
                    if perplexity_data and perplexity_data.get('choices'):
                        perplexity_content = perplexity_data['choices'][0]['message']['content']
                        
                        # Extract the main content and sources
                        main_content = perplexity_content
                        source_urls = []
                        
                        # Check for structured citations (top-level)
                        citations = perplexity_data.get("citations", [])
                        if not citations and perplexity_data.get("sources"):
                            citations = perplexity_data.get("sources", [])
                            
                        # If we have structured citations, use them
                        if citations and isinstance(citations, list):
                            for citation in citations:
                                if isinstance(citation, dict) and citation.get('url'):
                                    source_result = {
                                        'source': 'Perplexity',
                                        'url': citation.get('url'),
                                        'title': citation.get('title', citation.get('url')),
                                        'snippet': citation.get('text', '')[:200] + '...' if citation.get('text') else 'No preview available',
                                        'full_content': citation.get('text', '')
                                    }
                                    search_results.append(source_result)
                                    all_search_results['perplexity'].append(source_result)
                        
                        # If no structured citations, try to parse from text
                        elif "Sources:" in perplexity_content:
                            parts = perplexity_content.split("Sources:", 1)
                            main_content = parts[0].strip()
                            sources_text = parts[1].strip()
                            
                            # Extract URLs using regex
                            import re
                            urls = re.findall(r'https?://[^\s\)]+', sources_text)
                            
                            # For each URL, extract the surrounding text as a title
                            lines = sources_text.split('\n')
                            for line in lines:
                                line = line.strip()
                                if not line:
                                    continue
                                    
                                # Find URLs in this line
                                line_urls = re.findall(r'https?://[^\s\)]+', line)
                                if line_urls:
                                    url = line_urls[0]
                                    # Use text before URL as title, or the whole line if no other text
                                    title = line.replace(url, '').strip() or url
                                    source_result = {
                                        'source': 'Perplexity',
                                        'url': url,
                                        'title': title,
                                        'snippet': 'Source from Perplexity search',
                                        'full_content': main_content
                                    }
                                    search_results.append(source_result)
                                    all_search_results['perplexity'].append(source_result)
                                elif line and not any(url in line for url in urls):
                                    # This might be a source without URL
                                    source_result = {
                                        'source': 'Perplexity',
                                        'url': "#",
                                        'title': line,
                                        'snippet': 'Source from Perplexity search',
                                        'full_content': main_content
                                    }
                                    search_results.append(source_result)
                                    all_search_results['perplexity'].append(source_result)
                        
                        # If still no sources, add the content as a single result
                        if not all_search_results['perplexity']:
                            source_result = {
                                'source': 'Perplexity',
                                'url': "#",
                                'title': "Perplexity Research",
                                'snippet': main_content[:200] + '...' if len(main_content) > 200 else main_content,
                                'full_content': main_content
                            }
                            search_results.append(source_result)
                            all_search_results['perplexity'].append(source_result)
                        
                        # Add the content to additional_context
                        additional_context += f"\n\nAdditional Information from Perplexity:\n\n{main_content}\n"
                        
                    else:
                        print(f"Error: Invalid response from Perplexity API")
                        
                except Exception as e:
                    print(f"Error searching Perplexity: {e}")
                    additional_context += "\n\nPerplexity search was attempted but failed."
            
            # Search1API search for new information if requested
            if search_type in ['search1api', 'all'] and SEARCH1API_KEY:
                try:
                    print(f"Searching Search1API for follow-up information: {company_name} {question}")
                    search1api_query = f"{company_name} {question} earnings financial data"
                    
                    # Use Search1API's search endpoint
                    search1api_url = "https://api.search1api.com/search"
                    search1api_headers = {
                        "Authorization": f"Bearer {SEARCH1API_KEY}",
                        "Content-Type": "application/json"
                    }
                    search1api_payload = {
                        "query": search1api_query,
                        "search_service": "google",
                        "max_results": 3,
                        "crawl_results": 1,  # Crawl results to get content
                        "image": False,
                        "language": "en",
                        "time_range": "year"  # Use year for more comprehensive results
                    }
                    
                    search1api_response = requests.post(search1api_url, headers=search1api_headers, json=search1api_payload)
                    search1api_data = search1api_response.json()
                    
                    # Process search results
                    if search1api_data and "results" in search1api_data:
                        search1api_context = []
                        for result in search1api_data["results"]:
                            content = result.get('content') or result.get('snippet', '')
                            if content:
                                url = result.get('link', 'Unknown')
                                title = result.get('title', 'Unknown')
                                search1api_context.append(f"Source (Search1API): {url}\nTitle: {title}\nContent:\n{content}\n")
                                source_result = {
                                    'source': 'Search1API',
                                    'url': url,
                                    'title': title,
                                    'snippet': content[:200] + '...' if len(content) > 200 else content,
                                    'full_content': content
                                }
                                search_results.append(source_result)
                                all_search_results['search1api'].append(source_result)
                        
                        if search1api_context:
                            additional_context += "\n\nAdditional Search1API Results:\n\n" + "\n\n".join(search1api_context)
                    
                    # Also try the news endpoint for timely information
                    news_url = "https://api.search1api.com/news"
                    news_payload = {
                        "query": search1api_query,
                        "search_service": "google",
                        "max_results": 2,
                        "crawl_results": 1,
                        "image": False,
                        "language": "en",
                        "time_range": "month"  # More recent for news
                    }
                    
                    news_response = requests.post(news_url, headers=search1api_headers, json=news_payload)
                    news_data = news_response.json()
                    
                    # Process news results
                    if news_data and "results" in news_data:
                        news_context = []
                        for result in news_data["results"]:
                            content = result.get('content') or result.get('snippet', '')
                            if content:
                                url = result.get('link', 'Unknown')
                                title = result.get('title', 'Unknown')
                                news_context.append(f"Source (Search1API News): {url}\nTitle: {title}\nContent:\n{content}\n")
                                source_result = {
                                    'source': 'Search1API News',
                                    'url': url,
                                    'title': title,
                                    'snippet': content[:200] + '...' if len(content) > 200 else content,
                                    'full_content': content
                                }
                                search_results.append(source_result)
                                all_search_results['search1api_news'].append(source_result)
                        
                        if news_context:
                            additional_context += "\n\nAdditional Search1API News Results:\n\n" + "\n\n".join(news_context)
                    
                except Exception as e:
                    print(f"Error searching Search1API: {e}")
                    additional_context += "\n\nSearch1API search was attempted but failed."
            
            # If we're just doing the search phase, return the search results
            if not generate_answer and search_type != 'none':
                return jsonify({
                    'search_results': search_results,
                    'grouped_results': all_search_results,
                    'search_type': search_type,
                    'question': question,
                    'company_name': company_name,
                    'analysis_id': analysis_id
                })
            
            # Otherwise, generate the AI answer
            return generate_ai_answer(user_id, analysis_id, company_name, original_analysis, 
                                     fetched_context, conversation_history, additional_context, 
                                     question, search_type, search_results)
                
        else:
            return jsonify({'error': 'Database not available'}), 500
            
    except Exception as e:
        print(f"Error processing follow-up question: {e}")
        return jsonify({'error': str(e)}), 500

def generate_ai_answer(user_id, analysis_id, company_name, original_analysis, fetched_context, 
                      conversation_history, additional_context, question, search_type, search_results):
    """Helper function to generate AI answer using search results"""
    try:
        # Create the prompt for the follow-up
        if search_type == 'none':
            system_prompt = "You are a financial analyst helping with earnings call follow-up questions. Use only the information from the original transcript and analysis to answer."
        else:
            system_prompt = "You are a financial analyst helping with earnings call follow-up questions. You must cite your sources for factual information. When you mention a specific fact that comes from the search results, include a numbered citation like [1], [2], etc., and list the complete sources with their URLs at the end of your response. Each source should be clearly numbered to match your citations."
        
        user_prompt = f"""
        I previously asked for an analysis of {company_name}'s earnings call, and you provided the following analysis:
        
        {original_analysis}
        
        The original transcript information was:
        
        {fetched_context}
        
        Previous questions and answers about this analysis:
        {conversation_history}
        """
        
        if additional_context:
            user_prompt += f"""
            I've performed additional research to help answer this question and found:
            {additional_context}
            """
        
        user_prompt += f"""
        Based on all available information, please answer this follow-up question:
        {question}
        
        IMPORTANT FORMATTING INSTRUCTIONS:
        1. If information is available in the additional search results but wasn't in the original transcript, clearly indicate this new information in your answer.
        2. For each factual statement, especially numbers, statistics, or specific details from search results, include a numbered citation like [1].
        3. At the end of your answer, include a "Sources:" section that lists all the references you used, numbered to match your citations.
        4. For each source, include the full URL so users can verify the information.
        
        If you can't find the information to answer this question, please state that clearly.
        """
        
        # Use Claude for follow-up (similar to our main analysis)
        if os.environ.get("ANTHROPIC_API_KEY"):
            anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            
            response = anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=2000,
                temperature=0.3, # Lower temperature for more factual responses
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            answer = response.content[0].text
        else:
            # Fallback to OpenAI if Claude is not available
            if OPENAI_API_KEY:
                openai_client = OpenAI(api_key=OPENAI_API_KEY)
                completion = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3
                )
                answer = completion.choices[0].message.content
            else:
                return jsonify({'error': 'No AI service available'}), 500
        
        # Save the conversation history
        db.collection('conversation_threads').document().set({
            'analysis_id': analysis_id,
            'user_id': user_id,
            'question': question,
            'answer': answer,
            'search_type': search_type,
            'search_results': search_results,
            'timestamp': firestore.SERVER_TIMESTAMP
        })
        
        return jsonify({
            'answer': answer,
            'search_results': search_results,
            'search_type': search_type
        })
        
    except Exception as e:
        print(f"Error generating AI answer: {e}")
        return jsonify({'error': str(e)}), 500

# This standard Python construct checks if the script is being run directly.
if __name__ == '__main__':
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Run the Company Metric Finder web application')
    parser.add_argument('--port', type=int, default=8082, help='Port to run the server on (default: 8082)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on (default: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # For local development use debug mode and specified port
    # In production, the port will be provided by the hosting environment
    port = int(os.environ.get("PORT", args.port))
    host = os.environ.get("HOST", args.host)
    
    # Only use debug mode in local development if not explicitly set
    if args.debug:
        debug = True
    else:
        debug = os.environ.get('VERCEL_ENV') is None and os.environ.get('GAE_ENV') is None
    
    print(f"Starting server on {host}:{port} (debug={debug})")
    try:
        app.run(host=host, port=port, debug=debug)
    except OSError as e:
        if 'Address already in use' in str(e):
            print(f"Error: Port {port} is already in use.")
            print(f"Try running with a different port: python app.py --port=8083")
        else:
            print(f"Error starting server: {e}")
        sys.exit(1) 