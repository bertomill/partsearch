# app.py
# Import the Flask library. Think of this like getting a toolkit for building web apps.
from flask import Flask, render_template, request, redirect, url_for, jsonify
# Import the os module to access environment variables
import os
# Import the Tavily client
from tavily import TavilyClient
# Import the function to load .env files
from dotenv import load_dotenv
# Import re for pattern matching
import re
# Import datetime for date handling
from datetime import datetime
# Import OpenAI for advanced analysis
from openai import OpenAI
# Import Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, firestore
# Import for generating unique IDs
import uuid
# Import Firebase config
from firebase_config import FIREBASE_CONFIG

# Load environment variables from .env file *before* accessing them
load_dotenv()

# Create an instance of the Flask class. This is our actual web application object.
# __name__ tells Flask where to look for resources like templates.
app = Flask(__name__)

# Get API keys from environment variables
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Initialize Firebase Admin
try:
    # Initialize Firebase App if not already initialized
    if not firebase_admin._apps:
        # Check if the service account file exists
        cred_path = 'firebase-credentials.json'
        # Check for credentials in environment variable (for Vercel)
        firebase_creds_json = os.environ.get('FIREBASE_CREDENTIALS')
        
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
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            print("Firebase initialized with service account credentials")
        else:
            # For deployment: Use a certificate if provided, otherwise try application default credentials
            # For local development: Fall back to using just the project ID
            try:
                # Try to initialize with application default credentials first
                firebase_admin.initialize_app(options={
                    'projectId': FIREBASE_CONFIG['projectId'],
                })
                print("Firebase initialized with application default credentials")
            except Exception as credential_error:
                print(f"Falling back to unauthenticated mode: {credential_error}")
                # Fall back to a minimal configuration (limited capabilities)
                firebase_admin.initialize_app(options={
                    'projectId': FIREBASE_CONFIG['projectId'],
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
            # Initialize the Tavily client inside the request handler
            # using the key we fetched earlier.
            tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

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

# Route to save search results
@app.route('/save_search', methods=['POST'])
def save_search():
    if not db:
        return jsonify({"success": False, "error": "Firebase not initialized"}), 500
    
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

# Route to view saved searches
@app.route('/saved_searches', methods=['GET'])
def view_saved_searches():
    if not db:
        return render_template('saved_searches.html', 
                              error="Firebase not initialized", 
                              searches=[])
    
    try:
        # Get all saved searches from Firestore, ordered by timestamp (newest first)
        searches_ref = db.collection('saved_searches').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(50)
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

# This standard Python construct checks if the script is being run directly.
if __name__ == '__main__':
    # For local development use debug mode and a specific port
    # In production, the port will be provided by the hosting environment
    port = int(os.environ.get("PORT", 8081))
    # Only use debug mode in local development
    is_dev = os.environ.get('VERCEL_ENV') is None and os.environ.get('GAE_ENV') is None
    app.run(host='0.0.0.0', port=port, debug=is_dev) 