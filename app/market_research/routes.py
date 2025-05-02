from flask import Blueprint, render_template, request, jsonify, session
from tavily import TavilyClient
import firebase_admin
from firebase_admin import firestore
import uuid
import os

from app.config.api_keys import TAVILY_API_KEY
from app.utils.analysis import ai_analyze_results
from app.auth.routes import login_required
from app import db

market_research_bp = Blueprint('market_research', __name__, url_prefix='')

# Initialize Tavily Client globally
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

# Define a route for the homepage
@market_research_bp.route('/', methods=['GET', 'POST'])
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

# Route for saved searches
@market_research_bp.route('/saved_searches', methods=['GET'])
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

# Route to save a search
@market_research_bp.route('/save_search', methods=['POST'])
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
@market_research_bp.route('/delete_search/<search_id>', methods=['POST'])
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