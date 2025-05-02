from flask import Blueprint, render_template, request, jsonify, session
import firebase_admin
from firebase_admin import firestore
import uuid
import os
import tempfile
import requests
import json

from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic, APIError
from tavily import TavilyClient

from app.config.api_keys import (
    TAVILY_API_KEY, 
    OPENAI_API_KEY, 
    GEMINI_API_KEY, 
    PERPLEXITY_API_KEY, 
    SEARCH1API_KEY,
    ANTHROPIC_API_KEY
)
from app.auth.routes import login_required
from app import db

earnings_bp = Blueprint('earnings', __name__, url_prefix='')

# Initialize clients
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API initialized")

@earnings_bp.route('/earnings_call')
def earnings_call():
    """Route for earnings call analysis page"""
    return render_template('earnings_call.html')

@earnings_bp.route('/analyze_earnings_call', methods=['POST'])
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
        if ANTHROPIC_API_KEY:
            print(f"Using Claude for {company_name} earnings call analysis (large transcript)...")
            
            try:
                # Initialize Anthropic client
                anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
                
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
                    return jsonify({'error': f'Error using Claude API and OpenAI not available as fallback: {str(e)}'}), 500
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

@earnings_bp.route('/save_earnings_call/<search_id>', methods=['POST'])
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

@earnings_bp.route('/my_earnings_analyses')
@login_required
def my_earnings_analyses():
    """Page to view all saved earnings call analyses"""
    return render_template('my_earnings_analyses.html')

@earnings_bp.route('/api/my_earnings_analyses')
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

@earnings_bp.route('/view_earnings_analysis/<analysis_id>')
@login_required
def view_earnings_analysis(analysis_id):
    """Page to view a specific earnings call analysis with Q&A history"""
    return render_template('view_earnings_analysis.html', analysis_id=analysis_id)

@earnings_bp.route('/api/earnings_analysis/<analysis_id>')
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

@earnings_bp.route('/followup_question', methods=['POST'])
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
        if ANTHROPIC_API_KEY:
            anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
            
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