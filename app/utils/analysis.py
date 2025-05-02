import re
from app.config.api_keys import OPENAI_API_KEY
from openai import OpenAI

def ai_analyze_results(results, company, metric):
    """Function that uses OpenAI to analyze search results"""
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

def synthesize_answer(results, company, metric):
    """Function to synthesize a "smart" answer from multiple search results (used as fallback)"""
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