import os
import sys
import argparse
from app import create_app

# Create the Flask application
app = create_app()

if __name__ == '__main__':
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
            print(f"Try running with a different port: python main.py --port=8083")
        else:
            print(f"Error starting server: {e}")
        sys.exit(1) 