# PartsSearch

A Flask application for market research and earnings call analysis.

## Project Structure

The application uses a modular structure with Flask Blueprints:

```
├── app/                  # Main application package
│   ├── __init__.py       # App factory
│   ├── auth/             # Authentication routes and helpers
│   ├── market_research/  # Market research routes
│   ├── earnings/         # Earnings call analysis routes
│   ├── utils/            # Utility functions
│   └── config/           # Configuration modules
├── templates/            # HTML templates
├── main.py               # Application entry point
├── app.yaml              # Google App Engine configuration
└── vercel.json           # Vercel configuration
```

## Setup and Installation

1. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up environment variables in a `.env` file:
   ```
   TAVILY_API_KEY=your_key_here
   OPENAI_API_KEY=your_key_here
   GEMINI_API_KEY=your_key_here
   PERPLEXITY_API_KEY=your_key_here
   SEARCH1API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   FLASK_SECRET_KEY=your_secret_key_here
   ```

3. (Optional) For Firebase functionality, you need to provide Firebase credentials either through environment variables or a `firebase-credentials.json` file.

## Running the Application

To run the application locally:

```
python main.py
```

The application will be available at http://localhost:8082 by default.

### Command-line Arguments

- `--port`: Specify the port to run on (default: 8082)
- `--host`: Specify the host to bind to (default: 0.0.0.0)
- `--debug`: Run in debug mode

Example:
```
python main.py --port=8000 --debug
```

## Frontend Development

A Next.js frontend is available in the `frontend/` directory. To run it:

```
cd frontend
npm install
npm run dev
```

The frontend will be available at http://localhost:3000 by default.

## Deployment

The application is configured for deployment to:

- Google App Engine (via app.yaml)
- Vercel (via vercel.json)

## Features

- Market research with Tavily search integration
- Earnings call analysis using AI
- User authentication with Firebase
- Search and save functionality 