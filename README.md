# AI Text Evaluator

A FastAPI-based service for evaluating AI-generated text across multiple dimensions including toxicity, PII, bias, and hallucinations.

![swagger_ui](https://github.com/AOGbadamosi2018/AItext_evaluator/blob/main/swagger_ui.png))

## Features

- **Toxicity Detection**: Identifies harmful, offensive, or inappropriate content
- **PII Detection**: Finds personally identifiable information
- **Bias Detection**: Detects potential biases in the text
- **Hallucination Detection**: Identifies potential factual inaccuracies
- **Safety Scoring**: Provides an overall safety score (0-100)

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- (Optional) virtualenv or conda for virtual environment

## Setup

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <repository-url>
   cd ai-text-evaluator
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root with the following variables:
   ```
   HOST=0.0.0.0
   PORT=8000
   DEBUG=True
   DATABASE_URL=sqlite:///./ai_text_evaluator.db
   HUGGINGFACE_HUB_TOKEN=your_huggingface_token_here  # Optional but recommended
   ```

5. **Initialize the database**:
   ```bash
   python -m app.db.init_db
   ```

## Starting the Service

To start the FastAPI development server:

```bash
uvicorn app.main:app --reload
```

The service will start and be available at `http://localhost:8000`

## API Documentation

Once the service is running, you can access:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Example API Requests

### Evaluate Text
```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/evaluations/evaluate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog.",
    "context": "This is a sample context for hallucination detection.",
    "evaluations": ["toxicity", "pii", "bias", "hallucination"]
  }'
```

### Health Check
```bash
curl -X 'GET' \
  'http://localhost:8000/api/v1/evaluations/health' \
  -H 'accept: application/json'
```

## Project Structure

```
ai-text-evaluator/
├── app/
│   ├── api/                  # API endpoints and routes
│   ├── core/                 # Core configuration
│   ├── db/                   # Database configuration
│   ├── models/               # Database models
│   ├── schemas/              # Pydantic schemas
│   └── services/             # Evaluation services
├── tests/                   # Test files
├── .env                     # Environment variables
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

## Running Tests

To run the test suite:

```bash
pytest
```

## Using Docker (Alternative)

If you prefer using Docker:

1. Build the Docker image:
   ```bash
   docker build -t ai-text-evaluator .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 ai-text-evaluator
   ```

## Next Steps

- Add authentication for production use
- Implement rate limiting
- Add more evaluation metrics
- Set up monitoring and logging
- Create a frontend interface

## License

[Your License Here]
