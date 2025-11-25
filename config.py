import os
from dotenv import load_dotenv

# Load environment variables
#load_dotenv(dotenv_path=".env")

# Azure OpenAI configuration
#AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
#AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
#AZURE_DEPLOYMENT = "VARELab-GPT4o"
#AZURE_API_VERSION = "2024-08-01-preview"
#TEMPERATURE = 0.5

# Processing configuration
#BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
#REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))  # seconds
#RETRY_LIMIT = int(os.getenv("RETRY_LIMIT", "2"))
#RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))

# Output configuration
#OUTPUT_DIR = os.getenv("OUTPUT_DIR", "metadata_gen_output")
#EVALUATION_DIR = os.getenv("EVALUATION_DIR", "evaluation")


# Load environment variables
load_dotenv(dotenv_path=".env")

# Google Gemini configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5"))

# Processing configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
RETRY_LIMIT = int(os.getenv("RETRY_LIMIT", "3"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))

# Output configuration
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "metadata_gen_output")
EVALUATION_DIR = os.getenv("EVALUATION_DIR", "evaluation")