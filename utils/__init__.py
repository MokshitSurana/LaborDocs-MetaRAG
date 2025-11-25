# utils package
from .logger import setup_logger
from .pdf_utils import pdf_to_text

# from gpu_utils import GPUVerifier

# Initialize GPU verification
'''
# Removing gpu_verifier since:
a) logging and PDF reading don't need GPU, i.e. no file uses GPU from utils
b) main scripts (chunks.py, metadata_gen.py, embeddings.py) have their own GPU verification
'''
# gpu_verifier = GPUVerifier(require_gpu=True)

__all__ = ['setup_logger', 'pdf_to_text']