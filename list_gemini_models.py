from config import GEMINI_API_KEY
import google.generativeai as genai

genai.configure(api_key=GEMINI_API_KEY)
models = genai.list_models()
for m in models:
    if 'gemini' in m.name.lower():
        print(m.name, getattr(m, 'supported_generation_methods', None))
