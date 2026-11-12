import sys
import json
from parser import Parser

try:
    from pydantic import ValidationError
except ImportError:
    print("❌ [ERROR] Missing dependency: pydantic")
    print("Install it with: pip install pydantic")
    sys.exit(1)

""" try:
    from .llm_sdk import Llm
except ImportError:
    print("❌ [ERROR] Failed to import Small_LLM_Model.")
    print("Make sure llm_sdk and its dependencies are installed:")
    print(" - numpy")
    print(" - pydantic")
    print(" - torch")
    print(" - transformers")
    print(" - huggingface-hub")
    sys.exit(1) """

def main():
    parser = Parser()
    try:
        parser.parse()
    except ValueError:
        print("Files corrupted")



if __name__ =="__main__":
    main()