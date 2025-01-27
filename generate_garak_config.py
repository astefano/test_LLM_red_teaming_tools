import json
import os

def generate_garak_config():
    # Get API key from environment
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    # Define the configuration
    config = {
        "rest": {
            "RestGenerator": {
                "name": "Test Gemini",
                "uri": f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
                "method": "post",
                "req_template_json_object": {
                    "contents": [{
                        "parts": [{"text": "$INPUT"}]
                    }]
                }
            }
        }
    }

    # Write to file
    with open('gemini_rest_garak.json', 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    generate_garak_config()
