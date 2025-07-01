
# this is comment added to test if my pyton code is working or not for git push
# also removing my api key which i accidently pushed to public repo

import requests
import json

api_key = "Your API Key"  # remove this line and replace with your api key
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

headers = {
    "Content-Type": "application/json"
}

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": user_input}
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        data = response.json()
        try:
            output = data["candidates"][0]["content"]["parts"][0]["text"]
            print("Gemini:", output)
        except KeyError:
            print("Unexpected response format:", data)
    else:
        print(f"Error {response.status_code}:", response.text)
