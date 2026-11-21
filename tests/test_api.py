"""Sends test requests to the running API (start it first: python -m uvicorn src.app:app)."""
import requests

url = "http://127.0.0.1:8000/predict"

for test in ["google.com",
             "paypal-login-secure.xyz",
             "sp0tify.tk"]:
    response = requests.post(url, json={"url": test})
    print(test, "->", response.json())
