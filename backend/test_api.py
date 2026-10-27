import urllib.request
import json
import os

# Test health and status endpoints
req = urllib.request.urlopen("http://127.0.0.1:8000/api/status")
status_data = json.loads(req.read().decode())
print("API Status Response:")
print(json.dumps(status_data, indent=2))
