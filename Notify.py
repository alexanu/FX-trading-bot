import requests
import json
import sys

def notify_from_line(message, image=None):
	url = 'https://notify-api.line.me/api/notify'
	with open(sys.argv[1]) as f:
		auth_tokens = json.load(f)
		token = auth_tokens['line_token']
		print(f'<bot> Read token from {sys.argv[1]}')

	headers = {
		'Authorization' : 'Bearer {}'.format(token)
	}

	payload = {
		'message' :  message
	}
	if image is not None:
		try:
			files = {
				'imageFile': open(image, "rb")
			}
			response = requests.post(url ,headers=headers ,data=payload, files=files)
			return response
		except:
			pass

	else:
		try:
			response = requests.post(url ,headers=headers ,data=payload)
			return response
		except:
			pass