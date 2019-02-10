import json

def from_byte_to_dict(byte_line):
	"""
	byte型の文字列をdict型に変換する

	Parameters
	----------
	byte_line: byte
		byte型の文字列
	"""

	try:
		return json.loads(byte_line.decode("UTF-8"))
	except Exception as e:
		print("Caught exception when converting message into json : {}" .format(str(e)))
		return None

def from_response_to_dict(response):
	try:
		return json.loads(response.content.decode('UTF-8'))
	except Exception as e:
		print("Caught exception when converting message into json : {}" .format(str(e)))
		return None