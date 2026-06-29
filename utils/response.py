from flask import jsonify

def success_response(data: dict, status: int = 200):
    return jsonify({"success": True, "data": data}), status

def error_response(message: str, status: int = 400):
    return jsonify({"success": False, "error": message}), status
