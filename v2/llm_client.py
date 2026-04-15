import requests

class LLMClient:
    def __init__(self, server_url="http://127.0.0.1:8000/query", session_id="default", thinking=False):
        self.server_url = server_url
        self.session = requests.Session()
        self.thinking = thinking

    def query(self, prompt, max_new_tokens=64, session_id = "default"):
        self.session_id = session_id
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "session_id": session_id,
            "thinking": self.thinking
        }
        resp = self.session.post(self.server_url, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()["response"]