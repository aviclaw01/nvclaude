import importlib.util
import io
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError


spec = importlib.util.spec_from_file_location("nvclaude", "nvclaude.py")
nvclaude = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nvclaude)


class TestKeyValidation(unittest.TestCase):
    def test_rejects_non_nvidia_key_without_network_call(self):
        with patch.object(nvclaude, "http") as request:
            self.assertFalse(nvclaude.validate_api_key("wrong"))
        request.assert_not_called()

    def test_accepts_authenticated_response(self):
        response = io.BytesIO(b'{"data": []}')
        with patch.object(nvclaude, "http", return_value=response) as request:
            self.assertTrue(nvclaude.validate_api_key("nvapi-valid"))
        request.assert_called_once_with(
            nvclaude.UPSTREAM + "/models",
            headers={"Authorization": "Bearer nvapi-valid"},
            timeout=10,
        )

    def test_rejects_authentication_failure(self):
        error = HTTPError("https://example.test", 401, "unauthorized", {}, None)
        with patch.object(nvclaude, "http", side_effect=error):
            self.assertFalse(nvclaude.validate_api_key("nvapi-invalid"))

    def test_allows_network_failure(self):
        with patch.object(nvclaude, "http", side_effect=URLError("offline")):
            self.assertTrue(nvclaude.validate_api_key("nvapi-offline"))


if __name__ == "__main__":
    unittest.main()
