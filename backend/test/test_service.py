import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import base64
import json

from fastapi import HTTPException

# We need to add the parent directory to the path to import the app modules
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import the service
from app.service import InferenceService, HumanMessage

# --- Test Data ---
FAKE_IMAGE_BYTES = b"this is a fake image"
FAKE_ANALYSIS_RESULT = {
    "food_item": "Test Food",
    "calories": 100,
    "ingredients": {"fat": "10g"}
}
FAKE_ANALYSIS_JSON_STRING = json.dumps(FAKE_ANALYSIS_RESULT)


# We use IsolatedAsyncioTestCase for testing async methods
class TestInferenceService(unittest.IsolatedAsyncioTestCase):

    @patch('app.service.ChatOpenAI')
    @patch('app.service.ChatOllama')
    def setUp(self, mock_ollama, mock_chat_openai):
        """Set up the test environment before each test."""
        # Instantiate the service. The mocks prevent any real connections.
        self.service = InferenceService()
        self.test_image_bytes = FAKE_IMAGE_BYTES

    async def test_invoke_model_success(self):
        """
        Test that _invoke_model correctly calls the LLM and returns content.
        """
        # Create a mock LLM object with an async ainvoke method
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock()

        # Configure the mock response
        mock_response = MagicMock()
        mock_response.content = "Test response content"
        mock_llm.ainvoke.return_value = mock_response

        # Call the method under test
        result = await self.service._invoke_model(mock_llm, self.test_image_bytes)

        # Assertions
        self.assertEqual(result, "Test response content")

        # Check that ainvoke was called once
        mock_llm.ainvoke.assert_called_once()

        # Verify the structure of the message passed to ainvoke
        call_args = mock_llm.ainvoke.call_args[0][0]
        self.assertEqual(len(call_args), 1)
        message = call_args[0]
        self.assertIsInstance(message, HumanMessage)

        # Check the content of the message
        content_list = message.content
        self.assertEqual(len(content_list), 2)
        self.assertEqual(content_list[0]['type'], 'text')
        self.assertEqual(content_list[1]['type'], 'image_url')

        # Check the image encoding
        encoded_image = base64.b64encode(self.test_image_bytes).decode('utf-8')
        expected_image_url = f"data:image/jpeg;base64,{encoded_image}"
        self.assertEqual(content_list[1]['image_url']['url'], expected_image_url)

    async def test_invoke_model_raises_http_exception_on_error(self):
        """
        Test that _invoke_model catches exceptions and raises an HTTPException.
        """
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM provider is down"))

        # Use async with for checking exceptions in async code
        with self.assertRaises(HTTPException) as context:
            await self.service._invoke_model(mock_llm, self.test_image_bytes)

        self.assertEqual(context.exception.status_code, 503)
        self.assertIn("Inference service error", str(context.exception.detail))

    @patch('app.service.InferenceService._invoke_model', new_callable=AsyncMock)
    async def test_analyze_image_success_clean_json(self, mock_invoke_model):
        """
        Test analyze_image with a perfect JSON string response.
        """
        mock_invoke_model.return_value = FAKE_ANALYSIS_JSON_STRING

        result = await self.service.analyze_image(self.test_image_bytes)

        mock_invoke_model.assert_called_once_with(self.service.ollama_llm, self.test_image_bytes)
        self.assertEqual(result, FAKE_ANALYSIS_RESULT)

    @patch('app.service.InferenceService._invoke_model', new_callable=AsyncMock)
    async def test_analyze_image_handles_json_with_markdown(self, mock_invoke_model):
        """
        Test that the robust parsing handles JSON wrapped in markdown and text.
        """
        messy_response = f"Sure, here is the analysis:\n```json\n{FAKE_ANALYSIS_JSON_STRING}\n```\nI hope this helps!"
        mock_invoke_model.return_value = messy_response

        result = await self.service.analyze_image(self.test_image_bytes)

        self.assertEqual(result, FAKE_ANALYSIS_RESULT)

    @patch('app.service.InferenceService._invoke_model', new_callable=AsyncMock)
    async def test_analyze_image_handles_no_json_in_response(self, mock_invoke_model):
        """
        Test analyze_image when the response does not contain a JSON object.
        """
        raw_response = "I'm sorry, I cannot analyze this image."
        mock_invoke_model.return_value = raw_response

        result = await self.service.analyze_image(self.test_image_bytes)

        self.assertIn("error", result)
        self.assertEqual(result["error"], "Failed to parse analysis from the model.")
        self.assertEqual(result["raw_response"], raw_response)

    @patch('app.service.InferenceService._invoke_model', new_callable=AsyncMock)
    async def test_analyze_image_handles_malformed_json(self, mock_invoke_model):
        """
        Test analyze_image with a broken JSON string.
        """
        malformed_json = '{"food_item": "Test Food", "calories": 100,}' # Trailing comma
        mock_invoke_model.return_value = malformed_json

        result = await self.service.analyze_image(self.test_image_bytes)

        self.assertIn("error", result)
        self.assertEqual(result["error"], "Failed to parse analysis from the model.")
        self.assertEqual(result["raw_response"], malformed_json)

    @patch('app.service.InferenceService._invoke_model', new_callable=AsyncMock)
    async def test_analyze_image_propagates_http_exception(self, mock_invoke_model):
        """
        Test that exceptions from _invoke_model are correctly propagated.
        """
        mock_invoke_model.side_effect = HTTPException(status_code=500, detail="Test exception")

        with self.assertRaises(HTTPException) as context:
            await self.service.analyze_image(self.test_image_bytes)

        self.assertEqual(context.exception.status_code, 500)
        self.assertEqual(context.exception.detail, "Test exception")


if __name__ == '__main__':
    unittest.main()
