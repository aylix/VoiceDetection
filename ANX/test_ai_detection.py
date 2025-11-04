# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import unittest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the processor path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from processor import AIDetector


class TestAIDetectionProcessor(unittest.TestCase):
    """Test cases for AI Detection Processor."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        
        # Create mock configuration
        self.mock_config = Mock()
        self.mock_config.builder = Mock()
        self.mock_config.getOption = Mock()
        
        # Set up default config values
        config_defaults = {
            "voice.detection.enabled": True,
            "voice.detection.model": "wavefake",
            "voice.detection.threshold": 0.7,
            "voice.detection.sample_rate": 16000,
            "text.detection.enabled": True,
            "text.detection.model": "ollama",
            "text.detection.threshold": 0.6,
            "speech_to_text.model": "whisper-base",
            "speech_to_text.language": "auto",
            "ollama.uri": "http://ai01.server.trovent:11434",
            "ollama.model": "llama3.2",
            "output.detailed_results": True,
            "output.confidence_scores": True,
            "input.audio_field": "audio_data",
            "input.text_field": "text_content",
            "performance.batch_size": 1,
            "performance.max_audio_length_seconds": 300,
        }
        
        self.mock_config.getOption.side_effect = lambda key: config_defaults.get(key)
        
        # Create processor instance
        self.processor = AIDetector()
        self.processor.config = self.mock_config
        self.processor.logger = Mock()

    def test_prepare_config_schema(self):
        """Test configuration schema preparation."""
        
        with patch.object(self.processor, 'config'):
            self.processor.prepareConfigSchema()
            
        # Verify that addOption was called for key configuration items
        self.assertTrue(self.mock_config.builder.addOption.called)

    @patch('processor.whisper')
    @patch('processor.torch')
    def test_prepare_initialization(self, mock_torch, mock_whisper):
        """Test processor preparation and model initialization."""
        
        # Mock model loading
        mock_whisper.load_model.return_value = Mock()
        mock_torch.cuda.is_available.return_value = False
        
        # Mock the initialize methods
        with patch.object(self.processor, '_initialize_models'):
            self.processor.prepare()
            
        # Verify config values are set
        self.assertEqual(self.processor.voice_detection_enabled, True)
        self.assertEqual(self.processor.voice_model_name, "wavefake")
        self.assertEqual(self.processor.text_detection_enabled, True)

    def test_extract_audio_data_direct(self):
        """Test audio data extraction from document."""
        
        # Test direct audio data
        doc = {"audio_data": b"fake_audio_bytes"}
        
        result = self.processor._extract_audio_data(doc)
        
        self.assertEqual(result, b"fake_audio_bytes")

    def test_extract_text_data(self):
        """Test text data extraction from document."""
        
        # Test direct text data
        doc = {"text_content": "Sample text for analysis"}
        
        result = self.processor._extract_text_data(doc)
        
        self.assertEqual(result, "Sample text for analysis")

    def test_extract_text_data_alternative_fields(self):
        """Test text extraction from alternative field names."""
        
        doc = {"transcript": "Alternative field text"}
        
        result = self.processor._extract_text_data(doc)
        
        self.assertEqual(result, "Alternative field text")

    @patch('processor.requests.post')
    def test_analyze_text_ollama_success(self, mock_post):
        """Test successful Ollama text analysis."""
        
        # Mock successful Ollama response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": json.dumps({
                "confidence": 0.8,
                "reasoning": "Text shows patterns typical of AI generation",
                "indicators": ["repetitive phrasing", "formal structure"]
            })
        }
        mock_post.return_value = mock_response
        
        # Set up processor config
        self.processor.ollama_uri = "http://test:11434"
        self.processor.ollama_model = "test-model"
        
        result = self.processor._analyze_text_ollama("Test text for analysis")
        
        self.assertAlmostEqual(result["confidence"], 0.8)
        self.assertIn("reasoning", result["features"])
        mock_post.assert_called_once()

    @patch('processor.requests.post')
    def test_analyze_text_ollama_failure(self, mock_post):
        """Test Ollama text analysis with API failure."""
        
        # Mock failed Ollama response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        self.processor.ollama_uri = "http://test:11434"
        self.processor.ollama_model = "test-model"
        
        result = self.processor._analyze_text_ollama("Test text")
        
        self.assertEqual(result["confidence"], 0.0)
        self.assertIn("error", result)

    def test_perform_hybrid_analysis_fully_ai(self):
        """Test hybrid analysis for fully AI-generated content."""
        
        voice_result = {"confidence": 0.9, "is_ai_generated": True}
        text_result = {"confidence": 0.8, "is_ai_generated": True}
        
        self.processor.voice_threshold = 0.7
        self.processor.text_threshold = 0.6
        
        result = self.processor._perform_hybrid_analysis(voice_result, text_result)
        
        self.assertEqual(result["scenario"], "fully_ai_generated")
        self.assertAlmostEqual(result["confidence"], 0.85)  # Average of 0.9 and 0.8

    def test_perform_hybrid_analysis_voice_cloning(self):
        """Test hybrid analysis for voice cloning scenario."""
        
        voice_result = {"confidence": 0.9, "is_ai_generated": True}
        text_result = {"confidence": 0.3, "is_ai_generated": False}
        
        self.processor.voice_threshold = 0.7
        self.processor.text_threshold = 0.6
        
        result = self.processor._perform_hybrid_analysis(voice_result, text_result)
        
        self.assertEqual(result["scenario"], "ai_voice_human_content")
        self.assertAlmostEqual(result["confidence"], 0.9)

    def test_perform_hybrid_analysis_human(self):
        """Test hybrid analysis for human-generated content."""
        
        voice_result = {"confidence": 0.2, "is_ai_generated": False}
        text_result = {"confidence": 0.3, "is_ai_generated": False}
        
        self.processor.voice_threshold = 0.7
        self.processor.text_threshold = 0.6
        
        result = self.processor._perform_hybrid_analysis(voice_result, text_result)
        
        self.assertEqual(result["scenario"], "likely_human")

    def test_generate_overall_assessment_high_risk(self):
        """Test overall assessment generation for high-risk scenario."""
        
        results = {
            "hybrid_analysis": {
                "scenario": "ai_voice_human_content",
                "confidence": 0.9
            }
        }
        
        assessment = self.processor._generate_overall_assessment(results)
        
        self.assertEqual(assessment["primary_detection"], "ai_voice_cloning")
        self.assertEqual(assessment["risk_level"], "high")
        self.assertGreater(len(assessment["recommendations"]), 0)

    @patch.object(AIDetector, '_extract_audio_data')
    @patch.object(AIDetector, '_extract_text_data') 
    @patch.object(AIDetector, '_analyze_text')
    def test_on_data_process_json_text_only(self, mock_analyze_text, mock_extract_text, mock_extract_audio):
        """Test main processing method with text-only input."""
        
        # Mock data extraction
        mock_extract_audio.return_value = None
        mock_extract_text.return_value = "Sample text for analysis"
        
        # Mock text analysis
        mock_analyze_text.return_value = {
            "is_ai_generated": True,
            "confidence": 0.8,
            "model_used": "ollama"
        }
        
        # Set up processor
        self.processor.voice_detection_enabled = True
        self.processor.text_detection_enabled = True
        
        # Process document
        input_doc = {"text_content": "Sample text"}
        result = self.processor.onDataProcessJson(input_doc)
        
        # Verify results
        self.assertIn("ai_detection", result)
        self.assertEqual(result["ai_detection"]["status"], "success")
        self.assertIsNotNone(result["ai_detection"]["text_analysis"])

    def test_voice_analysis_wavefake(self):
        """Test voice analysis using WaveFake model."""
        
        self.processor.voice_model_name = "wavefake"
        self.processor.voice_threshold = 0.7
        
        # Mock audio data
        audio_data = b"fake_audio_bytes"
        
        result = self.processor._analyze_voice(audio_data)
        
        self.assertIn("confidence", result)
        self.assertIn("model_used", result)
        self.assertEqual(result["model_used"], "wavefake")
        self.assertIsInstance(result["confidence"], float)

    def test_error_handling_in_analysis(self):
        """Test error handling in analysis methods."""
        
        # Test voice analysis error handling
        with patch.object(self.processor, '_analyze_voice_wavefake', side_effect=Exception("Test error")):
            self.processor.voice_model_name = "wavefake"
            result = self.processor._analyze_voice(b"fake_audio")
            
            self.assertEqual(result["confidence"], 0.0)
            self.assertIn("error", result)

    def test_configuration_validation(self):
        """Test configuration validation and defaults."""
        
        # Test with missing configuration
        self.mock_config.getOption.side_effect = lambda key: None
        
        # Should handle missing config gracefully
        try:
            self.processor.prepare()
        except Exception as e:
            self.fail(f"Processor preparation failed with missing config: {e}")


class TestAIDetectionModels(unittest.TestCase):
    """Test cases for AI detection model integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = AIDetector()
        self.processor.logger = Mock()

    def test_wavefake_model_placeholder(self):
        """Test WaveFake model placeholder implementation."""
        
        result = self.processor._analyze_voice_wavefake(b"fake_audio")
        
        self.assertIn("confidence", result)
        self.assertIn("features", result)
        self.assertIsInstance(result["confidence"], float)

    def test_speechbrain_model_placeholder(self):
        """Test SpeechBrain model placeholder implementation."""
        
        result = self.processor._analyze_voice_speechbrain(b"fake_audio")
        
        self.assertIn("confidence", result)
        self.assertIsInstance(result["confidence"], float)

    def test_resemblyzer_model_placeholder(self):
        """Test Resemblyzer model placeholder implementation."""
        
        result = self.processor._analyze_voice_resemblyzer(b"fake_audio")
        
        self.assertIn("confidence", result)
        self.assertIn("voice_embedding", result["features"])


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)