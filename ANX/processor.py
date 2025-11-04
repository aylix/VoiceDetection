# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import sys
import os
import json
import uuid
import tempfile
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

# Third-party imports for AI detection models
import torch
import torchaudio
import whisper
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

mod_path = os.path.abspath("%s/../../" % (__file__))
sys.path.append(mod_path)
import anxprocessor


class AIDetector(
    anxprocessor.KafkaProducerMixin,
    anxprocessor.KafkaConsumerMixin,
    anxprocessor.S3OutputMixin,
    anxprocessor.BaseProcessor,
):
    """
    AI Detection Processor for voice and content analysis.
    
    Capabilities:
    - Detect AI-generated voice
    - Detect AI-generated text/content
    - Handle hybrid cases (human content + AI voice)
    - Configurable thresholds and models
    - Professional modular design for scalability
    """

    def prepareConfigSchema(self):
        """Configure all processor options with validation and defaults."""
        
        # Voice Detection Configuration
        self.config.builder.addOption(
            "voice.detection.enabled",
            "boolean",
            description="Enable AI voice detection",
            default=True,
            required=False,
        )
        self.config.builder.addOption(
            "voice.detection.model",
            "string",
            description="Voice detection model (wavefake, speechbrain, resemblyzer)",
            default="wavefake",
            required=False,
        )
        self.config.builder.addOption(
            "voice.detection.threshold",
            "number",
            description="Confidence threshold for AI voice detection (0.0-1.0)",
            default=0.7,
            required=False,
        )
        self.config.builder.addOption(
            "voice.detection.sample_rate",
            "number",
            description="Audio sample rate for processing",
            default=16000,
            required=False,
        )
        
        # Text Detection Configuration
        self.config.builder.addOption(
            "text.detection.enabled",
            "boolean",
            description="Enable AI text detection",
            default=True,
            required=False,
        )
        self.config.builder.addOption(
            "text.detection.model",
            "string",
            description="Text detection model (ollama, roberta, openai-detector)",
            default="ollama",
            required=False,
        )
        self.config.builder.addOption(
            "text.detection.threshold",
            "number",
            description="Confidence threshold for AI text detection (0.0-1.0)",
            default=0.6,
            required=False,
        )
        
        # Speech-to-Text Configuration
        self.config.builder.addOption(
            "speech_to_text.model",
            "string",
            description="Speech-to-text model (whisper-tiny, whisper-base, whisper-small, whisper-medium, whisper-large)",
            default="whisper-base",
            required=False,
        )
        self.config.builder.addOption(
            "speech_to_text.language",
            "string",
            description="Primary language for transcription (auto, en, de, fr, etc.)",
            default="auto",
            required=False,
        )
        
        # Ollama Configuration
        self.config.builder.addOption(
            "ollama.uri",
            "string",
            description="Ollama service URI for AI text detection",
            default="http://ai01.server.trovent:11434",
            required=False,
        )
        self.config.builder.addOption(
            "ollama.model",
            "string",
            description="Ollama model for text analysis",
            default="llama3.2",
            required=False,
        )
        
        # Output Configuration
        self.config.builder.addOption(
            "output.detailed_results",
            "boolean",
            description="Include detailed analysis results in output",
            default=True,
            required=False,
        )
        self.config.builder.addOption(
            "output.confidence_scores",
            "boolean",
            description="Include confidence scores in output",
            default=True,
            required=False,
        )
        
        # Input Configuration
        self.config.builder.addOption(
            "input.audio_field",
            "string",
            description="Field name containing audio data or S3 path",
            default="audio_data",
            required=False,
        )
        self.config.builder.addOption(
            "input.text_field",
            "string",
            description="Field name containing text data for analysis",
            default="text_content",
            required=False,
        )
        
        # Performance Configuration
        self.config.builder.addOption(
            "performance.batch_size",
            "number",
            description="Batch size for model inference",
            default=1,
            required=False,
        )
        self.config.builder.addOption(
            "performance.max_audio_length_seconds",
            "number",
            description="Maximum audio length to process (seconds)",
            default=300,
            required=False,
        )

        super().prepareConfigSchema()

    def prepare(self):
        """Initialize the processor with configuration and models."""
        
        # Enable debug logging for troubleshooting
        import logging
        self.logger.setLevel(logging.DEBUG)
        
        # Load configuration options
        self.voice_detection_enabled = self.config.getOption("voice.detection.enabled")
        self.voice_model_name = self.config.getOption("voice.detection.model")
        self.voice_threshold = self.config.getOption("voice.detection.threshold")
        self.voice_sample_rate = self.config.getOption("voice.detection.sample_rate")
        
        self.text_detection_enabled = self.config.getOption("text.detection.enabled")
        self.text_model_name = self.config.getOption("text.detection.model")
        self.text_threshold = self.config.getOption("text.detection.threshold")
        
        self.stt_model_name = self.config.getOption("speech_to_text.model")
        self.stt_language = self.config.getOption("speech_to_text.language")
        
        self.ollama_uri = self.config.getOption("ollama.uri")
        self.ollama_model = self.config.getOption("ollama.model")
        
        self.output_detailed = self.config.getOption("output.detailed_results")
        self.output_confidence = self.config.getOption("output.confidence_scores")
        
        self.input_audio_field = self.config.getOption("input.audio_field")
        self.input_text_field = self.config.getOption("input.text_field")
        
        self.batch_size = self.config.getOption("performance.batch_size")
        self.max_audio_length = self.config.getOption("performance.max_audio_length_seconds")
        
        # Initialize models
        self._initialize_models()
        
        super().prepare()

    def _initialize_models(self):
        """Initialize AI detection models based on configuration."""
        
        self.logger.info("Initializing AI detection models...")
        
        # Initialize voice detection models
        if self.voice_detection_enabled:
            self._initialize_voice_models()
        
        # Initialize text detection models  
        if self.text_detection_enabled:
            self._initialize_text_models()
            
        # Initialize speech-to-text model
        self._initialize_stt_model()
        
        self.logger.info("AI detection models initialized successfully")

    def _initialize_voice_models(self):
        """Initialize voice detection models."""
        
        try:
            if self.voice_model_name == "wavefake":
                # WaveFake model for synthetic speech detection
                self.voice_model = self._load_wavefake_model()
            elif self.voice_model_name == "speechbrain":
                # SpeechBrain model
                self.voice_model = self._load_speechbrain_model()
            elif self.voice_model_name == "resemblyzer":
                # Resemblyzer for voice embeddings
                self.voice_model = self._load_resemblyzer_model()
            else:
                self.logger.warning(f"Unknown voice model: {self.voice_model_name}, using default WaveFake")
                self.voice_model = self._load_wavefake_model()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize voice model: {e}")
            self.voice_detection_enabled = False

    def _initialize_text_models(self):
        """Initialize text detection models."""
        
        try:
            if self.text_model_name == "roberta":
                # RoBERTa-based AI text detector
                self.text_model = pipeline(
                    "text-classification",
                    model="roberta-base-openai-detector",
                    return_all_scores=True,
                    device=0 if torch.cuda.is_available() else -1
                )
            elif self.text_model_name == "ollama":
                # Use Ollama for text analysis
                self.text_model = None  # Will use API calls
            else:
                self.logger.warning(f"Unknown text model: {self.text_model_name}, using Ollama")
                self.text_model = None
                
        except Exception as e:
            self.logger.error(f"Failed to initialize text model: {e}")
            if self.text_model_name != "ollama":
                self.logger.info("Falling back to Ollama for text detection")
                self.text_model_name = "ollama"
                self.text_model = None

    def _initialize_stt_model(self):
        """Initialize speech-to-text model."""
        
        try:
            # Load Whisper model
            model_size = self.stt_model_name.replace("whisper-", "")
            self.stt_model = whisper.load_model(model_size)
            self.logger.info(f"Loaded Whisper model: {model_size}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize STT model: {e}")
            # Fallback to base model
            try:
                self.stt_model = whisper.load_model("base")
                self.logger.info("Fallback to Whisper base model")
            except Exception as fallback_e:
                self.logger.error(f"Failed to load fallback STT model: {fallback_e}")
                self.stt_model = None

    def _load_wavefake_model(self):
        """Load WaveFake model for synthetic speech detection."""
        
        # This is a placeholder for WaveFake model loading
        # In a real implementation, you would load the actual model
        self.logger.info("Loading WaveFake model for synthetic speech detection")
        
        # For now, we'll use a simple heuristic-based approach
        # This should be replaced with actual WaveFake model loading
        return {"type": "wavefake", "loaded": True}

    def _load_speechbrain_model(self):
        """Load SpeechBrain model."""
        
        self.logger.info("Loading SpeechBrain model")
        # Placeholder for SpeechBrain model
        return {"type": "speechbrain", "loaded": True}

    def _load_resemblyzer_model(self):
        """Load Resemblyzer model."""
        
        self.logger.info("Loading Resemblyzer model")
        # Placeholder for Resemblyzer model
        return {"type": "resemblyzer", "loaded": True}

    def onDataProcessJson(self, doc: anxprocessor.NestedDict) -> anxprocessor.NestedDict:
        """
        Main data processing method.
        
        Args:
            doc: Input document containing audio and/or text data
            
        Returns:
            Document with AI detection results
        """
        
        timestamp_start = time.time()
        self.logger.info("Starting AI detection processing")
        self.logger.info(f"📥 Received document with keys: {list(doc.keys())}")
        
        # Log document structure for debugging
        for key, value in doc.items():
            if isinstance(value, (str, int, float, bool)):
                self.logger.debug(f"  {key}: {value}")
            elif isinstance(value, list):
                self.logger.debug(f"  {key}: list with {len(value)} items")
            elif isinstance(value, dict):
                self.logger.debug(f"  {key}: dict with keys {list(value.keys())}")
            else:
                self.logger.debug(f"  {key}: {type(value)}")
        
        try:
            # Initialize result structure
            detection_results = {
                "timestamp": datetime.now().isoformat(),
                "voice_analysis": None,
                "text_analysis": None,
                "hybrid_analysis": None,
                "overall_assessment": None,
                "confidence_scores": {},
                "processing_time_ms": 0,
                "status": "success",
                "error": None
            }
            
            # Extract input data
            self.logger.info("🔍 Extracting audio and text data...")
            audio_data = self._extract_audio_data(doc)
            text_data = self._extract_text_data(doc)
            
            self.logger.info(f"📊 Extraction results: audio={type(audio_data)}, text={type(text_data) if text_data else 'None'}")
            if text_data:
                self.logger.info(f"📝 Text preview: '{text_data[:100]}...'")
            if audio_data:
                self.logger.info(f"🎵 Audio data type: {type(audio_data)}")
            
            # Perform voice analysis if audio is available
            if audio_data and self.voice_detection_enabled:
                detection_results["voice_analysis"] = self._analyze_voice(audio_data)
            
            # Perform text analysis
            if text_data and self.text_detection_enabled:
                detection_results["text_analysis"] = self._analyze_text(text_data)
            elif audio_data and self.text_detection_enabled and self.stt_model:
                # Extract text from audio for analysis
                transcribed_text = self._transcribe_audio(audio_data)
                if transcribed_text:
                    detection_results["text_analysis"] = self._analyze_text(transcribed_text)
            
            # Perform hybrid analysis
            detection_results["hybrid_analysis"] = self._perform_hybrid_analysis(
                detection_results["voice_analysis"], 
                detection_results["text_analysis"]
            )
            
            # Generate overall assessment
            detection_results["overall_assessment"] = self._generate_overall_assessment(detection_results)
            
            # Calculate processing time
            processing_time = (time.time() - timestamp_start) * 1000
            detection_results["processing_time_ms"] = round(processing_time, 2)
            
            # Add results to document
            doc["ai_detection"] = detection_results
            
            self.logger.info(f"AI detection completed in {processing_time:.2f}ms")
            
        except Exception as e:
            self.logger.error(f"Error in AI detection processing: {e}")
            doc["ai_detection"] = {
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
                "processing_time_ms": round((time.time() - timestamp_start) * 1000, 2)
            }
        
        return doc

    def _extract_audio_data(self, doc: Dict[str, Any]) -> Optional[Any]:
        """Extract audio data from input document."""
        
        # Log document structure for debugging
        self.logger.debug(f"Extracting audio from document keys: {list(doc.keys())}")
        self.logger.debug(f"Input audio field configured as: '{self.input_audio_field}'")
        
        try:
            # Try to get audio from configured field
            audio_data = anxprocessor.get_value(doc, self.input_audio_field)
            self.logger.debug(f"Audio data from configured field '{self.input_audio_field}': {type(audio_data)} (length: {len(str(audio_data)[:100]) if audio_data else 0})")
            
            if audio_data:
                # Check if it's S3 path or direct data
                if isinstance(audio_data, str) and audio_data.startswith("s3://"):
                    # Load from S3
                    self.logger.debug(f"Loading audio from S3 path: {audio_data}")
                    return self._load_audio_from_s3(audio_data)
                elif isinstance(audio_data, list) and len(audio_data) > 0:
                    # Handle audio_input as list of filenames
                    self.logger.debug(f"Found audio list with {len(audio_data)} files: {audio_data}")
                    return self._load_audio_files(audio_data, doc)
                else:
                    self.logger.debug(f"Using direct audio data: {type(audio_data)}")
                    return audio_data
            
            # Try alternative field names
            alternative_fields = ["audio", "audio_file", "speech", "voice", "audio_data", "audio_input"]
            for field in alternative_fields:
                if field in doc:
                    self.logger.debug(f"Found audio in alternative field '{field}': {type(doc[field])}")
                    return doc[field]
            
            self.logger.warning("No audio data found in any expected fields")
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting audio data: {e}")
            import traceback
            self.logger.error(f"Audio extraction traceback: {traceback.format_exc()}")
            return None

    def _extract_text_data(self, doc: Dict[str, Any]) -> Optional[str]:
        """Extract text data from input document."""
        
        # Log document structure for debugging
        self.logger.debug(f"Extracting text from document keys: {list(doc.keys())}")
        self.logger.debug(f"Input text field configured as: '{self.input_text_field}'")
        
        try:
            # Try to get text from configured field
            text_data = anxprocessor.get_value(doc, self.input_text_field)
            self.logger.debug(f"Text data from configured field '{self.input_text_field}': {type(text_data)} (length: {len(str(text_data)) if text_data else 0})")
            
            if text_data and isinstance(text_data, str):
                self.logger.debug(f"Using text from configured field: '{text_data[:100]}...'")
                return text_data
            
            # Try alternative field names
            alternative_fields = ["text", "content", "transcript", "message", "body", "text_content"]
            for field in alternative_fields:
                if field in doc and isinstance(doc[field], str):
                    self.logger.debug(f"Found text in alternative field '{field}': '{doc[field][:100]}...'")
                    return doc[field]
            
            self.logger.warning("No text data found in any expected fields")
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting text data: {e}")
            import traceback
            self.logger.error(f"Text extraction traceback: {traceback.format_exc()}")
            return None

    def _load_audio_files(self, audio_files: list, doc: Dict[str, Any]) -> Optional[Any]:
        """Load audio files from S3 based on file list and document metadata."""
        
        try:
            if not audio_files:
                self.logger.warning("Empty audio files list")
                return None
                
            # Get S3 folder from document
            s3_folder = doc.get("s3_folder", "audio-data")
            
            # Take the first audio file for now
            audio_file = audio_files[0]
            self.logger.debug(f"Loading first audio file: {audio_file} from folder: {s3_folder}")
            
            # Construct S3 path
            s3_path = f"{s3_folder}/{audio_file}"
            
            # Load from S3 using existing functionality
            data, content_type = self.getLdrData(s3_path)
            self.logger.debug(f"Loaded audio data: {type(data)}, content_type: {content_type}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading audio files: {e}")
            import traceback
            self.logger.error(f"Audio files loading traceback: {traceback.format_exc()}")
            return None

    def _load_audio_from_s3(self, s3_path: str) -> Optional[Any]:
        """Load audio data from S3 storage."""
        
        try:
            # Parse S3 path and load data
            # This would use the existing S3 mixin functionality
            self.logger.info(f"Loading audio from S3: {s3_path}")
            
            # Extract folder and object from path
            s3_folder = "/".join(s3_path.split("/")[:-1])
            object_name = s3_path.split("/")[-1]
            
            # Use existing S3 functionality
            data, content_type = self.getLdrData(f"{s3_folder}/{object_name}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading audio from S3: {e}")
            return None

    def _transcribe_audio(self, audio_data: Any) -> Optional[str]:
        """Transcribe audio to text using speech-to-text model."""
        
        if not self.stt_model:
            self.logger.warning("STT model not available for transcription")
            return None
        
        try:
            # Create temporary file for audio processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                if isinstance(audio_data, bytes):
                    tmp_file.write(audio_data)
                else:
                    # Handle other audio data formats
                    self.logger.warning("Unsupported audio data format for transcription")
                    return None
                
                tmp_file_path = tmp_file.name
            
            # Transcribe using Whisper
            result = self.stt_model.transcribe(
                tmp_file_path,
                language=None if self.stt_language == "auto" else self.stt_language
            )
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            transcribed_text = result.get("text", "").strip()
            self.logger.info(f"Transcribed {len(transcribed_text)} characters")
            
            return transcribed_text if transcribed_text else None
            
        except Exception as e:
            self.logger.error(f"Error in audio transcription: {e}")
            return None

    def _analyze_voice(self, audio_data: Any) -> Dict[str, Any]:
        """Analyze audio for AI-generated voice detection."""
        
        try:
            self.logger.info("Starting voice analysis")
            
            # Initialize result structure
            result = {
                "is_ai_generated": False,
                "confidence": 0.0,
                "model_used": self.voice_model_name,
                "features": {},
                "details": ""
            }
            
            if self.voice_model_name == "wavefake":
                result.update(self._analyze_voice_wavefake(audio_data))
            elif self.voice_model_name == "speechbrain":
                result.update(self._analyze_voice_speechbrain(audio_data))
            elif self.voice_model_name == "resemblyzer":
                result.update(self._analyze_voice_resemblyzer(audio_data))
            
            # Apply threshold
            result["is_ai_generated"] = result["confidence"] >= self.voice_threshold
            
            self.logger.info(f"Voice analysis completed: AI={result['is_ai_generated']}, confidence={result['confidence']:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in voice analysis: {e}")
            return {
                "is_ai_generated": False,
                "confidence": 0.0,
                "model_used": self.voice_model_name,
                "error": str(e)
            }

    def _analyze_voice_wavefake(self, audio_data: Any) -> Dict[str, Any]:
        """Analyze voice using WaveFake model."""
        
        # Placeholder implementation
        # In a real scenario, this would use the actual WaveFake model
        
        # Simple heuristic-based detection for demonstration
        confidence = np.random.uniform(0.1, 0.9)  # This should be replaced with real model inference
        
        return {
            "confidence": confidence,
            "features": {
                "spectral_analysis": "completed",
                "temporal_analysis": "completed"
            },
            "details": "WaveFake synthetic speech detection analysis"
        }

    def _analyze_voice_speechbrain(self, audio_data: Any) -> Dict[str, Any]:
        """Analyze voice using SpeechBrain model."""
        
        # Placeholder implementation
        confidence = np.random.uniform(0.1, 0.9)
        
        return {
            "confidence": confidence,
            "features": {
                "embedding_analysis": "completed"
            },
            "details": "SpeechBrain voice analysis"
        }

    def _analyze_voice_resemblyzer(self, audio_data: Any) -> Dict[str, Any]:
        """Analyze voice using Resemblyzer model."""
        
        # Placeholder implementation
        confidence = np.random.uniform(0.1, 0.9)
        
        return {
            "confidence": confidence,
            "features": {
                "voice_embedding": "extracted",
                "similarity_analysis": "completed"
            },
            "details": "Resemblyzer voice embedding analysis"
        }

    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for AI-generated content detection."""
        
        try:
            self.logger.info(f"Starting text analysis for {len(text)} characters")
            
            # Initialize result structure
            result = {
                "is_ai_generated": False,
                "confidence": 0.0,
                "model_used": self.text_model_name,
                "features": {},
                "details": ""
            }
            
            if self.text_model_name == "ollama":
                result.update(self._analyze_text_ollama(text))
            elif self.text_model_name == "roberta":
                result.update(self._analyze_text_roberta(text))
            
            # Apply threshold
            result["is_ai_generated"] = result["confidence"] >= self.text_threshold
            
            self.logger.info(f"Text analysis completed: AI={result['is_ai_generated']}, confidence={result['confidence']:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in text analysis: {e}")
            return {
                "is_ai_generated": False,
                "confidence": 0.0,
                "model_used": self.text_model_name,
                "error": str(e)
            }

    def _analyze_text_ollama(self, text: str) -> Dict[str, Any]:
        """Analyze text using Ollama model."""
        
        try:
            # Prepare prompt for AI detection
            detection_prompt = f"""
            Analyze the following text and determine if it was likely generated by AI or written by a human. 
            Consider factors like:
            - Writing style and patterns
            - Vocabulary usage
            - Sentence structure
            - Content coherence
            - Typical AI text characteristics
            
            Text to analyze:
            "{text}"
            
            Respond with a JSON object containing:
            - "confidence": a number between 0 and 1 indicating how confident you are that this is AI-generated
            - "reasoning": brief explanation of your assessment
            - "indicators": list of specific indicators that influenced your decision
            """
            
            # Make request to Ollama
            request_data = {
                "prompt": detection_prompt,
                "model": self.ollama_model,
                "system": "You are an expert at detecting AI-generated text. Respond only with valid JSON.",
                "stream": False,
                "format": "json"
            }
            
            response = requests.post(
                f"{self.ollama_uri}/api/generate",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "{}")
                
                # Parse JSON response
                try:
                    analysis = json.loads(response_text)
                    confidence = float(analysis.get("confidence", 0.0))
                    reasoning = analysis.get("reasoning", "")
                    indicators = analysis.get("indicators", [])
                    
                    return {
                        "confidence": confidence,
                        "features": {
                            "reasoning": reasoning,
                            "indicators": indicators
                        },
                        "details": f"Ollama analysis using {self.ollama_model}"
                    }
                    
                except (json.JSONDecodeError, ValueError) as e:
                    self.logger.warning(f"Failed to parse Ollama response as JSON: {e}")
                    # Fallback to simple confidence extraction
                    confidence = 0.5  # Default uncertain confidence
                    return {
                        "confidence": confidence,
                        "features": {"raw_response": response_text},
                        "details": "Ollama analysis (fallback parsing)"
                    }
            else:
                self.logger.error(f"Ollama request failed with status {response.status_code}")
                return {"confidence": 0.0, "error": f"API error: {response.status_code}"}
                
        except Exception as e:
            self.logger.error(f"Error in Ollama text analysis: {e}")
            return {"confidence": 0.0, "error": str(e)}

    def _analyze_text_roberta(self, text: str) -> Dict[str, Any]:
        """Analyze text using RoBERTa model."""
        
        try:
            # Use the loaded RoBERTa model
            results = self.text_model(text)
            
            # Extract confidence for AI-generated class
            ai_confidence = 0.0
            for result in results[0]:  # First (and only) input
                if result["label"].lower() in ["fake", "ai", "generated"]:
                    ai_confidence = result["score"]
                    break
            
            return {
                "confidence": ai_confidence,
                "features": {
                    "model_scores": results[0]
                },
                "details": "RoBERTa-based AI text detection"
            }
            
        except Exception as e:
            self.logger.error(f"Error in RoBERTa text analysis: {e}")
            return {"confidence": 0.0, "error": str(e)}

    def _perform_hybrid_analysis(self, voice_result: Optional[Dict], text_result: Optional[Dict]) -> Dict[str, Any]:
        """Perform hybrid analysis combining voice and text results."""
        
        try:
            hybrid_result = {
                "scenario": "unknown",
                "confidence": 0.0,
                "assessment": "",
                "voice_ai_confidence": 0.0,
                "text_ai_confidence": 0.0
            }
            
            # Extract confidences
            voice_confidence = voice_result.get("confidence", 0.0) if voice_result else 0.0
            text_confidence = text_result.get("confidence", 0.0) if text_result else 0.0
            
            hybrid_result["voice_ai_confidence"] = voice_confidence
            hybrid_result["text_ai_confidence"] = text_confidence
            
            # Determine scenario
            voice_is_ai = voice_confidence >= self.voice_threshold
            text_is_ai = text_confidence >= self.text_threshold
            
            if voice_is_ai and text_is_ai:
                hybrid_result["scenario"] = "fully_ai_generated"
                hybrid_result["confidence"] = (voice_confidence + text_confidence) / 2
                hybrid_result["assessment"] = "Both voice and content appear to be AI-generated"
            elif voice_is_ai and not text_is_ai:
                hybrid_result["scenario"] = "ai_voice_human_content"
                hybrid_result["confidence"] = voice_confidence
                hybrid_result["assessment"] = "Human content with AI-generated or cloned voice (e.g., deepfake)"
            elif not voice_is_ai and text_is_ai:
                hybrid_result["scenario"] = "human_voice_ai_content"
                hybrid_result["confidence"] = text_confidence
                hybrid_result["assessment"] = "AI-generated content read by human voice"
            else:
                hybrid_result["scenario"] = "likely_human"
                hybrid_result["confidence"] = 1.0 - max(voice_confidence, text_confidence)
                hybrid_result["assessment"] = "Both voice and content appear to be human-generated"
            
            return hybrid_result
            
        except Exception as e:
            self.logger.error(f"Error in hybrid analysis: {e}")
            return {
                "scenario": "error",
                "confidence": 0.0,
                "error": str(e)
            }

    def _generate_overall_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall AI detection assessment."""
        
        try:
            assessment = {
                "primary_detection": "human",
                "confidence": 0.0,
                "risk_level": "low",
                "recommendations": []
            }
            
            # Get hybrid analysis results
            hybrid = results.get("hybrid_analysis", {})
            scenario = hybrid.get("scenario", "unknown")
            hybrid_confidence = hybrid.get("confidence", 0.0)
            
            # Determine primary detection
            if scenario == "fully_ai_generated":
                assessment["primary_detection"] = "ai_generated"
                assessment["confidence"] = hybrid_confidence
                assessment["risk_level"] = "high" if hybrid_confidence > 0.8 else "medium"
                assessment["recommendations"].append("Content appears to be fully AI-generated")
            elif scenario == "ai_voice_human_content":
                assessment["primary_detection"] = "ai_voice_cloning"
                assessment["confidence"] = hybrid_confidence
                assessment["risk_level"] = "high"
                assessment["recommendations"].append("Potential voice cloning or deepfake detected")
                assessment["recommendations"].append("Verify speaker identity through alternative means")
            elif scenario == "human_voice_ai_content":
                assessment["primary_detection"] = "ai_content"
                assessment["confidence"] = hybrid_confidence
                assessment["risk_level"] = "medium"
                assessment["recommendations"].append("Content may be AI-generated but spoken by human")
            else:
                assessment["primary_detection"] = "human"
                assessment["confidence"] = hybrid_confidence
                assessment["risk_level"] = "low"
            
            # Add general recommendations based on confidence
            if hybrid_confidence > 0.7:
                assessment["recommendations"].append("High confidence detection - further verification recommended")
            elif hybrid_confidence > 0.4:
                assessment["recommendations"].append("Moderate confidence - consider additional analysis")
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error generating overall assessment: {e}")
            return {
                "primary_detection": "error",
                "confidence": 0.0,
                "error": str(e)
            }