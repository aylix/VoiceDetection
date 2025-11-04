# AI Detection Processor

A comprehensive AI detection component for voice and content analysis, designed to identify AI-generated content in various scenarios including voice cloning and deepfakes.

## Features

### Voice Detection
- **WaveFake Model**: Synthetic speech detection using spectral analysis
- **SpeechBrain**: Voice embedding analysis for AI detection  
- **Resemblyzer**: Voice similarity and cloning detection
- Configurable confidence thresholds
- Support for multiple audio formats

### Text Detection
- **Ollama Integration**: Uses your existing Ollama setup (http://ai01.server.trovent:11434)
- **RoBERTa Models**: Transformer-based AI text detection
- **OpenAI Detectors**: Pre-trained classifiers for GPT-generated content
- Configurable detection thresholds

### Hybrid Analysis
- **Fully AI Generated**: Both voice and content are AI-generated
- **AI Voice + Human Content**: Voice cloning scenarios (deepfakes)
- **Human Voice + AI Content**: AI-generated text read by humans
- **Likely Human**: Natural human-generated content

### Speech-to-Text Integration
- **Whisper Models**: tiny, base, small, medium, large variants
- **Multi-language Support**: Auto-detection or specified languages
- **Audio Preprocessing**: Automatic format conversion and optimization

## Configuration

### Voice Detection Settings
```json
{
  "voice.detection.enabled": true,
  "voice.detection.model": "wavefake",
  "voice.detection.threshold": 0.7,
  "voice.detection.sample_rate": 16000
}
```

### Text Detection Settings
```json
{
  "text.detection.enabled": true, 
  "text.detection.model": "ollama",
  "text.detection.threshold": 0.6
}
```

### Ollama Configuration
```json
{
  "ollama.uri": "http://ai01.server.trovent:11434",
  "ollama.model": "llama3.2"
}
```

### Speech-to-Text Settings
```json
{
  "speech_to_text.model": "whisper-base",
  "speech_to_text.language": "auto"
}
```

## Input Format

The processor accepts documents with audio and/or text fields:

```json
{
  "audio_data": "<audio_bytes_or_s3_path>",
  "text_content": "Text to analyze for AI generation",
  "s3_folder": "audio-samples",
  "language": "en"
}
```

## Output Format

Results include comprehensive analysis and confidence scores:

```json
{
  "ai_detection": {
    "timestamp": "2024-11-04T10:30:00.000Z",
    "voice_analysis": {
      "is_ai_generated": false,
      "confidence": 0.3,
      "model_used": "wavefake",
      "features": {
        "spectral_analysis": "completed",
        "temporal_analysis": "completed"
      }
    },
    "text_analysis": {
      "is_ai_generated": true,
      "confidence": 0.8,
      "model_used": "ollama",
      "features": {
        "reasoning": "Text shows repetitive patterns typical of AI",
        "indicators": ["formal structure", "repetitive phrasing"]
      }
    },
    "hybrid_analysis": {
      "scenario": "human_voice_ai_content",
      "confidence": 0.8,
      "assessment": "AI-generated content read by human voice",
      "voice_ai_confidence": 0.3,
      "text_ai_confidence": 0.8
    },
    "overall_assessment": {
      "primary_detection": "ai_content",
      "confidence": 0.8,
      "risk_level": "medium",
      "recommendations": [
        "Content may be AI-generated but spoken by human"
      ]
    },
    "processing_time_ms": 1250,
    "status": "success"
  }
}
```

## Model Requirements

### Ollama Models
The component integrates with your existing Ollama setup. Ensure the following models are available:

```bash
# Check available models
curl http://ai01.server.trovent:11434/api/tags

# Pull required models if needed
ollama pull llama3.2
ollama pull mistral
ollama pull codellama
```

### Whisper Models
Whisper models are downloaded automatically on first use. Larger models provide better accuracy but require more resources:

- `whisper-tiny`: Fastest, basic accuracy
- `whisper-base`: Good balance (default)
- `whisper-small`: Better accuracy
- `whisper-medium`: High accuracy
- `whisper-large`: Best accuracy, most resources

## Installation

1. **Build the Docker image**:
```bash
cd /path/to/ai-detection
docker build -t ai-detection-processor .
```

2. **Install dependencies locally** (for development):
```bash
pip install -r requirements.txt
```

## Performance Considerations

### Resource Usage
- **GPU Support**: Automatically detected and used when available
- **Memory Usage**: Varies by model size (1-8GB for larger models)
- **Processing Time**: 1-5 seconds per audio minute depending on model

### Optimization Settings
```json
{
  "performance.batch_size": 1,
  "performance.max_audio_length_seconds": 300,
  "performance.model_cache_enabled": true
}
```

## Testing

Run the comprehensive test suite:

```bash
python -m pytest test_ai_detection.py -v
```

Test categories:
- Configuration validation
- Audio/text extraction
- Model integration 
- Hybrid analysis logic
- Error handling
- Performance benchmarks

## Integration Examples

### Kafka Integration
The processor integrates seamlessly with existing Kafka infrastructure:

```json
{
  "input": {
    "kafka": {
      "host": "kafka",
      "port": 9092, 
      "topic": "audio-analysis"
    }
  },
  "output": {
    "kafka": {
      "host": "kafka",
      "port": 9092,
      "topic": "ai-detection-results"
    }
  }
}
```

### S3 Integration
Supports direct S3 audio file processing:

```json
{
  "audio_data": "s3://bucket/path/to/audio.wav",
  "s3_folder": "audio-samples"
}
```

## Security Considerations

- **Model Security**: Models are loaded from trusted sources
- **API Security**: Ollama API calls use configured endpoints only
- **Data Privacy**: Audio/text data is processed locally when possible
- **Input Validation**: All inputs are validated before processing

## Troubleshooting

### Common Issues

1. **Ollama Connection Issues**:
```bash
# Test Ollama connection
curl http://ai01.server.trovent:11434/api/tags
```

2. **Audio Format Issues**:
- Ensure ffmpeg is installed
- Check audio file format compatibility
- Verify S3 access permissions

3. **Memory Issues**:
- Reduce model size (use whisper-tiny/base)
- Adjust batch_size configuration
- Monitor GPU memory usage

### Logging
The processor provides detailed logging for troubleshooting:

```python
self.logger.info("Starting AI detection processing")
self.logger.error(f"Error in voice analysis: {e}")
```

## Roadmap

### Planned Enhancements
- **Real-time Detection**: Streaming audio analysis
- **Advanced Models**: Integration with latest detection models
- **Performance Optimization**: GPU acceleration improvements
- **Extended Language Support**: More Whisper language variants
- **Custom Model Training**: Support for domain-specific models

### Model Updates
- WaveFake model implementation
- SpeechBrain integration
- Resemblyzer voice embedding support
- Additional transformer-based text detectors

## License

This component follows the project's existing license terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review component logs
3. Validate configuration settings
4. Test with sample data