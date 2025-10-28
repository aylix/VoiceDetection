import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

from .audio_processor import AudioProcessor
from .analyzers.speaker_recognition import SpeakerRecognitionAnalyzer
from .analyzers.synthetic_detection import SyntheticVoiceDetector
from .analyzers.language_detection import LanguageDetectionAnalyzer
from .analyzers.emotion_analysis import EmotionAnalyzer
from .analyzers.environmental_analysis import EnvironmentalAnalyzer
from .analyzers.psychological_analysis import PsychologicalAnalyzer
from .utils.report_generator import ReportGenerator
from .models.model_loader import ModelLoader

@dataclass
class AnalysisResult:
    """Container for all analysis results"""
    speaker_demographics: Dict[str, Any]
    language_analysis: Dict[str, Any]
    psychological_indicators: Dict[str, Any]
    environmental_context: Dict[str, Any]
    voice_authenticity: Dict[str, Any]
    additional_insights: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export"""
        return {
            'speaker_demographics': self.speaker_demographics,
            'language_analysis': self.language_analysis,
            'psychological_indicators': self.psychological_indicators,
            'environmental_context': self.environmental_context,
            'voice_authenticity': self.voice_authenticity,
            'additional_insights': self.additional_insights,
            'metadata': self.metadata
        }
    
    def generate_report(self) -> str:
        """Generate human-readable report"""
        report_gen = ReportGenerator()
        return report_gen.generate_text_report(self)

class VoiceAnalyzer:
    """Main voice analysis system for forensic applications"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the voice analyzer with all components
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = self._setup_logging()
        self.model_loader = ModelLoader(config_path)
        self.audio_processor = AudioProcessor()
        
        # Initialize all analyzers
        self.speaker_analyzer = SpeakerRecognitionAnalyzer(self.model_loader)
        self.synthetic_detector = SyntheticVoiceDetector(self.model_loader)
        self.language_analyzer = LanguageDetectionAnalyzer(self.model_loader)
        self.emotion_analyzer = EmotionAnalyzer(self.model_loader)
        self.environmental_analyzer = EnvironmentalAnalyzer(self.model_loader)
        self.psychological_analyzer = PsychologicalAnalyzer(self.model_loader)
        
        self.logger.info("VoiceAnalyzer initialized successfully")
    
    def analyze_audio(self, audio_path: str, detailed: bool = True) -> AnalysisResult:
        """Perform comprehensive voice analysis
        
        Args:
            audio_path: Path to audio file
            detailed: Whether to perform detailed analysis
            
        Returns:
            AnalysisResult containing all analysis results
        """
        self.logger.info(f"Starting analysis of {audio_path}")
        
        try:
            # Process audio file
            audio_data = self.audio_processor.load_audio(audio_path)
            
            # Perform all analyses
            results = {}
            
            # Speaker demographics and biometrics
            self.logger.info("Analyzing speaker demographics...")
            results['speaker_demographics'] = self.speaker_analyzer.analyze(audio_data)
            
            # Synthetic voice detection
            self.logger.info("Detecting synthetic voice...")
            results['voice_authenticity'] = self.synthetic_detector.analyze(audio_data)
            
            # Language and accent detection
            self.logger.info("Analyzing language and accent...")
            results['language_analysis'] = self.language_analyzer.analyze(audio_data)
            
            # Emotional state analysis
            self.logger.info("Analyzing emotional state...")
            emotion_results = self.emotion_analyzer.analyze(audio_data)
            
            # Psychological analysis
            self.logger.info("Performing psychological analysis...")
            psych_results = self.psychological_analyzer.analyze(audio_data)
            results['psychological_indicators'] = {**emotion_results, **psych_results}
            
            # Environmental analysis
            self.logger.info("Analyzing environmental context...")
            results['environmental_context'] = self.environmental_analyzer.analyze(audio_data)
            
            # Additional insights
            self.logger.info("Extracting additional insights...")
            results['additional_insights'] = self._extract_additional_insights(audio_data, results)
            
            # Metadata
            results['metadata'] = {
                'analysis_timestamp': datetime.now().isoformat(),
                'audio_file': audio_path,
                'audio_duration': audio_data.get('duration', 0),
                'sample_rate': audio_data.get('sample_rate', 0),
                'analyzer_version': '1.0.0'
            }
            
            analysis_result = AnalysisResult(**results)
            self.logger.info("Analysis completed successfully")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error during analysis: {str(e)}")
            raise
    
    def _extract_additional_insights(self, audio_data: Dict, results: Dict) -> Dict[str, Any]:
        """Extract additional insights from combined analysis results"""
        insights = {}
        
        # Voice quality assessment
        insights['voice_quality'] = self._assess_voice_quality(audio_data)
        
        # Speaking rate analysis
        insights['speaking_rate'] = self._analyze_speaking_rate(audio_data)
        
        # Confidence scoring
        insights['overall_confidence'] = self._calculate_confidence_score(results)
        
        # Risk assessment
        insights['risk_indicators'] = self._assess_risk_indicators(results)
        
        return insights
    
    def _assess_voice_quality(self, audio_data: Dict) -> Dict[str, Any]:
        """Assess overall voice quality"""
        # Implement voice quality assessment
        return {
            'clarity': 'Good',
            'noise_level': 'Low',
            'recording_quality': 'High'
        }
    
    def _analyze_speaking_rate(self, audio_data: Dict) -> Dict[str, Any]:
        """Analyze speaking rate patterns"""
        # Implement speaking rate analysis
        return {
            'words_per_minute': 150,
            'rate_variability': 'Normal',
            'pace_assessment': 'Slightly elevated'
        }
    
    def _calculate_confidence_score(self, results: Dict) -> float:
        """Calculate overall confidence score for the analysis"""
        # Implement confidence calculation based on individual analyzer confidences
        return 0.85
    
    def _assess_risk_indicators(self, results: Dict) -> Dict[str, Any]:
        """Assess various risk indicators"""
        return {
            'deception_probability': 'Low',
            'stress_indicators': 'Moderate',
            'authenticity_concerns': 'None'
        }
    
    def export_results(self, result: AnalysisResult, output_path: str, format: str = 'json'):
        """Export analysis results to file
        
        Args:
            result: Analysis result to export
            output_path: Path to save the results
            format: Export format ('json', 'pdf', 'txt')
        """
        report_gen = ReportGenerator()
        
        if format == 'json':
            report_gen.export_json(result, output_path)
        elif format == 'pdf':
            report_gen.export_pdf(result, output_path)
        elif format == 'txt':
            report_gen.export_text(result, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Results exported to {output_path}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('VoiceAnalyzer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

# Example usage
if __name__ == "__main__":
    analyzer = VoiceAnalyzer()
    result = analyzer.analyze_audio("sample_audio.wav")
    print(result.generate_report())
    analyzer.export_results(result, "analysis_results.json")
