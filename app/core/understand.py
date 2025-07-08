"""Understanding phase - Parse and analyze user input with context."""

import re
import time
from typing import Dict, Any, Optional, List

from app.schemas import ChatRequest, UPEEResult, UPEEPhase, FileContext
from app.utils.logging_config import get_logger
from app.utils.file_processor import FileProcessor, ProcessedFile
from app.utils.text_summarizer import TextSummarizer
from app.llm_providers import LLMProviderManager
from app.settings import Settings


class UnderstandPhase:
    """
    Understanding phase of the UPEE loop.
    
    Responsible for:
    - Parsing user message and extracting intent
    - Processing file contexts
    - Analyzing complexity and requirements
    - Preparing context for subsequent phases
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger("understand_phase")
        
        # Initialize file processing components
        self.file_processor = FileProcessor(
            max_chunk_size=settings.max_context_tokens // 4,  # Quarter of max context for chunks
            overlap_size=200
        )
        self.llm_manager = LLMProviderManager(settings)
        self.text_summarizer = TextSummarizer(settings, self.llm_manager)
    
    async def process(
        self, 
        request: ChatRequest, 
        request_id: str
    ) -> UPEEResult:
        """
        Process the understanding phase.
        
        Args:
            request: The chat request to understand
            request_id: Request tracking ID
            
        Returns:
            UPEEResult with understanding analysis
        """
        self.logger.info(
            "Starting understanding phase",
            request_id=request_id,
            message_length=len(request.message),
            files_count=len(request.files) if request.files else 0
        )
        
        try:
            # Analyze user intent
            intent_analysis = await self._analyze_intent(request.message)
            
            # Process file contexts
            file_analysis = await self._process_file_contexts(request.files)
            
            # Calculate context requirements
            context_analysis = await self._analyze_context_requirements(
                request, intent_analysis, file_analysis
            )
            
            # Determine complexity
            complexity = await self._assess_complexity(
                request, intent_analysis, file_analysis
            )
            
            # Build understanding result
            content = self._build_understanding_summary(
                intent_analysis, file_analysis, context_analysis, complexity
            )
            
            metadata = {
                "intent": intent_analysis["intent"],
                "confidence": intent_analysis["confidence"],
                "complexity": complexity,
                "context_tokens": context_analysis["estimated_tokens"],
                "file_count": len(request.files) if request.files else 0,
                "requires_external_calls": self._needs_external_calls(intent_analysis),
                "language": intent_analysis.get("language", "english"),
                "topics": intent_analysis.get("topics", []),
                "entities": intent_analysis.get("entities", []),
                # Enhanced file processing metadata
                "file_processing": {
                    "total_chunks": file_analysis.get("total_chunks", 0),
                    "summarized_files": file_analysis.get("summarized_files", 0),
                    "file_types": file_analysis.get("file_types", []),
                    "processing_stats": file_analysis.get("processing_stats", {})
                },
                "processed_files": file_analysis.get("processed_files", [])
            }
            
            result = UPEEResult(
                phase=UPEEPhase.UNDERSTAND,
                content=content,
                metadata=metadata,
                completed=True
            )
            
            self.logger.info(
                "Understanding phase completed",
                request_id=request_id,
                intent=metadata["intent"],
                complexity=complexity,
                context_tokens=metadata["context_tokens"]
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Understanding phase failed",
                request_id=request_id,
                error=str(e),
                exc_info=True
            )
            
            return UPEEResult(
                phase=UPEEPhase.UNDERSTAND,
                content=f"Understanding failed: {str(e)}",
                metadata={"error": str(e)},
                completed=False,
                error=str(e)
            )
    
    async def _analyze_intent(self, message: str) -> Dict[str, Any]:
        """Analyze user message to determine intent and characteristics."""
        
        # Simple intent classification (will be enhanced with LLM later)
        intent_patterns = {
            "question": [
                r"\?", r"\bwhat\b", r"\bhow\b", r"\bwhy\b", r"\bwhen\b", 
                r"\bwhere\b", r"\bwho\b", r"\bwhich\b", r"\bcan you\b"
            ],
            "request": [
                r"\bplease\b", r"\bcould you\b", r"\bwould you\b", 
                r"\bhelp me\b", r"\bi need\b", r"\bi want\b"
            ],
            "task": [
                r"\bcreate\b", r"\bgenerate\b", r"\bbuild\b", r"\bmake\b", 
                r"\bwrite\b", r"\bimplement\b", r"\bdevelop\b"
            ],
            "analysis": [
                r"\banalyze\b", r"\bexplain\b", r"\breview\b", r"\bcheck\b", 
                r"\bexamine\b", r"\bevaluate\b"
            ],
            "conversation": [
                r"\bhello\b", r"\bhi\b", r"\bthanks\b", r"\bthank you\b", 
                r"\bgoodbye\b", r"\bbye\b"
            ]
        }
        
        message_lower = message.lower()
        intent_scores = {}
        
        for intent, patterns in intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, message_lower))
                score += matches
            intent_scores[intent] = score
        
        # Determine primary intent
        primary_intent = max(intent_scores, key=intent_scores.get)
        confidence = min(intent_scores[primary_intent] / 3.0, 1.0)  # Normalize
        
        if confidence < 0.3:
            primary_intent = "general"
            confidence = 0.5
        
        # Extract topics and entities (simplified)
        topics = self._extract_topics(message_lower)
        entities = self._extract_entities(message)
        
        return {
            "intent": primary_intent,
            "confidence": confidence,
            "language": "english",  # Simple detection
            "topics": topics,
            "entities": entities,
            "word_count": len(message.split()),
            "char_count": len(message)
        }
    
    async def _process_file_contexts(
        self, 
        files: Optional[List[FileContext]]
    ) -> Dict[str, Any]:
        """Process and analyze file contexts using intelligent file processing."""
        if not files:
            return {
                "file_count": 0,
                "total_characters": 0,
                "file_types": [],
                "requires_summarization": False,
                "processed_files": [],
                "total_chunks": 0,
                "summarized_files": 0
            }
        
        self.logger.info(
            "Processing file contexts",
            file_count=len(files),
            total_size=sum(len(f.content) for f in files)
        )
        
        processed_files = []
        total_chars = 0
        file_types = []
        total_chunks = 0
        summarized_files = 0
        
        for file_context in files:
            try:
                # Process file with intelligent chunking
                processed_file = self.file_processor.process_file(file_context)
                processed_files.append(processed_file)
                
                total_chars += processed_file.total_size
                total_chunks += len(processed_file.chunks)
                
                # Track file types
                if processed_file.file_type.value not in file_types:
                    file_types.append(processed_file.file_type.value)
                
                # Generate summary if file is large
                if processed_file.total_size > 2000:  # Summarize files larger than 2KB
                    try:
                        summary_result = await self.text_summarizer.summarize_file(processed_file)
                        if summary_result:
                            processed_file.summary = summary_result.summary
                            summarized_files += 1
                            
                        self.logger.info(
                            "File summarized",
                            path=processed_file.path,
                            original_size=processed_file.total_size,
                            summary_length=len(summary_result.summary) if summary_result else 0
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to summarize file {processed_file.path}: {e}")
                
            except Exception as e:
                self.logger.error(f"Failed to process file {file_context.path}: {e}")
                # Fallback to basic processing
                processed_files.append({
                    "path": file_context.path,
                    "size": len(file_context.content),
                    "type": "unknown",
                    "error": str(e)
                })
        
        # Calculate token estimates
        estimated_tokens = total_chars // 4  # Rough estimation
        requires_summarization = estimated_tokens > self.settings.max_context_tokens * 0.6
        
        analysis = {
            "file_count": len(files),
            "total_characters": total_chars,
            "estimated_tokens": estimated_tokens,
            "file_types": file_types,
            "requires_summarization": requires_summarization,
            "processed_files": processed_files,
            "total_chunks": total_chunks,
            "summarized_files": summarized_files,
            "processing_stats": {
                "avg_chunks_per_file": total_chunks / len(files) if files else 0,
                "summarization_rate": summarized_files / len(files) if files else 0,
                "token_density": estimated_tokens / total_chars if total_chars > 0 else 0
            }
        }
        
        self.logger.info(
            "File processing completed",
            file_count=len(files),
            total_chunks=total_chunks,
            summarized_files=summarized_files,
            estimated_tokens=estimated_tokens
        )
        
        return analysis
    
    async def _analyze_context_requirements(
        self, 
        request: ChatRequest,
        intent_analysis: Dict[str, Any],
        file_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze context requirements for the request."""
        
        # Base token count for message
        message_tokens = len(request.message) // 4  # Rough estimation
        
        # Add file context tokens
        file_tokens = file_analysis.get("estimated_tokens", 0)
        
        # Add overhead for system prompts and formatting
        overhead_tokens = 200  # Base overhead
        
        # Add extra overhead based on complexity
        if intent_analysis["intent"] in ["task", "analysis"]:
            overhead_tokens += 300
        
        total_estimated_tokens = message_tokens + file_tokens + overhead_tokens
        
        return {
            "message_tokens": message_tokens,
            "file_tokens": file_tokens,
            "overhead_tokens": overhead_tokens,
            "estimated_tokens": total_estimated_tokens,
            "within_limits": total_estimated_tokens <= self.settings.max_context_tokens,
            "utilization_percent": (total_estimated_tokens / self.settings.max_context_tokens) * 100
        }
    
    async def _assess_complexity(
        self,
        request: ChatRequest,
        intent_analysis: Dict[str, Any],
        file_analysis: Dict[str, Any]
    ) -> str:
        """Assess the complexity of the request."""
        
        complexity_score = 0
        
        # Intent-based complexity
        intent_complexity = {
            "conversation": 1,
            "question": 2,
            "request": 3,
            "analysis": 4,
            "task": 5
        }
        complexity_score += intent_complexity.get(intent_analysis["intent"], 3)
        
        # Message length complexity
        word_count = intent_analysis["word_count"]
        if word_count > 100:
            complexity_score += 2
        elif word_count > 50:
            complexity_score += 1
        
        # File context complexity
        if file_analysis["file_count"] > 0:
            complexity_score += 1
            if file_analysis["file_count"] > 3:
                complexity_score += 1
            if file_analysis["requires_summarization"]:
                complexity_score += 2
        
        # Model-specific complexity
        if request.model and "gpt-4" in request.model.lower():
            complexity_score -= 1  # More capable model
        
        # Map score to complexity level
        if complexity_score <= 3:
            return "simple"
        elif complexity_score <= 6:
            return "moderate"
        else:
            return "complex"
    
    def _extract_topics(self, message: str) -> List[str]:
        """Extract potential topics from the message."""
        # Simple keyword-based topic extraction
        topic_keywords = {
            "programming": ["code", "function", "class", "variable", "programming", "development"],
            "data": ["data", "database", "query", "analysis", "statistics"],
            "ai": ["ai", "machine learning", "neural network", "model", "training"],
            "web": ["html", "css", "javascript", "web", "frontend", "backend"],
            "documentation": ["documentation", "readme", "docs", "guide", "tutorial"]
        }
        
        found_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in message for keyword in keywords):
                found_topics.append(topic)
        
        return found_topics
    
    def _extract_entities(self, message: str) -> List[str]:
        """Extract named entities from the message."""
        # Simple pattern-based entity extraction
        entities = []
        
        # File paths
        file_patterns = [
            r'[a-zA-Z0-9_/.-]+\.[a-zA-Z]{2,4}',  # Files with extensions
            r'/[a-zA-Z0-9_/.-]+',  # Unix paths
            r'[A-Z]:[\\a-zA-Z0-9_\\.-]+',  # Windows paths
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, message)
            entities.extend(matches)
        
        # URLs
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, message)
        entities.extend(urls)
        
        return list(set(entities))  # Remove duplicates
    
    def _needs_external_calls(self, intent_analysis: Dict[str, Any]) -> bool:
        """Determine if external calls might be needed."""
        # Simple heuristic - more sophisticated logic can be added
        complex_intents = ["task", "analysis"]
        return intent_analysis["intent"] in complex_intents
    
    def _build_understanding_summary(
        self,
        intent_analysis: Dict[str, Any],
        file_analysis: Dict[str, Any],
        context_analysis: Dict[str, Any],
        complexity: str
    ) -> str:
        """Build a human-readable summary of the understanding."""
        
        summary_parts = [
            f"Intent: {intent_analysis['intent']} (confidence: {intent_analysis['confidence']:.2f})",
            f"Complexity: {complexity}",
            f"Context tokens: {context_analysis['estimated_tokens']}"
        ]
        
        if file_analysis["file_count"] > 0:
            file_summary = f"Files: {file_analysis['file_count']} files"
            if file_analysis.get("total_chunks", 0) > 0:
                file_summary += f", {file_analysis['total_chunks']} chunks"
            if file_analysis.get("summarized_files", 0) > 0:
                file_summary += f", {file_analysis['summarized_files']} summarized"
            if file_analysis["file_types"]:
                file_summary += f" ({', '.join(file_analysis['file_types'])})"
            summary_parts.append(file_summary)
        
        if intent_analysis["topics"]:
            summary_parts.append(f"Topics: {', '.join(intent_analysis['topics'])}")
        
        return " | ".join(summary_parts) 