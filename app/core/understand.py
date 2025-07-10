"""Understanding phase - Parse and analyze user input with context."""

import re
import time
from typing import Dict, Any, Optional, List

from app.schemas import ChatRequest, UPEEResult, UPEEPhase, FileContext, FileContent, ConversationMessage
from app.core.file_processor import file_processor
from app.core.file_processing_agent import get_file_processing_agent
from app.llm_providers import LLMProviderManager
from app.settings import Settings
from app.utils.logging_config import get_logger


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
        self.llm_manager = LLMProviderManager(settings)
    
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
            
            # Process conversation history for short-term memory
            history_analysis = file_processor.process_conversation_history(
                request.history, request.memory_limit or 10
            )
            
            # Ensure all expected keys exist
            if "history_truncated" not in history_analysis:
                history_analysis["history_truncated"] = False
            if "original_message_count" not in history_analysis:
                history_analysis["original_message_count"] = history_analysis.get("message_count", 0)
            
            # Process file contexts
            file_analysis = await self._process_file_contexts(request.files)
            
            # Create context summary
            context_summary = file_processor.create_context_summary(
                file_analysis.get("processed_files", []), 
                history_analysis
            )
            
            # Calculate context requirements
            context_analysis = await self._analyze_context_requirements(
                request, intent_analysis, file_analysis, history_analysis
            )
            
            # Determine complexity
            complexity = await self._assess_complexity(
                request, intent_analysis, file_analysis, history_analysis
            )
            
            # Build understanding result
            content = self._build_understanding_summary(
                intent_analysis, file_analysis, context_analysis, complexity, history_analysis, context_summary
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
                    "processed_files_count": len(file_analysis.get("processed_files", [])),
                    "file_types": [f.get("file_type") for f in file_analysis.get("processed_files", [])],
                    "total_file_size": sum(f.get("file_size", 0) for f in file_analysis.get("processed_files", [])),
                    "files_with_content": sum(1 for f in file_analysis.get("processed_files", []) if f.get("content")),
                    "files_with_signed_urls": sum(1 for f in file_analysis.get("processed_files", []) if f.get("source") == "signed_url"),
                    "agentic_processing": {
                        "files_requiring_agent": file_analysis.get("agentic_files_count", 0),
                        "successfully_processed": file_analysis.get("successfully_processed_agentic", 0),
                        "agentic_results": file_analysis.get("agentic_processing_results", [])
                    }
                },
                "processed_files": file_analysis.get("processed_files", []),
                # Short-term memory metadata
                "conversation_history": {
                    "message_count": history_analysis["message_count"],
                    "total_tokens_estimate": history_analysis["total_tokens_estimate"],
                    "files_in_history": len(history_analysis["files_in_history"]),
                    "history_truncated": history_analysis["history_truncated"],
                    "original_message_count": history_analysis["original_message_count"]
                },
                "context_summary": context_summary
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
    
    async def _process_file_contexts(self, files) -> Dict[str, Any]:
        """Process file contexts using the new file processor and agentic workflow."""
        if not files:
            return {
                "file_count": 0,
                "total_characters": 0,
                "estimated_tokens": 0,
                "processed_files": [],
                "agentic_processing_results": []
            }
        
        self.logger.info(f"Processing {len(files)} file contexts")
        
        # Use the standard file processor first
        processed_files = await file_processor.process_files(files)
        
        # Identify files that need agentic processing
        agentic_results = []
        files_needing_agent = [
            (f, original_file) for f, original_file in zip(processed_files, files)
            if f.get("requires_agentic_processing", False)
        ]
        
        if files_needing_agent:
            self.logger.info(
                f"Found {len(files_needing_agent)} files requiring agentic processing",
                files=[f[0]["file_name"] for f in files_needing_agent]
            )
            
            # Get the file processing agent
            agent = get_file_processing_agent(self.settings, self.llm_manager)
            
            # Process complex files with agentic workflow
            for file_info, original_file in files_needing_agent:
                try:
                    self.logger.info(
                        f"Starting agentic processing for {file_info['file_name']}",
                        file_type=file_info.get("file_type"),
                        file_size=file_info.get("file_size")
                    )
                    
                    # Generate a request ID for this file processing
                    import uuid
                    file_request_id = f"file-{uuid.uuid4().hex[:8]}"
                    
                    # Process with agentic workflow
                    agent_result = await agent.process_file(original_file, file_request_id)
                    
                    # Update file info with agent results
                    file_info.update({
                        "processed": agent_result.success,
                        "processing_status": "completed_agentic_workflow",
                        "agentic_result": {
                            "success": agent_result.success,
                            "confidence_score": agent_result.confidence_score,
                            "execution_time": agent_result.execution_time,
                            "tools_used": list(agent_result.tool_outputs.keys()),
                            "errors": agent_result.errors
                        }
                    })
                    
                    # If successful, update content
                    if agent_result.success and agent_result.extracted_content:
                        file_info["content"] = agent_result.extracted_content
                        file_info["char_count"] = len(agent_result.extracted_content)
                        file_info["line_count"] = len(agent_result.extracted_content.splitlines())
                    
                    agentic_results.append({
                        "file_name": file_info["file_name"],
                        "success": agent_result.success,
                        "confidence": agent_result.confidence_score,
                        "processing_time": agent_result.execution_time,
                        "content_extracted": bool(agent_result.extracted_content)
                    })
                    
                    self.logger.info(
                        f"Agentic processing completed for {file_info['file_name']}",
                        success=agent_result.success,
                        confidence=agent_result.confidence_score,
                        content_length=len(agent_result.extracted_content) if agent_result.extracted_content else 0
                    )
                    
                except Exception as e:
                    self.logger.error(
                        f"Agentic processing failed for {file_info['file_name']}: {str(e)}",
                        exc_info=True
                    )
                    
                    file_info.update({
                        "processed": False,
                        "processing_status": "agentic_workflow_failed",
                        "agentic_result": {
                            "success": False,
                            "error": str(e),
                            "confidence_score": 0.0
                        }
                    })
                    
                    agentic_results.append({
                        "file_name": file_info["file_name"],
                        "success": False,
                        "error": str(e),
                        "confidence": 0.0
                    })
        
        # Calculate totals including agentic processing results
        total_chars = sum(f.get("char_count", 0) for f in processed_files if f.get("processed"))
        estimated_tokens = total_chars // 4  # Rough estimation
        
        return {
            "file_count": len(files),
            "total_characters": total_chars,
            "estimated_tokens": estimated_tokens,
            "processed_files": processed_files,
            "agentic_processing_results": agentic_results,
            "agentic_files_count": len(files_needing_agent),
            "successfully_processed_agentic": sum(1 for r in agentic_results if r.get("success", False))
        }
    
    async def _analyze_context_requirements(
        self, 
        request: ChatRequest,
        intent_analysis: Dict[str, Any],
        file_analysis: Dict[str, Any],
        history_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze context requirements for the request."""
        
        # Base token count for message
        message_tokens = len(request.message) // 4  # Rough estimation
        
        # Add file context tokens
        file_tokens = file_analysis.get("estimated_tokens", 0)
        
        # Add conversation history tokens
        history_tokens = history_analysis.get("total_tokens_estimate", 0)
        
        # Add overhead for system prompts and formatting
        overhead_tokens = 200  # Base overhead
        
        # Add extra overhead based on complexity
        if intent_analysis["intent"] in ["task", "analysis"]:
            overhead_tokens += 300
        
        # Add overhead for file processing
        if file_analysis.get("file_count", 0) > 0:
            overhead_tokens += 100
        
        # Add overhead for conversation history
        if history_analysis.get("message_count", 0) > 0:
            overhead_tokens += 150
        
        total_estimated_tokens = message_tokens + file_tokens + history_tokens + overhead_tokens
        max_context = request.context_window_size or self.settings.max_context_tokens
        
        return {
            "message_tokens": message_tokens,
            "file_tokens": file_tokens,
            "history_tokens": history_tokens,
            "overhead_tokens": overhead_tokens,
            "estimated_tokens": total_estimated_tokens,
            "max_context_tokens": max_context,
            "within_limits": total_estimated_tokens <= max_context,
            "utilization_percent": (total_estimated_tokens / max_context) * 100
        }
    
    async def _assess_complexity(
        self,
        request: ChatRequest,
        intent_analysis: Dict[str, Any],
        file_analysis: Dict[str, Any],
        history_analysis: Dict[str, Any]
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
            if file_analysis["total_characters"] > 10000:
                complexity_score += 2
        
        # Conversation history complexity
        if history_analysis["message_count"] > 5:
            complexity_score += 1
        if history_analysis["total_tokens_estimate"] > 1000:
            complexity_score += 1
        
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
        complexity: str,
        history_analysis: Dict[str, Any],
        context_summary: str
    ) -> str:
        """Build a human-readable summary of the understanding."""
        
        summary_parts = [
            f"Intent: {intent_analysis['intent']} (confidence: {intent_analysis['confidence']:.2f})",
            f"Complexity: {complexity}",
            f"Context tokens: {context_analysis['estimated_tokens']}"
        ]
        
        if file_analysis["file_count"] > 0:
            file_summary = f"Files: {file_analysis['file_count']} files"
            successful_files = [f for f in file_analysis["processed_files"] if f.get("processed")]
            if successful_files:
                file_summary += f" ({len(successful_files)} processed)"
            summary_parts.append(file_summary)
        
        if history_analysis["message_count"] > 0:
            history_summary = f"History: {history_analysis['message_count']} messages"
            if history_analysis["history_truncated"]:
                history_summary += f" (truncated from {history_analysis['original_message_count']})"
            summary_parts.append(history_summary)
        
        if intent_analysis["topics"]:
            summary_parts.append(f"Topics: {', '.join(intent_analysis['topics'])}")
        
        # Add context summary as separate section
        full_summary = " | ".join(summary_parts)
        if context_summary.strip():
            full_summary += f"\n\n{context_summary}"
        
        return full_summary 