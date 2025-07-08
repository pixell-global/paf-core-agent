"""Text summarization utilities for large file processing."""

import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from app.llm_providers import LLMProviderManager, LLMRequest, LLMProviderError
from app.utils.file_processor import ProcessedFile, FileChunk, FileType
from app.utils.logging_config import get_logger
from app.settings import Settings

logger = get_logger("text_summarizer")


@dataclass
class SummaryResult:
    """Result of text summarization."""
    original_text: str
    summary: str
    summary_type: str  # "extractive", "abstractive", "structured"
    confidence: float
    token_reduction: float  # Percentage reduction in tokens
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TextSummarizer:
    """Intelligent text summarizer using LLM providers."""
    
    def __init__(self, settings: Settings, llm_manager: LLMProviderManager):
        self.settings = settings
        self.llm_manager = llm_manager
        self.logger = get_logger("text_summarizer")
        
        # Summarization models (prefer faster, cheaper models for summaries)
        self.summary_model_preferences = [
            "gpt-3.5-turbo",
            "gpt-4o-mini", 
            "claude-3-haiku",
            "claude-3-sonnet"
        ]
    
    async def summarize_file(self, processed_file: ProcessedFile) -> Optional[SummaryResult]:
        """Summarize an entire processed file."""
        if not processed_file.chunks:
            return None
        
        self.logger.info(
            "Starting file summarization",
            path=processed_file.path,
            file_type=processed_file.file_type.value,
            chunks_count=len(processed_file.chunks),
            total_size=processed_file.total_size
        )
        
        # If file is small enough, summarize directly
        if processed_file.total_size <= 4000:
            content = '\n'.join(chunk.content for chunk in processed_file.chunks)
            return await self._summarize_text(
                content, 
                processed_file.file_type,
                f"Complete {processed_file.file_type.value} file"
            )
        
        # For larger files, summarize chunks and then create overall summary
        chunk_summaries = []
        important_chunks = sorted(processed_file.chunks, key=lambda x: x.importance, reverse=True)
        
        # Summarize most important chunks first
        for chunk in important_chunks[:10]:  # Limit to top 10 chunks
            if chunk.token_count > 500:  # Only summarize substantial chunks
                chunk_summary = await self._summarize_text(
                    chunk.content,
                    processed_file.file_type,
                    f"File chunk {chunk.chunk_index} ({chunk.chunk_type})"
                )
                if chunk_summary:
                    chunk_summaries.append(chunk_summary.summary)
        
        # Create overall summary from chunk summaries
        if chunk_summaries:
            combined_summaries = '\n\n'.join(chunk_summaries)
            return await self._summarize_text(
                combined_summaries,
                processed_file.file_type,
                f"Overall summary of {processed_file.path}",
                is_meta_summary=True
            )
        
        return None
    
    async def summarize_chunks(self, chunks: List[FileChunk], file_type: FileType) -> List[FileChunk]:
        """Add summaries to file chunks."""
        updated_chunks = []
        
        # Summarize chunks in parallel (but limit concurrency)
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent summarizations
        
        async def summarize_chunk(chunk: FileChunk) -> FileChunk:
            async with semaphore:
                if chunk.token_count > 300:  # Only summarize substantial chunks
                    summary_result = await self._summarize_text(
                        chunk.content,
                        file_type,
                        f"Chunk {chunk.chunk_index} ({chunk.chunk_type})"
                    )
                    if summary_result:
                        chunk.summary = summary_result.summary
                return chunk
        
        # Process chunks concurrently
        tasks = [summarize_chunk(chunk) for chunk in chunks]
        updated_chunks = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid chunks
        valid_chunks = [chunk for chunk in updated_chunks if isinstance(chunk, FileChunk)]
        
        self.logger.info(
            "Chunk summarization completed",
            total_chunks=len(chunks),
            summarized_chunks=len([c for c in valid_chunks if c.summary]),
            file_type=file_type.value
        )
        
        return valid_chunks
    
    async def _summarize_text(
        self, 
        text: str, 
        file_type: FileType, 
        context: str,
        is_meta_summary: bool = False
    ) -> Optional[SummaryResult]:
        """Summarize a piece of text using LLM."""
        if not text.strip():
            return None
        
        # Get the best available model for summarization
        model = await self._get_best_summary_model()
        if not model:
            self.logger.warning("No models available for summarization")
            return None
        
        # Create type-specific prompt
        system_prompt = self._create_summary_prompt(file_type, is_meta_summary)
        user_prompt = self._create_user_prompt(text, context, file_type)
        
        try:
            llm_request = LLMRequest(
                model=model,
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,  # Lower temperature for more consistent summaries
                max_tokens=300,   # Reasonable summary length
                stream=False      # Don't need streaming for summaries
            )
            
            response = await self.llm_manager.get_completion(llm_request)
            
            if response and response.content:
                original_tokens = len(text) // 4  # Rough estimation
                summary_tokens = len(response.content) // 4
                token_reduction = ((original_tokens - summary_tokens) / original_tokens) * 100
                
                return SummaryResult(
                    original_text=text,
                    summary=response.content.strip(),
                    summary_type="abstractive",
                    confidence=0.8,  # Default confidence
                    token_reduction=max(0, token_reduction),
                    metadata={
                        "model_used": model,
                        "file_type": file_type.value,
                        "context": context,
                        "original_tokens": original_tokens,
                        "summary_tokens": summary_tokens
                    }
                )
        
        except LLMProviderError as e:
            self.logger.error(f"LLM summarization failed: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error during summarization: {e}", exc_info=True)
        
        return None
    
    async def _get_best_summary_model(self) -> Optional[str]:
        """Get the best available model for summarization."""
        try:
            available_models = await self.llm_manager.get_all_models()
            available_names = [m["name"] for m in available_models if m["available"]]
            
            # Return first preferred model that's available
            for preferred_model in self.summary_model_preferences:
                if preferred_model in available_names:
                    return preferred_model
            
            # Fallback to any available model
            return available_names[0] if available_names else None
        
        except Exception as e:
            self.logger.error(f"Error getting available models: {e}")
            return None
    
    def _create_summary_prompt(self, file_type: FileType, is_meta_summary: bool) -> str:
        """Create system prompt for summarization based on file type."""
        base_prompt = "You are an expert at creating concise, informative summaries. "
        
        if is_meta_summary:
            return base_prompt + "You will be given multiple summaries to combine into a single, coherent overview. Focus on the main themes, key points, and overall structure."
        
        type_specific_prompts = {
            FileType.PYTHON: "You specialize in summarizing Python code. Focus on main functions, classes, purpose, and key algorithms. Mention imports and dependencies when relevant.",
            
            FileType.JAVASCRIPT: "You specialize in summarizing JavaScript/TypeScript code. Focus on main functions, components, purpose, and key logic. Mention frameworks and dependencies when relevant.",
            
            FileType.MARKDOWN: "You specialize in summarizing Markdown documents. Focus on main sections, key points, and document structure. Preserve important headings and concepts.",
            
            FileType.JSON: "You specialize in summarizing JSON data. Focus on the structure, main data types, key fields, and overall purpose of the data.",
            
            FileType.CSV: "You specialize in summarizing CSV data. Focus on the number of rows/columns, data types, key patterns, and what the dataset represents.",
            
            FileType.TEXT: "You specialize in summarizing text documents. Focus on main topics, key arguments, conclusions, and overall message.",
            
            FileType.LOG: "You specialize in summarizing log files. Focus on error patterns, important events, timeframes, and system behavior.",
        }
        
        return base_prompt + type_specific_prompts.get(file_type, "Summarize the following content clearly and concisely.")
    
    def _create_user_prompt(self, text: str, context: str, file_type: FileType) -> str:
        """Create user prompt for summarization."""
        prompt_parts = [
            f"Please summarize the following {file_type.value} content:",
            f"Context: {context}",
            "",
            "Content to summarize:",
            "```",
            text,
            "```",
            "",
            "Provide a concise summary that captures the main purpose, key components, and important details. Keep it under 200 words."
        ]
        
        return "\n".join(prompt_parts)
    
    def create_extractive_summary(self, text: str, max_sentences: int = 3) -> SummaryResult:
        """Create an extractive summary by selecting key sentences."""
        import re
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return SummaryResult(
                original_text=text,
                summary=text,
                summary_type="extractive",
                confidence=1.0,
                token_reduction=0.0
            )
        
        # Simple scoring: prefer sentences with common words and avoid very short/long ones
        scored_sentences = []
        word_freq = {}
        
        # Count word frequencies
        all_words = text.lower().split()
        for word in all_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score sentences
        for i, sentence in enumerate(sentences):
            words = sentence.lower().split()
            if len(words) < 5 or len(words) > 50:  # Skip very short or long sentences
                continue
            
            # Score based on word frequency and position
            score = sum(word_freq.get(word, 0) for word in words) / len(words)
            score += (1.0 / (i + 1)) * 0.1  # Slight boost for earlier sentences
            
            scored_sentences.append((score, sentence))
        
        # Select top sentences
        scored_sentences.sort(reverse=True)
        selected_sentences = [sent for _, sent in scored_sentences[:max_sentences]]
        
        # Maintain original order
        summary_sentences = []
        for sentence in sentences:
            if sentence in selected_sentences:
                summary_sentences.append(sentence)
        
        summary = '. '.join(summary_sentences) + '.'
        
        original_tokens = len(text) // 4
        summary_tokens = len(summary) // 4
        token_reduction = ((original_tokens - summary_tokens) / original_tokens) * 100
        
        return SummaryResult(
            original_text=text,
            summary=summary,
            summary_type="extractive",
            confidence=0.7,
            token_reduction=max(0, token_reduction),
            metadata={
                "sentences_selected": len(summary_sentences),
                "total_sentences": len(sentences)
            }
        ) 