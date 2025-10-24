"""
Example plugin implementations.
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime, timezone

from .base.plugin_base import BasePlugin, PluginMetadata, PluginConfig, PluginType
from .base.interfaces import UPEEPluginInterface, WorkerPluginInterface

logger = logging.getLogger(__name__)


class ExampleUPEEEnhancer(BasePlugin, UPEEPluginInterface):
    """Example UPEE enhancement plugin."""
    
    METADATA = PluginMetadata(
        id="example-upee-enhancer",
        name="Example UPEE Enhancer",
        version="1.0.0",
        plugin_type=PluginType.UPEE,
        description="Example plugin that enhances UPEE loop phases",
        author="PAF Core Team",
        dependencies=[],
        required_capabilities=["text_processing", "llm_integration"],
        tags=["upee", "enhancement", "example"],
        license="MIT"
    )
    
    async def initialize(self) -> None:
        """Initialize the UPEE enhancer."""
        logger.info(f"Initializing {self.metadata.name}")
        self.enhancement_level = self.config.config.get("enhancement_level", "basic")
        self.custom_prompts = self.config.config.get("custom_prompts", {})
    
    async def activate(self) -> None:
        """Activate the UPEE enhancer."""
        logger.info(f"Activating {self.metadata.name}")
        # Register hooks for UPEE phases
        self.register_hook("upee_understand", self.enhance_understand)
        self.register_hook("upee_plan", self.enhance_plan)
        self.register_hook("upee_execute", self.enhance_execute)
        self.register_hook("upee_evaluate", self.enhance_evaluate)
    
    async def deactivate(self) -> None:
        """Deactivate the UPEE enhancer."""
        logger.info(f"Deactivating {self.metadata.name}")
    
    async def cleanup(self) -> None:
        """Cleanup the UPEE enhancer."""
        logger.info(f"Cleaning up {self.metadata.name}")
    
    async def enhance_understand(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance the Understand phase."""
        self.update_activity()
        
        enhanced_context = context.copy()
        
        # Add context analysis
        enhanced_context["enhanced_analysis"] = {
            "sentiment": "neutral",  # Would use actual sentiment analysis
            "complexity": self._analyze_complexity(context.get("user_input", "")),
            "enhancement_level": self.enhancement_level,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Add custom prompt if configured
        if "understand" in self.custom_prompts:
            enhanced_context["custom_prompt"] = self.custom_prompts["understand"]
        
        logger.debug(f"Enhanced understand phase for: {context.get('conversation_id')}")
        return enhanced_context
    
    async def enhance_plan(self, understanding: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance the Plan phase."""
        self.update_activity()
        
        enhanced_plan = understanding.copy()
        
        # Add planning enhancements
        enhanced_plan["planning_enhancements"] = {
            "strategy": "multi_step" if self.enhancement_level == "advanced" else "single_step",
            "resource_requirements": self._estimate_resources(understanding),
            "fallback_options": self._generate_fallbacks(understanding)
        }
        
        if "plan" in self.custom_prompts:
            enhanced_plan["custom_prompt"] = self.custom_prompts["plan"]
        
        logger.debug(f"Enhanced plan phase")
        return enhanced_plan
    
    async def enhance_execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance the Execute phase."""
        self.update_activity()
        
        enhanced_execution = plan.copy()
        
        # Add execution monitoring
        enhanced_execution["execution_monitoring"] = {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "performance_tracking": True,
            "error_handling": "enhanced" if self.enhancement_level == "advanced" else "basic"
        }
        
        if "execute" in self.custom_prompts:
            enhanced_execution["custom_prompt"] = self.custom_prompts["execute"]
        
        logger.debug(f"Enhanced execute phase")
        return enhanced_execution
    
    async def enhance_evaluate(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance the Evaluate phase."""
        self.update_activity()
        
        enhanced_evaluation = execution_result.copy()
        
        # Add comprehensive evaluation
        enhanced_evaluation["evaluation_enhancements"] = {
            "quality_score": self._calculate_quality_score(execution_result),
            "improvement_suggestions": self._generate_improvements(execution_result),
            "learning_insights": self._extract_insights(execution_result),
            "performance_metrics": {
                "plugin_executions": self.execution_count,
                "plugin_errors": self.error_count
            }
        }
        
        if "evaluate" in self.custom_prompts:
            enhanced_evaluation["custom_prompt"] = self.custom_prompts["evaluate"]
        
        logger.debug(f"Enhanced evaluate phase")
        return enhanced_evaluation
    
    def _analyze_complexity(self, text: str) -> str:
        """Analyze text complexity."""
        word_count = len(text.split())
        if word_count < 10:
            return "low"
        elif word_count < 50:
            return "medium"
        else:
            return "high"
    
    def _estimate_resources(self, understanding: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate resource requirements."""
        complexity = understanding.get("enhanced_analysis", {}).get("complexity", "medium")
        
        resource_map = {
            "low": {"cpu": "minimal", "memory": "low", "time": "fast"},
            "medium": {"cpu": "moderate", "memory": "medium", "time": "normal"},
            "high": {"cpu": "intensive", "memory": "high", "time": "extended"}
        }
        
        return resource_map.get(complexity, resource_map["medium"])
    
    def _generate_fallbacks(self, understanding: Dict[str, Any]) -> List[str]:
        """Generate fallback options."""
        return [
            "simplified_response",
            "request_clarification", 
            "delegate_to_human"
        ]
    
    def _calculate_quality_score(self, result: Dict[str, Any]) -> float:
        """Calculate quality score."""
        # Simple scoring based on presence of key fields
        score = 0.0
        if result.get("response"):
            score += 0.4
        if result.get("confidence", 0) > 0.7:
            score += 0.3
        if not result.get("errors"):
            score += 0.3
        
        return min(score, 1.0)
    
    def _generate_improvements(self, result: Dict[str, Any]) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []
        
        if result.get("confidence", 1.0) < 0.8:
            suggestions.append("Increase confidence through additional context")
        
        if result.get("errors"):
            suggestions.append("Implement better error handling")
        
        if not suggestions:
            suggestions.append("Consider response personalization")
        
        return suggestions
    
    def _extract_insights(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract learning insights."""
        return {
            "response_length": len(str(result.get("response", ""))),
            "processing_time": result.get("processing_time", 0),
            "user_satisfaction_predicted": "high" if result.get("confidence", 0) > 0.8 else "medium"
        }


class ExampleTextWorker(BasePlugin, WorkerPluginInterface):
    """Example text processing worker plugin."""
    
    METADATA = PluginMetadata(
        id="example-text-worker",
        name="Example Text Worker",
        version="1.0.0",
        plugin_type=PluginType.WORKER,
        description="Example worker plugin for text processing tasks",
        author="PAF Core Team",
        dependencies=[],
        required_capabilities=["text_processing", "nlp"],
        tags=["worker", "text", "nlp", "example"],
        license="MIT"
    )
    
    def __init__(self, metadata: PluginMetadata, config: PluginConfig):
        super().__init__(metadata, config)
        self.current_tasks = 0
        self.max_concurrent_tasks = 5
        self.supported_tasks = {
            "text_analysis",
            "text_summarization", 
            "text_translation",
            "text_classification"
        }
    
    async def initialize(self) -> None:
        """Initialize the text worker."""
        logger.info(f"Initializing {self.metadata.name}")
        self.max_text_length = self.config.config.get("max_text_length", 10000)
        self.supported_languages = self.config.config.get("supported_languages", ["en"])
    
    async def activate(self) -> None:
        """Activate the text worker."""
        logger.info(f"Activating {self.metadata.name}")
        # Register for worker task events
        self.register_hook("worker_task_request", self.handle_task_request)
    
    async def deactivate(self) -> None:
        """Deactivate the text worker."""
        logger.info(f"Deactivating {self.metadata.name}")
    
    async def cleanup(self) -> None:
        """Cleanup the text worker."""
        logger.info(f"Cleaning up {self.metadata.name}")
    
    async def can_handle_task(self, task_type: str, task_data: Dict[str, Any]) -> bool:
        """Check if this worker can handle the given task."""
        # Check task type
        if task_type not in self.supported_tasks:
            return False
        
        # Check capacity
        if self.current_tasks >= self.max_concurrent_tasks:
            return False
        
        # Check text length
        text = task_data.get("text", "")
        if len(text) > self.max_text_length:
            return False
        
        # Check language
        language = task_data.get("language", "en")
        if language not in self.supported_languages:
            return False
        
        return True
    
    async def execute_task(
        self, 
        task_type: str, 
        task_data: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute a text processing task."""
        self.update_activity()
        self.current_tasks += 1
        
        try:
            start_time = datetime.now(timezone.utc)
            
            # Route to appropriate handler
            if task_type == "text_analysis":
                result = await self._analyze_text(task_data)
            elif task_type == "text_summarization":
                result = await self._summarize_text(task_data)
            elif task_type == "text_translation":
                result = await self._translate_text(task_data)
            elif task_type == "text_classification":
                result = await self._classify_text(task_data)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
            
            end_time = datetime.now(timezone.utc)
            
            return {
                "success": True,
                "result": result,
                "processing_time": (end_time - start_time).total_seconds(),
                "worker_id": self.instance_id,
                "task_type": task_type
            }
            
        except Exception as e:
            self.increment_error()
            logger.error(f"Error executing task {task_type}: {e}")
            return {
                "success": False,
                "error": str(e),
                "worker_id": self.instance_id,
                "task_type": task_type
            }
        finally:
            self.current_tasks -= 1
    
    async def get_capabilities(self) -> List[str]:
        """Get worker capabilities."""
        return list(self.supported_tasks)
    
    async def get_current_load(self) -> int:
        """Get current workload (0-100)."""
        return int((self.current_tasks / self.max_concurrent_tasks) * 100)
    
    async def get_max_capacity(self) -> int:
        """Get maximum task capacity."""
        return self.max_concurrent_tasks
    
    async def handle_task_request(self, task_request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming task request."""
        task_type = task_request.get("task_type")
        task_data = task_request.get("task_data", {})
        
        if await self.can_handle_task(task_type, task_data):
            return await self.execute_task(task_type, task_data)
        else:
            return {
                "success": False,
                "error": "Cannot handle this task",
                "reason": "capacity_or_requirements"
            }
    
    async def _analyze_text(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze text content."""
        text = task_data.get("text", "")
        
        # Simulate text analysis
        await asyncio.sleep(0.1)  # Simulate processing time
        
        words = text.split()
        sentences = text.split(".")
        
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "reading_level": "intermediate",  # Would use actual algorithm
            "sentiment": "neutral",  # Would use actual sentiment analysis
            "key_topics": ["example", "text", "analysis"]  # Would use actual topic extraction
        }
    
    async def _summarize_text(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize text content."""
        text = task_data.get("text", "")
        max_length = task_data.get("max_length", 100)
        
        # Simulate summarization
        await asyncio.sleep(0.2)
        
        # Simple summarization - take first N words
        words = text.split()
        if len(words) <= max_length:
            summary = text
        else:
            summary = " ".join(words[:max_length]) + "..."
        
        return {
            "summary": summary,
            "original_length": len(words),
            "summary_length": len(summary.split()),
            "compression_ratio": len(summary.split()) / len(words) if words else 0
        }
    
    async def _translate_text(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Translate text content."""
        text = task_data.get("text", "")
        target_language = task_data.get("target_language", "en")
        
        # Simulate translation
        await asyncio.sleep(0.3)
        
        # Mock translation
        translation = f"[{target_language.upper()}] {text}"
        
        return {
            "translated_text": translation,
            "source_language": "auto-detected",
            "target_language": target_language,
            "confidence": 0.95
        }
    
    async def _classify_text(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify text content."""
        text = task_data.get("text", "")
        categories = task_data.get("categories", ["positive", "negative", "neutral"])
        
        # Simulate classification
        await asyncio.sleep(0.15)
        
        # Mock classification
        predicted_category = categories[0] if categories else "unknown"
        
        return {
            "predicted_category": predicted_category,
            "confidence": 0.85,
            "all_scores": {cat: 0.3 for cat in categories},
            "features_used": ["word_count", "sentiment", "keywords"]
        }