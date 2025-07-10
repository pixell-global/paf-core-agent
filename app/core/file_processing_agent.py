"""
Agentic File Processing Workflow using UPEE pattern.

This agent specializes in understanding, planning, executing, and evaluating 
file processing tasks for any file type through tool selection and execution.
"""

import asyncio
import base64
import json
import logging
import mimetypes
import os
import tempfile
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from app.schemas import FileContent, FileContext, UPEEResult, UPEEPhase
from app.utils.logging_config import get_logger
from app.llm_providers import LLMProviderManager
from app.settings import Settings

logger = get_logger("file_processing_agent")


class FileProcessingPhase(str, Enum):
    """File processing UPEE phases."""
    UNDERSTAND_FILE = "understand_file"
    PLAN_PROCESSING = "plan_processing"
    EXECUTE_PROCESSING = "execute_processing"
    EVALUATE_RESULT = "evaluate_result"


class ProcessingToolType(str, Enum):
    """Types of processing tools available."""
    PANDAS_EXCEL = "pandas_excel"
    PANDAS_CSV = "pandas_csv"
    TEXT_READER = "text_reader"
    JSON_PARSER = "json_parser"
    IMAGE_ANALYZER = "image_analyzer"
    PDF_EXTRACTOR = "pdf_extractor"
    BINARY_ANALYZER = "binary_analyzer"
    CUSTOM_PARSER = "custom_parser"


@dataclass
class ProcessingTool:
    """Definition of a file processing tool."""
    tool_type: ProcessingToolType
    name: str
    description: str
    supported_extensions: List[str]
    supported_mime_types: List[str]
    python_package: str
    installation_command: str
    confidence_score: float = 0.0


@dataclass
class ProcessingPlan:
    """Plan for processing a file."""
    file_analysis: Dict[str, Any]
    selected_tools: List[ProcessingTool]
    processing_strategy: str
    expected_output_format: str
    confidence: float
    steps: List[Dict[str, Any]]


@dataclass
class ProcessingResult:
    """Result of file processing execution."""
    success: bool
    extracted_content: str
    metadata: Dict[str, Any]
    execution_time: float
    confidence_score: float
    errors: List[str]
    tool_outputs: Dict[str, Any]


class FileProcessingAgent:
    """Agentic file processor using UPEE workflow."""
    
    def __init__(self, settings: Settings, llm_manager: LLMProviderManager):
        self.settings = settings
        self.llm_manager = llm_manager
        self.logger = get_logger("file_processing_agent")
        
        # Available processing tools
        self.available_tools = self._initialize_tools()
        
        # Processing thresholds
        self.success_threshold = 0.5
        self.max_retries = 3
    
    def _initialize_tools(self) -> Dict[ProcessingToolType, ProcessingTool]:
        """Initialize available file processing tools."""
        tools = {
            ProcessingToolType.PANDAS_EXCEL: ProcessingTool(
                tool_type=ProcessingToolType.PANDAS_EXCEL,
                name="Pandas Excel Reader",
                description="Read Excel files (.xlsx, .xls) with pandas",
                supported_extensions=[".xlsx", ".xls", ".xlsm", ".xlsb"],
                supported_mime_types=[
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "application/vnd.ms-excel"
                ],
                python_package="pandas[excel]",
                installation_command="pip install pandas openpyxl xlrd"
            ),
            ProcessingToolType.PANDAS_CSV: ProcessingTool(
                tool_type=ProcessingToolType.PANDAS_CSV,
                name="Pandas CSV Reader",
                description="Read CSV files with pandas",
                supported_extensions=[".csv", ".tsv"],
                supported_mime_types=["text/csv", "text/tab-separated-values"],
                python_package="pandas",
                installation_command="pip install pandas"
            ),
            ProcessingToolType.TEXT_READER: ProcessingTool(
                tool_type=ProcessingToolType.TEXT_READER,
                name="Text File Reader",
                description="Read plain text files",
                supported_extensions=[".txt", ".md", ".py", ".js", ".json", ".yaml", ".yml"],
                supported_mime_types=["text/plain", "text/markdown", "application/json"],
                python_package="built-in",
                installation_command="No installation required"
            ),
            ProcessingToolType.JSON_PARSER: ProcessingTool(
                tool_type=ProcessingToolType.JSON_PARSER,
                name="JSON Parser",
                description="Parse and analyze JSON files",
                supported_extensions=[".json", ".jsonl"],
                supported_mime_types=["application/json"],
                python_package="built-in",
                installation_command="No installation required"
            ),
            ProcessingToolType.PDF_EXTRACTOR: ProcessingTool(
                tool_type=ProcessingToolType.PDF_EXTRACTOR,
                name="PDF Text Extractor",
                description="Extract text from PDF files",
                supported_extensions=[".pdf"],
                supported_mime_types=["application/pdf"],
                python_package="PyPDF2",
                installation_command="pip install PyPDF2"
            ),
            ProcessingToolType.IMAGE_ANALYZER: ProcessingTool(
                tool_type=ProcessingToolType.IMAGE_ANALYZER,
                name="Image Analyzer",
                description="Analyze images and extract metadata",
                supported_extensions=[".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"],
                supported_mime_types=["image/jpeg", "image/png", "image/gif"],
                python_package="Pillow",
                installation_command="pip install Pillow"
            ),
            ProcessingToolType.BINARY_ANALYZER: ProcessingTool(
                tool_type=ProcessingToolType.BINARY_ANALYZER,
                name="Binary File Analyzer",
                description="Analyze binary files and extract metadata",
                supported_extensions=[],  # Fallback for unknown types
                supported_mime_types=[],
                python_package="built-in",
                installation_command="No installation required"
            )
        }
        return tools
    
    async def process_file(
        self, 
        file_item: Union[FileContent, FileContext], 
        request_id: str
    ) -> ProcessingResult:
        """
        Main entry point for agentic file processing.
        Uses UPEE workflow to process any file type.
        """
        self.logger.info(
            "Starting agentic file processing",
            request_id=request_id,
            file_name=getattr(file_item, 'file_name', getattr(file_item, 'path', 'unknown'))
        )
        
        # Enforce file size limits (100MB default)
        max_file_size = getattr(self.settings, 'max_file_size', 100 * 1024 * 1024)  # 100MB default
        
        if isinstance(file_item, FileContent):
            file_size = file_item.file_size
            file_name = file_item.file_name
        else:  # FileContext
            content_size = len(file_item.content.encode('utf-8')) if file_item.content else 0
            file_size = content_size
            file_name = os.path.basename(file_item.path)
        
        if file_size > max_file_size:
            error_msg = f"File '{file_name}' exceeds maximum size limit of {max_file_size / (1024*1024):.1f}MB (actual: {file_size / (1024*1024):.1f}MB)"
            self.logger.warning(error_msg, request_id=request_id)
            return ProcessingResult(
                success=False,
                extracted_content="",
                metadata={"error": error_msg, "file_size": file_size, "max_size": max_file_size},
                execution_time=0.0,
                confidence_score=0.0,
                errors=[error_msg],
                tool_outputs={}
            )
        
        attempt = 0
        while attempt < self.max_retries:
            try:
                # UPEE Workflow for File Processing
                understand_result = await self._understand_file(file_item, request_id)
                plan_result = await self._plan_processing(file_item, understand_result, request_id)
                execute_result = await self._execute_processing(file_item, plan_result, request_id)
                evaluate_result = await self._evaluate_result(execute_result, plan_result, request_id)
                
                # Check if result meets threshold
                if evaluate_result.confidence_score >= self.success_threshold:
                    self.logger.info(
                        "File processing completed successfully",
                        request_id=request_id,
                        confidence=evaluate_result.confidence_score,
                        attempt=attempt + 1
                    )
                    return evaluate_result
                else:
                    attempt += 1
                    self.logger.warning(
                        "File processing below threshold, retrying",
                        request_id=request_id,
                        confidence=evaluate_result.confidence_score,
                        threshold=self.success_threshold,
                        attempt=attempt,
                        max_retries=self.max_retries
                    )
                    
                    if attempt < self.max_retries:
                        # Wait before retry
                        await asyncio.sleep(1.0 * attempt)
                        
            except Exception as e:
                attempt += 1
                self.logger.error(
                    "Error in file processing attempt",
                    request_id=request_id,
                    attempt=attempt,
                    error=str(e),
                    exc_info=True
                )
                
                if attempt >= self.max_retries:
                    return ProcessingResult(
                        success=False,
                        extracted_content="",
                        metadata={"error": str(e)},
                        execution_time=0.0,
                        confidence_score=0.0,
                        errors=[f"Failed after {self.max_retries} attempts: {str(e)}"],
                        tool_outputs={}
                    )
        
        # If we get here, all retries failed
        return ProcessingResult(
            success=False,
            extracted_content="",
            metadata={"error": "All processing attempts failed"},
            execution_time=0.0,
            confidence_score=0.0,
            errors=[f"Failed to process file after {self.max_retries} attempts"],
            tool_outputs={}
        )
    
    async def _understand_file(
        self, 
        file_item: Union[FileContent, FileContext], 
        request_id: str
    ) -> Dict[str, Any]:
        """
        UNDERSTAND phase: Analyze the file to determine its type and characteristics.
        """
        self.logger.info("Starting file understanding phase", request_id=request_id)
        
        # Extract file information
        if isinstance(file_item, FileContent):
            file_name = file_item.file_name
            file_size = file_item.file_size
            file_type = file_item.file_type
            content_preview = file_item.content[:500] if file_item.content else None
        else:  # FileContext
            file_name = os.path.basename(file_item.path)
            file_type = mimetypes.guess_type(file_item.path)[0] or "unknown"
            file_size = len(file_item.content.encode('utf-8')) if file_item.content else 0
            content_preview = file_item.content[:500] if file_item.content else None
        
        # Analyze file extension and MIME type
        _, ext = os.path.splitext(file_name.lower())
        
        # Determine potential tools
        compatible_tools = []
        for tool_type, tool in self.available_tools.items():
            if ext in tool.supported_extensions or file_type in tool.supported_mime_types:
                tool.confidence_score = self._calculate_tool_confidence(ext, file_type, tool)
                compatible_tools.append(tool)
        
        # Sort by confidence
        compatible_tools.sort(key=lambda t: t.confidence_score, reverse=True)
        
        # Use LLM to analyze file characteristics
        analysis_prompt = f"""
        Analyze this file and provide insights for processing:
        
        File Name: {file_name}
        File Type: {file_type}
        File Size: {file_size} bytes
        Extension: {ext}
        
        Content Preview:
        {content_preview if content_preview else "No content preview available"}
        
        Based on this information, provide a JSON analysis with:
        1. file_characteristics: What type of data this likely contains
        2. complexity_assessment: How complex the file structure might be
        3. processing_challenges: Potential difficulties in processing
        4. recommended_approach: Suggested processing strategy
        5. expected_structure: What structure the content might have
        
        Return only valid JSON.
        """
        
        try:
            # Get LLM analysis
            analysis_response = await self._call_llm_for_analysis(analysis_prompt, request_id)
            llm_analysis = json.loads(analysis_response) if analysis_response else {}
        except Exception as e:
            self.logger.warning(f"LLM analysis failed: {e}", request_id=request_id)
            llm_analysis = {}
        
        understanding_result = {
            "file_info": {
                "name": file_name,
                "type": file_type,
                "size": file_size,
                "extension": ext
            },
            "compatible_tools": [
                {
                    "tool_type": tool.tool_type.value,
                    "name": tool.name,
                    "confidence": tool.confidence_score,
                    "package": tool.python_package
                }
                for tool in compatible_tools[:5]  # Top 5 tools
            ],
            "llm_analysis": llm_analysis,
            "content_preview": content_preview
        }
        
        self.logger.info(
            "File understanding completed", 
            request_id=request_id,
            compatible_tools=len(compatible_tools),
            top_tool=compatible_tools[0].name if compatible_tools else "none"
        )
        
        return understanding_result
    
    async def _plan_processing(
        self, 
        file_item: Union[FileContent, FileContext], 
        understand_result: Dict[str, Any], 
        request_id: str
    ) -> ProcessingPlan:
        """
        PLAN phase: Create a detailed plan for processing the file.
        """
        self.logger.info("Starting processing planning phase", request_id=request_id)
        
        # Get the top tools from understanding
        compatible_tools = understand_result.get("compatible_tools", [])
        file_info = understand_result.get("file_info", {})
        llm_analysis = understand_result.get("llm_analysis", {})
        
        # Create detailed planning prompt
        planning_prompt = f"""
        Create a detailed processing plan for this file:
        
        File Information:
        {json.dumps(file_info, indent=2)}
        
        Compatible Tools:
        {json.dumps(compatible_tools, indent=2)}
        
        LLM Analysis:
        {json.dumps(llm_analysis, indent=2)}
        
        Create a JSON plan with:
        1. processing_strategy: "sequential", "parallel", or "multi_step"
        2. selected_tools: List of tool types to use (in order)
        3. processing_steps: Detailed steps with parameters
        4. expected_output: What the final output should look like
        5. fallback_options: Alternative approaches if primary fails
        6. confidence: How confident you are in this plan (0.0-1.0)
        
        Consider:
        - Does the file have headers?
        - Multiple sheets/sections?
        - Specific encoding?
        - Data validation needs?
        - Size constraints?
        
        Return only valid JSON.
        """
        
        try:
            planning_response = await self._call_llm_for_analysis(planning_prompt, request_id)
            plan_data = json.loads(planning_response) if planning_response else {}
        except Exception as e:
            self.logger.warning(f"LLM planning failed: {e}", request_id=request_id)
            plan_data = {}
        
        # Select tools based on plan
        selected_tools = []
        for tool_type_str in plan_data.get("selected_tools", []):
            try:
                tool_type = ProcessingToolType(tool_type_str)
                if tool_type in self.available_tools:
                    selected_tools.append(self.available_tools[tool_type])
            except ValueError:
                continue
        
        # Fallback to top compatible tool if no tools selected
        if not selected_tools and compatible_tools:
            top_tool_type = compatible_tools[0]["tool_type"]
            try:
                tool_type = ProcessingToolType(top_tool_type)
                selected_tools = [self.available_tools[tool_type]]
            except (ValueError, KeyError):
                pass
        
        plan = ProcessingPlan(
            file_analysis=understand_result,
            selected_tools=selected_tools,
            processing_strategy=plan_data.get("processing_strategy", "sequential"),
            expected_output_format=plan_data.get("expected_output", "text"),
            confidence=plan_data.get("confidence", 0.5),
            steps=plan_data.get("processing_steps", [])
        )
        
        self.logger.info(
            "Processing plan created",
            request_id=request_id,
            strategy=plan.processing_strategy,
            tools_count=len(plan.selected_tools),
            confidence=plan.confidence
        )
        
        return plan
    
    async def _execute_processing(
        self, 
        file_item: Union[FileContent, FileContext], 
        plan: ProcessingPlan, 
        request_id: str
    ) -> ProcessingResult:
        """
        EXECUTE phase: Execute the processing plan using selected tools.
        """
        self.logger.info("Starting processing execution phase", request_id=request_id)
        
        start_time = datetime.now()
        tool_outputs = {}
        errors = []
        extracted_content = ""
        
        try:
            for tool in plan.selected_tools:
                self.logger.info(
                    f"Executing tool: {tool.name}",
                    request_id=request_id,
                    tool_type=tool.tool_type.value
                )
                
                try:
                    # Execute tool based on type
                    tool_result = await self._execute_tool(tool, file_item, plan, request_id)
                    tool_outputs[tool.tool_type.value] = tool_result
                    
                    # Combine results
                    if tool_result.get("success", False):
                        content = tool_result.get("content", "")
                        if content:
                            extracted_content += f"\n\n--- {tool.name} Output ---\n{content}"
                    else:
                        errors.append(f"{tool.name}: {tool_result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    error_msg = f"Tool {tool.name} failed: {str(e)}"
                    errors.append(error_msg)
                    self.logger.error(error_msg, request_id=request_id, exc_info=True)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Determine success
            success = bool(extracted_content.strip()) and len(errors) == 0
            
            result = ProcessingResult(
                success=success,
                extracted_content=extracted_content.strip(),
                metadata={
                    "plan": {
                        "strategy": plan.processing_strategy,
                        "tools_used": [t.name for t in plan.selected_tools],
                        "confidence": plan.confidence
                    },
                    "execution": {
                        "tools_executed": len(tool_outputs),
                        "errors_count": len(errors)
                    }
                },
                execution_time=execution_time,
                confidence_score=0.8 if success else 0.2,  # Will be refined in evaluation
                errors=errors,
                tool_outputs=tool_outputs
            )
            
            self.logger.info(
                "Processing execution completed",
                request_id=request_id,
                success=success,
                execution_time=execution_time,
                content_length=len(extracted_content)
            )
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Execution failed: {str(e)}"
            self.logger.error(error_msg, request_id=request_id, exc_info=True)
            
            return ProcessingResult(
                success=False,
                extracted_content="",
                metadata={"error": error_msg},
                execution_time=execution_time,
                confidence_score=0.0,
                errors=[error_msg],
                tool_outputs=tool_outputs
            )
    
    async def _evaluate_result(
        self, 
        result: ProcessingResult, 
        plan: ProcessingPlan, 
        request_id: str
    ) -> ProcessingResult:
        """
        EVALUATE phase: Assess the quality of the processing result.
        """
        self.logger.info("Starting result evaluation phase", request_id=request_id)
        
        # Calculate confidence score based on multiple factors
        confidence_factors = {
            "basic_success": 0.3 if result.success else 0.0,
            "content_length": min(len(result.extracted_content) / 1000, 0.3),
            "error_penalty": max(0, 0.2 - len(result.errors) * 0.1),
            "plan_confidence": plan.confidence * 0.2
        }
        
        base_confidence = sum(confidence_factors.values())
        
        # Use LLM to evaluate content quality if we have content
        if result.extracted_content:
            evaluation_prompt = f"""
            Evaluate the quality of this file processing result:
            
            Original Plan Expected Output: {plan.expected_output_format}
            Processing Strategy: {plan.processing_strategy}
            
            Extracted Content (first 1000 chars):
            {result.extracted_content[:1000]}
            
            Processing Metadata:
            {json.dumps(result.metadata, indent=2)}
            
            Errors:
            {result.errors}
            
            Rate the quality on a scale of 0.0 to 1.0 based on:
            1. Content completeness
            2. Structure preservation
            3. Data integrity
            4. Error handling
            5. Format appropriateness
            
            Return a JSON object with:
            - quality_score: float (0.0-1.0)
            - assessment: string explaining the rating
            - improvements: list of suggested improvements
            
            Return only valid JSON.
            """
            
            try:
                eval_response = await self._call_llm_for_analysis(evaluation_prompt, request_id)
                eval_data = json.loads(eval_response) if eval_response else {}
                llm_quality_score = eval_data.get("quality_score", 0.5)
                
                # Combine base confidence with LLM assessment
                final_confidence = (base_confidence + llm_quality_score) / 2
                
                # Update result metadata with evaluation
                result.metadata["evaluation"] = eval_data
                
            except Exception as e:
                self.logger.warning(f"LLM evaluation failed: {e}", request_id=request_id)
                final_confidence = base_confidence
        else:
            final_confidence = base_confidence
        
        # Update confidence score
        result.confidence_score = min(final_confidence, 1.0)
        
        self.logger.info(
            "Result evaluation completed",
            request_id=request_id,
            confidence_score=result.confidence_score,
            threshold=self.success_threshold,
            meets_threshold=result.confidence_score >= self.success_threshold
        )
        
        return result
    
    async def _execute_tool(
        self, 
        tool: ProcessingTool, 
        file_item: Union[FileContent, FileContext], 
        plan: ProcessingPlan, 
        request_id: str
    ) -> Dict[str, Any]:
        """Execute a specific processing tool."""
        
        # Get file content
        if isinstance(file_item, FileContent):
            content = file_item.content
            file_name = file_item.file_name
        else:
            content = file_item.content
            file_name = os.path.basename(file_item.path)
        
        if not content:
            return {"success": False, "error": "No content available"}
        
        try:
            if tool.tool_type == ProcessingToolType.TEXT_READER:
                return {
                    "success": True,
                    "content": content,
                    "metadata": {
                        "length": len(content),
                        "lines": len(content.splitlines()),
                        "encoding": "utf-8"
                    }
                }
            
            elif tool.tool_type == ProcessingToolType.JSON_PARSER:
                try:
                    parsed_json = json.loads(content)
                    formatted_json = json.dumps(parsed_json, indent=2)
                    return {
                        "success": True,
                        "content": f"JSON Structure:\n{formatted_json}",
                        "metadata": {
                            "valid_json": True,
                            "keys": list(parsed_json.keys()) if isinstance(parsed_json, dict) else None,
                            "type": type(parsed_json).__name__
                        }
                    }
                except json.JSONDecodeError as e:
                    return {
                        "success": False,
                        "error": f"Invalid JSON: {str(e)}",
                        "content": content[:500]  # Return partial content
                    }
            
            elif tool.tool_type in [ProcessingToolType.PANDAS_EXCEL, ProcessingToolType.PANDAS_CSV]:
                # Attempt actual Excel/CSV processing
                try:
                    if tool.tool_type == ProcessingToolType.PANDAS_EXCEL:
                        # For Excel files, we need to handle them as binary data
                        # Since we might receive base64 or binary content, try to process it
                        
                        # First, check if we have the required packages
                        try:
                            import pandas as pd
                            import openpyxl
                            
                            # If content looks like base64, decode it
                            if isinstance(content, str) and len(content) > 100:
                                try:
                                    import base64
                                    from io import BytesIO
                                    
                                    # Try to decode as base64
                                    if content.startswith('data:'):
                                        # Handle data URL format
                                        content = content.split(',', 1)[1]
                                    
                                    binary_content = base64.b64decode(content)
                                    excel_file = BytesIO(binary_content)
                                    
                                    # Read Excel file
                                    workbook = pd.ExcelFile(excel_file)
                                    sheet_names = workbook.sheet_names
                                    
                                    # Process each sheet
                                    sheets_data = []
                                    for sheet_name in sheet_names:
                                        df = pd.read_excel(excel_file, sheet_name=sheet_name)
                                        
                                        # Convert to readable text format
                                        sheet_info = f"\n--- Sheet: {sheet_name} ---\n"
                                        sheet_info += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
                                        sheet_info += f"Columns: {list(df.columns)}\n\n"
                                        
                                        # Show first few rows
                                        if not df.empty:
                                            sheet_info += "First 5 rows:\n"
                                            sheet_info += df.head().to_string(index=False)
                                            
                                            # Show summary statistics for numeric columns
                                            numeric_cols = df.select_dtypes(include=['number']).columns
                                            if len(numeric_cols) > 0:
                                                sheet_info += f"\n\nSummary statistics for numeric columns:\n"
                                                sheet_info += df[numeric_cols].describe().to_string()
                                        
                                        sheets_data.append(sheet_info)
                                    
                                    result_content = f"EXCEL FILE ANALYSIS for {file_name}:\n"
                                    result_content += f"Total sheets: {len(sheet_names)}\n"
                                    result_content += "".join(sheets_data)
                                    
                                    return {
                                        "success": True,
                                        "content": result_content,
                                        "metadata": {
                                            "sheets": sheet_names,
                                            "total_sheets": len(sheet_names),
                                            "processing_method": "pandas_excel",
                                            "file_format": "xlsx"
                                        }
                                    }
                                    
                                except Exception as decode_error:
                                    # If base64 decoding fails, treat as text content
                                    pass
                            
                            # If we get here, content might be text representation
                            return {
                                "success": True,
                                "content": f"""
EXCEL FILE DETECTED for {file_name}:

The file appears to be an Excel file but the content format needs verification.
Content type: {type(content).__name__}
Content length: {len(content) if content else 0}

To properly process this Excel file, the system needs:
1. Binary content (base64 encoded or raw bytes)
2. Proper Excel file structure
3. Pandas and openpyxl packages (✅ available)

Current content preview:
{content[:200] if content else 'No content available'}...

RECOMMENDATION: Ensure Excel files are uploaded as binary/base64 content.
                                """,
                                "metadata": {
                                    "tool_available": True,
                                    "content_type": type(content).__name__,
                                    "packages_installed": True
                                }
                            }
                            
                        except ImportError as import_error:
                            # Pandas/openpyxl not available
                            return {
                                "success": False,
                                "error": f"Required packages not available: {import_error}",
                                "content": f"""
EXCEL PROCESSING FAILED for {file_name}:

Required packages are not installed:
- pandas: For Excel file reading
- openpyxl: For .xlsx file support

Installation command: {tool.installation_command}

Please install these packages to enable Excel file processing.
                                """,
                                "metadata": {
                                    "tool_required": tool.python_package,
                                    "missing_packages": str(import_error)
                                }
                            }
                    
                    else:  # CSV processing
                        try:
                            import pandas as pd
                            from io import StringIO
                            
                            # Process CSV content
                            csv_data = StringIO(content)
                            df = pd.read_csv(csv_data)
                            
                            result_content = f"CSV FILE ANALYSIS for {file_name}:\n"
                            result_content += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
                            result_content += f"Columns: {list(df.columns)}\n\n"
                            result_content += "First 10 rows:\n"
                            result_content += df.head(10).to_string(index=False)
                            
                            # Summary statistics
                            numeric_cols = df.select_dtypes(include=['number']).columns
                            if len(numeric_cols) > 0:
                                result_content += f"\n\nSummary statistics:\n"
                                result_content += df[numeric_cols].describe().to_string()
                            
                            return {
                                "success": True,
                                "content": result_content,
                                "metadata": {
                                    "rows": df.shape[0],
                                    "columns": df.shape[1],
                                    "column_names": list(df.columns),
                                    "processing_method": "pandas_csv"
                                }
                            }
                            
                        except ImportError:
                            return {
                                "success": False,
                                "error": "Pandas not available for CSV processing",
                                "content": f"CSV file detected but pandas not installed: {tool.installation_command}"
                            }
                        except Exception as csv_error:
                            return {
                                "success": False,
                                "error": f"CSV parsing failed: {str(csv_error)}",
                                "content": f"Failed to parse CSV: {content[:200]}..."
                            }
                
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Excel/CSV processing failed: {str(e)}",
                        "content": f"Error processing {tool.tool_type.value} file: {str(e)}"
                    }
            
            else:
                # Generic binary analyzer
                return {
                    "success": True,
                    "content": f"""
BINARY FILE ANALYSIS for {file_name}:

File Type: {tool.tool_type.value}
Size: {len(content.encode('utf-8')) if isinstance(content, str) else len(content)} bytes
Tool: {tool.name}

This file requires specialized processing with {tool.python_package}.
Consider implementing specific handler for this file type.
                    """,
                    "metadata": {
                        "tool_needed": tool.name,
                        "binary_file": True
                    }
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
                "content": ""
            }
    
    def _calculate_tool_confidence(self, ext: str, file_type: str, tool: ProcessingTool) -> float:
        """Calculate confidence score for a tool's compatibility with a file."""
        confidence = 0.0
        
        # Extension match
        if ext in tool.supported_extensions:
            confidence += 0.6
        
        # MIME type match
        if file_type in tool.supported_mime_types:
            confidence += 0.4
        
        # Bonus for exact matches
        if ext in tool.supported_extensions and file_type in tool.supported_mime_types:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    async def _call_llm_for_analysis(self, prompt: str, request_id: str) -> Optional[str]:
        """Call LLM for analysis with error handling."""
        try:
            # Use the LLM manager to make the call
            # This is a placeholder - would need to implement actual LLM call
            # For now, return None to indicate LLM analysis unavailable
            self.logger.debug("LLM analysis requested but not implemented", request_id=request_id)
            return None
            
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}", request_id=request_id, exc_info=True)
            return None


# Global file processing agent instance
file_processing_agent: Optional[FileProcessingAgent] = None


def get_file_processing_agent(settings: Settings, llm_manager: LLMProviderManager) -> FileProcessingAgent:
    """Get or create the file processing agent instance."""
    global file_processing_agent
    if file_processing_agent is None:
        file_processing_agent = FileProcessingAgent(settings, llm_manager)
    return file_processing_agent