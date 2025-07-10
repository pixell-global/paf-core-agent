#!/usr/bin/env python3
"""
Test script for short-term memory and file content support features.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.schemas import (
    ChatRequest, FileContent, FileContext, ConversationMessage,
    UPEEPhase, UPEEResult
)
from app.core.file_processor import file_processor
from app.core.understand import UnderstandPhase
from app.core.plan import PlanPhase
from app.settings import get_settings


async def test_file_processor():
    """Test the file processor with different file types."""
    print("🧪 Testing File Processor...")
    
    # Test FileContent with inline content
    file_content_1 = FileContent(
        file_name="example.py",
        file_path="/path/to/example.py",
        content="def hello():\n    print('Hello, world!')\n\nhello()",
        file_type="text/x-python",
        file_size=50
    )
    
    # Test FileContent with signed URL (mock)
    file_content_2 = FileContent(
        file_name="large_file.txt",
        file_type="text/plain",
        file_size=1000,
        signed_url="https://example.com/signed-url",
        metadata={"source": "cloud_storage"}
    )
    
    # Test legacy FileContext
    file_context = FileContext(
        path="config.json",
        content='{"setting": "value", "debug": true}',
        line_start=1,
        line_end=2
    )
    
    files = [file_content_1, file_content_2, file_context]
    
    # Process files
    processed_files = await file_processor.process_files(files)
    
    print(f"✅ Processed {len(processed_files)} files:")
    for i, file_info in enumerate(processed_files):
        print(f"  {i+1}. {file_info.get('file_name', 'unknown')}")
        print(f"     Type: {file_info.get('file_type', 'unknown')}")
        print(f"     Processed: {file_info.get('processed', False)}")
        if file_info.get('language'):
            print(f"     Language: {file_info['language']}")
        if file_info.get('line_count'):
            print(f"     Lines: {file_info['line_count']}")
        print()
    
    return processed_files


async def test_conversation_history():
    """Test conversation history processing."""
    print("🧪 Testing Conversation History Processing...")
    
    # Create sample conversation history
    history = [
        ConversationMessage(
            role="user",
            content="Hello, can you help me with Python?",
            timestamp=datetime.now()
        ),
        ConversationMessage(
            role="assistant", 
            content="Of course! I'd be happy to help you with Python. What specific topic would you like to learn about?",
            timestamp=datetime.now()
        ),
        ConversationMessage(
            role="user",
            content="I need help with file handling",
            timestamp=datetime.now()
        ),
        ConversationMessage(
            role="assistant",
            content="Great! File handling in Python is straightforward. You can use the `open()` function to read and write files.",
            timestamp=datetime.now()
        ),
        ConversationMessage(
            role="user",
            content="Can you show me an example?",
            timestamp=datetime.now(),
            files=[FileContent(
                file_name="sample.txt",
                content="This is sample file content",
                file_type="text/plain",
                file_size=26
            )]
        )
    ]
    
    # Process conversation history
    memory_limit = 3
    history_info = file_processor.process_conversation_history(history, memory_limit)
    
    print(f"✅ Processed conversation history:")
    print(f"  Total messages: {history_info['message_count']}")
    print(f"  Original count: {history_info['original_message_count']}")
    print(f"  History truncated: {history_info['history_truncated']}")
    print(f"  Total tokens estimate: {history_info['total_tokens_estimate']}")
    print(f"  Files in history: {len(history_info['files_in_history'])}")
    print()
    
    return history_info


async def test_context_summary():
    """Test context summary generation."""
    print("🧪 Testing Context Summary...")
    
    # Use results from previous tests
    processed_files = await test_file_processor()
    history_info = await test_conversation_history()
    
    # Create context summary
    context_summary = file_processor.create_context_summary(processed_files, history_info)
    
    print("✅ Generated context summary:")
    print(context_summary)
    print()
    
    return context_summary


async def test_understand_phase():
    """Test the Understand phase with files and memory."""
    print("🧪 Testing Understand Phase with Files and Memory...")
    
    settings = get_settings()
    understand_phase = UnderstandPhase(settings)
    
    # Create chat request with files and history
    chat_request = ChatRequest(
        message="Analyze these Python files and explain the patterns you see based on our previous conversation",
        files=[
            FileContent(
                file_name="main.py",
                content="import os\nimport sys\n\ndef main():\n    print('Starting application')\n    process_files()\n\nif __name__ == '__main__':\n    main()",
                file_type="text/x-python",
                file_size=120
            ),
            FileContent(
                file_name="utils.py", 
                content="def process_files():\n    \"\"\"Process all files in directory\"\"\"\n    for file in os.listdir('.'):\n        if file.endswith('.py'):\n            print(f'Processing {file}')",
                file_type="text/x-python",
                file_size=150
            )
        ],
        history=[
            ConversationMessage(role="user", content="I'm working on a Python project"),
            ConversationMessage(role="assistant", content="Great! What kind of Python project are you building?"),
            ConversationMessage(role="user", content="A file processing application")
        ],
        memory_limit=5,
        show_thinking=True
    )
    
    # Process through understand phase
    result = await understand_phase.process(chat_request, "test-request-001")
    
    print(f"✅ Understand phase completed:")
    print(f"  Phase: {result.phase}")
    print(f"  Completed: {result.completed}")
    print(f"  Content length: {len(result.content)}")
    print(f"  Intent: {result.metadata.get('intent')}")
    print(f"  Complexity: {result.metadata.get('complexity')}")
    print(f"  File count: {result.metadata.get('file_count')}")
    print(f"  History messages: {result.metadata.get('conversation_history', {}).get('message_count', 0)}")
    print()
    print("Content:")
    print(result.content)
    print()
    
    return result


async def test_plan_phase():
    """Test the Plan phase with enhanced context."""
    print("🧪 Testing Plan Phase with Enhanced Context...")
    
    settings = get_settings()
    plan_phase = PlanPhase(settings)
    
    # Get understanding result first
    understand_result = await test_understand_phase()
    
    # Create chat request
    chat_request = ChatRequest(
        message="Create a comprehensive analysis of the code structure and suggest improvements",
        files=[FileContent(
            file_name="complex_app.py",
            content="# Large complex application code here...\n" * 50,
            file_type="text/x-python",
            file_size=2000
        )],
        history=[ConversationMessage(
            role="user",
            content="I need help optimizing my application",
            timestamp=datetime.now()
        )],
        memory_limit=10,
        temperature=0.7
    )
    
    # Process through plan phase
    result = await plan_phase.process(chat_request, "test-request-002", understand_result)
    
    print(f"✅ Plan phase completed:")
    print(f"  Phase: {result.phase}")
    print(f"  Completed: {result.completed}")
    print(f"  Strategy: {result.metadata.get('strategy')}")
    print(f"  Model: {result.metadata.get('model_recommendation')}")
    print(f"  File processing: {result.metadata.get('file_processing', {}).get('approach')}")
    print(f"  Memory usage: {result.metadata.get('memory_usage', {}).get('approach')}")
    print(f"  External calls needed: {result.metadata.get('needs_external_calls')}")
    print()
    print("Content:")
    print(result.content)
    print()
    
    return result


async def test_api_request_validation():
    """Test API request validation for new features."""
    print("🧪 Testing API Request Validation...")
    
    # Valid request with new features
    valid_request = ChatRequest(
        message="Test message",
        files=[
            FileContent(
                file_name="test.py",
                content="print('hello')",
                file_type="text/x-python",
                file_size=15
            )
        ],
        history=[
            ConversationMessage(role="user", content="Previous message")
        ],
        memory_limit=5
    )
    
    print("✅ Valid request created successfully")
    
    # Test various validation scenarios
    try:
        # Invalid file - no content or signed_url
        invalid_file = FileContent(
            file_name="empty.txt",
            file_type="text/plain", 
            file_size=0
        )
        print("❌ Should have failed validation for empty file")
    except Exception as e:
        print(f"✅ Correctly caught validation error: {e}")
    
    try:
        # Invalid history message
        invalid_history = ConversationMessage(
            role="invalid_role",
            content="test"
        )
        print("❌ Should have failed validation for invalid role")
    except Exception as e:
        print(f"✅ Correctly caught validation error: {e}")
    
    print()


async def main():
    """Run all tests."""
    print("🚀 Starting Short-term Memory and File Content Support Tests\n")
    
    try:
        # Test individual components
        await test_file_processor()
        await test_conversation_history()
        await test_context_summary()
        
        # Test UPEE phases
        await test_understand_phase()
        await test_plan_phase()
        
        # Test API validation
        await test_api_request_validation()
        
        print("🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)