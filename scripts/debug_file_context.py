#!/usr/bin/env python3
"""
Debug script to show exactly what file context the agent receives before UPEE processing.
"""

import asyncio
import base64
import json
import os
import sys
from io import BytesIO

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.schemas import ChatRequest, FileContent
from app.core.upee_engine import UPEEEngine
from app.settings import get_settings
from app.utils.logging_config import get_logger

logger = get_logger("debug_file_context")


async def create_test_excel_file():
    """Create a test Excel file identical to your campaign report."""
    try:
        import pandas as pd
        
        # Create sample campaign data matching your file structure
        campaign_data = {
            'Date': ['2025-07-02', '2025-07-03', '2025-07-04', '2025-07-05', '2025-07-06'],
            'Campaign': ['Sinsuru_USA', 'Sinsuru_USA', 'Sinsuru_USA', 'Sinsuru_USA', 'Sinsuru_USA'],
            'Impressions': [15000, 18000, 12000, 22000, 16000],
            'Clicks': [450, 540, 360, 660, 480],
            'Conversions': [23, 28, 18, 34, 25],
            'Cost': [892.50, 1070.40, 713.60, 1308.80, 952.00],
            'CTR': [3.0, 3.0, 3.0, 3.0, 3.0],
            'CPA': [38.80, 38.23, 39.64, 38.49, 38.08]
        }
        
        # Create DataFrame
        df = pd.DataFrame(campaign_data)
        
        # Save to BytesIO as Excel file
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Campaign_Report', index=False)
        
        # Get the Excel content as base64
        excel_buffer.seek(0)
        excel_content = excel_buffer.getvalue()
        base64_content = base64.b64encode(excel_content).decode('utf-8')
        
        return base64_content, df
        
    except ImportError:
        print("❌ Pandas/openpyxl not available for test file creation")
        return None, None


async def debug_upee_context():
    """Debug exactly what context UPEE receives for file processing."""
    print("🔍 Debugging UPEE File Context Reception\n")
    
    # Create test Excel file
    excel_content, original_df = await create_test_excel_file()
    
    if not excel_content:
        print("❌ Could not create test Excel file")
        return False
    
    print(f"✅ Created test Excel file ({len(excel_content)} base64 chars)")
    print(f"   Original data: {len(original_df)} rows × {len(original_df.columns)} columns")
    print(f"   Base64 preview: {excel_content[:50]}...\n")
    
    # Create FileContent exactly as the client would send it
    excel_file = FileContent(
        file_name="신스루_USA-Campaign Report-(2025-07-02 to 2025-07-09).xlsx",
        content=excel_content,  # Base64 encoded Excel content
        file_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        file_size=len(excel_content)
    )
    
    # Create ChatRequest with file
    chat_request = ChatRequest(
        message="Analyze this campaign report and suggest improvements for Sinsuru USA campaign performance",
        files=[excel_file],
        show_thinking=True
    )
    
    print("📋 ChatRequest Created:")
    print(f"   Message: {chat_request.message}")
    print(f"   Files count: {len(chat_request.files)}")
    print(f"   File name: {chat_request.files[0].file_name}")
    print(f"   File type: {chat_request.files[0].file_type}")
    print(f"   File size: {chat_request.files[0].file_size}")
    print(f"   Has content: {bool(chat_request.files[0].content)}")
    print(f"   Content length: {len(chat_request.files[0].content) if chat_request.files[0].content else 0}")
    print()
    
    # Add debug logging to understand phase
    from app.core.understand import UnderstandPhase
    
    # Monkey patch the understand phase to log more details
    original_process = UnderstandPhase.process
    
    async def debug_process(self, request, request_id):
        print("🔍 UNDERSTAND PHASE DEBUG:")
        print(f"   Request ID: {request_id}")
        print(f"   Message: {request.message}")
        print(f"   Files received: {len(request.files) if request.files else 0}")
        
        if request.files:
            for i, file_item in enumerate(request.files):
                print(f"   File {i+1}:")
                print(f"     Name: {file_item.file_name}")
                print(f"     Type: {file_item.file_type}")
                print(f"     Size: {file_item.file_size}")
                print(f"     Has content: {bool(file_item.content)}")
                print(f"     Content type: {type(file_item.content).__name__}")
                if file_item.content:
                    print(f"     Content length: {len(file_item.content)}")
                    print(f"     Content preview: {file_item.content[:100]}...")
                else:
                    print(f"     Content: None")
                print(f"     Has signed_url: {bool(file_item.signed_url)}")
        
        # Call the original process method
        result = await original_process(self, request, request_id)
        
        print(f"\n📊 UNDERSTAND PHASE RESULT:")
        print(f"   Phase: {result.phase}")
        print(f"   Completed: {result.completed}")
        print(f"   Content: {result.content}")
        print(f"   Metadata keys: {list(result.metadata.keys())}")
        
        if "processed_files" in result.metadata:
            processed_files = result.metadata["processed_files"]
            print(f"   Processed files: {len(processed_files)}")
            for i, pf in enumerate(processed_files):
                print(f"     File {i+1}:")
                print(f"       Name: {pf.get('file_name', 'unknown')}")
                print(f"       Processed: {pf.get('processed', False)}")
                print(f"       Has content: {bool(pf.get('content'))}")
                if pf.get('content'):
                    content_len = len(pf['content'])
                    print(f"       Content length: {content_len}")
                    print(f"       Content preview: {pf['content'][:200]}...")
                print(f"       Status: {pf.get('processing_status', 'unknown')}")
                
                # Check agentic processing results
                if 'agentic_result' in pf:
                    agentic = pf['agentic_result']
                    print(f"       Agentic success: {agentic.get('success', False)}")
                    print(f"       Agentic confidence: {agentic.get('confidence_score', 0.0)}")
                    print(f"       Agentic tools: {agentic.get('tools_used', [])}")
                    
        return result
    
    # Apply the debug patch
    UnderstandPhase.process = debug_process
    
    # Initialize UPEE engine
    settings = get_settings()
    upee_engine = UPEEEngine(settings)
    
    request_id = "debug-file-context-001"
    
    print("🚀 Starting UPEE Processing with Debug Logging...\n")
    
    try:
        # Process with detailed logging
        async for event in upee_engine.process_request(chat_request, request_id):
            event_type = event.get("event")
            
            if event_type == "thinking":
                print(f"💭 {event.get('data', '')}")
            elif event_type == "content":
                data = event.get('data', '')
                if isinstance(data, str):
                    try:
                        data_obj = json.loads(data)
                        content = data_obj.get('content', '')
                        print(f"📝 Content: {content[:100]}{'...' if len(content) > 100 else ''}")
                    except:
                        print(f"📝 Content: {data[:100]}{'...' if len(data) > 100 else ''}")
            elif event_type == "complete":
                print(f"✅ Complete: {event.get('data', '')}")
                break
            elif event_type == "error":
                print(f"❌ Error: {event.get('data', '')}")
                break
    
    except Exception as e:
        print(f"❌ UPEE Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    print("📋 DEBUG SUMMARY:")
    print("="*80)
    print("This debug shows exactly what file context the agent receives.")
    print("Look for:")
    print("1. Whether file content is properly received in ChatRequest")
    print("2. Whether the understand phase processes the file")
    print("3. Whether agentic processing extracts meaningful content")
    print("4. Whether the final LLM receives the extracted content")
    
    return True


async def main():
    """Run the debug analysis."""
    try:
        success = await debug_upee_context()
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)