#!/usr/bin/env python3
"""
Test actual Excel file processing to verify the agent reads files correctly.
"""

import asyncio
import base64
import os
import sys
import time
from io import BytesIO

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.schemas import FileContent
from app.core.file_processing_agent import FileProcessingAgent
from app.settings import get_settings
from app.llm_providers import LLMProviderManager


async def create_test_excel_file():
    """Create a test Excel file with campaign data."""
    try:
        import pandas as pd
        
        # Create sample campaign data
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
            
            # Add a summary sheet
            summary_data = {
                'Metric': ['Total Impressions', 'Total Clicks', 'Total Conversions', 'Total Cost', 'Avg CTR', 'Avg CPA'],
                'Value': [df['Impressions'].sum(), df['Clicks'].sum(), df['Conversions'].sum(), 
                         df['Cost'].sum(), df['CTR'].mean(), df['CPA'].mean()]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Get the Excel content as base64
        excel_buffer.seek(0)
        excel_content = excel_buffer.getvalue()
        base64_content = base64.b64encode(excel_content).decode('utf-8')
        
        return base64_content, df
        
    except ImportError:
        print("❌ Pandas/openpyxl not available for test file creation")
        return None, None


async def test_excel_file_reading():
    """Test if the agent can actually read Excel files."""
    print("🧪 Testing Excel File Reading...")
    
    # Create test Excel file
    excel_content, original_df = await create_test_excel_file()
    
    if not excel_content:
        print("❌ Could not create test Excel file")
        return False
    
    print(f"✅ Created test Excel file with {len(original_df)} rows of campaign data")
    
    # Create FileContent with the Excel data
    excel_file = FileContent(
        file_name="test_campaign_report.xlsx",
        content=excel_content,  # Base64 encoded Excel content
        file_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        file_size=len(excel_content)
    )
    
    # Test with file processing agent
    settings = get_settings()
    llm_manager = LLMProviderManager(settings)
    agent = FileProcessingAgent(settings, llm_manager)
    
    request_id = "excel-test-001"
    
    print(f"🔄 Processing Excel file through agentic workflow...")
    
    try:
        result = await agent.process_file(excel_file, request_id)
        
        print(f"\n✅ Processing Results:")
        print(f"   Success: {result.success}")
        print(f"   Confidence: {result.confidence_score:.2f}")
        print(f"   Execution time: {result.execution_time:.3f}s")
        print(f"   Tools used: {list(result.tool_outputs.keys())}")
        
        if result.errors:
            print(f"   Errors: {result.errors}")
        
        if result.extracted_content:
            print(f"\n📊 Extracted Content:")
            print("=" * 80)
            print(result.extracted_content)
            print("=" * 80)
            
            # Check if actual data was extracted
            if "Campaign_Report" in result.extracted_content and "Sinsuru_USA" in result.extracted_content:
                print(f"\n✅ SUCCESS: Agent successfully read the Excel file and extracted campaign data!")
                return True
            else:
                print(f"\n❌ ISSUE: Content extracted but doesn't contain expected campaign data")
                return False
        else:
            print(f"\n❌ ISSUE: No content was extracted from the Excel file")
            return False
            
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_streaming_issue():
    """Test for streaming cutoff issues."""
    print("\n🧪 Testing Streaming Issue...")
    
    # Check if the issue might be in the UPEE engine
    try:
        from app.core.upee_engine import UPEEEngine
        from app.schemas import ChatRequest
        
        settings = get_settings()
        upee_engine = UPEEEngine(settings)
        
        # Create a test request
        test_request = ChatRequest(
            message="This is a test message to check streaming functionality",
            show_thinking=True
        )
        
        request_id = "streaming-test-001"
        
        print("🔄 Testing UPEE engine streaming...")
        
        event_count = 0
        events_received = []
        start_time = time.time()
        timeout = 30  # 30 second timeout
        
        async for event in upee_engine.process_request(test_request, request_id):
            event_count += 1
            event_type = event.get("event")
            events_received.append(event_type)
            print(f"   Event {event_count}: {event_type}")
            
            # Check for completion events
            if event_type in ["complete", "done", "error"]:
                print(f"   ✅ Received completion event: {event_type}")
                break
            
            # Timeout check instead of event count limit
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"   ⏰ Timeout after {elapsed:.1f}s with {event_count} events...")
                break
            
            # Safety limit for runaway streams
            if event_count > 100:
                print("   🛑 Safety limit reached after 100 events...")
                break
        
        print(f"\n✅ Streaming test completed:")
        print(f"   Total events: {event_count}")
        print(f"   Event types: {set(events_received)}")
        
        # Check if we got a proper completion
        if "complete" in events_received or "done" in events_received:
            print(f"✅ Streaming completed properly")
            return True
        else:
            print(f"❌ Streaming may have cut off - no completion event received")
            return False
            
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_file_content_reception():
    """Test how file content is received and processed."""
    print("\n🧪 Testing File Content Reception...")
    
    # Test different content formats
    test_cases = [
        {
            "name": "Plain text content",
            "content": "This is plain text content",
            "file_type": "text/plain",
            "expected_in_result": "This is plain text content"
        },
        {
            "name": "JSON content",
            "content": '{"campaign": "sinsuru", "metrics": {"impressions": 1000}}',
            "file_type": "application/json",
            "expected_in_result": "sinsuru"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n  Testing: {test_case['name']}")
        
        file_content = FileContent(
            file_name=f"test_{test_case['name'].replace(' ', '_').lower()}.txt",
            content=test_case["content"],
            file_type=test_case["file_type"],
            file_size=len(test_case["content"])
        )
        
        # Process with file processor
        from app.core.file_processor import file_processor
        
        try:
            result = await file_processor.process_files([file_content])
            
            if result and len(result) > 0:
                processed_content = result[0].get("content", "")
                if test_case["expected_in_result"] in processed_content:
                    print(f"    ✅ Content correctly preserved and accessible")
                else:
                    print(f"    ❌ Expected content not found in result")
                    print(f"    Expected: {test_case['expected_in_result']}")
                    print(f"    Got: {processed_content[:100]}...")
            else:
                print(f"    ❌ No result returned from file processor")
                
        except Exception as e:
            print(f"    ❌ Processing failed: {e}")


async def main():
    """Run all tests to diagnose file processing and streaming issues."""
    print("🚀 Diagnosing File Processing and Streaming Issues\n")
    
    try:
        # Test 1: File content reception
        await test_file_content_reception()
        
        # Test 2: Actual Excel file reading
        excel_success = await test_excel_file_reading()
        
        # Test 3: Streaming functionality
        streaming_success = await test_streaming_issue()
        
        print("\n" + "="*80)
        print("📋 DIAGNOSIS SUMMARY:")
        print("="*80)
        
        if excel_success:
            print("✅ EXCEL PROCESSING: Agent can read and extract data from Excel files")
        else:
            print("❌ EXCEL PROCESSING: Issues with reading Excel file content")
            print("   RECOMMENDATION: Check file upload format (base64 encoding)")
        
        if streaming_success:
            print("✅ STREAMING: UPEE engine streaming works correctly")
        else:
            print("❌ STREAMING: Issues with streaming completion")
            print("   RECOMMENDATION: Check for exceptions in UPEE processing")
        
        print("\n💡 NEXT STEPS:")
        if not excel_success:
            print("1. Verify Excel files are uploaded as base64-encoded content")
            print("2. Check client-side file reading and encoding")
            print("3. Test with actual uploaded Excel file content format")
        
        if not streaming_success:
            print("1. Check UPEE engine error handling")
            print("2. Verify all phases complete properly")
            print("3. Check for unhandled exceptions that break the stream")
        
        return 0 if excel_success and streaming_success else 1
        
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)