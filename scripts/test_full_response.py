#!/usr/bin/env python3
"""
Test full response completion without cutoffs.
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


async def create_sinsuru_campaign_file():
    """Create a test Excel file with detailed campaign data."""
    try:
        import pandas as pd
        
        # Create more detailed campaign data
        campaign_data = {
            'Date': ['2025-07-02', '2025-07-03', '2025-07-04', '2025-07-05', '2025-07-06', '2025-07-07', '2025-07-08', '2025-07-09'],
            'Campaign': ['Sinsuru_USA'] * 8,
            'Ad_Group': ['Brand_Awareness', 'Conversion', 'Retargeting', 'Brand_Awareness', 'Conversion', 'Retargeting', 'Brand_Awareness', 'Conversion'],
            'Impressions': [15000, 18000, 12000, 22000, 16000, 14000, 19000, 17000],
            'Clicks': [450, 540, 360, 660, 480, 420, 570, 510],
            'Conversions': [23, 28, 18, 34, 25, 21, 31, 27],
            'Cost_USD': [892.50, 1070.40, 713.60, 1308.80, 952.00, 834.00, 1134.50, 1020.30],
            'CTR_Percent': [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            'CPA_USD': [38.80, 38.23, 39.64, 38.49, 38.08, 39.71, 36.60, 37.79],
            'ROAS': [2.5, 2.8, 2.1, 3.2, 2.6, 2.3, 3.0, 2.7]
        }
        
        # Create DataFrame
        df = pd.DataFrame(campaign_data)
        
        # Save to BytesIO as Excel file
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Campaign_Daily_Report', index=False)
            
            # Add performance summary
            summary_data = {
                'Metric': [
                    'Total Impressions', 'Total Clicks', 'Total Conversions', 
                    'Total Cost', 'Average CTR', 'Average CPA', 'Average ROAS',
                    'Best Performing Day', 'Worst Performing Day'
                ],
                'Value': [
                    df['Impressions'].sum(), 
                    df['Clicks'].sum(), 
                    df['Conversions'].sum(),
                    df['Cost_USD'].sum(), 
                    df['CTR_Percent'].mean(), 
                    df['CPA_USD'].mean(),
                    df['ROAS'].mean(),
                    df.loc[df['ROAS'].idxmax(), 'Date'],
                    df.loc[df['ROAS'].idxmin(), 'Date']
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Performance_Summary', index=False)
        
        # Get the Excel content as base64
        excel_buffer.seek(0)
        excel_content = excel_buffer.getvalue()
        base64_content = base64.b64encode(excel_content).decode('utf-8')
        
        return base64_content, df
        
    except ImportError:
        print("❌ Pandas/openpyxl not available for test file creation")
        return None, None


async def test_full_campaign_analysis():
    """Test full campaign analysis without response cutoffs."""
    print("🚀 Testing Full Campaign Analysis Response\n")
    
    # Create test Excel file
    excel_content, original_df = await create_sinsuru_campaign_file()
    
    if not excel_content:
        print("❌ Could not create test Excel file")
        return False
    
    print(f"✅ Created campaign Excel file with {len(original_df)} days of data")
    print(f"   Total impressions: {original_df['Impressions'].sum():,}")
    print(f"   Total conversions: {original_df['Conversions'].sum()}")
    print(f"   Average ROAS: {original_df['ROAS'].mean():.2f}\n")
    
    # Create FileContent
    excel_file = FileContent(
        file_name="신스루_USA-Campaign Report-(2025-07-02 to 2025-07-09).xlsx",
        content=excel_content,
        file_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        file_size=len(excel_content)
    )
    
    # Create detailed analysis request
    chat_request = ChatRequest(
        message="""Analyze this Sinsuru USA campaign report and provide comprehensive improvement recommendations. Please include:

1. Overall performance assessment
2. Day-by-day trend analysis  
3. Ad group performance comparison
4. Cost efficiency insights
5. ROAS optimization opportunities
6. Specific actionable recommendations for the next campaign period
7. Budget allocation suggestions
8. Target audience refinement ideas

Please be thorough and provide detailed insights based on the data.""",
        files=[excel_file],
        show_thinking=True,
        max_tokens=2000  # Allow for longer response
    )
    
    # Initialize UPEE engine
    settings = get_settings()
    upee_engine = UPEEEngine(settings)
    
    request_id = "full-analysis-test-001"
    
    print("🔄 Processing full campaign analysis request...\n")
    
    response_content = ""
    event_count = 0
    thinking_events = 0
    content_events = 0
    
    try:
        async for event in upee_engine.process_request(chat_request, request_id):
            event_count += 1
            event_type = event.get("event")
            
            if event_type == "thinking":
                thinking_events += 1
                if thinking_events <= 5:  # Show first few thinking events
                    data = event.get('data', '')
                    if isinstance(data, str):
                        try:
                            data_obj = json.loads(data)
                            phase = data_obj.get('phase', 'unknown')
                            content = data_obj.get('content', '')
                            print(f"💭 [{phase.upper()}] {content}")
                        except:
                            print(f"💭 {data[:100]}...")
                
            elif event_type == "content":
                content_events += 1
                data = event.get('data', '')
                if isinstance(data, str):
                    try:
                        data_obj = json.loads(data)
                        content = data_obj.get('content', '')
                        response_content += content
                        if content_events <= 5:  # Show first few content chunks
                            print(f"📝 Content chunk {content_events}: {content[:50]}...")
                    except:
                        print(f"📝 Raw content: {data[:50]}...")
                        response_content += data
                
            elif event_type == "complete":
                print(f"\n✅ Response completed!")
                print(f"   Total events: {event_count}")
                print(f"   Thinking events: {thinking_events}")
                print(f"   Content events: {content_events}")
                print(f"   Response length: {len(response_content)} characters")
                
                if len(response_content) > 200:
                    print(f"\n📊 FULL RESPONSE:")
                    print("=" * 80)
                    print(response_content)
                    print("=" * 80)
                    return True
                else:
                    print(f"\n⚠️  Response seems too short ({len(response_content)} chars)")
                    print(f"Response: {response_content}")
                    return False
                
            elif event_type == "error":
                print(f"\n❌ Error received: {event.get('data', '')}")
                return False
                
            # Safety timeout
            if event_count > 500:
                print(f"\n⏰ Test timeout after {event_count} events")
                break
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\nℹ️  Final stats:")
    print(f"   Events received: {event_count}")
    print(f"   Response length: {len(response_content)} chars")
    print(f"   Content events: {content_events}")
    
    return len(response_content) > 200  # Consider success if we got substantial content


async def main():
    """Run the full response test."""
    try:
        success = await test_full_campaign_analysis()
        if success:
            print("\n🎉 SUCCESS: Full response completed without cutoffs!")
        else:
            print("\n❌ ISSUE: Response was cut off or incomplete")
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)