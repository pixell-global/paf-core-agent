#!/bin/bash

# Test curl request for debug endpoint to check file reception
# This script creates a sample Excel file and sends it to the debug endpoint

echo "🧪 Testing file reception with debug endpoint..."

# Create a simple test Excel file using Python
python3 << 'EOF'
import base64
import json
from io import BytesIO

try:
    import pandas as pd
    
    # Create simple test data
    data = {
        'Campaign': ['Test_Campaign'] * 3,
        'Date': ['2025-07-10', '2025-07-11', '2025-07-12'],
        'Impressions': [1000, 1200, 950],
        'Clicks': [50, 60, 45],
        'Cost': [100.0, 120.0, 95.0]
    }
    
    df = pd.DataFrame(data)
    
    # Save to BytesIO as Excel
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Campaign_Data', index=False)
    
    # Get base64 content
    excel_buffer.seek(0)
    excel_content = excel_buffer.getvalue()
    base64_content = base64.b64encode(excel_content).decode('utf-8')
    
    # Create the request payload
    request_payload = {
        "message": "Test file reception - can you see this Excel file?",
        "files": [{
            "file_name": "test_campaign.xlsx",
            "content": base64_content,
            "file_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "file_size": len(base64_content)
        }],
        "model": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    # Write to file for curl
    with open('/tmp/debug_test_payload.json', 'w') as f:
        json.dump(request_payload, f, indent=2)
    
    print(f"✅ Created test Excel file with {len(df)} rows")
    print(f"📄 Base64 content length: {len(base64_content)} characters")
    print(f"💾 Payload saved to /tmp/debug_test_payload.json")
    
except ImportError:
    print("❌ pandas/openpyxl not available - creating minimal test")
    
    # Fallback: create a simple text file
    test_content = "Campaign,Date,Impressions\nTest_Campaign,2025-07-10,1000"
    base64_content = base64.b64encode(test_content.encode()).decode('utf-8')
    
    request_payload = {
        "message": "Test file reception - can you see this CSV content?",
        "files": [{
            "file_name": "test_campaign.csv", 
            "content": base64_content,
            "file_type": "text/csv",
            "file_size": len(base64_content)
        }],
        "model": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    with open('/tmp/debug_test_payload.json', 'w') as f:
        json.dump(request_payload, f, indent=2)
    
    print(f"✅ Created fallback CSV test file")
    print(f"📄 Base64 content length: {len(base64_content)} characters")

EOF

# Check if payload was created
if [ ! -f "/tmp/debug_test_payload.json" ]; then
    echo "❌ Failed to create test payload"
    exit 1
fi

echo ""
echo "📡 Sending request to debug endpoint..."
echo ""

# Make the curl request
curl -X POST "http://localhost:8000/api/debug/inspect-request" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d @/tmp/debug_test_payload.json \
  --silent \
  --show-error \
  --write-out "\n📊 HTTP Status: %{http_code}\n📏 Response Size: %{size_download} bytes\n⏱️  Total Time: %{time_total}s\n" | \
  python3 -m json.tool

echo ""
echo "🔍 Debug test completed!"
echo ""
echo "💡 If you see detailed file information above, files are being received correctly."
echo "💡 If you see errors or empty file details, there's a reception issue."
echo ""
echo "🗑️  Cleaning up test file..."
rm -f /tmp/debug_test_payload.json