import pytest
import asyncio
import pytest_asyncio
from src.core.upee_engine import UPEEEngine
from src.schemas import ChatRequest
from src.settings import Settings


class TestTableGeneration:
    """Test cases for table generation functionality"""

    @pytest_asyncio.fixture
    async def upee_engine(self):
        settings = Settings()
        return UPEEEngine(settings)

    async def get_response(self, upee_engine, prompt):
        """Helper to get response from UPEE engine"""
        request = ChatRequest(
            message=prompt,
            model="gpt-4o"
        )
        
        response_parts = []
        async for event in upee_engine.process_request(request, "test-request-id"):
            # Collect content from streaming events
            if event.get("event") == "content":
                import json
                data = json.loads(event.get("data", "{}"))
                if data.get("content"):
                    response_parts.append(data["content"])
        
        return ''.join(response_parts)

    @pytest.mark.asyncio
    async def test_simple_number_table(self, upee_engine):
        """Test creating a simple number table 1-500 with 10 columns"""
        prompt = "create a table of number 1-500. there should be ten columns"
        response = await self.get_response(upee_engine, prompt)
        
        # Check that response contains actual table data, not instructions
        assert "Open the Application" not in response
        assert "Microsoft Excel" not in response
        assert "Step-by-step" not in response.lower()
        
        # Check for presence of numbers in table format
        assert "1" in response and "500" in response
        assert "|" in response or "\t" in response  # Table formatting

    @pytest.mark.asyncio
    async def test_multiplication_table(self, upee_engine):
        """Test creating a multiplication table"""
        prompt = "Create a multiplication table for numbers 1 through 10"
        response = await self.get_response(upee_engine, prompt)
        
        # Should contain actual multiplication results
        assert "how to" not in response.lower()
        assert "12" in response  # 3x4 or 2x6
        assert "100" in response  # 10x10

    @pytest.mark.asyncio
    async def test_days_of_week_table(self, upee_engine):
        """Test creating a table with days of the week"""
        prompt = "Make a table showing the days of the week with their abbreviations"
        response = await self.get_response(upee_engine, prompt)
        
        # Should contain actual days
        assert "Monday" in response or "Mon" in response
        assert "Sunday" in response or "Sun" in response
        assert "spreadsheet" not in response.lower()

    @pytest.mark.asyncio
    async def test_comparison_table(self, upee_engine):
        """Test creating a comparison table"""
        prompt = "Create a table comparing Python, JavaScript, and Java with columns for typing, performance, and use cases"
        response = await self.get_response(upee_engine, prompt)
        
        # Should contain actual comparison data
        assert "Python" in response and "JavaScript" in response
        assert "typing" in response.lower() or "static" in response or "dynamic" in response
        assert "click and drag" not in response.lower()

    @pytest.mark.asyncio
    async def test_price_list_table(self, upee_engine):
        """Test creating a price list table"""
        prompt = "Generate a table with 5 products and their prices in USD"
        response = await self.get_response(upee_engine, prompt)
        
        # Should contain actual product data
        assert "$" in response or "USD" in response
        assert "autofill" not in response.lower()
        assert "|" in response or "\t" in response

    @pytest.mark.asyncio
    async def test_calendar_table(self, upee_engine):
        """Test creating a calendar table"""
        prompt = "Create a table showing a calendar for January 2024"
        response = await self.get_response(upee_engine, prompt)
        
        # Should contain actual calendar data
        assert "1" in response and "31" in response
        assert "Mon" in response or "Monday" in response
        assert "Excel" not in response

    @pytest.mark.asyncio
    async def test_data_summary_table(self, upee_engine):
        """Test creating a data summary table"""
        prompt = "Make a table summarizing sales data for Q1, Q2, Q3, Q4 with totals"
        response = await self.get_response(upee_engine, prompt)
        
        # Should contain actual data
        assert "Q1" in response and "Q4" in response
        assert "Total" in response or "total" in response
        assert "text editor" not in response.lower()

    @pytest.mark.asyncio
    async def test_alphabet_table(self, upee_engine):
        """Test creating an alphabet table"""
        prompt = "Create a table with the alphabet arranged in 5 columns"
        response = await self.get_response(upee_engine, prompt)
        
        # Should contain actual alphabet
        assert "A" in response and "Z" in response
        assert "launch" not in response.lower()
        assert "|" in response or "\t" in response

    @pytest.mark.asyncio
    async def test_conversion_table(self, upee_engine):
        """Test creating a unit conversion table"""
        prompt = "Generate a table converting kilometers to miles for 1-10 km"
        response = await self.get_response(upee_engine, prompt)
        
        # Should contain actual conversions
        assert "km" in response.lower() or "kilometers" in response.lower()
        assert "miles" in response.lower() or "mi" in response.lower()
        assert "0.62" in response or "1.6" in response  # Conversion factors
        assert "Google Sheets" not in response

    @pytest.mark.asyncio
    async def test_markdown_table(self, upee_engine):
        """Test creating a markdown formatted table"""
        prompt = "Create a markdown table with 3 columns and 5 rows of sample data"
        response = await self.get_response(upee_engine, prompt)
        
        # Should contain markdown table syntax
        assert "|" in response
        assert "---" in response or "|-" in response
        assert "step-by-step" not in response.lower()