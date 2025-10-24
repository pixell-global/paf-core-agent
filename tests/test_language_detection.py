"""Test language detection and response in appropriate language"""

import pytest
from unittest.mock import AsyncMock, patch
from src.core.understand import UnderstandPhase
from src.core.execute import ExecutePhase
from src.schemas import ChatRequest
from src.settings import Settings


@pytest.mark.asyncio
async def test_korean_language_detection():
    """Test that Korean messages are detected and language instructions added to prompt"""
    settings = Settings()
    
    # Create Korean request
    request = ChatRequest(
        message='레딧에서 "collagen"으로 "skincareaddiction" 서브레딧에서 포스트를 찾아줘',
        model="gpt-4o"
    )
    
    # Mock LLM response for language detection
    mock_llm_response = AsyncMock()
    mock_llm_response.content = '{"language": "Korean", "language_code": "ko", "confidence": "high"}'
    
    with patch('app.core.understand.UnderstandPhase._detect_language_with_llm') as mock_detect:
        mock_detect.return_value = {
            "language": "Korean",
            "language_code": "ko", 
            "confidence": "high"
        }
        
        understand_phase = UnderstandPhase(settings)
        understanding_result = await understand_phase.process(request, "test-123")
        
        # Verify language was detected correctly
        assert understanding_result.metadata["language"] == "Korean"
        assert understanding_result.metadata["language_code"] == "ko"
        
        # Test execute phase builds prompt with Korean instructions
        execute_phase = ExecutePhase(settings)
        prompt = await execute_phase._build_prompt(request, understanding_result, None)
        
        # Verify Korean instructions are in the prompt
        assert "Korean" in prompt
        assert "You MUST respond in Korean" in prompt
        assert "Write your entire response in Korean" in prompt


@pytest.mark.asyncio
async def test_english_language_no_special_instructions():
    """Test that English messages don't get language instructions"""
    settings = Settings()
    
    # Create English request  
    request = ChatRequest(
        message="Find posts about collagen on the skincareaddiction subreddit",
        model="gpt-4o"
    )
    
    with patch('app.core.understand.UnderstandPhase._detect_language_with_llm') as mock_detect:
        mock_detect.return_value = {
            "language": "English",
            "language_code": "en",
            "confidence": "high"
        }
        
        understand_phase = UnderstandPhase(settings)
        understanding_result = await understand_phase.process(request, "test-456")
        
        # Verify language was detected as English
        assert understanding_result.metadata["language"] == "English"
        assert understanding_result.metadata["language_code"] == "en"
        
        # Test execute phase builds prompt WITHOUT language instructions
        execute_phase = ExecutePhase(settings)
        prompt = await execute_phase._build_prompt(request, understanding_result, None)
        
        # Verify no language instructions for English
        assert "You MUST respond in English" not in prompt
        assert "Write your entire response in English" not in prompt


@pytest.mark.asyncio  
async def test_multiple_languages():
    """Test detection of various languages"""
    settings = Settings()
    
    test_cases = [
        ("Hola, ¿cómo estás?", "Spanish", "es"),
        ("Bonjour, comment allez-vous?", "French", "fr"),
        ("こんにちは、お元気ですか？", "Japanese", "ja"),
        ("你好，你好吗？", "Chinese", "zh"),
    ]
    
    for message, expected_language, expected_code in test_cases:
        request = ChatRequest(message=message, model="gpt-4o")
        
        with patch('app.core.understand.UnderstandPhase._detect_language_with_llm') as mock_detect:
            mock_detect.return_value = {
                "language": expected_language,
                "language_code": expected_code,
                "confidence": "high"
            }
            
            understand_phase = UnderstandPhase(settings)
            understanding_result = await understand_phase.process(request, f"test-{expected_code}")
            
            execute_phase = ExecutePhase(settings)
            prompt = await execute_phase._build_prompt(request, understanding_result, None)
            
            # Verify correct language instructions
            assert f"You MUST respond in {expected_language}" in prompt
            assert f"Write your entire response in {expected_language}" in prompt