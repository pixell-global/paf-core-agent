#!/usr/bin/env python3
"""A2A 기능 테스트 스크립트."""

import asyncio
import json
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.a2a_client import A2AClient
from app.settings import get_settings


async def test_a2a_discovery():
    """A2A 에이전트 탐색 테스트."""
    settings = get_settings()
    
    if not settings.a2a_enabled:
        print("❌ A2A 기능이 비활성화되어 있습니다.")
        print("환경 변수 A2A_ENABLED=true로 설정하세요.")
        return False
    
    print(f"🔍 A2A 서버 탐색 중: {settings.a2a_server_url}")
    
    client = A2AClient(settings.a2a_server_url, settings.a2a_timeout)
    
    try:
        # 에이전트 탐색 테스트
        agents = await client.discover_agents()
        
        if agents:
            print(f"✅ {len(agents)}개의 A2A 에이전트를 발견했습니다:")
            for i, agent in enumerate(agents, 1):
                print(f"  {i}. {agent.get('name', 'Unknown')}")
                print(f"     설명: {agent.get('description', 'N/A')}")
                print(f"     버전: {agent.get('version', 'N/A')}")
                print(f"     URL: {agent.get('url', 'N/A')}")
                if agent.get('skills'):
                    print(f"     스킬: {len(agent['skills'])}개")
                print()
        else:
            print("❌ A2A 에이전트를 찾을 수 없습니다.")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ A2A 탐색 실패: {e}")
        return False


async def test_a2a_health_check():
    """A2A 서버 상태 확인 테스트."""
    settings = get_settings()
    
    print(f"🏥 A2A 서버 상태 확인: {settings.a2a_server_url}")
    
    client = A2AClient(settings.a2a_server_url, settings.a2a_timeout)
    
    try:
        is_healthy = await client.health_check()
        
        if is_healthy:
            print("✅ A2A 서버가 정상적으로 응답합니다.")
            return True
        else:
            print("❌ A2A 서버에서 응답이 없습니다.")
            return False
            
    except Exception as e:
        print(f"❌ A2A 상태 확인 실패: {e}")
        return False


async def test_a2a_message():
    """A2A 메시지 전송 테스트."""
    settings = get_settings()
    
    print(f"💬 A2A 메시지 전송 테스트: {settings.a2a_server_url}")
    
    client = A2AClient(settings.a2a_server_url, settings.a2a_timeout)
    
    try:
        test_message = {
            "payload": {
                "message": "Hello from PAF Core Agent! This is a test message."
            }
        }
        
        response = await client.send_message_async(test_message)
        
        if response.get("status") == "success":
            print("✅ A2A 메시지 전송 성공!")
            print(f"응답: {json.dumps(response, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"❌ A2A 메시지 전송 실패: {response.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ A2A 메시지 전송 실패: {e}")
        return False


async def main():
    """메인 테스트 함수."""
    print("🚀 PAF Core Agent A2A 기능 테스트")
    print("=" * 50)
    
    # 설정 확인
    settings = get_settings()
    print(f"A2A 활성화: {settings.a2a_enabled}")
    print(f"A2A 서버 URL: {settings.a2a_server_url}")
    print(f"A2A 타임아웃: {settings.a2a_timeout}초")
    print()
    
    if not settings.a2a_enabled:
        print("❌ A2A 기능이 비활성화되어 있습니다.")
        print("환경 변수를 설정하고 다시 시도하세요:")
        print("export A2A_ENABLED=true")
        print("export A2A_SERVER_URL=http://localhost:9999")
        return
    
    # 테스트 실행
    tests = [
        ("A2A 서버 상태 확인", test_a2a_health_check),
        ("A2A 에이전트 탐색", test_a2a_discovery),
        ("A2A 메시지 전송", test_a2a_message),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"🧪 {test_name}")
        print("-" * 30)
        
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ 테스트 실행 중 오류: {e}")
            results.append((test_name, False))
        
        print()
    
    # 결과 요약
    print("📊 테스트 결과 요약")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ 통과" if success else "❌ 실패"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print()
    print(f"전체 테스트: {len(results)}개")
    print(f"통과: {passed}개")
    print(f"실패: {len(results) - passed}개")
    
    if passed == len(results):
        print("\n🎉 모든 A2A 테스트가 성공했습니다!")
    else:
        print(f"\n⚠️  {len(results) - passed}개의 테스트가 실패했습니다.")
        print("A2A 서버가 실행 중인지 확인하세요.")


if __name__ == "__main__":
    asyncio.run(main()) 