"""
Test Phase 4: Verify .env.example configuration

This test verifies:
1. .env.example exists at project root
2. Contains required environment variables
3. Has proper structure and comments
4. All LLM provider configs are present
"""

import sys
from pathlib import Path
import re


def test_env_example_exists():
    """Test that .env.example exists"""
    print("=" * 60)
    print("Phase 4: .env.example File Existence Test")
    print("=" * 60)

    errors = []
    successes = []

    project_root = Path(__file__).parent.parent
    env_example_file = project_root / ".env.example"

    if not env_example_file.exists():
        errors.append((".env.example", "File does not exist"))
        print("❌ .env.example does not exist at project root")
        return False, successes, errors, None
    else:
        successes.append(".env.example exists")
        print("✅ .env.example exists at project root")

    # Read file content
    try:
        with open(env_example_file, 'r') as f:
            content = f.read()
        successes.append(".env.example is readable")
        print("✅ .env.example is readable")
        return True, successes, errors, content
    except Exception as e:
        errors.append((".env.example reading", str(e)))
        print(f"❌ Failed to read .env.example: {e}")
        return False, successes, errors, None


def test_llm_provider_configs(content):
    """Test that LLM provider configurations are present"""
    print("\n" + "=" * 60)
    print("Phase 4: LLM Provider Configuration Test")
    print("=" * 60)

    errors = []
    successes = []

    # Required LLM provider variables
    required_providers = {
        "OpenAI": ["OPENAI_API_KEY", "OPENAI_MODEL"],
        "Anthropic": ["ANTHROPIC_API_KEY"],
        "AWS Bedrock": ["AWS_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
    }

    for provider_name, variables in required_providers.items():
        provider_found = False
        for var in variables:
            # Check if variable is mentioned (even if commented)
            if var in content:
                provider_found = True
                break

        if provider_found:
            successes.append(f"{provider_name} configuration present")
            print(f"✅ {provider_name} configuration present")

            # Check each variable
            for var in variables:
                if var in content:
                    successes.append(f"Variable {var} present")
                    print(f"   ✅ {var}")
                else:
                    errors.append((f"Variable {var}", "Missing"))
                    print(f"   ❌ {var} missing")
        else:
            errors.append((f"{provider_name} configuration", "Missing"))
            print(f"❌ {provider_name} configuration missing")

    return len(errors) == 0, successes, errors


def test_agent_configuration(content):
    """Test that agent configuration variables are present"""
    print("\n" + "=" * 60)
    print("Phase 4: Agent Configuration Test")
    print("=" * 60)

    errors = []
    successes = []

    # Required agent configuration variables
    required_vars = [
        "DEFAULT_MODEL",
        "MAX_CONTEXT_TOKENS",
        "DEBUG",
    ]

    for var in required_vars:
        if var in content:
            successes.append(f"Variable {var} present")
            print(f"✅ {var} present")
        else:
            errors.append((f"Variable {var}", "Missing"))
            print(f"❌ {var} missing")

    return len(errors) == 0, successes, errors


def test_a2a_configuration(content):
    """Test that A2A configuration variables are present"""
    print("\n" + "=" * 60)
    print("Phase 4: A2A Configuration Test")
    print("=" * 60)

    errors = []
    successes = []

    # A2A configuration variables
    a2a_vars = [
        "A2A_ENABLED",
        "A2A_SERVER_URL",
        "A2A_TIMEOUT",
    ]

    for var in a2a_vars:
        if var in content:
            successes.append(f"Variable {var} present")
            print(f"✅ {var} present")
        else:
            errors.append((f"Variable {var}", "Missing"))
            print(f"❌ {var} missing")

    return len(errors) == 0, successes, errors


def test_security_configuration(content):
    """Test that security configuration is present"""
    print("\n" + "=" * 60)
    print("Phase 4: Security Configuration Test")
    print("=" * 60)

    errors = []
    successes = []

    # Security variables (may be commented out)
    security_vars = [
        "JWT_SECRET_KEY",
        "HMAC_SECRET_KEY",
    ]

    security_section_found = False
    for var in security_vars:
        if var in content:
            security_section_found = True
            successes.append(f"Variable {var} present")
            print(f"✅ {var} present")

    if security_section_found:
        successes.append("Security configuration section present")
        print("✅ Security configuration section present")
    else:
        print("ℹ️  Security configuration section not found (may be optional)")

    return True, successes, errors  # Don't fail on missing security section


def test_documentation_comments(content):
    """Test that file has proper documentation comments"""
    print("\n" + "=" * 60)
    print("Phase 4: Documentation Comments Test")
    print("=" * 60)

    errors = []
    successes = []

    # Check for section headers (comments with ===)
    section_headers = re.findall(r'#.*=+', content)

    if len(section_headers) > 0:
        successes.append(f"Found {len(section_headers)} section headers")
        print(f"✅ Found {len(section_headers)} section headers")
    else:
        errors.append(("Section headers", "No section headers found"))
        print("❌ No section headers found")

    # Check for description at top
    lines = content.split('\n')
    if lines and lines[0].startswith('#'):
        successes.append("File has header comment")
        print(f"✅ File has header comment: {lines[0]}")
    else:
        errors.append(("Header comment", "Missing"))
        print("❌ File missing header comment")

    # Count total comment lines
    comment_lines = [line for line in lines if line.strip().startswith('#')]
    if len(comment_lines) > 5:
        successes.append(f"File has {len(comment_lines)} comment lines")
        print(f"✅ File has {len(comment_lines)} comment lines (well documented)")
    else:
        errors.append(("Documentation", f"Only {len(comment_lines)} comment lines"))
        print(f"⚠️  File has only {len(comment_lines)} comment lines")

    return len(errors) == 0, successes, errors


def test_no_sensitive_data(content):
    """Test that file doesn't contain actual secrets"""
    print("\n" + "=" * 60)
    print("Phase 4: Sensitive Data Check")
    print("=" * 60)

    errors = []
    successes = []

    # Patterns that indicate real secrets (not placeholders)
    suspicious_patterns = [
        (r'sk-[a-zA-Z0-9]{32,}', "OpenAI API key"),
        (r'sk-ant-[a-zA-Z0-9]{32,}', "Anthropic API key"),
        (r'AKIA[0-9A-Z]{16}', "AWS Access Key"),
    ]

    found_secrets = []
    for pattern, secret_type in suspicious_patterns:
        matches = re.findall(pattern, content)
        if matches:
            found_secrets.append((secret_type, matches[0]))

    if found_secrets:
        for secret_type, match in found_secrets:
            errors.append((f"Real {secret_type}", f"Found: {match[:20]}..."))
            print(f"❌ Found real {secret_type}: {match[:20]}...")
    else:
        successes.append("No real secrets found")
        print("✅ No real secrets found (all placeholders)")

    return len(errors) == 0, successes, errors


def run_all_tests():
    """Run all Phase 4 tests"""
    print("\n" + "🧪" * 30)
    print("PHASE 4: .ENV.EXAMPLE CONFIGURATION")
    print("🧪" * 30 + "\n")

    all_successes = []
    all_errors = []
    results = []

    # Test 1: File exists
    passed, successes, errors, content = test_env_example_exists()
    results.append(passed)
    all_successes.extend(successes)
    all_errors.extend(errors)

    if not passed or content is None:
        print("\n❌ Cannot proceed: .env.example failed to load")
        return False

    # Test 2: LLM provider configs
    passed, successes, errors = test_llm_provider_configs(content)
    results.append(passed)
    all_successes.extend(successes)
    all_errors.extend(errors)

    # Test 3: Agent configuration
    passed, successes, errors = test_agent_configuration(content)
    results.append(passed)
    all_successes.extend(successes)
    all_errors.extend(errors)

    # Test 4: A2A configuration
    passed, successes, errors = test_a2a_configuration(content)
    results.append(passed)
    all_successes.extend(successes)
    all_errors.extend(errors)

    # Test 5: Security configuration
    passed, successes, errors = test_security_configuration(content)
    results.append(passed)
    all_successes.extend(successes)
    all_errors.extend(errors)

    # Test 6: Documentation comments
    passed, successes, errors = test_documentation_comments(content)
    results.append(passed)
    all_successes.extend(successes)
    all_errors.extend(errors)

    # Test 7: No sensitive data
    passed, successes, errors = test_no_sensitive_data(content)
    results.append(passed)
    all_successes.extend(successes)
    all_errors.extend(errors)

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 4 SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {sum(results)}/{len(results)} test groups")
    print(f"✅ Total checks passed: {len(all_successes)}")
    print(f"❌ Total checks failed: {len(all_errors)}")

    if all_errors:
        print("\n⚠️  Failed checks:")
        for check_name, error in all_errors:
            print(f"   - {check_name}: {error}")

    overall_pass = all(results)
    if overall_pass:
        print("\n🎉 Phase 4: ALL TESTS PASSED")
    else:
        print("\n❌ Phase 4: SOME TESTS FAILED")

    return overall_pass


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
