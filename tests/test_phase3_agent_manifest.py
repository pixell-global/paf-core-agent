"""
Test Phase 3: Verify agent.yaml manifest

This test verifies:
1. agent.yaml exists at project root
2. Has correct structure and required fields
3. Entrypoints are correctly configured
4. Capabilities are defined
5. Validates against APKG schema (if pixell CLI available)
"""

import sys
import yaml
from pathlib import Path
import subprocess


def test_manifest_exists():
    """Test that agent.yaml exists"""
    print("=" * 60)
    print("Phase 3: Manifest File Existence Test")
    print("=" * 60)

    errors = []
    successes = []

    project_root = Path(__file__).parent.parent
    manifest_file = project_root / "agent.yaml"

    if not manifest_file.exists():
        errors.append(("agent.yaml", "File does not exist"))
        print("❌ agent.yaml does not exist at project root")
        return False, successes, errors, None
    else:
        successes.append("agent.yaml exists")
        print("✅ agent.yaml exists at project root")

    # Try to load YAML
    try:
        with open(manifest_file, 'r') as f:
            manifest = yaml.safe_load(f)
        successes.append("agent.yaml is valid YAML")
        print("✅ agent.yaml is valid YAML")
        return True, successes, errors, manifest
    except Exception as e:
        errors.append(("agent.yaml YAML parsing", str(e)))
        print(f"❌ Failed to parse agent.yaml: {e}")
        return False, successes, errors, None


def test_required_fields(manifest):
    """Test that required fields are present"""
    print("\n" + "=" * 60)
    print("Phase 3: Required Fields Test")
    print("=" * 60)

    errors = []
    successes = []

    required_fields = {
        "version": str,
        "name": str,
        "display_name": str,
        "description": str,
        "author": str,
        "entrypoint": str,
        "runtime": str,
    }

    for field, expected_type in required_fields.items():
        if field not in manifest:
            errors.append((f"Field '{field}'", "Missing"))
            print(f"❌ Required field '{field}' missing")
        else:
            value = manifest[field]
            if not isinstance(value, expected_type):
                errors.append((f"Field '{field}'", f"Expected {expected_type.__name__}, got {type(value).__name__}"))
                print(f"❌ Field '{field}' has wrong type: {type(value).__name__}")
            else:
                successes.append(f"Field '{field}' present and correct type")
                print(f"✅ Field '{field}': {value}")

    return len(errors) == 0, successes, errors


def test_entrypoints(manifest):
    """Test that entrypoints are correctly configured"""
    print("\n" + "=" * 60)
    print("Phase 3: Entrypoints Configuration Test")
    print("=" * 60)

    errors = []
    successes = []

    # Check main entrypoint
    entrypoint = manifest.get("entrypoint")
    if entrypoint == "src.par_adapter:mount":
        successes.append("Main entrypoint correct")
        print(f"✅ Main entrypoint: {entrypoint}")
    else:
        errors.append(("Main entrypoint", f"Expected 'src.par_adapter:mount', got '{entrypoint}'"))
        print(f"❌ Main entrypoint incorrect: {entrypoint}")

    # Check REST surface
    if "rest" in manifest:
        rest_config = manifest["rest"]
        if isinstance(rest_config, dict):
            rest_entry = rest_config.get("entry")
            if rest_entry == "src.par_adapter:mount":
                successes.append("REST entrypoint correct")
                print(f"✅ REST entry: {rest_entry}")
            else:
                errors.append(("REST entry", f"Expected 'src.par_adapter:mount', got '{rest_entry}'"))
                print(f"❌ REST entry incorrect: {rest_entry}")
        else:
            errors.append(("REST config", "Not a dict"))
            print("❌ REST config is not a dict")
    else:
        errors.append(("REST config", "Missing"))
        print("❌ REST configuration missing")

    # Check A2A surface
    if "a2a" in manifest:
        a2a_config = manifest["a2a"]
        if isinstance(a2a_config, dict):
            a2a_service = a2a_config.get("service")
            if a2a_service == "src.par_adapter:create_service":
                successes.append("A2A service entrypoint correct")
                print(f"✅ A2A service: {a2a_service}")
            else:
                errors.append(("A2A service", f"Expected 'src.par_adapter:create_service', got '{a2a_service}'"))
                print(f"❌ A2A service incorrect: {a2a_service}")
        else:
            errors.append(("A2A config", "Not a dict"))
            print("❌ A2A config is not a dict")
    else:
        errors.append(("A2A config", "Missing"))
        print("❌ A2A configuration missing")

    return len(errors) == 0, successes, errors


def test_capabilities(manifest):
    """Test that capabilities are defined"""
    print("\n" + "=" * 60)
    print("Phase 3: Capabilities Test")
    print("=" * 60)

    errors = []
    successes = []

    if "capabilities" not in manifest:
        errors.append(("capabilities", "Missing"))
        print("❌ 'capabilities' field missing")
        return False, successes, errors

    capabilities = manifest["capabilities"]

    if not isinstance(capabilities, list):
        errors.append(("capabilities", f"Expected list, got {type(capabilities).__name__}"))
        print(f"❌ 'capabilities' is not a list: {type(capabilities).__name__}")
        return False, successes, errors

    successes.append(f"capabilities is list with {len(capabilities)} items")
    print(f"✅ capabilities is list with {len(capabilities)} items")

    # Check for expected capabilities
    expected_capabilities = [
        "chat",
        "upee_processing",
        "multi_provider_llm",
        "a2a_protocol",
    ]

    for cap in expected_capabilities:
        if cap in capabilities:
            successes.append(f"Capability '{cap}' present")
            print(f"✅ Capability '{cap}' present")
        else:
            errors.append((f"Capability '{cap}'", "Missing"))
            print(f"⚠️  Capability '{cap}' missing (may be optional)")

    # List all capabilities
    print(f"\n   All capabilities: {', '.join(capabilities)}")

    return len(errors) == 0, successes, errors


def test_metadata(manifest):
    """Test optional metadata section"""
    print("\n" + "=" * 60)
    print("Phase 3: Metadata Test")
    print("=" * 60)

    errors = []
    successes = []

    if "metadata" in manifest:
        metadata = manifest["metadata"]
        successes.append("metadata section present")
        print("✅ metadata section present")

        # Check optional metadata fields
        optional_fields = ["version", "homepage", "extensive_description", "tags", "usage_guide"]

        for field in optional_fields:
            if field in metadata:
                successes.append(f"Metadata '{field}' present")
                print(f"✅ Metadata '{field}' present")

        # Check sub_agents if present
        if "sub_agents" in metadata:
            sub_agents = metadata["sub_agents"]
            if isinstance(sub_agents, list):
                successes.append(f"sub_agents defined with {len(sub_agents)} agents")
                print(f"✅ sub_agents defined with {len(sub_agents)} agents")
            else:
                print(f"ℹ️  sub_agents is not a list")
    else:
        print("ℹ️  metadata section not present (optional)")

    return True, successes, errors


def test_pixell_validate():
    """Test manifest validation using pixell CLI"""
    print("\n" + "=" * 60)
    print("Phase 3: Pixell CLI Validation Test")
    print("=" * 60)

    errors = []
    successes = []

    project_root = Path(__file__).parent.parent

    try:
        result = subprocess.run(
            ["pixell", "validate"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            successes.append("pixell validate passed")
            print("✅ pixell validate passed")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()}")
        else:
            errors.append(("pixell validate", result.stderr or result.stdout))
            print("❌ pixell validate failed")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()}")

    except FileNotFoundError:
        print("⚠️  pixell CLI not found (skipping validation)")
        successes.append("pixell CLI not available")
    except subprocess.TimeoutExpired:
        errors.append(("pixell validate", "Timeout"))
        print("❌ pixell validate timed out")
    except Exception as e:
        errors.append(("pixell validate", str(e)))
        print(f"⚠️  Error running pixell validate: {e}")

    return len(errors) == 0, successes, errors


def run_all_tests():
    """Run all Phase 3 tests"""
    print("\n" + "🧪" * 30)
    print("PHASE 3: AGENT.YAML MANIFEST")
    print("🧪" * 30 + "\n")

    all_successes = []
    all_errors = []
    results = []

    # Test 1: Manifest exists
    passed, successes, errors, manifest = test_manifest_exists()
    results.append(passed)
    all_successes.extend(successes)
    all_errors.extend(errors)

    if not passed or manifest is None:
        print("\n❌ Cannot proceed: agent.yaml failed to load")
        return False

    # Test 2: Required fields
    passed, successes, errors = test_required_fields(manifest)
    results.append(passed)
    all_successes.extend(successes)
    all_errors.extend(errors)

    # Test 3: Entrypoints
    passed, successes, errors = test_entrypoints(manifest)
    results.append(passed)
    all_successes.extend(successes)
    all_errors.extend(errors)

    # Test 4: Capabilities
    passed, successes, errors = test_capabilities(manifest)
    results.append(passed)
    all_successes.extend(successes)
    all_errors.extend(errors)

    # Test 5: Metadata (optional)
    passed, successes, errors = test_metadata(manifest)
    results.append(passed)
    all_successes.extend(successes)
    all_errors.extend(errors)

    # Test 6: Pixell validation
    passed, successes, errors = test_pixell_validate()
    results.append(passed)
    all_successes.extend(successes)
    all_errors.extend(errors)

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 3 SUMMARY")
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
        print("\n🎉 Phase 3: ALL TESTS PASSED")
    else:
        print("\n❌ Phase 3: SOME TESTS FAILED")

    return overall_pass


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
