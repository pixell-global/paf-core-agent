"""
Test Phase 1: Verify all imports work after renaming app/ to src/
"""

import sys
import importlib

def test_core_imports():
    """Test that core modules can be imported"""
    modules_to_test = [
        "src.main",
        "src.schemas",
        "src.settings",
        "src.core.upee_engine",
        "src.core.understand",
        "src.core.plan",
        "src.core.execute",
        "src.core.evaluate",
        "src.api.chat",
        "src.api.health",
        "src.api.debug",
        "src.llm_providers",
        "src.utils.logging_config",
    ]

    errors = []
    successes = []

    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            successes.append(module_name)
            print(f"✅ {module_name}")
        except Exception as e:
            errors.append((module_name, str(e)))
            print(f"❌ {module_name}: {str(e)}")

    print(f"\n{'='*60}")
    print(f"Results: {len(successes)} passed, {len(errors)} failed")
    print(f"{'='*60}")

    if errors:
        print("\nFailed imports:")
        for module, error in errors:
            print(f"  - {module}: {error}")
        return False
    else:
        print("\n✅ All imports successful!")
        return True

if __name__ == "__main__":
    success = test_core_imports()
    sys.exit(0 if success else 1)
