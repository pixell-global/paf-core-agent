"""
Test Phase 2: Verify PAR adapter loads and provides correct interfaces
"""

import sys
import importlib
import inspect


def test_par_adapter():
    """Test that PAR adapter module loads and has required functions"""
    print("="*60)
    print("Testing PAR Adapter (Phase 2)")
    print("="*60 + "\n")

    errors = []
    successes = []

    # Test 1: Import the module
    try:
        adapter = importlib.import_module("src.par_adapter")
        successes.append("Module import")
        print("✅ src.par_adapter imports successfully")
    except Exception as e:
        errors.append(("Module import", str(e)))
        print(f"❌ Failed to import src.par_adapter: {e}")
        return False

    # Test 2: Check for mount() function
    try:
        mount_func = getattr(adapter, "mount", None)
        if mount_func and callable(mount_func):
            successes.append("mount() function exists")
            print("✅ mount() function found")

            # Check signature
            sig = inspect.signature(mount_func)
            params = list(sig.parameters.keys())
            if params == ['app']:
                successes.append("mount() has correct signature")
                print("✅ mount(app: FastAPI) signature correct")
            else:
                errors.append(("mount() signature", f"Expected ['app'], got {params}"))
                print(f"❌ mount() signature incorrect: {params}")
        else:
            errors.append(("mount() function", "Not found or not callable"))
            print("❌ mount() function missing or not callable")
    except Exception as e:
        errors.append(("mount() check", str(e)))
        print(f"❌ Error checking mount(): {e}")

    # Test 3: Check for create_service() function
    try:
        create_service_func = getattr(adapter, "create_service", None)
        if create_service_func and callable(create_service_func):
            successes.append("create_service() function exists")
            print("✅ create_service() function found")

            # Check signature
            sig = inspect.signature(create_service_func)
            params = list(sig.parameters.keys())
            if len(params) == 0:
                successes.append("create_service() has correct signature")
                print("✅ create_service() signature correct (no params)")
            else:
                errors.append(("create_service() signature", f"Expected no params, got {params}"))
                print(f"❌ create_service() signature incorrect: {params}")
        else:
            errors.append(("create_service() function", "Not found or not callable"))
            print("❌ create_service() function missing or not callable")
    except Exception as e:
        errors.append(("create_service() check", str(e)))
        print(f"❌ Error checking create_service(): {e}")

    # Test 4: Check for initialize() function (optional)
    try:
        initialize_func = getattr(adapter, "initialize", None)
        if initialize_func and callable(initialize_func):
            successes.append("initialize() function exists")
            print("✅ initialize() function found (optional)")
        else:
            print("ℹ️  initialize() function not found (optional)")
    except Exception as e:
        print(f"ℹ️  Error checking initialize(): {e} (optional)")

    # Test 5: Check for shutdown() function (optional)
    try:
        shutdown_func = getattr(adapter, "shutdown", None)
        if shutdown_func and callable(shutdown_func):
            successes.append("shutdown() function exists")
            print("✅ shutdown() function found (optional)")
        else:
            print("ℹ️  shutdown() function not found (optional)")
    except Exception as e:
        print(f"ℹ️  Error checking shutdown(): {e} (optional)")

    # Test 6: Test create_service() returns correct structure
    try:
        service = adapter.create_service()
        if isinstance(service, dict):
            successes.append("create_service() returns dict")
            print("✅ create_service() returns dict")

            if "custom_handlers" in service:
                successes.append("create_service() has custom_handlers")
                print("✅ create_service() contains 'custom_handlers'")

                handlers = service["custom_handlers"]
                if isinstance(handlers, dict):
                    successes.append("custom_handlers is dict")
                    print(f"✅ custom_handlers is dict with {len(handlers)} handlers")

                    # Check for expected handlers
                    expected_handlers = ["chat", "health"]
                    for handler_name in expected_handlers:
                        if handler_name in handlers:
                            successes.append(f"Handler '{handler_name}' exists")
                            print(f"✅ Handler '{handler_name}' found")
                        else:
                            errors.append((f"Handler '{handler_name}'", "Not found"))
                            print(f"❌ Handler '{handler_name}' missing")
                else:
                    errors.append(("custom_handlers type", f"Expected dict, got {type(handlers)}"))
                    print(f"❌ custom_handlers is not a dict: {type(handlers)}")
            else:
                errors.append(("custom_handlers key", "Missing from service dict"))
                print("❌ 'custom_handlers' key missing from service dict")
        else:
            errors.append(("create_service() return type", f"Expected dict, got {type(service)}"))
            print(f"❌ create_service() returns {type(service)}, not dict")
    except Exception as e:
        errors.append(("create_service() execution", str(e)))
        print(f"❌ Error calling create_service(): {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Results: {len(successes)} passed, {len(errors)} failed")
    print(f"{'='*60}")

    if errors:
        print("\nFailed tests:")
        for test_name, error in errors:
            print(f"  - {test_name}: {error}")
        return False
    else:
        print("\n✅ All PAR adapter tests passed!")
        return True


if __name__ == "__main__":
    success = test_par_adapter()
    sys.exit(0 if success else 1)
