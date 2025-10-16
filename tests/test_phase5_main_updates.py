"""
Test Phase 5: Verify main.py updates

This test verifies:
1. src/main.py exists and has dev-mode warnings
2. Uses correct import path (src.main:app not app.main:app)
3. Has proper documentation about APKG vs dev mode
4. FastAPI app is configured correctly
"""

import sys
import ast
from pathlib import Path
import re


def test_main_file_exists():
    """Test that src/main.py exists"""
    print("=" * 60)
    print("Phase 5: main.py File Existence Test")
    print("=" * 60)

    errors = []
    successes = []

    project_root = Path(__file__).parent.parent
    main_file = project_root / "src" / "main.py"

    if not main_file.exists():
        errors.append(("src/main.py", "File does not exist"))
        print("❌ src/main.py does not exist")
        return False, successes, errors, None
    else:
        successes.append("src/main.py exists")
        print("✅ src/main.py exists")

    # Read file content
    try:
        with open(main_file, 'r') as f:
            content = f.read()
        successes.append("src/main.py is readable")
        print("✅ src/main.py is readable")
        return True, successes, errors, content
    except Exception as e:
        errors.append(("src/main.py reading", str(e)))
        print(f"❌ Failed to read src/main.py: {e}")
        return False, successes, errors, None


def test_apkg_documentation(content):
    """Test that file has proper APKG vs dev mode documentation"""
    print("\n" + "=" * 60)
    print("Phase 5: APKG Documentation Test")
    print("=" * 60)

    errors = []
    successes = []

    # Check for key documentation phrases
    required_phrases = [
        ("DEVELOPMENT MODE", "Development mode warning"),
        ("NOT used when running as an APKG", "APKG usage note"),
        ("PAR invokes", "PAR reference"),
        ("par_adapter", "PAR adapter reference"),
    ]

    for phrase, description in required_phrases:
        if phrase in content:
            successes.append(f"{description} present")
            print(f"✅ {description} present")
        else:
            errors.append((description, "Missing"))
            print(f"❌ {description} missing")

    return len(errors) == 0, successes, errors


def test_import_path(content):
    """Test that uvicorn uses correct import path"""
    print("\n" + "=" * 60)
    print("Phase 5: Import Path Test")
    print("=" * 60)

    errors = []
    successes = []

    # Check for correct import path in uvicorn.run()
    if 'uvicorn.run' in content:
        successes.append("uvicorn.run() call found")
        print("✅ uvicorn.run() call found")

        # Check for correct path
        if '"src.main:app"' in content or "'src.main:app'" in content:
            successes.append("Correct import path 'src.main:app'")
            print("✅ Correct import path 'src.main:app'")
        else:
            errors.append(("Import path", "Not 'src.main:app'"))
            print("❌ Import path is not 'src.main:app'")

        # Check for old incorrect path
        if '"app.main:app"' in content or "'app.main:app'" in content:
            errors.append(("Old import path", "Still contains 'app.main:app'"))
            print("❌ Still contains old 'app.main:app' path")
        else:
            successes.append("No old 'app.main:app' path")
            print("✅ No old 'app.main:app' path found")

    else:
        print("ℹ️  uvicorn.run() not found (may be using different approach)")

    return len(errors) == 0, successes, errors


def test_dev_mode_warnings(content):
    """Test that file has dev mode warnings"""
    print("\n" + "=" * 60)
    print("Phase 5: Dev Mode Warnings Test")
    print("=" * 60)

    errors = []
    successes = []

    # Check for warning messages
    warning_patterns = [
        (r'Running in DEVELOPMENT mode', "Development mode warning"),
        (r'not for production', "Production warning"),
        (r'pixell build', "Build command reference"),
        (r'pixell deploy', "Deploy command reference"),
    ]

    for pattern, description in warning_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            successes.append(f"{description} present")
            print(f"✅ {description} present")
        else:
            errors.append((description, "Missing"))
            print(f"⚠️  {description} missing")

    return len(errors) == 0, successes, errors


def test_fastapi_app_structure(content):
    """Test that FastAPI app is properly structured"""
    print("\n" + "=" * 60)
    print("Phase 5: FastAPI App Structure Test")
    print("=" * 60)

    errors = []
    successes = []

    try:
        # Parse the Python file
        tree = ast.parse(content)

        # Check for FastAPI app creation
        fastapi_found = False
        for node in ast.walk(tree):
            # Look for: app = FastAPI(...)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'app':
                        if isinstance(node.value, ast.Call):
                            if hasattr(node.value.func, 'id') and node.value.func.id == 'FastAPI':
                                fastapi_found = True
                            elif hasattr(node.value.func, 'attr') and node.value.func.attr == 'FastAPI':
                                fastapi_found = True

        if fastapi_found:
            successes.append("FastAPI app creation found")
            print("✅ FastAPI app = FastAPI(...) found")
        else:
            errors.append(("FastAPI app", "Creation not found"))
            print("❌ FastAPI app creation not found")

        # Check for common FastAPI components
        components = [
            ("include_router", "Router includes"),
            ("add_middleware", "Middleware configuration"),
            ("lifespan", "Lifespan handler"),
        ]

        for component, description in components:
            if component in content:
                successes.append(f"{description} present")
                print(f"✅ {description} present")
            else:
                print(f"ℹ️  {description} not found (may be optional)")

    except SyntaxError as e:
        errors.append(("Python syntax", str(e)))
        print(f"❌ Python syntax error: {e}")
        return False, successes, errors

    return len(errors) == 0, successes, errors


def test_if_main_block(content):
    """Test that __main__ block is present"""
    print("\n" + "=" * 60)
    print("Phase 5: __main__ Block Test")
    print("=" * 60)

    errors = []
    successes = []

    if '__name__ == "__main__"' in content or "__name__ == '__main__'" in content:
        successes.append("__main__ block present")
        print("✅ __main__ block present")

        # Check for uvicorn.run in __main__ block
        if 'uvicorn.run' in content:
            successes.append("uvicorn.run in __main__ block")
            print("✅ uvicorn.run() found")
        else:
            errors.append(("uvicorn.run", "Not found in __main__ block"))
            print("❌ uvicorn.run() not found")

    else:
        errors.append(("__main__ block", "Missing"))
        print("❌ __main__ block missing")

    return len(errors) == 0, successes, errors


def test_no_syntax_errors(content):
    """Test that file has no Python syntax errors"""
    print("\n" + "=" * 60)
    print("Phase 5: Syntax Validation Test")
    print("=" * 60)

    errors = []
    successes = []

    try:
        ast.parse(content)
        successes.append("No syntax errors")
        print("✅ File has valid Python syntax")
    except SyntaxError as e:
        errors.append(("Python syntax", f"Line {e.lineno}: {e.msg}"))
        print(f"❌ Syntax error at line {e.lineno}: {e.msg}")

    return len(errors) == 0, successes, errors


def run_all_tests():
    """Run all Phase 5 tests"""
    print("\n" + "🧪" * 30)
    print("PHASE 5: MAIN.PY UPDATES")
    print("🧪" * 30 + "\n")

    all_successes = []
    all_errors = []
    results = []

    # Test 1: File exists
    passed, successes, errors, content = test_main_file_exists()
    results.append(passed)
    all_successes.extend(successes)
    all_errors.extend(errors)

    if not passed or content is None:
        print("\n❌ Cannot proceed: src/main.py failed to load")
        return False

    # Test 2: APKG documentation
    passed, successes, errors = test_apkg_documentation(content)
    results.append(passed)
    all_successes.extend(successes)
    all_errors.extend(errors)

    # Test 3: Import path
    passed, successes, errors = test_import_path(content)
    results.append(passed)
    all_successes.extend(successes)
    all_errors.extend(errors)

    # Test 4: Dev mode warnings
    passed, successes, errors = test_dev_mode_warnings(content)
    results.append(passed)
    all_successes.extend(successes)
    all_errors.extend(errors)

    # Test 5: FastAPI structure
    passed, successes, errors = test_fastapi_app_structure(content)
    results.append(passed)
    all_successes.extend(successes)
    all_errors.extend(errors)

    # Test 6: __main__ block
    passed, successes, errors = test_if_main_block(content)
    results.append(passed)
    all_successes.extend(successes)
    all_errors.extend(errors)

    # Test 7: Syntax validation
    passed, successes, errors = test_no_syntax_errors(content)
    results.append(passed)
    all_successes.extend(successes)
    all_errors.extend(errors)

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 5 SUMMARY")
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
        print("\n🎉 Phase 5: ALL TESTS PASSED")
    else:
        print("\n❌ Phase 5: SOME TESTS FAILED")

    return overall_pass


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
