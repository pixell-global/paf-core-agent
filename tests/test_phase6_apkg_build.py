"""
Test Phase 6: Verify APKG package build

This test verifies:
1. APKG package can be built successfully
2. Package file exists and has reasonable size
3. Package contains required files
4. Package structure is correct
"""

import sys
import subprocess
from pathlib import Path
import zipfile
import shutil


def test_pixell_cli_available():
    """Test that pixell CLI is available"""
    print("=" * 60)
    print("Phase 6: Pixell CLI Availability Test")
    print("=" * 60)

    errors = []
    successes = []

    try:
        result = subprocess.run(
            ["pixell", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            successes.append("pixell CLI available")
            version = result.stdout.strip() or result.stderr.strip()
            print(f"✅ pixell CLI available: {version}")
            return True, successes, errors
        else:
            errors.append(("pixell CLI", "Command failed"))
            print("❌ pixell CLI command failed")
            return False, successes, errors

    except FileNotFoundError:
        errors.append(("pixell CLI", "Not found in PATH"))
        print("❌ pixell CLI not found in PATH")
        return False, successes, errors
    except Exception as e:
        errors.append(("pixell CLI", str(e)))
        print(f"❌ Error checking pixell CLI: {e}")
        return False, successes, errors


def test_build_apkg():
    """Test building the APKG package"""
    print("\n" + "=" * 60)
    print("Phase 6: APKG Build Test")
    print("=" * 60)

    errors = []
    successes = []

    project_root = Path(__file__).parent.parent
    dist_dir = project_root / "dist"

    # Clean dist directory
    if dist_dir.exists():
        print("ℹ️  Cleaning existing dist/ directory...")
        shutil.rmtree(dist_dir)

    try:
        print("ℹ️  Running 'pixell build --output ./dist'...")
        result = subprocess.run(
            ["pixell", "build", "--output", "./dist"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            successes.append("pixell build succeeded")
            print("✅ pixell build succeeded")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()}")
            return True, successes, errors
        else:
            errors.append(("pixell build", result.stderr or result.stdout))
            print("❌ pixell build failed")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()}")
            return False, successes, errors

    except subprocess.TimeoutExpired:
        errors.append(("pixell build", "Timeout after 120s"))
        print("❌ pixell build timed out")
        return False, successes, errors
    except Exception as e:
        errors.append(("pixell build", str(e)))
        print(f"❌ Error running pixell build: {e}")
        return False, successes, errors


def test_package_exists():
    """Test that APKG package file was created"""
    print("\n" + "=" * 60)
    print("Phase 6: Package File Existence Test")
    print("=" * 60)

    errors = []
    successes = []

    project_root = Path(__file__).parent.parent
    dist_dir = project_root / "dist"

    if not dist_dir.exists():
        errors.append(("dist/ directory", "Does not exist"))
        print("❌ dist/ directory does not exist")
        return False, successes, errors, None

    # Find .apkg files
    apkg_files = list(dist_dir.glob("*.apkg"))

    if len(apkg_files) == 0:
        errors.append(("APKG file", "No .apkg files found in dist/"))
        print("❌ No .apkg files found in dist/")
        return False, successes, errors, None

    apkg_file = apkg_files[0]
    successes.append(f"APKG file created: {apkg_file.name}")
    print(f"✅ APKG file created: {apkg_file.name}")

    # Check file size
    file_size = apkg_file.stat().st_size
    size_mb = file_size / (1024 * 1024)

    if file_size > 0:
        successes.append(f"Package size: {size_mb:.2f} MB")
        print(f"✅ Package size: {size_mb:.2f} MB ({file_size:,} bytes)")

        # Reasonable size check (0.1 MB to 100 MB)
        if 0.1 <= size_mb <= 100:
            successes.append("Package size is reasonable")
            print(f"✅ Package size is reasonable")
        else:
            errors.append(("Package size", f"{size_mb:.2f} MB seems unusual"))
            print(f"⚠️  Package size {size_mb:.2f} MB seems unusual")
    else:
        errors.append(("Package size", "File is empty"))
        print("❌ Package file is empty")
        return False, successes, errors, None

    return True, successes, errors, apkg_file


def test_package_structure(apkg_file):
    """Test that package contains required files"""
    print("\n" + "=" * 60)
    print("Phase 6: Package Structure Test")
    print("=" * 60)

    errors = []
    successes = []

    try:
        with zipfile.ZipFile(apkg_file, 'r') as zip_ref:
            file_list = zip_ref.namelist()

            successes.append(f"Package contains {len(file_list)} files")
            print(f"✅ Package contains {len(file_list)} files")

            # Check for required files
            required_files = [
                "agent.yaml",
                "requirements.txt",
                "src/par_adapter.py",
                "src/main.py",
                "src/schemas.py",
                "src/settings.py",
            ]

            for required_file in required_files:
                if required_file in file_list:
                    successes.append(f"File '{required_file}' present")
                    print(f"✅ {required_file} present")
                else:
                    errors.append((f"File '{required_file}'", "Missing from package"))
                    print(f"❌ {required_file} missing from package")

            # Check for .env (should be present, not .env.example)
            if ".env" in file_list:
                successes.append(".env file present")
                print("✅ .env file present")
            else:
                print("ℹ️  .env file not present (may use environment variables)")

            # Check that src/ directory has Python files
            src_files = [f for f in file_list if f.startswith("src/") and f.endswith(".py")]
            if len(src_files) > 10:
                successes.append(f"Package has {len(src_files)} Python files in src/")
                print(f"✅ Package has {len(src_files)} Python files in src/")
            else:
                errors.append(("src/ Python files", f"Only {len(src_files)} found"))
                print(f"❌ Only {len(src_files)} Python files in src/ (expected more)")

            # Check for core directories
            core_dirs = set()
            for file_path in file_list:
                if "/" in file_path:
                    core_dirs.add(file_path.split("/")[0])

            expected_dirs = ["src"]
            for dir_name in expected_dirs:
                if dir_name in core_dirs:
                    successes.append(f"Directory '{dir_name}/' present")
                    print(f"✅ {dir_name}/ directory present")
                else:
                    errors.append((f"Directory '{dir_name}/'", "Missing"))
                    print(f"❌ {dir_name}/ directory missing")

            # Show sample of files
            print(f"\n   Sample files in package:")
            for i, file_path in enumerate(sorted(file_list)[:10]):
                print(f"     - {file_path}")
            if len(file_list) > 10:
                print(f"     ... and {len(file_list) - 10} more files")

    except zipfile.BadZipFile:
        errors.append(("Package format", "Not a valid ZIP file"))
        print("❌ Package is not a valid ZIP file")
        return False, successes, errors
    except Exception as e:
        errors.append(("Package inspection", str(e)))
        print(f"❌ Error inspecting package: {e}")
        return False, successes, errors

    return len(errors) == 0, successes, errors


def test_package_validity():
    """Test that package can be validated"""
    print("\n" + "=" * 60)
    print("Phase 6: Package Validation Test")
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
            print("⚠️  pixell validate failed")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")

    except subprocess.TimeoutExpired:
        errors.append(("pixell validate", "Timeout"))
        print("❌ pixell validate timed out")
    except Exception as e:
        errors.append(("pixell validate", str(e)))
        print(f"⚠️  Error running pixell validate: {e}")

    return len(errors) == 0, successes, errors


def run_all_tests():
    """Run all Phase 6 tests"""
    print("\n" + "🧪" * 30)
    print("PHASE 6: APKG PACKAGE BUILD")
    print("🧪" * 30 + "\n")

    all_successes = []
    all_errors = []
    results = []

    # Test 1: Pixell CLI available
    passed, successes, errors = test_pixell_cli_available()
    results.append(passed)
    all_successes.extend(successes)
    all_errors.extend(errors)

    if not passed:
        print("\n❌ Cannot proceed: pixell CLI not available")
        print("   Install with: pip install pixell-cli")
        return False

    # Test 2: Build APKG
    passed, successes, errors = test_build_apkg()
    results.append(passed)
    all_successes.extend(successes)
    all_errors.extend(errors)

    if not passed:
        print("\n❌ Cannot proceed: APKG build failed")
        return False

    # Test 3: Package exists
    passed, successes, errors, apkg_file = test_package_exists()
    results.append(passed)
    all_successes.extend(successes)
    all_errors.extend(errors)

    if not passed or apkg_file is None:
        print("\n❌ Cannot proceed: APKG package not found")
        return False

    # Test 4: Package structure
    passed, successes, errors = test_package_structure(apkg_file)
    results.append(passed)
    all_successes.extend(successes)
    all_errors.extend(errors)

    # Test 5: Package validation
    passed, successes, errors = test_package_validity()
    results.append(passed)
    all_successes.extend(successes)
    all_errors.extend(errors)

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 6 SUMMARY")
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
        print("\n🎉 Phase 6: ALL TESTS PASSED")
        print(f"\n📦 APKG package ready for deployment!")
    else:
        print("\n❌ Phase 6: SOME TESTS FAILED")

    return overall_pass


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
