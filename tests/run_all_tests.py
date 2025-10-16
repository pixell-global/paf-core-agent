"""
Master Test Runner - Run all phase tests

This script runs all 6 phase tests in sequence and provides a summary.
"""

import sys
import subprocess
from pathlib import Path


def run_test(test_file, phase_name):
    """Run a single test file and return results"""
    print("\n" + "🔥" * 30)
    print(f"Running {phase_name}")
    print("🔥" * 30 + "\n")

    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            cwd=test_file.parent.parent,
            capture_output=True,
            text=True,
            timeout=180
        )

        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        success = result.returncode == 0
        return success, phase_name

    except subprocess.TimeoutExpired:
        print(f"❌ {phase_name} timed out after 180 seconds")
        return False, phase_name
    except Exception as e:
        print(f"❌ Error running {phase_name}: {e}")
        return False, phase_name


def main():
    """Run all phase tests"""
    print("=" * 80)
    print(" " * 20 + "PAF-CORE APKG CONVERSION TEST SUITE")
    print("=" * 80)

    tests_dir = Path(__file__).parent

    # Define all tests in order
    tests = [
        (tests_dir / "test_phase1_directory_rename.py", "Phase 1: Directory Rename"),
        (tests_dir / "test_phase2_par_adapter.py", "Phase 2: PAR Adapter"),
        (tests_dir / "test_phase3_agent_manifest.py", "Phase 3: Agent Manifest"),
        (tests_dir / "test_phase4_configuration.py", "Phase 4: Configuration"),
        (tests_dir / "test_phase5_main_updates.py", "Phase 5: Main.py Updates"),
        (tests_dir / "test_phase6_apkg_build.py", "Phase 6: APKG Build"),
    ]

    results = []
    passed_count = 0
    failed_count = 0

    # Run each test
    for test_file, phase_name in tests:
        if not test_file.exists():
            print(f"\n❌ Test file not found: {test_file}")
            results.append((False, phase_name))
            failed_count += 1
            continue

        success, name = run_test(test_file, phase_name)
        results.append((success, name))

        if success:
            passed_count += 1
        else:
            failed_count += 1

    # Final summary
    print("\n" + "=" * 80)
    print(" " * 25 + "FINAL TEST SUMMARY")
    print("=" * 80)

    for success, phase_name in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {phase_name}")

    print("\n" + "=" * 80)
    print(f"Total: {passed_count} passed, {failed_count} failed out of {len(results)} phases")
    print("=" * 80)

    if failed_count == 0:
        print("\n🎉 🎉 🎉 ALL TESTS PASSED! 🎉 🎉 🎉")
        print("\n✅ PAF-Core is ready for APKG deployment!")
        return 0
    else:
        print(f"\n❌ {failed_count} phase(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
