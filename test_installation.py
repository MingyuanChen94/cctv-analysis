"""
Test script to verify package installation.
"""

def test_imports():
    """Test importing package components."""
    print("Testing imports...")
    
    try:
        import cctv_analysis
        print("✓ Successfully imported cctv_analysis")
        print(f"  Package location: {cctv_analysis.__file__}")
        
        from cctv_analysis import CameraProcessor
        print("✓ Successfully imported CameraProcessor")
        
        from cctv_analysis.utils import setup_logging
        print("✓ Successfully imported utils")
        
        print("\nAll imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n✗ Import error: {str(e)}")
        return False

def check_package_structure():
    """Check if package files exist in the correct locations."""
    from pathlib import Path
    
    print("\nChecking package structure...")
    required_files = [
        "src/cctv_analysis/__init__.py",
        "src/cctv_analysis/camera_processor.py",
        "src/cctv_analysis/utils/__init__.py",
        "setup.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ Found {file_path}")
        else:
            print(f"✗ Missing {file_path}")
            all_exist = False
    
    return all_exist

if __name__ == "__main__":
    print("CCTV Analysis Package Installation Test")
    print("=" * 40)
    
    # Test imports
    imports_ok = test_imports()
    
    # Check structure
    structure_ok = check_package_structure()
    
    # Print summary
    print("\nTest Summary:")
    print(f"Imports: {'✓' if imports_ok else '✗'}")
    print(f"Structure: {'✓' if structure_ok else '✗'}")
    
    if not (imports_ok and structure_ok):
        print("\nTroubleshooting steps:")
        print("1. Make sure you're in the project root directory")
        print("2. Run: pip uninstall cctv-analysis -y")
        print("3. Run: pip install -e .")
        print("4. Check if all required files exist")