#!/usr/bin/env python3
"""
Development installation script for CCTV Analysis package.
This script sets up the package in development mode and verifies the installation.
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Get project root directory
    project_root = Path(__file__).parent.parent.absolute()
    
    print("Installing CCTV Analysis package in development mode...")
    
    # Install package in development mode
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(project_root)],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error installing package: {e}")
        sys.exit(1)
    
    # Verify installation
    print("\nVerifying installation...")
    try:
        import cctv_analysis
        print(f"Successfully installed CCTV Analysis version {cctv_analysis.__version__}")
        
        # Test imports of main components
        from cctv_analysis.utils.config import Config
        from cctv_analysis.analysis import CCTVAnalysis
        print("All main components successfully imported")
        
    except ImportError as e:
        print(f"Error importing package: {e}")
        sys.exit(1)
    
    print("\nInstallation complete!")
    print("\nYou can now use the package in Python:")
    print(">>> from cctv_analysis.utils.config import Config")
    print(">>> from cctv_analysis.analysis import CCTVAnalysis")

if __name__ == "__main__":
    main()
