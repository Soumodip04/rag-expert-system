"""
Tests for the RAG Expert System.
This file contains basic tests to ensure the system can be imported and initialized.
"""

import unittest
import os
import sys

# Add the parent directory to the path so we can import the system modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try importing key modules to ensure they're available
try:
    import src
    TEST_IMPORTS_WORK = True
except ImportError:
    TEST_IMPORTS_WORK = False


class TestRAGSystemBasics(unittest.TestCase):
    """Basic tests for the RAG Expert System."""

    def test_imports(self):
        """Test that the system modules can be imported."""
        self.assertTrue(TEST_IMPORTS_WORK, "Failed to import the 'src' module")

    def test_system_structure(self):
        """Test that the system directory structure exists."""
        self.assertTrue(os.path.exists("src"), "The 'src' directory does not exist")

    def test_env_example_exists(self):
        """Test that the .env.example file exists."""
        self.assertTrue(os.path.exists(".env.example"), "The '.env.example' file does not exist")


if __name__ == "__main__":
    unittest.main()
