import pytest
import sys
import io

# Redirect stdout to capture output
class CatchOut:
    def __init__(self):
        self.value = ""
    def write(self, txt):
        self.value += txt
    def flush(self):
        pass
    def isatty(self):
        return False

old_stdout = sys.stdout
old_stderr = sys.stderr
catcher = CatchOut()
sys.stdout = catcher
sys.stderr = catcher

try:
    # Run tests
    result = pytest.main(["tests/test_auth.py", "-v"])
finally:
    # Restore stdout
    sys.stdout = old_stdout
    sys.stderr = old_stderr

# Print captured output to a file that we can read
with open("test_results.log", "w", encoding="utf-8") as f:
    f.write(catcher.value)
    f.write(f"\nExit Code: {result}")

print("Tests completed. Check test_results.log")
