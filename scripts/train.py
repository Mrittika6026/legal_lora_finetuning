import sys
from pathlib import Path

# Add src to path if running from scripts directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from lora_finetune.main import main

if __name__ == "__main__":
    main()

