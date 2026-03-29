import os
from pathlib import Path

# Get project root directory (assuming config.py is in the project root)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Define paths relative to project root
PROMPTS_DIR = os.path.join(PROJECT_ROOT, "prompts")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SCENE_DIR = os.path.join(DATA_DIR, "preassembly_scenes")
MANUAL_DATA_PATH = os.path.join(DATA_DIR, "main_data.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

RECIPE_PATH = os.path.join(PROJECT_ROOT, "recipe.yaml")

# Default model settings
DEFAULT_MODEL = "gemini-pro"
DEFAULT_MAX_TOKENS = 65000
DEFAULT_TEMPERATURE = 0

# Output file names for --output_format modes
ACTIONS_FILENAME = "actions.json"
PDDL_DOMAIN_FILENAME = "domain.pddl"
PDDL_PROBLEM_FILENAME = "problem.pddl"
BEHAVIOR_TREE_FILENAME = "behavior_tree.xml"

# Static PDDL domain file (reusable across all furniture items)
PDDL_DIR = os.path.join(PROJECT_ROOT, "pddl")