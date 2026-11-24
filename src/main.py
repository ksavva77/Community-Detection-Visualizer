"""Run the whole project: dataset summary, raw graphs and the full comparison.

Run from the repository root::

    python src/main.py
"""

import community_detection
import comparison


def main():
    """Print the dataset summary, then run and save the full comparison."""
    community_detection.main()
    comparison.main()


if __name__ == "__main__":
    main()
