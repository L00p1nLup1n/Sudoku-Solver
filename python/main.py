"""CLI entry for the Python Sudoku app."""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true', help='Run the Tkinter GUI')
    args = parser.parse_args()
    if args.gui:
        from .gui import run_gui
        run_gui()

if __name__ == '__main__':
    main()
