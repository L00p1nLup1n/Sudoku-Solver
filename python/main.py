"""CLI entry for the Python Sudoku app."""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Python Sudoku App with OCR and Text Extraction"
    )
    parser.add_argument("--gui", action="store_true", help="Run the Tkinter GUI")
    parser.add_argument(
        "--text", metavar="IMAGE_PATH", help="Extract text from an image file"
    )
    parser.add_argument(
        "--output",
        metavar="OUTPUT_DIR",
        default="output",
        help="Output directory for extracted text (default: output)",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip preprocessing when extracting text",
    )
    parser.add_argument(
        "--method",
        choices=["adaptive", "otsu", "enhanced", "combined"],
        default="enhanced",
        help="Preprocessing method: adaptive, otsu, enhanced, or combined (default: enhanced)",
    )
    parser.add_argument(
        "--psm",
        type=int,
        choices=range(0, 14),
        default=3,
        help="Page segmentation mode (0-13). Common: 3=auto, 6=single block, 11=sparse text (default: 3)",
    )
    parser.add_argument(
        "--save-preprocessed",
        action="store_true",
        help="Save preprocessed image for debugging",
    )

    args = parser.parse_args()

    if args.gui:
        from gui import run_gui

        run_gui()
    elif args.text:
        from text_extractor import extract_and_save

        try:
            output_path = extract_and_save(
                args.text,
                output_dir=args.output,
                preprocess=not args.no_preprocess,
                method=args.method,
                psm=args.psm,
                save_preprocessed=args.save_preprocessed,
            )
            print(f"Successfully extracted text to: {output_path}")
        except Exception as e:
            print(f"Error: {e}")
            exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
