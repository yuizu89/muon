from cli import build_plot_argument_parser, validate_plot_args
from reporting.plotting import plot_results


def main() -> None:
    parser = build_plot_argument_parser()
    args = parser.parse_args()
    validate_plot_args(args)

    saved_paths = plot_results(
        input_path=args.input_path,
        output_dir=args.output_dir,
        file_format=args.format,
        dpi=args.dpi,
    )

    for path in saved_paths:
        print("Saved plot: {0}".format(path))


if __name__ == "__main__":
    main()
