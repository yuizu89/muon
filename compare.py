from cli import build_compare_argument_parser
from training.experiment import run_comparison


def main() -> None:
    parser = build_compare_argument_parser()
    args = parser.parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()
