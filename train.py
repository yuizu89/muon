from cli import build_train_argument_parser
from training.experiment import run_experiment


def main() -> None:
    parser = build_train_argument_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
