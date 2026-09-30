"""Command line interface: train, evaluate and predict.

Examples:
    python -m lunchlab.cli train --data data/lunches.csv --out model.json
    python -m lunchlab.cli evaluate --model-file model.json --data data/lunches.csv
    python -m lunchlab.cli predict --model-file model.json --input samples/new_lunches.csv
"""

import argparse
import csv
import sys
from typing import List, Optional

from lunchlab.data import load_csv, train_test_split
from lunchlab.metrics import classification_report
from lunchlab.models import MODELS
from lunchlab.models.knn import KNNClassifier
from lunchlab.models.logistic import LogisticRegression
from lunchlab.persistence import load_bundle, save_bundle
from lunchlab.preprocessing import SCALERS


def _print_report(report: dict) -> None:
    tn, fp = report["confusion_matrix"][0]
    fn, tp = report["confusion_matrix"][1]
    print(f"accuracy : {report['accuracy']:.3f}")
    print(f"precision: {report['precision']:.3f}")
    print(f"recall   : {report['recall']:.3f}")
    print(f"f1       : {report['f1']:.3f}")
    print("confusion matrix (rows = true, columns = predicted):")
    print(f"           pred 0  pred 1")
    print(f"  true 0   {tn:6d}  {fp:6d}")
    print(f"  true 1   {fn:6d}  {tp:6d}")


def _check_features(expected: List[str], got: List[str]) -> None:
    if expected != got:
        raise ValueError(f"feature columns do not match: model expects {expected}, file has {got}")


def _build_model(args: argparse.Namespace):
    return LogisticRegression(lr=args.lr, epochs=args.epochs)


def _train(args: argparse.Namespace) -> None:
    X, y, names = load_csv(args.data, target=args.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, seed=args.seed
    )

    scaler = SCALERS[args.scaler]()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = _build_model(args)
    model.fit(X_train, y_train)

    report = classification_report(y_test, model.predict(X_test))
    print(f"trained logistic on {len(X_train)} lunches, testing on {len(X_test)}")
    _print_report(report)

    save_bundle(args.out, model, scaler, names)
    print(f"saved to {args.out}")


def _evaluate(args: argparse.Namespace) -> None:
    model, scaler, names = load_bundle(args.model_file)
    X, y, got_names = load_csv(args.data, target=args.target)
    _check_features(names, got_names)

    report = classification_report(y, model.predict(scaler.transform(X)))
    print(f"evaluated on {len(X)} lunches")
    _print_report(report)


def _predict(args: argparse.Namespace) -> None:
    model, scaler, names = load_bundle(args.model_file)
    X, _, got_names = load_csv(args.input, target=None)
    _check_features(names, got_names)

    predictions = model.predict(scaler.transform(X))
    for i, p in enumerate(predictions, start=1):
        print(f"lunch {i}: {'good' if p == 1 else 'not good'}")

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["row", "good_lunch"])
            for i, p in enumerate(predictions, start=1):
                writer.writerow([i, int(p)])
        print(f"wrote {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lunchlab", description="Will the mess lunch be good?")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train a model and save it")
    train.add_argument("--data", required=True, help="CSV file with a header row")
    train.add_argument("--target", default="good_lunch", help="label column name")
    train.add_argument("--scaler", choices=sorted(SCALERS), default="minmax")
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--lr", type=float, default=0.1, help="logistic regression learning rate")
    train.add_argument("--epochs", type=int, default=1000, help="logistic regression epochs")
    train.add_argument("--out", default="model.json", help="where to save the trained model")
    train.set_defaults(func=_train)

    evaluate = sub.add_parser("evaluate", help="evaluate a saved model on a labelled CSV")
    evaluate.add_argument("--model-file", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--target", default="good_lunch")
    evaluate.set_defaults(func=_evaluate)

    predict = sub.add_parser("predict", help="predict for lunches in a CSV of features only")
    predict.add_argument("--model-file", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", help="optional CSV file to write predictions to")
    predict.set_defaults(func=_predict)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the CLI. Returns 0 on success and 1 on a user-facing error."""
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, KeyError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
