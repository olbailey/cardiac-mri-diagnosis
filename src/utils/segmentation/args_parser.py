import argparse
import yaml
import os
import re

def process_model_path(base_dir: str):
    """Build a type= function that turns a bare name into <base_dir>/<name><extension>.

    Args:
        base_dir (str): _description_
        extension (str, optional): _description_. Defaults to ".pth".

    Raises:
        argparse.ArgumentTypeError: model name cannot be empty
        argparse.ArgumentTypeError: Should be a name only, not a path
        argparse.ArgumentTypeError: Should not include a file extension
        argparse.ArgumentTypeError: Contains invalid characters (use letters, numbers, '_' and '-')
    """

    def convert(value: str):
        name = value.strip()

        if not name:
            raise argparse.ArgumentTypeError("model name cannot be empty")

        # Catch paths first so the error message is the most helpful one
        if os.path.basename(name) != name or name in (".", ".."):
            raise argparse.ArgumentTypeError(
                f"'{value}' should be a name only, not a path"
            )

        # No dots at all, so no extension can be supplied
        if "." in name:
            raise argparse.ArgumentTypeError(
                f"'{value}' should not include a file extension "
            )

        # fullmatch: whole string must be letters | digits | '_' | '-' (\w = letter, digit or _; \- = literal hyphen; + = one or more)
        if not re.fullmatch(r"[\w\-]+", name):
            raise argparse.ArgumentTypeError(
                f"'{value}' contains invalid characters "
                "(use letters, numbers, '_' and '-')"
            )

        full_path = os.path.join(os.path.abspath(base_dir), name)

        return full_path

    return convert


def parse_args(base_dir: str):
    # Stage 1: only find out which config file to use, since the
    # other defaults come from it
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="configs/unet.yaml",
                     help="Path to the YAML config file")
    pre_args, _ = pre.parse_known_args()

    with open(pre_args.config, "r") as f:
        configs = yaml.safe_load(f)

    # Stage 2: the full parser, with config values as defaults
    parser = argparse.ArgumentParser(
        description="Train an MRI segmentation model", parents=[pre]
    )
    parser.add_argument("--model-name", default=configs["data"]["model_name"],
                        help="Where the final model is saved", type=process_model_path(base_dir=base_dir))
    parser.add_argument("--train-path", default="data/processed/training")
    parser.add_argument("--test-path", default="data/processed/testing")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=configs["training"]["learning_rate"])
    parser.add_argument("--batch-size", type=int, default=configs["training"]["batch_size"])
    parser.add_argument("--test-slice", default="patient120_slice02_MINF.npy",
                        help="Slice to visualise on Ctrl+C")
    parser.add_argument("--save-model", action="store_true",
                        help="Save the model when training is interrupted")

    return parser.parse_args(), configs

if __name__ == "__main__":
    args, configs = parse_args()