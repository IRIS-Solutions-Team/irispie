
# Write a module to read pyproject.toml and bump the version number in it,
# depending on the argument passed to the script.

from typing import (Literal, )
import argparse
import toml


def _upgrade_version_string(version: str, bump: Literal["major", "minor", "patch"]) -> str:
    """Bump the version number"""
    def add_one(number: str, ) -> str:
        return str(int(number) + 1)
    major_minor_patch = version.split(".")
    if bump == "major":
        major_minor_patch[0] = add_one(major_minor_patch[0], )
        major_minor_patch[1] = "0"
        major_minor_patch[2] = "0"
    elif bump == "minor":
        major_minor_patch[1] = add_one(major_minor_patch[1], )
        major_minor_patch[2] = "0"
    elif bump == "patch":
        major_minor_patch[2] = add_one(major_minor_patch[2], )
    return ".".join(major_minor_patch, )


def main(args, ):
    with open(args.source_path, "rt", ) as f:
        file = toml.load(f, )
    current_version = file["project"]["version"]
    bumped_version = _upgrade_version_string(current_version, args.release_type, )
    file["project"]["version"] = bumped_version
    print(bumped_version, )
    if args.target_path:
        with open(args.target_path, "wt", ) as f:
            f.write(toml.dumps(file, ), )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release_type", choices=["major", "minor", "patch"])
    parser.add_argument("--source_path", )
    parser.add_argument("--target_path", default=None, )
    args = parser.parse_args()
    main(args, )

