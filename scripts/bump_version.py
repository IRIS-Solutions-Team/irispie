
# Write a module to read pyproject.toml and bump the version number in it,
# depending on the argument passed to the script.

from typing import (Literal, )
import argparse
import toml

def bump_version(version: str, bump: Literal["major", "minor", "patch"]) -> str:
    """Bump the version number"""
    major, minor, patch = version.split(".")
    if bump == "major":
        major = str(int(major) + 1)
    elif bump == "minor":
        minor = str(int(minor) + 1)
    elif bump == "patch":
        patch = str(int(patch) + 1)
    return ".".join((major, minor, patch, ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bump_type", choices=["major", "minor", "patch"])
    parser.add_argument("--pyproject_path", default="pyproject.toml")
    args = parser.parse_args()

    with open(args.pyproject_path, "rt", ) as f:
        pyproject = toml.load(f)
    current_version = pyproject["project"]["version"]
    bumped_version = bump_version(current_version, args.bump_type, )
    pyproject["project"]["version"] = bumped_version
    with open("test.toml", "wt", ) as f:
       toml.dump(pyproject, f)
    print(toml.dumps(pyproject))


if __name__ == "__main__":
    main()

