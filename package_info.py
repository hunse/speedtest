from __future__ import print_function

import argparse
import imp
import os
import subprocess


def _clean_output(lines):
    return [line for line in lines if line]


def package_info(package):
    print("Package %r" % package)
    print("Version %s" % package.__version__)

    path = os.path.dirname(package.__file__)
    print("Path %s" % path)

    try:
        branch = _clean_output(subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=path,
        ).decode('ascii').split('\n'))[0]
        commit = _clean_output(subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], cwd=path,
        ).decode('ascii').split('\n'))[0]
        print("Branch: %s, commit: %s" % (branch, commit))
    except subprocess.CalledProcessError:
        pass


parser = argparse.ArgumentParser(
    description="Info about Python package versions and Git repo")
parser.add_argument('packages', nargs='+', help="Packages to display")
args = parser.parse_args()

for package_name in args.packages:
    try:
        print(package_name)
        package = imp.load_module(package_name, *imp.find_module(package_name))
        package_info(package)
        print()
    except ImportError as e:
        print(e)
