import argparse
from dataclasses import dataclass


def range_type(astr, a=0, b=1, a_inclusive=False, b_inclusive=False):
    value = float(astr)
    if (a <= value if a_inclusive else a < value) and (value <= b if b_inclusive else value < b):
        return value
    else:
        raise argparse.ArgumentTypeError('value not in range %s-%s'%(a,b))


parser = argparse.ArgumentParser()
parser.add_argument("learning_rate",
                    type=lambda lr: range_type(lr, 0, 1),
                    nargs='?',
                    metavar='[0.0-1.0]',
                    help="Learning rate to be passed to each strategy trainer")

parser.add_argument("-r", "--reuse_mesh", help="Connects to existing mesh using information from env files",
                    action="store_false")
args = parser.parse_args()

@dataclass
class Test:
    learning_rate: float = 0.0

print(Test(args.learning_rate).learning_rate)
print(args.reuse_mesh)