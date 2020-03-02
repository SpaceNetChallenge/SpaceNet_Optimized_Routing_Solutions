import os
import argparse
from shutil import copyfile

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--indirs', nargs='+')
    parser.add_argument('--outdir', type=str, default='')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    for indir in args.indirs:
        if '/PS-MS' not in indir:
            indir = os.path.join(indir, 'PS-MS')

        for f in os.listdir(indir):
            n = 'AOI' + f.split('AOI')[-1]
            n = n.replace('_PS-MS', '')

            _from = os.path.join(indir, f)
            to = os.path.join(args.outdir, n)
            copyfile(_from, to)

