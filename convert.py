import os
import argparse

def convert(name, label, num):
    filepath = 'linemod/{}/'.format(name)
    txt = os.path.join(filepath, 'labels.txt')
    print("filepath: {}\ntxt: {}".format(filepath, txt))
    number = num + 1

    with open(txt, 'w') as f:
        for idx in range(number):
            filepath = 'linemod/{}/'.format(name)
            imageidx = str(10000 + idx)
            imageidx = imageidx[1:]
            imagename = 'real_' + imageidx + '_img.png'
            filepath = os.path.join(filepath, imagename)
            f.write(filepath + ' ' + str(label) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--label', default=0, type=int)
    parser.add_argument('--num', default=0, type=int)
    args = parser.parse_args()

    convert(args.name, args.label, args.num)
        