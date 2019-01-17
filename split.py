import os
import argparse

def split(name):
    filepath = 'linemod/cfg/' + name + "/"
    label = os.path.join(filepath, 'labels.txt')
    print('filepath: {}'.format(filepath))

    with open(label, 'r') as f:
        label_list = [x.strip() for x in f.readlines()]

    print(len(label_list))
    train_file = os.path.join(filepath, 'train.txt')
    test_file = os.path.join(filepath, 'test.txt')

    with open(train_file, 'w') as train:
        with open(test_file, 'w') as test:
            for i in range(len(label_list)):
                if i % 4 == 0:
                    test.write(label_list[i] + '\n')
                else:
                    train.write(label_list[i] + '\n')


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='ape', type=str)
    args = parser.parse_args()

    split(args.name)