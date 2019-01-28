import os
import cv2
import random
def mask():
    path = 'LINEMOD/ape/mask/'
    for ro, _, fi in os.walk(path):
        root, __, files = ro, _, fi
    image_name = files[0]
    mask = cv2.imread(os.path.join(path, image_name))
    bg_txt = 'VOCdevkit/VOC2012/ImageSets/Layout/trainval.txt'
    with open(bg_txt, 'r') as f:
        bg_files = [x.split()[0] for x in f.readlines()]
    rand_num = random.randint(0, 850)
    bg_file = bg_files[rand_num]
    bg_file_path = 'VOCdevkit/VOC2012/JPEGImages/' + bg_file + '.jpg'
    bg = cv2.imread(bg_file_path)
    bg = cv2.resize(bg, (640, 480))
    obj_path = 'LINEMOD/ape/JPEGImages/' + '00' + image_name[:-3] + 'jpg'
    obj = cv2.imread(obj_path)
    obj[mask == 0] = 0
    cv2.imwrite('obj.jpg', obj)
    bg[mask == 255] = 0
    cv2.imwrite('bg.jpg', bg)
    res = obj + bg
    cv2.imwrite('res.jpg', res)
    

    print("end")


if __name__ == "__main__":
    mask()