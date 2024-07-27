import os
import cv2 as cv
from glob import glob
from tqdm import tqdm
import random
from shutil import copyfile

def getImages(path:str):
    HR_imgs = glob(os.path.join(path, '*[0-9].png'))
    LR_imgs = glob(os.path.join(path, '*LR.png'))
    print(len(HR_imgs), HR_imgs[:5])
    print(len(LR_imgs), LR_imgs[:5])

    return HR_imgs,LR_imgs

def blurImgs(imgs):
    for path in tqdm(imgs):
        i = cv.imread(path)
        assert i is not None
        # i = cv.blur(i, (7, 7))
        f = cv.resize(i, (8, 8))
        cv.imwrite(path[:-4]+'_LR8_.png', f)
        # break
        # f = cv.resize(i, (64, 64))
        # cv.imwrite(path[:-4]+'_LR64_.png', f)

def getLabel():
    sortedImgs = {}
    id_label = {}
    for sub, substr in enumerate(['aia', 'bonnie', 'jules', 'malcolm', 'mery', 'ray']):
        id_label[substr] = sub
        sortedImgs[substr] = {}

    exp_label = {}
    for exp, expstr in enumerate(['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']):
        exp_label[expstr] = exp
        for key in sortedImgs:
            sortedImgs[key][expstr] = []


    imgs = glob('./imgs/*[0-9].png')
    print(len(imgs))
    imgs = [os.path.basename(img) for img in imgs]
    print(imgs[1])

    for i in imgs:
        substr, expstr = i.split('_')[:2]
        sortedImgs[substr][expstr] += [','.join([i, str(exp_label[expstr]), str(id_label[substr]), i[:-4]+'_LR.png'])+'\n']

    print(sortedImgs.keys(), sortedImgs['aia'].keys())
    print(sortedImgs['aia']['anger'])

    train = []
    test = []
    for sub in sortedImgs:
        for exp in sortedImgs[sub]:
            temp = set(random.sample(sortedImgs[sub][exp], round(0.15*len(sortedImgs[sub][exp]))))
            test += temp
            train += set(sortedImgs[sub][exp]) - temp
    print(len(train), len(test), len(train)+len(test), len(test)/(len(train)+len(test)))

    with open('train_ids3.csv', 'w') as f:
        for i in train:
            f.write(i)
    with open('./test_ids3.csv', 'w') as f:
        for i in test:
            f.write(i)


def select_for_paper():
    """ 挑选用于展示的 """
    test_data = open('train_ids3.csv','r').readlines()
    test_data = [i.strip().split(',') for i in test_data]
    selected = {i[1]:0 for i in test_data}

    save_dir = os.path.join('H:\实验室\服务器实验\samples', 'FERG')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for data in test_data:
        if selected[data[1]]<1:
            print(data)
            copyfile(os.path.join('imgs', data[3]),
                     os.path.join(save_dir, data[1]+'_'+str(selected[data[1]])+'.'+data[3].split('.')[-1]))
            selected[data[1]] += 1

def gen_16_64_labels(train_file_name, test_file_name):
    train_8, train_64 = [], []
    for line in open(train_file_name, 'r'):
        HR, exp, id, LR = line.strip().split(',')
        train_8.append(','.join([HR, exp, id, LR[:-4]+'8_.png']))
        # train_64.append(','.join([HR, exp, id, LR[:-4]+'64_.png']))

    with open('./train_ids_8.csv', 'w') as f:
        f.write('\n'.join(train_8))

    # with open('./train_ids_64.csv', 'w') as f:
    #     f.write('\n'.join(train_64))

    test_8, test_64 = [], []
    for line in open(test_file_name, 'r'):
        HR, exp, id, LR = line.strip().split(',')
        test_8.append(','.join([HR, exp, id, LR[:-4] + '8_.png']))
        # test_64.append(','.join([HR, exp, id, LR[:-4] + '64_.png']))

    with open('./test_ids_8.csv', 'w') as f:
        f.write('\n'.join(test_8))

    # with open('./test_ids_64.csv', 'w') as f:
    #     f.write('\n'.join(test_64))

def gen_128_labels(train_file_name, test_file_name):
    train_128 = []
    for line in open(train_file_name, 'r'):
        HR, exp, id, LR = line.strip().split(',')
        train_128.append(','.join([HR, exp, id, HR]))
        # train_64.append(','.join([HR, exp, id, LR[:-4]+'64_.png']))

    with open('./train_ids_128.csv', 'w') as f:
        f.write('\n'.join(train_128))

    test_128 = []
    for line in open(test_file_name, 'r'):
        HR, exp, id, LR = line.strip().split(',')
        test_128.append(','.join([HR, exp, id, HR]))
        # test_64.append(','.join([HR, exp, id, LR[:-4] + '64_.png']))

    with open('./test_ids_128.csv', 'w') as f:
        f.write('\n'.join(test_128))

def gen_partial_labels(train_file_name:str, test_file_name:str):
    ids_dict = {}
    with open(train_file_name, 'r') as f:
        for line in f:
            HR, exp, id, LR = line.strip().split(',')
            if id not in ids_dict:
                ids_dict[id] = {}
            if exp not in ids_dict[id]:
                ids_dict[id][exp] = []
            ids_dict[id][exp].append(line)

    with open(train_file_name.split('.')[0]+'_partial.csv', 'w') as f:
        for id in ids_dict:
            prompt = 'train '+id+': '
            for exp in ids_dict[id]:
                prompt += "%s: %d, " % (exp, int(len(ids_dict[id][exp])*0.34))
                for line in random.sample(ids_dict[id][exp], int(len(ids_dict[id][exp])*0.34)):
                    f.write(line)
            print(prompt)

    ids_dict = {}
    with open(test_file_name, 'r') as f:
        for line in f:
            HR, exp, id, LR = line.strip().split(',')
            if id not in ids_dict:
                ids_dict[id] = {}
            if exp not in ids_dict[id]:
                ids_dict[id][exp] = []
            ids_dict[id][exp].append(line)

    with open(test_file_name.split('.')[0]+'_partial.csv', 'w') as f:
        for id in ids_dict:
            prompt = 'test '+id+': '
            for exp in ids_dict[id]:
                prompt += "%s: %d, " % (exp, int(len(ids_dict[id][exp])*0.34))
                for line in random.sample(ids_dict[id][exp], int(len(ids_dict[id][exp])*0.34)):
                    f.write(line)
            print(prompt)

if __name__ == '__main__':
    # HR_imgs, _ = getImages('./imgs')
    # blurImgs(HR_imgs)
    # getLabel()
    # gen_16_64_labels('./train_ids3.csv', './test_ids3.csv')
    # select_for_paper()
    # gen_128_labels('train_ids_8.csv', 'test_ids_8.csv')
    gen_partial_labels('train_ids_32.csv', 'test_ids_32.csv')