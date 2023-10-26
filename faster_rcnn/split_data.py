import random
import os

def main():
    random.seed(0)  # 设置随机种子，保证随机结果可复现

    files_path = './VOCdevkit/VOC2012/Annotations'
    assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)

    total_files = sorted([file.split(".")[0] for file in os.listdir(files_path)])
    total_num = len(total_files) # 获取数据集总长度

    train_rate = 0.9
    val_rate = 0.05
    test_rate = 0.05

    train_num = int(total_num * train_rate)
    val_num = int(total_num * val_rate)
    test_num = int(total_num * test_rate)

    random.shuffle(total_files)  # 随机打乱数据集

    train_files = total_files[:train_num]
    val_files = total_files[train_num:train_num + val_num]
    test_files = total_files[train_num + val_num:]

    try:
        with open("VOCdevkit/VOC2012/ImageSets/Main/train.txt", "w") as train_f:
            train_f.write("\n".join(train_files))
        with open("VOCdevkit/VOC2012/ImageSets/Main/val.txt", "w") as eval_f:
            eval_f.write("\n".join(val_files))
        with open("VOCdevkit/VOC2012/ImageSets/Main/test.txt", "w") as test_f:
            test_f.write("\n".join(test_files))
    except FileExistsError as e:
        print(e)
        exit(1)

if __name__ == "__main__":
    main()
