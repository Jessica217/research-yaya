import os
import random


def main():
    random.seed(0)  # 设置随机种子，保证随机结果可复现

    files_path = "./VOCdevkit/VOC2012/Annotations"
    assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)
    train_rate = 0.9
    test_rate = 0.05
    val_rate = 0.05

    files_name = sorted([file.split(".")[0] for file in os.listdir(files_path)])
    files_num = len(files_name)
    train_index = random.sample(range(0, files_num), k=int(files_num*train_rate))
    test_index = random.sample(range(0, files_num), k=int(files_num*test_rate))
    val_index = random.sample(range(0, files_num), k=int(files_num*val_rate))

    train_files = []
    test_files = []
    val_files = []
    for index, file_name in enumerate(files_name):
        if index in val_index:
            val_files.append(file_name)
        elif index in train_index:
            train_files.append(file_name)
        elif index in test_index:
            test_files.append(file_name)

    try:
        train_f = open("train.txt", "x")
        eval_f = open("val.txt", "x")
        test_f = open("test.txt","x")
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))
        test_f.write("\n".join(test_files))

    except FileExistsError as e:
        print(e)
        exit(1)


if __name__ == '__main__':
    main()
