import os


def load_file(root, rel_path = "", img_list=[], rel_img_list = []):
    if os.path.isfile(root):
        img_list.append(root)
        rel_img_list.append(rel_path)
    elif os.path.isdir(root):
        for path_i in os.listdir(root):
            sub_root = os.path.join(root, path_i)
            sub_rel_path = os.path.join(rel_path, path_i)
            img_list, rel_img_list = load_file(sub_root, sub_rel_path, img_list, rel_img_list)
    return img_list, rel_img_list


def main():
    img_root = "/home/zkyang/Workspace/task/Pytorch_task/Pytorch_train/data/car/Data"
    img_list, rel_img_list = load_file(img_root, rel_path = "", img_list=[], rel_img_list = [])
    for i in img_list:
        print i
        # new_name = ''.join(i.split(" "))
        # os.rename(i,new_name)
        filepath = os.path.dirname(i)
        print filepath



if __name__ == '__main__':
    main()