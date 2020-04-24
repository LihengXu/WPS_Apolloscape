import numpy as np
rank_file_path = "./val_search/"
from re_ranking.re_ranking_feature import re_ranking
from sklearn.preprocessing import normalize

def k_near(scene_file='./data/merge_scene1.txt', batch_num=5, action=[0]*12):
    action = np.array(action, dtype=np.float)
    action_scope = np.array([4, 2, 5, 36, 8, 8, 6, 4, 25, 12, 6, 7])
    rank_num = batch_num
    rank_buffer = []
    with open(scene_file, 'r') as f:
        for x in f:
            if x[-1] == '\n':
                scene_list = x[:-1].split(' ')
            else:
                scene_list = x.split(' ')

            index = int(scene_list[0])
            encode = np.array([int(scene_list[2]), int(scene_list[3]), int(scene_list[4]), int(scene_list[5]),
                               int(scene_list[6]), int(scene_list[7]), int(scene_list[8]), int(scene_list[9]),
                               int(scene_list[10]), int(scene_list[11]), int(scene_list[12]), int(scene_list[13])])
            # 以下计算欧氏距离
            diff = encode - action
            nm_diff = diff / action_scope
            sqdiff = nm_diff ** 2
            sqdistance = sqdiff.sum(axis=0)
            distance = sqdistance ** 0.5

            if len(rank_buffer) < rank_num:
                rank_buffer.append([index, distance])
            else:
                for num, y in enumerate(rank_buffer):
                    if y[-1] > distance:
                        rank_buffer[num] = [index, distance]
                        break
        rank_index = np.array(rank_buffer, dtype=int)[:, 0].tolist()
        # print(rank_buffer)
        # print(rank_index)
        # print(len(rank_buffer))
        return rank_buffer, rank_index


def k_near2(scene_file='./data/merge_scene2.txt', batch_num=5, action=[0]*17):
    action = np.array(action, dtype=np.float)
    action_scope = np.array([4, 2, 5, 1, 1, 1, 1, 1, 1, 36, 6, 25, 12, 6, 7, 18, 57])
    rank_num = batch_num
    rank_buffer = []
    with open(scene_file, 'r') as f:
        for x in f:
            if x[-1] == '\n':
                scene_list = x[:-1].split(' ')
            else:
                scene_list = x.split(' ')

            index = int(scene_list[0])
            encode = np.array([int(scene_list[2]), int(scene_list[3]), int(scene_list[4]), int(scene_list[5]),
                               int(scene_list[6]), int(scene_list[7]), int(scene_list[8]), int(scene_list[9]),
                               int(scene_list[10]), int(scene_list[11]), int(scene_list[12]), int(scene_list[13]),
                               int(scene_list[14]), int(scene_list[15]), int(scene_list[16]), int(scene_list[17]),
                               int(scene_list[18])])
            # 以下计算欧氏距离
            diff = encode - action
            nm_diff = diff / action_scope
            sqdiff = nm_diff ** 2
            sqdistance = sqdiff.sum(axis=0)
            distance = sqdistance ** 0.5

            if len(rank_buffer) < rank_num:
                rank_buffer.append([index, distance])
            else:
                for num, y in enumerate(rank_buffer):
                    if y[-1] > distance:
                        rank_buffer[num] = [index, distance]
                        break
        rank_index = np.array(rank_buffer, dtype=int)[:, 0].tolist()
        # print(rank_buffer)
        # print(rank_index)
        # print(len(rank_buffer))
        return rank_buffer, rank_index


def re_rank(action=[0]*29, lamb=0.3, nums=10):
    """
    融合instance label和场景特征的距离进行rank
    :param action: scenario list
    :param lamb: rank_param
    :param nums: rank_num
    :return: rank_index
    """
    Fea = np.array(action)
    probFea = np.expand_dims(Fea, axis=0)
    probFea1 = probFea[:, 0:4]
    probFea2 = normalize(probFea[:, 4:], axis=1, norm='l2')

    galFea1 = np.load("./data/Fea/ins_galFea.npy")
    final_dist1 = re_ranking(probFea1, galFea1, 20, 6, 0.3)
    galFea2 = np.load("./data/Fea/pca_galFea.npy")
    final_dist2 = re_ranking(probFea2, galFea2, 20, 6, 0.3)

    try:
        assert final_dist1.shape == final_dist2.shape
    except AssertionError as e:
        print('两个距离矩阵shape不匹配')

    final_dist = (1-lamb)*final_dist1 + lamb*final_dist2
    final_rank = np.argsort(final_dist)

    return final_rank[:, :nums].flatten()


def action_txt(save_path=rank_file_path, scene_file='./data/merge_ann_bbox.txt', action=[0]*12, rank_index=None):
    assert (rank_index is not None)
    action_name = "_".join([str(i) for i in action]) + '.txt'
    with open(save_path + action_name, 'w') as m:
        with open(scene_file, 'r') as f:
            index = 0
            for x in f:
                if int(x.split(' ')[0]) in rank_index:
                    # print(x)
                    if index < len(rank_index)-1:
                        m.write(str(index) + ' ' + x.split(' ', 1)[-1][:-1] + '\n')
                    else:
                        m.write(str(index) + ' ' + x.split(' ', 1)[-1][:-1])
                    index += 1
    return save_path + action_name








