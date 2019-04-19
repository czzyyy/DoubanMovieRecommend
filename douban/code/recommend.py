# -*-coding:utf-8-*-
import numpy as np
import os
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from RecommendationSys.douban import models
import heapq

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
# 选择样式
# plt.style.use('bmh')


class DBRecommender(object):
    def __init__(self, clean_user_pickle_path, clean_movie_pickle_path, model_save_dir, clean_movie_user_csv_dir):
        self.model_save_dir = model_save_dir
        self.clean_user_pickle_path = clean_user_pickle_path
        self.clean_movie_pickle_path = clean_movie_pickle_path
        self.clean_movie_user_csv_dir = clean_movie_user_csv_dir
        self._load_datas()

    def _load_datas(self):
        try:
            with open(self.clean_user_pickle_path, 'rb') as f:
                self.clean_user_dict = pickle.load(f)
        except Exception:
            print('read user pickle error')
        try:
            with open(self.clean_movie_pickle_path, 'rb') as f:
                clean_movie_all_dict = pickle.load(f)
                self.clean_movie_dict = clean_movie_all_dict['clean_movie_dict']
            print('clean_user_dict', len(self.clean_user_dict))
            print('clean_movie_dict', len(self.clean_movie_dict))
        except Exception:
            print('read movie pickle error')
        try:
            with open(self.model_save_dir + 'movie/train_movie_embedding.pickle', 'rb') as f:
                self.train_movie_embedding = pickle.load(f)
            print('train_movie_embedding', len(self.train_movie_embedding))
            print('read train_movie_embedding.pickle end')
        except Exception:
            print('read train_movie_embedding.pickle error')
        try:
            with open(self.model_save_dir + 'movie_user/train_user_embedding.pickle', 'rb') as f:
                self.train_user_embedding = pickle.load(f)
            print('train_user_embedding', len(self.train_user_embedding))
            print('read train_user_embedding.pickle end')
        except Exception:
            print('read train_user_embedding.pickle error')
        self.reverse_movie_dict = dict(zip(self.clean_movie_dict.values(), self.clean_movie_dict.keys()))  # index, title
        self.reverse_user_dict = dict(zip(self.clean_user_dict.values(), self.clean_user_dict.keys()))  # index, username

    # 对用户 和 电影的 嵌入进行降维并可视化
    def visual_movie_embedding(self, point_num=400):
        # 只对前 point_num 个电影进行降维和可视化
        norm_all = np.sqrt(np.sum(np.square(self.train_movie_embedding), axis=1, keepdims=True))
        normalized_embedding_all = self.train_movie_embedding / norm_all  # [all, 64]
        if point_num > len(self.clean_movie_dict):
            point_num = len(self.clean_movie_dict)
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        two_dim_embedding = tsne.fit_transform(normalized_embedding_all[0:point_num])  # [point_num, 2]
        # 聚类？？
        titles = [self.reverse_movie_dict[i] for i in range(point_num)]
        plt.figure(figsize=(30, 30))  # in inches
        for i, title in enumerate(titles):
            x, y = two_dim_embedding[i, :]
            plt.scatter(x, y)
            plt.annotate(title, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
        plt.show()

    def visual_user_embedding(self, point_num=100):
        # 只对前 point_num 个用户进行降维和可视化
        print(self.train_user_embedding[0][0:10])
        norm_all = np.sqrt(np.sum(np.square(self.train_user_embedding), axis=1, keepdims=True))
        normalized_embedding_all = self.train_user_embedding / norm_all  # [all, 64]
        if point_num > len(self.clean_user_dict):
            point_num = len(self.clean_user_dict)
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        two_dim_embedding = tsne.fit_transform(normalized_embedding_all[0:point_num])  # [point_num, 2]

        user_names = [self.reverse_user_dict[i] for i in range(point_num)]
        plt.figure(figsize=(100, 100))  # in inches
        for i, user_name in enumerate(user_names):
            x, y = two_dim_embedding[i, :]
            plt.scatter(x, y)
            plt.annotate(user_name, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
        plt.show()

    def movie_similarity(self, title1, title2=None, top_n=10):
        if title2 is not None:
            # 计算两个相似度
            index1 = self.clean_movie_dict.get(title1)
            embedding1 = self.train_movie_embedding[int(index1)]
            index2 = self.clean_movie_dict.get(title2)
            embedding2 = self.train_movie_embedding[int(index2)]
            norm1 = np.sqrt(np.sum(np.square(embedding1)))
            normalized_embedding1 = embedding1 / norm1  # [64, 1]
            norm2 = np.sqrt(np.sum(np.square(embedding2)))
            normalized_embedding2 = embedding2 / norm2  # [64, 1]
            similarity = np.matmul(np.transpose(normalized_embedding1), normalized_embedding2)
            return similarity
        elif title2 is None:
            # 计算title1 最相似的5个
            index1 = self.clean_movie_dict.get(title1)
            embedding1 = self.train_movie_embedding[int(index1)]
            norm1 = np.sqrt(np.sum(np.square(embedding1)))
            normalized_embedding1 = embedding1 / norm1  # [64, 1]
            norm_all = np.sqrt(np.sum(np.square(self.train_movie_embedding), axis=1, keepdims=True))
            normalized_embedding_all = self.train_movie_embedding / norm_all  # [all, 64]
            similarity = list(np.matmul(normalized_embedding_all, np.transpose(normalized_embedding1)).data)  # [all, 1]
            # 获得前top_n个的索引
            top_n_index = list(map(similarity.index, heapq.nlargest(top_n + 1, similarity)))
            re_titles = list()
            for i in top_n_index:
                re_titles.append(str(self.reverse_movie_dict.get(i)))
            re_similarity = heapq.nlargest(top_n + 1, similarity)
            return re_similarity, re_titles

    def user_similarity(self, user_name1, user_name2=None, top_n=10):
        if user_name2 is not None:
            # 计算两个相似度
            index1 = self.clean_user_dict.get(user_name1)
            embedding1 = self.train_user_embedding[int(index1)]
            print('embedding1.shape', embedding1.shape)
            index2 = self.clean_user_dict.get(user_name2)
            embedding2 = self.train_user_embedding[int(index2)]
            norm1 = np.sqrt(np.sum(np.square(embedding1)))
            normalized_embedding1 = embedding1 / norm1  # [64, 1]
            norm2 = np.sqrt(np.sum(np.square(embedding2)))
            normalized_embedding2 = embedding2 / norm2  # [64, 1]
            similarity = np.matmul(np.transpose(normalized_embedding1), normalized_embedding2)
            return similarity
        elif user_name2 is None:
            # 计算 user_name1 最相似的5个
            index1 = self.clean_user_dict.get(user_name1)
            embedding1 = self.train_user_embedding[int(index1)]
            norm1 = np.sqrt(np.sum(np.square(embedding1)))
            normalized_embedding1 = embedding1 / norm1  # [64, 1]
            norm_all = np.sqrt(np.sum(np.square(self.train_user_embedding), axis=1, keepdims=True))
            normalized_embedding_all = self.train_user_embedding / norm_all  # [all, 64]
            similarity = list(np.matmul(normalized_embedding_all, np.transpose(normalized_embedding1)).data) # [all, 1]
            # 获得前top_n个的索引
            top_n_index = list(map(similarity.index, heapq.nlargest(top_n + 1, similarity)))
            re_user_names = list()
            for i in top_n_index:
                re_user_names.append(str(self.reverse_user_dict.get(i)))
            re_similarity = heapq.nlargest(top_n + 1, similarity)
            return re_similarity, re_user_names

    def user_recommend(self, user_name, top_n=10, policy=0):
        if policy == 0:
            # 直接根据用户最喜欢的 1 个电影 找出5个 movie_similarity最 近的其他电影 并推荐
            the_clean_movie_user_table_file = os.path.join(self.clean_movie_user_csv_dir, user_name + '.csv')
            with open(the_clean_movie_user_table_file, 'rb') as f:
                the_clean_movie_user_df = pd.read_csv(f, encoding="utf-8")
            max_titles = the_clean_movie_user_df.sort_values(by='ratings')['titles'].values
            s = list()
            candidates = list()
            count = 0
            for title in max_titles:
                if self.clean_movie_dict.get(title) is not None and count < top_n // 2:
                    re_s, re_t = self.movie_similarity(title1=title, top_n=2)
                    s = s + re_s[1:]
                    candidates = candidates + re_t[1:]
                    count = count + 1
            return candidates
        elif policy == 1:
            candidates = list()  # 保存候选电影的名字
            u_similarity, u_user_names = self.user_similarity(user_name1=user_name, top_n=top_n)
            for name in u_user_names[1:]:
                tmp_clean_movie_user_table_file = os.path.join(self.clean_movie_user_csv_dir, name + '.csv')
                with open(tmp_clean_movie_user_table_file, 'rb') as f:
                    tmp_clean_movie_user_df = pd.read_csv(f, encoding="utf-8")
                    # 找到评分最高的第一个电影名字
                    tmp_max_titles = tmp_clean_movie_user_df.sort_values(by='ratings')['titles'].values
                    tmp_max_title = None
                    for title in tmp_max_titles:
                        if self.clean_movie_dict.get(title) is not None:
                            tmp_max_title = title
                    candidates.append(tmp_max_title)
            return candidates
        elif policy == 2:
            the_clean_movie_user_table_file = os.path.join(self.clean_movie_user_csv_dir, user_name + '.csv')
            with open(the_clean_movie_user_table_file, 'rb') as f:
                the_clean_movie_user_df = pd.read_csv(f, encoding="utf-8")
            max_titles = the_clean_movie_user_df.sort_values(by='ratings')['titles'].values
            candidates = list()
            count = 0
            for title in max_titles:
                if self.clean_movie_dict.get(title) is not None and count < top_n // 2:
                    re_s, re_t = self.movie_similarity(title1=title, top_n=1)
                    candidates = candidates + re_t[1:]
                    count = count + 1
            u_similarity, u_user_names = self.user_similarity(user_name1=user_name, top_n=top_n // 2)
            for name in u_user_names[1:]:
                tmp_clean_movie_user_table_file = os.path.join(self.clean_movie_user_csv_dir, name + '.csv')
                with open(tmp_clean_movie_user_table_file, 'rb') as f:
                    tmp_clean_movie_user_df = pd.read_csv(f, encoding="utf-8")
                    # 找到评分最高的第一个电影名字
                    tmp_max_titles = tmp_clean_movie_user_df.sort_values(by='ratings')['titles'].values
                    tmp_max_title = None
                    for title in tmp_max_titles:
                        if self.clean_movie_dict.get(title) is not None:
                            tmp_max_title = title
                    candidates.append(tmp_max_title)
            return candidates
            # # 载入模型 进行评分
            # movie_data_path = '/home/ziyangcheng/python_code/SPECIFICexercise/RecommendationSys/douban/my_data/train_data/train_movie_data.pickle'
            # user_data_path = '/home/ziyangcheng/python_code/SPECIFICexercise/RecommendationSys/douban/my_data/train_data/train_user_data.pickle'
            # user_model = models.MovieUserRecommendModel(batch_size=64, learning_rate=0.001, embedding_dim=24,
            #                                             training_epochs=100, save_dir=self.model_save_dir,
            #                                             user_data_path=None, movie_data_path=None,
            #                                             movie_embedding_data_path=None)
            # with open(movie_data_path, 'rb') as f:
            #     movie_data = pickle.load(f)
            # movie_data = movie_data['movie_train_x']
            #
            # with open(user_data_path, 'rb') as f:
            #     user_data = pickle.load(f)
            # columns = user_data['columns']
            # user_model.user_num = user_data['user_num']
            # user_data = user_data['user_train_x']
            # the_user_index = int(self.clean_user_dict.get(user_name))
            # the_user_data = user_data[the_user_index][columns]
            # result_ratings = list()
            # for candidate in candidates:
            #     # 每一个都是电影名字 根据名字先找到电影的索引 然后找到
            #     the_movie_index = self.clean_movie_dict.get(candidate)
            #     the_movie_data = movie_data[the_movie_index]
            #     the_movie_embedding_data = self.train_movie_embedding[the_movie_index]
            #
            #     result_ratings.append(user_model.predict_score(user_predict=the_user_data,
            #                                                    movie_predict=the_movie_data,
            #                                                    movie_e_predict=the_movie_embedding_data,
            #                                                    user_predict_index=[the_user_index]))
            # # 找到最大的5个
            # top_n_index = list(map(result_ratings.index, heapq.nlargest(top_n, result_ratings)))
            # return [result_ratings[i] for i in top_n_index], [candidates[i] for i in top_n_index]


if __name__ == '__main__':
    user_path = './douban/my_data/clean_user.pickle'
    movie_path = './douban/my_data/clean_movie.pickle'
    save_dir = './douban/my_data/save/'
    movie_user_dir = './douban/my_data/clean_mv_us_tables'
    dbr = DBRecommender(clean_user_pickle_path=user_path, clean_movie_pickle_path=movie_path, model_save_dir=save_dir,
                        clean_movie_user_csv_dir=movie_user_dir)
    dbr.visual_movie_embedding(point_num=600)
    re_s, re_t = dbr.movie_similarity(title1='致命魔术')
    for a, b in zip(re_s[1:], re_t[1:]):
        print(a)
        print(b)
    dbr.visual_user_embedding(point_num=500)
    re_s, re_t = dbr.user_similarity(user_name1='SY不是游泳的')
    for a, b in zip(re_s[1:], re_t[1:]):
        print(a)
        print(b)
    # re_t = dbr.user_recommend(user_name='haha', top_n=10, policy=0)
    # for a in re_t:
    #     print(a)
    #
    # print('-------------')
    #
    # re_t = dbr.user_recommend(user_name='haha', top_n=10, policy=1)
    # for a in re_t:
    #     print(a)
    #
    # print('-------------')
    #
    # re_t = dbr.user_recommend(user_name='haha', top_n=10, policy=2)
    # for a in re_t:
    #     print(a)