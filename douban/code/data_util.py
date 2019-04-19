# -*-coding:utf-8-*-
import numpy as np
import os
import pandas as pd
import pickle


# 所有电影顺序都按照清洗之后的电影dict和反dict  用户顺序都按照清洗之后的用户dict和反dict
class DataUtil(object):
    def __init__(self):
        self.average_rating_language_path = './douban/my_data/average_rating_language.csv'
        self.average_rating_area_path = './douban/my_data/average_rating_area.csv'
        self.average_rating_movie_type_path = './douban/my_data/average_rating_movie_type.csv'
        self.average_rating_runtime_path = './douban/my_data/average_rating_runtime.csv'
        self.pickle_dir = './douban/my_data/'

        self.movie_csv_path = './douban/my_data/mv_table.csv'
        self.movie_user_csv_dir = './douban/my_data/mv_us_tables/'

        self.train_movie_data = dict()
        self.train_user_data = dict()
        self.train_data_dir = './douban/my_data/train_data/'
        self.clean_movie_csv_path = './douban/my_data/clean_mv_table.csv'
        self.clean_user_csv_path = './douban/my_data/clean_us_table.csv'
        self.clean_movie_user_csv_dir = './douban/my_data/clean_mv_us_tables/'

        self.clean_user_dict = None
        self.clean_movie_dict = None
        self.clean_director_dict = None
        self.clean_screenwriter_dict = None
        self.clean_actor_dict = None
        self.clean_movie_type_dict = None
        self.clean_area_dict = None
        self.clean_language_dict = None

    def get_movie_train_data(self, normalization=True):
        reverse_movie_dict = dict(zip(self.clean_movie_dict.values(), self.clean_movie_dict.keys()))  # index, title

        if not os.path.exists(self.clean_movie_csv_path):
            print(self.clean_movie_csv_path + ' not exist')
            return False
        with open(self.clean_movie_csv_path, 'rb') as f:
            clean_movie_df = pd.read_csv(f, encoding="utf-8")

        if not os.path.exists(self.average_rating_movie_type_path):
            print(self.average_rating_movie_type_path + ' not exist')
            return False
        with open(self.average_rating_movie_type_path, 'rb') as f:
            avg_movie_type_df = pd.read_csv(f, encoding="utf-8")

        if not os.path.exists(self.average_rating_area_path):
            print(self.average_rating_area_path + ' not exist')
            return False
        with open(self.average_rating_area_path, 'rb') as f:
            avg_area_df = pd.read_csv(f, encoding="utf-8")

        if not os.path.exists(self.average_rating_language_path):
            print(self.average_rating_language_path + ' not exist')
            return False
        with open(self.average_rating_language_path, 'rb') as f:
            avg_language_df = pd.read_csv(f, encoding="utf-8")

        if not os.path.exists(self.average_rating_runtime_path):
            print(self.average_rating_runtime_path + ' not exist')
            return False
        with open(self.average_rating_runtime_path, 'rb') as f:
            avg_runtime_df = pd.read_csv(f, encoding="utf-8")

        for m_type in avg_movie_type_df['movie_type'].values:
            indexes = clean_movie_df[(pd.isna(clean_movie_df['runtime'])) & (clean_movie_df['movie_type'] == m_type)].index
            clean_movie_df.loc[indexes, 'runtime'] = float(avg_movie_type_df[
                avg_movie_type_df['movie_type'] == m_type]['runtime'].values[0])

        total_movie_num = len(reverse_movie_dict)
        total_movie_type_num = len(self.clean_movie_type_dict)
        total_area_num = len(self.clean_area_dict)
        total_language_num = len(self.clean_language_dict)

        movie_train_x_list = list()
        movie_train_y_list = list()
        for index in range(total_movie_num):
            try:
                movie_title = reverse_movie_dict.get(index)
                the_movie_df = clean_movie_df[clean_movie_df['title'] == movie_title]
                vote_num = [float(the_movie_df['vote_num'].values[0])]
                movie_type = [0.0] * (total_movie_type_num + 1)
                movie_type[self.clean_movie_type_dict.get(str(the_movie_df['movie_type'].values[0]))] = 1.0
                movie_type[-1] = float(avg_movie_type_df[avg_movie_type_df['movie_type']
                                                         == the_movie_df['movie_type'].values[0]]['avg_star'].values[0])
                area = [0.0] * (total_area_num + 1)
                area[self.clean_area_dict.get(str(the_movie_df['area'].values[0]))] = 1.0
                area[-1] = float(avg_area_df[avg_area_df['area'] == the_movie_df['area'].values[0]]['avg_star'].values[0])
                language = [0.0] * (total_language_num + 1)
                language[self.clean_language_dict.get(str(the_movie_df['language'].values[0]))] = 1.0
                language[-1] = float(avg_language_df[avg_language_df['language']
                                                     == the_movie_df['language'].values[0]]['avg_star'].values[0])
                runtime = [0.0, 0.0]
                runtime[0] = float(the_movie_df['runtime'].values[0])
                runtime[1] = float(avg_runtime_df[(avg_runtime_df['runtime_up'] > runtime[0]) &
                                                  (avg_runtime_df['runtime_low'] <= runtime[0])]['avg_star'].values[0])
                date = [float(str(the_movie_df['date'].values[0])[0:4])]
                movie_train_x_list.append(np.array(np.concatenate([vote_num, movie_type, area, language, runtime, date]), dtype=np.float32))
                movie_train_y_list.append(np.array([the_movie_df['star'].values[0]], dtype=np.float32))
                if index % 1000 == 0:
                    print('get movie data', index)
                    print(movie_train_x_list[-1].shape)
                    print(movie_train_y_list[-1])
            except Exception:
                print('index ' + str(index) + ' movie ' + movie_title + ' get data failed')
        print('save train_movie_data.pickle start')
        try:
            movie_train_data_path = self.train_data_dir + 'train_movie_data.pickle'
            self.train_movie_data['movie_train_x'] = np.array(movie_train_x_list, dtype=np.float32)
            self.train_movie_data['movie_train_y'] = np.array(movie_train_y_list, dtype=np.float32)
            if normalization:
                vote_num_index = 0
                movie_type_index = total_movie_type_num + 1
                area_index = movie_type_index + total_area_num + 1
                language_index = area_index + total_language_num + 1
                runtime_index0 = language_index + 1
                runtime_index1 = runtime_index0 + 1
                date_index = runtime_index1 + 1
                columns = [vote_num_index, movie_type_index, area_index, language_index, runtime_index0, runtime_index1, date_index]
                means = np.mean(self.train_movie_data['movie_train_x'][:, columns], axis=0)
                stds = np.std(self.train_movie_data['movie_train_x'][:, columns], axis=0)
                self.train_movie_data['movie_train_x'][:, columns] = (self.train_movie_data['movie_train_x'][:, columns] - means) / stds
                self.train_movie_data['movie_train_y'] = self.train_movie_data['movie_train_y'] / 10.0  # 0.0~1.0
                self.train_movie_data['means'] = means
                self.train_movie_data['stds'] = stds
                self.train_movie_data['columns'] = columns
            print('movie_train_x shape', self.train_movie_data['movie_train_x'].shape)
            print('movie_train_y shape', self.train_movie_data['movie_train_y'].shape)
            if not os.path.exists(movie_train_data_path):
                open(movie_train_data_path, 'w')
                print('train_movie_data.pickle not exists, create it')
            with open(movie_train_data_path, 'wb') as f:
                pickle.dump(self.train_movie_data, f)
            print('save train_movie_data.pickle end')
        except Exception:
            print('save movie_train_data.pickle error')

    def get_user_train_data(self, normalization=True):
        reverse_user_dict = dict(zip(self.clean_user_dict.values(), self.clean_user_dict.keys()))  # index, username
        reverse_movie_type_dict = dict(zip(self.clean_movie_type_dict.values(), self.clean_movie_type_dict.keys()))  # index, movie_type

        # 读取 用户的 clean_user_csv ,并利用这个生成用户的特征，以及 clean_mv_us_tables，用这个找到对应的电影索引和对该电影的评分
        # user_name, avg_ratings, 类型, 类型_avg_ratings
        if not os.path.exists(self.clean_user_csv_path):
            print(self.clean_user_csv_path + ' not exist')
            return False
        with open(self.clean_user_csv_path, 'rb') as f:
            clean_user_df = pd.read_csv(f, encoding="utf-8")

        total_user_num = len(reverse_user_dict)
        total_movie_type_num = len(reverse_movie_type_dict)
        user_train_x_list = list()
        user_train_y_list = list()
        tmp_user_dict = dict()
        for index in range(total_user_num):
            try:
                # 首先根据index 从reverse_user_dict 获得用户的名字 然后根据名字获得mv_us_table和获得特定的clean_user_df
                user_name = reverse_user_dict.get(index)
                the_user_df = clean_user_df[clean_user_df['user_name'] == user_name]
                # 获得该用户的平均分 是list
                avg_rating = [float(the_user_df['avg_ratings'].values[0])]
                # 获得不同类型的电影个数以及平均分, 注意这里的类型顺序最好和电影特征中的类型顺序一致， 也就是按照类型字典中的值顺序从小到大
                type_count_ratings = list()
                for m_t_index in range(total_movie_type_num):
                    m_type = reverse_movie_type_dict.get(m_t_index)
                    # 获得该类型的电影个数
                    type_count_ratings.append(float(the_user_df[str(m_type)].values[0]))
                    # 获得该类型的电影平均分
                    type_count_ratings.append(float(the_user_df[str(m_type) + '_avg_ratings'].values[0]))
                # 打开该用户的mv_us_table 找到他的电影名字和打分 并利用reverse_movie_dict 得到电影的索引值保存即可
                the_clean_movie_user_table_file = os.path.join(self.clean_movie_user_csv_dir, user_name + '.csv')
                with open(the_clean_movie_user_table_file, 'rb') as f:
                    the_clean_movie_user_df = pd.read_csv(f, encoding="utf-8")
                # user_name, titles, ratings
                for m_u_index in range(len(the_clean_movie_user_df)):
                    m_title = str(the_clean_movie_user_df.loc[m_u_index, 'titles'])
                    m_index = self.clean_movie_dict.get(m_title)
                    if m_index is None:
                        # 如果该电影找不到　或者没有该电影的数据 就不算到训练集中
                        # print('movie ' + m_title + ' not in movie data!, forget it')
                        continue
                    movie_user_index = [float(m_index), float(index)]
                    # 注意，user_train_x_list 每一个的倒数第二位是电影 特征的索引， 其顺序和movie dict的值顺序一致，最后一位是用户索引
                    # 和用户的dict顺序一致 目的是在网络训练的时候 得到相应的嵌入特征
                    user_train_x_list.append(np.array(np.concatenate([avg_rating, type_count_ratings, movie_user_index]), dtype=np.float32))
                    user_train_y_list.append(np.array([the_clean_movie_user_df.loc[m_u_index, 'ratings']], dtype=np.float32))
                    if tmp_user_dict.get(user_name) is None:
                        tmp_user_dict[user_name] = len(tmp_user_dict)
                if index % 50 == 0:
                    print('get user data', user_name)
                    print(user_train_x_list[-1].shape)
                    print(user_train_y_list[-1])
            except Exception:
                print('index ' + str(index) + 'user ' + user_name + 'get data' + m_title + ' failed')
        print('save train_user_data.pickle start')
        try:
            user_train_data_path = self.train_data_dir + 'train_user_data.pickle'
            self.train_user_data['user_train_x'] = np.array(user_train_x_list, dtype=np.float32)
            self.train_user_data['user_train_y'] = np.array(user_train_y_list, dtype=np.float32)
            self.train_user_data['user_num'] = len(tmp_user_dict)
            print('user_num', len(tmp_user_dict))
            if normalization:
                # 标准化
                feature_dim = len(user_train_x_list[0])
                # 除了最后2列 其余都标准化 因为最后2列是电影和用户的索引
                columns = list(range(0, feature_dim - 2))

                print(self.train_user_data['user_train_x'][0, columns])

                means = np.mean(self.train_user_data['user_train_x'][:, columns], axis=0)
                stds = np.std(self.train_user_data['user_train_x'][:, columns], axis=0)
                self.train_user_data['user_train_x'][:, columns] = (self.train_user_data['user_train_x'][:,
                                                                    columns] - means) / stds
                self.train_user_data['user_train_y'] = self.train_user_data['user_train_y'] / 10.0  # 0.0~1.0
                self.train_user_data['means'] = means
                self.train_user_data['stds'] = stds
                self.train_user_data['columns'] = columns
            print('user_train_x shape', self.train_user_data['user_train_x'].shape)
            print('user_train_y shape', self.train_user_data['user_train_y'].shape)
            if not os.path.exists(user_train_data_path):
                open(user_train_data_path, 'w')
                print('train_user_data.pickle not exists, create it')
            with open(user_train_data_path, 'wb') as f:
                pickle.dump(self.train_user_data, f)
            print('save train_user_data.pickle end')
        except Exception:
            print('save train_user_data.pickle error')


    def read_clean_pickle(self):
        clean_user_pickle_path = self.pickle_dir + 'clean_user.pickle'
        clean_movie_pickle_path = self.pickle_dir + 'clean_movie.pickle'
        try:
            with open(clean_user_pickle_path, 'rb') as f:
                self.clean_user_dict = pickle.load(f)
        except Exception:
            print('read user pickle error')
        try:
            with open(clean_movie_pickle_path, 'rb') as f:
                clean_movie_all_dict = pickle.load(f)
                self.clean_movie_dict = clean_movie_all_dict['clean_movie_dict']
                self.clean_director_dict = clean_movie_all_dict['clean_director_dict']
                self.clean_screenwriter_dict = clean_movie_all_dict['clean_screenwriter_dict']
                self.clean_actor_dict = clean_movie_all_dict['clean_actor_dict']
                self.clean_movie_type_dict = clean_movie_all_dict['clean_movie_type_dict']
                self.clean_area_dict = clean_movie_all_dict['clean_area_dict']
                self.clean_language_dict = clean_movie_all_dict['clean_language_dict']
            print('clean_user_dict', len(self.clean_user_dict))
            print('clean_movie_dict', len(self.clean_movie_dict))
            print('clean_director_dict', len(self.clean_director_dict))
            print('clean_screenwriter_dict', len(self.clean_screenwriter_dict))
            print('clean_actor_dict', len(self.clean_actor_dict))
            print('clean_movie_type_dict', len(self.clean_movie_type_dict))
            print('clean_area_dict', len(self.clean_area_dict))
            print('clean_language_dict', len(self.clean_language_dict))
        except Exception:
            print('read movie pickle error')

    def save_clean_movie_pickle(self):
        clean_movie_pickle_path = self.pickle_dir + 'clean_movie.pickle'

        if not os.path.exists(self.clean_movie_csv_path):
            print(self.clean_movie_csv_path + ' not exist')
            return False
        with open(self.clean_movie_csv_path, 'rb') as f:
            clean_movie_df = pd.read_csv(f, encoding="utf-8")

        clean_movie_dict = dict()
        clean_director_dict = dict()
        clean_screenwriter_dict = dict()
        clean_actor_dict = dict()
        clean_movie_type_dict = dict()
        clean_area_dict = dict()
        clean_language_dict = dict()
        length = clean_movie_df['title'].count()
        for i in range(length):
            if clean_movie_dict.get(str(clean_movie_df['title'].values[i])) is None:
                clean_movie_dict[str(clean_movie_df['title'].values[i])] = len(clean_movie_dict)
            if clean_director_dict.get(str(clean_movie_df['director'].values[i])) is None:
                clean_director_dict[str(clean_movie_df['director'].values[i])] = len(clean_director_dict)
            if clean_screenwriter_dict.get(str(clean_movie_df['screenwriter'].values[i])) is None:
                clean_screenwriter_dict[str(clean_movie_df['screenwriter'].values[i])] = len(clean_screenwriter_dict)
            if clean_actor_dict.get(str(clean_movie_df['actor1'].values[i])) is None:
                clean_actor_dict[str(clean_movie_df['actor1'].values[i])] = len(clean_actor_dict)
            if clean_actor_dict.get(str(clean_movie_df['actor2'].values[i])) is None:
                clean_actor_dict[str(clean_movie_df['actor2'].values[i])] = len(clean_actor_dict)
            if clean_movie_type_dict.get(str(clean_movie_df['movie_type'].values[i])) is None:
                clean_movie_type_dict[str(clean_movie_df['movie_type'].values[i])] = len(clean_movie_type_dict)
            if clean_area_dict.get(str(clean_movie_df['area'].values[i])) is None:
                clean_area_dict[str(clean_movie_df['area'].values[i])] = len(clean_area_dict)
            if clean_language_dict.get(str(clean_movie_df['language'].values[i])) is None:
                clean_language_dict[str(clean_movie_df['language'].values[i])] = len(clean_language_dict)
        pickle_dicts = {
            'clean_movie_dict': clean_movie_dict,
            'clean_director_dict': clean_director_dict,
            'clean_screenwriter_dict': clean_screenwriter_dict,
            'clean_actor_dict': clean_actor_dict,
            'clean_movie_type_dict': clean_movie_type_dict,
            'clean_area_dict': clean_area_dict,
            'clean_language_dict': clean_language_dict
        }
        try:
            if not os.path.exists(clean_movie_pickle_path):
                open(clean_movie_pickle_path, 'w')
                print('clean_movie.pickle not exists, create it')
            with open(clean_movie_pickle_path, 'wb') as f:
                pickle.dump(pickle_dicts, f)
            print('save clean_movie.pickle', len(clean_movie_dict))
        except Exception:
            print('save clean_movie error')

    def save_clean_user_pickle(self):
        clean_user_pickle_path = self.pickle_dir + 'clean_user.pickle'
        clean_user_dict = dict()
        clean_movie_user_table_path_list = os.listdir(self.clean_movie_user_csv_dir)
        for clean_movie_user_table_path in clean_movie_user_table_path_list:
            if clean_user_dict.get(str(clean_movie_user_table_path)[0:-4]) is None:
                clean_user_dict[str(clean_movie_user_table_path)[0:-4]] = len(clean_user_dict)
        try:
            if not os.path.exists(clean_user_pickle_path):
                open(clean_user_pickle_path, 'w')
                print('clean_user.pickle not exists, create it')
            with open(clean_user_pickle_path, 'wb') as f:
                pickle.dump(clean_user_dict, f)
            print('save clean_user.pickle', len(clean_user_dict))
        except Exception:
            print('save clean_user error')

    def clean_movie_data(self):
        # https://blog.csdn.net/weixin_39750084/article/details/81750185
        # 去掉没有评分的 ， 没有投票数的, 没有日期的, 没有地区和语言的，以及电影名字重复的
        if not os.path.exists(self.movie_csv_path):
            print(self.movie_csv_path + ' not exist')
            return False
        with open(self.movie_csv_path, 'rb') as f:
            movie_df = pd.read_csv(f, encoding="utf-8")
        clean_movie_df = movie_df[(~pd.isna(movie_df['star'])) & (~pd.isna(movie_df['vote_num'])) &
                                  (~pd.isna(movie_df['date'])) & (~pd.isna(movie_df['area']))
                                  & (~pd.isna(movie_df['language']))]
        clean_movie_df = clean_movie_df.drop_duplicates(['title'])
        clean_movie_df = clean_movie_df[(clean_movie_df['title'] != '解放') & (clean_movie_df['title'] != '撒旦探戈')]
        # 写入清洗之后的数据 到 clean_movie_csv
        clean_movie_df.to_csv(self.clean_movie_csv_path, encoding="utf-8", index=False)
        self.save_clean_movie_pickle()

    def clean_user_data(self):
        # https://blog.csdn.net/weixin_39750084/article/details/81750185
        # 去掉每个用户中电影重复的
        movie_user_table_path_list = os.listdir(self.movie_user_csv_dir)
        for movie_user_table_path in movie_user_table_path_list:
            movie_user_table_file = os.path.join(self.movie_user_csv_dir, movie_user_table_path)
            with open(movie_user_table_file, 'rb') as f:
                movie_user_df = pd.read_csv(f, encoding="utf-8")
                clean_movie_user_df = movie_user_df.drop_duplicates(['titles'])
            # 写入清洗之后的数据 到 clean_movie_csv 有的用户没有数据 即空表 这些空表就不保存了
            try:
                if str(movie_user_table_path)[0:-4] == str(clean_movie_user_df['user_name'][0]):
                    file_name = str(movie_user_table_path)[0:-4]
                    clean_movie_user_df.to_csv(self.clean_movie_user_csv_dir + file_name + '.csv', encoding="utf-8", index=False)
                else:
                    print('name1', str(movie_user_table_path)[0:-4])
                    print('name2', clean_movie_user_df['user_name'][0])
            except Exception:
                print('user ' + str(movie_user_table_path)[0:-4] + ' has no data! forget it')
        self.save_clean_user_pickle()

    # 获得用户和对应的按照类型不同的电影数目和平均分以及该用户所有爬取到的电影的平均分
    def save_user_table(self):
        if not os.path.exists(self.clean_movie_csv_path):
            print(self.clean_movie_csv_path + ' not exist')
            return False
        with open(self.clean_movie_csv_path, 'rb') as f:
            clean_movie_df = pd.read_csv(f, encoding="utf-8")

        type_rating_columns = list()
        for key in self.clean_movie_type_dict.keys():
            type_rating_columns.append(str(key))
            type_rating_columns.append(str(key) + '_avg_ratings')
        clean_movie_user_table_path_list = os.listdir(self.clean_movie_user_csv_dir)
        for clean_movie_user_table_path in clean_movie_user_table_path_list:
            clean_movie_user_table_file = os.path.join(self.clean_movie_user_csv_dir, clean_movie_user_table_path)
            user_info = dict()
            with open(clean_movie_user_table_file, 'rb') as f:
                clean_movie_user_df = pd.read_csv(f, encoding="utf-8")
            user_info['user_name'] = str(clean_movie_user_df['user_name'][0])
            # mean会返回nan, count返回0
            if not np.isnan(clean_movie_user_df['ratings'].mean()):
                user_info['avg_ratings'] = clean_movie_user_df['ratings'].mean()
            else:
                user_info['avg_ratings'] = 0.0
            # 这里这样写而不用 for key in self.clean_movie_type_dict.keys(): 是担心两次的类型顺序不一致，导致列中的数据混乱
            for index in range(0, len(type_rating_columns), 2):
                key = type_rating_columns[index]
                key_rating = type_rating_columns[index + 1]
                titles_per_type = clean_movie_df[clean_movie_df['movie_type'] == key]['title']
                user_info[key] = clean_movie_user_df[clean_movie_user_df['titles'].isin(titles_per_type)]['titles'].count()
                if not np.isnan(clean_movie_user_df[clean_movie_user_df['titles'].isin(titles_per_type)]['ratings'].mean()):
                    user_info[key_rating] = clean_movie_user_df[clean_movie_user_df['titles'].isin(titles_per_type)]['ratings'].mean()
                else:
                    user_info[key_rating] = 0.0
            if not os.path.exists(self.clean_user_csv_path):
                df = pd.DataFrame(user_info, columns=['user_name', 'avg_ratings'] + type_rating_columns, index=[0])
                df.to_csv(self.clean_user_csv_path, encoding="utf-8", index=False)
            else:
                df = pd.DataFrame(user_info, columns=['user_name', 'avg_ratings'] + type_rating_columns, index=[0])
                df.to_csv(self.clean_user_csv_path, encoding="utf-8", mode='a', header=False, index=False)

    # 按照原始电影类型计算的平均分，以及该类型电影的数目
    def save_average_rating_movie_type(self):
        if not os.path.exists(self.clean_movie_csv_path):
            print(self.clean_movie_csv_path + ' not exist')
        else:
            with open(self.clean_movie_csv_path, 'rb') as f:
                clean_movie_df = pd.read_csv(f, encoding="utf-8")
                datas = []
                for key in self.clean_movie_type_dict.keys():
                    tmp_dict = dict()
                    tmp_dict['movie_type'] = key
                    tmp_dict['count'] = clean_movie_df[clean_movie_df['movie_type'] == key]['title'].count()
                    if not np.isnan(clean_movie_df[clean_movie_df['movie_type'] == key]['runtime'].mean()):
                        tmp_dict['runtime'] = clean_movie_df[clean_movie_df['movie_type'] == key]['runtime'].mean()
                    else:
                        tmp_dict['runtime'] = 0.0
                    if not np.isnan(clean_movie_df[clean_movie_df['movie_type'] == key]['vote_num'].mean()):
                        tmp_dict['vote_num'] = clean_movie_df[clean_movie_df['movie_type'] == key]['vote_num'].mean()
                    else:
                        tmp_dict['vote_num'] = 0.0
                    if not np.isnan(clean_movie_df[clean_movie_df['movie_type'] == key]['star'].mean()):
                        tmp_dict['avg_star'] = clean_movie_df[clean_movie_df['movie_type'] == key]['star'].mean()
                    else:
                        tmp_dict['avg_star'] = 0.0
                    datas.append(tmp_dict)
                movie_type_mean_df = pd.DataFrame(data=datas,
                                                  columns=['movie_type', 'count', 'runtime', 'vote_num', 'avg_star'],
                                                  index=range(len(datas)))
                movie_type_mean_df.to_csv(self.average_rating_movie_type_path, encoding="utf-8", index=False)

    # 按照原始电影地区计算的平均分，以及该类型电影的数目
    def save_average_rating_area(self):
        if not os.path.exists(self.clean_movie_csv_path):
            print(self.clean_movie_csv_path + ' not exist')
        else:
            with open(self.clean_movie_csv_path, 'rb') as f:
                clean_movie_df = pd.read_csv(f, encoding="utf-8")
                datas = []
                for key in self.clean_area_dict.keys():
                    tmp_dict = dict()
                    tmp_dict['area'] = key
                    tmp_dict['count'] = clean_movie_df[clean_movie_df['area'] == key]['title'].count()
                    if not np.isnan(clean_movie_df[clean_movie_df['area'] == key]['runtime'].mean()):
                        tmp_dict['runtime'] = clean_movie_df[clean_movie_df['area'] == key]['runtime'].mean()
                    else:
                        tmp_dict['runtime'] = 0.0
                    if not np.isnan(clean_movie_df[clean_movie_df['area'] == key]['vote_num'].mean()):
                        tmp_dict['vote_num'] = clean_movie_df[clean_movie_df['area'] == key]['vote_num'].mean()
                    else:
                        tmp_dict['vote_num'] = 0.0
                    if not np.isnan(clean_movie_df[clean_movie_df['area'] == key]['star'].mean()):
                        tmp_dict['avg_star'] = clean_movie_df[clean_movie_df['area'] == key]['star'].mean()
                    else:
                        tmp_dict['avg_star'] = 0.0
                    datas.append(tmp_dict)
                area_mean_df = pd.DataFrame(data=datas, columns=['area', 'count', 'runtime', 'vote_num', 'avg_star'],
                                            index=range(len(datas)))
                area_mean_df.to_csv(self.average_rating_area_path,  encoding="utf-8", index=False)

    # 按照原始电影语言计算的平均分，以及该类型电影的数目
    def save_average_rating_language(self):
        if not os.path.exists(self.clean_movie_csv_path):
            print(self.clean_movie_csv_path + ' not exist')
        else:
            with open(self.clean_movie_csv_path, 'rb') as f:
                clean_movie_df = pd.read_csv(f, encoding="utf-8")
                datas = []
                for key in self.clean_language_dict.keys():
                    tmp_dict = dict()
                    tmp_dict['language'] = key
                    tmp_dict['count'] = clean_movie_df[clean_movie_df['language'] == key]['title'].count()
                    if not np.isnan(clean_movie_df[clean_movie_df['language'] == key]['runtime'].mean()):
                        tmp_dict['runtime'] = clean_movie_df[clean_movie_df['language'] == key]['runtime'].mean()
                    else:
                        tmp_dict['runtime'] = 0.0
                    if not np.isnan(clean_movie_df[clean_movie_df['language'] == key]['vote_num'].mean()):
                        tmp_dict['vote_num'] = clean_movie_df[clean_movie_df['language'] == key]['vote_num'].mean()
                    else:
                        tmp_dict['vote_num'] = 0.0
                    if not np.isnan(clean_movie_df[clean_movie_df['language'] == key]['star'].mean()):
                        tmp_dict['avg_star'] = clean_movie_df[clean_movie_df['language'] == key]['star'].mean()
                    else:
                        tmp_dict['avg_star'] = 0.0
                    datas.append(tmp_dict)
                language_mean_df = pd.DataFrame(data=datas,
                                                columns=['language', 'count', 'runtime', 'vote_num', 'avg_star'],
                                                index=range(len(datas)))
                language_mean_df.to_csv(self.average_rating_language_path, encoding="utf-8", index=False)

    # 按照划分好的电影时长0-20 20-40 40-60...计算的平均分，以及该类型电影的数目
    def save_average_rating_runtime(self):
        if not os.path.exists(self.clean_movie_csv_path):
            print(self.clean_movie_csv_path + ' not exist')
            return False
        with open(self.clean_movie_csv_path, 'rb') as f:
            clean_movie_df = pd.read_csv(f, encoding="utf-8")
            datas = []
            for j in range(1, 16):
                tmp_dict = dict()
                tmp_dict['runtime_low'] = (j - 1) * 25
                tmp_dict['runtime_up'] = j * 25
                tmp_dict['count'] = clean_movie_df[(clean_movie_df['runtime'] < j * 25) & (
                        clean_movie_df['runtime'] >= (j - 1) * 25)]['title'].count()
                tmp_dict['vote_num'] = clean_movie_df[(clean_movie_df['runtime'] < j * 25) & (
                        clean_movie_df['runtime'] >= (j - 1) * 25)]['vote_num'].mean()
                tmp_dict['avg_star'] = clean_movie_df[(clean_movie_df['runtime'] < j * 25) & (
                        clean_movie_df['runtime'] >= (j - 1) * 25)]['star'].mean()
                datas.append(tmp_dict)
            runtime_mean_df = pd.DataFrame(data=datas, columns=['runtime_low', 'runtime_up','count', 'vote_num', 'avg_star'],
                                           index=range(len(datas)))
            runtime_mean_df.to_csv(self.average_rating_runtime_path, encoding="utf-8", index=False)

    def data_progress(self):
        # 清洗电影和用户数据 并在内部保存清洗之后的dict
        print('clean movie data start')
        self.clean_movie_data()
        print('clean movie data end')

        print('clean user data start')
        self.clean_user_data()
        print('clean user data end')

        # 读取保存的dict到内存
        self.read_clean_pickle()

        # 保存统计的表
        print('save user table start')
        self.save_user_table()
        print('save user table end')

        print('save average rating movie type start')
        self.save_average_rating_movie_type()
        print('save average rating movie type end')

        print('save average rating area start')
        self.save_average_rating_area()
        print('save average rating area end')

        print('save average rating language start')
        self.save_average_rating_language()
        print('save average rating language end')

        print('save average rating runtime start')
        self.save_average_rating_runtime()
        print('save average rating runtime end')

        # 获得电影和用户的训练数据并保存
        print('get movie train data start')
        self.get_movie_train_data(normalization=True)
        print('get movie train data end')

        print('get user train data start')
        self.get_user_train_data(normalization=True)
        print('get user train data end')


if __name__ == '__main__':
    DU = DataUtil()
    DU.data_progress()
