# -*-coding:utf-8-*-
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
# http://python.jobbole.com/87831/

# plt.style.available
PltStyleAvailable = ['seaborn-dark-palette',
                     'seaborn-bright',
                     '_classic_test',
                     'seaborn-pastel',
                     'seaborn-paper',
                     'tableau-colorblind10',
                     'Solarize_Light2',
                     'bmh',
                     'dark_background',
                     'seaborn-deep',
                     'ggplot',
                     'seaborn-talk',
                     'seaborn-muted',
                     'grayscale',
                     'seaborn-notebook',
                     'seaborn',
                     'fivethirtyeight',
                     'seaborn-whitegrid',
                     'seaborn-white',
                     'seaborn-darkgrid',
                     'fast',
                     'seaborn-ticks',
                     'seaborn-dark',
                     'seaborn-poster',
                     'classic',
                     'seaborn-colorblind']

# https://medium.com/marketingdatascience/%E8%A7%A3%E6%B1%BApython-3-matplotlib%E8%88%87seaborn%E8%A6%96%E8%A6%BA%E5%8C%96%E5%A5%97%E4%BB%B6%E4%B8%AD%E6%96%87%E9%A1%AF%E7%A4%BA%E5%95%8F%E9%A1%8C-f7b3773a889b
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
# 选择样式
plt.style.use('bmh')

# 电影["title", "director", "screenwriter", "actor1", "actor2", "movie_type",
#                                                    "area", "language", "date", "runtime", "vote_num", "star"]

# 用户["user_name", "titles", "ratings"]


# 统计不同分数段的电影个数
def x_ratings_y_count():
    if not os.path.exists(clean_movie_csv_path):
        print(clean_movie_csv_path + ' not exist')
        return False
    with open(clean_movie_csv_path, 'rb') as f:
        movie_df = pd.read_csv(f, encoding="utf-8")
        datas = []
        for i in range(10):
            tmp_dict = dict()
            tmp_dict['rating'] = str(i) + '-' + str(i+1)
            tmp_dict['count'] = movie_df[(i < movie_df['star']) & (movie_df['star'] < i + 1)]['title'].count()
            datas.append(tmp_dict)
        plot_df = pd.DataFrame(data=datas, columns=['rating', 'count'], index=range(10))
        plot_df.plot(kind='barh', y="count", x="rating")
        plt.show()


# 统计不同地区的高分电影数目和分数
def x_area_y_ratings():
    if not os.path.exists(clean_movie_csv_path):
        print(clean_movie_csv_path + ' not exist')
        return False
    with open(clean_movie_csv_path, 'rb') as f:
        movie_df = pd.read_csv(f, encoding="utf-8")
        datas = []
        for key in clean_area_dict.keys():
            tmp_dict = dict()
            count = 0
            tmp_dict['area'] = key
            for i in range(10):
                tmp_dict[str(i) + '-' + str(i+1)] = movie_df[(movie_df['area'] == key) & (i < movie_df['star']) &
                                                             (movie_df['star'] < i + 1)]['title'].count()
                count += tmp_dict[str(i) + '-' + str(i+1)]
            if count > 100:
                datas.append(tmp_dict)
        plot_df = pd.DataFrame(data=datas, columns=['area', '0-1', '1-2', '2-3', '3-4', '4-5', '5-6',
                                                    '6-7', '7-8', '8-9', '9-10'], index=range(len(datas)))
        plot_df.plot(kind='barh', stacked=True, x="area")
        plt.show()


# 统计不同的类型电影数目和分数
def x_movie_type_y_ratings():
    if not os.path.exists(clean_movie_csv_path):
        print(clean_movie_csv_path + ' not exist')
        return False
    with open(clean_movie_csv_path, 'rb') as f:
        movie_df = pd.read_csv(f, encoding="utf-8")
        datas = []
        for key in clean_movie_type_dict.keys():
            count = 0
            tmp_dict = dict()
            tmp_dict['movie_type'] = key
            for i in range(10):
                tmp_dict[str(i) + '-' + str(i + 1)] = movie_df[(movie_df['movie_type'] == key) & (i < movie_df['star']) &
                                                               (movie_df['star'] < i + 1)]['title'].count()
                count += tmp_dict[str(i) + '-' + str(i + 1)]
            if count > 100:
                datas.append(tmp_dict)
        plot_df = pd.DataFrame(data=datas, columns=['movie_type', '0-1', '1-2', '2-3', '3-4', '4-5', '5-6',
                                                    '6-7', '7-8', '8-9', '9-10'], index=range(len(datas)))
        plot_df.plot(kind='barh', stacked=True, x="movie_type")
        plt.show()


# 统计不同的语言和分数
def x_language_y_ratings():
    if not os.path.exists(clean_movie_csv_path):
        print(clean_movie_csv_path + ' not exist')
        return False
    with open(clean_movie_csv_path, 'rb') as f:
        movie_df = pd.read_csv(f, encoding="utf-8")
        datas = []
        for key in clean_language_dict.keys():
            count = 0
            tmp_dict = dict()
            tmp_dict['language'] = key
            for i in range(10):
                tmp_dict[str(i) + '-' + str(i + 1)] = movie_df[(movie_df['language'] == key) & (i < movie_df['star']) &
                                                               (movie_df['star'] < i + 1)]['title'].count()
                count += tmp_dict[str(i) + '-' + str(i + 1)]
            if count > 100:
                datas.append(tmp_dict)
        plot_df = pd.DataFrame(data=datas, columns=['language', '0-1', '1-2', '2-3', '3-4', '4-5', '5-6',
                                                    '6-7', '7-8', '8-9', '9-10'], index=range(len(datas)))
        plot_df.plot(kind='barh', stacked=True, x="language")
        plt.show()


# 统计时长和电影分数的关系
def x_runtime_y_ratings():
    if not os.path.exists(clean_movie_csv_path):
        print(clean_movie_csv_path + ' not exist')
        return False
    with open(clean_movie_csv_path, 'rb') as f:
        movie_df = pd.read_csv(f, encoding="utf-8")
        datas = []
        for j in range(1, 11):
            tmp_dict = dict()
            tmp_dict['runtime'] = str((j-1)*20) + '-' + str(j*20) + '分钟'
            for i in range(10):
                tmp_dict[str(i) + '-' + str(i + 1)] = movie_df[(movie_df['runtime'] < j*20)
                                                               & (movie_df['runtime'] > (j - 1)*20)
                                                               & (i < movie_df['star'])
                                                               & (movie_df['star'] < i + 1)]['title'].count()
            datas.append(tmp_dict)
        plot_df = pd.DataFrame(data=datas, columns=['runtime', '0-1', '1-2', '2-3', '3-4', '4-5', '5-6',
                                                    '6-7', '7-8', '8-9', '9-10'], index=range(len(datas)))
        plot_df.plot(kind='bar', stacked=True, x="runtime")
        plt.show()


# 统计评分人数和电影分数的关系
def x_vote_num_y_ratings():
    if not os.path.exists(clean_movie_csv_path):
        print(clean_movie_csv_path + ' not exist')
        return False
    with open(clean_movie_csv_path, 'rb') as f:
        movie_df = pd.read_csv(f, encoding="utf-8")
        plot_df = movie_df.groupby('star').mean()
        # print(plot_df)
        plot_df.plot(kind='line', legend=True, y='vote_num')
        plt.show()


# 统计上映日期和电影个数的关系
def x_date_y_ratings():
    if not os.path.exists(clean_movie_csv_path):
        print(clean_movie_csv_path + ' not exist')
        return False
    with open(clean_movie_csv_path, 'rb') as f:
        movie_df = pd.read_csv(f, encoding="utf-8")
        datas = []
        for y in range(1950, 2020, 3):
            tmp_dict = dict()
            tmp_dict['date'] = str(y)
            for i in range(10):
                tmp_dict[str(i) + '-' + str(i + 1)] = movie_df[(movie_df['date'].str.contains(str(y)))
                                                               & (i < movie_df['star'])
                                                               & (movie_df['star'] < i + 1)]['title'].count()
            datas.append(tmp_dict)
        plot_df = pd.DataFrame(data=datas, columns=['date', '0-1', '1-2', '2-3', '3-4', '4-5', '5-6',
                                                    '6-7', '7-8', '8-9', '9-10'], index=range(len(datas)))
        plot_df.plot(kind='bar', stacked=True, x="date")
        plt.show()


# 榜单
# 豆瓣评论数 > 100,000
# 豆瓣评分 >= 8.5分
def x_title_y_vote_num_top(choose):
    if not os.path.exists(clean_movie_csv_path):
        print(clean_movie_csv_path + ' not exist')
        return False
    with open(clean_movie_csv_path, 'rb') as f:
        movie_df = pd.read_csv(f, encoding="utf-8")
        if choose == 0:
            # 2018
            plot_df1 = movie_df[(movie_df['date'].str.contains(str(2018))) & (movie_df['vote_num'] >= 100000)
                                & (movie_df['star'] >= 8.5)][['title', 'star', 'vote_num']]
            plot_df1.plot(kind='bar', x="title", y=['vote_num','star'], secondary_y=['star'])
            plt.show()
        elif choose == 1:
            # 2019
            plot_df2 = movie_df[(movie_df['date'].str.contains(str(2019))) & (movie_df['vote_num'] >= 50000)
                                & (movie_df['star'] >= 7.5)][['title', 'star', 'vote_num']]
            plot_df2.plot(kind='bar', x="title", y=['vote_num', 'star'], secondary_y=['star'])
            plt.show()
        elif choose == 2:
            # all
            plot_df3 = movie_df[(movie_df['vote_num'] >= 500000)
                                & (movie_df['star'] >= 9.0)][['title', 'star', 'vote_num']]
            plot_df3.plot(kind='bar', x="title", y=['vote_num', 'star'], secondary_y=['star'])
            plt.show()


# 导演
def x_director_y_vote_num_top():
    if not os.path.exists(clean_movie_csv_path):
        print(clean_movie_csv_path + ' not exist')
        return False
    with open(clean_movie_csv_path, 'rb') as f:
        movie_df = pd.read_csv(f, encoding="utf-8")
        datas = []
        for key in clean_director_dict.keys():
            tmp_dict = dict()
            tmp_dict['director'] = key
            tmp_dict['movie_num'] = movie_df[(movie_df['director'] == key) & (movie_df['star'] > 7.5)]['title'].count()

            if tmp_dict['movie_num'] > 12:
                datas.append(tmp_dict)
        plot_df = pd.DataFrame(data=datas, columns=['director', 'movie_num'], index=range(len(datas)))
        plot_df = plot_df.sort_values('movie_num')
        plot_df.plot(kind='barh', x='director')
        plt.show()


# 编剧
def x_screenwriter_y_vote_num_top():
    if not os.path.exists(clean_movie_csv_path):
        print(clean_movie_csv_path + ' not exist')
        return False
    with open(clean_movie_csv_path, 'rb') as f:
        movie_df = pd.read_csv(f, encoding="utf-8")
        datas = []
        for key in clean_screenwriter_dict.keys():
            if key != 'None':
                tmp_dict = dict()
                tmp_dict['screenwriter'] = key
                tmp_dict['movie_num'] = movie_df[(movie_df['screenwriter'] == key) & (movie_df['star'] > 7.5)]['title'].count()

                if tmp_dict['movie_num'] > 12:
                    datas.append(tmp_dict)
        plot_df = pd.DataFrame(data=datas, columns=['screenwriter', 'movie_num'], index=range(len(datas)))
        plot_df = plot_df.sort_values('movie_num')
        plot_df.plot(kind='barh', x='screenwriter')
        plt.show()


# 演员
def x_actor_y_vote_num_top():
    if not os.path.exists(clean_movie_csv_path):
        print(clean_movie_csv_path + ' not exist')
        return False
    with open(clean_movie_csv_path, 'rb') as f:
        movie_df = pd.read_csv(f, encoding="utf-8")
        datas = []
        for key in clean_actor_dict.keys():
            if key != 'None':
                tmp_dict = dict()
                tmp_dict['actor'] = key
                tmp_dict['movie_num'] = movie_df[(movie_df['actor1'] == key) &
                                                 (movie_df['star'] > 8.0)]['title'].count() + movie_df[
                    (movie_df['actor2'] == key) & (movie_df['star'] > 8.0)]['title'].count()

                if tmp_dict['movie_num'] > 15:
                    datas.append(tmp_dict)
        plot_df = pd.DataFrame(data=datas, columns=['actor', 'movie_num'], index=range(len(datas)))
        plot_df = plot_df.sort_values('movie_num')
        plot_df.plot(kind='barh', x='actor')
        plt.show()


if __name__ == '__main__':
    from matplotlib.font_manager import _rebuild

    _rebuild()  # reload一下

    clean_movie_csv_path = './douban/my_data/clean_mv_table.csv'

    # 读取清洗过的dict
    pickle_dir = './douban/my_data/'
    clean_movie_pickle_path = pickle_dir + 'clean_movie.pickle'
    try:
        with open(clean_movie_pickle_path, 'rb') as f:
            clean_movie_all_dict = pickle.load(f)
            clean_movie_dict = clean_movie_all_dict['clean_movie_dict']
            clean_director_dict = clean_movie_all_dict['clean_director_dict']
            clean_screenwriter_dict = clean_movie_all_dict['clean_screenwriter_dict']
            clean_actor_dict = clean_movie_all_dict['clean_actor_dict']
            clean_movie_type_dict = clean_movie_all_dict['clean_movie_type_dict']
            clean_area_dict = clean_movie_all_dict['clean_area_dict']
            clean_language_dict = clean_movie_all_dict['clean_language_dict']
        print('clean_movie_dict', len(clean_movie_dict))
        print('clean_director_dict', len(clean_director_dict))
        print('clean_screenwriter_dict', len(clean_screenwriter_dict))
        print('clean_actor_dict', len(clean_actor_dict))
        print('clean_movie_type_dict', len(clean_movie_type_dict))
        print('clean_area_dict', len(clean_area_dict))
        print('clean_language_dict', len(clean_language_dict))
    except Exception:
        print('read pickle error')
    x_ratings_y_count()
    x_area_y_ratings()
    x_movie_type_y_ratings()
    x_language_y_ratings()
    x_runtime_y_ratings()
    x_vote_num_y_ratings()
    x_date_y_ratings()
    # 0-2018 1-2019 2-all
    x_title_y_vote_num_top(2)
    x_director_y_vote_num_top()
    x_screenwriter_y_vote_num_top()
    x_actor_y_vote_num_top()

    # 检测可用的中文字体
    # from matplotlib.font_manager import FontManager
    # import subprocess
    #
    # fm = FontManager()
    # mat_fonts = set(f.name for f in fm.ttflist)
    # # print(mat_fonts)
    # output = subprocess.check_output('fc-list :lang=zh -f "%{family}\n"', shell=True)
    # # print( '*' * 10, '系统可用的中文字体', '*' * 10)
    # # print (output)
    # zh_fonts = set(f.split(',', 1)[0] for f in output.decode('utf-8').split('\n'))
    # available = mat_fonts & zh_fonts
    # print('*' * 10, '可用的字体', '*' * 10)
    # for f in available:
    #     print(f)