# -*-coding:utf-8-*-
import requests
from bs4 import BeautifulSoup
import threading
import os
import pandas as pd
import re
import time
import pickle
import random
import string


class DouBanCrawler(object):
    def __init__(self, start_url):
        self.movie_visited = dict()
        self.user_visited = dict()

        self.director_dict = dict()
        self.screenwriter_dict = dict()
        self.actor_dict = dict()
        self.movie_type_dict = dict()
        self.area_dict = dict()
        self.language_dict = dict()

        self.movie_url_list = list()
        self.user_url_list = list([start_url])
        self.USER_WEB_HEADER = {
            "Host": "movie.douban.com",
            "scheme": "https",
            "Connection": "keep-alive",
            "version": "HTTP/1.1",
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q = 0.8",
            "accept-encoding": "gzip,deflate,br",
            "accept-language": "zh-CN,zh;q=0.8",
            "cache-control": "max-age=0",
            "cookie": '',  # add a cookie
            "referer": "movie.douban.com/people/84781545/collect",
            "upgrade-insecure -requests": "1",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/72.0.3626.121 Chrome/72.0.3626.121 Safari/537.36"
        }
        self.MOVIE_WEB_HEADER = {
            "Host": "movie.douban.com",
            "scheme": "https",
            "Connection": "keep-alive",
            "version": "HTTP/1.1",
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q = 0.8",
            "accept-encoding": "gzip,deflate,br",
            "accept-language": "zh-CN,zh;q=0.8",
            "cache-control": "max-age=0",
            "cookie": '',  # add a cookie
            "referer": "movie.douban.com/subject/26213252/",
            "upgrade-insecure -requests": "1",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/72.0.3626.121 Chrome/72.0.3626.121 Safari/537.36"
        }
        self.user_start_url = "https://movie.douban.com/people/84781545/collect"
        self.movie_csv_path = './douban/my_data/mv_table.csv'
        self.movie_user_csv_dir = './douban/my_data/mv_us_tables/'
        self.pickle_dir = './douban/my_data/'
        self.MOVIE_MAX_NUM_PER_USER = 200

    def _refresh_header(self):
        # https://zhuanlan.zhihu.com/p/24035574
        # refresh the cookie bid value
        start = ''
        end = ''
        bid = "bid=%s" % "".join(random.sample(string.ascii_letters + string.digits, 11))
        self.USER_WEB_HEADER["cookie"] = start + bid + end
        self.MOVIE_WEB_HEADER["cookie"] = start + bid + end
        # print(start + bid + end)
        print('refresh_header')

    def _get_movie_info(self, movie_html):
        # https://github.com/MashiMaroLjc/ML-and-DM-in-action/blob/master/DouBanMovie/douban.py
        soup = BeautifulSoup(movie_html, "html.parser")
        title_span = soup.find("span", attrs={"property": "v:itemreviewed"})
        # title
        try:
            title = str(title_span.string.split(' ')[0])
        except Exception:
            title = ""

        # print('title', title)
        if self.movie_visited.get(title) is not None:
            print('movie ' + title + ' visited')
            return False
        else:
            self.movie_visited[title] = len(self.movie_visited)

        director_a = soup.find("a", attrs={"rel": "v:directedBy"})
        try:
            director = str(director_a.string)
        except Exception:
            director = ""

        # print('director', director)
        if self.director_dict.get(director) is None:
            self.director_dict[director] = len(self.director_dict)

        screenwriter_a = soup.find(href=re.compile("/celebrity/\d{7}/"), attrs={"rel": ""})
        try:
            screenwriter = str(screenwriter_a.string)
        except Exception:
            screenwriter = ""

        # print('screenwriter', screenwriter)
        if self.screenwriter_dict.get(screenwriter) is None:
            self.screenwriter_dict[screenwriter] = len(self.screenwriter_dict)

        actors_a = soup.find_all(attrs={"rel": "v:starring"})[0:2]
        try:
            actors = [str(name.string) for name in actors_a]
        except Exception:
            actors = [""]

        # print('actors', actors)
        for a in actors:
            if self.actor_dict.get(a) is None:
                self.actor_dict[a] = len(self.actor_dict)

        movie_type_span = soup.find_all("span", attrs={"property": "v:genre"})[0]
        try:
            movie_type = str(movie_type_span.string)
        except Exception:
            movie_type = ""

        # print('movie_type', movie_type)
        if self.movie_type_dict.get(movie_type) is None:
            self.movie_type_dict[movie_type] = len(self.movie_type_dict)

        area_index = movie_html.find("制片国家/地区:</span>")
        end_index = movie_html.find("br", area_index)
        if area_index != -1 and end_index != -1:
            area = str(movie_html[area_index + 16:end_index - 1]).split(' ')[0]
        else:
            area = ""

        # print('area', area)
        if self.area_dict.get(area) is None:
            self.area_dict[area] = len(self.area_dict)

        language_index = movie_html.find("语言:</span>")
        end_index = movie_html.find("br", language_index)
        if language_index != -1 and end_index != -1:
            language = str(movie_html[language_index + 11:end_index - 1].split(' ')[0])
        else:
            language = ""

        # print('language', language)
        if self.language_dict.get(language) is None:
            self.language_dict[language] = len(self.language_dict)

        date_span = soup.find("span", attrs={"property": "v:initialReleaseDate"})
        try:
            date = date_span.string[0:10]
        except Exception:
            date = ""
        # print('date', date)

        runtime_span = soup.find("span", attrs={"property": "v:runtime"})
        try:
            runtime = runtime_span['content']
        except Exception:
            runtime = ""
        # print('runtime', runtime)

        vote_num_span = soup.find("span", attrs={"property": "v:votes"})
        try:
            vote_num = vote_num_span.string
        except Exception:
            vote_num = ""
        # print('vote_num', vote_num)

        star_strong = soup.find("strong", attrs={"property": "v:average"})
        try:
            star = star_strong.string
        except Exception:
            star = "-1"
        # print('star', star)

        info = {
            "title": title,
            "director": director,
            "screenwriter": screenwriter,
            "actor1": actors[0],
            "actor2": actors[1],
            "movie_type": movie_type,
            "area": area,
            "language": language,
            "date": date,
            "runtime": runtime,
            "vote_num": vote_num,
            "star": star
        }
        self._save_m_table(info)
        return True

    def _get_user_info(self, user_url, paras_left_string, paras_right_string, timeout):
        # 解析得到该用户对电影的评分，并保存到mv_us_table.csv中，并保存电影的url
        start_index = 0
        movie_url_count = 0
        u_url = user_url + paras_left_string + str(start_index) + paras_right_string
        try:
            time.sleep(random.randint(1, 3))
            u_response = requests.get(u_url, headers=self.USER_WEB_HEADER, timeout=timeout)
            if str(u_response.status_code) == '200':
                u_response.encoding = "utf-8"
                u_html = u_response.text
                soup = BeautifulSoup(u_html, "html.parser")
                info_h1 = soup.find("div", attrs={"class": "info"}).h1.string
                user_collect_max = int(info_h1[info_h1.index('(') + 1: -1])
                user_name = info_h1[0:info_h1.index("看过的电影")]
                print('user_collect_max', user_collect_max)
                print('user_name', user_name)
                if self.user_visited.get(user_name) is not None:
                    print('user ' + user_name + ' visited')
                    return False
                else:
                    self.user_visited[user_name] = len(self.user_visited)

                raw_movies_div = soup.find_all("div", attrs={"class": "info"})
                movies_div = [d for d in raw_movies_div if d.span is not None and "rating" in str(d.find_all("span"))]

                titles_li = [m.find("li", attrs={"class": "title"}) for m in movies_div]
                raw_movie_urls = [t.a['href'] for t in titles_li]
                titles_em = [t.em.string for t in titles_li]
                titles = [s.split(' ')[0] for s in titles_em]

                ratings_span = [m.find("span", attrs={"class": re.compile("^rating[0-5]")}) for m in movies_div]
                ratings_class = [r['class'][0] for r in ratings_span]
                ratings = [int(str(r)[6])*2 for r in ratings_class]
                user_info = {
                    "user_name": user_name,
                    "titles": titles,
                    "ratings": ratings
                }
                self._save_m_u_table(user_info)
                # 为了获得更多的电影，这里过滤掉已访问的电影
                movie_urls = []
                for m in range(len(titles)):
                    if self.movie_visited.get(titles[m]) is None:
                        movie_urls.append(raw_movie_urls[m])

                self.movie_url_list += movie_urls
                movie_url_count += len(movie_urls)
                start_index += 15
            else:
                self._refresh_header()
                print('user not 200 forget it')
                return False
            while start_index < (user_collect_max - 15) and movie_url_count < self.MOVIE_MAX_NUM_PER_USER:
                u_url = user_url + paras_left_string + str(start_index) + paras_right_string
                time.sleep(random.randint(1, 3))
                u_response = requests.get(u_url, headers=self.USER_WEB_HEADER, timeout=timeout)
                if str(u_response.status_code) == '200':
                    u_response.encoding = "utf-8"
                    u_html = u_response.text
                    soup = BeautifulSoup(u_html, "html.parser")
                    raw_movies_div = soup.find_all("div", attrs={"class": "info"})
                    movies_div = [d for d in raw_movies_div if d.span is not None and "rating" in str(d.find_all("span"))]

                    titles_li = [m.find("li", attrs={"class": "title"}) for m in movies_div]
                    raw_movie_urls = [t.a['href'] for t in titles_li]
                    titles_em = [t.em.string for t in titles_li]
                    titles = [s.split(' ')[0] for s in titles_em]

                    ratings_span = [m.find("span", attrs={"class": re.compile("^rating[0-5]")}) for m in movies_div]
                    ratings_class = [r['class'][0] for r in ratings_span]
                    ratings = [int(str(r)[6]) * 2 for r in ratings_class]
                    user_info = {
                        "user_name": user_name,
                        "titles": titles,
                        "ratings": ratings
                    }
                    self._save_m_u_table(user_info)
                    # 为了获得更多的电影，这里过滤掉已访问的电影
                    movie_urls = []
                    for m in range(len(titles)):
                        if self.movie_visited.get(titles[m]) is None:
                            movie_urls.append(raw_movie_urls[m])

                    self.movie_url_list += movie_urls
                    movie_url_count += len(movie_urls)
                    start_index += 15
                else:
                    self._refresh_header()
            print('movie_url_count', movie_url_count)
            return True
        except Exception:
            print(u_url + ' maybe time out')
            return False

    def _save_m_u_table(self, user_info):
        m_u_path = self.movie_user_csv_dir + '/' + user_info["user_name"] + '.csv'
        if not os.path.exists(m_u_path):
            df = pd.DataFrame(user_info, columns=["user_name", "titles", "ratings"])
            df.to_csv(m_u_path, encoding="utf-8", index=False)
        else:
            df = pd.DataFrame(user_info, columns=["user_name", "titles", "ratings"])
            df.to_csv(m_u_path, mode='a', encoding="utf-8", header=False, index=False)
        # print('save m_u_table')

    def _save_m_table(self, movie_info):
        if not os.path.exists(self.movie_csv_path):
            df = pd.DataFrame(movie_info, columns=["title", "director", "screenwriter", "actor1", "actor2", "movie_type",
                                                   "area", "language", "date", "runtime", "vote_num", "star"], index=[0])
            df.to_csv(self.movie_csv_path, encoding="utf-8", index=False)
        else:
            df = pd.DataFrame(movie_info, columns=["title", "director", "screenwriter", "actor1", "actor2", "movie_type",
                                                   "area", "language", "date", "runtime", "vote_num", "star"], index=[0])
            df.to_csv(self.movie_csv_path, mode='a',  encoding="utf-8", header=False, index=False)
        # print('save_m_table')

    def _get_user_url_from_movie(self, reviews_html):
        # 只要前5个 避免多次迭代深度过深 速度太慢
        soup = BeautifulSoup(reviews_html, "html.parser")
        start_string = 'https://movie.douban.com/people/'
        raw_urls_a = soup.find_all("a", attrs={"class": "avator"})[0:5]
        raw_urls = [r['href'] for r in raw_urls_a]
        key_words = [r.split('/')[-2] for r in raw_urls]
        urls = [start_string + k + '/collect' for k in key_words]
        self.user_url_list += urls
        print('user_url', len(urls))

    def run_user_crawler(self, timeout):
        print('user crawler start')
        count = 0
        while len(self.user_url_list) != 0:
            user_url = self.user_url_list.pop(0)
            paras_left_string = '?start='
            paras_right_string = '&sort=time&rating=all&filter=all&mode=grid'
            res = self._get_user_info(user_url, paras_left_string, paras_right_string, timeout)
            if res:
                count += 1
        print('user crawler stop')
        return count

    def _save_user_pickle(self):
        user_pickle_path = self.pickle_dir + 'user_visited.pickle'
        try:
            if not os.path.exists(user_pickle_path):
                open(user_pickle_path, 'w')
                print('user_visited.pickle not exists, create it')
            with open(user_pickle_path, 'wb') as f:
                pickle.dump(self.user_visited, f)
            print('save user dict to user_visited.pickle')
        except Exception:
            print('save user visited error')

    def run_movie_crawler(self, timeout):
        print('movie crawler start')
        count = 0
        user_count = 0
        res = True
        while len(self.movie_url_list) != 0:
            movie_url = self.movie_url_list.pop(0)
            try:
                time.sleep(random.randint(1, 5))
                m_response = requests.get(movie_url, headers=self.MOVIE_WEB_HEADER, timeout=timeout)
                if str(m_response.status_code) == '200':
                    m_response.encoding = "utf-8"
                    m_html = m_response.text
                    res = self._get_movie_info(m_html)
                    if res:
                        count += 1
                        if count % 50 == 0:
                            self._save_movie_pickle()
                            print('movie: ', count)
                else:
                    self._refresh_header()
                    print('movie not 200 forget it')
                if res and user_count < 2:
                    reviews_html = movie_url + '/reviews'
                    time.sleep(random.randint(1, 5))
                    r_response = requests.get(reviews_html, headers=self.MOVIE_WEB_HEADER, timeout=timeout)
                    if str(r_response.status_code) == '200':
                        r_response.encoding = "utf-8"
                        r_html = r_response.text
                        self._get_user_url_from_movie(r_html)
                    else:
                        self._refresh_header()
                        print('movie reviews not 200 forget it')
                    user_count += 1
            except Exception:
                print(movie_url + ' maybe time out')
        print('movie crawler stop')
        return count

    def _save_movie_pickle(self):
        movie_pickle_path = self.pickle_dir + 'movie.pickle'
        pickle_dicts = {
            'movie_visited': self.movie_visited,
            'director_dict': self.director_dict,
            'screenwriter_dict': self.screenwriter_dict,
            'actor_dict': self.actor_dict,
            'movie_type_dict': self.movie_type_dict,
            'area_dict': self.area_dict,
            'language_dict': self.language_dict
        }
        # https://stackoverflow.com/questions/22626003/pickle-dump-meet-runtimeerror-maximum-recursion-depth-exceeded-in-cmp
        try:
            if not os.path.exists(movie_pickle_path):
                open(movie_pickle_path, 'w')
                print(movie_pickle_path + ' not exists, create it')
            with open(movie_pickle_path, 'wb') as f:
                pickle.dump(pickle_dicts, f)
                print('save pickle dict to ' + movie_pickle_path)
        except Exception:
            print('save ' + movie_pickle_path + ' error')

    def run(self):
        total_movie_num = 0
        total_user_num = 0
        self._save_user_pickle()
        self._save_movie_pickle()
        while True:
            self.reload_user_pickle()
            self._read_pickle()
            stop1 = random.randint(a=20, b=120)
            print('stop1:', stop1)
            time.sleep(stop1)
            total_user_num += self.run_user_crawler(timeout=5)
            self._save_user_pickle()
            print('user num: ', total_user_num)

            stop2 = random.randint(a=20, b=120)
            print('stop2:', stop2)
            time.sleep(stop2)
            total_movie_num += self.run_movie_crawler(timeout=5)
            self._save_movie_pickle()
            print('movie num: ', total_movie_num)

    def _read_pickle(self):
        user_pickle_path = self.pickle_dir + 'user_visited.pickle'
        movie_pickle_path = self.pickle_dir + 'movie.pickle'
        try:
            with open(user_pickle_path, 'rb') as f:
                self.user_visited = pickle.load(f)
            with open(movie_pickle_path, 'rb') as f:
                movie_all_dict = pickle.load(f)
                self.movie_visited = movie_all_dict['movie_visited']
                self.director_dict = movie_all_dict['director_dict']
                self.screenwriter_dict = movie_all_dict['screenwriter_dict']
                self.actor_dict = movie_all_dict['actor_dict']
                self.movie_type_dict = movie_all_dict['movie_type_dict']
                self.area_dict = movie_all_dict['area_dict']
                self.language_dict = movie_all_dict['language_dict']
            print('user_visited', len(self.user_visited))
            print('movie_visited', len(self.movie_visited))
            print('director_dict', len(self.director_dict))
            print('screenwriter_dict', len(self.screenwriter_dict))
            print('actor_dict', len(self.actor_dict))
            print('movie_type_dict', len(self.movie_type_dict))
            print('area_dict', len(self.area_dict))
            print('language_dict', len(self.language_dict))
        except Exception:
            print('read pickle error')

    def reload_user_pickle(self):
        txt_files = os.listdir(self.movie_user_csv_dir)
        for f_name in txt_files:
            self.user_visited[str(f_name)[0:-4]] = len(self.user_visited)
        user_pickle_path = self.pickle_dir + 'user_visited.pickle'
        try:
            if not os.path.exists(user_pickle_path):
                open(user_pickle_path, 'w')
                print('user_visited.pickle not exists, create it')
            with open(user_pickle_path, 'wb') as f:
                pickle.dump(self.user_visited, f)
            print('save user dict to user_visited.pickle')
        except Exception:
            print('save user visited error')


if __name__ == '__main__':
    DB = DouBanCrawler(start_url="https://movie.douban.com/people/mr_tree/collect")
    DB.run()