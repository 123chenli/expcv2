import urllib.request, socket, re, sys, os

baseUrl = 'http://image.baidu.com/search/index?tn=baiduimage&ct=201326592&lm=-1&cl=2&ie=gbk&word=%B9%F3%B1%F6%C8%AE&fr=ala&ala=1&alatpl=adress&pos=0&hs=2&xthttps=000000'
# 定义文件保存路径
targetPath = 'D:\file\Images\husky'


def getContant(Weburl):
    Webheader = {'Upgrade-Insecure-Requests': '1',
              'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36'}
    req = urllib.request.Request(url=Weburl, headers=Webheader)
    response = urllib.request.urlopen(req)
    _contant = response.read()
    response.close()
    return str(_contant)


def openUrl(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 9_1 like Mac OS X) AppleWebKit/601.1.46 (KHTML, like Gecko) Version/9.0 Mobile/13B143 Safari/601.1'
    }
    req = urllib.request.Request(url=url, headers=headers)
    res = urllib.request.urlopen(req)
    data = res.read()
    downImg(data)


def downImg(data):
    for link, t in set(re.findall(r'([http|https]: [^\s]*?(jpa|png|gif))', str(data))):
        if link.startswith('s'):
            link = 'http' + link
        else:
            link = 'htt' + link
        print(link)
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(link, saveFile(link))
        except:
            print('失败')


def saveFile(path):
    # 检测当前路径的有效性
    if not os.path.isdir(targetPath):
        os.mkdir(targetPath)

    # 设置每个图片的路径
    pos = path.rindex('/')
    t = os.path.join(targetPath, path[pos+1: ])
    return t

url = 'http://image.baidu.com/cardserver/search?para=%5B%7B%22ct%22%3A%22simi%22%2C%22cv%22%3A%5B%7B%22provider%22%3A%22piclist%22%2C%22Https%22%3A%220%22%2C%22query%22%3A%22%E8%B4%B5%E5%AE%BE%E7%8A%AC%22%2C%22SimiCs%22%3A%221421890046%2C169180912%22%2C%22type%22%3A%22card%22%2C%22pn%22%3A%220%22%2C%22rn%22%3A%226%22%2C%22srctype%22%3A%22%22%2C%22bdtype%22%3A%22%22%2C%22os%22%3A%221435516746%2C4237255299%22%7D%5D%7D%5D'
openUrl(url)