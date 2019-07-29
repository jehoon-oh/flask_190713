from konlpy.corpus import kolaw
print(kolaw.fileids())
c = kolaw.open('constitution.txt').read()
print(c[:40])

from konlpy.corpus import kobill
print(kobill.fileids())
d = kobill.open('1809890.txt').read()
print(d[:40])

from konlpy.tag import *
hannanum = Hannanum()
kkma = Kkma()
komoran = Komoran()
# mecab = Mecab()
okt = Okt()
print(hannanum.nouns(c[:40]))
print(kkma.nouns(c[:40]))
print(komoran.nouns("\n".join([s for s in c[:40].split("\n") if s])))
print(okt.nouns(c[:40]))



from nltk import Text
import matplotlib.pyplot as plt

kolaw = Text(okt.nouns(c), name="kolaw")
kolaw.plot(30)
plt.show()

from wordcloud import WordCloud

# 자신의 컴퓨터 환경에 맞는 한글 폰트 경로를 설정
font_path = 'C:/Windows/Fonts/malgun.ttf'

wc = WordCloud(width = 1000, height = 600, background_color="white", font_path=font_path)
plt.imshow(wc.generate_from_frequencies(kolaw.vocab()))
plt.axis("off")
plt.show()