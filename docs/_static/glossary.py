from bs4 import BeautifulSoup
import json

# def strip_enclosing_tag(string, tag):
#     assert string.startswith('<{}'.format(tag))
#     assert string.endswith('</{}>'.format(tag))

source = 'build/html/glossary.html'
with open(source, 'r') as file:
    soup = BeautifulSoup(file.read(), features='html.parser')
    # print(soup.prettify())

glossary = {}
dt = soup.find_all(name='dt')
dd = soup.find_all(name='dd')

for key, value in zip(dt, dd):
    if 'id' in key.attrs:
        if 'term-' in key.attrs['id']: # found
            key = str(key.contents[0]).lower()
            value = str(value).replace('<dd>', '') \
                              .replace('</dd>', '') \
                              .replace('"', '&quot;') \
                              .replace('\\', '\\\\') \
                              .replace('\n', '\\n')
            glossary[key] = value

target = 'build/html/_static/glossary.json'
with open(target, 'w') as file:
    # https://stackoverflow.com/a/18283904
    file.write("glossary = '{}';".format(json.dumps(glossary)))
