<h1><center>Text preprocessing</center></h1>

## Main pipeline
* Character level:
    * Tokenization: splitting text into words
    * Splitting the text into sentences
* Word level (morphology):
    * Part-of-speech tagging
    * Resolving morphological ambiguity
    * Normalization (stemming or lemmatization)
* Sentence level (syntax):
    * Nominal and verb groups recognition
    * Semantic roles detection
    * Constituency and dependency trees
* Semantic level (semantics and discourse):
    * Coreference resolution
    * Finding synonyms and antonyms
    * Analysis of the argument-based relations

## Main problems
* Ambiguity
    * Lexical ambiguity: *орган, парить, рожки, атлас*
    * Morphological ambiguity: *Хранение денег в банке. Что делают белки в клетке?*
    * Syntactic ambiguity: *Мужу изменять нельзя. Его удивил простой солдат. Эти типы стали есть в цехе.*
* Neologisms: *печеньки, заинстаграммить, репостнуть, расшарить, биткоины*
* Different : *Россия, Российская Федерация, РФ*
* Non-standard spelling (including spelling errors and typos)

<img src="https://i.postimg.cc/pTzqjFkL/pipeline.png" alt="pipeline.png" style="width: 400px;"/>

### NLP-libraries

NLP Python libraries:
* Natural Language Toolkit (NLTK)
* Apache OpenNLP
* Stanford NLP suite
* Gate NLP library
* Spacy
* Yargy
* DeepPavlov
* CLTK (for ancient languages)
* and many others

The oldest and most famous is NLTK. NLTK does not have only various tools for text processing, but also various data — text corpora, pre-trained sentiment models and POS tagging models, stopwords lists for different languages, etc.

* [Book on NLTK](https://www.nltk.org/book/) from the authors of the library and [tutorials](https://github.com/hb20007/hands-on-nltk-tutorial) on solving NLP task using NLTK.
* [Spacy documnetation](https://spacy.io/)
* [Yargy documentation](https://yargy.readthedocs.io/)
* [DeepPavlop documentation](http://docs.deeppavlov.ai/)

## Text preprocessing

1. **Tokenization** is the very first step in text processing.
2. **Normalization** — mapping to the same lowercase form, removing punctuation, correcting typos, etc.
3.
    * **Stemming** — reducing the words to their word stem or root form. The objective of stemming is to reduce related words to the same stem even if the stem is not a dictionary word. For example, connection, connected, connecting word reduce to a common word “connect”.
    * **Lemmatization** — unlike stemming, lemmatization reduces words to their base (dictionary) form, reducing the inflected words properly and ensuring that the root word belongs to the language.
4. **Deleting stopwords** — the most common words in any language (like articles, prepositions, pronouns, conjunctions, etc) and does not add much information to the text. The list depends on the task!

**Important!** You don't always need all the stages, it all depends on the task!

## Tokenization

#### How many words are there in the following sentence?

*На дворе трава, на траве дрова, не руби дрова на траве двора.*

* 12 tokens: На, дворе, трава, на, траве, дрова, не, руби, дрова, на, траве, двора
* 8-9 word forms: На/на, дворе, трава, траве, дрова, не, руби, двора
* 6 lemmas: на, не, двор, трава, дрова, рубить


### Tokens and word forms

**Word form** – a unique word from the text

**Token** – a word form and its position in the text

The volume of the corpus is measured in tokens, the volume of the dictionary is measured in word forms or lexemes.


### Notation
$N$ = number of tokens

$V$ = dictionary (all wordforms)

$|V|$ = number of wordforms in the dictionary

### Token ≠ word


```python
# the most obvious tokenization approach: split text by space

text = '''
Продаётся LADA 4x4. ПТС 01.12.2018, куплена 20 января 19 года, 10 000 км пробега. 
Комплектация полная. Новая в салоне 750 000, отдам за 650 000. 
Возможен обмен на ВАЗ-2110 или ВАЗ 2109 с вашей доплатой. 
Краснодар, ул. Миклухо-Маклая, д. 4/5, подьезд 1 
Тел. 8(999)1234567, 8 903 987-65-43, +7 (351) 111 22 33 
И.И. Иванов (Иван Иванович) 
'''

tokens = text.split()
print(tokens)
len(tokens)
```

    ['Продаётся', 'LADA', '4x4.', 'ПТС', '01.12.2018,', 'куплена', '20', 'января', '19', 'года,', '10', '000', 'км', 'пробега.', 'Комплектация', 'полная.', 'Новая', 'в', 'салоне', '750', '000,', 'отдам', 'за', '650', '000.', 'Возможен', 'обмен', 'на', 'ВАЗ-2110', 'или', 'ВАЗ', '2109', 'с', 'вашей', 'доплатой.', 'Краснодар,', 'ул.', 'Миклухо-Маклая,', 'д.', '4/5,', 'подьезд', '1', 'Тел.', '8(999)1234567,', '8', '903', '987-65-43,', '+7', '(351)', '111', '22', '33', 'И.И.', 'Иванов', '(Иван', 'Иванович)']





    56




```python
text_en = '''
328i trim. FUEL EFFICIENT 28 MPG Hwy/18 MPG City!, 
PRICED TO MOVE $600 below Kelley Blue Book! Moonroof, Leather, Dual Zone A/C, CD Player, Rear Air, 
6-SPEED STEPTRONIC AUTOMATIC TRANSMIS... PREMIUM PKG, VALUE PKG. 
======KEY FEATURES INCLUDE: Rear Air, CD Player, Dual Zone A/C. BMW 328i with Crimson Red exterior and Oyster/Black Dakota Leather interior features a Straight 6 Cylinder Engine with 230 HP at 6500 RPM*. 
======OPTION PACKAGES: PREMIUM PKG Dakota leather seat trim, universal garage door opener, auto-dimming pwr folding exterior mirrors w/2-position memory, auto-dimming rearview mirror w/compass, pwr front seats w/4-way pwr lumbar, 2-position driver seat memory, BMW Assist, Bluetooth interface, pwr tilt/slide glass moonroof, VALUE PKG iPod & USB adapter, Dakota leather seat trim, 17' x 8.0 V-spoke alloy wheels (style 285), P225/45R17 run-flat performance tires, 6-SPEED STEPTRONIC AUTOMATIC TRANSMISSION normal, sport & manual shift modes. Remote Trunk Release, Keyless Entry, Steering Wheel Controls, Child Safety Locks, Electronic Stability Control. 
======AFFORDABILITY: This 328i is priced $600 below Kelley Blue Book. 
======OUR OFFERINGS: Here at DCH Subaru of Riverside, everything we do revolves around you. 
Our various teams are trained to address your needs from the moment you walk through the door, whether you're in the market for your next vehicle or tuning up your current one. Our Riverside showroom is the place to be if you're in the market for a new Subaru or quality pre-owned vehicle from today's top automakers. 
No matter what you are in search of, we deliver customer happiness! Pricing analysis performed on 8/22/2021. 
Horsepower calculations based on trim engine configuration. 
Fuel economy calculations based on original manufacturer data for trim engine configuration. 
Please confirm the accuracy of the included equipment by calling us prior to purchase.

DCH Subaru of Riverside
| 862 Lifetime Reviews
8043 Indiana Avenue

Riverside, California 92504

(951) 428-2314
'''

tokens = text_en.split()
print(tokens)
len(tokens)
```

    ['328i', 'trim.', 'FUEL', 'EFFICIENT', '28', 'MPG', 'Hwy/18', 'MPG', 'City!,', 'PRICED', 'TO', 'MOVE', '$600', 'below', 'Kelley', 'Blue', 'Book!', 'Moonroof,', 'Leather,', 'Dual', 'Zone', 'A/C,', 'CD', 'Player,', 'Rear', 'Air,', '6-SPEED', 'STEPTRONIC', 'AUTOMATIC', 'TRANSMIS...', 'PREMIUM', 'PKG,', 'VALUE', 'PKG.', '======KEY', 'FEATURES', 'INCLUDE:', 'Rear', 'Air,', 'CD', 'Player,', 'Dual', 'Zone', 'A/C.', 'BMW', '328i', 'with', 'Crimson', 'Red', 'exterior', 'and', 'Oyster/Black', 'Dakota', 'Leather', 'interior', 'features', 'a', 'Straight', '6', 'Cylinder', 'Engine', 'with', '230', 'HP', 'at', '6500', 'RPM*.', '======OPTION', 'PACKAGES:', 'PREMIUM', 'PKG', 'Dakota', 'leather', 'seat', 'trim,', 'universal', 'garage', 'door', 'opener,', 'auto-dimming', 'pwr', 'folding', 'exterior', 'mirrors', 'w/2-position', 'memory,', 'auto-dimming', 'rearview', 'mirror', 'w/compass,', 'pwr', 'front', 'seats', 'w/4-way', 'pwr', 'lumbar,', '2-position', 'driver', 'seat', 'memory,', 'BMW', 'Assist,', 'Bluetooth', 'interface,', 'pwr', 'tilt/slide', 'glass', 'moonroof,', 'VALUE', 'PKG', 'iPod', '&', 'USB', 'adapter,', 'Dakota', 'leather', 'seat', 'trim,', "17'", 'x', '8.0', 'V-spoke', 'alloy', 'wheels', '(style', '285),', 'P225/45R17', 'run-flat', 'performance', 'tires,', '6-SPEED', 'STEPTRONIC', 'AUTOMATIC', 'TRANSMISSION', 'normal,', 'sport', '&', 'manual', 'shift', 'modes.', 'Remote', 'Trunk', 'Release,', 'Keyless', 'Entry,', 'Steering', 'Wheel', 'Controls,', 'Child', 'Safety', 'Locks,', 'Electronic', 'Stability', 'Control.', '======AFFORDABILITY:', 'This', '328i', 'is', 'priced', '$600', 'below', 'Kelley', 'Blue', 'Book.', '======OUR', 'OFFERINGS:', 'Here', 'at', 'DCH', 'Subaru', 'of', 'Riverside,', 'everything', 'we', 'do', 'revolves', 'around', 'you.', 'Our', 'various', 'teams', 'are', 'trained', 'to', 'address', 'your', 'needs', 'from', 'the', 'moment', 'you', 'walk', 'through', 'the', 'door,', 'whether', "you're", 'in', 'the', 'market', 'for', 'your', 'next', 'vehicle', 'or', 'tuning', 'up', 'your', 'current', 'one.', 'Our', 'Riverside', 'showroom', 'is', 'the', 'place', 'to', 'be', 'if', "you're", 'in', 'the', 'market', 'for', 'a', 'new', 'Subaru', 'or', 'quality', 'pre-owned', 'vehicle', 'from', "today's", 'top', 'automakers.', 'No', 'matter', 'what', 'you', 'are', 'in', 'search', 'of,', 'we', 'deliver', 'customer', 'happiness!', 'Pricing', 'analysis', 'performed', 'on', '8/22/2021.', 'Horsepower', 'calculations', 'based', 'on', 'trim', 'engine', 'configuration.', 'Fuel', 'economy', 'calculations', 'based', 'on', 'original', 'manufacturer', 'data', 'for', 'trim', 'engine', 'configuration.', 'Please', 'confirm', 'the', 'accuracy', 'of', 'the', 'included', 'equipment', 'by', 'calling', 'us', 'prior', 'to', 'purchase.', 'DCH', 'Subaru', 'of', 'Riverside', '|', '862', 'Lifetime', 'Reviews', '8043', 'Indiana', 'Avenue', 'Riverside,', 'California', '92504', '(951)', '428-2314']





    301




```python
!pip install yargy
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting yargy
      Downloading yargy-0.15.0-py3-none-any.whl (41 kB)
    [K     |████████████████████████████████| 41 kB 109 kB/s 
    [?25hCollecting pymorphy2
      Downloading pymorphy2-0.9.1-py3-none-any.whl (55 kB)
    [K     |████████████████████████████████| 55 kB 3.5 MB/s 
    [?25hCollecting dawg-python>=0.7.1
      Downloading DAWG_Python-0.7.2-py2.py3-none-any.whl (11 kB)
    Collecting docopt>=0.6
      Downloading docopt-0.6.2.tar.gz (25 kB)
    Collecting pymorphy2-dicts-ru<3.0,>=2.4
      Downloading pymorphy2_dicts_ru-2.4.417127.4579844-py2.py3-none-any.whl (8.2 MB)
    [K     |████████████████████████████████| 8.2 MB 42.2 MB/s 
    [?25hBuilding wheels for collected packages: docopt
      Building wheel for docopt (setup.py) ... [?25l[?25hdone
      Created wheel for docopt: filename=docopt-0.6.2-py2.py3-none-any.whl size=13723 sha256=c9bf68fad8be34eea11468a596666f075ca4cf9f4a94bb8ea986fd27d895bdad
      Stored in directory: /root/.cache/pip/wheels/72/b0/3f/1d95f96ff986c7dfffe46ce2be4062f38ebd04b506c77c81b9
    Successfully built docopt
    Installing collected packages: pymorphy2-dicts-ru, docopt, dawg-python, pymorphy2, yargy
    Successfully installed dawg-python-0.7.2 docopt-0.6.2 pymorphy2-0.9.1 pymorphy2-dicts-ru-2.4.417127.4579844 yargy-0.15.0



```python
# the most detailed tokenization
from yargy.tokenizer import MorphTokenizer

tknzr = MorphTokenizer()
tokens = [_.value for _ in tknzr(text)]
print(tokens)
len(tokens)
```

    ['\n', 'Продаётся', 'LADA', '4', 'x', '4', '.', 'ПТС', '01', '.', '12', '.', '2018', ',', 'куплена', '20', 'января', '19', 'года', ',', '10', '000', 'км', 'пробега', '.', '\n', 'Комплектация', 'полная', '.', 'Новая', 'в', 'салоне', '750', '000', ',', 'отдам', 'за', '650', '000', '.', '\n', 'Возможен', 'обмен', 'на', 'ВАЗ', '-', '2110', 'или', 'ВАЗ', '2109', 'с', 'вашей', 'доплатой', '.', '\n', 'Краснодар', ',', 'ул', '.', 'Миклухо', '-', 'Маклая', ',', 'д', '.', '4', '/', '5', ',', 'подьезд', '1', '\n', 'Тел', '.', '8', '(', '999', ')', '1234567', ',', '8', '903', '987', '-', '65', '-', '43', ',', '+', '7', '(', '351', ')', '111', '22', '33', '\n', 'И', '.', 'И', '.', 'Иванов', '(', 'Иван', 'Иванович', ')', '\n']





    107




```python
!pip install --user spacy
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: spacy in /usr/local/lib/python3.7/dist-packages (3.4.1)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.7/dist-packages (from spacy) (57.4.0)
    Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.7/dist-packages (from spacy) (1.0.8)
    Requirement already satisfied: wasabi<1.1.0,>=0.9.1 in /usr/local/lib/python3.7/dist-packages (from spacy) (0.10.1)
    Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from spacy) (3.0.7)
    Requirement already satisfied: numpy>=1.15.0 in /usr/local/lib/python3.7/dist-packages (from spacy) (1.21.6)
    Requirement already satisfied: pathy>=0.3.5 in /usr/local/lib/python3.7/dist-packages (from spacy) (0.6.2)
    Requirement already satisfied: catalogue<2.1.0,>=2.0.6 in /usr/local/lib/python3.7/dist-packages (from spacy) (2.0.8)
    Requirement already satisfied: pydantic!=1.8,!=1.8.1,<1.10.0,>=1.7.4 in /usr/local/lib/python3.7/dist-packages (from spacy) (1.9.2)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.7/dist-packages (from spacy) (21.3)
    Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /usr/local/lib/python3.7/dist-packages (from spacy) (4.64.0)
    Requirement already satisfied: requests<3.0.0,>=2.13.0 in /usr/local/lib/python3.7/dist-packages (from spacy) (2.23.0)
    Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from spacy) (2.0.6)
    Requirement already satisfied: spacy-loggers<2.0.0,>=1.0.0 in /usr/local/lib/python3.7/dist-packages (from spacy) (1.0.3)
    Requirement already satisfied: langcodes<4.0.0,>=3.2.0 in /usr/local/lib/python3.7/dist-packages (from spacy) (3.3.0)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.7/dist-packages (from spacy) (2.11.3)
    Requirement already satisfied: spacy-legacy<3.1.0,>=3.0.9 in /usr/local/lib/python3.7/dist-packages (from spacy) (3.0.10)
    Requirement already satisfied: thinc<8.2.0,>=8.1.0 in /usr/local/lib/python3.7/dist-packages (from spacy) (8.1.0)
    Requirement already satisfied: typing-extensions<4.2.0,>=3.7.4 in /usr/local/lib/python3.7/dist-packages (from spacy) (4.1.1)
    Requirement already satisfied: typer<0.5.0,>=0.3.0 in /usr/local/lib/python3.7/dist-packages (from spacy) (0.4.2)
    Requirement already satisfied: srsly<3.0.0,>=2.4.3 in /usr/local/lib/python3.7/dist-packages (from spacy) (2.4.4)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from catalogue<2.1.0,>=2.0.6->spacy) (3.8.1)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging>=20.0->spacy) (3.0.9)
    Requirement already satisfied: smart-open<6.0.0,>=5.2.1 in /usr/local/lib/python3.7/dist-packages (from pathy>=0.3.5->spacy) (5.2.1)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0,>=2.13.0->spacy) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0,>=2.13.0->spacy) (2022.6.15)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0,>=2.13.0->spacy) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0,>=2.13.0->spacy) (1.24.3)
    Requirement already satisfied: blis<0.8.0,>=0.7.8 in /usr/local/lib/python3.7/dist-packages (from thinc<8.2.0,>=8.1.0->spacy) (0.7.8)
    Requirement already satisfied: click<9.0.0,>=7.1.1 in /usr/local/lib/python3.7/dist-packages (from typer<0.5.0,>=0.3.0->spacy) (7.1.2)
    Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.7/dist-packages (from jinja2->spacy) (2.0.1)



```python
import spacy
nlp = spacy.load("en_core_web_sm")
```


```python
doc = nlp(text_en)
tokens = [token.text for token in doc]
print(tokens)
len(tokens)
```

    ['\n', '328i', 'trim', '.', 'FUEL', 'EFFICIENT', '28', 'MPG', 'Hwy/18', 'MPG', 'City', '!', ',', '\n', 'PRICED', 'TO', 'MOVE', '$', '600', 'below', 'Kelley', 'Blue', 'Book', '!', 'Moonroof', ',', 'Leather', ',', 'Dual', 'Zone', 'A', '/', 'C', ',', 'CD', 'Player', ',', 'Rear', 'Air', ',', '\n', '6', '-', 'SPEED', 'STEPTRONIC', 'AUTOMATIC', 'TRANSMIS', '...', 'PREMIUM', 'PKG', ',', 'VALUE', 'PKG', '.', '\n', '=', '=', '=', '=', '=', '=', 'KEY', 'FEATURES', 'INCLUDE', ':', 'Rear', 'Air', ',', 'CD', 'Player', ',', 'Dual', 'Zone', 'A', '/', 'C.', 'BMW', '328i', 'with', 'Crimson', 'Red', 'exterior', 'and', 'Oyster', '/', 'Black', 'Dakota', 'Leather', 'interior', 'features', 'a', 'Straight', '6', 'Cylinder', 'Engine', 'with', '230', 'HP', 'at', '6500', 'RPM', '*', '.', '\n', '=', '=', '=', '=', '=', '=', 'OPTION', 'PACKAGES', ':', 'PREMIUM', 'PKG', 'Dakota', 'leather', 'seat', 'trim', ',', 'universal', 'garage', 'door', 'opener', ',', 'auto', '-', 'dimming', 'pwr', 'folding', 'exterior', 'mirrors', 'w/2', '-', 'position', 'memory', ',', 'auto', '-', 'dimming', 'rearview', 'mirror', 'w', '/', 'compass', ',', 'pwr', 'front', 'seats', 'w/4', '-', 'way', 'pwr', 'lumbar', ',', '2', '-', 'position', 'driver', 'seat', 'memory', ',', 'BMW', 'Assist', ',', 'Bluetooth', 'interface', ',', 'pwr', 'tilt', '/', 'slide', 'glass', 'moonroof', ',', 'VALUE', 'PKG', 'iPod', '&', 'USB', 'adapter', ',', 'Dakota', 'leather', 'seat', 'trim', ',', '17', "'", 'x', '8.0', 'V', '-', 'spoke', 'alloy', 'wheels', '(', 'style', '285', ')', ',', 'P225/45R17', 'run', '-', 'flat', 'performance', 'tires', ',', '6', '-', 'SPEED', 'STEPTRONIC', 'AUTOMATIC', 'TRANSMISSION', 'normal', ',', 'sport', '&', 'manual', 'shift', 'modes', '.', 'Remote', 'Trunk', 'Release', ',', 'Keyless', 'Entry', ',', 'Steering', 'Wheel', 'Controls', ',', 'Child', 'Safety', 'Locks', ',', 'Electronic', 'Stability', 'Control', '.', '\n', '=', '=', '=', '=', '=', '=', 'AFFORDABILITY', ':', 'This', '328i', 'is', 'priced', '$', '600', 'below', 'Kelley', 'Blue', 'Book', '.', '\n', '=', '=', '=', '=', '=', '=', 'OUR', 'OFFERINGS', ':', 'Here', 'at', 'DCH', 'Subaru', 'of', 'Riverside', ',', 'everything', 'we', 'do', 'revolves', 'around', 'you', '.', '\n', 'Our', 'various', 'teams', 'are', 'trained', 'to', 'address', 'your', 'needs', 'from', 'the', 'moment', 'you', 'walk', 'through', 'the', 'door', ',', 'whether', 'you', "'re", 'in', 'the', 'market', 'for', 'your', 'next', 'vehicle', 'or', 'tuning', 'up', 'your', 'current', 'one', '.', 'Our', 'Riverside', 'showroom', 'is', 'the', 'place', 'to', 'be', 'if', 'you', "'re", 'in', 'the', 'market', 'for', 'a', 'new', 'Subaru', 'or', 'quality', 'pre', '-', 'owned', 'vehicle', 'from', 'today', "'s", 'top', 'automakers', '.', '\n', 'No', 'matter', 'what', 'you', 'are', 'in', 'search', 'of', ',', 'we', 'deliver', 'customer', 'happiness', '!', 'Pricing', 'analysis', 'performed', 'on', '8/22/2021', '.', '\n', 'Horsepower', 'calculations', 'based', 'on', 'trim', 'engine', 'configuration', '.', '\n', 'Fuel', 'economy', 'calculations', 'based', 'on', 'original', 'manufacturer', 'data', 'for', 'trim', 'engine', 'configuration', '.', '\n', 'Please', 'confirm', 'the', 'accuracy', 'of', 'the', 'included', 'equipment', 'by', 'calling', 'us', 'prior', 'to', 'purchase', '.', '\n\n', 'DCH', 'Subaru', 'of', 'Riverside', '\n', '|', '862', 'Lifetime', 'Reviews', '\n', '8043', 'Indiana', 'Avenue', '\n\n', 'Riverside', ',', 'California', '92504', '\n\n', '(', '951', ')', '428', '-', '2314', '\n']





    438




```python
!pip install --user nltk
```

    Requirement already satisfied: nltk in /usr/local/lib/python3.7/dist-packages (3.2.5)
    Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from nltk) (1.15.0)



```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('snowball_data')
nltk.download('perluniprops')
nltk.download('universal_tagset')
nltk.download('stopwords')
nltk.download('nonbreaking_prefixes')
nltk.download('wordnet')
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package snowball_data to /root/nltk_data...
    [nltk_data]   Package snowball_data is already up-to-date!
    [nltk_data] Downloading package perluniprops to /root/nltk_data...
    [nltk_data]   Package perluniprops is already up-to-date!
    [nltk_data] Downloading package universal_tagset to /root/nltk_data...
    [nltk_data]   Package universal_tagset is already up-to-date!
    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package nonbreaking_prefixes to
    [nltk_data]     /root/nltk_data...
    [nltk_data]   Package nonbreaking_prefixes is already up-to-date!
    [nltk_data] Downloading package wordnet to /root/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!





    True




```python
from nltk.tokenize import word_tokenize, ToktokTokenizer

tokens = word_tokenize(text)
print(tokens)
len(tokens)
```

    ['Продаётся', 'LADA', '4x4', '.', 'ПТС', '01.12.2018', ',', 'куплена', '20', 'января', '19', 'года', ',', '10', '000', 'км', 'пробега', '.', 'Комплектация', 'полная', '.', 'Новая', 'в', 'салоне', '750', '000', ',', 'отдам', 'за', '650', '000', '.', 'Возможен', 'обмен', 'на', 'ВАЗ-2110', 'или', 'ВАЗ', '2109', 'с', 'вашей', 'доплатой', '.', 'Краснодар', ',', 'ул', '.', 'Миклухо-Маклая', ',', 'д', '.', '4/5', ',', 'подьезд', '1', 'Тел', '.', '8', '(', '999', ')', '1234567', ',', '8', '903', '987-65-43', ',', '+7', '(', '351', ')', '111', '22', '33', 'И.И', '.', 'Иванов', '(', 'Иван', 'Иванович', ')']





    81




```python
tokens = word_tokenize(text_en)
print(tokens)
len(tokens)
```

    ['328i', 'trim', '.', 'FUEL', 'EFFICIENT', '28', 'MPG', 'Hwy/18', 'MPG', 'City', '!', ',', 'PRICED', 'TO', 'MOVE', '$', '600', 'below', 'Kelley', 'Blue', 'Book', '!', 'Moonroof', ',', 'Leather', ',', 'Dual', 'Zone', 'A/C', ',', 'CD', 'Player', ',', 'Rear', 'Air', ',', '6-SPEED', 'STEPTRONIC', 'AUTOMATIC', 'TRANSMIS', '...', 'PREMIUM', 'PKG', ',', 'VALUE', 'PKG', '.', '======KEY', 'FEATURES', 'INCLUDE', ':', 'Rear', 'Air', ',', 'CD', 'Player', ',', 'Dual', 'Zone', 'A/C', '.', 'BMW', '328i', 'with', 'Crimson', 'Red', 'exterior', 'and', 'Oyster/Black', 'Dakota', 'Leather', 'interior', 'features', 'a', 'Straight', '6', 'Cylinder', 'Engine', 'with', '230', 'HP', 'at', '6500', 'RPM*', '.', '======OPTION', 'PACKAGES', ':', 'PREMIUM', 'PKG', 'Dakota', 'leather', 'seat', 'trim', ',', 'universal', 'garage', 'door', 'opener', ',', 'auto-dimming', 'pwr', 'folding', 'exterior', 'mirrors', 'w/2-position', 'memory', ',', 'auto-dimming', 'rearview', 'mirror', 'w/compass', ',', 'pwr', 'front', 'seats', 'w/4-way', 'pwr', 'lumbar', ',', '2-position', 'driver', 'seat', 'memory', ',', 'BMW', 'Assist', ',', 'Bluetooth', 'interface', ',', 'pwr', 'tilt/slide', 'glass', 'moonroof', ',', 'VALUE', 'PKG', 'iPod', '&', 'USB', 'adapter', ',', 'Dakota', 'leather', 'seat', 'trim', ',', '17', "'", 'x', '8.0', 'V-spoke', 'alloy', 'wheels', '(', 'style', '285', ')', ',', 'P225/45R17', 'run-flat', 'performance', 'tires', ',', '6-SPEED', 'STEPTRONIC', 'AUTOMATIC', 'TRANSMISSION', 'normal', ',', 'sport', '&', 'manual', 'shift', 'modes', '.', 'Remote', 'Trunk', 'Release', ',', 'Keyless', 'Entry', ',', 'Steering', 'Wheel', 'Controls', ',', 'Child', 'Safety', 'Locks', ',', 'Electronic', 'Stability', 'Control', '.', '======AFFORDABILITY', ':', 'This', '328i', 'is', 'priced', '$', '600', 'below', 'Kelley', 'Blue', 'Book', '.', '======OUR', 'OFFERINGS', ':', 'Here', 'at', 'DCH', 'Subaru', 'of', 'Riverside', ',', 'everything', 'we', 'do', 'revolves', 'around', 'you', '.', 'Our', 'various', 'teams', 'are', 'trained', 'to', 'address', 'your', 'needs', 'from', 'the', 'moment', 'you', 'walk', 'through', 'the', 'door', ',', 'whether', 'you', "'re", 'in', 'the', 'market', 'for', 'your', 'next', 'vehicle', 'or', 'tuning', 'up', 'your', 'current', 'one', '.', 'Our', 'Riverside', 'showroom', 'is', 'the', 'place', 'to', 'be', 'if', 'you', "'re", 'in', 'the', 'market', 'for', 'a', 'new', 'Subaru', 'or', 'quality', 'pre-owned', 'vehicle', 'from', 'today', "'s", 'top', 'automakers', '.', 'No', 'matter', 'what', 'you', 'are', 'in', 'search', 'of', ',', 'we', 'deliver', 'customer', 'happiness', '!', 'Pricing', 'analysis', 'performed', 'on', '8/22/2021', '.', 'Horsepower', 'calculations', 'based', 'on', 'trim', 'engine', 'configuration', '.', 'Fuel', 'economy', 'calculations', 'based', 'on', 'original', 'manufacturer', 'data', 'for', 'trim', 'engine', 'configuration', '.', 'Please', 'confirm', 'the', 'accuracy', 'of', 'the', 'included', 'equipment', 'by', 'calling', 'us', 'prior', 'to', 'purchase', '.', 'DCH', 'Subaru', 'of', 'Riverside', '|', '862', 'Lifetime', 'Reviews', '8043', 'Indiana', 'Avenue', 'Riverside', ',', 'California', '92504', '(', '951', ')', '428-2314']





    364




```python
tknzr = ToktokTokenizer()
tokens = tknzr.tokenize(text)
print(tokens)
len(tokens)
```

    ['Продаётся', 'LADA', '4x4.', 'ПТС', '01.12.2018', ',', 'куплена', '20', 'января', '19', 'года', ',', '10', '000', 'км', 'пробега.', 'Комплектация', 'полная.', 'Новая', 'в', 'салоне', '750', '000', ',', 'отдам', 'за', '650', '000.', 'Возможен', 'обмен', 'на', 'ВАЗ-2110', 'или', 'ВАЗ', '2109', 'с', 'вашей', 'доплатой.', 'Краснодар', ',', 'ул.', 'Миклухо-Маклая', ',', 'д.', '4/5', ',', 'подьезд', '1', 'Тел.', '8(', '999', ')', '1234567', ',', '8', '903', '987-65-43', ',', '+7', '(', '351', ')', '111', '22', '33', 'И.И.', 'Иванов', '(', 'Иван', 'Иванович', ')']





    71




```python
tknzr = ToktokTokenizer()
tokens = tknzr.tokenize(text_en)
print(tokens)
len(tokens)
```

    ['328i', 'trim.', 'FUEL', 'EFFICIENT', '28', 'MPG', 'Hwy/18', 'MPG', 'City', '!', ',', 'PRICED', 'TO', 'MOVE', '$', '600', 'below', 'Kelley', 'Blue', 'Book', '!', 'Moonroof', ',', 'Leather', ',', 'Dual', 'Zone', 'A/C', ',', 'CD', 'Player', ',', 'Rear', 'Air', ',', '6-SPEED', 'STEPTRONIC', 'AUTOMATIC', 'TRANSMIS', '...', 'PREMIUM', 'PKG', ',', 'VALUE', 'PKG.', '======KEY', 'FEATURES', 'INCLUDE', ':', 'Rear', 'Air', ',', 'CD', 'Player', ',', 'Dual', 'Zone', 'A/C.', 'BMW', '328i', 'with', 'Crimson', 'Red', 'exterior', 'and', 'Oyster/Black', 'Dakota', 'Leather', 'interior', 'features', 'a', 'Straight', '6', 'Cylinder', 'Engine', 'with', '230', 'HP', 'at', '6500', 'RPM*.', '======OPTION', 'PACKAGES', ':', 'PREMIUM', 'PKG', 'Dakota', 'leather', 'seat', 'trim', ',', 'universal', 'garage', 'door', 'opener', ',', 'auto-dimming', 'pwr', 'folding', 'exterior', 'mirrors', 'w/2-position', 'memory', ',', 'auto-dimming', 'rearview', 'mirror', 'w/compass', ',', 'pwr', 'front', 'seats', 'w/4-way', 'pwr', 'lumbar', ',', '2-position', 'driver', 'seat', 'memory', ',', 'BMW', 'Assist', ',', 'Bluetooth', 'interface', ',', 'pwr', 'tilt/slide', 'glass', 'moonroof', ',', 'VALUE', 'PKG', 'iPod', '&amp;', 'USB', 'adapter', ',', 'Dakota', 'leather', 'seat', 'trim', ',', '17', "'", 'x', '8.0', 'V-spoke', 'alloy', 'wheels', '(', 'style', '285', ')', ',', 'P225/45R17', 'run-flat', 'performance', 'tires', ',', '6-SPEED', 'STEPTRONIC', 'AUTOMATIC', 'TRANSMISSION', 'normal', ',', 'sport', '&amp;', 'manual', 'shift', 'modes.', 'Remote', 'Trunk', 'Release', ',', 'Keyless', 'Entry', ',', 'Steering', 'Wheel', 'Controls', ',', 'Child', 'Safety', 'Locks', ',', 'Electronic', 'Stability', 'Control.', '======AFFORDABILITY', ':', 'This', '328i', 'is', 'priced', '$', '600', 'below', 'Kelley', 'Blue', 'Book.', '======OUR', 'OFFERINGS', ':', 'Here', 'at', 'DCH', 'Subaru', 'of', 'Riverside', ',', 'everything', 'we', 'do', 'revolves', 'around', 'you.', 'Our', 'various', 'teams', 'are', 'trained', 'to', 'address', 'your', 'needs', 'from', 'the', 'moment', 'you', 'walk', 'through', 'the', 'door', ',', 'whether', 'you', "'", 're', 'in', 'the', 'market', 'for', 'your', 'next', 'vehicle', 'or', 'tuning', 'up', 'your', 'current', 'one.', 'Our', 'Riverside', 'showroom', 'is', 'the', 'place', 'to', 'be', 'if', 'you', "'", 're', 'in', 'the', 'market', 'for', 'a', 'new', 'Subaru', 'or', 'quality', 'pre-owned', 'vehicle', 'from', 'today', "'", 's', 'top', 'automakers.', 'No', 'matter', 'what', 'you', 'are', 'in', 'search', 'of', ',', 'we', 'deliver', 'customer', 'happiness', '!', 'Pricing', 'analysis', 'performed', 'on', '8/22/2021.', 'Horsepower', 'calculations', 'based', 'on', 'trim', 'engine', 'configuration.', 'Fuel', 'economy', 'calculations', 'based', 'on', 'original', 'manufacturer', 'data', 'for', 'trim', 'engine', 'configuration.', 'Please', 'confirm', 'the', 'accuracy', 'of', 'the', 'included', 'equipment', 'by', 'calling', 'us', 'prior', 'to', 'purchase.', 'DCH', 'Subaru', 'of', 'Riverside', '&#124;', '862', 'Lifetime', 'Reviews', '8043', 'Indiana', 'Avenue', 'Riverside', ',', 'California', '92504', '(', '951', ')', '428-2314']





    353




```python
# tokenizer that specifies on tweets
from nltk.tokenize import TweetTokenizer

tknzr = TweetTokenizer()
tweet = "@remy This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
tknzr.tokenize(tweet)
```




    ['@remy',
     'This',
     'is',
     'a',
     'cooool',
     '#dummysmiley',
     ':',
     ':-)',
     ':-P',
     '<3',
     'and',
     'some',
     'arrows',
     '<',
     '>',
     '->',
     '<--']




```python
word_tokenize(tweet)
```




    ['@',
     'remy',
     'This',
     'is',
     'a',
     'cooool',
     '#',
     'dummysmiley',
     ':',
     ':',
     '-',
     ')',
     ':',
     '-P',
     '<',
     '3',
     'and',
     'some',
     'arrows',
     '<',
     '>',
     '-',
     '>',
     '<',
     '--']




```python
# tokenizer based on regexp
from nltk.tokenize import RegexpTokenizer

s = "Good muffins cost $3.88 in New York.  Please buy me two of them. \n\nThanks."
tknzr = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
tknzr.tokenize(s)
```




    ['Good',
     'muffins',
     'cost',
     '$3.88',
     'in',
     'New',
     'York',
     '.',
     'Please',
     'buy',
     'me',
     'two',
     'of',
     'them',
     '.',
     'Thanks',
     '.']




```python

```

[A-zА-я0-9-] = \w

\S+

Actually, `nltk` possesses a wide range of tokenizers:


```python
from nltk import tokenize
dir(tokenize)[:16]
```




    ['BlanklineTokenizer',
     'LineTokenizer',
     'MWETokenizer',
     'PunktSentenceTokenizer',
     'RegexpTokenizer',
     'ReppTokenizer',
     'SExprTokenizer',
     'SpaceTokenizer',
     'StanfordSegmenter',
     'TabTokenizer',
     'TextTilingTokenizer',
     'ToktokTokenizer',
     'TreebankWordTokenizer',
     'TweetTokenizer',
     'WhitespaceTokenizer',
     'WordPunctTokenizer']



They are able to provide the beginning and the end indices of each token:


```python
wh_tok = tokenize.WhitespaceTokenizer()
list(wh_tok.span_tokenize("don't stop me"))
```




    [(0, 5), (6, 10), (11, 13)]



Some tokenizers are very specific:


```python
tokenize.TreebankWordTokenizer().tokenize("don't stop me")
```




    ['do', "n't", 'stop', 'me']



## Sentence splitting

Sentence splitting mostly relies on punctuation marks. "? ", "! "are usually unambiguous, thea problems arise with ". " Possible solution: develop a binary classifier for sentence segmentation. For each dot "." task is to fefine whether it is the end of the sentence or not.


```python
from nltk.tokenize import sent_tokenize

sents = sent_tokenize(text)
print(len(sents))
sents
```

    10





    ['\nПродаётся LADA 4x4.',
     'ПТС 01.12.2018, куплена 20 января 19 года, 10 000 км пробега.',
     'Комплектация полная.',
     'Новая в салоне 750 000, отдам за 650 000.',
     'Возможен обмен на ВАЗ-2110 или ВАЗ 2109 с вашей доплатой.',
     'Краснодар, ул.',
     'Миклухо-Маклая, д.',
     '4/5, подьезд 1 \nТел.',
     '8(999)1234567, 8 903 987-65-43, +7 (351) 111 22 33 \nИ.И.',
     'Иванов (Иван Иванович)']




```python
sents = sent_tokenize(text_en)
print(len(sents))
sents
```

    18





    ['\n328i trim.',
     'FUEL EFFICIENT 28 MPG Hwy/18 MPG City!, \nPRICED TO MOVE $600 below Kelley Blue Book!',
     'Moonroof, Leather, Dual Zone A/C, CD Player, Rear Air, \n6-SPEED STEPTRONIC AUTOMATIC TRANSMIS...',
     'PREMIUM PKG, VALUE PKG.',
     '======KEY FEATURES INCLUDE: Rear Air, CD Player, Dual Zone A/C.',
     'BMW 328i with Crimson Red exterior and Oyster/Black Dakota Leather interior features a Straight 6 Cylinder Engine with 230 HP at 6500 RPM*.',
     "======OPTION PACKAGES: PREMIUM PKG Dakota leather seat trim, universal garage door opener, auto-dimming pwr folding exterior mirrors w/2-position memory, auto-dimming rearview mirror w/compass, pwr front seats w/4-way pwr lumbar, 2-position driver seat memory, BMW Assist, Bluetooth interface, pwr tilt/slide glass moonroof, VALUE PKG iPod & USB adapter, Dakota leather seat trim, 17' x 8.0 V-spoke alloy wheels (style 285), P225/45R17 run-flat performance tires, 6-SPEED STEPTRONIC AUTOMATIC TRANSMISSION normal, sport & manual shift modes.",
     'Remote Trunk Release, Keyless Entry, Steering Wheel Controls, Child Safety Locks, Electronic Stability Control.',
     '======AFFORDABILITY: This 328i is priced $600 below Kelley Blue Book.',
     '======OUR OFFERINGS: Here at DCH Subaru of Riverside, everything we do revolves around you.',
     "Our various teams are trained to address your needs from the moment you walk through the door, whether you're in the market for your next vehicle or tuning up your current one.",
     "Our Riverside showroom is the place to be if you're in the market for a new Subaru or quality pre-owned vehicle from today's top automakers.",
     'No matter what you are in search of, we deliver customer happiness!',
     'Pricing analysis performed on 8/22/2021.',
     'Horsepower calculations based on trim engine configuration.',
     'Fuel economy calculations based on original manufacturer data for trim engine configuration.',
     'Please confirm the accuracy of the included equipment by calling us prior to purchase.',
     'DCH Subaru of Riverside\n| 862 Lifetime Reviews\n8043 Indiana Avenue\n\nRiverside, California 92504\n\n(951) 428-2314']




```python
!pip install rusenttokenize
```

    Requirement already satisfied: rusenttokenize in /root/.local/lib/python3.7/site-packages (0.0.5)



```python
from ru_sent_tokenize import ru_sent_tokenize
sents = ru_sent_tokenize(text)

print(len(sents))
sents
```


```python

```

## Normalization

### Delete punctuation


```python
text
```




    '\nПродаётся LADA 4x4. ПТС 01.12.2018, куплена 20 января 19 года, 10 000 км пробега. \nКомплектация полная. Новая в салоне 750 000, отдам за 650 000. \nВозможен обмен на ВАЗ-2110 или ВАЗ 2109 с вашей доплатой. \nКраснодар, ул. Миклухо-Маклая, д. 4/5, подьезд 1 \nТел. 8(999)1234567, 8 903 987-65-43, +7 (351) 111 22 33 \nИ.И. Иванов (Иван Иванович) \n'




```python
# First option
import re

# set of punctuation marks which depends on the task and texts
punct = '[!"#$%&()*\+,-\./:;<=>?@\[\]^_`{|}~„“«»†*\—/\-‘’]'
clean_text = re.sub(punct, r' ', text)
print(clean_text.split())

# Another point
clean_words = [w.strip(punct) for w in word_tokenize(text)]
print(clean_words)

clean_words == clean_text
```

    ['Продаётся', 'LADA', '4x4', 'ПТС', '01', '12', '2018', 'куплена', '20', 'января', '19', 'года', '10', '000', 'км', 'пробега', 'Комплектация', 'полная', 'Новая', 'в', 'салоне', '750', '000', 'отдам', 'за', '650', '000', 'Возможен', 'обмен', 'на', 'ВАЗ', '2110', 'или', 'ВАЗ', '2109', 'с', 'вашей', 'доплатой', 'Краснодар', 'ул', 'Миклухо', 'Маклая', 'д', '4', '5', 'подьезд', '1', 'Тел', '8', '999', '1234567', '8', '903', '987', '65', '43', '7', '351', '111', '22', '33', 'И', 'И', 'Иванов', 'Иван', 'Иванович']
    ['Продаётся', 'LADA', '4x4', '', 'ПТС', '01.12.2018', '', 'куплена', '20', 'января', '19', 'года', '', '10', '000', 'км', 'пробега', '', 'Комплектация', 'полная', '', 'Новая', 'в', 'салоне', '750', '000', '', 'отдам', 'за', '650', '000', '', 'Возможен', 'обмен', 'на', 'ВАЗ-2110', 'или', 'ВАЗ', '2109', 'с', 'вашей', 'доплатой', '', 'Краснодар', '', 'ул', '', 'Миклухо-Маклая', '', 'д', '', '4/5', '', 'подьезд', '1', 'Тел', '', '8', '', '999', '', '1234567', '', '8', '903', '987-65-43', '', '7', '', '351', '', '111', '22', '33', 'И.И', '', 'Иванов', '', 'Иван', 'Иванович', '']





    False




```python
'____jlgherg.'.strip(punctuation)
```




    'jlgherg'




```python
from string import punctuation
```


```python
punctuation
```




    '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'




```python
clean_words_en = [w.strip(punctuation) for w in word_tokenize(text_en)]
print(clean_words_en)
```

    ['328i', 'trim', '', 'FUEL', 'EFFICIENT', '28', 'MPG', 'Hwy/18', 'MPG', 'City', '', '', 'PRICED', 'TO', 'MOVE', '', '600', 'below', 'Kelley', 'Blue', 'Book', '', 'Moonroof', '', 'Leather', '', 'Dual', 'Zone', 'A/C', '', 'CD', 'Player', '', 'Rear', 'Air', '', '6-SPEED', 'STEPTRONIC', 'AUTOMATIC', 'TRANSMIS', '', 'PREMIUM', 'PKG', '', 'VALUE', 'PKG', '', 'KEY', 'FEATURES', 'INCLUDE', '', 'Rear', 'Air', '', 'CD', 'Player', '', 'Dual', 'Zone', 'A/C', '', 'BMW', '328i', 'with', 'Crimson', 'Red', 'exterior', 'and', 'Oyster/Black', 'Dakota', 'Leather', 'interior', 'features', 'a', 'Straight', '6', 'Cylinder', 'Engine', 'with', '230', 'HP', 'at', '6500', 'RPM', '', 'OPTION', 'PACKAGES', '', 'PREMIUM', 'PKG', 'Dakota', 'leather', 'seat', 'trim', '', 'universal', 'garage', 'door', 'opener', '', 'auto-dimming', 'pwr', 'folding', 'exterior', 'mirrors', 'w/2-position', 'memory', '', 'auto-dimming', 'rearview', 'mirror', 'w/compass', '', 'pwr', 'front', 'seats', 'w/4-way', 'pwr', 'lumbar', '', '2-position', 'driver', 'seat', 'memory', '', 'BMW', 'Assist', '', 'Bluetooth', 'interface', '', 'pwr', 'tilt/slide', 'glass', 'moonroof', '', 'VALUE', 'PKG', 'iPod', '', 'USB', 'adapter', '', 'Dakota', 'leather', 'seat', 'trim', '', '17', '', 'x', '8.0', 'V-spoke', 'alloy', 'wheels', '', 'style', '285', '', '', 'P225/45R17', 'run-flat', 'performance', 'tires', '', '6-SPEED', 'STEPTRONIC', 'AUTOMATIC', 'TRANSMISSION', 'normal', '', 'sport', '', 'manual', 'shift', 'modes', '', 'Remote', 'Trunk', 'Release', '', 'Keyless', 'Entry', '', 'Steering', 'Wheel', 'Controls', '', 'Child', 'Safety', 'Locks', '', 'Electronic', 'Stability', 'Control', '', 'AFFORDABILITY', '', 'This', '328i', 'is', 'priced', '', '600', 'below', 'Kelley', 'Blue', 'Book', '', 'OUR', 'OFFERINGS', '', 'Here', 'at', 'DCH', 'Subaru', 'of', 'Riverside', '', 'everything', 'we', 'do', 'revolves', 'around', 'you', '', 'Our', 'various', 'teams', 'are', 'trained', 'to', 'address', 'your', 'needs', 'from', 'the', 'moment', 'you', 'walk', 'through', 'the', 'door', '', 'whether', 'you', 're', 'in', 'the', 'market', 'for', 'your', 'next', 'vehicle', 'or', 'tuning', 'up', 'your', 'current', 'one', '', 'Our', 'Riverside', 'showroom', 'is', 'the', 'place', 'to', 'be', 'if', 'you', 're', 'in', 'the', 'market', 'for', 'a', 'new', 'Subaru', 'or', 'quality', 'pre-owned', 'vehicle', 'from', 'today', 's', 'top', 'automakers', '', 'No', 'matter', 'what', 'you', 'are', 'in', 'search', 'of', '', 'we', 'deliver', 'customer', 'happiness', '', 'Pricing', 'analysis', 'performed', 'on', '8/22/2021', '', 'Horsepower', 'calculations', 'based', 'on', 'trim', 'engine', 'configuration', '', 'Fuel', 'economy', 'calculations', 'based', 'on', 'original', 'manufacturer', 'data', 'for', 'trim', 'engine', 'configuration', '', 'Please', 'confirm', 'the', 'accuracy', 'of', 'the', 'included', 'equipment', 'by', 'calling', 'us', 'prior', 'to', 'purchase', '', 'DCH', 'Subaru', 'of', 'Riverside', '', '862', 'Lifetime', 'Reviews', '8043', 'Indiana', 'Avenue', 'Riverside', '', 'California', '92504', '', '951', '', '428-2314']


### Convert to lower/upper case


```python
clean_words = [w.lower() for w in clean_words if w != '']
print(clean_words)
```

    ['продаётся', 'lada', '4x4', 'птс', '01.12.2018', 'куплена', '20', 'января', '19', 'года', '10', '000', 'км', 'пробега', 'комплектация', 'полная', 'новая', 'в', 'салоне', '750', '000', 'отдам', 'за', '650', '000', 'возможен', 'обмен', 'на', 'ваз-2110', 'или', 'ваз', '2109', 'с', 'вашей', 'доплатой', 'краснодар', 'ул', 'миклухо-маклая', 'д', '4/5', 'подьезд', '1', 'тел', '8', '999', '1234567', '8', '903', '987-65-43', '7', '351', '111', '22', '33', 'и.и', 'иванов', 'иван', 'иванович']



```python
clean_words_en = [w.lower() for w in clean_words_en if w != '']
print(clean_words_en)
```

    ['328i', 'trim', 'fuel', 'efficient', '28', 'mpg', 'hwy/18', 'mpg', 'city', 'priced', 'to', 'move', '600', 'below', 'kelley', 'blue', 'book', 'moonroof', 'leather', 'dual', 'zone', 'a/c', 'cd', 'player', 'rear', 'air', '6-speed', 'steptronic', 'automatic', 'transmis', 'premium', 'pkg', 'value', 'pkg', 'key', 'features', 'include', 'rear', 'air', 'cd', 'player', 'dual', 'zone', 'a/c', 'bmw', '328i', 'with', 'crimson', 'red', 'exterior', 'and', 'oyster/black', 'dakota', 'leather', 'interior', 'features', 'a', 'straight', '6', 'cylinder', 'engine', 'with', '230', 'hp', 'at', '6500', 'rpm', 'option', 'packages', 'premium', 'pkg', 'dakota', 'leather', 'seat', 'trim', 'universal', 'garage', 'door', 'opener', 'auto-dimming', 'pwr', 'folding', 'exterior', 'mirrors', 'w/2-position', 'memory', 'auto-dimming', 'rearview', 'mirror', 'w/compass', 'pwr', 'front', 'seats', 'w/4-way', 'pwr', 'lumbar', '2-position', 'driver', 'seat', 'memory', 'bmw', 'assist', 'bluetooth', 'interface', 'pwr', 'tilt/slide', 'glass', 'moonroof', 'value', 'pkg', 'ipod', 'usb', 'adapter', 'dakota', 'leather', 'seat', 'trim', '17', 'x', '8.0', 'v-spoke', 'alloy', 'wheels', 'style', '285', 'p225/45r17', 'run-flat', 'performance', 'tires', '6-speed', 'steptronic', 'automatic', 'transmission', 'normal', 'sport', 'manual', 'shift', 'modes', 'remote', 'trunk', 'release', 'keyless', 'entry', 'steering', 'wheel', 'controls', 'child', 'safety', 'locks', 'electronic', 'stability', 'control', 'affordability', 'this', '328i', 'is', 'priced', '600', 'below', 'kelley', 'blue', 'book', 'our', 'offerings', 'here', 'at', 'dch', 'subaru', 'of', 'riverside', 'everything', 'we', 'do', 'revolves', 'around', 'you', 'our', 'various', 'teams', 'are', 'trained', 'to', 'address', 'your', 'needs', 'from', 'the', 'moment', 'you', 'walk', 'through', 'the', 'door', 'whether', 'you', 're', 'in', 'the', 'market', 'for', 'your', 'next', 'vehicle', 'or', 'tuning', 'up', 'your', 'current', 'one', 'our', 'riverside', 'showroom', 'is', 'the', 'place', 'to', 'be', 'if', 'you', 're', 'in', 'the', 'market', 'for', 'a', 'new', 'subaru', 'or', 'quality', 'pre-owned', 'vehicle', 'from', 'today', 's', 'top', 'automakers', 'no', 'matter', 'what', 'you', 'are', 'in', 'search', 'of', 'we', 'deliver', 'customer', 'happiness', 'pricing', 'analysis', 'performed', 'on', '8/22/2021', 'horsepower', 'calculations', 'based', 'on', 'trim', 'engine', 'configuration', 'fuel', 'economy', 'calculations', 'based', 'on', 'original', 'manufacturer', 'data', 'for', 'trim', 'engine', 'configuration', 'please', 'confirm', 'the', 'accuracy', 'of', 'the', 'included', 'equipment', 'by', 'calling', 'us', 'prior', 'to', 'purchase', 'dch', 'subaru', 'of', 'riverside', '862', 'lifetime', 'reviews', '8043', 'indiana', 'avenue', 'riverside', 'california', '92504', '951', '428-2314']



```python
# how to convert to upper case?
# your code here for text_en
```

### Stopwords

**Stop words** — are the most common words in any natural language. For the purpose of analyzing text data and building NLP models, these stopwords might not add much value to the meaning of the document. They make up the top of the frequency list in any language. The set of stop words is not universal, it will depend on your task!

NLTK has ready-made stop word lists for many languages.


```python
from nltk.corpus import stopwords

# language list
stopwords.fileids()
```




    ['arabic',
     'azerbaijani',
     'bengali',
     'danish',
     'dutch',
     'english',
     'finnish',
     'french',
     'german',
     'greek',
     'hungarian',
     'indonesian',
     'italian',
     'kazakh',
     'nepali',
     'norwegian',
     'portuguese',
     'romanian',
     'russian',
     'slovene',
     'spanish',
     'swedish',
     'tajik',
     'turkish']




```python
sw = stopwords.words('russian')
print(sw)
```

    ['и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между']



```python
print([w if w not in sw else print(w) for w in clean_words])
```

    в
    за
    на
    или
    с
    ['продаётся', 'lada', '4x4', 'птс', '01.12.2018', 'куплена', '20', 'января', '19', 'года', '10', '000', 'км', 'пробега', 'комплектация', 'полная', 'новая', None, 'салоне', '750', '000', 'отдам', None, '650', '000', 'возможен', 'обмен', None, 'ваз-2110', None, 'ваз', '2109', None, 'вашей', 'доплатой', 'краснодар', 'ул', 'миклухо-маклая', 'д', '4/5', 'подьезд', '1', 'тел', '8', '999', '1234567', '8', '903', '987-65-43', '7', '351', '111', '22', '33', 'и.и', 'иванов', 'иван', 'иванович']



```python
sw = stopwords.words('english')
print(sw)
```

    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]



```python
nltk_clean_words_en = [w for w in clean_words_en if w not in sw]
print(nltk_clean_words_en)
```

    ['328i', 'trim', 'fuel', 'efficient', '28', 'mpg', 'hwy/18', 'mpg', 'city', 'priced', 'move', '600', 'kelley', 'blue', 'book', 'moonroof', 'leather', 'dual', 'zone', 'a/c', 'cd', 'player', 'rear', 'air', '6-speed', 'steptronic', 'automatic', 'transmis', 'premium', 'pkg', 'value', 'pkg', 'key', 'features', 'include', 'rear', 'air', 'cd', 'player', 'dual', 'zone', 'a/c', 'bmw', '328i', 'crimson', 'red', 'exterior', 'oyster/black', 'dakota', 'leather', 'interior', 'features', 'straight', '6', 'cylinder', 'engine', '230', 'hp', '6500', 'rpm', 'option', 'packages', 'premium', 'pkg', 'dakota', 'leather', 'seat', 'trim', 'universal', 'garage', 'door', 'opener', 'auto-dimming', 'pwr', 'folding', 'exterior', 'mirrors', 'w/2-position', 'memory', 'auto-dimming', 'rearview', 'mirror', 'w/compass', 'pwr', 'front', 'seats', 'w/4-way', 'pwr', 'lumbar', '2-position', 'driver', 'seat', 'memory', 'bmw', 'assist', 'bluetooth', 'interface', 'pwr', 'tilt/slide', 'glass', 'moonroof', 'value', 'pkg', 'ipod', 'usb', 'adapter', 'dakota', 'leather', 'seat', 'trim', '17', 'x', '8.0', 'v-spoke', 'alloy', 'wheels', 'style', '285', 'p225/45r17', 'run-flat', 'performance', 'tires', '6-speed', 'steptronic', 'automatic', 'transmission', 'normal', 'sport', 'manual', 'shift', 'modes', 'remote', 'trunk', 'release', 'keyless', 'entry', 'steering', 'wheel', 'controls', 'child', 'safety', 'locks', 'electronic', 'stability', 'control', 'affordability', '328i', 'priced', '600', 'kelley', 'blue', 'book', 'offerings', 'dch', 'subaru', 'riverside', 'everything', 'revolves', 'around', 'various', 'teams', 'trained', 'address', 'needs', 'moment', 'walk', 'door', 'whether', 'market', 'next', 'vehicle', 'tuning', 'current', 'one', 'riverside', 'showroom', 'place', 'market', 'new', 'subaru', 'quality', 'pre-owned', 'vehicle', 'today', 'top', 'automakers', 'matter', 'search', 'deliver', 'customer', 'happiness', 'pricing', 'analysis', 'performed', '8/22/2021', 'horsepower', 'calculations', 'based', 'trim', 'engine', 'configuration', 'fuel', 'economy', 'calculations', 'based', 'original', 'manufacturer', 'data', 'trim', 'engine', 'configuration', 'please', 'confirm', 'accuracy', 'included', 'equipment', 'calling', 'us', 'prior', 'purchase', 'dch', 'subaru', 'riverside', '862', 'lifetime', 'reviews', '8043', 'indiana', 'avenue', 'riverside', 'california', '92504', '951', '428-2314']



```python
clean_text_en = re.sub(punct, r'', text_en.lower())

spacy_clean_words_en = [token.text for token in nlp(clean_text_en) if not token.is_stop and token.text.isalnum()]
print(spacy_clean_words_en)
```

    ['328i', 'trim', 'fuel', 'efficient', '28', 'mpg', 'hwy18', 'mpg', 'city', 'priced', '600', 'kelley', 'blue', 'book', 'moonroof', 'leather', 'dual', 'zone', 'ac', 'cd', 'player', 'rear', 'air', '6speed', 'steptronic', 'automatic', 'transmis', 'premium', 'pkg', 'value', 'pkg', 'key', 'features', 'include', 'rear', 'air', 'cd', 'player', 'dual', 'zone', 'ac', 'bmw', '328i', 'crimson', 'red', 'exterior', 'oysterblack', 'dakota', 'leather', 'interior', 'features', 'straight', '6', 'cylinder', 'engine', '230', 'hp', '6500', 'rpm', 'option', 'packages', 'premium', 'pkg', 'dakota', 'leather', 'seat', 'trim', 'universal', 'garage', 'door', 'opener', 'autodimming', 'pwr', 'folding', 'exterior', 'mirrors', 'w2position', 'memory', 'autodimming', 'rearview', 'mirror', 'wcompass', 'pwr', 'seats', 'w4way', 'pwr', 'lumbar', '2position', 'driver', 'seat', 'memory', 'bmw', 'assist', 'bluetooth', 'interface', 'pwr', 'tiltslide', 'glass', 'moonroof', 'value', 'pkg', 'ipod', 'usb', 'adapter', 'dakota', 'leather', 'seat', 'trim', '17', 'x', '80', 'vspoke', 'alloy', 'wheels', 'style', '285', 'p22545r17', 'runflat', 'performance', 'tires', '6speed', 'steptronic', 'automatic', 'transmission', 'normal', 'sport', 'manual', 'shift', 'modes', 'remote', 'trunk', 'release', 'keyless', 'entry', 'steering', 'wheel', 'controls', 'child', 'safety', 'locks', 'electronic', 'stability', 'control', 'affordability', '328i', 'priced', '600', 'kelley', 'blue', 'book', 'offerings', 'dch', 'subaru', 'riverside', 'revolves', 'teams', 'trained', 'address', 'needs', 'moment', 'walk', 'door', 'market', 'vehicle', 'tuning', 'current', 'riverside', 'showroom', 'place', 'market', 'new', 'subaru', 'quality', 'preowned', 'vehicle', 'today', 'automakers', 'matter', 'search', 'deliver', 'customer', 'happiness', 'pricing', 'analysis', 'performed', '8222021', 'horsepower', 'calculations', 'based', 'trim', 'engine', 'configuration', 'fuel', 'economy', 'calculations', 'based', 'original', 'manufacturer', 'data', 'trim', 'engine', 'configuration', 'confirm', 'accuracy', 'included', 'equipment', 'calling', 'prior', 'purchase', 'dch', 'subaru', 'riverside', '862', 'lifetime', 'reviews', '8043', 'indiana', 'avenue', 'riverside', 'california', '92504', '951', '4282314']



```python
len(nltk_clean_words_en)==len(spacy_clean_words_en), len(nltk_clean_words_en), len(spacy_clean_words_en)
```




    (False, 234, 223)



## Stemming

**Stemming** reduces words to their stem (root) by removing endings and suffixes. The remaining part is called stem. However, it should not necessarily match with the morphological basis of the word. The problem of stemming is that the same stems may  be obtained from the words with different roots and vice versa.

* 1st type of error: белый, белка, белье $\implies$  бел

* 2nd type of error: трудность, трудный $\implies$  трудност, труд

* 3rd type of error: быстрый, быстрее $\implies$  быст, побыстрее $\implies$  побыст

The simplest algorithm is the Porter algorithm. It consists of 5 loops with commands, on each loop there is an operation to delete / replace the suffix. Probabilistic extensions of the algorithm are possible.

### Snowball stemmer
An improved version of Porter's stemmer; unlike Porter's stemmer, it can work with multiple languages.


```python
from nltk.stem.snowball import SnowballStemmer

SnowballStemmer.languages  
```




    ('arabic',
     'danish',
     'dutch',
     'english',
     'finnish',
     'french',
     'german',
     'hungarian',
     'italian',
     'norwegian',
     'porter',
     'portuguese',
     'romanian',
     'russian',
     'spanish',
     'swedish')




```python
poem = '''
По морям, играя, носится
с миноносцем миноносица.
Льнет, как будто к меду осочка,
к миноносцу миноносочка.
И конца б не довелось ему,
благодушью миноносьему.
Вдруг прожектор, вздев на нос очки,
впился в спину миноносочки.
Как взревет медноголосина:
Р-р-р-астакая миноносина!
'''

words = [w.strip(punct).lower() for w in word_tokenize(poem)]
words = [w for w in words if w not in sw and w != '']
```


```python
snowball = SnowballStemmer("russian")

for w in words:
    print("%s: %s" % (w, snowball.stem(w)))
```

    по: по
    морям: мор
    играя: игр
    носится: нос
    с: с
    миноносцем: миноносц
    миноносица: миноносиц
    льнет: льнет
    как: как
    будто: будт
    к: к
    меду: мед
    осочка: осочк
    к: к
    миноносцу: миноносц
    миноносочка: миноносочк
    и: и
    конца: конц
    б: б
    не: не
    довелось: довел
    ему: ем
    благодушью: благодуш
    миноносьему: минонос
    вдруг: вдруг
    прожектор: прожектор
    вздев: вздев
    на: на
    нос: нос
    очки: очк
    впился: впил
    в: в
    спину: спин
    миноносочки: миноносочк
    как: как
    взревет: взревет
    медноголосина: медноголосин
    р-р-р-астакая: р-р-р-астак
    миноносина: миноносин



```python
snowball = SnowballStemmer("english")
```


```python
poem_en = '''
Twinkle, twinkle, little star,
How I wonder what you are.
Up above the world so high,
Like a diamond in the sky.
Twinkle, twinkle, little star,
How I wonder what you are!

When the blazing sun is gone,
When he nothing shines upon,
Then you show your little light,
Twinkle, twinkle, all the night.
Twinkle, twinkle, little star,
How I wonder what you are!
'''

words = [w.strip(punct).lower() for w in word_tokenize(poem_en)]
words = [w for w in words if w not in sw and w != '']
for w in words:
    print("%s: %s" % (w, snowball.stem(w)))
```

    twinkle: twinkl
    twinkle: twinkl
    little: littl
    star: star
    wonder: wonder
    world: world
    high: high
    like: like
    diamond: diamond
    sky: sky
    twinkle: twinkl
    twinkle: twinkl
    little: littl
    star: star
    wonder: wonder
    blazing: blaze
    sun: sun
    gone: gone
    nothing: noth
    shines: shine
    upon: upon
    show: show
    little: littl
    light: light
    twinkle: twinkl
    twinkle: twinkl
    night: night
    twinkle: twinkl
    twinkle: twinkl
    little: littl
    star: star
    wonder: wonder


## Morphological analysis

The are the following tasks of morphological analysis:

* Morphological segmentation — converting word to its normal form (lemma), extracting the basis (stem) and grammatical characteristics of the word
* Word form generation — word form generation according to its lemma and given grammatical characteristics

Morphological analysis is not exactly the strong suit of NLTK. You should better use `Spacy` for european languages and `pymorphy2` and `pymystem3` for Russian.

## Lemmatization

**Lemmatization** — the process of converting a word form to a lemma, (a normal, dictionary form). This is a more complex task than stemming, but it also gives much more meaningful results, especially for languages with rich morphology.

* кошке, кошку, кошкам, кошкой $\implies$ кошка
* бежал, бежит, бегу $\implies$  бежать
* белому, белым, белыми $\implies$ белый

## POS-tagging

**Частеречная разметка**, или **POS-tagging** _(part of speech tagging)_ —  the process of marking up a word in a text (corpus) as corresponding to a particular part of speech (tags), based on both its definition and its context.

Для большинства слов возможно несколько разборов (т.е. несколько разных лемм, несколько разных частей речи и т.п.). Теггер генерирует  все варианты, ранжирует их по вероятности и по умолчанию выдает наиболее вероятный. Выбор одного разбора из нескольких называется **снятием омонимии**, или **дизамбигуацией**.

Part-of-speech tagging is harder than just having a list of words and their parts of speech, because some word forms can represent more than one part of speech at different times. It means that in natural languages a large percentage of word forms are ambiguous. For example, even "dogs", which is usually thought of as just a plural noun, can also be a verb:

    He reads books \<plural noun\>
    He books \<3rd person singular verb\> tickets.

Correct grammatical tagging which chooses one of possible options is called **morphological disambiguation**.

### Tag sets

There are many sets of grammatical tags, or tagsets:
* НКРЯ
* Mystem
* UPenn
* OpenCorpora (is used in pymorphy2)
* Universal Dependencies
* ...

There is a [library](https://github.com/kmike/russian-tagsets) to convert tags from one system to another for the Russian language, `russian-tagsets`. However, you may loose important information when converting POS-tags from one system to another.

At the moment, the standard is **Universal Dependencies**. You can read more about the project [here](http://universaldependencies.org/), and [here](http://universaldependencies.org/u/pos/) about tags. Here below you can see the list of the main UD tags:

* ADJ: adjective
* ADP: adposition
* ADV: adverb
* AUX: auxiliary
* CCONJ: coordinating conjunction
* DET: determiner
* INTJ: interjection
* NOUN: noun
* NUM: numeral
* PART: particle
* PRON: pronoun
* PROPN: proper noun
* PUNCT: punctuation
* SCONJ: subordinating conjunction
* SYM: symbol
* VERB: verb
* X: other

### pymystem3

**pymystem3** — a wrapper for "an excellent morphological analyzer" for Russian language Yandex Mystem 3.1 released in June 2014. You can download it separately and use it from the console. An outstanding advantage of Mystem is that the system relies on the word context which is quite helpful when resolving ambiguity. 

* [Mystem documentation](https://tech.yandex.ru/mystem/doc/index-docpage/)
* [pymystem3 documentation](http://pythonhosted.org/pymystem3/)

Initialize Mystem with the default parameters. BTW, parameters are as follows:
* mystem_bin - path to `mystem`, if if there are several of them
* grammar_info - analyze grammatical information or lemmas only (grammatical information is included by default)
* disambiguation - morphological disambiguation (included by default)
* entire_input - save all characters from the input to the output (e.g. different types of space characters, \everything is included by default)

Mystem methods accept string as input and tokenizes it inside. You may provide tokens separately, however, then the context would not be taken into consideration.


```python
! pip install --user pymystem3
```

    Requirement already satisfied: pymystem3 in /usr/local/lib/python3.7/dist-packages (0.2.0)
    Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from pymystem3) (2.23.0)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->pymystem3) (2.10)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->pymystem3) (1.24.3)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->pymystem3) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->pymystem3) (2021.5.30)



```python
from pymystem3 import Mystem

m = Mystem()
lemmas = m.lemmatize(' '.join(words))
print(lemmas)
```


```python
from pymystem3 import Mystem

m = Mystem()
lemmas = m.stem(' '.join(words))
print(lemmas)
```


```python
parsed = m.analyze(poem)
parsed[:10]
```




    [{'text': '\n'},
     {'analysis': [{'gr': 'PR=', 'lex': 'по'}], 'text': 'По'},
     {'text': ' '},
     {'analysis': [{'gr': 'S,сред,неод=дат,мн', 'lex': 'море'}], 'text': 'морям'},
     {'text': ', '},
     {'analysis': [{'gr': 'V,несов,пе=непрош,деепр', 'lex': 'играть'}],
      'text': 'играя'},
     {'text': ', '},
     {'analysis': [{'gr': 'V,несов,нп=непрош,ед,изъяв,3-л', 'lex': 'носиться'}],
      'text': 'носится'},
     {'text': '\n'},
     {'analysis': [{'gr': 'PR=', 'lex': 'с'}], 'text': 'с'}]




```python
# how to get pos of the word

for word in parsed[:20]:
    if 'analysis' in word:
        gr = word['analysis'][0]['gr']
        pos = gr.split('=')[0].split(',')[0]
        print(word['text'], pos)
```

    По PR
    морям S
    играя V
    носится V
    с PR
    миноносцем S
    миноносица S
    Льнет V
    как ADVPRO



```python
spacy_poem_en = nlp(poem_en)
```


```python
for token in spacy_poem_en:
    if token.pos_ != "SPACE":
        print(token.text, token.lemma_, token.pos_)
```

    Twinkle Twinkle PROPN
    , , PUNCT
    twinkle twinkle NOUN
    , , PUNCT
    little little ADJ
    star star NOUN
    , , PUNCT
    How how ADV
    I -PRON- PRON
    wonder wonder VERB
    what what PRON
    you -PRON- PRON
    are be AUX
    . . PUNCT
    Up up ADP
    above above ADP
    the the DET
    world world NOUN
    so so ADV
    high high ADV
    , , PUNCT
    Like like SCONJ
    a a DET
    diamond diamond NOUN
    in in ADP
    the the DET
    sky sky NOUN
    . . PUNCT
    Twinkle Twinkle PROPN
    , , PUNCT
    twinkle twinkle NOUN
    , , PUNCT
    little little ADJ
    star star NOUN
    , , PUNCT
    How how ADV
    I -PRON- PRON
    wonder wonder VERB
    what what PRON
    you -PRON- PRON
    are be AUX
    ! ! PUNCT
    When when ADV
    the the DET
    blazing blaze VERB
    sun sun NOUN
    is be AUX
    gone go VERB
    , , PUNCT
    When when ADV
    he -PRON- PRON
    nothing nothing PRON
    shines shine VERB
    upon upon SCONJ
    , , PUNCT
    Then then ADV
    you -PRON- PRON
    show show VERB
    your -PRON- DET
    little little ADJ
    light light NOUN
    , , PUNCT
    Twinkle Twinkle PROPN
    , , PUNCT
    twinkle twinkle NOUN
    , , PUNCT
    all all DET
    the the DET
    night night NOUN
    . . PUNCT
    Twinkle Twinkle PROPN
    , , PUNCT
    twinkle twinkle NOUN
    , , PUNCT
    little little ADJ
    star star NOUN
    , , PUNCT
    How how ADV
    I -PRON- PRON
    wonder wonder VERB
    what what PRON
    you -PRON- PRON
    are be AUX
    ! ! PUNCT


###  pymorphy2

**pymorphy2** — is a full-fledged morphological analyzer, written entirely in Python. Unlike Mystem, it does not take into account the context, which means that the question of disambiguation should be resolved by our means. It also knows how to put words in the correct form (conjugate and incline).

[pymorphy2 documentation](https://pymorphy2.readthedocs.io/en/latest/)


```python
# ! pip install --user pymorphy2
```


```python
from pymorphy2 import MorphAnalyzer

morph = MorphAnalyzer()
p = morph.parse('стали')
p
```




    [Parse(word='стали', tag=OpencorporaTag('VERB,perf,intr plur,past,indc'), normal_form='стать', score=0.975342, methods_stack=((DictionaryAnalyzer(), 'стали', 945, 4),)),
     Parse(word='стали', tag=OpencorporaTag('NOUN,inan,femn sing,gent'), normal_form='сталь', score=0.010958, methods_stack=((DictionaryAnalyzer(), 'стали', 13, 1),)),
     Parse(word='стали', tag=OpencorporaTag('NOUN,inan,femn plur,nomn'), normal_form='сталь', score=0.005479, methods_stack=((DictionaryAnalyzer(), 'стали', 13, 6),)),
     Parse(word='стали', tag=OpencorporaTag('NOUN,inan,femn sing,datv'), normal_form='сталь', score=0.002739, methods_stack=((DictionaryAnalyzer(), 'стали', 13, 2),)),
     Parse(word='стали', tag=OpencorporaTag('NOUN,inan,femn sing,loct'), normal_form='сталь', score=0.002739, methods_stack=((DictionaryAnalyzer(), 'стали', 13, 5),)),
     Parse(word='стали', tag=OpencorporaTag('NOUN,inan,femn plur,accs'), normal_form='сталь', score=0.002739, methods_stack=((DictionaryAnalyzer(), 'стали', 13, 9),))]




```python
first = p[0]  # первый разбор
print('Word:', first.word)
print('Tag:', first.tag)
print('Lemma:', first.normal_form)
print('Proba:', first.score)
```

    Word: стали
    Tag: VERB,perf,intr plur,past,indc
    Lemma: стать
    Proba: 0.975342


You can get more detailed information from each tag. If the grammeme is in parsing, its value will be returned, if it is not, it will be returned

[list of grammems](https://pymorphy2.readthedocs.io/en/latest/user/grammemes.html)


```python
print(first.normalized)        # лемма
print(first.tag.POS)           # Part of Speech, часть речи
print(first.tag.animacy)       # одушевленность
print(first.tag.aspect)        # вид: совершенный или несовершенный
print(first.tag.case)          # падеж
print(first.tag.gender)        # род (мужской, женский, средний)
print(first.tag.involvement)   # включенность говорящего в действие
print(first.tag.mood)          # наклонение (повелительное, изъявительное)
print(first.tag.number)        # число (единственное, множественное)
print(first.tag.person)        # лицо (1, 2, 3)
print(first.tag.tense)         # время (настоящее, прошедшее, будущее)
print(first.tag.transitivity)  # переходность (переходный, непереходный)
print(first.tag.voice)         # залог (действительный, страдательный)
```

    Parse(word='стать', tag=OpencorporaTag('INFN,perf,intr'), normal_form='стать', score=1.0, methods_stack=((<DictionaryAnalyzer>, 'стать', 904, 0),))
    VERB
    None
    perf
    None
    None
    None
    indc
    plur
    None
    past
    intr
    None



```python
print(first.normalized)      
print(first.tag.POS)
print(first.tag.aspect)
print(first.tag.case)
```

    Parse(word='стать', tag=OpencorporaTag('INFN,perf,intr'), normal_form='стать', score=1.0, methods_stack=((<DictionaryAnalyzer>, 'стать', 904, 0),))
    VERB
    perf
    None


### mystem vs. pymorphy

1) Both of them can work with out-of-vocabulary words (OOV).

2) *Speed*. Mystem runs incredibly slow under Windows on large texts, but still very fast if you run it from the console on linux / mac os.

3) *Disambiguation*. Mystem is able to disambiguate words by context (although it does not always succeed), pymorphy2 takes one word as input and, accordingly, does not know how to disambiguate by context at all


```python
homonym1 = 'За время обучения я прослушал больше сорока курсов.'
homonym2 = 'Сорока своровала блестящее украшение со стола.'
mystem_analyzer = Mystem() # initialize object with default parameters

print(mystem_analyzer.analyze(homonym1)[-5])
print(mystem_analyzer.analyze(homonym2)[0])
```

    {'text': 'сорока', 'analysis': [{'lex': 'сорок', 'gr': 'NUM=(пр|дат|род|твор)'}]}
    {'text': 'Сорока', 'analysis': [{'lex': 'сорока', 'gr': 'S,жен,од=им,ед'}]}



```python
p = morph.parse('сорока')
```


```python
p
```




    [Parse(word='сорока', tag=OpencorporaTag('NUMR loct'), normal_form='сорок', score=0.285714, methods_stack=((<DictionaryAnalyzer>, 'сорока', 2802, 5),)),
     Parse(word='сорока', tag=OpencorporaTag('NOUN,inan,femn sing,nomn'), normal_form='сорока', score=0.142857, methods_stack=((<DictionaryAnalyzer>, 'сорока', 43, 0),)),
     Parse(word='сорока', tag=OpencorporaTag('NOUN,anim,femn sing,nomn'), normal_form='сорока', score=0.142857, methods_stack=((<DictionaryAnalyzer>, 'сорока', 403, 0),)),
     Parse(word='сорока', tag=OpencorporaTag('NUMR gent'), normal_form='сорок', score=0.142857, methods_stack=((<DictionaryAnalyzer>, 'сорока', 2802, 1),)),
     Parse(word='сорока', tag=OpencorporaTag('NUMR datv'), normal_form='сорок', score=0.142857, methods_stack=((<DictionaryAnalyzer>, 'сорока', 2802, 2),)),
     Parse(word='сорока', tag=OpencorporaTag('NUMR ablt'), normal_form='сорок', score=0.142857, methods_stack=((<DictionaryAnalyzer>, 'сорока', 2802, 4),))]



### Joining it all together:

Let us make a standard data preprocessing pipeline of texts from the Lenta.ru website


```python
!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1IMtWrTrs54SVzHlTgtBWqcQA990CCrQh' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1IMtWrTrs54SVzHlTgtBWqcQA990CCrQh" -O lenta-ru-news-full.csv && rm -rf /tmp/cookies.txt
```

    --2021-10-26 19:36:26--  https://docs.google.com/uc?export=download&confirm=94CX&id=1IMtWrTrs54SVzHlTgtBWqcQA990CCrQh
    Resolving docs.google.com (docs.google.com)... 74.125.134.113, 74.125.134.139, 74.125.134.100, ...
    Connecting to docs.google.com (docs.google.com)|74.125.134.113|:443... connected.
    HTTP request sent, awaiting response... 302 Moved Temporarily
    Location: https://doc-0c-bc-docs.googleusercontent.com/docs/securesc/82i407l4bglp5lqo6aftb2tgvf5s9g97/c4i97c0alcigkiodq3ec7ffqfjufgeqn/1635276975000/14013181690233399305/11514772734847751932Z/1IMtWrTrs54SVzHlTgtBWqcQA990CCrQh?e=download [following]
    --2021-10-26 19:36:26--  https://doc-0c-bc-docs.googleusercontent.com/docs/securesc/82i407l4bglp5lqo6aftb2tgvf5s9g97/c4i97c0alcigkiodq3ec7ffqfjufgeqn/1635276975000/14013181690233399305/11514772734847751932Z/1IMtWrTrs54SVzHlTgtBWqcQA990CCrQh?e=download
    Resolving doc-0c-bc-docs.googleusercontent.com (doc-0c-bc-docs.googleusercontent.com)... 172.217.204.132, 2607:f8b0:400c:c15::84
    Connecting to doc-0c-bc-docs.googleusercontent.com (doc-0c-bc-docs.googleusercontent.com)|172.217.204.132|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://docs.google.com/nonceSigner?nonce=uq550be0h2if8&continue=https://doc-0c-bc-docs.googleusercontent.com/docs/securesc/82i407l4bglp5lqo6aftb2tgvf5s9g97/c4i97c0alcigkiodq3ec7ffqfjufgeqn/1635276975000/14013181690233399305/11514772734847751932Z/1IMtWrTrs54SVzHlTgtBWqcQA990CCrQh?e%3Ddownload&hash=ab67k9r9b1sp1rsogml1udeavvave01f [following]
    --2021-10-26 19:36:26--  https://docs.google.com/nonceSigner?nonce=uq550be0h2if8&continue=https://doc-0c-bc-docs.googleusercontent.com/docs/securesc/82i407l4bglp5lqo6aftb2tgvf5s9g97/c4i97c0alcigkiodq3ec7ffqfjufgeqn/1635276975000/14013181690233399305/11514772734847751932Z/1IMtWrTrs54SVzHlTgtBWqcQA990CCrQh?e%3Ddownload&hash=ab67k9r9b1sp1rsogml1udeavvave01f
    Connecting to docs.google.com (docs.google.com)|74.125.134.113|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://doc-0c-bc-docs.googleusercontent.com/docs/securesc/82i407l4bglp5lqo6aftb2tgvf5s9g97/c4i97c0alcigkiodq3ec7ffqfjufgeqn/1635276975000/14013181690233399305/11514772734847751932Z/1IMtWrTrs54SVzHlTgtBWqcQA990CCrQh?e=download&nonce=uq550be0h2if8&user=11514772734847751932Z&hash=5rn3pfspspiui2ob0f9li8tqfcotjd7j [following]
    --2021-10-26 19:36:26--  https://doc-0c-bc-docs.googleusercontent.com/docs/securesc/82i407l4bglp5lqo6aftb2tgvf5s9g97/c4i97c0alcigkiodq3ec7ffqfjufgeqn/1635276975000/14013181690233399305/11514772734847751932Z/1IMtWrTrs54SVzHlTgtBWqcQA990CCrQh?e=download&nonce=uq550be0h2if8&user=11514772734847751932Z&hash=5rn3pfspspiui2ob0f9li8tqfcotjd7j
    Connecting to doc-0c-bc-docs.googleusercontent.com (doc-0c-bc-docs.googleusercontent.com)|172.217.204.132|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 2084746431 (1.9G) [text/csv]
    Saving to: ‘lenta-ru-news-full.csv’
    
    lenta-ru-news-full. 100%[===================>]   1.94G  78.1MB/s    in 21s     
    
    2021-10-26 19:36:48 (93.2 MB/s) - ‘lenta-ru-news-full.csv’ saved [2084746431/2084746431]
    



```python
import pandas as pd
pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', 800)

data = pd.read_csv('./lenta-ru-news-full.csv', usecols=['text'])
data.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>214459</th>
      <td>Число посетителей сайтов американских газет в 2007 году резко возросло, свидетельствуют статистические данные, обнародованные 24 января Газетной ассоциацией Америки (Newspaper Association of America), сообщает Reuters. В 2007 году в среднем за месяц на сайты американских газет заходили около 60 миллионов уникальных посетителей - на шесть процентов больше, чем в предыдущем году. При этом в четвертом квартале 2007 года аудитория интернет-версий газет, издающихся в США, демонстрировала прирост на 9 процентов в месяц. Интернет-версии газет посещали в 2007 году не менее 39 процентов от всей аудитории Сети. При этом средний пользователь проводил на сайтах газет не менее 44 минут в месяц. Статистические данные, опубликованные Газетной ассоциацией Америки, учитывают пользователей, выходящих в ...</td>
    </tr>
    <tr>
      <th>664833</th>
      <td>На Лондонском мосту микроавтобус наехал на пешеходов, проходит эвакуация людей, сообщает газета The Sun. Очевидцы также сообщают, что в районе моста слышны выстрелы, а несколько людей получили ножевые ранения. По информации BBC News, на месте происшествия работают вооруженные сотрудники полиции. В результате происшествия, как пишет The Telegraph, пострадали 15-20 человек. 22 марта в Лондоне уроженец юго-востока Англии Эдриан Рассел Аджао, сменивший имя на Халид Масуд, находясь за рулем автомобиля, направил его на пешеходов на Вестминстерском мосту, сбил несколько человек и доехал до здания парламента. Там машину остановил полицейский. Преступник зарезал его ножом, после чего был застрелен другим стражем порядка. Погибли пять человек.</td>
    </tr>
    <tr>
      <th>110006</th>
      <td>Американская компьютерная компания Altnet направила письма владельцам нескольких базирующихся в США файлообменных сетей с уведомлением о том, что они незаконно используют в своей работе технологию, патент на которую ей принадлежит. Как сообщает газета Washington Post, Altnet владеет патентом на технологию "хэшинга" (hashing), которая предусматривает снабжение каждого файла кратким описанием ("хэшем"), с которым пользователи сетей могут ознакомиться, чтобы узнать, что содержится в том или ином файле. Сети также используют хэши для управления публичными каталогами пользователей. Как известно, файлообменные сети предназначены для бесплатного обмена компьютерными файлами между пользователями Интернета, и большинство медиапродукции, распространяемой по таким сетям, является нелицензионной. ...</td>
    </tr>
    <tr>
      <th>774383</th>
      <td>Мэр Москвы Сергей Собянин заявил, что организаторы несанкционированных мероприятий в центре Москвы пытались втянуть участников в массовые беспорядки. Об этом он заявил в эфире телеканала «Россия-1», сообщает в воскресенье, 4 августа, радиостанция «Говорит Москва». По его словам, не все пришедшие на акцию граждане ожидали «такого развития событий». «Это все не для блага людей, а ради чьих-то политических, узкокорыстных целей…» — высказался градоначальник. Он добавил, что, по его данным, многие собравшиеся не имели «никакого отношения к Москве» и к выборам в Мосгордуму. «Я уверен, что москвичи это хорошо понимают», — заключил Собянин. Несогласованный митинг в поддержку не допущенных на выборы независимых кандидатов в Мосгордуму начался в Москве днем в субботу, 3 августа. По информации МВ...</td>
    </tr>
    <tr>
      <th>518138</th>
      <td>Сотрудники Следственного комитета России задержали в Санкт-Петербурге генерального директора компании «Экспо-тур» Игоря Рюрикова, сообщает официальный представитель ведомства Владимир Маркин. По данным следствия, задержанный знал, что его компанию исключат из Единого федерального реестра в июне 2014 года. Однако он дал указание сотрудникам продолжать работу и оформлять путевки. «В результате его действий были похищены денежные средства граждан на общую сумму более одного миллиона рублей», — подчеркнул Маркин. В настоящее время решается вопрос о его аресте и предъявлении обвинения. После проверок деятельности турфирм в Москве и Санкт-Петербурге за последнее время возбуждено шесть уголовных дел в отношении руководителей компаний, сообщает официальный представитель Генеральной прокуратуры...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# !pip install pymorphy2
```


```python
import pymorphy2
import re
```


```python
m = pymorphy2.MorphAnalyzer()

# delete non-alphabetic characters
regex = re.compile("[А-Яа-яA-z]+")

def words_only(text, regex=regex):
    try:
        return regex.findall(text.lower())
    except:
        return []
```


```python
print(data.text[0])
```

    Бои у Сопоцкина и Друскеник закончились отступлением германцев. Неприятель, приблизившись с севера к Осовцу начал артиллерийскую борьбу с крепостью. В артиллерийском бою принимают участие тяжелые калибры. С раннего утра 14 сентября огонь достиг значительного напряжения. Попытка германской пехоты пробиться ближе к крепости отражена. В Галиции мы заняли Дембицу. Большая колонна, отступавшая по шоссе от Перемышля к Саноку, обстреливалась с высот нашей батареей и бежала, бросив парки, обоз и автомобили. Вылазки гарнизона Перемышля остаются безуспешными. При продолжающемся отступлении австрийцев обнаруживается полное перемешивание их частей, захватываются новые партии пленных, орудия и прочая материальная часть. На перевале Ужок мы разбили неприятельский отряд, взяли его артиллерию и много пленных и, продолжая преследовать, вступили в пределы Венгрии. 
    «Русский инвалид», 16 сентября 1914 года.



```python
print(*words_only(data.text[0]))
```

    бои у сопоцкина и друскеник закончились отступлением германцев неприятель приблизившись с севера к осовцу начал артиллерийскую борьбу с крепостью в артиллерийском бою принимают участие тяжелые калибры с раннего утра сентября огонь достиг значительного напряжения попытка германской пехоты пробиться ближе к крепости отражена в галиции мы заняли дембицу большая колонна отступавшая по шоссе от перемышля к саноку обстреливалась с высот нашей батареей и бежала бросив парки обоз и автомобили вылазки гарнизона перемышля остаются безуспешными при продолжающемся отступлении австрийцев обнаруживается полное перемешивание их частей захватываются новые партии пленных орудия и прочая материальная часть на перевале ужок мы разбили неприятельский отряд взяли его артиллерию и много пленных и продолжая преследовать вступили в пределы венгрии русский инвалид сентября года


`@lru_cache` method creates a cache of the specified size for the `lemmatize` function, which allows you to speed up text lemmatization in general (which is very useful, since lemmatization is a resource-intensive process).


```python
import functools
from nltk.corpus import stopwords
```


```python
@functools.lru_cache(maxsize=128)
def lemmatize_word(token, pymorphy=m):
    return pymorphy.parse(token)[0].normal_form

def lemmatize_text(text):
    return [lemmatize_word(w) for w in text]
```


```python
tokens = words_only(data.text[0])

print(lemmatize_text(tokens))
```

    ['бой', 'у', 'сопоцкина', 'и', 'друскеник', 'закончиться', 'отступление', 'германец', 'неприятель', 'приблизиться', 'с', 'север', 'к', 'осовца', 'начать', 'артиллерийский', 'борьба', 'с', 'крепость', 'в', 'артиллерийский', 'бой', 'принимать', 'участие', 'тяжёлый', 'калибр', 'с', 'ранний', 'утро', 'сентябрь', 'огонь', 'достигнуть', 'значительный', 'напряжение', 'попытка', 'германский', 'пехота', 'пробиться', 'близкий', 'к', 'крепость', 'отразить', 'в', 'галиция', 'мы', 'занять', 'дембица', 'больший', 'колонна', 'отступать', 'по', 'шоссе', 'от', 'перемышль', 'к', 'санок', 'обстреливаться', 'с', 'высота', 'наш', 'батарея', 'и', 'бежать', 'бросить', 'парка', 'обоз', 'и', 'автомобиль', 'вылазка', 'гарнизон', 'перемышль', 'оставаться', 'безуспешный', 'при', 'продолжаться', 'отступление', 'австриец', 'обнаруживаться', 'полный', 'перемешивание', 'они', 'часть', 'захватываться', 'новый', 'партия', 'пленный', 'орудие', 'и', 'прочий', 'материальный', 'часть', 'на', 'перевал', 'ужок', 'мы', 'разбить', 'неприятельский', 'отряд', 'взять', 'он', 'артиллерия', 'и', 'много', 'пленный', 'и', 'продолжать', 'преследовать', 'вступить', 'в', 'предел', 'венгрия', 'русский', 'инвалид', 'сентябрь', 'год']



```python
mystopwords = stopwords.words('russian') 

def remove_stopwords(lemmas, stopwords = mystopwords):
    return [w for w in lemmas if not w in stopwords]
```


```python
lemmas = lemmatize_text(tokens)

print(*remove_stopwords(lemmas))
```

    бой сопоцкина друскеник закончиться отступление германец неприятель приблизиться север осовца начать артиллерийский борьба крепость артиллерийский бой принимать участие тяжёлый калибр ранний утро сентябрь огонь достигнуть значительный напряжение попытка германский пехота пробиться близкий крепость отразить галиция занять дембица больший колонна отступать шоссе перемышль санок обстреливаться высота наш батарея бежать бросить парка обоз автомобиль вылазка гарнизон перемышль оставаться безуспешный продолжаться отступление австриец обнаруживаться полный перемешивание часть захватываться новый партия пленный орудие прочий материальный часть перевал ужок разбить неприятельский отряд взять артиллерия пленный продолжать преследовать вступить предел венгрия русский инвалид сентябрь год



```python
def remove_stopwords(lemmas, stopwords = mystopwords):
    return [w for w in lemmas if not w in stopwords and len(w) > 3]
```


```python
print(*remove_stopwords(lemmas))
```

    сопоцкина друскеник закончиться отступление германец неприятель приблизиться север осовца начать артиллерийский борьба крепость артиллерийский принимать участие тяжёлый калибр ранний утро сентябрь огонь достигнуть значительный напряжение попытка германский пехота пробиться близкий крепость отразить галиция занять дембица больший колонна отступать шоссе перемышль санок обстреливаться высота батарея бежать бросить парка обоз автомобиль вылазка гарнизон перемышль оставаться безуспешный продолжаться отступление австриец обнаруживаться полный перемешивание часть захватываться новый партия пленный орудие прочий материальный часть перевал ужок разбить неприятельский отряд взять артиллерия пленный продолжать преследовать вступить предел венгрия русский инвалид сентябрь


Joining everything to one function:


```python
def clean_text(text):
    tokens = words_only(text)
    lemmas = lemmatize_text(tokens)
    
    return remove_stopwords(lemmas)
```


```python
print(*clean_text(data.text[3]))
```

    фотограф корреспондент daily mirror рассказывать случай который порадовать весь друг животное лейтенант бельгийский артиллерия руководить оборона фортов льеж хотеть расстаться свой собака бульдог пруссак пробраться фортов самый город офицер прийти голова доверить бульдог письмо который посылать успокоительный ведать свой родитель благородный честно исполнить свой миссия десять спустя бульдог проникнуть обратно форт принести ответ момент бульдог стать настоящий гонец пробираться линия германский войско нести спрятать ошейник шифровать депеша журнал нива сентябрь


If you need to preprocess large texts, you may also use `Pool` method of the `multiprocessing` library:


```python
from multiprocessing import Pool
from tqdm.notebook import tqdm

N = 200

with Pool(8) as p:
    lemmas = list(tqdm(p.imap(clean_text, data['text'][:N]), total=N))
```


      0%|          | 0/200 [00:00<?, ?it/s]



```python
data = data.head(200)
data['lemmas'] = lemmas
data.sample(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>lemmas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>146</th>
      <td>В четверг вечером представители Генеральной прокуратуры России приступили к обыскам на даче и двух московских квартирах Юрия Скуратова, отстраненного от должности генпрокурора. Как передает "Интерфакс", ссылаясь на дочь  Юрия Скуратова, по  данным  на 20 часов 40 минут по московскому времени  группа сотрудников  Генпрокуратуры из  шести  человек производила обыск на даче генпрокурора   в   подмосковном Архангельском. По словам дочери Скуратова, днем в этот же день были проведены обыски в  двух московских  квартирах, в  которых прописана  семья Юрия Скуратова и родители его жены. В Генеральной  прокуратуре РФ  "Интерфаксу" подтвердили, что обыски проводятся  с санкции  прокурора  в  рамках  расследования уголовного дела,  возбужденного 2  апреля в отношении Скуратова по статье "Злоупотр...</td>
      <td>[четверг, вечером, представитель, генеральный, прокуратура, россия, приступить, обыск, дача, московский, квартира, юрий, скуратов, отстранить, должность, генпрокурор, передавать, интерфакс, ссылаться, дочь, юрий, скуратов, данные, минута, московский, время, группа, сотрудник, генпрокуратура, шесть, человек, производить, обыск, дача, генпрокурор, подмосковный, архангельск, слово, дочь, скуратов, день, день, провести, обыск, московский, квартира, который, прописать, семья, юрий, скуратов, родитель, жена, генеральный, прокуратура, интерфакс, подтвердить, обыск, проводиться, санкция, прокурор, рамка, расследование, уголовный, дело, возбудить, апрель, отношение, скуратов, статья, злоупотребление, должностной, полномочие, скуратов, комментарий, воздержаться]</td>
    </tr>
    <tr>
      <th>165</th>
      <td>Нефтеперерабатывающий завод "Ярославнефтеоргсинтез" увеличил отпускные оптовые цены на наиболее ходовые марки бензина в среднем на 25% по отношению к ценам августа, - передает РИА "Новости" со ссылкой на отдел сбыта завода. Цена бензина Аи-92 возросла на 21% и теперь составляет 5 тысяч 500 рублей за тонну; бензина А-76 - на 33% (5 тысяч 200 рублей за тонну); бензина Аи-95 - на 18% (6 тысяч 400 рублей за тонну). Дизельное топливо завод отпускает теперь по цене 3 тысячи 600 рублей за тонну, что соответствует подорожанию на 23% по сравнению с прошлым месяцем. Нефтеперерабатывающий завод "Ярославнефтеоргсинтез" является одним из основных поставщиков нефтепродуктов на рынок Москвы и Центрального региона России в целом.</td>
      <td>[нефтеперерабатывающий, завод, ярославнефтеоргсинтез, увеличить, отпускной, оптовый, цена, наиболее, ходовой, марка, бензин, среднее, отношение, цена, август, передавать, новость, ссылка, отдел, сбыт, завод, цена, бензин, возрасти, составлять, тысяча, рубль, тонна, бензин, тысяча, рубль, тонна, бензин, тысяча, рубль, тонна, дизельный, топливо, завод, отпускать, цена, тысяча, рубль, тонна, соответствовать, подорожание, сравнение, прошлое, месяц, нефтеперерабатывающий, завод, ярославнефтеоргсинтез, являться, основный, поставщик, нефтепродукт, рынок, москва, центральный, регион, россия, целое]</td>
    </tr>
    <tr>
      <th>105</th>
      <td>Сегодня днем в столице Греции произошло землетрясение силой 5,9 балла по шкале Рихтера. Согласно данным греческих сейсмологов, эпицентр находился в 20 километрах к северо-западу от Афин. В результате подземных толчков вибрировали здания, некоторые из них покрылись трещинами, во многих местах осыпалась штукатурка. По данным полиции, в северной части Афин рухнул один дом. По радио было передано обращение местных властей к столичным жителям с указанием покинуть жилые помещения в связи с продолжающейся сейсмической активностью. Люди сильно озабочены происходящим, пытаются дозвониться до родственников или добраться на автомобиле туда, где находятся их дети - в школы, детские сады. Люди крайне взволнованы, все вспоминают о недавнем разрушительном землетрясении в соседней Турции. В результате...</td>
      <td>[сегодня, день, столица, греция, произойти, землетрясение, сила, балл, шкала, рихтер, согласно, данные, греческий, сейсмолог, эпицентр, находиться, километр, северо, запад, афины, результат, подземный, толчок, вибрировать, здание, некоторый, покрыться, трещина, многий, место, осыпаться, штукатурка, данные, полиция, северный, часть, афины, рухнуть, радио, передать, обращение, местный, власть, столичный, житель, указание, покинуть, жилой, помещение, связь, продолжаться, сейсмический, активность, человек, сильно, озаботить, происходить, пытаться, дозвониться, родственник, добраться, автомобиль, туда, находиться, ребёнок, школа, детский, человек, крайне, взволновать, вспоминать, недавний, разрушительный, землетрясение, соседний, турция, результат, подземный, толчок, афины, выйти, строй, ст...</td>
    </tr>
  </tbody>
</table>
</div>



## Now it is your time to analyse text in English


```python
from sklearn.datasets import fetch_20newsgroups
```


```python
def twenty_newsgroup_to_csv():
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

    df = pd.DataFrame([newsgroups_train.data, newsgroups_train.target.tolist()]).T
    df.columns = ['text', 'target']

    targets = pd.DataFrame( newsgroups_train.target_names)
    targets.columns=['title']

    out = pd.merge(df, targets, left_on='target', right_index=True)
    out['date'] = pd.to_datetime('now')
    return out
```


```python
df = twenty_newsgroup_to_csv()
df.head(5)
```


```python
# your code here
```

### Result:

- learned about standard text preprocessing pipeline
- learned how to work with morphological parsers


```python

```
