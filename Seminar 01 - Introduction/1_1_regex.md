<h1><center>Regular expressions</center></h1>



**Regular expression** _\(RegExp\)_ — is the specific, standard textual syntax for representing patterns for matching substrings in the text. In other words, it is a sequence of characters that specifies a pattern to search / replace with something those pieces of text that match with it.

**Useful materials:** https://habr.com/ru/post/115825/ and https://habr.com/ru/post/115436/

![rexexp](https://imgs.xkcd.com/comics/regular_expressions.png)

Python has a built-in package called `re`, which can be used to work with Regular Expressions. (you may read the docs [here](https://docs.python.org/3/library/re.html)). First, we need to import it, as any other library. Here are listed the most common methods of this library:

* re.match()
* re.search()
* re.findall()
* re.sub()
* re.compile()

### re.match()

This methods checks for a match **only at the beginning of the string**. For instance, if we apply the following pattern 'the' on the text "the cat is on the mat" with the `match()` method, we will find the first occurence of "the" and the program will end successfully. However, if we are looking for the pattern "cat", the result would be empty. 

`re.match()` accepts two arguments:

* pattern (what are we looking for?)
* string (where we want to search for it?)


```python
import re

re.match("[cmf]an", "dan")
```


```python
re.match("[^cmf]an", "fan")
```


```python
re.match("(c|m|f)an", "man")
```




    <re.Match object; span=(0, 3), match='man'>




```python
import re

re.match('the', 'the cat is on the mat')
```




    <re.Match object; span=(0, 3), match='the'>




```python
print(re.match('cat', 'the cat is on the mat'))
```

    None


"cat" is the precise pattern, now let us try to search with general patterns, so we need set ranges for that:

* **\[A-Z\]** — will match _any single_ uppercase letter \(latin script\)
* **\[a-z\]** — will match _any single_ lowercase letter \(latin script\)
* **\[А-Я\]** — will match _any single_ uppercase letter \(cyrillic script\)
* **\[а-я\]** — will match _any single_ lowercase letter \(cyrillic script\)
* **\[0-9\]** or **\d** — one digit

^ Matches a single character that is not contained within the brackets:
* **\[^0-9\]** or **\D** — _any single_ character apart from digit

Wildcard **.** (dot) — matches any single character.


```python
re.match('[a-z]', 'the cat is on the mat')
```




    <re.Match object; span=(0, 1), match='t'>




```python
re.match('[0-9]', 'the cat is on the mat')
```


```python
re.match('.', 'the cat is on the mat')
```




    <re.Match object; span=(0, 1), match='t'>




```python
re.match('.', ' the cat is on the mat')
```




    <re.Match object; span=(0, 1), match=' '>



Set ranges can be combined:

* **\[A-Za-z\]** — _any single_ uppercase and lowercase letter \(latin script\)
* **\[A-Za-z0-9\]** — _any single_ uppercase and lowercase letter \(latin script\) or digit
* **\[A-Za-z0-9\_\]** или **\w** — _any single_ uppercase and lowercase letter \(latin script\) or digit or \_
* **\[^A-Za-z0-9\_\]** или **\W** — anything except uppercase and lowercase letters \(latin script\), digits and \_

* You may choose any range from the [unicode table](https://unicode-table.com/ru/) like, for example, **[à-ÿ]** 


```python
re.match('[A-Za-z]', 'Uppercase letters'), re.match('[A-Za-z]', 'no uppercase letters')
```




    (<re.Match object; span=(0, 1), match='U'>,
     <re.Match object; span=(0, 1), match='n'>)




```python
re.match('[A-z]', 'Uppercase letters')
```




    <re.Match object; span=(0, 1), match='U'>




```python
re.match('[А-яЁё]', 'ёжик')
```




    <re.Match object; span=(0, 1), match='ё'>



### re.search()

While previous method checks for a match only at the beginning of the string, `re.search()` checks for a match anywhere in the string, however, it returns only the first occurence of the patter. Arguments of this method are the same `re.search(pattern, string)`.


```python
re.search('the', 'the cat is on the mat')
```




    <re.Match object; span=(0, 3), match='the'>




```python
re.search('cat', 'the cat is on the mat')
```




    <re.Match object; span=(4, 7), match='cat'>



How can we return not the matched object, but the string itself?


```python
re.search('cat', 'the cat is on the mat').group(0)
```




    'cat'




```python
re.search('[A-Za-z][A-Za-z][A-Za-z]', 'The cat is on the mat')[0]
```




    'The'




```python
re.match("waz{,5}up", 'wazzzup')
```




    <re.Match object; span=(0, 7), match='wazzzup'>



### re.findall()

Return **all** non-overlapping matches of pattern in string, as a list of strings or tuples. The string is scanned left-to-right, and matches are returned in the order found. Empty matches are included in the result. Returns a list of all matches.


```python
re.findall('the', 'the cat is on the mat')
```




    ['the', 'the']




```python
re.findall('the', 'the cat is on the mat')[1]
```




    'the'




```python
re.match("a+b*c+", 'aabbbbc')
```




    <re.Match object; span=(0, 7), match='aabbbbc'>



Another important feature that can be used in patterns, is quantification. We can define the desired length of characters we search in the pattern:

###### Quantifiers

* **?** — question mark indicates _zero or one_ occurrences of the preceding character/group
* **\*** — asterisk indicates _zero or more_ occurrences of the preceding character/group
* **+** — plus indicates _one or more_ occurrences of the preceding character/group
* **{n}** — preceding character/group is matched exactly _n times_
* **{n,}** — preceding character/group is matched _n or more times_
* **{,m**} — preceding character/group is matched _up to m times_
* **{n,m}** — preceding item is matched _at least n times, but not more than m times_


**NB!** Quantifiers __.__ __\\__ __\*__ __+__ etc. do not work inside \[ \]


```python
re.findall('[a-z]+', 'the cat is on the mat')
```




    ['the', 'cat', 'is', 'on', 'the', 'mat']




```python
re.findall('[a-z]{3}', 'the cat is on the mat')
```




    ['the', 'cat', 'the', 'mat']




```python
re.findall('[a-z]*', 'the cat is on the mat')
```




    ['the', '', 'cat', '', 'is', '', 'on', '', 'the', '', 'mat', '']




```python
re.findall('[a-z]?', 'the cat is on the mat')
```




    ['t',
     'h',
     'e',
     '',
     'c',
     'a',
     't',
     '',
     'i',
     's',
     '',
     'o',
     'n',
     '',
     't',
     'h',
     'e',
     '',
     'm',
     'a',
     't',
     '']




```python
text = '''Mission: successful
Last Mission: unsuccessful	To be completed
Next Mission: successful upon capture of target
Mission: successful upon capture
'''
```


```python
re.findall('^Mission: successful$', text, re.MULTILINE)
```




    ['Mission: successful']



###### Lazy  and possessive/greedy matching

Quantifiers are greedy by default. It means that they match the longest substring that matches the search they can find. In other words, they consume as many characters as possible. For example, we want to extract all substrings inside quotes from the string `'a "witch" and her "broom" is one'`. If we use the following pattern: `".+"`, we will get the following result:


```python
s = 'a "witch" and her "broom" is one'
re.findall('".+"', s)
```




    ['"witch" and her "broom"']




```python
for i in re.finditer('".+"', s):
    print(i)
```

    <re.Match object; span=(2, 25), match='"witch" and her "broom"'>


Maximum number of symbols between the first quotation mark and the last quotation mark is 21. Our greedy matching got this substring instead of "witch" and "broom". In order to retrive shorter substrings we should make our quantifier to **non-greedy** (or **lazy**) by adding quotation mark '?'.


```python
re.findall('".+?"', s)
```




    ['"witch"', '"broom"']



This operation can be applied to all quantifiers.

| Greedy quantifiers | Lazy quantifiers |
| :--- | :--- |
| \* | \*? |
| + | +? |
| {min, max} | {min, max}? |

###### Escape characters

The usual metacharacters are `{}[]()^$.|*+?` and `\`, however they do not have their literal character meaning. But what if we need to find such symbols in the text? For instance, we want to find all sentences, ending with quotation mark. The answer is as follows: we need to "escape" them, i.o.w. add __\\__ before such characters.

Let us retrieve emojis from tweets:


```python
tweet = 'have a good day :)'
re.findall('[:\)\(]+', tweet)
```




    [':)']



Or punctuation marks: 


```python
tweet = 'Дождь - это прекрасно, в дожде можно спрятать слезы...'
re.findall('[\-\.!?:;,]|[.]+', tweet)
```




    ['-', ',', '.', '.', '.']



Here we used one more metacharacter: **|**  which means **or**.


```python
tweet = 'Дождь - это прекрасно, в дожде можно спрятать слезы...'
re.findall('[.]+|[\-\.!?:;,]', tweet)
```




    ['-', ',', '...']



Pay attention to the ellipsis (dot-dot-dot) in this example!

Let us find formulae in the tweet:


```python
tweet = 'Формула всем известная: (a+b)^2 = a^2 + 2*a*b + b^2'
re.findall('[\^\+\(\)=\-\* 0-9a-z]{2,}', tweet)
```




    [' (a+b)^2 = a^2 + 2*a*b + b^2']



Question: how to escape backslash *\\*?


```python
tweet = "find \ here"

#Your code here
```

### re.sub()

Return the string obtained by replacing the leftmost non-overlapping occurrences of pattern in string by the replacement repl. If the pattern isn’t found, string is returned unchanged. The method accapts three arguments:

* what to substitute
* on what to substitute
* where substittute

All matched subsrings are replaced.


```python
re.sub('the', 'my', 'the cat is on the mat')
```




    'my cat is on my mat'



Here are a few more metacharacters in RegExp:

* **\t** — matches tabs
* **\s** — matches a whitespace character
* **\S** — matches anything but a whitespace
* **\n** — new line
* **^** — start of line
* **$** — end of line

Let us delete extra spaces from the following text:


```python
s  = 'а  я иду,     шагаю   по   Москве '
re.sub(' +', ' ', s)
```




    'а я иду, шагаю по Москве '



Or let us bring an excerpt of the play to a readable form:


```python
s = '''
Дездемона: 

Кто здесь? Отелло, ты?

Отелло: 

Я, Дездемона. 

Дездемона: 

Что ж не идешь ложиться ты, мой друг?

Отелло:

Молилась ли ты на ночь, Дездемона?

Дездемона:

Да, милый мой.

'''
print(s)
```

    
    Дездемона: 
    
    Кто здесь? Отелло, ты?
    
    Отелло: 
    
    Я, Дездемона. 
    
    Дездемона: 
    
    Что ж не идешь ложиться ты, мой друг?
    
    Отелло:
    
    Молилась ли ты на ночь, Дездемона?
    
    Дездемона:
    
    Да, милый мой.
    
    



```python
s = re.sub('\n+', '\n', s)
print(s)
```

    
    Дездемона: 
    Кто здесь? Отелло, ты?
    Отелло: 
    Я, Дездемона. 
    Дездемона: 
    Что ж не идешь ложиться ты, мой друг?
    Отелло:
    Молилась ли ты на ночь, Дездемона?
    Дездемона:
    Да, милый мой.
    



```python
s = re.sub(':[ \n]+', ': ', s)
print(s)
```

    
    Дездемона: Кто здесь? Отелло, ты?
    Отелло: Я, Дездемона. 
    Дездемона: Что ж не идешь ложиться ты, мой друг?
    Отелло: Молилась ли ты на ночь, Дездемона?
    Дездемона: Да, милый мой.
    


###  re.compile()

Compiles a regular expression pattern into a regular expression object, which can be used for all methods described above. Then ths object can be applied multiple times to different strings. Pros: we do not need to copy-paste patterns and program will work faster!


```python
# compile regexp to find cats
cats = re.compile('cat')

# now we can use this variable with any method
# here we do not provide "what to search/replace" argument anymore 
print(cats.search('the cat is on the mat').group(0))
print(cats.findall('my cat is black, my cat is fat, my cat likes rats, rats are gray and fat'))
print(cats.sub('dog', 'the cat is on the mat'))
```

    cat
    ['cat', 'cat', 'cat']
    the dog is on the mat



```python
cats = re.compile('cat')
texts = ['the cat is on the mat', 'the cat is on the mat', 'my cat is black, my cat is fat, my cat likes rats, rats are gray and fat']

for text in texts:
    print(cats.findall(text))
```

    ['cat']
    ['cat']
    ['cat', 'cat', 'cat']


## You have the floor now!

Write a regexp to search for the different name of Søren Kierkegaard in Russian.


```python
s = '''
Киркегор - датский философ, богослов и писатель, один из предшественников экзистенциализма. 
С. Кьеркегор окончил теологический факультет Копенгагенского университета в 1840 году. 
Степень магистра получил в 1841 году, защитив диссертацию “О понятии иронии, с постоянным обращением к Сократу”, посвященную концепциям иронии у древнегреческих авторов и романтиков. 
Работы С. Кьеркегора отличаются исключительной психологической точностью и глубиной. 
Вклад в развистие философии, сделанный Кьеркегаардом. неоценим. 
Сёрен Киркегаард: немецкое издание Сёрена Киркегаарда. 
Спецкурс “С. Керкегор и история христианства в XIX в.” посвящен датскому философу Серену Керкегору. 
'''
```


```python
philosopher = re.compile(r"К[иеь]е*ркег.+?рд?\w?\w?")
res = philosopher.findall(s)
res
```




    ['Киркегор',
     'Кьеркегор',
     'Кьеркегора',
     'Кьеркегаардом',
     'Киркегаард',
     'Киркегаарда',
     'Керкегор',
     'Керкегору']




```python
len(res)
```




    8



Write REgExps to search for emails and phone numbers:


```python
!wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hcw1REIuEZRG-Qe234eFviR1nBNz9IkM' -O instructions.txt
```


```python
with open ('instructions.txt', 'r') as f:
    text = f.read()
    
print(text)
```

    ИНСТРУКЦИЯ ПО ОФОРМЛЕНИЮ ЦИФРОВОГО ПРОПУСКА
    
    Технические вопросы
    Как заказать пропуск в Москву при условии, что на даче интернет работает плохо? При звонке по номеру телефона: +7 (495) 777-7777 приходится очень долго ждать.
    В связи с большим количеством звонков время ожидания по номеру телефона: 8-495-777-77-77 может достигать нескольких минут. Кроме того, вы можете воспользоваться возможностью получить пропуск по СМС на короткий номер 7377.
    
    Списываются ли деньги при отправке СМС для получения пропуска?
    Нет. За СМС средства не списывают. 
    
    Что делать если сайт не доступен, а при звонке на телефон: 84957777777 сеть занята?
    Обратиться на почту по адресу gosuslugi@mail.ru
    
    
    Как заказать оформленный пропуск на электронную почту через СМС?
    Для этого нужно отправить цель получения пропуска (в кавычках) и через пробел почту, например,
    "для поездок на работу", golikova_t67@gmail.com
    "для иных целей", natysik@ya.ru
    
    По любым вопросам пишите на круглосуточную линию поддержки support24@mos.ru.



```python
mails = re.compile('Your regexp here for emails')
mails.findall(text)
```




    ['gosuslugi@mail.ru',
     'golikova_t67@gmail.com',
     'natysik@ya.ru',
     'support24@mos.ru']




```python
phones = re.compile('Your regexp here for phone numbers')
phones.findall(text)
```




    ['+7 (495) 777-777', '8-495-777-77-77', '84957777777 ']



### Result:

We have learned how to use regular expressions for searching and substitutions in the text.
