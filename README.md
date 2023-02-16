[![Build Status](https://travis-ci.org/mepland/data_science_notes.svg?branch=master)](https://travis-ci.org/mepland/data_science_notes)
# Data Science Notes

### Matthew Epland, PhD
[![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/matthew-epland/)
&nbsp;
[![GitHub](https://i.stack.imgur.com/tskMh.png) GitHub](https://github.com/mepland)

All rights reserved

## Latex Editing Tips


### Searching for formatting problems with grep
```bash
grep -rIn ".  " *.tex > tmp.txt
grep -rIn '[[:blank:]]$' *.tex > tmp.txt
pcregrep -rInM '(\b[a-zA-Z]+)\s+\1\b' *.tex > tmp.txt
grep -rIn '.%' *.tex > tmp.txt
grep -rIn '([A-Z][A-Z]*)' *.tex > tmp.txt
grep -rIn '\\href{' *.tex > tmp.txt
grep -rIn ' $' *.tex > tmp.txt
```

### Checking for spelling mistakes
```bash
echo 'personal_ws-1.1 en 1' > dictionary.tmp | sed -e '/[^a-zA-Z]/d' ~/.vim/spell/en.utf-8.add >> dictionary.tmp | for f in $(find . -type f -name '*.tex') ; do aspell --home-dir=. --personal=dictionary.tmp --mode=tex list < $f ; done | sort | uniq > misspelled_words.txt && rm -rf dictionary.tmp
```

## Travis CI
Setup following the instructions on this [blog post](https://harshjv.com/blog/setup-latex-pdf-build-using-travis-ci/) and [associated repository](https://github.com/harshjv/travis-ci-latex-pdf).
