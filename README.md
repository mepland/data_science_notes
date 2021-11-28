[![Build Status](https://travis-ci.org/mepland/data_science_notes.svg?branch=master)](https://travis-ci.org/mepland/data_science_notes)
# Data Science Notes

Matthew Epland, PhD  
[![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/matthew-epland/)
&nbsp;
[![GitHub](https://i.stack.imgur.com/tskMh.png) GitHub](https://github.com/mepland)  

All rights reserved except, the rights granted by the Creative Commons Attribution 4.0 International Licence [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)  

## Latex Editing Tips


### Searching for formatting problems with grep
```bash
grep -rIn ".  " . > tmp.txt
grep -rIn '[[:blank:]]$' . > tmp.txt
pcregrep -rInM '(\b[a-zA-Z]+)\s+\1\b' . > tmp.txt
grep -rIn '.%' . > tmp.txt
grep -rIn '([A-Z][A-Z]*)' . > tmp.txt
```

### Checking for spelling mistakes  
```bash
for f in $(find . -type f -name '*.tex') ; do aspell list < $f ; done | sort | uniq > tmp.txt
```

## Travis CI
Setup following the instructions on this [blog post](https://harshjv.com/blog/setup-latex-pdf-build-using-travis-ci/) and [associated repository](https://github.com/harshjv/travis-ci-latex-pdf).
