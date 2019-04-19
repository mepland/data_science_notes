[![Build Status](https://travis-ci.org/mepland/data_science_notes.svg?branch=master)](https://travis-ci.org/mepland/data_science_notes)
# Data Science Notes
---
Matthew Epland  
matthew.epland@duke.edu  
2019-  

All rights reserved except, the rights granted by the Creative Commons Attribution 4.0 International Licence [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)  

## Latex Editing Tips
---

### Searching for formatting problems with grep
```bash
grep -rIn ".  " . > tmp.txt
grep -rIn '[[:blank:]]$' . > tmp.txt
pcregrep -rInM '(\b[a-zA-Z]+)\s+\1\b' . > tmp.txt
grep -rIn '.%' . > tmp.txt
```

### Checking for spelling mistakes  
```bash
for f in $(find . -type f -name '*.tex') ; do aspell list < $f ; done | sort | uniq > tmp.txt
```
