# Data Science Notes
---
Matthew Epland  
matthew.epland@duke.edu  
2019-  

All rights reserved except, the rights granted by the Creative Commons Attribution-Noncommercial Licence [CC BY-NC 3.0 US](http://creativecommons.org/licenses/by-nc/3.0/us/)  

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
