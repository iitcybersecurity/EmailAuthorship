#!/bin/bash

for f in *; 
    do 
    sed -i -e '0, /^[[:space:]]*$/d' -e '/---/, $d' $f; 
    tr '\n' '\v' < $f > temp;
    sed -e 's/\v\v\v.*$//' temp | tr '\v' '\n' > $f;
    #sed -i '/^[[:space:]]*$/d' $f;  
    rm temp;  
    done
find ./ -size 0 -print -delete  

exit 0