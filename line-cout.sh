find . -name "*.py" |xargs cat|grep -v ^$|wc -l
