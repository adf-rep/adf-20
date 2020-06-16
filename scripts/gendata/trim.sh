IMAGES="./train/*.jpg"
for file in $IMAGES
do
	echo "$file"
	convert $file -resize 32x32! -gravity center $file

done
