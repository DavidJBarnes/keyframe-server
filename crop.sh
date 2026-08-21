for i in 1 2 3 4; do
  convert ./inputs/$i.jpeg -resize 512x768^ -gravity center -extent 512x768 ./inputs/kf$i.png
done
