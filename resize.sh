for f in "/home/ubuntu/keras/enver/acrec/custom_data_generator/activity_recognition/activity_data/*/*/*/*.jpg"
do
     mogrify $f -resize 224x224! $f
done



