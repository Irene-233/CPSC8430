# CPSC8430HW2

Is only for class CPSC8430

Use python 2.7 to run the code

To use:

Run hw2_seq2seq2.sh like that :
./hw2_seq2seq2.sh <path_to_video_features> <output_filename>
The path_to_video_features is the path of the test vedio features folder. The output_filename is the name of the ouput file.
Use the existing test video path as
./hw2_seq2seq2.sh ./data/test_features output.txt

To train the model, set the train feattures path in the model.py then run:
python 2.7 train.py 
then will get model in models folder

To get the RGB features, run the extract.py like that :
python 2.7 extract.py <path_to_video> <path_to_save_features>
The <path_to_video> is the folder path of the video, and the <path_to_save_features> is the path to save the features.


