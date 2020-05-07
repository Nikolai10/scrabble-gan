from src.dinterface.iam_handwriting_db import convert_to_gan_reading_format_save


def init_reading(train_input, train_output, target_size, bucket_size):
    """
    Convert iamDB dataset (words) to "GAN format":

    Prerequisite:
    Download data/words and data/ascii into res/data/iamDB/words/ and res/data/iamDB/words.txt, respectively
    link: http://www.fki.inf.unibe.ch/DBs/iamDB/data/words/words.tgz
    link: http://www.fki.inf.unibe.ch/DBs/iamDB/data/ascii/words.txt

    Input:
    res/data/iamDB/words/
        ├── a01
            ├── ...
        ├── ...
        ├── r06
    res/data/iamDB/words.txt


    Output:
    res/data/iamDB/words-Reading/
        ├── Bucket 1
            ├── img_i.png
            ├── img_i.txt
            ├── ...
        ├── Bucket 2
            ├── ...
        ├── Bucket n

    notes:
    - img_i.txt contains the ground truth annotation (transcription) for img_i.png
    - Bucket x holds all samples with transcription length x


    :param train_input:
    :param train_output:
    :param target_size:
    :param bucket_size:
    :return:
    """
    print('convert iamDB words to GAN-Reading task...\n')
    convert_to_gan_reading_format_save(train_input, train_output, target_size, bucket_size)
