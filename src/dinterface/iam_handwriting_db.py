import cv2
import os
from collections import Counter


def convert_to_gan_reading_format_save(input_dir, output_dir, target_size, bucket_size):
    """
    Convert iamDB dataset to "GAN format"

    (1) get list of all files in directory tree res/data/iamDB/words/
    (2) get file transcriptions (stored in words.txt)
    (3) process/ filter images along its respective transcriptions
    (4) compute meta data (such as transcription-length distribution across iamDB words)

    :param input_dir:       directory containing iamDB (words)
    :param output_dir:      output location
    :param target_size:     (height, width, channels) - here height width ratio is 2:1 (32:16px) per char
    :param bucket_size:     max transcription length considered in this work
    :return:
    """

    h, w, _ = target_size
    valid_samples = 0
    transcription_lengths = []

    if not os.path.exists(output_dir):
        for i in range(bucket_size):
            # create buckets (in this work, words longer than 10 chars are for the sake of simplicity ignored)
            os.makedirs(output_dir + str(i + 1) + '/')

    # (1) get list of all files in directory tree res/data/iamDB/words/
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(input_dir):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    # (2)get file transcriptions (stored in words.txt)
    transcription_file = input_dir.rstrip('/') + '.txt'
    transcriptions = {}

    with open(transcription_file, 'r', encoding="utf8") as fi:
        for line in fi:
            if not line.startswith('#'):
                labels = line.split()
                file_nm = labels[0] + '.png'

                # mark not properly segmented words as '-1'
                if 'ok' == labels[1]:
                    transcription = labels[-1]
                    transcriptions[file_nm] = transcription.strip()
                else:
                    transcriptions[file_nm] = '-1'

    print('size of iamDB words: {}'.format(len(transcriptions)))

    # (3) process/ filter images along its respective transcriptions
    for idx, file in enumerate(listOfFiles):

        if file.endswith(".png"):

            print(file)
            # get file name and its corresponding transcription
            img_nm = os.path.basename(file)
            transcription = transcriptions[img_nm]

            # filter samples with chars other a-zA-Z
            if transcription.isalpha() and len(transcription) > 0:
                len_word = len(transcription)

                # read image (grayscale mode) and resize to common data size
                img = cv2.imread(os.path.join(input_dir, file), 0)

                try:
                    resized_img = cv2.resize(img, (int(h / 2) * len_word, h))

                    # compute transcription length and save to corresponding output bucket
                    transcription_length = len(transcription)
                    output_bucket = output_dir + str(transcription_length) + '/'
                    print(output_bucket)

                    if transcription_length <= bucket_size:
                        cv2.imwrite(os.path.join(output_bucket, img_nm), resized_img)
                        with open(os.path.join(output_bucket, os.path.splitext(img_nm)[0] + '.txt'), 'w',
                                  encoding="utf8") as fo:
                            fo.write(transcription)

                        valid_samples += 1
                        transcription_lengths.append(transcription_length)
                except:
                    print('error at: {}'.format(file))

    # (4) compute meta data (such as transcription-lenght distribution across iamDB words)
    print('size of valid iamDB words: {}'.format(valid_samples))
    print(Counter(transcription_lengths))
