import os
import numpy as np
import h5py
import json
import torch
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import re


def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'atlas', 'flickr8k', 'flickr30k'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    train_image_titles = []

    val_image_paths = []
    val_image_captions = []
    val_image_titles = []

    test_image_paths = []
    test_image_captions = []
    test_image_titles = []

    word_freq = Counter()
    title_word_freq = Counter()

    for img in data['images']:

        title = img['title']
        title_tokens = get_title_tokens(title, max_len)
        title_word_freq.update(title_tokens)

        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
            train_image_titles.append(title_tokens)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
            val_image_titles.append(title_tokens)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)
            test_image_titles.append(title_tokens)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map for captions
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create word map for titles
    title_words = [w for w in title_word_freq.keys() if title_word_freq[w] > min_word_freq]
    title_word_map = {k: v + 1 for v, k in enumerate(title_words)}
    title_word_map['<unk>'] = len(title_word_map) + 1
    title_word_map['<start>'] = len(title_word_map) + 1
    title_word_map['<end>'] = len(title_word_map) + 1
    title_word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save caption word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Save title word map to a JSON
    with open(os.path.join(output_folder, 'TITLE_WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(title_word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, imtitles, split in [(train_image_paths, train_image_captions, train_image_titles, 'TRAIN'),
                                             (val_image_paths, val_image_captions, val_image_titles, 'VAL'),
                                             (test_image_paths, test_image_captions, test_image_titles, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []
            enc_titles = []
            title_lens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

                title_tokens = imtitles[i]
                enc_title = encode_title(title_tokens, title_word_map, max_len)

                # Find title length
                title_len = len(title_tokens) + 2 # add 2 for start and end tokens

                enc_titles.append(enc_title)
                title_lens.append(title_len)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)

            # Save encoded titles and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_TITLES_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_titles, j)

            with open(os.path.join(output_folder, split + '_TITLELENS_' + base_filename + '.json'), 'w') as j:
                json.dump(title_lens, j)


def encode_title(title_tokens, title_word_map, max_len=15):
    enc_title = [title_word_map['<start>']] + \
                [title_word_map.get(word, title_word_map['<unk>']) for word in title_tokens] + \
                [title_word_map['<end>']] + [title_word_map['<pad>']] * (max_len - len(title_tokens))
    return enc_title


def get_title_tokens(title, max_len):
    title = title.lower()
    title = title.replace("'s", "")
    title_tokens = re.split(r'\s+', title)
    if len(title_tokens) > max_len:
        title_tokens = title_tokens[:max_len]
    return title_tokens

