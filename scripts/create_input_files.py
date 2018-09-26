from scripts.utils import create_input_files
import os

if __name__ == '__main__':
    # Create input files (along with word map)
    CURR_DIR = os.getcwd()
    create_input_files(dataset='atlas',
                       karpathy_json_path=CURR_DIR +'/datasets/toy_dataset/indian_clothing_with_title_toy.json',
                       image_folder=CURR_DIR +'/datasets/toy_dataset/',
                       captions_per_image=1,
                       min_word_freq=20,
                       output_folder=CURR_DIR +'/outputs/',
                       max_len=15)
