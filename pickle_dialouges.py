import path # to handdle path
import pickle # to pickle dataset
from sys import argv # to take command line args

file_name, input_dir, output_file = argv # to get all command line args

path_dialouges = path.Path(input_dir) # updated path

dialouges_files = path_dialouges.files() # to get list of files in the dir

# text var to read all the text
text = ''

# iterate into dialouge paths
for i in dialouges_files:
    with open(i, 'r', encoding="utf-8") as fp:
        text += fp.read()

# make a pickle file for our output file
with open(output_file, 'wb') as fp:
    pickle.dump(text, fp)