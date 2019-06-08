import path # to handdle path
import pickle # to pickle dataset

path_dialouges = path.Path('dialouges')

dialouges_files = path_dialouges.listdir()

# text var to read all the text
text = ''

# iterate into dialouge paths
for i in dialouges_files:
    with open(i, 'r', encoding="utf-8") as fp:
        text += fp.read()

# make a pickle file 
with open('season1.pkl', 'wb') as fp:
    pickle.dump(text, fp)