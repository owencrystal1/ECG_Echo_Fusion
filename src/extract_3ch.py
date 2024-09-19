import pandas as pd
import pickle
import re
from tqdm import tqdm

def generate_filenames(filepath, window=1):
    # Use regex to extract the base filename and the numeric part

    filename = filepath.rsplit('/', 1)[-1]
    just_path = filepath.split(filename)[0]
    match = re.match(r'^(.*?)(\d+)(\.[a-zA-Z0-9]+)$', filename)
    
    if not match:
        raise ValueError("Filename format is incorrect.")
    
    base = match.group(1)   # The part before the number
    num_part = int(match.group(2))  # Convert the number part to an integer
    ext = match.group(3)    # The file extension
    
    # Generate filenames with numbers in the range around the current number
    filenames = [just_path + f"{base}{num_part + i}{ext}" for i in range(-window, window + 1)]
    
    return filenames

def get_3ch(df, es_ed_frames):

    df.reset_index(inplace=True, drop=True)

    ind_to_keep_test = []
    df.reset_index(inplace=True, drop=True)
    for i in tqdm(range(len(df))):
        test_file = df['path'][i]
        test_fn = test_file.rsplit('/', 1)[-1]
        frame_num = test_fn.rsplit('_', 1)[-1]
        frame_num = int(frame_num.split('.')[0])
        test_fn = test_fn.rsplit('_', 1)[0]
        

        if df['label'][i] == 0:
            test_fn = 'AMY_' + test_fn
        elif df['label'][i] == 1:
            test_fn = 'HCM_' + test_fn
        elif df['label'][i] == 2:
            test_fn = 'HTN_' + test_fn

        test_fn = test_fn.rsplit('_', 1)[0]
        filt_df = ed_es_frames[ed_es_frames['Filename'] == test_fn] # create df with only this file name (all frames)
        if frame_num in filt_df['Frame'].tolist(): # then append if we can find that frame number 
            ind_to_keep_test.append(i)

    print('Test set:', len(ind_to_keep_test))

    test_paths = [df['path'].tolist()[i] for i in ind_to_keep_test]
    test_labels = [df['label'].tolist()[i] for i in ind_to_keep_test]
    test_paths.sort()

    # these are the indices that we should keep
    ch3_test_paths = []
    ch3_test_labels = []
    all_test_files = df['path'].tolist()
    for i in range(len(test_paths)):

        file_array = generate_filenames(test_paths[i], window=1) # generate [es-1, es, es+1] file_path array and corresponding label
        
        # ensure all 3 files in 3ch array exist
        if all(item in all_test_files for item in file_array):
            ch3_test_paths.append(file_array)
            ch3_test_labels.append(test_labels[i])

    return ch3_test_labels, ch3_test_paths

# read existing training data
pd_file = '/datasplits.pkl'
df = pd.read_pickle(pd_file)

# read in results from echonet
results = pd.read_csv('size_lvh.csv')
ed_frames = results[results[' ComputerLarge'] == 1]
es_frames = results[results['ComputerSmall'] == 1]

ed_es_frames = pd.concat([ed_frames, es_frames], ignore_index=True)
frames_oi = ed_es_frames['Filename'].tolist()

frames_oi = [name.rsplit('_', 1)[0] for name in frames_oi]
ed_es_frames['Filename'] = frames_oi


df_test = df['test']
df_val = df['val']
df_train = df['train']

ch3_test_labels, ch3_test_paths = get_3ch(df_test, ed_es_frames)
ch3_val_labels, ch3_val_paths = get_3ch(df_val, ed_es_frames)
ch3_train_labels, ch3_train_paths = get_3ch(df_train, ed_es_frames)

df_es_ed_train = pd.DataFrame()
df_es_ed_train['paths'] = ch3_train_paths
df_es_ed_train['label'] = ch3_train_labels

df_es_ed_val = pd.DataFrame()
df_es_ed_val['paths'] = ch3_val_paths
df_es_ed_val['label'] = ch3_val_labels

df_es_ed_test = pd.DataFrame()
df_es_ed_test['paths'] = ch3_test_paths
df_es_ed_test['label'] = ch3_test_labels

es_ed_df = {
    'train': df_es_ed_train,
    'val': df_es_ed_val,
    'test': df_es_ed_test
}

# input pickle path
pickle_path = '3ch_datasplits.pkl'

with open(pickle_path, 'wb') as file:
    pickle.dump(es_ed_df, file)
    
