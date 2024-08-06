Model pipeline that performs joint fusion (Type I and II) using electrocardiograms (ECGs) and echocardiograms (echos)
- Type I (ML+CNN): fuse extracted time domain ECG features with features from Echo CNN backbone --> MLP
- Type II (CNN+CNN): fuse features from both ECG CNN and Echo CNN --> MLP

main.py
- parameters are initialized in 'params' variable
- dataloaders are created based on ecg_fusion_type (dataloader_base.py)
- model is loaded based on fusion type (model_base.py)
- loss function and optimizer defined (training_base.py)
- training script called from training_base.py
- model weights are saved from best epoch (based on validation loss)
- test dataloader defined, model weights loaded for testing
- model_predict called to generate predictions on test set
- predictions and probabilities saved slice-wise to .csv file

dataloader_base.py
- ecg_dataloader called from main.py
- inputs = parameters dict(), ECG dataframe containing file names and time domain ECG features, mode (train,val,test)
- call load_png_wID class
- loads echo .png files with corresponding de-identified patient ID, diagnostic label, ECG data (time domain features or ECG signal)
- Saves png file, ID, label, ECG data as the sample within the dataloader
- Output of ecg_dataloader is a dict() containing a dataset for train, val, test

model_base.py
- ECG_ML_Echo_CNN_Fusion(): for Type I Fusion
  - loads in ResNext101 (or other architecture for echo CNN)
  - fc_layers defined in order to downsample the output features from the Echo CNN backbone
  - time domain ECG features concatenated with extracted, downsampled Eco features (both of the same length)
  - concatenated feature vector fed into final_mlp which downsamples to final 3 logits --> fed into cross entropy function during training
 
training_base.py
- new_train_epochs function called
- loops throug n number of epochs
  - Runs through train dataset
  - extracts echo inputs, labels, patient IDs, ECG inputs for each batch
  - echo inputs and ECG inputs fed into fusion model
  - output is logits --> cross entropy loss function
  - softmax applied to logits to generate predictions
  - same applied to validation dataset
  - classification reports generated for training and validation
  - all epoch-wise information stored in a logger
  - best model returned and saved
