# Generating Controllable and Realistic Images by Unsupervised Image to Image Translation

## Dependencies:

Install the following dependencies:

1. Python 3.6.
2. Tensorflow 2.2.
3. TensorFlow Addons 0.10.0.
4. OpenCV, scikit-image, tqdm, oyaml.

## Steps to run:

1. Clone this Repository.
2. Edit the output_dir variable in Main_CycleGAN_LS_loss.ipynb, Main_ForwardCycleGAN_BCE.ipynb, Main_ForwardCycleGAN_Wloss.ipynb, Main_CycleGAN_Wloss.ipynb, Main_CycleGAN_FeatureCycleLoss_WeightDecay.ipynb.
3. Give the dataset path of real and comic images in the variable dataset_real and dataset_comic respectively in Main_CycleGAN_LS_loss.ipynb, Main_ForwardCycleGAN_BCE.ipynb, Main_ForwardCycleGAN_Wloss.ipynb, Main_CycleGAN_Wloss.ipynb, Main_CycleGAN_FeatureCycleLoss_WeightDecay.ipynb.
4. Run any of the .ipynb file :

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;To run cycle gan full(forard and backward) with ls loss - Main_CycleGAN_LS_loss.ipynb (Baseline)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;To run cycle gan forward with BCE loss - Main_ForwardCycleGAN_BCE.ipynb (Our Experiment)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;To run cycle gan forward with Wloss - Main_ForwardCycleGAN_Wloss.ipynb (Our Experiment)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;To run cycle gan full(forward and backward) with Wloss - Main_CycleGAN_Wloss.ipynb (Our Proposal)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;To run cycle gan full(forward and backward) with Wloss, Weight decay, feature loss - Main_CycleGAN_FeatureCycleLoss_WeightDecay.ipynb (Our Proposal)



Once the steps are done the Checkpoints, Sample Training result and Sample Test result are stored in the output_dir that was given. The output of all the types of main file is stored in different folder with different name.

The already run .ipynb files are in the files - Main_CycleGAN_LS_loss_With_Output.ipynb, Main_ForwardCycleGAN_BCE_With_Output.ipynb, Main_ForwardCycleGAN_Wloss_With_Output.ipynb, Main_CycleGAN_Wloss_With_Output.ipynb, Main_CycleGAN_FeatureCycleLoss_WeightDecay_With_Output.ipynb.

## Results

### Cycle GAN Full with LS loss
#### Sample Training
##### Comic to Real to Comic and Real to Comic to Real
![Full_CycleGAN_LS_Training](/uploads/67125bf52ef664569641699139f0d1c1/Full_CycleGAN_LS_Training.PNG)
#### Sample Testing
##### Comic to Real to Comic
![Full_CycleGAN_LS_Testing_A2B](/uploads/b9239a22b49888675b1b3f8d2e9229fd/Full_CycleGAN_LS_Testing_A2B.PNG)
##### Real to Comic to Real
![Full_CycleGAN_LS_Testing_B2A](/uploads/917f68e633df07979fad7209558bd2be/Full_CycleGAN_LS_Testing_B2A.PNG)


### Cycle GAN Forward with BCE loss
#### Sample Training
##### Comic to Real to Comic
![Forward_CycleGAN_BCE_Training](/uploads/8b41e96d55dfb80fc7afbe2993103489/Forward_CycleGAN_BCE_Training.PNG)
#### Sample Testing
##### Comic to Real to Comic
![Forward_CycleGAN_BCE_Testing](/uploads/aa0c84962f56ceeb34e7702610a3b8bf/Forward_CycleGAN_BCE_Testing.PNG)

### Cycle GAN Forward with W loss
#### Sample Training
##### Comic to Real to Comic
![Forward_CycleGAN_W_Testing](/uploads/6376ac8b85f4d8884de2bce8cfe01452/Forward_CycleGAN_W_Testing.PNG)
#### Sample Testing
##### Comic to Real to Comic
![Forward_CycleGAN_W_Training](/uploads/6731bff202de8c2e045c7f125852fb3d/Forward_CycleGAN_W_Training.PNG)

### Cycle GAN Full with W loss
#### Sample Training
##### Comic to Real to Comic and Real to Comic to Real
![Full_CycleGAN_W_Training](/uploads/26d45012a28e50abfbf8348a4f3b3ec8/Full_CycleGAN_W_Training.PNG)
#### Sample Testing
##### Comic to Real to Comic
![Full_CycleGAN_W_Testing_A2B](/uploads/90622d80cf4538161b04f3792791bc36/Full_CycleGAN_W_Testing_A2B.PNG)
##### Real to Comic to Real
![Full_CycleGAN_W_Testing_B2A](/uploads/35e271ed94c45bfdb16912620b97c158/Full_CycleGAN_W_Testing_B2A.PNG)

### Cycle GAN Full with Feature loss and Weight Decay
#### Sample Training
##### Comic to Real to Comic and Real to Comic to Real
![Full_CycleGAN_Feature_Weight_Training](/uploads/951810ce32f609189f54070101939bde/Full_CycleGAN_Feature_Weight_Training.PNG)
#### Sample Testing
##### Comic to Real to Comic
![Full_CycleGAN_Feature_Weight_Testing_A2B](/uploads/fa2f37f2561538969f52810b19ff66ad/Full_CycleGAN_Feature_Weight_Testing_A2B.PNG)
##### Real to Comic to Real
![Full_CycleGAN_Feature_Weight_Testing_B2A](/uploads/2a45e64da2424280949a416833cd2ca1/Full_CycleGAN_Feature_Weight_Testing_B2A.PNG)

Reference repo : https://github.com/LynnHo/CycleGAN-Tensorflow-2.
