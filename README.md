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
Run any of the .ipynb file :
&nbsp&nbsp&nbsp&nbspTo run cycle gan full(forard and backward) with ls loss - Main_CycleGAN_LS_loss.ipynb (Baseline)
&nbsp&nbsp&nbsp&nbspTo run cycle gan forward with BCE loss - Main_ForwardCycleGAN_BCE.ipynb (Our Experiment)
&nbsp&nbsp&nbsp&nbspTo run cycle gan forward with Wloss - Main_ForwardCycleGAN_Wloss.ipynb (Our Experiment)
&nbsp&nbsp&nbsp&nbspTo run cycle gan full(forward and backward) with Wloss - Main_CycleGAN_Wloss.ipynb (Our Proposal)
&nbsp&nbsp&nbsp&nbspTo run cycle gan full(forward and backward) with Wloss, Weight decay, feature loss - Main_CycleGAN_FeatureCycleLoss_WeightDecay.ipynb (Our Proposal)

Once the steps are done the Checkpoints, Sample Training result and Sample Test result are stored in the output_dir that was given. The output of all the types of main file is stored in different folder with different name.



Reference repo : https://github.com/LynnHo/CycleGAN-Tensorflow-2.
