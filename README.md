===================================================
  LOW LIGHT IMAGE ENHANCEMENT PROJECT - README
===================================================

FOLDER STRUCTURE
-----------------
LowlightProject/
├── data/
│   ├── train_data/         ← Put training images (.jpg) here
│   ├── test_data/images/   ← Put test images here
│   └── result/images/      ← Enhanced output saved here
├── snapshots/              ← Trained model weights saved here
├── model.py
├── dataloader.py
├── Myloss.py
├── lowlight_train.py
├── lowlight_test.py
├── GUI.py
├── ssim.py
├── index.jpg
└── download.png

INSTALL DEPENDENCIES (run once)
---------------------------------
pip install torch torchvision
pip install opencv-python
pip install Pillow numpy matplotlib
pip install scikit-image easygui lpips

HOW TO RUN
-----------
STEP 1 - Train the model:
  > Add low-light .jpg images to data/train_data/
  > python lowlight_train.py --num_epochs 100
  > Saves model as snapshots/Epoch99.pth

STEP 2 - Test on a single image:
  > Add your image to data/test_data/images/
  > python lowlight_test.py
  > Output saved to data/result/images/

STEP 3 - Use the GUI:
  > python GUI.py
  > Click UPLOAD IMAGE → select your dark image
  > Click ENHANCE IMAGE → see the result
  > Click SAVE IMAGE → save where you want

STEP 4 - Check quality metrics:
  > python ssim.py

NOTE: GUI needs snapshots/Epoch99.pth to exist.
If you trained fewer epochs, rename your .pth file to Epoch99.pth
===================================================
