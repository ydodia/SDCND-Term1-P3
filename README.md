# Behavioral Cloning - P3

### Overview
This aim of this project was to use a gain experience and practice with behavioral cloning. I used the provided simulator to drive a car around track 1 and collected data based on my driving ability using the mouse to control steering angle and the keyboard to control acceleration and braking. I then constructed a model based on NVIDIA's and used it for training and validation. Using this model, the simulator's car drove around track 1 successfully for two laps without once touching the lane boundaries (yellow lane lines).

### Model Architecture and Strategy
The `model.py` file contains my code for training, validating and saving the convolutional neural network. The model used was directly from NVIDIA (lines 82-91) as referenced in the lecture video. The model architecture is:

1. Crop the image by cutting off the top 70 pixels and bottom 20pixels; nothing is cropped from the left or right sides. (line 79)
2. Normalization by using a Keras Lambda layer. (line 80)

3. Convolutional layer with a depth, filter size and stride window of 24, (5x5) and (2,2), with ReLU activation.
4. Convolutional layer with a 32, (5,5) and (2,2) stride, with ReLU activation.
5. Convolutional layer with a 48, (5,5) and (2,2) stride, with ReLU activation.
6. Convolutional layer with a 64, (3,3) and default stride, with ReLU activation.
7. Additional convolutional layer with a 64, (3,3) and default stride, with ReLU activation.
8. Flatten everything to 1-D.
9. Fully-connected layer of size 100.
10. Fully-connected layer of size 50.
11. Fully-connected layer of size 10.
12. Fully-connected layer of size 1.

I didn't experience any indication of overfitting and thus didn't make any model modifications in this respect. My data sizes were X data points in total. I did a data split and used X for training and Y for validation, representing a 80/20 split. Additionally, I implemented data augmentation steps by using the left, right and mirrored center images for a total of 4 images or data points per capture line. Finally, I used an angle correction method to compensate for the left and right cameras.
The model uses Mean Squared Error (MSE) as the loss metric and the Adam optimizer with default learning rate (0.001).

The figure below helps visual NVIDIA's model as used in my project.

[arch]: ./NVIDIA_arch.png
![NVIDIA End-to-End Deep Learning for Self Driving Cars][arch]

### Data Capturing
I used the mouse for steering and the keyboard for acceleration and braking. I drove 2 laps clockwise (CW) and 1 lap counter-clockwise while trying to stay as close to the center of the track as possible. I then captured data for critical turns at three points (ordered CW): the first left turn with water on the right; the left turn with the dirt road opening on the right; and lastly, on the right turn with water on the left side. For each of these segments I captured a handful (3-4) of segments of the car starting close to the edge of the track and then steering hard to get back to the center. This resulted in smoothly re-centering to center of the track if the car failed to turn hard into the turn and risked getting run off-track. Total data size was 17,248, representing a capture each for the left, center and right cameras with corresponding steering angle taken from the center camera.

### Model Parameter Tuning
For the data, I tuned the angle correction value through trial and error via running the autonomous driving mode. I settled on a value of 0.11 since higher values (greater than 0.2) exhibited greater steering angle oscillations between positive and negative values.
For the NVIDIA model architecture, I settled on a `batch size of 32` and number of `epochs to 10`. I settled on a smaller batch size to allow for quicker computation in training. I increased the number of epochs from 8 to 10 to allow for a little better training, as measured by MSE which still seemed to decreased.
The model also a

### Results
My model's MSE results were `0.0056` after the 10th run. The true test was running the simulator on autonomous mode where the car drove 2 laps flawlessly, without touching any of the side, yellow lane lines. I decided to test it further and managed to get 6+ laps on the next run.
