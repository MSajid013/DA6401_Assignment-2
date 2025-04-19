# CNN Model on iNaturalist Dataset
In this assignment, we build a CNN (Convolutional Neural Network) using PyTorch and Torchvision from scratch and train it on the iNaturalist dataset. We also try different settings (hyperparameters) to get the best performance.

I made this assignment in Kaggle. First of all I uploaded the iNaturalist dataset in dataset section on Kaggle, then I created a notebook for Part-A of this assignment and I copied the path to the uploaded dataset and used it in the code where needed.

# Model Details
The model has 5 convolutional layers.

Each layer uses an activation function (like ReLU or Mish).

Max Pooling is used after each activation to reduce image size.

I used Dropout to prevent overfitting.

There is one Dense (Fully Connected) layer before the output.

The Output layer has 10 classes (for 10 categories in iNaturalist).

# Things I Tried (Hyperparameters)
I tested different combinations to see what works best:

Kernel sizes:
[3, 3, 3, 3, 3], [3, 5, 5, 7, 7], [3, 5, 3, 5, 7], [5, 5, 5, 5, 5], [7, 7, 7, 7, 7]

Dropout rates:
0.2, 0.3

Activation functions:
ReLU, GELU, SiLU, Mish

Use of Batch Normalization:
Yes / No

Filter sizes (number of channels):
[32, 32, 32, 32, 32], [32, 64, 64, 128, 128], [128, 128, 64, 64, 32], [32, 64, 128, 256, 512]

Dense layer size:
128 or 256 neurons

Data Augmentation:
Yes / No (adding random changes to images to improve generalization)

#  Training Steps
Load and split the training data into 80% for training and 20% for validation.

Used Weights & Biases (wandb) to run a hyperparameter sweep (Bayesian method).

Trained the model with different settings.

Pick the model with the best validation accuracy.

#  Best Settings Found
These settings gave the highest performance:

Kernel Sizes: [5, 5, 5, 5, 5]

Dropout: 0.2

Activation: Mish

Batch Normalization: Yes

Filters: [32, 64, 128, 256, 512]

Dense Layer: 128 neurons

Data Augmentation: No

# Final Evaluation
After training, I tested the best model on the test set.

The final accuracy 35.60% was reported and provided sample images using the test data.

# To run the train.py file:
python trainA.py --data_dir ./dataset --epochs 10 --apply_augmentation Yes --activation_function gelu --dropout_rate 0.3
