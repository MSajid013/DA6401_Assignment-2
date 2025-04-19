# Part B: Fine-Tuning a Pre-Trained Model (ResNet50)

Overview:

- I fine-tuned a pre-trained ResNet50 model using the iNaturalist dataset.
- The model was initialized with ImageNet pre-trained weights, and modified to work with 10 classes instead of 1000.
- Training from scratch is slow and resource-heavy, so fine-tuning helps speed things up and improves performance.

 Fine-Tuning Strategies Tried:
1. Freeze all layers except the last layer
Only the final classification layer is trained.

2. Freeze the first k layers, train the rest
Tried freezing 25%, 50%, and 75% of the layers.

3. Freeze the last fully connected layers, train conv layers
This approach was less effective.

 Result Analysis:

- Freezing 75% of the layers gave good validation accuracy.

- Training only the last layer also gave strong performance.

- Training conv layers only (without updating classifier) did not perform well.

- Logged and compared validation accuracies across all strategies.

# To run train.py file:
python train_transfer.py \
  --data_dir ./path_to_data \
  --model_name resnet \
  --freeze_percent 0.5 \
  --freeze_all_except_last_layer No \
  --epochs 5 \
  --apply_augmentation Yes
