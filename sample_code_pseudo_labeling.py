import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

backbone = 'clip'
mode = 'single'

batch_size = 1900000
# note: data loading has been deleted for safety.

# Step 6: Define the model
#a) Linear Classifier
class LinearClassifier(nn.Module):
    def __init__(self, input_dim=768):
        """
        Args:
            input_dim (int): Dimension of the input features.
        """
        reduced_dim = 1 #768, 512, 248, 124, 64, 32, 1
        super(LinearClassifier, self).__init__()
        # Define all layers in __init__
        self.linear_single = nn.Linear(input_dim, 1)  # Single-layer binary classification
        self.linear1_multi = nn.Linear(input_dim, reduced_dim)  # First layer for multi
        self.linear2_multi = nn.Linear(reduced_dim, 1)  # Second layer for multi

        

        
    def forward(self, x, mode =  'single'):
        """
        Args:
            x (Tensor): Input tensor.
            mode (str): "single" for single linear layer, "multi" for two-layer ReLU setup.
        
        Returns:
            Tensor: Model output.
        """
        if mode == "single":
            # Single-layer binary classification
            return torch.sigmoid(self.linear_single(x))  # Use sigmoid activation for binary classification, needed for BCE loss
        elif mode == "multi":
            # Two-layer setup with ReLU activation
            return torch.sigmoid(self.linear2_multi(F.relu(self.linear1_multi(x))))
            #return torch.sigmoid(self.linear_single(x))


def train_model(train_loader, val_samples, val_labels, model=None, optimizer=None, epochs=1000, patience=10):
    input_dim = 768
    if model is None:
        model = LinearClassifier(input_dim)
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X, mode=mode).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_outputs = model(val_samples, mode=mode).squeeze()
            val_loss = criterion(val_outputs, val_labels)

        if val_loss.item() < best_val_loss - 1e-4:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                model.load_state_dict(best_model_state)
                break

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss.item():.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, optimizer



def generate_pseudo_labels(model, wild_data_tensor, used_indices, threshold=0.9):
    """
    Generate pseudo-labels from model predictions on wild data.

    Args:
        model: Trained PyTorch model with sigmoid already applied.
        wild_data_tensor: Torch tensor of shape [N, feature_dim].
        threshold: Confidence threshold for selecting pseudo-labels.

    Returns:
        confident_X: Tensor of high-confidence inputs.
        confident_y: Corresponding binary pseudo-labels.
    """
    model.eval()

    with torch.no_grad():
        probs = model(wild_data_tensor, mode=mode).squeeze()  # already sigmoid-ed
        pseudo_labels = (probs > 0.5).float()
        confidence = torch.abs(probs - 0.5) * 2  # confidence ∈ [0, 1]

    # Only keep new confident examples
    new_X = []
    new_y = []
    new_indices = []

    for idx in range(len(wild_data_tensor)):
        if confidence[idx] >= threshold and idx not in used_indices:
            new_X.append(wild_data_tensor[idx].cpu())
            new_y.append(pseudo_labels[idx].cpu())
            new_indices.append(idx)

    print(f"✅ Found {len(new_X)} new pseudo-labeled samples")

    # Update used indices
    used_indices.update(new_indices)

    if new_X:
        return torch.stack(new_X), torch.stack(new_y)
    else:
        return None, None


#pseudo_labeling

# Initial training on labeled data
# Initial training
train_data_labeled_DataLoader = torch.utils.data.DataLoader(train_data_labeled, batch_size=batch_size, shuffle=True)

model = LinearClassifier(input_dim=768)  # initialize once
optimizer = optim.Adam(model.parameters(), lr=0.001)  # initialize once

model, optimizer = train_model(
    train_data_labeled_DataLoader,
    val_labeled_samples,
    labels_val_labeled_samples,
    model=model,
    optimizer=optimizer
)

# Step 3: Iterative pseudo-labeling loop

num_iterations = 5
used_indices = set()  # to track which indices were already used

for iteration in range(num_iterations):
    print(f"\n🔁 Iteration {iteration + 1}")

    # Generate pseudo-labels
    confident_X, confident_y = generate_pseudo_labels(model, wild_data, used_indices, threshold=0.9)

    if confident_X is None or len(confident_X) == 0:
        print("⚠️ No confident pseudo-labels found. Stopping early.")
        break


    # Create combined dataset
    pseudo_dataset = torch.utils.data.TensorDataset(confident_X, confident_y)
    combined_dataset = torch.utils.data.ConcatDataset([train_data_labeled, pseudo_dataset])
    train_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    # retrain model on labeled + pseudo-labeled data (should switch to finetune later)
    model, optimizer = train_model(train_loader, val_labeled_samples, labels_val_labeled_samples, model=model, optimizer=optimizer)


print("✅ Pseudo-labeling and fine-tuning complete!")




# Step 10: Evaluate the model
# List of generator names
generator_names = list(test_data.keys())


model.eval()
result={}
with torch.no_grad():
    criterion = nn.BCELoss()
    print("BCE loss on labeled: ", criterion(model(samples), labels.view(-1, 1)).item())
    
    #test on target_model 3
    print(f"Evaluating target_model 3:")
    test_target_model_set = torch.tensor(test_target_model, dtype=torch.float32)
    labels_test_target_model = torch.zeros(test_target_model_set.size(0)) #target_model has labels 0
    y_pred_target_model = model(test_target_model_set).squeeze()

    all_preds = [y_pred_target_model.clone()]
    all_labels =[labels_test_target_model.clone()]

    all_hard_preds = [y_pred_target_model.clone()]
    all_hard_labels =[labels_test_target_model.clone()]

    hard_and_semi_hard_preds = [y_pred_target_model.clone()]
    hard_and_semi_hard_labels = [labels_test_target_model.clone()]

    no_data_generators_preds = [y_pred_target_model.clone()]
    no_data_generators_labels = [labels_test_target_model.clone()]



    #print("target_model prediction: ", y_pred_target_model[: 10])
    y_pred_labels_target_model = (y_pred_target_model >= 0.5).float() #true is converted to 1, false is converted to 0 in Pytorch
    #print("target_model prediction after converting: ", y_pred_labels_target_model[:10])

    true_positive = (y_pred_labels_target_model == labels_test_target_model).float().sum()
    false_negative= (y_pred_labels_target_model != labels_test_target_model).float().sum()
    #print("True positive: ", int(true_positive.item()), "False negative: ", int(false_negative.item()))
    
    target_model_accuracy = float( true_positive/ (true_positive + false_negative))
    target_model_ap = average_precision_score(labels_test_target_model, y_pred_target_model, pos_label=0)

   

    total_true_negative = 0
    total_false_positive = 0
    for generator in generator_names:
        print(f"Evaluating {generator}:")
        test_dataset = test_data[generator] 
        test_dataset = torch.tensor(test_dataset, dtype=torch.float32)
        labels_test = torch.ones(test_dataset.size(0)) #other generators have label 1)
        y_pred = model(test_dataset).squeeze()
        #print('raw_prediction', y_pred[:10])
        
        all_preds.append(y_pred)
        all_labels.append(labels_test)


        y_pred_labels = (y_pred >= 0.5).float()  # Thresholding at 0.5. if >=0.5 gets labeled = 1, else labeled= 0

        true_negative = (y_pred_labels == labels_test).float().sum() #number of being able to predict not target_model
        false_positive = (y_pred_labels != labels_test).float().sum()

        total_true_negative += true_negative
        total_false_positive += false_positive

        #print("True negative: ", int(true_negative.item()), "False positive: ", int(false_positive.item()))

        Accuracy = (true_negative)/( false_positive + true_negative )
        

        y_true_combined = torch.cat((labels_test_target_model, labels_test), dim=0)  # Combine labels
        y_scores_combined = torch.cat((y_pred_target_model, y_pred), dim=0)  # Combine scores

        # Convert to NumPy arrays for scikit-learn
        y_true_np = y_true_combined.cpu().numpy()
        y_scores_np = y_scores_combined.cpu().numpy()
        #print(y_true_np[:10])
        #print( y_scores_np[:10])
    

        # Calculate Average Precision
        ap = average_precision_score(y_true_np, y_scores_np, pos_label=1)
        roc_auc = roc_auc_score(y_true_np, y_scores_np)

        print( "AP: ", ap, "ROC_AUC: ", roc_auc )
        result[generator] = [None, None, float(true_negative), float(false_positive), Accuracy.item(),  ap]
   

   #some evaluations deleted for safety
