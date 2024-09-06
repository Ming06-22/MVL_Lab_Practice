# $${\color{yellow}Plant \space Seedlings \space Classification}$$
Source: [https://www.kaggle.com/competitions/plant-seedlings-classification/overview](https://www.kaggle.com/competitions/plant-seedlings-classification/overview)

## $${\color{red}Description}$$
Can you differentiate a weed from a crop seedling?

The ability to do so effectively can mean better crop yields and better stewardship of the environment.

The Aarhus University Signal Processing group, in collaboration with University of Southern Denmark, has recently released a dataset containing images of approximately 960 unique plants belonging to 12 species at several growth stages.

<img src="https://storage.googleapis.com/kaggle-media/competitions/seedlings-classify/seedlings.png"></img>

We're hosting this dataset as a Kaggle competition in order to give it wider exposure, to give the community an opportunity to experiment with different image recognition techniques, as well to provide a place to cross-pollenate ideas.

## $${\color{red}Submission \space File}$$

For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## $${\color{yellow}One-fold \space Validation \space Version}$$
### $${\color{red}Environment}$$
Python==3.11.9
torch==2.4.0
torchvision==0.19.0
pandas==2.2.2
seaborn==0.13.2
matplotlib==3.9.2
tqdm==4.66.5

### $${\color{red}Data \space Preprocess}$$
Model: [ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)<br>
Device: MPS<br>
Image Transformation:<br>
&ensp;&ensp;&ensp;&ensp;ToImage() -> Resize((224, 224)) -> ToDtype(torch.float32, scale = True) -><br>&ensp;&ensp;&ensp;&ensp;Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))<br>
Traing Set: Validation Set = 7: 3<br>

### $${\color{red}Hyperparameter \space Setting}$$
Weights: ResNet50_Weights.IMAGENET1K_V2<br>
Batch Size: 32<br>
Number of Epoch: 15<br>
Optimizer: Adam, learning rate = 0.001<br>
Loss Function: torch.nn.functional.cross_entropy

### $${\color{red}Training \space / \space Validation \space Loss \space Curve}$$
<img src="https://github.com/Ming06-22/MVL_Lab_Practice/blob/main/one-fold%20validation/loss_curve.png?raw=True"></img>

### $${\color{red}Confusion \space Matrix}$$
<img src="https://github.com/Ming06-22/MVL_Lab_Practice/blob/main/one-fold%20validation/confusion%20matrix_epoch%2015.png?raw=true"></img>

### $${\color{red}Submission \space Result}$$
<img src="https://github.com/Ming06-22/MVL_Lab_Practice/blob/main/prediction/one-fold%20validation%20submission%20result.png?raw=true"></img>

## $${\color{yellow}K-fold \space Validation \space Version}$$
### $${\color{red}Environment}$$
Python==3.11.9
torch==2.4.0
torchvision==0.19.0
pandas==2.2.2
seaborn==0.13.2
matplotlib==3.9.2
tqdm==4.66.5
sklearn==1.5.1
IPython==8.27.0

### $${\color{red}Data \space Preprocess}$$
Model: [ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)<br>
Device: MPS<br>
Image Transformation:<br>
&ensp;&ensp;&ensp;&ensp;ToImage() -> Resize((224, 224)) -> ToDtype(torch.float32, scale = True) -><br>&ensp;&ensp;&ensp;&ensp;Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))<br>

### $${\color{red}Hyperparameter \space Setting}$$
Weights: ResNet50_Weights.IMAGENET1K_V2<br>
Batch Size: 32<br>
Number of Fold: 5<br>
Number of Epoch: 15<br>
Optimizer: Adam, learning rate = 0.001<br>
Loss Function: torch.nn.functional.cross_entropy

### $${\color{red}Training \space / \space Validation \space Setting}$$
<img src="https://github.com/Ming06-22/MVL_Lab_Practice/blob/main/k-fold%20validation/loss_curve_fold%203.png?raw=True"></img>

### $${\color{red}Confusion \space Matrix}$$
<img src="https://github.com/Ming06-22/MVL_Lab_Practice/blob/main/k-fold%20validation/confusion%20matrix_fold%203_epoch%2015.png?raw=true"></img>

### $${\color{red}Submission \space Result}$$
<img src="https://github.com/Ming06-22/MVL_Lab_Practice/blob/main/prediction/k-fold%20validation%20submission%20result.png?raw=true"></img>