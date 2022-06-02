# HMS vision
This project is aimed at building a machine learning model that detects smoke plumes in GOES satellite imagery in the same way as HMS does.

## Data
Data required for HMS vision is stored as png files on Mickley lab storage. 

### Data Description
#### Input Variables
HMS is produced from GOES Geostationary Satellite imagery. GOES provides two bandwidths, visible and infrared, both of which are used for HMS production. We downloaded GOES imagery for every thirty minutes from the beginning of 2017 to the end of 2021, and crop them from -180E to 0 in longitude, and from 90N to 0 in latitude. The resulting images have dimension of 12384x6192. For our traininnng purposes, we split them up into square tiles. The side length of the tiles have to be $2^n$ where $n\in\mathbb{N}$.

#### Target Variable
Similarly to GOES imagery, we convert HMS into 12384x6192 images and split up into tiles. Each pixel of HMS is classified into four categories (0, 5, 16, and 27) depending on the concentration of PM2.5. We convert those categories into simple integer (0,1,2, and 3) so that the target variable is 2-D array where each element indicate the density level of smoke plume (no smoke, low, medium, and high).

## Model
### Network architecture
For our model, we use U-net architecture; U-net is a deep learning architecture developed by Olaf Ronneberger et al. for biomedical image segmentation. The original model takes cell images as input and classifies if each pixel is on cell border or within a cell. In our case, the model takes in seven images over certain region at certain timestamp, and classifies each pixel in the input domain into one of four category of smoke density using HMS as target variable. The model first "encodes" the input image by applying $3\times3$ convolution layers with ReLU and $2\times2$ max pooling operations, and "decode" by applying upsampling convolution layers to compute pixel classification of the input from the feature map. We modify the original model in that we add padding to each convolution layers to preserve the length of the edge (the output image is shrinked in the original U-net).

## Training
### Preprocessing
We filter the dataset by how much smoke is in the target HMS image. Since most of the images have no smoke, when we train the model on the whole dataset, the model learns to always output "no smoke" prediction. We omit images that do not contain heavy smoke (smoke density 3) since heavy smoke is most important and easier to detect.
### Optimization
We use Adam optimizer, which uses both Momentum (0.99) and RMSProp methods. Our loss function is categorical cross-entropy over pixel-wise softmax. Let $\Omega$ be the dimension of output image and $\hat{\mathbf{x}}$ be the model output vector of pixel where $\hat{\mathbf{x}}\in\Omega$. $\hat{\mathbf{x}}_i$ represents the model output for $i$th category in $\hat{\mathbf{x}}$, where in our case $i\in\{0,1,2,3\}$. Then softmax is defined as 
$$p_k(\hat{\mathbf{x}})=\frac{\exp(\hat{\mathbf{x}}_k)}{\sum_{i=0}^3\exp(\hat{\mathbf{x}}_i)}.$$
Where $p_k(\hat{\mathbf{x}})$ represents the probability of pixel being category $k$ predicted by model. Let $q_k(\mathbf{x})$ be the correct probability of pixel $\mathbf{x}$ being category $k$ (it is either $0$ or $1$). Then the loss function is 
$$\mathcal{L}=\sum_{\mathbf{x}\in\Omega}\sum_{i\in\{0,1,2,3\}}q_i(\mathbf{x})\log(p_i(\hat{\mathbf{x}})).$$
It will penalize when the model outputs high probability for wrong class.


## Running training
You can run your training by running below;
```python
python main.py --side-len=256 --batch-size=32 --smoke=dense --epoch-num=20 > log_train_$(date "+%Y.%m.%d-%H.%M.%S").txt 2>&1
```
`main.py` takes in some arguments to configure your training.
