This is the code implementation for the paper under review. 

**PRE-REQUISITES**
Python (3.6.4),
Pytorch (0.4.1),
CUDA, and
numpy

**
Use the following commands to run the code:**

**ResNet-18**:  python train_resnet.py 


**Wideresnet**: python train_wideresnet.py


**PRETRAINED MODELS**

**LRAT** WideResnet-34-10  https://drive.google.com/file/d/1r8mbv1TWB7Z9aDWCc8aap352nCWMo9_Q/view?usp=sharing


**LRLLAT** WideResnet-34-10  https://drive.google.com/file/d/1JU2Wx8D0pnMXGC70BAtJOBV3XqldVbiz/view?usp=sharing
**LRAT**  ResNet18 https://drive.google.com/file/d/1M7WSNrwnk3Jxvad0lrHFb8IfMo3UHuta/view?usp=sharing


ResNet-18
| Method              	| PGD (%) 	| CW (%)| Natural Accuracy(%)
|-----------------------|-----------------------|------------------|--------------------|
| LRAT (Proposed)   		|  54.67%   	|     52.74  		|     82.96%            |
| LRLLAT(Proposed)   		|  54.39%   	|     51.89  		|        81.69%             |






Wideresnet-34-10
| Method              	| PGD(%) 	|  CW(%) | Natural Accuracy(%) | AutoAttack(%)
|-----------------------|-----------------------|------------------|-------------|--------------|
| LRAT (Proposed)   		|  58.83   	|  56.70     		|        85.48     | 52.16      |
| LRLLAT(Proposed)   		|  58.37  	|   56.65 		|           85.36       | 53.51      |

  
