# Execute only once!
    """Example_California_Housing"""

import os
import sys
sys.path.append("..")
os.chdir("..")

%load_ext autoreload
%autoreload 2


import numpy as np
import pandas as pd
import logging
from sklearn import datasets


from utils import set_logging_level
from be_great import GReaT


import matplotlib.pyplot as plt

logger = set_logging_level(logging.INFO)


data = datasets.fetch_california_housing(as_frame=True).frame
data.head()

MedInc	HouseAge	AveRooms	AveBedrms	Population	AveOccup	Latitude	Longitude	MedHouseVal
0	8.3252	41.0	6.984127	1.023810	322.0	2.555556	37.88	-122.23	4.526
1	8.3014	21.0	6.238137	0.971880	2401.0	2.109842	37.86	-122.22	3.585
2	7.2574	52.0	8.288136	1.073446	496.0	2.802260	37.85	-122.24	3.521
3	5.6431	52.0	5.817352	1.073059	558.0	2.547945	37.85	-122.25	3.413
4	3.8462	52.0	6.281853	1.081081	565.0	2.181467	37.85	-122.25	3.422



great = GReaT("distilgpt2",                         # Name of the large language model used (see HuggingFace for more options)
              epochs=1,                             # Number of epochs to train (only one epoch for demonstration)
              save_steps=2000,                      # Save model weights every x steps
              logging_steps=50,                     # Log the loss and learning rate every x steps
              experiment_dir="trainer_california",  # Name of the directory where all intermediate steps are saved
              #lr_scheduler_type="constant",        # Specify the learning rate scheduler 
              #learning_rate=5e-5                   # Set the inital learning rate
             )



Step	Training Loss
50	2.680700
100	2.031900
150	1.939400
200	1.914700
250	1.875900
300	1.867900
350	1.851000
400	1.836000
450	1.800000
500	1.803000
550	1.806700
600	1.810800
650	1.815500
700	1.797400
750	1.778400
800	1.800900
850	1.793400
900	1.794900
950	1.794400
1000	1.777800
1050	1.773700
1100	1.769200
1150	1.781900
1200	1.772400
1250	1.769500


loss_hist = trainer.state.log_history.copy()
loss_hist.pop()

{'train_runtime': 111.2294,
 'train_samples_per_second': 185.562,
 'train_steps_per_second': 11.598,
 'total_flos': 464022201434112.0,
 'train_loss': 1.854305208191391,
 'epoch': 1.0,
 'step': 1290}

 loss = [x["loss"] for x in loss_hist]
epochs = [x["epoch"] for x in loss_hist]