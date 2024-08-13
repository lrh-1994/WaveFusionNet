# WaveFusionNet
Infrared and visible image fusion

******2024/7/10********
This is a official code for WaveFusionNet, which is proposed for infrared and visible image fsuion, 

demostrating good performance  in both visual perception and quantitative assessment.

The required environment and packages are listed in the requirements.txt.

The paper for WaveFusionNet is uder-review and will be available according to the policy of the journal.

Complete code and detailed results will be released soon after being accepted by the journal. 
************************
We provide three benchmark datasets: MSRS, RoadScene, TNO for performance assessment, 
source image pairs and fused images are contained in .\test_data and .\results directory, respectively.

The input and reuslts are organized as follows:

.\test_data
|
|___TNO
         |___IR
         |___VI
|___RoadScene
         |___IR
         |___VI
|___MSRS
         |___IR
         |___VI

.\results
|
|__TNO
|
|__RoadScene
|
|__MSRS

*************************************************************
Testing:

First, researchers can generate fused images for three test dtasets by implementing test_demo.py:

python test_demo.py

Then, you can test the perfoemance of fused images by implementing metric.py:

python metric.py

*************************************************************
We express our gratitude to the researchers who provide the evaluation code as reference, and extend our thanks to the authors of the datasets and comparison methods. 

