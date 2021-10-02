# MDA-CNN
   MiRNA (MicroRNA) is a type of single-stranded small RNA of about 18-25 nts that 
is ubiquitous in eukaryotic cells. MiRNA plays a very important role in regulating cell 
proliferation, differentiation, and apoptosis. In recent years, with the deepening of 
people's research, more and more evidence shows that there is a close relationship 
between miRNA and disease occurrence. Therefore, exploring the relationship between 
miRNA and disease has become the research topic of many researchers. In recent years, 
with the development of computer technology and bioinformatics, the simulation of 
computer methods to calculate the relationship between miRNA and disease has become 
the focus of current research. This method can reduce the workload of biological 
experimenters. It is also possible to use miRNA for disease judgment as soon as possible, 
and to develop miRNA targeted drugs for clinical treatment.  
   In this thesis, we obtain the disease similarity network, gene-gene association 
network, miRNA similarity network, miRNA-gene network and disease-gene network 
from the data in multiple databases, and obtain the three-layer miRNA-gene-disease 
network through the above five networks . Then extract the miRNA-gene and disease-gene characteristics from the three-layer network of miRNA-gene-disease. The general 
idea of this step is based on an innovative regression model to find the similarity score 
between different miRNAs and between different diseases. Similarity score, and then find 
the similarity score between miRNA (or disease) and different genes. Using a stack 
autoencoder, the obtained feature vectors are processed for dimensionality reduction and 
denoising. Finally, the convolutional neural network is used to classify the feature vectors. 
The ten-fold cross-validation method is used to obtain the final classification processing 
results. Various evaluation criteria are used to evaluate and compare with the 
experimental results of other models. Experimental results show that the proposed method 
can effectively and quickly predict the association between miRNA and disease.
