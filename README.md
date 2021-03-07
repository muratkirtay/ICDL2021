#ICDL2021
A repository for reproducing the results presented in ICDL-2021 submission.

>**Abstract:**Trust is an essential component in human-human and human-robot interactions. The factors that play potent roles in these interactions have been an attractive issue in robotics. However, the studies that aim at developing a computational model of robot trust in interaction partners remain relatively limited. In this study, we extend our emergent emotion model to propose that the robot’s trust in the interaction partner (i.e., trustee) can be established by the effect of the interactions on the computational energy budget of the robot  (i.e., trustor).  To be concrete, we show how high-level emotions (e.g., well-being) of an agent can be modeled by the computational cost of perceptual processing (e.g., visual stimulus processing for visual recalling) in a decision-making framework. To realize this approach, we endow the Pepper humanoid robot with two modules: an auto-associative memory that extracts the required computational energy to perform a visual recalling, and an internal reward mechanism guiding model-free reinforcement learning to yield computational energy cost-aware behaviors.   With this setup, the robot interacts with online instructors with different guiding strategies, namely random, less reliable, and reliable. Through interaction with the instructors, the robot associates the cumulative reward values based on the cost of perceptual processing to evaluate the instructors and determine which one should be trusted. Overall the results indicate that the robot can differentiate the guiding strategies of the instructors. Additionally, in the case of free choice, the robot trusts the reliable one that increases the total reward -- and therefore reduces the required computational energy (cognitive load)-- to perform the next task.

## Folder and file descriptions
+ **Assets:** this folder contains various assets to create the scene for visual processing, visual patterns to train associative memory, etc.  
+ **Source:** the source files to run the experiments, preprocess assets, postprocess the data, and visualize the results.  Note that **hopfield_helpers.py** contains the methods to run the experiment, and you need an actual Pepper robot to fully realize the same experiments. 
+ **Data:** this contains subfolders for the results of each instructor in .pkl format. Check visualization in the **Source folder** to obtain the result for the various runs for each instructor.  
+ **Figures:** the subfolders host the figures generated for the articles, the related figures (e.g., average energy, total visits, Q matrices, TD errors, cumulative reward curves) for all runs for each instructors. 
+ **Logs:** various logs to evaluate results: final policy, number of correct actions, etc. Note that the folders and logs with unreliable instructor renamed as less reliable instructor.  
+ **TheRobotTrustICDL2021.mp4:** A video file to show the experiment demo and the interactions with an instructor.  
+ **requirements.txt:** a text file contains the version numbers of the python packages for running the scripts; use **pip install -r requirements.txt** to install the packages with correct versions.  