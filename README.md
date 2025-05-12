# MC-Elmap

Implementation of Multi-Coordinate Elastic Maps (MC-Elmap)

Corresponding paper can be found for free [here](https://arxiv.org/abs/2505.06092), please read for method details.

To learn manipulation skills, robots need to understand the features of those skills. An easy way for robots to learn is through Learning from Demonstration (LfD), where the robot learns a skill from an expert demonstrator. While the main features of a skill might be captured in one differential coordinate (i.e., Cartesian), they could have meaning in other coordinates. For example, an important feature of a skill may be its shape or velocity profile, which are difficult to discover in Cartesian differential coordinate. In this work, we present a method which enables robots to learn skills from human demonstrations via encoding these skills into various differential coordinates, then determines the importance of each coordinate to reproduce the skill. We also introduce a modified form of Elastic Maps that includes multiple differential coordinates, combining statistical modeling of skills in these differential coordinate spaces. Elastic Maps, which are flexible and fast to compute, allow for the incorporation of several different types of constraints and the use of any number of demonstrations. Additionally, we propose methods for auto-tuning several parameters associated with the modified Elastic Map formulation. We validate our approach in several simulated experiments and a real-world writing task with a UR5e manipulator arm.

<img src="https://github.com/brenhertel/mc-elmap/blob/main/pictures/mc_elmap.jpg" alt="" width="600"/> 

This repository implements the method described in the paper above using Python. Scripts which perform individual experiments are included, as well as other necessary utilities. If you have any questions, please contact Brendan Hertel (brendan_hertel@student.uml.edu).

If you use the code present in this repository, please cite the following paper:
```
@inproceedings{hertel2025mcelmap,
  title={Robot Learning Using Multi-Coordinate Elastic Maps},
  author={Hertel, Brendan and Azadeh, Reza},
  booktitle={22nd International Conference on Ubiquitous Robots (UR)},
  year={2025},
}
```
