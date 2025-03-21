# Repository for Thesis
This repository contains code for the individual project component of MSc degree in Computing (Specialism) under Artificial Intelligence and Machine Learning: 

Evaluating the Adversarial Robustness of Reinforcement Learning System

Abstract:
The integration of Deep Learning into Reinforcement Learning has significantly expanded its potential applications across various domains, including robotics and autonomous vehicles. Offline Reinforcement Learning, which utilizes pre-existing datasets instead of continuous interaction with the environment, holds particular promise for extending Reinforcement Learning to use cases where real-time interaction is costly, impractical, or risky. However, the vulnerability of such systems to adversarial attacks remains poorly understood. This report explores adversarial artificial intelligence in Deep Reinforcement Learning, with a focus on evaluating the robustness of Offline Reinforcement Learning. We propose a threat model and attack framework aimed at introducing backdoor policies in the target agents. Three attacks, an Untargeted Attack, a Targeted Attack, and a Control Attack, are designed to implant a backdoor policy in different environments, each serving distinct objectives. Our evaluation reveals that Offline Reinforcement Learning systems are vulnerable to adversarial attacks, emphasizing the need for strong mitigation strategies to safeguard these systems.

----

This is a research that evaluates the adversarial robustness of Offline Reinforcement Learning agents and there are three attacks investigated in the report.

1. Untargeted Attack
2. Targeted Attack
3. Control Attack

The code for Untargeted Attack is in `mujoco_untargeted_attack` folder

The code for Targeted Attack is in `gridworld_targeted_control_attack` folder

The code for Control Attack (DISCRETE ENVIRONMENT) is in the `gridworld_targeted_control_attack` folder

The code for the Control Attack (CONTINUOUS ENVIRONMENT) is in `carracing_control_attack` folder