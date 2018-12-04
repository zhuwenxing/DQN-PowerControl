注：本代码是从Jun Fang教授的主页下载

**DQN-power-control** is the code for applying deep reinforcement learning for spectrum sharing in cognitive radios, available at http://www.junfang-uestc.net/codes/DQN-power-control.rar.

The code has been tested on Ubuntu 14.04 + tensorflow 1.0.0 + Python 2.7.

Reference:  Xingjian Li, Jun Fang, Wen Cheng, Huiping Duan, Zhi Chen, and Hongbin Li, "Intelligent Power Control for Spectrum Sharing in Cognitive Radios: A Deep Reinforcement Learning Approach," was accepted by IEEE Access, May 2018.

Written by: Wen Cheng

Email: JunFang@uestc.edu.cn


Get Start:

cd ./code

python main.py


Optional parameters (main.py):

noise - The estimation error of RSS

num_sensor - The number of sensors

policy - Power update policy of PU, suppose to be 1 or 2


