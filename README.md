# MicroRacer
A didactic car-racer micro environment for Deep Reinforcement Learning


## Aim and motivation
MicroRacer is a simple environment inspired by car racing and especially meant for the didactics of Deep Reiforcement Learning.
The complexity of the environment has been explicitly calibrated to allow to experiment with many different methods, networks and hyperparameters settings
without the need of sophisticated software and no fear of getting bored by too long training times.

## The MicroRacer environment
MacroRacer generates new random circular tracks at each episode. The Random track is defined by CubicSplines delimiting the inner and outer border; the number of turns and the width of the track are configurable. From this description we derive a dense matrix of points of dimension 1300x1300 providing information about points inside the track. This is the actual definition of the track used by the environment.
![micro_racer](https://user-images.githubusercontent.com/15980090/135791705-cd678320-c189-43b5-84fe-1ceb0dd01f0d.png)

## State and actions
MicroRacer **does not** intend to model realistic car dynamics. The model is explicitly meant to be as simple as possible, with the minimal amount of complexity that still makes learning interesting and challenging.

The **state** information available to actors is composed by:
  1. a lidar-like vision of the track from the car's frontal perspective. This is an array of 19 values, expressing the distance of the car from the track's borders along uniformly spaced angles in the range -30°,+30°. 
  2. the car scalar velocity.
  
The actor (the car) has no global knowledge of the track, and no information about its absolute or relative position/direction w.r.t. the track.

The actor is supposed to answer with two actions, both in the range [-1,1]: 
  1. acceleration/deceleration
  2. turning angle
Maximum values for acceleration and turning angles can be configured. 

## Available learning models
We currently equip the code with a basic actor trained with DDPG (weights included). Students are supposed to develop their own models. 

## Requirements
The project just requires basic libraries: tensorflow, numpy, pyplotlib and scipy.interpolate (for Cubic Splines). 

## Plans for future work
The current version of the MicroRacer is very basic.

We are extremely interested in collaborations, especially with colleagues teaching DRL at other Universities.
We plan to investigate the following directions
1. the computation of the lidar signal in pyhton is too slow. We plan to move it to C.
2. improve the modularity and configurability of the system of rewards 
3. create a "competion" mode, allowing the simultaneous execution of multiple (independent) cars. This is easy to do, but too slow, at present.
4. add "chicanes" 
5. add obstacles, e.g. in the form of other moving vehicles
6. add a policy-based sample code, in addition to DDPG (e.g. with PPO)

