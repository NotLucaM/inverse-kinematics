# Inverse Kinematics
A playground to test out inverse kinematics for my robotic arm.
I did not create the CAD for this arm, it can be found on [here](https://www.thingiverse.com/thing:1015238).

Code currently does inverse kinematics using `autograd` for gradient descent of the angles.
The cost function calculates the distance to the target point and returns a extremely high distance if the angle limits are not met.
