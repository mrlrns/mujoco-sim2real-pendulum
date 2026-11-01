"""MJCF description of the single pendulum used in the project.

One hinge (rotation around y), one motor (gear = 3, control in [-1, 1]),
contacts disabled so that only the actuator, gravity and joint damping act on the pole.
"""

XML = """
<mujoco>
  <option timestep="0.002">

    <flag contact="disable"/>
  </option>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="4096" offsamples="8"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.45 0.65 0.85" rgb2="0.05 0.05 0.1" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="6 6" reflectance="0.25"/>
    <material name="rod" rgba="0.85 0.85 0.9 1" specular="0.8" shininess="0.8"/>
    <material name="bob" rgba="0.95 0.35 0.2 1" specular="0.7" shininess="0.6" reflectance="0.15"/>
    <material name="pivot" rgba="0.2 0.2 0.25 1" specular="0.5"/>
  </asset>
  <worldbody>
    <light pos="1 -2 3" dir="-0.3 0.6 -1" diffuse="0.9 0.9 0.9" castshadow="true"/>
    <geom type="plane" size="4 4 0.1" material="grid"/>
    <geom type="capsule" fromto="0 0.15 0 0 0.15 1.2" size="0.03" material="pivot"/>
    <geom type="capsule" fromto="0 0.15 1.2 0 0 1.2" size="0.03" material="pivot"/>
    <camera name="cam" pos="0.8 -3 1.1" mode="targetbody" target="pole"/>
    <body name="pole" pos="0 0 1.2">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom type="cylinder" fromto="0 -0.07 0 0 0.07 0" size="0.05" mass="0.05"
      contype="0" conaffinity="0" material="bob"/>

      <geom type="capsule" fromto="0 0 0 0 0 -0.6" size="0.025"  mass="0.05" material="rod"/>
      <geom type="sphere" pos="0 0 -0.6" size="0.1" mass="1" material="bob"/>

    </body>
  </worldbody>
  <actuator>
    <motor joint="hinge" gear="3" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""
