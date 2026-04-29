# Robot Integration Guide

This guide explains how to integrate a new robot into the Robocasa framework.

## MJCF File Requirements
Location: [`robocasa/models/assets/robots/`](robocasa/models/robots)

### MJCF Structure Requirements
- The MJCF file must have a single root element containing all other elements (`<worldbody>`, `<asset>`, etc.)
- The robot's kinematic tree must start from a `<body name="base">` element that contains a freejoint and is placed directly under `<worldbody>`
- Joint definitions for the right arm must precede those of the left arm in the XML structure
- End effector attachments (grippers) are only supported on bodies named `<body name="right_eef">`

### Base Mounting Configuration
When operating in fixed-base mode, the body connected to the base joint will be anchored to the world frame. So choose this mounting point carefully, typically at the robot's pelvis/torso, to ensure proper fixed-base behavior.

## Controller Configuration
Location: [`robocasa/examples/third_party_controllers/`](robocasa/examples/third_party_controllers/)


### Configuration Requirements
1. Implement whole-body control configuration (excluding leg components for now)
2. Configure and tune `ik_posture_weight`
   - Remember to add "robot0_" prefix to all parameters


## Robot Registry Updates
Location: [`robocasa/models/robots/manipulators/`](robocasa/models/robots/manipulators/)

### Required Function Updates
Take a look at the existing robot class definitions for examples.
Remember to update the following functions with new robot's specifications:
1. `update_joints()`
   - Add joint names in correct order
   - Maintain right-before-left convention

2. `update_actuators()`
   - Add actuator names
   - Ensure order matches joint configuration

