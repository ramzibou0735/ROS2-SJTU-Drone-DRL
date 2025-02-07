# Project Prototype

This project is currently in its **prototype phase**. It is intended for use with the **SJTU Drone** package and serves as a foundation for future developments.

## Current Features
- The project can now be trained using an SB3 PPO but at this stage the reward structure is only good enough for the drone to learn not to run into obstacles after a few iterations.

## Future Updates
- This project will be updated **very frequently** to include new features and enhancements.
- Planned improvements include: 
  - Integration with reinforcment learning algorithms.
  - Enhanced target search and decision-making capabilities.
  - Improved reward functions and parameters.
  - better hyperparameters for the drone not to get stuck in local minimas.

## Usage
1. Ensure you have the **SJTU Drone** package properly set up as described in the repository READMD.md https://github.com/NovoG93/sjtu_drone/tree/ros2 .
2. Clone this repository inside your workspace and install all of the python librarties contained in the package.xml file:
3. With one terminal run:
    ```bash
   ros2 launch sjtu_drone_bringup sjtu_drone_gazebo.launch.py
   ```
4. With a second terminal run the test script:
    ```bash
   python3 sb3_train.py
   ```

## Disclaimer
As this project is still in development, it may lack certain functionalities and could contain bugs. Please use it with caution in experimental setups.

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests to help improve the project.

