# Anki Autonomous Racing
Autonomous racing with the Anki Overdrive.
Using the Overdrive-Python API (https://github.com/xerodotc/overdrive-python) develop sensing, state estimation, and control techniques for static obstacle avoidance using the Anki Overdrive vehicle.

Running the code:
Run "roslaunch anki_drive anki_cv.launch" to launch the USB camera tracking. Following this, you may run "rosru anki_drive planning.py" to run the obstacle avoidance routine for the Anki Overdrive. Within this file, in the "main" method, you may change the type of controller you'd like by setting its parameter to "True." For example, if you'd like to use MPC for obstacle avoidance, you may change the variable do_MPC to True, and change all other controller variables (do_P, do_PD, do_PD_setpoint) to False. The car will then use MPC to determine its path.

System ID:
If you wish to reperform system ID of the Anki Overdrive vehicle, where a model later used by MPC is fit to the vehicle, uncomment "trial_SysID.sysID_datacollect(pointFromPixel, last_image_service)" in the main method of mapping_track_to_model.py. This will enable and run the data collection, storage, and regression procedures. After uncommenting, run "rosrun anki_drive mapping_track_to_model.py" to perform the system ID.
