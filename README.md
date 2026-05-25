# Robotics path planning using RRT algorithm
Path planning in an indoor room with several obstacles for robotics using Rapidly-exploring Random Tree algorithm.
<br>
<br>
This version automates performance testing with different set of arbitrary parameters

<video src="./Assets/Demo.mp4" width="100%" controls></video>

How to run.
1. Install python3
2. (Optional) use anaconda to make new python environment
3. Install requirements via
    ```
    pip install -r requirements.txt
    ```
4. Git clone this repo or download
5. Go to repo's directory
6. Open terminal / cmd and run
    ```
    python main.p
    ```
7. Enjoy!

How to Set parameter
1. Open ```pref.py``` on text or code editor
2. Add, Remove, or change the value within this lines:
    ```
    EPOCH_VALUES       = [100, 250, 500, 1000]
    EXPAND_DIS_VALUES  = [1, 2, 3, 5]
    ```