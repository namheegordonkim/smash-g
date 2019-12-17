# SMASH-G: A System for Modeling, Analyzing, and Synthesizing Hand Gestures
Nam Hee Gordon Kim and Tim Straubinger

This is a supplementary code for our project: https://sites.google.com/view/smash-g

We used Python 3.6 on Ubuntu 18.04 LTS along with these libraries (not comprehensive):
* matplotlib
* scikit-learn
* imageio
* tensorflow

Also, an installation of Docker or Singularity is recommended.

## Instructions

### Step 1: Install MANO and Configure

* Download MANO codebase after signing up: https://mano.is.tue.mpg.de/
* Also install mesh https://github.com/MPI-IS/mesh
* MANO is written in Python 2.7, and therefore you must make tweaks to make it Python 3.6 compatible.
* Also, create `__init__.py` in every directory and subdirectory of MANO.

### Step 2: Deploy the 3D hand pose estimator (Zimmermann et al.)

* We employed a remote server with a GPU and used socket programming to process videos.
* We provide the Dockerfile used inside the hand3d subdirectory. If running Docker, `docker pull namheegordonkim/handgpu` should suffice.
* Use `docker shell` or similar to access the files inside the container.
* Once inside the container, run `python3 -u server.py` to listen to port 3333.
* If needed, set up an SSH tunnel so the client running on your edge device can communicate with the remote server.

### Step 3: Preprocess data

You can download our video data here: https://www.dropbox.com/s/ep8jhys2ie4kjda/smash-g-data.zip?dl=0

Put all the .mp4 files inside the subdirectory `./data/`.

To process all the data in one command, run this inside a bash shell:

```
for GESTURE in "ok" "thumbs_up" "paper" "scissors" "call_me" "lets_drink"
do
    for i in {00..08}
    do
        python preprocess_video_with_cc.py --input_file ./data/"$GESTURE"$i.mp4 --output_dir ./data/dynamic/$GESTURE/$i/
    done
done
```

### Step 4: Learn the dynamics

First, another preprocessing step for learning dynamics:

```
python preprocess_dynamics_data.py
```

Then run

```
python learn_dynamics.py
```

You should be able to visualize the dynamics with

```
python visualize_dynamics.py
```

### Step 6: Generate trajectories

You can now synthesize the data with the learned dynamics. Simply run:

```
python generate_trajectory.py --gesture_name <gesture_name>
```

where `<gesture_name>` is one of `{ok, thumbs_up, paper, scissors, call_me, lets_drink}`.

This generates three trajectories total, with varying speed parameter values.

### Step 5: Animate MANO Hand

Finally, animate with inverse kinematics using:

```
python mano_render.py --gesture_name <gesture_name> --speed <speed>
```

where `<speed>` is one of `{0.5, 1.0, 1.5}`.


