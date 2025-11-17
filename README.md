# Sutton & Barto Reinforcement Learning Implementations

A collection of implementations from *Reinforcement Learning: An Introduction* by Sutton and Barto.

## Problems Implemented

### Jack's Car Rental (`jack_car_rental.py`)
Policy iteration solution for the Jack's Car Rental problem (Chapter 4). Uses dynamic programming to find the optimal policy for managing car inventory between two rental locations.

### Windy Gridworld (`windy_gridworld.py`)
SARSA algorithm implementation for the Windy Gridworld problem. Learns to navigate a gridworld with wind effects using temporal difference learning.

### Racetrack (`racetrack.py`)
Off-policy Monte Carlo control for the racetrack problem. Learns to navigate a racetrack from start to finish using weighted importance sampling.

## Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

- numpy
- scipy
- tqdm

## Usage

Each file can be run independently:

```bash
python jack_car_rental.py
python windy_gridworld.py
python racetrack.py
```

