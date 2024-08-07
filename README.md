# Traffic_Tracking Video

https://www.youtube.com/watch?v=PJ5xXXcfuTc&list=PLcQZGj9lFR7y5WikozDSrdk6UCtAnM9mB&index=4

# Traffic Tracking

This project is a traffic tracking system that detects and counts cars in a video using OpenCV and a custom tracking module.

## Prerequisites

- Python 3.x
- OpenCV
- Numpy
- Pytest

## Installation

1.Clone the repository:

```sh
git clone https://github.com/HaileInnoTech/Traffic_Tracking.git
cd traffic-tracking
```

2.Create virtual environment:

```sh
python -m venv venv
source venv/bin/activate
```

#### On Windows use `venv\Scripts\activate`

3.Install required packages:

```sh

pip install -r requirements.txt

```

## Running program

```sh

python TrackingTraffic.py

```

## Testing all function

```sh

pytest
OR
python -m pytest

```

## Testing a function

```sh

pytest -s -vv test_car_counter_originalMutated
OR
python -m pytest  -s -vv test_car_counter_originalMutated

```
